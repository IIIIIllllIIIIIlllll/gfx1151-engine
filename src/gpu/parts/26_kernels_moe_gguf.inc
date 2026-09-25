// --- GGUF routed-expert F16 WMMA GEMM ---------------------------------------
//
// Ported from gufo (MIT, src/models/qwen38_flash_next/kernels/rocm/
// kernels.hip.cpp, RoutedF16GEMMKernel) and adapted to this engine's routing
// data: expert-sorted slots (d_eoff / d_tokidx), 64-slot MoeTile list from
// k_moe_tiles, outputs indexed by slot.
//
// Weights stay in their raw llama.cpp block formats (Q4_K / Q5_K for
// gate+up, Q5_1 / Q8_0 for down). Each stage (two 32-element K blocks) stages
// the packed codes plus one (scale, bias) half2 per (row, K block) in LDS;
// each wave then decodes only its own 16 rows to F16 right after the LDS read:
//
//   code -> half via v_perm into the mantissa of 1024.0 (0x6400 | q == 1024+q
//   exactly for q < 32), a packed subtract of 1024 (1152 for Q8_0, whose
//   signed bytes are carried as q+128 by flipping the sign bit), then one
//   packed FMA  w = q * scale + bias,
//   (scale, bias) = (d*sc, -dmin*mn) for Q4_K/Q5_K, (d, m) for Q5_1, (d, 0) Q8_0.
//
// The matrix core then accumulates the whole K extent in F32
// (wmma_f32_16x16x16_f16). Activations are F16 rows.
//
// Block = 256 threads = 8 waves, BM = 128 weight rows (16 per wave), BN = 64
// slots (one MoeTile), BK = 2 K blocks per stage.
//   kPair (gate/up): waves 0-3 compute 64 gate rows, waves 4-7 the matching
//     64 up rows; the epilogue writes F16 hid[slot][m] = up * silu(gate).
//     grid = (m / 64, tiles).
//   !kPair (down): F32 out[slot][m], grid = (m / 128, tiles).
// Rows of a tile past its count read zero activations and are not written.

namespace moegg {

constexpr int kQ4K = 12, kQ5K = 13, kQ51 = 7, kQ80 = 8;  // gguf::Type values
constexpr uint32_t kHalfMagic = 0x64646464U;              // 1024.0 high bytes

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8f wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

template <int T>
__host__ __device__ constexpr size_t row_bytes(size_t k) {
  return T == kQ4K ? (k / 256) * 144
       : T == kQ5K ? (k / 256) * 176
       : T == kQ51 ? (k / 32) * 24
                   : (k / 32) * 34;
}

__device__ __forceinline__ uint32_t header_byte(const uint4& h, uint32_t i) {
  const uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
  return (word >> (8U * (i & 3U))) & 0xFFU;
}

// Bit `s` of each of the four bytes of `w`, packed into bits 0-3.
__device__ __forceinline__ uint32_t gather_bit(uint32_t w, int s) {
  return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}

// Bits 0-3 of `bits` spread to bit 4 of each byte.
__device__ __forceinline__ uint32_t spread_high(uint32_t bits) {
  return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}

__device__ __forceinline__ void codes_to_halves(uint32_t codes, __half2 magic,
                                                __half2 scale2, __half2 bias2,
                                                __half2& lo, __half2& hi) {
  const uint32_t p0 = __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const uint32_t p1 = __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

}  // namespace moegg

// w: expert-stacked weights [E][m][row_bytes(k)] (gate for kPair), w_up: the
// up tensor (kPair only). x: F16 [*, k] rows; tokidx maps slot -> x row
// (nullptr: x row = slot, used by down which reads hid[slot]).
template <int kType, bool kPair, int BN = 64>
__launch_bounds__(256) __global__
    void k_moe_gg(const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_up,
                  const __half* __restrict__ x, const int* __restrict__ tokidx,
                  const MoeTile* __restrict__ tiles, const int* __restrict__ ntiles,
                  float* __restrict__ out, __half* __restrict__ out_half, int m, int k) {
  using namespace moegg;
  constexpr int BM = 128, BK = 2;
  static_assert(BN % 16 == 0 && BN <= 64, "BN must match the MoeTile size (<= 64)");
  constexpr int kTokTiles = BN / 16;
  constexpr bool kQ5 = kType == kQ51;
  constexpr bool kQ5Kt = kType == kQ5K;
  constexpr bool kQ8 = kType == kQ80;
  static_assert(!kPair || kType == kQ4K || kType == kQ5K, "pair = K-quant gate/up");
  constexpr int kChunks = kQ8 ? 2 * BK : BK;
  constexpr int kCodeBytes = BM * kChunks * 16;
  constexpr int kHighBytes = (kQ5 || kQ5Kt) ? BK * BM * 4 : 0;
  constexpr int kScaleBytes = BK * BM * 4;
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes + kActBytes;
  constexpr int kLdsBytes = kStageBytes > 8 * 1024 ? kStageBytes : 8 * 1024;
  __shared__ __attribute__((aligned(16))) uint8_t lds[kLdsBytes];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_high = reinterpret_cast<uint32_t*>(lds + kCodeBytes);
  auto* s_scale = reinterpret_cast<uint32_t*>(lds + kCodeBytes + kHighBytes);
  auto* s_act = reinterpret_cast<uint4*>(lds + kCodeBytes + kHighBytes + kScaleBytes);

  if ((int)blockIdx.y >= *ntiles) return;
  const MoeTile tile = tiles[blockIdx.y];
  const int expert = tile.expert;
  const int slot0 = tile.first;
  const int nrows = tile.count;  // <= BN
  const int live_tok_tiles = (nrows + 15) / 16;
  const int num_kb = k / 32;
  const size_t rb = row_bytes<kType>((size_t)k);

  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  constexpr int kRows = kPair ? BM / 2 : BM;
  const int r_block = (int)blockIdx.x * kRows;

  // Weight fetch: thread = (row tid/2, 16-byte chunk tid%2).
  const int f_c = tid & 1;
  const uint8_t* f_ptr;
  bool f_live;
  {
    const int r = r_block + (kPair ? (tid >> 1) % kRows : (tid >> 1));
    f_live = r < m;
    const uint8_t* base = w;
    if (kPair && (tid >> 1) >= kRows) base = w_up;
    f_ptr = base + (size_t)expert * m * rb + (size_t)(f_live ? r : (m - 1)) * rb;
  }
  uint4 f_header = make_uint4(0u, 0u, 0u, 0u);
  uint4 f_codes;
  uint4 code_cache[4];
  uint4 f_codes_hi;
  uint4 f_qh[2];
  uint32_t f_high = 0, f_dm = 0;
  int f_sb32 = 0;
  constexpr int kActFetch = 2;
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256, "activation fetch");
  uint4 a_data[kActFetch];
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + i * 256;
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    int src = -1;
    if (chunk < kActChunks && t < nrows) src = tokidx ? tokidx[slot0 + t] : slot0 + t;
    a_src[i] = src >= 0 ? x + (size_t)src * k + sub * 8 : nullptr;
    a_slot[i] = chunk < kActChunks ? sub * kActStride + t : -1;
  }

  const auto swizzle = [](int row, int c) {
    return row * kChunks + (c ^ (kChunks == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1)));
  };

  const auto fetch_stage = [&](int kb0) {
    if constexpr (kQ5) {
      const int kb = kb0 + f_c;
      const auto* words = reinterpret_cast<const uint2*>(f_ptr) + kb * 3;
      const uint2 w0 = words[0], w1 = words[1], w2 = words[2];
      f_codes = make_uint4(w1.x, w1.y, w2.x, w2.y);
      f_high = w0.y;
      f_dm = w0.x;
    } else if constexpr (kQ8) {
      const uint8_t* blk = f_ptr + (kb0 + f_c) * 34;
      f_dm = *reinterpret_cast<const uint16_t*>(blk);
      __builtin_memcpy(&f_codes, blk + 2, 16);
      __builtin_memcpy(&f_codes_hi, blk + 18, 16);
    } else {
      constexpr int kBlockChunks = kQ5Kt ? 11 : 9;
      constexpr int kCodeChunk = kQ5Kt ? 3 : 1;
      const int block = kb0 / 8;
      const auto* blk = reinterpret_cast<const uint4*>(f_ptr) + block * kBlockChunks;
      if (kb0 % 8 == 0) {
        f_header = blk[0];
        if constexpr (kPair) {
#pragma unroll
          for (int g = 0; g < 4; ++g) code_cache[g] = blk[kCodeChunk + g * 2 + f_c];
        }
        if constexpr (kQ5Kt) {
          f_qh[0] = blk[1];
          f_qh[1] = blk[2];
        }
      }
      const int sb32 = (kb0 % 8) + f_c;
      if constexpr (kPair) {
        const int g = sb32 / 2;
        f_codes = g == 0 ? code_cache[0] : g == 1 ? code_cache[1] : g == 2 ? code_cache[2] : code_cache[3];
      } else {
        f_codes = blk[kCodeChunk + (sb32 / 2) * 2 + f_c];
      }
      f_sb32 = sb32;
      if constexpr (kQ5Kt) {
        const uint32_t qh[8] = {f_qh[0].x, f_qh[0].y, f_qh[0].z, f_qh[0].w,
                                f_qh[1].x, f_qh[1].y, f_qh[1].z, f_qh[1].w};
        uint32_t high = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) high |= gather_bit(qh[i], sb32) << (4 * i);
        f_high = high;
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      a_data[i] = a_src[i] ? *reinterpret_cast<const uint4*>(a_src[i] + kb0 * 32)
                           : make_uint4(0u, 0u, 0u, 0u);
  };

  const auto commit_stage = [&]() {
    const int row = tid >> 1;
    uint32_t scale_bias = 0;
    if constexpr (kQ8) {
      s_codes[swizzle(row, 2 * f_c)] = f_codes;
      s_codes[swizzle(row, 2 * f_c + 1)] = f_codes_hi;
      scale_bias = f_live ? f_dm : 0U;  // half2 (d, 0)
    } else {
      s_codes[swizzle(row, f_c)] = f_codes;
    }
    if constexpr (kQ5Kt || kQ5) s_high[f_c * BM + row] = f_high;
    if constexpr (kQ8) {
    } else if constexpr (kQ5) {
      const __half2 dm = __builtin_bit_cast(__half2, f_dm);
      const float d = f_live ? __low2float(dm) : 0.0f;
      const float mn = f_live ? __high2float(dm) : 0.0f;
      scale_bias = __builtin_bit_cast(uint32_t, __floats2half2_rn(d, mn));
    } else {
      const int sb32 = f_sb32;
      uint32_t sc, mn;
      if (sb32 < 4) {
        sc = header_byte(f_header, 4 + sb32) & 0x3FU;
        mn = header_byte(f_header, 8 + sb32) & 0x3FU;
      } else {
        sc = (header_byte(f_header, 8 + sb32) & 0x0FU) | ((header_byte(f_header, sb32) >> 6U) << 4U);
        mn = (header_byte(f_header, 8 + sb32) >> 4U) | ((header_byte(f_header, 4 + sb32) >> 6U) << 4U);
      }
      const __half2 dm = __builtin_bit_cast(__half2, f_header.x);
      const float scale = f_live ? __low2float(dm) * (float)sc : 0.0f;
      const float offset = f_live ? __high2float(dm) * (float)mn : 0.0f;
      scale_bias = __builtin_bit_cast(uint32_t, __floats2half2_rn(scale, -offset));
    }
    s_scale[f_c * BM + row] = scale_bias;
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      if (a_slot[i] >= 0) s_act[a_slot[i]] = a_data[i];
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) acc[j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

  const __half2 magic = __floats2half2_rn(kQ8 ? -1152.0f : -1024.0f, kQ8 ? -1152.0f : -1024.0f);
  const auto compute_stage = [&]() {
    const int row = wave_id * 16 + sub_lane;
    uint4 raw[BK];
    if constexpr (!kQ8) {
#pragma unroll
      for (int c = 0; c < BK; ++c) raw[c] = s_codes[swizzle(row, c)];
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      const __half2 sb = __builtin_bit_cast(__half2, s_scale[kb * BM + row]);
      const __half2 scale2 = __low2half2(sb);
      const __half2 bias2 = __high2half2(sb);
      uint32_t nib[8];
      if constexpr (kQ8) {
        const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
        const uint4 c1 = s_codes[swizzle(row, 2 * kb + 1)];
        const uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) nib[i] = words[i] ^ 0x80808080U;
      } else if constexpr (kQ5) {
        const uint4 r = raw[kb];
        const uint32_t high = s_high[kb * BM + row];
        const uint32_t words[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          nib[i] = (words[i] & 0x0F0F0F0FU) | spread_high((high >> (4 * i)) & 0xFU);
          nib[4 + i] = ((words[i] >> 4U) & 0x0F0F0F0FU) | spread_high((high >> (16 + 4 * i)) & 0xFU);
        }
      } else {
        const unsigned shift = 4U * (unsigned)kb;
        const uint32_t words[8] = {raw[0].x, raw[0].y, raw[0].z, raw[0].w,
                                   raw[1].x, raw[1].y, raw[1].z, raw[1].w};
        const uint32_t high = kQ5Kt ? s_high[kb * BM + row] : 0U;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          nib[i] = (words[i] >> shift) & 0x0F0F0F0FU;
          if constexpr (kQ5Kt) nib[i] |= spread_high((high >> (4 * i)) & 0xFU);
        }
      }
      __half2 h[16];
#pragma unroll
      for (int i = 0; i < 8; ++i) codes_to_halves(nib[i], magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
      v16h a_lo, a_hi;
      __builtin_memcpy(&a_lo, &h[0], 32);
      __builtin_memcpy(&a_hi, &h[8], 32);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        asm volatile("" ::: "memory");  // keep one token tile's fragments live
        if (j >= live_tok_tiles) continue;
        const uint4* frag = s_act + (kb * 4) * kActStride + j * 16 + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) b[q] = frag[q * kActStride];
        v16h b_lo, b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
        acc[j] = wmma(a_lo, b_lo, acc[j]);
        acc[j] = wmma(a_hi, b_hi, acc[j]);
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) fetch_stage(kb0 + BK);
    compute_stage();
    __syncthreads();
  }

  if constexpr (kPair) {
    // Waves 0-3 hold gate rows, 4-7 the matching up rows: pair them in LDS.
    constexpr unsigned stride = 18;
    constexpr unsigned plane = 16 * stride;
    static_assert(8 * plane * sizeof(float) <= kLdsBytes, "pair epilogue LDS");
    float* base = reinterpret_cast<float*>(lds);
    float* scratch = base + wave_id * plane;
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      if (j >= live_tok_tiles) break;  // uniform across the block
#pragma unroll
      for (int l = 0; l < 8; ++l) scratch[sub_lane * stride + 2 * l + half_id] = acc[j][l];
      __syncthreads();
#pragma unroll
      for (int unit = 0; unit < 2; ++unit) {
        const int flat = (unit * 256 + tid) * 2;
        const int t = j * 16 + flat / kRows;
        const int r = flat % kRows;
        if (t < nrows && r_block + r < m) {
          const int idx = (r / 16) * plane + (flat / kRows) * stride + r % 16;
          __half vals[2];
#pragma unroll
          for (int v = 0; v < 2; ++v) {
            const float g = base[idx + v];
            const float u = base[idx + v + 4 * plane];
            vals[v] = __float2half(u * g * sigmoidf_(g));
          }
          const size_t o = (size_t)(slot0 + t) * m + r_block + r;
          *reinterpret_cast<__half2*>(out_half + o) = __halves2half2(vals[0], vals[1]);
        }
      }
      __syncthreads();
    }
    return;
  }

  // Down: transpose each 16x16 tile through LDS; one 64 B row segment per slot.
  float* tile_scratch = reinterpret_cast<float*>(lds) + wave_id * 256;
  const int r0 = r_block + wave_id * 16;
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    if (j >= live_tok_tiles) break;
#pragma unroll
    for (int l = 0; l < 8; ++l) tile_scratch[sub_lane * 16 + 2 * l + half_id] = acc[j][l];
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int flat = s * 32 + lane_id;
      const int t = j * 16 + (flat >> 4);
      const int r = r0 + (flat & 15);
      if (t < nrows && r < m) out[(size_t)(slot0 + t) * m + r] = tile_scratch[flat];
    }
    __builtin_amdgcn_wave_barrier();
  }
}

// F32 [P][d] -> F16 [P][d] (d % 4 == 0).
__global__ void k_f32_to_f16_v4(const float* __restrict__ x, __half* __restrict__ y, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const float4 v = reinterpret_cast<const float4*>(x)[i];
  __half2* o = reinterpret_cast<__half2*>(y) + 2 * i;
  o[0] = __floats2half2_rn(v.x, v.y);
  o[1] = __floats2half2_rn(v.z, v.w);
}

// Sort-free routing for tiny P (decode, graph-capturable): one single-slot
// tile per (token, top-k slot) pair in token-major order, so pairids is the
// identity and the weighted reduce sums in expert-rank order. Only sensible
// when tokens share no experts (P == 1): shared experts would be re-read.
__global__ void k_moe_gg_route1(const int* __restrict__ topids, const float* __restrict__ topw,
                                int P, int k, int id_stride, int* __restrict__ tokidx,
                                float* __restrict__ pw, int* __restrict__ pairids,
                                MoeTile* __restrict__ tiles, int* __restrict__ ntiles) {
  const int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (p == 0) *ntiles = P * k;
  if (p >= P * k) return;
  const int t = p / k, s = p % k;
  tokidx[p] = t;
  pw[p] = topw[t * id_stride + s];
  pairids[p] = p;
  tiles[p] = MoeTile{topids[t * id_stride + s], p, 1};
}

// Host launchers: return false for an unsupported type combination.
// gate/up: hid[slot][m] F16 = up * silu(gate), x = F16 [P][k] via tokidx.
static bool moe_gg_up(int type, const uint8_t* gate, const uint8_t* up, const __half* x,
                      const int* tokidx, const MoeTile* tiles, const int* ntiles, int max_tiles,
                      __half* hid, int m, int k, hipStream_t st) {
  if (m % 64 || k % 256) return false;
  dim3 grid(m / 64, max_tiles);
  if (type == moegg::kQ4K)
    k_moe_gg<moegg::kQ4K, true><<<grid, 256, 0, st>>>(gate, up, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  else if (type == moegg::kQ5K)
    k_moe_gg<moegg::kQ5K, true><<<grid, 256, 0, st>>>(gate, up, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  else
    return false;
  return true;
}

// down: pairs[slot][m] F32 (unweighted), hid = F16 [slot][k].
static bool moe_gg_down(int type, const uint8_t* dn, const __half* hid, const MoeTile* tiles,
                        const int* ntiles, int max_tiles, float* pairs, int m, int k,
                        hipStream_t st) {
  if (m % 128 || k % 64) return false;
  dim3 grid(m / 128, max_tiles);
  if (type == moegg::kQ51)
    k_moe_gg<moegg::kQ51, false><<<grid, 256, 0, st>>>(dn, nullptr, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  else if (type == moegg::kQ80)
    k_moe_gg<moegg::kQ80, false><<<grid, 256, 0, st>>>(dn, nullptr, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  else
    return false;
  return true;
}

// --- GGUF routed experts, small-P GEMV path (decode / spec verify) ----------
//
// The WMMA kernels above compute 16 rows per slot tile; at P = 1 they waste
// 15/16 of the matrix work and run ~40% below the hgn q4cp decode GEMVs.
// These kernels read the raw llama.cpp blocks with scalar FP32 math (FP32
// activations, no F16 rounding), one "unit" per lane step:
//   Q4_K / Q5_K: 64 elements = sub-block pair j of a 256 super-block
//                (low nibbles -> sub-block 2j, high nibbles -> 2j+1),
//   Q5_1 / Q8_0: one 32-element block.
// gate/up: one wave per (token, slot, row m); lanes 0-15 walk the gate row,
//   lanes 16-31 the up row; hid[tok*kpt+s][m] = up * silu(gate) (FP32).
// down: LPR lanes per output row, all KPT expert partials kept in registers
//   and summed in expert-rank order with the routing weights -> out[tok][r]
//   overwritten (deterministic, no atomics, no memset), like
//   k_q4cp_gemv_gd_topk_h16.

namespace moegg {

__device__ __forceinline__ uint32_t byte_of(const uint32_t* w, int i) {
  return (w[i >> 2] >> (8 * (i & 3))) & 0xFFu;
}
// ggml get_scale_min_k4 on the 12 packed scale bytes (as 3 words)
__device__ __forceinline__ void scale_min_k4(int j, const uint32_t* q, float& s, float& m) {
  if (j < 4) {
    s = (float)(byte_of(q, j) & 63);
    m = (float)(byte_of(q, j + 4) & 63);
  } else {
    s = (float)((byte_of(q, j + 4) & 0xF) | ((byte_of(q, j - 4) >> 6) << 4));
    m = (float)((byte_of(q, j + 4) >> 4) | ((byte_of(q, j) >> 6) << 4));
  }
}

template <int T>
__host__ __device__ constexpr int unit_elems() {
  return (T == kQ4K || T == kQ5K) ? 64 : 32;
}

// One weight "unit" of a raw GGUF row, loaded once (load) and dotted with
// any number of activation vectors (dot) — the expert-dedup kernels reuse one
// load for every token routed to the same expert. unit_dot = load + dot.
template <int T> struct UnitW;
template <> struct UnitW<kQ4K> {
  float d, dmin, s0, m0, s1, m1;
  uint32_t qw[8];
  int off;  // element offset of the unit in the row
  __device__ __forceinline__ void load(const uint8_t* __restrict__ row, int u) {
    const int sb = u >> 2, j = u & 3;
    const uint8_t* blk = row + (size_t)sb * 144;
    const uint32_t dd = *(const uint32_t*)blk;
    d = __half2float(__ushort_as_half((unsigned short)(dd & 0xFFFF)));
    dmin = __half2float(__ushort_as_half((unsigned short)(dd >> 16)));
    uint32_t sc[3];
    sc[0] = *(const uint32_t*)(blk + 4);
    sc[1] = *(const uint32_t*)(blk + 8);
    sc[2] = *(const uint32_t*)(blk + 12);
    scale_min_k4(2 * j, sc, s0, m0);
    scale_min_k4(2 * j + 1, sc, s1, m1);
    const uint4* qp = (const uint4*)(blk + 16 + 32 * j);
    const uint4 qa = qp[0], qb = qp[1];
    qw[0] = qa.x; qw[1] = qa.y; qw[2] = qa.z; qw[3] = qa.w;
    qw[4] = qb.x; qw[5] = qb.y; qw[6] = qb.z; qw[7] = qb.w;
    off = sb * 256 + 64 * j;
  }
  __device__ __forceinline__ float dot(const float* __restrict__ x) const {
    const float* x0 = x + off;
    float dl = 0.f, dh = 0.f, sl = 0.f, sh = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 xl = *(const float4*)(x0 + 4 * i);
      const float4 xh = *(const float4*)(x0 + 32 + 4 * i);
      const float xlv[4] = {xl.x, xl.y, xl.z, xl.w}, xhv[4] = {xh.x, xh.y, xh.z, xh.w};
#pragma unroll
      for (int b = 0; b < 4; b++) {
        const uint32_t q = (qw[i] >> (8 * b)) & 0xFFu;
        dl += (float)(q & 0xF) * xlv[b];
        dh += (float)(q >> 4) * xhv[b];
        sl += xlv[b];
        sh += xhv[b];
      }
    }
    return d * (s0 * dl + s1 * dh) - dmin * (m0 * sl + m1 * sh);
  }
};
template <> struct UnitW<kQ5K> {
  float d, dmin, s0, m0, s1, m1;
  uint32_t qw[8], hw[8];
  int off, j;
  __device__ __forceinline__ void load(const uint8_t* __restrict__ row, int u) {
    const int sb = u >> 2;
    j = u & 3;
    const uint8_t* blk = row + (size_t)sb * 176;
    const uint32_t dd = *(const uint32_t*)blk;
    d = __half2float(__ushort_as_half((unsigned short)(dd & 0xFFFF)));
    dmin = __half2float(__ushort_as_half((unsigned short)(dd >> 16)));
    uint32_t sc[3];
    sc[0] = *(const uint32_t*)(blk + 4);
    sc[1] = *(const uint32_t*)(blk + 8);
    sc[2] = *(const uint32_t*)(blk + 12);
    scale_min_k4(2 * j, sc, s0, m0);
    scale_min_k4(2 * j + 1, sc, s1, m1);
    const uint4* qp = (const uint4*)(blk + 48 + 32 * j);
    const uint4 qa = qp[0], qb = qp[1];
    qw[0] = qa.x; qw[1] = qa.y; qw[2] = qa.z; qw[3] = qa.w;
    qw[4] = qb.x; qw[5] = qb.y; qw[6] = qb.z; qw[7] = qb.w;
    const uint4* hp = (const uint4*)(blk + 16);
    const uint4 ha = hp[0], hb = hp[1];
    hw[0] = ha.x; hw[1] = ha.y; hw[2] = ha.z; hw[3] = ha.w;
    hw[4] = hb.x; hw[5] = hb.y; hw[6] = hb.z; hw[7] = hb.w;
    off = sb * 256 + 64 * j;
  }
  __device__ __forceinline__ float dot(const float* __restrict__ x) const {
    const float* x0 = x + off;
    float dl = 0.f, dh = 0.f, sl = 0.f, sh = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 xl = *(const float4*)(x0 + 4 * i);
      const float4 xh = *(const float4*)(x0 + 32 + 4 * i);
      const float xlv[4] = {xl.x, xl.y, xl.z, xl.w}, xhv[4] = {xh.x, xh.y, xh.z, xh.w};
#pragma unroll
      for (int b = 0; b < 4; b++) {
        const uint32_t q = (qw[i] >> (8 * b)) & 0xFFu;
        const uint32_t h = (hw[i] >> (8 * b)) & 0xFFu;
        const uint32_t lo = (q & 0xF) | (((h >> (2 * j)) & 1u) << 4);
        const uint32_t hi = (q >> 4) | (((h >> (2 * j + 1)) & 1u) << 4);
        dl += (float)lo * xlv[b];
        dh += (float)hi * xhv[b];
        sl += xlv[b];
        sh += xhv[b];
      }
    }
    return d * (s0 * dl + s1 * dh) - dmin * (m0 * sl + m1 * sh);
  }
};
template <> struct UnitW<kQ51> {
  float d, mn;
  uint32_t qh, qw[4];
  int off;
  __device__ __forceinline__ void load(const uint8_t* __restrict__ row, int u) {
    const uint8_t* blk = row + (size_t)u * 24;
    const uint32_t dm = *(const uint32_t*)blk;
    d = __half2float(__ushort_as_half((unsigned short)(dm & 0xFFFF)));
    mn = __half2float(__ushort_as_half((unsigned short)(dm >> 16)));
    qh = *(const uint32_t*)(blk + 4);
    const uint2 qa = *(const uint2*)(blk + 8), qb = *(const uint2*)(blk + 16);
    qw[0] = qa.x; qw[1] = qa.y; qw[2] = qb.x; qw[3] = qb.y;
    off = u * 32;
  }
  __device__ __forceinline__ float dot(const float* __restrict__ x) const {
    const float* x0 = x + off;
    float dq = 0.f, sx = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float4 xl = *(const float4*)(x0 + 4 * i);
      const float4 xh = *(const float4*)(x0 + 16 + 4 * i);
      const float xlv[4] = {xl.x, xl.y, xl.z, xl.w}, xhv[4] = {xh.x, xh.y, xh.z, xh.w};
#pragma unroll
      for (int b = 0; b < 4; b++) {
        const int e = 4 * i + b;
        const uint32_t q = (qw[i] >> (8 * b)) & 0xFFu;
        const uint32_t lo = (q & 0xF) | (((qh >> e) & 1u) << 4);
        const uint32_t hi = (q >> 4) | (((qh >> (e + 16)) & 1u) << 4);
        dq += (float)lo * xlv[b] + (float)hi * xhv[b];
        sx += xlv[b] + xhv[b];
      }
    }
    return d * dq + mn * sx;
  }
};
template <> struct UnitW<kQ80> {  // 34-byte blocks are only 2-byte aligned
  float d;
  uint32_t w[8];
  int off;
  __device__ __forceinline__ void load(const uint8_t* __restrict__ row, int u) {
    const uint16_t* blk = (const uint16_t*)(row + (size_t)u * 34);
    d = __half2float(__ushort_as_half(blk[0]));
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = (uint32_t)blk[1 + 2 * i] | ((uint32_t)blk[2 + 2 * i] << 16);
    off = u * 32;
  }
  __device__ __forceinline__ float dot(const float* __restrict__ x) const {
    const float* x0 = x + off;
    float dq = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 xv = *(const float4*)(x0 + 4 * i);
      dq += (float)(int8_t)(w[i] & 0xFF) * xv.x + (float)(int8_t)((w[i] >> 8) & 0xFF) * xv.y +
            (float)(int8_t)((w[i] >> 16) & 0xFF) * xv.z + (float)(int8_t)(w[i] >> 24) * xv.w;
    }
    return d * dq;
  }
};

// dot(row unit u, x[unit u elements]) for one weight row in raw GGUF layout
template <int T>
__device__ __forceinline__ float unit_dot(const uint8_t* __restrict__ row, int u,
                                          const float* __restrict__ x) {
  UnitW<T> w;
  w.load(row, u);
  return w.dot(x);
}

}  // namespace moegg

// x F32 [P][k]; ids/topw rows id_stride wide; hid F32 [P*kpt][m].
template <int T>
__launch_bounds__(256) __global__
void k_moe_gg_gemv_gu(const uint8_t* __restrict__ wg, const uint8_t* __restrict__ wu,
                      const float* __restrict__ x, const int* __restrict__ ids,
                      float* __restrict__ hid, int m, int k, int kpt, int id_stride, int P) {
  const int gw = (int)((blockIdx.x * blockDim.x + threadIdx.x) / warpSize);
  if (gw >= P * kpt * m) return;
  const int lane = threadIdx.x % warpSize, half = lane >> 4, l16 = lane & 15;
  const int r = gw % m, slot = gw / m, tok = slot / kpt, s = slot - tok * kpt;
  const int e = ids[tok * id_stride + s];
  const size_t rb = moegg::row_bytes<T>((size_t)k);
  const uint8_t* w = (half ? wu : wg) + ((size_t)e * m + r) * rb;
  const float* xt = x + (size_t)tok * k;
  const int nu = k / moegg::unit_elems<T>();
  float acc = 0.f;
  for (int u = l16; u < nu; u += 16) acc += moegg::unit_dot<T>(w, u, xt);
#pragma unroll
  for (int off = 8; off > 0; off >>= 1) acc += __shfl_down_sync(~0ull, acc, off, 16);
  const float up = __shfl_sync(~0ull, acc, 16);
  if (lane == 0) hid[(size_t)slot * m + r] = up * acc * moegg::sigmoidf_(acc);
}

template <int T, int LPR, int KPT>
__launch_bounds__(256) __global__
void k_moe_gg_gemv_dn(const uint8_t* __restrict__ wd, const float* __restrict__ hid,
                      const int* __restrict__ ids, const float* __restrict__ ws,
                      float* __restrict__ out, int m, int k, int id_stride, int P) {
  constexpr int RPW = 32 / LPR;  // rows per wave (wave32)
  const int gw = (int)((blockIdx.x * blockDim.x + threadIdx.x) / warpSize);
  const int rpt = (m + RPW - 1) / RPW;  // row groups per token
  if (gw >= P * rpt) return;
  const int lane = threadIdx.x % warpSize, sub = lane / LPR, ll = lane % LPR;
  const int tok = gw / rpt, r = (gw % rpt) * RPW + sub;
  const bool live = r < m;
  const size_t rb = moegg::row_bytes<T>((size_t)k);
  const int nu = k / moegg::unit_elems<T>();
  float acc[KPT];
#pragma unroll
  for (int s = 0; s < KPT; s++) {
    acc[s] = 0.f;
    const int e = ids[tok * id_stride + s];
    const uint8_t* w = wd + ((size_t)e * m + (live ? r : 0)) * rb;
    const float* xs = hid + ((size_t)tok * KPT + s) * k;
    for (int u = ll; u < nu; u += LPR) acc[s] += moegg::unit_dot<T>(w, u, xs);
  }
#pragma unroll
  for (int s = 0; s < KPT; s++)
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1)
      acc[s] += __shfl_down_sync(~0ull, acc[s], off, LPR);
  if (ll == 0 && live) {
    float y = 0.f;
#pragma unroll
    for (int s = 0; s < KPT; s++) y += ws[tok * id_stride + s] * acc[s];
    out[(size_t)tok * m + r] = y;
  }
}

// --- Expert dedup for 2 <= P <= 8 (spec verify): tokens of one step share
// many experts; read each unique expert once for all its (token, slot) pairs.
// meta (ints): [0] unique count U; [1+u] expert; [1+NM+u] member count;
// [1+2NM+8u+j] member j = tok*kpt+s (ascending). NM = 8*kpt max.
namespace moegg {
constexpr int kUqMaxP = 8, kUqNM = kUqMaxP * 10;
constexpr int kUqInts = 1 + 2 * kUqNM + kUqMaxP * kUqNM;
}  // namespace moegg
__launch_bounds__(128) __global__
void k_moe_gg_uniq(const int* __restrict__ ids, int* __restrict__ meta, int P, int kpt,
                   int id_stride) {
  constexpr int NM = moegg::kUqNM;
  __shared__ int e[NM], lead[NM];
  const int n = P * kpt;
  for (int i = threadIdx.x; i < n; i += blockDim.x) e[i] = ids[(i / kpt) * id_stride + i % kpt];
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int l = 1;
    for (int j = 0; j < i; j++)
      if (e[j] == e[i]) { l = 0; break; }
    lead[i] = l;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (!lead[i]) continue;
    int u = 0;
    for (int j = 0; j < i; j++) u += lead[j];
    meta[1 + u] = e[i];
    int c = 0;
    for (int j = i; j < n; j++)
      if (e[j] == e[i]) meta[1 + 2 * NM + moegg::kUqMaxP * u + c++] = j;
    meta[1 + NM + u] = c;
  }
  if (threadIdx.x == 0) {
    int U = 0;
    for (int i = 0; i < n; i++) U += lead[i];
    meta[0] = U;
  }
}

// gate/up: one wave per (unique expert u, row r); hid[member][r] as in
// k_moe_gg_gemv_gu (same per-lane unit order -> same values).
template <int T>
__launch_bounds__(256) __global__
void k_moe_gg_gemv_gu_uq(const uint8_t* __restrict__ wg, const uint8_t* __restrict__ wu,
                         const float* __restrict__ x, const int* __restrict__ meta,
                         float* __restrict__ hid, int m, int k, int kpt) {
  constexpr int NM = moegg::kUqNM, MP = moegg::kUqMaxP;
  const int gw = (int)((blockIdx.x * blockDim.x + threadIdx.x) / warpSize);
  const int u = gw / m, r = gw - u * m;
  if (u >= meta[0]) return;
  const int e = meta[1 + u], cnt = meta[1 + NM + u];
  const int* mem = meta + 1 + 2 * NM + MP * u;
  const int lane = threadIdx.x % warpSize, half = lane >> 4, l16 = lane & 15;
  const size_t rb = moegg::row_bytes<T>((size_t)k);
  const uint8_t* w = (half ? wu : wg) + ((size_t)e * m + r) * rb;
  int sl[MP];
  const float* xp[MP];
  float acc[MP];
#pragma unroll
  for (int j = 0; j < MP; j++) {
    sl[j] = mem[j < cnt ? j : 0];
    xp[j] = x + (size_t)(sl[j] / kpt) * k;
    acc[j] = 0.f;
  }
  const int nu = k / moegg::unit_elems<T>();
  for (int uu = l16; uu < nu; uu += 16) {
    moegg::UnitW<T> wv;
    wv.load(w, uu);
#pragma unroll
    for (int j = 0; j < MP; j++)
      if (j < cnt) acc[j] += wv.dot(xp[j]);
  }
#pragma unroll
  for (int j = 0; j < MP; j++) {
    if (j >= cnt) break;  // cnt is wave-uniform
    float a = acc[j];
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) a += __shfl_down_sync(~0ull, a, off, 16);
    const float up = __shfl_sync(~0ull, a, 16);
    if (lane == 0) hid[(size_t)sl[j] * m + r] = up * a * moegg::sigmoidf_(a);
  }
}

// down: one wave per (unique expert, RPW rows); unweighted pairs[member][r].
template <int T, int LPR>
__launch_bounds__(256) __global__
void k_moe_gg_gemv_dn_uq(const uint8_t* __restrict__ wd, const float* __restrict__ hid,
                         const int* __restrict__ meta, float* __restrict__ pairs, int m,
                         int k) {
  constexpr int NM = moegg::kUqNM, MP = moegg::kUqMaxP, RPW = 32 / LPR;
  const int gw = (int)((blockIdx.x * blockDim.x + threadIdx.x) / warpSize);
  const int rpt = (m + RPW - 1) / RPW;
  const int u = gw / rpt;
  if (u >= meta[0]) return;
  const int lane = threadIdx.x % warpSize, sub = lane / LPR, ll = lane % LPR;
  const int r = (gw - u * rpt) * RPW + sub;
  const bool live = r < m;
  const int e = meta[1 + u], cnt = meta[1 + NM + u];
  const int* mem = meta + 1 + 2 * NM + MP * u;
  const size_t rb = moegg::row_bytes<T>((size_t)k);
  const uint8_t* w = wd + ((size_t)e * m + (live ? r : 0)) * rb;
  int sl[MP];
  float acc[MP];
#pragma unroll
  for (int j = 0; j < MP; j++) {
    sl[j] = mem[j < cnt ? j : 0];
    acc[j] = 0.f;
  }
  const int nu = k / moegg::unit_elems<T>();
  for (int uu = ll; uu < nu; uu += LPR) {
    moegg::UnitW<T> wv;
    wv.load(w, uu);
#pragma unroll
    for (int j = 0; j < MP; j++)
      if (j < cnt) acc[j] += wv.dot(hid + (size_t)sl[j] * k);
  }
#pragma unroll
  for (int j = 0; j < MP; j++) {
    if (j >= cnt) break;
    float a = acc[j];
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1) a += __shfl_down_sync(~0ull, a, off, LPR);
    if (ll == 0 && live) pairs[(size_t)sl[j] * m + r] = a;
  }
}
// out[tok][r] = sum_s ws[tok][s] * pairs[tok*KPT+s][r], expert-rank order
// (same order as k_moe_gg_gemv_dn -> same values).
template <int KPT>
__global__ void k_moe_gg_pairs_sum(const float* __restrict__ pairs, const float* __restrict__ ws,
                                   float* __restrict__ out, int m, int id_stride) {
  const int tok = blockIdx.y, r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= m) return;
  float y = 0.f;
#pragma unroll
  for (int s = 0; s < KPT; s++) y += ws[tok * id_stride + s] * pairs[((size_t)tok * KPT + s) * m + r];
  out[(size_t)tok * m + r] = y;
}

#ifndef MOE_GG_DN_LPR
#define MOE_GG_DN_LPR 8
#endif
// Small-P GEMV path: x F32 [P][d] -> out F32 [P][d] (overwritten), hid =
// F32 scratch [P*10][mid]. topk must be 10. false = unsupported types.
static bool moe_gg_gemv(int tgu, int tdn, const uint8_t* gate, const uint8_t* up,
                        const uint8_t* down, const float* x, const int* ids, const float* ws,
                        float* hid, float* out, int P, int d, int mid, int id_stride,
                        hipStream_t st, int* uq_meta = nullptr, float* uq_pairs = nullptr) {
  constexpr int KPT = 10;
  if (d % 256 || mid % 32) return false;
  if (uq_meta && uq_pairs && P >= 2 && P <= moegg::kUqMaxP) {
    // expert dedup (meta/pairs scratch: kUqInts ints, P*KPT*d floats)
    if ((tgu != moegg::kQ4K && tgu != moegg::kQ5K) || (tdn != moegg::kQ51 && tdn != moegg::kQ80))
      return false;
    k_moe_gg_uniq<<<1, 128, 0, st>>>(ids, uq_meta, P, KPT, id_stride);
    const unsigned gb = (unsigned)(((size_t)P * KPT * mid + 7) / 8);
    if (tgu == moegg::kQ4K)
      k_moe_gg_gemv_gu_uq<moegg::kQ4K><<<gb, 256, 0, st>>>(gate, up, x, uq_meta, hid, mid, d, KPT);
    else
      k_moe_gg_gemv_gu_uq<moegg::kQ5K><<<gb, 256, 0, st>>>(gate, up, x, uq_meta, hid, mid, d, KPT);
    constexpr int LPR = MOE_GG_DN_LPR, RPW = 32 / LPR;
    const unsigned db = (unsigned)(((size_t)P * KPT * ((d + RPW - 1) / RPW) + 7) / 8);
    if (tdn == moegg::kQ51)
      k_moe_gg_gemv_dn_uq<moegg::kQ51, LPR><<<db, 256, 0, st>>>(down, hid, uq_meta, uq_pairs, d, mid);
    else
      k_moe_gg_gemv_dn_uq<moegg::kQ80, LPR><<<db, 256, 0, st>>>(down, hid, uq_meta, uq_pairs, d, mid);
    k_moe_gg_pairs_sum<KPT><<<dim3((d + 255) / 256, P), 256, 0, st>>>(uq_pairs, ws, out, d, id_stride);
    return true;
  }
  const unsigned gu_blocks = (unsigned)(((size_t)P * KPT * mid + 7) / 8);
  if (tgu == moegg::kQ4K)
    k_moe_gg_gemv_gu<moegg::kQ4K><<<gu_blocks, 256, 0, st>>>(gate, up, x, ids, hid, mid, d, KPT, id_stride, P);
  else if (tgu == moegg::kQ5K)
    k_moe_gg_gemv_gu<moegg::kQ5K><<<gu_blocks, 256, 0, st>>>(gate, up, x, ids, hid, mid, d, KPT, id_stride, P);
  else
    return false;
  constexpr int LPR = MOE_GG_DN_LPR, RPW = 32 / LPR;
  const unsigned dn_blocks = (unsigned)(((size_t)P * ((d + RPW - 1) / RPW) + 7) / 8);
  if (tdn == moegg::kQ51)
    k_moe_gg_gemv_dn<moegg::kQ51, LPR, KPT><<<dn_blocks, 256, 0, st>>>(down, hid, ids, ws, out, d, mid, id_stride, P);
  else if (tdn == moegg::kQ80)
    k_moe_gg_gemv_dn<moegg::kQ80, LPR, KPT><<<dn_blocks, 256, 0, st>>>(down, hid, ids, ws, out, d, mid, id_stride, P);
  else
    return false;
  return true;
}

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

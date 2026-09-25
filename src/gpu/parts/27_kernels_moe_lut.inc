// --- LUT-decoded 4-bit routed-expert F16 WMMA GEMM -------------------------
//
// Same pipeline / tiling / epilogues as k_moe_gg (26_kernels_moe_gguf.inc),
// for the 4-bit formats whose value is  lut[nibble] * scale  with one scale
// per 32-element block:
//   hgn q4cp (dtype 5): per-tensor 16 x fp32 codebook, fp16 scale per 32
//     (codes and scales in separate planes, byte b = elements 2b | 2b+1 << 4);
//   GGUF IQ4_NL (18 B / 32: fp16 d, qs[16]; element j = qs[j] & 15,
//     j + 16 = qs[j] >> 4) and IQ4_XS (136 B / 256: d, scales_h, scales_l[4],
//     qs[128]; 6-bit sub-block scale d * (ls - 32)), lut = kvalues_iq4nl.
//
// Each stage stages 16 code bytes per (row, K block) in "pair order" (byte b =
// elements 2b, 2b+1; IQ blocks are reordered with two v_perm on commit) plus a
// half2 (scale, scale). Decode = 16 lookups in an LDS pair table
// s_cbp[byte] = half2(lut[byte & 15], lut[byte >> 4]) and one packed multiply.
// Numerics: w = f16(f16(lut) * f16(scale)), F32 accumulate — like the GGUF path
// (IQ: lut / 256 and scale * 256, exact rescale keeping scales out of fp16 subnormals).
//
//   kPair (gate/up): hid F16 [slot][m] = up * silu(gate), grid = (m / 64, tiles)
//   !kPair (down):   pairs F32 [slot][m],                grid = (m / 128, tiles)

namespace moelut {

constexpr int kQ4CP = 100, kIQ4NL = 20, kIQ4XS = 23;  // IQ: gguf::Type values

__constant__ int8_t c_iq4_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                        1, 13, 25, 38, 53, 69, 89, 113};

// Expert-stacked weights. Row index of (expert e, row r) inside `w` / `w_up`:
//   e * e_rows + r            (gate, down, IQ up)
//   e * e_rows + up_off + r   (kPair upper half: q4cp fused gate_up -> up_off = mid)
struct LutW {
  const uint8_t* w;        // q4cp: code plane (tensor data + 64); IQ: raw blocks
  const uint8_t* w_up;     // kPair up rows (q4cp fused gate_up: == w)
  const uint8_t* sc;       // q4cp scale plane (fp16 per 32, sstride bytes / row)
  const uint8_t* sc_up;    // q4cp: == sc for fused gate_up
  const float* cb;         // q4cp codebook (16 x fp32, tensor data)
  int sstride;             // q4cp scale row stride in bytes (16 B multiple, rows padded)
  int e_rows, up_off;
};

template <int T>
__host__ __device__ constexpr size_t lut_row_bytes(size_t k) {
  return T == kQ4CP ? k / 2 : T == kIQ4NL ? (k / 32) * 18 : (k / 256) * 136;
}

// IQ block codes (qs[0..15]: low nibbles = elements 0..15, high = 16..31)
// -> pair order (byte b = elements 2b | 2b+1 << 4).
__device__ __forceinline__ uint32_t pair_lo(uint32_t w) {
  const uint32_t l = w & 0x0F0F0F0FU;
  return l | (l >> 4);  // bytes 0, 2 = (q0 | q1 << 4), (q2 | q3 << 4)
}
__device__ __forceinline__ uint32_t pair_hi(uint32_t w) {
  const uint32_t h = (w >> 4) & 0x0F0F0F0FU;
  return h | (h >> 4);
}
__device__ __forceinline__ uint4 iq_to_pairs(uint4 q) {
  constexpr uint32_t sel = 0x06040200U;  // bytes 0, 2 of b, then 0, 2 of a
  return make_uint4(__builtin_amdgcn_perm(pair_lo(q.y), pair_lo(q.x), sel),
                    __builtin_amdgcn_perm(pair_lo(q.w), pair_lo(q.z), sel),
                    __builtin_amdgcn_perm(pair_hi(q.y), pair_hi(q.x), sel),
                    __builtin_amdgcn_perm(pair_hi(q.w), pair_hi(q.z), sel));
}

}  // namespace moelut

template <int kLut, bool kPair, int BN = 64>
__launch_bounds__(256) __global__
    void k_moe_lut(moelut::LutW p, const __half* __restrict__ x, const int* __restrict__ tokidx,
                   const MoeTile* __restrict__ tiles, const int* __restrict__ ntiles,
                   float* __restrict__ out, __half* __restrict__ out_half, int m, int k) {
  using namespace moegg;
  using namespace moelut;
  constexpr int BM = 128, BK = 2;
  static_assert(BN % 16 == 0 && BN <= 64, "BN must match the MoeTile size (<= 64)");
  static_assert(kLut == kQ4CP || kLut == kIQ4NL || kLut == kIQ4XS, "LUT type");
  constexpr int kTokTiles = BN / 16;
  constexpr int kCodeBytes = BM * BK * 16;
  constexpr int kScaleBytes = BK * BM * 4;
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  constexpr int kStageBytes = kCodeBytes + kScaleBytes + kActBytes;
  constexpr int kLdsBytes = kStageBytes > 8 * 1024 ? kStageBytes : 8 * 1024;
  __shared__ __attribute__((aligned(16))) uint8_t lds[kLdsBytes];
  __shared__ uint32_t s_cbp[256];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_scale = reinterpret_cast<uint32_t*>(lds + kCodeBytes);
  auto* s_act = reinterpret_cast<uint4*>(lds + kCodeBytes + kScaleBytes);

  if ((int)blockIdx.y >= *ntiles) return;
  const int tid = threadIdx.x;
  {
    float lo, hi;
    if constexpr (kLut == kQ4CP) {
      lo = p.cb[tid & 15];
      hi = p.cb[tid >> 4];
    } else {
      lo = (float)c_iq4_values[tid & 15] * (1.0f / 256.0f);  // exact; scale carries * 256
      hi = (float)c_iq4_values[tid >> 4] * (1.0f / 256.0f);
    }
    s_cbp[tid] = __builtin_bit_cast(uint32_t, __floats2half2_rn(lo, hi));
  }  // visible after the first __syncthreads of the main loop

  const MoeTile tile = tiles[blockIdx.y];
  const int expert = tile.expert;
  const int slot0 = tile.first;
  const int nrows = tile.count;  // <= BN
  const int live_tok_tiles = (nrows + 15) / 16;
  const int num_kb = k / 32;
  const size_t rb = lut_row_bytes<kLut>((size_t)k);

  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  constexpr int kRows = kPair ? BM / 2 : BM;
  const int r_block = (int)blockIdx.x * kRows;

  // Weight fetch: thread = (row tid/2, K block kb0 + tid%2).
  const int f_c = tid & 1;
  const uint8_t* f_ptr;
  const uint8_t* f_sp = nullptr;
  bool f_live;
  {
    const bool upper = kPair && (tid >> 1) >= kRows;
    const int r = r_block + (kPair ? (tid >> 1) % kRows : (tid >> 1));
    f_live = r < m;
    const size_t row = (size_t)expert * p.e_rows + (upper ? p.up_off : 0) + (f_live ? r : (m - 1));
    f_ptr = (upper ? p.w_up : p.w) + row * rb;
    if constexpr (kLut == kQ4CP) f_sp = (upper ? p.sc_up : p.sc) + row * (size_t)p.sstride;
  }
  uint4 f_codes = make_uint4(0u, 0u, 0u, 0u);
  uint32_t f_s = 0;   // q4cp / IQ4_NL: fp16 scale bits; IQ4_XS: header word 0 (d | scales_h << 16)
  uint32_t f_sl = 0;  // IQ4_XS: scales_l[0..3]
  int f_ib = 0;       // IQ4_XS: sub-block
  uint4 f_scv = make_uint4(0u, 0u, 0u, 0u);  // q4cp: scales of K blocks kb0 & ~7 .. +7
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

  const auto swizzle = [](int row, int c) { return row * BK + (c ^ ((row >> 2) & 1)); };

  const auto fetch_stage = [&](int kb0) {
    const int kb = kb0 + f_c;
    if constexpr (kLut == kQ4CP) {
      f_codes = *reinterpret_cast<const uint4*>(f_ptr + (size_t)kb * 16);
      // 8 scales (16 B, rows padded to 16 B) per 4 stages, kept in registers.
      if (kb0 % 8 == 0) f_scv = *reinterpret_cast<const uint4*>(f_sp + (size_t)kb0 * 2);
      const int si = (kb0 % 8) + f_c;
      const uint32_t sw = si < 2 ? f_scv.x : si < 4 ? f_scv.y : si < 6 ? f_scv.z : f_scv.w;
      f_s = (si & 1) ? sw >> 16 : sw & 0xFFFFU;
    } else if constexpr (kLut == kIQ4NL) {
      // 18 B block at a 4 B (kb even) or 2 B (kb odd) boundary: 5 aligned
      // words cover it exactly (rows are 36-byte multiples, num_kb even).
      const uint8_t* blk = f_ptr + (size_t)kb * 18;
      const auto* wp = reinterpret_cast<const uint32_t*>(blk - 2 * f_c);
      const uint32_t w0 = wp[0], w1 = wp[1], w2 = wp[2], w3 = wp[3], w4 = wp[4];
      if (f_c) {
        f_s = w0 >> 16;
        f_codes = make_uint4(w1, w2, w3, w4);
      } else {
        f_s = w0 & 0xFFFFU;
        f_codes = make_uint4(__builtin_amdgcn_alignbyte(w1, w0, 2), __builtin_amdgcn_alignbyte(w2, w1, 2),
                             __builtin_amdgcn_alignbyte(w3, w2, 2), __builtin_amdgcn_alignbyte(w4, w3, 2));
      }
    } else {
      const uint8_t* blk = f_ptr + (size_t)(kb / 8) * 136;
      const uint2 hdr = *reinterpret_cast<const uint2*>(blk);
      const auto* qs = reinterpret_cast<const uint2*>(blk + 8 + 16 * (kb % 8));
      const uint2 q0 = qs[0], q1 = qs[1];
      f_s = hdr.x;
      f_sl = hdr.y;
      f_ib = kb % 8;
      f_codes = make_uint4(q0.x, q0.y, q1.x, q1.y);
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      a_data[i] = a_src[i] ? *reinterpret_cast<const uint4*>(a_src[i] + kb0 * 32)
                           : make_uint4(0u, 0u, 0u, 0u);
  };

  const auto commit_stage = [&]() {
    const int row = tid >> 1;
    uint32_t s16;
    if constexpr (kLut == kQ4CP) {
      s_codes[swizzle(row, f_c)] = f_codes;
      s16 = f_s;
    } else {
      s_codes[swizzle(row, f_c)] = iq_to_pairs(f_codes);
      // IQ scales are ~1e-5 (fp16 subnormal range): carry scale * 256 and
      // lut / 256 so the per-block scale keeps full fp16 precision.
      const float d = __half2float(__builtin_bit_cast(__half, (uint16_t)(f_s & 0xFFFFU))) * 256.0f;
      if constexpr (kLut == kIQ4NL) {
        s16 = __builtin_bit_cast(uint16_t, __float2half(d));
      } else {
        const int ib = f_ib;
        const int ls = (int)((f_sl >> (8 * (ib >> 1) + 4 * (ib & 1))) & 0xFU) |
                       (int)(((f_s >> (16 + 2 * ib)) & 3U) << 4);
        s16 = __builtin_bit_cast(uint16_t, __float2half(d * (float)(ls - 32)));
      }
    }
    s_scale[f_c * BM + row] = f_live ? (s16 | (s16 << 16)) : 0U;
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      if (a_slot[i] >= 0) s_act[a_slot[i]] = a_data[i];
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) acc[j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

  const auto compute_stage = [&]() {
    const int row = wave_id * 16 + sub_lane;
    uint4 raw[BK];
#pragma unroll
    for (int c = 0; c < BK; ++c) raw[c] = s_codes[swizzle(row, c)];
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      const __half2 scale2 = __builtin_bit_cast(__half2, s_scale[kb * BM + row]);
      const uint32_t words[4] = {raw[kb].x, raw[kb].y, raw[kb].z, raw[kb].w};
      __half2 h[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const uint32_t b = (words[i >> 2] >> (8 * (i & 3))) & 0xFFU;
        h[i] = __hmul2(__builtin_bit_cast(__half2, s_cbp[b]), scale2);
      }
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

namespace moelut {
static bool aligned(const void* ptr, uintptr_t a) { return ((uintptr_t)ptr & (a - 1)) == 0; }
static bool lut_ok(int type, const LutW& p, bool pair, int k) {
  if (k % (type == kIQ4XS ? 256 : 64)) return false;
  const uintptr_t a = type == kQ4CP ? 16 : type == kIQ4NL ? 4 : 8;
  if (!aligned(p.w, a) || (pair && !aligned(p.w_up, a))) return false;
  if (type == kQ4CP && (!p.cb || !aligned(p.sc, 16) || (pair && !aligned(p.sc_up, 16)) || p.sstride % 16 ||
                        p.sstride < (k / 256 + (k % 256 != 0)) * 16))
    return false;
  return type == kQ4CP || type == kIQ4NL || type == kIQ4XS;
}
}  // namespace moelut

// gate/up: hid[slot][m] F16 = up * silu(gate), x = F16 [P][k] via tokidx.
// type: moelut::kQ4CP / kIQ4NL / kIQ4XS. Returns false when unsupported.
static bool moe_lut_up(int type, const moelut::LutW& p, const __half* x, const int* tokidx,
                       const MoeTile* tiles, const int* ntiles, int max_tiles, __half* hid, int m,
                       int k, hipStream_t st) {
  if (m % 64 || !moelut::lut_ok(type, p, true, k)) return false;
  dim3 grid(m / 64, max_tiles);
  if (type == moelut::kQ4CP)
    k_moe_lut<moelut::kQ4CP, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  else if (type == moelut::kIQ4NL)
    k_moe_lut<moelut::kIQ4NL, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  else
    k_moe_lut<moelut::kIQ4XS, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  return true;
}

// down: pairs[slot][m] F32 (unweighted), hid = F16 [slot][k].
static bool moe_lut_down(int type, const moelut::LutW& p, const __half* hid, const MoeTile* tiles,
                         const int* ntiles, int max_tiles, float* pairs, int m, int k, hipStream_t st) {
  if (m % 128 || !moelut::lut_ok(type, p, false, k)) return false;
  dim3 grid(m / 128, max_tiles);
  if (type == moelut::kQ4CP)
    k_moe_lut<moelut::kQ4CP, false><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  else if (type == moelut::kIQ4NL)
    k_moe_lut<moelut::kIQ4NL, false><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  else
    k_moe_lut<moelut::kIQ4XS, false><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  return true;
}

// q4cp views of the hgn expert tensors (dtype 5; data = device pointer to
// [cb 64 B][codes][scales]). gate_up: [E][2*mid][d] fused, down: [E][d][mid].
static moelut::LutW lut_q4cp_view(const uint8_t* data, uint64_t rows, uint64_t cols,
                                  uint64_t scale_stride, int e_rows, int up_off) {
  moelut::LutW p{};
  p.cb = reinterpret_cast<const float*>(data);
  p.w = p.w_up = data + 64;
  p.sc = p.sc_up = data + 64 + rows * cols / 2;
  p.sstride = (int)scale_stride;
  p.e_rows = e_rows;
  p.up_off = up_off;
  return p;
}

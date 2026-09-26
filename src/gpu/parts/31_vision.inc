// ==================== Vision tower (ViT) — vision stage 4 ====================
// Ported from src/gpu/vit.cpp (stage 3, 27-layer parity vs the PyTorch
// reference, see HANDOVER_2026-09-13-VIT.md). Numeric path: fp16 GEMM I/O
// (weights bf16->fp16 host-side, exact at normal magnitudes) + split-fp16
// dual GEMM for patch_embed/fc1/fc2 inputs; residuals/LN/softmax fp32,
// eps=1e-6; MLP tanh-GELU, merger erf-GELU; non-causal WMMA flash attention.
// Loaded only via --vision-tower <sidecar.hgn>; without it the engine is
// bitwise identical to the text-only build (image requests are rejected).

__device__ inline uint16_t vit_f2h(float f) {  // fp32->fp16 RNE
  return __half_as_ushort(__float2half_rn(f));
}
__device__ inline float vit_h2f(uint16_t h) {
  return __half2float(__ushort_as_half(h));
}

__global__ void k_vit_f32_to_fp16(const float* x, uint16_t* y, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = vit_f2h(x[i]);
}

// split-fp16: x = hi + lo (lo keeps the residual hi rounded away; two-GEMM
// accumulation ~= fp32 input precision). Required for patch/fc1/fc2 inputs
// (stage-3 attribution: input rounding amplified over 27 layers).
__global__ void k_vit_f32_split_fp16(const float* x, uint16_t* hi, uint16_t* lo,
                                     uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float h = vit_h2f(vit_f2h(x[i]));
  hi[i] = vit_f2h(x[i]);
  lo[i] = vit_f2h(x[i] - h);
}

__global__ void k_vit_bias_add(float* y, const float* b, int P, int N) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (uint64_t)P * N) y[i] += b[i % N];
}

__global__ void k_vit_bias_resadd(float* h, const float* y, const float* b,
                                  int P, int N) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (uint64_t)P * N) h[i] += y[i] + b[i % N];
}

__global__ void k_vit_layernorm(const float* x, float* y, const float* w,
                                const float* b, int N, float eps) {
  int row = blockIdx.x;
  const float* xr = x + (uint64_t)row * N;
  float* yr = y + (uint64_t)row * N;
  __shared__ float red[256];
  float s = 0, ss = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float v = xr[i];
    s += v;
    ss += v * v;
  }
  red[threadIdx.x] = s;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  float mean = red[0] / N;
  __syncthreads();
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  float var = red[0] / N - mean * mean;
  float rstd = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    yr[i] = (xr[i] - mean) * rstd * w[i] + b[i];
}

// pos_embed gather + 4-tap bilinear weights, added into residual h (P x 1152)
__global__ void k_vit_pos_embed(float* h, const uint16_t* table,
                                const int* idx, const float* wts, int P) {
  int p = blockIdx.x;
  for (int d = threadIdx.x; d < 1152; d += blockDim.x) {
    float acc = 0;
    for (int t = 0; t < 4; t++)
      acc += wts[p * 4 + t] *
             vit_h2f(table[(uint64_t)idx[p * 4 + t] * 1152 + d]);
    h[(uint64_t)p * 1152 + d] += acc;
  }
}

// rope on q,k (fp32 qkv (P,3456)) -> fp16 q/k/v (P,1152). rotate_half with
// per-position (P,72) cos/sin broadcast across the 16 heads.
__global__ void k_vit_rope_qkv(const float* qkv, const float* cos_t,
                               const float* sin_t, uint16_t* qb, uint16_t* kb,
                               uint16_t* vb, int P) {
  int p = blockIdx.x;
  int h = blockIdx.y;
  int d = threadIdx.x;  // 36 threads
  if (d >= 36) return;
  const float* qr = qkv + (uint64_t)p * 3456 + h * 72;
  const float* kr = qr + 1152;
  const float* vr = qr + 2304;
  const float* cr = cos_t + (uint64_t)p * 72;
  const float* sr = sin_t + (uint64_t)p * 72;
  float c0 = cr[d], s0 = sr[d], c1 = cr[d + 36], s1 = sr[d + 36];
  float q0 = qr[d], q1 = qr[d + 36];
  float k0 = kr[d], k1 = kr[d + 36];
  uint64_t o = (uint64_t)p * 1152 + h * 72;
  qb[o + d] = vit_f2h(q0 * c0 - q1 * s0);
  qb[o + d + 36] = vit_f2h(q1 * c1 + q0 * s1);
  kb[o + d] = vit_f2h(k0 * c0 - k1 * s0);
  kb[o + d + 36] = vit_f2h(k1 * c1 + k0 * s1);
  vb[o + d] = vit_f2h(vr[d]);
  vb[o + d + 36] = vit_f2h(vr[d + 36]);
}

using vit_shortx16 = __attribute__((ext_vector_type(16))) short;
using vit_floatx8 = __attribute__((ext_vector_type(8))) float;
__device__ __forceinline__ vit_shortx16 vit_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  vit_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}

// WMMA fp16 flash attention (non-causal, single segment); gfx1151 fragment
// layout identical to k_qsa_wmma (see its comment). grid (ceil(P/128), 16
// heads), block 256 = 8 warps, warp w owns q rows [w*16,+16). Online softmax
// fp32; P quantized fp16 (stage-3: probs rounding negligible).
__global__ void __launch_bounds__(256)
k_vit_flash_wmma(const uint16_t* __restrict__ qg, const uint16_t* __restrict__ kg,
                 const uint16_t* __restrict__ vg, float* __restrict__ out,
                 int P, float scale) {
  __shared__ uint16_t Qs[128][88], Ks[64][88], Vs[64][88], Pbuf[128][72];
  const int r0 = blockIdx.x * 128;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31, w = tid >> 5;
  const int colbase = lane & 15;
  for (int i = tid; i < 128 * 44; i += 256) {
    int rr = i / 44, jj = i % 44;
    uint32_t val = 0;
    if (r0 + rr < P && jj < 36)
      val = ((const uint32_t*)(qg + (size_t)(r0 + rr) * 1152 + h * 72))[jj];
    ((uint32_t*)Qs[rr])[jj] = val;
  }
  __syncthreads();
  int qrowA = -1;
  if (lane < 16 && !(lane & 1)) qrowA = lane / 2;
  else if (lane >= 17 && (lane & 1)) qrowA = 8 + (lane - 17) / 2;
  float m_run[8], l_run[8];
  vit_floatx8 acc[5];
#pragma unroll
  for (int dc = 0; dc < 5; dc++) acc[dc] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int e = 0; e < 8; e++) {
    m_run[e] = -1e30f;
    l_run[e] = 0.f;
  }
  const int nkt = (P + 63) / 64;
  for (int kt = 0; kt < nkt; kt++) {
    const int c0 = kt * 64;
    __syncthreads();
    for (int i = tid; i < 64 * 44; i += 256) {
      int rr = i / 44, jj = i % 44;
      uint32_t kv = 0, vv = 0;
      if (c0 + rr < P && jj < 36) {
        kv = ((const uint32_t*)(kg + (size_t)(c0 + rr) * 1152 + h * 72))[jj];
        vv = ((const uint32_t*)(vg + (size_t)(c0 + rr) * 1152 + h * 72))[jj];
      }
      ((uint32_t*)Ks[rr])[jj] = kv;
      ((uint32_t*)Vs[rr])[jj] = vv;
    }
    __syncthreads();
    vit_floatx8 s[4];
#pragma unroll
    for (int f = 0; f < 4; f++) s[f] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int kk = 0; kk < 5; kk++) {
      vit_shortx16 a = {0};
      if (qrowA >= 0) a = vit_ld16(&Qs[w * 16 + qrowA][kk * 16]);
#pragma unroll
      for (int f = 0; f < 4; f++) {
        short bt[16];
        int krow = f * 16 + colbase;
#pragma unroll
        for (int k = 0; k < 16; k++)
          bt[k] = (short)Ks[krow][kk * 16 + k];
        vit_shortx16 b;
        memcpy(&b, bt, 32);
        s[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s[f]);
      }
    }
    float mx[8];
#pragma unroll
    for (int e = 0; e < 8; e++) mx[e] = -1e30f;
#pragma unroll
    for (int f = 0; f < 4; f++) {
      int col = c0 + f * 16 + colbase;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        float v = s[f][e] * scale;
        if (col >= P) v = -1e30f;
        s[f][e] = v;
        mx[e] = fmaxf(mx[e], v);
      }
    }
#pragma unroll
    for (int e = 0; e < 8; e++) {
#pragma unroll
      for (int o = 1; o <= 8; o <<= 1)
        mx[e] = fmaxf(mx[e], __shfl_xor(mx[e], o));
      float mnew = fmaxf(m_run[e], mx[e]);
      float corr = expf(m_run[e] - mnew);
      float es = 0.f;
#pragma unroll
      for (int f = 0; f < 4; f++) {
        float p = expf(s[f][e] - mnew);
        s[f][e] = p;
        es += p;
      }
#pragma unroll
      for (int o = 1; o <= 8; o <<= 1) es += __shfl_xor(es, o);
      l_run[e] = l_run[e] * corr + es;
      m_run[e] = mnew;
#pragma unroll
      for (int dc = 0; dc < 5; dc++) acc[dc][e] *= corr;
    }
#pragma unroll
    for (int f = 0; f < 4; f++)
#pragma unroll
      for (int e = 0; e < 8; e++)
        Pbuf[w * 16 + (lane < 16 ? e : 8 + e)][f * 16 + colbase] =
            vit_f2h(s[f][e]);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      vit_shortx16 a = {0};
      if (qrowA >= 0) a = vit_ld16(&Pbuf[w * 16 + qrowA][kk * 16]);
#pragma unroll
      for (int dc = 0; dc < 5; dc++) {
        short bt[16];
#pragma unroll
        for (int k = 0; k < 16; k++)
          bt[k] = (short)Vs[kk * 16 + k][dc * 16 + colbase];
        vit_shortx16 b;
        memcpy(&b, bt, 32);
        acc[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[dc]);
      }
    }
  }
#pragma unroll
  for (int e = 0; e < 8; e++) {
    int row = w * 16 + (lane < 16 ? e : 8 + e);
    if (r0 + row >= P) continue;
    float inv = 1.f / l_run[e];
#pragma unroll
    for (int dc = 0; dc < 5; dc++) {
      int col = dc * 16 + colbase;
      if (col < 72)
        out[(size_t)(r0 + row) * 1152 + h * 72 + col] = acc[dc][e] * inv;
    }
  }
}

// k-tile double-buffered k_vit_flash_wmma (2026-09-13, mirrors vit.cpp
// k_flash_wmma_db): next K/V tile staged in 24 uint4 registers issued at
// tile top (overlaps all compute); Ks written after S1, VsT after S2 (V
// stored transposed VsT[80][68] so the PV B-gather becomes wide
// ds_load_2addr_b64 instead of 320 scalar u16 loads). softmax expf ->
// __expf (v_exp_f32; probs quantize to fp16 downstream, three-image cmp
// verified). fp16 in / fp32 accumulate and online-softmax order unchanged.
__global__ void __launch_bounds__(256)
k_vit_flash_wmma_db(const uint16_t* __restrict__ qg,
                    const uint16_t* __restrict__ kg,
                    const uint16_t* __restrict__ vg, float* __restrict__ out,
                    int P, float scale) {
  __shared__ uint16_t Qs[128][88], Ks[64][88], VsT[80][68], Pbuf[128][72];
  const int r0 = blockIdx.x * 128;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31, w = tid >> 5;
  const int colbase = lane & 15;
  for (int i = tid; i < 128 * 44; i += 256) {
    int rr = i / 44, jj = i % 44;
    uint32_t val = 0;
    if (r0 + rr < P && jj < 36)
      val = ((const uint32_t*)(qg + (size_t)(r0 + rr) * 1152 + h * 72))[jj];
    ((uint32_t*)Qs[rr])[jj] = val;
  }
  for (int i = tid; i < 64 * 2; i += 256) {  // Ks padding cols [72,88)
    int rr = i >> 1, c9 = 9 + (i & 1);
    ((uint4*)Ks[rr])[c9] = uint4{0, 0, 0, 0};
  }
  for (int i = tid; i < 80 * 68; i += 256)  // VsT zero once (pad rows/cols)
    ((uint16_t*)VsT)[i] = 0;
  auto stage = [&](int c0, uint4* kr, uint4* vr) {
#pragma unroll
    for (int u = 0; u < 3; u++) {
      int i = tid + u * 256;
      uint4 k = {0, 0, 0, 0}, v = {0, 0, 0, 0};
      if (i < 576) {
        int rr = i / 9, c9 = i - rr * 9;
        if (c0 + rr < P) {
          k = *(const uint4*)(kg + (size_t)(c0 + rr) * 1152 + h * 72 + c9 * 8);
          v = *(const uint4*)(vg + (size_t)(c0 + rr) * 1152 + h * 72 + c9 * 8);
        }
      }
      kr[u] = k;
      vr[u] = v;
    }
  };
  auto storeK = [&](const uint4* r) {
#pragma unroll
    for (int u = 0; u < 3; u++) {
      int i = tid + u * 256;
      if (i < 576) {
        int rr = i / 9, c9 = i - rr * 9;
        ((uint4*)Ks[rr])[c9] = r[u];
      }
    }
  };
  auto storeV = [&](const uint4* r) {  // transposed: VsT[dim][key]
#pragma unroll
    for (int u = 0; u < 3; u++) {
      int i = tid + u * 256;
      if (i < 576) {
        int rr = i / 9, c9 = i - rr * 9;
        uint16_t tmp[8];
        *(uint4*)tmp = r[u];
#pragma unroll
        for (int j = 0; j < 8; j++) VsT[c9 * 8 + j][rr] = tmp[j];
      }
    }
  };
  const int nkt = (P + 63) / 64;
  uint4 kn[3], vn[3];
  stage(0, kn, vn);
  storeK(kn);
  storeV(vn);
  __syncthreads();
  int qrowA = -1;
  if (lane < 16 && !(lane & 1)) qrowA = lane / 2;
  else if (lane >= 17 && (lane & 1)) qrowA = 8 + (lane - 17) / 2;
  float m_run[8], l_run[8];
  vit_floatx8 acc[5];
#pragma unroll
  for (int dc = 0; dc < 5; dc++) acc[dc] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int e = 0; e < 8; e++) {
    m_run[e] = -1e30f;
    l_run[e] = 0.f;
  }
  for (int kt = 0; kt < nkt; kt++) {
    const int c0 = kt * 64;
    if (kt + 1 < nkt) stage(c0 + 64, kn, vn);  // prefetch next tile
    vit_floatx8 s[4];
#pragma unroll
    for (int f = 0; f < 4; f++) s[f] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int kk = 0; kk < 5; kk++) {
      vit_shortx16 a = {0};
      if (qrowA >= 0) a = vit_ld16(&Qs[w * 16 + qrowA][kk * 16]);
#pragma unroll
      for (int f = 0; f < 4; f++) {
        short bt[16];
        int krow = f * 16 + colbase;
#pragma unroll
        for (int k = 0; k < 16; k++)
          bt[k] = (short)Ks[krow][kk * 16 + k];
        vit_shortx16 b;
        memcpy(&b, bt, 32);
        s[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s[f]);
      }
    }
    float mx[8];
#pragma unroll
    for (int e = 0; e < 8; e++) mx[e] = -1e30f;
#pragma unroll
    for (int f = 0; f < 4; f++) {
      int col = c0 + f * 16 + colbase;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        float v = s[f][e] * scale;
        if (col >= P) v = -1e30f;
        s[f][e] = v;
        mx[e] = fmaxf(mx[e], v);
      }
    }
#pragma unroll
    for (int e = 0; e < 8; e++) {
#pragma unroll
      for (int o = 1; o <= 8; o <<= 1)
        mx[e] = fmaxf(mx[e], __shfl_xor(mx[e], o));
      float mnew = fmaxf(m_run[e], mx[e]);
      float corr = __expf(m_run[e] - mnew);
      float es = 0.f;
#pragma unroll
      for (int f = 0; f < 4; f++) {
        float p = __expf(s[f][e] - mnew);
        s[f][e] = p;
        es += p;
      }
#pragma unroll
      for (int o = 1; o <= 8; o <<= 1) es += __shfl_xor(es, o);
      l_run[e] = l_run[e] * corr + es;
      m_run[e] = mnew;
#pragma unroll
      for (int dc = 0; dc < 5; dc++) acc[dc][e] *= corr;
    }
#pragma unroll
    for (int f = 0; f < 4; f++)
#pragma unroll
      for (int e = 0; e < 8; e++)
        Pbuf[w * 16 + (lane < 16 ? e : 8 + e)][f * 16 + colbase] =
            vit_f2h(s[f][e]);
    __syncthreads();  // S1: scores done with Ks, Pbuf visible
    if (kt + 1 < nkt) storeK(kn);
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      vit_shortx16 a = {0};
      if (qrowA >= 0) a = vit_ld16(&Pbuf[w * 16 + qrowA][kk * 16]);
#pragma unroll
      for (int dc = 0; dc < 5; dc++) {
        short bt[16];
#pragma unroll
        for (int k = 0; k < 16; k++)
          bt[k] = (short)VsT[dc * 16 + colbase][kk * 16 + k];
        vit_shortx16 b;
        memcpy(&b, bt, 32);
        acc[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[dc]);
      }
    }
    __syncthreads();  // S2: PV done with VsT/Pbuf, Ks store visible
    if (kt + 1 < nkt) storeV(vn);  // next PV is past next S1: visible
  }
#pragma unroll
  for (int e = 0; e < 8; e++) {
    int row = w * 16 + (lane < 16 ? e : 8 + e);
    if (r0 + row >= P) continue;
    float inv = 1.f / l_run[e];
#pragma unroll
    for (int dc = 0; dc < 5; dc++) {
      int col = dc * 16 + colbase;
      if (col < 72)
        out[(size_t)(r0 + row) * 1152 + h * 72 + col] = acc[dc][e] * inv;
    }
  }
}

__global__ void k_vit_gelu_tanh_split(const float* x, uint16_t* hi,
                                      uint16_t* lo, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
  float g = 0.5f * v * (1.f + tanhf(u));
  float h = vit_h2f(vit_f2h(g));
  hi[i] = vit_f2h(g);
  lo[i] = vit_f2h(g - h);
}

__global__ void k_vit_gelu_erf_fp16(const float* x, uint16_t* y, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  y[i] = vit_f2h(0.5f * v * (1.f + erff(v * 0.7071067811865476f)));
}

// fp32 -> bf16 RNE (f2bf, same rounding as torch .to(bf16)); merger output
// is quantized before injection, matching HF masked_scatter of bf16
// image_embeds into bf16 inputs_embeds.
__global__ void k_vit_f32_to_bf16(const float* x, uint16_t* y, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = f2bf(x[i]);
}

// masked_scatter injection: overwrite prefill stream rows (chunk-local,
// [P,10240], 4 repeated 2560 branches) at image placeholder positions with
// the ViT embeds. One block per token; map[i] = (src embed row, dst row).
__global__ void k_img_inject_f32(const uint16_t* __restrict__ emb,
                                 const int2* __restrict__ map,
                                 float* __restrict__ Rb) {
  int src = map[blockIdx.x].x, dst = map[blockIdx.x].y;
  const uint16_t* s = emb + (size_t)src * 2560;
  float* r = Rb + (size_t)dst * 10240;
  for (int d = threadIdx.x; d < 2560; d += blockDim.x) {
    float v = bf2f(s[d]);
    r[d] = v;
    r[2560 + d] = v;
    r[5120 + d] = v;
    r[7680 + d] = v;
  }
}
__global__ void k_img_inject_bf16(const uint16_t* __restrict__ emb,
                                  const int2* __restrict__ map,
                                  uint16_t* __restrict__ Rb) {
  int src = map[blockIdx.x].x, dst = map[blockIdx.x].y;
  const uint16_t* s = emb + (size_t)src * 2560;
  uint16_t* r = Rb + (size_t)dst * 10240;
  for (int d = threadIdx.x; d < 2560; d += blockDim.x) {
    uint16_t v = s[d];
    r[d] = v;
    r[2560 + d] = v;
    r[5120 + d] = v;
    r[7680 + d] = v;
  }
}

// ---- npy I/O (--vision-test offline dumps only) ----
// FNV-1a over the raw patch bytes of every image in a request. Guards the
// M-RoPE cont path: placeholder token runs are pixel-agnostic, so prefix
// matching alone would pair a re-sent history with a DIFFERENT same-shape
// image's cached KV. `count`: hash only the first `count` images (the ones a
// cached prefix holds); 0 images hash to 0.
static uint64_t fnv1a_patches(const std::vector<std::vector<float>>& vp,
                              size_t count = SIZE_MAX) {
  count = std::min(count, vp.size());
  if (count == 0) return 0;
  uint64_t h = 14695981039346656037ull;
  for (size_t k = 0; k < count; k++) {
    const auto& v = vp[k];
    const uint8_t* p = (const uint8_t*)v.data();
    size_t n = v.size() * sizeof(float);
    for (size_t i = 0; i < n; i++) {
      h ^= p[i];
      h *= 1099511628211ull;
    }
  }
  return h;
}
struct VitNpy {
  std::vector<uint64_t> shape;
  char descr[8] = {0};
  std::vector<uint8_t> data;
  uint64_t numel() const {
    uint64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

static VitNpy vit_npy_read(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) throw std::runtime_error(std::string("open ") + path);
  uint8_t magic[6];
  if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6))
    throw std::runtime_error("bad npy magic");
  uint8_t ver[2];
  if (fread(ver, 1, 2, f) != 2) throw std::runtime_error("npy ver");
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t s;
    if (fread(&s, 2, 1, f) != 1) throw std::runtime_error("npy hlen");
    hlen = s;
  } else {
    if (fread(&hlen, 4, 1, f) != 1) throw std::runtime_error("npy hlen");
  }
  std::string hdr(hlen, '\0');
  if (fread(hdr.data(), 1, hlen, f) != hlen) throw std::runtime_error("npy hdr");
  VitNpy r;
  auto dp = hdr.find("'descr'");
  if (dp == std::string::npos) dp = hdr.find("\"descr\"");
  if (dp == std::string::npos) throw std::runtime_error("npy descr");
  auto q1 = hdr.find('\'', dp + 7);
  if (q1 == std::string::npos) q1 = hdr.find('"', dp + 7);
  sscanf(hdr.c_str() + q1 + 1, "%3[^'\"]", r.descr);
  if (hdr.find("False") == std::string::npos)
    throw std::runtime_error("fortran order unsupported");
  auto sp = hdr.find("'shape'");
  if (sp == std::string::npos) sp = hdr.find("\"shape\"");
  auto lp = hdr.find('(', sp), rp = hdr.find(')', sp);
  std::string dims = hdr.substr(lp + 1, rp - lp - 1);
  for (char& c : dims)
    if (c == ',') c = ' ';
  {
    std::istringstream iss(dims);
    uint64_t d;
    while (iss >> d) r.shape.push_back(d);
  }
  if (strcmp(r.descr, "<f4")) throw std::runtime_error("npy dtype must be <f4");
  r.data.resize(r.numel() * 4);
  if (fread(r.data.data(), 1, r.data.size(), f) != r.data.size())
    throw std::runtime_error("npy data short");
  fclose(f);
  return r;
}

static void vit_npy_write_f32(const char* path, const float* p, uint64_t rows,
                              uint64_t cols) {
  FILE* f = fopen(path, "wb");
  if (!f) throw std::runtime_error(std::string("write ") + path);
  char hdr[256];
  int n = snprintf(hdr, sizeof(hdr),
                   "{'descr': '<f4', 'fortran_order': False, 'shape': (%llu, %llu), }",
                   (unsigned long long)rows, (unsigned long long)cols);
  int total = 10 + n + 1;
  int pad = (64 - total % 64) % 64;
  fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  uint16_t hl = n + pad + 1;
  fwrite(&hl, 2, 1, f);
  fwrite(hdr, 1, n, f);
  for (int i = 0; i < pad; i++) fputc(' ', f);
  fputc('\n', f);
  fwrite(p, 4, rows * cols, f);
  fclose(f);
}

static void vit_npy_write_u16(const char* path, const uint16_t* p,
                              uint64_t rows, uint64_t cols) {
  FILE* f = fopen(path, "wb");
  if (!f) throw std::runtime_error(std::string("write ") + path);
  char hdr[256];
  int n = snprintf(hdr, sizeof(hdr),
                   "{'descr': '<u2', 'fortran_order': False, 'shape': (%llu, %llu), }",
                   (unsigned long long)rows, (unsigned long long)cols);
  int total = 10 + n + 1;
  int pad = (64 - total % 64) % 64;
  fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  uint16_t hl = n + pad + 1;
  fwrite(&hl, 2, 1, f);
  fwrite(hdr, 1, n, f);
  for (int i = 0; i < pad; i++) fputc(' ', f);
  fputc('\n', f);
  fwrite(p, 2, rows * cols, f);
  fclose(f);
}

// ---- VisionTower: weights + forward, handles borrowed from the engine ----
struct VisionTower {
  hipblasLtHandle_t lth = nullptr;
  rocblas_handle rbh = nullptr;
  void* d_ltws = nullptr;
  size_t lt_ws = 0;

  struct BlockW {
    uint16_t *qkv_w, *proj_w, *fc1_w, *fc2_w;
    float *qkv_b, *proj_b, *fc1_b, *fc2_b, *ln1_w, *ln1_b, *ln2_w, *ln2_b;
  };
  uint16_t* patch_w = nullptr;   // 1152x1536
  float* patch_b = nullptr;
  uint16_t* pos_table = nullptr; // 2304x1152
  BlockW blk[27];
  float *mg_ln_w = nullptr, *mg_ln_b = nullptr, *mg_fc1_b = nullptr,
        *mg_fc2_b = nullptr;
  uint16_t *mg_fc1_w = nullptr, *mg_fc2_w = nullptr;  // 4608x4608, 2560x4608

  struct LtEntry {
    hipblasLtMatmulDesc_t opd;
    hipblasLtMatrixLayout_t Al, Bl, Cl, Dl;
    hipblasLtMatmulAlgo_t algo;
  };
  using LtKey = std::tuple<int, int, int, int, int, int, int, int, int>;
  std::map<LtKey, LtEntry> lt_cache;

  // grow-only activation buffers (sized by the largest image seen)
  int capP = 0;
  float *d_h = nullptr, *d_n = nullptr, *d_qkv = nullptr, *d_attn = nullptr,
        *d_mlp = nullptr, *d_out = nullptr;
  uint16_t *d_xbf = nullptr, *d_xlo = nullptr, *d_qb = nullptr, *d_kb = nullptr,
           *d_vb = nullptr, *d_emb_bf16 = nullptr;
  float *d_cos = nullptr, *d_sin = nullptr;
  int* d_idx = nullptr;
  float* d_wts = nullptr;

  void lt_gemm(hipblasOperation_t opA, hipblasOperation_t opB, int M, int N,
               int K, const uint16_t* A, int lda, const uint16_t* B, int ldb,
               void* C, int ldc, float alpha, float beta, bool c16 = false) {
    LtKey key{(int)opA, (int)opB, M, N, K, lda, ldb, ldc, (int)c16};
    auto it = lt_cache.find(key);
    if (it == lt_cache.end()) {
      LtEntry e;
      hipblasLtMatmulDescCreate(&e.opd, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSA,
                                      &opA, sizeof(opA));
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSB,
                                      &opB, sizeof(opB));
      hipblasLtMatrixLayoutCreate(&e.Al, HIP_R_16F, opA == HIPBLAS_OP_N ? M : K,
                                  opA == HIPBLAS_OP_N ? K : M, lda);
      hipblasLtMatrixLayoutCreate(&e.Bl, HIP_R_16F, opB == HIPBLAS_OP_N ? K : N,
                                  opB == HIPBLAS_OP_N ? N : K, ldb);
      hipDataType cdt = c16 ? HIP_R_16F : HIP_R_32F;
      hipblasLtMatrixLayoutCreate(&e.Cl, cdt, M, N, ldc);
      hipblasLtMatrixLayoutCreate(&e.Dl, cdt, M, N, ldc);
      hipblasLtMatmulPreference_t pref;
      hipblasLtMatmulPreferenceCreate(&pref);
      hipblasLtMatmulPreferenceSetAttribute(
          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws,
          sizeof(lt_ws));
      hipblasLtMatmulHeuristicResult_t hres[16];
      int nres = 0;
      hipblasLtMatmulAlgoGetHeuristic(lth, e.opd, e.Al, e.Bl, e.Cl, e.Dl, pref,
                                      16, hres, &nres);
      float one = 1.f, zero = 0.f;
      int best = -1;
      double best_ms = 1e30;
      for (int i = 0; i < nres; i++) {
        if (hres[i].state != HIPBLAS_STATUS_SUCCESS) continue;
        if (hres[i].workspaceSize > lt_ws) continue;
        hipblasStatus_t st =
            hipblasLtMatmul(lth, e.opd, &one, A, e.Al, B, e.Bl, &zero, C, e.Cl,
                            C, e.Dl, &hres[i].algo, d_ltws, lt_ws, g_str);
        hipError_t sync_err = hipStreamSynchronize(g_str);
        if (st != HIPBLAS_STATUS_SUCCESS || sync_err != hipSuccess) {
          (void)hipGetLastError();
          continue;
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 3; r++)
          hipblasLtMatmul(lth, e.opd, &one, A, e.Al, B, e.Bl, &zero, C, e.Cl,
                          C, e.Dl, &hres[i].algo, d_ltws, lt_ws, g_str);
        sync_err = hipStreamSynchronize(g_str);
        (void)sync_err;
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        if (ms < best_ms) {
          best_ms = ms;
          best = i;
        }
      }
      if (best < 0) {
        if (c16) {
          fprintf(stderr, "vit gemm c16 no LT algo M=%d N=%d K=%d\n", M, N, K);
          exit(1);
        }
        rocblas_status st = rocblas_gemm_ex(
            rbh, (rocblas_operation)opA, (rocblas_operation)opB, M, N, K, &one,
            A, rocblas_datatype_f16_r, lda, B, rocblas_datatype_f16_r, ldb,
            &zero, C, rocblas_datatype_f32_r, ldc, C, rocblas_datatype_f32_r,
            ldc, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
        if (st != rocblas_status_success) {
          fprintf(stderr, "vit gemm failed M=%d N=%d K=%d (rb %d)\n", M, N, K,
                  (int)st);
          exit(1);
        }
        hipblasLtMatmulDescDestroy(e.opd);
        hipblasLtMatrixLayoutDestroy(e.Al);
        hipblasLtMatrixLayoutDestroy(e.Bl);
        hipblasLtMatrixLayoutDestroy(e.Cl);
        hipblasLtMatrixLayoutDestroy(e.Dl);
        return;
      }
      e.algo = hres[best].algo;
      it = lt_cache.emplace(key, e).first;
    }
    hipblasLtMatmul(lth, it->second.opd, &alpha, A, it->second.Al, B,
                    it->second.Bl, &beta, C, it->second.Cl, C, it->second.Dl,
                    &it->second.algo, d_ltws, lt_ws, g_str);
  }

  // row-major projection: Y(PxN) f32 = X(PxK) fp16 @ W(NxK) fp16^T
  void proj_gemm(const uint16_t* X, const uint16_t* W, float* Y, int P, int N,
                 int K) {
    lt_gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, N, P, K, W, K, X, K, Y, N, 1.f, 0.f);
  }
  void proj_gemm_split(const uint16_t* hi, const uint16_t* lo,
                       const uint16_t* W, float* Y, int P, int N, int K) {
    lt_gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, N, P, K, W, K, hi, K, Y, N, 1.f, 0.f);
    lt_gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, N, P, K, W, K, lo, K, Y, N, 1.f, 1.f);
  }

  static void* dup_dev(const void* p, size_t bytes) {
    void* d;
    DALLOC(&d, bytes);
    CK(hipMemcpy(d, p, bytes, hipMemcpyHostToDevice));
    return d;
  }
  static float* dup_dev_bf16_to_f32(const Tensor& t) {
    uint64_t n = t.numel();
    std::vector<float> h(n);
    const uint16_t* p = (const uint16_t*)t.data;
    for (uint64_t i = 0; i < n; i++) h[i] = hgn::bf16_to_f32(p[i]);
    return (float*)dup_dev(h.data(), n * 4);
  }

  void load(const char* path) {
    // *.gguf = llama.cpp mmproj (clip / qwen3vl_merger), mapped to the same
    // visual.* bf16 tensors in memory (gguf_map::build_vision, bit-exact vs hgn).
    const size_t pl = strlen(path);
    const bool is_gguf = pl > 5 && strcmp(path + pl - 5, ".gguf") == 0;
    std::unique_ptr<Checkpoint> ckp(is_gguf ? new Checkpoint() : new Checkpoint(path));
    Checkpoint& ck = *ckp;
    if (is_gguf) gguf_map::build_vision(ck, gguf::File(path));
    // GEMM weights bf16 -> fp16 host-side (exact at normal magnitudes).
    auto bf = [&](const char* name) -> uint16_t* {
      const Tensor& t = ck.at(name);
      if (t.dtype != 0) throw std::runtime_error(std::string("vit dtype ") + name);
      uint64_t n = t.numel();
      std::vector<uint16_t> h(n);
      const uint16_t* p = (const uint16_t*)t.data;
      for (uint64_t i = 0; i < n; i++)
        h[i] = vit_f2h_host(hgn::bf16_to_f32(p[i]));
      return (uint16_t*)dup_dev(h.data(), n * 2);
    };
    auto f32 = [&](const char* name) { return dup_dev_bf16_to_f32(ck.at(name)); };
    patch_w = bf("visual.patch_embed.proj.weight");
    patch_b = f32("visual.patch_embed.proj.bias");
    pos_table = bf("visual.pos_embed.weight");
    char nm[128];
    for (int i = 0; i < 27; i++) {
      BlockW& b = blk[i];
      snprintf(nm, 128, "visual.blocks.%d.attn.qkv.weight", i);
      b.qkv_w = bf(nm);
      snprintf(nm, 128, "visual.blocks.%d.attn.qkv.bias", i);
      b.qkv_b = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.attn.proj.weight", i);
      b.proj_w = bf(nm);
      snprintf(nm, 128, "visual.blocks.%d.attn.proj.bias", i);
      b.proj_b = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.mlp.linear_fc1.weight", i);
      b.fc1_w = bf(nm);
      snprintf(nm, 128, "visual.blocks.%d.mlp.linear_fc1.bias", i);
      b.fc1_b = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.mlp.linear_fc2.weight", i);
      b.fc2_w = bf(nm);
      snprintf(nm, 128, "visual.blocks.%d.mlp.linear_fc2.bias", i);
      b.fc2_b = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.norm1.weight", i);
      b.ln1_w = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.norm1.bias", i);
      b.ln1_b = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.norm2.weight", i);
      b.ln2_w = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.norm2.bias", i);
      b.ln2_b = f32(nm);
    }
    mg_ln_w = f32("visual.merger.norm.weight");
    mg_ln_b = f32("visual.merger.norm.bias");
    mg_fc1_w = bf("visual.merger.linear_fc1.weight");
    mg_fc1_b = f32("visual.merger.linear_fc1.bias");
    mg_fc2_w = bf("visual.merger.linear_fc2.weight");
    mg_fc2_b = f32("visual.merger.linear_fc2.bias");
    fprintf(stderr, "vision: tower loaded from %s (%zu tensors)\n", path,
            ck.tensor_count());
  }

  static uint16_t vit_f2h_host(float f) {
    __half h = __float2half_rn(f);
    return __half_as_ushort(h);
  }

  void ensure(int P) {
    if (P <= capP) return;
    if (capP) {
      (void)hipFree(d_h); (void)hipFree(d_n); (void)hipFree(d_qkv);
      (void)hipFree(d_attn);
      (void)hipFree(d_mlp); (void)hipFree(d_out); (void)hipFree(d_xbf);
      (void)hipFree(d_xlo);
      (void)hipFree(d_qb); (void)hipFree(d_kb); (void)hipFree(d_vb);
      (void)hipFree(d_emb_bf16);
      (void)hipFree(d_cos); (void)hipFree(d_sin); (void)hipFree(d_idx);
      (void)hipFree(d_wts);
    }
    int M = P / 4;
    CK(hipMalloc(&d_h, (size_t)P * 1152 * 4));
    CK(hipMalloc(&d_n, (size_t)P * 1152 * 4));
    CK(hipMalloc(&d_qkv, (size_t)P * 3456 * 4));
    CK(hipMalloc(&d_attn, (size_t)P * 1152 * 4));
    CK(hipMalloc(&d_mlp, (size_t)P * 4608 * 4));  // merger fc1 also needs 4608
    CK(hipMalloc(&d_out, (size_t)M * 2560 * 4));
    CK(hipMalloc(&d_xbf, (size_t)P * 4608 * 2));
    CK(hipMalloc(&d_xlo, (size_t)P * 4608 * 2));
    CK(hipMalloc(&d_qb, (size_t)P * 1152 * 2));
    CK(hipMalloc(&d_kb, (size_t)P * 1152 * 2));
    CK(hipMalloc(&d_vb, (size_t)P * 1152 * 2));
    CK(hipMalloc(&d_emb_bf16, (size_t)M * 2560 * 2));
    CK(hipMalloc(&d_cos, (size_t)P * 72 * 4));
    CK(hipMalloc(&d_sin, (size_t)P * 72 * 4));
    CK(hipMalloc(&d_idx, (size_t)P * 4 * 4));
    CK(hipMalloc(&d_wts, (size_t)P * 4 * 4));
    capP = P;
  }

  // pos_embed 4-tap indices/weights + rope cos/sin (block-major (h,w) order,
  // replicating transformers vision_utils; host fp64 -> fp32).
  static void host_pos_interp(int gh, int gw, std::vector<int>& idx,
                              std::vector<float>& wts) {
    const int side = 48, merge = 2;
    int P = gh * gw;
    idx.resize(P * 4);
    wts.resize(P * 4);
    int blocks_w = gw / merge;
    for (int p = 0; p < P; p++) {
      int in_col = p % merge, in_row = (p / merge) % merge;
      int block_col = (p / (merge * merge)) % blocks_w;
      int block_row = p / (merge * merge * blocks_w);
      int row = block_row * merge + in_row;
      int col = block_col * merge + in_col;
      double src_r = (double)row * (side - 1) / std::max(gh - 1, 1);
      double src_c = (double)col * (side - 1) / std::max(gw - 1, 1);
      int fr = (int)floor(src_r), fc = (int)floor(src_c);
      double wr[2] = {1.0 - (src_r - fr), src_r - fr};
      double wc[2] = {1.0 - (src_c - fc), src_c - fc};
      int tr[2] = {std::min(std::max(fr, 0), side - 1),
                   std::min(std::max(fr + 1, 0), side - 1)};
      int tc[2] = {std::min(std::max(fc, 0), side - 1),
                   std::min(std::max(fc + 1, 0), side - 1)};
      int t = 0;
      for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++) {
          idx[p * 4 + t] = tr[a] * side + tc[b];
          wts[p * 4 + t] = (float)(wr[a] * wc[b]);
          t++;
        }
    }
  }

  static void host_rope(int gh, int gw, std::vector<float>& cos_t,
                        std::vector<float>& sin_t) {
    const int merge = 2;
    int P = gh * gw;
    cos_t.resize(P * 72);
    sin_t.resize(P * 72);
    double inv_freq[18];
    for (int i = 0; i < 18; i++)
      inv_freq[i] = 1.0 / pow(10000.0, (2.0 * i) / 36.0);
    int blocks_w = gw / merge;
    for (int p = 0; p < P; p++) {
      int in_col = p % merge, in_row = (p / merge) % merge;
      int block_col = (p / (merge * merge)) % blocks_w;
      int block_row = p / (merge * merge * blocks_w);
      int row = block_row * merge + in_row;
      int col = block_col * merge + in_col;
      float rot[36];
      for (int i = 0; i < 18; i++) rot[i] = (float)(row * inv_freq[i]);
      for (int i = 0; i < 18; i++) rot[18 + i] = (float)(col * inv_freq[i]);
      for (int d = 0; d < 36; d++) {
        cos_t[p * 72 + d] = cos_t[p * 72 + 36 + d] = cosf(rot[d]);
        sin_t[p * 72 + d] = sin_t[p * 72 + 36 + d] = sinf(rot[d]);
      }
    }
  }

  // Full forward: patches fp32 host (P*1536, block-major (C,T,ph,pw) order,
  // preprocessing is the client's job) -> d_emb_bf16 [P/4, 2560]. Returns
  // M = P/4. dump_prefix != nullptr writes vit.cpp-compatible layer dumps
  // (offline --vision-test only; syncs the stream).
  int forward(const float* patches, int P, int gh, int gw,
              const char* dump_prefix = nullptr) {
    if (gh * gw != P || gh % 2 || gw % 2) {
      fprintf(stderr, "vision: bad grid (%d,%d) for P=%d\n", gh, gw, P);
      return -1;
    }
    ensure(P);
    int M = P / 4;
    std::string ddir;
    if (dump_prefix) {
      ddir = std::string(dump_prefix) + ".layers";
      mkdir(ddir.c_str(), 0777);
    }
    auto dump_f32 = [&](const char* name, const float* dev, uint64_t rows,
                        uint64_t cols) {
      if (!dump_prefix) return;
      std::vector<float> h(rows * cols);
      CK(hipMemcpy(h.data(), dev, h.size() * 4, hipMemcpyDeviceToHost));
      vit_npy_write_f32((ddir + "/" + name).c_str(), h.data(), rows, cols);
    };

    // rope cos/sin
    std::vector<float> hcos, hsin;
    host_rope(gh, gw, hcos, hsin);
    CK(hipMemcpy(d_cos, hcos.data(), hcos.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_sin, hsin.data(), hsin.size() * 4, hipMemcpyHostToDevice));
    if (dump_prefix) {
      vit_npy_write_f32((ddir + "/rope_cos.npy").c_str(), hcos.data(), P, 72);
      vit_npy_write_f32((ddir + "/rope_sin.npy").c_str(), hsin.data(), P, 72);
    }

    const float scale = powf(72.f, -0.5f);
    auto run_block = [&](int bi) {
      BlockW& B = blk[bi];
      k_vit_layernorm<<<P, 256, 0, g_str>>>(d_h, d_n, B.ln1_w, B.ln1_b, 1152,
                                            1e-6f);
      k_vit_f32_to_fp16<<<(P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_n, d_xbf, (size_t)P * 1152);
      proj_gemm(d_xbf, B.qkv_w, d_qkv, P, 3456, 1152);
      k_vit_bias_add<<<((size_t)P * 3456 + 255) / 256, 256, 0, g_str>>>(
          d_qkv, B.qkv_b, P, 3456);
      {
        dim3 grid(P, 16);
        k_vit_rope_qkv<<<grid, 36, 0, g_str>>>(d_qkv, d_cos, d_sin, d_qb,
                                               d_kb, d_vb, P);
      }
      {
        dim3 grid((P + 127) / 128, 16);
        k_vit_flash_wmma_db<<<grid, 256, 0, g_str>>>(d_qb, d_kb, d_vb, d_attn,
                                                     P, scale);
      }
      k_vit_f32_to_fp16<<<(P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_attn, d_xbf, (size_t)P * 1152);
      proj_gemm(d_xbf, B.proj_w, d_qkv, P, 1152, 1152);
      k_vit_bias_resadd<<<((size_t)P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_h, d_qkv, B.proj_b, P, 1152);
      k_vit_layernorm<<<P, 256, 0, g_str>>>(d_h, d_n, B.ln2_w, B.ln2_b, 1152,
                                            1e-6f);
      k_vit_f32_split_fp16<<<(P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_n, d_xbf, d_xlo, (size_t)P * 1152);
      proj_gemm_split(d_xbf, d_xlo, B.fc1_w, d_mlp, P, 4304, 1152);
      k_vit_bias_add<<<((size_t)P * 4304 + 255) / 256, 256, 0, g_str>>>(
          d_mlp, B.fc1_b, P, 4304);
      k_vit_gelu_tanh_split<<<((size_t)P * 4304 + 255) / 256, 256, 0, g_str>>>(
          d_mlp, d_xbf, d_xlo, (size_t)P * 4304);
      proj_gemm_split(d_xbf, d_xlo, B.fc2_w, d_qkv, P, 1152, 4304);
      k_vit_bias_resadd<<<((size_t)P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_h, d_qkv, B.fc2_b, P, 1152);
      if (dump_prefix) {
        char nm[32];
        snprintf(nm, sizeof nm, "block%02d.npy", bi);
        dump_f32(nm, d_h, P, 1152);
      }
    };

    // patch_embed: fp32 -> split-fp16, dual GEMM + bias
    CK(hipMemcpy(d_mlp, patches, (size_t)P * 1536 * 4,
                 hipMemcpyHostToDevice));  // d_mlp doubles as staging
    k_vit_f32_split_fp16<<<(P * 1536 + 255) / 256, 256, 0, g_str>>>(
        d_mlp, d_xbf, d_xlo, (size_t)P * 1536);
    proj_gemm_split(d_xbf, d_xlo, patch_w, d_h, P, 1152, 1536);
    k_vit_bias_add<<<((size_t)P * 1152 + 255) / 256, 256, 0, g_str>>>(
        d_h, patch_b, P, 1152);
    dump_f32("patchembed.npy", d_h, P, 1152);

    // pos_embed gather
    {
      std::vector<int> hidx;
      std::vector<float> hwts;
      host_pos_interp(gh, gw, hidx, hwts);
      CK(hipMemcpy(d_idx, hidx.data(), hidx.size() * 4, hipMemcpyHostToDevice));
      CK(hipMemcpy(d_wts, hwts.data(), hwts.size() * 4, hipMemcpyHostToDevice));
      k_vit_pos_embed<<<P, 256, 0, g_str>>>(d_h, pos_table, d_idx, d_wts, P);
    }
    dump_f32("embed.npy", d_h, P, 1152);

    for (int bi = 0; bi < 27; bi++) run_block(bi);

    // merger: LN -> fc1 -> erf-GELU -> fc2 (+bias), then bf16 RNE quantize
    k_vit_layernorm<<<P, 256, 0, g_str>>>(d_h, d_n, mg_ln_w, mg_ln_b, 1152,
                                          1e-6f);
    dump_f32("merger_ln.npy", d_n, M, 4608);
    k_vit_f32_to_fp16<<<((size_t)M * 4608 + 255) / 256, 256, 0, g_str>>>(
        d_n, d_xbf, (size_t)M * 4608);
    proj_gemm(d_xbf, mg_fc1_w, d_mlp, M, 4608, 4608);
    k_vit_bias_add<<<((size_t)M * 4608 + 255) / 256, 256, 0, g_str>>>(
        d_mlp, mg_fc1_b, M, 4608);
    dump_f32("merger_fc1.npy", d_mlp, M, 4608);
    k_vit_gelu_erf_fp16<<<((size_t)M * 4608 + 255) / 256, 256, 0, g_str>>>(
        d_mlp, d_xbf, (size_t)M * 4608);
    proj_gemm(d_xbf, mg_fc2_w, d_out, M, 2560, 4608);
    k_vit_bias_add<<<((size_t)M * 2560 + 255) / 256, 256, 0, g_str>>>(
        d_out, mg_fc2_b, M, 2560);
    k_vit_f32_to_bf16<<<((size_t)M * 2560 + 255) / 256, 256, 0, g_str>>>(
        d_out, d_emb_bf16, (size_t)M * 2560);
    dump_f32("merged.npy", d_out, M, 2560);
    if (dump_prefix) {
      std::vector<uint16_t> h((size_t)M * 2560);
      CK(hipMemcpy(h.data(), d_emb_bf16, h.size() * 2, hipMemcpyDeviceToHost));
      vit_npy_write_u16((ddir + "/embeds.bf16.npy").c_str(), h.data(), M, 2560);
    }
    return M;
  }
};

// --- model -------------------------------------------------------------------

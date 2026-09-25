// ============================ kernels =======================================

// --- Q4C-P GEMV: y[r] = sum_c cb[nib]*scale[r][c/32]*x[c]; one warp per row.
// v3 (decode GEMV governance): cb staged in LDS, fp32 x, 2 rows per warp.
// Bit-exact vs the original 1-row/warp version: same per-lane fma order,
// same shuffle reduction tree (verified d=0 in tools/decode_gemv_bench.cu).
__global__ void k_q4cp_gemv(const uint8_t* __restrict__ codes,
                            const uint8_t* __restrict__ scales,
                            const float* __restrict__ cb,
                            const float* __restrict__ x, float* __restrict__ y,
                            uint64_t rows, uint64_t cols, uint64_t scale_stride) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t groups_per_row = cols / 32;
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  bool has1 = row0 + 1 < rows;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  float acc0 = 0.f, acc1 = 0.f;
  for (uint64_t g = lane; g < groups_per_row; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + g * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + g * 2));
    }
    const float* xg = x + g * 32;
    float xv[32];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float4 t = *(const float4*)(xg + i * 4);
      xv[i * 4] = t.x; xv[i * 4 + 1] = t.y; xv[i * 4 + 2] = t.z;
      xv[i * 4 + 3] = t.w;
    }
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t v0 = w0[i], v1 = w1[i];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        a0 += cbf[v0 & 0xf] * xv[i * 8 + j];
        a1 += cbf[v1 & 0xf] * xv[i * 8 + j];
        v0 >>= 4; v1 >>= 4;
      }
    }
    acc0 += a0 * s0;
    acc1 += a1 * s1;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc0 += __shfl_down_sync(~0ull, acc0, off);
    acc1 += __shfl_down_sync(~0ull, acc1, off);
  }
  if (lane == 0) {
    y[row0] = acc0;
    if (has1) y[row0 + 1] = acc1;
  }
}

// --- q4cp multi-row GEMV: y[P,N] = x[P,K] . W[N,K]^T, weight streamed once ---
// Same dequant math as k_q4cp_gemv (16-entry codebook + fp16 group scale),
// but each warp applies its 2 weight rows to all P input rows. x is small and
// stays L2-resident across warps, so DRAM traffic ~ N*K/2 bytes regardless of
// P — this replaces the dequant->bf16->hipBLASLt path for small P, which
// costs ~4.5x the weight bytes in extra traffic.
template <int P, bool XBF16 = false>
__global__ void k_q4cp_gemv_mr(const uint8_t* __restrict__ codes,
                               const uint8_t* __restrict__ scales,
                               const float* __restrict__ cb,
                               const void* __restrict__ x,
                               float* __restrict__ y, uint64_t rows,
                               uint64_t cols, uint64_t scale_stride,
                               uint64_t xstride) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t groups_per_row = cols / 32;
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  bool has1 = row0 + 1 < rows;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    acc0[p] = 0.f;
    acc1[p] = 0.f;
  }
  for (uint64_t g = lane; g < groups_per_row; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + g * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + g * 2));
    }
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
#pragma unroll
    for (int p = 0; p < P; p++) {
      float xv[32];
      if (XBF16) {
        const uint16_t* xg =
            (const uint16_t*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          uint4 t = *(const uint4*)(xg + i * 8);
          const uint16_t* th = (const uint16_t*)&t;
#pragma unroll
          for (int j = 0; j < 8; j++)
            xv[i * 8 + j] = __bfloat162float(*(__hip_bfloat16*)(&th[j]));
        }
      } else {
        const float* xg = (const float*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          float4 t = *(const float4*)(xg + i * 4);
          xv[i * 4] = t.x;
          xv[i * 4 + 1] = t.y;
          xv[i * 4 + 2] = t.z;
          xv[i * 4 + 3] = t.w;
        }
      }
      uint32_t u0[4] = {w0[0], w0[1], w0[2], w0[3]};
      uint32_t u1[4] = {w1[0], w1[1], w1[2], w1[3]};
      float b0 = 0.f, b1 = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
          b0 += cbf[u0[i] & 0xf] * xv[i * 8 + j];
          b1 += cbf[u1[i] & 0xf] * xv[i * 8 + j];
          u0[i] >>= 4;
          u1[i] >>= 4;
        }
      }
      acc0[p] += b0 * s0;
      acc1[p] += b1 * s1;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      acc0[p] += __shfl_down_sync(~0ull, acc0[p], off);
      acc1[p] += __shfl_down_sync(~0ull, acc1[p], off);
    }
    if (lane == 0) {
      y[(size_t)p * rows + row0] = acc0[p];
      if (has1) y[(size_t)p * rows + row0 + 1] = acc1[p];
    }
  }
}

// --- bf16-weight multi-row GEMV: y[P,N] = x[P,K] . W[N,K]^T (bf16 weights) ---
template <int P>
__global__ void k_bf16_gemv_mr(const uint16_t* __restrict__ w,
                               const float* __restrict__ x,
                               float* __restrict__ y, uint64_t rows,
                               uint64_t cols, uint64_t xstride) {
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  bool has1 = row0 + 1 < rows;
  const uint16_t* r0 = w + row0 * cols;
  const uint16_t* r1 = w + (row0 + 1) * cols;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    acc0[p] = 0.f;
    acc1[p] = 0.f;
  }
  for (uint64_t c = lane * 8; c < cols; c += warpSize * 8) {
    uint4 pk0 = *(const uint4*)(r0 + c);
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    if (has1) pk1 = *(const uint4*)(r1 + c);
    const uint16_t* h0 = (const uint16_t*)&pk0;
    const uint16_t* h1 = (const uint16_t*)&pk1;
    float w0[8], w1[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      w0[j] = __bfloat162float(*(__hip_bfloat16*)(&h0[j]));
      w1[j] = __bfloat162float(*(__hip_bfloat16*)(&h1[j]));
    }
#pragma unroll
    for (int p = 0; p < P; p++) {
      const float* xg = x + (size_t)p * xstride + c;
      float4 xa = *(const float4*)xg;
      float4 xb = *(const float4*)(xg + 4);
      float b0 = w0[0] * xa.x + w0[1] * xa.y + w0[2] * xa.z + w0[3] * xa.w +
                 w0[4] * xb.x + w0[5] * xb.y + w0[6] * xb.z + w0[7] * xb.w;
      float b1 = w1[0] * xa.x + w1[1] * xa.y + w1[2] * xa.z + w1[3] * xa.w +
                 w1[4] * xb.x + w1[5] * xb.y + w1[6] * xb.z + w1[7] * xb.w;
      acc0[p] += b0;
      acc1[p] += b1;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      acc0[p] += __shfl_down_sync(~0ull, acc0[p], off);
      acc1[p] += __shfl_down_sync(~0ull, acc1[p], off);
    }
    if (lane == 0) {
      y[(size_t)p * rows + row0] = acc0[p];
      if (has1) y[(size_t)p * rows + row0 + 1] = acc1[p];
    }
  }
}

// --- q8g64 multi-row GEMV (overlay o_proj): [codes uint8 x cols][(s,m) fp16 per 64] ---
template <int P>
__global__ void k_q8g64_gemv_mr(const uint8_t* __restrict__ w,
                                const float* __restrict__ x,
                                float* __restrict__ y, uint64_t rows,
                                uint64_t cols, uint64_t xstride) {
  uint64_t stride = cols + cols / 64 * 4;
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  bool has1 = row0 + 1 < rows;
  const uint8_t* r0 = w + row0 * stride;
  const uint8_t* r1 = w + (row0 + 1) * stride;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    acc0[p] = 0.f;
    acc1[p] = 0.f;
  }
  for (uint64_t g = lane; g < cols / 64; g += warpSize) {
    const uint8_t* c0 = r0 + g * 64;
    const __half* sm0 = (const __half*)(r0 + cols + g * 4);
    float s0 = __half2float(sm0[0]), m0 = __half2float(sm0[1]);
    float s1 = 0.f, m1 = 0.f;
    const uint8_t* c1 = (has1 ? r1 : r0) + g * 64;  // !has1: dead, s1=m1=0
    const __half* sm1 = (const __half*)(r1 + cols + g * 4);
    if (has1) {
      s1 = __half2float(sm1[0]);
      m1 = __half2float(sm1[1]);
    }
#pragma unroll
    for (int p = 0; p < P; p++) {
      const float* xg = x + (size_t)p * xstride + g * 64;
      float b0 = 0.f, b1 = 0.f, xs = 0.f;
#pragma unroll
      for (int j = 0; j < 64; j += 4) {
        float4 t = *(const float4*)(xg + j);
        b0 += (float)c0[j] * t.x + (float)c0[j + 1] * t.y +
              (float)c0[j + 2] * t.z + (float)c0[j + 3] * t.w;
        b1 += (float)c1[j] * t.x + (float)c1[j + 1] * t.y +
              (float)c1[j + 2] * t.z + (float)c1[j + 3] * t.w;
        xs += t.x + t.y + t.z + t.w;
      }
      // q8g64 dequant: w = c*s + m  ->  w.x = s*(c.x) + m*sum(x)
      acc0[p] += b0 * s0 + m0 * xs;
      acc1[p] += b1 * s1 + m1 * xs;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      acc0[p] += __shfl_down_sync(~0ull, acc0[p], off);
      acc1[p] += __shfl_down_sync(~0ull, acc1[p], off);
    }
    if (lane == 0) {
      y[(size_t)p * rows + row0] = acc0[p];
      if (has1) y[(size_t)p * rows + row0 + 1] = acc1[p];
    }
  }
}

// --- q8g64 GEMV (overlay o_proj): per row [cols uint8][cols/64 x (fp16 s, fp16 m)]
__global__ void k_q8g64_gemv(const uint8_t* __restrict__ w, const float* __restrict__ x,
                             float* __restrict__ y, uint64_t rows, uint64_t cols) {
  uint64_t stride = cols + cols / 64 * 4;
  uint64_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint8_t* rp = w + row * stride;
  float acc = 0.f;
  for (uint64_t g = lane; g < cols / 64; g += warpSize) {
    const uint8_t* codes = rp + g * 64;
    const __half* sm = (const __half*)(rp + cols + g * 4);
    float s = __half2float(sm[0]), mn = __half2float(sm[1]);
    float a = 0.f, cs = 0.f;
    const float* xg = x + g * 64;
#pragma unroll
    for (int j = 0; j < 64; j += 4) {
      float4 xv = *(const float4*)(xg + j);
      a += codes[j] * xv.x + codes[j + 1] * xv.y + codes[j + 2] * xv.z +
           codes[j + 3] * xv.w;
      cs += xv.x + xv.y + xv.z + xv.w;
    }
    acc += a * s + mn * cs;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1)
    acc += __shfl_down_sync(~0ull, acc, off);
  if (lane == 0) y[row] = acc;
}

// --- q8g32 planar (GGUF Q8_0 repack, hgn.h dtype 8) --------------------------
// q: [rows][cols] int8 codes, s: [rows][cols/32] fp16 scales; w = q * s.
// y[P,N] = x[P,K] . W[N,K]^T, 2 rows per warp, lanes stride over 32-col
// groups (32 contiguous code bytes = 2 x uint4 per lane), weight read once
// for all P input rows. P=1 doubles as the plain decode GEMV.
__device__ inline float q8_b(uint32_t v, int k) {
  return (float)(int8_t)(uint8_t)(v >> (8 * k));
}
template <int P, bool XBF16 = false>
__global__ void k_q8g32_gemv_mr(const int8_t* __restrict__ q,
                                const __half* __restrict__ s,
                                const void* __restrict__ x,
                                float* __restrict__ y, uint64_t rows,
                                uint64_t cols, uint64_t xstride) {
  const uint64_t gpr = cols / 32;
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  bool has1 = row0 + 1 < rows;
  const int8_t* q0 = q + row0 * cols;
  const int8_t* q1 = q + (has1 ? row0 + 1 : row0) * cols;
  const __half* s0p = s + row0 * gpr;
  const __half* s1p = s + (has1 ? row0 + 1 : row0) * gpr;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    acc0[p] = 0.f;
    acc1[p] = 0.f;
  }
  for (uint64_t g = lane; g < gpr; g += warpSize) {
    const uint4* c0 = (const uint4*)(q0 + g * 32);
    const uint4* c1 = (const uint4*)(q1 + g * 32);
    uint4 a0 = c0[0], a1 = c0[1], b0 = c1[0], b1 = c1[1];
    uint32_t w0[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    uint32_t w1[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    float sc0 = __half2float(s0p[g]);
    float sc1 = has1 ? __half2float(s1p[g]) : 0.f;
#pragma unroll
    for (int p = 0; p < P; p++) {
      float xv[32];
      if (XBF16) {
        const uint16_t* xg = (const uint16_t*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          uint4 t = *(const uint4*)(xg + i * 8);
          const uint16_t* th = (const uint16_t*)&t;
#pragma unroll
          for (int j = 0; j < 8; j++)
            xv[i * 8 + j] = __bfloat162float(*(__hip_bfloat16*)(&th[j]));
        }
      } else {
        const float* xg = (const float*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          float4 t = *(const float4*)(xg + i * 4);
          xv[i * 4] = t.x;
          xv[i * 4 + 1] = t.y;
          xv[i * 4 + 2] = t.z;
          xv[i * 4 + 3] = t.w;
        }
      }
      float d0 = 0.f, d1 = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int k = 0; k < 4; k++) {
          d0 += q8_b(w0[i], k) * xv[i * 4 + k];
          d1 += q8_b(w1[i], k) * xv[i * 4 + k];
        }
      acc0[p] += d0 * sc0;
      acc1[p] += d1 * sc1;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      acc0[p] += __shfl_down_sync(~0ull, acc0[p], off);
      acc1[p] += __shfl_down_sync(~0ull, acc1[p], off);
    }
    if (lane == 0) {
      y[(size_t)p * rows + row0] = acc0[p];
      if (has1) y[(size_t)p * rows + row0 + 1] = acc1[p];
    }
  }
}

// --- q8g32 embedding gather: block b dequants row tok[b] into
// out + b*outstride (fp32), or with HC4 into four bf16 branch copies at
// out + b*4*cols (same layout as k_q4cp_row_b_hc_bf16). cols/32 threads.
template <bool HC4>
__global__ void k_q8g32_row_b(const int8_t* __restrict__ q, const __half* __restrict__ s,
                              const int* __restrict__ d_tokens, uint64_t cols,
                              void* __restrict__ out, uint64_t outstride) {
  uint64_t row = (uint64_t)d_tokens[blockIdx.x];
  uint64_t g = threadIdx.x;
  if (g * 32 >= cols) return;
  const uint4* c = (const uint4*)(q + row * cols + g * 32);
  uint4 a = c[0], b = c[1];
  uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  float sc = __half2float(s[row * (cols / 32) + g]);
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int k = 0; k < 4; k++) {
      uint64_t col = g * 32 + i * 4 + k;
      float v = q8_b(w[i], k) * sc;
      if (HC4) {
        uint32_t u;
        memcpy(&u, &v, 4);
        u += 0x7FFF + ((u >> 16) & 1);
        uint16_t h = (uint16_t)(u >> 16);
        uint16_t* o = (uint16_t*)out + (size_t)blockIdx.x * 4 * cols;
        o[col] = h;
        o[cols + col] = h;
        o[2 * cols + col] = h;
        o[3 * cols + col] = h;
      } else {
        ((float*)out)[(size_t)blockIdx.x * outstride + col] = v;
      }
    }
}

// --- BF16 GEMV: one warp per row, float4 (8 bf16) loads
__global__ void k_bf16_gemv(const uint16_t* __restrict__ w, const float* __restrict__ x,
                            float* __restrict__ y, uint64_t rows, uint64_t cols) {
  uint64_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint16_t* rw = w + row * cols;
  float acc = 0.f;
  for (uint64_t c = lane * 8; c < cols; c += warpSize * 8) {
    uint4 pk = *(const uint4*)(rw + c);  // 8 bf16
    float4 xa = *(const float4*)(x + c);
    float4 xb = *(const float4*)(x + c + 4);
    const uint16_t* h = (const uint16_t*)&pk;
    float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
    for (int j = 0; j < 8; j++) {
      uint32_t u = (uint32_t)h[j] << 16;
      float wf;
      memcpy(&wf, &u, 4);
      acc += wf * xv[j];
    }
  }
  for (int off = warpSize / 2; off > 0; off >>= 1)
    acc += __shfl_down_sync(~0ull, acc, off);
  if (lane == 0) y[row] = acc;
}

// --- Q4C-P single-row dequant (embedding gather): out[cols]
__global__ void k_q4cp_row(const uint8_t* __restrict__ codes,
                           const uint8_t* __restrict__ scales,
                           const float* __restrict__ cb, const int* __restrict__ d_token,
                           uint64_t cols, uint64_t scale_stride, float* __restrict__ out) {
  uint64_t row = (uint64_t)*d_token;
  uint64_t g = threadIdx.x;  // one block, cols/32 threads
  if (g * 32 >= cols) return;
  const uint8_t* rc = codes + row * (cols / 2) + g * 16;
  const uint8_t* rs = scales + row * scale_stride;
  float s = __half2float(*(const __half*)(rs + g * 2));
  uint4 pk = *(const uint4*)rc;
  uint32_t w[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t v = w[i];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      out[g * 32 + i * 8 + j] = __ldg(cb + (v & 0xf)) * s;
      v >>= 4;
    }
  }
}

// --- zero-centered RMSNorm, optionally grouped. grid = one block per group.
__global__ void k_rmsnorm_zc(const float* __restrict__ x, const float* __restrict__ w,
                             float* __restrict__ out, int n, float eps) {
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < n; i += blockDim.x) ss += x[i] * x[i];
  sm[t] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / n + eps);
  for (int i = t; i < n; i += blockDim.x) out[i] = x[i] * inv * (1.f + w[i]);
}

__device__ inline uint16_t f2bf(float f);

// grouped zc rmsnorm over flat (ngroups x gsize): one block per group.
// Batched: grid may exceed ngroups; weight index wraps with % ngroups.
template <bool Bf16 = false>
__global__ void k_rmsnorm_zc_grouped(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     float* __restrict__ out, int gsize, float eps,
                                     int ngroups, uint16_t* __restrict__ bf16 = nullptr) {
  const float* xg = x + (size_t)blockIdx.x * gsize;
  const float* wg = w + (size_t)(blockIdx.x % ngroups) * gsize;
  float* og = out + (size_t)blockIdx.x * gsize;
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < gsize; i += blockDim.x) ss += xg[i] * xg[i];
  sm[t] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / gsize + eps);
  for (int i = t; i < gsize; i += blockDim.x) {
    float value = xg[i] * inv * (1.f + wg[i]);
    og[i] = value;
    if (Bf16) bf16[(size_t)blockIdx.x * gsize + i] = f2bf(value);
  }
}

// --- GDN conv update: state (3,qkv) shifts; out = silu(sum_j w[j]*hist[j])
// hist[0..2] = previous three inputs (oldest first), cur = current input.
__global__ void k_gdn_conv(const float* __restrict__ cur, const float* __restrict__ cw,
                           float* __restrict__ convstate, float* __restrict__ out,
                           int qkv) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= qkv) return;
  float* st = convstate;  // 3 x qkv
  float a = cw[ch * 4 + 0] * st[ch] + cw[ch * 4 + 1] * st[qkv + ch] +
            cw[ch * 4 + 2] * st[2 * qkv + ch] + cw[ch * 4 + 3] * cur[ch];
  st[ch] = st[qkv + ch];
  st[qkv + ch] = st[2 * qkv + ch];
  st[2 * qkv + ch] = cur[ch];
  float sg = 1.f / (1.f + expf(-a));
  out[ch] = a * sg;
}

// --- GDN recurrence, one block per v-head (128 threads).
// S layout: [h][j(128 k)][i(128 v)] row-major. q,k already l2normed; q prescaled.
__global__ void k_gdn_step(const float* __restrict__ q, const float* __restrict__ k,
                           const float* __restrict__ v, const float* __restrict__ g,
                           const float* __restrict__ beta, float* __restrict__ S,
                           float* __restrict__ out, int dk, int dv, int hv, int hk) {
  int h = blockIdx.x;
  int group = hv / hk;
  const float* kh = k + (h / group) * dk;
  const float* qh = q + (h / group) * dk;
  const float* vh = v + h * dv;
  float* Sh = S + (size_t)h * dk * dv;
  float alpha = expf(g[h]);
  float be = beta[h];
  // 4 j-chunks x dv threads: thread t owns column i=t%dv, j-chunk jc=t/dv
  int t = threadIdx.x;
  int i = t % dv, jc = t / dv;
  int j0 = jc * (dk / 4), j1 = j0 + dk / 4;
  __shared__ float part[4][128];  // partial kv / o per j-chunk
  __shared__ float sdelta[128];
  // pass 1: decay + kv partial
  float kv = 0.f;
  for (int j = j0; j < j1; j++) {
    float s = Sh[j * dv + i] * alpha;
    Sh[j * dv + i] = s;
    kv += s * kh[j];
  }
  part[jc][i] = kv;
  __syncthreads();
  float kvf = part[0][i] + part[1][i] + part[2][i] + part[3][i];
  float delta = (vh[i] - kvf) * be;
  if (jc == 0) sdelta[i] = delta;
  __syncthreads();
  delta = sdelta[i];
  // pass 2: rank-1 update + pass 3: output partial
  float o = 0.f;
  for (int j = j0; j < j1; j++) {
    float s = Sh[j * dv + i] + kh[j] * delta;
    Sh[j * dv + i] = s;
    o += s * qh[j];
  }
  __syncthreads();
  part[jc][i] = o;
  __syncthreads();
  if (jc == 0) out[h * dv + i] = part[0][i] + part[1][i] + part[2][i] + part[3][i];
}

// Short speculative batches: keep the entire recurrent state in registers.
// The four partial reductions and accumulation order follow k_gdn_step.
// qkv and out may alias; all input values are read before output is stored.
__global__ void k_gdn_verify(const float* qkv, const float* g48,
                             const float* b48, float* S, float* out,
                             int P, int stride) {
  const int h = blockIdx.x, i = threadIdx.x % 128;
  const int jc = threadIdx.x / 128, j0 = jc * 32;
  float* Sh = S + (size_t)h * 128 * 128;
  float state[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) state[j] = Sh[(j0 + j) * 128 + i];
  __shared__ float part[4][128];
  __shared__ float delta_shared[128];
  for (int row = 0; row < P; ++row) {
    const float* q = qkv + (size_t)row * stride + (h / 3) * 128;
    const float* k = q + 2048;
    const float value = qkv[(size_t)row * stride + 4096 + h * 128 + i];
    const float alpha = expf(g48[row * 48 + h]);
    const float beta = b48[row * 48 + h];
    float kv = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      state[j] *= alpha;
      kv += state[j] * k[j0 + j];
    }
    part[jc][i] = kv;
    __syncthreads();
    float kvf = part[0][i] + part[1][i] + part[2][i] + part[3][i];
    float delta = (value - kvf) * beta;
    if (jc == 0) delta_shared[i] = delta;
    __syncthreads();
    delta = delta_shared[i];
    float result = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      state[j] += k[j0 + j] * delta;
      result += state[j] * q[j0 + j];
    }
    __syncthreads();
    part[jc][i] = result;
    __syncthreads();
    if (jc == 0)
      out[(size_t)row * stride + 4096 + h * 128 + i] =
          part[0][i] + part[1][i] + part[2][i] + part[3][i];
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < 32; ++j) Sh[(j0 + j) * 128 + i] = state[j];
}

// --- indexer proj snapshot: keep every verify row's raw ring value (the
// 128-float tail of the 640-wide proj) so a rollback can rebuild the 4-slot
// raw ring at any accepted depth.
__global__ void k_iproj_snap(const float* __restrict__ proj,
                             float* __restrict__ snap, int P) {
  int t = blockIdx.x, d = threadIdx.x;
  if (t < P) snap[(size_t)t * 128 + d] = proj[(size_t)t * 640 + 512 + d];
}

// --- GDN snapshot: serially advance S over the chunk's P rows (same math as
// k_gdn_step, no output), writing S after each row into snap[t]. S lives in
// registers (32/thread) and d_S is left untouched — the chunked scan after
// this kernel still sees and advances the pre-chunk state. Lets speculative
// rollback install the state at any accepted depth without a re-prefill.
// snap t-stride = ngdn*48*128*128.
__global__ void k_gdn_snap(const float* __restrict__ qkv,
                           const float* __restrict__ g48,
                           const float* __restrict__ b48,
                           const float* __restrict__ S, float* __restrict__ snap,
                           int P, int qkvstride, size_t snaptstride) {
  int h = blockIdx.x;  // 48 v-heads, group of 3 over 16 k-heads
  const int dk = 128, dv = 128;
  const float* Sh = S + (size_t)h * dk * dv;
  float* Sn = snap + (size_t)h * dk * dv;
  int t = threadIdx.x;
  int i = t % dv, jc = t / dv;
  int j0 = jc * (dk / 4);
  float sreg[dk / 4];
#pragma unroll
  for (int j = 0; j < dk / 4; j++) sreg[j] = Sh[(j0 + j) * dv + i];
  __shared__ float part[4][128];
  __shared__ float sdelta[128];
  for (int tt = 0; tt < P; tt++) {
    const float* row = qkv + (size_t)tt * qkvstride;
    const float* kh = row + 2048 + (h / 3) * dk;
    const float* vh = row + 4096 + h * dv;
    float alpha = expf(g48[tt * 48 + h]);
    float be = b48[tt * 48 + h];
    float kv = 0.f;
#pragma unroll
    for (int j = 0; j < dk / 4; j++) {
      sreg[j] *= alpha;
      kv += sreg[j] * kh[j0 + j];
    }
    part[jc][i] = kv;
    __syncthreads();
    float kvf = part[0][i] + part[1][i] + part[2][i] + part[3][i];
    float delta = (vh[i] - kvf) * be;
    if (jc == 0) sdelta[i] = delta;
    __syncthreads();
    delta = sdelta[i];
#pragma unroll
    for (int j = 0; j < dk / 4; j++) {
      sreg[j] += kh[j0 + j] * delta;
      Sn[(size_t)tt * snaptstride + (j0 + j) * dv + i] = sreg[j];
    }
    __syncthreads();
  }
}

// --- conv state snapshot: convst after row t = rows (t-2,t-1,t) of the raw
// qkv stream, with the live (pre-chunk) convst filling rows < 0. snap layout
// per t: [3][qkv], t-stride = ngdn*3*qkv.
__global__ void k_convst_snap(const float* __restrict__ qkvb,
                              const float* __restrict__ convst,
                              float* __restrict__ snap, int P, int qkv,
                              size_t snaptstride) {
  int t = blockIdx.x;
  float* out = snap + (size_t)t * snaptstride;
  for (int c = threadIdx.x; c < qkv; c += blockDim.x) {
    out[c] = t >= 2   ? qkvb[(size_t)(t - 2) * qkv + c]
             : t == 1 ? convst[2 * qkv + c]
                      : convst[qkv + c];
    out[qkv + c] = t >= 1 ? qkvb[(size_t)(t - 1) * qkv + c] : convst[2 * qkv + c];
    out[2 * qkv + c] = qkvb[(size_t)t * qkv + c];
  }
}

// --- indexer raw-ring rebuild after rollback: slot s must hold the proj row
// of the largest position p <= qpos with p%4==s (qpos = q+a+1). Rows from the
// verify chunk come from iproj_snap[qi][p-qbase], older rows from the
// checkpoint ring. One block per QSA layer.
__global__ void k_iraw_restore(const float* __restrict__ iproj_snap,
                               const float* __restrict__ ckpt_iraw,
                               float* __restrict__ iraw, int qbase, int qpos,
                               int snapstride) {
  int qi = blockIdx.x, d = threadIdx.x;
  const float* snapl = iproj_snap + (size_t)qi * snapstride * 128;
  for (int s = 0; s < 4; s++) {
    int p = qpos - ((qpos % 4 - s + 4) % 4);
    const float* src = p >= qbase
                           ? snapl + (size_t)(p - qbase) * 128
                           : ckpt_iraw + (size_t)qi * 512 + (size_t)(p % 4) * 128;
    iraw[(size_t)qi * 512 + s * 128 + d] = src[d];
  }
}

// --- GDN q/k L2 norm + q scale: one block per qk head
__global__ void k_l2norm_qk(float* __restrict__ q, float* __restrict__ k, int dk,
                            float qscale) {
  __shared__ float sm[128];
  int h = blockIdx.x;
  int t = threadIdx.x;
  float* qh = q + h * dk;
  float* kh = k + h * dk;
  float v = (t < dk) ? qh[t] : 0.f;
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float iq = rsqrtf(sm[0] + 1e-6f);
  __syncthreads();
  float kv = (t < dk) ? kh[t] : 0.f;
  sm[t] = kv * kv;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float ik = rsqrtf(sm[0] + 1e-6f);
  if (t < dk) {
    qh[t] = v * iq * qscale;
    kh[t] = kv * ik;
  }
}

// --- GDN gates: alpha_decay[g] = -exp(A_log)*softplus(a+dt_bias) (= log decay),
//     beta = sigmoid(b).  hv threads.  Batched: blockIdx.x = token row (stride).
__global__ void k_gdn_gates(const float* __restrict__ a, const float* __restrict__ b,
                            const float* __restrict__ A_log,
                            const float* __restrict__ dt_bias,
                            float* __restrict__ g_out, float* __restrict__ beta_out,
                            int hv, int stride) {
  int h = threadIdx.x;
  if (h >= hv) return;
  size_t off = (size_t)blockIdx.x * stride;
  a += off;
  b += off;
  g_out += off;
  beta_out += off;
  float sp = a[h] + dt_bias[h];
  sp = sp > 20.f ? sp : log1pf(expf(sp));
  g_out[h] = -expf(A_log[h]) * sp;
  beta_out[h] = 1.f / (1.f + expf(-b[h]));
}

// --- GDN gated norm: per v-head plain RMSNorm(128) * sigmoid(z)
// Batched: grid.y = token row; y/out row stride ys, z row stride zs.
// Bf16: additionally emit compact [grid.y, gridDim.x*dv] bf16 (same f2bf RNE
// as k_f32_to_bf16 on out, so the consumer GEMM can skip its conversion).
template <bool Bf16 = false>
__global__ void k_gdn_gatednorm(const float* __restrict__ y, const float* __restrict__ z,
                                const float* __restrict__ w, float* __restrict__ out,
                                int dv, float eps, int ys, int zs,
                                uint16_t* __restrict__ bf16 = nullptr) {
  __shared__ float sm[128];
  int h = blockIdx.x;
  int t = threadIdx.x;
  const float* yh = y + (size_t)blockIdx.y * ys + h * dv;
  const float* zh = z + (size_t)blockIdx.y * zs + h * dv;
  float* oh = out + (size_t)blockIdx.y * ys + h * dv;
  float v = (t < dv) ? yh[t] : 0.f;
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / dv + eps);
  if (t < dv) {
    float zz = zh[t];
    float value = v * inv * w[t] / (1.f + expf(-zz));
    oh[t] = value;
    if (Bf16) bf16[((size_t)blockIdx.y * gridDim.x + h) * dv + t] = f2bf(value);
  }
}

// --- GDN gated norm, warp-per-token (dv = 128 fixed): one wave32 per
// (token, head); lane owns dims lane, lane+32, lane+64, lane+96. Reduction
// reproduces the smem tree of k_gdn_gatednorm exactly: the off=64 layer
// pairs (d, d+64), off=32 adds the (d+32, d+96) pair, shfl_down 16..1 does
// the rest -> bit-exact, but one block covers 32 tokens instead of 1.
// grid (heads, ceil(P/32)), block 128 (4 warps); warp w takes tokens
// blockIdx.y*32 + k*4 + w, k = 0..7.
template <bool Bf16 = false>
__global__ void k_gdn_gatednorm_b(const float* __restrict__ y,
                                  const float* __restrict__ z,
                                  const float* __restrict__ w,
                                  float* __restrict__ out, float eps, int ys, int zs,
                                  uint16_t* __restrict__ bf16, int P) {
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float w0 = w[lane], w1 = w[lane + 32], w2 = w[lane + 64],
              w3 = w[lane + 96];
#pragma unroll 2
  for (int k = 0; k < 8; k++) {
    int t = blockIdx.y * 32 + k * 4 + warp;  // warp-uniform
    if (t >= P) return;
    const float* yh = y + (size_t)t * ys + h * 128;
    const float* zh = z + (size_t)t * zs + h * 128;
    float* oh = out ? out + (size_t)t * ys + h * 128 : nullptr;
    float v0 = yh[lane], v1 = yh[lane + 32], v2 = yh[lane + 64],
          v3 = yh[lane + 96];
    // Bit-exactness vs the smem tree: products must round individually before
    // the adds, but ffp-contract=fast would fuse mul+add into FMA (the old
    // kernel is saved by its smem round-trip). The empty asm barrier
    // materializes each product as a rounded VGPR value, no memory traffic.
    float q0 = v0 * v0, q1 = v1 * v1, q2 = v2 * v2, q3 = v3 * v3;
    asm volatile("" : "+v"(q0), "+v"(q1), "+v"(q2), "+v"(q3));
    float s64 = q0 + q2;
    float s96 = q1 + q3;
    float s = s64 + s96;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(~0ull, s, off);
    s = __shfl_sync(~0ull, s, 0);
    float inv = rsqrtf(s / 128 + eps);
    float r0 = v0 * inv * w0 / (1.f + expf(-zh[lane]));
    float r1 = v1 * inv * w1 / (1.f + expf(-zh[lane + 32]));
    float r2 = v2 * inv * w2 / (1.f + expf(-zh[lane + 64]));
    float r3 = v3 * inv * w3 / (1.f + expf(-zh[lane + 96]));
    if (out) {  // fp32 out is dead when the consumer GEMM reads only the bf16
      oh[lane] = r0;
      oh[lane + 32] = r1;
      oh[lane + 64] = r2;
      oh[lane + 96] = r3;
    }
    if (Bf16) {
      uint16_t* bp = bf16 + ((size_t)t * gridDim.x + h) * 128;
      bp[lane] = f2bf(r0);
      bp[lane + 32] = f2bf(r1);
      bp[lane + 64] = f2bf(r2);
      bp[lane + 96] = f2bf(r3);
    }
  }
}

// --- RoPE partial: rotate first rotary_dim dims of each head (half-split).
// vec: (nheads, dh). One warp per head... simple: one thread per pair.
// d_rdelta (M-RoPE decode): rope angle position is *d_pos + *d_rdelta;
// absent/zero for text-only requests (bitwise identical to before).
__global__ void k_rope(float* __restrict__ vec, int nheads, int dh, int rotary,
                       const int* __restrict__ d_pos, double theta,
                       const int* __restrict__ d_rdelta = nullptr) {
  int pos = *d_pos;
  if (d_rdelta) pos += *d_rdelta;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // over nheads * rotary/2
  int half = rotary / 2;
  if (idx >= nheads * half) return;
  int h = idx / half, i = idx % half;
  double ang = pos * pow(theta, -2.0 * i / rotary);
  float cs = (float)cos(ang), sn = (float)sin(ang);
  float* p = vec + h * dh;
  float x0 = p[i], x1 = p[i + half];
  p[i] = x0 * cs - x1 * sn;
  p[i + half] = x0 * sn + x1 * cs;
}

// --- QSA attention step: one block per q head, threads over positions.
// kcache/vcache: (pos, hkv*dh) accumulated; current token's k/v already appended.
template <typename KVT = float>
__global__ void k_qsa_step(const float* __restrict__ q, const KVT* __restrict__ kc,
                           const KVT* __restrict__ vc, float* __restrict__ out,
                           const int* __restrict__ d_pos, int dh, int hkv, int hq,
                           const int* selected = nullptr, bool skip_sparse = false,
                           const int* __restrict__ ptab = nullptr) {
  int visible = *d_pos + 1;
  bool sparse = selected && visible >= 2052;
  if (sparse && skip_sparse) return;
  int ntok = sparse ? 2048 + visible % 4 : visible;
  int h = blockIdx.x;
  int kvh = h / (hq / hkv);
  const float* qh = q + h * dh;
  float scale = 1.f / sqrtf((float)dh);
  // pass 1: scores + running max (threads strided over positions)
  int t = threadIdx.x;
  float mx = -1e30f;
  for (int p = t; p < ntok; p += blockDim.x) {
    const KVT* kp = kc + ((size_t)kv_row(ptab, qsa_source_pos(p, visible, selected)) * hkv + kvh) * dh;
    float acc = 0.f;
    for (int i = 0; i < dh; i++) acc += qh[i] * kv_ld1(kp + i);
    mx = fmaxf(mx, acc * scale);
  }
  // block max reduce
  __shared__ float red[1024];
  red[t] = mx;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) red[t] = fmaxf(red[t], red[t + off]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();
  // pass 2: exp sum
  float se = 0.f;
  for (int p = t; p < ntok; p += blockDim.x) {
    const KVT* kp = kc + ((size_t)kv_row(ptab, qsa_source_pos(p, visible, selected)) * hkv + kvh) * dh;
    float acc = 0.f;
    for (int i = 0; i < dh; i++) acc += qh[i] * kv_ld1(kp + i);
    se += expf(acc * scale - mx);
  }
  red[t] = se;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) red[t] += red[t + off];
    __syncthreads();
  }
  float invsum = 1.f / red[0];
  // pass 3: weighted v — threads over dh? need per-dim sums. dh=256, block 256.
  // recompute weights per position serially but parallel over dh.
  if (t < dh) {
    float acc = 0.f;
    for (int p = 0; p < ntok; p++) {
      int source = kv_row(ptab, qsa_source_pos(p, visible, selected));
      const KVT* kp = kc + ((size_t)source * hkv + kvh) * dh;
      float s = 0.f;
      for (int i = 0; i < dh; i++) s += qh[i] * kv_ld1(kp + i);
      float w = expf(s * scale - mx) * invsum;
      acc += w * kv_ld1(vc + ((size_t)source * hkv + kvh) * dh + t);
    }
    out[h * dh + t] = acc;
  }
}

// --- per-head zc rmsnorm for QSA q/k (dh=256, one block per head)
// (reuse k_rmsnorm_zc with n=256 per block over head pointer)

// --- sigmoid gate multiply: out[i] = a[i] * sigmoid(g[i])
__global__ void k_sigmoid_gate(const float* __restrict__ a, const float* __restrict__ g,
                               float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] / (1.f + expf(-g[i]));
}

// --- GR read: needs gemv results; combine on device.
// t = silu(Wd Rhat / 4) computed via gemv into t320 then k_silu_scale.
__global__ void k_silu_scale(float* __restrict__ v, float inv_div, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = v[i] * inv_div;
    v[i] = x / (1.f + expf(-x));
  }
}
__global__ void k_sigmoid_inplace(float* __restrict__ v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v[i] = 1.f / (1.f + expf(-v[i]));
}
// x_out[i] = mean_b G[b*d+i] * Rhat[b*d+i]
__global__ void k_gr_combine(const float* __restrict__ G, const float* __restrict__ Rhat,
                             float* __restrict__ x, int d, int branches) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  float acc = 0.f;
  for (int b = 0; b < branches; b++) acc += G[b * d + i] * Rhat[b * d + i];
  x[i] = acc / branches;
}
// GR write: s = 2*sigmoid(w/4); R[b*d+i] += s[b] * y[i]
__global__ void k_gr_write(const float* __restrict__ w4, float* __restrict__ R,
                           const float* __restrict__ y, int d, int branches) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  for (int b = 0; b < branches; b++) {
    float s = 2.f / (1.f + expf(-w4[b] * 0.25f));
    R[b * d + i] += s * y[i];
  }
}

// --- MoE router: softmax + top-k done on host (512 floats download is cheap),
//     experts gathered per selected expert via row-offset GEMV.
// silu(gate)*up for mid dims
__global__ void k_silu_mul(const float* __restrict__ g, const float* __restrict__ u,
                           float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float a = g[i];
    out[i] = a / (1.f + expf(-a)) * u[i];
  }
}
// acc[r] += w * y[r]
__global__ void k_axpy(float* __restrict__ acc, const float* __restrict__ y, float w,
                       int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) acc[i] += w * y[i];
}

// --- GPU router: softmax + top-k over n logits, one block. ------------------
// out: ids[k], w[k] with w = p_e / sum(selected p) (official norm_topk_prob).
// Batched: one block per token, ids/ws rows of 16.
__global__ void k_router_topk(const float* __restrict__ logits, int* __restrict__ ids,
                              float* __restrict__ ws, int n, int k) {
  logits += (size_t)blockIdx.x * n;
  ids += (size_t)blockIdx.x * 16;
  ws += (size_t)blockIdx.x * 16;
  __shared__ float p[512];
  __shared__ float sval[512];
  __shared__ int sidx[512];
  __shared__ float raw[16];
  int t = threadIdx.x;
  float v = logits[t];  // n == 512 == blockDim
  // block max
  sval[t] = v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sval[t] = fmaxf(sval[t], sval[t + off]);
    __syncthreads();
  }
  float mx = sval[0];
  p[t] = expf(v - mx);
  __syncthreads();
  sval[t] = p[t];
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sval[t] += sval[t + off];
    __syncthreads();
  }
  float inv_se = 1.f / sval[0];
  p[t] *= inv_se;
  __syncthreads();
  for (int j = 0; j < k; j++) {
    sval[t] = p[t];
    sidx[t] = t;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
      if (t < off && sval[t + off] > sval[t]) {
        sval[t] = sval[t + off];
        sidx[t] = sidx[t + off];
      }
      __syncthreads();
    }
    if (t == 0) {
      ids[j] = sidx[0];
      raw[j] = sval[0];
      p[sidx[0]] = -1e30f;
    }
    __syncthreads();
  }
  if (t == 0) {
    float sw = 0.f;
    for (int j = 0; j < k; j++) sw += raw[j];
    for (int j = 0; j < k; j++) ws[j] = raw[j] / sw;
  }
}

// --- grouped expert GEMV (shared x): slot-major y[(slot, r)] ---------------
// Batched: nslots = P*kpt; slot -> token tok = slot/kpt, expert s = slot%kpt;
// x = xs + tok*x_stride, expert id = ids[tok*id_stride + s].
// v3 (decode GEMV governance): cb in LDS + 2 rows per warp; bit-exact vs the
// original (same per-lane fma order and reduction). Grid must now cover
// nslots * ceil(rows_per/2) warps, not nslots * rows_per.
__global__ void k_q4cp_gemv_gg(const uint8_t* __restrict__ codes,
                               const uint8_t* __restrict__ scales,
                               const float* __restrict__ cb,
                               const float* __restrict__ x, float* __restrict__ y,
                               const int* __restrict__ ids, uint64_t rows_per,
                               uint64_t cols, uint64_t scale_stride, int nslots,
                               uint64_t x_stride, int id_stride, int kpt) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t pairs_per_slot = (rows_per + 1) / 2;
  uint64_t gw = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (gw >= (uint64_t)nslots * pairs_per_slot) return;
  int slot = gw / pairs_per_slot;
  int tok = slot / kpt, s = slot - tok * kpt;
  uint64_t row0 =
      (uint64_t)ids[tok * id_stride + s] * rows_per + (gw % pairs_per_slot) * 2;
  int lane = threadIdx.x % warpSize;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  bool has1 = (gw % pairs_per_slot) * 2 + 1 < rows_per;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  const float* xt = x + (size_t)tok * x_stride;
  float acc0 = 0.f, acc1 = 0.f;
  for (uint64_t g = lane; g < cols / 32; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + g * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + g * 2));
    }
    const float* xg = xt + g * 32;
    float xv[32];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float4 t = *(const float4*)(xg + i * 4);
      xv[i * 4] = t.x; xv[i * 4 + 1] = t.y; xv[i * 4 + 2] = t.z;
      xv[i * 4 + 3] = t.w;
    }
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t v0 = w0[i], v1 = w1[i];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        a0 += cbf[v0 & 0xf] * xv[i * 8 + j];
        a1 += cbf[v1 & 0xf] * xv[i * 8 + j];
        v0 >>= 4; v1 >>= 4;
      }
    }
    acc0 += a0 * s0;
    acc1 += a1 * s1;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc0 += __shfl_down_sync(~0ull, acc0, off);
    acc1 += __shfl_down_sync(~0ull, acc1, off);
  }
  if (lane == 0) {
    uint64_t out = (uint64_t)slot * rows_per + (gw % pairs_per_slot) * 2;
    y[out] = acc0;
    if (has1) y[out + 1] = acc1;
  }
}

// --- grouped silu(gate)*up: guv (nslots, 2*mid) -> hid (nslots, mid) --------
__global__ void k_silu_mul_g(const float* __restrict__ guv, float* __restrict__ hid,
                             int mid, int nslots) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nslots * mid) return;
  int slot = i / mid, r = i % mid;
  float a = guv[slot * 2 * mid + r];
  hid[i] = a / (1.f + expf(-a)) * guv[slot * 2 * mid + mid + r];
}

// --- grouped expert GEMV (per-slot x) with fused weighted accumulate --------
// acc[tok*acc_stride + r] += ws[tok*id_stride+s] * dot(W[ids[...]*rows_per + r],
// xs + slot*x_stride). v3: full warp per row, 2 rows per warp, cb in LDS;
// per-row math bit-exact vs the LPW=32 original on wave32 (the atomicAdd set
// per output element is unchanged). Grid must cover nslots*ceil(rows_per/2)
// warps.
__global__ void k_q4cp_gemv_gd(const uint8_t* __restrict__ codes,
                               const uint8_t* __restrict__ scales,
                               const float* __restrict__ cb,
                               const float* __restrict__ xs, float* __restrict__ acc,
                               const int* __restrict__ ids, const float* __restrict__ ws,
                               uint64_t rows_per, uint64_t cols, uint64_t scale_stride,
                               uint64_t x_stride, int nslots, int id_stride,
                               uint64_t acc_stride, int kpt) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t pairs_per_slot = (rows_per + 1) / 2;
  int lane = threadIdx.x % warpSize;
  uint64_t gw = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (gw >= (uint64_t)nslots * pairs_per_slot) return;
  int slot = gw / pairs_per_slot;
  uint64_t rp = gw % pairs_per_slot;
  int tok = slot / kpt, s = slot - tok * kpt;
  uint64_t row0 = (uint64_t)ids[tok * id_stride + s] * rows_per + rp * 2;
  const float* x = xs + (uint64_t)slot * x_stride;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  bool has1 = rp * 2 + 1 < rows_per;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  float acc0 = 0.f, acc1 = 0.f;
  for (uint64_t g = lane; g < cols / 32; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + g * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + g * 2));
    }
    const float* xg = x + g * 32;
    float xv[32];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float4 t = *(const float4*)(xg + i * 4);
      xv[i * 4] = t.x; xv[i * 4 + 1] = t.y; xv[i * 4 + 2] = t.z;
      xv[i * 4 + 3] = t.w;
    }
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t v0 = w0[i], v1 = w1[i];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        a0 += cbf[v0 & 0xf] * xv[i * 8 + j];
        a1 += cbf[v1 & 0xf] * xv[i * 8 + j];
        v0 >>= 4; v1 >>= 4;
      }
    }
    acc0 += a0 * s0;
    acc1 += a1 * s1;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc0 += __shfl_down_sync(~0ull, acc0, off);
    acc1 += __shfl_down_sync(~0ull, acc1, off);
  }
  if (lane == 0) {
    float w = ws[tok * id_stride + s];
    atomicAdd(acc + (size_t)tok * acc_stride + rp * 2, w * acc0);
    if (has1) atomicAdd(acc + (size_t)tok * acc_stride + rp * 2 + 1, w * acc1);
  }
}

// Small-P verify down projection: one wave covers two output rows and keeps
// all top-10 expert partials in registers. Two 16-lane subgroups avoid the
// 12 idle lanes of the K=640 full-wave mapping and eliminate atomic output
// accumulation. This is intentionally limited to topk=10 by the host route.
template <int KPT>
__global__ void k_q4cp_gemv_gd_topk_h16(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
    const float* __restrict__ cb, const float* __restrict__ xs,
    float* __restrict__ out, const int* __restrict__ ids,
    const float* __restrict__ ws, uint64_t rows_per, uint64_t cols,
    uint64_t scale_stride, uint64_t x_stride, int P, int id_stride,
    uint64_t out_stride) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t row_pairs = (rows_per + 1) / 2;
  uint64_t gw =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (gw >= (uint64_t)P * row_pairs) return;
  int tok = gw / row_pairs;
  uint64_t rp = gw % row_pairs;
  int half = (threadIdx.x % warpSize) / 16;
  int lane = threadIdx.x & 15;
  uint64_t r = rp * 2 + half;
  if (r >= rows_per) return;
  float acc[KPT] = {};
#pragma unroll
  for (int s = 0; s < KPT; s++) {
    uint64_t row = (uint64_t)ids[tok * id_stride + s] * rows_per + r;
    const uint8_t* rc = codes + row * (cols / 2);
    const uint8_t* rs = scales + row * scale_stride;
    const float* x = xs + ((size_t)tok * KPT + s) * x_stride;
    for (uint64_t g = lane; g < cols / 32; g += 16) {
      uint4 pk = *(const uint4*)(rc + g * 16);
      float sc = __half2float(*(const __half*)(rs + g * 2));
      const float* xg = x + g * 32;
      float xv[32];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        float4 t = *(const float4*)(xg + i * 4);
        xv[i * 4] = t.x;
        xv[i * 4 + 1] = t.y;
        xv[i * 4 + 2] = t.z;
        xv[i * 4 + 3] = t.w;
      }
      uint32_t w[4] = {pk.x, pk.y, pk.z, pk.w};
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        uint32_t v = w[i];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          a += cbf[v & 0xf] * xv[i * 8 + j];
          v >>= 4;
        }
      }
      acc[s] += a * sc;
    }
  }
#pragma unroll
  for (int s = 0; s < KPT; s++)
    for (int off = 8; off > 0; off >>= 1)
      acc[s] += __shfl_down_sync(~0ull, acc[s], off, 16);
  if (lane == 0) {
    float y = 0.f;
#pragma unroll
    for (int s = 0; s < KPT; s++) y += ws[tok * id_stride + s] * acc[s];
    out[(size_t)tok * out_stride + r] = y;
  }
}

// --- PLE conv (dilation 3, k=4): ring 9 x 10240; tap j pairs t-(3-j)*3
__global__ void k_ple_conv(const float* __restrict__ ring,
                           const int* __restrict__ d_ringpos,
                           const float* __restrict__ cw, float* __restrict__ out,
                           int n) {
  int ringpos = *d_ringpos;
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= n) return;
  float a = 0.f;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int back = (3 - j) * 3;
    int rp = ((ringpos - back) % 9 + 9) % 9;
    a += cw[ch * 4 + j] * ring[(size_t)rp * n + ch];
  }
  out[ch] = a / (1.f + expf(-a));  // silu
}

// grouped zc rmsnorm into PLE ring slot *d_ringpos (ring = 9 x 10240)
__global__ void k_rmsnorm_zc_ring(const float* __restrict__ x, const float* __restrict__ w,
                                  float* __restrict__ ring,
                                  const int* __restrict__ d_ringpos, int gsize,
                                  float eps) {
  float* out = ring + (size_t)(*d_ringpos) * 10240;
  const float* xg = x + (size_t)blockIdx.x * gsize;
  const float* wg = w + (size_t)blockIdx.x * gsize;
  float* og = out + (size_t)blockIdx.x * gsize;
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < gsize; i += blockDim.x) ss += xg[i] * xg[i];
  sm[t] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / gsize + eps);
  for (int i = t; i < gsize; i += blockDim.x) og[i] = xg[i] * inv * (1.f + wg[i]);
}

// --- QSA kv cache append at *d_pos (replaces pos-dependent memcpy) ----------
__global__ void k_store_kv(const float* __restrict__ kb, const float* __restrict__ vb,
                           float* __restrict__ kc, float* __restrict__ vc,
                           const int* __restrict__ d_pos,
                           const int* __restrict__ ptab = nullptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 512) return;
  size_t off = (size_t)kv_row(ptab, *d_pos) * 512 + i;
  kc[off] = kb[i];
  vc[off] = vb[i];
}

// BF16 variant of the above: same layout, values rounded RNE via f2bf.
// vct (optional, GDEC_QSA_WMMA_BTV): also write V into the block-transposed
// prefill cache [block][kvh][dim][slot] (layout of k_f32_to_bf16_v4_bt, with
// i = kvh*256 + dim), so later sparse prefill / verify reads of decoded
// positions see the same bits as the row-major cache.
// ptab (A1c, optional): paged addressing. The physical row is kv_row(pos) and
// the physical vct block is kv_row(pos) >> 2 == ik_blk(pos >> 2) (a page holds
// whole 4-token blocks), slot pos & 3 unchanged.
__global__ void k_store_kv_bf16(const float* __restrict__ kb,
                                const float* __restrict__ vb,
                                uint16_t* __restrict__ kc, uint16_t* __restrict__ vc,
                                const int* __restrict__ d_pos,
                                uint16_t* __restrict__ vct = nullptr,
                                const int* __restrict__ ptab = nullptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 512) return;
  const int pos = *d_pos;
  const int row = kv_row(ptab, pos);
  size_t off = (size_t)row * 512 + i;
  kc[off] = f2bf(kb[i]);
  const uint16_t v = f2bf(vb[i]);
  vc[off] = v;
  if (vct) vct[((size_t)(row >> 2) * 512 + i) * 4 + (pos & 3)] = v;
}

// --- end-of-step: pos++, ringpos = (ringpos+1)%9 ----------------------------
__global__ void k_step_incr(int* __restrict__ d_pos, int* __restrict__ d_ringpos) {
  *d_pos += 1;
  *d_ringpos = (*d_ringpos + 1) % 9;
}
// R[ch] += U[ch] + convout[ch]
__global__ void k_add2(float* __restrict__ R, const float* __restrict__ a,
                       const float* __restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) R[i] += a[i] + b[i];
}
__global__ void k_acc(float* __restrict__ acc, const float* __restrict__ a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) acc[i] += a[i];
}
// Copy P contiguous d-rows into the heads of 4d-stride rows (k_repeat4_b then
// expands them in place).
__global__ void k_row_head_copy(const float* __restrict__ x,
                                float* __restrict__ R, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // over P * d
  int t = i / d, c = i - t * d;
  R[(size_t)t * 4 * d + c] = x[(size_t)t * d + c];
}
// Copy P contiguous d-rows into branch slot b of 4d-stride rows.
__global__ void k_row_branch_copy(const float* __restrict__ x,
                                  float* __restrict__ R, int d, int b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // over P * d
  int t = i / d, c = i - t * d;
  R[(size_t)t * 4 * d + (size_t)b * d + c] = x[(size_t)t * d + c];
}
// Broadcast-add P contiguous d-rows onto all four branches of 4d-stride rows.
__global__ void k_acc_broadcast4(const float* __restrict__ x,
                                 float* __restrict__ R, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // over P * d
  int t = i / d, c = i - t * d;
  float v = x[(size_t)t * d + c];
  size_t base = (size_t)t * 4 * d + c;
  R[base] += v;
  R[base + d] += v;
  R[base + 2 * d] += v;
  R[base + 3 * d] += v;
}
// PLE per-branch gate: g_m = sum(norm_k_m * norm_q_m)/sqrt(d) -> signed sqrt ->
// sigmoid; U[m*d+i] = alpha_m * v[i]
__global__ void k_ple_gate(const float* __restrict__ kn, const float* __restrict__ qn,
                           const float* __restrict__ v, float* __restrict__ U, int d,
                           int branches) {
  int m = blockIdx.x;
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float acc = 0.f;
  for (int i = t; i < d; i += blockDim.x) acc += kn[m * d + i] * qn[m * d + i];
  sm[t] = acc;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float g = sm[0] / sqrtf((float)d);
  g = (g < 0 ? -1.f : 1.f) * sqrtf(fmaxf(fabsf(g), 1e-6f));
  float alpha = 1.f / (1.f + expf(-g));
  for (int i = t; i < d; i += blockDim.x) U[m * d + i] = alpha * v[i];
}


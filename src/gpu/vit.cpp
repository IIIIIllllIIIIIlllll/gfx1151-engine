// vit.cpp — 阶段 3:Qwen4-Exp 视觉塔 ViT 独立对拍二进制(gfx1151)。
// 只加载 vision sidecar(全部 bf16),读 data/vision-ref/ 的 patches.npy
// 跑前向,逐层 dump 与 tools/vision_ref.py --dump-layers 的基准对拍。
//
// 数值约定(与 reference/modeling_qwen4_exp.py 对齐):
//   - 残差流 fp32;GEMM 输入转 fp16(RNE),fp32 累加(hipBLASLt)。
//     (初版用 bf16 I/O,27 层漂移 merged cos≈0.9999 不达 0.99999 目标;
//     tools/vit_emu.py 仿真证明 fp16 I/O 同结构 merged cos≈0.9999986,
//     权重 bf16→fp16 在正常幅值范围精确,activation 最大值 ~5.7e3 ≪ 65504。)
//   - LN eps=1e-6(fp32);MLP GELU = tanh 近似;merger GELU = 精确 erf。
//   - rope:2D (h,w) block-major 位置,theta=10000,dim=36,rotate_half,
//     host 侧 fp64 算 inv_freq/sin/cos 后 cast fp32,device fp32 应用。
//   - 注意力:非因果全量,scaling = 72^-0.5(alpha 折进 scores GEMM),
//     scores fp32 → softmax fp32 → probs 转 fp16 → P@V。
//   - pos_embed:48x48 表按 (68,120) 双线性 align_corners=True,host 算
//     4-tap 索引/权重,device gather + 加权加进残差。
//
// Build: hipcc -O3 -o vit vit.cpp -lrocblas -lhipblaslt
// Usage: vit <sidecar.hgn> --patches x.npy --grid-thw g.npy --out-prefix D/N
//            [--blocks N] [--dump-layers] [--quiet]

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <vector>

#include "../hgn.h"

#define CK(x)                                                            \
  do {                                                                   \
    hipError_t e_ = (x);                                                 \
    if (e_ != hipSuccess) {                                              \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),  \
              __FILE__, __LINE__);                                       \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

static hipStream_t g_str;
static hipblasLtHandle_t lth;
static rocblas_handle rbh;
static void* d_ltws = nullptr;
static const size_t LT_WS = (size_t)64 << 20;

// ----------------------------- npy 读写 ----------------------------------
struct Npy {
  std::vector<uint64_t> shape;
  char descr[8] = {0};  // "<f4" / "<i8" / "<u2"
  std::vector<uint8_t> data;
  uint64_t numel() const {
    uint64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

static Npy npy_read(const char* path) {
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
  Npy r;
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
  uint64_t esz = 0;
  if (!strcmp(r.descr, "<f4") || !strcmp(r.descr, "<i4") || !strcmp(r.descr, "<u4"))
    esz = 4;
  else if (!strcmp(r.descr, "<i8"))
    esz = 8;
  else if (!strcmp(r.descr, "<u2"))
    esz = 2;
  else
    throw std::runtime_error(std::string("npy dtype ") + r.descr);
  r.data.resize(r.numel() * esz);
  if (fread(r.data.data(), 1, r.data.size(), f) != r.data.size())
    throw std::runtime_error("npy data short");
  fclose(f);
  return r;
}

static void npy_write_f32(const char* path, const float* p, uint64_t rows,
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

static void npy_write_u16(const char* path, const uint16_t* p, uint64_t rows,
                          uint64_t cols) {
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

// ----------------------------- hipBLASLt GEMM -----------------------------
// col-major C(M×N) = op(A)op(B),A/B fp16,C fp32(c16=false)或 fp16,fp32 累加。
// 行-major 视角:投影 Y(P×N)=X(P×K)@W(N×K)^T 用
//   lt_gemm(OP_T, OP_N, M=N, N=P, K, A=W(lda=K), B=X(ldb=K), C=Y(ldc=N))
using LtKey = std::tuple<int, int, int, int, int, int, int, int, int>;
struct LtEntry {
  hipblasLtMatmulDesc_t opd;
  hipblasLtMatrixLayout_t Al, Bl, Cl, Dl;
  hipblasLtMatmulAlgo_t algo;
};
static std::map<LtKey, LtEntry> lt_cache;

static void lt_gemm(hipblasOperation_t opA, hipblasOperation_t opB, int M, int N,
                    int K, const uint16_t* A, int lda,
                    const uint16_t* B, int ldb, void* C, int ldc,
                    float alpha, float beta, bool c16 = false) {
  LtKey key{(int)opA, (int)opB, M, N, K, lda, ldb, ldc, (int)c16};
  auto it = lt_cache.find(key);
  if (it == lt_cache.end()) {
    LtEntry e;
    hipblasLtMatmulDescCreate(&e.opd, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSA, &opA,
                                    sizeof(opA));
    hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSB, &opB,
                                    sizeof(opB));
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
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &LT_WS, sizeof(LT_WS));
    hipblasLtMatmulHeuristicResult_t hres[16];
    int nres = 0;
    hipblasLtMatmulAlgoGetHeuristic(lth, e.opd, e.Al, e.Bl, e.Cl, e.Dl, pref, 16,
                                    hres, &nres);
    float one = 1.f, zero = 0.f;
    int best = -1;
    double best_ms = 1e30;
    for (int i = 0; i < nres; i++) {
      if (hres[i].state != HIPBLAS_STATUS_SUCCESS) continue;
      if (hres[i].workspaceSize > LT_WS) continue;
      hipblasStatus_t st = hipblasLtMatmul(lth, e.opd, &one, A, e.Al, B, e.Bl,
                                           &zero, C, e.Cl, C, e.Dl, &hres[i].algo,
                                           d_ltws, LT_WS, g_str);
      if (st != HIPBLAS_STATUS_SUCCESS || hipStreamSynchronize(g_str) != hipSuccess) {
        (void)hipGetLastError();
        continue;
      }
      auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < 3; r++)
        hipblasLtMatmul(lth, e.opd, &one, A, e.Al, B, e.Bl, &zero, C, e.Cl, C,
                        e.Dl, &hres[i].algo, d_ltws, LT_WS, g_str);
      hipStreamSynchronize(g_str);
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
      if (ms < best_ms) {
        best_ms = ms;
        best = i;
      }
    }
    static const bool lt_dbg = getenv("VIT_LT_DBG") != nullptr;
    if (lt_dbg)
      fprintf(stderr,
              "[lt] opA=%d opB=%d M=%d N=%d K=%d lda=%d ldb=%d ldc=%d c16=%d "
              "nres=%d best=%d %.3fms\n",
              (int)opA, (int)opB, M, N, K, lda, ldb, ldc, (int)c16, nres, best,
              best_ms / 3);
    if (best < 0) {
      // rocBLAS 兜底(仅 fp32 C)
      if (c16) {
        fprintf(stderr, "gemm c16 no LT algo M=%d N=%d K=%d\n", M, N, K);
        exit(1);
      }
      rocblas_status st = rocblas_gemm_ex(
          rbh, (rocblas_operation)opA, (rocblas_operation)opB, M, N, K, &one, A,
          rocblas_datatype_f16_r, lda, B, rocblas_datatype_f16_r, ldb, &zero, C,
          rocblas_datatype_f32_r, ldc, C, rocblas_datatype_f32_r, ldc,
          rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
      if (st != rocblas_status_success) {
        fprintf(stderr, "gemm failed M=%d N=%d K=%d (no LT algo, rb %d)\n", M, N,
                K, (int)st);
        exit(1);
      }
      hipblasLtMatmulDescDestroy(e.opd);
      hipblasLtMatrixLayoutDestroy(e.Al);
      hipblasLtMatrixLayoutDestroy(e.Bl);
      hipblasLtMatrixLayoutDestroy(e.Cl);
      hipblasLtMatrixLayoutDestroy(e.Dl);
      // 不缓存,形状少,每次走 rb。
      return;
    }
    e.algo = hres[best].algo;
    it = lt_cache.emplace(key, e).first;
  }
  hipblasLtMatmul(lth, it->second.opd, &alpha, A, it->second.Al, B, it->second.Bl,
                  &beta, C, it->second.Cl, C, it->second.Dl, &it->second.algo,
                  d_ltws, LT_WS, g_str);
}

// 行-major 投影:Y(P×N) f32 = X(P×K) fp16 @ W(N×K) fp16^T
static void proj_gemm(const uint16_t* X, const uint16_t* W, float* Y,
                      int P, int N, int K) {
  lt_gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, N, P, K, W, K, X, K, Y, N, 1.f, 0.f);
}

// split 版:Y = W@(hi+lo),两次 GEMM,第二次 beta=1 累加
static void proj_gemm_split(const uint16_t* hi, const uint16_t* lo,
                            const uint16_t* W, float* Y, int P, int N, int K) {
  lt_gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, N, P, K, W, K, hi, K, Y, N, 1.f, 0.f);
  lt_gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, N, P, K, W, K, lo, K, Y, N, 1.f, 1.f);
}

// ----------------------------- kernels -------------------------------------
__device__ inline uint16_t f2h(float f) {  // fp32→fp16 RNE
  return __half_as_ushort(__float2half_rn(f));
}
__device__ inline float h2f(uint16_t h) {
  return __half2float(__ushort_as_half(h));
}

__global__ void k_f32_to_fp16(const float* x, uint16_t* y, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = f2h(x[i]);
}

// split-fp16:x = hi + lo(lo 保留 hi 舍掉的残差,两路 GEMM 累加 ≈ fp32 输入
// 精度)。归因结论(landscape,tools/vit_emu.py):patch_embed 输入 fp16
// 舍入经 27 层放大贡献几乎全部漂移;fc1/fc2 次之;qkv/proj/merger/probs
// 单 fp16 即可(合并 cos 预测 0.9999989)。
__global__ void k_f32_split_fp16(const float* x, uint16_t* hi, uint16_t* lo,
                                 uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float h = h2f(f2h(x[i]));
  hi[i] = f2h(x[i]);
  lo[i] = f2h(x[i] - h);
}

__global__ void k_bias_add(float* y, const float* b, int P, int N) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (uint64_t)P * N) y[i] += b[i % N];
}

// h += y + bias (残差融合)
__global__ void k_bias_resadd(float* h, const float* y, const float* b, int P,
                              int N) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (uint64_t)P * N) h[i] += y[i] + b[i % N];
}

// 一行一个 block,fp32,eps=1e-6
__global__ void k_layernorm(const float* x, float* y, const float* w,
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

// pos_embed gather + 4-tap 加权,加进残差 h(P×1152 f32)
__global__ void k_pos_embed(float* h, const uint16_t* table,
                            const int* idx, const float* wts, int P) {
  int p = blockIdx.x;
  for (int d = threadIdx.x; d < 1152; d += blockDim.x) {
    float acc = 0;
    for (int t = 0; t < 4; t++)
      acc += wts[p * 4 + t] *
             h2f(table[(uint64_t)idx[p * 4 + t] * 1152 + d]);
    h[(uint64_t)p * 1152 + d] += acc;
  }
}

// rope 应用到 q,k(fp32 d_qkv (P,3456)),输出 fp16 q/k/v (P,1152)。
// rotate_half:d<36 与 d+36 配对;cos/sin (P,72) 广播到头维。
__global__ void k_rope_qkv(const float* qkv, const float* cos_t,
                           const float* sin_t, uint16_t* qb,
                           uint16_t* kb, uint16_t* vb, int P) {
  // 每线程处理一个 (p, head, pair d<36)
  int p = blockIdx.x;
  int h = blockIdx.y;
  int d = threadIdx.x;  // 36 threads
  if (d >= 36) return;
  const float* qr = qkv + (uint64_t)p * 3456 + h * 72;
  const float* kr = qr + 1152;
  const float* vr = qr + 2304;
  const float* cr = cos_t + (uint64_t)p * 72;  // cos/sin 不按头变
  const float* sr = sin_t + (uint64_t)p * 72;
  float c0 = cr[d], s0 = sr[d], c1 = cr[d + 36], s1 = sr[d + 36];
  float q0 = qr[d], q1 = qr[d + 36];
  float k0 = kr[d], k1 = kr[d + 36];
  uint64_t o = (uint64_t)p * 1152 + h * 72;
  qb[o + d] = f2h(q0 * c0 - q1 * s0);
  qb[o + d + 36] = f2h(q1 * c1 + q0 * s1);
  kb[o + d] = f2h(k0 * c0 - k1 * s0);
  kb[o + d + 36] = f2h(k1 * c1 + k0 * s1);
  if (d < 36) {  // v 直接转 fp16(两个半维都由 36 线程写)
    vb[o + d] = f2h(vr[d]);
    vb[o + d + 36] = f2h(vr[d + 36]);
  }
}

// softmax:一行一个 block,读 fp16 scores,输出 fp16 probs(exp/max/sum 均 fp32)
__global__ void k_softmax_fp16(const uint16_t* scores, uint16_t* probs,
                               int P) {
  int row = blockIdx.x;
  const uint16_t* sr = scores + (uint64_t)row * P;
  uint16_t* pr = probs + (uint64_t)row * P;
  __shared__ float red[256];
  float m = -INFINITY;
  for (int i = threadIdx.x; i < P; i += blockDim.x) m = fmaxf(m, h2f(sr[i]));
  red[threadIdx.x] = m;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
    __syncthreads();
  }
  float mx = red[0];
  float s = 0;
  for (int i = threadIdx.x; i < P; i += blockDim.x) s += expf(h2f(sr[i]) - mx);
  __syncthreads();
  red[threadIdx.x] = s;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  float inv = 1.f / red[0];
  for (int i = threadIdx.x; i < P; i += blockDim.x)
    pr[i] = f2h(expf(h2f(sr[i]) - mx) * inv);
}

using fw_shortx16 = __attribute__((ext_vector_type(16))) short;
using fw_floatx8 = __attribute__((ext_vector_type(8))) float;
__device__ __forceinline__ fw_shortx16 fw_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  fw_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}

// WMMA fp16 flash 注意力(非因果,单段):grid (ceil(P/128), 16 头),block 256
// = 8 warps,warp w 拥有 q 行 [w*16,+16)。gfx1151 fragment 布局(引擎
// gdec.cpp k_qsa_wmma 实证注释):
//   A(r,k): r<8 -> lane 2r elem k; r>=8 -> lane 17+2(r-8) elem k (rest ignored)
//   B(k,c): lane c elem k AND lane 16+c elem k (duplicate)
//   C(r,c): lane c + 16*(r>=8), elem r%8
// LDS: Q 128×88 fp16(d≥72 置 0),K/V 64×88,Pbuf 128×72(C 布局写入,A 布局
// 读回)。在线 softmax fp32;P 量化 fp16(与朴素路径 fp16 probs 同精度)。
// 原型验证:/tmp/flash_wmma_test.cu,P=200/1000/8160 cos≥0.99999991 vs fp64。
__global__ void __launch_bounds__(256)
k_flash_wmma(const uint16_t* __restrict__ qg, const uint16_t* __restrict__ kg,
             const uint16_t* __restrict__ vg, float* __restrict__ out, int P,
             float scale) {
  __shared__ uint16_t Qs[128][88], Ks[64][88], Vs[64][88], Pbuf[128][72];
  const int r0 = blockIdx.x * 128;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31, w = tid >> 5;
  const int colbase = lane & 15;
  for (int i = tid; i < 128 * 44; i += 256) {
    int rr = i / 44, jj = i % 44;  // 44 uint32/行(88 fp16)
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
  fw_floatx8 acc[5];
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
    __syncthreads();  // 保护上轮 Pbuf/Vs 读取
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
    // scores:S(16 行 × 64 keys) = Q·Kᵀ,4 key-frag × 5 d-frag
    fw_floatx8 s[4];
#pragma unroll
    for (int f = 0; f < 4; f++) s[f] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int kk = 0; kk < 5; kk++) {
      fw_shortx16 a = {0};
      if (qrowA >= 0) a = fw_ld16(&Qs[w * 16 + qrowA][kk * 16]);
#pragma unroll
      for (int f = 0; f < 4; f++) {
        short bt[16];
        int krow = f * 16 + colbase;
#pragma unroll
        for (int k = 0; k < 16; k++)
          bt[k] = (short)Ks[krow][kk * 16 + k];
        fw_shortx16 b;
        memcpy(&b, bt, 32);
        s[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s[f]);
      }
    }
    // scale + mask + 在线 softmax(C 布局:lane l 持行 e(l<16)/8+e,列 l%16)
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
        float p = expf(s[f][e] - mnew);  // s=-1e30 → 0
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
    // P 量化 fp16 → Pbuf(C 布局)
#pragma unroll
    for (int f = 0; f < 4; f++)
#pragma unroll
      for (int e = 0; e < 8; e++)
        Pbuf[w * 16 + (lane < 16 ? e : 8 + e)][f * 16 + colbase] =
            f2h(s[f][e]);
    __syncthreads();
    // PV:O(16×80) += P(16×64)·V(64×80),4 key-frag × 5 dim-frag
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      fw_shortx16 a = {0};
      if (qrowA >= 0) a = fw_ld16(&Pbuf[w * 16 + qrowA][kk * 16]);
#pragma unroll
      for (int dc = 0; dc < 5; dc++) {
        short bt[16];
#pragma unroll
        for (int k = 0; k < 16; k++)
          bt[k] = (short)Vs[kk * 16 + k][dc * 16 + colbase];
        fw_shortx16 b;
        memcpy(&b, bt, 32);
        acc[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[dc]);
      }
    }
  }
  // epilogue:lane l 写行 e/8+e,列 dc*16 + l%16
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

// k-tile 双缓冲版 k_flash_wmma(2026-09-13):同样的 grid/数值口径,
// 区别在 K/V DRAM→LDS 流水与 V 的 LDS 布局。LDS 预算(64KB 静态上限,
// 现用量 63104B)装不下第二份 K/V tile,双缓冲用寄存器暂存实现:iter kt
// 顶部先把 tile kt+1 的全局读发射进 24 个 uint4 寄存器(64 行 × 9 uint4 /
// 256 线程),scores+softmax+PV 的全部计算与读重叠;Ks 在 S1 后(scores
// 已读完)写入,VsT 在 S2 后(PV 已读完)写入。同步纪律:
//   S1:scores 读完 Ks + Pbuf 写可见;S2:PV 读完 VsT/Pbuf + Ks 写可见。
//   VsT 写在 S2 之后、下轮 S1 之前,下轮 PV 在 S1 后,天然可见(省第三同步)。
// V 以转置 VsT[80][68] 存 LDS(store 侧 uint4 拆 8 标量):PV 的 B-gather
// 从 320 条标量 ds_load_u16 变成 40 条 ds_load_2addr_b64(SASS 实证),这是
// db→dbvt 25.6→21.3ms 的主要来源;行距 68 fp16 免 bank 冲突(132B 不行
// b128,但 b64 已 4 倍于标量)。全局读 uint32 对升级为 uint4(9 个/行)。
// softmax expf→__expf(v_exp_f32 单指令):probs 下游量化 fp16,精度影响
// 淹没在 fp16 舍入内(三图 cmp 验证,见 HANDOVER_2026-09-13-VIT.md)。
// 其余数值路径与 v1 同:fp16 输入 fp32 累加,online softmax 归约顺序不变。
// 微基准(tools/vit_attn_bench.cu,P=8160):v1 32.0ms → dbvt_x ~19.7ms/层。
__global__ void __launch_bounds__(256)
k_flash_wmma_db(const uint16_t* __restrict__ qg, const uint16_t* __restrict__ kg,
                const uint16_t* __restrict__ vg, float* __restrict__ out, int P,
                float scale) {
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
  for (int i = tid; i < 64 * 2; i += 256) {  // K 行 padding 列 [72,88) 置 0
    int rr = i >> 1, c9 = 9 + (i & 1);
    ((uint4*)Ks[rr])[c9] = uint4{0, 0, 0, 0};
  }
  for (int i = tid; i < 80 * 68; i += 256)  // VsT 全零一次(padding 行/列)
    ((uint16_t*)VsT)[i] = 0;
  // 每线程 3 个 uint4 槽(64 行 × 9 = 576 = 2×256 + 64)
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
  auto storeV = [&](const uint4* r) {  // 转置写:VsT[dim][key]
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
  fw_floatx8 acc[5];
#pragma unroll
  for (int dc = 0; dc < 5; dc++) acc[dc] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int e = 0; e < 8; e++) {
    m_run[e] = -1e30f;
    l_run[e] = 0.f;
  }
  for (int kt = 0; kt < nkt; kt++) {
    const int c0 = kt * 64;
    if (kt + 1 < nkt) stage(c0 + 64, kn, vn);  // 预取下 tile,与计算重叠
    // scores:S(16 行 × 64 keys) = Q·Kᵀ,4 key-frag × 5 d-frag
    fw_floatx8 s[4];
#pragma unroll
    for (int f = 0; f < 4; f++) s[f] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int kk = 0; kk < 5; kk++) {
      fw_shortx16 a = {0};
      if (qrowA >= 0) a = fw_ld16(&Qs[w * 16 + qrowA][kk * 16]);
#pragma unroll
      for (int f = 0; f < 4; f++) {
        short bt[16];
        int krow = f * 16 + colbase;
#pragma unroll
        for (int k = 0; k < 16; k++)
          bt[k] = (short)Ks[krow][kk * 16 + k];
        fw_shortx16 b;
        memcpy(&b, bt, 32);
        s[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s[f]);
      }
    }
    // scale + mask + 在线 softmax(__expf=v_exp_f32:probs 下游量化 fp16,
    // 精度影响淹没在 fp16 舍入内,三图 cmp 验证)
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
            f2h(s[f][e]);
    __syncthreads();  // S1:scores 读完 Ks、Pbuf 写可见
    if (kt + 1 < nkt) storeK(kn);  // Ks 已自由,写入下 tile
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      fw_shortx16 a = {0};
      if (qrowA >= 0) a = fw_ld16(&Pbuf[w * 16 + qrowA][kk * 16]);
#pragma unroll
      for (int dc = 0; dc < 5; dc++) {
        short bt[16];
#pragma unroll
        for (int k = 0; k < 16; k++)
          bt[k] = (short)VsT[dc * 16 + colbase][kk * 16 + k];
        fw_shortx16 b;
        memcpy(&b, bt, 32);
        acc[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[dc]);
      }
    }
    __syncthreads();  // S2:PV 读完 VsT/Pbuf、Ks 写可见
    if (kt + 1 < nkt) storeV(vn);  // 下轮 PV 在 S1 后,天然可见
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

__global__ void k_gelu_tanh_fp16(const float* x, uint16_t* y, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
  y[i] = f2h(0.5f * v * (1.f + tanhf(u)));
}

// gelu 输出直接 split 成 hi+lo(fc2 输入)
__global__ void k_gelu_tanh_split(const float* x, uint16_t* hi, uint16_t* lo,
                                  uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
  float g = 0.5f * v * (1.f + tanhf(u));
  float h = h2f(f2h(g));
  hi[i] = f2h(g);
  lo[i] = f2h(g - h);
}

__global__ void k_gelu_erf_fp16(const float* x, uint16_t* y, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  y[i] = f2h(0.5f * v * (1.f + erff(v * 0.7071067811865476f)));
}

// ----------------------------- 权重 ----------------------------------------
struct BlockW {
  uint16_t *qkv_w, *proj_w, *fc1_w, *fc2_w;
  float *qkv_b, *proj_b, *fc1_b, *fc2_b, *ln1_w, *ln1_b, *ln2_w, *ln2_b;
};
struct ViTW {
  uint16_t* patch_w;   // 1152×1536
  float* patch_b;
  uint16_t* pos_table; // 2304×1152
  BlockW blk[27];
  float *mg_ln_w, *mg_ln_b, *mg_fc1_b, *mg_fc2_b;
  uint16_t *mg_fc1_w, *mg_fc2_w;  // 4608×4608, 2560×4608
};

static void* dup_dev(const void* p, size_t bytes) {
  void* d;
  CK(hipMalloc(&d, bytes));
  CK(hipMemcpy(d, p, bytes, hipMemcpyHostToDevice));
  return d;
}

static float* dup_dev_bf16_to_f32(const hgn::Tensor& t) {
  uint64_t n = t.numel();
  std::vector<float> h(n);
  const uint16_t* p = (const uint16_t*)t.data;
  for (uint64_t i = 0; i < n; i++) h[i] = hgn::bf16_to_f32(p[i]);
  return (float*)dup_dev(h.data(), n * 4);
}

static void load_weights(hgn::Checkpoint& ck, ViTW& W) {
  // GEMM 权重:bf16 → fp16(host 转换)。正常幅值范围内精确(fp16 尾数
  // 11bit ⊃ bf16 8bit);<2^-14 的 bf16 小值落 fp16 denormal/0,贡献可忽略。
  auto bf = [&](const char* name) -> uint16_t* {
    const hgn::Tensor& t = ck.at(name);
    if (t.dtype != 0) throw std::runtime_error(std::string("dtype ") + name);
    uint64_t n = t.numel();
    std::vector<uint16_t> h(n);
    const uint16_t* p = (const uint16_t*)t.data;
    for (uint64_t i = 0; i < n; i++)
      h[i] = __half_as_ushort(__float2half_rn(hgn::bf16_to_f32(p[i])));
    return (uint16_t*)dup_dev(h.data(), n * 2);
  };
  auto f32 = [&](const char* name) { return dup_dev_bf16_to_f32(ck.at(name)); };
  W.patch_w = bf("visual.patch_embed.proj.weight");
  W.patch_b = f32("visual.patch_embed.proj.bias");
  W.pos_table = bf("visual.pos_embed.weight");
  char nm[128];
  for (int i = 0; i < 27; i++) {
    BlockW& b = W.blk[i];
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
  W.mg_ln_w = f32("visual.merger.norm.weight");
  W.mg_ln_b = f32("visual.merger.norm.bias");
  W.mg_fc1_w = bf("visual.merger.linear_fc1.weight");
  W.mg_fc1_b = f32("visual.merger.linear_fc1.bias");
  W.mg_fc2_w = bf("visual.merger.linear_fc2.weight");
  W.mg_fc2_b = f32("visual.merger.linear_fc2.bias");
}

// ----------------------------- host 预处理 ---------------------------------
// pos_embed 4-tap 索引/权重 + rope cos/sin,逐行复刻 transformers vision_utils。
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
    // taps 外积序 = (h_taps[:,None]*side + w_taps) flatten:h 主序
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

// ----------------------------- main ----------------------------------------
struct Prof {
  double patch = 0, pos = 0, ln = 0, qkv = 0, rope = 0, attn = 0, proj = 0,
         mlp = 0, merger = 0;
};
static Prof prof;
struct Tic {
  std::chrono::steady_clock::time_point t0;
  Tic() : t0(std::chrono::steady_clock::now()) {}
  double ms() {
    hipStreamSynchronize(g_str);
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
  }
};

int main(int argc, char** argv) {
  const char* sidecar = nullptr;
  const char *patches_path = nullptr, *grid_path = nullptr, *out_prefix = nullptr;
  int max_blocks = 27;
  bool dump_layers = false, quiet = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() { return argv[++i]; };
    if (a == "--patches")
      patches_path = next();
    else if (a == "--grid-thw")
      grid_path = next();
    else if (a == "--out-prefix")
      out_prefix = next();
    else if (a == "--blocks")
      max_blocks = atoi(next());
    else if (a == "--dump-layers")
      dump_layers = true;
    else if (a == "--quiet")
      quiet = true;
    else if (a[0] != '-')
      sidecar = argv[i];
    else {
      fprintf(stderr, "unknown arg %s\n", a.c_str());
      return 2;
    }
  }
  if (!sidecar || !patches_path || !grid_path || !out_prefix) {
    fprintf(stderr,
            "usage: vit <sidecar.hgn> --patches x.npy --grid-thw g.npy "
            "--out-prefix D/N [--blocks N] [--dump-layers] [--quiet]\n");
    return 2;
  }

  Npy patches = npy_read(patches_path);
  if (patches.shape.size() != 2 || patches.shape[1] != 1536 ||
      strcmp(patches.descr, "<f4"))
    throw std::runtime_error("patches npy must be <f4 (P,1536)");
  int P = (int)patches.shape[0];
  Npy grid = npy_read(grid_path);
  int64_t gt = ((int64_t*)grid.data.data())[0];
  int gh = (int)((int64_t*)grid.data.data())[1];
  int gw = (int)((int64_t*)grid.data.data())[2];
  if (gt != 1 || gh * gw != P)
    throw std::runtime_error("grid/patches mismatch");
  fprintf(stderr, "vit: P=%d grid=(%lld,%d,%d) blocks=%d\n", P, (long long)gt, gh,
          gw, max_blocks);
  mkdir((std::string(out_prefix) + ".layers").c_str(), 0777);  // 输出恒落盘至此

  CK(hipSetDevice(0));
  CK(hipStreamCreate(&g_str));
  hipblasLtCreate(&lth);
  rocblas_create_handle(&rbh);
  rocblas_set_stream(rbh, g_str);
  CK(hipMalloc(&d_ltws, LT_WS));

  hgn::Checkpoint ck(sidecar);
  ViTW W;
  load_weights(ck, W);
  fprintf(stderr, "vit: weights loaded (%zu tensors)\n", ck.tensor_count());

  // ---- device buffers
  float *d_h, *d_n, *d_qkv, *d_attn, *d_mlp;
  uint16_t* d_scores;  // fp16 scores(P×P)
  uint16_t *d_xbf, *d_xlo, *d_qb, *d_kb, *d_vb, *d_probs;
  CK(hipMalloc(&d_h, (size_t)P * 1152 * 4));
  CK(hipMalloc(&d_n, (size_t)P * 1152 * 4));
  CK(hipMalloc(&d_qkv, (size_t)P * 3456 * 4));
  CK(hipMalloc(&d_attn, (size_t)P * 1152 * 4));
  CK(hipMalloc(&d_mlp, (size_t)P * 4608 * 4));  // merger fc1 也要 4608
  CK(hipMalloc(&d_scores, (size_t)P * P * 2));  // fp16
  CK(hipMalloc(&d_xbf, (size_t)P * 4608 * 2));
  CK(hipMalloc(&d_xlo, (size_t)P * 4608 * 2));  // split-fp16 低位
  CK(hipMalloc(&d_qb, (size_t)P * 1152 * 2));
  CK(hipMalloc(&d_kb, (size_t)P * 1152 * 2));
  CK(hipMalloc(&d_vb, (size_t)P * 1152 * 2));
  CK(hipMalloc(&d_probs, (size_t)P * P * 2));

  // ---- rope cos/sin
  std::vector<float> hcos, hsin;
  host_rope(gh, gw, hcos, hsin);
  float *d_cos, *d_sin;
  CK(hipMalloc(&d_cos, hcos.size() * 4));
  CK(hipMalloc(&d_sin, hsin.size() * 4));
  CK(hipMemcpy(d_cos, hcos.data(), hcos.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_sin, hsin.data(), hsin.size() * 4, hipMemcpyHostToDevice));
  if (dump_layers) {
    std::string pc = std::string(out_prefix) + ".layers/rope_cos.npy";
    std::string ps = std::string(out_prefix) + ".layers/rope_sin.npy";
    npy_write_f32(pc.c_str(), hcos.data(), P, 72);
    npy_write_f32(ps.c_str(), hsin.data(), P, 72);
  }

  const float scale = powf(72.f, -0.5f);
  // ---- blocks(主体为 lambda,warmup 与正式循环共用)
  auto run_block = [&](int bi, bool dump) {
    BlockW& B = W.blk[bi];
    {  // LN1
      Tic t;
      k_layernorm<<<P, 256, 0, g_str>>>(d_h, d_n, B.ln1_w, B.ln1_b, 1152, 1e-6f);
      prof.ln += t.ms();
    }
    {  // qkv
      Tic t;
      k_f32_to_fp16<<<(P * 1152 + 255) / 256, 256, 0, g_str>>>(d_n, d_xbf,
                                                               (size_t)P * 1152);
      proj_gemm(d_xbf, B.qkv_w, d_qkv, P, 3456, 1152);
      k_bias_add<<<((size_t)P * 3456 + 255) / 256, 256, 0, g_str>>>(
          d_qkv, B.qkv_b, P, 3456);
      prof.qkv += t.ms();
    }
    {  // rope → fp16 q/k/v
      Tic t;
      dim3 grid(P, 16);
      k_rope_qkv<<<grid, 36, 0, g_str>>>(d_qkv, d_cos, d_sin, d_qb, d_kb, d_vb,
                                         P);
      prof.rope += t.ms();
    }
    {  // attention:默认 flash kernel;VIT_NAIVE_ATTN=1 回退三-kernel 朴素路径
      Tic t;
      static const bool naive_attn = getenv("VIT_NAIVE_ATTN") != nullptr;
      if (naive_attn) {
        for (int h = 0; h < 16; h++) {
          const uint16_t* qh = d_qb + h * 72;
          const uint16_t* kh = d_kb + h * 72;
          const uint16_t* vh = d_vb + h * 72;
          // scores_rm(P×P) = Q_h K_h^T:C_cm(P×P) = opA=T A=K_cm(72×P ld 1152)
          // opB=N B=Q_cm(72×P ld 1152);C 直接出 fp16(归因:probs 路径舍入可忽略)
          lt_gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, P, P, 72, kh, 1152, qh, 1152,
                  d_scores, P, scale, 0.f, true);
          k_softmax_fp16<<<P, 256, 0, g_str>>>(d_scores, d_probs, P);
          // out_rm(P×72) = probs @ V_h:C_cm(72×P) = opA=N A=V_cm(72×P ld 1152)
          // opB=N B=probs_cm(P×P ld P)
          lt_gemm(HIPBLAS_OP_N, HIPBLAS_OP_N, 72, P, P, vh, 1152,
                  (const uint16_t*)d_probs, P, d_attn + h * 72, 1152, 1.f,
                  0.f);
        }
      } else {
        dim3 grid((P + 127) / 128, 16);
        static const bool flash_v1 = getenv("VIT_FLASH_V1") != nullptr;
        if (flash_v1)
          k_flash_wmma<<<grid, 256, 0, g_str>>>(d_qb, d_kb, d_vb, d_attn, P,
                                                scale);
        else
          k_flash_wmma_db<<<grid, 256, 0, g_str>>>(d_qb, d_kb, d_vb, d_attn, P,
                                                   scale);
      }
      prof.attn += t.ms();
    }
    {  // proj + 残差
      Tic t;
      k_f32_to_fp16<<<(P * 1152 + 255) / 256, 256, 0, g_str>>>(d_attn, d_xbf,
                                                               (size_t)P * 1152);
      proj_gemm(d_xbf, B.proj_w, d_qkv, P, 1152, 1152);
      k_bias_resadd<<<((size_t)P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_h, d_qkv, B.proj_b, P, 1152);
      prof.proj += t.ms();
    }
    {  // MLP:LN2 → fc1(split) → gelu(tanh, split) → fc2(split) + 残差
      Tic t;
      k_layernorm<<<P, 256, 0, g_str>>>(d_h, d_n, B.ln2_w, B.ln2_b, 1152, 1e-6f);
      k_f32_split_fp16<<<(P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_n, d_xbf, d_xlo, (size_t)P * 1152);
      proj_gemm_split(d_xbf, d_xlo, B.fc1_w, d_mlp, P, 4304, 1152);
      k_bias_add<<<((size_t)P * 4304 + 255) / 256, 256, 0, g_str>>>(
          d_mlp, B.fc1_b, P, 4304);
      k_gelu_tanh_split<<<((size_t)P * 4304 + 255) / 256, 256, 0, g_str>>>(
          d_mlp, d_xbf, d_xlo, (size_t)P * 4304);
      proj_gemm_split(d_xbf, d_xlo, B.fc2_w, d_qkv, P, 1152, 4304);
      k_bias_resadd<<<((size_t)P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_h, d_qkv, B.fc2_b, P, 1152);
      prof.mlp += t.ms();
    }
    if (dump && dump_layers) {
      std::vector<float> h((size_t)P * 1152);
      CK(hipMemcpy(h.data(), d_h, h.size() * 4, hipMemcpyDeviceToHost));
      char p[512];
      snprintf(p, 512, "%s.layers/block%02d.npy", out_prefix, bi);
      npy_write_f32(p, h.data(), P, 1152);
    }
    if (dump && !quiet) fprintf(stderr, "vit: block%02d done\n", bi);
  };

  // ---- merger lambda(同样供 warmup 复用)
  int M = P / 4;
  float* d_out;
  CK(hipMalloc(&d_out, (size_t)M * 2560 * 4));
  auto run_merger = [&](bool dump) {
    Tic t;
    k_layernorm<<<P, 256, 0, g_str>>>(d_h, d_n, W.mg_ln_w, W.mg_ln_b, 1152,
                                      1e-6f);
    if (dump && dump_layers) {
      std::vector<float> h((size_t)P * 1152);
      CK(hipMemcpy(h.data(), d_n, h.size() * 4, hipMemcpyDeviceToHost));
      std::string p = std::string(out_prefix) + ".layers/merger_ln.npy";
      npy_write_f32(p.c_str(), h.data(), M, 4608);
    }
    k_f32_to_fp16<<<((size_t)M * 4608 + 255) / 256, 256, 0, g_str>>>(
        d_n, d_xbf, (size_t)M * 4608);
    proj_gemm(d_xbf, W.mg_fc1_w, d_mlp, M, 4608, 4608);
    k_bias_add<<<((size_t)M * 4608 + 255) / 256, 256, 0, g_str>>>(
        d_mlp, W.mg_fc1_b, M, 4608);
    if (dump && dump_layers) {
      std::vector<float> h((size_t)M * 4608);
      CK(hipMemcpy(h.data(), d_mlp, h.size() * 4, hipMemcpyDeviceToHost));
      std::string p = std::string(out_prefix) + ".layers/merger_fc1.npy";
      npy_write_f32(p.c_str(), h.data(), M, 4608);
    }
    k_gelu_erf_fp16<<<((size_t)M * 4608 + 255) / 256, 256, 0, g_str>>>(
        d_mlp, d_xbf, (size_t)M * 4608);
    proj_gemm(d_xbf, W.mg_fc2_w, d_out, M, 2560, 4608);
    k_bias_add<<<((size_t)M * 2560 + 255) / 256, 256, 0, g_str>>>(
        d_out, W.mg_fc2_b, M, 2560);
    prof.merger += t.ms();
  };

  // ---- warmup:一个 dummy block + merger(全 0 输入),触发全部 LT 形状的
  // 算法选择,正式计时不被一次性选择开销污染。warmup 在 patch_embed 之前,
  // 之后 d_h 被完全覆写,结果不受影响。
  {
    CK(hipMemset(d_h, 0, (size_t)P * 1152 * 4));
    Prof saved = prof;
    proj_gemm_split(d_xbf, d_xlo, W.patch_w, d_h, P, 1152, 1536);  // patch 形状
    run_block(0, false);
    run_merger(false);
    prof = saved;  // warmup 不计入
  }

  // ---- patch_embed:X fp32→split-fp16,两路 GEMM + bias
  {
    Tic t;
    CK(hipMemcpy(d_mlp, patches.data.data(), (size_t)P * 1536 * 4,
                 hipMemcpyHostToDevice));  // d_mlp (P*4608 f32) 兼作 staging
    k_f32_split_fp16<<<(P * 1536 + 255) / 256, 256, 0, g_str>>>(
        d_mlp, d_xbf, d_xlo, (size_t)P * 1536);
    proj_gemm_split(d_xbf, d_xlo, W.patch_w, d_h, P, 1152, 1536);
    k_bias_add<<<((size_t)P * 1152 + 255) / 256, 256, 0, g_str>>>(d_h, W.patch_b,
                                                                  P, 1152);
    prof.patch = t.ms();
  }
  if (dump_layers) {
    std::vector<float> h((size_t)P * 1152);
    CK(hipMemcpy(h.data(), d_h, h.size() * 4, hipMemcpyDeviceToHost));
    std::string p = std::string(out_prefix) + ".layers/patchembed.npy";
    npy_write_f32(p.c_str(), h.data(), P, 1152);
  }

  // ---- pos_embed:host 索引/权重 → device gather 加权
  {
    Tic t;
    std::vector<int> hidx;
    std::vector<float> hwts;
    host_pos_interp(gh, gw, hidx, hwts);
    int* d_idx;
    float* d_wts;
    CK(hipMalloc(&d_idx, hidx.size() * 4));
    CK(hipMalloc(&d_wts, hwts.size() * 4));
    CK(hipMemcpy(d_idx, hidx.data(), hidx.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_wts, hwts.data(), hwts.size() * 4, hipMemcpyHostToDevice));
    k_pos_embed<<<P, 256, 0, g_str>>>(d_h, W.pos_table, d_idx, d_wts, P);
    prof.pos = t.ms();
    hipFree(d_idx);
    hipFree(d_wts);
  }
  if (dump_layers) {
    std::vector<float> h((size_t)P * 1152);
    CK(hipMemcpy(h.data(), d_h, h.size() * 4, hipMemcpyDeviceToHost));
    std::string p = std::string(out_prefix) + ".layers/embed.npy";
    npy_write_f32(p.c_str(), h.data(), P, 1152);
  }

  for (int bi = 0; bi < max_blocks; bi++) run_block(bi, true);
  run_merger(true);

  // ---- 输出:merged fp32 + bf16(u16 视图;host RNE,与 torch .to(bf16) 一致)
  {
    std::vector<float> h((size_t)M * 2560);
    CK(hipMemcpy(h.data(), d_out, h.size() * 4, hipMemcpyDeviceToHost));
    std::string p = std::string(out_prefix) + ".layers/merged.npy";
    npy_write_f32(p.c_str(), h.data(), M, 2560);
    std::vector<uint16_t> hb(h.size());
    for (size_t i = 0; i < h.size(); i++) {
      uint32_t u;
      memcpy(&u, &h[i], 4);
      u += 0x7FFF + ((u >> 16) & 1);
      hb[i] = (uint16_t)(u >> 16);
    }
    std::string pb = std::string(out_prefix) + ".layers/embeds.bf16.npy";
    npy_write_u16(pb.c_str(), hb.data(), M, 2560);
  }

  if (!quiet) {
    fprintf(stderr,
            "vit prof (ms): patch=%.1f pos=%.1f ln=%.1f qkv=%.1f rope=%.1f "
            "attn=%.1f proj=%.1f mlp=%.1f merger=%.1f\n",
            prof.patch, prof.pos, prof.ln, prof.qkv, prof.rope, prof.attn,
            prof.proj, prof.mlp, prof.merger);
  }
  return 0;
}

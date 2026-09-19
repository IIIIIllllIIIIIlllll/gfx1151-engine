#define main gdec_real_main
#include "gdec.cpp"
#undef main
#include <random>

// Compare batched state handoff and the NEXT output with sequential decode.
// Nonzero, distinct history rows expose one/two-token tail and rollback errors.
int main() {
  CK(hipSetDevice(0));
  constexpr int C = 10240, MAXP = 65;
  std::mt19937 rng(91237);
  std::uniform_real_distribution<float> rnd(-1.f, 1.f);
  std::vector<float> hx((MAXP + 1)*C), hs(3*C), hw(4*C);
  for (auto& v : hx) v = rnd(rng);
  for (auto& v : hs) v = rnd(rng);
  for (auto& v : hw) v = rnd(rng);
  float *x, *w, *sr, *sg, *yr, *yg;
  CK(hipMalloc(&x, hx.size()*4)); CK(hipMalloc(&w, hw.size()*4));
  CK(hipMalloc(&sr, hs.size()*4)); CK(hipMalloc(&sg, hs.size()*4));
  CK(hipMalloc(&yr, C*4)); CK(hipMalloc(&yg, C*4));
  CK(hipMemcpy(x,hx.data(),hx.size()*4,hipMemcpyHostToDevice));
  CK(hipMemcpy(w,hw.data(),hw.size()*4,hipMemcpyHostToDevice));
  std::vector<float> ref(3*C), got(3*C), next_ref(C), next_got(C);
  int failed = 0;
  for (int P : {1,2,3,4,8,9,16,17,32,33,64,65}) {
    CK(hipMemcpy(sr,hs.data(),hs.size()*4,hipMemcpyHostToDevice));
    CK(hipMemcpy(sg,hs.data(),hs.size()*4,hipMemcpyHostToDevice));
    for(int t=0;t<P;++t)
      k_gdn_conv<<<(C+255)/256,256>>>(x+(size_t)t*C,w,sr,yr,C);
    k_convst_update<<<(C+255)/256,256>>>(sg,x,P,C);
    CK(hipMemcpy(ref.data(),sr,hs.size()*4,hipMemcpyDeviceToHost));
    CK(hipMemcpy(got.data(),sg,hs.size()*4,hipMemcpyDeviceToHost));
    size_t wrong=0;
    for(size_t i=0;i<ref.size();++i) wrong += ref[i]!=got[i];
    k_gdn_conv<<<(C+255)/256,256>>>(x+(size_t)P*C,w,sr,yr,C);
    k_gdn_conv<<<(C+255)/256,256>>>(x+(size_t)P*C,w,sg,yg,C);
    CK(hipMemcpy(next_ref.data(),yr,C*4,hipMemcpyDeviceToHost));
    CK(hipMemcpy(next_got.data(),yg,C*4,hipMemcpyDeviceToHost));
    double maxerr=0;
    for(int i=0;i<C;++i) maxerr=std::max(maxerr,(double)fabsf(next_ref[i]-next_got[i]));
    bool pass=wrong==0 && maxerr==0;
    failed+=!pass;
    printf("P=%d %s state_mismatch=%zu next_output_abs=%.9g\n",P,pass?"PASS":"FAIL",wrong,maxerr);
    fflush(stdout);
  }
  return failed?1:0;
}

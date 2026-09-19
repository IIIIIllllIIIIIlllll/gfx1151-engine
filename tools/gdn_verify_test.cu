#define main gdec_real_main
#include "gdec.cpp"
#undef main
#include <random>

int main() {
  CK(hipSetDevice(0));
  std::mt19937 rng(718);
  std::uniform_real_distribution<float> rnd(-1.f, 1.f);
  constexpr size_t NS = 48 * 128 * 128;
  float *x, *ref, *got, *s0, *sr, *sg, *g, *b, *ws;
  CK(hipMalloc(&x, 65*10240*4)); CK(hipMalloc(&ref, 65*10240*4));
  CK(hipMalloc(&got, 65*10240*4));
  CK(hipMalloc(&s0, NS*4)); CK(hipMalloc(&sr, NS*4)); CK(hipMalloc(&sg, NS*4));
  CK(hipMalloc(&g, 65*48*4)); CK(hipMalloc(&b, 65*48*4));
  CK(hipMalloc(&ws, (size_t)2*48*GDN_SPLIT_WS_FLOATS*4));
  std::vector<float> hx(65*10240), hs(NS), hg(65*48), hb(65*48);
  for (float& v:hx) v=rnd(rng);
  for (float& v:hs) v=.1f*rnd(rng);
  for (float& v:hg) v=-.2f*(1.f+rnd(rng));
  for (float& v:hb) v=.5f*(1.f+rnd(rng));
  CK(hipMemcpy(x,hx.data(),hx.size()*4,hipMemcpyHostToDevice));
  CK(hipMemcpy(s0,hs.data(),hs.size()*4,hipMemcpyHostToDevice));
  CK(hipMemcpy(g,hg.data(),hg.size()*4,hipMemcpyHostToDevice));
  CK(hipMemcpy(b,hb.data(),hb.size()*4,hipMemcpyHostToDevice));
  k_l2norm_qk_b<<<dim3(16,65),128>>>(x,128,1.f/sqrtf(128.f),10240,16);
  auto init=[&](float* dst,float* state,int P){
    CK(hipMemcpyAsync(dst,x,(size_t)P*10240*4,hipMemcpyDeviceToDevice));
    CK(hipMemcpyAsync(state,s0,NS*4,hipMemcpyDeviceToDevice));
  };
  auto old=[&](int P){
    k_gdn_intra<GDN_NT_INTRA><<<dim3((P+63)/64,48),GDN_NT_INTRA>>>(ref,g,b,ws,P,10240);
    k_gdn_inter_strip<<<dim3(4,48),GDN_NT_STRIP>>>(ref,sr,ref,ws,P,10240);
  };
  auto fast=[&](int P){k_gdn_verify<<<48,512>>>(got,g,b,sg,got,P,10240);};
  auto sequential=[&](int P){for(int t=0;t<P;++t){
    float* row=ref+(size_t)t*10240;
    k_gdn_step<<<48,512>>>(row,row+2048,row+4096,g+t*48,b+t*48,sr,row+4096,128,128,48,16);
  }};
  std::vector<float> hr(65*10240),ho(65*10240),rs(NS),gs(NS);
  int failures=0;
  for(int P: {1,2,4,8,9,16,17,32,33,64,65}) {
    init(ref,sr,P); init(got,sg,P); sequential(P); fast(P);
    CK(hipMemcpy(hr.data(),ref,(size_t)P*10240*4,hipMemcpyDeviceToHost));
    CK(hipMemcpy(ho.data(),got,(size_t)P*10240*4,hipMemcpyDeviceToHost));
    CK(hipMemcpy(rs.data(),sr,NS*4,hipMemcpyDeviceToHost));
    CK(hipMemcpy(gs.data(),sg,NS*4,hipMemcpyDeviceToHost));
    double oe=0,se=0; bool finite=true;
    for(int t=0;t<P;++t) for(int i=4096;i<10240;++i){
      size_t at=(size_t)t*10240+i;
      finite &= std::isfinite(ho[at]);
      oe=std::max(oe,(double)fabsf(hr[at]-ho[at]));
    }
    for(size_t i=0;i<NS;++i){finite &= std::isfinite(gs[i]); se=std::max(se,(double)fabsf(rs[i]-gs[i]));}
    bool pass=finite&&oe<2e-6&&se<2e-6;
    failures+=!pass;
    hipEvent_t start,end; CK(hipEventCreate(&start)); CK(hipEventCreate(&end));
    auto bench=[&](bool optimized){
      for(int w=0;w<3;++w){init(optimized?got:ref,optimized?sg:sr,P); if(optimized)fast(P);else old(P);}
      float sum=0;
      for(int rep=0;rep<20;++rep){
        init(optimized?got:ref,optimized?sg:sr,P);
        CK(hipEventRecord(start)); if(optimized)fast(P);else old(P);
        CK(hipEventRecord(end)); CK(hipEventSynchronize(end));
        float ms; CK(hipEventElapsedTime(&ms,start,end)); sum+=ms;
      }
      return sum/20;
    };
    float prev=bench(false), now=bench(true);
    printf("P=%d %s output_abs=%.3g state_abs=%.3g split=%.4fms recurrent=%.4fms speedup=%.2fx\n",P,pass?"PASS":"FAIL",oe,se,prev,now,prev/now);
    fflush(stdout);
    CK(hipEventDestroy(start));CK(hipEventDestroy(end));
  }
  return failures?1:0;
}

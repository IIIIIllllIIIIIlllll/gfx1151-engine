#ifndef _WIN32
// POSIX side of the socket shim (Windows side lives in os_win32.h): lets the
// serve loop below use one spelling on both platforms.
typedef int sockfd_t;
static inline bool wsa_init() { return true; }
static inline sockfd_t sock_tcp() { return socket(AF_INET, SOCK_STREAM, 0); }
static inline int sock_close(sockfd_t fd) { return close(fd); }
static inline ssize_t sock_recv(sockfd_t fd, void* buf, size_t n, bool dontwait) {
  return recv(fd, buf, n, dontwait ? MSG_DONTWAIT : 0);
}
static inline ssize_t sock_send(sockfd_t fd, const void* buf, size_t n) {
  return send(fd, buf, n, MSG_NOSIGNAL);
}
static inline int sock_errno() { return errno; }
static inline const char* sock_strerror(int e) { return strerror(e); }

// POSIX side of the os_* file shims (Windows side lives in os_win32.h).
static inline int os_open_rd(const char* path) { return open(path, O_RDONLY); }
static inline int os_open_wr(const char* path) {
  return open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
}
static inline int os_close(int fd) { return close(fd); }
static inline ssize_t os_write(int fd, const void* p, size_t n) {
  return write(fd, p, n);
}
static inline int64_t os_fsize(int fd) {
  struct stat sb;
  return fstat(fd, &sb) ? -1 : (int64_t)sb.st_size;
}
static inline void* os_map_ro(const char* path, size_t len) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return nullptr;
  void* p = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  return p == MAP_FAILED ? nullptr : p;
}
static inline void os_unmap_ro(void* p, size_t len) { munmap(p, len); }
#endif

static inline void sock_rcvtimeo(sockfd_t fd, int sec) {
#ifdef _WIN32
  DWORD ms = (DWORD)(sec * 1000);
  setsockopt((SOCKET)fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof ms);
#else
  timeval tv{sec, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
}

using hgn::Checkpoint;
using hgn::Tensor;

#define CK(x)                                                          \
  do {                                                                 \
    hipError_t e_ = (x);                                               \
    if (e_ != hipSuccess) {                                            \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), \
              __FILE__, __LINE__);                                     \
      exit(1);                                                         \
    }                                                                  \
  } while (0)

// "[2026-09-18 11:25:03.123]" — wall-clock prefix for human-facing logs.
// thread_local so concurrent loader/decode threads never share the buffer.
// One call per fprintf: two calls in one expression would alias.
static const char* log_ts() {
  static thread_local char b[32];
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  struct tm tmv;
  localtime_r(&ts.tv_sec, &tmv);
  snprintf(b, sizeof b, "[%04d-%02d-%02d %02d:%02d:%02d.%03d]",
           tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour,
           tmv.tm_min, tmv.tm_sec, (int)(ts.tv_nsec / 1000000));
  return b;
}

struct NgramMod {
  using token_t=int32_t; static constexpr token_t EMPTY=-1;
  int n=24,n_min=48,n_max=64; size_t learned=0; std::vector<token_t> table;
  NgramMod(){ if(const char* e=getenv("GDEC_NGRAM_MATCH")) n=std::max(4,std::min(64,atoi(e))); if(const char* e=getenv("GDEC_NGRAM_MIN")) n_min=std::max(1,std::min(64,atoi(e))); if(const char* e=getenv("GDEC_NGRAM_MAX")) n_max=std::max(1,std::min(64,atoi(e))); table.resize((size_t)4*1024*1024,EMPTY); }
  size_t slot(const std::vector<int64_t>&h,size_t off)const{uint64_t z=0;for(int i=0;i<n;i++) z=z*6364136223846793005ULL+(uint32_t)h[off+i];return(size_t)(z%table.size());}
  void reset(){std::fill(table.begin(),table.end(),EMPTY);learned=0;}
  void observe(const std::vector<int64_t>&h){if(h.size()<=(size_t)n)return;size_t lim=h.size()-n;if(learned>lim)learned=lim;for(size_t i=learned;i<lim;i++)table[slot(h,i)]=(token_t)h[i+n];learned=lim;}
  std::vector<int> draft(const std::vector<int64_t>&h,int pending,int want)const{std::vector<int>o;if(h.size()+1<(size_t)n)return o;std::vector<int64_t>c((size_t)n);for(int i=0;i<n-1;i++)c[i]=h[h.size()-(size_t)(n-1)+i];c[n-1]=pending;for(int i=0;i<std::min(want,n_max);i++){token_t t=table[slot(c,0)];if(t==EMPTY){if(i<n_min)o.clear();break;}o.push_back((int)t);for(int j=0;j<n-1;j++)c[j]=c[j+1];c[n-1]=(int)t;}return o;}
};


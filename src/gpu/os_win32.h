// os_win32.h — Windows 兼容层：为 gdec.cpp / hgn.h 提供 POSIX 等价物。
// 仅 _WIN32 下包含。设计原则：gdec.cpp 主体代码基本不动，差异全部收敛到这里。
#pragma once
#ifndef _WIN32
#error "os_win32.h 仅供 Windows 构建包含"
#endif

#ifndef NOMINMAX
#define NOMINMAX  // 屏蔽 windows.h 的 min/max 宏（std::min/size_t> 会被污染）
#endif
#include <winsock2.h>  // 必须先于 windows.h
#include <ws2tcpip.h>
#include <windows.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <string>
#include <sys/stat.h>
#include <time.h>

#pragma comment(lib, "ws2_32")

// ---------- madvise（纯提示，Windows 降级为 no-op；快路径走 IOCP 批量读） ----------
#ifndef MADV_WILLNEED
#define MADV_WILLNEED 0
#endif
static inline int madvise(void*, size_t, int) { return 0; }
#define isatty _isatty
#define mkdir(p, m) _mkdir(p)

// ---------- 杂项 ----------
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
typedef SSIZE_T ssize_t;
typedef intptr_t sockfd_t;  // SOCKET 在 Win64 是 64 位，不能截断成 int

static inline size_t strnlen_(const char* s, size_t n) {
  size_t i = 0;
  while (i < n && s[i]) i++;
  return i;
}
#define strnlen strnlen_

// clock_gettime(CLOCK_REALTIME) / localtime_r
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
static inline int clock_gettime(int, timespec* ts) {
  FILETIME ft;
  GetSystemTimePreciseAsFileTime(&ft);
  uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
  t -= 116444736000000000ull;  // 1601 -> 1970
  ts->tv_sec = (time_t)(t / 10000000ull);
  ts->tv_nsec = (long)((t % 10000000ull) * 100);
  return 0;
}
static inline struct tm* localtime_r(const time_t* t, struct tm* out) {
  return localtime_s(out, t) == 0 ? out : nullptr;
}

// ---------- Winsock ----------
static inline bool wsa_init() {
  WSADATA w;
  return WSAStartup(MAKEWORD(2, 2), &w) == 0;
}
static inline sockfd_t sock_tcp() {
  SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  return s == INVALID_SOCKET ? (sockfd_t)-1 : (sockfd_t)s;
}
static inline int sock_close(sockfd_t fd) { return closesocket((SOCKET)fd); }
static inline int sock_shutdown(sockfd_t fd) { return shutdown((SOCKET)fd, SD_BOTH); }
// flags 仅支持 0 与 MSG_DONTWAIT（后者通过临时切换非阻塞模式实现）
static inline ssize_t sock_recv(sockfd_t fd, void* buf, size_t n, bool dontwait) {
  SOCKET s = (SOCKET)fd;
  if (dontwait) {
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
    int r = recv(s, (char*)buf, (int)n, 0);
    nb = 0;
    ioctlsocket(s, FIONBIO, &nb);
    if (r == SOCKET_ERROR) return WSAGetLastError() == WSAEWOULDBLOCK ? 0 : -1;
    return r == 0 ? 0 : (ssize_t)r;  // 对端关闭按"此刻无数据"处理,下次阻塞读给出 EOF
  }
  int r = recv(s, (char*)buf, (int)n, 0);
  return r == SOCKET_ERROR ? -1 : (ssize_t)r;
}
static inline ssize_t sock_send(sockfd_t fd, const void* buf, size_t n) {
  int r = send((SOCKET)fd, (const char*)buf, (int)n, 0);
  return r == SOCKET_ERROR ? -1 : (ssize_t)r;
}
static inline int sock_errno() { return WSAGetLastError(); }
static inline const char* sock_strerror(int e) {
  static thread_local char b[96];
  snprintf(b, sizeof b, "WSA error %d", e);
  return b;
}

// ---------- 文件 ----------
// 二进制模式（MSVC 默认文本模式会改写 \n，必须显式 _O_BINARY）
static inline int os_open_rd(const char* path) {
  return _open(path, _O_RDONLY | _O_BINARY);
}
static inline int os_open_wr(const char* path) {  // create+truncate
  return _open(path, _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, 0644);
}
static inline int os_close(int fd) { return _close(fd); }
static inline ssize_t os_write(int fd, const void* p, size_t n) {
  // _write 单次限 4 GiB（int 返回值），调用方本就有循环
  return _write(fd, p, (unsigned int)n);
}
static inline int64_t os_fsize(int fd) { return _filelengthi64(fd); }

// 只读映射整个文件（mmap PROT_READ 等价物），返回 nullptr 失败
static inline void* os_map_ro(const char* path, size_t len) {
  HANDLE fh = CreateFileA(path, GENERIC_READ,
                          FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                          nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (fh == INVALID_HANDLE_VALUE) return nullptr;
  HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mh) {
    CloseHandle(fh);
    return nullptr;
  }
  void* p = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, len);
  CloseHandle(mh);
  CloseHandle(fh);
  return p;
}
static inline void os_unmap_ro(void* p, size_t) { UnmapViewOfFile(p); }

static inline std::string os_last_error(const char* what) {
  DWORD e = GetLastError();
  char buf[512];
  DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, e, 0, buf,
                           sizeof buf, nullptr);
  std::string s = what;
  s += ": ";
  s += n ? buf : "unknown error";
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  return s;
}

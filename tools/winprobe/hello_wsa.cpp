#include <winsock2.h>
#include <cstdio>
int main() {
  WSADATA w;
  int r = WSAStartup(MAKEWORD(2, 2), &w);
  std::printf("wsa %d\n", r);
  return 0;
}

// sockt.h — 跨平台 socket 句柄类型：POSIX int / Win64 SOCKET(intptr_t)。
// 保持轻量（不含 winsock2.h），供 http.h / engine_client.h 公开签名使用。
#pragma once

#include <cstdint>

#ifdef _WIN32
typedef intptr_t sock_t;
#else
typedef int sock_t;
#endif

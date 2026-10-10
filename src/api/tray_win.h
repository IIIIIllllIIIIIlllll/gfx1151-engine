// tray_win.h — qwenox-api 的 Windows 托盘模式（仅 _WIN32 编译）。
// 取代旧 start_win.exe 启动器的生态位：API 进程本身就是托盘程序，
// worker 线程跑 HTTP 服务，主线程跑托盘消息循环；引擎启停经 enginesup。
#pragma once

#include <functional>
#include <string>
#include <utility>

#include "../service_conf.h"

namespace http {
class Server;
}

struct TrayHooks {
    int api_port = 8731;
    std::string api_host;   // 监听地址（0.0.0.0/:: 时本机访问用 127.0.0.1）
    std::string api_log;    // main 重定向 stdout/stderr 的日志文件（绝对路径）
    // 当前有效 service.conf（网页 /admin/config 改过后启动引擎要用新值）
    std::function<svcconf::Conf()> conf_snapshot;
    // ENGINE_HOST/ENGINE_PORT 当前有效值（状态探测 / SNAP 用）
    std::function<std::pair<std::string, int>()> engine_endpoint;
};

// 进入托盘主循环：worker 线程跑 srv.run()，主线程消息循环，"退出"菜单
// 结束进程（引擎在 KILL_ON_JOB_CLOSE Job 里随之结束）。返回进程退出码。
int tray_run(http::Server& srv, const TrayHooks& hooks);

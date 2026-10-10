// engine_sup.h — API 进程的引擎监管：启动 / 停止 / 状态查询。
// Windows 上由 qwenox-api 自己 spawn 并看守 build\qwenox-engine-win.exe（与 start_win.exe
// 的拉起逻辑等价：同一套 conf 解析、argv 与 QWENOX_* 环境、KILL_ON_JOB_CLOSE）。
// Linux 不提供启停（start/stop 返回不支持），status 仍可探测外部引擎。
#pragma once

#include <string>

#include "../service_conf.h"

namespace enginesup {

struct Status {
    std::string state = "stopped";  // stopped | starting | running | exited
    int pid = 0;                    // 本进程拉起的引擎 PID（external 时为 0）
    unsigned long exit_code = 0;    // state == exited 时有效
    std::string log;                // 引擎日志路径
    bool external = false;          // 端口在听但不是本进程拉起的（如 start_win.exe）
    std::string detail;             // 附加说明（启动超时等）
};

bool supported();  // 仅 Windows 为 true

// host/port 是 ENGINE_HOST/ENGINE_PORT 的当前有效值（探测外部引擎用）。
Status status(const std::string& host, int port);

// root 是 service.conf 所在目录：引擎 exe 为 <root>/build/qwenox-engine-win.exe，
// cwd 与 logs\ 都以 root 为基准。已在运行/启动中返回 false。
// spawn 成功后立即返回（state=starting），就绪/退出由监管线程异步推进。
bool start(const std::string& root, const svcconf::Conf& conf, std::string* err);

// 停掉本进程拉起的引擎；外部引擎（external）拒绝并返回 false。
bool stop(std::string* err);

}  // namespace enginesup

// start_stub_win.cpp — start_win.exe：根目录最小入口，只做一件事：
// 以项目根为工作目录拉起 build\qwenox-win.exe（GUI 托盘程序，API 服务 +
// 控制台网页 + 引擎启停）。放在根目录是为了用户不必点进 build\。
// 旧启动器（配置面板/双进程看守）已归档到 attic/launcher/，不再维护。
#include <windows.h>

#include <cstdio>
#include <string>

namespace {

std::string exe_dir() {
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf, n ? static_cast<size_t>(n) : 0);
    const size_t slash = p.find_last_of("\\/");
    return slash == std::string::npos ? "." : p.substr(0, slash);
}

int fail(const std::string& msg) {
    MessageBoxA(nullptr, msg.c_str(), "gfx1151-engine",
                MB_ICONERROR | MB_OK | MB_SETFOREGROUND | MB_TOPMOST);
    return 1;
}

}  // namespace

int main() {
    const std::string root = exe_dir();
    const std::string exe = root + "\\build\\qwenox-win.exe";
    if (GetFileAttributesA(exe.c_str()) == INVALID_FILE_ATTRIBUTES)
        return fail(
            "Missing build\\qwenox-win.exe — build the project first (build_win.bat).\n"
            "缺少 build\\qwenox-win.exe —— 请先编译（build_win.bat）。");

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::string cmd = "\"" + exe + "\"";
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr,
                        root.c_str(), &si, &pi))
        return fail("Failed to start " + exe + ", error " +
                    std::to_string(GetLastError()));
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return 0;
}

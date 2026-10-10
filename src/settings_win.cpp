// settings_win.exe - Windows 设置工具：双击打开配置面板，编辑参数、检查环境、
// 保存 service.conf。不启动服务（启动用 start_win.exe / qwenox-win.exe）。
// 原生 Win32 GUI 子系统，双击不出黑色控制台，无任何脚本宿主依赖。
//
// 面板由本文件末尾的 settings_panel.inc 并入同一匿名命名空间；4 页
// （权重 / 上下文与并发 / 端口与监听 / 显存检查），每页带独立环境检查区，
// 底部 保存 / 关闭 / 语言切换。面板代码源自归档的旧启动器
// （attic/launcher/launch_panel.inc），去掉了"启动服务"流程。
//
// 配置来源（优先级从高到低）：环境变量 > 根目录 service.conf > 内置默认；
// 环境变量覆盖的键在面板检查区给出 [警告] 提示（面板改动不生效）。
// 权重版本与界面语言以注释标记持久化（"# start_win: weights=" /
// "# start_win: lang="），与托盘程序、网页控制台读写同一份标记。
#include "engine_net.h"
#include <windows.h>
#include <shellapi.h>
#include <uxtheme.h>
#include <dwmapi.h>
#include <gdiplus.h>  // 面板左侧立绘（PNG alpha 合成），系统自带组件
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "launch_win_icon.inc"  // kIconIco：窗口图标（tools/win_icon.py 生成）
#include "service_conf.h"       // svcconf：service.conf 共享读写层（与 gdec-api 共用）

// 配置面板在文件末尾以 settings_panel.inc 并入同一匿名命名空间。
namespace {
// kPanelRelang：面板内切换了界面语言（已写回 service.conf 标记），调用方重开面板。
enum PanelResult : int { kPanelClose, kPanelRelang };
PanelResult panel_run();
// 界面语言：定义与解析都在 settings_panel.inc；
// 解析顺序 = service.conf lang 标记 > UI_LANG > 系统 UI 语言（非中文→英文）。
extern bool g_lang_en;
void panel_lang_resolve();
}

// 双语文案：TR(中文, English)，按 g_lang_en 选择。
#define TR(zh, en) (g_lang_en ? (en) : (zh))

namespace {

std::string g_root;
std::wstring g_root_w;  // 宽字符版 exe 目录：ANSI 路径在非 ASCII 目录下会乱码，图片加载等用这条

std::wstring to_w(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
                                      nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), &w[0], n);
    return w;
}

int msgbox(const std::string& text, UINT flags) {
    return MessageBoxW(nullptr, to_w(text).c_str(), L"gfx1151-engine",
                       flags | MB_SETFOREGROUND | MB_TOPMOST);
}

bool file_exists(const std::string& path) {
    DWORD a = GetFileAttributesA(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

// 两个路径是否同一文件（卷序列号 + 文件 ID；打不开时退回字符串比较）。
bool same_file(const std::string& a, const std::string& b) {
    if (a == b) return true;
    BY_HANDLE_FILE_INFORMATION ia{}, ib{};
    auto info = [](const std::string& p, BY_HANDLE_FILE_INFORMATION* out) {
        HANDLE h = CreateFileA(p.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        const bool ok = GetFileInformationByHandle(h, out) != 0;
        CloseHandle(h);
        return ok;
    };
    if (!info(a, &ia) || !info(b, &ib)) return false;
    return ia.dwVolumeSerialNumber == ib.dwVolumeSerialNumber &&
           ia.nFileIndexHigh == ib.nFileIndexHigh && ia.nFileIndexLow == ib.nFileIndexLow;
}

bool dir_exists(const std::string& path) {
    DWORD a = GetFileAttributesA(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

// --- service.conf 解析 -------------------------------------------------------
// 实现在 service_conf.h（svcconf 命名空间，与 gdec-api 共享）：识别
// KEY="v" / KEY='v' / KEY=v 与 bash 缺省语法 "${KEY:-d}" / "${KEY-d}"，
// 值内 $VAR/${VAR} 展开；环境变量总是优先于 conf（置空也算显式设置）。

svcconf::Conf g_conf;

void load_conf(const std::string& path) { svcconf::load_conf(path, &g_conf); }

std::string cfg(const char* key, const std::string& builtin) {
    return svcconf::cfg(g_conf, key, builtin);
}

// MTP_FILE / VISION_FILE 等可选项：显式置空（env 或 conf）即禁用。
std::string cfg_optional(const char* key, const std::string& builtin) {
    return svcconf::cfg_optional(g_conf, key, builtin);
}

// 试探绑定：能 bind 说明端口空闲。
bool port_free(int port) {
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return true;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<u_short>(port));
    bool ok = bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    closesocket(s);
    return ok;
}

// 能连上配置的 host:port 说明对端已在 LISTEN（用于探测服务是否运行中）。
bool port_listening(int port, const std::string& host = engine_net::kDefaultHost) {
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return false;
    sockaddr_in addr{};
    if (!engine_net::address(engine_net::connect_host(host), port, &addr)) {
        closesocket(s);
        return false;
    }
    bool ok = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    closesocket(s);
    return ok;
}

// 从内嵌的 ICO（launch_win_icon.inc）里挑一个尺寸做成 HICON：
// 取不小于 px 的最小条目，都比 px 小就取最大的；由系统缩放到 px。
HICON app_icon(int px) {
    const unsigned char* d = kIconIco;
    const size_t total = sizeof(kIconIco);
    auto u16 = [&](size_t o) { return static_cast<unsigned>(d[o] | d[o + 1] << 8); };
    auto u32 = [&](size_t o) { return static_cast<DWORD>(u16(o) | u16(o + 2) << 16); };
    const unsigned n = u16(4);
    int best = -1, best_size = 0;
    for (unsigned i = 0; i < n && 6 + 16 * (i + 1) <= total; ++i) {
        const int s = d[6 + 16 * i] ? d[6 + 16 * i] : 256;
        const bool better = best < 0 || (s >= px ? (best_size < px || s < best_size)
                                                 : (best_size < px && s > best_size));
        if (better) best = static_cast<int>(i), best_size = s;
    }
    if (best < 0) return nullptr;
    const size_t e = 6 + 16 * static_cast<size_t>(best);
    const DWORD bytes = u32(e + 8), off = u32(e + 12);
    if (off > total || bytes > total - off) return nullptr;
    return CreateIconFromResourceEx(const_cast<PBYTE>(d + off), bytes, TRUE, 0x00030000, px, px,
                                    LR_DEFAULTCOLOR);
}

}  // namespace

// 设置面板（含 save_conf / panel_lang_resolve / panel_run）。
// 必须放在匿名命名空间之外：面板内部要 include <set> 等标准头。
#include "settings_panel.inc"

int main() {
    // 高 DPI 屏上面板图标取对应尺寸、弹框文字不发糊
    SetProcessDPIAware();

    char exe_path[MAX_PATH];
    GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    g_root = exe_path;
    size_t slash = g_root.find_last_of("\\/");
    g_root = slash == std::string::npos ? "." : g_root.substr(0, slash);
    wchar_t exe_w[MAX_PATH];
    GetModuleFileNameW(nullptr, exe_w, MAX_PATH);
    g_root_w = exe_w;
    const size_t wslash = g_root_w.find_last_of(L"\\/");
    if (wslash != std::wstring::npos) g_root_w.resize(wslash);
    SetCurrentDirectoryW(g_root_w.empty() ? L"." : g_root_w.c_str());

    // 界面语言：service.conf lang 标记 > UI_LANG > 系统 UI 语言（非中文→英文）。
    // 标记读文件不依赖 load_conf，尽早解析。
    panel_lang_resolve();

    // 防多开（避免两个设置窗口同时写 service.conf）。句柄不关，进程退出时系统回收。
    // 与 start_win.exe 的 mutex 不同名：服务跑着也能开设置（端口检查会降级为提示）。
    CreateMutexW(nullptr, FALSE, L"Local\\gfx1151-engine-settings");
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        msgbox(TR("设置窗口已经打开了。", "The settings window is already open."),
               MB_OK | MB_ICONINFORMATION);
        return 0;
    }

    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    load_conf(g_root + "\\service.conf");

    // kPanelRelang：语言切换后重开面板（新语言在 panel_run 里重新解析）
    PanelResult pr;
    do pr = panel_run(); while (pr == kPanelRelang);
    return 0;
}

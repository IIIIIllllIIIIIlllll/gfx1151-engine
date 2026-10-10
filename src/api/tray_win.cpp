// tray_win.cpp — 见 tray_win.h。托盘/菜单/气泡/图标逻辑移植自已归档的
// attic/launcher/launch_win.cpp（面板、tee 控制台透传、--check 不移植：
// 配置改在网页 #/engine 完成，日志直接落盘）。
#include "../engine_net.h"  // 须在 windows.h 之前（带来 winsock2）
#include <windows.h>
#include <shellapi.h>
#include <shobjidl.h>  // SetCurrentProcessExplicitAppUserModelID

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include "../launch_win_icon.inc"  // kIconIco：托盘图标（tools/win_icon.py 生成）
#include "../service_conf.h"
#include "engine_sup.h"
#include "http.h"
#include "tray_win.h"

namespace {

TrayHooks g_hooks;
std::string g_root;       // 项目根（main 已把 CWD 切到 conf 所在目录）
std::string g_base_url;   // 本机控制台地址
bool g_lang_en = false;

#define TR(zh, en) (g_lang_en ? (en) : (zh))

std::wstring to_w(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
                                      nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), &w[0], n);
    return w;
}

int msgbox(const std::string& text, UINT flags) {
    return MessageBoxW(nullptr, to_w(text).c_str(), L"Qwenox",
                       flags | MB_SETFOREGROUND | MB_TOPMOST);
}

// 用系统默认程序打开：网址 → 浏览器，文件夹 → 资源管理器，.log → 记事本
void shell_open(const std::string& target) {
    const INT_PTR r = reinterpret_cast<INT_PTR>(
        ShellExecuteA(nullptr, "open", target.c_str(), nullptr, g_root.c_str(), SW_SHOWNORMAL));
    if (r <= 32 && target.size() > 4 && target.compare(target.size() - 4, 4, ".log") == 0)
        ShellExecuteA(nullptr, "open", "notepad.exe", ("\"" + target + "\"").c_str(),
                      g_root.c_str(), SW_SHOWNORMAL);
}

// 界面语言：service.conf 的 "# start_win: lang=" 标记 > UI_LANG > 系统 UI 语言。
void lang_resolve() {
    std::string text;
    if (svcconf::read_text_file(g_root + "\\service.conf", &text)) {
        const char* marker = "# start_win: lang=";
        const size_t p = text.find(marker);
        if (p != std::string::npos) {
            size_t b = p + strlen(marker);
            size_t e = text.find_first_of("\r\n", b);
            std::string v = text.substr(b, e == std::string::npos ? e : e - b);
            const size_t a = v.find_first_not_of(" \t");
            const size_t z = v.find_last_not_of(" \t");
            v = a == std::string::npos ? "" : v.substr(a, z - a + 1);
            if (v == "en") return void(g_lang_en = true);
            if (v == "zh") return void(g_lang_en = false);
        }
    }
    if (const char* e = getenv("UI_LANG")) {
        if (_stricmp(e, "en") == 0) return void(g_lang_en = true);
        if (_stricmp(e, "zh") == 0) return void(g_lang_en = false);
    }
    g_lang_en = PRIMARYLANGID(GetUserDefaultUILanguage()) != LANG_CHINESE;
}

// ---- 图标（移植自 launch_win.cpp） ------------------------------------------
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

// ---- 托盘 --------------------------------------------------------------------
NOTIFYICONDATAW g_nid{};
volatile LONG g_tray_added = 0;
UINT g_wm_taskbar_created = 0;
HWND g_hwnd = nullptr;
const UINT WM_TRAY = WM_APP + 1;

enginesup::Status g_eng;  // 最近一次轮询的引擎状态（仅 UI 线程读写）
std::wstring g_perf;      // 最近一次 running 时的 SNAP 文本（仅 UI 线程）

const UINT WM_PROBE_DONE = WM_APP + 2;

constexpr UINT_PTR kTimerStatus = 1;
constexpr UINT_PTR kTimerPerf = 2;
constexpr uint64_t kPerfTtlMs = 6000;  // 两个 3s 轮询周期：数据必须会过期

enum MenuId : UINT {
    kIdConsole = 1, kIdCopyUrl, kIdEngStart, kIdEngStop,
    kIdEngineLog, kIdApiLog, kIdLogDir, kIdQuit
};

std::wstring state_text() {
    if (g_eng.state == "running")
        return g_eng.external ? TR(L"Qwenox引擎·运行中（外部）",
                                   L"Qwenox engine·running (external)")
                              : TR(L"Qwenox引擎·运行中", L"Qwenox engine·running");
    if (g_eng.state == "starting")
        return TR(L"Qwenox引擎·启动中…", L"Qwenox engine·starting…");
    if (g_eng.state == "exited")
        return TR(L"Qwenox引擎·已退出", L"Qwenox engine·exited");
    return TR(L"Qwenox引擎·已停止", L"Qwenox engine·stopped");
}

// ---- PerfSnap：3s 轮询引擎行协议的 SNAP 动词（移植自 launch_win.cpp）----------
std::wstring perf_count(uint64_t v) {  // 12K 风格缩写
    wchar_t b[24];
    if (v >= 10000) swprintf(b, ARRAYSIZE(b), L"%lluK", (unsigned long long)((v + 500) / 1000));
    else swprintf(b, ARRAYSIZE(b), L"%llu", (unsigned long long)v);
    return b;
}

// 空串 = 查询失败 / 已过期 / idle：tooltip 回归纯状态。
std::wstring perf_tip() {
    const auto ep = g_hooks.engine_endpoint();
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return {};
    const DWORD tv = 500;  // 本机回环，500ms 足够；挡住引擎卡死连累 UI 线程
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof tv);
    sockaddr_in addr{};
    engine_net::address(engine_net::connect_host(ep.first), ep.second, &addr);
    char buf[256];
    int n = -1;
    if (connect(s, (sockaddr*)&addr, sizeof addr) == 0 &&
        send(s, "SNAP\n", 5, 0) == 5)
        n = recv(s, buf, sizeof buf - 1, 0);
    closesocket(s);
    if (n <= 0) return {};
    buf[n] = '\0';
    // P ts_ms state req pp_toks done total tg_inst tg_avg accept_pct
    unsigned long long ts = 0, req = 0, done = 0, total = 0;
    unsigned st = 0;
    double pp = 0, tg_inst = 0, tg_avg = 0, accept = -1;
    if (sscanf(buf, "P %llu %u %llu %lf %llu %llu %lf %lf %lf", &ts, &st, &req, &pp,
               &done, &total, &tg_inst, &tg_avg, &accept) != 9)
        return {};
    if (st == 0) return {};
    const uint64_t now = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
    if (now - ts > kPerfTtlMs) return {};  // 引擎被杀也兜底：ts 不再更新
    wchar_t b[96];
    if (st == 1) {
        const std::wstring d = perf_count(done), t = perf_count(total);
        swprintf(b, ARRAYSIZE(b), L"pp %.0f tok/s · %ls/%ls", pp, d.c_str(), t.c_str());
    } else if (accept >= 0) {
        swprintf(b, ARRAYSIZE(b), TR(L"tg %.1f tok/s · 接受率 %.1f%%", L"tg %.1f tok/s · accept %.1f%%"),
                 tg_inst, accept);
    } else {
        swprintf(b, ARRAYSIZE(b), L"tg %.1f tok/s", tg_inst);
    }
    return b;
}

void tray_update_tip() {
    std::wstring tip = state_text();
    if (g_eng.state == "running" && !g_perf.empty()) tip += L"\n" + g_perf;
    g_nid.uFlags = NIF_TIP;
    lstrcpynW(g_nid.szTip, tip.c_str(), ARRAYSIZE(g_nid.szTip));
    if (g_tray_added) Shell_NotifyIconW(NIM_MODIFY, &g_nid);
}

void tray_balloon(const std::wstring& title, const std::wstring& text);  // 定义在下方

// ---- 状态探测 worker ---------------------------------------------------------
// UI 线程不做任何 socket 操作：status() 的端口探测（500ms 上限）与 perf_tip()
// 的 SNAP 查询（阻塞 connect，某些机器环回 RST 被过滤驱动延迟 ~2s）都会卡满
// 超时。探测放 worker 线程，结果经 WM_PROBE_DONE 回 UI 线程应用（气泡/菜单/
// 定时器只许在 UI 线程动）。g_eng/g_perf 仅 UI 线程访问；互斥锁只保护
// worker → UI 的交接字段。

std::mutex g_probe_mtx;
std::condition_variable g_probe_cv;
bool g_probe_kicked = false;         // 有在跑/待跑的探测
bool g_probe_quit = false;
bool g_probe_result_ready = false;
enginesup::Status g_probe_eng;
std::wstring g_probe_perf;

void probe_kick() {
    {
        std::lock_guard<std::mutex> lk(g_probe_mtx);
        if (g_probe_kicked || g_probe_quit) return;  // 上一次还没跑完：跳过本次
        g_probe_kicked = true;
    }
    g_probe_cv.notify_one();
}

void probe_worker() {
    std::unique_lock<std::mutex> lk(g_probe_mtx);
    for (;;) {
        g_probe_cv.wait(lk, [] { return g_probe_kicked || g_probe_quit; });
        if (g_probe_quit) return;
        lk.unlock();
        const auto ep = g_hooks.engine_endpoint();
        enginesup::Status eng = enginesup::status(ep.first, ep.second);
        std::wstring perf;
        if (eng.state == "running") perf = perf_tip();
        lk.lock();
        g_probe_eng = std::move(eng);
        g_probe_perf = std::move(perf);
        g_probe_kicked = false;
        g_probe_result_ready = true;
        PostMessageW(g_hwnd, WM_PROBE_DONE, 0, 0);
    }
}

// UI 线程（WM_PROBE_DONE）：应用 worker 的探测结果，并就绪/意外退出时弹气泡。
void apply_probe() {
    const std::string prev = g_eng.state;
    {
        std::lock_guard<std::mutex> lk(g_probe_mtx);
        if (!g_probe_result_ready) return;
        g_eng = g_probe_eng;
        g_perf = g_probe_perf;
        g_probe_result_ready = false;
    }
    if (g_eng.state == "running" && prev != "running") {
        SetTimer(g_hwnd, kTimerPerf, 3000, nullptr);
        if (prev == "starting")
            tray_balloon(TR(L"引擎已就绪", L"Engine ready"),
                         to_w(g_base_url + "/v1") + L"\n" +
                             TR(L"双击图标打开控制台，右键查看更多",
                                L"Double-click the icon to open the console; "
                                L"right-click for more"));
    } else if (g_eng.state != "running" && prev == "running") {
        KillTimer(g_hwnd, kTimerPerf);
    }
    if (g_eng.state == "exited" && (prev == "running" || prev == "starting")) {
        const std::string code = std::to_string(g_eng.exit_code);
        tray_balloon(TR(L"引擎已退出", L"Engine exited"),
                     to_w(TR("退出码 " , "exit code ") + code +
                          (g_eng.detail.empty() ? "" : " · " + g_eng.detail) +
                          TR("\n右键图标查看引擎日志",
                             "\nRight-click the icon to view the engine log")));
    }
    tray_update_tip();
}

void tray_add() {
    g_nid.cbSize = sizeof(g_nid);
    g_nid.hWnd = g_hwnd;
    g_nid.uID = 1;
    g_nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
    g_nid.uCallbackMessage = WM_TRAY;
    static HICON small_icon = app_icon(GetSystemMetrics(SM_CXSMICON));
    static HICON big_icon = app_icon(GetSystemMetrics(SM_CXICON));  // 通知气泡用
    g_nid.hIcon = small_icon ? small_icon : LoadIconA(nullptr, IDI_APPLICATION);
    g_nid.hBalloonIcon = big_icon;
    const std::wstring tip = state_text();
    lstrcpynW(g_nid.szTip, tip.c_str(), ARRAYSIZE(g_nid.szTip));
    if (Shell_NotifyIconW(NIM_ADD, &g_nid)) {
        InterlockedExchange(&g_tray_added, 1);
        // 128 字符多行 tooltip（uVersion 0 只有 64 字符）
        g_nid.uVersion = NOTIFYICON_VERSION;
        Shell_NotifyIconW(NIM_SETVERSION, &g_nid);
    }
}

void tray_remove() {
    if (InterlockedExchange(&g_tray_added, 0)) Shell_NotifyIconW(NIM_DELETE, &g_nid);
}

// 通知应用名注册：HKCU\SOFTWARE\Classes\AppUserModelIDs\Qwenox 提供
// DisplayName/IconUri（toast 标题与通知设置页用），再给本进程设同名
// AUMID。须在第一次 Shell_NotifyIcon 之前调用。每次启动幂等重写。
void register_app_id() {
    HKEY key = nullptr;
    if (RegCreateKeyExW(HKEY_CURRENT_USER,
                        L"SOFTWARE\\Classes\\AppUserModelIDs\\Qwenox", 0, nullptr,
                        0, KEY_WRITE, nullptr, &key, nullptr) == ERROR_SUCCESS) {
        const wchar_t* name = L"Qwenox";
        RegSetValueExW(key, L"DisplayName", 0, REG_SZ,
                       reinterpret_cast<const BYTE*>(name),
                       static_cast<DWORD>((wcslen(name) + 1) * sizeof(wchar_t)));
        const std::wstring ico = to_w(g_root + "\\src\\launch_win.ico");
        if (GetFileAttributesW(ico.c_str()) != INVALID_FILE_ATTRIBUTES)
            RegSetValueExW(key, L"IconUri", 0, REG_SZ,
                           reinterpret_cast<const BYTE*>(ico.c_str()),
                           static_cast<DWORD>((ico.size() + 1) * sizeof(wchar_t)));
        RegCloseKey(key);
    }
    SetCurrentProcessExplicitAppUserModelID(L"Qwenox");
}

void tray_balloon(const std::wstring& title, const std::wstring& text) {
    if (!g_tray_added) return;
    g_nid.uFlags = NIF_INFO;
    lstrcpynW(g_nid.szInfoTitle, title.c_str(), ARRAYSIZE(g_nid.szInfoTitle));
    lstrcpynW(g_nid.szInfo, text.c_str(), ARRAYSIZE(g_nid.szInfo));
    g_nid.dwInfoFlags = g_nid.hBalloonIcon ? (NIIF_USER | NIIF_LARGE_ICON) : NIIF_INFO;
    Shell_NotifyIconW(NIM_MODIFY, &g_nid);
}

void copy_text(const std::wstring& s) {
    if (!OpenClipboard(g_hwnd)) return;
    EmptyClipboard();
    const size_t bytes = (s.size() + 1) * sizeof(wchar_t);
    HGLOBAL g = GlobalAlloc(GMEM_MOVEABLE, bytes);
    if (g) {
        void* dst = GlobalLock(g);
        if (dst) {
            memcpy(dst, s.c_str(), bytes);
            GlobalUnlock(g);
            if (!SetClipboardData(CF_UNICODETEXT, g)) GlobalFree(g);
        } else {
            GlobalFree(g);
        }
    }
    CloseClipboard();
}

// ---- 引擎启停（菜单动作） -----------------------------------------------------
void engine_start() {
    std::string err;
    if (!enginesup::start(g_root, g_hooks.conf_snapshot(), &err)) {
        msgbox(TR("启动引擎失败：", "Failed to start the engine: ") + err,
               MB_OK | MB_ICONERROR);
        return;
    }
    tray_balloon(TR(L"引擎启动中", L"Engine starting"),
                 TR(L"冷启动可能要几分钟，就绪后会再提示。",
                    L"A cold start can take a few minutes; you'll be notified when ready."));
    // 立刻刷一次状态（不等 2s 轮询），让菜单/tooltip 马上变成"启动中"；
    // start() 已把状态置为 starting，status() 此刻不做端口探测（瞬时返回）
    const auto ep = g_hooks.engine_endpoint();
    g_eng = enginesup::status(ep.first, ep.second);
    tray_update_tip();
    probe_kick();  // 让 worker 尽快接上后续迁移（starting → running）
}

void engine_stop() {
    std::string err;
    if (!enginesup::stop(&err))
        msgbox(TR("停止引擎失败：", "Failed to stop the engine: ") + err,
               MB_OK | MB_ICONERROR);
}

[[noreturn]] void quit_all(UINT code) {
    {
        std::lock_guard<std::mutex> lk(g_probe_mtx);
        g_probe_quit = true;
    }
    g_probe_cv.notify_one();  // 唤醒探测 worker 退出（不 join：ExitProcess 兜底）
    tray_remove();
    ExitProcess(code);  // 引擎在 KILL_ON_JOB_CLOSE Job 里，随本进程结束
}

void on_menu(UINT id) {
    switch (id) {
        case kIdConsole: shell_open(g_base_url + "/"); break;
        case kIdCopyUrl:
            copy_text(to_w(g_base_url + "/v1"));
            tray_balloon(TR(L"已复制", L"Copied"), to_w(g_base_url + "/v1"));
            break;
        case kIdEngStart: engine_start(); break;
        case kIdEngStop: engine_stop(); break;
        case kIdEngineLog:
            if (!g_eng.log.empty()) shell_open(g_eng.log);
            break;
        case kIdApiLog:
            if (!g_hooks.api_log.empty()) shell_open(g_hooks.api_log);
            break;
        case kIdLogDir: shell_open(g_root + "\\logs"); break;
        case kIdQuit: {
            const bool ours_running =
                g_eng.state == "running" && !g_eng.external || g_eng.state == "starting";
            if (!ours_running ||
                msgbox(TR("退出会同时停止引擎，正在进行的生成会中断。\n确定退出？",
                          "Quitting stops the engine; any ongoing generation will be "
                          "interrupted.\nQuit now?"),
                       MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON2) == IDYES)
                quit_all(0);
            break;
        }
        default: break;
    }
}

void show_menu() {
    HMENU m = CreatePopupMenu();
    if (!m) return;
    AppendMenuW(m, MF_STRING | MF_GRAYED, 0, state_text().c_str());
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING, kIdConsole, TR(L"打开控制台", L"Open console"));
    AppendMenuW(m, MF_STRING, kIdCopyUrl, TR(L"复制 API 地址", L"Copy API URL"));
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    const bool can_start = g_eng.state == "stopped" || g_eng.state == "exited";
    const bool can_stop = (g_eng.state == "running" && !g_eng.external) ||
                          g_eng.state == "starting";
    AppendMenuW(m, MF_STRING | (can_start ? 0 : MF_GRAYED), kIdEngStart,
                TR(L"启动引擎", L"Start engine"));
    AppendMenuW(m, MF_STRING | (can_stop ? 0 : MF_GRAYED), kIdEngStop,
                TR(L"停止引擎", L"Stop engine"));
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING | (g_eng.log.empty() ? MF_GRAYED : 0), kIdEngineLog,
                TR(L"查看引擎日志", L"View engine log"));
    AppendMenuW(m, MF_STRING | (g_hooks.api_log.empty() ? MF_GRAYED : 0), kIdApiLog,
                TR(L"查看 API 日志", L"View API log"));
    AppendMenuW(m, MF_STRING, kIdLogDir, TR(L"打开日志文件夹", L"Open logs folder"));
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING, kIdQuit, TR(L"退出", L"Quit"));
    POINT pt;
    GetCursorPos(&pt);
    SetForegroundWindow(g_hwnd);  // 否则点菜单外面菜单不消失
    const UINT id = static_cast<UINT>(TrackPopupMenu(
        m, TPM_RETURNCMD | TPM_NONOTIFY | TPM_RIGHTBUTTON, pt.x, pt.y, 0, g_hwnd, nullptr));
    PostMessageW(g_hwnd, WM_NULL, 0, 0);
    DestroyMenu(m);
    on_menu(id);
}

// 2s 轮询引擎状态（kTimerStatus → probe_kick）：驱动菜单/tooltip，并就绪/
// 意外退出时弹气泡（迁移逻辑在 apply_probe，由 WM_PROBE_DONE 触发）。

LRESULT CALLBACK wnd_proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_TRAY) {
        switch (LOWORD(lp)) {
            case WM_RBUTTONUP:
            case WM_CONTEXTMENU: show_menu(); break;
            case WM_LBUTTONDBLCLK: on_menu(kIdConsole); break;
            default: break;
        }
        return 0;
    }
    if (msg == WM_TIMER) {
        probe_kick();  // kTimerStatus(2s) / kTimerPerf(3s) 都只是踢探测 worker
        return 0;
    }
    if (msg == WM_PROBE_DONE) {
        apply_probe();
        return 0;
    }
    if (g_wm_taskbar_created && msg == g_wm_taskbar_created) {
        InterlockedExchange(&g_tray_added, 0);
        tray_add();
        return 0;
    }
    if (msg == WM_ENDSESSION && wp) quit_all(0);  // 注销 / 关机
    return DefWindowProcW(h, msg, wp, lp);
}

}  // namespace

int tray_run(http::Server& srv, const TrayHooks& hooks) {
    g_hooks = hooks;
    char cwd[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, cwd);
    g_root = cwd;
    lang_resolve();
    // Win10+ 气泡被转成 toast，通知标题/设置页里的"应用名"取自进程
    // AppUserModelID，不注册就显示 exe 文件名（qwenox-win.exe）。
    // 注册 HKCU\...\AppUserModelIDs\Qwenox + 进程 AUMID 后显示"Qwenox"。
    register_app_id();
    const std::string& h = g_hooks.api_host;
    const std::string local = h.empty() || h == "0.0.0.0" || h == "::" ? "127.0.0.1" : h;
    g_base_url = "http://" + local + ":" + std::to_string(g_hooks.api_port);

    SetProcessDPIAware();
    HINSTANCE hi = GetModuleHandleW(nullptr);
    WNDCLASSW wc{};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = hi;
    wc.lpszClassName = L"gfx1151_qwenox_api";
    RegisterClassW(&wc);
    // 普通的隐藏顶层窗口（不 ShowWindow）：消息窗口收不到 TaskbarCreated 广播
    g_hwnd = CreateWindowExW(0, wc.lpszClassName, L"Qwenox", WS_OVERLAPPED, 0, 0, 0, 0,
                             nullptr, nullptr, hi, nullptr);
    if (!g_hwnd) {
        msgbox(TR("无法创建托盘窗口，错误码 ", "Cannot create tray window, error ") +
                   std::to_string(GetLastError()),
               MB_OK | MB_ICONERROR);
        return 1;
    }
    g_wm_taskbar_created = RegisterWindowMessageW(L"TaskbarCreated");

    // HTTP 服务在 worker 线程跑，主线程跑消息循环
    std::thread worker([&srv] { srv.run(); });
    worker.detach();
    std::thread(probe_worker).detach();  // 引擎状态/SNAP 探测（socket 操作）

    tray_add();
    probe_kick();  // 初始状态（可能有外部引擎已在听）
    SetTimer(g_hwnd, kTimerStatus, 2000, nullptr);
    tray_balloon(TR(L"Qwenox 已启动", L"Qwenox started"),
                 to_w(g_base_url + "/") + L"\n" +
                     TR(L"双击图标打开控制台；引擎需在控制台或右键菜单中手动启动。",
                        L"Double-click the icon to open the console; start the engine "
                        L"from the console or the right-click menu."));

    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0) > 0) {
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    quit_all(0);
}

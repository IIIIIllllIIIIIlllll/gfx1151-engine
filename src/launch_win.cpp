// start_win.exe - Windows 启动器：拉起引擎 + OpenAI API 双进程。
// 原生 Win32，无任何脚本宿主依赖（不需要 Git Bash / PowerShell）。
//
// 默认是托盘程序：双击后不开黑色控制台，只在任务栏右下角放一个图标。
// 右键图标：打开面板 / 复制 API 地址 / 查看日志 / 退出（同时停掉两个子进程）；
// 双击图标：已就绪时打开面板，否则打开引擎日志。出错或子进程退出时弹框提示。
// （控制台窗口在"快速编辑模式"下被鼠标点一下就会暂停输出，是 Windows 下
// 推理卡住的常见原因之一；托盘模式根本没有控制台。）
//
//   start_win.exe --console   旧的控制台模式（新开一个控制台窗口实时显示输出，
//                             Ctrl+C 或关窗停止），排查问题用
//   start_win.exe --check     只检查配置不启动（结果打印到调用它的控制台）
//
// 配置来源（优先级从高到低）：环境变量 > 根目录 service.conf > 内置默认。
// 换模型文件名、改上下文窗口等，直接编辑 service.conf（与 Linux 同一文件）。
//   set MAX_CONTEXT=131072 && start_win.exe   （环境变量临时覆盖）
//   VISION_FILE 置空 = 纯文本不挂视觉塔；MTP_FILE 置空 = 不用投机草稿；
//   OVERLAY_FILE 置空（默认）= 无覆盖层
// START_TIMEOUT 等计时项与平台相关，不读 conf（Windows 冷加载分钟级）。
//
// 子进程输出写入 logs\engine-win-<时间戳>.log / logs\api-win-<时间戳>.log，
// 启动器自己的输出写入 logs\launcher-win-<时间戳>.log（托盘模式）。
// 控制台模式下输出同时透传到控制台；控制台卡住（选择文本、终端忙）不会阻塞
// 子进程：日志文件照写，控制台显示最多缓冲 8 MiB，超出部分只在控制台上丢弃
// 并提示。快速编辑模式运行期间关闭。
// 两个子进程放在一个 Job 里：启动器无论怎样退出（包括被任务管理器结束），
// 引擎和 API 都会随之结束，不会留下占着显存的孤儿进程。
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

#include "launch_win_icon.inc"  // kIconIco：托盘图标（tools/win_icon.py 生成）

// 配置面板在文件末尾以 launch_panel.inc 并入同一匿名命名空间；
// report()/on_menu() 在其定义点之前使用这些入口，需要前向声明。
// PanelResult 的完整定义放在这里（枚举值的可见性要求），.inc 直接使用。
namespace {
// kPanelRelang：面板内切换了界面语言（已写回 service.conf 标记），调用方重开面板。
enum PanelResult : int { kPanelStart, kPanelCancel, kPanelRelang };
PanelResult panel_run(bool allow_start);
// 界面语言（托盘与面板共用）：定义与解析都在 launch_panel.inc；
// 解析顺序 = service.conf lang 标记 > UI_LANG > 系统 UI 语言（非中文→英文）。
extern bool g_lang_en;
void panel_lang_resolve();
}

// 双语文案：TR(中文, English)，按 g_lang_en 选择。
#define TR(zh, en) (g_lang_en ? (en) : (zh))

namespace {

std::string g_root;
std::wstring g_root_w;  // 宽字符版 exe 目录：ANSI 路径在非 ASCII 目录下会乱码，图片加载等用这条
std::string g_stamp;                        // 本次启动的时间戳，日志文件名用
HANDLE g_children[2] = {nullptr, nullptr};  // 0 = 引擎，1 = API
HANDLE g_job = nullptr;
bool g_own_console = false;  // 控制台是本进程 AllocConsole 出来的（退出前 pause）
bool g_tray = true;          // false = --console / --check
bool g_console_on = false;   // 子进程输出是否透传到控制台
bool g_logs_started = false; // 已开始写子进程日志（出错时提示去看日志）

void console_drain(DWORD ms);  // 定义在下方"控制台输出"一节
void console_mode_restore();
void tray_remove();            // 定义在下方"托盘"一节

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

// 用系统默认程序打开：网址 → 浏览器，文件夹 → 资源管理器，.log → 记事本
void shell_open(const std::string& target) {
    const INT_PTR r = reinterpret_cast<INT_PTR>(
        ShellExecuteA(nullptr, "open", target.c_str(), nullptr, g_root.c_str(), SW_SHOWNORMAL));
    if (r <= 32 && target.size() > 4 && target.compare(target.size() - 4, 4, ".log") == 0)
        ShellExecuteA(nullptr, "open", "notepad.exe", ("\"" + target + "\"").c_str(),
                      g_root.c_str(), SW_SHOWNORMAL);
}

void pause_if_own_console() {
    console_mode_restore();
    if (g_own_console) system("pause");
}

[[noreturn]] void fail(const std::string& msg) {
    console_drain(2000);  // 先让子进程最后的输出上屏，错误信息排在其后
    fprintf(stderr, "%s%s\n", TR("错误：", "Error: "), msg.c_str());
    fflush(stderr);
    for (HANDLE h : g_children)
        if (h) TerminateProcess(h, 1);
    if (g_tray) {
        tray_remove();
        if (g_logs_started) {
            if (msgbox(TR("错误：", "Error: ") + msg +
                           TR("\n\n是否打开日志文件夹？", "\n\nOpen the logs folder?"),
                       MB_YESNO | MB_ICONERROR) == IDYES)
                shell_open(g_root + "\\logs");
        } else {
            msgbox(TR("错误：", "Error: ") + msg, MB_OK | MB_ICONERROR);
        }
    } else {
        pause_if_own_console();
    }
    ExitProcess(1);
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

int env_int(const char* name, int fallback, int lo, int hi) {
    const char* v = getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    long n = strtol(v, &end, 10);
    if (!end || *end || n < lo || n > hi)
        fail(std::string(name) + " 必须是 " + std::to_string(lo) + "-" +
             std::to_string(hi) + " 的整数（当前为 \"" + v + "\"）");
    return static_cast<int>(n);
}

// --- service.conf 极简解析 -------------------------------------------------
// 识别 KEY="v" / KEY='v' / KEY=v 与 bash 缺省语法 "${KEY:-d}" / "${KEY-d}"，
// 值内支持 $VAR / ${VAR} 展开（取环境变量或 conf 中先解析的键）。
// 环境变量总是优先于 conf（置空也算显式设置，用于禁用可选文件）。

std::map<std::string, std::string> g_conf;

std::string expand(const std::string& in) {
    std::string out;
    for (size_t i = 0; i < in.size();) {
        if (in[i] != '$') {
            out += in[i++];
            continue;
        }
        std::string name;
        size_t j = i + 1;
        if (j < in.size() && in[j] == '{') {
            size_t k = in.find('}', j);
            if (k == std::string::npos) {
                out += in[i++];
                continue;
            }
            name = in.substr(j + 1, k - j - 1);
            i = k + 1;
        } else {
            while (j < in.size() && (isalnum(static_cast<unsigned char>(in[j])) ||
                                     in[j] == '_'))
                name += in[j++];
            if (name.empty()) {
                out += in[i++];
                continue;
            }
            i = j;
        }
        const char* e = getenv(name.c_str());
        auto it = g_conf.find(name);
        if (e) out += e;
        else if (it != g_conf.end()) out += it->second;
    }
    return out;
}

std::string trim(const std::string& s) {
    // Windows 上 service.conf 常带 CRLF（如 git autocrlf=true 签出）：
    // 不去掉 \r 会让行尾引号剥不掉、"${KEY:-d}" 识别失败，值变成 ""\r 之类
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

void load_conf(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return;
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
    fclose(f);
    size_t pos = 0;
    while (pos < text.size()) {
        size_t eol = text.find('\n', pos);
        std::string line = text.substr(pos, eol == std::string::npos
                                              ? std::string::npos : eol - pos);
        pos = eol == std::string::npos ? text.size() : eol + 1;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        if (key.empty() ||
            key.find_first_not_of(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_") !=
                std::string::npos)
            continue;
        std::string rhs = trim(line.substr(eq + 1));
        if (rhs.size() >= 2 && (rhs[0] == '"' || rhs[0] == '\'') &&
            rhs.back() == rhs[0])
            rhs = rhs.substr(1, rhs.size() - 2);
        const char* env = getenv(key.c_str());
        const std::string colon_def = "${" + key + ":-";
        const std::string plain_def = "${" + key + "-";
        if (rhs.compare(0, colon_def.size(), colon_def) == 0 && rhs.back() == '}') {
            if (env && *env) continue;  // :- 语义：非空环境变量胜出
            g_conf[key] = expand(rhs.substr(colon_def.size(),
                                            rhs.size() - colon_def.size() - 1));
        } else if (rhs.compare(0, plain_def.size(), plain_def) == 0 &&
                   rhs.back() == '}') {
            if (env) continue;  // - 语义：环境变量存在即胜出（置空也算）
            g_conf[key] = expand(rhs.substr(plain_def.size(),
                                            rhs.size() - plain_def.size() - 1));
        } else {
            if (env) continue;  // 字面量：环境变量优先
            g_conf[key] = expand(rhs);
        }
    }
}

std::string cfg(const char* key, const std::string& builtin) {
    const char* e = getenv(key);
    if (e && *e) return e;
    auto it = g_conf.find(key);
    if (it != g_conf.end() && !it->second.empty()) return it->second;
    return builtin;
}

// MTP_FILE / VISION_FILE 等可选项：显式置空（env 或 conf）即禁用。
std::string cfg_optional(const char* key, const std::string& builtin) {
    const char* e = getenv(key);
    if (e) return e;
    auto it = g_conf.find(key);
    if (it != g_conf.end()) return it->second;
    return builtin;
}

int cfg_int(const char* key, int fallback, int lo, int hi) {
    const std::string& v = cfg(key, "");
    if (v.empty()) return fallback;
    char* end = nullptr;
    long n = strtol(v.c_str(), &end, 10);
    if (!end || *end || n < lo || n > hi)
        fail(std::string(key) + " 必须是 " + std::to_string(lo) + "-" +
             std::to_string(hi) + " 的整数（当前为 \"" + v + "\"）");
    return static_cast<int>(n);
}

// service.conf 的小数项（ROPE_*）：完整解析、有限、非负；范围约束由调用方检查。
double parse_double(const char* key, const std::string& v) {
    char* end = nullptr;
    errno = 0;
    const double n = strtod(v.c_str(), &end);
    if (errno || end == v.c_str() || !end || *end || n != n || n < 0)
        fail(std::string(key) + " 必须为非负小数（当前为 \"" + v + "\"）");
    return n;
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

// 能连上配置的 host:port 说明对端已在 LISTEN。
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

std::string quote(const std::string& s) {
    return "\"" + s + "\"";
}

struct Child {
    HANDLE proc = nullptr;
    HANDLE pipe_read = nullptr;
    FILE* log = nullptr;
    std::string name;
};

// ---- 控制台输出：永不阻塞子进程（仅 --console 模式） ------------------------
// 子进程 stdout/stderr → 匿名管道 → tee_thread：先写日志文件（始终完整），
// 再放进有界内存队列，由唯一的 console_thread 写控制台。
// 旧实现在 tee_thread 里同步 WriteFile 控制台：控制台一暂停（快速编辑模式下
// 鼠标点一下进入"选择"、终端忙），tee 不再读管道，管道写满后引擎/API 的每个
// fprintf 都阻塞——decode 停在半路，kvsnap 写线程持锁打印还会连带卡住 GPU 线程，
// 日志也同时停住。现在控制台卡住最多丢掉控制台上的文字（队列满之后），
// 子进程照常跑，日志文件不丢。托盘模式没有控制台，tee 只写日志文件。
struct ConsoleQueue {
    SRWLOCK lock = SRWLOCK_INIT;
    CONDITION_VARIABLE cv = CONDITION_VARIABLE_INIT;
    std::string buf;        // 待写控制台的字节
    size_t dropped = 0;     // buf 之后被丢弃的字节数
    bool busy = false;      // console_thread 正在 WriteFile
};
ConsoleQueue g_cq;
const size_t kConsoleQueueMax = 8u << 20;  // 8 MiB ≈ 数万行日志

void console_push(const char* p, size_t n) {
    AcquireSRWLockExclusive(&g_cq.lock);
    // 一旦开始丢弃就一直丢到 console_thread 取走 buf，保证丢弃段是连续的
    if (g_cq.dropped || g_cq.buf.size() + n > kConsoleQueueMax)
        g_cq.dropped += n;
    else
        g_cq.buf.append(p, n);
    ReleaseSRWLockExclusive(&g_cq.lock);
    WakeAllConditionVariable(&g_cq.cv);
}

bool console_write_all(HANDLE out, const char* p, size_t n) {
    while (n) {
        DWORD w = 0;
        const DWORD want = n > (1u << 20) ? (1u << 20) : static_cast<DWORD>(n);
        if (!WriteFile(out, p, want, &w, nullptr) || w == 0) return false;
        p += w;
        n -= w;
    }
    return true;
}

DWORD WINAPI console_thread(LPVOID) {
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    std::string chunk;
    for (;;) {
        AcquireSRWLockExclusive(&g_cq.lock);
        while (g_cq.buf.empty() && !g_cq.dropped)
            SleepConditionVariableSRW(&g_cq.cv, &g_cq.lock, INFINITE, 0);
        chunk.swap(g_cq.buf);
        g_cq.buf.clear();
        const size_t dropped = g_cq.dropped;
        g_cq.dropped = 0;
        g_cq.busy = true;
        ReleaseSRWLockExclusive(&g_cq.lock);

        console_write_all(out, chunk.data(), chunk.size());  // 可能在这里卡住，但只卡本线程
        if (dropped) {
            char m[200];
            snprintf(m, sizeof m,
                     "\n[启动器] 控制台输出曾暂停，期间丢弃 %zu 字节（仅控制台显示，"
                     "logs\\ 下的日志文件完整）\n",
                     dropped);
            console_write_all(out, m, strlen(m));
        }
        chunk.clear();

        AcquireSRWLockExclusive(&g_cq.lock);
        g_cq.busy = false;
        ReleaseSRWLockExclusive(&g_cq.lock);
        WakeAllConditionVariable(&g_cq.cv);
    }
}

// 退出前把队列里剩下的输出写完（最多等 ms 毫秒，控制台卡住也不会永久挂起）
void console_drain(DWORD ms) {
    const ULONGLONG deadline = GetTickCount64() + ms;
    AcquireSRWLockExclusive(&g_cq.lock);
    while (!g_cq.buf.empty() || g_cq.dropped || g_cq.busy) {
        const ULONGLONG now = GetTickCount64();
        if (now >= deadline) break;
        SleepConditionVariableSRW(&g_cq.cv, &g_cq.lock, static_cast<DWORD>(deadline - now), 0);
    }
    ReleaseSRWLockExclusive(&g_cq.lock);
}

// 关闭控制台"快速编辑模式"：它让鼠标单击就进入选择状态并暂停所有输出。
// 退出时恢复原模式。
HANDLE g_con_in = INVALID_HANDLE_VALUE;
DWORD g_con_in_mode = 0;
bool g_con_mode_saved = false;
void console_quickedit_off() {
    g_con_in = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode = 0;
    if (g_con_in == INVALID_HANDLE_VALUE || !GetConsoleMode(g_con_in, &mode)) return;
    g_con_in_mode = mode;
    g_con_mode_saved = true;
    SetConsoleMode(g_con_in, (mode | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE);
}
void console_mode_restore() {
    if (g_con_mode_saved) SetConsoleMode(g_con_in, g_con_in_mode);
}

DWORD WINAPI tee_thread(LPVOID param) {
    Child* c = static_cast<Child*>(param);
    char buf[65536];
    DWORD n = 0;
    while (ReadFile(c->pipe_read, buf, sizeof(buf), &n, nullptr) && n > 0) {
        if (c->log) {  // 日志先写：控制台怎样都不影响它
            fwrite(buf, 1, n, c->log);
            fflush(c->log);
        }
        if (g_console_on) console_push(buf, n);
    }
    return 0;
}

Child spawn(const std::string& name, const std::string& exe,
            const std::vector<std::string>& args, const std::string& log_path) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE pipe_read = nullptr, pipe_write = nullptr;
    // 1 MiB 管道缓冲（默认只有约 4 KB）：tee 线程短暂落后时子进程也不必等
    if (!CreatePipe(&pipe_read, &pipe_write, &sa, 1u << 20))
        fail("CreatePipe 失败");
    SetHandleInformation(pipe_read, HANDLE_FLAG_INHERIT, 0);
    // 托盘模式没有控制台：子进程 stdin 接 NUL
    HANDLE in = g_tray ? CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                     &sa, OPEN_EXISTING, 0, nullptr)
                       : GetStdHandle(STD_INPUT_HANDLE);

    std::string cmd = quote(exe);
    for (const std::string& a : args) cmd += " " + quote(a);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = pipe_write;
    si.hStdError = pipe_write;
    si.hStdInput = in == INVALID_HANDLE_VALUE ? nullptr : in;
    PROCESS_INFORMATION pi{};
    std::vector<char> cmdline(cmd.begin(), cmd.end());
    cmdline.push_back('\0');
    // 托盘模式：CREATE_NO_WINDOW，否则每个控制台子进程都会弹出自己的黑窗
    const DWORD flags = g_tray ? CREATE_NO_WINDOW : 0;
    const BOOL ok = CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, TRUE, flags,
                                   nullptr, g_root.c_str(), &si, &pi);
    const DWORD err = GetLastError();
    if (g_tray && in != INVALID_HANDLE_VALUE) CloseHandle(in);
    CloseHandle(pipe_write);
    if (!ok) {
        CloseHandle(pipe_read);
        fail("启动失败（" + exe + "），错误码 " + std::to_string(err));
    }
    if (g_job) AssignProcessToJobObject(g_job, pi.hProcess);  // 失败也不影响运行
    CloseHandle(pi.hThread);

    Child c;
    c.proc = pi.hProcess;
    c.pipe_read = pipe_read;
    c.name = name;
    c.log = fopen(log_path.c_str(), "wb");
    if (!c.log) fprintf(stderr, "警告：无法写日志 %s\n", log_path.c_str());
    g_logs_started = true;
    HANDLE t = CreateThread(nullptr, 0, tee_thread, new Child(c), 0, nullptr);
    if (t) CloseHandle(t);
    return c;
}

bool alive(HANDLE h) {
    return WaitForSingleObject(h, 0) == WAIT_TIMEOUT;
}

void kill_child(Child& c) {
    if (c.proc && alive(c.proc)) TerminateProcess(c.proc, 1);
}

void kill_all_children() {
    for (HANDLE h : g_children)
        if (h) TerminateProcess(h, 1);
}

BOOL WINAPI on_ctrl(DWORD ev) {
    if (ev == CTRL_C_EVENT || ev == CTRL_BREAK_EVENT || ev == CTRL_CLOSE_EVENT) {
        kill_all_children();
        console_mode_restore();
        ExitProcess(1);
    }
    return FALSE;
}

// 本程序是 GUI 子系统（双击不出黑窗）。--console / --check 需要控制台：
// --check 先尝试挂到调用者（cmd）的控制台上，不行就新开一个窗口。
void console_open(bool attach_parent) {
    const bool attached = attach_parent && AttachConsole(ATTACH_PARENT_PROCESS);
    if (!attached) {
        if (!AllocConsole()) return;
        g_own_console = true;
    }
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;  // 子进程共用这个控制台
    HANDLE out = CreateFileA("CONOUT$", GENERIC_READ | GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    HANDLE in = CreateFileA("CONIN$", GENERIC_READ | GENERIC_WRITE,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    if (out != INVALID_HANDLE_VALUE) {
        SetStdHandle(STD_OUTPUT_HANDLE, out);
        SetStdHandle(STD_ERROR_HANDLE, out);
    }
    if (in != INVALID_HANDLE_VALUE) SetStdHandle(STD_INPUT_HANDLE, in);
    freopen("CONOUT$", "w", stdout);
    freopen("CONOUT$", "w", stderr);
    freopen("CONIN$", "r", stdin);
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCtrlHandler(on_ctrl, TRUE);
    if (attached) printf("\n");  // cmd 的提示符已经先打印了，另起一行
}

// 托盘模式：启动器自己的 printf/fprintf 写到 logs\launcher-win-<时间戳>.log
void launcher_log_open() {
    const std::string path = "logs\\launcher-win-" + g_stamp + ".log";
    FILE* f = fopen(path.c_str(), "wb");  // 先清空，stdout/stderr 再以追加方式共用它
    if (!f) return;
    fclose(f);
    if (freopen(path.c_str(), "ab", stdout)) setvbuf(stdout, nullptr, _IONBF, 0);
    if (freopen(path.c_str(), "ab", stderr)) setvbuf(stderr, nullptr, _IONBF, 0);
}

// ---- 服务：拉起引擎 → 等端口 → 拉起 API → 等端口 → 看守 -----------------------
struct Plan {
    std::vector<std::string> engine_args, api_args;
    std::string engine_log, api_log;
    std::string engine_host;
    std::string api_host;
    int engine_port = 0, api_port = 0;
    int start_timeout = 0;
};
Plan g_plan;

enum SvcState : LONG { kLoading = 0, kApiStarting = 1, kReady = 2 };
volatile LONG g_state = kLoading;
volatile LONG g_quitting = 0;  // 用户从托盘选了"退出"
HWND g_hwnd = nullptr;
const UINT WM_TRAY = WM_APP + 1;       // 托盘图标鼠标事件
const UINT WM_SVC_STATE = WM_APP + 2;  // 服务线程：状态变化
const UINT WM_SVC_EXIT = WM_APP + 3;   // 服务线程：子进程退出（wParam 谁，lParam 退出码）

void report(SvcState st) {
    InterlockedExchange(&g_state, st);
    if (g_hwnd) PostMessageW(g_hwnd, WM_SVC_STATE, st, 0);
}

// 本机访问地址：监听 0.0.0.0 时用 127.0.0.1
std::string local_base() {
    const std::string& h = g_plan.api_host;
    const std::string host = h.empty() || h == "0.0.0.0" || h == "::" ? "127.0.0.1" : h;
    return "http://" + host + ":" + std::to_string(g_plan.api_port);
}

// 返回先退出的子进程（0 引擎 / 1 API），*code 为其退出码；用户主动退出时返回 -1
int run_service(DWORD* code) {
    const Plan& p = g_plan;
    Child engine = spawn("engine", "build\\gdec-win.exe", p.engine_args, p.engine_log);
    g_children[0] = engine.proc;

    ULONGLONG deadline = GetTickCount64() + static_cast<ULONGLONG>(p.start_timeout) * 1000;
    while (!port_listening(p.engine_port, p.engine_host)) {
        if (!alive(engine.proc))
            fail("引擎提前退出，查看 " + p.engine_log);
        if (GetTickCount64() > deadline)
            fail("引擎启动超过 " + std::to_string(p.start_timeout) + " 秒，查看 " + p.engine_log);
        Sleep(1000);
    }

    report(kApiStarting);
    Child api = spawn("api", "build\\gdec-api-win.exe", p.api_args, p.api_log);
    g_children[1] = api.proc;

    deadline = GetTickCount64() + 30000;
    while (!port_listening(p.api_port)) {
        if (!alive(engine.proc)) fail("引擎已退出，查看 " + p.engine_log);
        if (!alive(api.proc)) fail("API 提前退出，查看 " + p.api_log);
        if (GetTickCount64() > deadline) fail("API 启动超时，查看 " + p.api_log);
        Sleep(500);
    }

    printf("服务已就绪：http://%s:%d/v1（0.0.0.0 表示监听所有网卡）\n", p.api_host.c_str(),
           p.api_port);
    printf("日志：%s %s%s\n", p.engine_log.c_str(), p.api_log.c_str(),
           g_tray ? "" : "；Ctrl+C 同时停止 API 和引擎。");
    fflush(stdout);
    report(kReady);

    HANDLE both[2] = {engine.proc, api.proc};
    const DWORD who = WaitForMultipleObjects(2, both, FALSE, INFINITE) - WAIT_OBJECT_0;
    if (g_quitting) return -1;
    const char* which = who == 0 ? "引擎" : "API";
    *code = 1;
    GetExitCodeProcess(both[who == 1 ? 1 : 0], code);
    console_drain(3000);  // 退出进程最后的输出（如错误原因）先上屏
    fprintf(stderr, "%s进程退出（%lu），正在停止服务。\n", which, *code);
    kill_child(engine);
    kill_child(api);
    return who == 1 ? 1 : 0;
}

// ---- 托盘 --------------------------------------------------------------------
NOTIFYICONDATAW g_nid{};
volatile LONG g_tray_added = 0;
UINT g_wm_taskbar_created = 0;  // 资源管理器重启后广播，需重新加图标

enum MenuId : UINT {
    kIdDashboard = 1, kIdCopyUrl, kIdEngineLog, kIdApiLog, kIdLogDir, kIdQuit
};

std::wstring status_text() {
    switch (g_state) {
        case kReady: return TR(L"已就绪：", L"Ready: ") + to_w(local_base()) + L"/v1";
        case kApiStarting: return TR(L"模型已加载，正在启动 API…", L"Model loaded, starting API…");
        default: return TR(L"正在加载模型…", L"Loading model…");
    }
}

void tray_update_tip() {
    // 就绪时 tooltip 只留状态本身（地址在右键菜单首项和气球里都有）
    const std::wstring tip = g_state == kReady
        ? TR(L"引擎就绪", L"Engine ready")
        : L"gfx1151-engine · " + status_text();
    g_nid.uFlags = NIF_TIP;
    lstrcpynW(g_nid.szTip, tip.c_str(), ARRAYSIZE(g_nid.szTip));
    if (g_tray_added) Shell_NotifyIconW(NIM_MODIFY, &g_nid);
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

void tray_add() {
    g_nid.cbSize = sizeof(g_nid);
    g_nid.hWnd = g_hwnd;
    g_nid.uID = 1;
    g_nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
    g_nid.uCallbackMessage = WM_TRAY;
    static HICON small_icon = app_icon(GetSystemMetrics(SM_CXSMICON));
    static HICON big_icon = app_icon(GetSystemMetrics(SM_CXICON));  // 通知气泡用
    g_nid.hIcon = small_icon ? small_icon
                             : LoadIconA(nullptr, IDI_APPLICATION);  // 未定义 UNICODE：IDI_* 是窄串
    g_nid.hBalloonIcon = big_icon;
    const std::wstring tip = L"gfx1151-engine · " + status_text();
    lstrcpynW(g_nid.szTip, tip.c_str(), ARRAYSIZE(g_nid.szTip));
    // 开机自启时任务栏可能还没起来：失败就等 TaskbarCreated 再加
    if (Shell_NotifyIconW(NIM_ADD, &g_nid)) InterlockedExchange(&g_tray_added, 1);
}

void tray_remove() {
    if (InterlockedExchange(&g_tray_added, 0)) Shell_NotifyIconW(NIM_DELETE, &g_nid);
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

[[noreturn]] void quit_all(UINT code) {
    InterlockedExchange(&g_quitting, 1);
    kill_all_children();
    tray_remove();
    ExitProcess(code);
}

void on_menu(UINT id) {
    switch (id) {
        case kIdDashboard: shell_open(local_base() + "/"); break;
        case kIdCopyUrl:
            copy_text(to_w(local_base() + "/v1"));
            tray_balloon(TR(L"已复制", L"Copied"), to_w(local_base() + "/v1"));
            break;
        case kIdEngineLog: shell_open(g_root + "\\" + g_plan.engine_log); break;
        case kIdApiLog: shell_open(g_root + "\\" + g_plan.api_log); break;
        case kIdLogDir: shell_open(g_root + "\\logs"); break;
        case kIdQuit:
            if (msgbox(TR("退出会同时停止引擎和 API，正在进行的生成会中断。\n确定退出？",
                          "Quitting stops the engine and the API; any ongoing generation "
                          "will be interrupted.\nQuit now?"),
                       MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON2) == IDYES)
                quit_all(0);
            break;
        default: break;
    }
}

void show_menu() {
    const LONG st = g_state;
    HMENU m = CreatePopupMenu();
    if (!m) return;
    AppendMenuW(m, MF_STRING | MF_GRAYED, 0, status_text().c_str());
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING | (st == kReady ? 0 : MF_GRAYED), kIdDashboard,
                TR(L"打开控制台", L"Open console"));
    AppendMenuW(m, MF_STRING, kIdCopyUrl, TR(L"复制 API 地址", L"Copy API URL"));
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING, kIdEngineLog, TR(L"查看引擎日志", L"View engine log"));
    AppendMenuW(m, MF_STRING | (st >= kApiStarting ? 0 : MF_GRAYED), kIdApiLog,
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

LRESULT CALLBACK wnd_proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_TRAY) {
        switch (LOWORD(lp)) {
            case WM_RBUTTONUP:
            case WM_CONTEXTMENU: show_menu(); break;
            case WM_LBUTTONDBLCLK:
                on_menu(g_state == kReady ? kIdDashboard : kIdEngineLog);
                break;
            default: break;
        }
        return 0;
    }
    if (msg == WM_SVC_STATE) {
        tray_update_tip();
        if (wp == static_cast<WPARAM>(kReady))
            tray_balloon(TR(L"服务已就绪", L"Service ready"),
                         to_w(local_base() + "/v1") + L"\n" +
                             TR(L"双击图标打开控制台，右键查看更多",
                                L"Double-click the icon to open the console; "
                                L"right-click for more"));
        return 0;
    }
    if (msg == WM_SVC_EXIT) {
        tray_remove();
        const bool is_api = wp == 1;
        const std::string log = is_api ? g_plan.api_log : g_plan.engine_log;
        const std::string code = std::to_string(static_cast<DWORD>(lp));
        const std::string text =
            g_lang_en
                ? "The " + std::string(is_api ? "API" : "engine") +
                      " process exited (code " + code +
                      "); the service has stopped.\n\nLog: " + log +
                      "\nOpen the logs folder?"
                : std::string(is_api ? "API" : "引擎") + "进程已退出（退出码 " + code +
                      "），服务已停止。\n\n日志：" + log + "\n是否打开日志文件夹？";
        if (msgbox(text, MB_YESNO | MB_ICONWARNING) == IDYES) shell_open(g_root + "\\logs");
        ExitProcess(static_cast<UINT>(lp));
    }
    if (g_wm_taskbar_created && msg == g_wm_taskbar_created) {
        InterlockedExchange(&g_tray_added, 0);
        tray_add();
        return 0;
    }
    if (msg == WM_ENDSESSION && wp) quit_all(0);  // 注销 / 关机
    return DefWindowProcW(h, msg, wp, lp);
}

DWORD WINAPI service_thread(LPVOID) {
    DWORD code = 1;
    const int who = run_service(&code);
    if (who >= 0) PostMessageW(g_hwnd, WM_SVC_EXIT, static_cast<WPARAM>(who),
                               static_cast<LPARAM>(code));
    return 0;
}

int tray_main() {
    HINSTANCE hi = GetModuleHandleW(nullptr);
    WNDCLASSW wc{};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = hi;
    wc.lpszClassName = L"gfx1151_start_win";
    RegisterClassW(&wc);
    // 普通的隐藏顶层窗口（不 ShowWindow）：消息窗口收不到 TaskbarCreated 广播
    g_hwnd = CreateWindowExW(0, wc.lpszClassName, L"gfx1151-engine", WS_OVERLAPPED, 0, 0, 0, 0,
                             nullptr, nullptr, hi, nullptr);
    if (!g_hwnd) fail("无法创建托盘窗口，错误码 " + std::to_string(GetLastError()));
    g_wm_taskbar_created = RegisterWindowMessageW(L"TaskbarCreated");
    tray_add();
    tray_balloon(TR(L"正在加载模型", L"Loading model"),
                 TR(L"冷启动可能要几分钟。就绪后会再提示；右键托盘图标可查看日志或退出。",
                    L"A cold start can take a few minutes. You'll be notified when it's "
                    L"ready; right-click the tray icon to view logs or quit."));

    HANDLE t = CreateThread(nullptr, 0, service_thread, nullptr, 0, nullptr);
    if (!t) fail("无法创建服务线程");
    CloseHandle(t);

    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0) > 0) {
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    quit_all(0);
}

}  // namespace

// 启动配置面板（含 save_conf / panel_lang_resolve / panel_run）。
// 必须放在匿名命名空间之外：面板内部要 include <set> 等标准头。
#include "launch_panel.inc"

int main(int argc, char** argv) {
    bool check_only = false, want_console = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--check") == 0) check_only = true;
        else if (strcmp(argv[i], "--console") == 0) want_console = true;
        // --setup 已废弃（面板现在总是显示），忽略不报错
    }
    g_tray = !check_only && !want_console;
    // 高 DPI 屏上托盘图标取对应尺寸、弹框/菜单文字不发糊
    if (g_tray) SetProcessDPIAware();

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

    // 界面语言（托盘与面板共用）：service.conf lang 标记 > UI_LANG > 系统
    // UI 语言（非中文→英文）。标记读文件不依赖 load_conf，尽早解析。
    panel_lang_resolve();

    {
        SYSTEMTIME st;
        GetLocalTime(&st);
        char stamp[32];
        snprintf(stamp, sizeof(stamp), "%04d%02d%02d-%02d%02d%02d", st.wYear, st.wMonth,
                 st.wDay, st.wHour, st.wMinute, st.wSecond);
        g_stamp = stamp;
    }
    if (g_tray) {
        CreateDirectoryA("logs", nullptr);
        launcher_log_open();
    } else {
        console_open(check_only);
    }

    // 防多开（--check 不受限）。句柄不关，进程退出时系统回收。
    if (!check_only) {
        CreateMutexW(nullptr, FALSE, L"Local\\gfx1151-engine-start_win");
        if (GetLastError() == ERROR_ALREADY_EXISTS)
            fail("start_win.exe 已经在运行了（看任务栏右下角的托盘图标，"
                 "可能藏在 ^ 展开区里）。");
    }

    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    load_conf(g_root + "\\service.conf");

    // 启动配置面板：双击（托盘模式）启动时总是显示，点"启动服务"才继续。
    // 面板在"启动服务"时已把新值写进 g_conf 和 service.conf，下面的 cfg()
    // 会读到新配置。--check / --console 不显示面板，行为与之前完全一致。
    if (!check_only && !want_console) {
        // kPanelRelang：语言切换后重开面板（新语言在 panel_run 里重新解析）
        PanelResult pr;
        do pr = panel_run(true); while (pr == kPanelRelang);
        if (pr != kPanelStart) return 0;  // 取消：直接退出（仅保存不关面板）
    }

    const std::string model_dir = cfg("MODEL_DIR", "models");
    const std::string model_file = cfg("MODEL_FILE", model_dir + "\\heretic.hgn");
    // PLE n-gram 表所在文件：默认同 MODEL_FILE（w4b 单文件）；halogen v2 指向独立的 *-ngram.hgn。
    const std::string ngram_file = cfg("NGRAM_FILE", model_file);
    const std::string mtp_file =
        cfg_optional("MTP_FILE", model_dir + "\\heretic-mtp.hgn");
    const std::string vision_file =
        cfg_optional("VISION_FILE", model_dir + "\\heretic-vision.hgn");
    // 覆盖层（可选的高精度替换张量，叠加在主权重之上、MTP 之前）：默认空。
    std::string overlay_file = cfg_optional("OVERLAY_FILE", "");
    const std::string tokenizer_dir =
        cfg("TOKENIZER_DIR", model_dir + "\\tokenizer");
    const std::string engine_host = cfg("ENGINE_HOST", engine_net::kDefaultHost);
    const int engine_port = cfg_int("ENGINE_PORT", 8730, 1, 65535);
    const std::string api_host = cfg("API_HOST", "0.0.0.0");
    const int api_port = cfg_int("API_PORT", 8731, 1, 65535);
    const int max_context = cfg_int("MAX_CONTEXT", 262144, 1024, 1 << 20);
    const int mtp_gamma = cfg_int("MTP_GAMMA", 0, 0, 8);  // 0 = 引擎按模式自选
    const int kvsnap_max_gb = cfg_int("KVSNAP_MAX_GB", 20, 0, 1 << 16);
    const int rckpt_max = cfg_int("RCKPT_MAX", 8, 0, 1 << 16);
    // 分页 KV（见 service.conf）：默认开启，页池 = 一条 MAX_CONTEXT 序列。
    const int kv_paged = cfg_int("KV_PAGED", 1, 0, 1);
    const int kv_pool_tokens = cfg_int("KV_POOL_TOKENS", 0, 0, 1 << 24);
    // 并发请求数（见 service.conf）：几条序列共享同一个页池，需要分页 KV。
    const int parallel = cfg_int("PARALLEL", 1, 1, 8);
    // 单请求图片数上限（见 service.conf）：多轮视觉对话会累计历史图片。
    const int max_images = cfg_int("MAX_IMAGES", 8, 1, 256);
    // prefill 分段（见 service.conf）：决定 prefill 工作区在 arena 里的尺寸
    // （8192≈6.9 GiB，4096≈3.5 GiB）。0 = 引擎内置默认（Windows 8192）。
    const int prefill_chunk = cfg_int("PREFILL_CHUNK", 0, 0, 1 << 20);
    const int start_timeout = env_int("START_TIMEOUT", 1800, 30, 86400);
    // YaRN（见 service.conf）：默认关闭（ROPE_FACTOR=1 即原生 RoPE）。字符串原样
    // 转发给引擎/API（GDEC_ROPE_*），这里只做与 start_win.sh 相同的校验。
    const std::string rope_factor_s = cfg("ROPE_FACTOR", "1");
    const std::string rope_fast_s = cfg("ROPE_BETA_FAST", "32");
    const std::string rope_slow_s = cfg("ROPE_BETA_SLOW", "1");
    const std::string rope_attn_s = cfg("ROPE_ATTN_SCALE", "0");
    const int rope_original = cfg_int("ROPE_ORIGINAL_CTX", 262144, 1, 1 << 30);

    sockaddr_in engine_addr{};
    if (!engine_net::address(engine_host, engine_port, &engine_addr))
        fail("ENGINE_HOST 必须是 IPv4 地址");
    if (engine_port == api_port) fail("ENGINE_PORT 与 API_PORT 必须不同");
    if (parallel > 1 && !kv_paged) fail("PARALLEL>1 需要 KV_PAGED=1");
    // Windows 设备内存是硬上限 95 GiB 的 arena：权重 62.1 GiB + prefill 工作区
    // 6.9 GiB 之后，512K 池（45,696 B/token ≈ 22.3 GiB）刚好放下，更大的池任何
    // 配置都放不下。有效池 = max(KV_POOL_TOKENS, MAX_CONTEXT)（0 = 跟随上限）。
    const int kv_pool_effective =
        kv_pool_tokens > max_context ? kv_pool_tokens : max_context;
    if (kv_pool_effective > 524288)
        fail("KV 页池有效值 " + std::to_string(kv_pool_effective) +
             " token 超过 Windows 上限 512K（524288）：95 GiB 显存 arena 中 512K 池"
             "已占约 22 GiB，更大的池放不下；调小 MAX_CONTEXT 或 KV_POOL_TOKENS");
    const double rope_factor = parse_double("ROPE_FACTOR", rope_factor_s);
    const double rope_beta_fast = parse_double("ROPE_BETA_FAST", rope_fast_s);
    const double rope_beta_slow = parse_double("ROPE_BETA_SLOW", rope_slow_s);
    parse_double("ROPE_ATTN_SCALE", rope_attn_s);  // 只校验非负小数，值由引擎使用
    if (rope_factor < 1 || rope_beta_fast < rope_beta_slow || rope_beta_slow <= 0)
        fail("ROPE_FACTOR 必须 >=1，beta_fast >= beta_slow > 0");
    // 与引擎 52_main.inc 一致：YaRN 开启时单条上限不得超过位置编码范围；
    // 并发容量用 KV_POOL_TOKENS 扩，不要拉伸 MAX_CONTEXT。
    if (rope_factor > 1 && max_context > rope_factor * rope_original)
        fail("MAX_CONTEXT=" + std::to_string(max_context) +
             " 超出 YaRN 上限 ROPE_FACTOR×ROPE_ORIGINAL_CTX；调小 MAX_CONTEXT 或调大"
             " ROPE_FACTOR（并发容量用 KV_POOL_TOKENS 扩）");
    if (rope_factor <= 1 && max_context > rope_original)
        fprintf(stderr, "警告：ROPE_FACTOR=1 且 MAX_CONTEXT=%d 超过原生 %d，超出部分"
                        "的位置编码未验证（需要时请设 ROPE_FACTOR）\n",
                max_context, rope_original);
    if (!file_exists("build\\gdec-win.exe")) fail("缺少 build\\gdec-win.exe");
    if (!file_exists("build\\gdec-api-win.exe")) fail("缺少 build\\gdec-api-win.exe");
    if (!file_exists(model_file)) fail("找不到模型：" + model_file + "（修改 service.conf）");
    if (!file_exists(ngram_file))
        fail("找不到 n-gram 表：" + ngram_file + "（w4b 与 MODEL_FILE 相同，修改 service.conf）");
    if (!mtp_file.empty() && !file_exists(mtp_file)) fail("找不到 MTP 权重：" + mtp_file);
    if (!vision_file.empty() && !file_exists(vision_file))
        fail("找不到视觉塔：" + vision_file + "（纯文本可 set VISION_FILE= 后启动）");
    // overlay 可选：文件不在只警告并跳过（例如换成 v2 权重后忘了改配置）。
    if (!overlay_file.empty() && !file_exists(overlay_file)) {
        fprintf(stderr, "警告：找不到 overlay：%s，已跳过（v2 权重不用 overlay；"
                        "不想看到这条可在 service.conf 设 OVERLAY_FILE=\"\"）\n",
                overlay_file.c_str());
        overlay_file.clear();
    }
    if (!file_exists(tokenizer_dir + "\\tokenizer.json"))
        fail("找不到 tokenizer：" + tokenizer_dir);
    if (!dir_exists("build\\rocblas\\library") || !dir_exists("build\\hipblaslt\\library"))
        fail("缺少 rocBLAS/hipBLASLt kernel db（build\\*\\library）");
    if (!port_free(engine_port)) fail("端口 " + std::to_string(engine_port) + " 已被占用");
    if (!port_free(api_port)) fail("端口 " + std::to_string(api_port) + " 已被占用");

    printf("项目：%s\n", g_root.c_str());
    printf("模型：%s\n", model_file.c_str());
    const std::string gamma_str =
        mtp_gamma ? std::to_string(mtp_gamma) : "auto（greedy 4 / 采样自适应）";
    printf("配置：%d 上下文，prefill chunk %d，MTP gamma=%s，engine %s:%d，API %s:%d\n",
           max_context, prefill_chunk ? prefill_chunk : 8192, gamma_str.c_str(),
           engine_host.c_str(), engine_port, api_host.c_str(), api_port);
    if (rope_factor > 1)
        printf("YaRN：factor=%s，original_ctx=%d\n",
               rope_factor_s.c_str(), rope_original);
    if (kv_paged) {
        printf("KV：分页，页池 %d token（%d 路并发共享），RAM 检查点 %d 个\n",
               kv_pool_tokens > max_context ? kv_pool_tokens : max_context, parallel,
               rckpt_max);
        if (parallel > 1)
            fprintf(stderr, "提示：每多一路并发约多占 0.12 GiB 设备内存，arena"
                            "（95 GiB 上限）放不下的部分会回退 hipMalloc\n");
        if (kv_pool_tokens > max_context)
            fprintf(stderr, "警告：KV_POOL_TOKENS 大于 MAX_CONTEXT，Windows arena"
                            "（95 GiB 上限）可能放不下，超出部分会回退 hipMalloc\n");
    } else {
        printf("KV：不分页（KV_PAGED=0）\n");
    }
    if (check_only) {
        printf("检查通过；没有启动引擎或 API。\n");
        fflush(stdout);
        pause_if_own_console();
        return 0;
    }

    // 生产选项与 Linux start.sh 一致。GDEC_PREFILL_CHUNK 只在 service.conf 的
    // PREFILL_CHUNK>0 时设置（0 = 不碰外部环境，引擎用内置默认：Windows 8192）。
    // 实测（128K，PORTING-WINDOWS.md）：4096 比 8192 省 ~3.6 GiB arena、稳态 PP
    // 仅慢 ~2%、首块冷启动更快；256K 下 16384 会顶破 95 GiB arena 上限；
    // maxctx ≤ 40K 时可设 16384 换 ~6% PP。
    const char* flags[] = {
        "GDEC_QSA_KV_BF16", "GDEC_QSA_WMMA", "GDEC_QSA_WMMA_BTV",
        "GDEC_MOE_LT", "GDEC_MOE_LT_BF16", "GDEC_GR_BF16",
        "GDEC_GDN_STREAM", "GDEC_GDN_WAVE", "GDEC_NOWARMUP",
        "GDEC_GEMM_WMMA", "GDEC_GDN_FUSED",
        "GDEC_INDEX_FUSED2", "GDEC_PP_MOE_OUT", "GDEC_INDEX_STREAM_SELECT",
    };
    for (const char* f : flags) SetEnvironmentVariableA(f, "1");
    SetEnvironmentVariableA("GDEC_KVSNAP", kvsnap_max_gb ? "1" : "0");
    SetEnvironmentVariableA("GDEC_KVSNAP_MAX_GB",
                            std::to_string(kvsnap_max_gb).c_str());
    SetEnvironmentVariableA("GDEC_RCKPT_MAX", std::to_string(rckpt_max).c_str());
    // MTP_GAMMA=0：不设（并删掉外部残留），引擎按请求模式自选（greedy 4 / 采样自适应）。
    SetEnvironmentVariableA("GDEC_SPEC_GAMMA",
                            mtp_gamma ? std::to_string(mtp_gamma).c_str() : nullptr);
    // 只在开启时设置；关闭时删掉外部环境的残留值（引擎子进程继承本进程环境）。
    SetEnvironmentVariableA("GDEC_KV_PAGED", kv_paged ? "1" : nullptr);
    SetEnvironmentVariableA("GDEC_KV_POOL_TOKENS",
                            kv_paged && kv_pool_tokens
                                ? std::to_string(kv_pool_tokens).c_str()
                                : nullptr);
    SetEnvironmentVariableA("GDEC_PARALLEL", std::to_string(parallel).c_str());
    SetEnvironmentVariableA("GDEC_API_MAX_IMAGES", std::to_string(max_images).c_str());
    SetEnvironmentVariableA("GDEC_ROPE_FACTOR", rope_factor_s.c_str());
    SetEnvironmentVariableA("GDEC_ROPE_ORIGINAL_CTX",
                            std::to_string(rope_original).c_str());
    SetEnvironmentVariableA("GDEC_ROPE_BETA_FAST", rope_fast_s.c_str());
    SetEnvironmentVariableA("GDEC_ROPE_BETA_SLOW", rope_slow_s.c_str());
    SetEnvironmentVariableA("GDEC_ROPE_ATTN_SCALE", rope_attn_s.c_str());
    // PREFILL_CHUNK>0 才设置；0 = 不碰外部环境变量（用户可直接 set GDEC_PREFILL_CHUNK）。
    if (prefill_chunk)
        SetEnvironmentVariableA("GDEC_PREFILL_CHUNK",
                                std::to_string(prefill_chunk).c_str());

    CreateDirectoryA("logs", nullptr);
    g_plan.engine_log = "logs\\engine-win-" + g_stamp + ".log";
    g_plan.api_log = "logs\\api-win-" + g_stamp + ".log";
    g_plan.engine_port = engine_port;
    g_plan.engine_host = engine_host;
    g_plan.api_port = api_port;
    g_plan.api_host = api_host;
    g_plan.start_timeout = start_timeout;

    std::vector<std::string>& engine_args = g_plan.engine_args;
    engine_args = {model_file};
    if (!overlay_file.empty()) engine_args.push_back(overlay_file);
    if (!same_file(ngram_file, model_file)) engine_args.push_back(ngram_file);
    if (!mtp_file.empty()) engine_args.push_back(mtp_file);
    engine_args.insert(engine_args.end(),
                       {"--serve", "--host", engine_host, "--port", std::to_string(engine_port),
                        "--maxctx", std::to_string(max_context)});
    if (!vision_file.empty()) engine_args.insert(engine_args.end(),
                                                 {"--vision-tower", vision_file});
    g_plan.api_args = {"--tokenizer", tokenizer_dir, "--engine",
                       engine_net::connect_host(engine_host) + ":" + std::to_string(engine_port), "--host", api_host,
                       "--port", std::to_string(api_port), "--context",
                       std::to_string(max_context)};

    // 子进程进 Job：本进程一退出（含崩溃、被任务管理器结束），Job 句柄关闭，
    // 引擎和 API 随之结束。
    g_job = CreateJobObjectA(nullptr, nullptr);
    if (g_job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(g_job, JobObjectExtendedLimitInformation, &li,
                                     sizeof(li))) {
            CloseHandle(g_job);
            g_job = nullptr;
        }
    }

    printf("加载模型中，日志：%s%s\n", g_plan.engine_log.c_str(),
           g_tray ? "" : "；Ctrl+C 停止。");
    fflush(stdout);
    if (g_tray) return tray_main();

    console_quickedit_off();
    {
        HANDLE t = CreateThread(nullptr, 0, console_thread, nullptr, 0, nullptr);
        if (!t) fail("无法创建控制台输出线程");
        CloseHandle(t);
        g_console_on = true;
    }
    DWORD code = 1;
    run_service(&code);
    pause_if_own_console();
    return static_cast<int>(code);
}

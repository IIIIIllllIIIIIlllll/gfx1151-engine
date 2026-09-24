// start_win.exe - Windows 启动器：拉起引擎 + OpenAI API 双进程。
// 原生 Win32，无任何脚本宿主依赖（不需要 Git Bash / PowerShell）。
// 双击即运行；Ctrl+C 或关窗同时停掉两个子进程。
//
// 配置来源（优先级从高到低）：环境变量 > 根目录 service.conf > 内置默认。
// 换模型文件名、改上下文窗口等，直接编辑 service.conf（与 Linux 同一文件）。
//   set MAX_CONTEXT=131072 && start_win.exe   （环境变量临时覆盖）
//   VISION_FILE 置空 = 纯文本不挂视觉塔；MTP_FILE 置空 = 不用投机草稿；
//   OVERLAY_FILE 置空（默认）= 无覆盖层
//   start_win.exe --check     只检查配置不启动
// START_TIMEOUT 等计时项与平台相关，不读 conf（Windows 冷加载分钟级）。
//
// 子进程输出实时透传到本控制台，同时写入 logs\*-win-<时间戳>.log。
#include <windows.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

std::string g_root;
HANDLE g_children[2] = {nullptr, nullptr};
bool g_own_console = false;

void pause_if_own_console() {
    if (g_own_console) system("pause");
}

[[noreturn]] void fail(const std::string& msg) {
    fprintf(stderr, "错误：%s\n", msg.c_str());
    fflush(stderr);
    for (HANDLE h : g_children)
        if (h) TerminateProcess(h, 1);
    pause_if_own_console();
    ExitProcess(1);
}

bool file_exists(const std::string& path) {
    DWORD a = GetFileAttributesA(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
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
    if (!end || *end || n < lo || n > hi) {
        fprintf(stderr, "错误：%s 必须是 %d-%d 的整数（当前为 \"%s\"）\n", name, lo, hi, v);
        ExitProcess(1);
    }
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
    size_t a = s.find_first_not_of(" \t");
    size_t b = s.find_last_not_of(" \t");
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

// 能连上 127.0.0.1:port 说明对端已在 LISTEN。
bool port_listening(int port) {
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<u_short>(port));
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

DWORD WINAPI tee_thread(LPVOID param) {
    Child* c = static_cast<Child*>(param);
    HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
    char buf[4096];
    DWORD n = 0;
    while (ReadFile(c->pipe_read, buf, sizeof(buf), &n, nullptr) && n > 0) {
        DWORD written = 0;
        WriteFile(console, buf, n, &written, nullptr);
        if (c->log) {
            fwrite(buf, 1, n, c->log);
            fflush(c->log);
        }
    }
    return 0;
}

Child spawn(const std::string& name, const std::string& exe,
            const std::vector<std::string>& args, const std::string& log_path) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE pipe_read = nullptr, pipe_write = nullptr;
    if (!CreatePipe(&pipe_read, &pipe_write, &sa, 0))
        fail("CreatePipe 失败");
    SetHandleInformation(pipe_read, HANDLE_FLAG_INHERIT, 0);

    std::string cmd = quote(exe);
    for (const std::string& a : args) cmd += " " + quote(a);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = pipe_write;
    si.hStdError = pipe_write;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::vector<char> cmdline(cmd.begin(), cmd.end());
    cmdline.push_back('\0');
    if (!CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, TRUE, 0,
                        nullptr, g_root.c_str(), &si, &pi)) {
        CloseHandle(pipe_read);
        CloseHandle(pipe_write);
        fail("启动失败（" + exe + "），错误码 " + std::to_string(GetLastError()));
    }
    CloseHandle(pipe_write);
    CloseHandle(pi.hThread);

    Child c;
    c.proc = pi.hProcess;
    c.pipe_read = pipe_read;
    c.name = name;
    c.log = fopen(log_path.c_str(), "wb");
    if (!c.log) fprintf(stderr, "警告：无法写日志 %s\n", log_path.c_str());
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

BOOL WINAPI on_ctrl(DWORD ev) {
    if (ev == CTRL_C_EVENT || ev == CTRL_BREAK_EVENT || ev == CTRL_CLOSE_EVENT) {
        for (HANDLE h : g_children)
            if (h) TerminateProcess(h, 1);
        ExitProcess(1);
    }
    return FALSE;
}

}  // namespace

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCtrlHandler(on_ctrl, TRUE);
    {
        DWORD procs[4];
        g_own_console = GetConsoleProcessList(procs, 4) == 1;
    }
    char exe_path[MAX_PATH];
    GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    g_root = exe_path;
    size_t slash = g_root.find_last_of("\\/");
    g_root = slash == std::string::npos ? "." : g_root.substr(0, slash);
    SetCurrentDirectoryA(g_root.c_str());

    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    load_conf(g_root + "\\service.conf");

    const std::string model_dir = cfg("MODEL_DIR", "models");
    const std::string model_file = cfg("MODEL_FILE", model_dir + "\\heretic.hgn");
    const std::string mtp_file =
        cfg_optional("MTP_FILE", model_dir + "\\heretic-mtp.hgn");
    const std::string vision_file =
        cfg_optional("VISION_FILE", model_dir + "\\heretic-vision.hgn");
    // 覆盖层（可选的高精度替换张量，叠加在主权重之上、MTP 之前）：默认空。
    const std::string overlay_file = cfg_optional("OVERLAY_FILE", "");
    const std::string tokenizer_dir =
        cfg("TOKENIZER_DIR", model_dir + "\\tokenizer");
    const int engine_port = cfg_int("ENGINE_PORT", 8730, 1, 65535);
    const std::string api_host = cfg("API_HOST", "0.0.0.0");
    const int api_port = cfg_int("API_PORT", 8731, 1, 65535);
    const int max_context = cfg_int("MAX_CONTEXT", 262144, 1024, 1 << 20);
    const int mtp_gamma = cfg_int("MTP_GAMMA", 3, 1, 8);
    const int kvsnap_max_gb = cfg_int("KVSNAP_MAX_GB", 20, 0, 1 << 16);
    const int rckpt_max = cfg_int("RCKPT_MAX", 8, 0, 1 << 16);
    // 分页 KV（见 service.conf）：默认开启，页池 = 一条 MAX_CONTEXT 序列。
    const int kv_paged = cfg_int("KV_PAGED", 1, 0, 1);
    const int kv_pool_tokens = cfg_int("KV_POOL_TOKENS", 0, 0, 1 << 24);
    // 并发请求数（见 service.conf）：几条序列共享同一个页池，需要分页 KV。
    const int parallel = cfg_int("PARALLEL", 1, 1, 8);
    const int start_timeout = env_int("START_TIMEOUT", 1800, 30, 86400);

    if (engine_port == api_port) fail("ENGINE_PORT 与 API_PORT 必须不同");
    if (parallel > 1 && !kv_paged) fail("PARALLEL>1 需要 KV_PAGED=1");
    if (!file_exists("build\\gdec-win.exe")) fail("缺少 build\\gdec-win.exe");
    if (!file_exists("build\\gdec-api-win.exe")) fail("缺少 build\\gdec-api-win.exe");
    if (!file_exists(model_file)) fail("找不到模型：" + model_file + "（修改 service.conf）");
    if (!mtp_file.empty() && !file_exists(mtp_file)) fail("找不到 MTP 权重：" + mtp_file);
    if (!vision_file.empty() && !file_exists(vision_file))
        fail("找不到视觉塔：" + vision_file + "（纯文本可 set VISION_FILE= 后启动）");
    if (!overlay_file.empty() && !file_exists(overlay_file))
        fail("找不到 overlay：" + overlay_file +
             "（无 overlay 可在 service.conf 设 OVERLAY_FILE=\"\"）");
    if (!file_exists(tokenizer_dir + "\\tokenizer.json"))
        fail("找不到 tokenizer：" + tokenizer_dir);
    if (!dir_exists("build\\rocblas\\library") || !dir_exists("build\\hipblaslt\\library"))
        fail("缺少 rocBLAS/hipBLASLt kernel db（build\\*\\library）");
    if (!port_free(engine_port)) fail("端口 " + std::to_string(engine_port) + " 已被占用");
    if (!port_free(api_port)) fail("端口 " + std::to_string(api_port) + " 已被占用");

    printf("项目：%s\n", g_root.c_str());
    printf("模型：%s\n", model_file.c_str());
    printf("配置：%d 上下文，MTP gamma=%d，API %s:%d\n", max_context, mtp_gamma,
           api_host.c_str(), api_port);
    if (kv_paged) {
        printf("KV：分页，页池 %d token（%d 路并发共享），RAM 检查点 %d 个\n",
               kv_pool_tokens > max_context ? kv_pool_tokens : max_context, parallel,
               rckpt_max);
        if (parallel > 1)
            fprintf(stderr, "提示：每多一路并发约多占 1.1 GiB 设备内存（256K 下），arena"
                            "（95 GiB 上限）放不下的部分会回退 hipMalloc\n");
        if (kv_pool_tokens > max_context)
            fprintf(stderr, "警告：KV_POOL_TOKENS 大于 MAX_CONTEXT，Windows arena"
                            "（95 GiB 上限）可能放不下，超出部分会回退 hipMalloc\n");
    } else {
        printf("KV：不分页（KV_PAGED=0）\n");
    }
    if (argc > 1 && strcmp(argv[1], "--check") == 0) {
        printf("检查通过；没有启动引擎或 API。\n");
        return 0;
    }

    // 生产选项与 Linux start.sh 一致，唯独不设 GDEC_PREFILL_CHUNK：
    // Windows 默认 8192（256K 下 16384 会顶破 95 GiB arena 上限，
    // 实测见 PORTING-WINDOWS.md；maxctx ≤ 40K 时可手动设 16384 换 ~6% PP）。
    const char* flags[] = {
        "GDEC_QSA_KV_BF16", "GDEC_QSA_WMMA", "GDEC_QSA_WMMA_BTV",
        "GDEC_MOE_LT", "GDEC_MOE_LT_BF16", "GDEC_GR_BF16",
        "GDEC_GDN_STREAM", "GDEC_GDN_WAVE", "GDEC_NOWARMUP",
        "GDEC_INDEX_FUSED2", "GDEC_PP_MOE_OUT", "GDEC_INDEX_STREAM_SELECT",
    };
    for (const char* f : flags) SetEnvironmentVariableA(f, "1");
    SetEnvironmentVariableA("GDEC_KVSNAP", kvsnap_max_gb ? "1" : "0");
    SetEnvironmentVariableA("GDEC_KVSNAP_MAX_GB",
                            std::to_string(kvsnap_max_gb).c_str());
    SetEnvironmentVariableA("GDEC_RCKPT_MAX", std::to_string(rckpt_max).c_str());
    SetEnvironmentVariableA("GDEC_SPEC_GAMMA", std::to_string(mtp_gamma).c_str());
    // 只在开启时设置；关闭时删掉外部环境的残留值（引擎子进程继承本进程环境）。
    SetEnvironmentVariableA("GDEC_KV_PAGED", kv_paged ? "1" : nullptr);
    SetEnvironmentVariableA("GDEC_KV_POOL_TOKENS",
                            kv_paged && kv_pool_tokens
                                ? std::to_string(kv_pool_tokens).c_str()
                                : nullptr);
    SetEnvironmentVariableA("GDEC_PARALLEL", std::to_string(parallel).c_str());

    CreateDirectoryA("logs", nullptr);
    SYSTEMTIME st;
    GetLocalTime(&st);
    char stamp[32];
    snprintf(stamp, sizeof(stamp), "%04d%02d%02d-%02d%02d%02d", st.wYear, st.wMonth,
             st.wDay, st.wHour, st.wMinute, st.wSecond);
    const std::string engine_log =
        "logs\\engine-win-" + std::string(stamp) + ".log";
    const std::string api_log = "logs\\api-win-" + std::string(stamp) + ".log";

    std::vector<std::string> engine_args = {model_file};
    if (!overlay_file.empty()) engine_args.push_back(overlay_file);
    if (!mtp_file.empty()) engine_args.push_back(mtp_file);
    engine_args.insert(engine_args.end(),
                       {"--serve", "--port", std::to_string(engine_port),
                        "--maxctx", std::to_string(max_context)});
    if (!vision_file.empty()) engine_args.insert(engine_args.end(),
                                                 {"--vision-tower", vision_file});

    printf("加载模型中，日志：%s；Ctrl+C 停止。\n", engine_log.c_str());
    Child engine = spawn("engine", "build\\gdec-win.exe", engine_args, engine_log);
    g_children[0] = engine.proc;

    ULONGLONG deadline = GetTickCount64() + static_cast<ULONGLONG>(start_timeout) * 1000;
    while (!port_listening(engine_port)) {
        if (!alive(engine.proc))
            fail("引擎提前退出，查看 " + engine_log);
        if (GetTickCount64() > deadline)
            fail("引擎启动超过 " + std::to_string(start_timeout) + " 秒，查看 " + engine_log);
        Sleep(1000);
    }

    Child api = spawn("api", "build\\gdec-api-win.exe",
                      {"--tokenizer", tokenizer_dir, "--engine",
                       "127.0.0.1:" + std::to_string(engine_port), "--host", api_host,
                       "--port", std::to_string(api_port), "--context",
                       std::to_string(max_context)},
                      api_log);
    g_children[1] = api.proc;

    deadline = GetTickCount64() + 30000;
    while (!port_listening(api_port)) {
        if (!alive(engine.proc)) fail("引擎已退出");
        if (!alive(api.proc)) fail("API 提前退出，查看 " + api_log);
        if (GetTickCount64() > deadline) fail("API 启动超时，查看 " + api_log);
        Sleep(500);
    }

    printf("服务已就绪：http://%s:%d/v1（0.0.0.0 表示监听所有网卡）\n", api_host.c_str(),
           api_port);
    printf("日志：%s %s；Ctrl+C 同时停止 API 和引擎。\n", engine_log.c_str(),
           api_log.c_str());
    fflush(stdout);

    HANDLE both[2] = {engine.proc, api.proc};
    DWORD who = WaitForMultipleObjects(2, both, FALSE, INFINITE);
    const char* which = who == WAIT_OBJECT_0 ? "引擎" : "API";
    DWORD code = 1;
    GetExitCodeProcess(both[who - WAIT_OBJECT_0], &code);
    fprintf(stderr, "%s进程退出（%lu），正在停止服务。\n", which, code);
    kill_child(engine);
    kill_child(api);
    pause_if_own_console();
    return static_cast<int>(code);
}

// engine_sup.cpp — 见 engine_sup.h。启停语义与 start_win.exe 一致：
// 停止 = 直接 TerminateProcess（与 Linux 脚本 kill 进程组等价），引擎不
// 需要控制协议动词。子进程进 KILL_ON_JOB_CLOSE 的 Job：API 进程无论怎样
// 退出，引擎都会随之结束，不留占着显存的孤儿进程。
#include "engine_sup.h"

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "../engine_net.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace enginesup {
namespace {

std::mutex g_mtx;
Status g_st;
bool g_stopping = false;  // stop() 主动停止：监管线程把退出记为 stopped 而非 exited

// 能连上 host:port 说明对端已在 LISTEN（跨平台探测，Linux 也用于 external 判定）。
// 带 500ms 超时：阻塞 connect 对黑洞地址（防火墙丢包的远程 IP）会挂 ~20s，
// 而 status() 被托盘 UI 线程每 2s 周期调用，不能忍受这种停顿。
bool port_listening(const std::string& host, int port) {
#ifdef _WIN32
    const SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return false;
#else
    const int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return false;
#endif
    sockaddr_in addr{};
    if (!engine_net::address(engine_net::connect_host(host), port, &addr)) {
#ifdef _WIN32
        closesocket(s);
#else
        close(s);
#endif
        return false;
    }
#ifdef _WIN32
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
    bool ok = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0;
    if (!ok && WSAGetLastError() == WSAEWOULDBLOCK) {
#else
    const int fl = fcntl(s, F_GETFL, 0);
    fcntl(s, F_SETFL, fl | O_NONBLOCK);
    bool ok = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0;
    if (!ok && errno == EINPROGRESS) {
#endif
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(s, &wfds);
        timeval tv{0, 500 * 1000};
        if (select(static_cast<int>(s) + 1, nullptr, &wfds, nullptr, &tv) > 0) {
            int soerr = 0;
            socklen_t sl = sizeof soerr;
            getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&soerr), &sl);
            ok = soerr == 0;
        }
    }
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
    return ok;
}

#ifdef _WIN32

HANDLE g_proc = nullptr;
HANDLE g_job = nullptr;
std::string g_host;  // 本次启动的引擎地址（监管线程探测用）
int g_port = 0;

bool file_exists(const std::string& path) {
    const DWORD a = GetFileAttributesA(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

bool dir_exists(const std::string& path) {
    const DWORD a = GetFileAttributesA(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
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

// 子进程 stdout/stderr → 日志文件（无控制台透传：API 独立运行时引擎输出只落盘）。
DWORD WINAPI tee_thread(LPVOID param) {
    HANDLE pipe_read = static_cast<HANDLE*>(param)[0];
    FILE* log = static_cast<FILE**>(param)[1];
    char buf[65536];
    DWORD n = 0;
    while (ReadFile(pipe_read, buf, sizeof(buf), &n, nullptr) && n > 0)
        if (log) {
            fwrite(buf, 1, n, log);
            fflush(log);
        }
    if (log) fclose(log);
    CloseHandle(pipe_read);
    delete[] static_cast<void**>(param);
    return 0;
}

// 监管线程：starting →（端口在听）→ running →（进程退出）→ exited/stopped；
// starting 超过 START_TIMEOUT 秒则杀掉并记 exited。
DWORD WINAPI monitor_thread(LPVOID) {
    long timeout = 1800;
    if (const char* e = getenv("START_TIMEOUT")) {
        char* end = nullptr;
        const long n = strtol(e, &end, 10);
        if (end && !*end && n >= 30 && n <= 86400) timeout = n;
    }
    const ULONGLONG deadline = GetTickCount64() + static_cast<ULONGLONG>(timeout) * 1000;
    for (;;) {
        if (WaitForSingleObject(g_proc, 0) == WAIT_OBJECT_0) break;  // 提前退出
        if (port_listening(g_host, g_port)) {
            std::lock_guard<std::mutex> lk(g_mtx);
            if (g_st.state == "starting") g_st.state = "running";
            break;
        }
        if (GetTickCount64() > deadline) {
            TerminateProcess(g_proc, 1);
            std::lock_guard<std::mutex> lk(g_mtx);
            g_st.detail = "start timeout (" + std::to_string(timeout) + "s)";
            break;
        }
        Sleep(1000);
    }
    WaitForSingleObject(g_proc, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(g_proc, &code);
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_st.exit_code = code;
        g_st.pid = 0;
        // 主动 stop() → stopped；提前退出 / 超时杀掉 / 运行中意外退出 → exited
        g_st.state = g_stopping ? "stopped" : "exited";
        g_stopping = false;
        CloseHandle(g_proc);
        g_proc = nullptr;
    }
    return 0;
}

#endif  // _WIN32

}  // namespace

bool supported() {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

Status status(const std::string& host, int port) {
    Status s;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        s = g_st;  // 只拷贝快照：端口探测在锁外做（connect 可能慢，不能拿着锁）
    }
    if (s.state == "stopped" || s.state == "exited") {
        if (port_listening(host, port)) {
            // 端口在听但不是本进程拉起的：外部引擎（start_win.exe / 脚本）
            s.state = "running";
            s.external = true;
            s.pid = 0;
        }
    }
    return s;
}

bool start(const std::string& root, const svcconf::Conf& conf, std::string* err) {
#ifdef _WIN32
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_st.state == "starting" || (g_st.state == "running" && !g_st.external)) {
        *err = "engine already running";
        return false;
    }
    // 与 launch_win.cpp 相同的取值与默认（env > conf > 内置）
    const std::string model_dir = svcconf::cfg(conf, "MODEL_DIR", "models");
    const std::string model_file = svcconf::cfg(conf, "MODEL_FILE", model_dir + "\\heretic.hgn");
    const std::string ngram_file = svcconf::cfg(conf, "NGRAM_FILE", model_file);
    const std::string mtp_file =
        svcconf::cfg_optional(conf, "MTP_FILE", model_dir + "\\heretic-mtp.hgn");
    const std::string vision_file =
        svcconf::cfg_optional(conf, "VISION_FILE", model_dir + "\\heretic-vision.hgn");
    std::string overlay_file = svcconf::cfg_optional(conf, "OVERLAY_FILE", "");
    const std::string engine_host = svcconf::cfg(conf, "ENGINE_HOST", engine_net::kDefaultHost);
    long engine_port = 8730, max_context = 262144, mtp_gamma = 0, kvsnap_max_gb = 20,
         rckpt_max = 8, kv_paged = 1, kv_pool_tokens = 0, parallel = 1, max_images = 8,
         prefill_chunk = 0, rope_original = 262144;
    svcconf::parse_int(svcconf::cfg(conf, "ENGINE_PORT", "8730"), &engine_port);
    svcconf::parse_int(svcconf::cfg(conf, "MAX_CONTEXT", "262144"), &max_context);
    svcconf::parse_int(svcconf::cfg(conf, "MTP_GAMMA", "0"), &mtp_gamma);
    svcconf::parse_int(svcconf::cfg(conf, "KVSNAP_MAX_GB", "20"), &kvsnap_max_gb);
    svcconf::parse_int(svcconf::cfg(conf, "RCKPT_MAX", "8"), &rckpt_max);
    svcconf::parse_int(svcconf::cfg(conf, "KV_PAGED", "1"), &kv_paged);
    svcconf::parse_int(svcconf::cfg(conf, "KV_POOL_TOKENS", "0"), &kv_pool_tokens);
    svcconf::parse_int(svcconf::cfg(conf, "PARALLEL", "1"), &parallel);
    svcconf::parse_int(svcconf::cfg(conf, "MAX_IMAGES", "8"), &max_images);
    svcconf::parse_int(svcconf::cfg(conf, "PREFILL_CHUNK", "0"), &prefill_chunk);
    svcconf::parse_int(svcconf::cfg(conf, "ROPE_ORIGINAL_CTX", "262144"), &rope_original);
    const std::string rope_factor_s = svcconf::cfg(conf, "ROPE_FACTOR", "1");
    const std::string rope_fast_s = svcconf::cfg(conf, "ROPE_BETA_FAST", "32");
    const std::string rope_slow_s = svcconf::cfg(conf, "ROPE_BETA_SLOW", "1");
    const std::string rope_attn_s = svcconf::cfg(conf, "ROPE_ATTN_SCALE", "0");

    // 启动前文件检查（同 launch_win.cpp 启动检查；overlay 缺失只警告并跳过）
    const std::string exe = root + "\\build\\qwenox-engine-win.exe";
    if (!file_exists(exe)) {
        *err = "missing " + exe;
        return false;
    }
    if (!file_exists(model_file)) {
        *err = "model not found: " + model_file;
        return false;
    }
    if (!file_exists(ngram_file)) {
        *err = "n-gram table not found: " + ngram_file;
        return false;
    }
    if (!mtp_file.empty() && !file_exists(mtp_file)) {
        *err = "MTP weights not found: " + mtp_file;
        return false;
    }
    if (!vision_file.empty() && !file_exists(vision_file)) {
        *err = "vision tower not found: " + vision_file;
        return false;
    }
    if (!overlay_file.empty() && !file_exists(overlay_file)) overlay_file.clear();
    if (!dir_exists(root + "\\build\\rocblas\\library") ||
        !dir_exists(root + "\\build\\hipblaslt\\library")) {
        *err = "missing rocBLAS/hipBLASLt kernel db (build\\*\\library)";
        return false;
    }

    // 生产选项与 launch_win.cpp / Linux start.sh 一致：设在自身进程，子进程继承
    const char* flags[] = {
        "QWENOX_QSA_KV_BF16", "QWENOX_QSA_WMMA", "QWENOX_QSA_WMMA_BTV",
        "QWENOX_MOE_LT", "QWENOX_MOE_LT_BF16", "QWENOX_GR_BF16",
        "QWENOX_GDN_STREAM", "QWENOX_GDN_WAVE", "QWENOX_NOWARMUP",
        "QWENOX_GEMM_WMMA", "QWENOX_GDN_FUSED",
        "QWENOX_INDEX_FUSED2", "QWENOX_PP_MOE_OUT", "QWENOX_INDEX_STREAM_SELECT",
    };
    for (const char* f : flags) SetEnvironmentVariableA(f, "1");
    SetEnvironmentVariableA("QWENOX_KVSNAP", kvsnap_max_gb ? "1" : "0");
    SetEnvironmentVariableA("QWENOX_KVSNAP_MAX_GB", std::to_string(kvsnap_max_gb).c_str());
    SetEnvironmentVariableA("QWENOX_RCKPT_MAX", std::to_string(rckpt_max).c_str());
    // 0 = 引擎按模式自选：不设（并删掉外部残留）
    SetEnvironmentVariableA("QWENOX_SPEC_GAMMA",
                            mtp_gamma ? std::to_string(mtp_gamma).c_str() : nullptr);
    SetEnvironmentVariableA("QWENOX_KV_PAGED", kv_paged ? "1" : nullptr);
    SetEnvironmentVariableA("QWENOX_KV_POOL_TOKENS",
                            kv_paged && kv_pool_tokens
                                ? std::to_string(kv_pool_tokens).c_str()
                                : nullptr);
    SetEnvironmentVariableA("QWENOX_PARALLEL", std::to_string(parallel).c_str());
    SetEnvironmentVariableA("QWENOX_API_MAX_IMAGES", std::to_string(max_images).c_str());
    SetEnvironmentVariableA("QWENOX_ROPE_FACTOR", rope_factor_s.c_str());
    SetEnvironmentVariableA("QWENOX_ROPE_ORIGINAL_CTX", std::to_string(rope_original).c_str());
    SetEnvironmentVariableA("QWENOX_ROPE_BETA_FAST", rope_fast_s.c_str());
    SetEnvironmentVariableA("QWENOX_ROPE_BETA_SLOW", rope_slow_s.c_str());
    SetEnvironmentVariableA("QWENOX_ROPE_ATTN_SCALE", rope_attn_s.c_str());
    if (prefill_chunk)
        SetEnvironmentVariableA("QWENOX_PREFILL_CHUNK", std::to_string(prefill_chunk).c_str());

    std::vector<std::string> args = {model_file};
    if (!overlay_file.empty()) args.push_back(overlay_file);
    if (!same_file(ngram_file, model_file)) args.push_back(ngram_file);
    if (!mtp_file.empty()) args.push_back(mtp_file);
    args.insert(args.end(),
                {"--serve", "--host", engine_host, "--port", std::to_string(engine_port),
                 "--maxctx", std::to_string(max_context)});
    if (!vision_file.empty())
        args.insert(args.end(), {"--vision-tower", vision_file});

    CreateDirectoryA((root + "\\logs").c_str(), nullptr);
    SYSTEMTIME st;
    GetLocalTime(&st);
    char stamp[32];
    snprintf(stamp, sizeof(stamp), "%04d%02d%02d-%02d%02d%02d", st.wYear, st.wMonth,
             st.wDay, st.wHour, st.wMinute, st.wSecond);
    const std::string log_path = root + "\\logs\\engine-api-" + stamp + ".log";

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE pipe_read = nullptr, pipe_write = nullptr;
    if (!CreatePipe(&pipe_read, &pipe_write, &sa, 1u << 20)) {
        *err = "CreatePipe failed";
        return false;
    }
    SetHandleInformation(pipe_read, HANDLE_FLAG_INHERIT, 0);
    HANDLE in = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                            OPEN_EXISTING, 0, nullptr);

    std::string cmd = "\"" + exe + "\"";
    for (const std::string& a : args) cmd += " \"" + a + "\"";
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = pipe_write;
    si.hStdError = pipe_write;
    si.hStdInput = in == INVALID_HANDLE_VALUE ? nullptr : in;
    PROCESS_INFORMATION pi{};
    std::vector<char> cmdline(cmd.begin(), cmd.end());
    cmdline.push_back('\0');
    const BOOL ok = CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW, nullptr, root.c_str(), &si, &pi);
    const DWORD create_err = GetLastError();
    if (in != INVALID_HANDLE_VALUE) CloseHandle(in);
    CloseHandle(pipe_write);
    if (!ok) {
        CloseHandle(pipe_read);
        *err = "CreateProcess failed (" + exe + "), error " + std::to_string(create_err);
        return false;
    }
    // API 进程退出（含被任务管理器结束）时引擎随之结束
    if (!g_job) {
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
    }
    if (g_job) AssignProcessToJobObject(g_job, pi.hProcess);  // 失败也不影响运行
    CloseHandle(pi.hThread);

    FILE* log = fopen(log_path.c_str(), "wb");
    if (!log)
        fprintf(stderr, "qwenox-api: warning: cannot write engine log %s\n", log_path.c_str());
    void** tee_args = new void*[2]{pipe_read, log};
    HANDLE t = CreateThread(nullptr, 0, tee_thread, tee_args, 0, nullptr);
    if (t) CloseHandle(t);

    g_proc = pi.hProcess;
    g_host = engine_host;
    g_port = static_cast<int>(engine_port);
    g_st = Status{};
    g_st.state = "starting";
    g_st.pid = static_cast<int>(pi.dwProcessId);
    g_st.log = log_path;
    g_stopping = false;
    HANDLE m = CreateThread(nullptr, 0, monitor_thread, nullptr, 0, nullptr);
    if (m) CloseHandle(m);
    fprintf(stderr, "qwenox-api: engine starting (pid %lu), log: %s\n",
            static_cast<unsigned long>(pi.dwProcessId), log_path.c_str());
    return true;
#else
    (void)root;
    (void)conf;
    *err = "engine start/stop is only supported on Windows";
    return false;
#endif
}

bool stop(std::string* err) {
#ifdef _WIN32
    std::lock_guard<std::mutex> lk(g_mtx);
    // 外部引擎（端口在听但非本进程拉起）由调用方先经 status() 判定并拒绝
    if (!g_proc) {
        *err = "engine not running";
        return false;
    }
    g_stopping = true;
    TerminateProcess(g_proc, 1);  // 锁内终止：避免与监管线程的 CloseHandle 竞态
    return true;                  // 状态由监管线程翻转为 stopped（前端轮询可见）
#else
    *err = "engine start/stop is only supported on Windows";
    return false;
#endif
}

}  // namespace enginesup

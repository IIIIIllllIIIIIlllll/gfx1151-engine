#include "power.h"

#include <mutex>

#ifdef _WIN32

#include <cstdlib>
#include <cstring>

#include <windows.h>

namespace {

// ADL PMLog 结构（ADL SDK adl_structures.h / adl_defines.h）：
// ADL2_New_QueryPMLogData_Get 返回全部传感器快照，按 ADL_PMLOG_SENSORS
// 枚举直接索引。
constexpr int ADL_PMLOG_MAX_SENSORS = 256;
constexpr int ADL_PMLOG_ASIC_POWER = 23;          // W，整封装
constexpr int ADL_PMLOG_GFX_POWER = 30;           // W，GPU 域
constexpr int ADL_PMLOG_CPU_POWER = 33;           // W，CPU 域
constexpr int ADL_PMLOG_SSPAIRED_ASICPOWER = 46;  // W，APU 整封装
constexpr int ADL_PMLOG_BOARD_POWER = 73;         // W，板卡

struct ADLSingleSensorData {
    int supported;
    int value;
};
struct ADLPMLogDataOutput {
    int size;
    ADLSingleSensorData sensors[ADL_PMLOG_MAX_SENSORS];
};  // 4 + 256*8 = 2052 字节

typedef void*(__stdcall* ADL_MALLOC_CB)(int);
typedef int(__stdcall* PFN_CREATE2)(ADL_MALLOC_CB, int, void**);
typedef int(__stdcall* PFN_PMLOG_QUERY)(void*, int, ADLPMLogDataOutput*);

void* __stdcall adl_malloc(int size) { return malloc(static_cast<size_t>(size)); }

// 进程级单例，刻意不析构（进程退出时系统回收，避免 at-exit 与在途查询竞争）。
HMODULE g_dll = nullptr;
void* g_ctx = nullptr;  // ADL_CONTEXT_HANDLE
PFN_PMLOG_QUERY g_query = nullptr;
int g_adapter = -1;     // 选定的适配器索引（第一个能读出功耗的）
int g_state = 0;        // 0 未初始化 / 1 可用 / -1 不可用
std::mutex g_mtx;

void adl_init() {
    g_dll = LoadLibraryA("atiadlxx.dll");
    if (!g_dll) g_dll = LoadLibraryA("atiadlxy.dll");
    if (!g_dll) {
        g_state = -1;
        return;
    }
    PFN_CREATE2 create2 =
        reinterpret_cast<PFN_CREATE2>(GetProcAddress(g_dll, "ADL2_Main_Control_Create"));
    g_query =
        reinterpret_cast<PFN_PMLOG_QUERY>(GetProcAddress(g_dll, "ADL2_New_QueryPMLogData_Get"));
    if (!create2 || !g_query || create2(adl_malloc, 1, &g_ctx) != 0 || !g_ctx) {
        g_state = -1;
        return;
    }
    // 探测适配器：取第一个 PMLog 查询成功且支持任一功耗传感器的。
    for (int i = 0; i < 8; ++i) {
        ADLPMLogDataOutput pm;
        memset(&pm, 0, sizeof(pm));
        pm.size = static_cast<int>(sizeof(pm));
        if (g_query(g_ctx, i, &pm) != 0) continue;
        if (pm.sensors[ADL_PMLOG_ASIC_POWER].supported ||
            pm.sensors[ADL_PMLOG_GFX_POWER].supported ||
            pm.sensors[ADL_PMLOG_SSPAIRED_ASICPOWER].supported) {
            g_adapter = i;
            g_state = 1;
            return;
        }
    }
    g_state = -1;
}

int sensor_value(const ADLPMLogDataOutput& pm, int sensor) {
    return pm.sensors[sensor].supported ? pm.sensors[sensor].value : -1;
}

}  // namespace

namespace power {

Reading read() {
    Reading out;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_state == 0) adl_init();
    if (g_state != 1) return out;
    ADLPMLogDataOutput pm;
    memset(&pm, 0, sizeof(pm));
    pm.size = static_cast<int>(sizeof(pm));
    if (g_query(g_ctx, g_adapter, &pm) != 0) return out;
    out.socket_watts = sensor_value(pm, ADL_PMLOG_ASIC_POWER);
    if (out.socket_watts < 0) out.socket_watts = sensor_value(pm, ADL_PMLOG_SSPAIRED_ASICPOWER);
    if (out.socket_watts < 0) out.socket_watts = sensor_value(pm, ADL_PMLOG_BOARD_POWER);
    out.gfx_watts = sensor_value(pm, ADL_PMLOG_GFX_POWER);
    out.cpu_watts = sensor_value(pm, ADL_PMLOG_CPU_POWER);
    out.available =
        out.socket_watts >= 0 || out.gfx_watts >= 0 || out.cpu_watts >= 0;
    return out;
}

}  // namespace power

#elif defined(__linux__)

#include <cstdio>
#include <cstring>
#include <string>

#include <dirent.h>

namespace {

// amdgpu: /sys/class/drm/cardN/device/hwmon/hwmonX/power1_average（µW）。
std::string g_power_path;
int g_state = 0;  // 0 未初始化 / 1 可用 / -1 不可用
std::mutex g_mtx;

void sysfs_init() {
    for (int card = 0; card < 8; ++card) {
        const std::string dev =
            "/sys/class/drm/card" + std::to_string(card) + "/device";
        FILE* f = fopen((dev + "/vendor").c_str(), "r");
        if (!f) continue;
        unsigned int vendor = 0;
        const int n = fscanf(f, "0x%x", &vendor);
        fclose(f);
        if (n != 1 || vendor != 0x1002) continue;
        const std::string hwmon_dir = dev + "/hwmon";
        DIR* d = opendir(hwmon_dir.c_str());
        if (!d) continue;
        while (dirent* de = readdir(d)) {
            if (strncmp(de->d_name, "hwmon", 5) != 0) continue;
            const std::string base = hwmon_dir + "/" + de->d_name;
            for (const char* name : {"power1_average", "power1_input"}) {
                FILE* pf = fopen((base + "/" + name).c_str(), "r");
                if (!pf) continue;
                fclose(pf);
                g_power_path = base + "/" + name;
                g_state = 1;
                closedir(d);
                return;
            }
        }
        closedir(d);
    }
    g_state = -1;
}

}  // namespace

namespace power {

Reading read() {
    Reading out;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_state == 0) sysfs_init();
    if (g_state != 1) return out;
    FILE* f = fopen(g_power_path.c_str(), "r");
    if (!f) return out;
    unsigned long long uw = 0;
    const int n = fscanf(f, "%llu", &uw);
    fclose(f);
    if (n != 1) return out;
    out.socket_watts = static_cast<int>(uw / 1000000);  // µW → W
    out.available = true;
    return out;
}

}  // namespace power

#else

namespace power {

Reading read() { return Reading{}; }

}  // namespace power

#endif

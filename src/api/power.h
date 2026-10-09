// power: APU 功耗读取，控制台"功耗"卡片的数据源。
// Windows: ADL PMLog（atiadlxx.dll 运行时装载）——APU/核显上唯一可靠的功耗
// 路径，OverdriveN/5 在 APU 上读不到功耗（参考 Easy-GPU-Tools backend_adl.c）。
// Linux: amdgpu hwmon sysfs power1_average。两条路都失败时 available=false，
// 前端显示不可用。读数是传感器瞬时值，纯驱动调用，不影响推理。
#pragma once

namespace power {

struct Reading {
    bool available = false;
    int socket_watts = -1;  // 整封装（APU 含 CPU+GPU+SoC）
    int gfx_watts = -1;     // GPU 域（Linux sysfs 无单独读数）
    int cpu_watts = -1;     // CPU 域（同上）
};

// 线程安全，不抛异常。每次调用读取当前值（传感器瞬时功耗）。
Reading read();

}  // namespace power

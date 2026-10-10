#!/usr/bin/env python
# Windows 大额 hipMalloc 探针：不依赖完整引擎环境，直接验证 devarena 报错的三种可能：
#   1) 实际加载的是哪份 amdhip64_7.dll（build/ 自包含 vs System32 旧 SDK）
#   2) GPU 看到的 VRAM 总量/空闲量（BIOS carve-out 是否够、是否有进程占用）
#   3) 单笔 hipMalloc 上限（旧 HIP SDK DLL 的特征是卡在 ~41 GiB）
#
# 用法（在目标机器上、发布包根目录下运行，纯 ctypes，无第三方依赖）：
#   python tools/hipalloc_probe.py            # 依次探测 build/ 与 System32 的 HIP DLL
#   python tools/hipalloc_probe.py <dll路径>  # 只探测指定 DLL
# 环境变量 PROBE_ALLOC_GIB 可改目标分配额（默认 93.48，即 256K 配置的 arena 大小）。
#
# 判读：
#   - resolved 行必须指向发布包 build/ 目录；指到 System32 就是部署缺 DLL。
#   - VRAM total 明显小于 ~95 GiB → BIOS carve-out 不够，调 96 GiB。
#   - free 明显小于 total（无引擎在跑时）→ 有别的进程占着 VRAM（含僵尸 qwenox-engine-win.exe）。
#   - max single block 卡在 ~41 GiB → 加载到的是旧 HIP SDK 的 DLL。

import ctypes
import os
import subprocess
import sys
from ctypes import c_int, c_size_t, c_void_p, byref, create_string_buffer
from ctypes.wintypes import DWORD, HMODULE, LPCWSTR, LPWSTR

GiB = 1 << 30
TARGET_GIB = float(os.environ.get("PROBE_ALLOC_GIB", "93.48"))

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.LoadLibraryExW.restype = HMODULE
k32.LoadLibraryExW.argtypes = [LPCWSTR, c_void_p, DWORD]
k32.GetModuleHandleW.restype = HMODULE
k32.GetModuleHandleW.argtypes = [LPCWSTR]
k32.GetModuleFileNameW.restype = DWORD
k32.GetModuleFileNameW.argtypes = [HMODULE, LPWSTR, DWORD]


def modpath(name):
    hm = k32.GetModuleHandleW(name)
    if not hm:
        return None
    buf = ctypes.create_unicode_buffer(1024)
    k32.GetModuleFileNameW(hm, buf, 1024)
    return buf.value


def probe(dll_path):
    print(f"----- probing {dll_path} -----")
    h = k32.LoadLibraryExW(dll_path, None, 0x08)  # LOAD_WITH_ALTERED_SEARCH_PATH
    if not h:
        print("LoadLibraryEx failed, gle =", ctypes.get_last_error())
        return
    hip = ctypes.CDLL(None, handle=h)
    hip.hipGetErrorString.restype = ctypes.c_char_p
    rc = hip.hipInit(0)
    if rc != 0:
        print("hipInit rc =", rc, hip.hipGetErrorString(rc).decode())
        return
    for n in ("amdhip64_7.dll", "amdhip64_6.dll", "amd_comgr.dll",
              "amd_comgr_2.dll", "amd_comgr_3.dll"):
        p = modpath(n)
        if p:
            print(f"resolved {n} -> {p}")
    drv = c_int(0)
    try:
        hip.hipDriverGetVersion(byref(drv))
        print("driver version:", drv.value)
    except Exception:
        pass
    cnt = c_int(0)
    hip.hipGetDeviceCount(byref(cnt))
    print("devices:", cnt.value)
    if cnt.value < 1:
        return
    name = create_string_buffer(256)
    hip.hipDeviceGetName(name, 256, 0)
    print("device 0:", name.value.decode(errors="replace"))
    hip.hipSetDevice(0)
    free, total = c_size_t(0), c_size_t(0)
    hip.hipMemGetInfo(byref(free), byref(total))
    print(f"VRAM free {free.value / GiB:.2f} GiB / total {total.value / GiB:.2f} GiB"
          f" (in use {(total.value - free.value) / GiB:.2f})")

    def try_alloc(gib):
        ptr = c_void_p()
        rc = hip.hipMalloc(byref(ptr), c_size_t(int(gib * GiB)))
        if rc == 0:
            hip.hipFree(ptr)
        return rc

    rc = try_alloc(TARGET_GIB)
    print(f"single hipMalloc({TARGET_GIB:.2f} GiB): rc={rc}"
          f" ({hip.hipGetErrorString(rc).decode()})")
    if rc != 0:
        lo, hi = 1.0, TARGET_GIB
        for _ in range(9):
            mid = (lo + hi) / 2
            if try_alloc(mid) == 0:
                lo = mid
            else:
                hi = mid
        print(f"max single block ~= {lo:.2f} GiB")


def main():
    if len(sys.argv) > 1:
        probe(sys.argv[1])
        return
    # 同名 DLL 一个进程只能加载一份，逐个子进程探测。
    win = os.environ.get("SystemRoot", r"C:\Windows")
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(os.getcwd(), "build", "amdhip64_7.dll"),
        os.path.join(here, "..", "build", "amdhip64_7.dll"),
        os.path.join(win, "System32", "amdhip64_7.dll"),
        os.path.join(win, "System32", "amdhip64_6.dll"),
    ]
    seen = []
    for c in cands:
        c = os.path.normpath(c)
        if os.path.exists(c) and c not in seen:
            seen.append(c)
    if not seen:
        print("找不到任何 amdhip64 DLL；请在发布包根目录下运行，或用参数指定 DLL 路径。")
        sys.exit(1)
    for c in seen:
        print("=" * 72)
        subprocess.run([sys.executable, os.path.abspath(__file__), c])
    print("=" * 72)
    print("判读见脚本头部注释：加载路径 / VRAM total / free / max single block 四项。")


if __name__ == "__main__":
    main()

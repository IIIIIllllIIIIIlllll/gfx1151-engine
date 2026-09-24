#!/usr/bin/env python3
"""rocprofv3 --kernel-trace 结果（run_results.db 或 *kernel_stats.csv）的耗时排行。

  python3 tools/ktrace_top.py <目录或 db> [N=30] [--group]

--group 把模板实参折叠（k_gemm_wmma<...> 视为一类）；默认按完整名。
db 模式下额外按 grid 形状拆 k_gemm_wmma / Cijk（hipBLASLt）以区分 GEMM 形状。
"""
import csv
import glob
import os
import re
import sqlite3
import sys


def short(name, group):
    name = re.sub(r"\(.*$", "", name)          # 去掉参数列表
    name = name.replace("void ", "")
    if group:
        name = re.sub(r"<.*>", "<>", name)
    return name[:120]


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 30
    group = "--group" in sys.argv
    agg = {}
    shapes = {}
    dbs = [path] if path.endswith(".db") else glob.glob(os.path.join(path, "**", "*.db"), recursive=True)
    if dbs:
        c = sqlite3.connect(dbs[0])
        for name, dur, gx, gy, wx in c.execute(
                "select name, duration, grid_x, grid_y, workgroup_x from kernels"):
            k = short(name, group)
            a = agg.setdefault(k, [0, 0])
            a[0] += dur
            a[1] += 1
            if "gemm_wmma" in name or name.startswith("Cijk"):
                s = shapes.setdefault((short(name, True), gx, gy, wx), [0, 0])
                s[0] += dur
                s[1] += 1
    else:
        f = glob.glob(os.path.join(path, "**", "*kernel_stats.csv"), recursive=True)[0]
        for r in csv.DictReader(open(f)):
            k = short(r["Name"], group)
            a = agg.setdefault(k, [0, 0])
            a[0] += float(r["TotalDurationNs"])
            a[1] += int(r["Calls"])
    tot = sum(v[0] for v in agg.values())
    print("kernel 总计 %.1f ms，%d 种" % (tot / 1e6, len(agg)))
    for k, (d, cnt) in sorted(agg.items(), key=lambda x: -x[1][0])[:n]:
        print("%9.1f ms %5.1f%% %7d  %s" % (d / 1e6, 100 * d / tot, cnt, k))
    if shapes:
        print("\nGEMM 按 grid 拆分（grid_x/grid_y/wg，前 15）：")
        for (k, gx, gy, wx), (d, cnt) in sorted(shapes.items(), key=lambda x: -x[1][0])[:15]:
            print("%9.1f ms %5.1f%% %6d  grid=%dx%d wg=%d  %s" % (
                d / 1e6, 100 * d / tot, cnt, gx, gy, wx, k[:60]))


if __name__ == "__main__":
    main()

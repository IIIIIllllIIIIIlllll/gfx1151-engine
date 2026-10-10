#!/usr/bin/env python3
"""D0-M3: 跨序列专家并集比 u（HANDOFF-CONCURRENCY §4-M3）。

输入：QWENOX_MOE_TOPIDS_DUMP 产出的 dump 文件，每个文件 = 一个不同 prompt 的
一次串行 spec 运行。行格式：L <layer> B <base> P <P> 后接 P*k 个专家 id
（k 由 (字段数-6)/P 推得）。

把每个文件看成一路序列：第 r 轮 verify 该序列贡献 P=γ+1 行、P*k 个槽。
对 B 路序列的同一轮取并集：u = |并集| / 总槽数。B=1 即同序列内（已知 ~0.63-0.68）。

用法: python3 tools/c0_union.py dump1 dump2 [dump3 dump4]
"""
import sys
from collections import defaultdict


def load(path):
    per_layer = defaultdict(list)  # l -> [(P, [ids...]), ...] 按调用顺序
    n_skip = 0
    for line in open(path, errors="ignore"):
        if not line.startswith("L "):
            continue
        f = line.split()
        try:
            l = int(f[1]); P = int(f[5])
        except (IndexError, ValueError):
            n_skip += 1
            continue
        k = (len(f) - 6) // P
        if k < 1 or 6 + P * k != len(f):
            n_skip += 1
            continue
        per_layer[l].append((P, [int(x) for x in f[6:6 + P * k]]))
    return per_layer, n_skip


def main():
    paths = sys.argv[1:]
    assert 2 <= len(paths) <= 4, "需要 2 或 4 个 dump 文件"
    seqs, skips = zip(*[load(p) for p in paths])
    Ps = sorted({P for s in seqs for l in s for (P, _) in s[l]})
    print(f"文件数={len(paths)} 出现的 P 值: {Ps} (跳过坏行 {list(skips)})")
    for p, s in zip(paths, seqs):
        ncalls = sum(len(v) for v in s.values())
        print(f"  {p}: layers={len(s)} calls={ncalls}")
    for P in Ps:
        gamma = P - 1
        for B in ([1, 2, 4] if len(paths) == 4 else [1, 2]):
            if B > len(paths):
                continue
            # B=2 时对 (0,1) 与 (2,3) 两组各算一次取平均（4 文件时）
            groups = [[0]] if B == 1 else None
            if B == 1:
                groups = [[i] for i in range(len(paths))]
            elif B == 2:
                groups = [[0, 1]] + ([[2, 3]] if len(paths) == 4 else [])
            else:
                groups = [list(range(4))]
            tot_u = tot_s = 0
            for g in groups:
                layers = set.intersection(*[set(seqs[i]) for i in g])
                for l in sorted(layers):
                    calls = [[c for c in seqs[i][l] if c[0] == P] for i in g]
                    R = min(len(c) for c in calls)
                    for r in range(R):
                        u = set()
                        slots = 0
                        for c in calls:
                            u.update(c[r][1])
                            slots += len(c[r][1])
                        tot_u += len(u)
                        tot_s += slots
            if tot_s:
                print(f"P={P} (γ={gamma}) B={B}: union/slots = "
                      f"{tot_u}/{tot_s} = {tot_u / tot_s:.3f} "
                      f"(平均每窗 |并集|={tot_u / max(1, tot_s // (B * P * 10)):.1f})")


if __name__ == "__main__":
    main()

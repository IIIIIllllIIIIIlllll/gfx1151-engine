#!/usr/bin/env python3
"""Summarise llama-perplexity / gdec --kld-base logs as one comparison table.

  python3 tools/kld_table.py NAME=logs/kld_x.log [NAME=logs/kld_y.log ...] [--unsloth]

Both tools print the same llama.cpp statistic lines, so one parser reads both.
--unsloth appends unsloth's published Qwen3.8-Flash-Next table (their BF16
base, undisclosed dataset: compare shapes/ratios, not absolute values).
"""
import re
import sys

PATTERNS = {
    "ppl_q": r"^Mean PPL\(Q\)\s*:\s*([\d.]+)",
    "ppl_base": r"^Mean PPL\(base\)\s*:\s*([\d.]+)",
    "kld": r"^Mean    KLD:\s*([\d.eE+-]+)\s*±\s*([\d.eE+-]+)",
    "p999": r"^99\.9%   KLD:\s*([\d.eE+-]+)",
    "p99": r"^99\.0%   KLD:\s*([\d.eE+-]+)",
    "median": r"^Median  KLD:\s*([\d.eE+-]+)",
    "rms_dp": r"^RMS Δp\s*:\s*([\d.]+)",
    "top": r"^Same top p:\s*([\d.]+)\s*±\s*([\d.]+)",
}

UNSLOTH = [
    ("unsloth UD-IQ1_S", 72.5, 77.325, 0.396070, 7.2126),
    ("unsloth UD-IQ1_M", 74.5, 79.691, 0.314739, 6.1965),
    ("unsloth UD-Q2_K_XL", 78.9, 82.715, 0.224607, 4.9121),
    ("unsloth UD-IQ3_XXS", 82.0, 85.414, 0.165120, 4.0375),
    ("unsloth UD-Q3_K_XL", 90.0, 88.315, 0.106504, 3.0538),
    ("unsloth UD-IQ4_XS", 93.7, 89.554, 0.083630, 2.3677),
    ("unsloth UD-Q4_K_XL", 111.3, 92.255, 0.046893, 1.5468),
    ("unsloth UD-Q5_K_XL", 158.3, 93.680, 0.030415, 1.0036),
    ("unsloth UD-Q6_K_XL", 169.2, 94.089, 0.027091, 0.8416),
    ("unsloth Q8_0", 188.2, 94.122, 0.026574, 0.8118),
]


def parse(path):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f.read().replace("\r", "\n").splitlines():
            for key, pat in PATTERNS.items():
                m = re.match(pat, line)
                if m:
                    out[key] = [float(g) for g in m.groups()]
            m = re.search(r"computing over (\d+) chunks|^kld: (\d+) chunks", line)
            if m:
                out["chunks"] = int(m.group(1) or m.group(2))
    return out


def main(argv):
    rows = []
    for arg in argv:
        if arg == "--unsloth":
            continue
        name, _, path = arg.partition("=")
        if not path:
            name, path = path or arg, arg
        rows.append((name, parse(path)))
    head = f"{'name':<28} {'top-1 %':>14} {'mean KLD':>20} {'99.9% KLD':>10} {'99% KLD':>9} {'median':>9} {'RMS Δp%':>8} {'PPL(Q)':>9} {'PPL(base)':>9}"
    print(head)
    print("-" * len(head))
    bad = 0
    for name, r in rows:
        if "kld" not in r:
            print(f"{name:<28} (无 KLD 结果)")
            bad += 1
            continue
        top = f"{r['top'][0]:.3f}±{r['top'][1]:.3f}"
        kld = f"{r['kld'][0]:.6f}±{r['kld'][1]:.6f}"
        print(f"{name:<28} {top:>14} {kld:>20} {r['p999'][0]:>10.4f} {r['p99'][0]:>9.4f} "
              f"{r['median'][0]:>9.5f} {r['rms_dp'][0]:>8.3f} {r['ppl_q'][0]:>9.4f} {r['ppl_base'][0]:>9.4f}")
    if "--unsloth" in argv:
        print("\nunsloth 公布值（BF16 基准，数据集未公开，只能比相对位置）")
        print(f"{'name':<28} {'GB':>6} {'top-1 %':>8} {'mean KLD':>10} {'99.9% KLD':>10}")
        for name, gb, top, kld, p999 in UNSLOTH:
            print(f"{name:<28} {gb:>6.1f} {top:>8.3f} {kld:>10.6f} {p999:>10.4f}")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1:]))

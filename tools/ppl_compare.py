#!/usr/bin/env python3
"""Compare per-position teacher-forced losses from two gdec --ppl runs."""

import argparse
import math
from pathlib import Path


def read_losses(path):
    rows = []
    summary_count = None
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 4 and fields[0] == "ppl_token" and fields[1] != "pos":
                position, token = int(fields[1]), int(fields[2])
                loss = float(fields[3])
                if position != len(rows) or not math.isfinite(loss):
                    raise ValueError(f"{path}: invalid position or loss at {position}")
                rows.append((token, loss))
            elif fields[0] == "ppl_summary":
                summary_count = int(fields[1].removeprefix("count="))
    if not rows or summary_count != len(rows):
        raise ValueError(f"{path}: incomplete ppl run or wrong summary count")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="serial run or reference chunk size")
    parser.add_argument("candidate", help="chunk size to inspect")
    parser.add_argument("--chunk", type=int, required=True, help="candidate chunk size")
    parser.add_argument("--window", type=int, default=16, help="tokens on either side of a seam")
    args = parser.parse_args()
    if args.chunk < 1 or args.window < 1:
        parser.error("chunk and window must be positive")
    try:
        baseline = read_losses(args.baseline)
        candidate = read_losses(args.candidate)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if len(baseline) != len(candidate) or any(left[0] != right[0] for left, right in zip(baseline, candidate)):
        parser.error("token sequences or scored lengths differ")

    differences = [right[1] - left[1] for left, right in zip(baseline, candidate)]
    count = len(differences)
    base_mean = sum(loss for _, loss in baseline) / count
    candidate_mean = sum(loss for _, loss in candidate) / count
    print(f"tokens={count} baseline_mean_nll={base_mean:.8f} candidate_mean_nll={candidate_mean:.8f}")
    print(f"baseline_ppl={math.exp(base_mean):.8f} candidate_ppl={math.exp(candidate_mean):.8f}")
    print(f"mean_abs_delta={sum(map(abs, differences)) / count:.8f} max_abs_delta={max(map(abs, differences)):.8f}")
    for position in sorted(range(count), key=lambda index: abs(differences[index]), reverse=True)[:10]:
        print(f"pos={position} next_id={baseline[position][0]} delta_nll={differences[position]:+.8f}")
    for seam in range(args.chunk, count + 1, args.chunk):
        start = max(0, seam - args.window)
        end = min(count, seam + args.window)
        region = differences[start:end]
        print(f"seam={seam} range=[{start},{end}) mean_abs_delta={sum(map(abs, region)) / len(region):.8f} max_abs_delta={max(map(abs, region)):.8f}")


if __name__ == "__main__":
    main()

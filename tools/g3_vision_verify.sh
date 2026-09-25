#!/usr/bin/env bash
# G3.2 视觉塔一键验证：mmproj GGUF vs hgn 视觉文件。
#   1. tools/g3_vision_check.cpp：333 个 visual.* 张量逐位相同 + 覆盖完整（CPU）
#   2. 引擎离线 --vision-test：同一份随机 patches 分别用 hgn / GGUF 视觉塔跑前向，
#      所有层 dump 逐字节相同
# 用法：[BIN=build/gdec-gguf] [VISION_HGN=...] [MMPROJ=...] bash tools/g3_vision_verify.sh
set -u
cd "$(dirname "$0")/.."
D=${GGUF_DIR:-$HOME/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL}
MMPROJ=${MMPROJ:-$D/mmproj-BF16.gguf}
VISION_HGN=${VISION_HGN:-models/qwen38-flash-next-vision.hgn}
BIN=${BIN:-build/gdec-gguf}
W=${W:-/tmp/g3_vision}
fail=0
rm -rf "$W"; mkdir -p "$W"

echo "== 1. 张量对照（CPU）"
g++ -O2 -std=c++17 -pthread -Isrc tools/g3_vision_check.cpp -o build/g3_vision_check || exit 1
build/g3_vision_check --mmproj "$MMPROJ" --hgn "$VISION_HGN" | tail -3 | tee "$W/check.txt"
grep -q '^PASS$' "$W/check.txt" || fail=1

echo "== 2. 端到端前向（GPU，--vision-test，grid 1,16,16）"
python3 - "$W/patches.npy" <<'EOF'
import sys, array, random
random.seed(1234)
P, C = 256, 1536
a = array.array('f', (random.gauss(0.0, 1.0) for _ in range(P * C)))
hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (P, C)
hdr += ' ' * (63 - (10 + len(hdr)) % 64) + '\n'
with open(sys.argv[1], 'wb') as f:
    f.write(b'\x93NUMPY\x01\x00' + len(hdr).to_bytes(2, 'little') + hdr.encode())
    a.tofile(f)
EOF
for k in hgn gguf; do
  V=$VISION_HGN; [[ $k == gguf ]] && V=$MMPROJ
  "$BIN" - --vision-test "$W/$k" --vision-tower "$V" --patches "$W/patches.npy" \
    --mrope-grid 1,16,16 > "$W/$k.log" 2>&1
  rc=$?
  grep -E 'vision(-test)?:' "$W/$k.log" | tail -2
  [[ $rc == 0 ]] || { echo "$k: rc=$rc"; tail -5 "$W/$k.log"; fail=1; }
done
nh=$(ls "$W/hgn.layers" 2>/dev/null | wc -l)
if [[ $nh -gt 0 ]] && diff -rq "$W/hgn.layers" "$W/gguf.layers" > "$W/diff.txt"; then
  echo "dump 文件 $nh 个，逐字节相同"
else
  echo "dump 不同或缺失（hgn $nh 个）："; head "$W/diff.txt"; fail=1
fi

[[ $fail == 0 ]] && echo PASS || echo FAIL

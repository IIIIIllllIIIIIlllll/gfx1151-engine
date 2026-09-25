#!/usr/bin/env bash
# 一键 kvsnap 保存→重启→恢复 A/B，使用生产 kernel 配置（BF16 KV + WMMA + BTV）。
# 前台运行，结束时给出 ALL MATCH / DIFF。
#   bash tools/btv_kvsnap_verify.sh                 # 32768 prompt，maxctx 65536，不加载 MTP
#   N=8192 bash tools/btv_kvsnap_verify.sh
#   MTP_FILE=models/qwen38-flash-next-mtp.hgn bash tools/btv_kvsnap_verify.sh
#   PROBE_BINARY=/path/old/gdec bash tools/btv_kvsnap_verify.sh   # 用旧二进制对照（预期 DIFF）
# 流程：
#   1. 空快照目录起引擎 → GEN P(16 tok) 触发保存 → 同一引擎 GEN Q=P+out+tail，得 live 金标准
#   2. 重启引擎 → GEN Q，应命中 P 的快照（日志 "kvsnap: restored"）→ 结果必须与 live 逐 token 一致
# 前提：生产服务已停；已 bash build.sh engine。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/probe_lib.sh

N="${N:-32768}"
[[ -f data/qsa-oracle/$N.json ]] || {
  echo "缺少 data/qsa-oracle/$N.json；先试 git checkout HEAD -- data/qsa-oracle" >&2; exit 1; }
probe_precheck || exit 1

# 与 tools/serve_common.sh 的生产环境变量保持一致
export GDEC_QSA_KV_BF16=1 GDEC_QSA_WMMA=1 GDEC_QSA_WMMA_BTV=1
export GDEC_GEMM_WMMA=1 GDEC_GDN_FUSED=1
export GDEC_MOE_LT=1 GDEC_MOE_LT_BF16=1 GDEC_GR_BF16=1
export GDEC_GDN_STREAM=1 GDEC_GDN_WAVE=1 GDEC_NOWARMUP=1
export GDEC_PREFILL_CHUNK="${GDEC_PREFILL_CHUNK:-16384}"
export GDEC_INDEX_FUSED2=1 GDEC_PP_MOE_OUT=1 GDEC_INDEX_STREAM_SELECT=1
unset GDEC_KV_PAGED
# A1c：SAVE_KVP / RESTORE_KVP 分别给保存引擎、恢复引擎设 GDEC_KV_PAGED（默认都不分页）
#   SAVE_KVP=2 RESTORE_KVP=2 bash tools/btv_kvsnap_verify.sh   # 反转页表下保存+恢复
#   RESTORE_KVP=2 bash tools/btv_kvsnap_verify.sh              # 不分页保存，反转页表恢复
kvp_env() { if [[ -n "${1:-}" ]]; then export GDEC_KV_PAGED="$1"; else unset GDEC_KV_PAGED; fi; }
kvp_log() { local l; l="$(grep -m1 '\[kvpage\]' "$PROBE_LOG" || true)"; echo "页表日志: ${l:-（无 [kvpage] 行，不分页）}"; }
DIR="${KVSNAP_DIR:-data/kvsnap-btv}"
STATE="logs/kvsnap-btv.json"
export GDEC_KVSNAP=1 GDEC_KVSNAP_DIR="$DIR"
rm -rf "$DIR"; rm -f "$STATE"

MODEL_DIR="${MODEL_DIR:-models}"
engine=("${PROBE_BINARY:-build/gdec}" "$MODEL_DIR/qwen38-flash-next-w4b.hgn"
        "$MODEL_DIR/qwen38-flash-next-w4b.overlay.hgn")
[[ -z "${MTP_FILE:-}" ]] || engine+=("$MTP_FILE")
engine+=(--serve --port 8732 --maxctx "${CTX:-65536}")

trap probe_stop EXIT
trap 'exit 130' INT TERM
fail() { echo "BTV KVSNAP VERIFY: FAIL — $*" >&2; exit 1; }

echo "================ 1/2 保存（N=$N，快照目录 $DIR）================"
kvp_env "${SAVE_KVP:-}"
probe_start btv-save "${engine[@]}" || fail "引擎启动失败"
kvp_log
python3 tools/kvsnap_ab.py save --n "$N" --state "$STATE" || fail "save 请求失败，见 $PROBE_LOG"
# 第二个请求结束后也会保存一次；等它写完再停，避免停在写盘中途
for _ in {1..120}; do (( $(grep -c 'kvsnap: saved' "$PROBE_LOG") >= 2 )) && break; sleep 1; done
grep 'kvsnap: \(saved\|save skipped\)' "$PROBE_LOG" || true
grep -q 'kvsnap: saved' "$PROBE_LOG" || fail "日志里没有 'kvsnap: saved'，快照没存下来"
probe_stop

echo
echo "================ 2/2 重启并恢复 ================"
kvp_env "${RESTORE_KVP:-}"
probe_start btv-restore "${engine[@]}" || fail "引擎启动失败"
kvp_log
python3 tools/kvsnap_ab.py run --tag restore --state "$STATE" || fail "run 请求失败，见 $PROBE_LOG"
restored="$(grep -m1 'kvsnap: restored' "$PROBE_LOG" || true)"
probe_stop
trap - EXIT
[[ -n "$restored" ]] || fail "没有命中快照（日志无 'kvsnap: restored'），对比无意义；见 logs/btv-restore.log"
echo "$restored"

echo
echo "================ 对比 ================"
python3 tools/kvsnap_ab.py compare --state "$STATE"
rc=$?
(( rc == 0 )) && echo "BTV KVSNAP VERIFY: PASS" || echo "BTV KVSNAP VERIFY: FAIL（恢复后续写与 live 不一致）"
exit $rc

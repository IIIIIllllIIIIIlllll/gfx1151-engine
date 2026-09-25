#!/usr/bin/env bash
# A5 C 场景隔离实验：每组条件下起 off / p1 两个引擎，只跑多轮对话 T1->T2->T3，逐 token 对比。
#   bash tools/a5_c.sh                         # 默认全部变体
#   bash tools/a5_c.sh base serial nosnap      # 只跑指定变体
#   MODES="off off2 p1 p1b" bash tools/a5_c.sh base   # 以 off 开头的模式不分页，其余按生产分页；都和第一个比
# 变体格式见 VARIANTS：名字|额外环境变量(空格分隔)|a5_c.py 参数
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/probe_lib.sh
mkdir -p logs/a5c

declare -A VARIANTS=(
  [base]="||--drafters 4,4,1 --snaps 1"
  [serial]="||--drafters 0,0,0 --snaps 1"
  [t2serial]="||--drafters 4,0,0 --snaps 1"
  [nosnap]="||--drafters 4,4,1 --snaps 0"
  [mtp]="||--drafters 1,1,1 --snaps 1"
  [ngram]="||--drafters 3,3,3 --snaps 1"
  [norckpt]="GDEC_RCKPT_MAX=0||--drafters 4,4,1 --snaps 1"
  [nokvsnap]="GDEC_KVSNAP=0||--drafters 4,4,1 --snaps 1"
  [oldsel]="GDEC_INDEX_OLDSEL=1||--drafters 4,4,1 --snaps 1"
  # 先跑 a5_ab 的 F/S 用例（全部或部分）再跑 C，用来二分是哪个前置请求让 C 分叉
  [full]="||--drafters 4,4,1 --snaps 1 --pre all"
  [preF]="||--drafters 4,4,1 --snaps 1 --pre F"
  [preS]="||--drafters 4,4,1 --snaps 1 --pre S"
  # 只用 32K 旧数据当前置（F-L32-d0，1 个请求）：串行 / chain+MTP / 纯 MTP
  [l32]="||--drafters 4,4,1 --snaps 1 --pre F-L32-d0"
  [l32ser]="||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  [l32mtp]="||--drafters 1,1,1 --snaps 1 --pre F-L32-d0"
  # 调试：GDEC_KV_ZERO=1 让两种模式里"本序列没写过的行"都读成 0
  [zero]="GDEC_KV_ZERO=1||--drafters 4,4,1 --snaps 1 --pre all"
  # 串行 + 32K 前置，只清零某一块 buffer（k/v/t=BTV 转置 V/i=indexer key），看哪块让 off==p1
  [zk]="GDEC_KV_ZERO=k||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  [zv]="GDEC_KV_ZERO=v||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  [zt]="GDEC_KV_ZERO=t||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  [zi]="GDEC_KV_ZERO=i||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  # 每次 prefill 前核对转置 V 与行主序 V 在 [0, base) 是否一致（结果在引擎日志 [vctchk]）
  [vchk]="GDEC_VCT_CHECK=1||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  # prefill 写完转置 V 后把尾块里 >= base+P 的 slot 清零（两种模式都做）
  [vtail]="GDEC_VCT_TAILZERO=1||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  # 投毒：尾块旧 slot 写 bf16 大数(0x7000≈1.6e38) / NaN(0x7fc0)，看是否被读到
  [vbig]="GDEC_VCT_TAILZERO=1 GDEC_VCT_TAILVAL=7000||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  [vone]="GDEC_VCT_TAILZERO=1 GDEC_VCT_TAILVAL=3f80||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  [vinf]="GDEC_VCT_TAILZERO=1 GDEC_VCT_TAILVAL=7f80||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  [vsub]="GDEC_VCT_TAILZERO=1 GDEC_VCT_TAILVAL=0001||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  [ahash]="GDEC_ATTN_HASH=1||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  [vneg]="GDEC_VCT_TAILZERO=1 GDEC_VCT_TAILVAL=bf80||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  [vrand]="GDEC_VCT_TAILZERO=1 GDEC_VCT_TAILVAL=ffff||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  [vnan]="GDEC_VCT_TAILZERO=1 GDEC_VCT_TAILVAL=7fc0||--drafters 0,0,0 --snaps 1 --pre F-L32-d0"
  # 历史依赖：同一引擎里 T1 → 32K 无关 prompt → T1，看同一个全新 prompt 的输出是否变化
  [hist]="GDEC_RCKPT=0 GDEC_KVSNAP=0|hist|--drafters 0,1,3,4"
)
ORDER=(base serial t2serial nosnap mtp ngram norckpt nokvsnap oldsel)
(( $# )) && ORDER=("$@")

probe_precheck || exit 1
for f in start_hgn.sh start_gguf.sh tools/serve_common.sh service.conf; do grep -q $'\r' "$f" && sed -i 's/\r$//' "$f"; done
chk="$(bash start_hgn.sh --check 2>&1)" || { echo "$chk"; echo "start_hgn.sh --check 失败"; exit 1; }
mapfile -t PENV < <(sed -n 's/^ENV //p' <<<"$chk")
CMDLINE="$(sed -n 's/^CMD //p' <<<"$chk")"
eval "ENGINE=($CMDLINE)"
for i in "${!ENGINE[@]}"; do [[ "${ENGINE[$i]}" == --port ]] && ENGINE[$((i + 1))]=8732; done
PROBE_CAP_GB="$(source service.conf; echo "${MEMORY_CAP_GB:-86}")"; export PROBE_CAP_GB
SNAPDIR="$PWD/data/kvsnap-a5c"
trap 'probe_stop; rm -rf "$SNAPDIR"' EXIT
trap 'exit 130' INT TERM

declare -A RES
read -ra MODES <<<"${MODES:-off p1}"
for v in "${ORDER[@]}"; do
  spec="${VARIANTS[$v]:-}"; [[ -n "$spec" ]] || { echo "未知变体 $v"; exit 2; }
  IFS='|' read -r extra cmd pyargs <<<"$spec"
  cmd="${cmd:-run}"
  for t in "${MODES[@]}"; do
    for e in $(compgen -e | grep '^GDEC_'); do unset "$e"; done
    for e in "${PENV[@]}"; do export "$e"; done
    export GDEC_KVSNAP_DIR="$SNAPDIR"
    [[ $t == off* ]] && unset GDEC_KV_PAGED GDEC_KV_POOL_TOKENS
    for e in $extra; do export "$e"; done
    rm -rf "$SNAPDIR"; mkdir -p "$SNAPDIR"
    probe_start "a5c-$v-$t" "${ENGINE[@]}" || { RES[$v]="引擎启动失败($t)"; continue 2; }
    out="$(python3 tools/a5_c.py "$cmd" --tag "$v-$t" $pyargs 2>&1)"; echo "$out"
    [[ $cmd == hist ]] && RES[$v]+="$t: $(grep -c SAME <<<"$out") SAME / $(grep -c "DIFF@" <<<"$out") DIFF  "
    grep -E 'hipError|Segmentation|Aborted|FATAL|GUARD PAGE' "$PROBE_LOG" | head -3
    probe_stop
  done
  if [[ $cmd == run ]]; then
    RES[$v]=""
    for t in "${MODES[@]:1}"; do
      RES[$v]+="${MODES[0]}~$t: $(python3 tools/a5_c.py compare --a "$v-${MODES[0]}" --b "$v-$t" 2>&1) | "
    done
  fi
  echo ">>> $v: ${RES[$v]}"
done

echo
echo "==== A5 C 隔离实验汇总（off vs p1）===="
for v in "${ORDER[@]}"; do printf '%-10s %-40s %s\n' "$v" "${VARIANTS[$v]}" "${RES[$v]:-未运行}"; done

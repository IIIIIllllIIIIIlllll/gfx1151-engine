# start_hgn.sh / start_gguf.sh 的公共部分（由启动器 source，不单独运行）。
# 启动器依次：
#   serve_init "$@"       参数、service.conf、随包运行库、数值与编译产物检查
#   need 说明 路径 [提示]  登记本格式需要的权重文件，缺失的在 serve_run 里一次性列出
#   设 FORMAT / MAIN_MODEL / MODEL_ARGS / VISION / MISSING_HINT，需要时 export GDEC_GGUF_*
#   serve_run             tokenizer、端口、进程、内存检查，生产环境变量，--check 或启动
# 调用前需已设置 ROOT（项目根目录，已 cd 进去）和 LAUNCHER（启动器文件名）。

fail() { echo "错误：$*" >&2; exit 1; }

MISSING=()
need() {
  [[ -f "$2" && -r "$2" ]] || MISSING+=("$1：$2${3:+（$3）}")
}

serve_init() {
  local usage="用法：bash $LAUNCHER [--check]"
  [[ $# -le 1 ]] || fail "$usage"
  case "${1:-}" in
    -h|--help) echo "$usage；--check 只检查配置，Ctrl+C 停止服务。"; exit 0 ;;
    '') CHECK=0 ;;
    --check) CHECK=1 ;;
    *) fail "$usage" ;;
  esac
  source "$ROOT/service.conf"
  ENGINE_HOST="${ENGINE_HOST:-127.0.0.1}"
  # 旧版 service.conf 没有这几项时的缺省值（与 service.conf 相同）
  KV_PAGED="${KV_PAGED:-1}"
  KV_POOL_TOKENS="${KV_POOL_TOKENS:-0}"
  PARALLEL="${PARALLEL:-1}"
  CONC_PREFILL_CHUNK="${CONC_PREFILL_CHUNK:-8192}"
  PREFILL_CHUNK="${PREFILL_CHUNK:-0}"
  MAX_IMAGES="${MAX_IMAGES:-8}"
  ROPE_FACTOR="${ROPE_FACTOR:-1}"
  ROPE_ORIGINAL_CTX="${ROPE_ORIGINAL_CTX:-262144}"
  ROPE_BETA_FAST="${ROPE_BETA_FAST:-32}"
  ROPE_BETA_SLOW="${ROPE_BETA_SLOW:-1}"
  ROPE_ATTN_SCALE="${ROPE_ATTN_SCALE:-0}"
  if [[ -f "$ROOT/build/bundled-runtime.conf" ]]; then
    # --bundle 产物优先使用随包库；kernel db 使用绝对路径，不依赖 cwd。
    [[ -d "$ROOT/build/lib" ]] || fail '缺少 build/lib，请重新运行 bash build.sh --bundle'
    export LD_LIBRARY_PATH="$ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ROCBLAS_TENSILE_LIBPATH="$ROOT/build/lib/rocblas/library"
    export HIPBLASLT_TENSILE_LIBPATH="$ROOT/build/lib/hipblaslt/library"
    [[ -d "$ROCBLAS_TENSILE_LIBPATH" && -d "$HIPBLASLT_TENSILE_LIBPATH" ]] || \
      fail '缺少 rocBLAS/hipBLASLt kernel db，请重新运行 bash build.sh --bundle'
    local pattern
    for pattern in 'libamdhip64.so.*' 'librocblas.so.*' 'libhipblaslt.so.*' \
                   'libpng16.so.*' 'libjpeg.so.*' 'libwebp.so.*'; do
      compgen -G "$ROOT/build/lib/$pattern" >/dev/null || \
        fail "缺少随包运行库：$pattern，请重新运行 bash build.sh --bundle"
    done
  fi
  local key value cmd
  [[ "$ENGINE_HOST" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || fail 'ENGINE_HOST 必须是 IPv4 地址'
  local engine_octets octet
  IFS=. read -r -a engine_octets <<< "$ENGINE_HOST"
  for octet in "${engine_octets[@]}"; do
    [[ "$octet" =~ ^(0|[1-9][0-9]{0,2})$ ]] && (( 10#$octet <= 255 )) || fail 'ENGINE_HOST 必须是 IPv4 地址'
  done
  for key in ENGINE_PORT API_PORT MAX_CONTEXT MEMORY_CAP_GB MIN_AVAILABLE_GB START_TIMEOUT STALL_TIMEOUT ROPE_ORIGINAL_CTX; do
    value="${!key}"
    [[ "$value" =~ ^[1-9][0-9]*$ && ${#value} -le 8 ]] || fail "$key 必须为正整数"
  done
  for key in KVSNAP_MAX_GB RCKPT_MAX KV_POOL_TOKENS MTP_GAMMA CONC_PREFILL_CHUNK PREFILL_CHUNK; do
    value="${!key}"
    [[ "$value" =~ ^(0|[1-9][0-9]*)$ && ${#value} -le 8 ]] || fail "$key 必须为非负整数"
  done
  [[ "$PLE_URING" =~ ^[01]$ ]] || fail 'PLE_URING 必须为 0 或 1'
  [[ "$KV_PAGED" =~ ^[01]$ ]] || fail 'KV_PAGED 必须为 0 或 1'
  [[ "$PARALLEL" =~ ^[1-8]$ ]] || fail 'PARALLEL 范围为 1–8'
  [[ "$MAX_IMAGES" =~ ^[1-9][0-9]*$ && "$MAX_IMAGES" -le 256 ]] || fail 'MAX_IMAGES 范围为 1–256'
  [[ "$ROPE_FACTOR" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail 'ROPE_FACTOR 必须为非负小数'
  [[ "$ROPE_BETA_FAST" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail 'ROPE_BETA_FAST 必须为非负小数'
  [[ "$ROPE_BETA_SLOW" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail 'ROPE_BETA_SLOW 必须为非负小数'
  [[ "$ROPE_ATTN_SCALE" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail 'ROPE_ATTN_SCALE 必须为非负小数'
  awk "BEGIN { exit !($ROPE_FACTOR >= 1 && $ROPE_BETA_FAST > 0 && $ROPE_BETA_SLOW > 0 && $ROPE_ATTN_SCALE >= 0) }" || \
    fail 'ROPE_FACTOR 必须 >=1，beta 必须 >0，ROPE_ATTN_SCALE 必须 >=0'
  # 单条上限不得超出位置编码范围（与引擎 52_main.inc 的校验一致）：YaRN 开启时
  # MAX_CONTEXT 上限 = ROPE_FACTOR × ROPE_ORIGINAL_CTX；想给并发备更多 KV，
  # 用 KV_POOL_TOKENS 扩大共享池，不要拉伸 MAX_CONTEXT。
  awk "BEGIN { exit !($ROPE_FACTOR <= 1 || $MAX_CONTEXT <= $ROPE_FACTOR * $ROPE_ORIGINAL_CTX) }" || \
    fail "MAX_CONTEXT=$MAX_CONTEXT 超出 YaRN 上限 ROPE_FACTOR×ROPE_ORIGINAL_CTX；调小 MAX_CONTEXT 或调大 ROPE_FACTOR（并发容量用 KV_POOL_TOKENS 扩）"
  awk "BEGIN { exit !($ROPE_FACTOR > 1 || $MAX_CONTEXT <= $ROPE_ORIGINAL_CTX) }" || \
    echo "警告：ROPE_FACTOR=1 且 MAX_CONTEXT=$MAX_CONTEXT 超过原生 $ROPE_ORIGINAL_CTX，超出部分的位置编码未验证（需要时请设 ROPE_FACTOR）" >&2
  (( PARALLEL == 1 || KV_PAGED )) || fail 'PARALLEL>1 需要 KV_PAGED=1'
  (( ENGINE_PORT <= 65535 && API_PORT <= 65535 && ENGINE_PORT != API_PORT )) || fail '端口必须为不同的 1–65535 整数'
  (( MTP_GAMMA <= 8 )) || fail 'MTP_GAMMA 范围为 0–8（0=引擎按模式自选 greedy 4 / 采样自适应）'
  for cmd in flock ss systemctl stat awk pgrep setsid; do command -v "$cmd" >/dev/null || fail "缺少命令：$cmd"; done
  systemctl --user show-environment >/dev/null || fail 'systemd 用户会话不可用，请通过普通用户 SSH 登录运行'
  [[ -x build/gdec && -x build/gdec-api ]] || fail '缺少编译产物，请先运行 bash build.sh'
  # 权重格式只由启动器决定：外部残留的 GDEC_GGUF* 一律清掉，GGUF 启动器再按 service.conf 导出。
  unset GDEC_GGUF GDEC_GGUF_DENSE GDEC_GGUF_MTP GDEC_GGUF_MTP_EXPERTS GDEC_GGUF_PLE GDEC_GGUF_DENSE_FILTER
}

serve_run() {
  need tokenizer "$TOKENIZER_DIR/tokenizer.json" '两种权重共用，取自原模型 HF 仓库'
  if (( ${#MISSING[@]} )); then
    {
      echo "错误：$LAUNCHER（$FORMAT 权重）缺少以下文件："
      printf '  %s\n' "${MISSING[@]}"
      echo "$MISSING_HINT"
      echo '路径在 service.conf 中配置（MODEL_DIR 默认为项目下的 ./models），同名环境变量优先。'
    } >&2
    exit 1
  fi
  local port processes available
  for port in "$ENGINE_PORT" "$API_PORT"; do
    [[ -z "$(ss -H -ltn "sport = :$port")" ]] || fail "端口 $port 已被占用"
  done
  if processes="$(pgrep -af '(^|/)(gdec[^/[:space:]]*|flash_serve|serve_api\.py)([[:space:]]|$)')"; then
    fail "已有引擎或 API 进程，请先停止：$processes"
  fi
  available="$(awk '/MemAvailable:/{print int($2/1048576)}' /proc/meminfo)"
  (( available >= MIN_AVAILABLE_GB )) || fail "可用内存 ${available} GiB，要求至少 ${MIN_AVAILABLE_GB} GiB"
  # Existing production options; no closed-source modules or experimental paths.
  # （放在 --check 之前：--check 会打印这些变量，tools/a5_verify.sh 等据此复现生产环境）
  export GDEC_QSA_KV_BF16=1 GDEC_QSA_WMMA=1 GDEC_QSA_WMMA_BTV=1
  # 自写 WMMA dense GEMM（Phase 3g）：8K +2%、32K +3%，对拍 8049 token 仅尾部分歧 4 个。
  export GDEC_GEMM_WMMA=1
  #export GDEC_PROF=1        # 临时诊断：每个 prefill chunk 打印 ple_host/ple_wait 等耗时
  # GDN 融合持久化 kernel（Phase 3c）：intra+strip 全融合、ws 不落 DRAM，kernel 3.34×，
  # 8K +5.7%、32K +4.4%，ids 对拍 8054 token 0 分歧。与 GDEC_GDN_PIPE2 互斥（fused 优先）。
  export GDEC_GDN_FUSED=1
  export GDEC_MOE_LT=1 GDEC_MOE_LT_BF16=1 GDEC_GR_BF16=1
  export GDEC_GDN_STREAM=1 GDEC_GDN_WAVE=1 GDEC_NOWARMUP=1
  # 32768: 32K prompt 单 chunk 实测 +7.4%（1155 vs 1076 tok/s）；65536 超内存 PSI 上限。
  # 32768性能最佳但是吃的显存太多，8192吃的最少但是性能最差，16384折中一下，性能损失不大，吃的显存更少
  # service.conf 的 PREFILL_CHUNK>0 时优先（同名环境变量再优先于 conf），否则 16384。
  if (( PREFILL_CHUNK > 0 )); then
    export GDEC_PREFILL_CHUNK="$PREFILL_CHUNK"
  else
    export GDEC_PREFILL_CHUNK=16384
  fi
  # 并发（D1a）：引擎 serve 且 GDEC_PARALLEL>1 时用这个分段代替上面的 16384。prefill 只在
  # 层间让出 GPU，32K prompt 时别的会话最长卡顿 16384 约 0.8–1.4 s、8192 约 0.6 s，
  # 单独 PP 不变（09-29 d1a_verify）。由引擎判断并发，离线工具（pp_prod/kld 等复用这些 ENV）不受影响。
  unset GDEC_CONC_PREFILL_CHUNK
  if (( CONC_PREFILL_CHUNK > 0 )); then export GDEC_CONC_PREFILL_CHUNK="$CONC_PREFILL_CHUNK"; fi
  export GDEC_INDEX_FUSED2=1 GDEC_PP_MOE_OUT=1 GDEC_INDEX_STREAM_SELECT=1
  if (( KVSNAP_MAX_GB )); then export GDEC_KVSNAP=1; else export GDEC_KVSNAP=0; fi
  # PLE io_uring 聚集由 service.conf 的 PLE_URING 控制；引擎只查 GDEC_PLE_URING
  # 的存在性（设 0 也会开），故 0 时必须不导出。
  if (( PLE_URING )); then export GDEC_PLE_URING=1; fi
  export GDEC_KVSNAP_MAX_GB="$KVSNAP_MAX_GB"
  export GDEC_RCKPT_MAX="$RCKPT_MAX"
  # 分页 KV（A5 起默认开启）。引擎按 atoi 解析，0 即关闭；这里仍然只在开启时导出，
  # 并清掉外部环境里可能残留的值，保证 service.conf 说了算。
  unset GDEC_KV_PAGED GDEC_KV_POOL_TOKENS
  if (( KV_PAGED )); then
    export GDEC_KV_PAGED=1
    if (( KV_POOL_TOKENS )); then export GDEC_KV_POOL_TOKENS="$KV_POOL_TOKENS"; fi
  fi
  export GDEC_PARALLEL="$PARALLEL"
  export GDEC_API_MAX_IMAGES="$MAX_IMAGES"
  export GDEC_ROPE_FACTOR="$ROPE_FACTOR"
  export GDEC_ROPE_ORIGINAL_CTX="$ROPE_ORIGINAL_CTX"
  export GDEC_ROPE_BETA_FAST="$ROPE_BETA_FAST"
  export GDEC_ROPE_BETA_SLOW="$ROPE_BETA_SLOW"
  export GDEC_ROPE_ATTN_SCALE="$ROPE_ATTN_SCALE"
  # --serve reads GDEC_SPEC_GAMMA; --gamma is for offline --spec-gen.
  # MTP_GAMMA=0：不导出，引擎按请求模式自选（greedy 4 / 采样自适应）。
  if (( MTP_GAMMA > 0 )); then export GDEC_SPEC_GAMMA="$MTP_GAMMA"; fi
  local engine_connect_host="$ENGINE_HOST"
  if [[ "$engine_connect_host" == 0.0.0.0 ]]; then engine_connect_host=127.0.0.1; fi
  local engine=("$ROOT/build/gdec" "${MODEL_ARGS[@]}" --serve --host "$ENGINE_HOST" --port "$ENGINE_PORT" --maxctx "$MAX_CONTEXT")
  [[ -z "$VISION" ]] || engine+=(--vision-tower "$VISION")

  echo "项目：$ROOT"
  echo "权重：$FORMAT，$MAIN_MODEL"
  echo "配置：${MAX_CONTEXT} 上下文，MTP gamma=$([[ $MTP_GAMMA == 0 ]] && echo 'auto（greedy 4 / 采样自适应）' || echo "$MTP_GAMMA")，engine ${ENGINE_HOST}:${ENGINE_PORT}，API ${API_HOST}:${API_PORT}"
  if (( KV_PAGED )); then
    echo "KV：分页，页池 $(( (KV_POOL_TOKENS > MAX_CONTEXT ? KV_POOL_TOKENS : MAX_CONTEXT) )) token（${PARALLEL} 路并发共享），RAM 检查点 ${RCKPT_MAX} 个"
  else
    echo "KV：不分页（KV_PAGED=0）"
  fi
  if (( PARALLEL > 1 && CONC_PREFILL_CHUNK > 0 )); then
    local conc=$CONC_PREFILL_CHUNK
    (( conc > GDEC_PREFILL_CHUNK )) && conc=$GDEC_PREFILL_CHUNK
    echo "prefill 分段：${conc}（并发，取 CONC_PREFILL_CHUNK 与单路 ${GDEC_PREFILL_CHUNK} 的较小值）"
  fi
  if (( CHECK )); then
    # 机器可读：引擎环境变量与命令行（tools/a5_verify.sh、pp_prod.sh 等解析这两段）
    env | grep '^GDEC_' | sort | sed 's/^/ENV /'
    printf 'CMD'; printf ' %q' "${engine[@]}"; echo
    echo '检查通过；没有启动引擎或 API。'
    exit 0
  fi

  mkdir -p logs
  exec 9>logs/.service.lock
  flock -n 9 || fail '本目录已有启动脚本在运行'
  exec 8>"${XDG_RUNTIME_DIR:-/run/user/$UID}/hgn-work.lock"
  flock -n 8 || fail '另一个 gfx1151-engine 编译或启动任务正在运行'
  RUN_LOG_DIR="logs/$(date +%Y%m%d-%H%M%S)-$$"
  mkdir "$RUN_LOG_DIR"
  ENGINE_LOG="$RUN_LOG_DIR/engine.log"
  API_LOG="$RUN_LOG_DIR/api.log"
  engine_pid=''
  api_pid=''
  scope=''
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP

  echo "加载模型中，日志：$ENGINE_LOG；Ctrl+C 停止。"
  setsid bash tools/run_capped.sh "$MEMORY_CAP_GB" -- "${engine[@]}" >"$ENGINE_LOG" 2>&1 9>&- 8>&- &
  engine_pid=$!
  scope="capped-${engine_pid}.scope"

  local begin=$SECONDS last_progress=$SECONDS previous='' current result
  until grep -q 'serve: listening' "$ENGINE_LOG"; do
    kill -0 "$engine_pid" 2>/dev/null || { tail -n 15 "$ENGINE_LOG" >&2; fail '引擎提前退出'; }
    (( SECONDS - begin < START_TIMEOUT )) || fail "引擎启动超过 ${START_TIMEOUT} 秒，查看 $ENGINE_LOG"
    current="$(progress_stamp)"
    if [[ "$current" != "$previous" ]]; then previous="$current"; last_progress=$SECONDS; fi
    (( SECONDS - last_progress < STALL_TIMEOUT )) || fail "加载停滞 ${STALL_TIMEOUT} 秒，查看 $ENGINE_LOG"
    sleep 1
  done
  kill -0 "$engine_pid" 2>/dev/null || fail '引擎已退出'
  "$ROOT/build/gdec-api" --tokenizer "$TOKENIZER_DIR" \
    --engine "$engine_connect_host:$ENGINE_PORT" --host "$API_HOST" \
    --port "$API_PORT" --context "$MAX_CONTEXT" >"$API_LOG" 2>&1 9>&- 8>&- &
  api_pid=$!
  begin=$SECONDS
  until [[ -n "$(ss -H -ltn "sport = :$API_PORT")" ]]; do
    kill -0 "$engine_pid" 2>/dev/null || fail '引擎已退出'
    kill -0 "$api_pid" 2>/dev/null || { tail -n 15 "$API_LOG" >&2; fail 'API 提前退出'; }
    (( SECONDS - begin < 15 )) || fail "API 启动超时，查看 $API_LOG"
    sleep 0.2
  done
  echo "服务已就绪：http://${API_HOST}:${API_PORT}/v1（0.0.0.0 表示监听所有网卡）"
  echo "日志：$RUN_LOG_DIR；Ctrl+C 同时停止 API 和引擎。"
  if wait -n "$engine_pid" "$api_pid"; then
    fail '服务进程意外结束'
  else
    result=$?
    echo "服务进程退出（$result），查看 $RUN_LOG_DIR" >&2
    exit "$result"
  fi
}

cleanup() {
  local result=$?
  trap - EXIT INT TERM HUP
  # Stop only the scope created by this invocation, including GPU descendants.
  if [[ -n "$api_pid" ]]; then kill -TERM "$api_pid" 2>/dev/null || true; fi
  # setsid makes the wrapper and its children our private process group. This
  # also handles interruption before systemd has finished creating the scope.
  if [[ -n "$engine_pid" ]]; then kill -TERM -- "-$engine_pid" 2>/dev/null || true; fi
  if [[ -n "$scope" ]]; then
    timeout -k 2 8 systemctl --user stop "$scope" >/dev/null 2>&1 || \
      systemctl --user kill --signal=KILL "$scope" >/dev/null 2>&1 || true
  fi
  if [[ -n "$engine_pid" ]]; then kill -KILL -- "-$engine_pid" 2>/dev/null || true; fi
  if [[ -n "$api_pid" ]]; then
    for _ in {1..20}; do kill -0 "$api_pid" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$api_pid" 2>/dev/null || true
    wait "$api_pid" 2>/dev/null || true
  fi
  if [[ -n "$engine_pid" ]]; then wait "$engine_pid" 2>/dev/null || true; fi
  echo "本次服务已停止。日志：$RUN_LOG_DIR"
  exit "$result"
}

# Track actual engine I/O and GPU allocation as well as log output while loading.
progress_stamp() {
  local cg pid f
  stat -c %s "$ENGINE_LOG"
  cg="$(systemctl --user show "$scope" -p ControlGroup --value 2>/dev/null || true)"
  if [[ -n "$cg" && -r "/sys/fs/cgroup$cg/cgroup.procs" ]]; then
    while read -r pid; do
      awk '/^(rchar|wchar|read_bytes|write_bytes):/{print}' "/proc/$pid/io" 2>/dev/null || true
    done <"/sys/fs/cgroup$cg/cgroup.procs"
  fi
  for f in /sys/class/drm/card*/device/mem_info_gtt_used; do
    [[ ! -r "$f" ]] || cat "$f"
  done
}

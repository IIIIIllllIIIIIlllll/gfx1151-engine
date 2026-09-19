#!/usr/bin/env bash
# 统一编译入口:引擎(hipcc)与 API(g++)全部输出到 build/,默认并行编译。
#
# 用法:
#   bash build.sh                 # all:引擎 + API(服务器+CLI工具),并行
#   bash build.sh engine [名字]   # 只编引擎 → build/<名字>(默认 gdec)
#   bash build.sh api             # 只编 API 服务器 + CLI 工具
#   bash build.sh test            # 编 ktest 并运行
#
# 产物:
#   build/gdec    引擎(src/gpu/gdec.cpp)
#   build/gdec-api      API 服务器(src/api/*.cpp)
#   build/{tok_cli,tpl_cli,eng_cli,http_selftest,toolparse_test,vision_test}
#   build/ktest         引擎内核测试
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() { sed -n '2,14p' "$0"; }
[[ $# -le 2 ]] || { usage >&2; exit 2; }
TARGET="${1:-all}"
ENGINE_NAME="${2:-gdec}"
case "$TARGET" in
  -h|--help) usage; exit 0 ;;
  all|engine|api|test) ;;
  *) usage >&2; exit 2 ;;
esac
command -v flock >/dev/null || { echo '缺少 flock，请安装 util-linux' >&2; exit 1; }
if processes="$(pgrep -af '(^|/)(gdec[^/[:space:]]*|flash_serve|serve_api\.py)([[:space:]]|$)')"; then
  echo "已有引擎或 API 在运行，请先在原终端停止服务：$processes" >&2
  exit 1
fi
mkdir -p build
exec 9>build/.build.lock
flock -n 9 || { echo '本目录已有编译任务在运行' >&2; exit 1; }
exec 8>"${XDG_RUNTIME_DIR:-/run/user/$UID}/hgn-work.lock"
flock -n 8 || { echo '另一个 gfx1151-engine 编译或启动任务正在运行' >&2; exit 1; }

GPU_ARCH="${GPU_ARCH:-gfx1151}"
if [[ "$TARGET" != api ]]; then
  if [[ -z "${HIPCC:-}" ]]; then
    HIPCC="$(command -v hipcc || true)"
    HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
  fi
  command -v "$HIPCC" >/dev/null || { echo '找不到 hipcc，请安装 ROCm 或设置 HIPCC' >&2; exit 1; }
fi
CXX="${CXX:-g++}"

# run <内存cap GB> <超时秒> <cmd...>:内存限额 + 超时保护下执行编译命令
run() {
  local cap="$1" secs="$2"; shift 2
  # 8>&- 9>&-: 编译子进程(含 systemd scope 包装)不得继承锁 fd,否则它们
  # 比父进程晚退出时,紧跟的下一个 build.sh 调用会误判"已有编译任务"
  bash tools/run_capped.sh "$cap" -- timeout -k 3 "$secs" "$@" 8>&- 9>&-
}
# compile <输出> <cap> <超时> <cmd...>:先编到临时文件,成功后原子替换
compile() {
  local output="$1" cap="$2" secs="$3"; shift 3
  local tmp="${output}.tmp.$$"
  echo "[编译] $output"
  if run "$cap" "$secs" "$@" -o "$tmp"; then
    mv -f -- "$tmp" "$output"
  else
    rm -f -- "$tmp"
    return 1
  fi
}

build_engine() {
  compile "build/$ENGINE_NAME" 8 600 "$HIPCC" -O3 -Werror \
    --offload-arch="$GPU_ARCH" src/gpu/gdec.cpp -lrocblas -lhipblaslt
}

API_FLAGS=(-O2 -std=c++17 -Isrc/api -Wall -Wextra -Wpedantic -Werror)
# tokenizer/chat_template 由 CLI 与服务器共用,每个目标须显式列出源文件
# (否则 main.cpp 会与 CLI 的 main 冲突)。
build_api() {
  compile build/gdec-api 8 120 "$CXX" "${API_FLAGS[@]}" \
    src/api/http.cpp src/api/engine_client.cpp src/api/tokenizer.cpp \
    src/api/chat_template.cpp src/api/json_py.cpp src/api/toolparse.cpp \
    src/api/vision.cpp src/api/main.cpp -lpng -ljpeg -lwebp -lpthread || return 1
  compile build/tok_cli 8 120 "$CXX" "${API_FLAGS[@]}" \
    src/api/tokenizer.cpp src/api/tok_cli.cpp || return 1
  compile build/tpl_cli 8 120 "$CXX" "${API_FLAGS[@]}" \
    src/api/chat_template.cpp src/api/json_py.cpp src/api/tpl_cli.cpp || return 1
  compile build/eng_cli 8 120 "$CXX" "${API_FLAGS[@]}" \
    src/api/engine_client.cpp src/api/tokenizer.cpp src/api/eng_cli.cpp || return 1
  compile build/http_selftest 8 120 "$CXX" "${API_FLAGS[@]}" \
    src/api/http.cpp src/api/http_selftest.cpp -lpthread || return 1
  compile build/toolparse_test 8 120 "$CXX" "${API_FLAGS[@]}" \
    src/api/toolparse.cpp src/api/json_py.cpp src/api/toolparse_test.cpp || return 1
  compile build/vision_test 8 120 "$CXX" "${API_FLAGS[@]}" \
    src/api/vision.cpp src/api/tokenizer.cpp src/api/vision_test.cpp \
    -lpng -ljpeg -lwebp -lpthread || return 1
}

build_test() {
  compile build/ktest 8 600 "$HIPCC" -O3 -Werror --offload-arch="$GPU_ARCH" \
    -I src/gpu tools/ktest.cu -lrocblas -lhipblaslt || return 1
  run 8 600 build/ktest
}

case "$TARGET" in
  engine) build_engine ;;
  api)    build_api ;;
  test)   build_test ;;
  all)
    pids=()
    build_engine & pids+=($!)
    build_api    & pids+=($!)
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [[ "$fail" == 0 ]] || { echo '[失败] 见上方编译输出' >&2; exit 1; }
    ;;
esac
echo '[完成] 编译输出位于 build/'

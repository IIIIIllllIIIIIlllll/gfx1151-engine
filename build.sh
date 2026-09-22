#!/usr/bin/env bash
# 统一编译入口:引擎(hipcc)与 API(g++)全部输出到 build/,默认并行编译。
#
# 用法:
#   bash build.sh                 # all:引擎 + API(服务器+CLI工具),并行
#   bash build.sh --bundle        # 同上，并打包分发所需的全部运行库
#   bash build.sh engine [名字]   # 只编引擎 → build/<名字>(默认 gdec)
#   bash build.sh api             # 只编 API 服务器 + CLI 工具
#   bash build.sh test            # 编 ktest 并运行
#
# 产物:
#   build/gdec    引擎(src/gpu/gdec.cpp)
#   build/gdec-api      API 服务器(src/api/*.cpp)
#   build/{tok_cli,tpl_cli,eng_cli,http_selftest,toolparse_test,vision_test}
#   build/ktest         引擎内核测试
#   build/lib/          --bundle 时生成的运行库与 gfx1151 kernel db
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() { sed -n '2,15p' "$0"; }
BUNDLE_RUNTIME=0
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --bundle) BUNDLE_RUNTIME=1 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "未知选项: $arg" >&2; usage >&2; exit 2 ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done
set -- "${POSITIONAL[@]}"
[[ $# -le 2 ]] || { usage >&2; exit 2; }
TARGET="${1:-all}"
ENGINE_NAME="${2:-gdec}"
case "$TARGET" in
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
BUNDLE_RPATH=()
if (( BUNDLE_RUNTIME )); then
  # 旧式 DT_RPATH 让 $ORIGIN/lib 同时覆盖 ROCm 库的间接依赖。
  BUNDLE_RPATH=(-Wl,-rpath,'$ORIGIN/lib' -Wl,--disable-new-dtags)
fi
BUNDLE_SCAN_DIR="build/.runtime-scan.$$"
cleanup_build() { rm -rf -- "$BUNDLE_SCAN_DIR"; }
trap cleanup_build EXIT

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

# glibc、libstdc++ 与编译器 ABI 仍由目标 Linux 提供；其余实际动态依赖
# （ROCm 闭包、libdrm/libelf 等依赖，以及 API 图像库闭包）复制到 build/lib。
is_base_system_lib() {
  case "$1" in
    libc.so.*|libm.so.*|libpthread.so.*|libdl.so.*|librt.so.*|\
    libresolv.so.*|libutil.so.*|libgcc_s.so.*|libstdc++.so.*) return 0 ;;
    *) return 1 ;;
  esac
}

bundle_binary_libs() {
  local binary="$1" scan ldd_output line soname arrow resolved rest tmp
  [[ -x "$binary" ]] || { echo "找不到待打包 ELF: $binary" >&2; return 1; }
  command -v ldd >/dev/null || { echo '缺少 ldd，无法收集 Linux 运行库' >&2; return 1; }
  mkdir -p build/lib "$BUNDLE_SCAN_DIR"

  # 在无 lib/ 邻居的临时目录解析，避免上次打包的 $ORIGIN/lib 干扰本次 ldd。
  scan="$BUNDLE_SCAN_DIR/$(basename "$binary")"
  cp -f -- "$binary" "$scan"
  if ! ldd_output="$(LC_ALL=C ldd "$scan")"; then
    echo "无法解析 $binary 的运行库" >&2
    return 1
  fi
  while IFS= read -r line; do
    read -r soname arrow resolved rest <<<"$line"
    [[ "$arrow" == '=>' ]] || continue
    if [[ "$resolved" == 'not' ]]; then
      echo "$binary 缺少运行库: $soname" >&2
      return 1
    fi
    [[ "$resolved" == /* ]] || continue
    is_base_system_lib "$soname" && continue
    tmp="build/lib/${soname}.tmp.$$"
    cp -Lf --preserve=mode,timestamps -- "$resolved" "$tmp"
    mv -f -- "$tmp" "build/lib/$soname"
    echo "[运行库] $soname"
  done <<<"$ldd_output"
}

resolve_rocm_root() {
  local hipcc_path candidate
  if [[ -n "${ROCM_PATH:-}" ]]; then
    readlink -f -- "$ROCM_PATH"
    return
  fi
  hipcc_path="$(readlink -f -- "$(command -v "$HIPCC")")"
  candidate="$(dirname -- "$(dirname -- "$hipcc_path")")"
  if [[ -d "$candidate/lib/rocblas/library" ||
        -d "$candidate/lib64/rocblas/library" ]]; then
    printf '%s\n' "$candidate"
    return
  fi
  [[ -d /opt/rocm ]] && { readlink -f /opt/rocm; return; }
  echo '无法从 HIPCC 推断 ROCm 根目录，请设置 ROCM_PATH' >&2
  return 1
}

find_kernel_db() {
  local rocm_root="$1" component="$2" candidate
  for candidate in \
    "$rocm_root/lib/$component/library" \
    "$rocm_root/lib64/$component/library"; do
    [[ -d "$candidate" ]] && { printf '%s\n' "$candidate"; return; }
  done
  candidate="$(find -L "$rocm_root" -maxdepth 5 -type d \
    -path "*/$component/library" -print -quit 2>/dev/null || true)"
  [[ -n "$candidate" ]] && { printf '%s\n' "$candidate"; return; }
  echo "找不到 $component kernel db（ROCm: $rocm_root）" >&2
  return 1
}

bundle_kernel_db() {
  local rocm_root="$1" component="$2" src dst stage item base copied=0
  local -a matches
  src="$(find_kernel_db "$rocm_root" "$component")" || return 1
  dst="build/lib/$component/library"
  stage="build/lib/$component/.library.tmp.$$"
  rm -rf -- "$stage"
  mkdir -p "$stage"

  # 新版 ROCm 按 gfx 架构分目录；旧版使用顶层带架构名的分片，两种都收集。
  if [[ -d "$src/$GPU_ARCH" ]]; then
    cp -aL -- "$src/$GPU_ARCH" "$stage/"
    copied=1
  fi
  matches=("$src/"*"$GPU_ARCH"*)
  for item in "${matches[@]}"; do
    [[ -e "$item" || -L "$item" ]] || continue
    base="$(basename -- "$item")"
    [[ "$base" == "$GPU_ARCH" ]] && continue
    cp -aL -- "$item" "$stage/"
    copied=1
  done
  (( copied )) || {
    rm -rf -- "$stage"
    echo "$component kernel db 中没有 $GPU_ARCH: $src" >&2
    return 1
  }
  rm -rf -- "$dst"
  mv -- "$stage" "$dst"
  echo "[kernel db] $dst"
}

bundle_gpu_runtime() {
  local binary="$1" rocm_root
  bundle_binary_libs "$binary"
  require_bundled_lib 'libamdhip64.so.*'
  require_bundled_lib 'librocblas.so.*'
  require_bundled_lib 'libhipblaslt.so.*'
  rocm_root="$(resolve_rocm_root)"
  bundle_kernel_db "$rocm_root" rocblas
  bundle_kernel_db "$rocm_root" hipblaslt
}

require_bundled_lib() {
  local pattern="$1"
  compgen -G "build/lib/$pattern" >/dev/null || {
    echo "运行库未打包: $pattern" >&2
    return 1
  }
}

bundle_api_runtime() {
  bundle_binary_libs "$1"
  require_bundled_lib 'libpng16.so.*'
  require_bundled_lib 'libjpeg.so.*'
  require_bundled_lib 'libwebp.so.*'
}

build_engine() {
  compile "build/$ENGINE_NAME" 8 600 "$HIPCC" -O3 -Werror \
    --offload-arch="$GPU_ARCH" src/gpu/gdec.cpp -lrocblas -lhipblaslt \
    "${BUNDLE_RPATH[@]}"
}

API_FLAGS=(-O2 -std=c++17 -Isrc/api -Ithird_party -Wall -Wextra -Wpedantic
           -Werror "${BUNDLE_RPATH[@]}")
# tokenizer/chat_template 由 CLI 与服务器共用,每个目标须显式列出源文件
# (否则 main.cpp 会与 CLI 的 main 冲突)。
build_api() {
  compile build/gdec-api 8 120 "$CXX" "${API_FLAGS[@]}" \
    src/api/http.cpp src/api/engine_client.cpp src/api/tokenizer.cpp \
    src/api/chat_template.cpp src/api/json_py.cpp src/api/toolparse.cpp \
    src/api/vision.cpp src/api/reqstat.cpp src/api/main.cpp -lpng -ljpeg -lwebp -lpthread || return 1
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
    -I src/gpu tools/ktest.cu -lrocblas -lhipblaslt \
    "${BUNDLE_RPATH[@]}" || return 1
}

case "$TARGET" in
  engine)
    build_engine
    (( ! BUNDLE_RUNTIME )) || bundle_gpu_runtime "build/$ENGINE_NAME"
    ;;
  api)
    build_api
    (( ! BUNDLE_RUNTIME )) || bundle_api_runtime build/gdec-api
    ;;
  test)
    build_test
    if (( BUNDLE_RUNTIME )); then
      bundle_gpu_runtime build/ktest
      run 8 600 env \
        "LD_LIBRARY_PATH=$ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "ROCBLAS_TENSILE_LIBPATH=$ROOT/build/lib/rocblas/library" \
        "HIPBLASLT_TENSILE_LIBPATH=$ROOT/build/lib/hipblaslt/library" \
        build/ktest
    else
      run 8 600 build/ktest
    fi
    ;;
  all)
    pids=()
    build_engine & pids+=($!)
    build_api    & pids+=($!)
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [[ "$fail" == 0 ]] || { echo '[失败] 见上方编译输出' >&2; exit 1; }
    if (( BUNDLE_RUNTIME )); then
      bundle_gpu_runtime "build/$ENGINE_NAME"
      bundle_api_runtime build/gdec-api
    fi
    ;;
esac
if (( BUNDLE_RUNTIME )); then
  printf 'GPU_ARCH=%s\n' "$GPU_ARCH" >build/bundled-runtime.conf
  echo '[完成] 分发产物与私有运行库位于 build/'
else
  # 仅在本轮编译成功后清理旧分发库，避免编译失败破坏上次可用产物。
  rm -rf -- build/lib
  rm -f -- build/bundled-runtime.conf
  echo '[完成] 编译输出位于 build/（使用系统运行库）'
fi

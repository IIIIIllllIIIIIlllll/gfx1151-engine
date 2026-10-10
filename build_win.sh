#!/usr/bin/env bash
# Windows (TheRock) 编译入口：与 Linux build.sh 并列，产物输出到 build/。
#
# 用法:
#   bash build_win.sh            # 全部产物：引擎、benchmark、API、根目录入口
#   bash build_win.sh bench      # 性能测试工具 → build/qwenox-bench.exe
#   bash build_win.sh api        # OpenAI HTTP 前端 → build/qwenox-win.exe
#                                #   （GUI 托盘程序：API 服务 + 控制台网页 + 引擎启停）
#   bash build_win.sh launcher   # 根目录最小入口 → ./start_win.exe（双击拉起 API）
#   bash build_win.sh test       # 编 build/ktest-win.exe 并运行 kernel 单测
#
# 前置：TheRock 多架构包（默认 C:\therock-dist-windows-multiarch-10.0.0\...，
# 可用 THEROCK=/path 覆盖）；GPU_ARCH 默认 gfx1151。
#
# 脚本同时准备好运行环境（只需做一次，之后重复运行为幂等跳过）：
# - winlibs 暂存：hipcc 的 -l<name> 解析找 <name>.lib；TheRock 的 hipBLASLt
#   只有 libhipblaslt.dll.a —— 改名暂存一份（lld-link 直接吃 .dll.a 内容）。
# - 运行依赖 DLL 复制到 exe 旁（屏蔽 System32 里 HIP SDK 7.2 的旧 DLL）;
#   origami.dll 是 libhipblaslt.dll 的静态依赖（lld 的 "?" 报错不点名它）。
# - rocBLAS/hipBLASLt kernel db junction：rocBLAS 按 <exe目录>/rocblas/library
#   找 db（缺主索引 TensileLibrary.dat 时回退扫描 gfx1151/ 分片目录，目录必须
#   能枚举），TheRock 的 db 实际在 bin/rocblas/library —— mklink /J 免管理员。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
TR="${THEROCK:-/c/therock-dist-windows-gfx1151-10.1.0}"
HIPCC="$TR/bin/hipcc.exe"
[[ -x "$HIPCC" ]] || { echo "找不到 TheRock hipcc: $HIPCC（设 THEROCK=...）" >&2; exit 1; }
# hipcc 会读 HIP_PATH 定位 clang（实测指向坏路径直接编译失败；指向 HIP SDK 7.2
# 时靠布局差异侥幸回退自定位，不该依赖）。强制 HIP_PATH 指向 TheRock，屏蔽机器
# 上其他 HIP/ROCm 安装（SDK 6.4/7.1/7.2 会设 HIP_PATH/HIP_PATH_64/HIP_PATH_72）。
# ROCM_PATH 则必须不设：clang 看到它会去 $ROCM_PATH/amdgcn/bitcode 找设备库，
# 而 TheRock 的设备库在 lib/llvm/amdgcn/bitcode，靠 clang 自定位才找得到。
TR_WIN="$(cygpath -w "$TR")"
export HIP_PATH="$TR_WIN"
unset ROCM_PATH HIP_PATH_64 HIP_PATH_71 HIP_PATH_72
GPU_ARCH="${GPU_ARCH:-gfx1151}"
TARGET="${1:-all}"
[[ "$TARGET" == all || "$TARGET" == engine || "$TARGET" == bench || "$TARGET" == api || "$TARGET" == launcher || "$TARGET" == test ]] || { sed -n '2,11p' "$0" >&2; exit 2; }

mkdir -p build build/winlibs
[[ -f build/winlibs/rocblas.lib ]]   || cp "$TR/lib/rocblas.lib" build/winlibs/
[[ -f build/winlibs/amdhip64.lib ]]  || cp "$TR/lib/amdhip64.lib" build/winlibs/
[[ -f build/winlibs/hipblaslt.lib ]] || cp "$TR/lib/libhipblaslt.dll.a" build/winlibs/hipblaslt.lib

for d in amdhip64_7.dll rocm_kpack.dll amd_comgr.dll rocblas.dll libhipblaslt.dll origami.dll; do
  [[ -f "build/$d" ]] || cp "$TR/bin/$d" build/
done
# MSVC 运行时（微软官方可再分发），覆盖没装 VC++ Redistributable 的裸机
for d in msvcp140.dll vcruntime140.dll vcruntime140_1.dll; do
  [[ -f "build/$d" ]] || cp "/c/windows/System32/$d" build/
done

# kernel db 拷真身而非 junction：rocBLAS 按 <exe目录>/rocblas/library 找 db
# （缺主索引 TensileLibrary.dat 时回退枚举 gfx<arch>/ 分片目录，目录必须能
# 枚举）。只拷本机架构分片（rocblas 17M + hipblaslt 13M；全架构是 703M+530M）。
# 拷完 build/ 即自包含：目标机器无需安装 TheRock/ROCm 运行库，拷走即用。
for d in rocblas hipblaslt; do
  [[ ! -L "build/$d" ]] || rm "build/$d"  # 旧版 junction：删链接换真身
  if [[ ! -d "build/$d/library/$GPU_ARCH" ]]; then
    mkdir -p "build/$d/library"
    cp -r "$TR/bin/$d/library/$GPU_ARCH" "build/$d/library/" \
      || { echo "kernel db 拷贝失败：$TR/bin/$d/library/$GPU_ARCH" >&2; exit 1; }
    echo "[db] build/$d/library/$GPU_ARCH 已拷贝"
  fi
done
# rocBLAS 顶层另有按 arch 命名的 fallback 分片（KB 级），一并带上
for f in "$TR/bin/rocblas/library/"*"$GPU_ARCH"*; do
  [[ -f "$f" ]] || continue
  b="$(basename "$f")"
  [[ -f "build/rocblas/library/$b" ]] || cp "$f" build/rocblas/library/
done

# _CRT_NONSTDC_NO_DEPRECATE：屏蔽 MSVC 头文件对 strdup 等 POSIX 名的
# deprecated 标记（_CRT_NONSTDC_DEPRECATE → __declspec(deprecated)）。
FLAGS=(-O3 -std=c++17 --offload-arch="$GPU_ARCH" -D_CRT_SECURE_NO_WARNINGS
       -D_CRT_NONSTDC_NO_DEPRECATE
       -I"$TR/include" -Lbuild/winlibs -lrocblas -lhipblaslt)

build_engine() {
    echo "[编译] build/qwenox-engine-win.exe"
    # 先编到临时文件再原子替换，编译失败保留上次成功的二进制（对齐 Linux build.sh）
    "$HIPCC" "${FLAGS[@]}" src/gpu/qwenox.cpp -o build/qwenox-engine-win.exe.tmp \
      && mv -f build/qwenox-engine-win.exe.tmp build/qwenox-engine-win.exe
}

build_bench() {
    echo "[编译] build/qwenox-bench.exe"
    "$HIPCC" "${FLAGS[@]}" -std=c++17 -Ithird_party src/gpu/bench_main.cpp \
      -o build/qwenox-bench.exe.tmp \
      && mv -f build/qwenox-bench.exe.tmp build/qwenox-bench.exe
}

# 静态页嵌入：编译生成器并重新生成 src/api/static_gen.inc（复用调用方设好的 CXX）。
# 生成器输出是确定性的，内容未变不会重写文件；static_gen.inc 是生成物，
# 不入库（.gitignore），此处按需重新生成。
gen_static_inc() {
    [[ -x build/gen_static_inc.exe && ! tools/gen_static_inc.cpp -nt build/gen_static_inc.exe ]] || {
        echo "[编译] build/gen_static_inc.exe"
        "$CXX" -O2 -std=c++17 tools/gen_static_inc.cpp -o build/gen_static_inc.exe || exit 1
    }
    echo "[生成] src/api/static_gen.inc"
    ./build/gen_static_inc.exe src/api/static src/api/static_gen.inc || exit 1
}

# GUI 子系统链接选项（托盘/入口程序，双击不出控制台）；入口仍是 main()。
# 链接器选项用 - 前缀：Git Bash 会把 / 开头的参数当路径改写。
GUI_LDFLAGS=()
gui_ldflags() {
    [[ ${#GUI_LDFLAGS[@]} -gt 0 ]] && return
    if "$CXX" -dumpmachine | grep -q msvc; then
      GUI_LDFLAGS=(-Xlinker -subsystem:windows -Xlinker -entry:mainCRTStartup)
    else
      GUI_LDFLAGS=(-mwindows)
    fi
}

build_api() {
    # OpenAI HTTP 前端：纯主机 C++，用 TheRock 自带 clang++（不拖 HIP 依赖）。
    # vision.cpp 的图片解码在 Windows 上走 stb_image（vendor 单头文件，
    # 编译进 exe，零新增 DLL），支持 PNG/JPEG；WebP 明确报错。
    # Windows 上是 GUI 子系统托盘程序（默认无控制台，--console 恢复），
    # 取代旧 start_win.exe 启动器的生态位：托盘菜单/网页 #/engine 启停引擎。
    CXX="$TR/lib/llvm/bin/clang++.exe"
    [[ -x "$CXX" ]] || { echo "找不到 TheRock clang++: $CXX" >&2; exit 1; }
    gen_static_inc
    gui_ldflags
    echo "[编译] build/qwenox-win.exe"
    "$CXX" -O2 -std=c++17 -D_CRT_SECURE_NO_WARNINGS -Isrc/api -Ithird_party -I"$TR/include" \
      src/api/http.cpp src/api/engine_client.cpp src/api/tokenizer.cpp \
      src/api/chat_template.cpp src/api/json_py.cpp src/api/toolparse.cpp \
      src/api/vision.cpp src/api/reqstat.cpp src/api/reqstat_read.cpp \
      src/api/power.cpp src/api/engine_sup.cpp src/api/tray_win.cpp src/api/main.cpp \
      -lws2_32 -lshell32 -luser32 -lgdi32 -ladvapi32 "${GUI_LDFLAGS[@]}" -o build/qwenox-win.exe
}

build_launcher() {
    # 根目录最小入口 start_win.exe：只负责以项目根为工作目录拉起
    # build\qwenox-win.exe（托盘程序）。旧启动器（配置面板/双进程看守）
    # 已归档到 attic/launcher/，不再编译。
    CXX="$TR/lib/llvm/bin/clang++.exe"
    [[ -x "$CXX" ]] || { echo "找不到 TheRock clang++: $CXX" >&2; exit 1; }
    gui_ldflags
    RES=()
    if "$CXX" -dumpmachine | grep -q msvc; then
      # exe 文件图标（可选）：优先 TheRock 的 llvm-rc，没有则退回 Windows SDK
      # 的 rc.exe（取最新版本目录）；都没有也不影响 API 的托盘图标。
      RC="$TR/lib/llvm/bin/llvm-rc.exe"
      RCFLAGS=(-no-preprocess)
      if [[ ! -x "$RC" ]]; then
        RC="$(ls "/c/Program Files (x86)/Windows Kits/10/bin"/*/x64/rc.exe 2>/dev/null | sort -V | tail -1)"
        RCFLAGS=()
      fi
      if [[ -n "$RC" && -x "$RC" ]] && "$RC" "${RCFLAGS[@]}" -fo build/start_stub.res src/start_stub.rc; then
        RES=(build/start_stub.res)
      else
        echo "提示：llvm-rc 不可用，start_win.exe 文件不带图标（托盘图标不受影响）" >&2
      fi
    fi
    echo "[编译] start_win.exe"
    STUB_SRC=(-O2 -std=c++17 -D_CRT_SECURE_NO_WARNINGS src/start_stub_win.cpp
              -luser32 -lshell32 "${GUI_LDFLAGS[@]}" -o start_win.exe)
    if ! "$CXX" "${STUB_SRC[@]}" ${RES[@]+"${RES[@]}"}; then
      [[ ${#RES[@]} -gt 0 ]] || exit 1
      echo "提示：带图标资源链接失败，改为不带文件图标重试" >&2
      "$CXX" "${STUB_SRC[@]}"
    fi
}

build_test() {
    python tools/kv_admission_test.py --cxx "$TR/lib/llvm/bin/clang++.exe"
    echo "[编译] build/ktest-win.exe"
    "$HIPCC" "${FLAGS[@]}" -I src/gpu tools/ktest.cu -o build/ktest-win.exe
    echo "[运行] ktest-win（预期末尾 ALL PASS）"
    (cd build && HIP_PATH="$TR_WIN" ROCM_PATH="$TR_WIN" ./ktest-win.exe)
}

case "$TARGET" in
  all)
    build_engine
    build_bench
    build_api
    build_launcher
    ;;
  engine) build_engine ;;
  bench) build_bench ;;
  api) build_api ;;
  launcher) build_launcher ;;
  test) build_test ;;
esac
echo '[完成] 编译输出位于 build/；启动服务用 ./start_win.exe（或 bash start_win.sh）'

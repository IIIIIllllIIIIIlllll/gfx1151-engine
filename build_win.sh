#!/usr/bin/env bash
# Windows (TheRock) 编译入口：与 Linux build.sh 并列，产物输出到 build/。
#
# 用法:
#   bash build_win.sh            # 引擎 → build/gdec-win.exe
#   bash build_win.sh api        # OpenAI HTTP 前端 → build/gdec-api-win.exe
#   bash build_win.sh launcher   # 免脚本启动器 → ./start_win.exe（双击即用）
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
TR="${THEROCK:-/c/therock-dist-windows-multiarch-10.0.0/therock-dist-windows-multiarch-10.0.0}"
HIPCC="$TR/bin/hipcc.exe"
[[ -x "$HIPCC" ]] || { echo "找不到 TheRock hipcc: $HIPCC（设 THEROCK=...）" >&2; exit 1; }
GPU_ARCH="${GPU_ARCH:-gfx1151}"
TARGET="${1:-engine}"
[[ "$TARGET" == engine || "$TARGET" == api || "$TARGET" == launcher || "$TARGET" == test ]] || { sed -n '2,10p' "$0" >&2; exit 2; }

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
TR_WIN="$(cygpath -w "$TR")"
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

FLAGS=(-O3 -std=c++17 --offload-arch="$GPU_ARCH" -D_CRT_SECURE_NO_WARNINGS
       -I"$TR/include" -Lbuild/winlibs -lrocblas -lhipblaslt)

case "$TARGET" in
  engine)
    echo "[编译] build/gdec-win.exe"
    "$HIPCC" "${FLAGS[@]}" src/gpu/gdec.cpp -o build/gdec-win.exe
    ;;
  api)
    # OpenAI HTTP 前端：纯主机 C++，用 TheRock 自带 clang++（不拖 HIP 依赖）。
    # vision.cpp 的图片解码在 Windows 上走 stb_image（vendor 单头文件，
    # 编译进 exe，零新增 DLL），支持 PNG/JPEG；WebP 明确报错。
    CXX="$TR/lib/llvm/bin/clang++.exe"
    [[ -x "$CXX" ]] || { echo "找不到 TheRock clang++: $CXX" >&2; exit 1; }
    echo "[编译] build/gdec-api-win.exe"
    "$CXX" -O2 -std=c++17 -D_CRT_SECURE_NO_WARNINGS -Isrc/api -I"$TR/include" \
      src/api/http.cpp src/api/engine_client.cpp src/api/tokenizer.cpp \
      src/api/chat_template.cpp src/api/json_py.cpp src/api/toolparse.cpp \
      src/api/vision.cpp src/api/reqstat.cpp src/api/main.cpp -lws2_32 -o build/gdec-api-win.exe
    ;;
  launcher)
    # 免脚本启动器：原生 Win32，双击即用（不需要 Git Bash / PowerShell）。
    CXX="$TR/lib/llvm/bin/clang++.exe"
    [[ -x "$CXX" ]] || { echo "找不到 TheRock clang++: $CXX" >&2; exit 1; }
    echo "[编译] start_win.exe"
    "$CXX" -O2 -std=c++17 -D_CRT_SECURE_NO_WARNINGS \
      src/launch_win.cpp -lws2_32 -o start_win.exe
    ;;
  test)
    echo "[编译] build/ktest-win.exe"
    "$HIPCC" "${FLAGS[@]}" -I src/gpu tools/ktest.cu -o build/ktest-win.exe
    echo "[运行] ktest-win（预期末尾 ALL PASS）"
    (cd build && HIP_PATH="$TR_WIN" ROCM_PATH="$TR_WIN" ./ktest-win.exe)
    ;;
esac
echo '[完成] 编译输出位于 build/；启动服务用 ./start_win.exe（或 bash start_win.sh）'

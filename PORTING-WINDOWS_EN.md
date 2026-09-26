# Windows Porting Feasibility Analysis (2026-09-19)

*中文版:[PORTING-WINDOWS.md](PORTING-WINDOWS.md)*

> **Revision note**: This is the third version of this document. The first
> version's "two walls" conclusion was inferred from public sources; the
> second version was based on **measured data** from this machine (Windows +
> gfx1151, BIOS set to 96 GiB of video memory, 128 GiB of RAM). The probe
> source code is in `tools/winprobe/` and can be re-run.
> **Third version: the port is complete and verified**, see the next section.

## Port Results (wrapped up 2026-09-19, all measured)

**Port complete: the engine and the OpenAI-compatible API run at full
functionality on Windows + gfx1151.** The root-level `build_win.sh` builds
everything in one shot (four targets: `engine` / `api` / `launcher` / `test`,
parallel to the Linux `build.sh`), producing `build/gdec-win.exe` +
`build/gdec-api-win.exe` + `start_win.exe`; `start_win.exe` launches both
the engine and API processes in one step (a native Win32 launcher,
double-click to use, no Git Bash / PowerShell needed; clients connect to
port 8731 via the standard OpenAI interface). Under Git Bash, `start_win.sh`
is the counterpart of the Linux `start_hgn.sh` (Windows supports hgn weights only).

Verification matrix (all PASS):

| Verification item | Result |
|---|---|
| Kernel unit tests `ktest-win` (gdn/ple/index/convstate etc.) | `== ALL PASS ==` |
| Loader byte-level comparison `loadcmp` (64 random ranges incl. unaligned ones, NO_BUFFERING direct read vs mapped memcpy) | ALL MATCH |
| CLI smoke (68 GiB weights + MTP, `--tokens 1,2,3 --gen 8`) | correct output, prefill 15.9 tok/s, **decode 32.5 tok/s** (30.8 ms/token, same DRAM tier as Linux) |
| serve mode 256K (`--maxctx 262144` + vision tower) | arena estimate 91.03 GiB → max reservation 95.00 GiB with no overflow; `state allocated (maxctx=262144)`, graph capture, vision tower (333 tensors), `listening on :8730` all pass |
| socket protocol (PING/INFO/GEN, `tools/winprobe/serve_probe.py`) | GEN output token sequence matches the CLI **exactly** ([4 5 6 24218 10 4838 1665 15]) |
| OpenAI API full-stack acceptance (`tools/api_live_smoke.py` → `gdec-api-win` → engine :8730) | **RESULT PASS** (8/8: health / models / 400 parameter validation / non-streaming content / streaming vs stop equivalence / logprobs / responses non-streaming + SSE event sequence) |
| hipGraph | decode graphs captured (ple=1), works on Windows |
| hipBLASLt MoE path (GDEC_MOE_LT=1) | passed along with the smoke test |
| 32K prefill measurement (32768 random ids, `--tokens-file`, maxctx=40960) | chunk=8192: **~920 tok/s** (4 chunks in sequence: 913.9/936.9/922.3/907.7, no degradation); chunk=16384: **~979 tok/s** (987.6/970.4, arena 91.95 GiB with no truncation). **Doubling the chunk buys only ~6% PP** — GEMM efficiency at 8192 is already near saturation, so being locked at 8192 by the 95 GiB cap at 256K costs very little |

> **Update 2026-09-25 (faster hgn prefill; takes effect on Windows after a rebuild)**: hgn routed-expert
> prefill now runs a LUT-decode WMMA kernel (`27_kernels_moe_lut.inc`, default on, `GDEC_MOE_Q4W=0`
> disables it). Measured on Linux: pp8K @ chunk 2048 816 → 1215 tok/s, pp32K @ chunk 16384 1237 → 1422;
> KLD and decode are unchanged (see the "hgn 路由专家也走 WMMA" section in [GGUF.md](GGUF.md)). With it on,
> `GDEC_MOE_LT=1` has no effect and its dequant/gather buffers (~0.4 GiB at chunk 8192, ~0.8 GiB at 16384)
> are not allocated; the arena estimate shrinks accordingly. So Windows stays on hgn (~11 GiB less memory
> than GGUF).
>
> **Measured on Windows (2026-09-26, heretic.hgn, same env as the start_win.sh production
> config)**: the kernel works but the gain is markedly smaller than on Linux. 32K @ chunk 16384
> (maxctx 40960) A/B: warm chunk 1068.3 tok/s vs 1002.8 tok/s with `GDEC_MOE_Q4W=0` (layer time
> 13.42 s vs 14.60 s, ple phase unchanged) — only **+6.5%** overall, ~1.2 s saved per 16K chunk,
> about 60% of the Linux absolute gain (~1.7 s per 16K chunk). Presumably WDDM kernel-launch
> overhead eats part of it (the LUT path launches many small per-expert-group kernels); not yet
> confirmed with a kernel-level trace. 128K @ chunk 8192 (maxctx 139264): 16 chunks declining
> gently from 1015.8 to 938.4 tok/s, **~1000 tok/s average**, no cliff; the per-chunk decline
> comes from PLE on-demand reads growing with prefix length (ple_wait rises chunk over chunk
> in prof), which is expected.
>
> **Update 2026-09-26 (high-quality hgn)**: the new 8-bit dense overlay + imatrix routed-expert
> base (same hgn format, see [HGN-HQ.md](HGN-HQ.md)) brings KLD down from 0.163 to GGUF levels;
> weight arena +2.2 GiB, estimated ~93.2 GiB at 256K / chunk 8192 (cap 95). Only the files need
> swapping (`MODEL_FILE` / `OVERLAY_FILE`). Not yet measured on Windows.

Key implementation points (differences relative to the Linux version):

- **Single-arena memory architecture**: `g_devarena` + `dalloc_arena` (bump
  allocation, falls back to hipMalloc on overflow with a one-time warning);
  all large device allocations (weights/KV/indices/prefill workspace) go
  into the process's very first allocation. `devarena_estimate` mirrors all
  of GpuModel's allocation formulas; reservation = estimate + slack
  (`GDEC_ARENA_SLACK_GB`, default 4 GiB), capped at `GDEC_ARENA_CAP_GB`
  (default **95.0**, matching the measured upper bound of the 96 GiB BIOS
  split). Windows defaults to prefill chunk 8192 (`eff_maxbatch`; Linux
  32768). **Do not set `GDEC_PREFILL_CHUNK=16384` at 256K** (it triggers
  truncation + overflow fallback); with maxctx ≤ 40K, 16384 fits and PP is
  ~6% faster (see the measurements in the verification matrix).
- **New weight loader** (`load_arena` Windows branch): CreateFile
  (OVERLAPPED|NO_BUFFERING) + 4 threads × 4 slots × 16 MiB pinned +
  overlapped ReadFile direct reads + H2D direct upload; expert weights also
  go into the arena. Measured **68 GiB in ~9 minutes, steady ~130 MB/s**
  (this machine's NVMe is weak, disk-bound); the old mmap+memcpy path took
  54 minutes on Windows due to page-cache eviction and degraded to ~18 MB/s
  near the end.
- **rocBLAS kernel db**: TheRock does not ship the main index
  `TensileLibrary.dat`, so rocBLAS falls back to enumerating the per-arch
  directory `<exe_dir>/rocblas/library/gfx1151/` — that directory must be
  enumerable before the process starts. The missing-index "Cannot read
  TensileLibrary.dat" is only a warning. **The final form is copying the
  real files rather than a junction**: only the gfx1151 slices are copied
  (rocblas 17M + hipblaslt 13M, vs 703M+530M for all architectures);
  `build_win.sh` does this idempotently. After copying, build/ is
  self-contained and TheRock can be deleted (verified by renaming it to
  sever the dependency).
- **PLE on-disk on-demand reads**: io_uring → IOCP (PleWin, DEPTH=512,
  free-slot queue); prefill batched reads go through the `PLE_PUMP` macro,
  so the main code has the same shape on both platforms.
- **socket/serve**: WinSock2 shim (`sock_*`/`os_*`), zero protocol changes.
- **API frontend** (`src/api` → `gdec-api-win.exe`): the socket layers of
  http.cpp / engine_client.cpp use the same `os_win32.h` shim (`sock_t`
  type, WSAPoll, `SO_RCVTIMEO` as DWORD milliseconds); compiled with
  TheRock's bare clang++ (no hipcc needed, zero HIP dependencies), linking
  only the system `ws2_32`; nlohmann/json is provided by TheRock includes.
  Beware the WinSock trap: the `peer_gone` probe must grab
  `WSAGetLastError()` **immediately** after recv fails and only then do the
  `ioctlsocket` to restore blocking — a successful call overwrites
  last-error, which was misjudged as "client disconnected" and caused
  streaming requests to drop within seconds (fixed). Image decoding uses
  stb_image (public-domain single header, `src/api/vendor/`) supporting
  PNG/JPEG, compiled into the exe with zero new DLLs; end-to-end PNG image
  Q&A verified. WebP has no equivalent single-header implementation and
  reports an explicit error — the only remaining gap.
- **origami.dll** is a static dependency of `libhipblaslt.dll` and must ship
  alongside the exe (the lld link error doesn't name it, so it's easy to
  miss).

Outstanding items: the API frontend has been ported (`gdec-api-win.exe`,
with image decoding via stb_image supporting PNG/JPEG and verified; only
WebP reports an explicit unsupported error); a fully-filled 256K prefill has
not been measured (the memory layout is verified; loading a full 256K tokens
on this machine would take hours and is of limited value); long-run
stability (WDDM paging, switches like `HSA_ENABLE_SDMA=0`) remains to be
observed.

## Conclusion (second version)

**The port is feasible**. Passages have been found through both walls of the
original judgment:

| Originally presumed wall | Measured result |
|---|---|
| gfx1151 has no Windows ROCm support | **Not true**. Both HIP SDK 7.2 and TheRock 10.0 support it natively: `hipInfo` enumerates the 8060S (gfx1151, 20 CU, 107.87 GiB visible memory), and kernels compile and run normally |
| WDDM does not allow 65 GiB pinned direct reads | **Bypassed**. After setting 96 GiB of video memory in the BIOS, a single large `hipMalloc` (measured up to 88 GiB) lands entirely in real VRAM at ~235 GB/s bandwidth (on par with Linux direct reads at ~220 GB/s), and data-integrity checks pass. The `hipHostRegister` architecture is no longer needed |

Port shape: all weights go into a **single device arena** (the first
allocation at startup); PLE tables stay on disk, gathered on demand via
Windows file IO. Expected performance is on par with the Linux version (same
DRAM, measured bandwidth identical).

## Measurement Log

Environment: HIP SDK 7.2 (`C:\Program Files\AMD\ROCm\7.2`) +
TheRock 10.0 (`C:\therock-dist-windows-multiarch-10.0.0`),
Adrenalin driver 32.0.22018.5 (2025-09-17). The host also has an RTX 2080 Ti
(not involved in HIP enumeration).

### Toolchain and Libraries (all ready)

- Compilation: both toolchains compile for gfx1151. A trial build of
  `gdec.cpp` with the SDK's hipcc showed that **the only blocking error is
  the missing `sys/mman.h`** (with `-std=c++17` added, rocPRIM and
  everything else passes); the warp intrinsics / inline assembly of the 127
  kernels are ISA-level and compile as-is.
- Runtime libraries: rocBLAS SGEMM measured 3.2 TFLOPS with correct results;
  hipBLASLt bf16 GEMM works (heuristic hits). Both, plus the rocPRIM
  headers, exist in both SDK 7.2 and TheRock; TheRock ships the gfx1151
  Tensile kernel db (`bin/rocblas`, `bin/hipblaslt`).
- Probe: `tools/winprobe/blassmoke.cpp`.

### Memory Allocation (the crux of the port; rules now established)

HIP allocations on Windows follow clear routing rules (consistent with the
upstream issue
[ROCm#5940](https://github.com/ROCm/legacy-rocm-build/issues/5940)):

1. **While the process's cumulative VRAM allocation is < 32 GiB, allocations
   land in VRAM**; past that, new allocations are routed to shared memory
   (host RAM, of which this machine has only ~21 GiB free), and pushing on
   from there gets you paged or failed.
2. **A single large allocation is routed as a whole**: making the very first
   allocation large → all of it lands in VRAM. Measured (TheRock runtime,
   `tools/winprobe/bigalloc.cpp`):

   | Single hipMalloc | Result | Read bandwidth | Integrity |
   |---|---|---|---|
   | 40–95 GiB | OK | 223–237 GB/s | Pass |
   | 96 GiB | out of memory | — | — |

   So the per-process usable ceiling is **~95 GiB** — nearly the full 96 GiB
   split is usable (the driver keeps only ~1 GiB).
3. Counter-example (an allocation pattern that must be avoided): consecutive
   4 GiB chunks go out of bounds after ~44 GiB — the SDK 7.2 runtime
   **silently corrupts data** (read-back after write is damaged;
   `integrity.cpp` measured 30 corrupted blocks), while the TheRock runtime
   raises a loud fault. **Neither runtime tolerates this pattern; "one large
   block at startup" is the only safe path**. Two concurrent processes each
   holding 40 GiB pass integrity checks (`holdprobe.cpp`), proving the limit
   is per-process routing, not a global shortage.
4. After `hipFree` within a process, the budget is not fully reclaimed
   (fragmentation); the engine allocates once for everything anyway, so this
   is harmless.

### Memory Budget (heretic.hgn, 115.5 GiB file)

| Content | Size | Where it goes on Windows |
|---|---|---|
| Expert weights `.mlp.experts.*` | ~65 GiB | arena (memcpy in at startup, replacing hipHostRegister) |
| Backbone weights | ~2.8 GiB | arena (same as Linux) |
| KV cache + QSA index @256K (bf16 KV) | ~9.4 GiB | arena |
| prefill workspace (scales with `GDEC_PREFILL_CHUNK`) | 27.6 / 13.8 / 6.9 GiB @chunk 32768/16384/8192 | arena |
| MTP sidecar + vision tower + misc | ~1.5 GiB | arena |
| **arena total @256K** | **106 / 92.6 / 85.7 GiB** | ceiling ~95 GiB: chunk 16384 is tight (2.4 GiB headroom), **8192 is comfortable (~9 GiB headroom)** |
| PLE tables (dtype 10) | 47.7 GiB | **stay on disk**, read on demand 16×160 B per token (same architecture as Linux; io_uring replaced by ReadFile/overlapped IO) |

**Key constraint: all large device memory must fit into the single arena
that is "the process's first allocation"** (see the threshold rule above: by
the time of a second large allocation the cumulative total is already >32
GiB and it gets routed to shared memory). Implementation-wise, it suffices
to extend `load_arena`'s "sum first, single hipMalloc, sub-allocate
internally" pattern to the KV/index/prefill workspace; the rocBLAS/Lt handle
workspaces (~64 MiB scale) are created after the arena and land in shared
memory, which is small enough to be harmless.

The 256K context itself (bf16 KV 6.4 GiB + index ~3 GiB) is not the
bottleneck; the prefill workspace is. The Linux default chunk=32768 was
chosen for 122 GiB of system memory; on Windows use
`GDEC_PREFILL_CHUNK=8192` (this switch is already built into the engine;
8192 is still a large enough GEMM, so prefill efficiency is essentially
unharmed). With `MAX_CONTEXT=131072`, the KV is halved and chunk 16384 also
fits comfortably.

PLE does only 2.5 KB of random reads per token (row addresses are a hash
modulo of the token history — **no sequential locality**, `gdec.cpp:8266`);
the Windows page cache absorbs hot rows, and cold rows hit the NVMe. Decode
reads 16 rows per token ≈ worst case +1 ms (period 18–33 ms), harmless;
**prefill does 131K random 4K reads per chunk (512 MB equivalent), which
must be submitted concurrently in batches via overlapped IO (QD≥32,
~0.3–0.5 s/chunk); per-row synchronous ReadFile is forbidden (QD1 needs 5+
s/chunk)**. This is the correct Windows equivalent of the io_uring path.

## Code Porting Checklist (reassessed after the measurements)

| Item | Scale | Approach |
|---|---|---|
| `sys/mman.h`, `mmap`/`madvise`/`munmap` | ~20 sites | Map weights with `MapViewOfFile` or stream them directly into the arena; drop the `madvise(WILLNEED)` prefetch (replaced by the PLE on-demand read path); KV snapshots use plain `fread`/`fwrite` |
| io_uring PLE gathering | ~150 lines | `ReadFile` + `OVERLAPPED` batched offset reads, landing in pinned staging (byte-for-byte semantics identical to the Linux path) |
| Expert-weight `hipHostRegister` direct reads | one branch in `load_arena` | On Windows, experts go into the arena like ordinary tensors (i.e. revert to the old all-in-arena form, Windows only) |
| POSIX sockets (serve mode + `src/api/http.cpp` etc.) | ~10 sites | WinSock2 (similar API, mechanical replacement) |
| `clock_gettime`/`localtime_r`/`unistd.h` misc | a few | standard replacements |
| `build.sh` | single script | `build.bat`/CMake: TheRock hipcc + `-O3 -std=c++17 --offload-arch=gfx1151`, linking `rocblas.lib` + `libhipblaslt.dll.a` (note: the hipcc wrapper rejects `-l:` syntax and bare `.lib` paths; the file must be renamed to `hipblaslt.lib`) |
| API frontend image libraries | libpng/jpeg/webp | vcpkg or prebuilt; decoupled from the engine, can be deferred |
| `tools/run_capped.sh` | — | no port needed (a local deployment tool) |

## Deployment Form (verified, self-contained)

**Copy build/ and run it as-is; the target machine needs no ROCm/TheRock
runtime installed** (verified by severing the dependency: with the TheRock
directory renamed and HIP_PATH/ROCM_PATH cleared, ktest still reports ALL
PASS). All runtime dependencies sit next to the exe:

- TheRock DLL ×6: `amdhip64_7.dll`, `rocm_kpack.dll`, `amd_comgr.dll`,
  `rocblas.dll`, `libhipblaslt.dll`, `origami.dll` (the exe directory takes
  precedence over System32 — **the old SDK 7.2 DLLs in System32 must be
  masked out**; the old DLLs cap a single allocation at 41 GiB and silently
  corrupt data beyond it). ldd confirms no other hidden dependencies.
- MSVC runtime ×3: `msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`
  (officially redistributable; covers bare machines without the VC++
  Redistributable).
- Kernel db real files: `rocblas/library/gfx1151/` (17M) +
  `hipblaslt/library/gfx1151/` (13M), resolved relative to the exe and
  independent of cwd.
- `gdec-api-win.exe` (OpenAI API frontend) has zero extra dependencies:
  links only the system `ws2_32`, copies along with build/ and runs; image
  decoding not wired in (text is fully functional).
- No device bitcode is needed at runtime (no JIT: engine kernels are
  embedded at compile time; rocBLAS/Lt load hsaco/.co directly), so neither
  `HIP_PATH` nor `ROCM_PATH` needs to be set.
- It is recommended to add an arena self-check at engine startup
  (write-read pattern verification) as a one-vote veto against driver
  regressions.

## Risks and Outstanding Items

1. **Allocation-order sensitivity**: the arena must be the process's first
   large allocation; afterwards there must be no more large device
   allocations (past the >32 GiB threshold they get routed to shared
   memory). Keep this in mind when auditing paths such as KV snapshot
   rebuilds.
2. Slower startup: the 68 GiB of weights must be genuinely read from disk
   into the arena (NVMe-bound, estimated in minutes), whereas the Linux
   version's mmap+register is nearly instant.
3. Long-run stability untested: WDDM paging behavior when host memory is
   tight (only 32 GiB), and the community-suggested stability switches
   `HSA_ENABLE_SDMA=0`/`HSA_USE_SVM=0` remain to be validated.
4. Keeping the 96 GiB BIOS split as-is is fine; no VGM adjustment is needed
   (the single large-allocation path does not pass through its limits). The
   2080 Ti does not affect HIP device enumeration.
5. ~~API frontend untouched~~ ported (see the section above); the only
   remaining functional gap is WebP decoding, and the systemd tooling does
   not apply to Windows.

## Next Steps (all done, kept for reference)

1. ~~Write Windows compatibility shims (mman/socket/unistd) + arena-ize
   `load_arena`, get `gdec.cpp` to compile.~~ ✅ `src/gpu/os_win32.h` +
   single-arena architecture
2. ~~Load heretic.hgn for a smoke test: short CLI generation, compare against
   the Linux baseline output.~~ ✅ correct output, decode 32.5 tok/s
3. ~~Windows batched-read path for PLE + performance comparison
   (prefill/decode tok/s).~~ ✅ the IOCP path passed along with the
   smoke/serve tests
4. ~~Kernel unit tests (`tools/ktest.cu`) all green on Windows.~~ ✅ ALL PASS
5. ~~API frontend and service-ization wrap-up.~~ ✅ `gdec-api-win.exe` +
   `start_win.sh` dual processes, `api_live_smoke.py` 8/8 PASS, multimodal
   (PNG image Q&A) verified end-to-end; remaining optional item: WebP
   decoding (needs a prebuilt libwebp)

The actual work concentrated on the arena architecture rework and the new
loader (NO_BUFFERING direct reads), consistent with the "person-week scale"
estimate; all runtime risk items have been ruled out or circumvented by
measurement.

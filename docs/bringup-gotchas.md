# Runbook: bringing up llama.cpp on a new Android phone without stepping on the same rakes

Reference device: OPPO Reno7 5G CN (`PFJM10`), Snapdragon 778G, Adreno 642L, Android 13, Termux (sshd on
port 8022), model `qwen2.5-3b-instruct-q4_k_m.gguf`.
This is a concentrated list of **every** failure we hit, with the exact symptom and the cure.
Read it before running anything.

---

## 0. Fastest path (if all you want is a working LLM)

CPU-only, ~15 minutes, no patches:

```bash
pkg update && pkg install -y clang cmake ninja git python
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j4
./build/bin/llama-bench -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf -ngl 0 -p 256 -n 64 -r 3 -t 4
```

On Snapdragon 778G this gives **pp256 ≈ 29 t/s, tg64 ≈ 7.5 t/s** — and that is the best result obtainable
on this device at all. **The OpenCL path costs 2–3 extra hours and ends up slower than the CPU** (§3).
If your objective function is speed, don't touch OpenCL.

---

## 1. Termux infrastructure

| Symptom | Cause | Cure |
|---|---|---|
| `bash: /tmp/xxx: No such file or directory` | Termux has **no writable** `/tmp` | use `$TMPDIR` (always set) or `$HOME/tmp`: `D=${TMPDIR:-$HOME/tmp}; mkdir -p "$D"` |
| `gzip: stdin: unexpected end of file` / `tar` dies while unpacking | Termux's system `gzip`/`tar` are broken on multi-threaded archives | unpack with python: `python -c "import tarfile;tarfile.open('x.tar.gz').extractall('.')"` |
| `dlopen("/vendor/lib64/libOpenCL.so") failed: library ... not accessible for namespace` | Android 12+ linker namespaces hide vendor libs from app processes | copy the needed `.so` files to `~/ocl` and set rpath (§3.1) |
| `run-as: package not debuggable` | OPPO firmware is not debuggable | don't use `run-as`; drive the phone via `adb shell am start` + `input` |
| sshd gone, SSH won't connect | OOM killer took `sshd` (usually when two llama processes run at once) | revive over ADB: `adb shell am start -n com.termux/.HomeActivity`, then `input keyevent 111`, `input text "sshd"`, `input keyevent 66` |
| `ssh` drops mid-benchmark | phone slept / Doze | keep the screen on (`adb shell svc power stayon true`), run long jobs under `timeout` with logging to a file |

Rules for working with the phone:

- **One heavy process at a time.** Two llama processes (even `llama-bench` + `llama-cli`) → OOM → sshd is
  killed and the session has to be revived over ADB.
- Long runs: `timeout 900 cmd > log 2>&1` in the background, then read the log. Don't hold an interactive
  SSH channel for hours.
- State checks: `ps -A | grep llama`, `free -m`, `cat /sys/class/thermal/thermal_zone0/temp`.

A note on Android's phantom-process killer: on MIUI/ColorOS-class firmware the shell user gets processes
reaped mid-run. `adb shell device_config put activity_manager max_phantom_processes 1024` and
`adb shell svc power stayon true` (or `stay_on_while_plugged_in=7`) make long benchmarks reliable.
Both are ADB-settable without root and both are reversible.

---

## 2. Building llama.cpp

Working parameters (verified):

```bash
# CPU
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j4          # ~6-8 min on 778G

# OpenCL
cmake -B build-cl -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENCL=ON \
  -DOpenCL_LIBRARY=$HOME/ocl/libOpenCL.so \
  -DOpenCL_INCLUDE_DIR=$HOME/opencl-headers \
  -DGGML_OPENCL_TARGET_VERSION=200 \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
  -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build-cl -j4       # ~8-10 min
```

Rakes:

- `-j8` on 8 GB RAM → OOM during link. `-j4` only.
- `LLAMA_CURL=OFF` is mandatory, otherwise cmake pulls curl and fails or spends forever looking for it.
- `GGML_NATIVE=ON` is mandatory for dotprod/i8mm detection; without it tg drops 2–3×.
- OpenCL headers: Termux has **no** package shipping `CL/cl.h` for 2.0 — keep your own copy in
  `~/opencl-headers` (header-only Khronos; `cl2.hpp` is not needed, only `CL/*.h`).
- Rebuilding one file: `cmake --build build-cl -j4` after editing `ggml-opencl.cpp` ≈ 2–3 min (not 8).
- The **OpenCL program cache** (`$PREFIX/tmp/llama.cpp/cl-cache`) hides kernel recompilation: after any
  change to a `.cl` file or to `ggml-opencl.cpp`, `rm -rf $PREFIX/tmp/llama.cpp/cl-cache`. Failed kernel
  builds do not enter the cache; successful ones do.
- On Termux, `std::aligned_alloc` may be missing depending on the target API level (bionic). If the build
  fails there, replace it with `posix_memalign` locally.

---

## 3. OpenCL on Adreno: the headache (skip if you don't need the GPU)

### 3.1. Curing the `dlopen` namespace problem

```bash
mkdir -p ~/ocl && cd ~/ocl
for f in libOpenCL.so libCB.so libgsl.so libadreno_utils.so libllvm-glnext.so; do
  cp /vendor/lib64/$f . 2>/dev/null
done
for f in *.so; do patchelf --set-rpath '$ORIGIN' $f; done
# run ONLY like this:
LD_LIBRARY_PATH=$HOME/ocl ./build-cl/bin/llama-bench ...
```

Without `LD_LIBRARY_PATH=$HOME/ocl` you get the namespace error again. This applies to `llama-cli` and to
every script.

### 3.2. What `clinfo` reported (for reference — re-check on another phone!)

```
OpenCL 2.0 / OpenCL C 2.0, Adreno 642L
max_compute_units = 2          <-- very few
max_work_group_size = 1024
global_mem_size = 3648 MB
max_mem_alloc_size = 912 MB    <-- key limit
cl_khr_fp16, cl_khr_subgroups, cl_qcom_subgroup_shuffle, cl_qcom_reqd_sub_group_size
NO cl_khr_il_program
```

No `cl_khr_il_program` means source-only compilation — every kernel is compiled from OpenCL C source at
runtime by the vendor compiler, which is where most of the pain below comes from.

### 3.3. Patches without which nothing works

All three are in `scripts/` and apply to `~/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp`.
Order matters. Each script makes its own backup (`.preps`, `.pre-enq`, `.pre-supports`).

| Patch | What it fixes | Without it |
|---|---|---|
| `patch_rms.py` | a failed kernel load becomes **non-fatal** (flag `g_cl_loading_kernels`), and the `rms_norm` program is built from the plain kernel only | any kernel that fails to compile kills the process |
| `patch_kernels2.py` | `first_kernel_only()` — truncates multi-kernel `.cl` sources to the first kernel; fusions are enabled only when both kernels exist | the vendor compiler dies with SIGSEGV (§3.4) |
| `patch_enq2.py` | on `CL_INVALID_WORK_GROUP_SIZE`, retry the enqueue with `NULL` local size instead of aborting | the process dies on `-54` with no diagnostics |
| manual: `supports_op` gate | the `GGML_OP_MUL_MAT` F16 branch requires `kernel_mul_mat_f16_f32_l4 && ..._mrow` | `exit=134` via null kernel (§3.5) |
| manual: `GGML_OPENCL_LMHEAD_CPU` | keep `lm_head` on the CPU | `lm_head` on the GPU is the slowest link |

The last two were applied by hand on the reference build, so the scripts don't ship them; both are a handful
of lines and are quoted in §3.5 and §3.9. Treat them as a recipe, not as a diff.

Key point: **all of these patches work around vendor driver bugs, not upstream bugs.** On a different phone
(different Adreno/driver) they may be unnecessary or actively harmful — always try a stock build first.

### 3.4. SIGSEGV inside the vendor compiler

Symptom:

```
QGPUPeepholeOptimizer::SplitMBBAt  ... SIGSEGV (rc=-11)
```

Cause: the Adreno compiler crashes on programs that contain **several kernels** in one `.cl` source
(typically `rms_norm`, `mean`, `sum_rows`, `ssm_scan`, `mul_mv_f16_f32_l4`).
Cure: `first_kernel_only()` (`scripts/patch_kernels2.py`) — truncate the source at the first `__kernel`.
After that `mean`/`sum_rows` load with rc=0, while `ssm_scan`/`mul_mv_f16_f32_l4` still crash with rc=-11,
so those kernels simply don't exist — which is handled by the `supports_op` gate.

### 3.5. `exit=134` (SIGABRT) via null kernel

Symptom: `Aborted (core dumped)`, `exit=134`, and nearby in the log a series of
`clCreateKernel(...) error -44` for `mul_mv_f16_f32_l4`/`_mrow`, then `GGML_ABORT("OpenCL enqueue failed")`
or `std::out_of_range: map::at`.

Cause: the kernel didn't compile (`clCreateKernel` → NULL) but `supports_op` still says yes, so the backend
writes into a null kernel and aborts.

Cure: in `ggml_opencl_supports_op`, `GGML_OP_MUL_MAT` branch for F16:

```cpp
if (!(backend_ctx->kernel_mul_mat_f16_f32_l4 && backend_ctx->kernel_mul_mat_f16_f32_mrow)) return false;
```

Consequence: f16 matrix multiplications (attention) run on the CPU — that is the expected state on this
driver.

### 3.6. Flash-attention does not compile

```
ggml_opencl: kernel compile error (err=-6) building fa f32_f16:
ggml_opencl: kernel compile error (err=-6) building fa f32_f16 split:
```

With `-fa auto` (the default!) and `-ngl > 0` the run then dies with:

```
libc++abi: terminating due to uncaught exception of type St12out_of_range: map::at:  key not found
Aborted (core dumped)   # exit=134
```

Cure: **always `-fa off` when `-ngl > 0`.** With `-ngl 0`, `-fa auto` is fine (verified: pp16 28.96 /
tg4 7.67). `-fa auto` with GPU layers = guaranteed 134.

### 3.7. `CL_INVALID_WORK_GROUP_SIZE` and garbage text

`kernel_gemv_noshuffle_q6_K_f32_o4_global` requires **exactly 4 subgroups** (lws `{64,4,1}`):

```
clEnqueueNDRangeKernel ... error -54 (CL_INVALID_WORK_GROUP_SIZE)
```

The retry with `NULL` local size formally succeeds but produces **garbage text** ("powered", "秆", "urse").
Cure: `GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=0` (keeps the image path `_o4`, also with `NSUBGROUPS=4`, but with a
valid lws). Correct output: `The capital of France is Paris.`

Working env knobs (all three needed for the correct and fastest GPU mode):

```bash
export GGML_OPENCL_Q6K_GEMV_O4=1          # image path for q6_K GEMV
export GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=0   # the __global variant gives -54 / garbage
export GGML_OPENCL_LMHEAD_CPU=1           # lm_head on CPU
```

`GGML_OPENCL_Q6K_GEMV_O4` and `..._O4_GLOBAL` are reads by our patched build: the `GGML_OP_MUL_MAT`
q6_K branch selects between the `_o4` image kernel and the `_o4_global` buffer kernel based on these
variables. If your checkout does not read them, add the `getenv` checks where the kernel is chosen.

### 3.8. Noise you can ignore

```
clReleaseKernel ... error -30
clReleaseProgram ... error -44
```

and the `error -44` streaks from `clCreateKernel` for `mul_mv_f16_f32_mrow*`/`_l4` are expected (those
programs never built) and don't affect results. `error -46` is the same story.

### 3.9. Other knobs

- `GGML_OPENCL_DISABLE_FUSION=1` — disable fusions (use if fused kernels produce artefacts).
- `GGML_OPENCL_OPFILTER` — regex to run only selected ops (useful for isolating a guilty kernel).
- `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF` — **untested**; might fix FA/f16 attention compilation (generic
  kernels are simpler); costs ~10 min of build time. It is the only obvious route to a better GPU pp.

---

## 4. CLI/bench flags that don't exist or lie

| Attempt | Result | Correct form |
|---|---|---|
| `llama-cli -no-cnv` | `invalid argument` | `-st, --single-turn` + `--no-warmup` |
| `llama-bench -fa 0` | `invalid argument` | `-fa off` (accepts `on/off/auto`) |
| `llama-bench -dev CPU` | `error: invalid device: CPU` (`--list-devices` shows only `GPUOpenCL: QUALCOMM Adreno...`) | CPU mode is `-ngl 0`, not `-dev` |
| `llama-bench -fitt 256` | **`rc=134`** (abort in the fitting path) | don't use `-fitt` at all |
| `-ngl` omitted | default `-1` = **auto-offload**: on a phone with OpenCL the model goes to the GPU by itself | always pass `-ngl 0` explicitly for a CPU baseline |
| `-t` omitted | default 8 → tg drops ~40 % (A55 cores + migrations) | pass `-t 4` explicitly |
| `llama-bench` without `-r` | default 5 repetitions — slow; `±` is near zero anyway | `-r 2..3` for screening, `-r 5` for the final number |
| progress spinner hides INFO logs | you can't see why it failed | redirect to a file (`> log 2>&1`) or `-v`, then `tr "\r" "\n" < log`; `llama-bench` has no `-lv` |

`llama-bench --help` in this version (0.7.0-dev, commit `19ab5a5`) has `-fitt/-fitc/-lm/-lzm`, `-dev` with
slash-separated devices (`dev0/dev1`), and `--list-devices`.

---

## 5. Memory and OOM

- `MemTotal` on Reno7 5G: 7.47 GB, swap ~7.8 GB. Model 1.95 GiB + KV cache + OpenCL buffers → little slack.
- **One heavy process.** A second llama process = OOM kill, and the usual victim is sshd because it's small.
- Before a run: `free -m`, `ps -A | grep -i llama`. After a failure, check for zombies.
- `max_mem_alloc_size = 912 MB` does **not** mean more than ~14 layers fail to fit: llama.cpp splits
  weights across several buffers, and `-ngl 37` (all layers) technically works. A single `clCreateBuffer`
  above 1024 MB is impossible (`-61`); 512 MB is fine.
- `-nkvo 1` and `-fa off` are safe. `-fitt` is not.

---

## 6. Vulkan: a dead end on this driver (don't spend time on it)

```
ggml_vulkan: ... SPIR-V 1.5 requires StorageBuffer8BitAccess
ErrorUnknown
```

The Qualcomm Vulkan 1.1.128 driver lacks `VK_KHR_8bit_storage`, while ggml-vulkan emits SPIR-V 1.5 with
8-bit storage. There is no workaround without patching shaders or rebuilding glslc with a different target.
The `build-vk/` directory remains as a monument.

---

## 7. Measurement methodology (so the numbers are comparable)

Canonical run:

```bash
llama-bench -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -ngl {0,8,14,20,37} -fa off -p 256 -n 64 -r 2..3 -t 4
```

- Same binary for CPU and GPU (`build-cl/bin/llama-bench` with `-ngl 0` matches the CPU-only `build/`:
  28.8/7.45 vs 28.98/7.57 — ~1 % apart).
- Several rounds **interleaved** (CPU → GPU → CPU → GPU) so thermal drift doesn't land on one variant.
- Temperature during runs 44–48 °C; the A78 cores throttle, so absolute numbers depend on device state.
- **Thermal throttling was measured:** at 45–47 °C `scaling_max_freq` drops from 2 208 000 to 1 862 400
  (cpu7) and from 2 131 200 to 1 651 200 (cpu4); on a cold device it sits exactly at the maximum. Readable
  from the shell: `cat /sys/devices/system/cpu/cpu{4,7}/cpufreq/scaling_max_freq` (the file belongs to
  `system`, so writing needs root).
- Hence pp depends strongly on temperature and tg does not: cold (36–41 °C) pp256 **40.47 ± 0.14** /
  tg128 7.53; hot **28.9 ± 0.1** / 7.4. ~29 % on pp, ~2 % on tg. Never take a single run as truth; compare
  everything in the same thermal state.
- Within-run spread `±`: 0.01–0.09 t/s; between runs up to 2 % (except that boost case).
- For a report, record: `-t`, `-ngl`, `-fa`, `-p`, `-n`, `-r`, temperature, commit.

Result on Reno7 (medians): CPU `-t 4` **28.8 / 7.45**; GPU `-ngl 8` 25.5/5.25; `-ngl 14` 22.7/4.14;
`-ngl 20` 20.7/3.32; `-ngl 37` 17.4/2.24. **The GPU is monotonically worse** → conclusion: CPU-only.
Full report: [`cpu-vs-gpu.md`](cpu-vs-gpu.md).

---

## 8. Checklist for a NEW phone

1. `nproc`, `cat /proc/cpuinfo | grep -E "Features|model name"` — dotprod/i8mm present? Without them
   `GGML_NATIVE` won't save you.
2. `free -m` — enough for 3B Q4_K (~2 GB) plus KV cache?
3. **CPU-only first** (§0), and take a baseline at `-t 1,2,4,6,8` — the optimum may not be 4.
4. Only then, if you really need the GPU: `~/ocl` + `LD_LIBRARY_PATH`, `--list-devices`, `clinfo`.
5. On a new Adreno, check `max_compute_units`, `max_mem_alloc_size`, and whether `cl_khr_fp16`,
   `cl_khr_subgroups`, `cl_qcom_reqd_sub_group_size` exist, plus the driver version. That decides which of
   the §3.3 patches you need.
6. Stock OpenCL build → if it crashes: `patch_rms.py` → `patch_kernels2.py` → `patch_enq2.py` →
   `supports_op` gate → env knobs.
7. `-fa off` whenever `-ngl > 0`; `-ngl 0` explicitly for CPU.
8. Sweep `-ngl {0,4,8,14,20,99}` and look at **monotonicity**: if more layers means slower, the GPU is
   pointless on that SoC — stop and don't spend hours on kernels.
9. One process at a time; logs to files; `timeout` on everything.

---

## 9. Diagnostics one-liners

```bash
cat /sys/class/thermal/thermal_zone0/temp                 # temperature (milli-°C)
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do echo "$c $(cat $c)"; done
free -m; ps -A | grep -i llama
rm -rf $PREFIX/tmp/llama.cpp/cl-cache                     # after changing kernels
LD_LIBRARY_PATH=$HOME/ocl ./build-cl/bin/llama-bench --list-devices
```

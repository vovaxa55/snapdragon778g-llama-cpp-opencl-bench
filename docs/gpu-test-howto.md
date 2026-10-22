# How to test GPU offload on an Android phone, step by step

Companion to [`bringup-gotchas.md`](bringup-gotchas.md). This is the order of operations that actually
works; each step has a pass/fail criterion, so you stop early instead of debugging for three hours.

Reference device: Snapdragon 778G / Adreno 642L, Android 13, Termux. Numbers quoted are from that device.

---

## Step 0. CPU baseline first (mandatory)

Never start with the GPU. You need a number to beat, and on many phones the GPU can't beat it.

```bash
cd ~/llama.cpp
cmake --build build -j4
./build/bin/llama-bench -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -ngl 0 -fa off -p 256 -n 64 -r 3 -t 4
```

Pass criterion: the run finishes and gives something like `pp256 ≈ 29`, `tg64 ≈ 7.5` on a 778G.
Sweep `-t 2,4,6,8` and keep the best. Write both numbers down — that is your target.

---

## Step 1. Packages and sources

```bash
pkg install -y clang cmake ninja git python patchelf
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp && git log -1 --oneline     # record the commit
```

OpenCL headers: Termux ships no usable `CL/cl.h` for 2.0. Put a header-only Khronos copy in
`~/opencl-headers` (`CL/*.h` is enough).

---

## Step 2. Vendor libraries → `~/ocl`

```bash
mkdir -p ~/ocl && cd ~/ocl
for f in libOpenCL.so libCB.so libgsl.so libadreno_utils.so libllvm-glnext.so; do
  cp /vendor/lib64/$f . 2>/dev/null
done
for f in *.so; do patchelf --set-rpath '$ORIGIN' $f; done
```

Reason: Android's linker namespaces make `/vendor/lib64` invisible to Termux binaries. Copy + rpath is the
root-free workaround. From now on, **every** GPU run needs `LD_LIBRARY_PATH=$HOME/ocl`.

Pass criterion: `LD_LIBRARY_PATH=$HOME/ocl ./build/bin/llama-cli --help` still runs, and a test program
linked against `~/ocl/libOpenCL.so` can call `clGetPlatformIDs` without a namespace error.

---

## Step 3. Build with OpenCL

```bash
./scripts/build-cl.sh          # cmake line from bringup-gotchas.md §2
cmake --build build-cl -j4
```

Pass criterion: `build-cl/bin/llama-bench` exists. Warnings about missing kernels are fine; errors are not.
`-j4`, never `-j8`.

---

## Step 4. Is the driver even visible?

```bash
LD_LIBRARY_PATH=$HOME/ocl ./build-cl/bin/llama-bench --list-devices
```

Pass criterion: a line like `GPUOpenCL: QUALCOMM Adreno (TM) 642L ...`. Note that `--list-devices` shows
**only** GPU devices — the CPU has no device entry, so `-dev CPU` is invalid; CPU mode is `-ngl 0`.

Useful next: `clinfo`-style output — `max_compute_units`, `max_work_group_size`, `global_mem_size`,
`max_mem_alloc_size`, extensions (`cl_khr_fp16`, `cl_khr_subgroups`, `cl_khr_il_program`). On Adreno 642L:
2 CUs, 1024 WG size, 3648 MB global, 912 MB max alloc, no `cl_khr_il_program`.

---

## Step 5. First run: exactly one layer

```bash
export LD_LIBRARY_PATH=$HOME/ocl
./build-cl/bin/llama-bench -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -ngl 1 -fa off -p 32 -n 8 -r 1 -t 4
```

Pass criterion: finishes with `exit=0` (not 134). On the reference device this required the three patches
from `scripts/`; stock llama.cpp segfaulted in the vendor compiler.

---

## Step 6. Correctness before benchmarking

A fast wrong answer is worthless:

```bash
./build-cl/bin/llama-cli -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -ngl 8 -fa off -t 4 -st --no-warmup -n 16 -p "The capital of France is"
```

Pass criterion: `The capital of France is Paris.` — literally. If you see "powered", "秆", "urse", the
q6_K GEMV path is broken: set `GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=0` and `GGML_OPENCL_LMHEAD_CPU=1`.

Never time a configuration you haven't passed this test with.

---

## Step 7. Sweep how many layers on the GPU

```bash
./scripts/gpu-test.sh -p 256 -n 64 -r 2 -t 4 -l "0 8 14 20 37"
```

Or manually, interleaving CPU and GPU so thermal drift affects both equally:

```bash
for g in 0 8 14 20 37; do
  ./build-cl/bin/llama-bench -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    -ngl $g -fa off -p 256 -n 64 -r 2 -t 4 | grep '^|'
  cat /sys/class/thermal/thermal_zone0/temp
done
```

Pass criterion: results are **monotonic in the useful direction**. If not, stop here.

---

## Step 8. How to read the result (and when to stop)

- Look at the trend, not the noise. On Reno7 5G: `ngl 0` 28.8/7.45 → `ngl 8` 25.5/5.25 → `ngl 14` 22.7/4.14
  → `ngl 20` 20.7/3.32 → `ngl 37` 17.4/2.24. Every additional layer makes it slower → **the GPU is
  useless on this SoC; stop.**
- If pp improves and tg drops, you have the classic memory-bound case: offload never pays for interactive
  generation, it can pay for batch prefill on large-memory GPUs (not on a phone with shared LPDDR).
- Check the memory roofline: `tg × model_GB ≈ achieved GB/s`; compare with the SoC's spec bandwidth.
  Reno7: 7.45 × 2.094 ≈ 15.6 GB/s of 17.06 GB/s peak = 91 %. At 91 % there is nothing left to win.
- If the device is hot, repeat: pp can differ by 29 % between 40 °C and 47 °C on the same binary.

---

## Step 9. Automated run: `scripts/gpu-test.sh`

Does steps 4–8 in one pass and writes a log:

```bash
./scripts/gpu-test.sh -p 256 -n 64 -r 2 -t 4 -l "0 8 14 20 37"
MODEL=/path/to/other.gguf LLAMA=/path/to/llama.cpp ./scripts/gpu-test.sh -h
```

It exports the three `GGML_OPENCL_*` knobs, checks device visibility, does an `ngl 1` smoke test, verifies
the France→Paris output, sweeps `-ngl`, and greps its own log for compile errors and aborts.
Interpreting its output is step 10.

---

## Step 10. Fix ladder when it breaks

In order, one at a time, re-testing after each:

1. `-fa off` (with `-ngl > 0` this is not optional).
2. `rm -rf $PREFIX/tmp/llama.cpp/cl-cache` — a stale program cache masks kernel changes.
3. `patch_rms.py` → `patch_kernels2.py` → `patch_enq2.py`, rebuild, clear the cache again.
4. Add the `supports_op` F16 gate (`bringup-gotchas.md` §3.5).
5. `GGML_OPENCL_Q6K_GEMV_O4=1`, `GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=0`, `GGML_OPENCL_LMHEAD_CPU=1`.
6. `GGML_OPENCL_DISABLE_FUSION=1`, then `GGML_OPENCL_OPFILTER` to isolate an op.
7. `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF` rebuild (untested; the only route to FA/f16 compiling).
8. Re-check temperature: at 45 °C+ everything looks broken and isn't.

If you reach step 8 with worse-than-CPU numbers: write down the numbers, keep `-ngl 0`, and move on.
That is the correct engineering outcome, not a failure.

---

## Step 11. Minimal "I ran the GPU test" checklist

- [ ] commit hash, `-DGGML_*` flags, driver version recorded
- [ ] CPU baseline at the chosen `-t`, same model file, same `-p/-n/-r`
- [ ] `--list-devices` output saved
- [ ] `ngl 1` smoke test `exit=0`
- [ ] France→Paris correctness check passed
- [ ] `-ngl` sweep with temperatures per round
- [ ] raw logs committed (`timeout` + redirect, not terminal scrollback)
- [ ] verdict stated as a trend, plus which `-t`/`-ngl` you recommend

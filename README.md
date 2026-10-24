# llama.cpp on Snapdragon 778G and 8 Gen 2: CPU vs Adreno (OpenCL) — measurements, patches, and why the GPU loses

Benchmark of **Qwen2.5-3B-Instruct Q4_K_M** on two Android phones (one mid-range, one flagship), comparing
a pure CPU run with GPU offload through the `ggml-opencl` backend, plus the driver bugs that had to be
patched before the GPU would run at all.

Devices: OPPO Reno7 5G CN (`PFJM10`), Snapdragon 778G / Adreno 642L, 8 GB RAM, Android 13 — and, since
2026-10-24, OnePlus 11 (`PHB110`), Snapdragon 8 Gen 2 / Adreno 740, 14 GB RAM, Android 16 (§11).
Stack: Termux, clang 21.1.8, llama.cpp `19ab5a5` (Reno7) and `6b9ba4f` (OnePlus), fork `0.7.0-dev`.
Benchmark only — no server, no chat UI.

**Short version: on both SoCs, offloading layers to the Adreno GPU is a net loss at every depth.
Pure CPU with 4 threads is the fastest configuration, and it is already at ~91 % of the memory
roofline, so there is almost nothing left to win.**

The same experiment on a flagship SoC (Snapdragon 8 Gen 2 / Adreno 740, 6 compute units) is in §11 — the
GPU is much less slow there, but still never wins.

---

## 1. Results

`llama-bench -fa off -p 256 -n 64 -t 4`, medians of interleaved rounds, same binary and model file for
CPU and GPU:

| configuration | prompt 256 (pp) | generation 64 (tg) | vs CPU |
|---|---:|---:|---|
| **CPU, `-ngl 0`, `-t 4`** | **28.8 t/s** | **7.45 t/s** | — |
| GPU `-ngl 8` (8 of 37 layers) | 25.5 t/s | 5.25 t/s | pp −11 %, tg −30 % |
| GPU `-ngl 14` | 22.7 t/s | 4.14 t/s | pp −21 %, tg −44 % |
| GPU `-ngl 20` | 20.74 t/s | 3.32 t/s | pp −28 %, tg −55 % |
| GPU `-ngl 37` (all layers) | 17.37 t/s | 2.24 t/s | pp −40 %, tg −70 % |

Monotonic: every additional layer moved to the Adreno 642L makes the model slower.
There is no crossover point where partial offload helps.

Thread count on the CPU path (the only tuning knob that does move the needle):

| `-t` | pp256, t/s | tg64, t/s |
|---:|---:|---:|
| 2 | 13.6 | 6.21 |
| **4** | **26.4–28.8** | **7.47** |
| 6 | 24.3 | 5.65 |
| 8 | 25.5 | 4.47 |

4 threads = the four Cortex-A78 cores. Adding A55 cores raises prompt throughput slightly and drops
generation by 25–40 %, because decode becomes migration- and memory-contention bound.

### 1.1 Thermal state dominates the prompt number

This is easy to miss when comparing benchmarks on phones. The vendor thermal stack lowers
`scaling_max_freq` for the A78 cores once the SoC reaches ~45 °C:

```
/sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq   2208000 → 1862400
/sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq   2131200 → 1651200
```

Measured on the same binary and model:

| SoC temperature | pp256 | tg |
|---|---:|---:|
| 36–41 °C (clocks at spec max) | **40.47 ± 0.14** | 7.53 ± 0.08 |
| 45–47 °C (clocks throttled) | **28.9 ± 0.1** | 7.4 ± 0.1 |

That is **−29 % on prompt processing and only −2 % on generation**. Decode is bandwidth-bound, so it barely
cares about core clock; prompt processing is compute-bound and is throttled hard.
All CPU-vs-GPU comparisons above were taken in the hot state (rounds interleaved), so they are
self-consistent; read the absolute pp number as "28.9 sustained / up to 40.5 when cold".
`scaling_max_freq` is owned by `system:system` and is not writable by the Termux shell user, so this cap
cannot be removed without root.

---

## 2. Why the GPU loses (five concrete reasons)

1. **Shared memory bandwidth.** Decode reads the whole 2.09 GB of weights per token, so tg is a DRAM
   benchmark. 7.45 t/s × 2.094 GB ≈ **15.6 GB/s ≈ 91 % of the 17.06 GB/s peak** of this phone's
   LPDDR4x-4266 (2×16-bit). The CPU is already within ~9 % of the theoretical ceiling; offloading
   layers cannot create bandwidth, and the GPU path pays extra traffic for cross-backend copies.
2. **Two compute units.** `clinfo` reports `max_compute_units = 2`, `max_work_group_size = 1024`,
   `global_mem_size = 3648 MB`, `max_mem_alloc_size = 912 MB` for Adreno 642L. Quantized GEMV at this size
   is launch-latency bound, not throughput bound.
3. **Attention stayed on the CPU.** Flash-attention kernels do not compile on this driver
   (`err = -6 building fa f32_f16`), and the f16 kernels needed for attention are unusable, so attention ran
   on the CPU with a copy of the KV cache across backends each layer.
4. **Per-layer dispatch + `clFinish`.** 37 layers × several kernels, each a small submission with
   synchronization, on a driver that is not designed for low-latency single-stream inference.
5. **The GPU is not idle-fast either.** At `-ngl 37` the whole model is on the GPU, and tg is 2.24 t/s —
   three times worse than the CPU. This is not a "not enough layers offloaded" problem.

**Vulkan is a dead end on this device.** The Vulkan driver advertises 1.1.128 and lacks
`VK_KHR_8bit_storage`, while ggml's SPIR-V requires `StorageBuffer8BitAccess`; every shader fails with
`ErrorUnknown`. Don't spend time on it.

---

## 3. Getting OpenCL to work in Termux at all (what had to be patched)

Nothing below is optional on this device — with stock llama.cpp the GPU path does not survive
`-ngl 1`.

| Symptom | Cause | Fix |
|---|---|---|
| `dlopen failed: library "/vendor/lib64/libOpenCL.so" ... namespace` | Android linker namespaces hide `/vendor` from Termux binaries | copy vendor libs to `$HOME/ocl`, `patchelf --set-rpath '$ORIGIN'` each `.so`, run with `LD_LIBRARY_PATH=$HOME/ocl` |
| `SIGSEGV` inside `libllvm-glnext.so` / `QGPUPeepholeOptimizer::SplitMBBAt` at kernel build | Adreno 6xx compiler crashes when several kernels of one `.cl` file are compiled together; each kernel compiles fine alone | build only the first kernel of the offending sources (`first_kernel_only()`), see `scripts/patch_kernels2.py`, `scripts/patch_rms.py` |
| `CL_INVALID_WORK_GROUP_SIZE` → `GGML_ABORT` → exit 134 | dispatch tables request a local size larger than the compiled kernel allows | retry `clEnqueueNDRangeKernel` with `NULL` local size, see `scripts/patch_enq2.py` |
| `err = -6 building fa f32_f16` then `std::out_of_range: map::at`, exit 134 | no `cl_khr_il_program` (source-only compile) + missing f16 kernels | always `-fa off` with `-ngl > 0` |
| garbage output, `error -54` | `GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=1` path is broken on this driver | `GGML_OPENCL_Q6K_GEMV_O4=1`, `GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=0` |
| slow + wrong logits spread | `lm_head` on GPU | `GGML_OPENCL_LMHEAD_CPU=1` |
| `ssm_scan` / `mul_mv_f16_f32_l4` return `-11` | kernels unsupported by the driver | gate them in `supports_op` |

Working environment for every GPU run in this repo:

```sh
export LD_LIBRARY_PATH=$HOME/ocl
export GGML_OPENCL_Q6K_GEMV_O4=1
export GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=0
export GGML_OPENCL_LMHEAD_CPU=1
llama-bench -m model.gguf -ngl 8 -fa off -p 256 -n 64 -t 4
```

Correctness was checked before timing anything: `ngl 8` with the prompt `The capital of France is` must
return `Paris`. Without the three env knobs it returns plausible-looking noise, and `llama-bench` will
happily report fast numbers for garbage.

---

## 4. Root: what it would and would not fix

| Change | Effect |
|---|---|
| Override `scaling_max_freq` / thermal driver | pp 28.9 → ~40 t/s (+25–35 %), tg ~+2 %; cost is a 48–55 °C chassis |
| LMK / `max_phantom_processes` / zram tuning | stability (sshd stops being killed mid-run), not speed |
| `cpuset`/`taskset` (pin 4×A78 to llama, sshd on A55), `chrt`/`nice` | +1–3 % |
| Access to `devfreq`, `cooling_device*`, `kgsl` sysfs | diagnostics only |
| Swapping vendor blobs (`libOpenCL.so`, `libllvm-glnext.so`) | the only route to `cl_khr_il_program` / f16 attention, but SoC-specific: a gamble that can break graphics |

Root does **not** provide memory bandwidth, more compute units (`max_compute_units = 2` is the hardware),
or `VK_KHR_8bit_storage` in the Vulkan driver. It turns "28.9 / 7.4 sustained" into "~40 / 7.5 while cold"
and makes the runs reproducible. It does not make GPU offload worthwhile. A meaningfully faster tg needs
different hardware (LPDDR5 + more CUs), not a different software configuration.

---

## 5. Reproduce

Preconditions: Termux, ~10 GB free, the 1.95 GiB GGUF, and a phone that is **cool** (screen off for
10 minutes) if you want comparable prompt numbers.

```sh
# 1. baseline first — always
pkg install -y clang cmake ninja git python
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j4
./build/bin/llama-bench -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf -ngl 0 -p 256 -n 64 -t 4

# 2. OpenCL backend (see docs/bringup-gotchas.md for the vendor-lib dance)
./scripts/build-cl.sh          # or the cmake line from docs/bringup-gotchas.md §2
./scripts/patch_rms.py && ./scripts/patch_kernels2.py && ./scripts/patch_enq2.py
cmake --build build-cl -j4

# 3. smoke test, then sweep
./scripts/gpu-test.sh -p 256 -n 64 -r 2 -t 4 -l "0 8 14 20 37"
```

Build budget on this phone: ~8 min for `build-cl` from scratch, 2–3 min incremental, `-j4` maximum.
Never run two llama processes at once — the phantom-process killer takes down sshd first.

Full gotcha list, section by section, with the exact error strings: **[docs/bringup-gotchas.md](docs/bringup-gotchas.md)**.
Step-by-step GPU test procedure: **[docs/gpu-test-howto.md](docs/gpu-test-howto.md)**.
Full report with all measurements: **[docs/cpu-vs-gpu.md](docs/cpu-vs-gpu.md)**.

---

## 6. Data and machine-readable summary

- `data/bench-inter.log` — interleaved CPU/GPU rounds with SoC temperature per round (the primary source
  for the table in §1)
- `data/bench-threads.log` — thread sweep
- `data/bench-cpu.log` — cross-check on a CPU-only build (`29.0 / 7.57` at `-t 4`, within 1 % of the
  OpenCL build's CPU path)
- `data/bench-gpu.log` — full GPU sweep including failures
- `data/results.csv` — the numbers above in one flat table
- `data/results-onplus-sm8550.csv`, `data/thermal-onplus-sm8550-*.csv`, `logs/` — the second-device data
  from §11 (see `docs/oneplus-sm8550-cpu-vs-gpu.md` §11 for the column layout)
- `llms.txt` — compact, factual summary intended for LLM/RAG ingestion
- Dataset mirror on Hugging Face, with `device.json` and the logs:
  [huggingface.co/datasets/vovaxa44/snapdragon778g-llama-cpp-opencl-bench](https://huggingface.co/datasets/vovaxa44/snapdragon778g-llama-cpp-opencl-bench)

## 7. Citing

```
llama.cpp on Snapdragon 778G: CPU vs Adreno 642L (OpenCL) benchmark, 2026.
https://github.com/vovaxa55/snapdragon778g-llama-cpp-opencl-bench
```

`CITATION.cff` is included for reference managers and for tools that read it.

## 8. License

Scripts and patches: MIT. Text and data: CC BY 4.0. See `LICENSE` and `LICENSE-docs`.

## 9. Scope

Two models of phone, two SoCs (SD778G/Adreno 642L and SD 8 Gen 2/Adreno 740), one model file, one
llama-bench protocol. The numbers are still not a general statement about "Android GPUs" — they are
statements about these two SoC + memory + vendor-driver combinations. If you are on a different Adreno
generation, run `llama-bench` yourself; §3 and `docs/bringup-gotchas.md` tell you what will break, and
`docs/oneplus-sm8550-cpu-vs-gpu.md` shows what a newer driver does *not* fix.

---

## 10. Contributing numbers for other hardware

The interesting open question is whether the crossover point moves on SoCs with more GPU compute units
(Adreno 642L has only 2). **Partially answered:** §11 repeats the whole experiment on a Snapdragon 8 Gen 2
with an Adreno 740 and 6 compute units, and the crossover still does not appear — the GPU is merely less
catastrophically slow than on the 778G. If you measure another device, open an issue with the
[device report template](.github/ISSUE_TEMPLATE/device-report.md) — it asks exactly the fields needed to make
the numbers comparable: exact `llama-bench` command, `-fa` state, thread count, cmake flags, env knobs, SoC
temperature, and the roofline check (`tg × bytes_per_token` vs spec bandwidth).

Rules that keep the dataset useful:

1. Always report the CPU baseline from the same binary and model, same session, interleaved with the GPU runs.
2. Report SoC temperature, or at least whether the device was cold or hot.
3. Never report GPU numbers without a correctness check first (see `docs/gpu-test-howto.md` §7).
4. State the llama.cpp commit and driver version; both change results.

---

## 11. Second device: Snapdragon 8 Gen 2 / Adreno 740 — the crossover still does not appear

The obvious objection to everything above is "the Adreno 642L only has 2 compute units, of course it loses".
So the same experiment was repeated on a flagship-class SoC, with the same model, the same
`-p 256 -n 64 -t 4 -fa off` protocol, interleaved rounds, sustained `-r 10` runs and 1 Hz thermal telemetry.

Device: OnePlus 11 (`PHB110`), Snapdragon 8 Gen 2 (`SM8550`), Adreno 740 with **6 compute units**, 14 GB
RAM, Android 16. Full write-up: [`docs/oneplus-sm8550-cpu-vs-gpu.md`](docs/oneplus-sm8550-cpu-vs-gpu.md).

| metric | Reno7 (SD778G, 2 CU) | OnePlus 11 (SD 8 Gen 2, 6 CU) |
|---|---:|---:|
| CPU pp256, cool | 40.5 t/s | **45.1 t/s** |
| CPU pp256, warm / sustained | 28.9 t/s | **42.1 / 44.3 t/s** |
| CPU tg64, cool | 7.45 t/s | **13.81 t/s** |
| GPU ~50 %: pp | 22.7 t/s (44 % of layers), −21 % | 37.0 t/s (18/36 layers), −12 % |
| GPU ~50 %: tg | 4.14 t/s, −44 % | 12.97 t/s, −6 % |
| GPU 100 %: pp / tg | 17.4 / 2.24 t/s, −40 % / −70 % | 31.8 / 13.4 t/s, −24 % / −3 % |
| best `-ngl` for the GPU | none | none |
| sustained GPU penalty (`-r 10`) | not measured | **−37 % pp / −25 % tg** |
| OpenCL patches needed | segfaults, FA `err=-6`, wrong output | none |

What changed and what did not:

- **The driver matured.** The Adreno 740 needs none of the 778G workarounds, flash attention compiles and
  runs (it is still slower, see below), and the OpenCL path produces correct output out of the box.
- **The CPU moved further ahead.** Decode goes from 7.45 to 13.81 t/s — a measured ~15.6 → ~29 GB/s of
  effective decode bandwidth, i.e. the 8 Gen 2 roughly doubled per-token memory throughput, and the CPU path
  was already at ~91 % of the roofline on the 778G. There is even less headroom for the GPU to exploit.
- **Full offload stopped being catastrophic but stayed a loss.** Prompt processing falls monotonically with
  every layer moved to the GPU (42 → 32 t/s, −24 % at 100 % offload, versus −40 % on the Reno7); generation is
  essentially flat (−3 %). No layer count beats `-ngl 0`.
- **The throttling story is harsher, not gentler.** The trigger is SoC-wide (`scaling_max_freq` is cut at
  ~50–52 °C package / 85–91 °C big-core regardless of backend), the GPU run takes 176 s instead of 117 s for
  the same work, and while it runs the vendor perf-HAL pins the prime core at **864 MHz in 155 of 157 sampled
  seconds** (versus 1.84–2.48 GHz on the CPU run). Offload relocates heat; it does not remove it.
- **Memory is still the only reason to offload:** 2084 → 1350 MiB peak RSS, i.e. ~730 MiB freed, on a phone
  where that no longer matters.

Raw data: [`data/results-onplus-sm8550.csv`](data/results-onplus-sm8550.csv),
[`data/thermal-onplus-sm8550-cpu.csv`](data/thermal-onplus-sm8550-cpu.csv),
[`data/thermal-onplus-sm8550-gpu.csv`](data/thermal-onplus-sm8550-gpu.csv),
plus the two `-probe.csv` files and the run logs in [`logs/`](logs). The same data is mirrored to the
Hugging Face dataset.

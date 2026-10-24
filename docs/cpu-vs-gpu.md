# CPU vs GPU (OpenCL) benchmark: Qwen2.5-3B-Instruct Q4_K_M on OPPO Reno7 5G

Date: 2026-10-22. Device: `PFJM10` (OPPO Reno7 5G CN), Snapdragon 778G / Adreno 642L, 8 GB RAM, Android 13.
Scope: benchmark only — no `llama-server`, no chat UI, no autostart.

## 1. Summary

**GPU offload does not help on this device — it slows the model down monotonically.**
The best result comes from a pure CPU run (`-ngl 0`, 4 threads).

| configuration | prompt 256 tok | generation 64 tok | vs CPU |
|---|---:|---:|---|
| **CPU, `-ngl 0`, `-t 4`** | **28.8 t/s** | **7.45 t/s** | — |
| GPU `-ngl 8` (8/37 layers) | 25.5 t/s | 5.25 t/s | pp −11 %, tg −30 % |
| GPU `-ngl 14` | 22.7 t/s | 4.14 t/s | pp −21 %, tg −44 % |
| GPU `-ngl 20` | 20.7 t/s | 3.32 t/s | pp −28 %, tg −55 % |
| GPU `-ngl 37` (all layers) | 17.4 t/s | 2.24 t/s | pp −40 %, tg −70 % |

The more layers are given to the GPU, the slower it gets. Each layer computed on the Adreno 642L is more
expensive than the same layer on the CPU, so the optimal strategy is **not to offload anything**.

## 2. What was tested

- Model: `qwen2.5-3b-instruct-q4_k_m.gguf` (1.95 GiB, 3.40 B parameters, 181 f32 + 217 q4_K + 37 q6_K tensors).
- llama.cpp: fork `0.7.0-dev`, commit `19ab5a5`, built in Termux (clang 21, Release, `-DGGML_NATIVE=ON`).
- CPU backend: 4×Cortex-A78 (1×2.4 + 3×2.2 GHz) + 4×Cortex-A55 @1.9 GHz; NEON + dotprod, no i8mm.
- GPU backend: `GGML_OPENCL=ON`, OpenCL C 2.0, Qualcomm vendor libraries (`libOpenCL.so` from
  `/vendor/lib64`, copied to `$HOME/ocl` + `patchelf --set-rpath`), Adreno-optimised kernels
  (`GGML_OPENCL_USE_ADRENO_KERNELS=ON`, embedded).
- Working environment for GPU runs: `GGML_OPENCL_Q6K_GEMV_O4=1`, `GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=0`,
  `GGML_OPENCL_LMHEAD_CPU=1` (lm_head on CPU — otherwise slower and borderline incorrect),
  `-fa off` (flash-attention kernels do not compile on this driver).

## 3. Method

One binary (`build-cl/bin/llama-bench`) for both CPU and GPU, one model file, identical parameters:

```
llama-bench -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \
            -ngl {0,8,14,20,37} -fa off -p 256 -n 64 -r 2..3 -t 4
```

- `-t 4` — the optimal thread count (§4), identical for all runs.
- Several rounds interleaved (CPU → GPU → CPU → GPU) to average out frequency/temperature drift.
- SoC temperature during runs: 44–48 °C. **Thermal throttling is real and measurable:** the vendor thermal
  stack lowers `scaling_max_freq` on the A78 cores — at 45–47 °C `cpu7` goes 2 208 000 → **1 862 400** Hz,
  `cpu4` 2 131 200 → **1 651 200** Hz (a cold device runs at exactly the spec maximum).
- Consequence: **pp depends on temperature, tg almost does not.** Cold device (36–41 °C, clocks at max):
  pp256 **40.47 ± 0.14**, tg128 **7.53 ± 0.08**. Hot (45–47 °C, clocks cut): pp256 **28.9 ± 0.1**,
  tg64 **7.4 ± 0.1**. That is ~29 % loss on pp and ~2 % on tg. All CPU-vs-GPU comparisons below were taken
  in the hot state (rounds interleaved), so they are consistent among themselves, but the absolute pp should
  be read as "28.9 sustained / up to 40.5 when cold".
- Reproducibility: within-run spread ±0.01…0.09 t/s, between runs ≤2 % at equal temperature.
- Cross-check on a separate CPU-only build (`build/`, label `CPU` in the llama-bench table): `-t 4` →
  pp256 **28.98**, tg64 **7.57** t/s; `-t 8` → 29.59 / 5.88. The CPU numbers from `build-cl` (28.8 / 7.45)
  match the CPU-only build within ~1 %, so the comparison is sound.

## 4. CPU-only: thread count

| `-t` | pp256, t/s | tg64, t/s |
|---:|---:|---:|
| 2 | 13.6 | 6.21 |
| **4** | **26.4–28.8** | **7.47** |
| 6 | 24.3 | 5.65 |
| 8 | 25.5 | 4.47 |

Four threads (the A78 cluster) is the optimum. More threads add A55 cores and cross-cluster migrations:
pp barely moves, tg drops 25–40 %. For interactive generation `-t 4` is the unambiguous choice.

## 5. Why the GPU loses

1. **Memory is shared.** Adreno 642L has no VRAM of its own: CPU and GPU read the same weights from
   LPDDR4x. Generating a token is limited by memory bandwidth (~1.95 GiB of weights per token), so moving
   computation to the GPU adds no bandwidth — only overhead.
2. **Effective read speed on the GPU is low.** At `-ngl 37` we get 2.24 t/s, i.e. ~4.7 GB/s of effective
   bandwidth — several times below both the theoretical LPDDR4x bandwidth and what the CPU achieves
   (7.45 t/s × 2.09 GB ≈ **15.6 GB/s**, ~91 % of the 17 GB/s peak for LPDDR4x-4266 with a 32-bit bus).
   The CPU decoder is therefore already against the memory ceiling: ~9 % of headroom, and no offload scheme
   gets around it. The GPU fails because of small GEMV kernels (batch = 1) with poor locality and a
   dispatch bottleneck across 2 CUs.
3. **Too few compute units.** Adreno 642L has only 2 CUs (`max_compute_units = 2`) — too few for prompt
   processing even against 4×A78 with dotprod.
4. **Part of the graph stays on the CPU anyway.** On this driver the f16 attention kernels
   (`mul_mv_f16_f32_l4`, `..._mrow`) and flash-attention (`fa f32_f16` → `err = -6`) do not compile, so
   attention runs on the CPU and tensors are shuttled between backends. Those copies eat any gain.
5. **Synchronisation overhead.** One `clFinish` plus several small enqueues per layer; with 37 layers and
   64 tokens that is thousands of calls, each costing tens of microseconds on a mobile driver.

Practical conclusion: on this SoC, **GPU offload in llama.cpp makes sense only as an experiment**, not as a
working mode.

## 6. Correctness

- `GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=1` (the image path for q6_K GEMV, `lws = {64,4,1}`) produces **garbage
  text** on Adreno 642L ("powered", "秆", "urse") — the kernel requires exactly 4 subgroups, the driver does
  not honour the requested local size, and the retry with `NULL` local size produces a formally successful
  but incorrect launch.
- With `GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=0` (and any `GGML_OPENCL_Q6K_GEMV_O4`) the answers are correct:
  `The capital of France is Paris.`
- `lm_head` on the GPU has the worst time/benefit ratio: moving it to the CPU raises tg from 4.9 to 5.4 t/s
  (`llama-cli`, `-ngl 8`).

## 7. How to reproduce

```bash
# CPU
cd ~/llama.cpp
LD_LIBRARY_PATH=$HOME/ocl ./build-cl/bin/llama-bench \
  -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf -ngl 0 -fa off -p 256 -n 64 -r 3 -t 4

# GPU (best, but still slower than CPU)
export LD_LIBRARY_PATH=$HOME/ocl
export GGML_OPENCL_Q6K_GEMV_O4=1 GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=0 GGML_OPENCL_LMHEAD_CPU=1
./build-cl/bin/llama-bench \
  -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf -ngl 8 -fa off -p 256 -n 64 -r 3 -t 4
```

Raw logs: `data/bench-inter.log`, `data/bench-threads.log`, `data/bench-gpu.log`, `data/bench-cpu.log`.

## 8. What could be tried next (if the GPU ever matters)

- Build with `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF` (generic kernels) — might fix the compilation of
  FA/f16 attention and improve pp, but is unlikely to change tg (memory bound).
- `-DGGML_OPENCL_EMBED_KERNELS=OFF` + a program cache, to iterate over kernel variants faster.
- Check `cl_qcom_large_buffer` / `clCreateBuffer` with allocations > 912 MB (the current
  `max_mem_alloc_size = 912 MB`, but llama.cpp splits weights across buffers, so `-ngl 37` works).
- Compare against Vulkan — impossible on this driver: SPIR-V 1.5 requires `StorageBuffer8BitAccess`, and the
  Vulkan 1.1.128 driver lacks `VK_KHR_8bit_storage` (`ErrorUnknown`).

## 9. Would root help?

Short answer: **not for the GPU; partly for the CPU, and mostly on pp.**

| What root gives | Effect here |
|---|---|
| Override `scaling_max_freq` / thermal driver (the file is owned by `system`) | pp 28.9 → ~40 t/s (+25–35 %), tg ~+2 %; cost: 48–55 °C chassis |
| LMK / `max_phantom_processes` / zram (7.8 GB) | sshd stops being killed during runs; larger context — stability, not speed |
| `cpuset`/`taskset` (4×A78 for llama, sshd on A55), `chrt`/`nice` | +1–3 % |
| Access to `devfreq`, `/sys/class/thermal/cooling_device*`, `kgsl` sysfs | diagnostics, not speed |
| Swapping vendor blobs (`libOpenCL.so`, `libllvm-glnext.so`) | the only route to `cl_khr_il_program` / `err=-6`, but SoC-specific: a gamble that can break graphics |

What root does **not** give: memory bandwidth (GPU and CPU read the same LPDDR4x, and CPU decode is already
at ~91 % of the ceiling), `max_compute_units = 2` in Adreno 642L, `VK_KHR_8bit_storage` in the Vulkan driver.
So root turns "28.9 / 7.4 sustained" into "~40 / 7.5 while cold", but does not make GPU offload worthwhile.
A substantially better tg needs a different SoC — LPDDR5 and more CUs.

## 10. Update (2026-10-24): the same test on LPDDR5 and 6 compute units

Exactly that SoC was measured afterwards: OnePlus 11, Snapdragon 8 Gen 2 (`SM8550`), Adreno 740 with 6
compute units, LPDDR5X, Android 16, same model and protocol. Report:
[`oneplus-sm8550-cpu-vs-gpu.md`](oneplus-sm8550-cpu-vs-gpu.md), data:
[`data/results-onplus-sm8550.csv`](../data/results-onplus-sm8550.csv).

- Decode did improve as predicted: tg 7.45 → **13.81 t/s** (~15.6 → ~29 GB/s effective), and pp 28.8 → 45.1.
- More compute units did **not** produce a crossover: pp still falls monotonically with `-ngl`
  (42.1 → 31.8 t/s at 100 % offload, −24 %), and tg stays flat. Best config is still `-ngl 0 -t 4`.
- The Adreno 740 driver needs none of the 778G patches, so the "GPU is unusable" part of this report is
  specific to the old driver; the "GPU is slower than 4 ARM cores" part is not.
- Under sustained load the GPU path is *worse* than the short-run numbers suggest (−37 % pp / −25 % tg),
  because the whole SoC throttles and the prime core gets pinned at 864 MHz.

# CPU vs GPU (OpenCL): Qwen2.5-3B-Instruct Q4_K_M on OnePlus 11 (Snapdragon 8 Gen 2)

Date: 2026-10-24. Device: OnePlus `PHB110` (`OP591BL1`), Snapdragon 8 Gen 2 (`SM8550`, `kalama`) /
Adreno 740, 14 GB RAM + 9 GB swap, Android 16, kernel 5.15.180-android13.
Scope: benchmark only — no `llama-server`, no chat UI, no autostart. This is the follow-up to
[`cpu-vs-gpu.md`](cpu-vs-gpu.md) (OPPO Reno7 5G / Snapdragon 778G), and it answers the question left open
there: *does the crossover point move on a flagship SoC with 3× the GPU compute units?*

## 1. Summary

**GPU offload still does not pay, not even on an Adreno 740 with 6 compute units.** Pure CPU with 4 threads
is the fastest configuration in all four interleaved rounds and in both sustained runs: prompt processing
falls monotonically with every layer moved to the GPU, generation never moves more than −6 % / +4 % around
the CPU baseline, and under sustained load the GPU path is worse on both axes.

Two interleaved sessions (CPU → GPU → CPU → GPU), same binary (`build-cl/bin/llama-bench`), same model,
`-fa off`, `-t 4`, medians. Session A was started immediately after the build, i.e. on an already warm phone;
between A and B the phone was left to cool for ~3 minutes. `zone0` is the main thermal zone at the start →
end of that run — it was only sampled in session B (the runner script in session A did not read it), so those
cells say `n/a` rather than a made-up number:

| # | session | configuration | zone0 start → end | prompt 256 (pp) | generation 64 (tg) | vs CPU of the same pair |
|---:|---|---|---|---:|---:|---|
| 1 | A (hot) | CPU `-ngl 0`, `-t 4` | n/a | **42.05 ± 0.74** | 12.88 ± 0.53 | — |
| 2 | A (hot) | GPU `-ngl 18` (18/36 layers) | n/a | 36.96 ± 0.04 | 12.20 ± 0.80 | pp −12 %, tg −5 % |
| 3 | A (hot) | CPU `-ngl 0` | n/a | 36.59 ± 0.00 | 12.03 ± 0.52 | — |
| 4 | A (hot) | GPU `-ngl 18` | n/a | 33.61 ± 0.11 | 11.60 ± 0.76 | pp −8 %, tg −4 % |
| 5 | B (cooled) | CPU `-ngl 0` | 39.6 → 47.0 °C | **45.11 ± 0.74** | **13.81 ± 0.56** | — |
| 6 | B (cooled) | GPU `-ngl 18` | 47.0 → 49.3 °C | 36.97 ± 0.02 | 12.97 ± 0.71 | pp −18 %, tg −6 % |
| 7 | B (cooled) | CPU `-ngl 0` | 50.5 → 52.0 °C | 40.85 ± 0.01 | 12.59 ± 0.44 | — |
| 8 | B (cooled) | GPU `-ngl 18` | 52.7 → 54.1 °C | 35.99 ± 0.12 | 12.53 ± 0.74 | pp −12 %, tg −0.5 % |

Within each session there is no cooldown between the pairwise rounds, so the numbers drift down; between the
sessions there was one, and the CPU went back from 36.59 to 45.11 t/s — direct evidence that a large part of
the loss is thermal and not structural. Across the whole sequence the **CPU path loses 19 % of pp**
(45.11 → 36.59) while the **GPU path loses 9 %** (36.97 → 33.61): heat costs the CPU more, because the CPU
is doing all of the prompt compute. But the CPU stays ahead at every temperature. Best CPU:
45.11 / 13.81. Worst GPU: 33.61 / 11.60.

Sustained (`-r 10`: 117 s of continuous load for CPU, 176 s for GPU, each after a cooldown):

| configuration | pp256, t/s | tg64, t/s |
|---|---:|---:|
| **CPU `-ngl 0`, `-t 4`** | **44.30 ± 0.95** | **12.52 ± 0.44** |
| GPU `-ngl 18`, `-t 4` | 27.84 ± 0.73 | 9.41 ± 0.10 |

At its best (session B, round 1) the short-run penalty is mild — −18 % pp / −6 % tg, far better than the
−11…−70 % on the Reno7 — but under sustained load the gap widens to **−37 % pp / −25 % tg**: the GPU
configuration needs ~50 % more wall-clock time for the same 10 repetitions (176 s vs 117 s of sampled load),
and the longer you are hot, the harder the SoC throttles (§7).

## 2. What was tested

- Model: `qwen2.5-3b-instruct-q4_k_m.gguf`, 2 104 932 768 bytes, sha256
  `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d` (verified), 36 layers.
- llama.cpp: master commit `6b9ba4f`, `0.7.0-dev (build 1)`, built in Termux with clang 21.1.8,
  `-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON`, OpenCL backend with
  `-DGGML_OPENCL_USE_ADRENO_KERNELS=ON -DGGML_OPENCL_EMBED_KERNELS=ON`.
- One binary (`build-cl/bin/llama-bench`) for **both** CPU and GPU runs, exactly as in the Reno7 report,
  so the backend comparison cannot be polluted by different compilers or compiler flags.
- OpenCL: 3.0, Adreno 740, driver `0676.76.1`, `max_compute_units = 6`, `global_mem_size = 7.39 GiB`.
  No patches were needed here: the stack from Reno7 (vendor libs copied to `$HOME/ocl` + `patchelf
  --set-rpath`) works, flash-attention kernels now *compile and run* on this driver, and no
  `GGML_OPENCL_*` workaround variables were set for the numbers below.
- CPU: 1× Cortex-X3 + 4× Cortex-A715 + 3× Cortex-A510, `-t 4` on purpose (same as Reno7, and Termux'
  `nproc` reports 6 of the 8 CPUs anyway).
- GPU 50 % = `-ngl 18` of 36 layers.

## 3. Method

```
llama-bench -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf -ngl {0,18} -fa off -p 256 -n 64 -r 2..3 -t 4
```

- Rounds interleaved (CPU → GPU → CPU → GPU) in two states — first right after the build (hot), then again
  after a ~3-minute cooldown (cool) — so backend effect and thermal drift can be separated.
- Sustained runs: `-r 10`, each configuration preceded by a **5-minute cooldown** with the screen off.
- One llama/bench process at a time; device was on the charger during all runs (adds heat, see §7).
- Reproducibility: within-run σ ≤ 0.74 t/s (pp) and ≤ 0.80 t/s (tg); between rounds at equal temperature
  ≤ 2 %.

## 4. Layer sweep

Thermally warm device, `-t 4 -fa off`, one run per point (`-r 1`, `-r 2` for 18 layers); `*` = the warm-device
reference from session A (`-r 3`), which agrees with the sweep run at the same `-ngl` to within ±0.1 t/s pp
and ±0.8 t/s tg.

| `-ngl` | layers on GPU | pp256, t/s | tg64, t/s |
|---:|---:|---:|---:|
| 0 | 0 % | 42.1 `*` | 12.9 `*` |
| 9 | 25 % | 41.2 | 13.3 |
| 18 | 50 % | 37.0 | 13.0 |
| 27 | 75 % | 34.3 | 13.3 |
| 36 | 100 % | 31.8 | 13.4 |

Same shape as on the Reno7, but a different slope: **pp falls monotonically with offload** (42 → 32 t/s,
−24 % at full offload, versus −40 % there), and **tg is flat** (13.0–13.4 t/s) instead of collapsing
(−70 % there). The Adreno 740 is a competent GPU; it is simply not faster than 4 ARM cores at this
batch size.

## 5. Flash attention on Adreno 740

Unlike Adreno 642L, the FA kernels compile here (`err=-6` does not appear), so `-fa on` is now a real
choice — and it is still a bad one for decode:

| configuration | pp256, t/s | tg64, t/s |
|---|---:|---:|
| `-ngl 18 -fa on` | 41.68 | **7.55** ← tg nearly halves |
| `-ngl 0 -fa on` | 40.76 | 12.55 |

With 18 layers on the GPU, each generated token pays extra kernel dispatches for the GPU-side attention,
and per-token serialization dominates once more. Keep `-fa off`.

## 6. Memory

Peak process RSS (`VmHWM`):

| configuration | peak RSS |
|---|---:|
| CPU `-ngl 0` | 2084 MiB |
| GPU `-ngl 18` | 1350 MiB |

~730 MiB of weights move into GPU-visible memory. On a 14 GB phone this is irrelevant; on the 8 GB Reno7
it was the only reason to consider offload at all.

## 7. Thermal: does GPU offload throttle less?

This is the question that benchmarks with `-r 1..3` cannot answer, so it was measured directly: 1 Hz
sampling of every thermal zone (`cpu-0-0` little cluster, `cpu-1-0` big cluster, `cpuss-0`, `gpuss-0`,
battery, `thermal_zone0` in µ°C) and of `scaling_max_freq` for cpu0–cpu7, with `-r 10` runs and 5-minute
cooldowns.

| metric | CPU `-ngl 0` | GPU `-ngl 18` |
|---|---:|---:|
| `thermal_zone0` start → end | 34.4 → 53.2 °C | 36.3 → 53.3 °C |
| peak big-core zone (`cpu-1-0`) | 92.3 °C | 86.0 °C |
| peak mid-core zone (`cpuss-0`) | 88.4 °C | 82.9 °C |
| peak GPU zone (`gpuss-0`) | 69.0 °C | 72.2 °C |
| mean big-core zone | 86.0 °C | 68.4 °C |
| `scaling_max_freq` (cpu7) cut at | t = 102 s, zone0 **51.9 °C**, big **90.7 °C** | t = 144 s, zone0 **50.3 °C**, big **84.8 °C** |
| actual cpu7 (prime) clock | 1.84–2.48 GHz, varies | **864 MHz, pinned from t = 3 s** |
| lowest cpu3 clock actually reached | 1.65 GHz | **730 MHz** |
| sampled load duration | 117 s | 176 s |
| throughput | pp **44.30** / tg **12.52** | pp **27.84** / tg **9.41** |

Short probe (`-r 3`, 39 s of sampling for CPU and 49 s for GPU, similar starting temperature):

| metric | CPU `-ngl 0` | GPU `-ngl 18` |
|---|---:|---:|
| zone0 start → end | 37.0 → 47.1 °C | 35.0 → 46.3 °C |
| peak big / mid / GPU zone | 91.9 / 87.2 / 65.9 °C | 90.7 / 84.0 / 71.0 °C |
| mean big-core zone over the run | 82.2 °C | **70.3 °C** |
| cpu7 frequency cap | 2.09 GHz (unchanged) | 2.09 → 3.19 GHz (HAL *raised* it) |

Three conclusions:

1. **The throttle trigger is shared by the whole SoC.** `scaling_max_freq` gets cut at zone0 ≈ 50–52 °C and
   big-core ≈ 85–91 °C regardless of which backend produced the heat. There is no "GPU-friendly" thermal
   policy to exploit.
2. **Offload relocates heat, it does not remove it.** ~6 °C come off the big cores (86.0 vs 92.3 °C peak,
   68.4 vs 86.0 °C mean, both in the sustained runs) and ~3 °C go into the GPU zone (72.2 vs 69.0 °C).
   Package temperature is identical: 53.3 vs 53.2 °C at the end.
3. **Therefore GPU offload ends up throttling *more*.** The same 10 repetitions take 176 s instead of 117 s,
   so the device sits in the throttled regime for longer, and the A715 cores actually clock down to 730 MHz
   — versus 1.65 GHz on the CPU run. The prime core is the extreme case: the CPU run had it at 1.84–2.48 GHz
   for the whole 117 s, while in the GPU run it sat at **864 MHz in 155 of the 157 sampled seconds** (the
   only two exceptions are the samples before the load starts), i.e. the vendor HAL took away the fastest
   core precisely when the GPU was loaded. Slower → hot for longer → slower again.
   This is the mechanism behind the −37 % pp / −25 % tg sustained numbers in §1.

The caps are not monotone, which is why a single short probe is not enough: in the 49 s GPU probe the vendor
HAL actually *raised* cpu7 to its nominal 3.19 GHz (big cores were only 70 °C on average) and cpu3–6 to
2.80 GHz, while the CPU probe kept cpu7 pinned at 2.09 GHz. Read the ranking from the sustained measurements.

Note on the absolute numbers: this phone was on the charger during the runs, and the vendor perf-HAL caps
the prime core well below its nominal 3.19 GHz, moving the cap during the run. In the sustained CPU run
`cpu7` `scaling_max_freq` was raised to 2 476 800 Hz at t = 7 s and then walked down
2 342 400 → 2 092 800 → 1 843 200 Hz; in the interleaved sessions it went on down to 1 593 600 and
1 478 400 Hz. The A715 cluster is capped at 2 803 200 Hz nominal and the HAL moved it between
1 785 600 and 2 457 600 Hz on a per-sample basis. `cmd thermalservice override-status 0` does **not** lift
the caps, `dumpsys battery unplug` does not either, and `/sys/.../cpufreq` is not writable from the Termux
user. Treat "45 / 13.8" as what this phone actually delivers in its stock thermal configuration, not as an
SoC limit.

## 8. Why the GPU still loses here

1. **Decode is a DRAM benchmark and the GPU shares the same DRAM.** 12.52–13.81 t/s × 2.105 GB ≈
   26–29 GB/s of weight traffic per token. Offloading layers cannot create bandwidth: `-ngl 36` gives
   13.4 t/s, i.e. the same 28 GB/s. For scale, the Reno7's CPU reached 15.6 GB/s — this SoC is ~+86 %,
   which is what the newer LPDDR5X buys, not the Adreno.
2. **Six compute units is still not enough for this batch size.** Prompt processing at `-p 256` is a
   throughput problem, and 4 A715-class cores with dotprod beat 6 Adreno CUs on quantized GEMV once
   per-dispatch overhead and cross-backend copies are counted.
3. **Attention is serialized per token.** Flash attention works on this driver, but with offload it is
   *slower* (§5) — each token pays GPU dispatch latency for a tiny amount of work.
4. **The CPU path has no copies.** Weights live in ordinary `mmap`ed memory; the GPU path has to keep a
   second copy in GPU-visible allocations and round-trips the KV cache when the layer is on the CPU.

## 9. Reno7 vs OnePlus: what actually changed

| metric | Reno7 (SD778G / Adreno 642L) | OnePlus 11 (SD 8 Gen 2 / Adreno 740) | change |
|---|---:|---:|---:|
| CPU pp256, cold | 40.5 ± 0.1 | **45.1 ± 0.7** | +11 % |
| CPU pp256, warm / sustained | 28.9 ± 0.1 | **42.1 (warm) / 44.3 (sustained)** | +46–53 % |
| CPU tg64, cold | 7.53 (tg128) | **13.81 ± 0.56** | +83 % |
| CPU decode bandwidth | ~15.6 GB/s | **~29 GB/s** | +86 % |
| GPU ~22 %: pp (sweep) | 25.5 t/s, −11 % (`-ngl 8`, 8/37) | 41.2 t/s, −2 % (`-ngl 9`, 9/36) | similar |
| GPU ~38 %: pp | 22.7 t/s, −21 % (`-ngl 14`, 14/37) | 37.0 t/s, −18 % (`-ngl 18`) | same |
| GPU ~50 %: tg, short | **−30 % … −44 %** | **−6 %** | much better |
| GPU 100 %: pp | 17.4 t/s, −40 % | 31.8 t/s, −24 % | better |
| GPU 100 %: tg | 2.24 t/s, −70 % | 13.4 t/s, −3 % | much better |
| GPU ~50 %: tg, sustained | — (not measured) | **−25 %** | still a loss |
| OpenCL patches needed | segfaults, FA `err=-6`, wrong output | none | fixed on this driver |
| Sustained GPU penalty | n/a (no sustained test) | **−37 % pp / −25 % tg** | new finding |

The driver story improved a lot: the Adreno 740 needs none of the 778G patches, and full offload is no
longer catastrophic. But the *economic* answer did not change: there is no layer count at which offload
beats pure CPU, and under sustained load the gap is large.

## 10. Recommendation

```
llama-bench -m qwen2.5-3b-instruct-q4_k_m.gguf -ngl 0 -fa off -p 256 -n 64 -r 3 -t 4
```

- `-ngl 0` (pure CPU), `-t 4`, `-fa off`. Expect ~45 t/s pp and ~13.8 t/s tg cold, ~44 / 12.5 sustained.
- Use GPU offload (`-ngl 18`) only if you need to free ~730 MiB of RAM — and accept ~37 % lower pp and
  ~25 % lower tg in long generations.
- Do not tune for short bursts: the ranking holds in every thermal state, and gets worse for the GPU the
  longer you run.

## 11. Data

- [`data/results-onplus-sm8550.csv`](../data/results-onplus-sm8550.csv) — every number above, machine-readable.
- [`data/thermal-onplus-sm8550-cpu.csv`](../data/thermal-onplus-sm8550-cpu.csv),
  [`data/thermal-onplus-sm8550-gpu.csv`](../data/thermal-onplus-sm8550-gpu.csv) — raw 1 Hz telemetry of the
  sustained runs.
- [`data/thermal-onplus-sm8550-cpu-probe.csv`](../data/thermal-onplus-sm8550-cpu-probe.csv),
  [`data/thermal-onplus-sm8550-gpu-probe.csv`](../data/thermal-onplus-sm8550-gpu-probe.csv) — the same for the
  short probes in §7 (`pp32/tg8`), where the vendor HAL raised the caps instead of cutting them.

All four telemetry files share one header: `t_s,zone0_c,cpu_little_c,cpu_big_c,cpuss_c,gpu_c,batt_c,
cpu7_max_khz,cpu7_cur_khz,cpu3_max_khz,cpu3_cur_khz`, one row per second sampled from the first sample
(`t_s` = 0); temperatures are the sysfs millidegree values / 1000, clocks are integers in kHz. The probe files
leave `cpu3_cur_khz` empty because the probe script did not read that node. The raw bench output for every run
is in [`logs/`](../logs): `bench-cold.log` = session B, with a `STATE` line per round; `bench-run.log` =
session A, without them; `probe-*.log` = the short probes.

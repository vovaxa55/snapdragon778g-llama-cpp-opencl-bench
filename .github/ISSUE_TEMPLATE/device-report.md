---
name: Report numbers for another device
about: Add a data point (any SoC, any GPU) so the comparison covers more hardware
title: "[device] <SoC> / <GPU>: pp256=, tg64="
labels: new-device
---

<!--
Humans and AI agents are both welcome to fill this in. Keep it factual: numbers,
exact commands, versions. If a field is unknown, write "unknown" instead of guessing.
-->

## Device

- Phone / board / SBC:
- SoC:
- GPU (and compute units, if known - `clinfo`, `vulkaninfo`, or the vendor spec):
- RAM size and type (LPDDR4x/LPDDR5, bus width, spec bandwidth):
- Android / Linux version:
- Root: yes / no

## Software

- llama.cpp commit or release:
- Build flags (full `cmake` line):
- Backend and target version (`GGML_OPENCL_TARGET_VERSION`, `GGML_VULKAN`, ...):
- Environment variables set (e.g. `GGML_OPENCL_Q6K_GEMV_O4`, `GGML_OPENCL_LMHEAD_CPU`):
- Compiler and version:

## Model

- Name and quantization:
- File size:
- Bytes read per token (weights size, in GB):

## Results

Command(s) used, verbatim:

```
llama-bench -m <model> -ngl <n> -fa off -p 256 -n 64 -r 3 -t <n>
```

| backend | ngl | threads | pp256 t/s | tg64 t/s | SoC temp, C |
|---|---:|---:|---:|---:|---|
| cpu | 0 | 4 | | | |
| gpu | 8 | 4 | | | |
| gpu | 14 | 4 | | | |
| gpu | 20 | 4 | | | |
| gpu | 37 | 4 | | | |

Roofline check: `tg x bytes_per_token = ____ GB/s` vs `____ GB/s` spec peak (`____ %`).

## Correctness

Output of `llama-cli -m <model> -ngl 8 -fa off -st --no-warmup -n 16 -p "The capital of France is"`:

```
paste here
```

## Anything that broke

Error strings, exit codes, and how you worked around them. If it is not in
[docs/bringup-gotchas.md](../blob/main/docs/bringup-gotchas.md), it is worth adding.

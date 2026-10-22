#!/data/data/com.termux/files/usr/bin/bash
# GPU (OpenCL) smoke-test + layer sweep for llama.cpp on Adreno.
# Usage: ./gpu-test.sh [-p N] [-n N] [-r R] [-l "ngl..."] [-t THREADS]
set -u
P=256; N=64; R=2; T=4; NGLS="0 4 8 14 20 37"
while getopts "p:n:r:t:l:" o; do case $o in
  p) P=$OPTARG;; n) N=$OPTARG;; r) R=$OPTARG;; t) T=$OPTARG;; l) NGLS=$OPTARG;;
  *) echo "usage: $0 [-p n_prompt] [-n n_gen] [-r reps] [-t threads] [-l 'ngl list']"; exit 2;;
esac; done

# Vendor OpenCL libs copied to ~/ocl (see docs/bringup-gotchas.md 3.1)
export LD_LIBRARY_PATH=$HOME/ocl
# Knobs required for a correct and fastest GPU run on Adreno 642L
export GGML_OPENCL_Q6K_GEMV_O4=1
export GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=0
export GGML_OPENCL_LMHEAD_CPU=1

LLAMA=${LLAMA:-$HOME/llama.cpp}
BIN=$LLAMA/build-cl/bin/llama-bench
CLI=$LLAMA/build-cl/bin/llama-cli
MODEL=${MODEL:-$HOME/models/qwen2.5-3b-instruct-q4_k_m.gguf}
D=${TMPDIR:-$HOME/tmp}; mkdir -p "$D"
LOG=$HOME/gputest-$(date +%H%M%S).log

[ -x "$BIN" ] || { echo "missing $BIN - run build-cl.sh first"; exit 1; }
[ -f "$MODEL" ] || { echo "missing model $MODEL"; exit 1; }

echo "== 0. environment ==" | tee "$LOG"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tee -a "$LOG"
"$BIN" --list-devices 2>&1 | grep -E "device|Available|GPU" | tee -a "$LOG"

echo "== 1. driver: first run, ngl 1 ==" | tee -a "$LOG"
timeout 600 "$BIN" -m "$MODEL" -ngl 1 -fa off -p 32 -n 8 -r 1 -t "$T" >>"$LOG" 2>&1
echo "exit=$? (0 = ok; 134 = abort, check the log)" | tee -a "$LOG"

echo "== 2. output correctness (ngl 8) ==" | tee -a "$LOG"
timeout 600 "$CLI" -m "$MODEL" -ngl 8 -fa off -t "$T" -st --no-warmup -n 16 \
  -p "The capital of France is" >"$D/gpu-correct.out" 2>&1
echo "exit=$?" | tee -a "$LOG"
OUT=$(tr "\r" "\n" <"$D/gpu-correct.out" | grep -vE "^ggml_opencl|^llama|^load_|^print_info|^common_|^main:|^>|^$" \
      | tail -3 | tr -d "\r")
echo "output: $OUT" | tee -a "$LOG"
case "$OUT" in *Paris*) echo "OK: output is correct" | tee -a "$LOG";;
  *) echo "!!! GARBAGE: check GGML_OPENCL_Q6K_GEMV_O4_GLOBAL=0 and LMHEAD_CPU=1" | tee -a "$LOG";; esac

echo "== 3. ngl sweep: $NGLS ==" | tee -a "$LOG"
for g in $NGLS; do
  echo "--- ngl $g ---" | tee -a "$LOG"
  timeout 900 "$BIN" -m "$MODEL" -ngl "$g" -fa off -p "$P" -n "$N" -r "$R" -t "$T" 2>/dev/null \
    | grep -E "^\|" | tee -a "$LOG"
done

echo "== 4. failures in the log ==" | tee -a "$LOG"
grep -cE "kernel compile error" "$LOG" | xargs echo "kernel compile error:" | tee -a "$LOG"
grep -E "Aborted|GGML_ABORT|std::out_of_range|error -54" "$LOG" | sort -u | head -5 | tee -a "$LOG"
echo "log: $LOG"
echo
echo "== 5. reminders ==" | tee -a "$LOG"
echo "CPU baseline for comparison: the same call with -ngl 0" | tee -a "$LOG"
echo "if every extra ngl is slower -> GPU offload makes no sense on this SoC" | tee -a "$LOG"

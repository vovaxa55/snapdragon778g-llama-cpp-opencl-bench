#!/data/data/com.termux/files/usr/bin/bash
# Build llama.cpp with the Adreno OpenCL backend (Termux, no root).
# Prerequisites: clang cmake ninja, vendor libs in ~/ocl, headers in ~/opencl-headers.
# Usage: ./build-cl.sh [path-to-llama.cpp]   (default: ~/llama.cpp)
set -eu

LLAMA=${1:-$HOME/llama.cpp}
OCL=${OCL:-$HOME/ocl}
OCL_INC=${OCL_INC:-$HOME/opencl-headers}
JOBS=${JOBS:-4}

[ -f "$OCL/libOpenCL.so" ] || { echo "missing $OCL/libOpenCL.so - see docs/bringup-gotchas.md 3.1"; exit 1; }
[ -d "$OCL_INC/CL" ] || { echo "missing $OCL_INC/CL/*.h - put Khronos OpenCL headers there"; exit 1; }

cd "$LLAMA"
cmake -B build-cl -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENCL=ON \
  -DOpenCL_LIBRARY="$OCL/libOpenCL.so" \
  -DOpenCL_INCLUDE_DIR="$OCL_INC" \
  -DGGML_OPENCL_TARGET_VERSION=200 \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
  -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF
cmake --build build-cl -j"$JOBS"

# A stale program cache hides kernel edits - drop it after every rebuild.
rm -rf "${PREFIX:-/data/data/com.termux/files/usr}/tmp/llama.cpp/cl-cache"

echo
echo "done: $LLAMA/build-cl/bin/llama-bench"
echo "run with: LD_LIBRARY_PATH=$OCL $LLAMA/build-cl/bin/llama-bench --list-devices"

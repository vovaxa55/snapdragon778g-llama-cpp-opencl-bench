import os, sys
P = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp")
src = open(P, encoding="utf-8").read()
orig_len = len(src)

# ---------- 1. add the first_kernel_only() helper before build_program_from_source_ex ----------
anchor = "static cl_program build_program_from_source_ex("
assert src.count(anchor) == 1, "build_program_from_source_ex anchor"
helper = '''// Some programs of this backend crash the Adreno 6xx driver compiler (SIGSEGV inside
// llccib's QGPUPeepholeOptimizer::SplitMBBAt) when several of their kernels are built
// together, although every single kernel compiles fine on its own. For those files only
// the first kernel is built; the remaining ones stay unused (they are not dispatched for
// the model shapes this path is used with).
static std::string first_kernel_only(const std::string & src) {
    const size_t first = src.find("kernel void ");
    if (first == std::string::npos) {
        return src;
    }
    const size_t next = src.find("kernel void ", first + 1);
    if (next == std::string::npos) {
        return src;
    }
    size_t cut = src.rfind("\\n//----", next);
    if (cut == std::string::npos || cut < first) {
        cut = next;
    }
    return src.substr(0, cut);
}

'''
src = src.replace(anchor, helper + anchor, 1)

# ---------- 2. rms_norm: use the helper instead of the inline truncation ----------
old_rms = '''        std::string kernel_src_norm = kernel_src;
        {
            const size_t fused_pos = kernel_src_norm.find("kernel void kernel_rms_norm_mul(");
            if (fused_pos != std::string::npos) {
                const size_t cut = kernel_src_norm.rfind("//----", fused_pos);
                kernel_src_norm.resize(cut == std::string::npos ? fused_pos : cut);
            }
        }
        backend_ctx->program_rms_norm =
            build_program_from_source(backend_ctx, kernel_src_norm.c_str(), compile_opts);
'''
new_rms = '''        backend_ctx->program_rms_norm =
            build_program_from_source(backend_ctx, first_kernel_only(kernel_src).c_str(), compile_opts);
'''
assert src.count(old_rms) == 1, "rms_norm custom block anchor"
src = src.replace(old_rms, new_rms, 1)

# ---------- 3. first-kernel-only build for the other crashers ----------
targets = ["mean.cl", "sum_rows.cl", "ssm_scan.cl", "mul_mv_f16_f32_l4.cl"]
plain = "build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts)"
patched = "build_program_from_source(backend_ctx, first_kernel_only(kernel_src).c_str(), compile_opts)"
for name in targets:
    marker = '#include "%s.h"' % name
    idx = src.find(marker)
    assert idx != -1, "missing include for " + name
    bidx = src.find(plain, idx)
    assert bidx != -1, "missing build call for " + name
    src = src[:bidx] + patched + src[bidx + len(plain):]

open(P, "w", encoding="utf-8").write(src)
print("patched ok, delta bytes:", len(src) - orig_len)

import io, os, sys
P = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp")
src = open(P, encoding="utf-8").read()
orig = src

# ---- patch 1: build the rms_norm program from the plain kernel only ----
old1 = '''        backend_ctx->program_rms_norm =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_rms_norm     = clCreateKernel(backend_ctx->program_rms_norm, "kernel_rms_norm", &err), err));
        CL_CHECK((backend_ctx->kernel_rms_norm_mul = clCreateKernel(backend_ctx->program_rms_norm, "kernel_rms_norm_mul", &err), err));
        CL_CHECK((backend_ctx->kernel_rms_norm_mul_add = clCreateKernel(backend_ctx->program_rms_norm, "kernel_rms_norm_mul_add", &err), err));
'''
new1 = '''        // The Adreno 6xx compiler (llccib) segfaults inside QGPUPeepholeOptimizer when the
        // subgroup-reduction kernels of this file are built together in one program; every
        // one of them compiles fine on its own. Keep the plain rms_norm kernel in the
        // program, and leave the fused variants unused (the unfused path is taken).
        std::string kernel_src_norm = kernel_src;
        {
            const size_t fused_pos = kernel_src_norm.find("kernel void kernel_rms_norm_mul(");
            if (fused_pos != std::string::npos) {
                const size_t cut = kernel_src_norm.rfind("//----", fused_pos);
                kernel_src_norm.resize(cut == std::string::npos ? fused_pos : cut);
            }
        }
        backend_ctx->program_rms_norm =
            build_program_from_source(backend_ctx, kernel_src_norm.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_rms_norm = clCreateKernel(backend_ctx->program_rms_norm, "kernel_rms_norm", &err), err));
        backend_ctx->kernel_rms_norm_mul     = nullptr;
        backend_ctx->kernel_rms_norm_mul_add = nullptr;
'''
assert src.count(old1) == 1, "patch1 anchor not found"
src = src.replace(old1, new1)

# ---- patch 2: never dispatch the fused variants when their kernels are absent ----
old2 = '''        if (!backend_ctx->disable_fusion && backend_ctx->fuse_rms_add &&
            ggml_opencl_can_fuse(backend_ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD })) {'''
new2 = '''        if (!backend_ctx->disable_fusion && backend_ctx->fuse_rms_add &&
            backend_ctx->kernel_rms_norm_mul_add != nullptr &&
            ggml_opencl_can_fuse(backend_ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD })) {'''
assert src.count(old2) == 1, "patch2 anchor not found"
src = src.replace(old2, new2)

old3 = '''        if (!backend_ctx->disable_fusion && ggml_opencl_can_fuse(backend_ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {'''
new3 = '''        if (!backend_ctx->disable_fusion && backend_ctx->kernel_rms_norm_mul != nullptr &&
            ggml_opencl_can_fuse(backend_ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {'''
assert src.count(old3) == 1, "patch3 anchor not found"
src = src.replace(old3, new3)

open(P, "w", encoding="utf-8").write(src)
print("patched, delta bytes:", len(src) - len(orig))

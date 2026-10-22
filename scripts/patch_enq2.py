import os, sys
P = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp")
s = open(P, encoding="utf-8").read()

old_start = "        const cl_int enq_err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);"
old_end = '            GGML_ABORT("OpenCL enqueue failed");\n        }'
i = s.find(old_start)
j = s.find(old_end, i)
assert i != -1 and j != -1, "debug block not found"
j += len(old_end)

new = '''        cl_int enq_err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        if (enq_err == CL_INVALID_WORK_GROUP_SIZE && local_work_size != nullptr) {
            // Some Adreno kernels are compiled for a smaller maximum work-group size than the
            // one the dispatch tables request, so the driver rejects the local size. Let the
            // driver pick a valid one instead of failing the whole graph.
            char kname[128] = {0};
            clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kname), kname, NULL);
            GGML_LOG_WARN("ggml_opencl: local work size rejected for %s, retrying with NULL local size\\n", kname);
            enq_err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, NULL, 0, NULL, NULL);
        }
        if (enq_err != CL_SUCCESS) {
            char kname[128] = {0};
            if (kernel != nullptr) {
                clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kname), kname, NULL);
            }
            GGML_LOG_ERROR("ggml_opencl: enqueue failed err=%d kernel=\\"%s\\" work_dim=%u\\n", enq_err, kname, work_dim);
            GGML_ABORT("OpenCL enqueue failed");
        }'''
s = s[:i] + new + s[j:]
open(P, "w", encoding="utf-8").write(s)
print("retry patch applied")

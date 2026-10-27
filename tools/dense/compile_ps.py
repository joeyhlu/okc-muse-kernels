import ast, sys, time, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
tree = ast.parse(src)
vals = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        n = node.targets[0].id
        if n in ("CPP_SRC", "CUDA_SRC"):
            vals[n] = ast.literal_eval(node.value)
from torch.utils.cpp_extension import load_inline
t0 = time.time()
mod = load_inline(name="ps_lab_compile_test", cpp_sources=vals["CPP_SRC"], cuda_sources=vals["CUDA_SRC"],
                  functions=["ps_gemm", "ps_info", "ps_smem", "r32_gemv", "r32_info", "ps_empty"],
                  extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"],
                  verbose=True, build_directory=sys.argv[2])
print("BUILD OK", round(time.time() - t0, 1), "s")

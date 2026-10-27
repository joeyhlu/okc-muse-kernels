"""va_local.py: local checks of VA1 (tools/va_patches.py) from a generated layer file, without importing it.

  sm_90a compile only (ptxas registers / spills):  ... cudaenv.sh python kernels/lab/va_local.py <layer.py> sass
  RTX 3070 (sm_86) correctness of the portable path: ... cudaenv.sh python kernels/lab/va_local.py <layer.py> run
(the K1 arch guard is lowered to sm_80 for the run build; red.v4 / try_wait take their sm_80 fallbacks)
"""
import ast
import pathlib
import subprocess
import sys
import sysconfig
import tempfile
import time

import torch

NAMES = ("K1_CUDA_SRC", "VA1_CASES", "RMS_EPS", "FP8")
K1_MAIN = "template <int L, int ST>\n__global__ void __launch_bounds__(NTHR, (L == 1) ? 2 : K1_L8_MINB) k1_main(const Params p) {"
LAB_CPP = """
#include <torch/extension.h>
int64_t lab_occ(int64_t ST);
void lab_va(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor page_table, torch::Tensor k_cache,
            torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor ws_o, torch::Tensor ws_l,
            torch::Tensor ws_c, torch::Tensor out, int64_t B, int64_t page_shift, int64_t G, int64_t segk, double qk_sm, int64_t ST);
"""
LAB_HOST = """
}  // namespace k1
template <int ST> static int lab_go(const k1::Params* p, dim3 grid) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(k1::va1_main<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, k1::va_smem<ST>()));
  int nb = 0;
  if (p) k1::va1_main<ST><<<grid, 256, k1::va_smem<ST>(), at::cuda::getCurrentCUDAStream()>>>(*p);
  else C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k1::va1_main<ST>, 256, k1::va_smem<ST>()));
  return nb;
}
int64_t lab_occ(int64_t ST) { return ST == 3 ? lab_go<3>(nullptr, dim3(1)) : lab_go<4>(nullptr, dim3(1)); }
void lab_va(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor page_table, torch::Tensor k_cache,
            torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor ws_o, torch::Tensor ws_l,
            torch::Tensor ws_c, torch::Tensor out, int64_t B, int64_t page_shift, int64_t G, int64_t segk, double qk_sm, int64_t ST) {
  k1::Params p{};
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()); p.seq_lens = seq_lens.data_ptr<int>();
  p.cache_loc = cache_loc.data_ptr<int64_t>(); p.page_table = page_table.data_ptr<int>();
  p.k_cache = k_cache.data_ptr<uint8_t>(); p.v_cache = v_cache.data_ptr<uint8_t>();
  p.k_scale = k_scale.data_ptr<float>(); p.v_scale = v_scale.data_ptr<float>();
  p.part_o = ws_o.data_ptr<float>(); p.part_lse = ws_l.data_ptr<float>(); p.counters = ws_c.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.B = (int)B; p.Hq = 16; p.Hkv = 2; p.W = B == 1 ? 8 : 1; p.page_shift = (int)page_shift; p.pt_stride = (int)page_table.size(1);
  p.Gx = (int)G; p.min_chunk = (int)segk; p.use_rope = 0; p.fold = 1; p.qk_sm = (float)qk_sm;
  if (ST == 3) lab_go<3>(&p, dim3((unsigned)G, 1)); else lab_go<4>(&p, dim3((unsigned)G, 1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
FUNCS = ("_k1_mk_case", "_k1_ref_attention", "_k1_ref_new_kv", "_k1_rel_rows", "_k1_rms_t", "_k1_rope_t")


def load(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    ns = {"torch": torch}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in NAMES for t in node.targets):
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    return ns


def source(ns, portable):
    if not portable:
        return ns["K1_CUDA_SRC"]
    src = ns["K1_CUDA_SRC"][:ns["K1_CUDA_SRC"].index(K1_MAIN)] + LAB_HOST
    g = "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)\n#define K1_ARCH_OK 1"
    assert src.count(g) == 1
    return src.replace(g, "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)\n#define K1_ARCH_OK 1")


def sass(ns):
    from torch.utils.cpp_extension import include_paths
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "va1.cu"
        p.write_text(source(ns, False))
        inc = sum((["-I", d] for d in include_paths("cuda")), []) + ["-I", sysconfig.get_paths()["include"]]
        r = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-O3", "--expt-relaxed-constexpr", "-Xptxas=-v", "-std=c++20",
                            *inc, "-o", str(p.with_suffix(".cubin")), str(p)], capture_output=True, text=True)
        out = r.stderr
        if r.returncode != 0:
            print(out[-6000:])
            return 1
        for ln in out.splitlines():
            if "va1_main" in ln or ("registers" in ln and "va1" in out) or "spill" in ln:
                pass
        keep, cur = [], None
        for ln in out.splitlines():
            if "Compiling entry function" in ln:
                cur = ln
            if cur and ("va1_main" in cur or "k1_main" in cur) and ("Used" in ln or "spill" in ln):
                keep.append(cur.split("for")[-1].strip()[:60])
                keep.append(ln.strip())
        print("\n".join(keep) or out[-3000:])
        return 0


def run(ns, cases):
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    mod = load_inline(name="va1_lab86b", cpp_sources=LAB_CPP, cuda_sources=source(ns, True), functions=["lab_va", "lab_occ"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_86,code=sm_86"])
    print(f"[va_local] built for sm_86 in {time.time() - t0:.0f}s; occupancy st3 {mod.lab_occ(3)} st4 {mod.lab_occ(4)}")
    dev = torch.device("cuda", 0)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ws = (torch.zeros((16 * 1024, 64), dtype=torch.float32, device=dev), torch.zeros(16 * 1024, dtype=torch.float32, device=dev),
          torch.zeros(1024, dtype=torch.int32, device=dev))
    FP8 = ns["FP8"]
    bad = 0
    for (B, ctx, page, kvs, seed) in cases:
        for st, segk, cps in ((4, 400, 1), (3, 0, 1), (4, 64, 3)):
            occ = mod.lab_occ(st)
            G = sms * max(1, min(cps, occ)) if cps <= occ else sms * cps   # cps > occ: more CTAs than slots (waves)
            d = ns["_k1_mk_case"](dev, B, ctx, 1, False, page, kvs, seed)
            cfg = d["config"]
            out = torch.empty((B, 1, 16 * 64), dtype=torch.bfloat16, device=dev)
            mod.lab_va(d["qkv"], d["seq_lens"], d["cache_loc"], d["page_table"], d["k_cache"].view(torch.uint8),
                       d["v_cache"].view(torch.uint8), d["k_scale"], d["v_scale"], ws[0], ws[1], ws[2], out, B,
                       page.bit_length() - 1, G, max(1, segk), float(cfg["qk_scale_factor"] * cfg["sm_scale"]), st)
            torch.cuda.synchronize()
            pos = d["seq_lens"].long()[:, None]
            slots = torch.gather(d["page_table"].long(), 1, pos // page) * page + pos % page
            k_new = d["k_cache"].view(torch.uint8).view(-1, 2, 64)[slots]
            v_new = d["v_cache"].view(torch.uint8).view(-1, 2, 64)[slots]
            ref = ns["_k1_ref_attention"](d, k_new, v_new)
            tot, row = ns["_k1_rel_rows"](out, ref)
            kref, vref = ns["_k1_ref_new_kv"](d)
            kt, _ = ns["_k1_rel_rows"](k_new.view(FP8).float(), kref)
            vt, _ = ns["_k1_rel_rows"](v_new.view(FP8).float(), vref)
            clean = int(ws[2].abs().sum()) == 0 and float(ws[0].abs().sum()) == 0.0 and float(ws[1].abs().sum()) == 0.0
            ok = bool(torch.isfinite(out.float()).all()) and clean and tot < 2e-2 and row < 6e-2 and kt < 3e-2 and vt < 3e-2
            bad += not ok
            print(f"[va_local] B={B} ctx={ctx} page={page} kvs={kvs:g} st={st} segk={segk} G={G}: rel {tot:.3e} row {row:.3e} "
                  f"newK {kt:.2e} newV {vt:.2e} ws {'clean' if clean else 'DIRTY'} -> {'ok' if ok else 'FAIL'}", flush=True)
    print(f"[va_local] {'ALL OK' if not bad else f'{bad} FAILED'}")
    return 1 if bad else 0


def main():
    ns = load(sys.argv[1])
    what = sys.argv[2] if len(sys.argv) > 2 else "run"
    if what == "sass":
        return sass(ns)
    cases = ns["VA1_CASES"] + ((1, 500, 1, 1.0, 401), (64, 40, 1, 1.0, 402), (2, 20, 64, 1.0, 306))
    return run(ns, cases)


if __name__ == "__main__":
    sys.exit(main())

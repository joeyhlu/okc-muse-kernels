"""M32 lab (MoE bs16/32 fresh designer, notes/study/moe_bs32_fresh.md; box only via tools/box/m32_run.py): K2A vid 3
(HK2 NCH 4, NST 3) with a HOST-ONLY change: the grid of K2 (probe bit 25) and / or K1 (probe bit 26) is halved to one
CTA per SM (132). Device code is byte-identical to the shipped k2a_v37 module (the probe bits 25 / 26 are read only in
launch36 on the host; vid 3's device code reads probe & 15 and probe & 16 only), so the lab copy without the bits is the
codegen / noise control.

Hypothesis tested (layer routings, box_out/sa5/ship per-CTA marks): at T = 16 / 32 the SM streams ~30-33 GB/s whenever it
holds a cold CTA, whatever the partner (C|C 2 x 18.4 KB / 1.19 us, C|H2 18.4 / 1.0 + 18.4 / 1.5); the time above a stream
is idle-SM time at the ends of K1 / K2. K2 has 13-20 stages per CTA (< one 16-stage item), so its p50 -> max spread
(6.5-12.7 us) and its late PDL landers are a large share of it. With 132 K2 CTAs every K2 CTA lands in the first freed slot
of its SM (~10 us before K1 ends), so none lands late, and no SM pairs two K2 CTAs (no age priority, no partner-dependent
speed); the cost is the lone-CTA rate (~1.52 vs ~1.7 stages / us per SM, R-tail data).
"""
import ast
import hashlib
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
K2A_FILE = ROOT / "kernels/muse_moe_w4a16/k2a_v37.py"
K2G, K1G = 25, 26   # probe bits (host only)

PATCHES = [
    ("K1 grid (probe bit 26)",
     "    k3::k1_kernel36<NCH, NST><<<G, 320, SM, st>>>(p);\n",
     "    const int g1m = ((p.probe >> 26) & 1) ? G / 2 : G;   // M32 lab: K1G\n"
     "    k3::k1_kernel36<NCH, NST><<<g1m, 320, SM, st>>>(p);\n"),
    ("K2 grid (probe bit 25)",
     "    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;\n"
     "    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel36<NCH, NST>, p));\n",
     "    cfg.gridDim = dim3(((p.probe >> 25) & 1) ? G / 2 : G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;   // M32 lab: K2G\n"
     "    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel36<NCH, NST>, p));\n"),
]


def _shipped():
    tree = ast.parse(K2A_FILE.read_text(encoding="utf-8"))
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("KERNEL_SRC_V3", "CPP_SRC_V3") for t in node.targets):
            exec(compile(ast.Module([node], []), str(K2A_FILE), "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name == "_wg_specs":
            exec(compile(ast.Module([node], []), str(K2A_FILE), "exec"), ns)
    return ns["KERNEL_SRC_V3"], ns["CPP_SRC_V3"], ns["_wg_specs"]()


def source():
    src, cpp, wg = _shipped()
    for what, old, new in PATCHES:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"m32 patch {what!r}: anchor found {n} times")
        src = src.replace(old, new)
    src = (src.replace("@CASES3@", "    case 0: launch36<4, 3>(p, (int)G, (int)G3, (int)mask); break;")
              .replace("@ATTRS3@", "    case 0: return attrs36<4, 3>();").replace("@WGSPECS@", wg))
    return src, cpp


def build():
    from torch.utils.cpp_extension import load_inline
    src, cpp = source()
    tag = hashlib.sha1((cpp + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"m32_k2g_{tag}", cpp_sources=cpp, cuda_sources=src, functions=["run3", "attrs3v"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills",
                                         "-gencode=arch=compute_90a,code=sm_90a"])
    return mod, time.time() - t0


def call(mod, st, d, hot, bits=0):
    import torch
    import k2a_v37 as D
    x = d["hidden_states"].contiguous()
    out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
    mod.run3(0, 7, x, d["topk_ids"].to(torch.int32).contiguous(), d["topk_weights"].float().contiguous(),
             st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], st["h"], st["inter"], st["y"], st["meta"],
             st["cnt"], st["err"], out, st["sink"], st["ss"], st["cnt2"], st["xw"], st["xw2"],
             D.hot_code(D.V3_VARIANTS[3], hot), x.shape[0], d["config"]["local_start"], st["grid"][0], st["sms"] * 2,
             D.V3_HINT | bits)
    return out

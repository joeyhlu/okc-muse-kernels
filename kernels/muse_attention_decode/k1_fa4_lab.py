"""K1-FA4 feasibility lab (2026-10-25): can FA4's CuTe-DSL Hopper forward serve L=8 full-layer attention?

Stages (comma list via --stage):
  env    versions of the CuTe DSL / FA4 / quack stack in the judge image; copies the image's vendored FA4 cute
         sources to box_out/fa4_image/ (to read the exact code the image runs).
  bench  attention core only (no QK-norm / cache write / gate), same process, CUDA graph replay, cold L2 before
         every replay (k1_v3_bench.py timing), on meta.json shapes (default verify16 full = 9, verify32 full = 11):
           fa3      sgl_kernel FA3 flash_attn_with_kvcache, E4M3 Q/K/V (what SGLang runs)
           fa3_l2   same, page table folded into a 16K-row pool (L2-resident: compute/latency-bound time)
           fa4_*    vendored FA4 sm90 (sglang.kernels.ops.attention.flash_attn.cute), F16 Q/K/V (FP8 -> F16 is
                    exact), paged page_size 1 (cp.async gather), pack-GQA, split-KV + FA4 combine
             *_l2   L2-resident pool  -> the FA4 pipeline's compute/latency-bound time at the full key count
             *_half half the context  -> the same bytes as FP8 at full context, half the compute
         An FP8-load/F16-compute FA4 derivative is bounded below by max(FP8 bytes stream, fa4_l2 + conversion).
Numerics: each variant vs an exact FP32 softmax on two requests (rel L2).
"""
import argparse
import hashlib
import importlib
import importlib.metadata as md
import importlib.util
import json
import os
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROB = ROOT / "problems" / "muse_attention_decode"
OUT = ROOT / "box_out"

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="env,bench")
ap.add_argument("--shapes", default="9,11")
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--variants", default="")
args = ap.parse_args()
stages = set(args.stage.split(","))
OUT.mkdir(exist_ok=True)


def say(*a):
    print(*a, flush=True)


VENDOR = "sglang.kernels.ops.attention.flash_attn.cute"
# Process-local CuTe JIT cache: patched variants must never hit (or pollute) the persistent AOT cache.
os.environ["SGLANG_CUTE_AOT_CACHE_DIR"] = ""
PATCH_DIR = Path("/tmp/k1fa4")


def load_patched_sm90():
    """The image's vendored FA4 sm90 forward with three knobs exposed through a module dict _K1, read at trace
    time: launch min_blocks_per_mp, the (consumer, producer) setmaxnreg pair for one MMA warpgroup, and the K/V
    pipeline depth. Defaults reproduce the stock kernel. The patch is applied to the image's own file at run time
    (written to /tmp so CuTe's AST preprocessor can read the source); no FA4 code lives in this repo."""
    src = importlib.import_module(VENDOR + ".flash_fwd_sm90")
    txt = Path(src.__file__).read_text()
    reps = [("min_blocks_per_mp=1,", 'min_blocks_per_mp=_K1["minb"],'),
            ("1: (256, 56),", '1: _K1["regs1"],'),
            ("        super().__init__(*args, **kwargs)\n",
             '        kwargs["num_stages"] = _K1.get("stages", kwargs.get("num_stages", 2))\n'
             "        super().__init__(*args, **kwargs)\n")]
    for a, b in reps:
        assert txt.count(a) == 1, ("patch anchor", a, txt.count(a))
        txt = txt.replace(a, b)
    txt = '_K1 = {"minb": 1, "regs1": (256, 56), "stages": 2}\n' + txt
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    f = PATCH_DIR / "k1_fwd_sm90_patched.py"
    f.write_text(txt)
    spec = importlib.util.spec_from_file_location("k1_fwd_sm90_patched", f)
    m = importlib.util.module_from_spec(spec)
    sys.modules["k1_fwd_sm90_patched"] = m
    spec.loader.exec_module(m)
    return m

if "env" in stages:
    say("ENV python", sys.version.split()[0])
    for pkg in ("torch", "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-cu13",
                "flash-attn-4", "flash-attn", "flash_attn", "quack-kernels", "apache-tvm-ffi", "sglang",
                "sgl-kernel", "sglang-kernel", "flashinfer-python", "tilelang", "triton", "cuda-python",
                "cuda-bindings", "torch-c-dlpack-ext"):
        try:
            say(f"ENV pkg {pkg}=={md.version(pkg)}")
        except Exception:  # noqa: BLE001
            say(f"ENV pkg {pkg} MISSING")
    try:
        import cutlass
        import cutlass.cute as cute  # noqa: F401
        say("ENV cutlass", getattr(cutlass, "__version__", "?"), cutlass.__file__)
        from cutlass.cute.nvgpu import warpgroup
        say("ENV warpgroup:", [n for n in dir(warpgroup) if not n.startswith("_")])
        import cutlass.utils.hopper_helpers as hh
        say("ENV hopper_helpers:", [n for n in dir(hh) if not n.startswith("_")][:60])
        say("ENV cute.arch cvt/fence:", [n for n in dir(cute.arch) if any(s in n.lower() for s in
                                                                       ("cvt", "fence", "fp8", "e4m3", "f16"))])
    except Exception:  # noqa: BLE001
        say("ENV cutlass import FAILED\n" + traceback.format_exc()[-1500:])
    for modname in (VENDOR + ".interface", "flash_attn.cute.interface"):
        try:
            t0 = time.time()
            m = importlib.import_module(modname)
            say(f"ENV import {modname} OK ({time.time() - t0:.1f}s) {m.__file__}")
        except Exception:  # noqa: BLE001
            say(f"ENV import {modname} FAILED\n" + traceback.format_exc()[-1200:])
    try:
        m = importlib.import_module(VENDOR)
        src = Path(m.__file__).parent
        dst = OUT / "fa4_image"
        dst.mkdir(exist_ok=True)
        for f in sorted(src.glob("*.py")):
            shutil.copy(f, dst / f.name)
        hs = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:12] for f in sorted(src.glob("*.py"))}
        say("ENV vendored sha:", json.dumps(hs))
        q = importlib.import_module("quack")
        say("ENV quack", q.__file__, getattr(q, "__version__", "?"))
    except Exception:  # noqa: BLE001
        say("ENV vendored copy FAILED\n" + traceback.format_exc()[-800:])

if "bench" not in stages:
    sys.exit(0)

import torch  # noqa: E402

sys.path.insert(0, str(PROB))
import reference  # noqa: E402

dev = torch.device("cuda")
FP8 = torch.float8_e4m3fn
from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: E402

fa4i = importlib.import_module(VENDOR + ".interface")
PM = load_patched_sm90()
fa4i.FlashAttentionForwardSm90 = PM.FlashAttentionForwardSm90
K1_DEFAULT = dict(PM._K1)
_cur_patch = [None]
_keep = []  # keep compiled modules alive for the life of the process


def set_patch(p):
    key = tuple(sorted(p.items()))
    if key == _cur_patch[0]:
        return
    _keep.append(fa4i._flash_attn_fwd.compile_cache)
    PM._K1.clear()
    PM._K1.update(K1_DEFAULT)
    PM._K1.update(p)
    fa4i._flash_attn_fwd.compile_cache = fa4i._get_jit_cache("fwd")
    _cur_patch[0] = key

flush_buf = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
sink = torch.empty((), dtype=torch.float32, device=dev)


def time_graph(g, reps):
    ts = []
    for _ in range(reps):
        torch.sum(flush_buf, dim=0, out=sink)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(ts)


eg = torch.cuda.CUDAGraph()
with torch.cuda.graph(eg):
    pass
floor = time_graph(eg, 50)
say(f"BENCH empty-graph replay {floor:.2f} us (subtracted)")

POOL = 16384
_K1MOD = []
_K2MOD = []
_K4MOD = []


def get_k4():
    if not _K4MOD:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import k1_fa4v4
        _K4MOD.append(k1_fa4v4)
    return _K4MOD[0]


def get_k2():
    if not _K2MOD:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import k1_fa4v2
        _K2MOD.append(k1_fa4v2)
    return _K2MOD[0]


def get_k1():
    if not _K1MOD:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import k1_fa4
        _K1MOD.append(k1_fa4)
    return _K1MOD[0]


benches = json.loads((PROB / "meta.json").read_text())["benchmarks"]
P2 = {"minb": 2, "regs1": (200, 56)}
VARIANTS = [
    # name, kind, opts
    ("fa3", "fa3", {}),
    ("fa3_l2", "fa3", {"l2": True}),
    ("fa3_half", "fa3", {"half": True}),
    ("fa4_t64n128", "fa4", {"tile": (64, 128)}),
    ("fa4_t64n128_s2x", "fa4", {"tile": (64, 128), "split_mul": 2}),
    ("fa4_t64n128_noOL", "fa4", {"tile": (64, 128), "iwo": False}),
    ("fa4_t64n64", "fa4", {"tile": (64, 64)}),
    ("fa4_t128n128", "fa4", {"tile": (128, 128)}),
    ("fa4_t64n128_l2", "fa4", {"tile": (64, 128), "l2": True}),
    ("fa4_t64n64_l2", "fa4", {"tile": (64, 64), "l2": True}),
    ("fa4_t64n128_half", "fa4", {"tile": (64, 128), "half": True}),
    ("fa4_noOL_l2", "fa4", {"tile": (64, 128), "iwo": False, "l2": True}),
    ("fa4_noOL_half", "fa4", {"tile": (64, 128), "iwo": False, "half": True}),
    ("p2_IWO", "fa4", {"tile": (64, 128), "patch": P2, "split_mul": 2}),
    ("p2_IWO_l2", "fa4", {"tile": (64, 128), "patch": P2, "split_mul": 2, "l2": True}),
    ("p2_IWO_half", "fa4", {"tile": (64, 128), "patch": P2, "split_mul": 2, "half": True}),
    ("p2_noOL", "fa4", {"tile": (64, 128), "patch": P2, "split_mul": 2, "iwo": False}),
    ("p2_noOL_l2", "fa4", {"tile": (64, 128), "patch": P2, "split_mul": 2, "iwo": False, "l2": True}),
    ("p2_noOL_half", "fa4", {"tile": (64, 128), "patch": P2, "split_mul": 2, "iwo": False, "half": True}),
    ("st3_noOL", "fa4", {"tile": (64, 128), "patch": {"stages": 3}, "iwo": False}),
    ("st3_noOL_l2", "fa4", {"tile": (64, 128), "patch": {"stages": 3}, "iwo": False, "l2": True}),
    ("p2st3_noOL", "fa4", {"tile": (64, 128), "patch": dict(P2, stages=3), "split_mul": 2, "iwo": False}),
    ("p2st3_noOL_l2", "fa4", {"tile": (64, 128), "patch": dict(P2, stages=3), "split_mul": 2, "iwo": False, "l2": True}),
    ("k4", "k4", {}),
    ("k4_l2", "k4", {"l2": True}),
    ("k4_st4", "k4", {"stages": 4}),
    ("k2", "k2", {}),
    ("k2_l2", "k2", {"l2": True}),
    ("k2_half", "k2", {"half": True}),
    ("k2_s2x", "k2", {"split_mul": 2}),
    ("k2_st3", "k2", {"stages": 3}),
    ("k2_st3_l2", "k2", {"stages": 3, "l2": True}),
    ("k1_bar", "k1", {"cvt": "direct", "map": "bar"}),
    ("k1_bar_skip", "k1", {"cvt": "skip", "map": "bar"}),
    ("k1_own_skip", "k1", {"cvt": "skip", "map": "own"}),
    ("k1_own_l2", "k1", {"cvt": "direct", "map": "own", "l2": True}),
    ("k1_own_s2x", "k1", {"cvt": "direct", "map": "own", "split_mul": 2}),
    ("k1_fp8", "k1", {"cvt": "direct"}),
    ("k1_fp8_l2", "k1", {"cvt": "direct", "l2": True}),
    ("k1_fp8_half", "k1", {"cvt": "direct", "half": True}),
    ("k1_fp8_s2x", "k1", {"cvt": "direct", "split_mul": 2}),
    ("k1_fp8f32", "k1", {"cvt": "f32"}),
    ("k1_fp8f32_l2", "k1", {"cvt": "f32", "l2": True}),
]
if args.variants:
    keep = set(args.variants.split(","))
    VARIANTS = [v for v in VARIANTS if v[0] in keep]


def exact_ref(qn, k_cache, v_cache, page_table, seqlens, b, L, hq, hkv, scale):
    """FP32 softmax attention for request b: qn [B, L, Hq, 64] unit-RMS queries (pre qk factor)."""
    n = int(seqlens[b])
    rows = page_table[b, :n].long()
    kk = k_cache.view(-1, hkv, 64)[rows].float()  # page size 1
    vv = v_cache.view(-1, hkv, 64)[rows].float()
    g = hq // hkv
    q = qn[b].float().view(L, hkv, g, 64)
    s = torch.einsum("lkgd,nkd->kgln", q, kk) * scale
    pos = torch.arange(n, device=q.device)
    lim = n - L + torch.arange(L, device=q.device)
    s = s.masked_fill(pos[None, :] > lim[:, None], float("-inf"))
    p = torch.softmax(s, -1)
    o = torch.einsum("kgln,nkd->lkgd", p, vv)
    return o.reshape(L, hq, 64)


for i in [int(x) for x in args.shapes.split(",")]:
    bm = benches[i]
    kw = {k: v for k, v in bm.items() if k not in ("load", "weight")}
    t0 = time.time()
    data = reference.generate_input(**kw)
    cfg = data["config"]
    B, L, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    assert cfg["page_size"] == 1
    label = f"[{i:2d}] {bm['load']:9s} {bm['layer']:7s} B{B} L{L}"
    q_in = data["qkv"].view(B, L, -1)[..., :hq * 64].reshape(B, L, hq, 64)
    qn = reference._rms(q_in)  # unit RMS, fp32
    scale = cfg["sm_scale"] * cfg["qk_scale_factor"]
    q8 = qn.reshape(B * L, hq, 64).to(FP8)
    q16 = qn.to(torch.float16).contiguous()  # [B, L, Hq, 64]
    q8b = qn.to(FP8).contiguous()  # [B, L, Hq, 64] E4M3 (stock query precision) for the FP8-QK build
    kc, vc = data["k_cache"], data["v_cache"]  # [P, 1, Hkv, 64] fp8
    k16, v16 = kc.to(torch.float16), vc.to(torch.float16)
    pt = data["page_table"]
    pt_l2 = torch.remainder(pt, POOL).to(torch.int32)
    sl = data["cache_seqlens"]
    sl_half = (sl // 2).to(torch.int32)
    cu_q = data["cu_seqlens_q"]
    maxk = int(sl.max())
    say(f"{label}: inputs in {time.time() - t0:.1f}s, mean ctx {sl.float().mean():.0f}, pool rows {kc.shape[0]}")
    refs = {}
    for b in (0, B - 1):
        for tag, ptab, sls in (("", pt, sl), ("l2", pt_l2, sl), ("half", pt, sl_half)):
            refs[(tag, b)] = exact_ref(qn, kc, vc, ptab, sls, b, L, hq, hkv, scale)

    def mk(kind, o):
        ptab = pt_l2 if o.get("l2") else pt
        sls = sl_half if o.get("half") else sl
        mk_ = int(sls.max())
        if kind == "fa3":
            def fn():
                out = flash_attn_with_kvcache(q=q8, k_cache=kc, v_cache=vc, page_table=ptab, cache_seqlens=sls,
                                              cu_seqlens_q=cu_q, max_seqlen_q=L, softmax_scale=scale, causal=True,
                                              window_size=(-1, -1))
                return (out[0] if isinstance(out, tuple) else out).view(B, L, hq, 64)
            return fn, None
        if kind == "k4":
            k4 = get_k4()
            k4.set_stages(o.get("stages", 3), "full")
            ns4 = min(k4.MAX_SPLITS, k4.num_splits_for(B, hkv, mk_) * o.get("split_mul", 1))

            def fn4():
                return k4.fwd(q8b, kc, vc, ptab, sls, mk_, scale, ns4)
            return fn4, ns4
        if kind == "k2":
            k2 = get_k2()
            k2.set_stages(o.get("stages", 2))
            ns2 = min(k2.MAX_SPLITS, k2.num_splits_for(B, hkv, mk_) * o.get("split_mul", 1))

            def fn2():
                return k2.fwd(q8b, kc, vc, ptab, sls, mk_, scale, ns2)
            return fn2, ns2
        if kind == "k1":
            k1 = get_k1()
            k1.set_cvt(o.get("cvt", "direct"), o.get("map", "own"))
            ns1 = k1.num_splits_for(B, hkv, mk_) * o.get("split_mul", 1)

            def fn1():
                return k1.fwd(q16, kc, vc, ptab, sls, mk_, scale, ns1)
            return fn1, ns1
        set_patch(o.get("patch", {}))
        tile = o["tile"]
        nmb = (L * (hq // hkv) + tile[0] - 1) // tile[0]
        nnb = (mk_ + tile[1] - 1) // tile[1]
        ns = fa4i.num_splits_heuristic(B * hkv * nmb, torch.cuda.get_device_properties(dev).multi_processor_count,
                                       nnb, 128) * o.get("split_mul", 1)

        def fn():
            out, _ = fa4i._flash_attn_fwd(q16, k16, v16, seqused_k=sls, max_seqlen_k=mk_, page_table=ptab,
                                          softmax_scale=scale, causal=True, tile_mn=tile, num_splits=ns,
                                          pack_gqa=True, intra_wg_overlap=o.get("iwo", True))
            return out
        return fn, ns

    row = []
    for name, kind, o in VARIANTS:
        try:
            fn, ns = mk(kind, o)
            t1 = time.time()
            out = fn()
            torch.cuda.synchronize()
            first = time.time() - t1
            tag = "l2" if o.get("l2") else ("half" if o.get("half") else "")
            errs = []
            for b in (0, B - 1):
                r = refs[(tag, b)]
                errs.append(float((out[b].float() - r).norm() / r.norm()))
            for _ in range(2):
                fn()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            t = time_graph(g, args.reps) - floor
            row.append(f"{name} {t:7.2f}us (ns {ns}, first {first:.1f}s, relL2 {max(errs):.2e})")
            del g
        except Exception as ex:  # noqa: BLE001
            row.append(f"{name} ERR {type(ex).__name__}: {str(ex)[:300]}")
            say(chr(10).join("TB| " + ln for ln in traceback.format_exc()[-2500:].splitlines()))
        say(f"BENCH {label} | {row[-1]}")
    say(f"SUMMARY {label} | " + " | ".join(row))
    del data, k16, v16, kc, vc
    torch.cuda.empty_cache()

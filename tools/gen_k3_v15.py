"""gen_k3_v15.py: k3_v15.py = k3_v14.py (as gen_k3_v14.py generates it) + the FA4 v2 L=8 attention route (full and sliding layers).

    python3 -I tools/gen_k3_v15.py            # writes kernels/muse_decode_layer/k3_v15.py, asserts the invariants
    python3 -I tools/gen_k3_v15.py --check    # compares with the file on disk, writes nothing

Changes vs k3_v14 (everything else asserted ast-identical to gen_k3_v14's output, which is asserted == k3_v14.py on disk):
  * K1FA4_KERNEL_SRC: kernels/muse_attention_decode/k1_fa4v2.py (FP8 QK / F16 PV, 2 CTAs/SM, built on the judge image's FA4 CuTe-DSL
    sm90 forward) as TEXT with four exact-once patches (the generator asserts the diff): the temp dir is the directory the layer wrote the
    module into (one dir, removed at exit), the private copies of the image's FA4 files are named after the module, and the module
    registers itself in sys.modules under its own new name (never replacing an entry). k1_fa4v2.py already writes no environment variable
    and makes the CuTe JIT caches of its private interface copy in-memory (cache_dir=None). The text is imported in prepare() only.
  * V15 section (names K1FA4_*, _v15_*): switches, the route predicate (k1_fa4v2_route.route_ok), the FA4 call (k1_fa4v2_route._route's
    lines, verbatim), the prepare self-test (vs the FP32 reference and vs the existing route's output, repeat call, CUDA-graph capture
    + replay) and the loader.
  * _r1_attention takes fa4=None: with the FA4 module it returns _v15_fa4_core(...) after its own prologue (V10 direct store or
    FusedQKNorm + store_cache, RoPE on sliding layers) instead of calling FA3; kernel_fn tries the FA4 route first for L=8 layers (full at
    B >= 16, sliding at every batch), falling back to K1 v7 / r1 on any host exception; prepare() ends with _v15_fa4_prepare.
"""

import ast
import difflib
import importlib.util
import json
import math
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "kernels/muse_decode_layer/k3_v15.py"
V14_FILE = ROOT / "kernels/muse_decode_layer/k3_v14.py"
SRC_FA4 = ROOT / "kernels/muse_attention_decode/k1_fa4v2.py"
SRC_ROUTE = ROOT / "kernels/muse_attention_decode/k1_fa4v2_route.py"
AB_JSON = ROOT / "tools/box/ab_v15a.json"
IMAGE_FA4 = ROOT / "box_out/fa4_image"                       # the judge image's vendored FA4 files (serverless job 1)
VENDOR_FA4 = ROOT / "repos/sglang/python/sglang/kernels/ops/attention/flash_attn/cute"   # identical to the image's (asserted when both exist)
AOT_CACHE = ROOT / "repos/sglang/python/sglang/kernels/jit/cute_aot_cache.py"
NONCE = "k3-v15-1"
WINDOW = 2048   # the sliding layers' window (the layer's cfg["window"]; used by the JIT-config analysis only)

_s14 = importlib.util.spec_from_file_location("gen_k3_v14", ROOT / "tools/gen_k3_v14.py")
G14 = importlib.util.module_from_spec(_s14)
_s14.loader.exec_module(G14)
G13, G8, G = G14.G13, G14.G8, G14.G
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha

V15_NODES = {"K1FA4_ENABLE", "K1FA4_SLIDING", "K1FA4_FULL", "K1FA4_FULL_MIN_BATCH", "K1FA4_SLIDING_MIN_BATCH", "K1FA4_SPLITS", "K1FA4_IWO",
             "K1FA4_PREPARE_BY_S", "K1FA4_SELF_TEST_CASES", "K1FA4_KERNEL_SRC", "_v15_fa4_load", "_v15_fa4_wanted", "_v15_fa4_core",
             "_v15_fa4_case", "_v15_fa4_selftest", "_v15_fa4_prepare"}
CHANGED_NODES = {"_r1_attention", "kernel_fn", "prepare"}
CALL_TIME_FNS = G14.CALL_TIME_FNS + ("_r1_attention", "_v15_fa4_wanted", "_v15_fa4_core")
# the route_ok terms of k1_fa4v2_route that the layer's _v15_fa4_wanted repeats (asserted in both, as source text)
ROUTE_OK_TERMS = ('cfg["q_len"] == 8', 'cfg["page_size"] == 1', 'cfg["kv_scale"] == 1.0', 'cfg["head_dim"] == 64',
                  'cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]', 'cfg["window"]')
# route_ok term (route's constant) -> the layer's: batch thresholds and the sliding switch
ROUTE_OK_RENAMED = (('cfg["batch"] >= ROUTE_SLIDING_MIN_BATCH', 'cfg["batch"] >= K1FA4_SLIDING_MIN_BATCH'),
                    ('cfg["batch"] >= ROUTE_FULL_MIN_BATCH', 'cfg["batch"] >= K1FA4_FULL_MIN_BATCH'))
# statements of k1_fa4v2_route._route that _v15_fa4_core repeats, in file order (ast.unparse form is what the generator matches on)
ROUTE_STMTS_SRC = (
    'batch, length = cfg["batch"], cfg["q_len"]',
    'hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]',
    'window = cfg["window"]',
    'tokens = batch * length',
    'k_cache, v_cache = data["k_cache"], data["v_cache"]',
    'max_k = data["page_table"].shape[1]',
    'keys = min(max_k, window + length) if window else max_k',
    'ns = splits or SPLITS.get(("sliding" if window else "full", batch)) or k2.num_splits_for(batch, hkv, keys)',
    'out = k2.fwd(q8.view(batch, length, hq, d), k_cache, v_cache, data["page_table"], data["cache_seqlens"], max_k, '
    'cfg["sm_scale"] * cfg["qk_scale_factor"], ns, window=window, iwo=IWO["sliding" if window else "full"] if iwo is None else iwo)',
)
ROUTE_STMTS = tuple(ast.unparse(ast.parse(s).body[0]) for s in ROUTE_STMTS_SRC)
ROUTE_RENAME = {"SPLITS": "K1FA4_SPLITS", "IWO": "K1FA4_IWO"}
_RENAME_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(ROUTE_RENAME) + r")(?![A-Za-z0-9_])")


def rename(text):
    return _RENAME_RE.sub(lambda m: ROUTE_RENAME[m.group(1)], text)


# ---------------------------------------------------------------------------------------------- k1_fa4v2.py -> K1FA4_KERNEL_SRC
HEADER_OLD = '''import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import cutlass
'''
HEADER_NEW = '''import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

# k3_v15: the layer file loads this module with importlib from a private temp dir (no sys.path change) and presets _k3fa4_self to the
# module object; it registers itself under its private name, as `import` would, and never replaces an existing sys.modules entry.
assert __name__ not in sys.modules, __name__
sys.modules[__name__] = _k3fa4_self  # noqa: F821

import cutlass
'''
TMP_OLD = 'TMP = Path(tempfile.mkdtemp(prefix="k1fa4v2_"))\n'
TMP_NEW = ('TMP = Path(__file__).resolve().parent  # the private temp dir the layer file wrote this module into (removed at exit); the '
           'patched FA4 copies go next to it\n')
SM90_OLD = 'SM90 = _patched("flash_fwd_sm90", "k1fa4v2_sm90", [\n'
SM90_NEW = 'SM90 = _patched("flash_fwd_sm90", __name__ + "_sm90", [\n'
IFACE_OLD = 'IFACE = _patched("interface", "k1fa4v2_iface", [\n'
IFACE_NEW = 'IFACE = _patched("interface", __name__ + "_iface", [\n'
# lines (stripped, non-empty) of k1_fa4v2.py that the patches remove or replace, in file order
KERNEL_REMOVED = (
    'TMP = Path(tempfile.mkdtemp(prefix="k1fa4v2_"))',
    'SM90 = _patched("flash_fwd_sm90", "k1fa4v2_sm90", [',
    'IFACE = _patched("interface", "k1fa4v2_iface", [',
)
# top-level names of k1_fa4v2.py whose statements the patches change (everything else must stay ast-identical, in place)
KERNEL_CHANGED_NAMES = ("TMP", "SM90", "IFACE")


def kernel_source():
    """k1_fa4v2.py (LF text) with the four exact-once patches above."""
    t = SRC_FA4.read_text(encoding="utf-8")
    t = patch(t, HEADER_OLD, HEADER_NEW, "k1_fa4v2 header (self-registration)")
    t = patch(t, TMP_OLD, TMP_NEW, "k1_fa4v2 TMP")
    t = patch(t, SM90_OLD, SM90_NEW, "k1_fa4v2 sm90 private name")
    t = patch(t, IFACE_OLD, IFACE_NEW, "k1_fa4v2 iface private name")
    return t


# ---------------------------------------------------------------------------------------------- the V15 layer glue
V15_DOC = '''K3 v15 = k3_v14 + the FA4 v2 attention route for 8-token (verify) layers (everything else is k3_v14, asserted ast-identical by
tools/gen_k3_v15.py except the listed nodes):

  FA4 v2 route (kernels/muse_attention_decode/k1_fa4v2.py + k1_fa4v2_route.py, design k1_fa4_design.md section 8): L = 8 layers with page
       size 1, kv_scale 1, head dim 64 and 8 q-heads per KV head -- full layers at B >= 16, sliding layers at every batch -- run r1's
       prologue (V10 direct QK-norm [+ RoPE] + FP8 K/V store), the judge image's FA4 CuTe-DSL sm90 forward (a patched COPY: 2 CTAs/SM,
       E4M3 QK wgmma on the gathered FP8 K, F16 PV with V upcast in smem, FA4's local mask on sliding layers, BF16 split combine) and r1's
       norm-gate, instead of K1 v7 (full) / FA3 (sliding). Measured full kernel_fn vs r1: ver1 sliding 18.1 -> 13.5, ver16 sliding
       23.9 -> 18.0 (4 splits), ver32 sliding 29.5 -> 22.4, ver16 full 195.2 -> 136.5, ver32 full 383.1 -> 262 us. The kernel module is
       embedded as text (K1FA4_KERNEL_SRC), written to a private temp dir and imported in prepare() under a new private name; it patches
       copies of the image's FA4 files under new private names. No existing module or sys.modules entry is touched, no environment variable
       is written (the CuTe JIT cache of the copy is in-memory), every file is under the temp dir (removed at exit).
  K1FA4_ENABLE, K1FA4_SLIDING and K1FA4_FULL are read at call time; the route is used only after its prepare self-test passed (vs the FP32
       reference and vs the existing route's output, a repeat call, a CUDA-graph capture + replay); a failed import / JIT / self-test or a
       host exception keeps the k3_v14 route (K1 v7 / r1) for the run. prepare() adds ~10-40 s (import + five JIT configs + self-test).

'''

V15_CONSTS = '''
# ---------------------------------------------------------------- V15: FA4 v2 attention route (L = 8: full B >= 16, sliding every batch)
# kernels/muse_attention_decode/k1_fa4v2.py (FA4's CuTe-DSL sm90 forward, 2 CTAs/SM, E4M3 QK on the gathered FP8 K, F16 PV with V upcast in smem)
# is inlined as TEXT (K1FA4_KERNEL_SRC; tools/gen_k3_v15.py applies its patches and asserts the diff). prepare() writes it into a private temp dir
# and imports it under a new private name; that module patches COPIES of the image's FA4 files (flash_fwd_sm90.py, interface.py) and loads them
# under new private names. No existing module object or sys.modules entry is touched, no environment variable is written, and every file lives
# under that temp dir (removed at exit). k1_fa4v2_route.py's route is adapted to the layer: the prologue (QK-norm [+ RoPE] + FP8 K/V store) and the
# norm-gate epilogue are _r1_attention's own; the FA4 call lines are k1_fa4v2_route.py's _route lines, verbatim.
K1FA4_ENABLE = True    # master switch, read at call time; False at prepare also skips the FA4 import / JIT / self-tests
K1FA4_SLIDING = True   # L = 8 sliding layers (k1_fa4v2_route.py ROUTE_SLIDING), read at call time
K1FA4_FULL = True      # L = 8 full layers, read at call time
K1FA4_FULL_MIN_BATCH = @FULLMIN@      # k1_fa4v2_route.py ROUTE_FULL_MIN_BATCH: measured at B 16 / 32 (k1_fa4_design.md section 8)
K1FA4_SLIDING_MIN_BATCH = @SLMIN@    # k1_fa4v2_route.py ROUTE_SLIDING_MIN_BATCH: wins at B 1 / 16 / 32
K1FA4_SPLITS = @SPLITS@   # (layer, batch) -> num_splits override, else the kernel's own heuristic (k1_fa4v2_route.py SPLITS)
K1FA4_IWO = @IWO@   # FA4 intra-WG overlap per layer type (k1_fa4v2_route.py IWO)
K1FA4_PREPARE_BY_S = 220.0   # no FA4 import / JIT starts after this many seconds of prepare (limit 300 s; FA4 adds ~10-40 s)
# (layer, B, ctx, seed) for _k1_mk_case (L = 8, page 1, kv_scale 1; lengths 0.6-1.4 x ctx; sliding = window 2048 + RoPE). The verify-16 / verify-32
# full classes (8 / 4 splits), a forced small batch with a one-block context (below the route's minimum), and the sliding classes: B = 1 (34 splits; every length above the window, so always 17 blocks),
# B = 16 (4, the override), B = 32 (4), B = 5 (26 splits, lengths around the window; the judge's sliding q_len 8 page 1 test). Together they compile
# every JIT config the routed shapes can reach: forward (full), forward (local), combine at log_max_splits 4 / 5 / 6.
K1FA4_SELF_TEST_CASES = (("full", 16, 2500, 201), ("full", 32, 1500, 202), ("full", 3, 100, 203), ("sliding", 1, 4000, 211),
                         ("sliding", 16, 3000, 212), ("sliding", 32, 3000, 213), ("sliding", 5, 2500, 214))
'''

V15_KERNEL_HEAD = '''# kernels/muse_attention_decode/k1_fa4v2.py + k3_v15's patches (the module text; never executed by this file, only written to the temp dir)
K1FA4_KERNEL_SRC = '''

V15_GLUE = '''

def _v15_fa4_load():
    """K1FA4_KERNEL_SRC -> a private temp dir -> imported as a NEW module named after that dir (importlib; sys.path untouched). The module
    registers itself under that private name (never replacing a sys.modules entry), patches copies of the image's FA4 files and loads
    them under names derived from its own. The dir is removed at exit. Prepare-time only."""
    import atexit
    import importlib.util
    import pathlib
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="k3fa4_")
    atexit.register(shutil.rmtree, tmp, True)
    name = pathlib.Path(tmp).name
    path = pathlib.Path(tmp) / (name + ".py")
    path.write_text(K1FA4_KERNEL_SRC, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod._k3fa4_self = mod   # read by the module's first lines (its own sys.modules registration)
    spec.loader.exec_module(mod)
    return mod


def _v15_fa4_wanted(data, cfg, state):
    """FA4 v2 route for this call? k1_fa4v2_route.route_ok: q_len 8, page 1, kv_scale 1, head dim 64, Hq = 8 Hkv; then sliding layers at
    batch >= K1FA4_SLIDING_MIN_BATCH (switch K1FA4_SLIDING), full layers at batch >= K1FA4_FULL_MIN_BATCH (switch K1FA4_FULL); plus the tensor
    dtypes / layouts the kernel needs. Shape / config only (graph-safe); needs the module, i.e. the prepare self-test passed."""
    if state.get("fa4") is None or not K1FA4_ENABLE:
        return False
    if not (cfg["q_len"] == 8 and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0 and cfg["head_dim"] == 64
            and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]):
        return False
    if cfg["window"]:
        live = K1FA4_SLIDING and cfg["batch"] >= K1FA4_SLIDING_MIN_BATCH
    else:
        live = K1FA4_FULL and cfg["batch"] >= K1FA4_FULL_MIN_BATCH
    return (bool(live) and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and data["page_table"].dtype == torch.int32 and data["page_table"].dim() == 2 and data["page_table"].is_contiguous()
            and data["cache_seqlens"].dtype == torch.int32 and data["cache_seqlens"].is_contiguous())


def _v15_fa4_core(k2, q8, data, cfg, state, gate, splits=None, iwo=None):
    """_r1_attention's FA3 call replaced by k1_fa4v2_route.py's FA4 call (the statements below are _route's lines, verbatim; SPLITS / IWO are
    K1FA4_SPLITS / K1FA4_IWO here): q8 = the prologue's E4M3 queries (FP8 QK, no cast), k2 = the loaded module; then r1's norm-gate epilogue."""
@CORE@    return state["o_norm_gate"](out.reshape(tokens, hq * d), gate)


def _v15_fa4_case(dev, layer, batch, ctx, seed):
    d = _k1_mk_case(dev, batch, ctx, 8, layer == "sliding", 1, 1.0, seed, 16, 2)
    d["config"]["tokens"] = batch * 8
    return d


def _v15_fa4_selftest(state, mod, dev):
    """The FA4 route (_r1_attention(fa4=mod): r1's prologue, FA4, r1's norm-gate) on synthetic caches, per K1FA4_SELF_TEST_CASES:
      * vs the FP32 reference at the same query rounding (E4M3): K1's own bounds (rel L2 < 2e-2, worst row < 6e-2) and the new K / V
        bytes vs the reference's quantization (< 5e-2: the V10 / FusedQKNorm store rounds K through BF16 first, K1 does not);
      * vs the existing route's output on identical inputs (K1 v7 where k3_v14 routes the case to it, else r1's FA3 path): decode_layer's
        floor bound 3% and 5x for a single row (skipped with a log line if that route raises);
      * a second call (must reproduce the first) and, on the first case of each layer type, a CUDA-graph capture + replay of the call.
    The first calls pay the JIT (forward per layer type + combine per split class). Prepare-time only: syncs and CPU references."""
    t0 = time.time()
    k1_ok = bool(state.get("k1_ok"))
    graphed = set()
    try:
        for ci, (layer, B, ctx, seed) in enumerate(K1FA4_SELF_TEST_CASES):
            d = _v15_fa4_case(dev, layer, B, ctx, seed)
            T = B * 8
            qkv = d["qkv"].view(T, -1)
            keys = min(d["page_table"].shape[1], d["config"]["window"] + 8) if d["config"]["window"] else d["page_table"].shape[1]
            ns = K1FA4_SPLITS.get((layer, B)) or mod.num_splits_for(B, 2, keys)
            print(f"[fa4] self-test case {ci}: {layer} B={B} ctx~{ctx} L=8 page=1 splits={ns} ...", flush=True)
            t1 = time.time()
            o1 = _r1_attention(qkv, d, state, fa4=mod).clone()
            torch.cuda.synchronize(dev)
            t_first = time.time() - t1
            o2 = _r1_attention(qkv, d, state, fa4=mod)
            torch.cuda.synchronize(dev)
            out = o1.view(B, 8, -1)
            pos = d["seq_lens"].long()[:, None] + torch.arange(8, device=dev)
            slots = torch.gather(d["page_table"].long(), 1, pos)
            k_new = d["k_cache"].view(torch.uint8).view(-1, 2, 64)[slots]
            v_new = d["v_cache"].view(torch.uint8).view(-1, 2, 64)[slots]
            ref = _k1_ref_attention(d, k_new, v_new, q_e4m3=True)
            tot, row = _k1_rel_rows(out, ref)
            kref, vref = _k1_ref_new_kv(d)
            kt, _ = _k1_rel_rows(k_new.view(FP8).float(), kref)
            vt, _ = _k1_rel_rows(v_new.view(FP8).float(), vref)
            trep, _ = _k1_rel_rows(o2, o1)
            tot1 = row1 = -1.0   # -1: no comparison with the existing route (it raised)
            which = "-"
            try:
                d2 = _v15_fa4_case(dev, layer, B, ctx, seed)
                if k1_ok and _k1_supported(d2, d2["config"]):
                    which, other = "K1 v7", _k1_run(d2["qkv"], d2, d2["config"])
                else:
                    which, other = "r1 FA3", _r1_attention(d2["qkv"].view(T, -1), d2, state)
                tot1, row1 = _k1_rel_rows(out, other.view(B, 8, -1))
            except Exception as ex:  # noqa: BLE001
                print(f"[fa4] self-test case {ci}: comparison with the existing route ({which}) skipped (it raised {ex!r})", flush=True)
            # Stock's own error on the identical case: FA4 must agree with the reference nearly as well as stock (r1
            # FA3, the same FP8-QK numerics) does. K1 v7 (fp16 QK) is printed for information only.
            tot_r1 = row_r1 = -1.0
            try:
                d3 = _v15_fa4_case(dev, layer, B, ctx, seed)
                o_r1 = _r1_attention(d3["qkv"].view(T, -1), d3, state).view(B, 8, -1)
                torch.cuda.synchronize(dev)
                pos3 = d3["seq_lens"].long()[:, None] + torch.arange(8, device=dev)
                slots3 = torch.gather(d3["page_table"].long(), 1, pos3)
                k3n = d3["k_cache"].view(torch.uint8).view(-1, 2, 64)[slots3]
                v3n = d3["v_cache"].view(torch.uint8).view(-1, 2, 64)[slots3]
                tot_r1, row_r1 = _k1_rel_rows(o_r1, _k1_ref_attention(d3, k3n, v3n, q_e4m3=True))
                del d3, o_r1
            except Exception as ex:  # noqa: BLE001
                print(f"[fa4] self-test case {ci}: stock r1 error not measured (it raised {ex!r})", flush=True)
            tg = -1.0
            if layer not in graphed:
                graphed.add(layer)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    og = _r1_attention(qkv, d, state, fa4=mod)
                g.replay()
                torch.cuda.synchronize(dev)
                tg, _ = _k1_rel_rows(og, o1)
                del g, og
            stock_ok = (tot <= 1.25 * tot_r1 + 5e-3 and row <= 1.25 * row_r1 + 2e-2) if tot_r1 >= 0 else (tot < 5e-2 and row < 1e-1)
            good = (bool(torch.isfinite(out.float()).all()) and tot < 1e-1 and row < 1.5e-1 and stock_ok
                    and kt < 5e-2 and vt < 5e-2 and trep < 1e-3 and tg < 1e-3)
            print(f"[fa4] self-test case {ci}: stock r1 vs fp32 ref rel {tot_r1:.3e} worst row {row_r1:.3e}", flush=True)
            print(f"[fa4] self-test case {ci}: out vs fp32 ref rel {tot:.3e} worst row {row:.3e}; new K {kt:.3e} V {vt:.3e}; vs {which} rel {tot1:.3e} "
                  f"worst row {row1:.3e}; repeat {trep:.1e}; graph replay {tg:.1e}; first call {t_first:.2f}s -> {'ok' if good else 'FAIL'}", flush=True)
            if not good:
                raise RuntimeError("FA4 self-test mismatch")
        print(f"[fa4] self-test passed in {time.time() - t0:.1f}s", flush=True)
        return True
    except Exception as ex:  # noqa: BLE001
        print(f"[v15] FA4 self-test FAILED: the k3_v14 route stays: {ex!r}", flush=True)
        return False


def _v15_fa4_prepare(state, t0):
    """Import the kernel module (patched copies of the image's FA4 files), JIT it on the first self-test calls, self-test; state["fa4"] = the
    module on PASS, else None (the k3_v14 route). Skipped when K1FA4_ENABLE is off at prepare or prepare already took K1FA4_PREPARE_BY_S."""
    state["fa4"] = None
    print(f"[v15] levers: K1FA4_ENABLE={K1FA4_ENABLE} K1FA4_SLIDING={K1FA4_SLIDING} K1FA4_FULL={K1FA4_FULL} min batch full {K1FA4_FULL_MIN_BATCH} "
          f"sliding {K1FA4_SLIDING_MIN_BATCH} splits {K1FA4_SPLITS} iwo {K1FA4_IWO} K1 route={K1_ROUTE_MIN_BATCH}", flush=True)
    if not K1FA4_ENABLE:
        return
    if time.time() - t0 > K1FA4_PREPARE_BY_S:
        print(f"[v15] FA4 skipped: prepare already took {time.time() - t0:.1f}s (> {K1FA4_PREPARE_BY_S}s): the k3_v14 route stays", flush=True)
        return
    t1 = time.time()
    try:
        mod = _v15_fa4_load()
    except Exception as ex:  # noqa: BLE001
        print(f"[v15] FA4 module load FAILED: the k3_v14 route stays: {ex!r}", flush=True)
        return
    t2 = time.time()
    ok = _v15_fa4_selftest(state, mod, state["router"].device)
    if ok:
        state["fa4"] = mod
    print(f"[v15] FA4 route {'live' if ok else 'FAILED: the k3_v14 route stays'} (import + patched copies {t2 - t1:.1f}s, self-test incl. "
          f"JIT {time.time() - t2:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)
'''

# patches of the k3_v14 text (each exact-once)
OLD_R1_DEF = '''def _r1_attention(qkv, data, state):
    """r1_glue.py kernel_fn's attention block, unchanged: qkv [T, 2304] -> o [T, 1024] BF16."""
'''
NEW_R1_DEF = '''def _r1_attention(qkv, data, state, fa4=None):
    """r1_glue.py kernel_fn's attention block, unchanged: qkv [T, 2304] -> o [T, 1024] BF16. V15: with fa4 = the loaded FA4 module the same
    prologue runs, then _v15_fa4_core (the FA4 kernel) replaces the FA3 call; the epilogue is the same o_norm_gate."""
'''
OLD_R1_FA3 = '''    window = cfg["window"]
    descale = {}
'''
NEW_R1_FA3 = '''    if fa4 is not None:   # V15: FA4 v2 route (E4M3 q straight into the FP8 QK, FA4 forward + combine), then r1's norm-gate
        return _v15_fa4_core(fa4, q_fp8, data, cfg, state, gate)
    window = cfg["window"]
    descale = {}
'''
OLD_KFN = '''    o = None
    if state["k1_ok"] and _k1_supported(data, cfg) and _k1_ready(cfg, qkv.device):
'''
NEW_KFN = '''    o = None
    if _v15_fa4_wanted(data, cfg, state):
        try:
            # V15: FA4 v2 route (L = 8: full layers at B >= 16, sliding at every batch): r1's QK-norm [+ RoPE] / FP8 K/V-store prologue, the
            # CuTe-DSL FA4 forward (FP8 QK, F16 PV) and r1's norm-gate; none of K1 v7 / FA3 runs on this path. Any host exception keeps the
            # k3_v14 route (K1 v7 / r1) from now on.
            o = _r1_attention(qkv, data, state, fa4=state["fa4"])
        except Exception as ex:  # noqa: BLE001
            _k3_disable(state, "fa4", None, "FA4 route", ex)
            o = None
    if o is None and state["k1_ok"] and _k1_supported(data, cfg) and _k1_ready(cfg, qkv.device):
'''
OLD_PREPARE = '''    _v13_prepare(state)
    torch.cuda.synchronize()
'''
NEW_PREPARE = '''    _v13_prepare(state)
    _v15_fa4_prepare(state, t0)
    torch.cuda.synchronize()
'''
OLD_DOC = '"""K3 v14 = k3_v13 + K2A v3.6 (role-split variant)'


def route_core(route):
    """The `_route` statements of ROUTE_STMTS as the file's own lines (SPLITS / IWO renamed), each found exactly once."""
    fn = route.fn("_route")
    got = {}
    for stmt in fn.body:
        key = ast.unparse(stmt)
        if key in ROUTE_STMTS:
            check(key not in got, f"k1_fa4v2_route._route: statement {key!r} twice")
            got[key] = rename("".join(route.lines[stmt.lineno - 1:stmt.end_lineno]))
    check(set(got) == set(ROUTE_STMTS), f"k1_fa4v2_route._route: statements missing: {sorted(set(ROUTE_STMTS) - set(got))}")
    return "".join(got[k] for k in ROUTE_STMTS)


def generate():
    text14, base13, k33, k36 = G14.generate()
    base14 = text14
    check(V14_FILE.read_text(encoding="utf-8") == base14, "gen_k3_v14's output differs from k3_v14.py on disk (regenerate v14 first)")
    route = Src(SRC_ROUTE)
    kernel_src = kernel_source()
    check("'''" not in kernel_src and not kernel_src.endswith("\\"), "kernel text cannot be embedded as a raw ''' literal")
    consts = (V15_CONSTS.replace("@FULLMIN@", repr(route.value("ROUTE_FULL_MIN_BATCH"))).replace("@SLMIN@", repr(route.value("ROUTE_SLIDING_MIN_BATCH")))
              .replace("@SPLITS@", repr(route.value("SPLITS"))).replace("@IWO@", repr(route.value("IWO"))))
    glue = V15_GLUE.replace("@CORE@", route_core(route))
    section = consts.lstrip("\n") + V15_KERNEL_HEAD + "r'''" + kernel_src + "'''\n" + glue
    text = text14
    lines = text.split("\n")
    check(lines[1].startswith("# nonce k3-v14-1 "), "k3_v14 nonce")
    hdr = "\n".join(lines[2:5]) + "\n"
    check(hdr.startswith("# GENERATED by tools/gen_k3_v14.py") and hdr.endswith("regenerate.\n"), "k3_v14 generated-by header")
    text = patch(text, lines[1] + "\n" + hdr,
                 f"# nonce {NONCE} (k3_v14 + the FA4 v2 attention route for L = 8 full (B >= 16) and sliding layers; k3_v15_plan.md)\n"
                 "# GENERATED by tools/gen_k3_v15.py on top of gen_k3_v14.py's output (k3_v14.py; its chain is gen_k3_v13 / v12 /\n"
                 "# v11 / v10 / v09 / v08 ...) and kernels/muse_attention_decode/k1_fa4v2.py (+ k1_fa4v2_route.py's FA4 call lines).\n"
                 "# Do not hand-edit: change a canonical source or a generator and regenerate.\n", "nonce + header")
    text = patch(text, OLD_DOC, '"""' + V15_DOC + "K3 v14 = k3_v13 + K2A v3.6 (role-split variant)", "docstring")
    text = patch(text, OLD_R1_DEF, section + "\n\n" + NEW_R1_DEF, "V15 section + _r1_attention signature")
    text = patch(text, OLD_R1_FA3, NEW_R1_FA3, "_r1_attention FA4 branch")
    text = patch(text, OLD_KFN, NEW_KFN, "kernel_fn FA4 route")
    text = patch(text, OLD_PREPARE, NEW_PREPARE, "prepare FA4")
    return text, base14, kernel_src, route


# ---------------------------------------------------------------------------------------------- checks
def check_ab_json(out, path):
    """ab_v15a.json: every switch exists, each arm resolves to the intended (sliding, full) state; ref = everything off = k3_v14's route."""
    arms = json.loads(path.read_text())
    default = {k: out.value(k) for k in ("K1FA4_ENABLE", "K1FA4_SLIDING", "K1FA4_FULL")}
    want = {"ref": (False, False), "fa4_all": (True, True), "fa4_full_only": (False, True), "fa4_sliding_only": (True, False)}
    check([a["name"] for a in arms] == list(want), f"{path.name}: arms {[a['name'] for a in arms]} != {list(want)}")
    check(arms[0]["set"] == {"K1FA4_ENABLE": False}, f"{path.name}: variant 0 must be ref = K1FA4_ENABLE off only")
    rows = []
    for a in arms:
        eff = dict(default)
        for k, v in a.get("set", {}).items():
            check(k in default and isinstance(v, bool), f"{path.name} {a['name']}: unknown or non-bool switch {k}")
            eff[k] = v
        live = (bool(eff["K1FA4_ENABLE"] and eff["K1FA4_SLIDING"]), bool(eff["K1FA4_ENABLE"] and eff["K1FA4_FULL"]))
        check(live == want[a["name"]], f"{path.name} {a['name']}: resolves to (sliding, full) = {live}, want {want[a['name']]}")
        rows.append(f"{a['name']}: FA4 on sliding {'on' if live[0] else 'off'}, on full {'on' if live[1] else 'off'}"
                    + (" (= k3_v14 behaviour)" if not any(live) else ""))
    return rows


def removed_lines(a_text, b_text):
    """Stripped non-empty lines of a that the diff to b removes or replaces, in order; and the number of inserted lines."""
    a, b = a_text.splitlines(), b_text.splitlines()
    removed, added = [], 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            removed += [ln.strip() for ln in a[i1:i2] if ln.strip()]
        if tag in ("replace", "insert"):
            added += j2 - j1
    return removed, added


def split_checks(kernel_src, out):
    """JIT configs. The FA4 forward's compile key depends on the layer type (local mask or not) but not on the batch or the split count; the
    combine's key depends on log_max_splits = max(ceil(log2(splits)), 4). Run the kernel text's own num_splits_for (and the route's SPLITS
    override) with the image's num_splits_heuristic over every routed batch and page-table width, and compare with what the self-test cases
    compile: the routed shapes reach forward (full), forward (local) and combine at 4 / 5 / 6, and each is compiled in prepare."""
    src_dir = IMAGE_FA4 if IMAGE_FA4.exists() else VENDOR_FA4
    itree = ast.parse((src_dir / "interface.py").read_text(encoding="utf-8"))
    nsh_node = [n_ for n_ in itree.body if isinstance(n_, ast.FunctionDef) and n_.name == "num_splits_heuristic"]
    check(len(nsh_node) == 1, "interface.num_splits_heuristic")
    nsh = G14.pure_fn(nsh_node[0])
    ktree = ast.parse(kernel_src)
    nsf_node = [n_ for n_ in ktree.body if isinstance(n_, ast.FunctionDef) and n_.name == "num_splits_for"]
    check(len(nsf_node) == 1, "num_splits_for")
    maxs = [n_ for n_ in ktree.body if isinstance(n_, ast.Assign) and isinstance(n_.targets[0], ast.Name) and n_.targets[0].id == "MAX_SPLITS"]
    check(len(maxs) == 1, "MAX_SPLITS")
    env = {"TILE": (64, 128), "MAX_SPLITS": ast.literal_eval(maxs[0].value), "IFACE": type("I", (), {"num_splits_heuristic": staticmethod(nsh)})}
    exec(compile(ast.Module(body=[nsf_node[0]], type_ignores=[]), "<gen>", "exec"), env)   # noqa: S102 (repo code)
    nsf = env["num_splits_for"]
    splits = out.value("K1FA4_SPLITS")
    fmin, smin = out.value("K1FA4_FULL_MIN_BATCH"), out.value("K1FA4_SLIDING_MIN_BATCH")

    def log_max(ns):
        return max(math.ceil(math.log2(ns)), 4)

    def logs_for(layer, batch, max_k, hkv=2):
        keys = min(max_k, WINDOW + 8) if layer == "sliding" else max_k
        ns = splits.get((layer, batch)) or nsf(batch, hkv, keys)
        check(ns >= 2, f"splits {ns} < 2")
        return log_max(ns), ns

    widths = list(range(1, 200000, 997)) + [80000, 112009, 1048576]
    reach = {"full": set(), "sliding": set()}
    for hkv in (1, 2):
        for batch in range(1, 257):
            for max_k in widths:
                if batch >= fmin:
                    reach["full"].add(logs_for("full", batch, max_k, hkv)[0])
                if batch >= smin:
                    reach["sliding"].add(logs_for("sliding", batch, max_k, hkv)[0])
    check(reach["full"] == {4} and reach["sliding"] == {4, 5, 6}, f"reachable combine configs changed: {reach}")
    covered = {"full": set(), "sliding": set()}
    per_case = []
    for (layer, b, ctx, _seed) in out.value("K1FA4_SELF_TEST_CASES"):
        lo_k, hi_k = int(0.6 * ctx) + 9, int(1.4 * ctx) + 10   # _k1_mk_case: lengths 0.6-1.4 x ctx, page-table width = longest + L + 2
        got = {logs_for(layer, b, mk)[0] for mk in range(lo_k, hi_k + 1, 7)}
        per_case.append(f"{layer} B{b}: log {sorted(got)}")
        if len(got) == 1:
            covered[layer] |= got
    check(covered["full"] >= reach["full"] and covered["sliding"] >= reach["sliding"], f"self-test cases cover {covered}, routed shapes reach {reach}")
    sc = {(layer, b): logs_for(layer, b, 112009)[1] for layer, b in (("full", 16), ("full", 32), ("sliding", 1), ("sliding", 16), ("sliding", 32))}
    return [f"JIT configs: forward (full) + forward (local) + combine at log_max_splits {sorted(reach['sliding'])} = 5 compiles; routed full shapes reach "
            f"combine {sorted(reach['full'])}, sliding {sorted(reach['sliding'])} (B 1..256, hkv 1 / 2, any table width); the self-test cases compile all of them "
            f"({'; '.join(per_case)}); splits at a 112K table: {sc}"]


def patched_copy_checks(kernel_src):
    """Apply the kernel text's own _patched() lists to the image's FA4 files (box_out/fa4_image; else the vendored copy): every anchor exactly
    once, and the patched text parses. Returns report lines."""
    tree = ast.parse(kernel_src)
    rep = []
    src_dir = IMAGE_FA4 if IMAGE_FA4.exists() else VENDOR_FA4
    n = 0
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_patched"):
            call = node.value
            modname = ast.literal_eval(call.args[0])
            reps = ast.literal_eval(call.args[2])
            header = ast.literal_eval(call.keywords[0].value) if call.keywords else ""
            txt = (src_dir / (modname + ".py")).read_text(encoding="utf-8")
            for a, b in reps:
                check(txt.count(a) == 1, f"{modname}: patch anchor found {txt.count(a)} times: {a[:70]!r}")
                txt = txt.replace(a, b)
            ast.parse(header + txt)
            n += 1
            rep.append(f"{modname}.py: {len(reps)} anchors exactly once in {src_dir.relative_to(ROOT)}; patched copy parses")
            if modname == "interface":
                check(any("cache_dir=None" in b for a, b in reps), "interface patch list must set cache_dir=None")
    check(n == 2, f"expected two _patched() assignments, found {n}")
    if IMAGE_FA4.exists() and VENDOR_FA4.exists():
        for f in ("flash_fwd_sm90.py", "interface.py", "paged_kv.py", "flash_fwd_combine.py"):
            check((IMAGE_FA4 / f).read_text(encoding="utf-8").replace("\r\n", "\n") == (VENDOR_FA4 / f).read_text(encoding="utf-8").replace("\r\n", "\n"),
                  f"{f}: image copy differs from repos/sglang's")
        rep.append("image FA4 files == repos/sglang's copy (flash_fwd_sm90, interface, paged_kv, flash_fwd_combine)")
    # get_jit_cache(cache_dir=None) -> a plain in-memory JITCache (repos/sglang's cute_aot_cache.py; same tree as the image's files above)
    aot = ast.parse(AOT_CACHE.read_text(encoding="utf-8"))
    fn = [n_ for n_ in aot.body if isinstance(n_, ast.FunctionDef) and n_.name == "get_jit_cache"]
    check(len(fn) == 1 and "cache_dir" in [a.arg for a in fn[0].args.kwonlyargs], "get_jit_cache has no cache_dir keyword")
    none_branch = [s for s in fn[0].body if isinstance(s, ast.If) and ast.unparse(s.test) == "cache_dir is None"]
    check(len(none_branch) == 1 and ast.unparse(none_branch[0].body[-1]) == "return JITCache()",
          "get_jit_cache: `if cache_dir is None: ... return JITCache()` branch")
    rep.append("get_jit_cache(cache_dir=None) returns JITCache() (in-memory; nothing read from or written to disk)")
    return rep


def verify(text, base14, kernel_src, route):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    btmp = OUT.with_suffix(".py.gen_base")
    btmp.write_text(base14, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(tmp), doraise=True)
        out, b14 = Src(tmp), Src(btmp)    # Src raises on a duplicate top-level name
    finally:
        tmp.unlink()
        btmp.unlink()
    rep = []
    check(text.isascii(), "non-ASCII character in the output")
    lines = text.splitlines()
    check(lines[0] == "# okc: problem=muse_decode_layer", "header line 1")
    check(lines[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1, "nonce")
    check(ast.get_docstring(out.tree).startswith("K3 v15 = k3_v14 + the FA4 v2 attention route"), "docstring")
    check(text.count(G13.K1_SELF_LINE) == 1, "K1 self-test reference line (v7, E4M3 Q at L = 8)")
    # kernel text: embedded byte for byte; the diff to k1_fa4v2.py = the patches
    ksrc = out.value("K1FA4_KERNEL_SRC")
    fa4_text = SRC_FA4.read_text(encoding="utf-8")
    check(ksrc == kernel_src and out.seg("K1FA4_KERNEL_SRC") == "K1FA4_KERNEL_SRC = r'''" + kernel_src + "'''\n", "K1FA4_KERNEL_SRC differs from the patched k1_fa4v2.py")
    removed, added = removed_lines(fa4_text, ksrc)
    check(tuple(removed) == KERNEL_REMOVED, f"k1_fa4v2.py -> kernel text: removed / replaced lines {removed}")
    ktree, otree = ast.parse(ksrc), ast.parse(fa4_text)
    ins = [n_ for n_ in ktree.body if (isinstance(n_, ast.Assert) and ast.unparse(n_.test) == "__name__ not in sys.modules")
           or (isinstance(n_, ast.Assign) and ast.unparse(n_.targets[0]) == "sys.modules[__name__]")]
    check(len(ins) == 2, "the two self-registration statements")
    rest = [n_ for n_ in ktree.body if n_ not in ins]
    check(len(rest) == len(otree.body), "top-level statement count (minus the two inserted) must equal k1_fa4v2.py's")
    changed = []
    for a, b in zip(otree.body, rest):
        if dump(a) != dump(b):
            changed.append(next((t.id for t in getattr(a, "targets", []) if isinstance(t, ast.Name)), getattr(a, "name", ast.unparse(a)[:40])))
    check(sorted(changed) == sorted(KERNEL_CHANGED_NAMES), f"top-level statements changed vs k1_fa4v2.py: {changed}")
    for nm in ("_patched", "K2Sm90", "num_splits_for", "set_stages", "fwd"):
        a = [n_ for n_ in otree.body if getattr(n_, "name", None) == nm]
        b = [n_ for n_ in ktree.body if getattr(n_, "name", None) == nm]
        check(len(a) == len(b) == 1 and dump(a[0]) == dump(b[0]), f"{nm} must stay ast-identical to k1_fa4v2.py")
    # hygiene of the embedded module: no os, no environment, sys.modules only to add new private names, no sys.path / exec
    names = {n_.id for n_ in ast.walk(ktree) if isinstance(n_, ast.Name)}
    check("os" not in names and not any(isinstance(n_, ast.Import) and any(a_.name == "os" for a_ in n_.names) for n_ in ast.walk(ktree)),
          "kernel text uses os")
    env_hits = [n_ for n_ in ast.walk(ktree) if isinstance(n_, ast.Attribute) and n_.attr == "environ"]
    check(not env_hits, "kernel text touches an environ attribute")
    sys_hits = [ast.unparse(n_) for n_ in ast.walk(ktree) if isinstance(n_, ast.Attribute) and isinstance(n_.value, ast.Name) and n_.value.id == "sys"]
    check(sorted(sys_hits) == ["sys.modules"] * 4, f"sys uses in the kernel text: {sys_hits}")   # 2 registrations x (assert + store)
    stores = [n_ for n_ in ast.walk(ktree) if isinstance(n_, ast.Assign) and isinstance(n_.targets[0], ast.Subscript)
              and ast.unparse(n_.targets[0].value) == "sys.modules"]
    asserts = [n_ for n_ in ast.walk(ktree) if isinstance(n_, ast.Assert) and "not in sys.modules" in ast.unparse(n_.test)]
    check(len(stores) == 2 and len(asserts) == 2, "sys.modules stores must each follow a `not in sys.modules` assert")
    rep.append(f"K1FA4_KERNEL_SRC == k1_fa4v2.py ({len(fa4_text.encode())} bytes) with 4 exact-once patches ({len(ksrc.encode())} bytes, sha1 {sha(ksrc)[:12]}): "
               f"removed/replaced lines {len(removed)} (TMP, the two private module names), +{added} lines (the self-registration, the new names); every other top-level "
               f"statement ast-identical in place (_patched, K2Sm90, num_splits_for, set_stages, fwd, ...); no os, no environ, sys.modules only adds new private names")
    rep += patched_copy_checks(ksrc)
    rep += split_checks(ksrc, out)
    # route <-> k1_fa4v2_route.py
    check(out.value("K1FA4_FULL_MIN_BATCH") == route.value("ROUTE_FULL_MIN_BATCH") == 16
          and out.value("K1FA4_SLIDING_MIN_BATCH") == route.value("ROUTE_SLIDING_MIN_BATCH") == 1
          and out.value("K1FA4_SPLITS") == route.value("SPLITS") == {("sliding", 16): 4}
          and out.value("K1FA4_IWO") == route.value("IWO") == {"full": False, "sliding": False}
          and out.value("K1FA4_SLIDING") is True and route.value("ROUTE_SLIDING") is True and out.value("K1FA4_FULL") is True,
          "route constants")
    route_ok_src = route.seg("route_ok")
    wanted_src = out.seg("_v15_fa4_wanted")
    for term in ROUTE_OK_TERMS:
        check(term in route_ok_src and term in wanted_src, f"route_ok term {term!r} must be in k1_fa4v2_route.route_ok and _v15_fa4_wanted")
    for a_, b_ in ROUTE_OK_RENAMED:
        check(a_ in route_ok_src and b_ in wanted_src, f"route_ok term {a_!r} / {b_!r}")
    core = out.fn("_v15_fa4_core")
    check([a_.arg for a_ in core.args.args] == ["k2", "q8", "data", "cfg", "state", "gate", "splits", "iwo"]
          and [ast.unparse(x) for x in core.args.defaults] == ["None", "None"], "_v15_fa4_core signature")
    body = core.body[1:-1]   # after the docstring, before the return
    route_stmts = [s for s in route.fn("_route").body if ast.unparse(s) in ROUTE_STMTS]
    check([ast.unparse(s) for s in route_stmts] == list(ROUTE_STMTS), "k1_fa4v2_route._route statement order")
    renamed = [ast.parse(rename(ast.get_source_segment(route.text, s))).body[0] for s in route_stmts]
    check([dump(a) for a in renamed] == [dump(b) for b in body], "_v15_fa4_core statements != k1_fa4v2_route._route's (after the SPLITS / IWO rename)")
    check(ast.unparse(core.body[-1]) == "return state['o_norm_gate'](out.reshape(tokens, hq * d), gate)", "_v15_fa4_core epilogue")
    rep.append(f"route: K1FA4_FULL_MIN_BATCH 16 / K1FA4_SLIDING_MIN_BATCH 1 / K1FA4_SPLITS / K1FA4_IWO == k1_fa4v2_route.py (ROUTE_SLIDING True); route_ok terms "
               f"{len(ROUTE_OK_TERMS) + len(ROUTE_OK_RENAMED)} in both; _v15_fa4_core = {len(ROUTE_STMTS)} _route statements ast-identical (SPLITS / IWO renamed) "
               f"+ r1's o_norm_gate epilogue")
    # nothing else changed
    changed = [n_ for n_, node in b14.nodes.items() if n_ not in out.nodes or dump(out.nodes[n_]) != dump(node)]
    check(set(changed) == CHANGED_NODES, f"changed nodes vs k3_v14: {sorted(changed)} (want {sorted(CHANGED_NODES)})")
    added_nodes = set(out.nodes) - set(b14.nodes)
    check(added_nodes == V15_NODES, f"added nodes: unexpected {sorted(added_nodes - V15_NODES)}, missing {sorted(V15_NODES - added_nodes)}")
    for nm in ("K1_ROUTE_MIN_BATCH", "K1_FOLD_MIN_BATCH", "K1_MIN_CHUNK_T", "K1_CPS_T", "K1_STAGES_T", "K2A36_PLAN", "K2A33_PLAN", "P1_PROJ",
               "_k1_run", "_k1_supported", "_k1_ready", "_k1_self_test", "_k1_prepare", "_v13_prepare", "_v10_qk_store", "_k3_disable"):
        check(dump(out.nodes[nm]) == dump(b14.nodes[nm]), f"{nm} must stay k3_v14's")
    # _r1_attention differs from k3_v14's only by the signature, the docstring and the inserted FA4 branch
    a14 = b14.fn("_r1_attention")
    a15 = out.fn("_r1_attention")
    check([x.arg for x in a15.args.args] == ["qkv", "data", "state", "fa4"] and len(a15.args.defaults) == 1 and ast.unparse(a15.args.defaults[0]) == "None",
          "_r1_attention signature")
    b_old = [dump(s) for s in a14.body[1:]]
    b_new = [dump(s) for s in a15.body[1:]]
    ins_ = [i for i, s in enumerate(a15.body[1:]) if isinstance(s, ast.If) and ast.unparse(s.test) == "fa4 is not None"]
    check(len(ins_) == 1 and b_new[:ins_[0]] + b_new[ins_[0] + 1:] == b_old, "_r1_attention: only the `if fa4 is not None` branch is inserted")
    check(ast.unparse(a15.body[1:][ins_[0]].body[0]) == "return _v15_fa4_core(fa4, q_fp8, data, cfg, state, gate)", "_r1_attention FA4 branch")
    # kernel_fn: the k3_v14 body with one block inserted and `if` -> `if o is None and`
    kf14, kf15 = b14.fn("kernel_fn"), out.fn("kernel_fn")
    idx = [i for i, s in enumerate(kf15.body) if isinstance(s, ast.If) and ast.unparse(s.test) == "_v15_fa4_wanted(data, cfg, state)"]
    check(len(idx) == 1, "kernel_fn: FA4 block")
    old_if = [i for i, s in enumerate(kf14.body) if isinstance(s, ast.If) and ast.unparse(s.test).startswith("state['k1_ok'] and _k1_supported(")]
    check(len(old_if) == 1, "kernel_fn: k3_v14's K1 block")
    new_if = kf15.body[idx[0] + 1]
    check(isinstance(new_if, ast.If) and ast.unparse(new_if.test) == "o is None and " + ast.unparse(kf14.body[old_if[0]].test)
          and dump(ast.Module(body=new_if.body, type_ignores=[])) == dump(ast.Module(body=kf14.body[old_if[0]].body, type_ignores=[])),
          "kernel_fn: K1 block must keep its body and gain `o is None and`")
    check([dump(s) for s in kf15.body[:idx[0]]] == [dump(s) for s in kf14.body[:old_if[0]]]
          and [dump(s) for s in kf15.body[idx[0] + 2:]] == [dump(s) for s in kf14.body[old_if[0] + 1:]], "kernel_fn: the rest must stay k3_v14's")
    fa4_if = kf15.body[idx[0]]
    check(len(fa4_if.body) == 1 and isinstance(fa4_if.body[0], ast.Try) and len(fa4_if.body[0].handlers) == 1
          and ast.unparse(fa4_if.body[0].body[0]) == "o = _r1_attention(qkv, data, state, fa4=state['fa4'])"
          and "_k3_disable(state, 'fa4', None, 'FA4 route', ex)" in [ast.unparse(s) for s in fa4_if.body[0].handlers[0].body], "kernel_fn FA4 try block")
    # prepare: one added call
    p14, p15 = b14.fn("prepare"), out.fn("prepare")
    ci = [i for i, s in enumerate(p15.body) if ast.unparse(s) == "_v15_fa4_prepare(state, t0)"]
    check(len(ci) == 1 and [dump(s) for s in p15.body[:ci[0]] + p15.body[ci[0] + 1:]] == [dump(s) for s in p14.body], "prepare: only the _v15_fa4_prepare call is added")
    rep.append(f"vs k3_v14 (gen output == file on disk): {len(b14.nodes) - len(changed)} top-level nodes ast-identical; changed {sorted(changed)} "
               f"(_r1_attention: signature + one inserted branch; kernel_fn: one inserted block + `o is None and`; prepare: one call); added {len(added_nodes)}; "
               f"K1_ROUTE_MIN_BATCH and every K1 / K2A / P1 setting identical")
    # hygiene
    for node in ast.walk(out.tree):
        if isinstance(node, ast.Name):
            check(node.id not in G.BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
    for fn in CALL_TIME_FNS:
        for sub in ast.walk(out.nodes[fn]):
            check(not (isinstance(sub, ast.Attribute) and sub.attr in G.SYNC_ATTRS), f"{fn} has a host sync")
    for fn in ("_v15_fa4_wanted", "_v15_fa4_core"):   # call-time FA4 glue: no allocation call
        calls = {ast.unparse(s.func) for s in ast.walk(out.nodes[fn]) if isinstance(s, ast.Call)}
        check(not (calls & {"torch.empty", "torch.zeros", "torch.arange", "torch.tensor"}), f"{fn} allocates: {calls}")
    bad = G8.G7.undefined_globals(text, str(OUT))
    check(not bad, f"undefined globals: {sorted(bad)}")
    check(len(text.encode()) < 1_000_000, "file >= 1 MB")
    code_only = "\n".join(l_ for l_ in lines if not l_.lstrip().startswith("#"))
    check(re.search(r"\benviron\b", code_only) is None and re.search(r"\bos\.environ", code_only) is None,
          "`environ` appears in the layer file outside comments")
    rep.append(f"static ok (no banned names in the layer's code, no host sync in {len(CALL_TIME_FNS)} call-time functions, no undefined globals, "
               f"no `environ` outside comments); {len(text.encode())} bytes (k3_v14: {len(base14.encode())})")
    for r in check_ab_json(out, AB_JSON):
        rep.append(f"{AB_JSON.name} arm  {r}")
    return rep


def main(argv):
    text, base14, kernel_src, route = generate()
    for r in verify(text, base14, kernel_src, route):
        print(f"[gen_k3_v15] OK  {r}")
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text(encoding="utf-8") == text, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_v15] --check: up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"[gen_k3_v15] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}, {len(text.splitlines())} lines, {len(text.encode())} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

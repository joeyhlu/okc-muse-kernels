"""gen_k3_v16.py: k3_v16.py = k3_v15.py (as gen_k3_v15.py generates it) + the FA4 v5 attention routes (decode and verify, sliding and full layers).

    python3 -I tools/gen_k3_v16.py            # writes kernels/muse_decode_layer/k3_v16.py, asserts the invariants
    python3 -I tools/gen_k3_v16.py --check    # compares with the file on disk, writes nothing

Changes vs k3_v15 (everything else asserted ast-identical to gen_k3_v15's output, which is asserted == k3_v15.py on disk):
  * K1FA4V5_KERNEL_SRC: kernels/muse_attention_decode/k1_fa4v5.py (Triton prologue -> FA4 v2 core (CuTe DSL, FP8 QK / F16 PV, 2 CTAs/SM) ->
    Triton combine + RMS-norm + sigmoid gate, chained with programmatic dependent launch) as TEXT with four exact-once patches (the generator
    asserts the diff): the temp dir is the directory the layer wrote the module into, the private copies of the image's FA4 files are named after
    the module, and the module registers itself in sys.modules under its own new name (never replacing an entry). The module writes no
    environment variable and makes the CuTe JIT caches of its private interface copy in-memory (cache_dir=None). Imported in prepare() only.
  * V16 section (names K1FA4V5_*, _v16_*): switches (master + one per route group), the route predicate (k1_fa4v5.route_ok and
    k1_fa4v5_route.route_ok terms), the v5 call (k1_fa4v5.run's own statements, generated from that file and AST-checked against the emitted
    function), the cache-row-0 helper (k1_fa4v5_route.force_row0, verbatim), the prepare self-test and the loader.
  * kernel_fn tries v5 first (decode sliding all B, decode full B = 1, verify sliding all B, verify full B = 1 and B >= 16), then falls through to
    k3_v15's routes (FA4 v2 for L = 8, K1 v7, r1); prepare() runs the v5 self-test, then k3_v15's FA4 v2 prepare (always: it is the fallback and
    the A/B reference; K1FA4V5_V2_ALWAYS False: only if v5 is not live).
"""

import ast
import importlib.util
import json
import math
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "kernels/muse_decode_layer/k3_v16.py"
V15_FILE = ROOT / "kernels/muse_decode_layer/k3_v15.py"
SRC_FA5 = ROOT / "kernels/muse_attention_decode/k1_fa4v5.py"
SRC_ROUTE5 = ROOT / "kernels/muse_attention_decode/k1_fa4v5_route.py"
AB_JSON = ROOT / "tools/box/ab_v16a.json"
META = ROOT / "problems/muse_decode_layer/meta.json"
NONCE = "k3-v16-1"
WINDOW = 2048   # the sliding layers' window (analysis only)
HQ, HKV = 16, 2   # heads on this GPU (the layer problem's HQ / HKV)

_s15 = importlib.util.spec_from_file_location("gen_k3_v15", ROOT / "tools/gen_k3_v15.py")
G15 = importlib.util.module_from_spec(_s15)
_s15.loader.exec_module(G15)
G14, G13, G8, G = G15.G14, G15.G13, G15.G8, G15.G
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha

V16_NODES = {"K1FA4V5_ENABLE", "K1FA4V5_DECODE_SLIDING", "K1FA4V5_DECODE_FULL_B1", "K1FA4V5_VERIFY_SLIDING", "K1FA4V5_VERIFY_FULL",
             "K1FA4V5_DECODE_FULL_MAX_BATCH", "K1FA4V5_VERIFY_FULL_MIN_BATCH", "K1FA4V5_SPLITS", "K1FA4V5_V2_ALWAYS", "K1FA4V5_PREPARE_BY_S",
             "K1FA4V5_SELF_TEST_CASES", "K1FA4V5_KERNEL_SRC", "_v16_fa4v5_load", "_v16_fa4v5_wanted", "_v16_fa4v5_core", "_v16_force_row0",
             "_v16_fa4v5_case", "_v16_fa4v5_selftest", "_v16_fa4v5_prepare", "_v16_prepare"}
CHANGED_NODES = {"kernel_fn", "prepare"}
CALL_TIME_FNS = G15.CALL_TIME_FNS + ("_v16_fa4v5_wanted", "_v16_fa4v5_core")
GROUPS = ("K1FA4V5_DECODE_SLIDING", "K1FA4V5_DECODE_FULL_B1", "K1FA4V5_VERIFY_SLIDING", "K1FA4V5_VERIFY_FULL")
# route_ok terms of k1_fa4v5.route_ok (kernel text) and k1_fa4v5_route.route_ok (route file) that _v16_fa4v5_wanted repeats (asserted as source text)
MODULE_TERMS = ('cfg["q_len"] in (1, 8)', 'cfg["page_size"] == 1', 'cfg["kv_scale"] == 1.0', 'cfg["head_dim"] == 64',
                'cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]')
ROUTE_TERMS = ('cfg["window"]', 'cfg["q_len"] == 8')
ROUTE_RENAMED = (('cfg["batch"] <= FULL_L1_MAX_BATCH', 'cfg["batch"] <= K1FA4V5_DECODE_FULL_MAX_BATCH'),)

# --------------------------------------------------------------------------- run() -> _v16_fa4v5_core
# statements of k1_fa4v5.run that the layer function drops or rewrites (everything else is taken from the file, renamed, and AST-compared)
RUN_DROPPED = tuple(ast.unparse(ast.parse(s).body[0]) for s in ('qkv = data["qkv"].view(tokens, -1)',))   # the layer passes the projection output itself
RUN_COS_OLD = "cos_sin = _cos_sin(theta, d, device) if theta else qkv"
RUN_COS_NEW_CALL = ("_cos_sin(theta, d, device)", 'state["cos_sin"]')   # r1's own [2^21, 64] float32 cos | sin table (same formula; no 512 MB copy)
RUN_RENAME = {"_k5_prologue": "k5._k5_prologue", "_k5_combine_gate": "k5._k5_combine_gate", "num_splits_for": "k5.num_splits_for",
              "set_pdl": "k5.set_pdl", "fwd_partials": "k5.fwd_partials", "SPLITS": "K1FA4V5_SPLITS"}
_RUN_RE = re.compile(r"(?<![A-Za-z0-9_.])(" + "|".join(RUN_RENAME) + r")(?![A-Za-z0-9_])")


def run_rename(text):
    return _RUN_RE.sub(lambda m: RUN_RENAME[m.group(1)], text)


# --------------------------------------------------------------------------- k1_fa4v5.py -> K1FA4V5_KERNEL_SRC
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

# k3_v16: the layer file loads this module with importlib from a private temp dir (no sys.path change) and presets _k3fa5_self to the
# module object; it registers itself under its private name, as `import` would, and never replaces an existing sys.modules entry.
assert __name__ not in sys.modules, __name__
sys.modules[__name__] = _k3fa5_self  # noqa: F821

import cutlass
'''
TMP_OLD = 'TMP = Path(tempfile.mkdtemp(prefix="k1fa4v5_"))\n'
TMP_NEW = ('TMP = Path(__file__).resolve().parent  # the private temp dir the layer file wrote this module into (removed at exit); the '
           'patched FA4 copies go next to it\n')
SM90_OLD = 'SM90 = _patched("flash_fwd_sm90", "k1fa4v5_sm90", [\n'
SM90_NEW = 'SM90 = _patched("flash_fwd_sm90", __name__ + "_sm90", [\n'
IFACE_OLD = 'IFACE = _patched("interface", "k1fa4v5_iface", [\n'
IFACE_NEW = 'IFACE = _patched("interface", __name__ + "_iface", [\n'
KERNEL_REMOVED = (
    'TMP = Path(tempfile.mkdtemp(prefix="k1fa4v5_"))',
    'SM90 = _patched("flash_fwd_sm90", "k1fa4v5_sm90", [',
    'IFACE = _patched("interface", "k1fa4v5_iface", [',
)
KERNEL_CHANGED_NAMES = ("TMP", "SM90", "IFACE")
# top-level functions of the module that must stay ast-identical to k1_fa4v5.py
KERNEL_SAME_FNS = ("_patched", "K5Sm90", "num_splits_for", "set_stages", "set_pdl", "fwd_partials", "_k5_prologue", "_k5_combine_gate",
                   "_cos_sin", "route_ok", "run")


def kernel_source():
    """k1_fa4v5.py (LF text) with the four exact-once patches above."""
    t = SRC_FA5.read_text(encoding="utf-8")
    t = patch(t, HEADER_OLD, HEADER_NEW, "k1_fa4v5 header (self-registration)")
    t = patch(t, TMP_OLD, TMP_NEW, "k1_fa4v5 TMP")
    t = patch(t, SM90_OLD, SM90_NEW, "k1_fa4v5 sm90 private name")
    t = patch(t, IFACE_OLD, IFACE_NEW, "k1_fa4v5 iface private name")
    return t


# --------------------------------------------------------------------------- the V16 layer glue
V16_DOC = '''K3 v16 = k3_v15 + the FA4 v5 attention routes (everything else is k3_v15, asserted ast-identical by tools/gen_k3_v16.py except the
listed nodes):

  FA4 v5 (kernels/muse_attention_decode/k1_fa4v5.py + k1_fa4v5_route.py, design k1_fa4_design.md section 10): three launches chained by
       programmatic dependent launch -- a Triton prologue (SGLang FusedQKNorm's arithmetic, E4M3 query, FP8 K/V written straight into the cache
       rows incl. row 0; bit-exact with r1's stored bytes), the FA4 v2 core (unchanged CuTe-DSL kernel: FP8 QK, F16 PV, 2 CTAs/SM; PDL wait in
       its loader) and a Triton combine + RMS-norm + sigmoid gate -- instead of K1 (decode) / the FA4 v2 route (verify). Routes (k1_fa4v5_route.route_ok
       terms; page 1, kv_scale 1, head dim 64, Hq = 8 Hkv): decode sliding at every batch, decode full at batch 1 (B16 / B32 stay on K1), verify
       sliding at every batch, verify full at batch 1 and batch >= 16 (B 2..15 unmeasured: unchanged). Measured full kernel_fn (us): dec1 sliding
       12.7 -> 8.2, dec16 15.7 -> 12.5, dec32 19.1 -> 16.2, dec1 full 24.1 -> 15.7, ver1 sliding 13.5 -> 9.9, ver16 18.0 -> 16.4, ver32 22.3 -> 20.6,
       ver1 full 30.4 -> 19.4, ver16 full 137.2 -> 133.7, ver32 full 261.9 -> 261.0. The kernel module is embedded as text (K1FA4V5_KERNEL_SRC),
       written to a private temp dir and imported in prepare() under a new private name, like k3_v15's v2 module; no existing module or
       sys.modules entry is touched, no environment variable is written, the CuTe JIT cache of its FA4 copies is in-memory.
  K1FA4V5_ENABLE and one switch per route group (K1FA4V5_DECODE_SLIDING, _DECODE_FULL_B1, _VERIFY_SLIDING, _VERIFY_FULL) are read at call time. A
       route is used only after the prepare self-test passed per case (FP32 reference with the stock-relative rule: error <= 1.25 x stock r1's own
       + 5e-3; new K/V cache bytes bit-exact vs stock r1's FusedQKNorm + store_cache, incl. a case whose new token lands in cache row 0; a repeat
       call and a CUDA-graph replay identical). Any failure keeps k3_v15's routes (FA4 v2 / K1 v7 / r1) for the run; the FA4 v2 route stays
       prepared (K1FA4V5_V2_ALWAYS) as that fallback and as the A/B reference.

'''

V16_CONSTS = '''
# ---------------------------------------------------------------- V16: FA4 v5 attention (Triton prologue -> FA4 v2 core -> Triton combine + norm-gate, PDL-chained)
# kernels/muse_attention_decode/k1_fa4v5.py is inlined as TEXT (K1FA4V5_KERNEL_SRC; tools/gen_k3_v16.py applies its patches and asserts the diff).
# prepare() writes it into a private temp dir and imports it under a new private name; that module patches COPIES of the image's FA4 files
# (flash_fwd_sm90.py, interface.py) and loads them under new private names. No existing module object or sys.modules entry is touched, no
# environment variable is written, and its files live under that temp dir (removed at exit; Triton's own cache dir is the one every Triton
# kernel in this file already uses). The layer's call is k1_fa4v5.run's own statements (generated from that file), with r1's cos | sin table.
K1FA4V5_ENABLE = True            # master switch, read at call time; False at prepare also skips the v5 import / JIT / self-test
K1FA4V5_DECODE_SLIDING = True    # L = 1, sliding layers, every batch (replaces K1)
K1FA4V5_DECODE_FULL_B1 = True    # L = 1, full layers, batch <= K1FA4V5_DECODE_FULL_MAX_BATCH (replaces K1; B16 / B32 stay on K1: v5 is slower there)
K1FA4V5_VERIFY_SLIDING = True    # L = 8, sliding layers, every batch (replaces the FA4 v2 route)
K1FA4V5_VERIFY_FULL = True       # L = 8, full layers at batch 1 (replaces r1 FA3) and batch >= K1FA4V5_VERIFY_FULL_MIN_BATCH (replaces the FA4 v2 route)
K1FA4V5_DECODE_FULL_MAX_BATCH = @L1MAX@     # k1_fa4v5_route.py FULL_L1_MAX_BATCH
K1FA4V5_VERIFY_FULL_MIN_BATCH = 16    # k3_v15's K1FA4_FULL_MIN_BATCH: full verify at batch 2..15 is unmeasured and keeps today's route
K1FA4V5_SPLITS = @SPLITS@   # (layer, batch) -> num_splits override, else the kernel's own heuristic (k1_fa4v5.py SPLITS)
K1FA4V5_V2_ALWAYS = True   # also prepare k3_v15's FA4 v2 route (its fallback and the A/B reference arm); False: only when v5 is not live
K1FA4V5_PREPARE_BY_S = 200.0   # no v5 import / JIT starts after this many seconds of prepare (limit 300 s)
# (layer, L, B, ctx, seed, cache-row-0) for _k1_mk_case (page 1, kv_scale 1; lengths 0.6-1.4 x ctx; sliding = window 2048 + RoPE). Every routed
# scored class (decode / verify x B 1 / 16 / 32, sliding; decode full B 1; verify full B 1 / 16 / 32) with its scored split count (sliding B1 32 [the
# override], B16 8, B32 4; full B1 128; full B16 8; full B32 4; the generator checks every scored Triton key is compiled), a tiny one-block context, the judge tests' odd classes (sliding L = 1 B = 9, L = 8
# B = 5), and four cases whose first new token is stored in cache row 0 (r1 stages that row, v5 writes it).
K1FA4V5_SELF_TEST_CASES = (("sliding", 1, 1, 4000, 301, False), ("sliding", 1, 1, 10, 302, False), ("sliding", 1, 16, 3000, 303, True),
                           ("sliding", 1, 32, 3000, 304, False), ("full", 1, 1, 20000, 305, True), ("sliding", 8, 1, 4000, 306, False),
                           ("sliding", 8, 16, 3000, 307, False), ("sliding", 8, 32, 3000, 308, True), ("full", 8, 1, 20000, 309, False),
                           ("full", 8, 16, 2500, 310, True), ("full", 8, 32, 1500, 311, False), ("sliding", 1, 9, 5000, 312, False),
                           ("sliding", 8, 5, 2500, 313, False))
'''

V16_KERNEL_HEAD = '''# kernels/muse_attention_decode/k1_fa4v5.py + k3_v16's patches (the module text; never executed by this file, only written to the temp dir)
K1FA4V5_KERNEL_SRC = '''

V16_GLUE = '''

def _v16_fa4v5_load():
    """K1FA4V5_KERNEL_SRC -> a private temp dir -> imported as a NEW module named after that dir (importlib; sys.path untouched). The module
    registers itself under that private name (never replacing a sys.modules entry), patches copies of the image's FA4 files and loads
    them under names derived from its own. The dir is removed at exit. Prepare-time only."""
    import atexit
    import importlib.util
    import pathlib
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="k3fa5_")
    atexit.register(shutil.rmtree, tmp, True)
    name = pathlib.Path(tmp).name
    path = pathlib.Path(tmp) / (name + ".py")
    path.write_text(K1FA4V5_KERNEL_SRC, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod._k3fa5_self = mod   # read by the module's first lines (its own sys.modules registration)
    spec.loader.exec_module(mod)
    return mod


def _v16_fa4v5_wanted(qkv, data, cfg, state):
    """FA4 v5 route for this call? k1_fa4v5.route_ok (q_len 1 or 8, page 1, kv_scale 1, head dim 64, Hq = 8 Hkv) and k1_fa4v5_route.route_ok
    (sliding layers and q_len 8 at every batch, full decode only at batch <= K1FA4V5_DECODE_FULL_MAX_BATCH), narrowed by the layer's measured
    classes (full verify at batch 1 or >= K1FA4V5_VERIFY_FULL_MIN_BATCH) and one switch per route group; plus the tensor dtypes / layouts the
    kernels need. Shape / config only (graph-safe); needs the module, i.e. the prepare self-test passed."""
    if state.get("fa4v5") is None or not K1FA4V5_ENABLE:
        return False
    if not (cfg["q_len"] in (1, 8) and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0 and cfg["head_dim"] == 64
            and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]):
        return False
    if cfg["window"]:
        live = K1FA4V5_VERIFY_SLIDING if cfg["q_len"] == 8 else K1FA4V5_DECODE_SLIDING
    elif cfg["q_len"] == 8:
        live = K1FA4V5_VERIFY_FULL and (cfg["batch"] == 1 or cfg["batch"] >= K1FA4V5_VERIFY_FULL_MIN_BATCH)
    else:
        live = K1FA4V5_DECODE_FULL_B1 and cfg["batch"] <= K1FA4V5_DECODE_FULL_MAX_BATCH
    return (bool(live) and cfg["rope_theta"] in (0.0, ROPE_THETA)
            and qkv.dtype == torch.bfloat16 and qkv.dim() == 2 and qkv.stride(1) == 1
            and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and data["page_table"].dtype == torch.int32 and data["page_table"].dim() == 2 and data["page_table"].is_contiguous()
            and data["cache_seqlens"].dtype == torch.int32 and data["cache_seqlens"].is_contiguous()
            and data["cache_loc"].is_contiguous() and data["positions"].is_contiguous())


def _v16_fa4v5_core(k5, qkv, data, state, splits=None, pdl=True):
    """k1_fa4v5.run's statements (the lines below are generated from that file, names prefixed k5. / K1FA4V5_SPLITS, AST-checked by the
    generator) on the layer's own qkv projection [T, 2304] (q | k | v | gate): Triton prologue (QK-norm [+ RoPE], E4M3 q, FP8 K/V rows
    written at cache_loc), FA4 v2 core with PDL, Triton combine + RMS-norm x sigmoid(gate). cos | sin is r1's table (same formula as v5's own).
    Returns [B, L, Hq * 64] bf16."""
@CORE@


def _v16_force_row0(data):
@FORCE@


def _v16_fa4v5_case(dev, layer, L, batch, ctx, seed, row0):
    d = _k1_mk_case(dev, batch, ctx, L, layer == "sliding", 1, 1.0, seed, 16, 2)
    d["config"]["tokens"] = batch * L
    return _v16_force_row0(d) if row0 else d


def _v16_fa4v5_selftest(state, mod, dev, pdl):
    """The FA4 v5 route (_v16_fa4v5_core) on synthetic caches, per K1FA4V5_SELF_TEST_CASES:
      * vs the FP32 reference at the same query rounding (E4M3) under the stock-relative rule: the output error must be <= 1.25 x stock r1's own
        error on the identical case + 5e-3 (worst row: + 2e-2), with r1's own path = FusedQKNorm + staging + store_cache + FA3 + norm-gate;
      * the whole new K / V cache must equal stock r1's byte for byte (cases with the first new token in cache row 0 included);
      * a second call and, on every case, a CUDA-graph capture + replay of the call must reproduce the first call bit for bit.
    The first calls pay the JIT (forward per layer type, Triton prologue / combine per specialization). Prepare-time only: syncs and CPU references."""
    t0 = time.time()
    stock = dict(state, v10qk=False)   # a copy: r1's staging path (FusedQKNorm.forward_triton + store_cache), then FA3 and o_norm_gate
    try:
        for ci, (layer, L, B, ctx, seed, row0) in enumerate(K1FA4V5_SELF_TEST_CASES):
            d = _v16_fa4v5_case(dev, layer, L, B, ctx, seed, row0)
            T = B * L
            qkv = d["qkv"].view(T, -1)
            keys = min(d["page_table"].shape[1], d["config"]["window"] + L) if d["config"]["window"] else d["page_table"].shape[1]
            ns = K1FA4V5_SPLITS.get((layer, B)) or mod.num_splits_for(B, 2, keys)
            print(f"[fa4v5] self-test case {ci}: {layer} L={L} B={B} ctx~{ctx} page=1 splits={ns}{' cache-row-0' if row0 else ''} ...", flush=True)
            t1 = time.time()
            o1 = _v16_fa4v5_core(mod, qkv, d, state, pdl=pdl).clone()
            torch.cuda.synchronize(dev)
            t_first = time.time() - t1
            o2 = _v16_fa4v5_core(mod, qkv, d, state, pdl=pdl)
            torch.cuda.synchronize(dev)
            out = o1
            pos = d["seq_lens"].long()[:, None] + torch.arange(L, device=dev)
            slots = torch.gather(d["page_table"].long(), 1, pos)
            k_new = d["k_cache"].view(torch.uint8).view(-1, 2, 64)[slots]
            v_new = d["v_cache"].view(torch.uint8).view(-1, 2, 64)[slots]
            ref = _k1_ref_attention(d, k_new, v_new, q_e4m3=True)
            tot, row = _k1_rel_rows(out, ref)
            kref, vref = _k1_ref_new_kv(d)
            kt, _ = _k1_rel_rows(k_new.view(FP8).float(), kref)
            vt, _ = _k1_rel_rows(v_new.view(FP8).float(), vref)
            rep_same = bool(torch.equal(o2, o1))
            # Stock's own error and cache bytes on the identical case: r1's path (the same FP8-query numerics).
            tot_r1 = row_r1 = -1.0
            kv_same = None
            try:
                d3 = _v16_fa4v5_case(dev, layer, L, B, ctx, seed, row0)
                o_r1 = _r1_attention(d3["qkv"].view(T, -1), d3, stock).view(B, L, -1)
                torch.cuda.synchronize(dev)
                tot_r1, row_r1 = _k1_rel_rows(o_r1, ref)
                kv_same = bool(torch.equal(d["k_cache"].view(torch.uint8), d3["k_cache"].view(torch.uint8))
                               and torch.equal(d["v_cache"].view(torch.uint8), d3["v_cache"].view(torch.uint8)))
                if not kv_same:
                    nk = int((d["k_cache"].view(torch.uint8) != d3["k_cache"].view(torch.uint8)).sum())
                    nv = int((d["v_cache"].view(torch.uint8) != d3["v_cache"].view(torch.uint8)).sum())
                    print(f"[fa4v5] self-test case {ci}: cache bytes differ from stock r1: K {nk} V {nv}", flush=True)
                del d3, o_r1
            except Exception as ex:  # noqa: BLE001
                print(f"[fa4v5] self-test case {ci}: stock r1 not run (it raised {ex!r}): absolute bounds only, cache bytes not compared", flush=True)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                og = _v16_fa4v5_core(mod, qkv, d, state, pdl=pdl)
            g.replay()
            torch.cuda.synchronize(dev)
            graph_same = bool(torch.equal(og, o1))
            del g, og
            stock_ok = (tot <= 1.25 * tot_r1 + 5e-3 and row <= 1.25 * row_r1 + 2e-2) if tot_r1 >= 0 else (tot < 5e-2 and row < 1e-1)
            good = (bool(torch.isfinite(out.float()).all()) and tot < 1e-1 and row < 1.5e-1 and stock_ok and kt < 5e-2 and vt < 5e-2
                    and kv_same is not False and rep_same and graph_same)
            print(f"[fa4v5] self-test case {ci}: stock r1 vs fp32 ref rel {tot_r1:.3e} worst row {row_r1:.3e}; cache bytes == stock r1: {kv_same}", flush=True)
            print(f"[fa4v5] self-test case {ci}: out vs fp32 ref rel {tot:.3e} worst row {row:.3e}; new K {kt:.3e} V {vt:.3e}; repeat identical {rep_same}; "
                  f"graph replay identical {graph_same}; first call {t_first:.2f}s -> {'ok' if good else 'FAIL'}", flush=True)
            if not good:
                raise RuntimeError("FA4 v5 self-test mismatch")
            del d, o1, o2
        print(f"[fa4v5] self-test passed in {time.time() - t0:.1f}s", flush=True)
        return True
    except Exception as ex:  # noqa: BLE001
        print(f"[v16] FA4 v5 self-test FAILED: k3_v15's routes stay: {ex!r}", flush=True)
        return False


def _v16_fa4v5_prepare(state, t0):
    """Import the v5 module (patched copies of the image's FA4 files), JIT it on the first self-test calls, self-test; state["fa4v5"] = the
    module on PASS, else None (k3_v15's routes). Skipped when K1FA4V5_ENABLE is off at prepare or prepare already took K1FA4V5_PREPARE_BY_S."""
    state["fa4v5"] = None
    state["fa4v5_pdl"] = False
    print(f"[v16] levers: K1FA4V5_ENABLE={K1FA4V5_ENABLE} decode_sliding={K1FA4V5_DECODE_SLIDING} decode_full_b1={K1FA4V5_DECODE_FULL_B1} "
          f"(max batch {K1FA4V5_DECODE_FULL_MAX_BATCH}) verify_sliding={K1FA4V5_VERIFY_SLIDING} verify_full={K1FA4V5_VERIFY_FULL} "
          f"(batch 1 or >= {K1FA4V5_VERIFY_FULL_MIN_BATCH}) splits {K1FA4V5_SPLITS} v2 always {K1FA4V5_V2_ALWAYS}", flush=True)
    if not K1FA4V5_ENABLE:
        return
    if time.time() - t0 > K1FA4V5_PREPARE_BY_S:
        print(f"[v16] FA4 v5 skipped: prepare already took {time.time() - t0:.1f}s (> {K1FA4V5_PREPARE_BY_S}s): k3_v15's routes stay", flush=True)
        return
    t1 = time.time()
    try:
        mod = _v16_fa4v5_load()
    except Exception as ex:  # noqa: BLE001
        print(f"[v16] FA4 v5 module load FAILED: k3_v15's routes stay: {ex!r}", flush=True)
        return
    t2 = time.time()
    try:
        from sglang.kernels.jit.utils import is_arch_support_pdl
        pdl = bool(is_arch_support_pdl())
    except Exception:  # noqa: BLE001
        pdl = False
    ok = _v16_fa4v5_selftest(state, mod, state["router"].device, pdl)
    if ok:
        state["fa4v5"] = mod
        state["fa4v5_pdl"] = pdl
    print(f"[v16] FA4 v5 route {'live' if ok else 'FAILED: k3_v15 routes stay'} (pdl {pdl}; import + patched copies {t2 - t1:.1f}s, self-test incl. "
          f"JIT {time.time() - t2:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)


def _v16_prepare(state, t0):
    """FA4 v5 first, then k3_v15's FA4 v2 route (always with K1FA4V5_V2_ALWAYS: it is the fallback for every v5 shape that is L = 8 and the A/B
    reference; otherwise only when v5 is not live)."""
    _v16_fa4v5_prepare(state, t0)
    if K1FA4V5_V2_ALWAYS or state["fa4v5"] is None:
        _v15_fa4_prepare(state, t0)
    else:
        state["fa4"] = None
        print("[v16] FA4 v2 route not prepared (v5 is live and K1FA4V5_V2_ALWAYS is False)", flush=True)
'''

# patches of the k3_v15 text (each exact-once)
OLD_R1_DEF = "def _r1_attention(qkv, data, state, fa4=None):\n"
OLD_KFN = '''    o = None
    if _v15_fa4_wanted(data, cfg, state):
'''
NEW_KFN = '''    o = None
    if _v16_fa4v5_wanted(qkv, data, cfg, state):
        try:
            # V16: FA4 v5 (Triton QK-norm [+ RoPE] / E4M3 q / FP8 K/V-store prologue -> FA4 v2 core -> Triton combine + norm-gate, PDL-chained):
            # decode sliding, decode full B = 1, verify sliding, verify full B = 1 and B >= 16. Any host exception keeps k3_v15's routes
            # (FA4 v2 / K1 v7 / r1) from now on.
            o = _v16_fa4v5_core(state["fa4v5"], qkv, data, state, pdl=state["fa4v5_pdl"]).view(tokens, hq * d)
        except Exception as ex:  # noqa: BLE001
            _k3_disable(state, "fa4v5", None, "FA4 v5 route", ex)
            o = None
    if o is None and _v15_fa4_wanted(data, cfg, state):
'''
OLD_PREPARE = '''    _v15_fa4_prepare(state, t0)
    torch.cuda.synchronize()
'''
NEW_PREPARE = '''    _v16_prepare(state, t0)
    torch.cuda.synchronize()
'''
OLD_DOC = '"""K3 v15 = k3_v14 + the FA4 v2 attention route'


def core_source(fa5):
    """The statements of k1_fa4v5.run (after its docstring) as the layer function's body: RUN_DROPPED removed, the cos | sin table swapped, names
    renamed (RUN_RENAME). Returns (text, [statements kept as unparse strings, before the rename])."""
    fn = fa5.fn("run")
    out, kept, dropped = [], [], []
    for st in fn.body[1:]:
        key = ast.unparse(st)
        seg = "".join(fa5.lines[st.lineno - 1:st.end_lineno])
        if key in RUN_DROPPED:
            dropped.append(key)
            continue
        if key == RUN_COS_OLD:
            check(seg.count(RUN_COS_NEW_CALL[0]) == 1, "run(): cos | sin statement")
            seg = seg.replace(*RUN_COS_NEW_CALL)
        kept.append(key)
        out.append(run_rename(seg))
    check(tuple(dropped) == RUN_DROPPED, f"run(): dropped statements {dropped}")
    check(kept and ast.unparse(fn.body[-1]) == "return out", "run(): must end with `return out`")
    return "".join(out), kept


def force_source(route5):
    """k1_fa4v5_route.force_row0 as `_v16_force_row0`: the def line is the layer's, the body lines are the file's, verbatim."""
    node = route5.fn("force_row0")
    seg = "".join(route5.lines[node.lineno - 1:node.end_lineno])
    head = "def force_row0(data):\n"
    check(seg.startswith(head), "force_row0 def line")
    return seg[len(head):]


def generate():
    text15, base14, kernel_src2, route2 = G15.generate()
    check(V15_FILE.read_text(encoding="utf-8") == text15, "gen_k3_v15's output differs from k3_v15.py on disk (regenerate v15 first)")
    fa5, route5 = Src(SRC_FA5), Src(SRC_ROUTE5)
    kernel_src = kernel_source()
    check("'''" not in kernel_src and not kernel_src.endswith("\\"), "kernel text cannot be embedded as a raw ''' literal")
    consts = (V16_CONSTS.replace("@L1MAX@", repr(route5.value("FULL_L1_MAX_BATCH"))).replace("@SPLITS@", repr(fa5.value("SPLITS"))))
    core_body, _ = core_source(fa5)
    force_body = force_source(route5)
    glue = V16_GLUE.replace("@CORE@", core_body.rstrip("\n")).replace("@FORCE@", force_body.rstrip("\n"))
    section = consts.lstrip("\n") + V16_KERNEL_HEAD + "r'''" + kernel_src + "'''\n" + glue
    text = text15
    lines = text.split("\n")
    check(lines[1].startswith("# nonce k3-v15-1 "), "k3_v15 nonce")
    hdr = "\n".join(lines[2:5]) + "\n"
    check(hdr.startswith("# GENERATED by tools/gen_k3_v15.py") and hdr.endswith("regenerate.\n"), "k3_v15 generated-by header")
    text = patch(text, lines[1] + "\n" + hdr,
                 f"# nonce {NONCE} (k3_v15 + the FA4 v5 attention routes: decode sliding / decode full B1 / verify sliding / verify full B1 + B>=16; k3_v16_plan.md)\n"
                 "# GENERATED by tools/gen_k3_v16.py on top of gen_k3_v15.py's output (k3_v15.py; its chain is gen_k3_v14 / v13 /\n"
                 "# v12 / v11 / v10 ...) and kernels/muse_attention_decode/k1_fa4v5.py (+ k1_fa4v5_route.py's route terms and force_row0).\n"
                 "# Do not hand-edit: change a canonical source or a generator and regenerate.\n", "nonce + header")
    text = patch(text, OLD_DOC, '"""' + V16_DOC + "K3 v15 = k3_v14 + the FA4 v2 attention route", "docstring")
    text = patch(text, OLD_R1_DEF, section + "\n\n" + OLD_R1_DEF, "V16 section")
    text = patch(text, OLD_KFN, NEW_KFN, "kernel_fn FA4 v5 route")
    text = patch(text, OLD_PREPARE, NEW_PREPARE, "prepare FA4 v5")
    return text, text15, kernel_src, fa5, route5


# --------------------------------------------------------------------------- checks
def check_ab_json(out, path):
    """ab_v16a.json: every switch exists and is a bool; ref = the master switch off only (k3_v15 behaviour: v2 / K1 / r1 routes); v5_all = defaults;
    one arm per route group = only that group on."""
    arms = json.loads(path.read_text())
    default = {k: out.value(k) for k in ("K1FA4V5_ENABLE",) + GROUPS}
    check(all(v is True for v in default.values()), "switch defaults must all be True")
    want = {"ref": (), "v5_all": GROUPS, "v5_decode_sliding": ("K1FA4V5_DECODE_SLIDING",), "v5_decode_full_b1": ("K1FA4V5_DECODE_FULL_B1",),
            "v5_verify_sliding": ("K1FA4V5_VERIFY_SLIDING",), "v5_verify_full": ("K1FA4V5_VERIFY_FULL",)}
    check([a["name"] for a in arms] == list(want), f"{path.name}: arms {[a['name'] for a in arms]} != {list(want)}")
    check(arms[0]["set"] == {"K1FA4V5_ENABLE": False}, f"{path.name}: variant 0 must be ref = K1FA4V5_ENABLE off only")
    rows = []
    for a in arms:
        eff = dict(default)
        for k, v in a.get("set", {}).items():
            check(k in default and isinstance(v, bool), f"{path.name} {a['name']}: unknown or non-bool switch {k}")
            eff[k] = v
        live = tuple(g for g in GROUPS if eff["K1FA4V5_ENABLE"] and eff[g])
        check(live == want[a["name"]], f"{path.name} {a['name']}: resolves to {live}, want {want[a['name']]}")
        rows.append(f"{a['name']}: v5 groups on = {[g.replace('K1FA4V5_', '').lower() for g in live] or 'none (= k3_v15 behaviour)'}")
    return rows


def make_nsf(kernel_src):
    """The kernel text's own num_splits_for with the image's num_splits_heuristic (pure functions, extracted by AST)."""
    src_dir = G15.IMAGE_FA4 if G15.IMAGE_FA4.exists() else G15.VENDOR_FA4
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
    return env["num_splits_for"]


def routed(layer, L, B, out):
    """The generator's own copy of the route table (k1_fa4v5_route.route_ok + the layer's full-verify batches), for the JIT-config analysis."""
    if layer == "sliding":
        return True
    if L == 8:
        return B == 1 or B >= out.value("K1FA4V5_VERIFY_FULL_MIN_BATCH")
    return B <= out.value("K1FA4V5_DECODE_FULL_MAX_BATCH")


def jit_checks(kernel_src, out):
    """JIT configs the self-test compiles vs the ones the scored shapes reach. FA4 forward: the compile key separates local from full only
    (not batch, split count or q_len; v5 skips FA4's own combine) = 2 compiles. Triton specializes integer arguments on `== 1` and `% 16 == 0`:
    prologue key = (RoPE, BLOCK_M, class of n_rows), combine key = (class of ns / B / L, next_pow2(ns), num_warps). The scored shapes' split
    counts come from the kernel text's own num_splits_for with the image's heuristic at the shapes' exact page-table widths (meta.json +
    reference.generate_input's sizing); every key a routed scored shape reaches must be produced by exactly one deterministic self-test case."""
    nsf = make_nsf(kernel_src)
    splits = out.value("K1FA4V5_SPLITS")

    def cls(v):
        return "1" if v == 1 else ("D" if v % 16 == 0 else "-")

    def pkey(layer, L, B):
        n_rows = B * L * (HQ + 2 * HKV)
        return ("rope" if layer == "sliding" else "norope", 4 if n_rows < 1024 else (8 if n_rows < 8192 else 16), cls(n_rows))

    def ckey(L, B, ns):
        return (cls(ns), cls(B), cls(L), 1 << (ns - 1).bit_length(), 2 if ns <= 8 else 4)

    def ns_for(layer, L, B, width):
        keys = min(width, WINDOW + L) if layer == "sliding" else width
        ns = splits.get((layer, B)) or nsf(B, HKV, keys)
        check(ns >= 2, f"splits {ns} < 2")
        return ns

    meta = json.loads(META.read_text(encoding="utf-8"))
    scored = {}
    for b in meta["benchmarks"]:
        L, B, ctx = b.get("q_len", 1), b["batch"], b["ctx"]
        longest = ctx if b.get("spread", "fixed") == "fixed" else math.ceil(ctx * 1.075)   # reference._longest ("agentic" spread 0.075)
        check(b.get("spread", "fixed") in ("fixed", "agentic"), "unknown spread in meta.json")
        width = longest + L   # page 1: max_pages = longest + q_len
        if routed(b["layer"], L, B, out):
            ns = ns_for(b["layer"], L, B, width)
            scored[(b["layer"], L, B)] = (pkey(b["layer"], L, B), ckey(L, B, ns), ns)
    check(len(scored) == 10, f"routed scored shapes: {sorted(scored)}")
    covered_p, covered_c, per_case = set(), set(), []
    for (layer, L, B, ctx, _seed, _row0) in out.value("K1FA4V5_SELF_TEST_CASES"):
        lo_k, hi_k = int(0.6 * ctx) + L + 1, int(1.4 * ctx) + L + 1   # _k1_mk_case: page-table width = longest length + L + 1
        cks = {ckey(L, B, ns_for(layer, L, B, w)) for w in range(lo_k, hi_k + 1, 7)}
        pk = pkey(layer, L, B)
        per_case.append(f"{layer} L{L} B{B}: {'1' if len(cks) == 1 else len(cks)} combine key")
        if len(cks) == 1:
            covered_c |= cks
        covered_p.add(pk)
    for shape, (pk, ck, ns) in sorted(scored.items()):
        check(pk in covered_p, f"scored shape {shape}: prologue key {pk} is not compiled by any self-test case")
        check(ck in covered_c, f"scored shape {shape}: combine key {ck} (splits {ns}) is not compiled by a deterministic self-test case")
    sc = {f"{l[0]}{L}B{B}": ns for (l, L, B), (_p, _c, ns) in sorted(scored.items())}
    return [f"JIT configs (estimate, Triton's int specialization model): FA4 forward x2 (full, local; independent of B, L and splits); Triton prologue "
            f"{len(covered_p)} keys {sorted(covered_p)}; combine {len(covered_c)} keys; every one of the 10 routed scored shapes' keys is compiled by the self-test "
            f"(scored split counts {sc}); the judge's capture() warms any other shape eagerly first"]


def pdl_audit(ktree, out):
    """PDL staleness audit by source (ld.global.nc / __restrict__ / plain loads of a predecessor's output issued before griddepcontrol.wait read stale data).
    (1) FA4 core, secondary of the prologue: in K5Sm90.load (the only code that reads Q, K, V, the new cache rows, the page table) the wait is the first statement
        after the thread-id setup, before any name of a global tensor / the paged-KV manager / a smem stage is used; the kernel's other pre-wait code is the
        TMA-descriptor prefetch, barrier init and the consumers' read of cache_seqlens (a layer input no kernel of the layer writes).
    (2) combine + norm-gate, secondary of the FA4 core: gdc_wait is its first statement (before the LSE, partial and gate loads). (3) prologue, secondary of the
        qkv projection: gdc_wait is its first statement, then the trigger, then every load. Both Triton kernels take USE_PDL from the layer's pdl flag and launch
        with launch_pdl only then (the statements of run(), asserted ast-identical above)."""
    forbidden = {"mQ", "mK", "mV", "mPageTable", "pkm", "sQ", "sK", "sV", "sVr", "tma_atom_Q", "tma_atom_K", "tma_atom_V", "gmem_tiled_copy_Q",
                 "PagedKVManager", "pipeline_k", "pipeline_v", "pipeline_q", "load_Q", "gQ", "mQ_cur"}
    cls = [n_ for n_ in ktree.body if isinstance(n_, ast.ClassDef) and n_.name == "K5Sm90"]
    check(len(cls) == 1, "K5Sm90")
    load = [n_ for n_ in cls[0].body if isinstance(n_, ast.FunctionDef) and n_.name == "load"]
    check(len(load) == 1, "K5Sm90.load")
    body = load[0].body
    idx = [i for i, s in enumerate(body) if isinstance(s, ast.If) and ast.unparse(s.test) == "const_expr(SM90._K2['pdl'])"
           and ast.unparse(s.body[0]) == "cute.arch.griddepcontrol_wait()"]
    check(len(idx) == 1 and idx[0] <= 3, "K5Sm90.load: griddepcontrol_wait must be among the first statements")
    pre = {n_.id for s in body[:idx[0]] for n_ in ast.walk(s) if isinstance(n_, ast.Name)}
    check(not (pre & forbidden), f"K5Sm90.load: names used before the wait: {sorted(pre & forbidden)}")
    for nm, what in (("_k5_prologue", "prologue"), ("_k5_combine_gate", "combine")):
        fn = [n_ for n_ in ktree.body if isinstance(n_, ast.FunctionDef) and n_.name == nm]
        check(len(fn) == 1, nm)
        b = fn[0].body
        first = b[1] if isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) else b[0]
        check(isinstance(first, ast.If) and ast.unparse(first.test) == "USE_PDL" and ast.unparse(first.body[0]) == "tl.extra.cuda.gdc_wait()",
              f"{nm}: gdc_wait must be the first statement")
        loads = [s_.lineno for s_ in ast.walk(fn[0]) if isinstance(s_, ast.Call) and ast.unparse(s_.func) in ("tl.load", "tl.store")]
        check(loads and min(loads) > first.body[0].lineno, f"{nm}: a tl.load / tl.store precedes the wait")
    core = out.seg("_v16_fa4v5_core")
    check(core.count("USE_PDL=pdl") == 2 and core.count("**pdl_kw") == 2 and 'pdl_kw = {"launch_pdl": True} if pdl else {}' in core,
          "both Triton launches must take USE_PDL and launch_pdl from the same pdl flag")
    return ["PDL audit (source): FA4 core - griddepcontrol_wait is the first statement of K5Sm90.load (before every use of Q / K / V / page table / smem stages), the "
            "consumers' only pre-wait read is cache_seqlens (a layer input); combine - gdc_wait is the first statement, before the LSE / partial / gate loads; prologue - "
            "gdc_wait first, then the trigger, then all loads and stores; USE_PDL and launch_pdl come from one flag. No violation, nothing patched "
            "(empirical check: tools/box/v16_soak.py)"]


def verify(text, text15, kernel_src, fa5, route5):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    btmp = OUT.with_suffix(".py.gen_base")
    btmp.write_text(text15, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(tmp), doraise=True)
        out, b15 = Src(tmp), Src(btmp)    # Src raises on a duplicate top-level name
    finally:
        tmp.unlink()
        btmp.unlink()
    rep = []
    check(text.isascii(), "non-ASCII character in the output")
    lines = text.splitlines()
    check(lines[0] == "# okc: problem=muse_decode_layer", "header line 1")
    check(lines[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1 and "k3-v15-1" not in text, "nonce")
    check(ast.get_docstring(out.tree).startswith("K3 v16 = k3_v15 + the FA4 v5 attention routes"), "docstring")
    check(text.count(G13.K1_SELF_LINE) == 1, "K1 self-test reference line (v7, E4M3 Q at L = 8)")
    # kernel text: embedded byte for byte; the diff to k1_fa4v5.py = the patches
    ksrc = out.value("K1FA4V5_KERNEL_SRC")
    fa5_text = SRC_FA5.read_text(encoding="utf-8")
    check(ksrc == kernel_src and out.seg("K1FA4V5_KERNEL_SRC") == "K1FA4V5_KERNEL_SRC = r'''" + kernel_src + "'''\n", "K1FA4V5_KERNEL_SRC differs from the patched k1_fa4v5.py")
    removed, added = G15.removed_lines(fa5_text, ksrc)
    check(tuple(removed) == KERNEL_REMOVED, f"k1_fa4v5.py -> kernel text: removed / replaced lines {removed}")
    ktree, otree = ast.parse(ksrc), ast.parse(fa5_text)
    ins = [n_ for n_ in ktree.body if (isinstance(n_, ast.Assert) and ast.unparse(n_.test) == "__name__ not in sys.modules")
           or (isinstance(n_, ast.Assign) and ast.unparse(n_.targets[0]) == "sys.modules[__name__]")]
    check(len(ins) == 2, "the two self-registration statements")
    rest = [n_ for n_ in ktree.body if n_ not in ins]
    check(len(rest) == len(otree.body), "top-level statement count (minus the two inserted) must equal k1_fa4v5.py's")
    changed = []
    for a, b in zip(otree.body, rest):
        if dump(a) != dump(b):
            changed.append(next((t.id for t in getattr(a, "targets", []) if isinstance(t, ast.Name)), getattr(a, "name", ast.unparse(a)[:40])))
    check(sorted(changed) == sorted(KERNEL_CHANGED_NAMES), f"top-level statements changed vs k1_fa4v5.py: {changed}")
    for nm in KERNEL_SAME_FNS:
        a = [n_ for n_ in otree.body if getattr(n_, "name", None) == nm]
        b = [n_ for n_ in ktree.body if getattr(n_, "name", None) == nm]
        check(len(a) == len(b) == 1 and dump(a[0]) == dump(b[0]), f"{nm} must stay ast-identical to k1_fa4v5.py")
    # hygiene of the embedded module: no os, no environment, sys.modules only to add new private names, no sys.path / exec
    names = {n_.id for n_ in ast.walk(ktree) if isinstance(n_, ast.Name)}
    check("os" not in names and not any(isinstance(n_, ast.Import) and any(a_.name == "os" for a_ in n_.names) for n_ in ast.walk(ktree)),
          "kernel text uses os")
    check(not [n_ for n_ in ast.walk(ktree) if isinstance(n_, ast.Attribute) and n_.attr == "environ"], "kernel text touches an environ attribute")
    sys_hits = [ast.unparse(n_) for n_ in ast.walk(ktree) if isinstance(n_, ast.Attribute) and isinstance(n_.value, ast.Name) and n_.value.id == "sys"]
    check(sorted(sys_hits) == ["sys.modules"] * 4, f"sys uses in the kernel text: {sys_hits}")   # 2 registrations x (assert + store)
    stores = [n_ for n_ in ast.walk(ktree) if isinstance(n_, ast.Assign) and isinstance(n_.targets[0], ast.Subscript)
              and ast.unparse(n_.targets[0].value) == "sys.modules"]
    asserts = [n_ for n_ in ast.walk(ktree) if isinstance(n_, ast.Assert) and "not in sys.modules" in ast.unparse(n_.test)]
    check(len(stores) == 2 and len(asserts) == 2, "sys.modules stores must each follow a `not in sys.modules` assert")
    rep.append(f"K1FA4V5_KERNEL_SRC == k1_fa4v5.py ({len(fa5_text.encode())} bytes) with 4 exact-once patches ({len(ksrc.encode())} bytes, sha1 {sha(ksrc)[:12]}): "
               f"removed/replaced lines {len(removed)} (TMP, the two private module names), +{added} lines (the self-registration, the new names); the other "
               f"top-level statements ast-identical in place ({', '.join(KERNEL_SAME_FNS)}); no os, no environ, sys.modules only adds new private names")
    rep += G15.patched_copy_checks(ksrc)
    rep += pdl_audit(ktree, out)
    # route <-> k1_fa4v5.py / k1_fa4v5_route.py
    check(out.value("K1FA4V5_DECODE_FULL_MAX_BATCH") == route5.value("FULL_L1_MAX_BATCH") == 1
          and out.value("K1FA4V5_SPLITS") == fa5.value("SPLITS") == {("sliding", 1): 32}
          and out.value("K1FA4V5_VERIFY_FULL_MIN_BATCH") == 16 == out.value("K1FA4_FULL_MIN_BATCH")
          and all(out.value(k) is True for k in ("K1FA4V5_ENABLE",) + GROUPS) and out.value("K1FA4V5_V2_ALWAYS") is True, "route constants")
    mod_route_ok, file_route_ok = fa5.seg("route_ok"), route5.seg("route_ok")
    wanted_src = out.seg("_v16_fa4v5_wanted")
    for term in MODULE_TERMS:
        check(term in mod_route_ok and term in wanted_src, f"route_ok term {term!r} must be in k1_fa4v5.route_ok and _v16_fa4v5_wanted")
    for term in ROUTE_TERMS:
        check(term in file_route_ok and term in wanted_src, f"route_ok term {term!r} must be in k1_fa4v5_route.route_ok and _v16_fa4v5_wanted")
    for a_, b_ in ROUTE_RENAMED:
        check(a_ in file_route_ok and b_ in wanted_src, f"route_ok term {a_!r} / {b_!r}")
    # _v16_fa4v5_core = k1_fa4v5.run's statements (dropped / rewritten ones listed above), after the rename
    core = out.fn("_v16_fa4v5_core")
    check([a_.arg for a_ in core.args.args] == ["k5", "qkv", "data", "state", "splits", "pdl"] and [ast.unparse(x) for x in core.args.defaults] == ["None", "True"],
          "_v16_fa4v5_core signature")
    run_fn = fa5.fn("run")
    check([a_.arg for a_ in run_fn.args.args] == ["data", "splits", "pdl"] and [ast.unparse(x) for x in run_fn.args.defaults] == ["None", "True"], "k1_fa4v5.run signature")
    expect = []
    for st in run_fn.body[1:]:
        key = ast.unparse(st)
        if key in RUN_DROPPED:
            continue
        seg = ast.get_source_segment(fa5.text, st)
        if key == RUN_COS_OLD:
            seg = seg.replace(*RUN_COS_NEW_CALL)
        expect.append(ast.parse(run_rename(seg)).body[0])
    got = core.body[1:]
    check(len(got) == len(expect) and [dump(a_) for a_ in got] == [dump(b_) for b_ in expect], "_v16_fa4v5_core statements != k1_fa4v5.run's (after the rename)")
    empties = [ast.unparse(s.func) for s in ast.walk(core) if isinstance(s, ast.Call) and ast.unparse(s.func) == "torch.empty"]
    check(len(empties) == 2, f"_v16_fa4v5_core must have exactly run()'s two shape-only torch.empty calls, has {len(empties)}")
    # force_row0: the file's function, renamed
    f_got, f_want = out.fn("_v16_force_row0"), route5.fn("force_row0")
    check(dump(ast.Module(body=f_got.body, type_ignores=[])) == dump(ast.Module(body=f_want.body, type_ignores=[]))
          and [a_.arg for a_ in f_got.args.args] == [a_.arg for a_ in f_want.args.args], "_v16_force_row0 != k1_fa4v5_route.force_row0")
    rep.append(f"route: K1FA4V5_DECODE_FULL_MAX_BATCH 1 / K1FA4V5_SPLITS == the route / module files; route_ok terms {len(MODULE_TERMS) + len(ROUTE_TERMS) + len(ROUTE_RENAMED)} "
               f"in both; _v16_fa4v5_core = k1_fa4v5.run's {len(expect)} statements ast-identical after the rename ({', '.join(RUN_RENAME)}, the cos | sin table, "
               f"`qkv = data[\"qkv\"].view(tokens, -1)` dropped); _v16_force_row0 == force_row0")
    rep += jit_checks(ksrc, out)
    # self-test lines taken from k3_v15's, verbatim
    for line in ("stock_ok = (tot <= 1.25 * tot_r1 + 5e-3 and row <= 1.25 * row_r1 + 2e-2) if tot_r1 >= 0 else (tot < 5e-2 and row < 1e-1)",):
        check(line in [ln.strip() for ln in out.seg("_v15_fa4_selftest").splitlines()] and line in [ln.strip() for ln in out.seg("_v16_fa4v5_selftest").splitlines()],
              f"self-test line must be verbatim in both self-tests: {line}")
    for nm in ("_k1_mk_case", "_k1_ref_attention", "_k1_ref_new_kv", "_k1_rel_rows"):
        check(dump(out.nodes[nm]) == dump(b15.nodes[nm]), f"{nm} (the self-test's reference) must stay k3_v15's")
    # nothing else changed
    changed = [n_ for n_, node in b15.nodes.items() if n_ not in out.nodes or dump(out.nodes[n_]) != dump(node)]
    check(set(changed) == CHANGED_NODES, f"changed nodes vs k3_v15: {sorted(changed)} (want {sorted(CHANGED_NODES)})")
    added_nodes = set(out.nodes) - set(b15.nodes)
    check(added_nodes == V16_NODES, f"added nodes: unexpected {sorted(added_nodes - V16_NODES)}, missing {sorted(V16_NODES - added_nodes)}")
    for nm in ("K1_ROUTE_MIN_BATCH", "K1_FOLD_MIN_BATCH", "K1_MIN_CHUNK_T", "K1_CPS_T", "K1_STAGES_T", "K2A36_PLAN", "K2A33_PLAN", "P1_PROJ", "_k1_run", "_k1_supported",
               "_k1_ready", "_k1_self_test", "_k1_prepare", "_v13_prepare", "_v10_qk_store", "_k3_disable", "_r1_attention", "K1FA4_KERNEL_SRC",
               "_v15_fa4_load", "_v15_fa4_wanted", "_v15_fa4_core", "_v15_fa4_selftest", "_v15_fa4_prepare", "K1FA4_SELF_TEST_CASES", "K1FA4_SPLITS"):
        check(dump(out.nodes[nm]) == dump(b15.nodes[nm]), f"{nm} must stay k3_v15's")
    # kernel_fn: k3_v15's body with one block inserted and `if` -> `if o is None and` on the FA4 v2 block
    kf15, kf16 = b15.fn("kernel_fn"), out.fn("kernel_fn")
    idx = [i for i, s in enumerate(kf16.body) if isinstance(s, ast.If) and ast.unparse(s.test) == "_v16_fa4v5_wanted(qkv, data, cfg, state)"]
    check(len(idx) == 1, "kernel_fn: FA4 v5 block")
    old_if = [i for i, s in enumerate(kf15.body) if isinstance(s, ast.If) and ast.unparse(s.test) == "_v15_fa4_wanted(data, cfg, state)"]
    check(len(old_if) == 1, "kernel_fn: k3_v15's FA4 v2 block")
    new_if = kf16.body[idx[0] + 1]
    check(isinstance(new_if, ast.If) and ast.unparse(new_if.test) == "o is None and " + ast.unparse(kf15.body[old_if[0]].test)
          and dump(ast.Module(body=new_if.body, type_ignores=[])) == dump(ast.Module(body=kf15.body[old_if[0]].body, type_ignores=[]))
          and dump(ast.Module(body=new_if.orelse, type_ignores=[])) == dump(ast.Module(body=kf15.body[old_if[0]].orelse, type_ignores=[])),
          "kernel_fn: the FA4 v2 block must keep its body and gain `o is None and`")
    check([dump(s) for s in kf16.body[:idx[0]]] == [dump(s) for s in kf15.body[:old_if[0]]]
          and [dump(s) for s in kf16.body[idx[0] + 2:]] == [dump(s) for s in kf15.body[old_if[0] + 1:]], "kernel_fn: the rest must stay k3_v15's")
    v5_if = kf16.body[idx[0]]
    check(len(v5_if.body) == 1 and isinstance(v5_if.body[0], ast.Try) and len(v5_if.body[0].handlers) == 1 and not v5_if.orelse
          and ast.unparse(v5_if.body[0].body[0]) == "o = _v16_fa4v5_core(state['fa4v5'], qkv, data, state, pdl=state['fa4v5_pdl']).view(tokens, hq * d)"
          and "_k3_disable(state, 'fa4v5', None, 'FA4 v5 route', ex)" in [ast.unparse(s) for s in v5_if.body[0].handlers[0].body], "kernel_fn FA4 v5 try block")
    # prepare: the FA4 v2 call replaced by _v16_prepare
    p15, p16 = b15.fn("prepare"), out.fn("prepare")
    ci = [i for i, s in enumerate(p16.body) if ast.unparse(s) == "_v16_prepare(state, t0)"]
    check(len(ci) == 1 and [dump(s) for s in p16.body[:ci[0]] + p16.body[ci[0] + 1:]] == [dump(s) for s in p15.body if ast.unparse(s) != "_v15_fa4_prepare(state, t0)"]
          and ast.unparse(p15.body[ci[0]]) == "_v15_fa4_prepare(state, t0)", "prepare: only the _v15_fa4_prepare call is replaced by _v16_prepare")
    rep.append(f"vs k3_v15 (gen output == file on disk): {len(b15.nodes) - len(changed)} top-level nodes ast-identical; changed {sorted(changed)} "
               f"(kernel_fn: one inserted block + `o is None and`; prepare: one call); added {len(added_nodes)}; the FA4 v2 route (K1FA4_*, _v15_*), "
               f"_r1_attention and every K1 / K2A / P1 setting identical")
    # hygiene
    for node in ast.walk(out.tree):
        if isinstance(node, ast.Name):
            check(node.id not in G.BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
    for fn in CALL_TIME_FNS:
        for sub in ast.walk(out.nodes[fn]):
            check(not (isinstance(sub, ast.Attribute) and sub.attr in G.SYNC_ATTRS), f"{fn} has a host sync")
    for fn in ("_v16_fa4v5_wanted", "_v16_fa4v5_core"):   # call-time glue: no fill / value-dependent allocation
        calls = {ast.unparse(s.func) for s in ast.walk(out.nodes[fn]) if isinstance(s, ast.Call)}
        check(not (calls & {"torch.zeros", "torch.arange", "torch.tensor", "torch.ones", "torch.full"}), f"{fn} allocates: {calls}")
    check("torch.empty" not in {ast.unparse(s.func) for s in ast.walk(out.nodes["_v16_fa4v5_wanted"]) if isinstance(s, ast.Call)}, "_v16_fa4v5_wanted allocates")
    bad = G8.G7.undefined_globals(text, str(OUT))
    check(not bad, f"undefined globals: {sorted(bad)}")
    check(len(text.encode()) < 1_000_000, "file >= 1 MB")
    code_only = "\n".join(l_ for l_ in lines if not l_.lstrip().startswith("#"))
    check(re.search(r"\benviron\b", code_only) is None and re.search(r"\bos\.environ", code_only) is None,
          "`environ` appears in the layer file outside comments")
    rep.append(f"static ok (no banned names in the layer's code, no host sync in {len(CALL_TIME_FNS)} call-time functions, no fill / arange / tensor call in the v5 glue, "
               f"no undefined globals, no `environ` outside comments); {len(text.encode())} bytes (k3_v15: {len(text15.encode())}; limit 1,000,000)")
    for r in check_ab_json(out, AB_JSON):
        rep.append(f"{AB_JSON.name} arm  {r}")
    return rep


def main(argv):
    text, text15, kernel_src, fa5, route5 = generate()
    for r in verify(text, text15, kernel_src, fa5, route5):
        print(f"[gen_k3_v16] OK  {r}")
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text(encoding="utf-8") == text, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_v16] --check: up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"[gen_k3_v16] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}, {len(text.splitlines())} lines, {len(text.encode())} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

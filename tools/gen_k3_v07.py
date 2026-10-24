"""gen_k3_v07.py: k3_v07.py = k3_v06.py with the routed experts on K2A v2.2 (kernels/muse_moe_w4a16/k2a_v22.py),
per token bucket, Marlin (r1) for any bucket the route table sends there (lessons.md rules R2/R3).

    python3 -I tools/gen_k3_v07.py            # verifies the chain (gen_k3 --check, v06 = v05 + its 2 patches), writes
                                              # k3_v07.py, prints a diff summary, asserts the invariants
    python3 -I tools/gen_k3_v07.py --check    # same checks, compares with k3_v07.py on disk, writes nothing

Canonical sources (read, never edited):
  kernels/muse_decode_layer/k3_v06.py     everything outside the K2a slice (r1 glue + levers incl. BLOCK_SIZE_M_RANGES,
                                          K1 v1 slice, kernel_fn, _experts, _r1_prepare); must equal what
                                          gen_k3.py + gen_k3_v06.py produce (checked first)
  kernels/muse_moe_w4a16/k2a_v22.py       K2A v2.2: KERNEL_SRC / HOST_TEMPLATE / CPP_SRC (byte-identical, sha1 pinned
                                          below), repack, build, v2_call / v2_prepare / v2_reset, FP32 reference,
                                          synthetic inputs, variants, buckets and the measured per-bucket V2_PLAN

The K2a v0 slice of k3_v06 (between its banner and the Rung-1 banner) is replaced by a K2A v2.2 slice: k2a_v22.py's
pieces copied verbatim under a K2A_/_k2a_ rename map (compared by ast.dump after the rename), plus four layer-glue
functions with the SAME call surface as v06 (_k2a_prepare, _k2a_experts, _k2a_supported, _k2a_self_test_inputs), so
kernel_fn is untouched. Head: nonce, docstring, USE_K2A = True, K2A_MIN/MAX_TOKENS -> per-bucket K2A_ROUTE, the
[k3] switches print. Tail: prepare's summary print. Only stdlib; no torch import.
"""

import ast
import builtins
import difflib
import hashlib
import importlib.util
import pathlib
import py_compile
import re
import subprocess
import symtable
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_LAYER = ROOT / "kernels/muse_decode_layer/k3_v06.py"
SRC_V05 = ROOT / "kernels/muse_decode_layer/k3_v05.py"
SRC_K2A = ROOT / "kernels/muse_moe_w4a16/k2a_v22.py"
OUT = ROOT / "kernels/muse_decode_layer/k3_v07.py"
NONCE = "k3-v07-1"

_spec = importlib.util.spec_from_file_location("gen_k3", ROOT / "tools/gen_k3.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha

# ------------------------------------------------------------------------------------------------ intended values
# K2A v2.2 sources, pinned (k2a_v22.py nonce k2a-v22-1, box judge clone 1.471x): any change there must be deliberate.
K2A_SHA1 = {"KERNEL_SRC": "eb8d222136dc216ed19aa7534611acb4c0092545",
            "HOST_TEMPLATE": "e36f9818d59c039d433b4e2e194fce3e933e350f",
            "CPP_SRC": "f69ceba4ad999c44783c414c79126e3ab3f848e3"}
WANT_PLAN = {1: (0, 2, 1, 0), 8: (0, 1, 2, 0), 16: (0, 1, 1, 0), 32: (0, 2, 2, 0), 128: (1, 2, 2, 0), 256: (1, 2, 4, 0)}
WANT_BUCKETS = ((1, 4, 1), (5, 12, 8), (13, 24, 16), (25, 48, 32), (49, 192, 128), (193, 320, 256))
# Experts route per token bucket (key = bucket representative), decided IN THE LAYER on the box (rule R2):
# "v2" = K2A v2.2, "marlin" = r1's Marlin path. See kernels/muse_decode_layer/k3_v07.md for the measurements.
# Box 2026-10-24: in-layer paired A/B (same process, same judge inputs, 30-60 pairs per shape) Marlin/K2A:
#   T=1 1.023-1.027 sl / 1.005-1.010 fu, T=8 1.033 / 1.046, T=16 1.053 / 1.037, T=32 0.958-0.967 sl / 1.000 fu,
#   T=128 1.075 / 1.042, T=256 1.067 / 1.042. T=32 (bucket 25-48) loses on the sliding shape, ties on full -> Marlin.
WANT_ROUTE = {1: "v2", 8: "v2", 16: "v2", 32: "marlin", 128: "v2", 256: "v2"}
ROUTE_NOTE = ("in-layer A/B on the box: K2A wins T=1/8/16/128/256, T=32 Marlin (sliding 0.96x, full 1.00x);"
              " k3_v07.md")
# extra layer self-test cases (bs, seed, dist): every scored bucket at its scored T, skewed where the layer routes skewed
LAYER_CASES = ((8, 17, "normal"), (16, 18, "skewed"), (256, 19, "skewed"))
K2A_BUILD_NAME = "k3_k2a_v22"   # load_inline name prefix (distinct from the moe file's k2a_v2_<tag> cache entry)

# k2a_v22 name -> layer name (one regex pass, identifier boundaries; applies inside f-strings and docstrings too)
RENAME = {
    "E_LOC": "K2A_E_LOC", "D_HIDDEN": "K2A_D_HIDDEN", "D_EXPERT": "K2A_D_EXPERT", "TOP_K": "K2A_TOP_K",
    "POST_NORM_EPS": "K2A_POST_NORM_EPS", "M_MAX": "K2A_M_MAX", "META_HDR": "K2A_META_HDR", "BITPOS": "K2A_BITPOS",
    "W_SHIFT": "K2A_W_SHIFT", "VKEYS": "K2A_VKEYS", "variant": "_k2a_variant", "V2_VARIANTS": "K2A_V2_VARIANTS",
    "BUCKETS": "K2A_BUCKETS", "V2_PLAN": "K2A_V2_PLAN",
    "reencode_scales": "_k2a_reencode_scales", "pack_words": "_k2a_pack_words", "tile_row_perm": "_k2a_tile_row_perm",
    "prepack_expert": "_k2a_prepack_expert", "bucket_of": "_k2a_bucket_of",
    "KERNEL_SRC": "K2A_KERNEL_SRC", "HOST_TEMPLATE": "K2A_HOST_TEMPLATE", "CPP_SRC": "K2A_CPP_SRC",
    "cfg_str": "_k2a_cfg_str", "gen_host": "_k2a_gen_host", "build_ext": "_k2a_build_ext",
    "PH_ROUTE": "K2A_PH_ROUTE", "PH_G1": "K2A_PH_G1", "PH_G2": "K2A_PH_G2", "PH_FIN": "K2A_PH_FIN",
    "HOT_LAUNCH_COMPILED": "K2A_HOT_LAUNCH_COMPILED", "FUSE_MASKS": "K2A_FUSE_MASKS",
    "v2_call": "_k2a_v2_call", "v2_prepare": "_k2a_v2_prepare", "v2_reset": "_k2a_v2_reset",
    "_GRID": "_K2A_GRID", "_dequant": "_k2a_dequant", "ref_fp32": "_k2a_ref_fp32", "rel_err": "_k2a_rel_err",
    "synth_inputs": "_k2a_synth_inputs",
}
_RENAME_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(sorted(map(re.escape, RENAME), key=len, reverse=True))
                        + r")(?![A-Za-z0-9_])")
SOURCES = ("KERNEL_SRC", "HOST_TEMPLATE", "CPP_SRC")
# emitted in this order (named top-level nodes of k2a_v22.py; the PH_* tuple assignment is handled by line)
V22_CONSTS = ("E_LOC", "D_HIDDEN", "D_EXPERT", "TOP_K", "POST_NORM_EPS", "M_MAX", "META_HDR", "BITPOS", "W_SHIFT",
              "VKEYS", "variant", "V2_VARIANTS", "BUCKETS", "V2_PLAN")
V22_REPACK = ("reencode_scales", "pack_words", "tile_row_perm", "prepack_expert", "bucket_of")
V22_BUILD = ("cfg_str", "gen_host", "build_ext")
V22_LAUNCH = ("HOT_LAUNCH_COMPILED", "FUSE_MASKS", "v2_call", "v2_prepare", "v2_reset")
V22_REF = ("_GRID", "_dequant", "ref_fp32", "rel_err", "synth_inputs")
V22_ALL = V22_CONSTS + V22_REPACK + SOURCES + V22_BUILD + V22_LAUNCH + V22_REF
GLUE_FNS = ("_k2a_self_test_inputs", "_k2a_experts", "_k2a_prepare", "_k2a_supported")
# names of the v0 slice / v06 head that must be gone from k3_v07 (not defined, not referenced)
V0_GONE = ("K2A_CUDA_SRC", "K2A_KS2_MAX", "_k2a_row_perm", "_k2a_ng_for_bs", "_k2a_build", "_K2A_GRID_VALUES",
           "_k2a_rel", "K2A_MIN_TOKENS", "K2A_MAX_TOKENS")
SYNC_ATTRS = G.SYNC_ATTRS


def val(src, name, **env):
    """Literal value of a top-level assignment, evaluated with the given names (e.g. M_MAX) and no builtins."""
    node = src.nodes[name]
    return eval(compile(ast.Expression(node.value), str(src.path), "eval"), {"__builtins__": {}, **env})  # noqa: S307


def rename(text):
    return _RENAME_RE.sub(lambda m: RENAME[m.group(1)], text)


# ================================================================================================ chain check
def check_chain():
    chk = subprocess.run([sys.executable, "-I", str(ROOT / "tools/gen_k3.py"), "--check"], capture_output=True, text=True)
    check(chk.returncode == 0 and "up to date" in chk.stdout, "k3_v05.py is not what gen_k3.py produces:\n" + chk.stdout[-2000:]
          + chk.stderr[-2000:])
    # gen_k3_v06.py's two patches, replayed (it has no --check mode and always writes)
    s = SRC_V05.read_text().split("\n")
    check(s[1].startswith("# nonce k3-v05-1"), "k3_v05 nonce")
    s[1] = "# nonce k3-v06-1 (k3_v05 + Marlin block_size_m 16 at >=49 tokens only; review kernels/review_scored_delta_1.md)"
    s = "\n".join(s)
    old = "BLOCK_SIZE_M_RANGES = ((9, 24, 16), (25, 48, 8), (49, 1 << 30, 16))"
    check(s.count(old) == 1, "v05 ranges line")
    s = s.replace(old, "BLOCK_SIZE_M_RANGES = ((49, 1 << 30, 16),)  # judge: only >=49 gains (bs128 1.37x, bs256 1.29x); "
                       "small-T ranges were noise or negative", 1)
    check(SRC_LAYER.read_text() == s, "k3_v06.py on disk differs from gen_k3_v06.py's output (regenerate the chain)")
    return sha(s)


# ================================================================================================ the K2A v2.2 slice
def k2a_section(k2a):
    for nm in SOURCES:
        v = k2a.value(nm)
        check(sha(v) == K2A_SHA1[nm], f"k2a_v22.py {nm} sha1 {sha(v)} != pinned {K2A_SHA1[nm]}")
        check('"""' not in v, f"{nm} contains triple quotes")
        check(k2a.seg(nm).startswith(f'{nm} = r"""'), f"{nm} is not a raw triple-quoted literal")
    check(k2a.value("V2_PLAN") == WANT_PLAN, f"k2a_v22 V2_PLAN {k2a.value('V2_PLAN')} != pinned")
    check(val(k2a, "BUCKETS", M_MAX=320) == WANT_BUCKETS and k2a.value("M_MAX") == 320, "k2a_v22 BUCKETS / M_MAX")
    check(set(WANT_ROUTE) == {b[2] for b in WANT_BUCKETS} and set(WANT_ROUTE.values()) <= {"v2", "marlin"}, "route keys")
    seg = {nm: (("K2A_" + k2a.seg(nm)) if nm in SOURCES else rename(k2a.seg(nm))) for nm in V22_ALL}
    # the PH_* tuple assignment (unnamed for Src)
    ph = [n for n in k2a.tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
          and [e.id for e in n.targets[0].elts] == ["PH_ROUTE", "PH_G1", "PH_G2", "PH_FIN"]]
    check(len(ph) == 1, "PH_* assignment")
    ph_text = rename("".join(k2a.lines[ph[0].lineno - 1:ph[0].end_lineno]))
    # v22's self-test cases (inside prepare), hoisted
    cases = [n for n in ast.walk(k2a.fn("prepare")) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id == "cases"]
    check(len(cases) == 1, "k2a_v22 prepare: one `cases = (...)`")
    v22_cases = ast.literal_eval(cases[0].value)
    check(len(v22_cases) == 6 and {c[0] for c in v22_cases} == {1, 5, 33, 100, 128, 257}, "v22 cases")
    all_cases = tuple(v22_cases) + LAYER_CASES
    check({k2a_bucket(c[0]) for c in all_cases} >= set(WANT_PLAN), "self-test covers every bucket's plan")

    banner = "# " + "=" * 94 + "\n"
    glue = GLUE_TEXT.replace("@BUILD_NAME@", K2A_BUILD_NAME)
    parts = [
        banner,
        "# K2a: custom NVFP4 W4A16 experts v2.2 (kernels/muse_moe_w4a16/k2a_v22.py, nonce k2a-v22-1; box judge clone 1.471x\n"
        "# on muse_moe_w4a16, box_out/v22b.log). GENERATED by tools/gen_k3_v07.py: KERNEL_SRC / HOST_TEMPLATE / CPP_SRC\n"
        "# byte-identical (sha1 pinned in the generator); repack, build, launch, reference and synthetic inputs copied\n"
        "# verbatim with names prefixed K2A_/_k2a_; per-bucket launch plan = k2a_v22.py's measured V2_PLAN. Layer glue\n"
        "# (last four functions) keeps k3_v06's call surface: ids are r1's table-mapped local ids (remote = -1), lo = 0.\n",
        banner, "\n",
        "".join(seg[n] for n in V22_CONSTS[:9]), "\n",
        "# Variant table (template parameters of k2a2::Cfg<NC, U, PF, TW, XE, MINB>), as in k2a_v22.py.\n",
        seg["VKEYS"], "\n\n", seg["variant"], "\n\n", seg["V2_VARIANTS"],
        "# Per bucket: (lo, hi, representative scored bs); the route table K2A_ROUTE (top of file) is keyed by the rep.\n",
        seg["BUCKETS"],
        "# v2 plan per bucket: (variant index, fuse, ks_target, asg), measured on the box (k2a_v22.md, tools/box/v22_bench.py).\n",
        seg["V2_PLAN"], "\n",
        "# (bs, seed, dist): k2a_v22.py's self-test cases, then layer cases (every bucket's plan at its scored T). Each case\n"
        "# runs the plan's fuse mode through the layer entry point, plus fuse 0 and 2; remote ids are -1 as the layer passes.\n",
        "K2A_SELF_TEST_CASES = " + repr(all_cases) + "\n\n\n",
        "".join(seg[n] + "\n\n" for n in V22_REPACK),
        seg["KERNEL_SRC"], "\n", seg["HOST_TEMPLATE"], "\n", seg["CPP_SRC"], "\n\n",
        "".join(seg[n] + "\n\n" for n in V22_BUILD),
        ph_text, seg["HOT_LAUNCH_COMPILED"], seg["FUSE_MASKS"], "\n\n",
        "".join(seg[n] + "\n\n" for n in V22_LAUNCH[2:]),
        seg["_GRID"], "\n\n",
        "".join(seg[n] + "\n\n" for n in V22_REF[1:]),
        glue, "\n",
    ]
    return "".join(parts), all_cases


def k2a_bucket(m):
    for lo, hi, rep in WANT_BUCKETS:
        if lo <= m <= hi:
            return rep
    return None


GLUE_TEXT = '''def _k2a_self_test_inputs(bs, seed, device, dist="normal"):
    """k2a_v22.py's synthetic inputs (this GPU's experts are 0..31, lo = 0), with remote ids mapped to -1 when
    R1_LOCAL_ID_TABLE is on: the id convention the layer actually hands K2A (r1's device-side local-id table)."""
    d = _k2a_synth_inputs(bs, seed, device, dist)
    if R1_LOCAL_ID_TABLE:
        ids = d["topk_ids"]
        d["topk_ids"] = torch.where(ids < K2A_E_LOC, ids, torch.full_like(ids, -1))
    return d


def _k2a_experts(x, local_ids, topk_weights, st):
    """K2A v2.2 on the layer (k2a_v22.py kernel_fn's v2 branch). x [T, 4096] BF16 = the pre-expert-normed pooled
    activations; local_ids [T, 8] int32 = r1's table-mapped ids (this GPU's experts 0..31; -1, or anything outside
    [0, 32), is remote: the kernel's route prologue tests 0 <= id - lo < 32), so lo = 0 and local_start is never read
    on the host; topk_weights [T, 8] FP32. Launch plan = K2A_V2_PLAN of T's bucket (shape-only). Returns
    sum_k w_k * RMSNorm(expert_k(x) * bf16(1 + post_expert_norm)) as BF16 [T, 4096]."""
    x = x.contiguous()
    if x.data_ptr() % 16:   # X is staged with 16-byte loads (a pointer property, not a value)
        x = x.clone()
    rep = _k2a_bucket_of(x.shape[0])
    vid, fuse, kst, asg, hv, ht = (tuple(K2A_V2_PLAN[rep]) + (-1, 0))[:6]
    data = {"hidden_states": x, "topk_ids": local_ids, "topk_weights": topk_weights, "config": {"local_start": 0}}
    return _k2a_v2_call(st, data, vid, fuse, kst, asg, 0, 0, hv, ht)


def _k2a_prepare(weights, t0):
    """k2a_v22.py's v2_prepare (lossless repack, nvcc, attributes) on the layer's weights (post_expert_norm is
    v2.2's post_norm_w; v2_prepare binds bf16(1 + w) itself), then the self-test. Returns K2A's state, or None:
    the experts stay on r1's Marlin path for every bucket."""
    if not any(v == "v2" for v in K2A_ROUTE.values()):
        print("[k3] K2A_ROUTE sends every bucket to Marlin: K2A not built", flush=True)
        return None
    dev = weights["w13"].device
    w = {"w13": weights["w13"], "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"],
         "w2": weights["w2"], "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"],
         "post_norm_w": weights["post_expert_norm"]}
    t1 = time.time()
    try:
        assert tuple(weights["w13"].shape) == (K2A_E_LOC, 2 * K2A_D_EXPERT, K2A_D_HIDDEN // 2), "w13 shape"
        assert tuple(weights["w2"].shape) == (K2A_E_LOC, K2A_D_HIDDEN, K2A_D_EXPERT // 2), "w2 shape"
        st = _k2a_v2_prepare(w, K2A_V2_VARIANTS, name="@BUILD_NAME@", log=lambda s: print(s, flush=True))
        _K3_BUILD_S["k2a"] = time.time() - t1
    except Exception as ex:  # noqa: BLE001
        print("[k3] K2A v2.2 repack/build FAILED: experts stay on the r1 Marlin path. Tail:\\n" + str(ex)[-12000:],
              flush=True)
        return None
    if not K2A_SELF_TEST:
        print("[k3] K2A_SELF_TEST=False: K2A used untested", flush=True)
        return st
    if time.time() - t0 > K2A_SELF_TEST_START_BY_S:
        print(f"[k3] K2A self-test not started ({time.time() - t0:.0f}s into prepare > {K2A_SELF_TEST_START_BY_S:.0f}s):"
              f" experts stay on the r1 Marlin path", flush=True)
        return None
    t2 = time.time()
    n = 0
    try:
        for bs, seed, dist in K2A_SELF_TEST_CASES:
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            ref = _k2a_ref_fp32(d, w)
            rep = _k2a_bucket_of(bs)
            vid, fuse, kst, asg, hv, ht = (tuple(K2A_V2_PLAN[rep]) + (-1, 0))[:6]
            for fz in sorted({fuse, 0, 2}):
                if fz == fuse:   # the layer's own entry point, exactly as kernel_fn calls it
                    got = _k2a_experts(d["hidden_states"], d["topk_ids"], d["topk_weights"], st)
                else:
                    got = _k2a_v2_call(st, d, vid, fz, kst, asg, 0, 0, -1, 0)
                torch.cuda.synchronize()
                n += 1
                tot, row = _k2a_rel_err(got, ref)
                errf = int(st["err"][0])
                print(f"[k2a2] self-test bs={bs} {dist} v{vid} fuse={fz}{' (layer entry)' if fz == fuse else ''}: "
                      f"rel {tot:.3e} worst row {row:.3e} err={errf}", flush=True)
                if not (tot < 1.0e-2 and row < 5.0e-2) or errf or not bool(torch.isfinite(got.float()).all()):
                    raise RuntimeError(f"self-test mismatch at bs={bs} fuse={fz}")
    except Exception as ex:  # noqa: BLE001
        print(f"[k3] K2A self-test FAILED: experts stay on the r1 Marlin path: {ex!r}", flush=True)
        return None
    _k2a_v2_reset(st)
    print(f"[k2a2] layer self-test passed ({n} calls) in {time.time() - t2:.1f}s; prepare {time.time() - t1:.1f}s",
          flush=True)
    return st


def _k2a_supported(cfg, tokens):
    """K2A v2.2 takes the call when T's bucket routes to it (K2A_ROUTE, measured in the layer) and the expert config
    is the one it was built for; anything else (a "marlin" bucket, T > 320) takes r1's Marlin path."""
    rep = _k2a_bucket_of(tokens)
    return (rep is not None and K2A_ROUTE.get(rep) == "v2" and cfg["local_experts"] == K2A_E_LOC
            and cfg["top_k"] == K2A_TOP_K and cfg["n_experts"] == N_EXPERTS)
'''


# ================================================================================================ splice
def generate():
    lay, k2a = Src(SRC_LAYER), Src(SRC_K2A)
    i0 = lay.find_line("# K2a: custom NVFP4 W4A16 experts (kernels/muse_moe_w4a16/k2a_v0.py, nonce k2a-v0-1, review_k2a_v0.md).") - 1
    i1 = lay.find_line("# Rung 1 (kernels/muse_decode_layer/r1_glue.py, nonce r1-1): unchanged, and the fallback of both pieces.") - 1
    check(lay.lines[i0].startswith("# ====") and lay.lines[i1].startswith("# ===="), "K2a slice banners")
    head, tail = "".join(lay.lines[:i0]), "".join(lay.lines[i1:])
    section, cases = k2a_section(k2a)

    head = patch(head, "# nonce k3-v06-1 (k3_v05 + Marlin block_size_m 16 at >=49 tokens only; review kernels/review_scored_delta_1.md)\n"
                       "# GENERATED by tools/gen_k3.py from k3_v04.py (layer), k1_v1.py (K1 sources/helpers/self-test) and k1_v2.py\n"
                       "# (K1 host tables). Do not hand-edit: change a canonical source or the generator and regenerate.\n",
                 f"# nonce {NONCE} (k3_v06 + K2A v2.2 routed experts per token bucket, Marlin fallback per bucket)\n"
                 "# GENERATED by tools/gen_k3_v07.py from k3_v06.py (gen_k3.py + gen_k3_v06.py: r1 glue, K1 v1, Marlin bsm 16 at\n"
                 "# >=49) and k2a_v22.py (K2A v2.2 sources/repack/plan). Do not hand-edit: change a canonical source or a\n"
                 "# generator and regenerate.\n", "nonce")
    head = patch(head, "  USE_K2A (kernels/muse_moe_w4a16/k2a_v0.py): the custom NVFP4 W4A16 experts (route, W13 GEMM + SiLU*up, W2\n"
                       "       GEMM, post-expert norm x topk weight sum: 4 launches) replace r1's Marlin path. Input = the\n",
                 "  USE_K2A (kernels/muse_moe_w4a16/k2a_v22.py, K2A v2.2): the custom NVFP4 W4A16 experts (routing folded in, W13\n"
                 "       GEMM + SiLU*up, W2 GEMM, post-expert norm x topk weight sum: 1-2 launches per the measured per-bucket\n"
                 "       plan) replace r1's Marlin path for every token bucket K2A_ROUTE sends to \"v2\". Input = the\n",
                 "docstring K2A line")
    head = patch(head, "(K1_CPP_SRC/K1_CUDA_SRC, K2A_CPP_SRC/K2A_CUDA_SRC) and\n",
                 "(K1_CPP_SRC/K1_CUDA_SRC, K2A_CPP_SRC/K2A_KERNEL_SRC/K2A_HOST_TEMPLATE) and\n", "docstring sources line")
    head = patch(head, "fold mode, vs an FP32 reference, plus the fold-counter reset check; K2a: 6\n"
                       "synthetic routings over the layer's real expert weights vs an FP32 reference) before first use.",
                 "fold mode, vs an FP32 reference, plus the fold-counter reset check; K2A: 9\n"
                 "synthetic routings over the layer's real expert weights, each in the bucket's plan through the layer entry point\n"
                 "and fuse modes 0/2, vs an FP32 reference, grid-barrier error flag checked) before first use.",
                 "docstring self-test line")
    head = patch(head, "USE_K2A = False              # custom NVFP4 experts (K2a); False = r1's Marlin path\n",
                 "USE_K2A = True               # custom NVFP4 experts (K2A v2.2, per bucket via K2A_ROUTE); False = r1's Marlin path\n",
                 "USE_K2A")
    head = patch(head, "K2A_MIN_TOKENS = 65          # measured: K2a loses to Marlin at T<=32 and T=256, wins at T=128\n"
                       "K2A_MAX_TOKENS = 192\n",
                 "# Experts per token bucket (key = bucket rep: T 1-4 -> 1, 5-12 -> 8, 13-24 -> 16, 25-48 -> 32, 49-192 -> 128,\n"
                 "# 193-320 -> 256): \"v2\" = K2A v2.2, \"marlin\" = r1's Marlin path. Measured in the layer: " + ROUTE_NOTE + ".\n"
                 "K2A_ROUTE = " + repr(WANT_ROUTE) + "\n", "K2A route table")
    head = patch(head, " K2A_TOKENS={K2A_MIN_TOKENS}..{K2A_MAX_TOKENS}\",", " K2A=v2.2 route={K2A_ROUTE}\",",
                 "switches print")
    tail = patch(tail, "f\" route {K1_ROUTE_MIN_BATCH}, cps {K1_CPS_T}, fold counters {'ok' if _K1_CNT else 'none'};\"\n",
                 "f\" route {K1_ROUTE_MIN_BATCH}, cps {K1_CPS_T}, fold counters {'ok' if _K1_CNT else 'none'};\"\n"
                 "          f\" k2a v2.2 route {K2A_ROUTE} plan {K2A_V2_PLAN};\"\n", "prepare summary")
    return head + section + tail, (lay, k2a, i0, i1, cases)


# ================================================================================================ verify
def undefined_globals(text, fname):
    """Global names read inside any function scope that are neither module-level bindings, imports nor builtins."""
    top = symtable.symtable(text, fname, "exec")
    module_names = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    bad = set()

    def walk(t):
        for ch in t.get_children():
            if isinstance(ch, symtable.Function):
                for s in ch.get_symbols():
                    if s.is_global() and s.is_referenced():
                        n = s.get_name()
                        if n not in module_names and not hasattr(builtins, n):
                            bad.add((ch.get_name(), n))
            walk(ch)
    walk(top)
    return bad


def verify(text, ctx, chain_sha):
    lay, k2a, i0, i1, cases = ctx
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text)
    try:
        py_compile.compile(str(tmp), doraise=True)
        out = Src(tmp)
    finally:
        tmp.unlink()
    rep = [f"chain: gen_k3.py --check OK; k3_v06.py == gen_k3_v06 output (sha1 {chain_sha[:12]})"]

    check(text.splitlines()[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1, "nonce")
    # K2A sources: emitted values == k2a_v22's == pinned sha1; raw literal text identical
    for nm in SOURCES:
        v = out.value("K2A_" + nm)
        check(v == k2a.value(nm) and sha(v) == K2A_SHA1[nm], f"K2A_{nm} differs from k2a_v22.py {nm}")
        check(out.seg("K2A_" + nm) == "K2A_" + k2a.seg(nm), f"K2A_{nm} raw literal text differs")
        rep.append(f"K2A_{nm}: byte-identical to k2a_v22.py {nm} (sha1 {sha(v)[:12]}, {len(v)} chars)")
    # K1 sources and everything K1: identical to v06
    for nm in ("K1_CPP_SRC", "K1_CUDA_SRC"):
        check(out.seg(nm) == lay.seg(nm), f"{nm} changed vs k3_v06")
    rep.append(f"K1 sources identical to k3_v06 (sha1 {sha(out.value('K1_CUDA_SRC'))[:12]})")
    # measured constants
    check(out.value("K2A_V2_PLAN") == WANT_PLAN == k2a.value("V2_PLAN"), "K2A_V2_PLAN")
    check(val(out, "K2A_BUCKETS", K2A_M_MAX=320) == WANT_BUCKETS == val(k2a, "BUCKETS", M_MAX=320), "K2A_BUCKETS")
    check(out.value("K2A_ROUTE") == WANT_ROUTE, "K2A_ROUTE")
    check(out.value("K2A_SELF_TEST_CASES") == cases, "K2A_SELF_TEST_CASES")
    check(out.value("USE_K2A") is True and out.value("USE_K1") is True and out.value("K2A_SELF_TEST") is True, "switches")
    check(out.value("BLOCK_SIZE_M_RANGES") == ((49, 1 << 30, 16),), "BLOCK_SIZE_M_RANGES")
    rep.append(f"K2A_V2_PLAN = k2a_v22 V2_PLAN {WANT_PLAN}; K2A_ROUTE = {WANT_ROUTE}; self-test cases {len(cases)}")

    # v22 pieces: ast-identical to k2a_v22.py after the rename map
    same = []
    for nm in V22_ALL:
        ref = ast.parse(rename(k2a.seg(nm)) if nm not in SOURCES else "K2A_" + k2a.seg(nm)).body[0]
        got = out.nodes[RENAME[nm]]
        check(dump(ref) == dump(got), f"{RENAME[nm]} differs from k2a_v22.py {nm} after renames")
        same.append(RENAME[nm])
    rep.append(f"k2a_v22 pieces ast-identical after renames ({len(same)}): {', '.join(same)}")
    for nm in GLUE_FNS:
        check(nm in out.nodes, f"glue {nm} missing")

    # outside the K2a slice: every top-level node ast-identical to k3_v06, except the listed ones
    slice_v06 = {n for n, node in lay.nodes.items() if i0 < node.lineno <= i1}
    allowed = {"prepare", "USE_K2A"}
    removed = {"K2A_MIN_TOKENS", "K2A_MAX_TOKENS"}
    changed = []
    for name, node in lay.nodes.items():
        if name in slice_v06 or name in removed:
            continue
        check(name in out.nodes, f"{name} lost")
        if dump(out.nodes[name]) != dump(node):
            changed.append(name)
    check(set(changed) <= allowed, f"unexpected changes outside the K2a slice: {sorted(set(changed) - allowed)}")
    for nm in V0_GONE:
        check(nm not in out.nodes, f"{nm} should be gone")
        check(re.search(r"(?<![A-Za-z0-9_])" + nm + r"(?![A-Za-z0-9_])", text) is None, f"{nm} still referenced")
    pv, pn = ast.unparse(lay.nodes["prepare"]).splitlines(), ast.unparse(out.nodes["prepare"]).splitlines()
    pdiff = [ln for ln in difflib.unified_diff(pv, pn, lineterm="", n=0) if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    check(all("k2a v2.2 route" in ln or "fold counters" in ln for ln in pdiff), f"prepare diff beyond summary: {pdiff}")
    for keep in ("kernel_fn", "_experts", "_r1_attention", "_r1_prepare", "_k3_disable", "_k1_run", "_k1_prepare",
                 "_k1_self_test", "BLOCK_SIZE_M_RANGES", "K1_ROUTE_MIN_BATCH", "_marlin_bank", "_local_id_table"):
        check(keep not in changed and dump(out.nodes[keep]) == dump(lay.nodes[keep]), f"{keep} changed")
    rep.append(f"outside the K2a slice: {len(lay.nodes) - len(slice_v06) - len(removed) - len(changed)} top-level "
               f"defs/assignments ast-identical to k3_v06 (kernel_fn, _experts, _r1_attention, _r1_prepare, K1 slice, "
               f"r1 levers incl. BLOCK_SIZE_M_RANGES); changed: {sorted(changed)} (prepare: summary print only); "
               f"removed: {sorted(removed)}; new: K2A_ROUTE")

    def unnamed(src, lo, hi):
        return [n for n in src.tree.body if not Src._names(n) and not (lo < n.lineno <= hi)
                and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    o0 = out.find_line("# K2a: custom NVFP4 W4A16 experts v2.2 (kernels/muse_moe_w4a16/k2a_v22.py, nonce k2a-v22-1; box judge clone 1.471x") - 1
    o1 = out.find_line("# Rung 1 (kernels/muse_decode_layer/r1_glue.py, nonce r1-1): unchanged, and the fallback of both pieces.") - 1
    u6, u7 = unnamed(lay, i0, i1), unnamed(out, o0, o1)
    check(len(u6) == len(u7), "unnamed top-level statement count changed")
    ud = [b for a, b in zip(u6, u7) if dump(a) != dump(b)]
    check(len(ud) == 1 and "[k3] switches" in ast.unparse(ud[0]), f"unnamed statements changed: {[ast.unparse(n)[:80] for n in ud]}")
    inslice = [n for n in out.tree.body if o0 < n.lineno <= o1 and not Src._names(n)]
    check(len(inslice) == 1 and isinstance(inslice[0].targets[0], ast.Tuple), "K2a slice: only the PH_* tuple is unnamed")
    rep.append("unnamed top-level statements: identical except the [k3] switches print")

    # static scan
    for node in ast.walk(out.tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                check(m.split(".")[0] not in G.BANNED_NAMES, f"banned import {m}")
        if isinstance(node, ast.Name):
            check(node.id not in G.BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
        if isinstance(node, ast.Attribute):
            check(node.attr not in ("environ", "system", "popen"), f"banned attribute {node.attr}")

    def sync_fns(src):
        res = set()
        for node in src.tree.body:
            if isinstance(node, ast.FunctionDef):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and sub.attr in SYNC_ATTRS:
                        res.add(node.name)
        return res
    s6, s7 = sync_fns(lay), sync_fns(out)
    allowed_new = {"_k2a_prepare", "_k2a_v2_prepare", "_k2a_v2_reset", "_k2a_ref_fp32"}
    check(s7 - s6 <= allowed_new, f"new functions with host syncs: {sorted(s7 - s6 - allowed_new)}")
    for hot in ("kernel_fn", "_k2a_experts", "_k2a_v2_call", "_k2a_supported", "_k2a_bucket_of", "_experts"):
        check(hot not in s7, f"{hot} contains a host-sync call")
    bad = undefined_globals(text, str(OUT))
    check(not bad, f"undefined globals: {sorted(bad)}")
    rep.append(f"static: no os/sys/subprocess/environ/threading/eval/exec/open; no undefined globals; host-sync calls "
               f"only in {', '.join(sorted(s7))}")
    calls = [n for n in ast.walk(out.nodes["_k2a_v2_call"]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "run"]
    check(len(calls) == 1 and len(calls[0].args) == 29, "mod.run call has 29 args")
    return out, rep


def diff_summary(new_text, ctx):
    lay, i0, i1 = ctx[0], ctx[2], ctx[3]
    a, b = lay.text.splitlines(), new_text.splitlines()
    o0 = b.index("# K2a: custom NVFP4 W4A16 experts v2.2 (kernels/muse_moe_w4a16/k2a_v22.py, nonce k2a-v22-1; box judge clone 1.471x") - 1
    o1 = b.index("# Rung 1 (kernels/muse_decode_layer/r1_glue.py, nonce r1-1): unchanged, and the fallback of both pieces.") - 1
    lines = []
    for (a0_, a1_, b0_, b1_) in ((0, i0, 0, o0), (i1, len(a), o1, len(b))):
        sm = difflib.SequenceMatcher(a=a[a0_:a1_], b=b[b0_:b1_], autojunk=False)
        for tag, x0, x1, y0, y1 in sm.get_opcodes():
            if tag == "equal":
                continue
            first = (b[b0_ + y0] if y1 > y0 else a[a0_ + x0]).strip()[:70]
            lines.append(f"  {tag:7s} k3_v06 l.{a0_ + x0 + 1}-{a0_ + x1} -> k3_v07 l.{b0_ + y0 + 1}-{b0_ + y1}: {first}")
        if a0_ == 0:
            lines.append(f"  replace k3_v06 l.{i0 + 1}-{i1} (K2a v0 slice, {i1 - i0} lines) -> k3_v07 l.{o0 + 1}-{o1}"
                         f" (K2A v2.2 slice, {o1 - o0} lines)")
    return lines, len(a), len(b)


def main(argv):
    chain_sha = check_chain()
    text, ctx = generate()
    out, report = verify(text, ctx, chain_sha)
    hunks, na, nb = diff_summary(text, ctx)
    print(f"[gen_k3_v07] {SRC_LAYER.relative_to(ROOT)} ({na} lines) + {SRC_K2A.relative_to(ROOT)} -> "
          f"{OUT.relative_to(ROOT)} ({nb} lines), nonce {NONCE}")
    print(f"[gen_k3_v07] diff vs k3_v06: {len(hunks)} hunks")
    for h in hunks:
        print(h)
    for r in report:
        print(f"[gen_k3_v07] OK  {r}")
    if "--check" in argv:
        disk = OUT.read_text() if OUT.exists() else ""
        check(disk == text, f"{OUT.name} on disk differs from the generator's output (regenerate)")
        print(f"[gen_k3_v07] --check: {OUT.name} on disk is up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[gen_k3_v07] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}); all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

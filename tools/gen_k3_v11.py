"""gen_k3_v11.py: k3_v11.py = k3_v10.py (as gen_k3_v10.py generates it) + K2A v3.3 as the expert path.

    python3 -I tools/gen_k3_v11.py            # writes kernels/muse_decode_layer/k3_v11.py, asserts the invariants
    python3 -I tools/gen_k3_v11.py --check    # compares with the file on disk, writes nothing

Changes vs k3_v10 (everything else asserted ast-identical to gen_k3_v10's output):
  * K2A v3.3 (kernels/muse_moe_w4a16/k2a_v33.py): sources byte-identical, host pieces ast-identical after a
    K2A33_/_k2a33_ rename; its packers / BITPOS3 are asserted identical to v3.1's and reuse the layer's _k2a31_ ones.
    The layer call passes v3.3's per-bucket warm threshold (V3_HOT_T) as its own self-test does.
  * Expert chain in _v8_experts: v3.3 -> v3.1 -> v3 -> v2.2 -> Marlin; prepare builds v3.1 / v3 / v2.2 only for buckets
    every newer kernel failed.
  * The single combined CUDA build now carries v3.3 instead of v3.1 (both use run3 / attrs3v in namespace k3), compiled
    for sm_90a (v3.3's wgmma). v3.1 builds itself only as a fallback.
"""

import ast
import importlib.util
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "kernels/muse_decode_layer/k3_v11.py"
SRC_K2A33 = ROOT / "kernels/muse_moe_w4a16/k2a_v33.py"
NONCE = "k3-v11-1"

_s10 = importlib.util.spec_from_file_location("gen_k3_v10", ROOT / "tools/gen_k3_v10.py")
G10 = importlib.util.module_from_spec(_s10)
_s10.loader.exec_module(G10)
G9, G8, G = G10.G9, G10.G8, G10.G
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha

K2A33_ROUTE = {1: "v33", 8: "v33", 16: "v33", 32: "v33", 128: "v33", 256: "v33"}
K2A33_NAMES = ("HOT_T", "HOT_W4", "M3", "SW3", "V3_VARIANTS", "V3_PLAN", "V3_HOT_T", "stage_repack", "KERNEL_SRC_V3",
               "CPP_SRC_V3", "_wg_specs", "build_ext_v3", "v3_prepare", "V3_HINT", "v3_call", "v3_reset",
               "V3_SELF_TEST_CASES")
K2A33_RENAME = {
    "HOT_T": "K2A33_HOT_T", "HOT_W4": "K2A33_HOT_W4", "M3": "K2A33_M3", "SW3": "K2A33_SW3",
    "V3_VARIANTS": "K2A33_VARIANTS", "V3_PLAN": "K2A33_PLAN", "V3_HOT_T": "K2A33_HOT_T_BUCKET",
    "stage_repack": "_k2a33_stage_repack", "KERNEL_SRC_V3": "K2A33_KERNEL_SRC", "CPP_SRC_V3": "K2A33_CPP_SRC",
    "_wg_specs": "_k2a33_wg_specs", "build_ext_v3": "_k2a33_build_ext", "v3_prepare": "_k2a33_prepare_st",
    "V3_HINT": "K2A33_HINT", "v3_call": "_k2a33_call", "v3_reset": "_k2a33_reset",
    "V3_SELF_TEST_CASES": "K2A33_SELF_TEST_CASES",
    # identical to v3.1's (asserted): the layer's _k2a31_ copies
    "BITPOS3": "K2A31_BITPOS3", "pack_words": "_k2a31_pack_words", "prepack_expert": "_k2a31_prepack_expert",
    # shared with v2.2 (asserted)
    "E_LOC": "K2A_E_LOC", "D_HIDDEN": "K2A_D_HIDDEN", "D_EXPERT": "K2A_D_EXPERT", "W_SHIFT": "K2A_W_SHIFT",
    "META_HDR": "K2A_META_HDR", "TOP_K": "K2A_TOP_K", "BITPOS": "K2A_BITPOS", "reencode_scales": "_k2a_reencode_scales",
    "tile_row_perm": "_k2a_tile_row_perm",
}
_K2A33_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(sorted(map(re.escape, K2A33_RENAME), key=len, reverse=True))
                       + r")(?![A-Za-z0-9_])")


def r33(text):
    return _K2A33_RE.sub(lambda m: K2A33_RENAME[m.group(1)], text)


def k2a33_glue():
    g = G9.k2a31_glue()
    for old, new in (("_v9_k2a31_", "_v11_k2a33_"), ("V9_K2A31_ROUTE", "V11_K2A33_ROUTE"), ("_k2a31_", "_k2a33_"),
                     ("K2A31_", "K2A33_"), ('"k2a31"', '"k2a33"'), ("[k2a31]", "[k2a33]"), ("K2A v3.1", "K2A v3.3"),
                     ('v == "v31"', 'v == "v33"'), ("experts stay on K2A v3 / v2.2 / Marlin",
                                                      "experts stay on K2A v3.1 / v3 / v2.2 / Marlin")):
        check(old in g, f"glue rename {old}")
        g = g.replace(old, new)
    # v3.3's call takes the bucket's warm threshold (k2a_v33.py _v3_try / kernel_fn pass V3_HOT_T.get(rep))
    g = patch(g, "    return _k2a33_call(st, data, K2A33_PLAN[_k2a_bucket_of(x.shape[0])])\n",
              "    rep = _k2a_bucket_of(x.shape[0])\n"
              "    return _k2a33_call(st, data, K2A33_PLAN[rep], hot_t=K2A33_HOT_T_BUCKET.get(rep))\n", "v33 call hot_t")
    return g


def generate():
    text, _ = G10.generate()
    base10 = text
    k31, k33 = Src(G9.SRC_K2A31), Src(SRC_K2A33)
    k22 = Src(ROOT / "kernels/muse_moe_w4a16/k2a_v22.py")
    for nm in ("BITPOS3", "pack_words", "prepack_expert"):
        check(dump(k31.nodes[nm]) == dump(k33.nodes[nm]), f"k2a_v33 {nm} differs from k2a_v31's")
    for nm in G9.K2A31_SHARED:
        check(dump(k22.nodes[nm]) == dump(k33.nodes[nm]), f"k2a_v33 {nm} differs from k2a_v22's")
    lines = text.split("\n")
    check(lines[1].startswith("# nonce k3-v10-1 "), "k3_v10 nonce")
    text = patch(text, lines[1] + "\n", f"# nonce {NONCE} (k3_v10 + K2A v3.3 experts in the combined build; k3_v11.md)\n",
                 "nonce")
    parts = ["# K2A v3.3 slice: kernels/muse_moe_w4a16/k2a_v33.py (nonce k2a-v33-1), sources byte-identical, host code\n"
             "# renamed K2A33_/_k2a33_ (packers identical to v3.1's: the layer's _k2a31_ copies).\n"
             "V11_K2A33_ROUTE = " + repr(K2A33_ROUTE) + "   # bucket rep -> \"v33\" | else K2A v3.1 / v3 / v2.2 / Marlin\n"
             "V11_BUILD_V31_ALWAYS = False   # lab A/B only: build K2A v3.1 even when v3.3 covers every bucket\n"]
    for nm in K2A33_NAMES:
        if nm in ("KERNEL_SRC_V3", "CPP_SRC_V3"):
            parts.append(K2A33_RENAME[nm] + k33.seg(nm)[len(nm):] + "\n")
        else:
            parts.append(r33(k33.seg(nm)) + "\n")
    parts.append(k2a33_glue())
    banner = ("# ==============================================================================================\n"
              "# Rung 1 (kernels/muse_decode_layer/r1_glue.py, nonce r1-1): unchanged, and the fallback of both pieces.\n")
    text = patch(text, banner, "".join(parts) + "\n\n" + banner, "K2A v3.3 section")
    # expert chain: v3.3 first
    text = patch(text, '''    rep = _k2a_bucket_of(tokens)
    k31 = state.get("k2a31")
''', '''    rep = _k2a_bucket_of(tokens)
    k33 = state.get("k2a33")
    if (k33 is not None and rep is not None and V11_K2A33_ROUTE.get(rep) == "v33" and k33["ok"].get(rep)
            and tokens <= K2A33_M3 and cfg["local_experts"] == K2A_E_LOC and cfg["top_k"] == K2A_TOP_K
            and cfg["n_experts"] == N_EXPERTS):
        try:
            return _v11_k2a33_call(expert_x, local_ids, topk_weights, k33["st"])
        except Exception as ex:  # noqa: BLE001
            _k3_disable(state, "k2a33", None, "K2A v3.3", ex)
    k31 = state.get("k2a31")
''', "_v8_experts chain")
    # prepare chain: v3.3 -> v3.1 (only for buckets v3.3 failed) -> v3 -> v2.2
    text = patch(text, '''    state["k2a31"] = None
    if USE_K2A:
        state["k2a31"] = _v9_k2a31_prepare(weights, t0)
        k31ok = state["k2a31"]["ok"] if state["k2a31"] is not None else {}
''', '''    state["k2a31"] = None
    state["k2a33"] = None
    if USE_K2A:
        state["k2a33"] = _v11_k2a33_prepare(weights, t0)
        k33ok = state["k2a33"]["ok"] if state["k2a33"] is not None else {}
        need_v31 = [rep for rep in V9_K2A31_ROUTE if not (V11_K2A33_ROUTE.get(rep) == "v33" and k33ok.get(rep))]
        state["k2a31"] = _v9_k2a31_prepare(weights, t0) if (need_v31 or V11_BUILD_V31_ALWAYS) else None
        k31ok = dict(state["k2a31"]["ok"]) if state["k2a31"] is not None else {}
        for rep, ok in k33ok.items():
            if ok and V11_K2A33_ROUTE.get(rep) == "v33":
                k31ok[rep] = True   # covered by v3.3: older kernels are not needed for it
''', "prepare chain")
    text = patch(text, '''experts={'K2A v3.1' if state.get('k2a31') is not None else''',
                 '''experts={'K2A v3.3' if state.get('k2a33') is not None else 'K2A v3.1' if state.get('k2a31') is not None else''',
                 "prepare summary")
    # combined build: v3.3 instead of v3.1, sm_90a; v3.1 builds itself
    text = patch(text, '''    cases = "\\n".join("    case %d: launch3<%d, %d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, *v)
                      for i, v in enumerate(K2A31_VARIANTS))
    attrs = "\\n".join("    case %d: return attrs3<%d, %d, %d>();" % (i, *v) for i, v in enumerate(K2A31_VARIANTS))
    k31 = K2A31_KERNEL_SRC.replace("@CASES3@", cases).replace("@ATTRS3@", attrs)
    cpp = "\\n".join((K1_CPP_SRC, K2A31_CPP_SRC, R1B_CPP_SRC, V8_TAIL_CPP_SRC))
    cuda = "\\n".join((K1_CUDA_SRC, k31, R1B_CUDA_SRC, V8_TAIL_CUDA_SRC))''', '''    vv = [tuple(v) + (0,) * (4 - len(v)) for v in K2A33_VARIANTS]   # exactly as _k2a33_build_ext
    cases = "\\n".join("    case %d: launch3<%d, %d, %d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, *v)
                      for i, v in enumerate(vv))
    attrs = "\\n".join("    case %d: return attrs3<%d, %d, %d, %d>();" % (i, *v) for i, v in enumerate(vv))
    k33 = (K2A33_KERNEL_SRC.replace("@CASES3@", cases).replace("@ATTRS3@", attrs)
           .replace("@WGSPECS@", _k2a33_wg_specs()))
    cpp = "\\n".join((K1_CPP_SRC, K2A33_CPP_SRC, R1B_CPP_SRC, V8_TAIL_CPP_SRC))
    cuda = "\\n".join((K1_CUDA_SRC, k33, R1B_CUDA_SRC, V8_TAIL_CUDA_SRC))''', "combined sources")
    text = patch(text, '''            extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-gencode=arch=compute_90,code=sm_90"],''',
                 '''            extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a"],''',
                 "combined arch sm_90a")
    text = patch(text, '''        print(f"[v10] combined build (K1 + K2A v3.1 + R1B + tail) in''',
                 '''        print(f"[v10] combined build (K1 + K2A v3.3 + R1B + tail) in''', "combined log")
    text = patch(text, '''    """K1 + K2A v3.1 + R1B + V8 tail in ONE load_inline:''', '''    """K1 + K2A v3.3 + R1B + V8 tail in ONE load_inline:''',
                 "combined doc")
    # v3.1 now builds itself; v3.3 takes the combined module
    text = patch(text, '''def _v10_k2a31_build(name, variants, verbose=False):
    """_k2a31_build_ext's contract, served by the combined module (built with K2A31_VARIANTS)."""
    if _V10_MOD is None or list(variants) != list(K2A31_VARIANTS):
        return _K2A31_OWN_BUILD(name, variants, verbose)
    return _V10_MOD, 0.0''', '''def _v10_k2a31_build(name, variants, verbose=False):
    """v11: the combined module carries v3.3, so v3.1 (a fallback only) always builds itself."""
    return _K2A31_OWN_BUILD(name, variants, verbose)


def _v11_k2a33_build(name, variants, verbose=False):
    """_k2a33_build_ext's contract, served by the combined module (built with K2A33_VARIANTS)."""
    if _V10_MOD is None or list(variants) != list(K2A33_VARIANTS):
        return _K2A33_OWN_BUILD(name, variants, verbose)
    return _V10_MOD, 0.0''', "build hooks")
    text = patch(text, "\ndef _k2a33_prepare_st(", "\n_K2A33_OWN_BUILD = _k2a33_build_ext\n\n\ndef _k2a33_prepare_st(",
                 "k2a33 own build alias")
    text = patch(text, '''        st = _k2a33_prepare_st(w, log=lambda s: print(s, flush=True))
''', '''        global _k2a33_build_ext
        _k2a33_build_ext = _v11_k2a33_build   # the combined module (falls back to _K2A33_OWN_BUILD)
        st = _k2a33_prepare_st(w, log=lambda s: print(s, flush=True))
''', "k2a33 module")
    return text, base10


def verify(text, base10):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text)
    btmp = OUT.with_suffix(".py.gen_base")
    btmp.write_text(base10)
    try:
        py_compile.compile(str(tmp), doraise=True)
        out, b10 = Src(tmp), Src(btmp)
    finally:
        tmp.unlink()
        btmp.unlink()
    rep = []
    check(text.splitlines()[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1, "nonce")
    k33 = Src(SRC_K2A33)
    for nm in K2A33_NAMES:
        if nm in ("KERNEL_SRC_V3", "CPP_SRC_V3"):
            check(out.value(K2A33_RENAME[nm]) == k33.value(nm), f"{nm} not byte-identical")
        else:
            check(dump(ast.parse(r33(k33.seg(nm))).body[0]) == dump(out.nodes[K2A33_RENAME[nm]]), f"k2a_v33 {nm} differs")
    rep.append(f"k2a_v33: sources byte-identical (sha1 {sha(k33.value('KERNEL_SRC_V3'))[:12]}), "
               f"{len(K2A33_NAMES) - 2} host pieces ast-identical after renames; packers == v3.1's, shared == v2.2's")
    allowed = {"prepare", "_v8_experts", "_v10_build_all", "_v10_k2a31_build"}
    changed = [n for n, node in b10.nodes.items() if n not in out.nodes or dump(out.nodes[n]) != dump(node)]
    check(set(changed) <= allowed, f"unexpected changes vs k3_v10: {sorted(set(changed) - allowed)}")
    rep.append(f"vs k3_v10 (gen output): {len(b10.nodes) - len(changed)} top-level nodes ast-identical; changed {sorted(changed)}")
    for node in ast.walk(out.tree):
        if isinstance(node, ast.Name):
            check(node.id not in G.BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
    for fn in ("kernel_fn", "_v8_experts", "_v11_k2a33_call", "_k2a33_call"):
        for sub in ast.walk(out.nodes[fn]):
            check(not (isinstance(sub, ast.Attribute) and sub.attr in G.SYNC_ATTRS), f"{fn} has a host sync")
    bad = G8.G7.undefined_globals(text, str(OUT))
    check(not bad, f"undefined globals: {sorted(bad)}")
    check(len(text.encode()) < 1_000_000, "file >= 1 MB")
    rep.append(f"static ok; {len(text.encode())} bytes")
    return rep


def main(argv):
    text, base10 = generate()
    for r in verify(text, base10):
        print(f"[gen_k3_v11] OK  {r}")
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text() == text, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_v11] --check: up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text)
    print(f"[gen_k3_v11] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}, {len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

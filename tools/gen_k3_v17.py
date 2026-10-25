"""gen_k3_v17.py: k3_v17.py = k3_v16.py (as gen_k3_v16.py generates it) + K2A v3.7 in place of K2A v3.6, with ONE new route: bucket 128 (T = 128, the
verify-16 load) on vid 5.

    python3 -I tools/gen_k3_v17.py            # writes kernels/muse_decode_layer/k3_v17.py, asserts the invariants
    python3 -I tools/gen_k3_v17.py --check    # compares with the file on disk, writes nothing

Changes vs k3_v16 (everything else asserted ast-identical to gen_k3_v16's output, which is asserted == k3_v16.py on disk):
  * K2A v3.7 (kernels/muse_moe_w4a16/k2a_v37.py) replaces the K2A v3.6 module inside the combined build. Its CUDA source is v3.6's plus additions
    (asserted: six v3.6 lines change, each adding a defaulted template parameter whose default restores the old expression; nothing else is
    removed), so vids 0-3 are v3.6's code and ONE module serves every bucket. K2A33_KERNEL_SRC is k2a_v37's KERNEL_SRC_V3 byte for byte;
    K2A33_VARIANTS gains (1, 3, 2, 3, 1) and (1, 3, 2, 3, 4) (vids 4, 5 = launch37<1, 3> and launch37<4, 3>); _k2a33_prepare_st is k2a_v37's
    (default build name only); _k2a36_variant_cases (HK 3 -> launch37), _k2a36_gate_cases, _k2a36_gate_attrs and _k2a36_hot_code (HK 3 shares the
    role-split code of HK 2) are k2a_v37's (names kept).
  * V17 section: K2A37_ENABLE / K2A37_PLAN / K2A37_DQ, k2a_v37's dyn_code and v37_call (renamed _k2a37_*), and _v17_k2a37_prepare (the bucket's
    k2a_v37 self-test cases, two calls each, plus the ticket counters back at zero).
  * _v11_k2a33_call: the v3.7 route first (K2A37_PLAN, its prepare self-test passed, K2A37_ENABLE on; read at call time), then k3_v16's v3.6 route,
    then v3.3's plan; _v11_k2a33_prepare calls _v17_k2a37_prepare after _v14_k2a36_prepare; _v10_build_all only changes its log line.
"""

import ast
import importlib.util
import json
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "kernels/muse_decode_layer/k3_v17.py"
V16_FILE = ROOT / "kernels/muse_decode_layer/k3_v16.py"
SRC_K2A36 = ROOT / "kernels/muse_moe_w4a16/k2a_v36.py"
SRC_K2A37 = ROOT / "kernels/muse_moe_w4a16/k2a_v37.py"
SRC_BENCH = ROOT / "tools/box/k37_bench.py"
AB_JSON = ROOT / "tools/box/ab_v17a.json"
NONCE = "k3-v17-1"

_s16 = importlib.util.spec_from_file_location("gen_k3_v16", ROOT / "tools/gen_k3_v16.py")
G16 = importlib.util.module_from_spec(_s16)
_s16.loader.exec_module(G16)
G15, G14, G13, G8, G = G16.G15, G16.G14, G16.G13, G16.G8, G16.G
G11 = G14.G11
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha

# bucket rep -> (vid, hot, dyn): k37_bench row `v37` = k2a_v37.V37_PLAN (layer routing, -13.9 us; moe routing -5.4 us) at bucket 128 only
K2A37_PLAN = {128: (5, (8, 0, 12), (8, 0, 3))}
V17_NODES = {"K2A37_ENABLE", "K2A37_PLAN", "K2A37_DQ", "_k2a37_dyn_code", "_k2a37_call", "_v17_k2a37_prepare"}
CHANGED_NODES = {"K2A33_VARIANTS", "K2A33_KERNEL_SRC", "_k2a33_prepare_st", "_k2a36_variant_cases", "_k2a36_gate_cases", "_k2a36_gate_attrs",
                 "_k2a36_hot_code", "_v10_build_all", "_v11_k2a33_prepare", "_v11_k2a33_call"}
CALL_TIME_FNS = G16.CALL_TIME_FNS + ("_k2a37_call", "_k2a37_dyn_code")
# k2a_v36 nodes (as named in k2a_v36 / k2a_v37) that k2a_v37 changes, among the ones k3_v14 embedded
SWAPPED = ("V3_VARIANTS", "KERNEL_SRC_V3", "v3_prepare")                       # replaced in place under their K2A33_ / _k2a33_ names
SWAPPED_SLICES = ("variant_cases", "_gate_cases", "_gate_attrs", "hot_code")   # replaced in place under their _k2a36_ names
NEW_SLICES = ("dyn_code", "v37_call")                                          # new, renamed _k2a37_*
R37_EXTRA = {"dyn_code": "_k2a37_dyn_code", "v37_call": "_k2a37_call"}
R37_MAP = {**G14.R36_MAP, **R37_EXTRA}
_R37_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(sorted(map(re.escape, R37_MAP), key=len, reverse=True)) + r")(?![A-Za-z0-9_])")
# the lines of k2a_v36's KERNEL_SRC_V3 that k2a_v37's changes (stripped, in file order), and the opcode shape of the whole diff
KERNEL_REMOVED = (
    "template <int NC, int TPH = 64, bool HGE = false>",
    "template <int NST, int STG, int MODE>",
    "flush_item<1, MODE>(P, acc, whole, e, rb, c0, ntok, npart, !whole);",
    "template <int NQ, int NST, int STG, int MODE>",
    "if (NOSWI && whole && !ho) { atomicExch(P.err, 1); atomicExch(P.err + 3, 1); }   // barrier-id budget exhausted",
    "flush_item<NQ, MODE, NOSWI>(P, acc, whole, e, rb, c0, ntok, npart, NOSWI ? (ho || !whole) : !whole);",
)
KERNEL_OPS = (("replace", 1, 2), ("insert", 0, 1), ("replace", 1, 1), ("replace", 1, 1), ("replace", 1, 1), ("replace", 2, 2), ("insert", 0, 415),
              ("insert", 0, 47))
# every line of the k2a_v37 kernel text that names the two new defaulted template parameters (stripped), by parameter
MB_LINES = ("template <int NST, int STG, int MODE, bool MB = false>",
            "flush_item<1, MODE>(P, acc, whole, e, rb, c0, ntok, npart, MB ? false : !whole);",
            "template <int NQ, int NST, int STG, int MODE, bool MB = false>",
            "if (!MB && NOSWI && whole && !ho) { atomicExch(P.err, 1); atomicExch(P.err + 3, 1); }   // barrier-id budget exhausted",
            "flush_item<NQ, MODE, NOSWI>(P, acc, whole, e, rb, c0, ntok, npart, MB ? false : (NOSWI ? (ho || !whole) : !whole));")
LIST_LINES = ("// LIST (v3.7): false skips the scatter pass (s_list / meta rows); K2 never reads s_list (its X rows are route positions).",
              "template <int NC, int TPH = 64, bool HGE = false, bool LIST = true>", "if (!LIST) return;")
# the only references to the K2A meta buffer in the kernel text (stripped): MH = 128 rows at and above, DQ37 = 120 ticket counters below
META_LINES = ("float* h; __nv_bfloat16* inter; float* y; int* meta; int* cnt; int* err;",
              "if (write_meta) P.meta[MH + p] = c;", "P.meta[MH + p] = -1;",
              "const int c = __ldcg(P.meta + MH + tok * 8 + tid);", "int* dq = P.meta + DQ37 + ((MODE == 1) ? 0 : 2);")
PRED_WRITTEN = re.compile(r"\bP\.(inter|h|y|ss|cnt|cnt2|meta|xw|xw2|out|sink)\b")   # buffers a kernel of the MoE call writes


def r37(text):
    return _R37_RE.sub(lambda m: R37_MAP[m.group(1)], text)


V17_DOC = '''K3 v17 = k3_v16 + K2A v3.7 (T = 128 route; everything else is k3_v16, asserted ast-identical by tools/gen_k3_v17.py except the
listed nodes):

  K2A v3.7 (kernels/muse_moe_w4a16/k2a_v37.py, k2a_v37_design.md): its CUDA source is v3.6's plus additions (only six v3.6 lines change, each
       adding a defaulted template parameter whose default restores the old expression, asserted), so vids 0-3 are v3.6's code and the one
       combined build serves every bucket: K2A33_KERNEL_SRC, K2A33_VARIANTS, _k2a33_prepare_st, _k2a36_variant_cases, _k2a36_gate_cases,
       _k2a36_gate_attrs, _k2a36_hot_code are k2a_v37's. New instantiations: vid 4 = launch37<1, 3> and vid 5 = launch37<4, 3> (HK 3: v3.6's
       role-split consumer code + a dynamic tail queue + a mailbox epilogue, any M <= 320). K1 stays a normal launch, K2 / K3 PDL.
  Route (K2A37_PLAN, tools/box/k37_bench.py row v37, layer routing decides): bucket 128 (T = 128, the verify-16 load) -> vid 5 with hot
       (8, 0, 12) and dyn (8, 0, 3) (K1 tail queue 8/64 of the stages in chunks of 8, K2 static): -13.9 us per call (3 jobs -12.6 .. -15.5; moe
       routing -1.6 .. -5.7). Buckets 1 / 8 (vid 0), 16 / 32 (vid 3, r8 / r12) and 256 (vid 2) are k3_v16's: the queue loses there.
  K2A37_ENABLE (read at call time) switches the route; it is used only after its prepare self-test passed (the bucket's k2a_v37 cases, M = 100
       and 128, two calls each, no error flag, ticket counters meta[120..123] back at zero); any failure or host exception keeps k3_v16's
       route (vid 2 at bucket 128, gated by v3.3's own self-test).

'''

V17_GLUE_HEAD = '''
# ---------------------------------------------------------------- V17: K2A v3.7 route (bucket 128: tail queue + mailbox epilogue)
K2A37_ENABLE = True   # master switch (read at call time; False at prepare also skips the v3.7 self-tests)
K2A37_PLAN = ''' + repr(K2A37_PLAN) + '''   # bucket rep -> (vid, hot, dyn); k37_bench row v37 = k2a_v37.V37_PLAN (hot = (hot_t, hot_mode, hot_w4), dyn = (f1, f2, log2 D): tail-queue fractions of K1 / K2 in 64ths, chunk = 2^log2 D stages)
K2A37_DQ = 120   # the HK 3 kernels' ticket counters meta[120..123] (K1 ticket, K1 done, K2 ticket, K2 done): zero at allocation (K2A_META_HDR = 128), written by nothing else, back at zero after every call

# kernels/muse_moe_w4a16/k2a_v37.py, verbatim after the K2A33_ / _k2a33_ / _k2a36_ / _k2a37_ renames (_k2a36_variant_cases, _k2a36_gate_cases,
# _k2a36_gate_attrs and _k2a36_hot_code above are k2a_v37's too).
'''

V17_GLUE = '''

def _v17_k2a37_prepare(st, w, dev):
    """K2A v3.7's routes (K2A37_PLAN) per bucket: k2a_v37's self-test cases of the bucket (bucket 128: M = 100 normal, M = 128 skewed) through
    _k2a37_call on the layer's id convention, two calls per case (the second checks that h / the counters were restored), rel-L2 < 1e-2,
    worst row < 5e-2, no timeout flag, finite, and the ticket counters meta[K2A37_DQ .. +3] back at zero after every call. Sets
    st["r37"] = {bucket rep: (vid, hot, dyn)} for the buckets that pass; any other bucket (a failure, a variant not built or below its CTAs/SM
    target, K2A37_ENABLE off at prepare) keeps k3_v16's route. Prepare-time only."""
    st["r37"] = {}
    print(f"[v17] levers: K2A37_ENABLE={K2A37_ENABLE} K2A37_PLAN={K2A37_PLAN} K2A36_PLAN={K2A36_PLAN} K2A33_PLAN={K2A33_PLAN} "
          f"K2A33 variants={len(K2A33_VARIANTS)}", flush=True)
    if not K2A37_ENABLE:
        return
    t2 = time.time()
    for rep, (vid, hot, dyn) in sorted(K2A37_PLAN.items()):
        if vid >= len(st["variants"]) or st["grid"][vid] < st["sms"] * st["variants"][vid][2]:
            print(f"[k2a37] bucket {rep}: v{vid} not built or below its CTAs/SM target -> k3_v16's route", flush=True)
            continue
        good_all = True
        for bs, seed, dist in K2A36_SELF_TEST_CASES:
            if _k2a_bucket_of(bs) != rep:
                continue
            try:
                d = _v14_self_test_inputs(bs, seed, dev, dist)
                ref = _k2a_ref_fp32(d, w)
                data = {"hidden_states": d["hidden_states"], "topk_ids": d["topk_ids"], "topk_weights": d["topk_weights"],
                        "config": {"local_start": 0}}
                for k in range(2):
                    st["err"].zero_()
                    got = _k2a37_call(st, data, vid, hot, dyn)
                    torch.cuda.synchronize()
                    tot, row = _k2a_rel_err(got, ref)
                    errf = int(st["err"][0])
                    dq = st["meta"][K2A37_DQ:K2A37_DQ + 4].tolist()
                    good = tot < 1.0e-2 and row < 5.0e-2 and not errf and not any(dq) and bool(torch.isfinite(got.float()).all())
                    print(f"[k2a37] layer self-test bs={bs} {dist} v{vid} hot={hot} dyn={dyn} call {k}: rel {tot:.3e} worst row "
                          f"{row:.3e} err={errf} counters={dq} -> {'ok' if good else 'FAIL'}", flush=True)
                    good_all &= good
            except Exception as ex:  # noqa: BLE001
                print(f"[k2a37] self-test bs={bs} v{vid} raised {ex!r}", flush=True)
                good_all = False
            _k2a33_reset(st)
            st["err"].zero_()
            if not good_all:
                break
        if good_all:
            st["r37"][rep] = (vid, tuple(hot), tuple(dyn))
        print(f"[k2a37] bucket {rep}: v{vid} hot {hot} dyn {dyn} -> {'route live' if good_all else 'FAILED: k3_v16 route'}", flush=True)
    print(f"[v17] K2A v3.7 routes live {st['r37']} (self-tests {time.time() - t2:.1f}s)", flush=True)
'''

OLD_PREP = '''    _v14_k2a36_prepare(st, w, dev)
    print(f"[v8] K2A v3.3 buckets {ok} in'''
NEW_PREP = '''    _v14_k2a36_prepare(st, w, dev)
    _v17_k2a37_prepare(st, w, dev)
    print(f"[v8] K2A v3.3 buckets {ok} in'''
OLD_CALL_DOC = '''with a line; the call is made on v3.3's plan, nothing was launched). Otherwise exactly k3_v13's call."""
'''
NEW_CALL_DOC = '''with a line; the call is made on v3.3's plan, nothing was launched). v17: K2A v3.7's vid 5 first where K2A37_PLAN routes the bucket,
    st["r37"] holds it (its prepare self-test passed) and K2A37_ENABLE allows it (read at call time); a host exception from that call drops
    the v3.7 routes for the run (once, with a line; the call then goes on with the v3.6 / v3.3 routes below, nothing was launched).
    Otherwise exactly k3_v16's call."""
'''
OLD_CALL_BODY = '''    rep = _k2a_bucket_of(x.shape[0])
    r36 = st["r36"].get(rep) if (K2A36_ENABLE and K2A36_BUCKETS.get(rep) and st.get("r36")) else None
'''
NEW_CALL_BODY = '''    rep = _k2a_bucket_of(x.shape[0])
    r37 = st["r37"].get(rep) if (K2A37_ENABLE and st.get("r37")) else None
    if r37 is not None:
        try:
            return _k2a37_call(st, data, r37[0], r37[1], r37[2])
        except Exception as ex:  # noqa: BLE001
            print(f"[k3] K2A v3.7 route raised ({ex!r}): v3.6 / v3.3 routes from here on", flush=True)
            st["r37"].clear()
    r36 = st["r36"].get(rep) if (K2A36_ENABLE and K2A36_BUCKETS.get(rep) and st.get("r36")) else None
'''
OLD_LOG = "combined build (K1 + K2A v3.6 [v3.3 = vids 0-2] + R1B + tail) in"
NEW_LOG = "combined build (K1 + K2A v3.7 [v3.3 = vids 0-2, v3.6 = vid 3, v3.7 = vids 4-5] + R1B + tail) in"
VARIANTS_COMMENT_V14 = ("   # (NC, NST, CTAs/SM, HK[, NCH]): HK 0 cold, 1 v3.3 HOTCAP (wgmma warm role), 2 v3.6 (cold NC1 + hot passes of 8 * NCH "
                        "tokens); vid 0-2 = v3.3's list\n\n")
VARIANTS_COMMENT_V17 = ("   # (NC, NST, CTAs/SM, HK[, NCH]): HK 0 cold, 1 v3.3 HOTCAP (wgmma warm role), 2 v3.6 (cold NC1 + hot passes of 8 * NCH "
                        "tokens), 3 v3.7 (HK 2's code + tail queue + mailbox epilogue, M <= 320); vid 0-2 = v3.3's list, 3 = v3.6, 4-5 = v3.7 "
                        "(vid 4 unrouted)\n\n")


def piece(src, nm):
    """The text of one K2A33 node as gen_k3_v14 emits it, for k2a_v37 (kernel / CPP sources keep k2a's text after the new name)."""
    if nm in ("KERNEL_SRC_V3", "CPP_SRC_V3"):
        return G11.K2A33_RENAME[nm] + src.seg(nm)[len(nm):] + "\n"
    return r37(src.seg(nm)) + "\n"


def generate():
    text, text15, kernel_src, fa5, route5 = G16.generate()
    check(V16_FILE.read_text(encoding="utf-8") == text, "gen_k3_v16's output differs from k3_v16.py on disk (regenerate v16 first)")
    base16 = text
    tmp = OUT.with_suffix(".py.gen_base")
    tmp.write_text(base16, encoding="utf-8", newline="\n")
    try:
        b16 = Src(tmp)
    finally:
        tmp.unlink()
    k36, k37 = Src(SRC_K2A36), Src(SRC_K2A37)
    # k2a_v37 vs k2a_v36: which K2A33 nodes changed
    for nm in G11.K2A33_NAMES:
        same = dump(k36.nodes[nm]) == dump(k37.nodes[nm])
        check(same != (nm in SWAPPED), f"k2a_v37 {nm}: expected {'changed' if nm in SWAPPED else 'ast-identical'} vs k2a_v36")
    for nm in SWAPPED_SLICES + ("v36_call", "synth_inputs", "V36_HOT", "V3_SELF_TEST_CASES"):
        same = dump(k36.nodes[nm]) == dump(k37.nodes[nm])
        check(same != (nm in SWAPPED_SLICES), f"k2a_v37 {nm}: expected {'changed' if nm in SWAPPED_SLICES else 'ast-identical'} vs k2a_v36")
    removed, _ = G14.diff_report(k36.value("KERNEL_SRC_V3"), k37.value("KERNEL_SRC_V3"))
    check(tuple(removed) == KERNEL_REMOVED, f"KERNEL_SRC_V3 v3.6 -> v3.7 removed / replaced lines: {removed}")
    lines = text.split("\n")
    check(lines[1].startswith("# nonce k3-v16-1 "), "k3_v16 nonce")
    hdr = "\n".join(lines[2:5]) + "\n"
    check(hdr.startswith("# GENERATED by tools/gen_k3_v16.py") and hdr.endswith("regenerate.\n"), "k3_v16 generated-by header")
    text = patch(text, lines[1] + "\n" + hdr,
                 f"# nonce {NONCE} (k3_v16 + K2A v3.7 in place of v3.6, T = 128 (bucket 128) on vid 5: K1 tail queue + mailbox epilogue; k3_v17_plan.md)\n"
                 "# GENERATED by tools/gen_k3_v17.py on top of gen_k3_v16.py's output (k3_v16.py; its chain is gen_k3_v15 / v14 /\n"
                 "# v13 / v12 ...) and kernels/muse_moe_w4a16/k2a_v37.py (K2A v3.7).\n"
                 "# Do not hand-edit: change a canonical source or a generator and regenerate.\n", "nonce + header")
    text = patch(text, '"""K3 v16 = k3_v15 + the FA4 v5 attention routes', '"""' + V17_DOC + 'K3 v16 = k3_v15 + the FA4 v5 attention routes', "docstring")
    # K2A33 nodes -> k2a_v37's (old text = the node as k3_v16 has it, taken from its own parse)
    for nm in SWAPPED:
        want = G11.K2A33_RENAME[nm]
        new = piece(k37, nm).rstrip("\n") + "\n"
        if nm == "V3_VARIANTS":
            new = (new.rstrip("\n") + VARIANTS_COMMENT_V17).rstrip("\n") + "\n"
            old = b16.seg(want)
            check(old.rstrip("\n") + "\n\n" == piece(k36, nm).rstrip("\n") + VARIANTS_COMMENT_V14, "K2A33_VARIANTS: k3_v16's text is not the expected one")
        else:
            old = b16.seg(want)
        text = patch(text, old, new, f"K2A v3.6 -> v3.7: {want}")
    for nm in SWAPPED_SLICES:
        want = R37_MAP[nm]
        text = patch(text, b16.seg(want), r37(k37.seg(nm)), f"K2A v3.6 -> v3.7: {want}")
    text = patch(text, OLD_LOG, NEW_LOG, "combined build log")
    # prepare / call
    text = patch(text, OLD_PREP, NEW_PREP, "prepare v17")
    text = patch(text, OLD_CALL_DOC, NEW_CALL_DOC, "k2a33 call doc")
    text = patch(text, OLD_CALL_BODY, NEW_CALL_BODY, "k2a33 call v3.7 route")
    # the V17 section, before the V12 levers banner (after the V14 section)
    banner = "# ---------------------------------------------------------------- V12 levers\n"
    slices = "".join(r37(k37.seg(nm)) + "\n" for nm in NEW_SLICES)
    text = patch(text, banner, V17_GLUE_HEAD + slices + V17_GLUE + "\n\n" + banner, "v17 section")
    return text, base16, b16, k36, k37, kernel_src


# --------------------------------------------------------------------------- checks
def check_ab_json(out, path):
    """ab_v17a.json: K2A37_ENABLE exists and is a bool; ref = K2A37_ENABLE off only (= k3_v16 behaviour), k37_on = defaults."""
    arms = json.loads(path.read_text())
    default = out.value("K2A37_ENABLE")
    check(default is True, "K2A37_ENABLE default must be True")
    want = {"ref": False, "k37_on": True}
    check([a["name"] for a in arms] == list(want), f"{path.name}: arms {[a['name'] for a in arms]} != {list(want)}")
    check(arms[0]["set"] == {"K2A37_ENABLE": False}, f"{path.name}: variant 0 must be ref = K2A37_ENABLE off only")
    rows = []
    for a in arms:
        eff = default
        for k, v in a.get("set", {}).items():
            check(k == "K2A37_ENABLE" and isinstance(v, bool), f"{path.name} {a['name']}: unknown or non-bool switch {k}")
            eff = v
        check(eff is want[a["name"]], f"{path.name} {a['name']}: K2A37_ENABLE resolves to {eff}, want {want[a['name']]}")
        rows.append(f"{a['name']}: K2A37_ENABLE {'on' if eff else 'off'}" + ("" if eff else " (= k3_v16 behaviour)"))
    return rows


def bench_check(k37):
    """K2A37_PLAN[128] == k2a_v37.V37_PLAN[128] == k37_bench's `v37` row (hk3 default hot (8, 0, 12) at bucket 128) == row k1q8d8's dyn (8, 0, 3)."""
    tree = ast.parse(SRC_BENCH.read_text(encoding="utf-8"))
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    hk3 = fns["hk3"]
    check([a.arg for a in hk3.args.args] == ["rep", "dyn", "hot32"] and [ast.literal_eval(d) for d in hk3.args.defaults] == [(8, 0, 12)],
          "k37_bench.hk3 signature / hot32 default")
    rows = [n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "ROWS"][0].value
    row = {ast.literal_eval(k): v for k, v in zip(rows.keys, rows.values)}
    check(ast.literal_eval(row["v37"].elts[1]) is None and "V37_PLAN" in ast.unparse(row["v37"].elts[2]), "k37_bench row v37 is k2a_v37.V37_PLAN")
    check("hk3(r, (8, 0, 3))" in ast.unparse(row["k1q8d8"].elts[1]), "k37_bench row k1q8d8 is hk3(r, (8, 0, 3))")
    return k37.value("V37_PLAN")


def pdl_audit_k2a(ksrc):
    """PDL staleness audit of the K2A source by text (ld.global.nc / __ldg / plain loads of a predecessor's output issued before griddepcontrol.wait
    read stale data). K2A call = K1 (normal launch), K2 (PDL secondary of K1), K3 (PDL secondary of K2).
    (1) launch chain of the HK 3 variants: K1 is `<<<>>>` (no programmatic attribute: everything before it, the router included, is complete),
        K2 / K3 go through cudaLaunchKernelEx with the programmatic-stream-serialization attribute.
    (2) __ldg (the only read-only-cache load in the text; no ld.global.nc / .nc / __restrict__) is applied to layer inputs and prepare-time
        tensors only (ids, wts, gsc13 / gsc2, nw, x): never to inter / h / y / ss / cnt / cnt2 / meta / xw / out.
    (3) K2 reads the one buffer its predecessor K1 writes (inter) with a bulk copy that is issued only if `waited` (after pdl_wait) or from the
        take_wait lambda, whose first statement is pdl_wait(); a producer that never took the wait does so before returning.
    (4) the ticket counters of K1 (meta[120], [121]) and K2 (meta[122], [123]) are different words, and meta[120..123] lie below MH = 128."""
    i0 = ksrc.index("static void launch37(")
    i1 = ksrc.index("static std::vector<int64_t> attrs37()", i0)
    launch = ksrc[i0:i1]
    check(launch.count("k3::k1_kernel37<NCH, NST><<<G, 320, SM, st>>>(p);") == 1 and "cudaLaunchKernelEx(&cfg, k3::k1_kernel37" not in launch
          and launch.count("cudaLaunchKernelEx(&cfg, k3::k2_kernel37<NCH, NST>, p)") == 1 and launch.count("cudaLaunchKernelEx(&cfg, k3::k3_kernel, p)") == 1
          and launch.count("cudaLaunchAttributeProgrammaticStreamSerialization") == 1 and launch.count("numAttrs = 1") == 1,
          "launch37: K1 must be a normal launch, K2 / K3 PDL")
    check(all(tok not in ksrc for tok in ("ld.global.nc", "__restrict__", ".nc.", "ld.global")), "K2A source: a read-only / non-coherent load spelling")
    ldg = [ln.strip() for ln in ksrc.splitlines() if "__ldg(" in ln]
    bad = [ln for ln in ldg if PRED_WRITTEN.search(ln)]
    check(ldg and not bad, f"__ldg on a buffer the MoE call writes: {bad}")
    g0 = ksrc.index("__device__ void gemm37(")
    g1 = ksrc.index("// K1 / K2 of the HK3 variants", g0)
    gemm = ksrc[g0:g1]
    tw0 = gemm.index("auto take_wait = [&]() {")
    tw1 = gemm.index("};", tw0)
    take = gemm[tw0:tw1]
    check(take.split("\n")[1].strip() == "pdl_wait();" and take.index("pdl_wait();") < take.index("bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);")
          and gemm.count("bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);") == 1, "gemm37.take_wait: pdl_wait must come first, before the deferred copies")
    inter_reads = [m.start() for m in re.finditer(r"= P\.inter \+", gemm)]
    check(len(inter_reads) == 1, "gemm37: exactly one read-side address of P.inter")
    tail = gemm[inter_reads[0]:]
    pat = re.compile(r"= P\.inter \+ [^;]*;\s*if \(waited\) \{\s*if \(lane == 0\) bulk_g2s\(dst \+ SW, src, xbytes, fb\);\s*\} else \{[^}]*dSrc\[d\] = src;")
    check(pat.match(tail), "gemm37: the X copy of K2 must be issued under `if (waited)` or deferred to take_wait")
    check(gemm.count("    if (!waited) take_wait();\n    return;") == 1, "gemm37: a producer that never waited must take the wait before returning")
    check(gemm.count("int* dq = P.meta + DQ37 + ((MODE == 1) ? 0 : 2);") == 1 and gemm.count("atomicAdd(dq, 1)") == 2 and gemm.count("atomicAdd(dq + 1, 1)") == 1
          and gemm.count("atomicExch(dq, 0); atomicExch(dq + 1, 0);") == 1, "gemm37: ticket counters")
    dq37 = int(re.search(r"constexpr int DQ37 = (\d+);", ksrc).group(1))
    mh = int(re.search(r"constexpr int MH = (\d+);", ksrc).group(1))
    check(dq37 == 120 and mh == 128 and dq37 + 4 <= mh, "DQ37 / MH")
    meta_lines = tuple(ln.strip() for ln in ksrc.splitlines() if re.search(r"\bP\.meta\b|\bint\* meta\b", ln))
    check(sorted(meta_lines) == sorted(META_LINES), f"references to the meta buffer: {meta_lines}")
    return dq37, mh, len(ldg)


def verify(text, base16, b16, k36, k37, kernel_src):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(tmp), doraise=True)
        out = Src(tmp)    # Src raises on a duplicate top-level name
    finally:
        tmp.unlink()
    rep = []
    check(text.isascii(), "non-ASCII character in the output")
    lines = text.splitlines()
    check(lines[0] == "# okc: problem=muse_decode_layer", "header line 1")
    check(lines[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1 and "k3-v16-1" not in text, "nonce")
    check(ast.get_docstring(out.tree).startswith("K3 v17 = k3_v16 + K2A v3.7"), "docstring")
    check(text.count(G13.K1_SELF_LINE) == 1, "K1 self-test reference line (v7, E4M3 Q at L = 8)")
    check(text.count("stock_ok = (tot <= 1.25 * tot_r1 + 5e-3 and row <= 1.25 * row_r1 + 2e-2) if tot_r1 >= 0 else (tot < 5e-2 and row < 1e-1)") == 2,
          "FA4 self-test reference lines (stock-relative rule) must stay verbatim in both self-tests")
    # K2A sources: v3.7's KERNEL_SRC_V3 byte for byte, the CPP source unchanged
    ksrc = out.value("K2A33_KERNEL_SRC")
    check(ksrc == k37.value("KERNEL_SRC_V3") and out.seg("K2A33_KERNEL_SRC") == piece(k37, "KERNEL_SRC_V3").rstrip("\n") + "\n"
          and '"""' not in ksrc, "K2A33_KERNEL_SRC is not k2a_v37's KERNEL_SRC_V3 byte for byte")
    check(out.value("K2A33_CPP_SRC") == k37.value("CPP_SRC_V3") == k36.value("CPP_SRC_V3") and dump(out.nodes["K2A33_CPP_SRC"]) == dump(b16.nodes["K2A33_CPP_SRC"]),
          "K2A33_CPP_SRC (run3's signature) must stay k3_v16's")
    old_k = b16.value("K2A33_KERNEL_SRC")
    check(old_k == k36.value("KERNEL_SRC_V3"), "k3_v16's K2A33_KERNEL_SRC is not k2a_v36's")
    removed, added = G14.diff_report(old_k, ksrc)
    check(tuple(removed) == KERNEL_REMOVED, "kernel source: removed / replaced v3.6 lines")
    import difflib
    ops = tuple((t, i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_k.splitlines(), ksrc.splitlines(), autojunk=False).get_opcodes()
                if t != "equal")
    check(ops == KERNEL_OPS, f"kernel diff shape {ops}")
    # the six replaced lines are neutral for every v3.6 instantiation: the new template parameters default to the old behaviour, and are named only
    # on those lines (plus the HK 3 block, which passes them)
    ksl = [ln.strip() for ln in ksrc.splitlines()]
    mb = tuple(ln for ln in ksl if re.search(r"\bMB\b", ln))
    lst = tuple(ln for ln in ksl if re.search(r"\bLIST\b", ln))
    check(mb == MB_LINES and lst == LIST_LINES, f"uses of the new template parameters: MB {mb} LIST {lst}")
    old_calls = [ln.strip() for ln in old_k.splitlines() if re.search(r"\b(cold_item|hc_item|route|flush_item)<", ln)]
    new_calls = [ln.strip() for ln in ksrc.splitlines() if re.search(r"\b(cold_item|hc_item|route|flush_item)<", ln)]
    extra = [ln for ln in new_calls if ln not in old_calls]
    check(all(ln in new_calls for ln in old_calls if ln not in KERNEL_REMOVED), "every v3.6 call site of cold_item / hc_item / route survives verbatim")
    check(len(extra) == 8 and sum(", true>(P, ring, full_a, empty_a" in ln for ln in extra) == 5 and sum("route<1, 8 * NCH, true, false>" in ln for ln in extra) == 1
          and sum("MB ?" in ln for ln in extra) == 2, f"new call sites outside v3.6's: {extra}")
    start37 = ksrc.index("// v3.7 (HK3 variants): v3.6's HK2 consumer")
    check(all(ksrc.index(ln) > start37 for ln in extra if ", true>(P, ring" in ln or "true, false>" in ln), "HK 3-only call sites after the v3.7 banner")
    check("#define" not in ksrc[start37:] and "#undef" not in ksrc[start37:] and "#pragma once" not in ksrc[start37:], "the v3.7 blocks define / undefine no macro")
    rep.append(f"K2A33_KERNEL_SRC == k2a_v37 KERNEL_SRC_V3 (sha1 {sha(ksrc)[:12]}, {len(ksrc.splitlines())} lines) vs k3_v16's = k2a_v36's (sha1 {sha(old_k)[:12]}, "
               f"{len(old_k.splitlines())} lines): +{added} lines in {len(ops)} hunks, {len(removed)} v3.6 lines replaced = the six template-parameter edits (route LIST = true, "
               f"cold_item / flush MB = false defaults restore the old expression; `MB` / `LIST` occur only on those lines), every v3.6 call site of cold_item / hc_item / route / "
               f"flush_item verbatim, the +462 lines are the HK 3 block (gemm37, k1/k2_kernel37, launch37, attrs37, smem_bytes37) and one `if (!LIST) return;`; no macro")
    # variants: v3.6's four + (1, 3, 2, 3, 1), (1, 3, 2, 3, 4)
    variants = out.value("K2A33_VARIANTS")
    check(variants == k37.value("V3_VARIANTS") and variants[:4] == b16.value("K2A33_VARIANTS") and variants[4:] == [(1, 3, 2, 3, 1), (1, 3, 2, 3, 4)],
          "K2A33_VARIANTS must be k3_v16's four + (1, 3, 2, 3, 1), (1, 3, 2, 3, 4)")
    # swapped / new host nodes ast-identical to k2a_v37 after the renames; the kept v3.3 pieces too
    for nm in SWAPPED:
        if nm == "KERNEL_SRC_V3":
            continue
        want = G11.K2A33_RENAME[nm]
        check(dump(ast.parse(r37(k37.seg(nm))).body[0]) == dump(out.nodes[want]), f"{want} differs from k2a_v37 {nm}")
    for nm in SWAPPED_SLICES + NEW_SLICES:
        want = R37_MAP[nm]
        check(dump(ast.parse(r37(k37.seg(nm))).body[0]) == dump(out.nodes[want]), f"{want} differs from k2a_v37 {nm}")
    kept = [nm for nm in G11.K2A33_NAMES if nm not in SWAPPED and nm not in G14.KEPT_DIFFERENT and nm != "CPP_SRC_V3"]
    for nm in kept:
        want = G11.K2A33_RENAME[nm]
        check(dump(ast.parse(r37(k37.seg(nm))).body[0]) == dump(out.nodes[want]), f"{want} differs from k2a_v37 {nm}")
    for nm in ("V36_HOT", "V3_SELF_TEST_CASES", "synth_inputs", "v36_call"):
        want = R37_MAP.get(nm, nm)
        check(dump(ast.parse(r37(k37.seg(nm))).body[0]) == dump(out.nodes[want]) , f"{want} differs from k2a_v37 {nm}")
    check(out.value("K2A36_SELF_TEST_CASES") == k37.value("V3_SELF_TEST_CASES") and out.value("K2A33_SELF_TEST_CASES") == b16.value("K2A33_SELF_TEST_CASES"),
          "self-test case tables")
    bucket128 = [c for c in out.value("K2A36_SELF_TEST_CASES") if 49 <= c[0] <= 192]
    check([c[0] for c in bucket128] == [100, 128], f"bucket 128 self-test cases {bucket128}")
    # a changed-node diff for the two small patches
    check(ast.unparse(b16.nodes["_v10_build_all"]).replace(OLD_LOG, NEW_LOG) == ast.unparse(out.nodes["_v10_build_all"]), "_v10_build_all: only the log line")
    check(ast.unparse(b16.nodes["_k2a33_prepare_st"]).replace("name='k2a_v36'", "name='k2a_v37'") == ast.unparse(out.nodes["_k2a33_prepare_st"]),
          "_k2a33_prepare_st: only the default build name")
    p16, p17 = b16.fn("_v11_k2a33_prepare"), out.fn("_v11_k2a33_prepare")
    ci = [i for i, s in enumerate(p17.body) if ast.unparse(s) == "_v17_k2a37_prepare(st, w, dev)"]
    check(len(ci) == 1 and ast.unparse(p17.body[ci[0] - 1]) == "_v14_k2a36_prepare(st, w, dev)"
          and [dump(s) for s in p17.body[:ci[0]] + p17.body[ci[0] + 1:]] == [dump(s) for s in p16.body], "_v11_k2a33_prepare: one inserted call after _v14_k2a36_prepare")
    c16, c17 = b16.fn("_v11_k2a33_call"), out.fn("_v11_k2a33_call")
    ri = [i for i, s in enumerate(c17.body) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "r37"]
    check(len(ri) == 1 and isinstance(c17.body[ri[0] + 1], ast.If) and ast.unparse(c17.body[ri[0] + 1].test) == "r37 is not None"
          and [dump(s) for s in c17.body[1:ri[0]] + c17.body[ri[0] + 2:]] == [dump(s) for s in c16.body[1:]] and ast.get_docstring(c17).startswith(ast.get_docstring(c16)[:200]),
          "_v11_k2a33_call: one inserted `r37` block in front of the v3.6 route")
    call_src = "\n".join(ast.unparse(s) for s in c17.body[1:])   # without the docstring
    check(call_src.index("r37") < call_src.index("r36") and "K2A37_ENABLE" in call_src and "_k2a37_call(st, data, r37[0], r37[1], r37[2])" in call_src
          and "return _k2a33_call(st, data, K2A33_PLAN[rep], hot_t=K2A33_HOT_T_BUCKET.get(rep))" in call_src, "call order r37 -> r36 -> v3.3 plan")
    # plan <-> the author's tables and the bench
    plan37 = bench_check(k37)
    check(out.value("K2A37_PLAN") == K2A37_PLAN and plan37[128] == (5, (8, 0, 12), (8, 0, 3)) == K2A37_PLAN[128], "K2A37_PLAN[128] != k2a_v37.V37_PLAN[128]")
    vid, hot, dyn = K2A37_PLAN[128]
    check(variants[vid] == (1, 3, 2, 3, 4) and out.value("K2A37_ENABLE") is True and out.value("K2A37_DQ") == 120, "K2A37 constants")
    dyn_fn = G14.pure_fn(k37.nodes["dyn_code"])
    check((16 | dyn_fn(dyn)) == (16 | 8 << 8 | 3 << 20) == 3147792, "probe = 16 | 8 << 8 | 3 << 20")
    hot_fn_src = out.fn("_k2a36_hot_code")
    check(ast.unparse(hot_fn_src).count("hk in (2, 3)") == 1, "hot_code: HK 3 shares HK 2's role-split code")
    # every other bucket keeps k3_v16's route: the author's ship table == the layer's K2A33_PLAN / K2A36_PLAN, K2A37_PLAN has no other bucket
    ship = k37.value("V36_SHIP")
    p36, p33 = out.value("K2A36_PLAN"), out.value("K2A33_PLAN")
    check(set(K2A37_PLAN) == {128} and out.value("K2A36_PLAN") == b16.value("K2A36_PLAN") and p33 == b16.value("K2A33_PLAN")
          and out.value("K2A36_BUCKETS") == {16: True, 32: True}, "plan tables must stay k3_v16's, K2A37_PLAN only bucket 128")
    check(all(ship[b][0] == p33[b] and ship[b][1:] == (None, None) for b in (1, 8, 256)) and ship[128] == (2, None, None) and p33[128] == 2
          and all((ship[b][0], ship[b][1]) == p36[b] and ship[b][2] is None for b in (16, 32)),
          "k2a_v37.V36_SHIP (the buckets v3.7 leaves alone) != the layer's K2A33_PLAN / K2A36_PLAN")
    rep.append(f"routes: bucket 128 -> vid {vid} hot {hot} dyn {dyn} (probe 16 | 8 << 8 | 3 << 20 = {16 | dyn_fn(dyn)}) == k2a_v37.V37_PLAN == k37_bench row v37 (layer -13.9 us); "
               f"bucket 1 / 8 -> vid {p33[1]} / {p33[8]}, 16 / 32 -> vid 3 {p36[16][1]} / {p36[32][1]}, 256 -> vid {p33[256]}: k3_v16's tables unchanged and == k2a_v37.V36_SHIP; "
               f"fallback at 128 = vid {p33[128]} (v3.3's own self-test)")
    # instantiations of the combined build
    cases, attrs = G14.pure_fn(k37.nodes["variant_cases"])(variants)
    gc, ga = G14.pure_fn(k37.nodes["_gate_cases"])(), G14.pure_fn(k37.nodes["_gate_attrs"])()
    check((cases, attrs) == (gc, ga), "variant_cases(K2A33_VARIANTS) != the gate cases (sass_gate's @CASES3@ / @ATTRS3@)")
    check((cases, attrs) == G14.pure_fn(out.nodes["_k2a36_variant_cases"])(variants) == (G14.pure_fn(out.nodes["_k2a36_gate_cases"])(), G14.pure_fn(out.nodes["_k2a36_gate_attrs"])()),
          "the layer's variant / gate case functions disagree")
    inst = re.findall(r"launch(3|36|37)<([0-9, ]+)>", cases)
    old_cases, _ = G14.pure_fn(b16.nodes["_k2a36_variant_cases"])(b16.value("K2A33_VARIANTS"))
    old_inst = re.findall(r"launch(3|36|37)<([0-9, ]+)>", old_cases)
    check(inst == [("3", "1, 3, 2, 0"), ("3", "2, 3, 2, 0"), ("3", "2, 2, 2, 1"), ("36", "4, 3"), ("37", "1, 3"), ("37", "4, 3")] and inst[:4] == old_inst,
          f"instantiations: {inst}")
    rep.append("combined-build instantiations: " + ", ".join(f"launch{k}<{a}>" for k, a in inst) + " (k3_v16: the first four); new: launch37<1, 3> (vid 4, unrouted: kept "
               "so vids match k2a_v37 / k37_bench) and launch37<4, 3> (vid 5) = k1_kernel37 + k2_kernel37 each (+4 template kernels; k3_kernel shared; route<..., LIST = false> "
               "once more); standalone gate (box_out/k37_gate4.log): K1 / K2 regs 78 / 75 (vid 4), 89 / 95 (vid 5), 0 spills; occupancy 2 CTAs/SM, grid 264; prepare increase est. "
               "+2..8 s (v3.6's vid 3 cost +1.6..2.1 s of the 52-54 s combined build; k37_check: 4 -> 6 variants +0.5 s standalone)")
    # R9: bounded waits (v3.6's checks on the new text) -- every mbar_wait passes P.err; the new mbar_wait_ok is bounded the same way
    check(ksrc.count("mbar_wait(") == len(re.findall(r"mbar_wait\([^;]*P\.err\);", ksrc)) + 1 and ksrc.count("void mbar_wait(") == 1, "every mbar_wait call passes P.err")
    check("__trap" not in ksrc and "trap;" not in ksrc, "no trap in the K2A source")
    mwo = ksrc[ksrc.index("bool mbar_wait_ok("):]
    mwo = mwo[:mwo.index("\n}\n")]
    check("WAIT_BUDGET" in mwo and "atomicExch(err, 1)" in mwo and "return false" in mwo and "__trap" not in mwo, "mbar_wait_ok must be bounded (clock64 budget, err flag, return)")
    check(len(re.findall(r"mbar_try\(", ksrc)) == 5 and ksrc.count("mbar_wait_ok(") == 3, "mbar_try only inside the two bounded waits")
    for macro, val in (("K36_LAUNDER", "1"), ("K36_KB_ROLL", "0"), ("K36_HALF_FENCE", "0")):
        check(f"#define {macro} {val}" in ksrc, f"{macro} default (the gated configuration)")
    rep.append(f"R9: {ksrc.count('mbar_wait(') - 1} mbar_wait call sites, all pass P.err; mbar_wait_ok (3 uses) is bounded the same way (clock64 budget, err flag, return); no trap; "
               "K36 macros at the gated defaults")
    dq37, mh, n_ldg = pdl_audit_k2a(ksrc)
    rep.append(f"PDL audit (source): launch37 K1 = `<<<>>>` normal launch, K2 / K3 cudaLaunchKernelEx with programmatic stream serialization; {n_ldg} __ldg lines, none on inter / h / y / ss / "
               f"cnt / cnt2 / meta / xw / out (no ld.global.nc, no __restrict__); K2's only read of K1's output (inter) is a bulk copy under `if (waited)` or deferred into take_wait, "
               f"whose first statement is pdl_wait(); a producer that never waited waits before returning; ticket counters meta[{dq37}..{dq37 + 3}] (K1 pair +0/+1, K2 pair +2/+3) below MH = {mh}, "
               f"the meta buffer is referenced in {len(META_LINES)} places only (MH rows by route / K3, DQ37 by gemm37)")
    # meta[120..123]: allocated zero, nothing else writes: the layer's K2A states
    check(out.value("K2A_META_HDR") == 128 and "meta=torch.zeros((K2A_META_HDR + 8 * K2A33_M3,), dtype=torch.int32, device=dev)," in ast.get_source_segment(text, out.nodes["_k2a33_prepare_st"]),
          "K2A state meta must be zeros((K2A_META_HDR + 8 * K2A33_M3,)), K2A_META_HDR = 128")
    uses = re.findall(r'[^\n]*\["meta"\][^\n]*', text)
    args = [u for u in uses if re.search(r'st\["meta"\],', u)]
    other = [u.strip() for u in uses if u not in args]
    check(args and other == ['dq = st["meta"][K2A37_DQ:K2A37_DQ + 4].tolist()'] and not re.search(r'\["meta"\](\.(zero_|fill_|copy_|add_|mul_)|\[[^\]]*\]\s*=[^=])', text),
          'st["meta"] only as a run3 argument, plus the self-test read of the four counters; never written on the host')
    rep.append(f'meta[120..123]: allocated by torch.zeros((K2A_META_HDR + 8 * K2A33_M3,)) with K2A_META_HDR = 128; the host passes st["meta"] to the kernels in {len(args)} argument lists and '
               "never writes it (the self-test reads the four counters); in the kernel only gemm37 touches words 120..123")
    # nothing else changed
    changed = [n for n, node in b16.nodes.items() if n not in out.nodes or dump(out.nodes[n]) != dump(node)]
    check(set(changed) == CHANGED_NODES, f"changed nodes vs k3_v16: {sorted(changed)} (want {sorted(CHANGED_NODES)})")
    added_nodes = set(out.nodes) - set(b16.nodes)
    check(added_nodes == V17_NODES, f"added nodes: unexpected {sorted(added_nodes - V17_NODES)}, missing {sorted(V17_NODES - added_nodes)}")
    for nm in ("K2A36_PLAN", "K2A36_BUCKETS", "K2A33_PLAN", "K2A33_HOT_T_BUCKET", "K1_ROUTE_MIN_BATCH", "P1_PROJ", "P1_ENABLE", "kernel_fn", "prepare", "_v8_experts", "_k2a33_call",
               "_k2a36_call", "_v14_k2a36_prepare", "_v14_self_test_inputs", "_k2a36_synth_inputs", "K1FA4V5_KERNEL_SRC", "_v16_fa4v5_core", "_v16_prepare", "_r1_attention", "K1FA4_KERNEL_SRC"):
        check(dump(out.nodes[nm]) == dump(b16.nodes[nm]), f"{nm} must stay k3_v16's")
    rep.append(f"vs k3_v16 (gen output == file on disk): {len(b16.nodes) - len(changed)} top-level nodes ast-identical; changed {sorted(changed)} (kernel text / variants / prepare_st / "
               f"case + hot_code functions = k2a_v37's; _v10_build_all: log line; _v11_k2a33_prepare: one call; _v11_k2a33_call: one block); added {sorted(added_nodes)}; the FA4 v5 / v2 "
               f"routes, P1, K1 and every plan table (K2A33_PLAN, K2A36_PLAN, K2A36_BUCKETS, K2A33_HOT_T_BUCKET), kernel_fn and prepare identical")
    # hygiene
    for node in ast.walk(out.tree):
        if isinstance(node, ast.Name):
            check(node.id not in G.BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
    for fn in CALL_TIME_FNS:
        for sub in ast.walk(out.nodes[fn]):
            check(not (isinstance(sub, ast.Attribute) and sub.attr in G.SYNC_ATTRS), f"{fn} has a host sync")
    bad = G8.G7.undefined_globals(text, str(OUT))
    check(not bad, f"undefined globals: {sorted(bad)}")
    code_only = "\n".join(l_ for l_ in lines if not l_.lstrip().startswith("#"))
    check(re.search(r"\benviron\b", code_only) is None and re.search(r"\bos\.environ", code_only) is None, "`environ` appears in the layer file outside comments")
    nbytes = len(text.encode())
    check(nbytes < 1_000_000, "file >= 1 MB")
    rep.append(f"static ok (no banned names, no host sync in {len(CALL_TIME_FNS)} call-time functions, no undefined globals, no `environ` outside comments); "
               f"{nbytes} bytes (k3_v16: {len(base16.encode())}; limit 1,000,000)")
    for r in check_ab_json(out, AB_JSON):
        rep.append(f"{AB_JSON.name} arm  {r}")
    return rep


def main(argv):
    text, base16, b16, k36, k37, kernel_src = generate()
    for r in verify(text, base16, b16, k36, k37, kernel_src):
        print(f"[gen_k3_v17] OK  {r}")
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text(encoding="utf-8") == text, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_v17] --check: up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"[gen_k3_v17] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}, {len(text.splitlines())} lines, {len(text.encode())} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

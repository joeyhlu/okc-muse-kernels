"""gen_k3_v14.py: k3_v14.py = k3_v13.py (as gen_k3_v13.py generates it) + K2A v3.6 in place of K2A v3.3, role-split routes at buckets 16 / 32.

    python3 -I tools/gen_k3_v14.py            # writes kernels/muse_decode_layer/k3_v14.py, asserts the invariants
    python3 -I tools/gen_k3_v14.py --check    # compares with the file on disk, writes nothing

Changes vs k3_v13 (everything else asserted ast-identical to gen_k3_v13's output, which is asserted == k3_v13.py on disk):
  * K2A v3.6 (kernels/muse_moe_w4a16/k2a_v36.py) replaces the K2A v3.3 module inside the combined build. Its CUDA source is
    v3.3's plus additions (asserted: five v3.3 lines change, nothing else is removed), so vids 0-2 are v3.3's code and ONE
    module serves every bucket. K2A33_KERNEL_SRC is k2a_v36's KERNEL_SRC_V3 byte for byte; K2A33_VARIANTS, _k2a33_build_ext
    and _k2a33_prepare_st are k2a_v36's nodes after the K2A33_ / _k2a33_ renames (the CPP source, packers, call, reset,
    hint, plan tables and the v3.3 self-test cases are v3.3's, unchanged). The only new instantiation is vid 3 =
    (NC 1, NST 3, 2 CTAs/SM, HK 2, NCH 4) = launch36<4, 3> (k1_kernel36<4, 3> + k2_kernel36<4, 3>).
  * New nodes (V14 section): K2A36_ENABLE / K2A36_BUCKETS / K2A36_PLAN, k2a_v36's variant_cases, gate cases, hot_code,
    v36_call, synth_inputs (hot8) and self-test cases (renamed _k2a36_* / K2A36_*), and the glue _v14_self_test_inputs /
    _v14_k2a36_prepare.
  * _v11_k2a33_call: v3.6's role-split variant where K2A36_PLAN routes the bucket, its prepare self-test passed and
    K2A36_ENABLE / K2A36_BUCKETS allow it (read at call time); otherwise exactly k3_v13's call.
  * _v10_build_all takes the variant cases from _k2a36_variant_cases (launch3 for HK 0 / 1, launch36 for HK 2).
"""

import ast
import difflib
import importlib.util
import json
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "kernels/muse_decode_layer/k3_v14.py"
V13_FILE = ROOT / "kernels/muse_decode_layer/k3_v13.py"
SRC_K2A33 = ROOT / "kernels/muse_moe_w4a16/k2a_v33.py"
SRC_K2A36 = ROOT / "kernels/muse_moe_w4a16/k2a_v36.py"
SRC_BENCH = ROOT / "tools/box/k36_bench.py"
AB_JSON = ROOT / "tools/box/ab_v14a.json"
NONCE = "k3-v14-1"

_s13 = importlib.util.spec_from_file_location("gen_k3_v13", ROOT / "tools/gen_k3_v13.py")
G13 = importlib.util.module_from_spec(_s13)
_s13.loader.exec_module(G13)
G12, G11, G10, G9, G8, G = G13.G12, G13.G11, G13.G10, G13.G9, G13.G8, G13.G
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha

# bucket rep -> (vid, (hot_t, hot_mode, hot_w4)): k36_bench rows v36_r8 (bucket 16) and v36_r12 (bucket 32); hot_mode 0 = role split
K2A36_PLAN = {16: (3, (8, 0, 8)), 32: (3, (8, 0, 12))}
K2A36_BUCKETS = {16: True, 32: True}
BENCH_ROW = {16: "v36_r8", 32: "v36_r12"}
V14_NODES = {"K2A36_ENABLE", "K2A36_BUCKETS", "K2A36_PLAN", "K2A36_HOT_DEFAULT", "K2A36_SELF_TEST_CASES", "_k2a36_synth_inputs",
             "_k2a36_variant_cases", "_k2a36_gate_cases", "_k2a36_gate_attrs", "_k2a36_hot_code", "_k2a36_call",
             "_v14_self_test_inputs", "_v14_k2a36_prepare"}
CHANGED_NODES = {"K2A33_VARIANTS", "K2A33_KERNEL_SRC", "_k2a33_build_ext", "_k2a33_prepare_st", "_v10_build_all",
                 "_v11_k2a33_prepare", "_v11_k2a33_call"}
# k2a_v33 node -> v13 node is G11.K2A33_RENAME; the v3.3 nodes k2a_v36 changes (the rest of K2A33_NAMES is ast-identical, asserted)
SWAPPED = ("V3_VARIANTS", "KERNEL_SRC_V3", "build_ext_v3", "v3_prepare")
# v3.3 nodes that stay k3_v13's although k2a_v36's differs: the v3.3 plan self-test cases (k2a_v36's list is K2A36_SELF_TEST_CASES)
# and the plan table (k3_v13's lever E: bucket 32 on variant 1)
KEPT_DIFFERENT = ("V3_SELF_TEST_CASES", "V3_PLAN")
# k2a_v36 nodes copied verbatim (renamed) into the V14 section, in this order
NEW_SLICES = ("V36_HOT", "V3_SELF_TEST_CASES", "synth_inputs", "variant_cases", "_gate_cases", "_gate_attrs", "hot_code", "v36_call")
R36_EXTRA = {
    "V3_SELF_TEST_CASES": "K2A36_SELF_TEST_CASES",   # overrides the v3.3 target: the v3.3 list stays K2A33_SELF_TEST_CASES
    "synth_inputs": "_k2a36_synth_inputs", "variant_cases": "_k2a36_variant_cases", "_gate_cases": "_k2a36_gate_cases",
    "_gate_attrs": "_k2a36_gate_attrs", "hot_code": "_k2a36_hot_code", "v36_call": "_k2a36_call", "V36_HOT": "K2A36_HOT_DEFAULT",
}
R36_MAP = {**G11.K2A33_RENAME, **R36_EXTRA}
_R36_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(sorted(map(re.escape, R36_MAP), key=len, reverse=True)) + r")(?![A-Za-z0-9_])")
# the only lines of k2a_v33's KERNEL_SRC_V3 that k2a_v36's changes (stripped), in file order
KERNEL_REMOVED = (
    "int hot_w4;                  // v3.3: warm-work scale in the CTA role split's byte model, x8",
    "template <int NC>",
    "const bool hot = P.hot_t > 0 && c > P.hot_t;   // v3.3: warm expert -> 64-token wgmma passes",
    "const int np = hot ? (c + 63) / 64 : (c + 8 * NC - 1) / (8 * NC);",
    "p.hot_w4 = (hot_t >= 1000) ? (int)(hot_t / 1000) : 8;   // x8",
)
CALL_TIME_FNS = G13.CALL_TIME_FNS + ("_v8_experts", "_v11_k2a33_call", "_k2a33_call", "_k2a36_call", "_k2a36_hot_code")


def r36(text):
    return _R36_RE.sub(lambda m: R36_MAP[m.group(1)], text)


V14_DOC = '''K3 v14 = k3_v13 + K2A v3.6 (role-split variant) at the decode 16 / decode 32 buckets (everything else is k3_v13, asserted
ast-identical by tools/gen_k3_v14.py except the listed nodes):

  K2A v3.6 (kernels/muse_moe_w4a16/k2a_v36.py): its CUDA source is K2A v3.3's plus additions (only five v3.3 lines change,
       asserted), so vids 0-2 are v3.3's code and the one combined build serves every bucket: K2A33_KERNEL_SRC,
       K2A33_VARIANTS, _k2a33_build_ext, _k2a33_prepare_st are k2a_v36's. The only new instantiation is vid 3 (NC 1, NST 3,
       2 CTAs/SM, HK 2, NCH 4): hot experts (more than 8 tokens) are read once, in passes of up to 32 tokens.
  Routes (K2A36_PLAN, from tools/box/k36_bench.py on the H200, layer routing decides): T = 16 -> vid 3 with the role split
       (8, 0, 8) (bench row v36_r8, -2.8 us); T = 32 -> vid 3 with (8, 0, 12) (v36_r12, -7.3 us against NC1, -3.7 us against
       k3_v13's NC2 plan). T <= 8 and T >= 128 are k3_v13's (v3.3 NC1; the wgmma warm path at 128 / 256).
  K2A36_ENABLE and K2A36_BUCKETS (per bucket) are read at call time; a v3.6 route is used only after its prepare
       self-test passed (k2a_v36's cases of the bucket incl. the 4-chunk hot8 cases, two calls each); any failure or host
       exception keeps v3.3's plan (K2A33_PLAN) for that bucket.

'''

V14_GLUE_HEAD = '''
# ---------------------------------------------------------------- V14: K2A v3.6 role-split routes
K2A36_ENABLE = True   # master switch (read at call time; False at prepare also skips the v3.6 self-tests)
K2A36_BUCKETS = ''' + repr(K2A36_BUCKETS) + '''   # per bucket rep (read at call time)
K2A36_PLAN = ''' + repr(K2A36_PLAN) + '''   # bucket rep -> (vid, (hot_t, hot_mode, hot_w4)); tools/box/k36_bench.py rows v36_r8 / v36_r12 (hot_mode 0 = role split, hot_w4 = sixteenths of a stage per extra chunk of a hot pass)

# kernels/muse_moe_w4a16/k2a_v36.py, verbatim after the K2A33_ / _k2a33_ / _k2a36_ renames (hot_code's default tuple is the
# dual-range plan that lost the bench: the layer always passes its own K2A36_PLAN entry).
'''

V14_GLUE = '''

def _v14_self_test_inputs(bs, seed, device, dist="normal"):
    """_k2a_self_test_inputs on k2a_v36's synthetic inputs (which add the "hot8" distribution): this GPU's experts are
    0..31, remote ids mapped to -1 when R1_LOCAL_ID_TABLE is on (the id convention the layer hands K2A)."""
    d = _k2a36_synth_inputs(bs, seed, device, dist)
    if R1_LOCAL_ID_TABLE:
        ids = d["topk_ids"]
        d["topk_ids"] = torch.where(ids < K2A_E_LOC, ids, torch.full_like(ids, -1))
    return d


def _v14_k2a36_prepare(st, w, dev):
    """K2A v3.6's routes (K2A36_PLAN) per bucket: k2a_v36's self-test cases of the bucket (normal / skewed / hot8 = every
    token on the same 8 experts, i.e. 4-chunk hot passes) through _k2a36_call on the layer's id convention, two calls per
    case (the second checks that h / the counters were restored), rel-L2 < 1e-2, worst row < 5e-2, no timeout flag. Sets
    st["r36"] = {bucket rep: (vid, hot)} for the buckets that pass; any other bucket (a failure, a variant not built or
    below its CTAs/SM target, K2A36_ENABLE off at prepare) keeps v3.3's plan. Prepare-time only."""
    st["r36"] = {}
    print(f"[v14] levers: K2A36_ENABLE={K2A36_ENABLE} K2A36_BUCKETS={K2A36_BUCKETS} K2A36_PLAN={K2A36_PLAN} "
          f"K2A33_PLAN={K2A33_PLAN} K2A33 variants={len(K2A33_VARIANTS)}", flush=True)
    if not K2A36_ENABLE:
        return
    t2 = time.time()
    for rep, (vid, hot) in sorted(K2A36_PLAN.items()):
        if vid >= len(st["variants"]) or st["grid"][vid] < st["sms"] * st["variants"][vid][2]:
            print(f"[k2a36] bucket {rep}: v{vid} not built or below its CTAs/SM target -> v3.3's plan", flush=True)
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
                    got = _k2a36_call(st, data, vid, hot)
                    torch.cuda.synchronize()
                    tot, row = _k2a_rel_err(got, ref)
                    errf = int(st["err"][0])
                    good = tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(got.float()).all())
                    print(f"[k2a36] layer self-test bs={bs} {dist} v{vid} hot={hot} call {k}: rel {tot:.3e} worst row "
                          f"{row:.3e} err={errf} -> {'ok' if good else 'FAIL'}", flush=True)
                    good_all &= good
            except Exception as ex:  # noqa: BLE001
                print(f"[k2a36] self-test bs={bs} v{vid} raised {ex!r}", flush=True)
                good_all = False
            _k2a33_reset(st)
            st["err"].zero_()
            if not good_all:
                break
        if good_all:
            st["r36"][rep] = (vid, tuple(hot))
        print(f"[k2a36] bucket {rep}: v{vid} hot {hot} -> {'route live' if good_all else 'FAILED: v3.3 plan'}", flush=True)
    print(f"[v14] K2A v3.6 routes live {st['r36']} (self-tests {time.time() - t2:.1f}s)", flush=True)
'''

OLD_CALL_DOC = '''    """K2A v3.3 on the layer: x [T, 4096] BF16 (pre-expert-normed), local ids (0..31, remote -1; lo = 0)."""
'''
NEW_CALL_DOC = '''    """K2A v3.3 on the layer: x [T, 4096] BF16 (pre-expert-normed), local ids (0..31, remote -1; lo = 0). v14: K2A v3.6's
    role-split variant where K2A36_PLAN routes the bucket, st["r36"] holds it (its prepare self-test passed) and K2A36_ENABLE /
    K2A36_BUCKETS allow it (read at call time); a host exception from that call drops the v3.6 routes for the run (once,
    with a line; the call is made on v3.3's plan, nothing was launched). Otherwise exactly k3_v13's call."""
'''
OLD_CALL_TAIL = '''    rep = _k2a_bucket_of(x.shape[0])
    return _k2a33_call(st, data, K2A33_PLAN[rep], hot_t=K2A33_HOT_T_BUCKET.get(rep))
'''
NEW_CALL_TAIL = '''    rep = _k2a_bucket_of(x.shape[0])
    r36 = st["r36"].get(rep) if (K2A36_ENABLE and K2A36_BUCKETS.get(rep) and st.get("r36")) else None
    if r36 is not None:
        try:
            return _k2a36_call(st, data, r36[0], r36[1])
        except Exception as ex:  # noqa: BLE001
            print(f"[k3] K2A v3.6 route raised ({ex!r}): v3.3's plan from here on", flush=True)
            st["r36"].clear()
    return _k2a33_call(st, data, K2A33_PLAN[rep], hot_t=K2A33_HOT_T_BUCKET.get(rep))
'''
OLD_BUILD_CASES = r'''    vv = [tuple(v) + (0,) * (4 - len(v)) for v in K2A33_VARIANTS]   # exactly as _k2a33_build_ext
    cases = "\n".join("    case %d: launch3<%d, %d, %d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, *v)
                      for i, v in enumerate(vv))
    attrs = "\n".join("    case %d: return attrs3<%d, %d, %d, %d>();" % (i, *v) for i, v in enumerate(vv))
'''
NEW_BUILD_CASES = '''    cases, attrs = _k2a36_variant_cases(K2A33_VARIANTS)   # exactly as _k2a33_build_ext: launch3 for HK 0 / 1, launch36 for HK 2
'''


def piece(src, nm, rename):
    """The text of one K2A33 node as gen_k3_v11 emits it (kernel / CPP sources keep k2a's text after the new name)."""
    if nm in ("KERNEL_SRC_V3", "CPP_SRC_V3"):
        return G11.K2A33_RENAME[nm] + src.seg(nm)[len(nm):] + "\n"
    return rename(src.seg(nm)) + "\n"


def diff_report(a_text, b_text):
    """(removed-or-replaced v3.3 lines stripped, inserted line count) of difflib's opcodes, v3.3 -> v3.6."""
    a, b = a_text.splitlines(), b_text.splitlines()
    removed, added = [], 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            removed += [ln.strip() for ln in a[i1:i2]]
        if tag in ("replace", "insert"):
            added += j2 - j1
    return removed, added


def generate():
    text, base12, p1 = G13.generate()
    base13 = text
    check(V13_FILE.read_text(encoding="utf-8") == base13, "gen_k3_v13's output differs from k3_v13.py on disk (regenerate v13 first)")
    k33, k36 = Src(SRC_K2A33), Src(SRC_K2A36)
    # k2a_v36 vs k2a_v33: which nodes changed
    for nm in G11.K2A33_NAMES:
        same = dump(k33.nodes[nm]) == dump(k36.nodes[nm])
        check(same != (nm in SWAPPED or nm == "V3_SELF_TEST_CASES"),
              f"k2a_v36 {nm}: expected {'changed' if nm in SWAPPED or nm == 'V3_SELF_TEST_CASES' else 'ast-identical'} vs k2a_v33")
    removed, _ = diff_report(k33.value("KERNEL_SRC_V3"), k36.value("KERNEL_SRC_V3"))
    check(tuple(removed) == KERNEL_REMOVED, f"KERNEL_SRC_V3 v3.3 -> v3.6 removed / replaced lines: {removed}")
    lines = text.split("\n")
    check(lines[1].startswith("# nonce k3-v13-1 "), "k3_v13 nonce")
    hdr = "\n".join(lines[2:5]) + "\n"
    check(hdr.startswith("# GENERATED by tools/gen_k3_v13.py") and hdr.endswith("regenerate.\n"), "k3_v13 generated-by header")
    text = patch(text, lines[1] + "\n" + hdr,
                 f"# nonce {NONCE} (k3_v13 + K2A v3.6 role-split variants at buckets 16 / 32, one combined module; k3_v14_plan.md)\n"
                 "# GENERATED by tools/gen_k3_v14.py on top of gen_k3_v13.py's output (k3_v13.py; its chain is gen_k3_v12 / v11 /\n"
                 "# v10 / v09 / v08 ...) and kernels/muse_moe_w4a16/k2a_v36.py (K2A v3.6).\n"
                 "# Do not hand-edit: change a canonical source or a generator and regenerate.\n", "nonce + header")
    text = patch(text, '"""K3 v13 = k3_v12 + P1 + two screened levers', '"""' + V14_DOC + 'K3 v13 = k3_v12 + P1 + two screened levers',
                 "docstring")
    # K2A v3.3 nodes -> k2a_v36's (same names; text = gen_k3_v11's slice of v3.3 -> the same slice of v3.6)
    for nm in SWAPPED:
        new = piece(k36, nm, r36)
        if nm == "V3_VARIANTS":
            new = new.rstrip("\n") + ("   # (NC, NST, CTAs/SM, HK[, NCH]): HK 0 cold, 1 v3.3 HOTCAP (wgmma warm role), 2 v3.6 "
                                      "(cold NC1 + hot passes of 8 * NCH tokens); vid 0-2 = v3.3's list\n\n")
        text = patch(text, piece(k33, nm, G11.r33), new, f"K2A v3.3 -> v3.6: {nm}")
    # combined build: the variant cases come from _k2a36_variant_cases
    text = patch(text, OLD_BUILD_CASES, NEW_BUILD_CASES, "combined build cases")
    text = patch(text, "combined build (K1 + K2A v3.3 + R1B + tail) in", "combined build (K1 + K2A v3.6 [v3.3 = vids 0-2] + R1B + tail) in",
                 "combined build log")
    # prepare: v3.6 routes self-tested after v3.3's own cases
    text = patch(text, '''        _k2a33_reset(st)
    print(f"[v8] K2A v3.3 buckets {ok} in {time.time() - t1:.1f}s; route {V11_K2A33_ROUTE}", flush=True)
''', '''        _k2a33_reset(st)
    _v14_k2a36_prepare(st, w, dev)
    print(f"[v8] K2A v3.3 buckets {ok} in {time.time() - t1:.1f}s; route {V11_K2A33_ROUTE}", flush=True)
''', "prepare v14")
    # call
    text = patch(text, OLD_CALL_DOC, NEW_CALL_DOC, "k2a33 call doc")
    text = patch(text, OLD_CALL_TAIL, NEW_CALL_TAIL, "k2a33 call")
    # the V14 section, before the V12 levers banner
    banner = "# ---------------------------------------------------------------- V12 levers\n"
    slices = "".join(r36(k36.seg(nm)) + "\n" for nm in NEW_SLICES)
    text = patch(text, banner, V14_GLUE_HEAD + slices + V14_GLUE + "\n\n" + banner, "v14 section")
    return text, base13, k33, k36


def check_ab_json(out, path):
    """ab_v14a.json: every switch exists, dict values keep the default's key set (a layer_ab2 variant replaces the whole value),
    and each arm resolves to the intended per-bucket K2A36 state; ref = K2A36 off = k3_v13's behaviour."""
    def keyfix(v):   # layer_ab2.keyfix
        if isinstance(v, dict) and any(isinstance(k, str) and "," in k for k in v):
            return {tuple(int(p) if p.isdigit() else p for p in k.split(",")): keyfix(x) for k, x in v.items()}
        if isinstance(v, dict):
            return {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): keyfix(x) for k, x in v.items()}
        return v

    arms = json.loads(path.read_text())
    default = {k: out.value(k) for k in ("K2A36_ENABLE", "K2A36_BUCKETS")}
    want = {"ref": (False, False), "k36_all": (True, True), "k36_16only": (True, False), "k36_32only": (False, True)}
    check([a["name"] for a in arms] == list(want), f"{path.name}: arms {[a['name'] for a in arms]} != {list(want)}")
    check(keyfix(arms[0]["set"]) == {"K2A36_ENABLE": False}, f"{path.name}: variant 0 must be ref = K2A36_ENABLE off only")
    rows = []
    for a in arms:
        eff = dict(default)
        for k, v in keyfix(a.get("set", {})).items():
            check(k in default, f"{path.name} {a['name']}: unknown switch {k}")
            if isinstance(default[k], dict):
                check(isinstance(v, dict) and set(v) == set(default[k]) and all(isinstance(x, bool) for x in v.values()),
                      f"{path.name} {a['name']}: {k} keys {sorted(map(str, v))} != default's {sorted(map(str, default[k]))}")
            else:
                check(isinstance(v, bool), f"{path.name} {a['name']}: {k} must be a bool")
            eff[k] = v
        live = tuple(bool(eff["K2A36_ENABLE"] and eff["K2A36_BUCKETS"].get(b)) for b in (16, 32))
        check(live == want[a["name"]], f"{path.name} {a['name']}: resolves to live buckets (16, 32) = {live}, want {want[a['name']]}")
        rows.append(f"{a['name']}: v3.6 at bucket 16 {'on' if live[0] else 'off'}, bucket 32 {'on' if live[1] else 'off'}"
                    + (" (= k3_v13 behaviour)" if not any(live) else ""))
    return rows


def bench_rows():
    """k36_bench.py's ROWS literal (name -> (module, vid, hot, mask, text)) without importing the file."""
    tree = ast.parse(SRC_BENCH.read_text(encoding="utf-8"))
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
            if nm == "PLAN_HOT":
                env[nm] = ast.literal_eval(node.value)
            elif nm == "ROWS":
                rows = {}
                for k, v in zip(node.value.keys, node.value.values):
                    rows[ast.literal_eval(k)] = tuple(env[e.id] if isinstance(e, ast.Name) else ast.literal_eval(e) for e in v.elts)
                return rows
    raise SystemExit("[gen_k3_v14] k36_bench.py: no ROWS")


def pure_fn(node):
    """Run a zero-global module function from its ast node (like sass_gate's call:FUNC)."""
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<gen>", "exec"), ns)   # noqa: S102 (repo code)
    return ns[node.name]


def verify(text, base13, k33, k36):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    btmp = OUT.with_suffix(".py.gen_base")
    btmp.write_text(base13, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(tmp), doraise=True)
        out, b13 = Src(tmp), Src(btmp)    # Src raises on a duplicate top-level name
    finally:
        tmp.unlink()
        btmp.unlink()
    rep = []
    check(text.isascii(), "non-ASCII character in the output")
    lines = text.splitlines()
    check(lines[0] == "# okc: problem=muse_decode_layer", "header line 1")
    check(lines[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1, "nonce")
    check(ast.get_docstring(out.tree).startswith("K3 v14 = k3_v13 + K2A v3.6"), "docstring")
    check(text.count(G13.K1_SELF_LINE) == 1, "K1 self-test reference line (v7, E4M3 Q at L = 8)")
    # K2A sources: v3.6's KERNEL_SRC_V3 byte for byte, the v3.3 CPP source unchanged, v3.3's list is vids 0-2
    ksrc = out.value("K2A33_KERNEL_SRC")
    check(ksrc == k36.value("KERNEL_SRC_V3") and out.seg("K2A33_KERNEL_SRC") == piece(k36, "KERNEL_SRC_V3", r36).rstrip("\n") + "\n"
          and '"""' not in ksrc, "K2A33_KERNEL_SRC is not k2a_v36's KERNEL_SRC_V3 byte for byte")
    check(out.value("K2A33_CPP_SRC") == k36.value("CPP_SRC_V3") == k33.value("CPP_SRC_V3")
          and dump(out.nodes["K2A33_CPP_SRC"]) == dump(b13.nodes["K2A33_CPP_SRC"]), "K2A33_CPP_SRC")
    removed, added = diff_report(k33.value("KERNEL_SRC_V3"), ksrc)
    check(tuple(removed) == KERNEL_REMOVED, "kernel source: removed / replaced v3.3 lines")
    rep.append(f"K2A33_KERNEL_SRC == k2a_v36 KERNEL_SRC_V3 (sha1 {sha(ksrc)[:12]}, {len(ksrc.splitlines())} lines); vs k2a_v33's "
               f"(sha1 {sha(k33.value('KERNEL_SRC_V3'))[:12]}, {len(k33.value('KERNEL_SRC_V3').splitlines())} lines): +{added} lines, "
               f"{len(removed)} v3.3 lines replaced (hot_w4 field + hot_mode, route template TPH / HGE, hot / np, run3's hot_t decode)")
    variants = out.value("K2A33_VARIANTS")
    check(variants == k36.value("V3_VARIANTS") and variants[:3] == k33.value("V3_VARIANTS") and variants[3] == (1, 3, 2, 2, 4),
          "K2A33_VARIANTS must be v3.3's three + vid 3 (1, 3, 2, 2, 4)")
    # every swapped / new node ast-identical to k2a_v36 after the renames
    for nm in SWAPPED:
        if nm == "KERNEL_SRC_V3":
            continue
        want = G11.K2A33_RENAME[nm]
        check(dump(ast.parse(r36(k36.seg(nm))).body[0]) == dump(out.nodes[want]), f"{want} differs from k2a_v36 {nm}")
    for nm in NEW_SLICES:
        want = R36_MAP.get(nm, nm)
        check(dump(ast.parse(r36(k36.seg(nm))).body[0]) == dump(out.nodes[want]), f"{want} differs from k2a_v36 {nm}")
    # the v3.3 pieces that stay: ast-identical to k2a_v36's too (same node), so the one module is v3.6's host code throughout
    kept = [nm for nm in G11.K2A33_NAMES if nm not in SWAPPED and nm not in KEPT_DIFFERENT and nm != "CPP_SRC_V3"]
    for nm in kept:
        want = G11.K2A33_RENAME[nm]
        check(dump(ast.parse(G11.r33(k36.seg(nm))).body[0]) == dump(out.nodes[want]), f"{want} differs from k2a_v36 {nm}")
    # self-test inputs: v3.6's synth_inputs == the layer's _k2a_synth_inputs plus the hot8 branch
    syn = ast.parse(r36(k36.seg("synth_inputs"))).body[0]
    chain = [n for n in syn.body if isinstance(n, ast.If)]
    check(len(chain) == 1 and isinstance(chain[0].orelse[0], ast.If) and ast.unparse(chain[0].orelse[0].test) == "dist == 'hot8'",
          "synth_inputs: hot8 branch")
    chain[0].orelse = chain[0].orelse[0].orelse
    syn.name = "_k2a_synth_inputs"
    check(dump(syn) == dump(out.nodes["_k2a_synth_inputs"]), "_k2a36_synth_inputs minus its hot8 branch != the layer's _k2a_synth_inputs")
    check(out.value("K2A36_SELF_TEST_CASES") == k36.value("V3_SELF_TEST_CASES")
          and out.value("K2A33_SELF_TEST_CASES") == b13.value("K2A33_SELF_TEST_CASES"), "self-test case tables")
    # routes <-> the measured bench rows
    rows = bench_rows()
    for b, nm in BENCH_ROW.items():
        mod, vid, hot, mask, _ = rows[nm]
        check(mod == "36" and mask == 7 and out.value("K2A36_PLAN")[b] == (vid, hot) == K2A36_PLAN[b]
              and variants[vid] == (1, 3, 2, 2, 4) and hot[1] == 0, f"K2A36_PLAN[{b}] != k36_bench row {nm}")
    check(set(out.value("K2A36_PLAN")) == set(out.value("K2A36_BUCKETS")) == {16, 32}, "K2A36 bucket keys")
    check(out.value("K2A36_ENABLE") is True and out.value("K2A36_BUCKETS") == {16: True, 32: True}, "K2A36 switches")
    rep.append(f"routes {out.value('K2A36_PLAN')} == k36_bench rows {BENCH_ROW}; buckets 1 / 8 / 128 / 256 keep k3_v13's K2A33_PLAN "
               f"{out.value('K2A33_PLAN')}")
    # instantiations of the combined build
    cases, attrs = pure_fn(k36.nodes["variant_cases"])(variants)
    gc, ga = pure_fn(k36.nodes["_gate_cases"])(), pure_fn(k36.nodes["_gate_attrs"])()
    check((cases, attrs) == (gc, ga), "variant_cases(K2A33_VARIANTS) != the gate cases (sass_gate's @CASES3@ / @ATTRS3@)")
    inst = re.findall(r"launch(3|36)<([0-9, ]+)>", cases)
    check(len(inst) == 4 and [i[0] for i in inst] == ["3", "3", "3", "36"] and inst[3][1] == "4, 3", f"instantiations: {inst}")
    old_inst = re.findall(r"launch3<([0-9, ]+)>", "\n".join("    case %d: launch3<%d, %d, %d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, *(tuple(v) + (0,) * (4 - len(v))))
                                                           for i, v in enumerate(b13.value("K2A33_VARIANTS"))))
    check([i[1] for i in inst[:3]] == old_inst, "vids 0-2 instantiations must be k3_v13's")
    rep.append("combined-build instantiations: " + ", ".join(f"launch{k}<{a}>" for k, a in inst) + " (k3_v13: the first three); new: "
               "launch36<4, 3> = k1_kernel36<4, 3> + k2_kernel36<4, 3> (+2 template kernels; k3_kernel shared); nothing else added")
    # R9: bounded waits (mbarrier waits all through mbar_wait with a clock budget and the err flag; no trap)
    check(ksrc.count("mbar_wait(") == len(re.findall(r"mbar_wait\([^;]*P\.err\);", ksrc)) + 1 and ksrc.count("void mbar_wait(") == 1,
          "every mbar_wait call passes P.err")
    check(len(re.findall(r"mbar_try\(", ksrc)) == 3 and "WAIT_BUDGET" in ksrc and "atomicExch(err, 1)" in ksrc, "mbar_try only inside the bounded mbar_wait")
    check("__trap" not in ksrc and "trap;" not in ksrc, "no trap in the K2A source")
    for macro, val in (("K36_LAUNDER", "1"), ("K36_KB_ROLL", "0"), ("K36_HALF_FENCE", "0")):
        check(f"#define {macro} {val}" in ksrc, f"{macro} default (the gated configuration)")
    rep.append(f"R9: {ksrc.count('mbar_wait(') - 1} mbar_wait call sites, all pass P.err (clock64 budget, err flag, return; no trap); "
               "K36 macros at the gated defaults (LAUNDER 1, KB_ROLL 0, HALF_FENCE 0)")
    # nothing else changed
    changed = [n for n, node in b13.nodes.items() if n not in out.nodes or dump(out.nodes[n]) != dump(node)]
    check(set(changed) == CHANGED_NODES, f"changed nodes vs k3_v13: {sorted(changed)} (want {sorted(CHANGED_NODES)})")
    added_nodes = set(out.nodes) - set(b13.nodes)
    check(added_nodes == V14_NODES, f"added nodes: unexpected {sorted(added_nodes - V14_NODES)}, missing {sorted(V14_NODES - added_nodes)}")
    for nm, val in (("K2A33_PLAN", None), ("K1_ROUTE_MIN_BATCH", None), ("P1_PROJ", None), ("P1_ENABLE", None), ("K2A33_HOT_T_BUCKET", None),
                    ("kernel_fn", None), ("prepare", None), ("_v8_experts", None)):
        check(dump(out.nodes[nm]) == dump(b13.nodes[nm]), f"{nm} must stay k3_v13's")
    rep.append(f"vs k3_v13 (gen output == file on disk): {len(b13.nodes) - len(changed)} top-level nodes ast-identical; changed "
               f"{sorted(changed)}; added {len(added_nodes)}; every P1 / lever setting (P1_*, K1_ROUTE_MIN_BATCH, K2A33_PLAN, kernel_fn, prepare) identical")
    # the v3.3 call survives verbatim as the fallback
    call = ast.unparse(out.nodes["_v11_k2a33_call"])
    check(text.count("return _k2a33_call(st, data, K2A33_PLAN[rep], hot_t=K2A33_HOT_T_BUCKET.get(rep))") == 1
          and text.count("_k2a36_call(st, data, r36[0], r36[1])") == 1 and "K2A36_ENABLE" in call and "K2A36_BUCKETS" in call,
          "v3.3 fallback call / v3.6 call sites")
    # hygiene
    for node in ast.walk(out.tree):
        if isinstance(node, ast.Name):
            check(node.id not in G.BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
    for fn in CALL_TIME_FNS:
        for sub in ast.walk(out.nodes[fn]):
            check(not (isinstance(sub, ast.Attribute) and sub.attr in G.SYNC_ATTRS), f"{fn} has a host sync")
    bad = G8.G7.undefined_globals(text, str(OUT))
    check(not bad, f"undefined globals: {sorted(bad)}")
    check(len(text.encode()) < 1_000_000, "file >= 1 MB")
    rep.append(f"static ok (no banned names, no host sync in {len(CALL_TIME_FNS)} call-time functions, no undefined globals); "
               f"{len(text.encode())} bytes (k3_v13: {len(base13.encode())})")
    for r in check_ab_json(out, AB_JSON):
        rep.append(f"{AB_JSON.name} arm  {r}")
    return rep


def main(argv):
    text, base13, k33, k36 = generate()
    for r in verify(text, base13, k33, k36):
        print(f"[gen_k3_v14] OK  {r}")
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text(encoding="utf-8") == text, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_v14] --check: up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"[gen_k3_v14] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}, {len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

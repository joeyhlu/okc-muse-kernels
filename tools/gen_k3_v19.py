"""gen_k3_v19.py: k3_v19.py = k3_v18.py (as gen_k3_v18.py generates it) + the three measured G2 glue riders.

    python -I tools/gen_k3_v19.py            # writes kernels/muse_decode_layer/k3_v19.py, asserts the invariants, prints the checker scan
    python -I tools/gen_k3_v19.py --check    # compares with the file on disk, writes nothing

The riders (kernels/muse_decode_layer/g2_design.md "Results"; box A/B vs the k3_v17 path, which has the same attention-adjacent glue):
  * G2-B  P1 GEMV launched with programmatic stream serialization (o and unpooler; call-time switch G2_P1_PDL, prepare self-test pdl 1 == pdl 0).
  * G2-A  `fused` at T = 1 ONLY: P1 pooler + FP32 router + routing / pre-expert-norm tail in ONE kernel (call-time switches G2_PRT / G2_PRT_MODE,
          prepare self-test vs the router32 + P1 pooler + tail path, fallback = k3_v18's path).
  * router place `side_first` at the T = 8 bucket (V8_ROUTER_PLACE[8], the existing call-time dict).
NOT included: G2-A at T = 8, G2-A `main` / `fusedr`, G2-C (FFN boundary kernel): their code is not in the file (kernels/muse_decode_layer/g2_glue_v19.py
is g2_glue.py without them; only prt_kernel<1, 1> is instantiated, so the G2 build is smaller than k3_g2's).

Changes vs k3_v18 (everything else asserted ast-identical to gen_k3_v18's output, which is asserted == k3_v18.py on disk; and, stronger, the output with
the patches reversed and the G2 section cut out is asserted byte-identical to k3_v18.py):
  * G2 section (before _k3_disable): every top-level node of g2_glue_v19.py verbatim.
  * kernel_fn: one `g2out = _g2_prt(...)` statement after the attention boundary; `if tail == "cat":` becomes the elif of `if g2out is not None:`.
  * prepare: `_g2_prepare(state, t0)` after `_v18_fa4_prepare(state, t0)`.
  * p1_gemv: `pdl=_g2_p1_pdl(proj)` passed to p1_call.
  * V8_ROUTER_PLACE: bucket 8 'side' -> 'side_first'.
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
OUT = ROOT / "kernels/muse_decode_layer/k3_v19.py"
V18_FILE = ROOT / "kernels/muse_decode_layer/k3_v18.py"
SRC_G2 = ROOT / "kernels/muse_decode_layer/g2_glue_v19.py"
SRC_G2_FULL = ROOT / "kernels/muse_decode_layer/g2_glue.py"
AB_JSON = ROOT / "tools/box/ab_v19a.json"
NONCE = "k3-v19-1"

_s18 = importlib.util.spec_from_file_location("gen_k3_v18", ROOT / "tools/gen_k3_v18.py")
G18 = importlib.util.module_from_spec(_s18)
_s18.loader.exec_module(G18)
G, G8 = G18.G, G18.G8
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha

CHANGED_NODES = {"kernel_fn", "prepare", "p1_gemv", "V8_ROUTER_PLACE"}
G2_CALL_TIME_FNS = ("_g2_prt", "_g2_prt_run", "_g2_p1_pdl", "p1_gemv")
CALL_TIME_FNS = tuple(f for f in G18.CALL_TIME_FNS if f != "p1_gemv") + G2_CALL_TIME_FNS
# nodes of g2_glue.py that g2_glue_v19.py drops (G2-C and its helpers), keeps untouched, or keeps with edits (the audit prints which)
G2_REMOVED = {"G2_FFN", "G2_FFN_T1", "_g2_ffn_run", "_g2_ffn", "_g2_selftest_ffn", "_g2_prepare_ffn"}
G2_EDITED = {"G2_PRT_MODE", "G2_SELF_TEST_CASES", "G2_CPP_SRC", "G2_CUDA_SRC", "_g2_build", "_g2_prt_run", "_g2_prt", "_g2_selftest_prt",
             "_g2_selftest_p1pdl", "_g2_prepare"}   # (G2_PRT and G2_PREPARE_BY_S differ in their comments only: ast-identical)

G2_DOC = '''K3 v19 = k3_v18 + the three G2 glue riders (kernels/muse_decode_layer/g2_design.md; everything else is k3_v18, asserted ast-identical by
tools/gen_k3_v19.py except kernel_fn / prepare / p1_gemv / V8_ROUTER_PLACE, and byte-identical once the riders' patches and section are taken out):

  G2-B (G2_P1_PDL, per projection, read at call time): P1 GEMV launches with programmatic stream serialization (o, unpooler on; qkv, pooler off).
       p1::stream issues only prepare-time weight loads before its griddepcontrol.wait (x, output and split-K workspace after it; audited on the
       source by the generator). Live after a bitwise pdl 1 == pdl 0 self-test.
  G2-A (G2_PRT + G2_PRT_MODE, read at call time) at T = 1 ONLY: the P1 pooler GEMV, the FP32 router (router32's math) and the routing / pre-expert
       norm / local-id tail (tail_kernel's math) in ONE launch (256 CTAs; router weights cp.async'd before the PDL wait; the last CTA by ticket runs
       the tail). Replaces the side-stream router, the stream fork / join, the P1 pooler launch and the tail. Live only after its prepare self-test vs
       the three-kernel path (ids / local ids equal, xn / weights within 1e-3 / 1e-6, ticket back at zero, repeat identical); else k3_v18's path.
       Not built at T = 8 (it lost there), no router-first / register-weight variants, no FFN-boundary kernel (G2-C lost).
  Router place 'side_first' at the T = 8 bucket (V8_ROUTER_PLACE[8]; the existing call-time dict): router32 captured before the P1 pooler.
  G2 is skipped (all three call-time paths = k3_v18's, except the router place) when prepare is already past G2_PREPARE_BY_S = 190 s.

'''

OLD_KFN = '''    ffn_input, stream, router_input = _v9_bnd("attn", data["stream"], attn_output,
                                              None if split_router else torch.float32, state, tokens)
    if tail == "cat":
'''
NEW_KFN = '''    ffn_input, stream, router_input = _v9_bnd("attn", data["stream"], attn_output,
                                              None if split_router else torch.float32, state, tokens)
    g2out = _g2_prt(ffn_input, router_input, state, data, tokens, mode, tail)   # G2-A: pooler + FP32 router + tail in one kernel (T = 1), else None
    if g2out is not None:
        expert_x, topk_weights, topk_ids, local_ids = g2out
    elif tail == "cat":
'''
OLD_PREP = "    _v18_fa4_prepare(state, t0)\n    torch.cuda.synchronize()\n"
NEW_PREP = "    _v18_fa4_prepare(state, t0)\n    _g2_prepare(state, t0)\n    torch.cuda.synchronize()\n"
OLD_PLACE = "V8_ROUTER_PLACE = {1: 'side', 8: 'side', 16: 'side', 32: 'side', 128: 'side', 256: 'main'}"
NEW_PLACE = "V8_ROUTER_PLACE = {1: 'side', 8: 'side_first', 16: 'side', 32: 'side', 128: 'side', 256: 'main'}"
PLACE_NOTE = "   # G2: bucket 8 side_first (router32 captured before the P1 pooler; box A/B g2_ab4: +0.5..1.0% at verify 1 sliding, +0.2..0.6% full)"
OLD_GEMV = '    p1_call(st["mod"], P, x, out, n_rows, ldo, geo, mode=mode)\n    return out\n'
NEW_GEMV = '    p1_call(st["mod"], P, x, out, n_rows, ldo, geo, pdl=_g2_p1_pdl(proj), mode=mode)   # G2-B: PDL launch where switched on\n    return out\n'
ANCHOR = "def _k3_disable(state, key, value, piece, ex):\n"
BANNER = "# ---------------------------------------------------------------- G2 (kernels/muse_decode_layer/g2_glue_v19.py, verbatim): riders G2-A (T = 1) and G2-B\n"
DOC_OLD = '"""K3 v18 = k3_v17 with the FA4 attention path replaced by K1-FA4 v6'


def g2_nodes(src):
    """(name, text) of every named top-level node of a g2_glue file, in file order (imports and the docstring are the layer's)."""
    out = []
    for node in src.tree.body:
        names = Src._names(node)
        if not names:
            check(isinstance(node, (ast.Import, ast.ImportFrom)) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)),
                  f"{src.path.name}: unexpected unnamed top-level node at line {node.lineno}")
            continue
        out.append((names[0], "".join(src.lines[node.lineno - 1:node.end_lineno])))
    return out


def generate():
    text, base17, v6, cases = G18.generate()
    check(V18_FILE.read_text(encoding="utf-8") == text, "gen_k3_v18's output differs from k3_v18.py on disk (regenerate v18 first)")
    base18 = text
    g2 = Src(SRC_G2)
    lines = text.split("\n")
    check(lines[1].startswith("# nonce k3-v18-1 "), "k3_v18 nonce")
    hdr = "\n".join(lines[2:4]) + "\n"
    check(hdr.startswith("# GENERATED by tools/gen_k3_v18.py") and hdr.endswith("regenerate.\n") and lines[4].startswith('"""K3 v18 = k3_v17'), "k3_v18 generated-by header")
    text = patch(text, lines[1] + "\n" + hdr,
                 f"# nonce {NONCE} (k3_v18 + the G2 riders: P1 PDL launches, pooler + FP32 router + tail in one kernel at T = 1, router side_first at T = 8; k3_v19_plan.md)\n"
                 "# GENERATED by tools/gen_k3_v19.py on top of gen_k3_v18.py's output (k3_v18.py; its chain is gen_k3_v17 / v16 / v15 / v14 ...) and\n"
                 "# kernels/muse_decode_layer/g2_glue_v19.py (G2). Do not hand-edit: change a canonical source or a generator and regenerate.\n", "nonce + header")
    text = patch(text, DOC_OLD, '"""' + G2_DOC + "K3 v18 = k3_v17 with the FA4 attention path replaced by K1-FA4 v6", "docstring")
    section = BANNER + "".join(seg.rstrip("\n") + "\n\n\n" for _, seg in g2_nodes(g2))
    text = patch(text, ANCHOR, section + ANCHOR, "G2 section")
    text = patch(text, OLD_KFN, NEW_KFN, "kernel_fn G2-A branch")
    text = patch(text, OLD_PREP, NEW_PREP, "prepare G2")
    text = patch(text, OLD_PLACE, NEW_PLACE + PLACE_NOTE, "router place")
    text = patch(text, OLD_GEMV, NEW_GEMV, "p1_gemv G2-B")
    return text, base18, g2


def unpatch(text, g2):
    """Undo generate()'s patches (each new text must occur exactly once) and cut the G2 section: must give k3_v18.py byte for byte."""
    section = BANNER + "".join(seg.rstrip("\n") + "\n\n\n" for _, seg in g2_nodes(g2))
    text = patch(text, NEW_GEMV, OLD_GEMV, "unpatch p1_gemv")
    text = patch(text, NEW_PLACE + PLACE_NOTE, OLD_PLACE, "unpatch router place")
    text = patch(text, NEW_PREP, OLD_PREP, "unpatch prepare")
    text = patch(text, NEW_KFN, OLD_KFN, "unpatch kernel_fn")
    text = patch(text, section + ANCHOR, ANCHOR, "unpatch G2 section")
    text = patch(text, '"""' + G2_DOC + "K3 v18 = k3_v17 with the FA4 attention path replaced by K1-FA4 v6", DOC_OLD, "unpatch docstring")
    i = text.index("# nonce " + NONCE)
    j = text.index('"""K3 v18 = k3_v17')
    return text[:i] + "<HDR>" + text[j:]


# --------------------------------------------------------------------------- checks
RIDERS = ("g2a_t1", "g2b_pdl", "side_first_t8")
ARM_WANT = {"ref": (), "all_on": RIDERS, "g2a_t1": ("g2a_t1",), "g2b_pdl": ("g2b_pdl",), "sf8": ("side_first_t8",)}


def check_ab_json(out, base, path):
    """ab_v19a.json: arm 0 = every rider off (= k3_v18's behaviour), all_on = the defaults, one arm per rider; each arm only sets legal switches."""
    arms = json.loads(path.read_text(encoding="utf-8"))
    check([a["name"] for a in arms] == list(ARM_WANT), f"{path.name}: arms {[a['name'] for a in arms]} != {list(ARM_WANT)}")
    dflt_pdl, dflt_place = out.value("G2_P1_PDL"), out.value("V8_ROUTER_PLACE")
    v18_place = base.value("V8_ROUTER_PLACE")
    check(dflt_place == {**v18_place, 8: "side_first"} and v18_place[8] == "side", "V8_ROUTER_PLACE default")
    check(out.value("G2_PRT") is True and out.value("G2_ENABLE") is True and any(dflt_pdl.values()), "rider defaults must be on")
    rows = []
    for a in arms:
        eff_prt, eff_pdl, eff_place = out.value("G2_PRT"), dict(dflt_pdl), dict(dflt_place)
        for k, v in a["set"].items():
            check(k in ("G2_PRT", "G2_P1_PDL", "V8_ROUTER_PLACE"), f"{path.name} {a['name']}: unknown switch {k}")
            if k == "G2_PRT":
                check(isinstance(v, bool), f"{path.name} {a['name']}: G2_PRT not bool")
                eff_prt = v
            elif k == "G2_P1_PDL":
                check(set(v) == set(dflt_pdl) and all(isinstance(x, bool) for x in v.values()), f"{path.name} {a['name']}: G2_P1_PDL keys / values")
                eff_pdl = dict(v)
            else:
                kv = {int(x): y for x, y in v.items()}
                check(set(kv) == set(dflt_place) and all(y in ("side", "side_first", "main") for y in kv.values()), f"{path.name} {a['name']}: V8_ROUTER_PLACE keys / values")
                eff_place = kv
        got = tuple(r for r, on in zip(RIDERS, (eff_prt, any(eff_pdl.values()), eff_place[8] == "side_first")) if on)
        check(got == ARM_WANT[a["name"]], f"{path.name} {a['name']}: riders on {got}, want {ARM_WANT[a['name']]}")
        if a["name"] == "ref":
            check(eff_prt is False and not any(eff_pdl.values()) and eff_place == v18_place, f"{path.name}: ref must equal k3_v18's behaviour (G2_PRT off, no P1 PDL, v18 router places)")
        elif a["name"] != "all_on":
            check(eff_place[1] == v18_place[1] and {k: v for k, v in eff_place.items() if k != 8} == {k: v for k, v in v18_place.items() if k != 8}, f"{path.name} {a['name']}: only bucket 8 may differ")
        rows.append(f"{a['name']}: riders on = {list(got) or 'none (= k3_v18)'}")
    check(arms[1]["set"] == {}, f"{path.name}: all_on must be the defaults")
    return rows


def seg(s, a, b, what):
    check(s.count(a) == 1 and s.count(b) == 1 and s.index(a) < s.index(b), f"seg {what}: anchors")
    return s[s.index(a):s.index(b)]


def provenance(g2_v19, g2_full):
    """Relation of g2_glue_v19.py to g2_glue.py (k3_g2's source): nodes dropped / kept untouched / edited, and for the CUDA text the retained kernel text."""
    n19, nf = dict(g2_nodes(g2_v19)), dict(g2_nodes(g2_full))
    removed = set(nf) - set(n19)
    check(removed == G2_REMOVED, f"g2_glue_v19 drops {sorted(removed ^ G2_REMOVED)} unexpected / missing nodes")
    check(set(n19) <= set(nf), f"g2_glue_v19 adds nodes: {sorted(set(n19) - set(nf))}")
    same = {n for n in n19 if dump(g2_v19.nodes[n]) == dump(g2_full.nodes[n])}
    edited = set(n19) - same
    check(edited == G2_EDITED, f"edited nodes: {sorted(edited ^ G2_EDITED)}")
    for n in ("_g2_stream_h", "_g2_p1_pdl", "G2_ENABLE", "G2_PRT_PDL", "G2_P1_PDL", "_G2_LIVE"):
        check(n in same, f"{n} must stay g2_glue.py's text")
    # CUDA / C++ text: the kernel template, launch and info are g2_glue.py's text byte for byte; the file is a subsequence of it up to the listed edits
    c19, cf = g2_v19.value("G2_CUDA_SRC"), g2_full.value("G2_CUDA_SRC")
    end_info = "return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)occ, (int64_t)smem};\n}\n"
    a19 = c19[c19.index("namespace g2 {"):c19.index(end_info) + len(end_info)]
    af = cf[cf.index("namespace g2 {"):cf.index(end_info) + len(end_info)]
    check(a19 == af and "__global__ void __launch_bounds__(512, 2) prt_kernel(" in a19, "prt_kernel / prt_launch / prt_info text must equal g2_glue.py's")
    w19 = seg(c19, "void g2_prt(torch::Tensor x,", "  TORCH_CHECK(M == 1 && rt == 1", "g2_prt wrapper")
    wf = seg(cf, "void g2_prt(torch::Tensor x,", '  TORCH_CHECK(rt >= 0 && rt <= 2, "g2_prt: router mode 0..2");', "g2_prt wrapper (full)")
    check(w19 == wf, "g2_prt wrapper checks / argument packing must equal g2_glue.py's")
    check(not re.search(r"ffn|FFN|G2-C|ld_acquire|finfo|prt_launch<8|prt_launch<1, [02]>|prt_info<8|prt_info<1, [02]>", c19), "G2-C / T = 8 / RM 0, 2 leftovers in the CUDA text")
    check(len(re.findall(r"prt_launch<\d+, \d+>\(", c19)) == 1 and len(re.findall(r"prt_info<\d+, \d+>\(\)", c19)) == 1, "exactly one prt instantiation")
    cp19, cpf = g2_v19.value("G2_CPP_SRC"), g2_full.value("G2_CPP_SRC")
    check("g2_ffn" not in cp19 and "g2_finfo" not in cp19 and "void g2_prt(" in cp19 and "std::vector<int64_t> g2_info(" in cp19, "G2_CPP_SRC declarations")
    sm = difflib.SequenceMatcher(None, cpf.splitlines(), cp19.splitlines(), autojunk=False)
    check(all(op[0] in ("equal", "delete") for op in sm.get_opcodes()), "G2_CPP_SRC: only deletions vs g2_glue.py")
    nl19, nlf = len(c19.splitlines()), len(cf.splitlines())
    kept = sum(b - a for tag, a, b, _c, _d in difflib.SequenceMatcher(None, cf.splitlines(), c19.splitlines(), autojunk=False).get_opcodes() if tag == "equal")
    return (f"g2_glue_v19.py vs g2_glue.py: {len(same)} nodes untouched ({', '.join(sorted(same))}), {len(edited)} edited ({', '.join(sorted(edited))}), "
            f"{len(removed)} dropped ({', '.join(sorted(removed))}); CUDA text {nl19} of {nlf} lines ({kept} lines identical; prt_kernel / prt_launch / prt_info and the "
            f"g2_prt argument checks byte-identical; one instantiation prt_kernel<1, 1>; G2-C, T = 8, router-first and register-weight code gone)")


def pdl_audit(p1_src, g2_src, stream_h):
    """PDL staleness audit by text (rule: no ld.global.nc / read-only loads of predecessor-written data before griddepcontrol.wait).
    (1) p1::stream: before its one griddepcontrol.wait only A.pk / A.toff / A.eoff / A.esc (prepare-time pack) are read; x (ldx),
        the output and the split-K workspace only after it; p1s_kernel reads A.fmt (prepare-time) before stream and A.out / ws /
        cnt only after it returns. This is what lets G2-B launch P1 with PDL after the combine (o) and after K2A (unpooler).
    (2) stream_h = stream + one `hook();` right after the wait. (3) prt_kernel: before its stream_h call only G.w32 (prepare-time
        router weight) is read; the router input (G.x32) only inside the hook; the pooler output / partials / ticket / tail outputs
        only after stream_h; the tail reads the other CTAs' outputs with __ldcg after a ticket + __threadfence."""
    i = p1_src.index("__device__ __forceinline__ void stream(")
    j = p1_src.index("// D = 1: <= 64 registers", i)
    st = p1_src[i:j]
    check(st.count('asm volatile("griddepcontrol.wait;" ::: "memory");') == 1, "p1::stream: one wait")
    pre, post = st.split('asm volatile("griddepcontrol.wait;" ::: "memory");')
    pre_fields = set(re.findall(r"\bA\.(\w+)", pre))
    check(pre_fields <= {"pk", "toff", "eoff", "esc", "M", "x", "ldx"}, f"p1::stream pre-wait A fields {pre_fields}")
    check("ldx(" not in pre and "A.out" not in pre and "A.ws" not in pre and "ld.global.nc" not in pre and "__ldg(" not in pre, "p1::stream: x / out / ws / read-only load before the wait")
    check(re.findall(r"[^\n]*\bA\.x\b[^\n]*", pre) == ["  const uint32_t* xw = reinterpret_cast<const uint32_t*>(A.x + (size_t)(xon ? g : 0) * A.ldx + (size_t)sb_lo * 64 + 2 * t);"],
          "p1::stream: A.x before the wait only as the xw address")
    k = p1_src.index("__global__ void __launch_bounds__(512, Bounds<D, MODE>::minb) p1s_kernel(")
    kern = p1_src[k:p1_src.index("// Bit-exact unpack", k)]
    first_stream = kern.index("stream<D,")
    check(not re.search(r"\bA\.(out|ws|cnt|x)\b", kern[:first_stream]), "p1s_kernel: output / workspace / x before stream")
    check("ld.global.nc" not in kern[:first_stream] and "__ldg(" not in kern[:first_stream], "p1s_kernel: read-only load before stream")
    check(stream_h.count("hook();") == 1 and 'asm volatile("griddepcontrol.wait;" ::: "memory");\n  hook();\n' in stream_h, "stream_h: hook right after the wait")
    check(stream_h.replace("  hook();\n", "", 1).split("{\n", 1)[1] == st.split("{\n", 1)[1].rstrip("\n") + "\n", "stream_h body != p1::stream body")
    k0 = g2_src.index("__global__ void __launch_bounds__(512, 2) prt_kernel(")
    body = g2_src[k0:g2_src.index("template <int MT, int RM>\nstatic void prt_launch(", k0)]
    h0, h1 = body.index("auto hook = [&]() {"), body.index("stream_h<1, 0>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc, hook);")
    before, hook, after = body[:h0], body[h0:h1], body[h1:]
    pre_g = set(re.findall(r"\bG\.(\w+)", before))
    check(pre_g == {"w32"} and not re.search(r"\bA\.(out|x|ws|cnt)\b", before), f"prt_kernel pre-wait reads {pre_g}")
    check(set(re.findall(r"\bG\.(\w+)", hook)) == {"x32"} and "cp16(" in hook, "prt_kernel: the hook stages G.x32 only")
    check(body.count("stream_h<") == 1 and body.count("hook") == 2, "prt_kernel: one stream_h call, hook defined once")
    tail = after[after.index("if (!s_last) return;"):]
    check(len(re.findall(r"__ldcg\(", tail)) == 5 and "__threadfence();" in tail.split("atomicExch")[0], "tail: __ldcg reads after the ticket fence")
    check("ld.global.nc" not in g2_src and "__ldg(" not in g2_src and "__restrict__" not in body, "G2 source: a read-only load spelling")
    check("griddepcontrol" not in g2_src.replace("@STREAM_H@", ""), "G2 source: its only PDL wait is stream_h's")
    check("__trap" not in g2_src and "while" not in body, "prt_kernel: no trap, no spin (ticket pattern)")
    return len(re.findall(r"\bG\.\w+", body))


def verify(text, base18, g2, g2_full):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(tmp), doraise=True)
        out = Src(tmp)
        out_tree = ast.parse(text)
    finally:
        tmp.unlink()
    tmp = OUT.with_suffix(".py.gen_base")
    tmp.write_text(base18, encoding="utf-8", newline="\n")
    try:
        b18 = Src(tmp)
    finally:
        tmp.unlink()
    rep = []
    check(text.isascii() and g2.text.isascii(), "non-ASCII character")
    lines = text.splitlines()
    check(lines[0] == "# okc: problem=muse_decode_layer", "header line 1")
    check(lines[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1 and "k3-v18-1" not in text, "nonce")
    check(ast.get_docstring(out.tree).startswith("K3 v19 = k3_v18 + the three G2 glue riders"), "docstring")
    # 0. byte-identity to k3_v18 outside the riders
    back = unpatch(text, g2)
    b_i, b_j = base18.index("# nonce k3-v18-1"), base18.index('"""K3 v18 = k3_v17')
    check(back == base18[:b_i] + "<HDR>" + base18[b_j:], "text with the riders taken out != k3_v18.py")
    rep.append("byte-identity: output with the 5 patches reversed, the G2 section cut and the docstring paragraph dropped == k3_v18.py (only the nonce / header lines differ)")
    # 1. nodes
    names = [n for n, _ in g2_nodes(g2)]
    changed = [n for n, node in b18.nodes.items() if n not in out.nodes or dump(out.nodes[n]) != dump(node)]
    check(set(changed) == CHANGED_NODES, f"changed nodes vs k3_v18: {sorted(changed)}")
    added = set(out.nodes) - set(b18.nodes)
    check(added == set(names), f"added nodes {sorted(added)} != g2_glue_v19's {sorted(names)}")
    for n in names:
        check(dump(out.nodes[n]) == dump(g2.nodes[n]), f"{n} differs from g2_glue_v19.py")
    kf_new, kf_old = out.fn("kernel_fn"), b18.fn("kernel_fn")
    src_new, src_old = ast.unparse(kf_new), ast.unparse(kf_old)
    ins = ("g2out = _g2_prt(ffn_input, router_input, state, data, tokens, mode, tail)\n    if g2out is not None:\n"
           "        expert_x, topk_weights, topk_ids, local_ids = g2out\n    elif tail == 'cat':")
    check(src_new.count(ins) == 1 and src_old.count("\n    if tail == 'cat':") == 1, "kernel_fn: G2-A block")
    check(ast.dump(ast.parse(src_new.replace(ins, "if tail == 'cat':"))) == ast.dump(ast.parse(src_old)), "kernel_fn: anything but the G2-A block changed")
    p_new, p_old = out.fn("prepare"), b18.fn("prepare")
    ci = [i for i, s in enumerate(p_new.body) if ast.unparse(s) == "_g2_prepare(state, t0)"]
    check(len(ci) == 1 and ast.unparse(p_new.body[ci[0] - 1]) == "_v18_fa4_prepare(state, t0)"
          and [dump(s) for s in p_new.body[:ci[0]] + p_new.body[ci[0] + 1:]] == [dump(s) for s in p_old.body], "prepare: one inserted call")
    pl_new, pl_old = out.value("V8_ROUTER_PLACE"), b18.value("V8_ROUTER_PLACE")
    check({k: v for k, v in pl_new.items() if k != 8} == {k: v for k, v in pl_old.items() if k != 8} and pl_old[8] == "side" and pl_new[8] == "side_first",
          "V8_ROUTER_PLACE: only bucket 8 -> side_first")
    gv_new, gv_old = ast.unparse(out.fn("p1_gemv")), ast.unparse(b18.fn("p1_gemv"))
    check(gv_new.replace("pdl=_g2_p1_pdl(proj), ", "") == gv_old and gv_new.count("pdl=_g2_p1_pdl(proj)") == 1, "p1_gemv: only the pdl argument")
    rep.append(f"vs k3_v18 (gen output == file on disk): {len(b18.nodes) - len(changed)} top-level nodes ast-identical; changed {sorted(changed)} "
               f"(kernel_fn: one G2-A block, prepare: one call, p1_gemv: pdl argument, V8_ROUTER_PLACE: bucket 8 side_first); added {len(added)} nodes == g2_glue_v19.py's, ast-identical")
    # 2. provenance of the trimmed G2 source
    rep.append(provenance(g2, g2_full))
    # 3. PDL audit
    p1_src, g2_src = out.value("P1_CUDA_SRC"), out.value("G2_CUDA_SRC")
    check(p1_src == b18.value("P1_CUDA_SRC") and out.value("P1_CPP_SRC") == b18.value("P1_CPP_SRC"), "P1 sources must stay k3_v18's")
    ns = {}
    exec(compile(ast.Module(body=[out.nodes["_g2_stream_h"]], type_ignores=[]), "g2", "exec"), ns)  # noqa: S102 (pure string function; generator only)
    stream_h = ns["_g2_stream_h"](p1_src)
    n_g = pdl_audit(p1_src, g2_src, stream_h)
    check(g2_src.count("@STREAM_H@") == 1, "G2_CUDA_SRC placeholder")
    rep.append("PDL audit: p1::stream reads only the prepare-time pack (A.pk / toff / eoff / esc) before its one griddepcontrol.wait; x (ldx), output and split-K workspace only "
               "after it; p1s_kernel reads A.fmt before stream and A.out / ws / cnt after; no ld.global.nc / __ldg before the wait. stream_h = stream + one hook() right after "
               f"the wait (body otherwise byte-identical); prt_kernel reads only G.w32 (prepare-time) before the wait, G.x32 only in the hook, other CTAs' outputs only with __ldcg "
               f"after the ticket fence; no ld.global.nc / __ldg / spin / trap in G2 ({n_g} G.* references). New PDL launches: P1 o (after the FA4 combine), P1 unpooler (after K2A), prt_kernel")
    # 4. switches
    check(all(out.value(k) is True for k in ("G2_ENABLE", "G2_PRT", "G2_PRT_PDL")) and out.value("G2_PRT_MODE") == {1: "fused"}
          and out.value("G2_PREPARE_BY_S") == 190.0 and "G2_FFN" not in out.nodes, "G2 defaults")
    check(out.value("G2_P1_PDL") == {"qkv": False, "o": True, "pooler": False, "unpooler": True}, "G2_P1_PDL default")
    check(all(t == 1 for t, _l, _s in out.value("G2_SELF_TEST_CASES")), "G2 self-test cases are T = 1")
    # 5. hygiene
    v6 = G18.SRC_V6.read_bytes().decode("utf-8")
    check(text.count(v6) == 1, "k1_fa4v6.py text must occur once")
    v6_off = text.index(v6)
    v6_lines = (text[:v6_off].count(chr(10)) + 1, text[:v6_off].count(chr(10)) + v6.count(chr(10)))
    for node in ast.walk(out_tree):
        if isinstance(node, ast.Name) and not (v6_lines[0] <= node.lineno <= v6_lines[1] and node.id in ("os", "sys")):
            check(node.id not in G.BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
    for fn in CALL_TIME_FNS:
        for sub in ast.walk(out.nodes[fn]):
            check(not (isinstance(sub, ast.Attribute) and sub.attr in G.SYNC_ATTRS), f"{fn} has a host sync")
    bad = G8.G7.undefined_globals(text, str(OUT))
    check(not bad, f"undefined globals: {sorted(bad)}")
    nbytes = len(text.encode())
    check(nbytes < 1_000_000, "file >= 1 MB")
    rep.append(f"static ok (no banned names, no host sync in {len(CALL_TIME_FNS)} layer call-time functions, no undefined globals); {nbytes} bytes "
               f"(k3_v18: {len(base18.encode())}; limit 1,000,000), {len(lines)} lines")
    # 6. checker scan (comments and docstrings included) + module-attribute writes by AST
    hits = G18.scan_text(text)
    bad_hits = {k: v for k, v in hits.items() if v}
    rep.append("CHECKER SCAN of k3_v19.py (comments and docstrings included): " + ", ".join(f"{k}: {len(v)}" for k, v in hits.items()))
    check(not bad_hits, "checker scan hits: " + "; ".join(f"{k}: {v[:6]}" for k, v in bad_hits.items()))
    full_names = G18.bound_names(out_tree)
    mods = {n for n, nodes in full_names.items() if all(isinstance(x, (ast.Import, ast.ImportFrom)) for x in nodes)}
    for node in ast.walk(out_tree):
        tg = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
        for t in tg:
            if isinstance(t, ast.Attribute):
                root = t
                while isinstance(root, ast.Attribute):
                    root = root.value
                check(not (isinstance(root, ast.Name) and root.id in mods), f"line {node.lineno}: assignment to an attribute of the imported name {root.id}: {ast.unparse(t)}")
        check(not isinstance(node, ast.Delete) or not any(isinstance(t, ast.Attribute) for t in node.targets), f"line {getattr(node, 'lineno', 0)}: del of an attribute")
    g2_text = "\n".join(seg_ for _, seg_ in g2_nodes(g2))
    g2_hits = {k: v for k, v in G18.scan_text(g2_text).items() if v}
    check(not g2_hits, f"G2 section scan hits {g2_hits}")
    rep.append("AST scan: no assignment / del to an attribute of any imported name (module attribute writes); the G2 section writes only its own dict items (_G2_LIVE, state, g2)")
    for r in check_ab_json(out, b18, AB_JSON):
        rep.append(f"{AB_JSON.name} arm  {r}")
    return rep


def main(argv):
    text, base18, g2 = generate()
    g2_full = Src(SRC_G2_FULL)
    for r in verify(text, base18, g2, g2_full):
        print(f"[gen_k3_v19] OK  {r}")
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text(encoding="utf-8") == text, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_v19] --check: up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"[gen_k3_v19] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}, {len(text.splitlines())} lines, {len(text.encode())} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

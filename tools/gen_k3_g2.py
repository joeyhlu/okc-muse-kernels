"""gen_k3_g2.py: k3_g2.py = k3_v17.py (as gen_k3_v17.py generates it) + the G2 glue fusions at T <= 8 (kernels/muse_decode_layer/
g2_glue.py, g2_design.md).

    python -I tools/gen_k3_g2.py            # writes kernels/muse_decode_layer/k3_g2.py, asserts the invariants
    python -I tools/gen_k3_g2.py --check    # compares with the file on disk, writes nothing

Changes vs k3_v17 (everything else asserted ast-identical to gen_k3_v17's output, which is asserted == k3_v17.py on disk):
  * G2 section (before _k3_disable): every top-level node of g2_glue.py verbatim (G2_* switches, the CUDA source, build, call,
    self-tests, _g2_prepare).
  * kernel_fn: one `g2out = _g2_prt(...)` statement after the attention boundary; `if tail == "cat":` becomes the elif of
    `if g2out is not None:` (G2-A replaces router / pooler / tail when it returns a result; None = k3_v17's code unchanged).
  * prepare: `_g2_prepare(state, t0)` after `_v16_prepare(state, t0)`.
  * p1_gemv: `pdl=_g2_p1_pdl(proj)` passed to p1_call (G2-B; 0 = k3_v17's launch).
  * kernel_fn: `g2f = _g2_ffn(...)` before the unpooler; a result returns (next input, stream, topk_ids) at once (G2-C replaces the
    unpooler + FFN boundary); None = k3_v17's code unchanged.
"""

import ast
import importlib.util
import json
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "kernels/muse_decode_layer/k3_g2.py"
V17_FILE = ROOT / "kernels/muse_decode_layer/k3_v17.py"
SRC_G2 = ROOT / "kernels/muse_decode_layer/g2_glue.py"
AB_JSON = ROOT / "tools/box/ab_g2.json"
NONCE = "k3-g2-1"

_s17 = importlib.util.spec_from_file_location("gen_k3_v17", ROOT / "tools/gen_k3_v17.py")
G17 = importlib.util.module_from_spec(_s17)
_s17.loader.exec_module(G17)
G, G8 = G17.G, G17.G8
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha

CHANGED_NODES = {"kernel_fn", "prepare", "p1_gemv", "V8_ROUTER_PLACE"}
CALL_TIME_FNS = G17.CALL_TIME_FNS + ("_g2_prt", "_g2_prt_run", "_g2_p1_pdl", "p1_gemv", "_g2_ffn", "_g2_ffn_run")

G2_DOC = '''K3 G2 = k3_v17 + the G2 glue fusions at T <= 8 (kernels/muse_decode_layer/g2_design.md; everything else is k3_v17, asserted
ast-identical by tools/gen_k3_g2.py except kernel_fn / prepare / p1_gemv):

  G2-A (G2_PRT, read at call time): the P1 pooler GEMV, the T <= 8 FP32 router (router32's math) and the routing / pre-expert
       norm / local-id tail (tail_kernel's math) in ONE launch (256 CTAs; router weights cp.async'd before the PDL wait; the last
       CTA by ticket runs the tail). Replaces the side-stream router, the stream fork / join, the P1 pooler launch and the tail.
       Live only after its prepare self-test vs the three-kernel path (ids / local ids equal, xn / weights within 1e-3 / 1e-6,
       ticket back at zero, repeat identical); otherwise k3_v17's path.
  G2-B (G2_P1_PDL, per projection, read at call time): P1 GEMV launches with programmatic stream serialization (o, unpooler
       on; the pooler only matters with G2-A off). Live after a bitwise pdl 1 == pdl 0 self-test.

'''

OLD_KFN = '''    ffn_input, stream, router_input = _v9_bnd("attn", data["stream"], attn_output,
                                              None if split_router else torch.float32, state, tokens)
    if tail == "cat":
'''
NEW_KFN = '''    ffn_input, stream, router_input = _v9_bnd("attn", data["stream"], attn_output,
                                              None if split_router else torch.float32, state, tokens)
    g2out = _g2_prt(ffn_input, router_input, state, data, tokens, mode, tail)   # G2-A: pooler + FP32 router + tail in one kernel (T <= 8), else None
    if g2out is not None:
        expert_x, topk_weights, topk_ids, local_ids = g2out
    elif tail == "cat":
'''
OLD_KFN2 = '''    ffn_output = _v13_unpool(expert_output, state, tokens)   # V13: P1 into the pre-zeroed buffer (T <= 8, routed), else None
'''
NEW_KFN2 = '''    g2f = _g2_ffn(expert_output, stream, state, tokens)   # G2-C: P1 unpooler + the FFN boundary in one kernel (T <= 8), else None
    if g2f is not None:
        return g2f[0], g2f[1], topk_ids
    ffn_output = _v13_unpool(expert_output, state, tokens)   # V13: P1 into the pre-zeroed buffer (T <= 8, routed), else None
'''
OLD_PLACE = "V8_ROUTER_PLACE = {1: 'side', 8: 'side', 16: 'side', 32: 'side', 128: 'side', 256: 'main'}"
NEW_PLACE = "V8_ROUTER_PLACE = {1: 'side', 8: 'side_first', 16: 'side', 32: 'side', 128: 'side', 256: 'main'}"
OLD_PREP = "    _v16_prepare(state, t0)\n    torch.cuda.synchronize()\n"
NEW_PREP = "    _v16_prepare(state, t0)\n    _g2_prepare(state, t0)\n    torch.cuda.synchronize()\n"
OLD_GEMV = '    p1_call(st["mod"], P, x, out, n_rows, ldo, geo, mode=mode)\n    return out\n'
NEW_GEMV = '    p1_call(st["mod"], P, x, out, n_rows, ldo, geo, pdl=_g2_p1_pdl(proj), mode=mode)   # G2-B: PDL launch where switched on\n    return out\n'
ANCHOR = "def _k3_disable(state, key, value, piece, ex):\n"
BANNER = "# ---------------------------------------------------------------- G2 (kernels/muse_decode_layer/g2_glue.py, verbatim)\n"


def g2_nodes(src):
    """(name, text) of every named top-level node of g2_glue.py, in file order (imports and the docstring are the layer's)."""
    out = []
    for node in src.tree.body:
        names = Src._names(node)
        if not names:
            check(isinstance(node, (ast.Import, ast.ImportFrom)) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)),
                  f"g2_glue.py: unexpected unnamed top-level node at line {node.lineno}")
            continue
        out.append((names[0], "".join(src.lines[node.lineno - 1:node.end_lineno])))
    return out


def generate():
    text, base16, b16, k36, k37, kernel_src = G17.generate()
    check(V17_FILE.read_text(encoding="utf-8") == text, "gen_k3_v17's output differs from k3_v17.py on disk (regenerate v17 first)")
    base17 = text
    g2 = Src(SRC_G2)
    lines = text.split("\n")
    check(lines[1].startswith("# nonce k3-v17-1 "), "k3_v17 nonce")
    hdr = "\n".join(lines[2:5]) + "\n"
    check(hdr.startswith("# GENERATED by tools/gen_k3_v17.py") and hdr.endswith("regenerate.\n"), "k3_v17 generated-by header")
    text = patch(text, lines[1] + "\n" + hdr,
                 f"# nonce {NONCE} (k3_v17 + G2 glue fusions at T <= 8: pooler + FP32 router + tail in one kernel, P1 PDL launches; g2_design.md)\n"
                 "# GENERATED by tools/gen_k3_g2.py on top of gen_k3_v17.py's output (k3_v17.py; its chain is gen_k3_v16 / v15 /\n"
                 "# v14 / v13 ...) and kernels/muse_decode_layer/g2_glue.py (G2).\n"
                 "# Do not hand-edit: change a canonical source or a generator and regenerate.\n", "nonce + header")
    text = patch(text, '"""K3 v17 = k3_v16 + K2A v3.7', '"""' + G2_DOC + 'K3 v17 = k3_v16 + K2A v3.7', "docstring")
    section = BANNER + "".join(seg.rstrip("\n") + "\n\n\n" for _, seg in g2_nodes(g2))
    text = patch(text, ANCHOR, section + ANCHOR, "G2 section")
    text = patch(text, OLD_KFN, NEW_KFN, "kernel_fn G2-A branch")
    text = patch(text, OLD_KFN2, NEW_KFN2, "kernel_fn G2-C branch")
    text = patch(text, OLD_PREP, NEW_PREP, "prepare G2")
    text = patch(text, OLD_PLACE, NEW_PLACE + "   # G2: bucket 8 side_first (router32 launched before the P1 pooler; box A/B g2_ab4 +0.6/+1.0% at verify 1)", "router place")
    text = patch(text, OLD_GEMV, NEW_GEMV, "p1_gemv G2-B")
    return text, base17, g2


# --------------------------------------------------------------------------- checks
def check_ab_json(out, path):
    """ab_g2.json: arm 0 = every G2 switch off (= k3_v17 behaviour); the other arms only set G2 switches to legal values."""
    arms = json.loads(path.read_text(encoding="utf-8"))
    check(arms[0]["name"] == "ref" and arms[0]["set"].get("G2_PRT") is False and arms[0]["set"].get("G2_FFN") is False
          and set(arms[0]["set"]) == {"G2_PRT", "G2_P1_PDL", "G2_FFN"} and not any(arms[0]["set"]["G2_P1_PDL"].values()),
          f"{path.name}: arm 0 must be ref = G2_PRT off + G2_FFN off + every G2_P1_PDL off")
    keys = set(out.value("G2_P1_PDL"))
    rows = []
    for a in arms:
        for k, v in a["set"].items():
            check(k in ("G2_PRT", "G2_PRT_PDL", "G2_P1_PDL", "G2_FFN", "G2_FFN_T1"), f"{path.name} {a['name']}: unknown switch {k}")
            if k == "G2_P1_PDL":
                check(set(v) == keys and all(isinstance(x, bool) for x in v.values()), f"{path.name} {a['name']}: G2_P1_PDL keys / values")
            else:
                check(isinstance(v, bool), f"{path.name} {a['name']}: {k} not bool")
        rows.append(f"{a['name']}: {a['set'] or 'defaults'}")
    return rows


def pdl_audit(p1_src, g2_src, stream_h):
    """PDL staleness audit by text (rule: no ld.global.nc / read-only loads of predecessor-written data before griddepcontrol.wait).
    (1) p1::stream: before its one griddepcontrol.wait only A.pk / A.toff / A.eoff / A.esc (prepare-time pack) are read; x (ldx),
        the output and the split-K workspace only after it; p1s_kernel reads A.fmt (prepare-time) before stream and A.out / ws /
        cnt only after it returns. (2) stream_h = stream + one `hook();` right after the wait. (3) prt_kernel: before its
        stream_h call only G.w32 (prepare-time router weight) is read; the router input (G.x32) only inside the hook; the pooler
        output / partials / ticket / tail outputs only after stream_h; the tail reads the other CTAs' outputs with __ldcg after a
        ticket + __threadfence."""
    i = p1_src.index("__device__ __forceinline__ void stream(")
    j = p1_src.index("// D = 1: <= 64 registers", i)
    st = p1_src[i:j]
    pre, post = st.split('asm volatile("griddepcontrol.wait;" ::: "memory");')
    pre_fields = set(re.findall(r"\bA\.(\w+)", pre))
    check(pre_fields <= {"pk", "toff", "eoff", "esc", "M", "x", "ldx"}, f"p1::stream pre-wait A fields {pre_fields}")
    check("ldx(" not in pre and "A.out" not in pre and "A.ws" not in pre, "p1::stream: x / out / ws before the wait")
    # A.x / A.ldx appear before the wait only in the address computation of xw (no load): assert that
    check(re.findall(r"[^\n]*\bA\.x\b[^\n]*", pre) == ["  const uint32_t* xw = reinterpret_cast<const uint32_t*>(A.x + (size_t)(xon ? g : 0) * A.ldx + (size_t)sb_lo * 64 + 2 * t);"],
          "p1::stream: A.x before the wait only as the xw address")
    k = p1_src.index("__global__ void __launch_bounds__(512, Bounds<D, MODE>::minb) p1s_kernel(")
    kern = p1_src[k:p1_src.index("// Bit-exact unpack", k)]
    first_stream = kern.index("stream<D,")
    check(not re.search(r"\bA\.(out|ws|cnt|x)\b", kern[:first_stream]), "p1s_kernel: output / workspace / x before stream")
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
    # G2-C ffn_kernel: prepare-time gate / norm weights before the wait, the stream only in the hook, one bounded barrier spin
    f0 = g2_src.index("__global__ void __launch_bounds__(512, 2) ffn_kernel(")
    fb = g2_src[f0:g2_src.index("template <int MT>\nstatic void ffn_launch(", f0)]
    q0, q1 = fb.index("auto hook = [&]() {"), fb.index("stream_h<1, 0>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc, hook);")
    f_pre, f_hook, f_post = fb[:q0], fb[q0:q1], fb[q1:]
    pre_f = set(re.findall(r"\bF\.(\w+)", f_pre))
    check(pre_f == {"chw", "alpha", "beta", "wpost", "wpre"} and not re.search(r"\bA\.(out|x|ws|cnt)\b", f_pre), f"ffn_kernel pre-wait reads {pre_f}")
    check(set(re.findall(r"\bF\.(\w+)", f_hook)) == {"s"} and "__ldcg(F.s" in f_hook, "ffn_kernel: the hook loads the stream only")
    check(fb.count("while") == 1 and "while (ld_acquire(F.bar) - target < 0) {\n      if (++n > (1 << 20)) { atomicExch(F.bar + 1, 1); break; }" in f_post,
          "ffn_kernel: one bounded spin with a timeout flag")
    check(f_post.index("const float* wp = F.ws") > f_post.index("ld_acquire(F.bar)") > f_post.index("__threadfence();") > f_post.index("__stcg(F.ws")
          and f_post.count("__ldcg(wp + ") == 4,
          "ffn_kernel: partials stored, fenced, barrier, then read with __ldcg")
    return len(re.findall(r"\bG\.\w+", body)) + len(re.findall(r"\bF\.\w+", fb))


def verify(text, base17, g2):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(tmp), doraise=True)
        out = Src(tmp)
    finally:
        tmp.unlink()
    tmp = OUT.with_suffix(".py.gen_base")
    tmp.write_text(base17, encoding="utf-8", newline="\n")
    try:
        b17 = Src(tmp)
    finally:
        tmp.unlink()
    rep = []
    check(text.isascii() and g2.text.isascii(), "non-ASCII character")
    lines = text.splitlines()
    check(lines[0] == "# okc: problem=muse_decode_layer", "header line 1")
    check(lines[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1 and "k3-v17-1" not in text, "nonce")
    check(ast.get_docstring(out.tree).startswith("K3 G2 = k3_v17 + the G2 glue fusions"), "docstring")
    # nodes
    names = [n for n, _ in g2_nodes(g2)]
    changed = [n for n, node in b17.nodes.items() if n not in out.nodes or dump(out.nodes[n]) != dump(node)]
    check(set(changed) == CHANGED_NODES, f"changed nodes vs k3_v17: {sorted(changed)}")
    added = set(out.nodes) - set(b17.nodes)
    check(added == set(names), f"added nodes {sorted(added)} != g2_glue's {sorted(names)}")
    for n in names:
        check(dump(out.nodes[n]) == dump(g2.nodes[n]), f"{n} differs from g2_glue.py")
    # the three patches, as AST diffs
    kf_new, kf_old = out.fn("kernel_fn"), b17.fn("kernel_fn")
    src_new, src_old = ast.unparse(kf_new), ast.unparse(kf_old)
    ins = ("g2out = _g2_prt(ffn_input, router_input, state, data, tokens, mode, tail)\n    if g2out is not None:\n"
           "        expert_x, topk_weights, topk_ids, local_ids = g2out\n    elif tail == 'cat':")
    check(src_new.count(ins) == 1 and src_old.count("\n    if tail == 'cat':") == 1, "kernel_fn: G2-A block")
    ins2 = "g2f = _g2_ffn(expert_output, stream, state, tokens)\n    if g2f is not None:\n        return (g2f[0], g2f[1], topk_ids)\n    "
    check(src_new.count(ins2) == 1, "kernel_fn: G2-C block")
    rebuilt = src_new.replace(ins, "if tail == 'cat':").replace(ins2, "")
    # in the old function the `if tail == 'cat'` chain is the same statement with the old indentation
    check(ast.dump(ast.parse(rebuilt)) == ast.dump(ast.parse(src_old)), "kernel_fn: anything but the G2-A block changed")
    p_new, p_old = out.fn("prepare"), b17.fn("prepare")
    ci = [i for i, s in enumerate(p_new.body) if ast.unparse(s) == "_g2_prepare(state, t0)"]
    check(len(ci) == 1 and ast.unparse(p_new.body[ci[0] - 1]) == "_v16_prepare(state, t0)"
          and [dump(s) for s in p_new.body[:ci[0]] + p_new.body[ci[0] + 1:]] == [dump(s) for s in p_old.body], "prepare: one inserted call")
    pl_new, pl_old = out.value("V8_ROUTER_PLACE"), b17.value("V8_ROUTER_PLACE")
    check({k: v for k, v in pl_new.items() if k != 8} == {k: v for k, v in pl_old.items() if k != 8} and pl_old[8] == "side" and pl_new[8] == "side_first",
          "V8_ROUTER_PLACE: only bucket 8 -> side_first")
    gv_new, gv_old = ast.unparse(out.fn("p1_gemv")), ast.unparse(b17.fn("p1_gemv"))
    check(gv_new.replace("pdl=_g2_p1_pdl(proj), ", "") == gv_old and gv_new.count("pdl=_g2_p1_pdl(proj)") == 1, "p1_gemv: only the pdl argument")
    rep.append(f"vs k3_v17 (gen output == file on disk): {len(b17.nodes) - len(changed)} top-level nodes ast-identical; changed {sorted(changed)} "
               f"(kernel_fn: one G2-A block, prepare: one call, p1_gemv: pdl argument); added {len(added)} nodes == g2_glue.py's, ast-identical")
    # G2 CUDA: derived stream, PDL audit, sources
    p1_src, g2_src = out.value("P1_CUDA_SRC"), out.value("G2_CUDA_SRC")
    check(p1_src == b17.value("P1_CUDA_SRC") and out.value("P1_CPP_SRC") == b17.value("P1_CPP_SRC"), "P1 sources must stay k3_v17's")
    ns = {}
    exec(compile(ast.Module(body=[out.nodes["_g2_stream_h"]], type_ignores=[]), "g2", "exec"), ns)  # noqa: S102 (pure string function)
    stream_h = ns["_g2_stream_h"](p1_src)
    n_g = pdl_audit(p1_src, g2_src, stream_h)
    check(g2_src.count("@STREAM_H@") == 1, "G2_CUDA_SRC placeholder")
    rep.append(f"PDL audit: p1::stream reads only the prepare-time pack before its griddepcontrol.wait (x / out / workspace after); stream_h = "
               f"stream + one hook() right after the wait (body otherwise byte-identical); prt_kernel reads only G.w32 before the wait, "
               f"G.x32 only in the hook, other CTAs' outputs only with __ldcg after the ticket fence; ffn_kernel reads only the gate / norm weights before the wait, the stream in the hook, partials via __stcg -> fence -> one bounded barrier spin (timeout flag) -> __ldcg; no ld.global.nc / __ldg / spin / trap in G2 "
               f"({n_g} G.* references)")
    # switches
    check(all(out.value(k) is True for k in ("G2_ENABLE", "G2_PRT", "G2_PRT_PDL", "G2_FFN_T1")) and out.value("G2_FFN") is False
          and out.value("G2_PRT_MODE") == {1: "fused", 8: "off"}, "G2 defaults")
    check(out.value("G2_P1_PDL") == {"qkv": False, "o": True, "pooler": False, "unpooler": True}, "G2_P1_PDL default")
    # hygiene (as gen_k3_v17)
    for node in ast.walk(out.tree):
        if isinstance(node, ast.Name):
            check(node.id not in G.BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
    for fn in CALL_TIME_FNS:
        for sub in ast.walk(out.nodes[fn]):
            check(not (isinstance(sub, ast.Attribute) and sub.attr in G.SYNC_ATTRS), f"{fn} has a host sync")
    bad = G8.G7.undefined_globals(text, str(OUT))
    check(not bad, f"undefined globals: {sorted(bad)}")
    code_only = "\n".join(l_ for l_ in lines if not l_.lstrip().startswith("#"))
    check(re.search(r"\benviron\b", code_only) is None, "`environ` outside comments")
    nbytes = len(text.encode())
    check(nbytes < 1_000_000, "file >= 1 MB")
    rep.append(f"static ok (no banned names, no host sync in {len(CALL_TIME_FNS)} call-time functions, no undefined globals); {nbytes} bytes "
               f"(k3_v17: {len(base17.encode())}; limit 1,000,000)")
    for r in check_ab_json(out, AB_JSON):
        rep.append(f"{AB_JSON.name} arm  {r}")
    return rep


def main(argv):
    text, base17, g2 = generate()
    for r in verify(text, base17, g2):
        print(f"[gen_k3_g2] OK  {r}")
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text(encoding="utf-8") == text, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_g2] --check: up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"[gen_k3_g2] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}, {len(text.splitlines())} lines, {len(text.encode())} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

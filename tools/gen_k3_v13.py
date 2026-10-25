"""gen_k3_v13.py: k3_v13.py = k3_v12.py (as gen_k3_v12.py generates it) + P1 packed-weight projection GEMVs + the screened lever bundle.

    python3 -I tools/gen_k3_v13.py            # writes kernels/muse_decode_layer/k3_v13.py, asserts the invariants
    python3 -I tools/gen_k3_v13.py --check    # compares with the file on disk, writes nothing

Changes vs k3_v12 (everything else asserted ast-identical to gen_k3_v12's output, which is asserted == k3_v12.py on disk):
  * P1 (kernels/muse_bf16_proj/p1_module.py) inlined verbatim from its first definition (P1_C) to the end of the file; its
    docstring and its three imports (hashlib, time, torch) are k3_v12's own. Names: p1_*, P1_*, _P1_MOD, _wrap32 (asserted
    free in k3_v12). The CUDA / C++ literals are asserted byte-identical to p1_module's and free of triple quotes.
    Only p1s_kernel<1, 0> is built (p1_prepare's default layer_only=True -> -DP1_LAYER_ONLY=1).
  * Call sites in kernel_fn (each falls back to the incumbent expression when P1 returns None): qkv, o, the three pooler
    calls on the main stream, and the T <= 8 unpooler into the pre-zeroed buffer. Module switches P1_ENABLE / P1_PROJ,
    read at call time, so tools/box/layer_ab2.py can flip them per variant.
  * prepare(): _v13_prepare after _v12_prepare (pack, bit-exact unpack check, one-hot + random-x self-test per routed
    (projection, T) against FP64 and the incumbent path; a failure turns that route off).
  * Lever bundle (notes/study/ab_screen_plan.md, analyze.py's pick AE): K1_ROUTE_MIN_BATCH[('full', 8)] 32 -> 16 and
    K2A33_PLAN[32] 0 -> 1.
"""

import ast
import importlib.util
import json
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "kernels/muse_decode_layer/k3_v13.py"
V12_FILE = ROOT / "kernels/muse_decode_layer/k3_v12.py"
SRC_P1 = ROOT / "kernels/muse_bf16_proj/p1_module.py"
AB_JSON = ROOT / "tools/box/ab_v13a.json"
AB_JSON_B = ROOT / "tools/box/ab_v13b.json"   # ref = k3_v12-equivalent: the bundle against v12
NONCE = "k3-v13-1"

_s12 = importlib.util.spec_from_file_location("gen_k3_v12", ROOT / "tools/gen_k3_v12.py")
G12 = importlib.util.module_from_spec(_s12)
_s12.loader.exec_module(G12)
G11, G10, G9, G8, G = G12.G11, G12.G10, G12.G9, G12.G8, G12.G
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha

K1_ROUTE = {("sliding", 1): 1, ("full", 1): 1, ("full", 8): 16}     # A: v12 had ("full", 8): 32
K2A33_PLAN = {1: 0, 8: 0, 16: 0, 32: 1, 128: 2, 256: 2}              # E: v12 had 32: 0
P1_PROJ = {"qkv": True, "o": True, "pooler": True, "unpooler": True}
V13_NODES = {"P1_ENABLE", "P1_PROJ", "_v13_p1_ok", "_v13_p1", "_v13_pooler", "_v13_unpool", "_v13_today",
             "_v13_p1_selftest", "_v13_prepare"}
CALL_TIME_FNS = ("kernel_fn", "_v13_p1_ok", "_v13_p1", "_v13_pooler", "_v13_unpool", "p1_bucket", "p1_route", "p1_gemv",
                 "p1_call", "p1_mode_of")    # no host sync inside (prepare-only code is not checked)
K1_SELF_LINE = "                ref = _k1_ref_attention(d, k_new, v_new, q_e4m3=(L == 8))   # v5: L=8 E4M3 Q (wgmma) + E4M3 P\n"

V13_DOC = '''K3 v13 = k3_v12 + P1 + two screened levers (everything else is k3_v12, asserted ast-identical by tools/gen_k3_v13.py):

  P1 (kernels/muse_bf16_proj/p1_module.py, inlined verbatim; names p1_*, P1_*, _P1_MOD, _wrap32): the four dense
       projections keep a lossless 12-bit packed copy of their BF16 weights (packed, checked bit-exact and
       self-tested in prepare) and run a streaming GEMV on it for T <= 8 (T = 1: decode 1; T = 8: verify 1): CUDA-graph
       safe, deterministic, one extra load_inline build holding only p1s_kernel<1, 0>. Routes per (projection, T bucket)
       from the measured nets (P1_ROUTE): qkv T1 / T8, o T1, pooler T1 / T8 (main stream only), unpooler T8 (the
       1024-column slice into the persistent pre-zeroed buffer). A projection whose pack, self-test or call fails
       keeps today's path. Switches P1_ENABLE and P1_PROJ (per projection) are read at call time.
  A    K1_ROUTE_MIN_BATCH[('full', 8)] 32 -> 16: K1 v7's L=8 path also at verify 16 full.
  E    K2A33_PLAN[32] 0 -> 1: K2A v3.3 variant 1 (NC 2, 16 tokens per pass) at bucket 32 (decode 32).
  (A and E: the k3_v12 lever screen, notes/study/ab_screen_plan.md; both are runtime switches.)

'''

V13_GLUE = '''

def _v13_p1_ok(x, k):
    """The host checks of p1s_gemv for x, shape / stride / pointer only (graph safe): bf16 [T, K] contiguous, 16 B aligned."""
    return x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == k and x.is_contiguous() and x.data_ptr() % 16 == 0


def _v13_p1(state, proj, x, out=None, **kw):
    """P1's packed-weight GEMV for (projection, x.shape[0] tokens), or None -> the caller keeps today's path: P1_ENABLE /
    P1_PROJ off, state["p1"] unset, the route table says no (T > 8, bucket not routed, projection dropped), or x fails the
    kernel's host checks. Shape-only branches, no sync. A host exception disables P1 for the rest of the run (once, with a
    line), as K1 / K2A do; p1s_gemv raises before it launches anything, so nothing is left half-captured."""
    st = state.get("p1")
    if not P1_ENABLE or st is None or not P1_PROJ.get(proj):
        return None
    if p1_route(st, proj, x.shape[0]) is None or not _v13_p1_ok(x, st["proj"][proj]["K"]):
        return None
    try:
        return p1_gemv(st, proj, x, out, **kw)
    except Exception as ex:  # noqa: BLE001
        _k3_disable(state, "p1", None, "P1", ex)
        return None


def _v13_pooler(ffn_input, dense, state):
    """The pooler GEMM on the main stream: P1 where routed (T <= 8), else the incumbent (the cat GEMM at T >= 16 and the FP32
    router on the side stream are not touched)."""
    y = _v13_p1(state, "pooler", ffn_input)
    return y if y is not None else _v8_gemm(ffn_input, "pooler", dense["pooler"], state)


def _v13_unpool(expert_output, state, tokens):
    """T <= 8 unpooler on P1: this GPU's 1024 columns straight into the persistent pre-zeroed [8, 8192] buffer (row stride
    8192; columns >= 1024 are never written, so they stay zero), or None -> today's path (v12's zero buffer / R1B GEMV)."""
    zb = state.get("v12_zbuf")
    if not (V12_ZBUF and zb is not None and R1_UNPOOL_INTO_ZEROS and tokens <= SLICED_UNPOOL_MAX_TOKENS
            and expert_output.dtype == zb.dtype):
        return None
    ffn_output = zb[:tokens]
    y = _v13_p1(state, "unpooler", expert_output, ffn_output, n_rows=UNPOOL_SLICE, ldo=HIDDEN)
    return ffn_output if y is not None else None


def _v13_today(proj, x, state, dense):
    """The incumbent path of each P1 projection for x.shape[0] tokens (the self-test's reference)."""
    tokens = x.shape[0]
    if proj == "qkv":
        return (_v8_gemm(x, "qkv_gate", dense["qkv_gate"], state) if tokens == 1
                else _v12_gemm(x, dense["qkv_gate"], "qkv", tokens))
    if proj == "o":
        return _v12_gemm(x, dense["o"], "o", tokens)
    if proj == "pooler":
        return _v8_gemm(x, "pooler", dense["pooler"], state)
    return torch.mm(x, dense["unpooler"][:UNPOOL_SLICE].t())


def _v13_p1_selftest(st, state, proj, T, gen):
    """(ok, message) for one routed (projection, T): one-hot probes (x = e_k gives y = W[:, k] exactly; escape columns
    first, the tensor core's subnormal flush counted apart, rows with inf / NaN weights skipped), then random x against
    FP64 (rel-L2 < 4e-3, the lab's tolerance) and against the incumbent path (< 1e-2, v12's Lt tolerance), and a repeat
    that must be bit-identical. Prepare-time only."""
    dense = state["dense"]
    P = st["proj"][proj]
    w = dense["qkv_gate" if proj == "qkv" else proj]
    n_rows, K, dev = P["rows_t8"], P["K"], w.device
    wr = w[:n_rows]
    fin = torch.isfinite(wr.float()).all(1)
    esc = sorted(set(p1_esc_columns(P["esc"]).tolist())) if P["n_esc"] else []
    esc = esc[::max(1, len(esc) // 48)][:48]
    rnd = torch.randint(0, K, (64,), generator=gen, device=dev).tolist()
    cols = esc + rnd[:16]
    cols += rnd[16:16 + (-len(cols)) % 8]
    chunks = [cols[i:i + T] for i in range(0, len(cols), T)] if T > 1 else [[c] for c in cols[:6] + cols[-2:]]
    bad = flushed = 0
    for cc in chunks:
        x = torch.zeros((T, K), dtype=torch.bfloat16, device=dev)
        ci = torch.tensor(cc, device=dev)
        x[torch.arange(T, device=dev), ci] = 1.0
        got = p1_gemv(st, proj, x).float()
        ref = wr[:, ci].t().float()
        mis = (got != ref) & ~(torch.isnan(got) & torch.isnan(ref)) & fin[None, :]
        sub = mis & (got == 0) & (ref.abs() < 2.0 ** -126)
        flushed += int(sub.sum())
        bad += int((mis & ~sub).sum())
    x = torch.randn((T, K), generator=gen, device=dev).to(torch.bfloat16)
    got = p1_gemv(st, proj, x)
    again = p1_gemv(st, proj, x)
    ref64 = x.double() @ wr.double().t()
    today = _v13_today(proj, x, state, dense)
    torch.cuda.synchronize()
    e64 = float((got.double() - ref64).norm() / ref64.norm())
    etoday = float((got.float() - today.float()).norm() / today.float().norm())
    same = bool(torch.equal(got, again))
    ok = bad == 0 and e64 < 4e-3 and etoday < 1e-2 and same
    return ok, (f"{len(cols)} one-hot cols ({len(esc)} escape) bad {bad} (subnormal flush {flushed}); random x rel vs fp64 "
                f"{e64:.2e} vs today {etoday:.2e}; repeat identical {same}")


def _v13_prepare(state):
    """P1 for the four dense projections (after state["dense"] and the zero buffer exist). state["p1"] stays None (today's
    path everywhere) when P1_ENABLE is off, the build or pack fails, or no route survives its self-test."""
    state["p1"] = None
    print(f"[v13] levers: P1_ENABLE={P1_ENABLE} P1_PROJ={P1_PROJ} K1 route={K1_ROUTE_MIN_BATCH} K2A33_PLAN={K2A33_PLAN}",
          flush=True)
    if not P1_ENABLE:
        return None
    t0 = time.time()
    dense = state["dense"]
    try:
        st = p1_prepare({"qkv": dense["qkv_gate"], "o": dense["o"], "pooler": dense["pooler"], "unpooler": dense["unpooler"]})
        if st is None:
            return None
        gen = torch.Generator(device=dense["qkv_gate"].device).manual_seed(1313)
        for proj in sorted(st["proj"]):
            P = st["proj"][proj]
            if P["mode_t8"] != 0:   # RAW row tiles: this build holds the packed kernel only (p1s_kernel<1, 0>)
                for b in (1, 8):
                    st["route"][(proj, b)] = False
                print(f"[p1] {proj}: RAW row tiles in rows [0, {P['rows_t8']}) (mode {P['mode_t8']}), the layer build has "
                      f"mode 0 only -> today's path", flush=True)
                continue
            for T in (1, 8):
                b = p1_bucket(T)
                if not st["route"].get((proj, b)) or p1_route(st, proj, T) is None:
                    continue
                ok, msg = _v13_p1_selftest(st, state, proj, T, gen)
                print(f"[p1] self-test {proj} T={T}: {msg} -> {'ok' if ok else 'FAIL: route off'}", flush=True)
                if not ok:
                    st["route"][(proj, b)] = False
    except Exception as ex:  # noqa: BLE001
        print(f"[v13] P1 prepare / self-test FAILED -> today's path everywhere: {ex!r}", flush=True)
        return None
    routed = sorted(f"{p}/{b}" for (p, b), on in st["route"].items() if on and p in st["proj"])
    print(f"[v13] P1 routes live {routed} (prepare {time.time() - t0:.1f}s)", flush=True)
    state["p1"] = st if routed else None
    return state["p1"]
'''


def p1_slice():
    """(text, Src): p1_module.py verbatim from its first definition (P1_C) to the end of the file."""
    p1 = Src(SRC_P1)
    first = p1.nodes["P1_C"].lineno
    head = ast.parse("".join(p1.lines[:first - 1]))
    kinds = [type(n).__name__ for n in head.body]
    imports = [n.names[0].name for n in head.body if isinstance(n, ast.Import)]
    check(kinds == ["Expr", "Import", "Import", "Import"] and imports == ["hashlib", "time", "torch"],
          f"p1_module.py preamble is not docstring + imports hashlib, time, torch: {kinds} {imports}")
    for nm in ("P1_CPP_SRC", "P1_CUDA_SRC"):
        check(p1.seg(nm).startswith(f'{nm} = r"""') and '"""' not in p1.value(nm), f"p1 {nm} literal")
    return "".join(p1.lines[first - 1:]), p1


def generate():
    text, _ = G12.generate()
    base12 = text
    check(V12_FILE.read_text() == base12, "gen_k3_v12's output differs from k3_v12.py on disk (regenerate v12 first)")
    p1text, p1 = p1_slice()
    lines = text.split("\n")
    check(lines[1].startswith("# nonce k3-v12-1 "), "k3_v12 nonce")
    hdr = "\n".join(lines[2:5]) + "\n"
    check(hdr.startswith("# GENERATED by tools/gen_k3_v10.py") and hdr.endswith("regenerate.\n"), "k3_v12 generated-by header")
    text = patch(text, lines[1] + "\n" + hdr,
                 f"# nonce {NONCE} (k3_v12 + P1 packed-weight projection GEMVs at T <= 8, K1 L=8 at verify 16 full, K2A v3.3 "
                 "plan 1 at bucket 32; k3_v13_plan.md)\n"
                 "# GENERATED by tools/gen_k3_v13.py on top of gen_k3_v12.py's output (k3_v12.py; its chain is gen_k3_v11 / v10 /\n"
                 "# v09 / v08 ...) and kernels/muse_bf16_proj/p1_module.py (P1).\n"
                 "# Do not hand-edit: change a canonical source or a generator and regenerate.\n", "nonce + header")
    text = patch(text, '"""K3 v0: SGLang\'s decoder layer (baseline.py) with three already-reviewed pieces composed, each behind its own\n',
                 '"""' + V13_DOC + "K3 v0: SGLang's decoder layer (baseline.py) with three already-reviewed pieces composed, each behind its own\n",
                 "docstring")
    # levers A and E
    old = [ln for ln in text.split("\n") if ln.startswith("K1_ROUTE_MIN_BATCH = ")]
    check(len(old) == 1, "K1_ROUTE_MIN_BATCH line")
    text = patch(text, old[0] + "\n", "K1_ROUTE_MIN_BATCH = " + repr(K1_ROUTE)
                 + "  # v13 (A): K1 v7's L=8 (wgmma) at verify 16 and 32 full (v12: 32 only; ab_screen: verify 16 full +1.1%)\n",
                 "K1 route A")
    old = [ln for ln in text.split("\n") if ln.startswith("K2A33_PLAN = ")]
    check(len(old) == 1 and old[0].startswith("K2A33_PLAN = {1: 0, 8: 0, 16: 0, 32: 0, 128: 2, 256: 2}"), "K2A33_PLAN line")
    text = patch(text, old[0] + "\n", "K2A33_PLAN = " + repr(K2A33_PLAN)
                 + "   # v3.3: bs 128/256 with warm experts on wgmma; v13 (E): bucket 32 on variant 1 (NC 2; ab_screen: decode 32 +1.5%)\n",
                 "K2A33_PLAN E")
    # the V13 section, before the rung-1 banner (after v12's section)
    banner = ("# ==============================================================================================\n"
              "# Rung 1 (kernels/muse_decode_layer/r1_glue.py, nonce r1-1): unchanged, and the fallback of both pieces.\n")
    head = ("# ---------------------------------------------------------------- V13: P1 projection kernels\n"
            "P1_ENABLE = True   # master switch (read at call time; False also skips the P1 build in prepare)\n"
            "P1_PROJ = " + repr(P1_PROJ) + "   # per projection (read at call time)\n\n"
            "# kernels/muse_bf16_proj/p1_module.py, verbatim from its first definition to the end of the file (its docstring\n"
            "# and imports are this file's own). The layer builds only p1s_kernel<1, 0> (p1_prepare's default layer_only).\n")
    text = patch(text, banner, head + p1text + V13_GLUE + "\n\n" + banner, "v13 section")
    # call sites
    text = patch(text, '''    qkv = (_v8_gemm(data["attn_input"], "qkv_gate", dense["qkv_gate"], state) if tokens == 1
           else _v12_gemm(data["attn_input"], dense["qkv_gate"], "qkv", tokens))
''', '''    qkv = _v13_p1(state, "qkv", data["attn_input"])   # V13: P1 packed-weight GEMV (T <= 8, routed), else today's path
    if qkv is None:
        qkv = (_v8_gemm(data["attn_input"], "qkv_gate", dense["qkv_gate"], state) if tokens == 1
               else _v12_gemm(data["attn_input"], dense["qkv_gate"], "qkv", tokens))
''', "qkv P1")
    text = patch(text, '''    attn_output = _v12_gemm(o, dense["o"], "o", tokens)  # all-reduce: identity
''', '''    attn_output = _v13_p1(state, "o", o)   # V13: P1 (T = 1), else today's path
    if attn_output is None:
        attn_output = _v12_gemm(o, dense["o"], "o", tokens)  # all-reduce: identity
''', "o P1")
    text = patch(text, '''expert_input = _v8_gemm(ffn_input, "pooler", dense["pooler"], state)''',
                 '''expert_input = _v13_pooler(ffn_input, dense, state)''', "pooler P1 (3 call sites, main stream)", count=3)
    text = patch(text, '''    ffn_output = _v8_unpool_t1(expert_output, state) if tokens == 1 else None
''', '''    ffn_output = _v13_unpool(expert_output, state, tokens)   # V13: P1 into the pre-zeroed buffer (T <= 8, routed), else None
    if ffn_output is None and tokens == 1:
        ffn_output = _v8_unpool_t1(expert_output, state)
''', "unpooler P1")
    text = patch(text, "    _v12_prepare(state)\n", "    _v12_prepare(state)\n    _v13_prepare(state)\n", "prepare v13")
    return text, base12, p1


def check_ab_json(out, path, ref_name, ref_set):
    """A layer_ab2 variants file (ab_v13a.json / ab_v13b.json): every switch it sets exists in k3_v13, dict values keep the default's key set (a layer_ab2
    variant replaces the whole value), and each arm resolves to the intended P1 / A / E setting."""
    def keyfix(v):   # layer_ab2.keyfix
        if isinstance(v, dict) and any(isinstance(k, str) and "," in k for k in v):
            return {tuple(int(p) if p.isdigit() else p for p in k.split(",")): keyfix(x) for k, x in v.items()}
        if isinstance(v, dict):
            return {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): keyfix(x) for k, x in v.items()}
        return v

    arms = json.loads(path.read_text())
    default = {k: out.value(k) for k in ("P1_ENABLE", "P1_PROJ", "K1_ROUTE_MIN_BATCH", "K2A33_PLAN")}
    v12_k1 = {**default["K1_ROUTE_MIN_BATCH"], ("full", 8): 32}
    v12_plan = {**default["K2A33_PLAN"], 32: 0}
    check(arms[0]["name"] == ref_name and keyfix(arms[0]["set"]) == keyfix(ref_set), f"{path.name}: variant 0 must be {ref_name}")
    names, rows = set(), []
    for a in arms:
        check(a["name"] not in names, f"{path.name}: duplicate arm {a['name']}")
        names.add(a["name"])
        eff = dict(default)
        for k, v in keyfix(a.get("set", {})).items():
            check(k in default, f"{path.name} {a['name']}: unknown switch {k}")
            if isinstance(default[k], dict):
                check(isinstance(v, dict) and set(v) == set(default[k]),
                      f"{path.name} {a['name']}: {k} keys {sorted(map(str, v))} != default's {sorted(map(str, default[k]))}")
            eff[k] = v
        p1 = [p for p, on in eff["P1_PROJ"].items() if on] if eff["P1_ENABLE"] else []
        rows.append(f"{a['name']}: P1 {','.join(p1) or 'off'}; A {'on' if eff['K1_ROUTE_MIN_BATCH'] == default['K1_ROUTE_MIN_BATCH'] else 'off' if eff['K1_ROUTE_MIN_BATCH'] == v12_k1 else '??'}; "
                    f"E {'on' if eff['K2A33_PLAN'] == default['K2A33_PLAN'] else 'off' if eff['K2A33_PLAN'] == v12_plan else '??'}")
        check("??" not in rows[-1], f"{path.name} {a['name']}: A / E value is neither the v13 default nor v12's")
    return rows


def verify(text, base12, p1):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    btmp = OUT.with_suffix(".py.gen_base")
    btmp.write_text(base12, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(tmp), doraise=True)
        out, b12 = Src(tmp), Src(btmp)    # Src raises on a duplicate top-level name, i.e. a collision with p1_module's
    finally:
        tmp.unlink()
        btmp.unlink()
    rep = []
    check(text.isascii(), "non-ASCII character in the output")
    lines = text.splitlines()
    check(lines[0] == "# okc: problem=muse_decode_layer", "header line 1")
    check(lines[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1, "nonce")
    check(ast.get_docstring(out.tree).startswith("K3 v13 = k3_v12 + P1"), "docstring")
    check(text.count(K1_SELF_LINE) == 1 and base12.count(K1_SELF_LINE) == 1, "K1 self-test reference line (v7, E4M3 Q at L = 8)")
    # P1: every p1_module node is in the output, ast-identical, the literals byte-identical, the names unprefixed-safe
    for n, node in p1.nodes.items():
        check(n in out.nodes and dump(out.nodes[n]) == dump(node), f"p1 node {n} differs from p1_module.py")
        check(n not in b12.nodes, f"p1 name {n} collides with k3_v12")
        check(n.startswith(("p1_", "P1_", "_P1_")) or n == "_wrap32", f"unexpected p1 name {n}")
    for nm in ("P1_CPP_SRC", "P1_CUDA_SRC"):
        check(out.seg(nm) == p1.seg(nm) and out.value(nm) == p1.value(nm) and '"""' not in out.value(nm), f"{nm} not byte-identical")
    rep.append(f"P1: {len(p1.nodes)} top-level nodes of p1_module.py ast-identical, CUDA source byte-identical "
               f"(sha1 {sha(p1.value('P1_CUDA_SRC'))[:12]}), no name collision")
    check(text.count("= p1_prepare(") == 1 and text.count("p1_build(layer_only=layer_only)") == 1
          and "def p1_prepare(weights, geometry=None, route=None, build=True, layer_only=True):" in text
          and "layer_only" not in V13_GLUE, "P1 build must be the layer-only default (p1_prepare called once, no layer_only argument)")
    check(text.count("-DP1_LAYER_ONLY=") == 1 and 'f"-DP1_LAYER_ONLY={1 if layer_only else 0}"' in text, "layer-only define")
    # the levers
    check(out.value("K1_ROUTE_MIN_BATCH") == K1_ROUTE and out.value("K2A33_PLAN") == K2A33_PLAN, "lever values A / E")
    check(out.value("P1_ENABLE") is True and out.value("P1_PROJ") == P1_PROJ, "P1 switches")
    # nothing else changed
    allowed = {"prepare", "kernel_fn", "K1_ROUTE_MIN_BATCH", "K2A33_PLAN"}
    changed = [n for n, node in b12.nodes.items() if n not in out.nodes or dump(out.nodes[n]) != dump(node)]
    check(set(changed) <= allowed and set(changed) == allowed, f"changed nodes vs k3_v12: {sorted(changed)} (want {sorted(allowed)})")
    added = set(out.nodes) - set(b12.nodes)
    check(added == set(p1.nodes) | V13_NODES, f"added nodes: unexpected {sorted(added - set(p1.nodes) - V13_NODES)}, "
          f"missing {sorted((set(p1.nodes) | V13_NODES) - added)}")
    rep.append(f"vs k3_v12 (gen output == file on disk): {len(b12.nodes) - len(changed)} top-level nodes ast-identical; changed "
               f"{sorted(changed)}; added {len(added)} ({len(p1.nodes)} P1 + {len(V13_NODES)} v13)")
    # call sites: static review facts
    kf = out.nodes["kernel_fn"]
    calls = {}
    for sub in ast.walk(kf):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id.startswith(("_v13_", "p1_")):
            calls.setdefault(sub.func.id, []).append(sub.lineno)
    check({k: len(v) for k, v in calls.items()} == {"_v13_p1": 2, "_v13_pooler": 3, "_v13_unpool": 1},
          f"kernel_fn P1 call sites: {calls}")
    check(text.count('_v8_gemm(ffn_input, "pooler"') == 1 and text.count("= _v8_unpool_t1(expert_output, state)") == 1,
          "the incumbent pooler / unpooler calls must survive exactly once, as the fallbacks")
    for node in ast.walk(kf):   # item 7: nothing from the side stream
        if isinstance(node, ast.With) and any(ast.unparse(it.context_expr.func) == "torch.cuda.stream"
                                              for it in node.items if isinstance(it.context_expr, ast.Call)):
            for sub in ast.walk(node):
                check(not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                           and sub.func.id.startswith(("_v13_", "p1_"))), f"P1 call inside a side-stream block (line {getattr(sub, 'lineno', '?')})")
    rep.append(f"kernel_fn P1 call sites: { {k: len(v) for k, v in calls.items()} } (qkv, o | pooler x3, main stream | unpooler); "
               "none inside a torch.cuda.stream(...) block")
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
               f"{len(text.encode())} bytes")
    v12_set = {"P1_ENABLE": False, "K1_ROUTE_MIN_BATCH": {"sliding,1": 1, "full,1": 1, "full,8": 32},
               "K2A33_PLAN": {"1": 0, "8": 0, "16": 0, "32": 0, "128": 2, "256": 2}}
    for path, ref_name, ref_set in ((AB_JSON, "ref", {"P1_ENABLE": False}), (AB_JSON_B, "v12eq", v12_set)):
        for r in check_ab_json(out, path, ref_name, ref_set):
            rep.append(f"{path.name} arm  {r}")
    return rep


def main(argv):
    text, base12, p1 = generate()
    for r in verify(text, base12, p1):
        print(f"[gen_k3_v13] OK  {r}")
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text(encoding="utf-8") == text, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_v13] --check: up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"[gen_k3_v13] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}, {len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

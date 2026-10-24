"""gen_k3_v12.py: k3_v12.py = k3_v11.py (as gen_k3_v11.py generates it) + v12 layer levers, each A/B'd in the layer.

    python3 -I tools/gen_k3_v12.py            # writes kernels/muse_decode_layer/k3_v12.py, asserts the invariants
    python3 -I tools/gen_k3_v12.py --check    # compares with the file on disk, writes nothing
    python3 -I tools/gen_k3_v12.py --k1v4     # lab only: the same file with K1 v4 kept (v08lab/k3_v12_k1v4.py)

Changes vs k3_v11 (everything else asserted ast-identical to gen_k3_v11's output):
  * K1 v7 (kernels/muse_attention_decode/k1_v7.py) in place of K1 v4, in the same combined build: CUDA source
    byte-identical; _k1_build ast-identical after gen_k3's renames; K1_STAGES_T and the L = 8 self-test reference
    (E4M3 Q, as v7's wgmma path) exactly as k1_v7.py. Routed at L = 8 full for batch >= 32 only (verify 32 full).
  * Unpooler at T <= 8 writes into a persistent pre-zeroed [8, 8192] buffer (columns >= 1024 are never written, so they
    stay zero): no zero-fill kernel per call.
  * cuBLASLt (addmm with a zero bias) for the dense projections, per (projection, bucket) route (V12_LT_ROUTE).
"""

import ast
import importlib.util
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "kernels/muse_decode_layer/k3_v12.py"
SRC_K1V4 = ROOT / "kernels/muse_attention_decode/k1_v4.py"
SRC_K1V7 = ROOT / "kernels/muse_attention_decode/k1_v7.py"
NONCE = "k3-v12-1"
USE_K1V7 = "--k1v4" not in sys.argv

_s11 = importlib.util.spec_from_file_location("gen_k3_v11", ROOT / "tools/gen_k3_v11.py")
G11 = importlib.util.module_from_spec(_s11)
_s11.loader.exec_module(G11)
G10, G9, G8, G = G11.G10, G11.G9, G11.G8, G11.G
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha

K1_ROUTE = {("sliding", 1): 1, ("full", 1): 1, ("full", 8): 32}
LT_ROUTE = {}   # filled from the in-layer A/B: {("qkv" | "o" | "unpool", bucket rep): True}
ZBUF = True

V12_GLUE = '''

def _v12_gemm(x, w, name, tokens):
    """x @ w.T on cuBLASLt (addmm with a zero bias: the GEMM + bias epilogue; the bias adds exact zeros) where
    V12_LT_ROUTE routes (projection, bucket), else k3_v11's _gemm."""
    if V12_LT_ROUTE.get((name, _k2a_bucket_of(tokens))):
        b = _V12_ZERO_BIAS.get((w.shape[0], w.dtype, str(w.device)))
        if b is not None:
            return torch.addmm(b, x, w.t())
    return _gemm(x, w)


_V12_ZERO_BIAS = {}


def _v12_prepare(state):
    """Zero biases for the cuBLASLt route (built here, never under capture) + the persistent zero buffer for the
    T <= 8 unpooler output; checks the Lt route against _gemm on every routed (projection, bucket)."""
    dense = state["dense"]
    dev = dense["qkv_gate"].device
    for w in (dense["qkv_gate"], dense["o"], dense["unpooler"]):
        _V12_ZERO_BIAS[(w.shape[0], w.dtype, str(w.device))] = torch.zeros((w.shape[0],), dtype=w.dtype, device=dev)
    state["v12_zbuf"] = torch.zeros((SLICED_UNPOOL_MAX_TOKENS, HIDDEN), dtype=torch.bfloat16, device=dev) if V12_ZBUF else None
    gen = torch.Generator(device=dev).manual_seed(1212)
    try:
        for (name, rep), on in sorted(V12_LT_ROUTE.items(), key=str):
            if not on:
                continue
            w = {"qkv": dense["qkv_gate"], "o": dense["o"], "unpool": dense["unpooler"]}[name]
            x = torch.randn((rep, w.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            ref, got = _gemm(x, w), _v12_gemm(x, w, name, rep)
            torch.cuda.synchronize()
            err = float((got.float() - ref.float()).norm() / ref.float().norm())
            print(f"[v12] Lt self-test {name} T={rep}: rel {err:.1e}", flush=True)
            if not err < 1e-2:
                raise RuntimeError(f"Lt {name} T={rep}")
    except Exception as ex:  # noqa: BLE001
        print(f"[v12] Lt self-test FAILED -> _gemm everywhere: {ex!r}", flush=True)
        V12_LT_ROUTE.clear()
    return True
'''


def k1v7_stage(text):
    v4, v7 = Src(SRC_K1V4), Src(SRC_K1V7)
    check(v4.value("CPP_SRC") == v7.value("CPP_SRC"), "k1_v7 host API differs from v4 (CPP_SRC)")
    check(v7.seg("CUDA_SRC").startswith('CUDA_SRC = r"""') and '"""' not in v7.value("CUDA_SRC"), "v7 CUDA literal")
    text = patch(text, "K1_" + v4.seg("CUDA_SRC"), "K1_" + v7.seg("CUDA_SRC"), "K1_CUDA_SRC -> v7")
    text = patch(text, G.rename(v4.seg("_build")), G.rename(v7.seg("_build")), "_k1_build -> v7")
    text = patch(text, v4.seg("K1_STAGES_T"), v7.seg("K1_STAGES_T"), "K1_STAGES_T -> v7")
    text = patch(text, "                ref = _k1_ref_attention(d, k_new, v_new, q_e4m3=False)   # v4: fp16 Q at both L\n",
                 "                ref = _k1_ref_attention(d, k_new, v_new, q_e4m3=(L == 8))   # v5: L=8 E4M3 Q (wgmma) + E4M3 P\n",
                 "K1 self-test reference")
    old = [ln for ln in text.split("\n") if ln.startswith("K1_ROUTE_MIN_BATCH = ")]
    check(len(old) == 1, "K1_ROUTE_MIN_BATCH line")
    text = patch(text, old[0] + "\n", "K1_ROUTE_MIN_BATCH = " + repr(K1_ROUTE)
                 + "  # v12: K1 v7's L=8 (wgmma) at verify 32 full only (k1_v7: 1.09x over rung-1 there)\n", "K1 route")
    text = patch(text, "K1=v4 route=", "K1=v7 route=", "switches print")
    text = patch(text, "k1 v4 occupancy", "k1 v7 occupancy", "prepare summary")
    return text


def generate():
    text, _ = G11.generate()
    base11 = text
    lines = text.split("\n")
    check(lines[1].startswith("# nonce k3-v11-1 "), "k3_v11 nonce")
    text = patch(text, lines[1] + "\n", f"# nonce {NONCE} (k3_v11 + K1 v7 at verify 32 full, zero-buffer unpooler, Lt"
                 " routes; k3_v12.md)\n", "nonce")
    if USE_K1V7:
        text = k1v7_stage(text)
    head = ("# ---------------------------------------------------------------- V12 levers\n"
            "V12_LT_ROUTE = " + repr(LT_ROUTE) + "   # (projection, bucket rep) -> cuBLASLt addmm (zero bias)\n"
            "V12_ZBUF = " + repr(ZBUF) + "   # T <= 8 unpooler output in a persistent pre-zeroed buffer\n")
    banner = ("# ==============================================================================================\n"
              "# Rung 1 (kernels/muse_decode_layer/r1_glue.py, nonce r1-1): unchanged, and the fallback of both pieces.\n")
    text = patch(text, banner, head + V12_GLUE + "\n\n" + banner, "v12 section")
    # zero buffer: T == 1 GEMV path and the T <= 8 cuBLAS path
    text = patch(text, '''    out = torch.zeros((1, HIDDEN), dtype=expert_output.dtype, device=expert_output.device)
    mt, rpw, wk, split = route''', '''    zb = state.get("v12_zbuf")
    out = zb[:1] if (V12_ZBUF and zb is not None) else torch.zeros((1, HIDDEN), dtype=expert_output.dtype,
                                                                     device=expert_output.device)
    mt, rpw, wk, split = route''', "unpool t1 zbuf")
    text = patch(text, '''            ffn_output = torch.zeros((tokens, HIDDEN), dtype=expert_output.dtype, device=expert_output.device)
            torch.mm(expert_output, dense["unpooler"][:UNPOOL_SLICE].t(), out=ffn_output[:, :UNPOOL_SLICE])''',
                 '''            zb = state.get("v12_zbuf")
            if V12_ZBUF and zb is not None and expert_output.dtype == zb.dtype:
                ffn_output = zb[:tokens]   # V12: columns >= 1024 are never written, so they stay zero
            else:
                ffn_output = torch.zeros((tokens, HIDDEN), dtype=expert_output.dtype, device=expert_output.device)
            torch.mm(expert_output, dense["unpooler"][:UNPOOL_SLICE].t(), out=ffn_output[:, :UNPOOL_SLICE])''',
                 "unpool t<=8 zbuf")
    # Lt routes
    text = patch(text, '''    attn_output = _gemm(o, dense["o"])  # all-reduce: identity''',
                 '''    attn_output = _v12_gemm(o, dense["o"], "o", tokens)  # all-reduce: identity''', "o Lt")
    text = patch(text, '''        ffn_output = _gemm(expert_output, dense["unpooler"])''',
                 '''        ffn_output = _v12_gemm(expert_output, dense["unpooler"], "unpool", tokens)''', "unpool Lt")
    text = patch(text, '''    qkv = _v8_gemm(data["attn_input"], "qkv_gate", dense["qkv_gate"], state)''',
                 '''    qkv = (_v8_gemm(data["attn_input"], "qkv_gate", dense["qkv_gate"], state) if tokens == 1
           else _v12_gemm(data["attn_input"], dense["qkv_gate"], "qkv", tokens))''', "qkv Lt")
    text = patch(text, '''    state["v10qk"] = _v10_qk_prepare(state)
''', '''    state["v10qk"] = _v10_qk_prepare(state)
    _v12_prepare(state)
''', "prepare v12")
    return text, base11


def verify(text, base11):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text)
    btmp = OUT.with_suffix(".py.gen_base")
    btmp.write_text(base11)
    try:
        py_compile.compile(str(tmp), doraise=True)
        out, b11 = Src(tmp), Src(btmp)
    finally:
        tmp.unlink()
        btmp.unlink()
    rep = []
    check(text.splitlines()[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1, "nonce")
    allowed = {"prepare", "kernel_fn", "_v8_unpool_t1"}
    if USE_K1V7:
        v7 = Src(SRC_K1V7)
        check(out.value("K1_CUDA_SRC") == v7.value("CUDA_SRC"), "K1_CUDA_SRC != k1_v7's")
        check(dump(ast.parse(G.rename(v7.seg("_build"))).body[0]) == dump(out.nodes["_k1_build"]), "_k1_build != v7's")
        allowed |= {"K1_CUDA_SRC", "_k1_build", "K1_STAGES_T", "_k1_self_test", "K1_ROUTE_MIN_BATCH"}
        rep.append(f"K1 v7: CUDA source byte-identical (sha1 {sha(v7.value('CUDA_SRC'))[:12]}), _k1_build ast-identical")
    changed = [n for n, node in b11.nodes.items() if n not in out.nodes or dump(out.nodes[n]) != dump(node)]
    check(set(changed) <= allowed, f"unexpected changes vs k3_v11: {sorted(set(changed) - allowed)}")
    rep.append(f"vs k3_v11 (gen output): {len(b11.nodes) - len(changed)} top-level nodes ast-identical; changed {sorted(changed)}")
    for node in ast.walk(out.tree):
        if isinstance(node, ast.Name):
            check(node.id not in G.BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
    for fn in ("kernel_fn", "_v12_gemm", "_v8_unpool_t1"):
        for sub in ast.walk(out.nodes[fn]):
            check(not (isinstance(sub, ast.Attribute) and sub.attr in G.SYNC_ATTRS), f"{fn} has a host sync")
    bad = G8.G7.undefined_globals(text, str(OUT))
    check(not bad, f"undefined globals: {sorted(bad)}")
    check(len(text.encode()) < 1_000_000, "file >= 1 MB")
    rep.append(f"static ok; {len(text.encode())} bytes")
    return rep


def main(argv):
    text, base11 = generate()
    for r in verify(text, base11):
        print(f"[gen_k3_v12] OK  {r}")
    if "--k1v4" in argv:
        alt = ROOT / "kernels/muse_decode_layer/v08lab/k3_v12_k1v4.py"
        alt.write_text(text.replace(f"# nonce {NONCE} ", f"# nonce {NONCE}-k1v4-lab "))
        print(f"[gen_k3_v12] wrote {alt.relative_to(ROOT)} (lab A/B only)")
        return 0
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text() == text, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_v12] --check: up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text)
    print(f"[gen_k3_v12] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}, {len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

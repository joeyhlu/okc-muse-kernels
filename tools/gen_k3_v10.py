"""gen_k3_v10.py: k3_v10.py = k3_v09.py (as gen_k3_v09.py generates it) + prepare-time cut + v10 layer pieces.

    python3 -I tools/gen_k3_v10.py            # writes kernels/muse_decode_layer/k3_v10.py, asserts the invariants
    python3 -I tools/gen_k3_v10.py --check    # compares with the file on disk, writes nothing

Changes vs k3_v09 (everything else asserted ast-identical to gen_k3_v09's output):
  * ONE load_inline for K1 + K2A v3.1 + R1B + the V8 tail (sources verbatim, concatenated; box lab
    tools/box/build_combo.py: 41-45 s cold vs 44 + 35 + 34 + 35 s separately), built in prepare (no import-time
    nvcc). Each piece takes its functions from the combined module; any build failure falls back to that piece's
    own build exactly as in k3_v09.
  * Boundaries: one SGLang JIT config (cluster 16) for T <= 32 instead of three configs (each config is its own JIT
    module, ~20 s cold); T >= 49 back to the module default. Measured cost of the dropped configs: <= 0.4 us at T = 16.
"""

import ast
import importlib.util
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "kernels/muse_decode_layer/k3_v10.py"
NONCE = "k3-v10-1"

_s9 = importlib.util.spec_from_file_location("gen_k3_v09", ROOT / "tools/gen_k3_v09.py")
G9 = importlib.util.module_from_spec(_s9)
_s9.loader.exec_module(G9)
G8, G = G9.G8, G9.G
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha

BND_ROUTE = {("attn", 1): (16, None), ("attn", 8): (16, None), ("attn", 16): (16, None), ("attn", 32): (16, None),
             ("ffn", 1): (16, None), ("ffn", 8): (16, None), ("ffn", 16): (16, None), ("ffn", 32): (16, None)}

QK = r'''

# ---------------------------------------------------------------- V10: QK-norm + RoPE + FP8 KV store straight into the cache
# SGLang's _qk_rms_norm_fwd_kernel (fused_qk_norm.py, Triton) skips cache row 0 (its CUDA-graph padding slot), which
# is a live row here, so k3_v07..v09 stage the rows and copy them with store_cache. This is the same kernel with only
# that one mask removed (kv_mask = mask): q -> FP8, normed/roped k and raw v -> FP8 rows at cache_loc, one launch.
V10_QK_DIRECT = True
import triton.language as tl  # noqa: E402


@triton.jit
def _v10_qk_fwd_kernel(
    QKV, Q_OUT, K_OUT, POS, COS_SIN, K_CACHE, V_CACHE, CACHE_LOC, eps, stride_qkv, n_rows, kv_capacity,
    NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BLOCK_M: tl.constexpr,
    HAS_ROPE: tl.constexpr, HAS_KV_STORE: tl.constexpr, FP8_KV_STORE: tl.constexpr, USE_PDL: tl.constexpr = False,
    FP8_MAX: tl.constexpr = 448.0,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    cols = tl.arange(0, BD)
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]
    if HAS_KV_STORE:
        HEADS: tl.constexpr = NUM_Q_HEADS + 2 * NUM_KV_HEADS
    else:
        HEADS: tl.constexpr = NUM_Q_HEADS + NUM_KV_HEADS
    token = rows // HEADS
    head = rows % HEADS
    is_q = head < NUM_Q_HEADS
    is_k = (head >= NUM_Q_HEADS) & (head < NUM_Q_HEADS + NUM_KV_HEADS)
    is_v = head >= NUM_Q_HEADS + NUM_KV_HEADS
    src = token * stride_qkv + head * D
    raw = tl.load(QKV + src[:, None] + cols[None, :], mask=mask, other=0.0)
    x = raw.to(tl.float32)
    var = tl.sum(tl.where(mask, x * x, 0.0), axis=1) / D
    rstd = tl.rsqrt(var + eps)
    x = x * rstd[:, None]
    if HAS_ROPE:
        x = x.to(QKV.dtype.element_ty).to(tl.float32)
        partner_cols = cols ^ 1
        xp = tl.load(QKV + src[:, None] + partner_cols[None, :], mask=mask, other=0.0).to(tl.float32)
        xp = (xp * rstd[:, None]).to(QKV.dtype.element_ty).to(tl.float32)
        pos = tl.load(POS + token, mask=row_mask, other=0).to(tl.int64)
        freq = cols // 2
        cs_mask = row_mask[:, None] & (cols < D)[None, :]
        cos = tl.load(COS_SIN + pos[:, None] * D + freq[None, :], mask=cs_mask, other=1.0)
        sin = tl.load(COS_SIN + pos[:, None] * D + D // 2 + freq[None, :], mask=cs_mask, other=0.0)
        is_even = (cols % 2 == 0)[None, :]
        x = tl.where(is_even, x * cos - xp * sin, xp * sin + x * cos)
    if FP8_KV_STORE:
        x = x.to(QKV.dtype.element_ty).to(tl.float32)
    if HAS_KV_STORE:
        x = tl.where(is_v[:, None], raw.to(tl.float32), x)
        if FP8_KV_STORE:
            x = tl.minimum(tl.maximum(x, -FP8_MAX), FP8_MAX)
        loc = tl.load(CACHE_LOC + token, mask=row_mask, other=0).to(tl.int64)
        kv_mask = mask   # V10: every row is live (no slot-0 skip)
        kv_head = tl.where(is_v, head - NUM_Q_HEADS - NUM_KV_HEADS, head - NUM_Q_HEADS)
        kv_dst = loc * (NUM_KV_HEADS * D) + kv_head * D
        tl.store(K_CACHE + kv_dst[:, None] + cols[None, :], x, mask=kv_mask & is_k[:, None])
        tl.store(V_CACHE + kv_dst[:, None] + cols[None, :], x, mask=kv_mask & is_v[:, None])
    q_dst = tl.where(is_q, (token * NUM_Q_HEADS + head) * D, 0)
    k_dst = tl.where(is_k, (token * NUM_KV_HEADS + head - NUM_Q_HEADS) * D, 0)
    tl.store(Q_OUT + q_dst[:, None] + cols[None, :], x, mask=mask & is_q[:, None])
    tl.store(K_OUT + k_dst[:, None] + cols[None, :], x, mask=mask & is_k[:, None])


def _v10_qk_store(qkv, positions, cos_sin, k_rows, v_rows, cache_loc, hq, hkv, d):
    """FusedQKNorm._forward_triton's launch (same grid / block_m / num_warps / PDL), on the real cache rows."""
    tokens = qkv.shape[0]
    q_out = torch.empty((tokens, hq * d), dtype=FP8, device=qkv.device)
    k_out = torch.empty((tokens, hkv * d), dtype=qkv.dtype, device=qkv.device)
    n_rows = tokens * (hq + 2 * hkv)
    block_m = 4 if n_rows < 1024 else 8 if n_rows < 8192 else 16 if n_rows < 65536 else 32
    pdl = {"USE_PDL": True, "launch_pdl": True} if _V10_PDL_OK else {}
    has_rope = positions is not None
    _v10_qk_fwd_kernel[(triton.cdiv(n_rows, block_m),)](
        qkv, q_out, k_out, positions if has_rope else qkv, cos_sin if has_rope else qkv, k_rows, v_rows,
        cache_loc, RMS_EPS, qkv.stride(0), n_rows, k_rows.shape[0], hq, hkv, d, triton.next_power_of_2(d), block_m,
        HAS_ROPE=has_rope, HAS_KV_STORE=True, FP8_KV_STORE=True, num_warps=4, **pdl)
    return q_out


_V10_PDL_OK = False


def _v10_qk_prepare(state):
    """Direct-store QK-norm vs k3_v09's staging path (FusedQKNorm + store_cache) on random caches whose cache_loc
    includes row 0: q (FP8) and every cache byte must be bit-identical. Returns True to enable V10_QK_DIRECT."""
    global _V10_PDL_OK
    if not V10_QK_DIRECT:
        return False
    try:
        from sglang.kernels.jit.utils import is_arch_support_pdl
        _V10_PDL_OK = bool(is_arch_support_pdl())
    except Exception:  # noqa: BLE001
        _V10_PDL_OK = False
    dev = state["router"].device
    gen = torch.Generator(device=dev).manual_seed(1010)
    try:
        for T, rope in ((8, True), (8, False), (128, True), (256, False), (5, True)):
            rows = 4096
            qkv = torch.randn((T, 2304), generator=gen, device=dev).to(torch.bfloat16)
            pos = torch.randint(0, 100000, (T,), generator=gen, device=dev)
            # distinct rows (as every real cache_loc), row 0 included
            loc = torch.cat((torch.zeros(1, dtype=torch.int64, device=dev),
                             torch.randperm(rows - 1, generator=gen, device=dev)[:T - 1].to(torch.int64) + 1))
            kc = (torch.randn((rows, 128), generator=gen, device=dev)).to(FP8)
            vc = (torch.randn((rows, 128), generator=gen, device=dev)).to(FP8)
            kc2, vc2 = kc.clone(), vc.clone()
            rp, cs = (pos, state["cos_sin"]) if rope else (None, None)
            k_stage = torch.empty((T + 1, 128), dtype=FP8, device=dev)
            v_stage = torch.empty((T + 1, 128), dtype=FP8, device=dev)
            sl = torch.arange(1, T + 1, dtype=torch.int64, device=dev)
            q_ref, _ = state["qk_norm"].forward_triton(qkv, rp, cs, (k_stage, v_stage, sl))
            store_cache(k_stage[1:].view(torch.uint8), v_stage[1:].view(torch.uint8), kc.view(torch.uint8),
                        vc.view(torch.uint8), loc, reserved_skip_index=-1)
            q_got = _v10_qk_store(qkv, rp, cs, kc2, vc2, loc, 16, 2, 64)
            torch.cuda.synchronize()
            same = (torch.equal(q_ref.view(torch.uint8), q_got.view(torch.uint8))
                    and torch.equal(kc.view(torch.uint8), kc2.view(torch.uint8))
                    and torch.equal(vc.view(torch.uint8), vc2.view(torch.uint8)))
            print(f"[v10] qk direct self-test T={T} rope={rope}: bit-identical {same}", flush=True)
            if not same:
                raise RuntimeError(f"qk direct mismatch T={T}")
    except Exception as ex:  # noqa: BLE001
        print(f"[v10] qk direct self-test FAILED -> staging path: {ex!r}", flush=True)
        return False
    return True
'''

COMBO = '''
# ---------------------------------------------------------------- V10: one combined CUDA build (prepare-time cut)
V10_COMBINED_BUILD = True    # False = k3_v09's four separate builds (K1 at import)
_V10_MOD = None


def _v10_build_all():
    """K1 + K2A v3.1 + R1B + V8 tail in ONE load_inline: the four sources verbatim, concatenated (distinct
    namespaces and entry points), K2A v3.1's variant cases substituted exactly as _k2a31_build_ext does.
    Returns the module or None (then each piece builds itself as in k3_v09)."""
    global _V10_MOD
    if _V10_MOD is not None:
        return _V10_MOD
    from torch.utils.cpp_extension import load_inline
    cases = "\\n".join("    case %d: launch3<%d, %d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, *v)
                      for i, v in enumerate(K2A31_VARIANTS))
    attrs = "\\n".join("    case %d: return attrs3<%d, %d, %d>();" % (i, *v) for i, v in enumerate(K2A31_VARIANTS))
    k31 = K2A31_KERNEL_SRC.replace("@CASES3@", cases).replace("@ATTRS3@", attrs)
    cpp = "\\n".join((K1_CPP_SRC, K2A31_CPP_SRC, R1B_CPP_SRC, V8_TAIL_CPP_SRC))
    cuda = "\\n".join((K1_CUDA_SRC, k31, R1B_CUDA_SRC, V8_TAIL_CUDA_SRC))
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    try:
        _V10_MOD = load_inline(
            name=f"k3_v10_all_{tag}", cpp_sources=cpp, cuda_sources=cuda,
            functions=["k1_forward", "k1_occupancy", "k1_prof_read", "run3", "attrs3v", "r1b_gemv", "r1b_reduce",
                       "v8_tail", "v8_router32"],
            extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-gencode=arch=compute_90,code=sm_90"],
            verbose=False)
        _K3_BUILD_S["all"] = time.time() - t0
        print(f"[v10] combined build (K1 + K2A v3.1 + R1B + tail) in {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:  # noqa: BLE001
        print("[v10] combined build FAILED -> separate builds. Tail:\\n" + str(ex)[-8000:], flush=True)
        _V10_MOD = None
    return _V10_MOD


def _v10_k2a31_build(name, variants, verbose=False):
    """_k2a31_build_ext's contract, served by the combined module (built with K2A31_VARIANTS)."""
    if _V10_MOD is None or list(variants) != list(K2A31_VARIANTS):
        return _K2A31_OWN_BUILD(name, variants, verbose)
    return _V10_MOD, 0.0
'''


def generate():
    text, _base8 = G9.generate()
    base9 = text
    lines = text.split("\n")
    check(lines[1].startswith("# nonce k3-v09-1 "), "k3_v09 nonce")
    text = patch(text, lines[1] + "\n", f"# nonce {NONCE} (k3_v09 + one combined CUDA build in prepare, one boundary JIT"
                 " config; k3_v10.md)\n", "nonce")
    text = patch(text, "# GENERATED by tools/gen_k3_v09.py on top of gen_k3_v08.py's output",
                 "# GENERATED by tools/gen_k3_v10.py on top of gen_k3_v09.py / gen_k3_v08.py's output", "generated-by")
    # no import-time nvcc when the combined build is on
    text = patch(text, "if USE_K1:\n    _t_build = time.time()\n", "if USE_K1 and not V10_COMBINED_BUILD:\n    _t_build = time.time()\n",
                 "import-time K1 build")
    text = patch(text, "USE_K1 = True ", "V10_COMBINED_BUILD = True    # one combined CUDA build in prepare (k3_v10)\nUSE_K1 = True ",
                 "combined lever early")
    # combined build section before the Rung-1 banner
    banner = ("# ==============================================================================================\n"
              "# Rung 1 (kernels/muse_decode_layer/r1_glue.py, nonce r1-1): unchanged, and the fallback of both pieces.\n")
    text = patch(text, banner, QK + "\n\n" + banner, "qk direct section")
    text = patch(text, banner, COMBO.replace("V10_COMBINED_BUILD = True    # False = k3_v09's four separate builds (K1 at import)\n", "")
                 + "\n\n" + banner, "combined section")
    # prepare: build first; K1 takes the combined module (or builds itself if that failed)
    text = patch(text, '''    state["k1_ok"] = False
    state["k2a"] = None
    if USE_K1:
''', '''    state["k1_ok"] = False
    state["k2a"] = None
    if V10_COMBINED_BUILD:
        global _K1_MOD
        mod = _v10_build_all()
        if USE_K1 and _K1_MOD is None:
            if mod is not None:
                _K1_MOD = mod
            else:
                try:
                    _K1_MOD = _k1_build()
                except Exception as ex:  # noqa: BLE001
                    print(f"[k3] K1 CUDA build FAILED: attention stays on the r1 FA3 path: {ex!r}", flush=True)
    if USE_K1:
''', "prepare build")
    # pieces take the combined module
    text = patch(text, '''        st["mod"] = _r1b_build()
''', '''        st["mod"] = _V10_MOD if _V10_MOD is not None else _r1b_build()
''', "r1b module")
    text = patch(text, '''        mod = _v8_tail_build()
''', '''        mod = _V10_MOD if _V10_MOD is not None else _v8_tail_build()
''', "tail module")
    text = patch(text, '''        st = _k2a31_prepare_st(w, log=lambda s: print(s, flush=True))
''', '''        global _k2a31_build_ext
        _k2a31_build_ext = _v10_k2a31_build   # the combined module (falls back to _K2A31_OWN_BUILD)
        st = _k2a31_prepare_st(w, log=lambda s: print(s, flush=True))
''', "k2a31 module")
    text = patch(text, "\ndef _k2a31_prepare_st(", "\n_K2A31_OWN_BUILD = _k2a31_build_ext\n\n\ndef _k2a31_prepare_st(",
                 "k2a31 own build alias")
    text = patch(text, """    if R1_QK_FUSED_FP8_STAGE and not scaled:
        # FusedQKNorm's Triton kernel""", """    if R1_QK_FUSED_FP8_STAGE and not scaled and V10_QK_DIRECT and state.get("v10qk"):
        # V10: QK-norm (+ RoPE), q -> FP8 and the FP8 K/V rows written straight into the cache at cache_loc.
        q8 = _v10_qk_store(qkv, rope[0], rope[1], k_cache.view(-1, hkv * d), v_cache.view(-1, hkv * d),
                           data["cache_loc"], hq, hkv, d)
        q_fp8 = q8.view(tokens, hq, d)
    elif R1_QK_FUSED_FP8_STAGE and not scaled:
        # FusedQKNorm's Triton kernel""", "r1_attention direct qk")
    text = patch(text, '''    state["v9bnd"] = _v9_bnd_prepare(state)
''', '''    state["v9bnd"] = _v9_bnd_prepare(state)
    state["v10qk"] = _v10_qk_prepare(state)
''', "prepare qk")
    # one boundary JIT config
    old = [ln for ln in text.split("\n") if ln.startswith("V9_BND_ROUTE = ")]
    check(len(old) == 1, "V9_BND_ROUTE line")
    text = patch(text, old[0] + "\n", "V9_BND_ROUTE = " + repr(BND_ROUTE) + "   # v10: one JIT config (cluster 16)\n",
                 "boundary route")
    return text, base9


def verify(text, base9):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text)
    btmp = OUT.with_suffix(".py.gen_base")
    btmp.write_text(base9)
    try:
        py_compile.compile(str(tmp), doraise=True)
        out, b9 = Src(tmp), Src(btmp)
    finally:
        tmp.unlink()
        btmp.unlink()
    rep = []
    check(text.splitlines()[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1, "nonce")
    allowed = {"prepare", "_v8_prepare", "_v8_tail_prepare", "_v9_k2a31_prepare", "V9_BND_ROUTE", "_r1_attention"}
    changed = [n for n, node in b9.nodes.items() if n not in out.nodes or dump(out.nodes[n]) != dump(node)]
    check(set(changed) <= allowed, f"unexpected changes vs k3_v09: {sorted(set(changed) - allowed)}")
    rep.append(f"vs k3_v09 (gen output): {len(b9.nodes) - len(changed)} top-level nodes ast-identical; changed {sorted(changed)}")
    for node in ast.walk(out.tree):
        if isinstance(node, ast.Name):
            check(node.id not in G.BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
    for fn in ("kernel_fn", "_v8_experts", "_v9_bnd", "_v8_tail"):
        for sub in ast.walk(out.nodes[fn]):
            check(not (isinstance(sub, ast.Attribute) and sub.attr in G.SYNC_ATTRS), f"{fn} has a host sync")
    bad = G8.G7.undefined_globals(text, str(OUT))
    check(not bad, f"undefined globals: {sorted(bad)}")
    check(len(text.encode()) < 1_000_000, "file >= 1 MB")
    rep.append(f"static ok; {len(text.encode())} bytes")
    return rep


def main(argv):
    text, base9 = generate()
    for r in verify(text, base9):
        print(f"[gen_k3_v10] OK  {r}")
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text() == text, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_v10] --check: up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text)
    print(f"[gen_k3_v10] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}, {len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

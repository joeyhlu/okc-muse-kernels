"""midr_patches.py: MIDR (notes/study/fresh_eyes_nonmoe.md s2) + its T >= 128 dirty-L2 riders, text patches on a layer file
(k3_sfx_2 = k3_gc_20 or any later gc_* base) -> kernels/lab/<out>.py.

  R6a TAIL_DISC  (wonder_optimization.md Rank 6a) tail_kernel2 issues discard.global.L2 on the FP32 cat output y32 [T, 4608]
                 after its own reads: the norm warps after their bar.sync (each 128 B line read by 4 threads of the 16 norm
                 warps, all consumed before the barrier; thread tx % 4 == 0 discards), the router warp after its top-k
                 (__reduce_max_sync: every lane's la / lb loads consumed; lane % 4 == 0 discards its la and lb lines).
                 Only the T >= 128 torch.mm cat branch of the layer sets it (y32 is a fresh graph-pool tensor whose only
                 reader is _v8_tail; xn / tw / tid / lid are new tensors). Flag = bit 8 of the kernel's `ns` (<= 8), set by
                 the host only for tail_kernel2 (pdl bit 4 from _v8_tail(..., disc=True)).
  MIDR           (R-midr-0b GO: store cost 4.05 / 1.09 us at T 256 / 128) at T in MIDR_TS (128, 256): the attention boundary
                 runs the vendored single-CTA kernel (tools/midr_kernel.cuh = SGLang norm_residual.cuh arithmetic, own source,
                 SGLang's JIT loader / headers) without the FP32 stream store and hands the ffn boundary ("midr", s_in, o_out);
                 the ffn boundary kernel rebuilds mid in registers with the same code and runs SGLang's ffn arithmetic. Prepare
                 checks attention out, ffn out and the final stream bitwise against the module pair, else MIDR stays off.
                 Rider R7a (discard o_out) is skipped: MIDR reads o_out again in the ffn boundary.

  python -I tools/midr_patches.py [--base=kernels/lab/k3_sfx_2.py] [--out=k3_midr_1] [--dry]
"""
import hashlib
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
NL = chr(10)


def _sub(s, old, new, n=1):
    if s.count(old) != n:
        raise SystemExit(f"[midr_patches] anchor count {s.count(old)} != {n}: {old[:70]!r}")
    return s.replace(old, new)


def _sub_after(s, before, old, new):
    if s.count(before) != 1:
        raise SystemExit(f"[midr_patches] anchor count {s.count(before)} != 1: {before[:70]!r}")
    i = s.index(before) + len(before)
    j = s.index(old, i)
    return s[:j] + new + s[j + len(old):]


def patch_tail_disc(t):
    k2 = "__global__ void __launch_bounds__(NT + 32) tail_kernel2(\n"
    t = _sub_after(t, k2, "  __shared__ float red[NT / 32];\n",
                   "  __shared__ float red[NT / 32];\n"
                   "  const bool disc = (ns & 256) != 0;   // R6a TAIL_DISC: y is dead after this kernel's reads\n"
                   "  ns &= 255;\n")
    t = _sub_after(t, k2, "    if (lane == 0) {\n      float sum = 0.f;\n",
                   "    if (disc && (lane & 3) == 0) {   // R6a: la / lb lines, every lane's loads consumed by the top-k\n"
                   "      asm volatile(\"discard.global.L2 [%0], 128;\" ::\"l\"(la + (long)t * l_ld + lane * 8) : \"memory\");\n"
                   "      if (lb != nullptr) asm volatile(\"discard.global.L2 [%0], 128;\" ::\"l\"(lb + (long)t * l_ld + lane * 8) : \"memory\");\n"
                   "    }\n"
                   "    if (lane == 0) {\n      float sum = 0.f;\n")
    bar = '  asm volatile("bar.sync 1, %0;\\n" ::"r"(NT) : "memory");   // the 16 norm warps only\n'
    t = _sub_after(t, k2, bar, bar +
                   "  if (Y32 && disc && (tx & 3) == 0)   // R6a: this 128 B line's 4 readers all passed the barrier\n"
                   "    asm volatile(\"discard.global.L2 [%0], 128;\" ::\"l\"(reinterpret_cast<const float*>(y) + (long)t * y_ld + tx * 8) : \"memory\");\n")
    t = _sub(t, "  const long y_ld = y.stride(0);\n  auto* xp = reinterpret_cast<__nv_bfloat16*>(xn.data_ptr());\n",
             "  const long y_ld = y.stride(0);\n  const int nsd = (int)ns | (((pdl & 6) == 6) ? 256 : 0);   // R6a: discard flag, tail_kernel2 only\n"
             "  auto* xp = reinterpret_cast<__nv_bfloat16*>(xn.data_ptr());\n")
    t = _sub(t, "(long)l_ld, (int)ns,\n", "(long)l_ld, nsd,\n", n=4)
    t = _sub(t, "def _v8_tail(y, logits, state, data):\n", "def _v8_tail(y, logits, state, data, disc=False):\n")
    t = _sub(t, "RMS_EPS, 0.1767766952966369, 1.0, 1e-15, (1 if V9_TAIL_PDL else 0) | (2 if TL1_TAIL2 else 0))\n",
             "RMS_EPS, 0.1767766952966369, 1.0, 1e-15, (1 if V9_TAIL_PDL else 0) | (2 if TL1_TAIL2 else 0)\n"
             "                      | (4 if (disc and TAIL_DISC and logits is None and y.dtype == torch.float32) else 0))\n")
    t = _sub(t, "            y32 = torch.mm(ffn_input, state[\"v8\"][\"wcat\"].t(), out_dtype=torch.float32)\n"
                "        expert_x, topk_weights, topk_ids, local_ids = _v8_tail(y32, None, state, data)\n",
             "            y32 = torch.mm(ffn_input, state[\"v8\"][\"wcat\"].t(), out_dtype=torch.float32)\n"
             "            expert_x, topk_weights, topk_ids, local_ids = _v8_tail(y32, None, state, data, disc=True)\n"
             "        else:\n"
             "            expert_x, topk_weights, topk_ids, local_ids = _v8_tail(y32, None, state, data)\n")
    t = _sub(t, "V9_TAIL_PDL = True\n", "V9_TAIL_PDL = True\nTAIL_DISC = True   # R6a (tools/midr_patches.py): discard the dead FP32 cat output in tail_kernel2\n")
    return t


MIDR_PY = r'''

MIDR_ENABLE = True   # MIDR (tools/midr_patches.py): T >= 128 attention boundary skips the FP32 mid stream; ffn boundary rebuilds it
MIDR_TS = (128, 256)
MIDR_CUH = r"""
@@CUH@@"""


def _midr_w(op):
    op._ensure_cached_weights(torch.bfloat16)
    return (op._cc_post_weight if op._post_norm.has_weight else None, op._cc_pre_weight if op._pre_norm.has_weight else None,
            op._cc_alpha if op._use_residual_gate else None, op._cc_beta if op._use_residual_gate else None,
            bool(op._post_norm_is_input_scale), float(op._post_norm.eps), float(op._pre_norm.eps))


def _midr_prepare(state):
    """JIT-build the vendored MIDR kernels (SGLang's loader + headers, own source) for MIDR_TS's launch configs and check them
    bitwise against the module pair (attention output, ffn output, final stream) on random rows; None = MIDR off."""
    if not MIDR_ENABLE:
        return None
    try:
        import pathlib
        import tempfile
        from sglang.kernels.ops.muse_spark_v12 import norm_residual as nr
        oa, of = state["attn_boundary"], state["ffn_boundary"]
        A, Fw = _midr_w(oa), _midr_w(of)
        src = pathlib.Path(tempfile.gettempdir()) / f"okc_midr_{hashlib.sha1(MIDR_CUH.encode()).hexdigest()[:12]}.cuh"
        if not src.exists():
            src.write_text(MIDR_CUH)
        M = {"a": (A[0], A[1], A[2], A[3], A[5], A[6]), "f": (A[0], A[2], A[3], A[5], Fw[0], Fw[1], Fw[2], Fw[3], Fw[5], Fw[6])}
        dev = state["router"].device
        gen = torch.Generator(device=dev).manual_seed(912)
        for T in MIDR_TS:
            nt = nr._cluster_launch_config(HIDDEN, T, 1)[0]
            args = nr.make_cpp_args(HIDDEN, nt, nr.is_arch_support_pdl(), A[4], A[0] is not None, A[2] is not None,
                                    Fw[4], Fw[0] is not None, Fw[2] is not None)
            M[T] = nr.load_jit("okc_midr", *args, cuda_files=[str(src)],
                               cuda_wrappers=[("runA", f"OkcMidr<{args}>::runA"), ("runF", f"OkcMidr<{args}>::runF")])
            s = torch.randn((T, HIDDEN), generator=gen, device=dev) * 2
            o = torch.randn((T, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
            f = torch.randn((T, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
            r1, rs, _ = oa(s, o, branch_dtype=torch.bfloat16, router_dtype=None)
            r2, rs2, _ = of(rs, f, branch_dtype=torch.bfloat16, router_dtype=None)
            g1, g2, gs = torch.empty_like(o), torch.empty_like(f), torch.empty_like(s)
            M[T].runA(o, s, g1, *M["a"])
            M[T].runF(o, s, f, g2, gs, *M["f"])
            torch.cuda.synchronize()
            if not (torch.equal(g1, r1) and torch.equal(g2, r2) and torch.equal(gs, rs2)):
                print(f"[midr] T={T} not bitwise vs the module pair -> MIDR off", flush=True)
                return None
        print(f"[midr] LIVE at T {MIDR_TS} (bitwise vs the module pair)", flush=True)
        return M
    except Exception as ex:
        print(f"[midr] unavailable -> module boundaries: {ex!r}", flush=True)
        return None


def _midr_bnd(which, stream, branch, router_dtype, state, tokens):
    M = state.get("midr")
    if which == "ffn" and isinstance(stream, tuple):
        _, s_in, b1 = stream
        out, ns = torch.empty_like(branch), torch.empty_like(s_in)
        M[tokens].runF(b1, s_in, branch, out, ns, *M["f"])
        return out, ns, None
    if (which != "attn" or not M or tokens not in M or router_dtype is not None or V9_BND_ROUTE.get(("attn", _k2a_bucket_of(tokens)))
            or stream.dtype != torch.float32 or branch.dtype != torch.bfloat16 or not stream.is_contiguous()
            or not branch.is_contiguous()):
        return None
    out = torch.empty_like(branch)
    M[tokens].runA(branch, stream, out, *M["a"])
    return out, ("midr", stream, branch), None   # stream = what the ffn boundary rebuilds mid from
'''


def patch_midr(t):
    cuh = (ROOT / "tools/midr_kernel.cuh").read_text(encoding="utf-8")
    assert '"""' not in cuh and "\\" not in cuh.replace("\\n", "")
    t = _sub(t, "\n\ndef _v9_bnd(which, stream, branch, router_dtype, state, tokens):\n",
             MIDR_PY.replace("@@CUH@@", cuh) + "\n\ndef _v9_bnd(which, stream, branch, router_dtype, state, tokens):\n"
             "    if MIDR_ENABLE and (isinstance(stream, tuple) or tokens >= MIDR_TS[0]):\n"
             "        r = _midr_bnd(which, stream, branch, router_dtype, state, tokens)\n"
             "        if r is not None:\n"
             "            return r\n")
    t = _sub(t, '    state["v9bnd"] = _v9_bnd_prepare(state)\n',
             '    state["v9bnd"] = _v9_bnd_prepare(state)\n    state["midr"] = _midr_prepare(state)\n')
    return t


def patch(t, midr=True):
    t = patch_tail_disc(t)
    return patch_midr(t) if midr else t


def main():
    a = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
    base = ROOT / a.get("base", "kernels/lab/k3_sfx_2.py")
    outn = a.get("out", "k3_midr_1")
    out = ROOT / f"kernels/lab/{outn}.py"
    dry = "--dry" in sys.argv
    if not dry and out.exists():
        raise SystemExit(f"[midr_patches] {out.name} exists")
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    if not bn.startswith("# nonce "):
        raise SystemExit("[midr_patches] line 2 is not a nonce line")
    t = patch(b)
    t = t.replace(bn, f"# nonce {outn.replace('_', '-')} ({base.stem} + MIDR + R6a tail discard; tools/midr_patches.py)", 1)
    assert t.isascii() and len(t.encode()) < 1_000_000, len(t.encode())
    assert max(len(x) for x in t.split(NL)) <= max(len(x) for x in b.split(NL))
    assert len(re.findall(r"importlib|sys\.modules", t)) == len(re.findall(r"importlib|sys\.modules", b))
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[midr_patches] OK {len(t.encode())} B (base {len(b.encode())} B)")
    if not dry:
        out.write_text(t, encoding="utf-8", newline=NL)
        print(f"[midr_patches] wrote {out.relative_to(ROOT)} sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()

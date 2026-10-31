"""graft_patches.py (R-graft-1): alygg77's Gluon row boundary `_graft_row_boundary` (Apache-2.0, notice kept) in place of SGLang's
stock fused_norm_residual for BOTH layer boundaries (attn: post-attention, ffn: end of layer) at K2A buckets 128 / 256 (T > 48).

  python -I tools/graft_patches.py --base=<in.py> --out=<out.py> [--route=128,256] [--force]
  python -I tools/graft_patches.py --emit-kern=<path.py>      (the kernel block alone, for kernels/lab/gft_lab.py)

Their source: third_party/alygg77-okc-muse-spark-kernels-6cae49a/kernels/muse_decode_layer.py (sha256 pinned), the
`@_glx.jit def _graft_row_boundary` body taken verbatim; their dispatch for T > 32 (`_graft_boundary`): one program per row,
BlockedLayout([4], [32], [NW]), NW = num_warps = 32 up to 192 rows and 16 above, launch_pdl=True. Changes: the module aliases
(_glx / _gl / _gdc_wait / _gdc_launch -> _gft_*), the two eps literals (1.e-8 post, 1.e-5 pre) become constexprs PE / QE fed
from the stock op's own eps, and the weights are the stock op's cached `_cc_*` tensors (exactly what fused_norm_residual reads).
Hunks (each anchor must occur once): the GFT block (notice, guarded gluon import, kernel, `_gft_bnd`, `_gft_prepare`) before
`V9_BND_ROUTE = {`; a 4-line route hook first in `_v9_bnd`; `_gft_prepare(state, t0)` at the end of prepare() (after the hyblo
prepare, behind a 240 s prepare-time guard). Arms: module global GFT_ROUTE (bucket -> bool); {} = byte-for-byte stock dispatch.
Gates: AST parse, ascii, < 1,000,000 B, each decorated def re-parses (lead_gate), the kernel text equals theirs modulo the renames.
"""
import argparse
import ast
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TP = ROOT / "third_party" / "alygg77-okc-muse-spark-kernels-6cae49a" / "kernels" / "muse_decode_layer.py"
PIN = "8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a"   # = tools/hyb_patches.py PIN
LIMIT = 1_000_000
NL = "\n"
RENAMES = (("_glx.jit", "_gft_glx.jit"), ("_gl.", "_gft_gl."), ("_gdc_wait()", "_gft_wait()"), ("_gdc_launch()", "_gft_launch()"),
           ("NW: _gft_gl.constexpr):", "NW: _gft_gl.constexpr,\n                        PE: _gft_gl.constexpr, QE: _gft_gl.constexpr):"),
           ("/ H + 1.e-8)", "/ H + PE)"), ("/ H + 1.e-5)", "/ H + QE)"))


def check(c, msg):
    if not c:
        raise SystemExit(f"[graft_patches] FAIL: {msg}")


def their_kernel():
    b = TP.read_bytes()
    check(hashlib.sha256(b).hexdigest() == PIN, "alygg77 source sha256 differs from the pin")
    lines = b.decode("utf-8").split(NL)
    i = lines.index("def _graft_row_boundary(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,")
    check(lines[i - 1] == "@_glx.jit", "decorator line")
    j = i + 1
    while lines[j].startswith("    ") or lines[j].startswith("                        "):
        j += 1
    src = NL.join(lines[i - 1:j]).rstrip()
    out = src
    for a, b2 in RENAMES:
        check(a in out, f"rename anchor {a!r}")
        out = out.replace(a, b2)
    for k in ("1.e-8", "1.e-5", "_gl.load", "_gdc_"):
        check(k not in out.replace("_gft_gl.load", ""), f"leftover {k!r}")
    return src, out


NOTICE = '''"""[GFT] Begin of code derived from alygg77/okc-muse-spark-kernels@6cae49a kernels/muse_decode_layer.py (Apache-2.0;
built on vjeux/okc-2026@f35a41e, Apache-2.0), modified by team joeyhlu (2026-10-31): see THIRD_PARTY_NOTICES."""
_GFT_NOTICE = r"""
==== alygg77/okc-muse-spark-kernels (Apache-2.0) -> the GFT row boundary (search: _graft_row_boundary, _gft_) ====
Source: https://github.com/alygg77/okc-muse-spark-kernels commit 6cae49a639ee6d14b929a5fd4a63c627f96d7dc0,
kernels/muse_decode_layer.py: _graft_row_boundary (their Gluon row boundary, router copy removed) and its T > 32 launch shape
(one program per row; 32 warps up to 192 rows, 16 above; PDL launch). Licensed under the Apache License, Version 2.0; that
repository's NOTICE and the license text are reproduced in the HYB section of THIRD_PARTY_NOTICES. Modified by team joeyhlu:
the two norm eps are constexpr parameters (theirs: 1e-8 / 1e-5 literals), module aliases renamed (_gft_*), the weights are
SGLang's own cached boundary weights; the route, prepare self-test and stock fallback are ours.
"""
THIRD_PARTY_NOTICES += _GFT_NOTICE
'''

GUARD = '''GFT_ROUTE = {@ROUTE@}   # K2A bucket -> alygg77's row boundary for both layer boundaries (R-graft-1)
GFT_PREP_GUARD_S = 240.0
try:
    from triton.experimental import gluon as _gft_glx
    from triton.experimental.gluon import language as _gft_gl
    from triton.language.extra.cuda import gdc_wait as _gft_wait, gdc_launch_dependents as _gft_launch
    _GFT_IMPORT = None
except Exception as _gft_ex:
    _GFT_IMPORT = repr(_gft_ex)
    _gft_glx = types.SimpleNamespace(jit=lambda f: f)
    _gft_gl = types.SimpleNamespace(constexpr=None)
    _gft_wait = _gft_launch = None
'''

GLUE = '''
def _gft_bnd(which, stream, branch, state, tokens):
    """alygg77's T > 32 row boundary on the stock op's cached weights: (out bf16, new_stream fp32, None) or None (stock)."""
    g = state.get("gft")
    e = g.get(which) if g else None
    if (e is None or branch.dtype != torch.bfloat16 or stream.dtype != torch.float32 or tuple(branch.shape) != (tokens, HIDDEN)
            or tuple(stream.shape) != (tokens, HIDDEN) or not branch.is_contiguous() or not stream.is_contiguous()
            or branch.data_ptr() % 16 or stream.data_ptr() % 16):
        return None
    h_out = torch.empty_like(branch)
    s_out = torch.empty_like(stream)
    nw = 16 if tokens > 192 else 32
    try:
        _graft_row_boundary[(tokens,)](branch, stream, e[0], e[1], e[2], e[3], h_out, s_out, HIDDEN, HIDDEN, e[4], 4, nw, e[5], e[6],
                                       num_warps=nw, launch_pdl=True)
    except Exception as ex:
        print(f"[gft] {which} T={tokens} launch failed -> stock boundaries: {ex!r:.200}", flush=True)
        state["gft"] = None
        return None
    return h_out, s_out, None
def _gft_prepare(state, t0):
    state["gft"] = None
    if not GFT_ROUTE or not any(GFT_ROUTE.values()):
        return
    if _GFT_IMPORT is not None:
        print(f"[gft] gluon unavailable -> stock boundaries: {_GFT_IMPORT}", flush=True)
        return
    if time.time() - t0 > GFT_PREP_GUARD_S:
        print(f"[gft] prepare at {time.time() - t0:.0f}s > {GFT_PREP_GUARD_S:.0f}s guard -> stock boundaries", flush=True)
        return
    tb = time.time()
    try:
        g = {}
        for which, name in (("attn", "attn_boundary"), ("ffn", "ffn_boundary")):
            op = state[name]
            op._ensure_cached_weights(torch.bfloat16)
            post = bool(op._post_norm.has_weight)
            if post and not op._post_norm_is_input_scale:
                raise RuntimeError(f"{which}: post weight is not an input scale")
            if not op._pre_norm.has_weight or not op._use_residual_gate:
                raise RuntimeError(f"{which}: no pre weight / residual gate")
            pw, qw, al, bt = (op._cc_post_weight if post else op._cc_pre_weight), op._cc_pre_weight, op._cc_alpha, op._cc_beta
            for t, dt in ((pw, torch.bfloat16), (qw, torch.bfloat16), (al, torch.float32), (bt, torch.float32)):
                if t.dtype != dt or tuple(t.shape) != (HIDDEN,) or not t.is_contiguous() or t.data_ptr() % 16:
                    raise RuntimeError(f"{which}: weight {tuple(t.shape)} {t.dtype}")
            g[which] = (pw, qw, al, bt, post, float(op._post_norm.eps), float(op._pre_norm.eps))
        state["gft"] = g
        dev = state["router"].device
        gen = torch.Generator(device=dev).manual_seed(4711)
        for which, name in (("attn", "attn_boundary"), ("ffn", "ffn_boundary")):
            op = state[name]
            for T in (49, 128, 192, 193, 256):
                stream = torch.randn((T, HIDDEN), generator=gen, device=dev) * 2
                branch = (torch.randn((T, HIDDEN), generator=gen, device=dev) * 3).to(torch.bfloat16)
                ref = op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=None)
                got = _gft_bnd(which, stream, branch, state, T)
                torch.cuda.synchronize()
                if got is None:
                    raise RuntimeError(f"{which} T={T}: not taken")
                e0 = float((got[0].float() - ref[0].float()).norm() / ref[0].float().norm())
                e1 = float((got[1] - ref[1]).norm() / ref[1].norm())
                nd = float((got[0] != ref[0]).float().mean())
                ok = e0 < 1e-2 and e1 < 1e-4 and bool(torch.isfinite(got[1]).all()) and bool(torch.isfinite(got[0].float()).all())
                print(f"[gft] self-test {which} T={T} post {g[which][4]} eps {g[which][5]:.0e}/{g[which][6]:.0e}: out rel {e0:.1e}"
                      f" (bf16 differ {nd:.1e}) stream rel {e1:.1e} -> {'ok' if ok else 'FAIL'}", flush=True)
                if not ok:
                    raise RuntimeError(f"{which} T={T}")
    except Exception as ex:
        state["gft"] = None
        print(f"[gft] prepare FAILED -> stock boundaries: {ex!r:.300}", flush=True)
        return
    print(f"[gft] row boundary live at buckets {sorted(k for k, v in GFT_ROUTE.items() if v)} ({time.time() - tb:.1f}s)", flush=True)
'''

END = '"""[GFT] End of code derived from alygg77/okc-muse-spark-kernels (see _GFT_NOTICE)."""\n'

HOOK_OLD = "def _v9_bnd(which, stream, branch, router_dtype, state, tokens):\n"
HOOK_NEW = HOOK_OLD + ("    if router_dtype is None and GFT_ROUTE.get(_k2a_bucket_of(tokens)):\n"
                       "        gr = _gft_bnd(which, stream, branch, state, tokens)\n"
                       "        if gr is not None:\n"
                       "            return gr\n")
PREP_OLD = "    builds = \" \".join(f\"{k}={v:.0f}s\" for k, v in _K3_BUILD_S.items()) or \"none\"\n"
PREP_NEW = "    _gft_prepare(state, t0)\n" + PREP_OLD
BLOCK_AT = "V9_BND_ROUTE = {"


def block(route):
    _, kern = their_kernel()
    r = ", ".join(f"{b}: True" for b in route)
    return NOTICE + GUARD.replace("@ROUTE@", r) + kern + NL + GLUE.lstrip(NL) + END


def apply(t, route):
    for a, n in ((HOOK_OLD, 1), (PREP_OLD, 1), (NL + BLOCK_AT, 1), ("\nimport types\n", 1), ("\nimport time\n", 1)):
        check(t.count(a) == n, f"anchor {a[:50]!r}: {t.count(a)} != {n}")
    check("_gft_" not in t, "base already carries GFT")
    t = t.replace(HOOK_OLD, HOOK_NEW, 1).replace(PREP_OLD, PREP_NEW, 1)
    k = t.index(NL + BLOCK_AT) + 1
    return t[:k] + block(route) + t[k:]


def gate(t, base):
    tree = ast.parse(t)
    check(t.isascii(), "ascii")
    n = len(t.encode())
    check(n < LIMIT, f"size {n:,d} B >= {LIMIT:,d}")
    defs = {x.name for x in tree.body if isinstance(x, ast.FunctionDef)}
    check({"_graft_row_boundary", "_gft_bnd", "_gft_prepare"} <= defs, "defs missing")
    src, _ = their_kernel()
    print(f"[graft_patches] gates: ast ok, ascii ok, {n:,d} B (+{n - len(base.encode()):,d}; under the cap by {LIMIT - n:,d}); "
          f"their kernel {len(src)} chars pinned; decorated defs checked by lead_gate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--out")
    ap.add_argument("--route", default="128,256")
    ap.add_argument("--emit-kern")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    route = [int(x) for x in a.route.split(",") if x]
    if a.emit_kern:
        _, kern = their_kernel()
        txt = ("# R-graft-1 lab copy of the GFT kernel (generated by tools/graft_patches.py --emit-kern; alygg77, Apache-2.0)\n"
               "import types\n" + GUARD.replace("@ROUTE@", "") + kern + NL)
        pathlib.Path(a.emit_kern).write_text(txt, encoding="utf-8", newline=NL)
        print(f"[graft_patches] kernel block -> {a.emit_kern} ({len(txt)} B)")
        return
    src, dst = ROOT / a.base, ROOT / a.out
    check(a.force or not dst.exists(), f"{dst} exists")
    base = src.read_bytes().decode("utf-8")
    t = apply(base, route)
    gate(t, base)
    dst.write_bytes(t.encode("utf-8"))
    print(f"[graft_patches] {a.base} -> {a.out}: route {route}")


if __name__ == "__main__":
    main()

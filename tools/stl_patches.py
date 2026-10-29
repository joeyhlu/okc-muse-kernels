"""stl_patches.py (R-stl-0 arm b): tail_kernel / tail_kernel2 (V8_TAIL_CUDA_SRC) read their predecessor's data (y, the logit
partials la / lb) through const __restrict__ pointers -> LDG.CONSTANT (.nc), which is undefined for data written while the
kernel's lifetime overlaps its producer (PDL) and lets ptxas hoist the loads above griddepcontrol.wait. This patch loads them
with __ldcg (ld.global.cg, coherent, ordered after the wait). Weights / graph inputs (nw, bias, lstart) are unchanged.

  python -I tools/stl_patches.py --n=1 [--base=k3_gc_20]   -> kernels/lab/k3_stl_<n>.py
"""
import pathlib
import py_compile
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)

REPL = [
    ("qa[q][0] = pq[0];", "qa[q][0] = __ldcg(pq);"),
    ("qa[q][1] = pq[1];", "qa[q][1] = __ldcg(pq + 1);"),
    ("qb0 = pb[0];", "qb0 = __ldcg(pb);"),
    ("qb1 = pb[1];", "qb1 = __ldcg(pb + 1);"),
    ("const float4 a = p[0], c = p[1];", "const float4 a = __ldcg(p), c = __ldcg(p + 1);"),
    ("const uint4 q = *(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(y) + (long)t * y_ld) + tx);",
     "const uint4 q = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(y) + (long)t * y_ld) + tx);"),
]


def main():
    n = ARGS["n"]
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_20')}.py"
    out = ROOT / f"kernels/lab/k3_stl_{n}.py"
    if out.exists():
        sys.exit(f"{out.name} exists")
    t = base.read_text(encoding="utf-8")
    a0 = t.index('V8_TAIL_CUDA_SRC = r"""')
    a1 = t.index('"""', a0 + 30)
    c = t[a0:a1]
    for old, new in REPL:
        k = c.count(old)
        if k != 2:
            sys.exit(f"{old[:40]}: {k} != 2 (tail_kernel + tail_kernel2)")
        c = c.replace(old, new)
    t = t[:a0] + c + t[a1:]
    bn = t.split(NL)[1]
    t = t.replace(bn, f"# nonce k3-stl-{n} ({base.stem} + tail_kernel / tail_kernel2 predecessor loads (y, la, lb) via __ldcg; "
                      "tools/stl_patches.py)", 1)
    if not t.isascii() or len(t.encode()) >= 1_000_000:
        sys.exit("ascii / size")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[stl_patches] wrote {out.relative_to(ROOT).as_posix()} ({len(t.encode())} B)")


if __name__ == "__main__":
    main()

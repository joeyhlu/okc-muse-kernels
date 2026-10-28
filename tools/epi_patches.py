"""epi_patches.py: EPI = attention-epilogue micro-changes from the 2026-10-28 external review, verified in code (VFA builder):
  * KS1 / KS3 finalize (vsl_fin1 / vsl_fin3 and their two inlined copies): the 4 (DPT) general IEEE divides `a[i] / dd` by the same
    denominator become one approximate reciprocal `__fdividef(1.f, dd)` (MUFU.RCP) and multiplies (bf16 rounding `bfr` follows);
    their gate sigmoids `1.f / (1.f + __expf(-g))` -> `__fdividef(1.f, ...)` (rcp.approx instead of the IEEE rcp.rn);
  * k1::sigmoidf_ (K1 merge, VA1 finalize): `1.f / (1.f + expf(-g))` -> `__fdividef(1.f, 1.f + __expf(-g))`.
  Routing sigmoids (expf(-lg ...)) are untouched: top-k ties must stay bitwise (parent, 2026-10-28).
Rejected after verification: VFA's "duplicate post-wait fence block" is empty inline asm (zero SASS), and its per-tile causal-mask
test is one uniform compare per 64-key tile (~1 % of a tile) - no variant built.

    python -I tools/epi_patches.py --base=k3_gc_4 --n=1   -> kernels/muse_decode_layer/k3_epi_<n>.py + kernels/lab/epi_k3_epi_<n>_ab.py
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
NL = chr(10)

EDITS = (
    # attention gate sigmoid: approximate reciprocal (MUFU.RCP) + fast exp; the routing sigmoids (expf(-lg ...)) stay bitwise
    ("__device__ __forceinline__ float sigmoidf_(float g) { return 1.f / (1.f + expf(-g)); }",
     "__device__ __forceinline__ float sigmoidf_(float g) { return __fdividef(1.f, 1.f + __expf(-g)); }", 1),
    ("(1.f / (1.f + __expf(-", "(__fdividef(1.f, 1.f + __expf(-", 12),
    # KS1 / KS3 normalisation: one approximate reciprocal of the row denominator, then multiplies (bfr's bf16 rounding follows)
    ("const float dd = (den > 0.f) ? den : 1.f;", "const float dd = (den > 0.f) ? den : 1.f; const float rd = __fdividef(1.f, dd);", 4),
    ("y[i] = bfr(a[i] / dd); ss = fmaf(y[i], y[i], ss); }", "y[i] = bfr(a[i] * rd); ss = fmaf(y[i], y[i], ss); }", 2),
    ("const float y0 = bfr(a.x / dd), y1 = bfr(a.y / dd), y2 = bfr(a.z / dd), y3 = bfr(a.w / dd);",
     "const float y0 = bfr(a.x * rd), y1 = bfr(a.y * rd), y2 = bfr(a.z * rd), y3 = bfr(a.w * rd);", 2),
)



def apply(text):
    for old, new, n in EDITS:
        assert text.count(old) == n, (text.count(old), n, old[:80])
        text = text.replace(old, new)
    return text


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM  # noqa: E402
    sys.argv = _argv
    n = args.get("n", "1")
    base = ROOT / f"kernels/muse_decode_layer/{args.get('base', 'k3_gc_4')}.py"
    out = ROOT / f"kernels/muse_decode_layer/k3_epi_{n}.py"
    lab = ROOT / f"kernels/lab/epi_k3_epi_{n}_ab.py"
    tag = f"k3-epi-{n}"
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    try:
        _, chk_base, _ = GM.G37.generate()
    except (SystemExit, Exception):   # noqa: BLE001
        chk_base = b
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{base.name} is not a finish() fixed point")
    nonce = f"# nonce {tag} ({base.stem} + EPI: attention epilogues with approximate reciprocals (__fdividef) and __expf: KS1 / KS3 normalisation + gate, k1::sigmoidf_)"
    rep = [f"base {base.name} {len(b.encode())} B is a finish() fixed point"]
    s = GM.finish(apply(b.replace(bn, nonce)), nonce, tag, chk_base, rep)
    for r in rep:
        print(f"[epi_patches] OK  {r}")
    out.write_text(s, encoding="utf-8", newline=NL)
    lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[epi_patches] wrote {out.relative_to(ROOT)} and {lab.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

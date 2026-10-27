"""gen_t1.py: the T1 files on top of k3_m32_3 (tools/gen_k3_m32_3.py chain: k3_m32_2 pre-strip text + apply_sk1):
  * kernels/lab/k3_t1m_ab.py      = k3_m32_3 + T1 fix (tools/t1_patches.py, T1FIX_ENABLE) + per-CTA marks
                                     (tools/t1_marks_patches.py). LAB ONLY (box runs, tools/box/t1_run.py).
  * kernels/lab/k3_t1p_ab.py      = k3_m32_3 + T1P0 (W2 L2 prefetch by vid 0's K1, T1_W2PF) + marks. LAB ONLY.
  (k3_m32_4 = k3_m32_3 + T1P0 was dropped after R-t1-2: T1P0 killed.)
Both go through gen_k3_m32.finish (strip_layer / strip_cxx equivalence, compiles, ASCII, nonce, checker scan, < 1 MB).

    python -I tools/gen_t1.py
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32 as GM  # noqa: E402
sys.argv = _argv
import gd_patches  # noqa: E402
import m32_patches  # noqa: E402
import sk1_patches  # noqa: E402
import t1_marks_patches  # noqa: E402
import t1_patches  # noqa: E402

OUTS = (   # (tag, nonce, T1 patch, marks, paths). R-t1-1 killed the vid 7 fix (k3_t1m_ab.py); R-t1-2 tests T1P0.
    ("k3-t1m", "# nonce k3-t1m (LAB: k3_m32_3 + T1 fix vid 7 at buckets 1 / 8 (T1FIX_ENABLE) + per-CTA marks in xw2's tail)",
     "t1", True, [ROOT / "kernels/lab/k3_t1m_ab.py"]),
    ("k3-t1p", "# nonce k3-t1p (LAB: k3_m32_3 + T1P0 W2 L2 prefetch from vid 0's K1 (T1_W2PF) + per-CTA marks in xw2's tail)",
     "t1p0", True, [ROOT / "kernels/lab/k3_t1p_ab.py"]),
    # k3_m32_4 (k3_m32_3 + T1P0) was dropped: R-t1-2 measured T1P0 +3..+10 us at T=8 (killed).
)


def main():
    v39, base, _ = GM.G37.generate()
    GM.check(GM.G37.V39 and v39 == GM.GD.V39.read_text(encoding="utf-8"), "gen_k3_v37 --v39 does not reproduce k3_v39.py")
    gd = gd_patches.apply(v39).replace(GM.G37.NONCE, GM.GD.NONCE)
    GM.check(GM.strip_layer.strip(gd) == GM.GD1.read_text(encoding="utf-8"), "the gd chain does not reproduce k3_gd_1.py")
    path2, tag2, nonce2, px2 = GM.OUTS[2]
    t2 = m32_patches.apply(gd, px=px2).replace(GM.GD.NONCE, nonce2)
    m33 = ROOT / "kernels/muse_decode_layer/k3_m32_3.py"
    import gen_k3_m32_3 as G3
    s3 = GM.finish(sk1_patches.apply_sk1(t2.replace(nonce2, G3.NONCE)), G3.NONCE, G3.TAG, base, [])
    GM.check(s3 == m33.read_text(encoding="utf-8"), "the chain does not reproduce k3_m32_3.py on disk")
    rep = ["k3_m32_3 reproduced byte-identically"]
    for tag, nonce, which, marks, paths in OUTS:
        t = sk1_patches.apply_sk1(t2.replace(nonce2, nonce))
        t = t1_patches.apply_t1(t) if which == "t1" else t1_patches.apply_t1p0(t)
        if marks:
            t = t1_marks_patches.apply_marks(t)
        s = GM.finish(t, nonce, tag, base, rep)
        for p in paths:
            p.write_text(s, encoding="utf-8", newline="\n")
        rep.append(f"wrote {', '.join(str(p.relative_to(ROOT)) for p in paths)} ({len(s.encode())} B, sha1 "
                   f"{hashlib.sha1(s.encode()).hexdigest()[:12]})")
    for r in rep:
        print(f"[gen_t1] OK  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

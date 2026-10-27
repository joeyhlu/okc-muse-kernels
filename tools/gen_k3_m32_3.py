"""gen_k3_m32_3.py: kernels/muse_decode_layer/k3_m32_3.py = k3_m32_2 + SK1 (K1-W dropped after R-sk-3; --k1w adds it back).

Chain: tools/gen_k3_m32.py's k3_m32_2 pre-strip text (GD chain asserted to reproduce k3_gd_1, m32_patches.apply(px=True);
its strip is asserted equal to k3_m32_2.py on disk), the new nonce, tools/at_k1w_patches.py apply_k1w (attention L = 1
whole-row loader, bitwise equal), tools/k1w_switch.py (K1-W as k1_main<L, ST, KW> + K1W_ENABLE, so one file serves both A/B
arms), tools/sk1_patches.py apply_sk1 (vid 6 = vid 2 + route_fr + 8-token hot K2 finalize, SK1_ENABLE, self-tested), then
gen_k3_m32.finish (strip_layer + strip_cxx with AST-equivalence checks, compiles, ASCII, line 1, nonce, no duplicate names,
judge-checker scan <= base, < 1 MB). Writes the file and a byte-identical box copy kernels/lab/k3_m32_3_ab.py.

    python -I tools/gen_k3_m32_3.py [--check] [--k1w]
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
CHECK = "--check" in sys.argv
K1W = "--k1w" in sys.argv   # R-sk-3: K1-W lost decode16 / 32 (-0.18 / -0.24 %) in the layer: off by default
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32 as GM  # noqa: E402
sys.argv = _argv
import at_k1w_patches  # noqa: E402
import gd_patches  # noqa: E402
import k1w_switch  # noqa: E402
import m32_patches  # noqa: E402
import sk1_patches  # noqa: E402

OUT = ROOT / "kernels/muse_decode_layer/k3_m32_3.py"
LAB = ROOT / "kernels/lab/k3_m32_3_ab.py"
TAG = "k3-m32-3"
NONCE = ("# nonce k3-m32-3 (k3_m32_2 + SK1 at bucket 256: vid 6 = vid 2 + one-batch route + 8-token hot K2 finalize "
         "(SK1_ENABLE), self-tested with fallback)")
NONCE_K1W = ("# nonce k3-m32-3 (k3_m32_2 + K1-W whole-row L = 1 K1 loader (K1W_ENABLE) + SK1 at bucket 256: vid 6 = vid 2 + "
             "one-batch route + 8-token hot K2 finalize (SK1_ENABLE), self-tested with fallback)")


def main():
    rep = []
    v39, base, rep0 = GM.G37.generate()
    GM.check(GM.G37.V39 and v39 == GM.GD.V39.read_text(encoding="utf-8"), "gen_k3_v37 --v39 does not reproduce k3_v39.py")
    gd = gd_patches.apply(v39)
    GM.check(gd.count(GM.G37.NONCE) == 1, "k3_v39 nonce line")
    gd = gd.replace(GM.G37.NONCE, GM.GD.NONCE)
    GM.check(GM.strip_layer.strip(gd) == GM.GD1.read_text(encoding="utf-8"), "the gd chain does not reproduce k3_gd_1.py")
    path2, tag2, nonce2, px2 = GM.OUTS[2]
    t2 = m32_patches.apply(gd, px=px2)
    GM.check(t2.count(GM.GD.NONCE) == 1, "gd nonce line")
    t2 = t2.replace(GM.GD.NONCE, nonce2)
    rep2 = []
    s2 = GM.finish(t2, nonce2, tag2, base, rep2)
    disk = path2.read_text(encoding="utf-8")
    # today's strip_cxx keeps a few more // comment lines than the one that wrote k3_m32_2.py: compare without them
    nocom = lambda s: [ln for ln in s.split("\n") if not ln.strip().startswith("//")]  # noqa: E731
    GM.check(s2 == disk or nocom(s2) == nocom(disk), "the m32_2 chain does not reproduce k3_m32_2.py on disk (beyond // lines)")
    rep.append(f"k3_m32_2 reproduced ({'byte-identical' if s2 == disk else 'identical up to // comment lines'}: "
               f"{len(disk.encode())} B on disk, {len(s2.encode())} B now); pre-strip {len(t2.encode())} B")
    nonce = NONCE_K1W if K1W else NONCE
    t = t2.replace(nonce2, nonce)
    if K1W:
        t = k1w_switch.k1w_switch(at_k1w_patches.apply_k1w(t))
    t = sk1_patches.apply_sk1(t)
    rep.append(f"{'K1-W + switch + ' if K1W else ''}SK1 applied: pre-strip {len(t.encode())} B")
    s = GM.finish(t, nonce, TAG, base, rep)
    for r in rep:
        print(f"[gen_k3_m32_3] OK  {r}")
    if CHECK:
        GM.check(OUT.exists() and OUT.read_text(encoding="utf-8") == s, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_m32_3] --check: {OUT.name} up to date")
        return 0
    OUT.write_text(s, encoding="utf-8", newline="\n")
    LAB.write_text(s, encoding="utf-8", newline="\n")
    print(f"[gen_k3_m32_3] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""gen_k3_v37.py: kernels/muse_decode_layer/k3_v37.py = k3_v36 + RS1 + SBB1 [+ FIN1], every rider switchable at call time.

Composition (unstripped chain, then tools/strip_layer.py; each step asserts its anchors):
  1. k3_v36's chain (tools/gen_k3_v36.py): k3_v34 -> su1 + su1b + x_4 config -> KW1 -> PR206 -> PT1 -> TIE1 -> TL1 -> K3P.
  2. --fin1: study-a's tools/fin1_patches.py (vid 3 handed-item finalize on all 9 warps; probe bit 6 = FIN1_ENABLE, prepare
     bitwise self-test). Only if R-study-a-6 adopts it. Must precede RS1 (both edit the _k2a33 / _k2a36 launch lines).
  3. tools/rs1_patches.py: warp-parallel K2A setup (probe bit 7 = RS1_ENABLE).
  4. tools/sbb1_patches.py (study-b): PDL-aware attention residual boundary at T <= 32 (SBB1_ENABLE / SBB1_ROUTE; PT1 job).
Then the nonce, the strip and gen_k3_v35's checks (compiles, ASCII, line 1, nonce, no duplicate names, checker scan <= base,
< 1 MB). Writes kernels/muse_decode_layer/k3_v37.py and the byte-identical lab copy kernels/lab/k3_v37_ab.py.

    python -I tools/gen_k3_v37.py [--fin1] [--check]
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
V39 = "--v39" in sys.argv   # k3_v39 = k3_v37 + SU1 before PR206 in prepare (R15 (e), parent 2026-10-27)
V38 = "--v38" in sys.argv   # k3_v38 = the R12 binary (FIN1 code present) with SS3 on and FIN1 off by default
FIN1 = "--fin1" in sys.argv or V38
SS3 = "--ss3" in sys.argv or V38   # lab: + tools/ss3_patches.py (ss in K3, 4-CTA clusters; R12 A/B file)
CHECK = "--check" in sys.argv
FR1 = "--fr1" in sys.argv   # lab: + tools/fr1_patches.py (R15: fresh-review levers as call-time switches)
PM8 = "--pm8" in sys.argv   # lab: + tools/pm8_patches.py (persistent T <= 8 MoE; component / in-layer A/B file)
_argv, sys.argv = sys.argv, [sys.argv[0]]   # gen_k3_v35 reads its flags at import; library use only
import gen_k3_v35 as G  # noqa: E402
sys.argv = _argv
import fa206_patches  # noqa: E402
import k3p_patches  # noqa: E402
import kw1_patches  # noqa: E402
import fr1_patches  # noqa: E402
import pm8_patches  # noqa: E402
import pt1_patches  # noqa: E402
import rs1_patches  # noqa: E402
import ss3_patches  # noqa: E402
import sbb1_patches  # noqa: E402
import strip_layer  # noqa: E402
import tie1_patches  # noqa: E402
import tl1_patches  # noqa: E402

_LABNAME = "k3_v37" + ("_ss3" if SS3 else "") + ("_fin1" if FIN1 else "") + ("_pm8" if PM8 else "") + ("_fr1" if FR1 else "")
OUT = ROOT / ("kernels/muse_decode_layer/k3_v39.py" if V39 else "kernels/muse_decode_layer/k3_v38.py" if V38 else "kernels/muse_decode_layer/k3_v37.py" if not (SS3 or FIN1 or PM8 or FR1)
              else f"kernels/lab/{_LABNAME}_ab.py")
LAB = ROOT / ("kernels/lab/k3_v39_ab.py" if V39 else "kernels/lab/k3_v38_ab.py" if V38 else f"kernels/lab/{_LABNAME}_ab.py")
NONCE = ("# nonce k3-v39-1 (k3_v37 [k3_v36 + RS1 + SBB1] + SU1 built / self-tested before PR206 in prepare)" if V39 else
         "# nonce k3-v38-1 (k3_v37 [k3_v36 + RS1 + SBB1] + SS3 ss in K3 via 4-CTA DSMEM clusters on; FIN1 code present, off)"
         if V38 else
         "# nonce k3-v37-1 (k3_v36 [v35_fa_pt1 + TIE1 + TL1 + K3P] + RS1 warp-parallel K2A setup + SBB1 attention boundary"
         + (" + FIN1 vid-3 9-warp finalize" if FIN1 else "") + (" + PM8 persistent T<=8 MoE lab" if PM8 else "") + (" + FR1 fresh-review switches lab" if FR1 else "") + ")")


def check(cond, msg):
    if not cond:
        raise SystemExit(f"[gen_k3_v37] FAIL: {msg}")


def generate():
    base = G.BASE.read_text(encoding="utf-8")
    check(hashlib.sha1(base.encode()).hexdigest() == G.BASE_SHA1, "k3_v34 changed (base sha1)")
    rep = []
    text = kw1_patches.apply(G.x4_levers(base))
    text = fa206_patches.apply(text, [p for p in fa206_patches.PATCHES if p[0] != "nonce"])
    text = k3p_patches.apply(tl1_patches.apply(tie1_patches.apply(pt1_patches.apply(text))))
    rep.append("k3_v36 chain: su1 + su1b + x_4 + KW1 + PR206 + PT1 + TIE1 + TL1 + K3P")
    if FIN1:
        import fin1_patches   # study-a's module, copied into tools/ when R-study-a-6 adopts FIN1
        text = fin1_patches.apply(text)
        check(text.count("((1 << 6) if FIN1_ENABLE else 0)") == 2, "FIN1 call sites")
        rep.append("FIN1 applied (study-a; probe bit 6)")
    text = rs1_patches.apply(text)
    check(text.count("((1 << 7) if RS1_ENABLE else 0)") == 2, "RS1 call sites")
    rep.append("RS1 applied (probe bit 7)")
    if SS3:
        text = ss3_patches.apply(text)
        check(text.count("| _ss3_bits(m)") == 2 and text.count("k3_kernel_ss(const Params P)") == 1, "SS3 anchors")
        rep.append("SS3 applied (probe bit 23 + FIN1_K1_ONLY bit 5; K3 4-CTA cluster ss; K2 no ss work; prepare envelope self-test)")
    text = sbb1_patches.apply(text)
    check(text.count("SBB1_ENABLE = True") == 1 and 'jobs.append(("sbb1", _sbb1_build))' in text, "SBB1 + PT1 job")
    rep.append("SBB1 applied (attention boundary, PT1 job)")
    if FR1:
        text = fr1_patches.apply(text)
        check(text.count("FR1_P1_GEO.get(") == 1 and text.count("if FR1_SU1_FIRST:") == 1, "FR1 anchors")
        rep.append("FR1 applied (call-time P1 geometry overrides; SU1 before PR206 in prepare)")
    if PM8:
        text = pm8_patches.apply(text)
        check(text.count("pm8_kernel(const Params P)") == 1 and text.count("    _pm8_prepare(state, weights)   #") == 1, "PM8 anchors")
        rep.append("PM8 applied (run3 mask bit 8 at vid 0, T <= 8; prepare self-test + fault injection)")
    if V39:   # R15 (e)
        text = fr1_patches.apply_order(text)
        check(text.index("\n    _su1_prepare(state, t0)") < text.index("\n    _p206_prepare(state, t0)"), "v39 order")
        rep.append("k3_v39: _su1_prepare before _p206_prepare (R15 (e))")
    if V38:   # R12: SS3 qualified, FIN1 arms did not
        check(text.count("FIN1_ENABLE = True   #") == 1, "FIN1 default anchor")
        text = text.replace("FIN1_ENABLE = True   #", "FIN1_ENABLE = False   # R12: off by default;")
        rep.append("k3_v38 defaults: SS3_ENABLE True, FIN1_ENABLE False (R12)")
    check(text.count(G.NONCE_OLD) == 1, "base nonce line")
    text = text.replace(G.NONCE_OLD, NONCE)
    before = len(text.encode())
    stripped = strip_layer.strip(text)
    ok, msg = strip_layer.equivalent(text, stripped)
    check(ok, f"strip: {msg}")
    lic0 = [l for l in text.splitlines() if strip_layer.LICENSE.search(l)]
    lic1 = [l for l in stripped.splitlines() if strip_layer.LICENSE.search(l)]
    check(lic0 == lic1, "strip: license lines changed")
    rep.append(f"stripped {before} -> {len(stripped.encode())} B; {msg}; {len(lic1)} license lines kept")
    return stripped, base, rep


def verify(text, base):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(tmp), doraise=True)
    finally:
        tmp.unlink()
    check(text.isascii(), "non-ASCII character")
    check(text.split("\n", 1)[0] == "# okc: problem=muse_decode_layer", "line 1")
    tag = "k3-v39-1" if V39 else "k3-v38-1" if V38 else "k3-v37-1"
    check(text.count(tag) == 1 and not any(f"k3-v3{v}-1" in text for v in (4, 5, 6)), "nonce")
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None)
             for n in ast.parse(text).body]
    dup = sorted({n for n in names if n and names.count(n) > 1})
    check(not dup, f"duplicate top-level names {dup}")
    s0, s1 = G.scan(base), G.scan(text)
    check(all(s1[k] <= s0[k] for k in s0), f"judge-checker scan grew: { {k: (s0[k], s1[k]) for k in s0 if s1[k] > s0[k]} }")
    check(len(text.encode()) < 1_000_000, "file >= 1 MB")
    return [f"compiles, ASCII, line 1, nonce, no duplicate names, checker scan <= base, {len(text.encode())} B"]


def main():
    text, base, rep = generate()
    rep += verify(text, base)
    for r in rep:
        print(f"[gen_k3_v37] OK  {r}")
    if CHECK:
        check(OUT.exists() and OUT.read_text(encoding="utf-8") == text, f"{OUT.name} on disk differs (regenerate)")
        print("[gen_k3_v37] --check: up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    LAB.write_text(text, encoding="utf-8", newline="\n")
    print(f"[gen_k3_v37] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} (sha1 {hashlib.sha1(text.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

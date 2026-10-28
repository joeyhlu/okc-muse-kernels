"""m32e_compose.py: k3_gc_<n> = an already m32e-patched layer (default k3_gc_5) + the T-routed WG5 JN 72 VML variant 'w72'
(R-m32e-16): applies only the incremental m32e patches (W5T + SPAN = VM_WG5 2, scratch for NTM <= 10, cost_ext NTM 8..10)
to the base's _vml_ block, adds _VML_DEFS['w72'] = VML_VAR's defines + WG5 2 / PREG 32 / CREG 112 / GV 2 / JN 72 / XSN 72,
and routes the given buckets to it (VML_BVAR; prepare self-tests each (bucket, w72) pair, a failure falls back to VML_VAR).
Only VML_VAR and w72 are compiled. Writes kernels/muse_decode_layer/<out>.py and the identical kernels/lab/<out>_ab.py.

    python -I tools/m32e_compose.py --base=k3_gc_5 --out=k3_gc_6 --buckets=128,256
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32 as GM  # noqa: E402
sys.argv = _argv
import m32e_patches as P  # noqa: E402

BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_5')}.py"
OUTN = ARGS.get("out", "k3_gc_6")
OUT, LAB = ROOT / f"kernels/muse_decode_layer/{OUTN}.py", ROOT / f"kernels/lab/{OUTN}_ab.py"
BUCKETS = [int(b) for b in ARGS.get("buckets", "128,256").split(",")]
NL = chr(10)
W72 = '"-DVM_WG5=2", "-DVM_PREG=32", "-DVM_CREG=112", "-DVM_GV=2", "-DVM_JN=72", "-DVM_XSN=72"'
# R-m32e-17: the base builds VML variants one after another inside the 'vml' PT1 job, which put the w72 compile on prepare's
# critical path (cold prepare 294.6 s, GD guard tripped at 241.9 s); build them in parallel threads instead.
BUILD_SEQ = """def _vml_build_job():
    for v in (VML_VAR,) + tuple(dict.fromkeys(a for a in tuple(VML_ABVARS) + tuple(VML_BVAR.values()) if a != VML_VAR)):
        _VML["mods"][v], _ = _vml_build(defines=_VML_DEFS[v])
    _VML["mod"] = _VML["mods"][VML_VAR]
"""
BUILD_PAR = """def _vml_build_job():
    import threading
    vs = (VML_VAR,) + tuple(dict.fromkeys(a for a in tuple(VML_ABVARS) + tuple(VML_BVAR.values()) if a != VML_VAR))
    errs = {}

    def one(v):   # a failed extra variant is just absent: its (bucket, variant) pairs never pass the self-test, VML_VAR runs
        try:
            _VML["mods"][v], _ = _vml_build(defines=_VML_DEFS[v])
        except Exception as ex:
            errs[v] = ex
    ths = [threading.Thread(target=one, args=(v,)) for v in vs[1:]]
    for th in ths:
        th.start()
    one(vs[0])
    for th in ths:
        th.join()
    if VML_VAR in errs:
        raise errs[VML_VAR]
    _VML["mod"] = _VML["mods"][VML_VAR]
"""


def main():
    GM.check("--dry" in sys.argv or not (OUT.exists() or LAB.exists()), f"{OUT.name} or its lab copy exists")
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(bn.startswith("# nonce "), f"{BASE.name}: line 2 is not a nonce line")
    t = b
    a0, b0 = P.cuda_span(t, "_vml_CUDA_T")
    cuda = t[a0:b0]
    for i, (old, new) in enumerate(P.W5T_PATCHES):
        cuda = P.sub1(cuda, old, new, f"w5t patch {i}")
    for s0, s1, old, new, cnt in P.SPAN_PATCHES:
        a = cuda.index(s0)
        z = cuda.index(s1, a)
        GM.check(cuda[a:z].count(old) == cnt, f"span patch {old!r}")
        cuda = cuda[:a] + cuda[a:z].replace(old, new) + cuda[z:]
    t = t[:a0] + cuda + t[b0:]
    t = P.sub1(t, '"fix": z((2 * (_vml_SLOTS + 384) * 4 * _vml_FIXW,), torch.float32)',
               '"fix": z((2 * (_vml_SLOTS + 384) * 4 * max(_vml_FIXW, 10 * 512),), torch.float32)', "fix scratch")
    t = P.sub1(t, '    assert ntm == 8, "m32e scratch (fix regions of 4096 floats) supports NTM 8 builds only"',
               '    assert 8 <= ntm <= 10, "m32e scratch (fix regions of max(FIXW, 5120) floats) supports NTM 8..10 builds"', "cost_ext assert")
    t = P.sub1(t, "        out += h + [h[7] + step * (k - 7) for k in range(8, ntm + 1)]",
               "        out += h + [h[7] + step * (k - 8) for k in range(9, ntm + 1)]", "cost_ext nt9+")
    var = next(ln for ln in t.split(NL) if ln.startswith("VML_VAR = ")).split("=", 1)[1].strip().strip("'\"")
    old = '"bf2": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1")}\n'
    GM.check(var == "bf2", f"VML_VAR {var!r}: w72 composes on bf2 only")
    t = P.sub1(t, old, '"bf2": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1"),\n'
               '             "w72": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", ' + W72 + ')}\n', "w72 defs")
    t = P.sub1(t, '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2") else _VML["sc"]',
               '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]', "scx")
    t = P.sub1(t, BUILD_SEQ, BUILD_PAR, "parallel VML variant builds")
    t = P.sub1(t, "\nVML_BVAR = {}\n", "\nVML_BVAR = " + repr({k: "w72" for k in sorted(BUCKETS, reverse=True)}) + "\n", "VML_BVAR")
    tag = OUTN.replace("_", "-")
    nonce = (f"# nonce {tag} ({BASE.stem} + T-routed VML w72 (WG5 2: setmaxnreg 112 / 32, JN 72) at buckets {sorted(BUCKETS)}, "
             f"bf2 elsewhere; tools/m32e_compose.py)")
    t = t.replace(bn, nonce)
    rep = [f"{len(P.W5T_PATCHES)} W5T + {len(P.SPAN_PATCHES)} span patches, scratch / cost_ext for NTM <= 10, w72 at {BUCKETS}"]
    s = t   # the base came from another generator (not a gen_k3_m32.finish fixed point): own checks instead
    import py_compile
    import tempfile
    GM.check(s.isascii(), "non-ASCII")
    GM.check(len(s.encode()) < 1_000_000, f"{len(s.encode())} B >= 1 MB")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(s, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    GM.check(s.count("# nonce ") >= 1 and s.split(NL)[1] == nonce, "nonce line")
    rep.append(f"py_compile ok, ASCII, {len(s.encode())} B")
    for r in rep:
        print(f"[m32e_compose] OK  {r}")
    if "--dry" in sys.argv:
        print(f"[m32e_compose] --dry: {len(s.encode())} B")
        return 0
    OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[m32e_compose] wrote {OUT.relative_to(ROOT)} + {LAB.relative_to(ROOT)} ({len(s.encode())} B, headroom "
          f"{1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""mb_gen18.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_15 + tools/vse_patches.py (VSE_PS arm bf2ps at bucket 32, as mb_gen17) +
wide-job VML builds under the real register cap: w88 / w96 = w72 defines + JN / XSN 88 or 96 + NX 2 (CREG 112 as w72: the
640-thread launch gets 96 regs / thread and setmaxnreg.inc only draws what the producer WG's .dec frees, so 112 is the cap;
mb_16's CREG 120 w112 hung in the b128 sweep). Experts of 73-88 / 73-96 tokens stream their weights once instead of twice
(model: bs256 mean -4.8 / -5.8 us, bs128 -0.9 / -1.6). Local ptxas vm_main spills: w72 16 / 8 B, w88 44 / 48, w96 64 / 160.
Arms 128 / 256: w72 / w88 / w96 (all :193:L; 'L' extends linearly to the wider nt). Host: NTM <= 14, fix scratch 14 x 512.
Refuses to overwrite.
    python -I tools/mb_gen18.py [--n=18] [--src=mb_15] [--wide=88,96]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vse_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 18))
SRCN = ARGS.get("src", "mb_15")
SRC = ROOT / f"kernels/muse_moe_w4a16/{SRCN}.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
WIDE = [int(v) for v in ARGS.get("wide", "88,96").split(",") if v]
NL = chr(10)


def _assign(src, name):
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return ast.get_source_segment(src, n), ast.literal_eval(n.value)
    raise KeyError(name)


def _fmt(d):
    return "{" + NL + "".join(f"    {k!r}: {v!r}," + NL for k, v in d.items()) + "}"


def _rep(s, old, new):
    assert s.count(old) == 1, (old[:70], s.count(old))
    return s.replace(old, new)


def main():
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    s = P.apply(SRC.read_text(encoding="utf-8"))
    ob, builds = _assign(s, "MB_BUILDS")
    builds["bf2ps"] = ("vml", tuple(builds["bf2"][1]) + ("-DVSE_PS=1",))
    for jn in WIDE:
        rep = {"-DVM_JN": str(jn), "-DVM_XSN": str(jn), "-DVM_NX": "2"}
        builds[f"w{jn}"] = ("vml", tuple(f for f in builds["w72"][1] if f.split("=")[0] not in rep) + tuple(f"{k}={v}" for k, v in rep.items()))
    oa, arms = _assign(s, "MB_ARMS")
    arms[32] = ["bf2:193:L", "bf2ps:193:L", "w72:193:L"]
    arms[128] = ["w72:193:L"] + [f"w{jn}:193:L" for jn in WIDE]
    arms[256] = ["w72:193:L"] + [f"w{jn}:193:L" for jn in WIDE]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    s = _rep(s, '    assert 8 <= ntm <= 10, "m32e scratch (fix regions of max(FIXW, 5120) floats) supports NTM 8..10 builds"',
             '    assert 8 <= ntm <= 14, "m32e scratch (fix regions of max(FIXW, 7168) floats) supports NTM 8..14 builds"')
    s = _rep(s, "max(_vml_FIXW, 10 * 512)", "max(_vml_FIXW, 14 * 512)")
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""{SRCN} (', f'"""mb_{N} (tools/mb_gen18.py: {SRCN} + vse_patches bf2ps@32 + wide-job builds '
                  f'{"/".join(f"w{j}" for j in WIDE)}@128/256, CREG 112, NX 2). {SRCN} (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen18] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms {arms}")


if __name__ == "__main__":
    main()

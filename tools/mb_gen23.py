"""mb_gen23.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_22 with VML arms re-set (runtime only, no new builds):
  flags 192 = 193 without bit 0 (split_point rotation off; parent's plan-search model: ~-1 us at b32 / 128 / 256, never measured)
  ck 'f' = cost table fitted in prepare on MB_FITS board routings from per-CTA marks (_mb_fit), next to the in-layer 'L' tables.
Arms 32: bf2ps:193:L (ref) / bf2ps:192:L / bf2ps:193:f / bf2:193:L; 128 and 256: w72:193:L (ref) / w72:192:L / w72:193:f / w72:192:f.
VMS buckets keep mb_22's arms. Refuses to overwrite.
    python -I tools/mb_gen23.py [--n=23] [--src=mb_22]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 23))
SRCN = ARGS.get("src", "mb_22")
SRC = ROOT / f"kernels/muse_moe_w4a16/{SRCN}.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
NL = chr(10)


def _assign(src, name):
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return ast.get_source_segment(src, n), ast.literal_eval(n.value)
    raise KeyError(name)


def _fmt(d):
    return "{" + NL + "".join(f"    {k!r}: {v!r}," + NL for k, v in d.items()) + "}"


def main():
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    s = SRC.read_text(encoding="utf-8")
    oa, arms = _assign(s, "MB_ARMS")
    arms[32] = ["bf2ps:193:L", "bf2ps:192:L", "bf2ps:193:f", "bf2:193:L"]
    arms[128] = ["w72:193:L", "w72:192:L", "w72:193:f", "w72:192:f"]
    arms[256] = ["w72:193:L", "w72:192:L", "w72:193:f", "w72:192:f"]
    s = s.replace(oa, "MB_ARMS = " + _fmt(arms))
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""{SRCN} (', f'"""mb_{N} (tools/mb_gen23.py: {SRCN} + VML arms flags 192 / fitted tables f at 32 / 128 / 256). '
                  f'{SRCN} (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen23] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); arms {arms}")


if __name__ == "__main__":
    main()

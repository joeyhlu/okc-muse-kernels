"""mb_gen40.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_29 + tools/vff_patches.py on the VML source (VM_FF 0 = SASS identical) and a
fused-finalize build w72ff (= w72 + -DVM_FF=1: per-token finalize inside vm_main, no vm_fin launch; local gate: w72 spills 16 / 8 B
as the base, 6 LDL/STL as the base) as an arm at 128 / 256. mb_29's marks measured the vm_fin tail after vm_main's last CTA at
b32 / 128 / 256 = 4.3 / 6.4 / 7.3 us (board). bf2 (bucket 32) stays on vm_fin: its VM_FF build adds ~160 spill instructions.
MKW-64 marks builds dropped. Arms 32: bf2ps:193:L; 128 / 256: w72:193:L / w72ff:193:L. Refuses to overwrite.
    python -I tools/mb_gen40.py [--n=40] [--src=mb_29]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vff_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 40))
SRCN = ARGS.get("src", "mb_29")
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
    s = P.apply(SRC.read_text(encoding="utf-8"))
    ob, builds = _assign(s, "MB_BUILDS")
    for k in ("w72m", "bf2psm"):
        builds.pop(k)
    builds["w72ff"] = ("vml", tuple(builds["w72"][1]) + ("-DVM_FF=1",))
    oa, arms = _assign(s, "MB_ARMS")
    arms[32] = ["bf2ps:193:L"]
    arms[128] = ["w72:193:L", "w72ff:193:L"]
    arms[256] = ["w72:193:L", "w72ff:193:L"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""mb_{SRCN[3:]} (', f'"""mb_{N} (tools/mb_gen40.py: {SRCN} + vff fused per-token finalize arm w72ff at 128 / 256).{NL}'
                  f'mb_{SRCN[3:]} (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen40] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms {arms}")


if __name__ == "__main__":
    main()

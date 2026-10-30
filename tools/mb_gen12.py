"""mb_gen12.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_11 + VM_RH (tools/vrh_patches.py: phase-A doneA releases issued by a
producer-WG helper warp instead of the consumer WG) as a new VML build 'w72rh' (= w72 defines + -DVM_RH=1); arms
32: bf2 / w72 / w72rh, 128: w72 / w72rh / bf2, 256: w72 / w72rh (all :193:L; mb_11's Lx arms dropped: killed). The in-prepare
sweep (16 board routings, paired r vs the first arm) measures w72rh vs w72 on the judge GPU. Refuses to overwrite.
    python -I tools/mb_gen12.py [--n=12] [--src=mb_11]
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vrh_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 12))
SRC = ROOT / f"kernels/muse_moe_w4a16/{ARGS.get('src', 'mb_11')}.py"
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
    builds["w72rh"] = ("vml", tuple(builds["w72"][1]) + ("-DVM_RH=1",))
    oa, arms = _assign(s, "MB_ARMS")
    arms[32] = ["bf2:193:L", "w72:193:L", "w72rh:193:L"]
    arms[128] = ["w72:193:L", "w72rh:193:L", "bf2:193:L"]
    arms[256] = ["w72:193:L", "w72rh:193:L"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    assert s.count(f"MB_N = 11\n") == 1
    s = s.replace("MB_N = 11\n", f"MB_N = {N}\n")
    s = s.replace('"""mb_11 (tools/mb_gen11.py:', f'"""mb_{N} (tools/mb_gen12.py: mb_11 + VM_RH release-helper arm w72rh). mb_11 (tools/mb_gen11.py:', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen12] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")


if __name__ == "__main__":
    main()

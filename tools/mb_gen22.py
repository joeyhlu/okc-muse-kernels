"""mb_gen22.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_19 + tools/vxp_patches.py on the layer-VMS source (VMS_L2H 0: weight copies
without evict_first; VMS_XPF 1: next-A-job x rows prefetched to L1 one unit ahead) as lvms arms at buckets 1 / 8 / 16, and the
marks printer fixed for lvms arms (the layer VMS writes the 'vms' kind's 10 phase marks at 16 int64 per CTA; mb_10..19 read them
with the VML layout). Arms 1: lv / lvf1 / lvaq / lvf1l2 / lvf1xp, 8: lv / lvf1 / lvl2 / lvxp, 16: lv / lvl2 / lvxp / lvaq.
Buckets >= 32 unchanged. Refuses to overwrite.
    python -I tools/mb_gen22.py [--n=22] [--src=mb_19]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vxp_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 22))
SRCN = ARGS.get("src", "mb_19")
SRC = ROOT / f"kernels/muse_moe_w4a16/{SRCN}.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
NL = chr(10)
LVQ = "MB_LV_CUDA = r'''"


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
    s = SRC.read_text(encoding="utf-8")
    a = s.index(LVQ) + len(LVQ)
    b = s.index("'''", a)
    s = s[:a] + P.apply(s[a:b]) + s[b:]
    s = _rep(s, '    vms = arm["kind"] == "vms"\n    W = 64 if vms else 32\n',
             '    vms = arm["kind"] in ("vms", "lvms")   # mb_gen22: the layer VMS writes the same 10 phase marks\n'
             '    W = 64 if arm["kind"] == "vms" else 16 if arm["kind"] == "lvms" else 32\n')
    ob, builds = _assign(s, "MB_BUILDS")
    builds["lvl2"] = ("lvms", ("-DVMS_L2H=0",))
    builds["lvxp"] = ("lvms", ("-DVMS_XPF=1",))
    builds["lvf1l2"] = ("lvms", ("-DVMS_FX=1", "-DVMS_L2H=0"))
    builds["lvf1xp"] = ("lvms", ("-DVMS_FX=1", "-DVMS_XPF=1"))
    oa, arms = _assign(s, "MB_ARMS")
    arms[1] = ["lv", "lvf1", "lvaq", "lvf1l2", "lvf1xp"]
    arms[8] = ["lv", "lvf1", "lvl2", "lvxp"]
    arms[16] = ["lv", "lvl2", "lvxp", "lvaq"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""{SRCN} (', f'"""mb_{N} (tools/mb_gen22.py: {SRCN} + vxp L2H / XPF lvms arms at 1 / 8 / 16, lvms marks fix). '
                  f'{SRCN} (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen22] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms {arms}")


if __name__ == "__main__":
    main()

"""mb_gen29.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_28 with the killed 't' / flags-192 arms dropped and VM_MKW 64 VML builds
(w72m, bf2psm = + -DVM_MKW=64) as marks-only arms at 32 / 128 / 256, to measure the vm_fin tail on the judge GPU before building a
fused finalize: _mb_marks reads 64-wide VML marks, seeds vm_fin's atomicMin slot (marks[G * 64]) and prints, for every MKW-64 arm,
route / gather / A end / vm_main end (WG ends) and fin0 / fin1 (first vm_fin post-wait, last vm_fin end; vm_fin writes them under
VM_MKW 64). fin1 - end max = what a fused finalize could remove. Timing arms otherwise as mb_28's live ones. No overwrite.
    python -I tools/mb_gen29.py [--n=29] [--src=mb_28]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 29))
SRCN = ARGS.get("src", "mb_28")
SRC = ROOT / f"kernels/muse_moe_w4a16/{SRCN}.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
NL = chr(10)

OLD_W = ('    vms = arm["kind"] in ("vms", "lvms")   # mb_gen22: the layer VMS writes the same 10 phase marks\n'
         '    W = 64 if arm["kind"] == "vms" else 16 if arm["kind"] == "lvms" else 32\n'
         '    marks = torch.zeros((G * W,), dtype=torch.int64, device=st["dev"])\n')
NEW_W = ('    vms = arm["kind"] in ("vms", "lvms")   # mb_gen22: the layer VMS writes the same 10 phase marks\n'
         '    W = 64 if arm["kind"] == "vms" else 16 if arm["kind"] == "lvms" else 32\n'
         '    w64 = arm["kind"] == "vml" and _vml_cfg(_MB["mods"][arm["b"]])[4] == 64   # mb_gen29: VM_MKW 64 VML build\n'
         '    if w64:\n'
         '        W = 64\n'
         '    marks = torch.zeros((G * W + 64,), dtype=torch.int64, device=st["dev"])\n')
OLD_Z = "        marks.zero_()\n        with torch.cuda.stream(stream):\n            torch.sum(flush, dim=0, out=sink)\n            _mb_call(arm, st,"
NEW_Z = ("        marks.zero_()\n        if w64:\n            marks[G * W] = 1 << 62   # vm_fin's first post-wait gtime (atomicMin)\n"
         "        with torch.cuda.stream(stream):\n            torch.sum(flush, dim=0, out=sink)\n            _mb_call(arm, st,")
OLD_MV = "        mv = marks.view(G, W).cpu().double()\n        t0 = float(mv[:, 0].min())\n"
NEW_MV = ("        mall = marks.cpu().double()\n        mv = mall[:G * W].view(G, W)\n        t0 = float(mv[:, 0].min())\n")
OLD_VML = ('            acc["Bwait"] = [float(x) / 1e3 for x in mv[:, 24]]\n'
           '            acc["Xwait"] = [float(x) / 1e3 for x in mv[:, 26]]\n')
NEW_VML = ('            if w64:   # mb_gen29: vm_fin first post-wait / last end (gtime), not the 32-wide B / X wait columns\n'
           '                acc["fin0"] = [(float(mall[G * W]) - t0) / 1e3]\n'
           '                acc["fin1"] = [(float(mall[G * W + 1]) - t0) / 1e3]\n'
           '            else:\n'
           '                acc["Bwait"] = [float(x) / 1e3 for x in mv[:, 24]]\n'
           '                acc["Xwait"] = [float(x) / 1e3 for x in mv[:, 26]]\n')
OLD_DIAG = "        diag = [a for a in (best, fb) if a is not None]\n"
NEW_DIAG = ("        diag = [a for a in (best, fb) if a is not None]\n"
            "        diag += [a for a in ok_arms if a['kind'] == 'vml' and _vml_cfg(_MB['mods'][a['b']])[4] == 64]   # mb_gen29\n")


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
    for o, n in ((OLD_W, NEW_W), (OLD_Z, NEW_Z), (OLD_MV, NEW_MV), (OLD_VML, NEW_VML), (OLD_DIAG, NEW_DIAG)):
        assert s.count(o) == 1, (o[:70], s.count(o))
        s = s.replace(o, n)
    ob, builds = _assign(s, "MB_BUILDS")
    builds["w72m"] = ("vml", tuple(builds["w72"][1]) + ("-DVM_MKW=64",))
    builds["bf2psm"] = ("vml", tuple(builds["bf2ps"][1]) + ("-DVM_MKW=64",))
    oa, arms = _assign(s, "MB_ARMS")
    arms[32] = ["bf2ps:193:L", "bf2psm:193:L"]
    arms[128] = ["w72:193:L", "w72m:193:L"]
    arms[256] = ["w72:193:L", "w72m:193:L"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""mb_{SRCN[3:]} (', f'"""mb_{N} (tools/mb_gen29.py: {SRCN} + VM_MKW 64 marks arms, vm_fin tail marks).{NL}'
                  f'mb_{SRCN[3:]} (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen29] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms {arms}")


if __name__ == "__main__":
    main()

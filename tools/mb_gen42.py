"""mb_gen42.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_41 with the killed VM_UD2 builds dropped and VMS_MK2 (per-CTA cycle counters
of consumer warp 0 and the helper) arms lvmk / lvf1mk at buckets 1 / 8 / 16, to measure where a T <= 16 unit's time goes on the
judge GPU before redesigning the VMS consumer: _mb_marks reads 32-wide lvms MK2 rows and prints, per bucket and arm, the per-unit
medians over CTAs and routings (us, clock -> time by each CTA's own clock / globaltimer ratio): F = full-barrier wait (data), C =
compute (after the data wait to the partial-buffer wait), E = pempty wait (helper behind), S = per-unit CTA sync, plus the doneA wait
W per CTA, helper F (pfull wait) / U (busy) per unit, and units / A units / jobs. Plus tools/vhi_patches.py on the layer-VMS
source: VMS_HI arms lvhi / lvf1hi (helper refills the ring, no per-unit CTA sync) and lvhimk. Timing arms otherwise as mb_41.
    python -I tools/mb_gen42.py [--n=42] [--src=mb_41]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vhi_patches as PH  # noqa: E402
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 42))
SRCN = ARGS.get("src", "mb_41")
SRC = ROOT / f"kernels/muse_moe_w4a16/{SRCN}.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
NL = chr(10)

OLD_W = ('    w64 = arm["kind"] == "vml" and _vml_cfg(_MB["mods"][arm["b"]])[4] == 64   # mb_gen29: VM_MKW 64 VML build\n'
         '    if w64:\n'
         '        W = 64\n')
NEW_W = (OLD_W +
         '    mk2 = arm["kind"] == "lvms" and "-DVMS_MK2=1" in tuple(MB_BUILDS[arm["b"]][1])   # mb_gen42: 32-wide MK2 rows\n'
         '    if mk2:\n'
         '        W = 32\n')
OLD_ACC = ('        if vms:\n'
           '            cols = {"route": 1, "first": 2, "bw0": 3, "bw1": 4, "comp": 5, "hA": 6, "hend": 7, "bar": 8, "fin": 9}\n'
           '            for k, c in cols.items():\n'
           '                acc[k] = [(float(x) - t0) / 1e3 for x in mv[:, c] if x > 0]\n')
NEW_ACC = (OLD_ACC +
           '            if mk2:   # mb_gen42: per-unit cycle split of consumer warp 0 (tid 0) and the helper, in us\n'
           '                for row in mv:\n'
           '                    nq = int(row[13]) & 0xFFFF\n'
           '                    span_ns, span_ck = float(row[9] - row[0]), float(row[17] - row[16])\n'
           '                    if nq <= 0 or span_ns <= 0 or span_ck <= 0:\n'
           '                        continue\n'
           '                    us = span_ns / span_ck / 1e3   # us per clock of this CTA\n'
           '                    for k, c in (("F", 10), ("C", 19), ("E", 11), ("S", 12), ("hF", 14), ("hU", 23)):\n'
           '                        acc.setdefault(k, []).append(float(row[c]) * us / nq)\n'
           '                    acc.setdefault("W", []).append(float(row[18]) * us)\n'
           '                    acc.setdefault("nq", []).append(float(nq))\n'
           '                    acc.setdefault("nA", []).append(float((int(row[13]) >> 16) & 0xFFFF))\n'
           '                    acc.setdefault("jobs", []).append(float(int(row[13]) >> 32))\n'
           '                    acc.setdefault("ghz", []).append(span_ck / span_ns)\n')
OLD_Q = '        return f"{sum(p50) / len(p50):.1f}/{sum(mx) / len(mx):.1f}" if p50 else "-"\n'
NEW_Q = ('        if p50 and k in ("F", "C", "E", "S", "hF", "hU", "ghz"):   # mb_gen42: per-unit values need 2-3 decimals\n'
         '            return f"{sum(p50) / len(p50):.3f}/{sum(mx) / len(mx):.3f}"\n' + OLD_Q)
OLD_DIAG = "        diag += [a for a in ok_arms if a['kind'] == 'vml' and _vml_cfg(_MB['mods'][a['b']])[4] == 64]   # mb_gen29\n"
NEW_DIAG = (OLD_DIAG + "        diag += [a for a in ok_arms if a['kind'] == 'lvms' and '-DVMS_MK2=1' in tuple(MB_BUILDS[a['b']][1])]   "
            "# mb_gen42\n")


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
    lq = "MB_LV_CUDA = r'''"
    la = s.index(lq) + len(lq)
    lb = s.index("'''", la)
    s = s[:la] + PH.apply(s[la:lb]) + s[lb:]
    for o, n in ((OLD_W, NEW_W), (OLD_ACC, NEW_ACC), (OLD_Q, NEW_Q), (OLD_DIAG, NEW_DIAG)):
        assert s.count(o) == 1, (o[:70], s.count(o))
        s = s.replace(o, n)
    ob, builds = _assign(s, "MB_BUILDS")
    for k in ("w72ud", "bf2psud"):
        builds.pop(k, None)
    builds["lvmk"] = ("lvms", ("-DVMS_MK2=1",))
    builds["lvf1mk"] = ("lvms", ("-DVMS_FX=1", "-DVMS_MK2=1"))
    builds["lvhi"] = ("lvms", ("-DVMS_HI=1",))
    builds["lvf1hi"] = ("lvms", ("-DVMS_FX=1", "-DVMS_HI=1"))
    builds["lvhimk"] = ("lvms", ("-DVMS_HI=1", "-DVMS_MK2=1"))
    oa, arms = _assign(s, "MB_ARMS")
    arms[1] = ["lvf1", "lv", "lvhi", "lvf1hi", "lvmk", "lvhimk"]
    arms[8] = ["lv", "lvf1", "lvhi", "lvmk", "lvhimk"]
    arms[16] = ["lv", "lvhi", "lvmk", "lvhimk"]
    arms[32] = ["bf2ps:193:L"]
    arms[128] = ["w72:193:L"]
    arms[256] = ["w72:193:L"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""mb_{SRCN[3:]} (', f'"""mb_{N} (tools/mb_gen42.py: {SRCN} + VMS_MK2 cycle-split + VMS_HI (helper refill, no per-unit sync) arms at 1 / 8 / 16).{NL}'
                  f'mb_{SRCN[3:]} (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen42] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms {arms}")


if __name__ == "__main__":
    main()

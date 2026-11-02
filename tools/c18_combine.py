"""combo18 lab = k3_moexr_c1 + HYBL2 (tools/hybl2_patches.py) + VFR1 (tools/vfr1_patches.py) + OX8 (o-proj GD at T=8 served by the
o/1 route record, XP 6 = coherent pre-wait weight loads; KSC SASS gate showed the XP 0 o/8 instance loads all weights below ACQBULK).
Writes kernels/lab/k3_c18_ab.py only."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import hybl2_patches as H
import vfr1_patches as V

def once(s, a, b):
    assert s.count(a) == 1, (a[:80], s.count(a))
    return s.replace(a, b)

src = H.BASE.read_text(encoding="utf-8").replace("\r\n", "\n")
cu = H.patch_cuda(H.cuda_source(src))
src = once(src, 'def _moexr_bank():\n', '_HYBL2_CUDA = ' + repr(cu) + '\n' + H.GLUE + '\ndef _moexr_bank():\n')
src = once(src, ' return _HYB if MOEXR else None\n', ' if not MOEXR: return None\n H = _HYBL2.get(HYBL2)\n return H if H is not None and H["ok"] else _HYB\n')
src = once(src, ' _moexr_prepare_bank(st,w,dev,_HYB)\n', ' _moexr_prepare_bank(st,w,dev,_HYB)\n _hybl2_prepare(st,w,dev)\n')
src = once(src, '  jobs.append(("hyb",_hyb_build))\n', '  jobs.append(("hyb",_hyb_build))\n  jobs.append(("hybl2",_hybl2_build))\n')
src = V.generate(src)
# OX8: call-time switch; o/1's route record (MT 1 -> T 8, XP 6) serves o at T=8 when OX8 = 1
src = once(src, 'def _gd_live(state,proj,tokens):\n st=state.get("gd")\n if not GD_ENABLE or st is None:\n  return None\n b=_gd_bucket(tokens)\n',
           'OX8=0\ndef _gd_live(state,proj,tokens):\n st=state.get("gd")\n if not GD_ENABLE or st is None:\n  return None\n b=_gd_bucket(tokens)\n'
           ' if OX8 and proj=="o" and b==8:\n  r=st["route"].get(("o",1))\n  if r is not None and tokens<=r["T"]:\n   return r\n')
compile(src, "k3_c18_ab.py", "exec")
out = ROOT / "kernels/lab/k3_c18_ab.py"
out.write_bytes(src.encode("ascii"))
print("wrote", out, len(src.encode("ascii")), "B")

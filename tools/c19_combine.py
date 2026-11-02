"""combo19 lab = kernels/lab/k3_catt_ab.py (tools/catt_patches.py on k3_moexr_c1) + OX8 (same patch as tools/c18_combine.py).
Writes kernels/lab/k3_c19_ab.py only."""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
def once(s, a, b):
    assert s.count(a) == 1, (a[:80], s.count(a))
    return s.replace(a, b)
src = (ROOT / "kernels/lab/k3_catt_ab.py").read_text(encoding="utf-8").replace("\r\n", "\n")
src = once(src, 'def _gd_live(state,proj,tokens):\n st=state.get("gd")\n if not GD_ENABLE or st is None:\n  return None\n b=_gd_bucket(tokens)\n',
           'OX8=0\ndef _gd_live(state,proj,tokens):\n st=state.get("gd")\n if not GD_ENABLE or st is None:\n  return None\n b=_gd_bucket(tokens)\n'
           ' if OX8 and proj=="o" and b==8:\n  r=st["route"].get(("o",1))\n  if r is not None and tokens<=r["T"]:\n   return r\n')
compile(src, "k3_c19_ab.py", "exec")
out = ROOT / "kernels/lab/k3_c19_ab.py"
out.write_bytes(src.encode("ascii"))
print("wrote", out, len(src.encode("ascii")), "B")

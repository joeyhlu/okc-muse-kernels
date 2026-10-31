"""vfx_patches.py (R-v32): give the w72 VML build its own scratch, so its VSE_PS split partials cannot land in bf2's middle region.

Bug (R-v32-1..7, notes/memory/gpu_requests.md): VML bucket 32 (bf2) returned non-finite / wrong rows on concentrated routings
(3-4 local experts, 69+ routes at T 17..48) on the first call after prepare. Root cause: bf2 (buckets 32, NTM 8) and w72 (buckets
128 / 256, VM_JN 72 -> NTM 9) share one scratch dict (_VML["scb"]), but their device fix-region stride differs (FIXW = max(4096,
NTM x 512): 4096 vs 4608 floats). VSE_PS keeps head / tail split partials as plain stores (never zeroed) in regions 0 / 1 and middle
partials in region 2, which must be zero at a call's entry (red.add; only the split unit's last arriver re-zeroes it). w72's
plain-stored regions overlap bf2's region-2 address range, so after any w72 call (prepare: _m32e_vartest, HYB's VML comparisons;
production whenever T >= 49 falls back from HYB to VML) the next bf2 call with middle shares (few jobs = concentrated routing)
sums that garbage. Fix (host only, no CUDA text change, SASS identical): w72 gets its own _vml_alloc scratch (sharing only the err
flag the self-tests read); bf2 keeps _VML["scb"].
    python -I tools/vfx_patches.py --base=kernels/lab/k3_hh_3rr.py --out=kernels/lab/k3_hh3_vfx_r.py
    python -I tools/slim_patches.py --base=kernels/lab/k3_hh3_vfx_r.py --out=kernels/lab/k3_hh3_vfx.py   (LAST)
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)

O_ALLOC = '        _VML["scb"] = dict(_VML["sc"], y=torch.zeros_like(_VML["sc"]["y"]))\n'
N_ALLOC = (O_ALLOC +
           '        # R-v32: w72 (NTM 9: fix stride 4608 floats) must not share bf2\'s fix scratch (stride 4096): its plain-stored VSE_PS\n'
           '        # head / tail partials overlapped bf2\'s middle region, which must be zero at entry (tools/vfx_patches.py)\n'
           '        _VML["scw"] = dict(_vml_alloc(dev, sms), err=_VML["sc"]["err"])\n')
O_SCX = '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]\n'
N_SCX = (O_SCX + '    if var == "w72" and _VML.get("scw") is not None:\n'
         '        scx = _VML["scw"]   # R-v32: own scratch (fix stride differs from bf2\'s)\n')
O_DICT = '_VML = {"mod": None, "mods": {}, "sc": None, "scb": None, '
N_DICT = '_VML = {"mod": None, "mods": {}, "sc": None, "scb": None, "scw": None, '
PATCHES = [("alloc", O_ALLOC, N_ALLOC), ("route scratch", O_SCX, N_SCX), ("state dict", O_DICT, N_DICT)]


def apply(s):
    for tag, old, _ in PATCHES:
        assert s.count(old) == 1, f"vfx anchor '{tag}' found {s.count(old)} times"
    for _, old, new in PATCHES:
        s = s.replace(old, new)
    return s


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3rr.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hh3_vfx_r.py")
    assert not out.exists(), f"refusing to overwrite {out}"
    s = base.read_text(encoding="utf-8")
    s = apply(s)
    lines = s.split(NL)
    assert lines[1].startswith("# nonce ")
    lines[1] = f"# nonce k3-hh3-vfx (k3_hh_3rr + w72 VML scratch of its own: bucket-32 VSE_PS middle-region fix; tools/vfx_patches.py)"
    s = NL.join(lines)
    assert s.isascii()
    out.write_text(s, encoding="utf-8", newline=NL)
    import py_compile
    py_compile.compile(str(out), doraise=True)
    print(f"[vfx] {base.name} -> {out.name} ({len(s.encode())} B)")


if __name__ == "__main__":
    main()

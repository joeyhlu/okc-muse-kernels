"""vfx_patches.py (R-v32): the w72 VML build gets its own scratch, so its VSE_PS split partials cannot land in bf2's middle region.

  python -I tools/vfx_patches.py [--base=kernels/lab/k3_hh_3k.py] [--out=kernels/lab/k3_hh_3kv.py] [--check]
  python -I tools/lead_gate.py file kernels/lab/k3_hh_3kv.py

Why (R-v32-1..8, notes/memory/gpu_requests.md, notes/lab.md 2026-10-31 03:25): VML bucket 32 (bf2) returned non-finite / wrong rows
on concentrated routings (3-4 local experts, 69+ routes at T 17..48) on the first such call after prepare. bf2 (buckets 32, NTM 8)
and w72 (buckets 128 / 256, VM_JN 72 -> NTM 9) shared one scratch dict (_VML["scb"]), but their device fix-region stride differs
(FIXW = max(4096, NTM x 512): 4096 vs 4608 floats). VSE_PS keeps head / tail split partials as plain stores (never zeroed) in fix
regions 0 / 1 and middle partials in region 2, which must be zero at a call's entry (red.add; only the split unit's last arriver
re-zeroes it). w72's regions 0 / 1 overlap bf2's region-2 addresses, so after any w72 call (prepare: _m32e_vartest and HYB's VML
comparisons; production whenever T >= 49 falls back from HYB to VML) the next bf2 call with middle shares (few jobs = concentrated
routing) summed that garbage. Box: ship 8 / 8 first-call + 24 / 24 w72 -> bf2 interleave bad, fixed 0 / 0 (R-v32-8).

Edits (each anchor must occur exactly once; plain-text replacements of code lines, no comment anchors; host Python only, the CUDA
text and therefore every SASS are unchanged):
  1. nonce line (line 2).
  2. _VML state dict: + "scw": None.
  3. _vml_prepare: after _VML["scb"] is built, _VML["scw"] = its own _vml_alloc scratch for w72, sharing only the err flag (the
     self-tests read _VML["sc"]["err"]).
  4. _vml_route: var w72 takes _VML["scw"] when it exists; bf2 and every other variant keep their scratch.
--check: no write; verifies that --out equals apply(--base) byte for byte, and the $0 gates on it (AST parse, ASCII, < 1,000,000 B,
every decorated def re-parses from its own source text via tools/pyslim.py, every inserted anchor present once).
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[vfx_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


A_STATE = '_VML = {"mod": None, "mods": {}, "sc": None, "scb": None, '
N_STATE = '_VML = {"mod": None, "mods": {}, "sc": None, "scb": None, "scw": None, '

A_ALLOC = '        _VML["scb"] = dict(_VML["sc"], y=torch.zeros_like(_VML["sc"]["y"]))\n'
N_ALLOC = A_ALLOC + '        _VML["scw"] = dict(_vml_alloc(dev, sms), err=_VML["sc"]["err"])\n'

A_SCX = '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]\n'
N_SCX = A_SCX + '    if var == "w72" and _VML.get("scw") is not None:\n        scx = _VML["scw"]\n'


def nonce(base_name):
    return (f"# nonce {base_name.replace('_', '-')}v ({base_name} + R-v32: w72 VML scratch of its own, bucket-32 VSE_PS middle-region fix; "
            f"tools/vfx_patches.py)")


def apply(s, base_name):
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] = nonce(base_name)
    s = NL.join(lines)
    s = sub(s, A_STATE, N_STATE, "_VML state dict")
    s = sub(s, A_ALLOC, N_ALLOC, "_vml_prepare scb alloc")
    s = sub(s, A_SCX, N_SCX, "_vml_route scratch choice")
    return s


def gates(path, s):
    import pyslim
    n = len(s.encode())
    check(n < LIMIT, f"size {n} B >= {LIMIT}")
    check(s.isascii(), "non-ASCII byte")
    ast.parse(s)
    d = pyslim.decorated_sources(s)
    bad = [x for x in d if x.startswith("UNPARSABLE")]
    check(not bad, f"decorated defs that do not re-parse from their own text: {bad[:2]}")
    for a in ('"scb": None, "scw": None, ', '_VML["scw"] = dict(_vml_alloc(dev, sms), err=_VML["sc"]["err"])',
              'if var == "w72" and _VML.get("scw") is not None:'):
        check(s.count(a) == 1, f"inserted anchor {a!r} present {s.count(a)}x")
    print(f"vfx_patches: gates OK on {path.name}: {n:,} B ({LIMIT - n:,} B under the cap), {len(d)} decorated defs re-parse")


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3k.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hh_3kv.py")
    new = apply(base.read_bytes().decode("ascii"), base.stem)
    if "check" in FLAGS:
        check(out.exists(), f"{out} missing")
        check(out.read_bytes().decode("ascii") == new, f"{out.name} differs from apply({base.name})")
        gates(out, new)
        print(f"vfx_patches: --check OK ({out.name} == apply({base.name}))")
        return
    ast.parse(new)
    out.write_bytes(new.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"vfx_patches: {base.name} -> {out.name} ({len(new):,} B, +{len(new) - base.stat().st_size:,} B)")
    gates(out, new)


if __name__ == "__main__":
    main()

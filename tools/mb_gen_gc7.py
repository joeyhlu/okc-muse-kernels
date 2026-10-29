"""mb_gen_gc7.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_8 (MoE board loop file, tools/mb_gen.py) with its VML block
replaced by k3_gc_7's shipped VML (FSW / BRED / D2 / SYNC7 / NOB9 / WG5 2 w72 variants, verbatim slice) and its stage
layout (K2A31_BITPOS3 = UNP L2). VMS (T <= 16) stays mb_8's (s3 / xw6). Arms: bf2 and w72 at T >= 32, flags 193, the
layer cost tables (== k3_gc_7 VML_COST). Check only, never scored. Refuses to overwrite.

    python -I tools/mb_gen_gc7.py [--n=9]
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 9))
SRC = ROOT / "kernels/muse_moe_w4a16/mb_8.py"
LAYER = ROOT / "kernels/muse_decode_layer/k3_gc_7.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"

VML_START = "_vml_E_LOC, _vml_TOPK, _vml_KDIM = 32, 8, 4096\n"
MB_GLUE = "\n\n# ================================================================ mb glue (tools/mb_gen.py)\n"
LAYER_VML_END = "\n\nVML_ENABLE = True\n"

BUILDS = {
    "s3": ("vms", 3, 16, 0),
    "xw6": ("vms", 3, 16, 1152, 0, 6),
}
ARMS = {1: ["s3", "xw6"], 8: ["s3", "xw6"], 16: ["s3", "xw6"],
        32: ["bf2:193:L", "w72:193:L"], 128: ["w72:193:L", "bf2:193:L"], 256: ["w72:193:L", "bf2:193:L"]}


def _fmt(d):
    nl = chr(10)
    return "{" + nl + "".join(f"    {k!r}: {v!r}," + nl for k, v in d.items()) + "}"


def _assign_src(src, name):
    t = ast.parse(src)
    for n in t.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return ast.get_source_segment(src, n), ast.literal_eval(n.value)
    raise KeyError(name)


def main():
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    s = SRC.read_text(encoding="utf-8")
    L = LAYER.read_text(encoding="utf-8")
    # 1. VML block
    a0 = s.index(VML_START)
    a1 = s.index(MB_GLUE)
    b0 = L.index(VML_START)
    b1 = L.index(LAYER_VML_END, b0)
    vml = L[b0:b1]
    assert vml.count("def _vml_call(") == 1 and "_pt1_wait" not in vml and "_v11_" not in vml
    s = s[:a0] + vml + s[a1:]
    s = s.replace("# ======== VML (kernels/lab/vml_moe.py via tools/vml_patches.lab_block, + flags bit 10)",
                  "# ======== VML (verbatim k3_gc_7 slice, tools/mb_gen_gc7.py)")
    # 2. stage layout bit positions (UNP L2)
    old_bp, _ = _assign_src(s, "K2A31_BITPOS3")
    new_bp, _ = _assign_src(L, "K2A31_BITPOS3")
    assert s.count(old_bp) == 1
    s = s.replace(old_bp, new_bp)
    # 3. builds / arms (VML defines from k3_gc_7's _VML_DEFS)
    _, defs = _assign_src(L, "_VML_DEFS")
    builds = dict(BUILDS, bf2=("vml", tuple(defs["bf2"])), w72=("vml", tuple(defs["w72"])))
    ob, _ = _assign_src(s, "MB_BUILDS")
    oa, _ = _assign_src(s, "MB_ARMS")
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(ARMS))
    # layer cost tables == k3_gc_7 VML_COST
    _, lc = _assign_src(L, "VML_COST")
    _, mc = _assign_src(s, "MB_LAYER_COST")
    assert {k: list(v) for k, v in lc.items()} == {k: list(v) for k, v in mc.items()}, "layer cost tables differ"
    s = s.replace("# okc: problem=muse_moe_w4a16\n\"\"\"mb_8:", f"# okc: problem=muse_moe_w4a16\n\"\"\"mb_{N} (tools/mb_gen_gc7.py: "
                  "mb_8 + k3_gc_7 VML bf2 / w72 + UNP L2 stage layout):", 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    dup = sorted({n for n in names if n and names.count(n) > 1})
    assert not dup, dup
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    long_lines = [(i + 1, len(x)) for i, x in enumerate(s.split("\n")) if len(x) >= 1000]
    assert not long_lines, long_lines[:5]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline="\n")
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen_gc7] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")


if __name__ == "__main__":
    main()

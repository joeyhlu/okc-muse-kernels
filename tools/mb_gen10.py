"""mb_gen10.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_9 (MoE board loop file) brought to the current layer ship's engines:
  * VML block replaced by kernels/lab/k3_b23_1.py's VML (verbatim slice; same _vml_call / _vml_alloc / _vml_build interface
    as gc_7's), bf2 / w72 builds take b23's _VML_DEFS (VM_RT, VTR riders, VCP, SFX, FNC acq, ...); cost tables asserted ==.
  * the layer VMS (b23's VMS_CUDA_SRC verbatim: UNP L2 layout, VMS_RO) as a new build 'lv' (own module, own prepack with the
    layer's VMS_POS, flags as the layer's _vms_build) next to mb's own VMS builds s3 / xw6; arm 'lv' at buckets 1 / 8 / 16.
  * probe arm 's32' (mb's VMS at MAXT 32) at bucket 32 next to VML bf2 / w72.
The in-prepare sweep (board-like routings, cold-L2 graph replays) picks the faster arm per bucket, as in every mb file.
Check only until a check proves a gain. Refuses to overwrite.

    python -I tools/mb_gen10.py [--n=10] [--src=mb_9] [--layer=kernels/lab/k3_b23_1.py]
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 10))
SRC = ROOT / f"kernels/muse_moe_w4a16/{ARGS.get('src', 'mb_9')}.py"
LAYER = ROOT / ARGS.get("layer", "kernels/lab/k3_b23_1.py")
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
NL = chr(10)
VML_START = "_vml_E_LOC, _vml_TOPK, _vml_KDIM = 32, 8, 4096\n"
MB_GLUE = "\n\n# ================================================================ mb glue (tools/mb_gen.py)\n"
LAYER_VML_END = "\n\nVML_ENABLE = True\n"


def _assign(src, name):
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return ast.get_source_segment(src, n), ast.literal_eval(n.value)
    raise KeyError(name)


def _func(src, name):
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
    raise KeyError(name)


def _fmt(d):
    return "{" + NL + "".join(f"    {k!r}: {v!r}," + NL for k, v in d.items()) + "}"


def rep(s, old, new, what):
    n = s.count(old)
    assert n == 1, f"{what}: {n} matches"
    return s.replace(old, new)


def main():
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    s = SRC.read_text(encoding="utf-8")
    L = LAYER.read_text(encoding="utf-8")
    # 1. VML block
    a0, a1 = s.index(VML_START), s.index(MB_GLUE)
    b0 = L.index(VML_START)
    b1 = L.index(LAYER_VML_END, b0)
    vml = L[b0:b1]
    assert vml.count("def _vml_call(") == 1 and "_pt1_wait" not in vml and "_v11_" not in vml
    s = s[:a0] + vml + s[a1:]
    assert _assign(s, "K2A31_BITPOS3")[1] == _assign(L, "K2A31_BITPOS3")[1], "stage layout differs"
    _, lc = _assign(L, "VML_COST")
    _, mc = _assign(s, "MB_LAYER_COST")
    assert {k: list(v) for k, v in lc.items()} == {k: list(v) for k, v in mc.items()}, "layer cost tables differ"
    # 2. layer VMS as build 'lv'
    i = L.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    lv_cuda = L[i:L.index('"""', i)]
    i = L.index('VMS_CPP_SRC = r"""') + len('VMS_CPP_SRC = r"""')
    lv_cpp = L[i:L.index('"""', i)]
    assert "'''" not in lv_cuda and "'''" not in lv_cpp
    pos_src, _ = _assign(L, "VMS_POS")
    pw = _func(L, "_vms_pack_words").replace("def _vms_pack_words(", "def _mb_lv_pack_words(").replace("VMS_POS[", "MB_LV_POS[")
    pe = (_func(L, "_vms_prepack_expert").replace("def _vms_prepack_expert(", "def _mb_lv_prepack_expert(")
          .replace("_vms_pack_words(", "_mb_lv_pack_words("))
    lv_block = (NL + NL + "# ================================================================ layer VMS (tools/mb_gen10.py: verbatim "
                + LAYER.name + " VMS_CUDA_SRC, UNP L2 layout, built as the layer's _vms_build)" + NL
                + "MB_LV_CUDA = r'''" + lv_cuda + "'''" + NL + "MB_LV_CPP = r'''" + lv_cpp + "'''" + NL
                + pos_src.replace("VMS_POS", "MB_LV_POS", 1) + NL + NL + NL
                + "def _mb_build_lv(G):" + NL
                + "    from torch.utils.cpp_extension import load_inline" + NL
                + '    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]' + NL
                + '    tag = hashlib.sha1((MB_LV_CUDA + " ".join(flags)).encode()).hexdigest()[:10]' + NL
                + '    return load_inline(name=f"mblv_{tag}", cpp_sources=MB_LV_CPP, cuda_sources=MB_LV_CUDA, functions=["run", "attrs"],' + NL
                + "                       extra_cuda_cflags=flags)" + NL + NL + NL + pw + NL + NL + NL + pe + NL)
    s = rep(s, MB_GLUE, lv_block + MB_GLUE, "lv block")
    # 3. builds / arms
    _, defs = _assign(L, "_VML_DEFS")
    _, ob_val = _assign(s, "MB_BUILDS")
    builds = dict(ob_val, bf2=("vml", tuple(defs["bf2"])), w72=("vml", tuple(defs["w72"])), lv=("lvms",), s32=("vms", 3, 32, 0))
    arms = {1: ["s3", "lv", "xw6"], 8: ["s3", "lv", "xw6"], 16: ["s3", "lv", "xw6"],
            32: ["bf2:193:L", "w72:193:L", "s32"], 128: ["w72:193:L", "bf2:193:L"], 256: ["w72:193:L", "bf2:193:L"]}
    ob, _ = _assign(s, "MB_BUILDS")
    oa, _ = _assign(s, "MB_ARMS")
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    s = rep(s, "MB_N = 8\n", f"MB_N = {N}\n", "MB_N")
    # 4. glue: build / prepack / scratch / call / prepare for kind 'lvms'
    s = rep(s, "        if b[0] == \"vms\":\n            _MB[\"mods\"][name] = _mb_build_vms(",
            "        if b[0] == \"lvms\":\n            _MB[\"mods\"][name] = _mb_build_lv(G)\n"
            "        elif b[0] == \"vms\":\n            _MB[\"mods\"][name] = _mb_build_vms(", "build_one")
    s = rep(s, "        pk[\"vms\"] = {\"wA\": wA, \"wB\": wB, \"gsA\": (w[\"w13_gs\"][:E].float() * two62).contiguous(),\n"
               "                     \"gsB\": (w[\"w2_gs\"][:E].float() * two62).contiguous(), \"nw\": (1.0 + w[\"post_norm_w\"].float()).contiguous()}\n",
            "        pk[\"vms\"] = {\"wA\": wA, \"wB\": wB, \"gsA\": (w[\"w13_gs\"][:E].float() * two62).contiguous(),\n"
            "                     \"gsB\": (w[\"w2_gs\"][:E].float() * two62).contiguous(), \"nw\": (1.0 + w[\"post_norm_w\"].float()).contiguous()}\n"
            "        lA = torch.empty((E, 512, 36864), dtype=torch.uint8, device=dev)   # mb_gen10: layer VMS layout (UNP L2)\n"
            "        lB = torch.empty((E, 256, 36864), dtype=torch.uint8, device=dev)\n"
            "        for e in range(E):\n"
            "            lA[e] = _mb_lv_prepack_expert(w[\"w13\"][e], w[\"w13_sf\"][e], True)\n"
            "            lB[e] = _mb_lv_prepack_expert(w[\"w2\"][e], w[\"w2_sf\"][e], False)\n"
            "        pk[\"lvms\"] = dict(pk[\"vms\"], wA=lA, wB=lB)\n", "prepack")
    s = rep(s, "    b = MB_BUILDS[name]\n    if b[0] == \"vms\":\n        mp = 8 * b[2]\n",
            "    b = MB_BUILDS[name]\n    if b[0] in (\"vms\", \"lvms\"):\n        mp = 8 * (b[2] if b[0] == \"vms\" else 16)\n", "scratch")
    s = rep(s, "                \"nomarks\": torch.empty((0,), dtype=torch.int64, device=dev), \"maxt\": b[2]}\n",
            "                \"nomarks\": torch.empty((0,), dtype=torch.int64, device=dev), \"maxt\": b[2] if b[0] == \"vms\" else 16}\n",
            "scratch maxt")
    s = rep(s, "    if arm[\"kind\"] == \"vms\":\n        pk = st[\"pk\"][\"vms\"]\n",
            "    if arm[\"kind\"] == \"lvms\":\n        pk = st[\"pk\"][\"lvms\"]\n"
            "        mod.run(x, ids, tw, pk[\"wA\"], pk[\"wB\"], pk[\"gsA\"], pk[\"gsB\"], pk[\"nw\"], sc[\"inter\"], sc[\"ybuf\"], sc[\"ss\"], sc[\"ctr\"],\n"
            "                out, sc[\"nomarks\"] if marks is None else marks, m, MB_EPS, 0, 0)\n"
            "        return out\n"
            "    if arm[\"kind\"] == \"vms\":\n        pk = st[\"pk\"][\"vms\"]\n", "call")
    s = rep(s, "    return int(sc[\"ctr\"][2]) if arm[\"kind\"] == \"vms\" else int(sc[\"err\"][0])\n",
            "    return int(sc[\"ctr\"][2]) if arm[\"kind\"] in (\"vms\", \"lvms\") else int(sc[\"err\"][0])\n", "err")
    s = rep(s, "        if MB_BUILDS[name][0] == \"vms\":\n            try:\n",
            "        if MB_BUILDS[name][0] in (\"vms\", \"lvms\"):\n            try:\n", "attrs print")
    s = rep(s, "        need = \"vms\" if MB_BUILDS[name][0] == \"vms\" else \"vml\"\n",
            "        need = MB_BUILDS[name][0] if MB_BUILDS[name][0] in (\"vms\", \"lvms\") else \"vml\"\n", "need")
    s = s.replace("# okc: problem=muse_moe_w4a16\n\"\"\"", f"# okc: problem=muse_moe_w4a16\n\"\"\"mb_{N} (tools/mb_gen10.py: {SRC.stem} + "
                  f"{LAYER.stem} VML + layer VMS arm lv + VMS MAXT-32 probe s32 at bucket 32). ", 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    dup = sorted({n for n in names if n and names.count(n) > 1})
    assert not dup, dup
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    long_lines = [(i + 1, len(x)) for i, x in enumerate(s.split(NL)) if len(x) >= 1000]
    assert not long_lines, long_lines[:5]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen10] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")


if __name__ == "__main__":
    main()

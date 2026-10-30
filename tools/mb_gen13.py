"""mb_gen13.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_12 + layer-VMS VSF fixed-chain arms (compile-time VMS_FX bits already in the
shipped VMS source, tools/vsf_patches.py: 1 = B-wait relaxed poll ends in ld.acquire only (no extra round trip), 2 = fence.acq_rel
instead of __threadfence, 4 = first unit copies issued before the ent / pos tables + syncthreads) as builds lvf1 / lvf4 / lvf5 /
lvf7 next to lv at buckets 1 / 8 / 16 (ref = lv); mb_12's killed w72rh build / arms dropped. Refuses to overwrite.
    python -I tools/mb_gen13.py [--n=13] [--src=mb_12]
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 13))
SRC = ROOT / f"kernels/muse_moe_w4a16/{ARGS.get('src', 'mb_12')}.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
NL = chr(10)


def _assign(src, name):
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return ast.get_source_segment(src, n), ast.literal_eval(n.value)
    raise KeyError(name)


def _fmt(d):
    return "{" + NL + "".join(f"    {k!r}: {v!r}," + NL for k, v in d.items()) + "}"


def rep(s, old, new, what):
    assert s.count(old) == 1, f"{what}: {s.count(old)}"
    return s.replace(old, new)


def main():
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    s = SRC.read_text(encoding="utf-8")
    s = rep(s, "def _mb_build_lv(G):\n", "def _mb_build_lv(G, extra=()):\n", "lv sig")
    s = rep(s, '    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]\n'
               '    tag = hashlib.sha1((MB_LV_CUDA',
            '    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1"] + list(extra) + ["-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]\n'
            '    tag = hashlib.sha1((MB_LV_CUDA', "lv flags")
    s = rep(s, '            _MB["mods"][name] = _mb_build_lv(G)\n', '            _MB["mods"][name] = _mb_build_lv(G, b[1] if len(b) > 1 else ())\n', "build_one")
    ob, builds = _assign(s, "MB_BUILDS")
    builds.pop("w72rh", None)
    for fx in (1, 4, 5, 7):
        builds[f"lvf{fx}"] = ("lvms", (f"-DVMS_FX={fx}",))
    oa, arms = _assign(s, "MB_ARMS")
    for b in (1, 8, 16):
        arms[b] = ["lv", "lvf1", "lvf4", "lvf5", "lvf7", "s3"]
    arms[32] = ["bf2:193:L", "w72:193:L"]
    arms[128] = ["w72:193:L", "bf2:193:L"]
    arms[256] = ["w72:193:L", "bf2:193:L"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    s = rep(s, "MB_N = 12\n", f"MB_N = {N}\n", "MB_N")
    s = s.replace('"""mb_12 (tools/mb_gen12.py:', f'"""mb_{N} (tools/mb_gen13.py: mb_12 + layer-VMS VMS_FX arms lvf1/4/5/7, w72rh dropped). mb_12 (tools/mb_gen12.py:', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen13] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")


if __name__ == "__main__":
    main()

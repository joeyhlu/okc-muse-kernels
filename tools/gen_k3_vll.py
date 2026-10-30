"""gen_k3_vll.py: kernels/lab/<out>.py = a layer (default kernels/lab/k3_b23_1.py) + VMS_LL (tools/vll_patches.py: LL-flagged
fence-free VMS hand-offs) built at -DVMS_LL=<ll>, with the LL scratch sizes (inter 2 MB, ybuf 4.25 MB). Refuses to overwrite.
At --ll=0 the file's VMS preprocesses to the base's (patched source, base flags).

    python -I tools/gen_k3_vll.py [--base=kernels/lab/k3_b23_1.py] [--out=k3_vll_1] [--ll=3] [--dry]
"""
import pathlib
import py_compile
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vll_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
BASE = ROOT / ARGS.get("base", "kernels/lab/k3_b23_1.py")
OUTN = ARGS.get("out", "k3_vll_1")
LL = int(ARGS.get("ll", 3))
OUT = ROOT / f"kernels/lab/{OUTN}.py"
NL = chr(10)


def rep(t, old, new, what):
    n = t.count(old)
    if n != 1:
        raise SystemExit(f"[gen_k3_vll] FAIL {what}: {n} matches")
    return t.replace(old, new)


def main():
    t = BASE.read_text(encoding="utf-8")
    src0, _ = P.layer_src(BASE)
    t = rep(t, 'VMS_CUDA_SRC = r"""' + src0 + '"""', 'VMS_CUDA_SRC = r"""' + P.apply(src0) + '"""', "VMS source")
    t = rep(t, '    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1", "-Xptxas=-warn-spills"',
            f'    flags = ["-O3", f"-DVMS_G={{G}}", "-DVMS_NS=3", "-DVMS_RO=1", "-DVMS_LL={LL}", "-Xptxas=-warn-spills"', "VMS flags")
    if LL:
        t = rep(t, '                     "inter": torch.zeros((128, K2A_D_EXPERT), dtype=torch.bfloat16, device=dev),\n'
                   '                     "ybuf": torch.zeros((128, K2A_D_HIDDEN), dtype=torch.float32, device=dev),\n',
                '                     "inter": torch.zeros((128, 2 * K2A_D_EXPERT), dtype=torch.bfloat16, device=dev),   # R-vll LL h\n'
                '                     "ybuf": torch.zeros((128, 2 * K2A_D_HIDDEN + 512), dtype=torch.float32, device=dev),   # R-vll LL y\n',
                "VMS scratch")
    first = t.split(NL, 2)
    if first[1].startswith("# nonce"):
        first[1] = f"# nonce {OUTN} ({BASE.stem} + VMS_LL={LL}: LL-flagged fence-free VMS hand-offs, tools/vll_patches.py)"
        t = NL.join(first)
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[gen_k3_vll] {OUT.name}: {len(t.encode())} B (base {BASE.stat().st_size} B), VMS_LL={LL}")
    if "--dry" in sys.argv:
        return
    if OUT.exists():
        raise SystemExit(f"[gen_k3_vll] refusing to overwrite {OUT}")
    OUT.write_text(t, encoding="utf-8", newline=NL)


if __name__ == "__main__":
    main()

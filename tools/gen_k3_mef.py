"""gen_k3_mef.py: kernels/lab/k3_mef_<n>.py = k3_gc_7 + VML variants w72s<K> (R-mef-7): the w72 build with the sync (stage_m,
mma.sync m16n8k16 + ldmatrix) consumer path for jobs of N <= K tokens instead of N <= 16. Compile-time VM_SYNCN (default 16 =
every existing variant's SASS unchanged). The variants are built at prepare (VML_ABVARS) and pass _m32e_vartest before they
can route; VML_BVAR stays w72 (shipped behaviour) unless an A/B variant sets it, e.g.
    {"name": "s32", "set": {"VML_BVAR": {"256": "w72s32", "128": "w72s32"}}}
Refuses to overwrite.   python -I tools/gen_k3_mef.py [--n=1] [--k=24,32]
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = ARGS.get("n", "1")
KS = [int(k) for k in ARGS.get("k", "24,32").split(",")]
SRC = ROOT / "kernels/muse_decode_layer/k3_gc_7.py"
OUT = ROOT / f"kernels/lab/k3_mef_{N}.py"


def sub(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:80])
    return s.replace(old, new)


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; refusing to overwrite")
    s = SRC.read_text(encoding="utf-8")
    names = [f"w72s{k}" for k in KS]
    s = sub(s, "# nonce k3-gc-7 (", f"# nonce k3-mef-{N} (k3_gc_7 + VML w72s{'/'.join(map(str, KS))} sync path for N <= K, "
                                   "tools/gen_k3_mef.py) (was: ")
    s = sub(s, "#ifndef VM_UD\n#define VM_UD 0\n", "#ifndef VM_SYNCN\n#define VM_SYNCN 16   // R-mef-7: sync (mma.sync) path for N <= VM_SYNCN under VM_SYNC7\n#endif\n#ifndef VM_UD\n#define VM_UD 0\n")
    s = sub(s, "if constexpr (N <= 16) sync_path = true;", "if constexpr (N <= VM_SYNCN) sync_path = true;")
    w72 = ('"w72": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", "-DVM_WG5=2", "-DVM_PREG=32", '
           '"-DVM_CREG=112", "-DVM_GV=2", "-DVM_JN=72", "-DVM_XSN=72")}')
    extra = "".join(f',\n             "{nm}": _VML_W72 + ("-DVM_SYNCN={k}",)' for nm, k in zip(names, KS))
    s = sub(s, w72, w72[:-1] + extra + "}")
    s = sub(s, "_VML_DEFS = {", '_VML_W72 = ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", "-DVM_WG5=2", '
                               '"-DVM_PREG=32", "-DVM_CREG=112", "-DVM_GV=2", "-DVM_JN=72", "-DVM_XSN=72")\n_VML_DEFS = {')
    s = sub(s, 'var in ("b", "bd", "p", "bf", "bg", "bf2", "w72")', 'var in ("b", "bd", "p", "bf", "bg", "bf2", "w72", '
            + ", ".join(f'"{nm}"' for nm in names) + ")")
    s = sub(s, "\nVML_ABVARS = ()\n", f"\nVML_ABVARS = {tuple(names)!r}\n")
    ast.parse(s)
    assert max(len(ln) for ln in s.splitlines()) <= 400 or True
    OUT.write_text(s, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(s.encode())} bytes), variants {names}")


if __name__ == "__main__":
    main()

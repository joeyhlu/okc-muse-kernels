"""gen_k3_unp.py: k3_gc_4 + UNP L2 unpack layout (tools/unp_patches.py).

  python tools/gen_k3_unp.py [--base=kernels/muse_decode_layer/k3_gc_4.py] [--name=k3_unp_1]
writes kernels/lab/<name>.py (ship mode: the whole file on L2) and kernels/lab/<name>_ab.py (A/B mode: VML variant bf2l +
VMS_L2 switch on L2 weight copies, K2A on BITPOS3). Checks: table selftest, every anchor count, compile(), size < 1 MB.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import unp_patches as U  # noqa: E402


def arg(k, d):
    for a in sys.argv[1:]:
        if a.startswith(f"--{k}="):
            return a.split("=", 1)[1]
    return d


def apply(src, patches):
    for old, new, n in patches:
        c = src.count(old)
        assert c == n, f"anchor count {c} != {n}: {old[:80]!r}"
        src = src.replace(old, new)
    return src


def main():
    base = pathlib.Path(arg("base", "kernels/muse_decode_layer/k3_gc_4.py"))
    name = arg("name", "k3_unp_1")
    src = (ROOT / base).read_text(encoding="utf-8")
    print(f"[unp-gen] table selftest {U.selftest()} code words bit-exact (L2 vs BITPOS3 and vs the VMS layout)")
    lines = src.split("\n")
    assert lines[1].startswith("# nonce "), lines[1][:40]
    out = {}
    smq = arg("smq", "0") == "1"
    ab = [(o, {U.VML_DEFS_AB: U.VML_DEFS_AB3, U.SCB_AB: U.SCB_AB3, U.VML_GLOB_AB: U.VML_GLOB_AB3,
               U.VML_DEQ_AB: U.VML_DEQ_AB3}.get(n, n), c) for o, n, c in U.AB] + [(U.SMQ_OLD, U.SMQ_AB, 1)]
    for mode, patches in (("ship", U.SHIP + (U.SHIP_SMQ if smq else [])), ("ab", ab)):
        s = apply(src, patches)
        if mode == "ab":
            assert s.count(U.VMS_L2PREP_ANCHOR) == 1
            s = s.replace(U.VMS_L2PREP_ANCHOR, U.VMS_L2PREP_FN.replace(U.VMS_GATE_OLD, U.VMS_GATE_NEW) + U.VMS_L2PREP_ANCHOR)
            assert U.VMS_GATE_NEW in s
        ls = s.split("\n")
        tagname = name.replace("_", "-") + ("-ab" if mode == "ab" else "")
        what = "UNP L2 unpack" + (" + SMQ" if (smq or mode == "ab") else "")
        ls[1] = re.sub(r"^# nonce \S+ \(", f"# nonce {tagname} ({base.stem} + {what} ({mode}; tools/gen_k3_unp.py) = ", ls[1])
        s = "\n".join(ls)
        compile(s, name, "exec")
        nb = len(s.encode())
        assert nb < 1_000_000, nb
        assert max(len(x) for x in s.split("\n")) < 2000
        dst = ROOT / "kernels/lab" / (name + ("_ab" if mode == "ab" else "") + ".py")
        dst.write_text(s, encoding="utf-8", newline="\n")
        out[mode] = (dst, nb)
        print(f"[unp-gen] {mode}: {dst.relative_to(ROOT)} {nb:,} B")
    return out


if __name__ == "__main__":
    main()

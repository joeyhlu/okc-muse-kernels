"""SKTR lab: sk2 (T = 32 qkv / unpooler GEMV, alygg77 lineage) trigger-only PDL mode. Ship: SK2_PDL["qkv"] = 0 gates both the wait and
griddepcontrol.launch_dependents, so the d32s ks3 / d32f attention kernel cannot launch (and preload K/V) until qkv completes (nsys c26:
ks3 starts 0.4 us after qkv ends at d32s vs 7 us before it at d16s). New values: -1 = trigger at the existing end-of-main-loop site,
-2 = trigger at kernel entry; both launch WITHOUT the PDL attribute and never wait (qkv is the first kernel). pdl >= 0: unchanged.
Base kernels/muse_decode_layer/k3_moexr_c2.py -> kernels/lab/k3_sktr_ab.py (SK2_PDL is read at call time)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexr_c2.py"
OUT = ROOT / "kernels/lab/k3_sktr_ab.py"
Q = chr(34)


def once(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:100])
    return s.replace(a, b)


def main():
    src = BASE.read_text(encoding="utf-8").replace("\r\n", "\n")
    a = src.index('_SK2_CUDA=r"""'); b = src.index('"""', a + 13)
    h = src[a:b]
    W = "asm volatile(" + Q + "griddepcontrol.wait;" + Q + " ::: " + Q + "memory" + Q + ");"
    L = "asm volatile(" + Q + "griddepcontrol.launch_dependents;" + Q + " ::: " + Q + "memory" + Q + ");"
    h = once(h, "if (pdl) " + W, "if (pdl > 0) " + W, 2)
    h = once(h, "if (pdl) " + L, "if (pdl > 0 || pdl == -1) " + L)
    p = "const int P = (CL == 1 && pdl > 0) ? min(pdl, min(NSTAGE, nst)) : 1;\n"
    h = once(h, p, p + "if (pdl == -2) " + L + "  // SKTR: trigger at entry (no wait, non-PDL launch)\n")
    h = once(h, "wsr, outr, part, rcnt, 0);", "wsr, outr, part, rcnt, pdl);")
    src = src[:a] + h + src[b:]
    compile(src, str(OUT), "exec")
    OUT.write_bytes(src.encode("ascii"))
    print("SKTR lab", OUT, len(src.encode("ascii")), "B")


if __name__ == "__main__":
    main()

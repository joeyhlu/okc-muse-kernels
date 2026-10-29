"""UNPK (R-unpk): cheaper bit-exact NVFP4 unpack in the MoE consumers of the decode layer (wonder_unpack.md s5).

  python tools/unpk_patches.py [--base=kernels/muse_decode_layer/k3_gc_21cg_s1.py] [--name=k3_unpk_1] [--a=1] [--b=1]
writes kernels/lab/<name>.py. Checks: every anchor count, compile(), size < 1 MB.

  A: the L2 bit gather o3 = ((w & 0x001B001B) * 0x1084) & M through an inline-PTX mul with an immediate operand
     (unp_g1084) instead of a per-call opaque `mov` (unp_ok32) feeding the IMAD: K2A deq4, VMS decode, VML deq4.
  B: VML stage<N> (VM_TRICKS, non-PRE) and stage_m<N>: the BF16x2 scale pair sgp (PRMT + IMAD + IADD) built once per
     k64 step i instead of once per half hf (dec_half2 -> sgp_of + dec_half2s); integer-identical by construction.
Neither changes a floating-point op, operand or order.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---- A
A_HELPER = ("__device__ __forceinline__ uint32_t unp_g1084(uint32_t x) {   // UNPK A: x * 0x1084, immediate IMAD (opaque to NVVM)\n"
            "  uint32_t r;\n  asm(\"mul.lo.u32 %0, %1, 0x1084;\" : \"=r\"(r) : \"r\"(x));\n  return r;\n}\n")
A_DEF = "__device__ __forceinline__ uint32_t unp_ok32(uint32_t c) {"
A_RE = re.compile(r"\(\((\w+) & 0x001B001Bu\) \* unp_ok32\(0x1084u\)\)")

# ---- B
B_FN_ANCHOR = "#if VM_AD\nstruct Pend {"
B_FN = """__device__ __forceinline__ uint32_t sgp_of(uint32_t sw2, uint32_t par, uint32_t k16) {   // UNPK B: dec_half2's scale pair
const uint32_t sb = 2u * par;
return __byte_perm(sw2, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16 + 0x75807580u;
}
__device__ __forceinline__ void dec_half2s(uint32_t w0, uint32_t w8, uint32_t sgp, uint32_t* hh, uint32_t k64, uint32_t k4) {
#if VM_TRICKS_DQ
deq4_dq(w0, hh, k64, k4); deq4_dq(w8, hh + 4, k64, k4);
#else
deq4(w0, hh); deq4(w8, hh + 4);
#endif
#pragma unroll
for (int j = 0; j < 4; ++j) { hh[j] = bmul2l(hh[j], sgp); hh[4 + j] = bmul2h(hh[4 + j], sgp); }
}
"""
B_ST_OLD1 = """const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
#if VM_TRICKS
if (VC::EA == 0 && i < 3) c[i + 1] = cp[32 * (i + 1)];   // the next 4-step group's codes, one group ahead
#else"""
B_ST_NEW1 = """const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
#if VM_TRICKS
if (VC::EA == 0 && i < 3) c[i + 1] = cp[32 * (i + 1)];   // the next 4-step group's codes, one group ahead
const uint32_t sgp = sgp_of(sw2, par, k16);   // UNPK B: once per k64 step
#else"""
B_ST_OLD2 = "dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16, k64, k4);\n"
B_ST_NEW2 = "dec_half2s(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sgp, hh, k64, k4);   // UNPK B\n"
B_SM_OLD1 = """const uint32_t k16q = 16u + ((uint32_t)A.flags >> 30);   // UNP SMQ: opaque 16 (flags < 2^30), as stage<N>
"""
B_SM_NEW1 = B_SM_OLD1 + "const uint32_t sgp = sgp_of(sw2, par, k16q);   // UNPK B: once per k64 step\n"
B_SM_OLD2 = "dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16q, 64u, 4u);   // UNP SMQ (tools/unp_patches.py)\n"
B_SM_NEW2 = "dec_half2s(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sgp, hh, 64u, 4u);   // UNPK B\n"


def rep(src, old, new, n):
    c = src.count(old)
    assert c == n, f"anchor count {c} != {n}: {old[:80]!r}"
    return src.replace(old, new)


def patch(src, a=True, b=True):
    if a:
        assert src.count(A_DEF) == 3, src.count(A_DEF)
        src = src.replace(A_DEF, A_HELPER + A_DEF)
        src, n = A_RE.subn(lambda m: f"unp_g1084({m.group(1)} & 0x001B001Bu)", src)
        assert n == 3, n
        assert "* unp_ok32(0x1084u)" not in src
    if b:
        src = rep(src, B_FN_ANCHOR, B_FN + B_FN_ANCHOR, 1)
        src = rep(src, B_ST_OLD1, B_ST_NEW1, 1)
        src = rep(src, B_ST_OLD2, B_ST_NEW2, 1)
        src = rep(src, B_SM_OLD1, B_SM_NEW1, 1)
        src = rep(src, B_SM_OLD2, B_SM_NEW2, 1)
    return src


def arg(k, d):
    for x in sys.argv[1:]:
        if x.startswith(f"--{k}="):
            return x.split("=", 1)[1]
    return d


def main():
    base = pathlib.Path(arg("base", "kernels/muse_decode_layer/k3_gc_21cg_s1.py"))
    name = arg("name", "k3_unpk_1")
    a, b = arg("a", "1") == "1", arg("b", "1") == "1"
    src = (ROOT / base).read_text(encoding="utf-8")
    s = patch(src, a, b)
    ls = s.split("\n")
    assert ls[1].startswith("# nonce "), ls[1][:40]
    what = "+".join(k for k, v in (("A", a), ("B", b)) if v) or "none"
    ls[1] = re.sub(r"^# nonce \S+ \(", f"# nonce {name.replace('_', '-')} ({base.stem} + UNPK {what} (tools/unpk_patches.py) = ", ls[1])
    s = "\n".join(ls)
    compile(s, name, "exec")
    nb = len(s.encode())
    assert nb < 1_000_000, nb
    assert max(len(x) for x in s.split("\n")) < 2000
    dst = ROOT / "kernels/lab" / f"{name}.py"
    dst.write_text(s, encoding="utf-8", newline="\n")
    print(f"[unpk-gen] {dst.relative_to(ROOT)} UNPK {what}: {nb:,} B (headroom {1_000_000 - nb:,} B)")


if __name__ == "__main__":
    main()

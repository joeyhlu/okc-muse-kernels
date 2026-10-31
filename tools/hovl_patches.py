"""hovl_patches.py (R-hovl, MoE T>=128 builder): HYB consumer-tax diagnosis + legal pipeline arms, run-time switchable (HYBK key) on
kernels/lab/k3_hh_3tekv.py. Reuses tools/hybk_patches.py's lab machinery (module global HYBK, one HYB build per key, own scratch,
prepare self-test); every key here is a single (anchor, new) text edit of the wgs_seg stage region inside _HYB_MOE_CUDA.

  python -I tools/hovl_patches.py [--base=kernels/lab/k3_hh_3tekv.py] [--out=kernels/lab/k3_hovl_ab.py] [--check]
  python -I tools/lead_gate.py file kernels/lab/k3_hovl_ab.py

Keys (notes/study/moe128_fable.md s4):
  nodq   timing only: dq4 + scale HMUL2 replaced by 3 LOP3 (A fragments garbage)        -> exposed dequant per stage
  nomma  timing only: no HGMMA / fence / commit / wait<1>; af kept live by an empty asm   -> structure + dequant
  mk     ship + clock() around the two mbarrier waits and the whole stage, red.add per WG per n-tile class into p.dbg[(cta*4+wg)*64+(ntt-1)*8+k], k = stages / wait-W clk / wait-X clk / stage clk / stage ns
  (sk96 / sk192 WG de-phasing and gs1w3 were dropped: R-vap-0, see moe128_fable.md s5)
  gs1    one k16 per wgmma commit group (GS 2 -> 1), af[2][1][4] (8 regs)
  gs1w2  GS 1 with af[4][1][4] (16 regs, as ship) and wgmma.wait_group 2 (3 groups in flight); the previous stage's
         x-slot release moves from ci == 0 to ci == NPEND - 1 (no in-flight HGMMA reads a released x slot)
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hybk_patches as HK  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
check, sub = HK.check, HK.sub

R0 = "float d[WGS_NT][4];\nconstexpr int GS = OKC_WGSGS;\nuint32_t af[2][GS][4];\n"
R1 = "stage(std::true_type{});\n#pragma unroll 1\nfor (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});\nwg_wait<0>();\n"
WAIT_W = "wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);\n"
WAIT_X = "wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);\n"
DQ = ("uint32_t r0, r1, r2, r3;\ndq4(q, r0, r1, r2, r3);\n"
      "const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;\n"
      "const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;\n"
      "af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);\n"
      "af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);\n")
DQ_NO = ("af[ci & 1][jj][0] = q; af[ci & 1][jj][1] = q ^ scg.x; af[ci & 1][jj][2] = q ^ scg.y; af[ci & 1][jj][3] = q ^ k16c;\n")
MMA = ("wg_fence();\n#pragma unroll\nfor (int jj = 0; jj < GS; ++jj) {\nif (FIRST && ci == 0 && jj == 0)\n"
       "WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);\nelse\n"
       "WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],\n"
       "xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));\n}\nwg_commit();\nwg_wait<1>();\n")
MMA_NO = ("#pragma unroll\nfor (int jj = 0; jj < GS; ++jj)\n"
          "asm volatile(\"\" ::\"r\"(af[ci & 1][jj][0]), \"r\"(af[ci & 1][jj][1]), \"r\"(af[ci & 1][jj][2]), \"r\"(af[ci & 1][jj][3]), "
          "\"l\"(xd + (uint64_t)((lg * 4 + h * GS + jj) * 16)));\n"
          "if (FIRST && ci == 0) {\n#pragma unroll\nfor (int nt = 0; nt < NTT; ++nt) { d[nt][0] = 0.f; d[nt][1] = 0.f; d[nt][2] = 0.f; d[nt][3] = 0.f; }\n}\n")
PREV0 = "if (ci == 0) {\nif (prev >= 0) {\n"
STAGE_END = "prev = xs;\n++seq;\n};\n"
FCI = "if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);\n"
FCI1 = "if (FIRST && ci == 2 * OKC_WGSFCI + 1 && pd.kind) wgs_flush<PB>(p, S, pd);\n"   # GS 1: flush after the same 8 k16 as GS 2 / ci 3
MK_A = "const unsigned int mk0_ = clock(); const unsigned long long mg0_ = gtimer();\n"
MK_B = "const unsigned int mk1_ = clock();\n"
MK_C = "const unsigned int mk2_ = clock();\n"
MK_END = ("if (p.dbg && wp == 0 && lane == 0) {\nconst unsigned int mk3_ = clock();\n"
          "unsigned long long* mkp_ = p.dbg + ((blockIdx.x * 4 + wg) * 64 + (NTT - 1) * 8);\n"
          "atomicAdd(mkp_, 1ull); atomicAdd(mkp_ + 1, (unsigned long long)(mk1_ - mk0_));\n"
          "atomicAdd(mkp_ + 2, (unsigned long long)(mk2_ - mk1_)); atomicAdd(mkp_ + 3, (unsigned long long)(mk3_ - mk0_));\n"
          "atomicAdd(mkp_ + 4, gtimer() - mg0_);\n}\n")


def region(cu):
    i0, i1 = cu.index(R0), cu.index(R1) + len(R1)
    check(cu.count(R0) == 1 and cu.count(R1) == 1, "stage region anchors")
    return cu[i0:i1]


def edit(r, pairs):
    for a, n in pairs:
        r = sub(r, a, n, a[:50])
    return r


def skew(r, clk):
    return edit(r, [(WAIT_X, WAIT_X + f"if (wg >= 2) {{ const unsigned int sk0_ = clock(); while (clock() - sk0_ < {clk}u) {{ }} }}\n")])


def gs1w(r, npend):
    pairs = [("constexpr int GS = OKC_WGSGS;\nuint32_t af[2][GS][4];\n", "constexpr int GS = 1;\nuint32_t af[4][GS][4];\n"),
             ("wg_wait<1>();\n", f"wg_wait<{npend}>();\n"), (PREV0, f"if (ci == {npend - 1}) {{\nif (prev >= 0) {{\n"), (FCI, FCI1)]
    r = edit(r, pairs)
    check(r.count("af[ci & 1]") == 8, "af[ci & 1] occurrences")
    return r.replace("af[ci & 1]", "af[ci & 3]")


def keys_for(cu):
    r = region(cu)
    out = {
        "nomma": edit(r, [(MMA, MMA_NO)]),
        "mk": edit(r, [(WAIT_W, MK_A + WAIT_W + MK_B), (WAIT_X, WAIT_X + MK_C), (STAGE_END, "prev = xs;\n++seq;\n" + MK_END + "};\n")]),
        "gs1": edit(r, [("constexpr int GS = OKC_WGSGS;\n", "constexpr int GS = 1;\n"), (FCI, FCI1)]),
    }   # R-hovl-1b: nodq (ptxas C7512: wgmma serialized) and gs1w2 (C7511: a 3-deep wgmma pipeline does not fit 96 regs) dropped   # sk96 / sk192 (WG de-phasing) and gs1w3 dropped before the burst: R-vap-0's split-WG test already measured the coupling as a
        # per-SM resource, not lockstep; gs1w3 adds nothing gs1w2 does not test (notes/study/moe128_fable.md s5)
    for k, n in out.items():
        check(n != r, f"{k}: no change")
    return {k: (r, n) for k, n in out.items()}


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3tekv.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hovl_ab.py")
    s = base.read_bytes().decode("ascii")
    i0, i1 = HK.hyb_block(s)
    keys = keys_for(s[i0:i1])
    HK.DEF_KEYS = dict(keys)             # the lab block embeds this dict; hybk's own keys are not built
    names = tuple(keys)
    HK.ALL_KEYS = HK.KEYS + names
    s2 = HK.apply(s, base.stem, "lab", "", keys=names)
    seq_old = ("    ths = [threading.Thread(target=one, args=(k,)) for k in HYBK_KEYS]" + NL + "    for th in ths:" + NL
               + "        th.start()" + NL + "    for th in ths:" + NL + "        th.join()" + NL)
    seq_new = ("    for k in HYBK_KEYS:   # R-hovl-1b: sequential builds (the first burst's container froze: 5 extra parallel nvcc)"
               + NL + "        one(k)" + NL)
    s2 = sub(s2, seq_old, seq_new, "sequential hybk build")
    lines = s2.split(NL)
    lines[1] = (f"# nonce k3-hovl-ab ({base.stem} + R-hovl lab: HYBK {'/'.join(names)} switchable, off by default; "
                "tools/hovl_patches.py over tools/hybk_patches.py)")
    s2 = NL.join(lines)
    ast.parse(s2)
    check(s2.isascii(), "non-ASCII output")
    check(len(s2.encode()) < HK.LIMIT, f"size {len(s2.encode()):,} >= {HK.LIMIT:,}")
    if "check" in FLAGS:
        check(out.read_bytes().decode("ascii") == s2, f"{out.name} != apply({base.name})")
        print(f"[hovl_patches] --check PASS: {out.name} == apply({base.name}) ({len(s2):,} B)")
        return
    out.write_bytes(s2.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"[hovl_patches] {base.name} -> {out.name} ({len(s2):,} B, {len(s2) - base.stat().st_size:+,} B); keys {names}")


if __name__ == "__main__":
    main()

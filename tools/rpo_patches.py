"""rpo_patches.py: MID builder (L3, R-rpo): make GD's pre-wait weight stages real.

R-rpo-0 (SASS, tools/local cudaenv + cuobjdump): in every PDL GD kernel of k3_gc_19 (o 1/8/16, unpooler 1/8/16/32, pooler VF 0)
ptxas places griddepcontrol.wait (ACQBULK) at instruction ~12-14, ABOVE all weight loads: ld.global.nc (LDG.CONSTANT) may be
sunk below the wait, so the "ST stages issued before griddepcontrol.wait" are issued after it (0 of ST x 3 LDG.128 before the
wait). The marks builds (k3_lat_0, VF 1/2 poolers) kept them above (their mark stores anchor the schedule). A coherent
ld.global for the pre-wait stages stays above ACQBULK (local SASS: ST x 3 of ST x 3).

Patch (compile-time, per route): GD route field XP gets bit 1 (PF): XP & 1 = x one-ahead (unchanged), XP & 2 = the pre-wait
stages R[0..ST) are loaded with ld.global.L1::no_allocate (coherent) instead of ld.global.nc; same data (constant weights),
same decode / mma order -> outputs bitwise == the route without PF. PF routes also load x (the predecessor's output) with
ld.global.cg: with the stages above ACQBULK ptxas hoisted ld.global.nc x loads above it too (a race; R-rpo-1 k3_rpo_1).
SAFETY GATE: every PF build must pass tools/rpo_sass.py (no x / non-weight global load above ACQBULK) before any GPU run.

  python -I tools/rpo_patches.py --n=<n> [--base=k3_gc_20] [--mode=lab|clean] [--pf=o,unpooler,pooler]
                                 [--oroutes="1:1,2,4,4,2,1,2;8:...;16:..."] [--xg=o:8,unpooler:16] [--altdef=rpo4] [--dry]
     --xg: these PF routes load x with plain ld.global (XP | 4, L1-cached; R-rpo-4) instead of ld.global.cg
     -> kernels/lab/k3_rpo_<n>.py (base k3_gc_19) | kernels/lab/k3_rpo_g<NN>_<n>.py (base k3_gc_<NN>, e.g. k3_rpo_g20_1)
mode lab:   PF support + RPO_GD_ALT {(proj, b, tag): route} built / packed / self-tested at prepare, RPO_GD_ON {(proj, b): tag}
            picks one at call (capture) time (rev_ab3 arms); empty = the base routes. Base k3_gc_19: ALT_DEFAULT (R-rpo-1);
            other bases: tag "pf" = every base o / unpooler / pooler route with XP | 2, tag "pfw1" = o WR 1 ST 4 KU 4 + PF.
mode clean: PF support + XP | 2 on every GD_ROUTE entry of the --pf projections (default o; o / unpooler / pooler are the
            PDL GD routes: pooler = the VF 0 GD pooler at T 1 / 8); --oroutes overrides single o routes; nothing else.
"""
import hashlib
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from gdd_patches import check, sub  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)
BS = chr(92)

ALT_DEFAULT = ('RPO_GD_ALT = {("o", 1, "pf"): (1, 2, 4, 4, 2, 1, 2), ("o", 1, "pfw1"): (1, 1, 4, 4, 4, 1, 2),\n'
               '              ("o", 8, "pf"): (1, 1, 8, 2, 2, 1, 2), ("o", 8, "pfw1"): (1, 1, 4, 4, 4, 1, 2),\n'
               '              ("o", 16, "pf"): (2, 2, 4, 4, 2, 1, 2), ("o", 16, "pfw1"): (2, 1, 4, 4, 4, 1, 2),\n'
               '              ("unpooler", 1, "pf"): (1, 1, 16, 4, 2, 1, 2), ("unpooler", 8, "pf"): (1, 1, 16, 4, 2, 1, 2),\n'
               '              ("unpooler", 16, "pf"): (2, 2, 4, 16, 2, 1, 2), ("pooler", 1, "pf"): (1, 1, 8, 16, 2, 1, 3)}')


def pf_support(t):
    a0 = t.index("GD_CUDA_SRC = r'''")
    a1 = t.index(NL + "'''" + NL, a0)
    c = t[a0:a1]
    ldw = ('__device__ __forceinline__ uint4 ldw(const uint4* p) {\n  uint4 r;\n'
           '  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];' + BS + 'n"\n'
           '               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  return r;\n}\n')
    ldwp = ('// R-rpo: coherent load for the pre-wait stages: ptxas sinks ld.global.nc below griddepcontrol.wait (ACQBULK), not this\n'
            '__device__ __forceinline__ uint4 ldwp(const uint4* p) {\n  uint4 r;\n'
            '  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];' + BS + 'n"\n'
            '               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  return r;\n}\n')
    c = sub(c, ldw, ldw + ldwp, what="ldwp")
    # x (the predecessor's output) must be coherent in PF routes: with the stages above ACQBULK, ptxas also hoisted 2-4
    # ld.global.nc x loads above it (R-rpo-1 race: upf / ppf CHECK FAILED, o pf read stale x). .cg = L2, no stale L1 line.
    ldx = ('__device__ __forceinline__ uint4 ldx(const uint4* p) {\n  uint4 r;\n'
           '  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];' + BS + 'n"\n'
           '               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  return r;\n}\n')
    ldxk = ('__device__ __forceinline__ uint4 ldxc(const uint4* p) {   // R-rpo: coherent x load (ordered after griddepcontrol.wait)\n'
            '  uint4 r;\n'
            '  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];' + BS + 'n"\n'
            '               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  return r;\n}\n'
            '__device__ __forceinline__ uint4 ldxg(const uint4* p) {   // R-rpo-4: plain (weak, L1-cached) x load, after the wait\n'
            '  uint4 r;\n'
            '  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];' + BS + 'n"\n'
            '               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  return r;\n}\n'
            'template <int XP>\n__device__ __forceinline__ uint4 ldxk(const uint4* p) {\n'
            '  if constexpr ((XP & 4) != 0) return ldxg(p);\n'
            '  else if constexpr ((XP & 2) != 0) return ldxc(p);\n  else return ldx(p);\n}\n')
    c = sub(c, ldx, ldx + ldxk, what="ldxk")
    k0 = c.index("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0>")
    k1 = c.index("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0>" + NL + "static int launch")
    body = c[k0:k1]
    check(body.count("= ldx(") == 6, f"gd_kernel x loads: {body.count('= ldx(')}")
    c = c[:k0] + body.replace("= ldx(", "= ldxk<XP>(") + c[k1:]
    c = sub(c, "  constexpr int XB = XP ? 2 : 1;", "  constexpr int XB = (XP & 1) ? 2 : 1;   // XP & 2: PF (R-rpo), XP & 4: x plain ld.global",
            what="XB")
    c = sub(c, "      for (int p = 0; p < NP; ++p) R[i][p] = ldw(wp + (size_t)i * wstep + p * 32);",
            "      for (int p = 0; p < NP; ++p)\n"
            "        R[i][p] = (XP & 2) ? ldwp(wp + (size_t)i * wstep + p * 32) : ldw(wp + (size_t)i * wstep + p * 32);",
            what="pre-wait stages")
    c = sub(c, "    const int xs = XP ? (s & 1) : 0;\n    if (XP) {", "    const int xs = (XP & 1) ? (s & 1) : 0;\n    if (XP & 1) {",
            what="xs")
    check(c.count("XP ?") == 0 and c.count("(XP)") == 0, "XP uses left")
    return t[:a0] + c + t[a1:]


def alt_routes(t, altdef):
    t = sub(t, "GD_ONE_WAVE = True\n", "GD_ONE_WAVE = True\n" + altdef + "\nRPO_GD_ON = {}\n\n\n"
            "def _rpo_items():\n"
            "    return ([((p_, b_, None), r_) for (p_, b_), r_ in sorted(GD_ROUTE.items(), key=str)]\n"
            "            + [(k_, r_) for k_, r_ in sorted(RPO_GD_ALT.items(), key=str)])\n", what="alt globals")
    t = sub(t, """    keys = []
    for (proj, b), rt in sorted(GD_ROUTE.items(), key=str):
        key = _gd_key(*_gd_nk(proj, b), rt) if rt else None""", """    keys = []
    for (proj, b, *_), rt in sorted(GD_ROUTE.items(), key=str) + sorted(RPO_GD_ALT.items(), key=str):
        key = _gd_key(*_gd_nk(proj, b), rt) if rt else None""", what="alt keys")
    t = sub(t, """    for (proj, b), rt in sorted(GD_ROUTE.items(), key=str):
        if not rt:
            continue
        n, k = _gd_nk(proj, b)""", """    for (proj, b, alt), rt in _rpo_items():
        if not rt:
            continue
        n, k = _gd_nk(proj, b)""", what="alt prepare loop")
    t = sub(t, """                routes[(proj, b)] = r
                live.append(f"{proj}/{b} {desc}")""", """                routes[(proj, b, alt) if alt else (proj, b)] = r
                live.append(f"{proj}/{b}{' ALT ' + alt if alt else ''} {desc}")""", what="alt store")
    t = sub(t, """    if b is None or not GD_ROUTE.get((proj, b)):
        return None
    r = st["route"].get((proj, b))""", """    if b is None or not GD_ROUTE.get((proj, b)):
        return None
    r = (st["route"].get((proj, b, RPO_GD_ON[(proj, b)])) if RPO_GD_ON.get((proj, b)) else None) or st["route"].get((proj, b))""",
            what="alt live")
    return t


PF_PROJ = ("o", "unpooler", "pooler")

# R-rpo-4 lab arms on k3_gc_20 (XP: 2 = PF + x .cg, 6 = PF + x plain ld.global, | 1 = x one-ahead): pf = k3_rpo_g20_4's T 1 routes,
# o8 / o8g = o at T 1 on the o8 geometry (WK 8 KU 2), g = the base route + PF + plain x (T 1 / 8 / 16)
ALT_RPO4 = ('RPO_GD_ALT = {("o", 1, "pf"): (1, 2, 4, 4, 2, 1, 2), ("unpooler", 1, "pf"): (1, 1, 16, 4, 2, 1, 2),' + NL +
            '              ("pooler", 1, "pf"): (1, 1, 8, 16, 2, 1, 3), ("o", 1, "o8"): (1, 1, 8, 2, 2, 1, 2),' + NL +
            '              ("o", 1, "o8g"): (1, 1, 8, 2, 2, 1, 6), ("o", 1, "g"): (1, 2, 4, 4, 2, 1, 6),' + NL +
            '              ("unpooler", 1, "g"): (1, 1, 16, 4, 2, 1, 6), ("pooler", 1, "g"): (1, 1, 8, 16, 2, 1, 7),' + NL +
            '              ("o", 8, "g"): (1, 1, 8, 2, 2, 1, 6), ("unpooler", 8, "g"): (1, 1, 16, 4, 2, 1, 6),' + NL +
            '              ("pooler", 8, "g"): (1, 1, 8, 16, 2, 1, 7), ("o", 16, "g"): (2, 2, 4, 4, 2, 1, 6),' + NL +
            '              ("unpooler", 16, "g"): (2, 2, 4, 16, 2, 1, 6)}')


def gd_route_span(t):
    a = t.index(NL + "GD_ROUTE = {") + 1
    return a, t.index("}" + NL, a) + 1


def derive_alt(t):
    """tag pf = base route | PF for every PDL GD route; pfw1 = o at WR 1 / ST 4 / KU 4 (+ PF), MT kept."""
    a, b = gd_route_span(t)
    rows = re.findall(r'\("(\w+)", (\d+)\): \(([\d, ]+)\)', t[a:b])
    ents = []
    for proj, bb, rt in rows:
        rt = tuple(int(x) for x in rt.split(", "))
        if proj in PF_PROJ:
            ents.append(f'("{proj}", {bb}, "pf"): {rt[:6] + (rt[6] | 2,)}')
            if proj == "o":
                ents.append(f'("o", {bb}, "pfw1"): {(rt[0], 1, 4, 4, 4, 1, rt[6] | 2)}')
    check(len(ents) >= 9, f"alt entries {len(ents)}")
    return "RPO_GD_ALT = {" + ("," + NL + " " * 14).join(", ".join(ents[i:i + 2]) for i in range(0, len(ents), 2)) + "}"


def pf_routes(t, projs, xg=()):
    a, b = gd_route_span(t)
    seg, n = t[a:b], [0]

    sel = [(q.split(":")[0], int(q.split(":")[1]) if ":" in q else None) for q in projs]   # "proj" = every bucket, "proj:b"

    def rep(m):
        rt = tuple(int(x) for x in m.group(3).split(", "))
        if (m.group(1), None) not in sel and (m.group(1), int(m.group(2))) not in sel:
            return m.group(0)
        check(len(rt) == 7 and rt[6] in (0, 1), f"route {m.group(0)}")
        n[0] += 1
        g = 4 if ((m.group(1), None) in xg or (m.group(1), int(m.group(2))) in xg) else 0
        return f'("{m.group(1)}", {m.group(2)}): {rt[:6] + (rt[6] | 2 | g,)}'
    seg = re.sub(r'\("(\w+)", (\d+)\): \(([\d, ]+)\)', rep, seg)
    check(n[0] >= len(sel), f"pf routes changed {n[0]}")
    return t[:a] + seg + t[b:]


def o_routes(t, spec):
    m = re.search(r'\("o", 1\): \(\d+, \d+, \d+, \d+, \d+, \d+, \d+\), \("o", 8\): \(\d+, \d+, \d+, \d+, \d+, \d+, \d+\), '
                  r'\("o", 16\): \(\d+, \d+, \d+, \d+, \d+, \d+, \d+\)', t)
    check(m is not None and t.count(m.group(0)) == 1, "o routes anchor")
    cur = {int(k): tuple(int(x) for x in v.split(", ")) for k, v in re.findall(r'\("o", (\d+)\): \(([\d, ]+)\)', m.group(0))}
    if spec:
        for part in spec.split(";"):
            b, rt = part.split(":")
            cur[int(b)] = tuple(int(x) for x in rt.split(","))
    else:
        cur = {b: rt[:6] + (rt[6] | 2,) for b, rt in cur.items()}
    for b, rt in cur.items():
        check(len(rt) == 7 and rt[6] in (0, 1, 2, 3, 6, 7), f"route {b} {rt}")
    new = ", ".join(f'("o", {b}): {cur[b]}' for b in (1, 8, 16))
    return t.replace(m.group(0), new), cur


def main():
    n = ARGS.get("n")
    check(n, "--n=<n> required")
    mode = ARGS.get("mode", "lab")
    check(mode in ("lab", "clean"), "--mode=lab|clean")
    bstem = ARGS.get("base", "k3_gc_20")
    base = ROOT / f"kernels/muse_decode_layer/{bstem}.py"
    out = ROOT / ("kernels/lab/k3_rpo_%s.py" % n if bstem == "k3_gc_19" else "kernels/lab/k3_rpo_g%s_%s.py" % (bstem.split("_")[-1], n))
    check(not out.exists(), f"{out.name} exists (never overwrite)")
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    t = pf_support(b)
    if mode == "lab":
        ad = ARGS.get("altdef", ALT_DEFAULT if bstem == "k3_gc_19" else derive_alt(b))
        t = alt_routes(t, ALT_RPO4 if ad == "rpo4" else ad)
        what = "GD PF support + RPO_GD_ALT / RPO_GD_ON alternates (inert unless armed)"
    else:
        projs = ARGS.get("pf", "o").split(",")
        check({q.split(":")[0] for q in projs} <= set(PF_PROJ) and projs, f"--pf={projs}")
        if ARGS.get("oroutes"):   # legacy: explicit o routes (MID k3_rpo_3)
            t, cur = o_routes(t, ARGS["oroutes"])
            projs = [q for q in projs if q.split(":")[0] != "o"]
        xg = [(q.split(":")[0], int(q.split(":")[1]) if ":" in q else None) for q in ARGS.get("xg", "").split(",") if q]
        t = pf_routes(t, projs, xg) if projs else t
        a_, b_ = gd_route_span(t)
        what = "GD pre-wait stages coherent (PF) on " + "/".join(projs) + ": " + " ".join(
            f"{p_}{bb}:{rt.replace(', ', '')}" for p_, bb, rt in re.findall(r'\("(\w+)", (\d+)\): \(([\d, ]+)\)', t[a_:b_])
            if int(rt.split(", ")[6]) & 2)
    nonce = f"# nonce {out.stem.replace('_', '-')} ({base.stem} + {what}; tools/rpo_patches.py)"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    check(re.search(r"importlib|sys\.modules", t) is None or re.search(r"importlib|sys\.modules", b) is not None, "checker")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[rpo_patches] OK {mode} {len(t.encode())} B (base {len(b.encode())} B, +{len(t.encode()) - len(b.encode())})")
    if "--dry" in sys.argv:
        return 0
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[rpo_patches] wrote {out.relative_to(ROOT).as_posix()} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

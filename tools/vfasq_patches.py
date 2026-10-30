"""vfasq_patches.py (R-vfasq): split-QK consumer for the VFA full-attention kernel (vfa_main, verify q_len 8).

Idea (ext_muse_attn.md lever A, notes/lab.md:698): each tile's QK (today one m64n64k32 pair, one commit group, wait_group 0)
becomes two m64n32k32 pairs (keys 0..31 -> sc[0..15], keys 32..63 -> sc[16..31]), each its own commit group. After
wait_group 1 the consumer exponentiates / packs keys 0..31 (pa[0]) while the tensor pipe still runs QK of keys 32..63;
then wait_group 0, empty arrive, pack keys 32..63, the unchanged saturation test (both halves) and PV. The lazy offset
(cr0 / cr1 fixed after a segment's first tile) is what makes the halves independent: the first tile of a segment keeps
today's order (both halves, row maxima, pack). Same registers as ship (sc[32] + pa[2][4] + acc[36]), same arithmetic.

Flags in the generated file:
  VFA_SQ    (Python) 1 = DIAG 0 uses the split QK, 0 = ship consumer (bit-for-bit the base's VFA source otherwise).
  VFA_SQLAB (Python) 1 = also build the lab arms (ST 4, 32-bit offsets only): DIAG 21 = split QK, 22 = ship consumer with a
            nocopy producer (copies only the first ST tiles, then only arrives: consumer-only time), 23 = split QK nocopy.

  python tools/vfasq_patches.py --base=kernels/muse_decode_layer/k3_gc_21cg_s1.py --out=kernels/lab/k3_vfasq_1.py [--sq=1]
  WSL: tools/local/cudaenv.sh python tools/vfasq_patches.py --sass=<dir> <layer.py>   (ptxas -v of ship + every arm)
"""
import ast
import pathlib
import re
import subprocess
import sys

NL = chr(10)
ROOT = pathlib.Path(__file__).resolve().parents[1]
CUOBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"


def one(t, old, new, what):
    n = t.count(old)
    if n != 1:
        raise SystemExit(f"vfasq_patches: anchor '{what}' found {n} times (need exactly 1)")
    return t.replace(old, new)


HELPERS_ANCHOR = "__device__ __forceinline__ void wg_pv(float (&d)[36], const uint32_t (&a)[4], uint64_t db) {"
HELPERS = r'''__device__ __forceinline__ void wg_wait1() { asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory"); }   // VFASQ
__device__ __forceinline__ void wg_qk32(float* d, uint64_t da, uint64_t db, int scale_d) {   // VFASQ: m64n32k32, 16 keys x 2 rows
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
               : "l"(da), "l"(db), "r"(scale_d));
}
'''

PVT_OLD = "template <int DIAG> constexpr bool pvt_on() { return (DIAG >= 18 && DIAG <= 20) || (DIAG == 0 && true); }   // VFA3-PVT"
PVT_NEW = "template <int DIAG> constexpr bool pvt_on() { return (DIAG >= 18 && DIAG <= 23) || (DIAG == 0 && true); }   // VFA3-PVT (+ VFASQ 21..23)"

# producer: nocopy lab arms (22 / 23) only arrive on full after the first ST tiles
PROD_OLD = '''      const uint32_t kb = so + st * STAGE_BYTES, vb = kb + off_v<ST>();
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int pos = k0 + x * TILE + r0 + 16 * k;'''
PROD_NEW = '''      const uint32_t kb = so + st * STAGE_BYTES, vb = kb + off_v<ST>();
      constexpr bool NOC = (DIAG == 22 || DIAG == 23);   // VFASQ lab: stages already resident (consumer-only time)
      if (NOC && n >= ST) {
        mbar_arrive(full + st);
      } else {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int pos = k0 + x * TILE + r0 + 16 * k;'''
PROD_OLD2 = '''      cp_async_mbar_arrive(full + st);
      if constexpr (PVT) { if (n >= LAG) vt_tile(n - LAG); }'''
PROD_NEW2 = '''      cp_async_mbar_arrive(full + st);
      }
      if constexpr (PVT) { if (n >= LAG) vt_tile(n - LAG); }'''

FLAG_OLD = "  constexpr bool PVS = (DIAG == 13 || DIAG == 14) || (DIAG == 0 && false);   // FA16-PVS" + NL
FLAG_NEW = FLAG_OLD + "  constexpr bool SQ = (DIAG == 21 || DIAG == 23) || (DIAG == 0 && VFA_SQ_ON);   // VFASQ: split QK (two commit groups)" + NL

QK_OLD = '''        if (DIAG != 3) {
          wg_fence();
          wg_qk(sc, HI64 | qlo, HI128 | kl, 0);
          wg_qk(sc, HI64 | (qlo + 2u), HI128 | (kl + 2u), 1);
          wg_commit();
        } else {'''
QK_NEW = '''        if (DIAG != 3) {
          wg_fence();
          if constexpr (SQ) {   // VFASQ: keys 0..31 (group 1) and keys 32..63 (group 2, rows 32.. of the stage = +4 KB)
            wg_qk32(sc, HI64 | qlo, HI128 | kl, 0);
            wg_qk32(sc, HI64 | (qlo + 2u), HI128 | (kl + 2u), 1);
            wg_commit();
            wg_qk32(sc + 16, HI64 | qlo, HI128 | (kl + 256u), 0);
            wg_qk32(sc + 16, HI64 | (qlo + 2u), HI128 | (kl + 258u), 1);
            wg_commit();
          } else {
          wg_qk(sc, HI64 | qlo, HI128 | kl, 0);
          wg_qk(sc, HI64 | (qlo + 2u), HI128 | (kl + 2u), 1);
          wg_commit();
          }
        } else {'''

WAIT_OLD = '''        if (DIAG != 3) wg_wait0();                               // QK(i) (and, deferred, PV(i - 1))
#pragma unroll
        for (int e = 0; e < 32; ++e) VFA_RF(sc[e]);'''
WAIT_NEW = '''        if constexpr (SQ) {   // VFASQ: pack keys 0..31 while QK(keys 32..63) runs (lazy offset fixed: not the first tile)
          wg_wait1();
#pragma unroll
          for (int e = 0; e < 16; ++e) VFA_RF(sc[e]);
          if (!first) {
            const int tq = k0 + i * TILE;
            if (tq + TILE - 1 >= lim) {                          // edge tile: keys 0..31 masked as below (idempotent there)
              const int hi0 = min(k1 - 1, seq + 2 * w), hi1 = min(k1 - 1, seq + 2 * w + 1);
#pragma unroll
              for (int nt8 = 0; nt8 < 4; ++nt8)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                  const int pos = tq + 8 * nt8 + 2 * tig + e;
                  if (pos > hi0) sc[4 * nt8 + e] = neg_inf();
                  if (pos > hi1) sc[4 * nt8 + 2 + e] = neg_inf();
                }
            }
            const float* sv = sc;   // = packh(0) below (PEX 0, H16 off in every SQ arm)
            pa[0][0] = pk4(ex2(fmaf(sv[0], ssc, -cr0)), ex2(fmaf(sv[1], ssc, -cr0)), ex2(fmaf(sv[4], ssc, -cr0)), ex2(fmaf(sv[5], ssc, -cr0)));
            pa[0][1] = pk4(ex2(fmaf(sv[2], ssc, -cr1)), ex2(fmaf(sv[3], ssc, -cr1)), ex2(fmaf(sv[6], ssc, -cr1)), ex2(fmaf(sv[7], ssc, -cr1)));
            pa[0][2] = pk4(ex2(fmaf(sv[8], ssc, -cr0)), ex2(fmaf(sv[9], ssc, -cr0)), ex2(fmaf(sv[12], ssc, -cr0)), ex2(fmaf(sv[13], ssc, -cr0)));
            pa[0][3] = pk4(ex2(fmaf(sv[10], ssc, -cr1)), ex2(fmaf(sv[11], ssc, -cr1)), ex2(fmaf(sv[14], ssc, -cr1)), ex2(fmaf(sv[15], ssc, -cr1)));
          }
          wg_wait0();
#pragma unroll
          for (int e = 16; e < 32; ++e) VFA_RF(sc[e]);
        } else {
        if (DIAG != 3) wg_wait0();                               // QK(i) (and, deferred, PV(i - 1))
#pragma unroll
        for (int e = 0; e < 32; ++e) VFA_RF(sc[e]);
        }'''

PACK_OLD = '''        pack();
        if (!first && DIAG != 4) {   // lazy offset'''
PACK_NEW = '''        if (SQ && !first) packh(1); else pack();   // VFASQ: keys 0..31 already packed after wait_group 1
        if (!first && DIAG != 4) {   // lazy offset'''

DISP_OLD = '''static void vfa_diag(const Params& p, dim3 grid, cudaStream_t stream, bool pdl, int diag) {
#if VFA_LAB'''
DISP_NEW = '''static void vfa_diag(const Params& p, dim3 grid, cudaStream_t stream, bool pdl, int diag) {
#if VFA_SQLAB
  if constexpr (O32 && ST == 4) {   // VFASQ lab arms: 21 split QK, 22 ship consumer nocopy, 23 split QK nocopy
    switch (diag) {
      case 21: vfa_go<ST, 21, true>(p, grid, stream, pdl); return;
      case 22: vfa_go<ST, 22, true>(p, grid, stream, pdl); return;
      case 23: vfa_go<ST, 23, true>(p, grid, stream, pdl); return;
      default: break;
    }
  }
  TORCH_CHECK(diag < 21 || diag > 23, "vfa: VFASQ lab arms need stages 4 and 32-bit cache offsets");
#endif
#if VFA_LAB'''

PY_OLD = "VFA_DIAG = 0" + NL
BUILD_OLD = '''    src = VFA_CUDA_SRC.replace("#if VFA_LAB", "#if 1" if lab else "#if 0")'''
BUILD_NEW = '''    src = VFA_CUDA_SRC.replace("#if VFA_LAB", "#if 1" if lab else "#if 0")
    src = src.replace("#if VFA_SQLAB", "#if 1" if VFA_SQLAB else "#if 0").replace("VFA_SQ_ON", "true" if VFA_SQ else "false")   # VFASQ'''


def apply(t, sq=1):
    """t: layer text carrying the ship VFA block (k3_gc_21cg_s1 / k3_b23_1 family)."""
    t = one(t, HELPERS_ANCHOR, HELPERS + HELPERS_ANCHOR, "wg_pv def")
    t = one(t, PVT_OLD, PVT_NEW, "pvt_on")
    t = one(t, PROD_OLD, PROD_NEW, "producer copy loop")
    t = one(t, PROD_OLD2, PROD_NEW2, "producer full arrive")
    t = one(t, FLAG_OLD, FLAG_NEW, "PVS flag")
    t = one(t, QK_OLD, QK_NEW, "QK issue")
    t = one(t, WAIT_OLD, WAIT_NEW, "QK wait")
    t = one(t, PACK_OLD, PACK_NEW, "pack call")
    t = one(t, DISP_OLD, DISP_NEW, "vfa_diag dispatch")
    t = one(t, PY_OLD, PY_OLD + f"VFA_SQ = {int(sq)}   # VFASQ: 1 = split-QK consumer at DIAG 0, 0 = ship consumer" + NL
            + "VFA_SQLAB = 0   # VFASQ: 1 = also build lab DIAG 21 / 22 / 23 (bench only)" + NL, "VFA_DIAG constant")
    t = one(t, BUILD_OLD, BUILD_NEW, "_vfa_build source replace")
    return t


def sass(layer, od):
    """Device-only sm_90a cubin of vfa_main<4, D, true>: ship (VFA_SQ_ON false, D 0), SQ (true, D 0), lab 21 / 22 / 23."""
    od = pathlib.Path(od)
    od.mkdir(parents=True, exist_ok=True)
    tree = ast.parse(pathlib.Path(layer).read_text(encoding="utf-8"))
    s = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "VFA_CUDA_SRC")
    cut = s.index("template <int ST, int DIAG, bool O32>" + NL + "static void vfa_go(")
    dev = re.sub(r"#include <(torch|ATen|c10)[^>]*>" + NL, "", s[:cut]) + "}  // namespace vfa" + NL
    rc = 0
    for name, on, diags in (("ship", "false", (0,)), ("sq", "true", (0, 21, 22, 23))):
        src = dev.replace("VFA_SQ_ON", on) + "".join(
            f"template __global__ void vfa::vfa_main<4, {d}, true>(const __grid_constant__ vfa::Params);" + NL for d in diags)
        cu, cub = od / f"vfa_{name}.cu", od / f"vfa_{name}.cubin"
        cu.write_text(src)
        r = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-O3", "--expt-relaxed-constexpr", "-Xptxas=-v", "-o", str(cub), str(cu)],
                           capture_output=True, text=True)
        (od / f"vfa_{name}.ptxas.txt").write_text(r.stderr)
        if r.returncode:
            print(f"[{name}] COMPILE FAILED" + NL + r.stderr[-3000:])
            return 2
        fn = ""
        for ln in r.stderr.splitlines():
            m = re.search(r"Compiling entry function '(\S+)'", ln)
            if m:
                fn = re.sub(r".*vfa_mainILi4ELi(\d+)ELb1E.*", r"D\1", m.group(1))
            if "warning" in ln.lower() or "C75" in ln:
                print(f"[{name} {fn}] {ln.strip()[:220]}")
                rc = 1
            elif "Used" in ln or "spill" in ln:
                print(f"[{name} {fn}] {ln.strip()[:160]}")
        out = subprocess.run([CUOBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
        (od / f"vfa_{name}.sass").write_text(out)
        for blk in out.split("Function : ")[1:]:
            fname = blk.split(NL, 1)[0]
            ins = re.findall(r"/\*[0-9a-f]{4,}\*/\s+([^;]*);", blk)
            ops = [re.sub(r"^@!?U?P\w+\s+", "", x).split()[0] if x.strip() else "?" for x in ins]
            cnt = {k: sum(1 for o in ops if o.startswith(k)) for k in ("QGMMA", "WARPGROUP", "MUFU", "FFMA", "F2FP", "STL", "LDL")}
            tag = re.sub(r".*vfa_mainILi4ELi(\d+)ELb1E.*", r"D\1", fname)
            print(f"[{name} {tag}] SASS {len(ins)} instr; " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
    return rc


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if "sass" in args:
        pos = [a for a in sys.argv[1:] if not a.startswith("--")]
        sys.exit(sass(pos[0], args["sass"]))
    base = pathlib.Path(args.get("base", "kernels/muse_decode_layer/k3_gc_21cg_s1.py"))
    out = pathlib.Path(args.get("out", "kernels/lab/k3_vfasq_1.py"))
    t = base.read_text(encoding="utf-8")
    t2 = apply(t, int(args.get("sq", "1")))
    out.write_text(t2, encoding="utf-8", newline=NL)
    print(f"[vfasq] {base} -> {out}: {len(t2) - len(t)} bytes added, VFA_SQ = {args.get('sq', '1')}")


if __name__ == "__main__":
    main()

"""fidx_patches.py (notes/study/full_attn_gap.md, R-fg-2 GO): k3_gc_21 -> kernels/lab/k3_fidx_1.py.

  python tools/fidx_patches.py [--base=kernels/muse_decode_layer/k3_gc_21.py] [--out=kernels/lab/k3_fidx_1.py] [--va=1] [--vfa=1]

FIDX = a rolling per-warp smem index ring in the full-attention producers (VA decode, VFA verify). Today each producer thread
loads its 4 page-table entries for tile x + 1 while it issues tile x (`pgn`); under a ~4 TB/s gather the loaded latency is
~3 us, so the producer waits on the index load every tile (fg_probe: 3.80-3.93 TB/s -> 4.02-4.22 with staged indices).
FIDX: per producer warp, 2 chunks x FIDX_TPC tiles x 16 rows of int (1 KB at TPC 8; 4 KB per CTA). Lane L fetches the 4
entries (rows 4w + (L & 3) + 16k) of tile L >> 2 of chunk c + 2 into registers at the end of chunk c and stores them at the
end of chunk c + 1; __syncwarp at each chunk entry publishes the chunk to the warp. Readers take exactly the entries the
direct path loads (same rows, same order, same -1 past the segment), so outputs are bitwise equal to the ship.
Segment starts load chunk 0 directly (one index latency per segment, as today). Correct for any page size and length.
Flags (module globals, rev_ab3-selectable): VA1_FIDX (stage codes 22x / 23x = va1_main<ST, true>), VFA_FIDX (diag 21 =
vfa_main<ST, 0, O32, true>). Ship kernels are kept, the new instantiations sit beside them.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)


def _rep(src, old, new, what, n=1):
    c = src.count(old)
    if c != n:
        raise SystemExit(f"[fidx_patches] {what}: expected {n} match(es), got {c}")
    return src.replace(old, new)


HELPERS = r"""#ifndef FIDX_TPC
#define FIDX_TPC 8
#endif
static_assert(FIDX_TPC == 8, "FIDX lane map: lane >> 2 = tile of the chunk");
__device__ __forceinline__ int fx_lds(uint32_t a) { int v; asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ void fx_sts(uint32_t a, int v) { asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(a), "r"(v) : "memory"); }
"""

VA_IX = r"""      if constexpr (IX) {   // FIDX: rolling per-warp smem index ring (tools/fidx_patches.py)
        const uint32_t fx_ib = va_sa(smem + va_smem<ST, false>()) + (uint32_t)((warp - 4) * (128 * FIDX_TPC));
        const int fx_lt = lane >> 2, fx_lj = lane & 3, fx_wb = 4 * (warp - 4), fx_j = r0 & 3;
        int fx_rr[4];
        auto fx_fetch = [&](int c) {
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int pos = k0 + (c * FIDX_TPC + fx_lt) * TILE + fx_wb + fx_lj + 16 * k;
            fx_rr[k] = (pos < cend) ? __ldg(pt + (pos >> psh)) : -1;
          }
        };
        auto fx_put = [&](int c) {
#pragma unroll
          for (int k = 0; k < 4; ++k) fx_sts(fx_ib + 4u * ((c & 1) * (16 * FIDX_TPC) + fx_lt * 16 + 4 * k + fx_lj), fx_rr[k]);
        };
        __syncwarp();
        fx_fetch(0);
        fx_put(0);
        fx_fetch(1);
        __syncwarp();
        for (int x = 0; x < nt; ++x, ++n) {
          const int c = x / FIDX_TPC, xi = x - c * FIDX_TPC;
          if (xi == 0 && x > 0) __syncwarp();
#pragma unroll
          for (int k = 0; k < 4; ++k) pg[k] = fx_lds(fx_ib + 4u * ((c & 1) * (16 * FIDX_TPC) + xi * 16 + 4 * k + fx_j));
          const int st = n % ST;
          if (n >= ST) va_wait(empt + st, ((n / ST) - 1) & 1);
          uint8_t* stg = smem + st * STAGE_BYTES;
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int r = r0 + 16 * k, pos = k0 + x * TILE + r;
            const size_t go = pg[k] >= 0 ? (size_t)((pg[k] << psh) | (pos & pm)) * rowB : (size_t)0;
            cp_async16(stg + r * ROWB + 16 * kswz<1>(r, ch), kc + go, pg[k] >= 0 ? 16 : 0);
            cp_async16(stg + TILE * ROWB + r * ROWB + 16 * vswz<1>(r, ch), vc + go, pg[k] >= 0 ? 16 : 0);
          }
          asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(va_sa(full + st)) : "memory");
          if (xi == FIDX_TPC - 1) {
            __syncwarp();
            fx_put(c + 1);
            fx_fetch(c + 2);
          }
        }
        continue;
      }
"""


def patch_va(s):
    s = _rep(s, "#define K1_ARCH_OK 0\n#endif\n", "#define K1_ARCH_OK 0\n#endif\n" + HELPERS, "K1 helpers")
    s = _rep(s, "template <int ST> constexpr int va_smem() { return ST * STAGE_BYTES + NEW_BYTES + 2048 + 16 * ST + 4 * 1032; }",
             "template <int ST, bool IX = false> constexpr int va_smem() {\n"
             "  return ST * STAGE_BYTES + NEW_BYTES + 2048 + 16 * ST + 4 * 1032 + (IX ? 4 * 128 * FIDX_TPC : 0);   // FIDX ring\n}",
             "va_smem")
    s = _rep(s, "template <int ST>\n__global__ void __launch_bounds__(256, 2) va1_main(const Params p) {",
             "template <int ST, bool IX = false>\n__global__ void __launch_bounds__(256, 2) va1_main(const Params p) {", "va1_main")
    old = ("      const int* pt = p.page_table + (size_t)b * p.pt_stride;\n      int pg[4];\n#pragma unroll\n"
           "      for (int k = 0; k < 4; ++k) pg[k] = (k0 + r0 + 16 * k < cend) ? __ldg(pt + ((k0 + r0 + 16 * k) >> psh)) : -1;\n")
    s = _rep(s, old, old.replace("      int pg[4];\n", "      int pg[4];\n" + VA_IX), "VA producer")
    s = _rep(s, "template <int ST>\nstatic int va_go(const k1::Params* p, dim3 grid, cudaStream_t s, bool pdl = false) {",
             "template <int ST, bool IX = false>\nstatic int va_go(const k1::Params* p, dim3 grid, cudaStream_t s, bool pdl = false) {", "va_go")
    s = _rep(s, "k1::va1_main<ST>", "k1::va1_main<ST, IX>", "va_go kernel refs", 3)
    s = _rep(s, "k1::va_smem<ST>()", "k1::va_smem<ST, IX>()", "va_go smem refs", 3)
    s = _rep(s, "ST == 213 || ST == 214)", "ST == 213 || ST == 214 || ST == 223 || ST == 224 || ST == 233 || ST == 234)", "stages_ok")
    s = _rep(s, "  if (ST > 200) return (int64_t)(ST % 10 == 3 ? va_go<3>(nullptr, dim3(1), 0) : va_go<4>(nullptr, dim3(1), 0));",
             "  if (ST > 220) return (int64_t)(ST % 10 == 3 ? va_go<3, true>(nullptr, dim3(1), 0) : va_go<4, true>(nullptr, dim3(1), 0));\n"
             "  if (ST > 200) return (int64_t)(ST % 10 == 3 ? va_go<3>(nullptr, dim3(1), 0) : va_go<4>(nullptr, dim3(1), 0));",
             "k1_occupancy")
    s = _rep(s, "    if (stages % 10 == 3) va_go<3>(&p, grid, st, stages > 210);\n    else va_go<4>(&p, grid, st, stages > 210);",
             "    const int vmode = (int)((stages / 10) % 10);   // 0 plain, 1 PDL, 2 FIDX, 3 FIDX + PDL\n"
             "    if (vmode >= 2) {\n"
             "      if (stages % 10 == 3) va_go<3, true>(&p, grid, st, vmode & 1);\n"
             "      else va_go<4, true>(&p, grid, st, vmode & 1);\n"
             "    } else if (stages % 10 == 3) va_go<3>(&p, grid, st, vmode & 1);\n"
             "    else va_go<4>(&p, grid, st, vmode & 1);", "k1_forward VA dispatch")
    s = _rep(s, "VA1_PDL = True\n", "VA1_PDL = True\nVA1_FIDX = %s   # FIDX smem index ring (tools/fidx_patches.py)\n"
             % ("True" if ARGS.get("va", "1") == "1" else "False"), "VA1_FIDX flag")
    s = _rep(s, "    st += 10 if pdl else 0\n    ws = _VA1_WS[str(dev)]\n",
             "    st += 10 if pdl else 0\n    st += 20 if VA1_FIDX else 0\n    ws = _VA1_WS[str(dev)]\n", "_va1_run stage code")
    return s


VFA_INIT_OLD = """    int pg[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int pos = k0 + r0 + 16 * k;
      pg[k] = pos < k1 ? __ldg(pt + (pos >> lp)) : -1;
    }
"""
VFA_INIT_NEW = """    int pg[4];
    const uint32_t fx_ib = sa(smem) + (uint32_t)(smem_bytes<ST>() - 1024 + (q >> 5) * (128 * FIDX_TPC));   // FIDX ring
    const int fx_lt = (q & 31) >> 2, fx_lj = q & 3, fx_wb = 4 * (q >> 5), fx_j = r0 & 3;
    int fx_rr[4];
    auto fx_fetch = [&](int c) {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int pos = k0 + (c * FIDX_TPC + fx_lt) * TILE + fx_wb + fx_lj + 16 * k;
        fx_rr[k] = pos < k1 ? __ldg(pt + (pos >> lp)) : -1;
      }
    };
    auto fx_put = [&](int c) {
#pragma unroll
      for (int k = 0; k < 4; ++k) fx_sts(fx_ib + 4u * ((c & 1) * (16 * FIDX_TPC) + fx_lt * 16 + 4 * k + fx_lj), fx_rr[k]);
    };
    (void)fx_fetch; (void)fx_put; (void)fx_j;
    if constexpr (IX) {
      __syncwarp();
      fx_fetch(0);
      fx_put(0);
      fx_fetch(1);
      __syncwarp();
    } else {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int pos = k0 + r0 + 16 * k;
        pg[k] = pos < k1 ? __ldg(pt + (pos >> lp)) : -1;
      }
    }
"""
VFA_PGN_OLD = """      int pgn[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {   // next tile's pages: in flight while this tile's copies issue
        const int pos = k0 + (x + 1) * TILE + r0 + 16 * k;
        pgn[k] = pos < k1 ? __ldg(pt + (pos >> lp)) : -1;
      }
"""
VFA_PGN_NEW = """      int pgn[4];
      if constexpr (IX) {
        const int fx_c = x / FIDX_TPC, fx_xi = x - fx_c * FIDX_TPC;
        if (fx_xi == 0 && x > 0) __syncwarp();
#pragma unroll
        for (int k = 0; k < 4; ++k) pg[k] = fx_lds(fx_ib + 4u * ((fx_c & 1) * (16 * FIDX_TPC) + fx_xi * 16 + 4 * k + fx_j));
      } else {
#pragma unroll
        for (int k = 0; k < 4; ++k) {   // next tile's pages: in flight while this tile's copies issue
          const int pos = k0 + (x + 1) * TILE + r0 + 16 * k;
          pgn[k] = pos < k1 ? __ldg(pt + (pos >> lp)) : -1;
        }
      }
"""
VFA_END_OLD = """      if constexpr (PVT) { if (n >= LAG) vt_tile(n - LAG); }   // VFA3-PVT: tile n - LAG landed long ago
#pragma unroll
      for (int k = 0; k < 4; ++k) pg[k] = pgn[k];
"""
VFA_END_NEW = """      if constexpr (PVT) { if (n >= LAG) vt_tile(n - LAG); }   // VFA3-PVT: tile n - LAG landed long ago
      if constexpr (IX) {
        const int fx_c = x / FIDX_TPC, fx_xi = x - fx_c * FIDX_TPC;
        if (fx_xi == FIDX_TPC - 1) {
          __syncwarp();
          fx_put(fx_c + 1);
          fx_fetch(fx_c + 2);
        }
      } else {
#pragma unroll
        for (int k = 0; k < 4; ++k) pg[k] = pgn[k];
      }
"""


def patch_vfa(s):
    s = _rep(s, "template <int DIAG> constexpr bool pvt_on()", HELPERS + "template <int DIAG> constexpr bool pvt_on()", "VFA helpers")
    s = _rep(s, "template <int ST, bool O32, int DIAG>\n__device__ __forceinline__ void producer(",
             "template <int ST, bool O32, int DIAG, bool IX = false>\n__device__ __forceinline__ void producer(", "VFA producer tmpl")
    s = _rep(s, VFA_INIT_OLD, VFA_INIT_NEW, "VFA producer init")
    s = _rep(s, VFA_PGN_OLD, VFA_PGN_NEW, "VFA producer pgn")
    s = _rep(s, VFA_END_OLD, VFA_END_NEW, "VFA producer end")
    s = _rep(s, "template <int ST, int DIAG, bool O32>\n__global__ void __launch_bounds__(NTHR, 2) vfa_main(",
             "template <int ST, int DIAG, bool O32, bool IX = false>\n__global__ void __launch_bounds__(NTHR, 2) vfa_main(", "vfa_main tmpl")
    s = _rep(s, "producer<ST, O32, DIAG>(p, smem, seg, nseg, full, empt, tid - NCONS, pair);",
             "producer<ST, O32, DIAG, IX>(p, smem, seg, nseg, full, empt, tid - NCONS, pair);", "vfa_main producer call")
    a = s.index("template <int ST, int DIAG, bool O32>\nstatic void vfa_go(")
    b = s.index("template <int ST, bool O32>\nstatic void vfa_diag(")
    go = s[a:b]
    go2 = go.replace("template <int ST, int DIAG, bool O32>\nstatic void vfa_go(", "template <int ST, int DIAG, bool O32, bool IX = false>\nstatic void vfa_go(")
    go2 = go2.replace("vfa_main<ST, DIAG, O32>", "vfa_main<ST, DIAG, O32, IX>").replace("smem_bytes<ST>()", "(smem_bytes<ST>() + (IX ? 4096 : 0))")   # host code: no FIDX_TPC (device-only macro)
    if go2.count("vfa_main<ST, DIAG, O32, IX>") != 3 or go2.count("(IX ? 4096 : 0)") != 2:
        raise SystemExit("[fidx_patches] vfa_go refs")
    s = s[:a] + go2 + s[b:]
    s = _rep(s, "  (void)diag;\n  vfa_go<ST, 0, O32>(p, grid, stream, pdl);",
             "  if (diag == 21) { vfa_go<ST, 0, O32, true>(p, grid, stream, pdl); return; }   // FIDX\n"
             "  (void)diag;\n  vfa_go<ST, 0, O32>(p, grid, stream, pdl);", "vfa_diag")
    s = _rep(s, "VFA_DIAG = 0\n", "VFA_DIAG = 0\nVFA_FIDX = %s   # FIDX smem index ring (tools/fidx_patches.py): diag 21\n"
             % ("True" if ARGS.get("vfa", "1") == "1" else "False"), "VFA_FIDX flag")
    s = _rep(s, "int(VFA_DIAG if diag is None else diag)",
             "int(((21 if VFA_FIDX else 0) if VFA_DIAG == 0 else VFA_DIAG) if diag is None else diag)", "_vfa_run diag")
    return s


def main():
    base = ROOT / ARGS.get("base", "kernels/muse_decode_layer/k3_gc_21.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_fidx_1.py")
    s = base.read_text(encoding="utf-8")
    s = patch_va(s)
    s = patch_vfa(s)
    ast.parse(s)
    if any(len(line) > 400 for line in s.splitlines()):
        print("[fidx_patches] note: lines > 400 chars exist (base has them too?)")
    out.write_text(s, encoding="utf-8", newline="\n")
    print(f"[fidx_patches] wrote {out.relative_to(ROOT)} ({len(s.encode())} bytes; VA1_FIDX {ARGS.get('va', '1')}, "
          f"VFA_FIDX {ARGS.get('vfa', '1')})")


if __name__ == "__main__":
    main()

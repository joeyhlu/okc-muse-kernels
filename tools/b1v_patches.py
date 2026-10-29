"""b1v_patches.py: B1V = B1A (tools/b1a_patches.py) + FP8 wgmma compute + a lighter barrier, notes/handoff_attn.md (B1A-v2).

Edits on the B1A block text (after att1_patches.b1a_block_edits, so B1A_DBG stays). Every edit is B = 1 only (b1a_ module).
* WG (template parameter, B1A_WG = {q_len: bool} picks the instance at call time): warpgroup h = KV head h does one tile of 64 keys
  per step for all 64 rows (row 16 w + g = token 2 w, head 8 h + g; q_len 1: only warp 0's rows g are real):
  QK = 2 x wgmma m64n64k32 e4m3 RS (A = B1A's Q fragment registers, B = the swizzled K tile, desc + h x 64 B), V^T of the tile
  by the 4 warps (ldmatrix.trans + prmt, keys permuted into P's register order, rows 64..71 = e4m3 ones: the row sums), lazy
  offset (first tile = row max, repack only when a P byte exceeds 256), PV = 2 x wgmma m64n72k32 RS. One V^T buffer per head
  (named barrier before the next tile's transpose). The new-key unit is a 64-row tile (rows >= L zero). K / V tiles stay
  where B1A's one-burst cp.async puts them; the smem base is 1024-B aligned. Removes the mma.sync path's 4x K / V re-reads
  and its e4m3 emulation.
* FIN (Params.fin, B1A_FIN): every CTA arrives by red.release.gpu on a per-q_len monotonic counter (bar[4], bar[5]; a
  multiple of G between calls); finalizer CTAs read the base after griddepcontrol.wait and poll (bounded by %globaltimer,
  sticky flag, no trap, no sleep) until base - base % G + G. No pre-arrival generation load, no last-arriver release.
* Self-test: every case for each (wg, fin) of B1A_TEST_VARIANTS.

    python -I tools/b1v_patches.py --ptxas [--nvcc=PATH] [--work=DIR]     compile-only sm_90a check of the B1V device code
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
NL = chr(10)

KNOBS = ("B1A_WG = {1: True, 8: True}" + NL + "B1A_FIN = 1" + NL +
         "B1A_TEST_VARIANTS = ((1, 1), (0, 0), (1, 0), (0, 1))" + NL)

LAY_OLD_HEAD = "template <int L, int NT> struct Lay {"
LAY_NEW = r"""template <int L, int NT, int WG> struct Lay {    // shared memory (bytes from a 128-B (WG: 1024-B) aligned base)
  static constexpr int NUR = WG ? 64 : NU;       // rows of the new-key unit
  static constexpr int K = 0;                    // NT K tiles [64 keys][128 B], 16-B chunk c of row r at ((c ^ (r & 7)) << 4)
  static constexpr int V = NT * TILEB;           // NT V tiles, same layout
  static constexpr int NK = 2 * NT * TILEB;      // new-key unit: K [NUR][128 B], then V
  static constexpr int NV = NK + NUR * ROWB;
  static constexpr int VT = NV + NUR * ROWB;     // WG: V^T per KV head [72 rows][64 keys] e4m3, 64B swizzle (one buffer)
  static constexpr int OS = VT + (WG ? 2 * VTB : 0);   // q_len 1 (mma.sync): [4 key quarters][16 rows][64] f32 partials
  static constexpr int LS = OS + ((L == 1 && !WG) ? 4 * 16 * D * 4 : 0);
  static constexpr int BAR = LS + ((L == 1 && !WG) ? 4 * 16 * 4 : 0);
  static constexpr int MISC = BAR + NT * 8;
  static constexpr int AL = WG ? 1024 : 128;
  static constexpr int BYTES = MISC + 16 + AL;   // + alignment slack
};
"""

HELPERS = r"""
// ---- B1V: FP8 wgmma helpers (tools/b1v_patches.py) ----
#define B1V_RF(x) asm volatile("" : "+f"(x))
#define B1V_RR(x) asm volatile("" : "+r"(x))
#define B1V_D32(d) "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
#define B1V_R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
#define B1V_R36 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}"
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
__device__ __forceinline__ void sts32(uint32_t addr, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;" ::"r"(addr), "r"(v) : "memory"); }
__device__ __forceinline__ void bar_named(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void red_release(unsigned* a, unsigned v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory"); }
// QK: D[64 x 64] (+)= A[64 x 32] (Q fragment registers, the mma.sync m16n8k32 A layout per warp) x B[64 keys x 32] (K tile desc)
__device__ __forceinline__ void wg_qk_rs(float (&d)[32], const uint32_t (&a)[4], uint64_t db, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " B1V_R32 ", {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
               : B1V_D32(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(scale_d));
}
// PV: D[64 x 72] += A[64 x 32] (P registers) x B[72 x 32] (V^T desc): 64 dims + 8 copies of the row sum
__device__ __forceinline__ void wg_pv(float (&d)[36], const uint32_t (&a)[4], uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 " B1V_R36 ", {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
               : B1V_D32(d), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
}

// one 64-key tile for warpgroup h (all 4 warps call it together): kt / vtile = 1024-B aligned K / V tile, vt = this head's V^T
// buffer, cnt = valid keys from the tile start (NEW: the block's own keys, causal per row token), sync_vt = the V^T buffer was
// used by the previous tile (wait for every warp's PV)
template <int L, bool NEW>
__device__ __forceinline__ void wtile(uint32_t kt, uint32_t vtile, uint32_t vt, int cnt, int h, int w, int lane, int tok0, int tok1,
                                      bool live, const uint32_t (&qa)[2][4], float (&acc)[36], float& cr0, float& cr1, bool& first,
                                      float ssc, bool sync_vt) {
  const int g = lane >> 2, tig = lane & 3;
  constexpr uint64_t HI64 = ((uint64_t)(512 >> 4) | (2ull << 30)) << 32;
  constexpr uint64_t HI128 = ((uint64_t)(1024 >> 4) | (1ull << 30)) << 32;
  const uint32_t klo = ((kt + (uint32_t)h * 64u) >> 4) | 0x10000u;
  float sc[32];
#pragma unroll
  for (int e = 0; e < 32; ++e) sc[e] = 0.f;
  wg_fence();
  wg_qk_rs(sc, qa[0], HI128 | klo, 0);
  wg_qk_rs(sc, qa[1], HI128 | (klo + 2u), 1);
  wg_commit();
  if (sync_vt) bar_named(1 + h, 128);
  {   // V^T: ldmatrix rows = key 16 w + 8 ((lane >> 3) & 1) + (lane & 7), chunk (4 h + (lane >> 4)) ^ (lane & 7); stores: row
      // n = 16 c' + 2 g (+1), key chunk w, bytes 4 tig at n * 64 + ((w ^ (g & 3)) << 4); lanes g >= 4 write the odd row first
    const int vkey = 16 * w + 8 * ((lane >> 3) & 1) + (lane & 7);
    const uint32_t vsrc = vtile + (uint32_t)vkey * ROWB + ((((uint32_t)(4 * h + (lane >> 4))) ^ (uint32_t)(lane & 7)) << 4);
    const uint32_t vst = vt + 128u * g + ((uint32_t)(w ^ (g & 3)) << 4) + 4u * tig;
    const uint32_t vsel_a = (g >= 4) ? 0x7531u : 0x6420u, vsel_b = (g >= 4) ? 0x6420u : 0x7531u;
    const uint32_t da = vst + ((g >= 4) ? 64u : 0u), db = vst + ((g >= 4) ? 0u : 64u);
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      uint32_t r[4];
      ldsm_x4_t(r, u ? (vsrc ^ 32u) : vsrc);
      sts32(da + 1024u * (2 * u), prmt(r[0], r[1], vsel_a));
      sts32(db + 1024u * (2 * u), prmt(r[0], r[1], vsel_b));
      sts32(da + 1024u * (2 * u + 1), prmt(r[2], r[3], vsel_a));
      sts32(db + 1024u * (2 * u + 1), prmt(r[2], r[3], vsel_b));
    }
  }
  wg_wait0();
#pragma unroll
  for (int e = 0; e < 32; ++e) B1V_RF(sc[e]);
  uint32_t pa[2][4] = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
  if (live) {
    if (NEW || cnt < TK) {   // keys past the range; new keys: causal inside the block
#pragma unroll
      for (int nt8 = 0; nt8 < 8; ++nt8)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int key = 8 * nt8 + 2 * tig + e;
          if (!(key < cnt && (!NEW || key <= tok0))) sc[4 * nt8 + e] = neg_inf();
          if (!(key < cnt && (!NEW || key <= tok1))) sc[4 * nt8 + 2 + e] = neg_inf();
        }
    }
    auto rowmax = [&](float& m0, float& m1) {
      m0 = neg_inf(); m1 = neg_inf();
#pragma unroll
      for (int nt8 = 0; nt8 < 8; ++nt8) {
        m0 = fmaxf(m0, fmaxf(sc[4 * nt8], sc[4 * nt8 + 1]));
        m1 = fmaxf(m1, fmaxf(sc[4 * nt8 + 2], sc[4 * nt8 + 3]));
      }
      m0 = quad_max(m0) * ssc;
      m1 = (L == 8) ? quad_max(m1) * ssc : 0.f;
    };
    if (first) {   // per-row offsets = the first tile's row maxima (log2 units)
      float m0, m1;
      rowmax(m0, m1);
      cr0 = (m0 == neg_inf()) ? 0.f : m0;
      cr1 = (m1 == neg_inf()) ? 0.f : m1;
    }
    auto pack = [&]() {   // P = e4m3(ex2(s x ssc - offset)), keys permuted into the PV A-fragment order
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const float* sv = sc + 16 * kk;
        pa[kk][0] = pk4(ex2(fmaf(sv[0], ssc, -cr0)), ex2(fmaf(sv[1], ssc, -cr0)), ex2(fmaf(sv[4], ssc, -cr0)), ex2(fmaf(sv[5], ssc, -cr0)));
        pa[kk][2] = pk4(ex2(fmaf(sv[8], ssc, -cr0)), ex2(fmaf(sv[9], ssc, -cr0)), ex2(fmaf(sv[12], ssc, -cr0)), ex2(fmaf(sv[13], ssc, -cr0)));
        if (L == 8) {
          pa[kk][1] = pk4(ex2(fmaf(sv[2], ssc, -cr1)), ex2(fmaf(sv[3], ssc, -cr1)), ex2(fmaf(sv[6], ssc, -cr1)), ex2(fmaf(sv[7], ssc, -cr1)));
          pa[kk][3] = pk4(ex2(fmaf(sv[10], ssc, -cr1)), ex2(fmaf(sv[11], ssc, -cr1)), ex2(fmaf(sv[14], ssc, -cr1)), ex2(fmaf(sv[15], ssc, -cr1)));
        } else {
          pa[kk][1] = 0u;
          pa[kk][3] = 0u;
        }
      }
    };
    pack();
    if (!first) {   // lazy offset: a byte > 0x78 (P > 256) sets bit 7 of byte + 7 -> offsets up to this tile's maxima, repack
      constexpr uint32_t PB = 0x07070707u;
      const uint32_t ov = ((pa[0][0] + PB) | (pa[0][1] + PB) | (pa[0][2] + PB) | (pa[0][3] + PB) |
                           (pa[1][0] + PB) | (pa[1][1] + PB) | (pa[1][2] + PB) | (pa[1][3] + PB)) & 0x80808080u;
      if (__any_sync(FULL, ov != 0u)) {
        float m0, m1;
        rowmax(m0, m1);
        const float n0 = fmaxf(m0, cr0), n1 = fmaxf(m1, cr1);
        const float a0 = ex2(cr0 - n0), a1 = ex2(cr1 - n1);
#pragma unroll
        for (int t = 0; t < 9; ++t) { acc[4 * t] *= a0; acc[4 * t + 1] *= a0; acc[4 * t + 2] *= a1; acc[4 * t + 3] *= a1; }
        cr0 = n0; cr1 = n1;
        pack();
      }
    }
    first = false;
  }
#pragma unroll
  for (int e = 0; e < 36; ++e) B1V_RF(acc[e]);
#pragma unroll
  for (int e = 0; e < 8; ++e) B1V_RR(pa[e >> 2][e & 3]);
  fence_proxy_async();      // V^T (st.shared) -> async proxy
  bar_named(1 + h, 128);    // all 4 warps' V^T rows written
  const uint32_t vl = (vt >> 4) | 0x10000u;
  wg_fence();
  wg_pv(acc, pa[0], HI64 | vl);
  wg_pv(acc, pa[1], HI64 | (vl + 2u));
  wg_commit();
  wg_wait0();
#pragma unroll
  for (int e = 0; e < 36; ++e) B1V_RF(acc[e]);
#pragma unroll
  for (int e = 0; e < 8; ++e) B1V_RR(pa[e >> 2][e & 3]);
}
"""

WG_COMPUTE = r"""  } else {
  // 3w. B1V: warpgroup h = KV head h over every tile of this CTA (64 rows: row 16 wq + g = token tok0, row + 8 = tok1)
  const bool live = (L == 8) || (wq == 0);
  float acc[36];
#pragma unroll
  for (int e = 0; e < 36; ++e) { acc[e] = 0.f; B1V_RF(acc[e]); }
  float cr0 = 0.f, cr1 = 0.f;
  bool first = true;
  const float ssc = p.qk_sm * ksc * LOG2E;   // raw e4m3 x e4m3 score -> log2 units
  const uint32_t vt = sb + S::VT + (uint32_t)(h * VTB);
  for (int i = 0; i < ntl; ++i) {
    mbar_wait(bars + i, i < redo ? 1u : 0u);
    fence_proxy_async();   // cp.async rows (generic proxy) -> wgmma
    wtile<L, false>(sb + S::K + i * TILEB, sb + S::V + i * TILEB, vt, kc - TK * i, h, wq, lane, tok0, tok1, live, qa, acc, cr0, cr1,
                    first, ssc, i > 0);
  }
  if (tail) wtile<L, true>(sb + S::NK, sb + S::NV, vt, L, h, wq, lane, tok0, tok1, live, qa, acc, cr0, cr1, first, ssc, ntl > 0);
  if ((p.sync >> 4) == 2) {   // lab: loads + compute only (the store keeps the compute alive)
    if (acc[0] + acc[32] == 1234.5f) p.out[0] = __float2bfloat16(0.f);
    return;
  }
  // 4w. merge: lanes (tig, tig ^ 1) swap halves so each holds 4 consecutive dims: 8 red.v4 + the row sums per live thread
  if (live && ((kc > 0) || tail)) {
    const float f0 = ex2(fminf(cr0, 126.f)), f1 = ex2(fminf(cr1, 126.f));
    float* wo = p.ws_o + (size_t)(c % p.W) * ROWS * D;
    float* wl = p.ws_l + (size_t)(c % p.W) * ROWS;
    const int R0 = tok0 * 16 + 8 * h + g, R1 = tok1 * 16 + 8 * h + g;
    const bool odd = (t & 1) != 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int j = 2 * q;
#pragma unroll
      for (int rr = 0; rr < (R2 ? 2 : 1); ++rr) {
        const int b0 = 4 * j + 2 * rr, b1 = 4 * (j + 1) + 2 * rr;
        const float kx = odd ? acc[b1] : acc[b0], ky = odd ? acc[b1 + 1] : acc[b0 + 1];
        const float sx = odd ? acc[b0] : acc[b1], sy = odd ? acc[b0 + 1] : acc[b1 + 1];
        const float x = __shfl_xor_sync(FULL, sx, 1), y = __shfl_xor_sync(FULL, sy, 1);
        const int dim = odd ? 8 * (j + 1) + 2 * (t - 1) : 8 * j + 2 * t;
        const float f = rr ? f1 : f0;
        float* dst = wo + (rr ? R1 : R0) * D + dim;
        if (odd) red4(dst, x * f, y * f, kx * f, ky * f);
        else red4(dst, kx * f, ky * f, x * f, y * f);
      }
    }
    if (t == 0) {
      red1(wl + R0, acc[32] * f0);
      if (R2) red1(wl + R1, acc[34] * f1);
    }
  }
  }
"""

FIN_OLD = "  if (tid == 0) {" + NL + "    int ok = 1;" + NL + "    const unsigned g0 = ld_acquire(p.bar + 1);"
FIN_NEW = r"""  if (p.fin) {   // B1V: arrive by red.release on the per-q_len counter (a multiple of G between calls); finalizers poll it
    if (tid == 0) {
      unsigned* cnt = p.bar + 4 + (L == 8 ? 1 : 0);
      red_release(cnt, 1u);
      int ok = 1;
      if (c < NFIN) {
        const unsigned tgt = fin_base - fin_base % (unsigned)G + (unsigned)G;
        const long long t0 = gtime();
        while ((int)(ld_acquire(cnt) - tgt) < 0) {
          if (gtime() - t0 > p.spin_ns) {
            atomicExch(p.bar + 2, 1u);
            ok = 0;
            break;
          }
        }
      }
      misc[0] = ok;
    }
  } else if (tid == 0) {
    int ok = 1;
    const unsigned g0 = ld_acquire(p.bar + 1);"""

EDITS = [
    ("B1A_DBG = 0" + NL, "B1A_DBG = 0" + NL + KNOBS),
    ('_B1A = {"mod": None, "ws": {}, "grid": {}, "occ": {}, "nt": None}',
     '_B1A = {"mod": None, "ws": {}, "grid": {}, "occ": {}, "nt": None, "force": None}'),
    # signatures (CPP prototypes + CUDA definitions)
    ("                 int64_t spin_ns);" + NL + "int64_t b1a_occupancy(int64_t L, int64_t nt);",
     "                 int64_t spin_ns, int64_t wg, int64_t fin);" + NL + "int64_t b1a_occupancy(int64_t L, int64_t nt, int64_t wg);"),
    ("                 int64_t spin_ns) {", "                 int64_t spin_ns, int64_t wg, int64_t fin) {"),
    ("constexpr int MAXW = 8;", "constexpr int MAXW = 8;" + NL + "constexpr int VTB = 72 * 64;                   // B1V V^T bytes per KV head"),
    ("  float eps;" + NL + "};", "  float eps;" + NL + "  int fin;                      // B1V: counter arrival + polling finalizers" + NL + "};"),
    # kernel
    ("#endif" + NL + NL + "template <int L, int NT>" + NL + "__global__ void __launch_bounds__(NTHR, 2) b1a_kernel(",
     HELPERS + "#endif" + NL + NL + "template <int L, int NT, int WG>" + NL + "__global__ void __launch_bounds__(NTHR, 2) b1a_kernel("),
    ("  using S = Lay<L, NT>;", "  using S = Lay<L, NT, WG>;"),
    ("  uint8_t* smem = smem_raw + ((128u - (sa(smem_raw) & 127u)) & 127u);",
     "  uint8_t* smem = smem_raw + (((unsigned)S::AL - (sa(smem_raw) & (unsigned)(S::AL - 1))) & (unsigned)(S::AL - 1));"),
    ("i < 2 * NU * ROWB / 16; i += NTHR)", "i < 2 * S::NUR * ROWB / 16; i += NTHR)"),
    ("  mbar_fence_init();" + NL + "  __syncthreads();" + NL + "  issue_copies",
     "  if (WG) {   // V^T rows 64..71 of both heads: e4m3 ones (every byte equal: no swizzle)" + NL +
     "    for (int i = tid; i < 2 * 128; i += NTHR)" + NL +
     "      *reinterpret_cast<uint32_t*>(smem + S::VT + (i >> 7) * VTB + 4096 + 4 * (i & 127)) = ONES;" + NL +
     "  }" + NL + "  mbar_fence_init();" + NL + "  __syncthreads();" + NL + "  issue_copies"),
    ("  if (p.trig == 1) pdl_trigger();" + NL,
     "  if (p.trig == 1) pdl_trigger();" + NL + "  unsigned fin_base = 0u;   // B1V: counter value before this CTA's arrival (finalizers)" + NL +
     "  if (p.fin && tid == 0 && c < NFIN) fin_base = ld_acquire(p.bar + 4 + (L == 8 ? 1 : 0));" + NL),
    ("  __syncthreads();   // the new-key unit is in shared memory" + NL,
     "  if (WG) fence_proxy_async();   // new-key rows / zero rows (generic stores) -> wgmma" + NL +
     "  __syncthreads();   // the new-key unit is in shared memory" + NL),
    ("  // 3. attention over this CTA's keys as the tiles land" + NL,
     "  if constexpr (WG == 0) {" + NL + "  // 3. attention over this CTA's keys as the tiles land" + NL),
    ("  if ((p.sync >> 4) == 1) return;   // lab: no barrier / finalize" + NL,
     WG_COMPUTE + "  if ((p.sync >> 4) == 1) return;   // lab: no barrier / finalize" + NL),
    (FIN_OLD, FIN_NEW),
    # host
    ("template <int L, int NT>" + NL + "static void b1a_go(", "template <int L, int NT, int WG>" + NL + "static void b1a_go("),
    ("template <int L, int NT>" + NL + "static int64_t b1a_occ()", "template <int L, int NT, int WG>" + NL + "static int64_t b1a_occ()"),
]
REPL_ALL = [   # (old, new, expected count)
    ("b1a::Lay<L, NT>::BYTES", "b1a::Lay<L, NT, WG>::BYTES", 2),
    ("b1a::b1a_kernel<L, NT>", "b1a::b1a_kernel<L, NT, WG>", 4),
]
HOST_EDITS = [
    ("int64_t b1a_occupancy(int64_t L, int64_t nt) {" + NL +
     '  TORCH_CHECK((L == 1 || L == 8) && (nt == 5 || nt == 10), "b1a: L 1 / 8, nt 5 / 10");' + NL +
     "  if (L == 1) return nt == 10 ? b1a_occ<1, 10>() : b1a_occ<1, 5>();" + NL +
     "  return nt == 10 ? b1a_occ<8, 10>() : b1a_occ<8, 5>();",
     "int64_t b1a_occupancy(int64_t L, int64_t nt, int64_t wg) {" + NL +
     '  TORCH_CHECK((L == 1 || L == 8) && (nt == 5 || nt == 10) && (wg == 0 || (wg == 1 && nt == 5)), "b1a: L 1 / 8, nt 5 / 10, wg nt 5");' + NL +
     "  if (wg) return L == 1 ? b1a_occ<1, 5, 1>() : b1a_occ<8, 5, 1>();" + NL +
     "  if (L == 1) return nt == 10 ? b1a_occ<1, 10, 0>() : b1a_occ<1, 5, 0>();" + NL +
     "  return nt == 10 ? b1a_occ<8, 10, 0>() : b1a_occ<8, 5, 0>();"),
    ('  TORCH_CHECK((L == 1 || L == 8) && (nt == 5 || nt == 10), "b1a: L 1 / 8, nt 5 / 10");' + NL + "  TORCH_CHECK(W >= 1",
     '  TORCH_CHECK((L == 1 || L == 8) && (nt == 5 || nt == 10) && (wg == 0 || (wg == 1 && nt == 5)) && (fin == 0 || fin == 1),' + NL +
     '              "b1a: L 1 / 8, nt 5 / 10, wg nt 5, fin");' + NL + "  TORCH_CHECK(W >= 1"),
    ("  TORCH_CHECK(bar.scalar_type() == at::kInt && bar.is_contiguous() && bar.numel() >= 4,",
     "  TORCH_CHECK(bar.scalar_type() == at::kInt && bar.is_contiguous() && bar.numel() >= 6,"),
    ("  p.eps = (float)eps;" + NL, "  p.eps = (float)eps;" + NL + "  p.fin = (int)fin;" + NL),
    ("  if (L == 1) {" + NL + "    if (nt == 10) b1a_go<1, 10>(p, (int)G, stream, pdl != 0); else b1a_go<1, 5>(p, (int)G, stream, pdl != 0);" + NL +
     "  } else {" + NL + "    if (nt == 10) b1a_go<8, 10>(p, (int)G, stream, pdl != 0); else b1a_go<8, 5>(p, (int)G, stream, pdl != 0);" + NL + "  }",
     "  if (wg) {" + NL + "    if (L == 1) b1a_go<1, 5, 1>(p, (int)G, stream, pdl != 0); else b1a_go<8, 5, 1>(p, (int)G, stream, pdl != 0);" + NL +
     "  } else if (L == 1) {" + NL + "    if (nt == 10) b1a_go<1, 10, 0>(p, (int)G, stream, pdl != 0); else b1a_go<1, 5, 0>(p, (int)G, stream, pdl != 0);" + NL +
     "  } else {" + NL + "    if (nt == 10) b1a_go<8, 10, 0>(p, (int)G, stream, pdl != 0); else b1a_go<8, 5, 0>(p, (int)G, stream, pdl != 0);" + NL + "  }"),
    ("void* b1a_instances[] = {(void*)b1a::b1a_kernel<1, 5>, (void*)b1a::b1a_kernel<8, 5>, (void*)b1a::b1a_kernel<1, 10>," + NL +
     "                         (void*)b1a::b1a_kernel<8, 10>};",
     "void* b1a_instances[] = {(void*)b1a::b1a_kernel<1, 5, 0>, (void*)b1a::b1a_kernel<8, 5, 0>, (void*)b1a::b1a_kernel<1, 10, 0>," + NL +
     "                         (void*)b1a::b1a_kernel<8, 10, 0>, (void*)b1a::b1a_kernel<1, 5, 1>, (void*)b1a::b1a_kernel<8, 5, 1>};"),
    # python
    ("def _b1a_run(qkv, data, cfg, pdl=None, spec=None, trig=None, copies=None):" + NL + '    L = cfg["q_len"]' + NL,
     "def _b1a_run(qkv, data, cfg, pdl=None, spec=None, trig=None, copies=None):" + NL + '    L = cfg["q_len"]' + NL +
     '    force = _B1A["force"]' + NL + "    wg = int(force[0]) if force else (1 if B1A_WG.get(L) else 0)" + NL +
     "    fin = int(force[1]) if force else int(B1A_FIN)" + NL),
    ("int(B1A_SPIN_MS * 1000000))", "int(B1A_SPIN_MS * 1000000), wg, fin)"),
    ('            occ = int(mod.b1a_occupancy(L, _B1A["nt"]))',
     '            occ = min(int(mod.b1a_occupancy(L, _B1A["nt"], w)) for w in ((0, 1) if _B1A["nt"] == 5 else (0,)))'),
    ("        for rep in range(2):" + NL + '            out = _b1a_run(d["qkv"].view(L, -1), d, d["config"], pdl=bool(rep))' + NL,
     "        for rep in range(2 * len(B1A_TEST_VARIANTS)):" + NL + '            _B1A["force"] = B1A_TEST_VARIANTS[rep >> 1]' + NL +
     "            try:" + NL + '                out = _b1a_run(d["qkv"].view(L, -1), d, d["config"], pdl=bool(rep & 1))' + NL +
     "            finally:" + NL + '                _B1A["force"] = None' + NL),
    ('width+{extra} #{rep}: rel', 'width+{extra} #{rep} (wg, fin) {B1A_TEST_VARIANTS[rep >> 1]}: rel'),
]


def _rep(text, old, new, count=1):
    n = text.count(old)
    assert n == count, f"anchor found {n} times (want {count}): {old[:140]!r}"
    return text.replace(old, new)


def edit_block(block):
    """B1A block text (after att1_patches.b1a_block_edits) -> B1V block text."""
    assert "B1A_DBG = 0" in block, "apply att1_patches.b1a_block_edits first"
    head = block.index(LAY_OLD_HEAD)
    end = block.index("};", head) + 3
    block = block[:head] + LAY_NEW + block[end:]
    for old, new in EDITS:
        block = _rep(block, old, new)
    for old, new, n in REPL_ALL:
        block = _rep(block, old, new, n)
    for old, new in HOST_EDITS:
        block = _rep(block, old, new)
    assert "Lay<L, NT>" not in block and "b1a_kernel<L, NT>" not in block
    return block


def _patched_bp():
    import att1_patches as AP
    import b1a_patches as BP
    BP.B1A_BLOCK = edit_block(AP.b1a_block_edits(BP.B1A_BLOCK))
    return BP


def main():
    args = sys.argv[1:]
    BP = _patched_bp()
    if "--cuda" in args:
        sys.stdout.write(BP.cuda_source(True))
        return 0
    if "--ptxas" in args:
        nvcc = next((a[7:] for a in args if a.startswith("--nvcc=")), "nvcc")
        work = next((a[7:] for a in args if a.startswith("--work=")), None)
        return BP.ptxas_check(nvcc, (), work, full="--full" in args)
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())

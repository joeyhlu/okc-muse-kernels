"""fac_patches.py (R-fac): VA1 (decode full, q_len 1) consumer variants for k3_gc_21-family layer files.

  python tools/fac_patches.py [--base=kernels/muse_decode_layer/k3_gc_21.py] [--out=kernels/lab/k3_fac_1.py] [--fac=2]

VA1_FAC (module global, rev_ab3-selectable; stage code + 20 * VA1_FAC, ST 4 only):
  0 ship tile_compute1<4>.
  1 TP: transposed MMAs: QK as S^T = K Q^T (M = keys, N = the 8 query heads: 8 HMMA instead of 16 with padded rows),
    P^T by movmatrix.trans, PV as O^T = V^T P^T (M = dims, N = heads: 8 HMMA instead of 16); same smem reads and
    F2FP count; 16 accumulators instead of 32 (88 vs 108 regs in kernels/lab/fac_bench.py).
  2 TP + ER: TP with early release: every K (LDS.128) and V (LDS.64 + PRMT) fragment of the tile is read at tile start and
    the stage's empty arrive follows at once (address-dependent on every load, R-sfx2 style); compute from registers.
Numerics: identical operands and fp16 P; only the MMA accumulation order differs (bench: 1.4e-7 vs ship).
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)


def _rep(src, old, new, what, n=1):
    c = src.count(old)
    if c != n:
        raise SystemExit(f"[fac_patches] anchor '{what}' found {c}x (want {n})")
    return src.replace(old, new)


TP_DEV = r"""
__device__ __forceinline__ uint32_t fac_movt(uint32_t x) {   // R-fac: 8x8 b16 transpose of an mma C-fragment register
#if K1_ARCH_OK
  uint32_t y;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
#else
  return x;
#endif
}
__device__ __forceinline__ uint32_t fac_kw(const uint4& v, int s) { return s == 0 ? v.x : s == 1 ? v.y : s == 2 ? v.z : v.w; }

// R-fac TP: S^T = K Q^T (rows = keys kw + 16 mt + g / + 8, cols = heads 2t / 2t + 1), P^T via movmatrix, O^T = V^T P^T
// (rows g / g + 8 of m-tile pm = dims 8 g + 2 pm / + 1). ER: all fragments read first, empty arrive at once. ARR: arrive at all.
template <int NMT, bool ER, bool ARR>
__device__ __forceinline__ void tile_tp(const uint8_t* __restrict__ Kbuf, const uint8_t* __restrict__ Vbuf, const int kw,
                                        const int h, const int g, const int t, const int lane, const uint32_t (&qb)[4][2],
                                        float (&oT)[4][4], float (&mT)[2], float (&lT)[2], const float cl2, const bool need,
                                        const int tile_abs, const int key_lim, const int seq, const uint32_t ea,
                                        const uint32_t zero) {
  uint4 kA[NMT], kB[NMT];
  uint32_t w01[NMT][4], w23[NMT][4];
#pragma unroll
  for (int mt = 0; mt < NMT; ++mt) {
    const int ka = kw + 16 * mt + g, kb = ka + 8;
    kA[mt] = *reinterpret_cast<const uint4*>(Kbuf + ka * ROWB + 16 * ((4 * h + t) ^ (4 * (ka & 1))));
    kB[mt] = *reinterpret_cast<const uint4*>(Kbuf + kb * ROWB + 16 * ((4 * h + t) ^ (4 * (kb & 1))));
  }
  if (ER) {
    uint32_t dep = 0u;
#pragma unroll
    for (int kk = 0; kk < NMT; ++kk) {
      v_frags(Vbuf, kw + 16 * kk + 2 * t, h, g, w01[kk], w23[kk]);
      dep ^= kA[kk].x ^ kB[kk].x ^ w01[kk][0] ^ w23[kk][0];
    }
    __syncwarp();
    if (ARR && lane == 0) asm volatile("{\n.reg .b64 s;\nmbarrier.arrive.shared.b64 s, [%0];\n}\n" ::"r"(ea + (dep & zero)) : "memory");
  }
  float sT[NMT][4];
#pragma unroll
  for (int mt = 0; mt < NMT; ++mt) {
    sT[mt][0] = 0.f; sT[mt][1] = 0.f; sT[mt][2] = 0.f; sT[mt][3] = 0.f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t wa = fac_kw(kA[mt], s), wb = fac_kw(kB[mt], s);
      mma16816(sT[mt], e4m3x2_to_h2(wa), e4m3x2_to_h2(wb), e4m3x2_to_h2(wa >> 16), e4m3x2_to_h2(wb >> 16), qb[s][0], qb[s][1]);
    }
  }
  if (need) {
#pragma unroll
    for (int mt = 0; mt < NMT; ++mt) {
      const int ak = tile_abs + kw + 16 * mt + g;
      if (!((ak < key_lim) && (ak <= seq))) { sT[mt][0] = neg_inf(); sT[mt][1] = neg_inf(); }
      if (!((ak + 8 < key_lim) && (ak + 8 <= seq))) { sT[mt][2] = neg_inf(); sT[mt][3] = neg_inf(); }
    }
  }
  float x0 = fmaxf(sT[0][0], sT[0][2]), x1 = fmaxf(sT[0][1], sT[0][3]);
#pragma unroll
  for (int mt = 1; mt < NMT; ++mt) { x0 = fmaxf(x0, fmaxf(sT[mt][0], sT[mt][2])); x1 = fmaxf(x1, fmaxf(sT[mt][1], sT[mt][3])); }
#pragma unroll
  for (int d = 4; d < 32; d <<= 1) {
    x0 = fmaxf(x0, __shfl_xor_sync(FULL, x0, d));
    x1 = fmaxf(x1, __shfl_xor_sync(FULL, x1, d));
  }
  const float n0 = fmaxf(mT[0], x0 * cl2), n1 = fmaxf(mT[1], x1 * cl2);
  const float u0 = (n0 == neg_inf()) ? 0.f : n0, u1 = (n1 == neg_inf()) ? 0.f : n1;
  const float a0 = ex2(mT[0] - u0), a1 = ex2(mT[1] - u1);
  const float b0 = u0 - P_OFFSET, b1 = u1 - P_OFFSET;
  mT[0] = n0; mT[1] = n1;
  uint32_t pT[NMT][2];
  float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
  for (int mt = 0; mt < NMT; ++mt) {
    const float p0 = ex2(fmaf(sT[mt][0], cl2, -b0)), p1 = ex2(fmaf(sT[mt][1], cl2, -b1));
    const float p2 = ex2(fmaf(sT[mt][2], cl2, -b0)), p3 = ex2(fmaf(sT[mt][3], cl2, -b1));
    ps0 += p0 + p2; ps1 += p1 + p3;
    pT[mt][0] = fac_movt(pack_h2(p0, p1));
    pT[mt][1] = fac_movt(pack_h2(p2, p3));
  }
  lT[0] = lT[0] * a0 + ps0;
  lT[1] = lT[1] * a1 + ps1;
  if (__any_sync(FULL, (a0 != 1.f) || (a1 != 1.f))) {
#pragma unroll
    for (int pm = 0; pm < 4; ++pm) { oT[pm][0] *= a0; oT[pm][1] *= a1; oT[pm][2] *= a0; oT[pm][3] *= a1; }
  }
  uint32_t vdep = 0u;
#pragma unroll
  for (int kk = 0; kk < NMT; ++kk) {
    if (!ER) {
      v_frags(Vbuf, kw + 16 * kk + 2 * t, h, g, w01[kk], w23[kk]);
      if (kk == NMT - 1) vdep = w01[kk][0] ^ w23[kk][0];
    }
#pragma unroll
    for (int pm = 0; pm < 4; ++pm)
      mma16816(oT[pm], e4m3x2_to_h2(w01[kk][pm]), e4m3x2_to_h2(w01[kk][pm] >> 16), e4m3x2_to_h2(w23[kk][pm]),
               e4m3x2_to_h2(w23[kk][pm] >> 16), pT[kk][0], pT[kk][1]);
  }
  if (!ER) {
    __syncwarp();
    if (ARR && lane == 0) asm volatile("{\n.reg .b64 s;\nmbarrier.arrive.shared.b64 s, [%0];\n}\n" ::"r"(ea + (vdep & zero)) : "memory");
  }
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const uint8_t* p) {"""

SHIP_HEAD = """    uint32_t qf[4][4];
    const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs) + (h * 8 + g) * 32;
"""
SHIP_RB = "    const size_t r0 = (size_t)(b * HP + hp) * nc * 16, rb = r0 + (blockIdx.x % nc) * 16;   // nc copies (W = nc)\n"
SHIP_TAIL = "    if (t == 0) atomicAdd(p.part_lse + rb + h * 8 + g, lA * f);\n"
TP_BLOCK = """    } else {   // R-fac TP / TP + ER consumer
    uint32_t qb[4][2];
    const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs) + (h * 8 + g) * 32;
#pragma unroll
    for (int s = 0; s < 4; ++s) { qb[s][0] = qa[8 * t + 2 * s]; qb[s][1] = qa[8 * t + 2 * s + 1]; }
    float oT[4][4];
#pragma unroll
    for (int pm = 0; pm < 4; ++pm) { oT[pm][0] = 0.f; oT[pm][1] = 0.f; oT[pm][2] = 0.f; oT[pm][3] = 0.f; }
    float mT[2] = {neg_inf(), neg_inf()}, lT[2] = {0.f, 0.f};
    const uint32_t zr = (uint32_t)p.page_shift >> 16;   // runtime 0: arrive address depends on the loads
    for (int i = 0; i < nt; ++i, ++n) {
      const int st = n % ST, ts = k0 + i * TILE;
      va_wait(full + st, (n / ST) & 1);
      const uint8_t* Kb = smem + st * STAGE_BYTES;
      tile_tp<2, CV == 2, true>(Kb, Kb + TILE * ROWB, kw, h, g, t, lane, qb, oT, mT, lT, cl2, ts + TILE > cend, ts, cend, seq,
                                va_sa(empt + st), zr);
    }
    if (k1 == sN[b] && kw == 0) tile_tp<1, false, false>(newk, newv, 0, h, g, t, lane, qb, oT, mT, lT, cl2, true, seq, seq + 1, seq, 0u, 0u);
#pragma unroll
    for (int d = 4; d < 32; d <<= 1) { lT[0] += __shfl_xor_sync(FULL, lT[0], d); lT[1] += __shfl_xor_sync(FULL, lT[1], d); }
    const float f0 = (mT[0] == neg_inf()) ? 0.f : ex2(fminf(mT[0] - P_OFFSET, 126.f));
    const float f1 = (mT[1] == neg_inf()) ? 0.f : ex2(fminf(mT[1] - P_OFFSET, 126.f));
    float* w0 = p.part_o + (rb + h * 8 + 2 * t) * 64 + 8 * g;   // head 2t, dims 8g .. 8g + 7; head 2t + 1 at + 64
    va_red4(w0, f0, oT[0][0], oT[0][2], oT[1][0], oT[1][2]);
    va_red4(w0 + 4, f0, oT[2][0], oT[2][2], oT[3][0], oT[3][2]);
    va_red4(w0 + 64, f1, oT[0][1], oT[0][3], oT[1][1], oT[1][3]);
    va_red4(w0 + 68, f1, oT[2][1], oT[2][3], oT[3][1], oT[3][3]);
    if (g == 0) {
      atomicAdd(p.part_lse + rb + h * 8 + 2 * t, lT[0] * f0);
      atomicAdd(p.part_lse + rb + h * 8 + 2 * t + 1, lT[1] * f1);
    }
    }
"""


def patch(s):
    fac = int(ARGS.get("fac", "2"))
    a = "      mma16816(o[jn], pa[kk][0], 0u, pa[kk][1], 0u, b0, b1);\n    }\n  }\n}\n"
    s = _rep(s, a + "\n__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const uint8_t* p) {", a + TP_DEV, "tile_tp insert")
    s = _rep(s, "template <int ST>\n__global__ void __launch_bounds__(256, 2) va1_main(const Params p) {",
             "template <int ST, int CV = 0>\n__global__ void __launch_bounds__(256, 2) va1_main(const Params p) {", "va1_main template")
    i0 = s.index(SHIP_HEAD)
    i1 = s.index(SHIP_TAIL, i0) + len(SHIP_TAIL)
    block = s[i0:i1]
    if block.count(SHIP_RB) != 1:
        raise SystemExit("[fac_patches] rb line not inside the ship consumer block")
    block = block.replace(SHIP_RB, "")
    s = s[:i0] + SHIP_RB + "    if constexpr (CV == 0) {\n" + block + TP_BLOCK + s[i1:]
    s = _rep(s, "template <int ST>\nstatic int va_go(const k1::Params* p, dim3 grid, cudaStream_t s, bool pdl = false) {",
             "template <int ST, int CV = 0>\nstatic int va_go(const k1::Params* p, dim3 grid, cudaStream_t s, bool pdl = false) {", "va_go")
    s = _rep(s, "k1::va1_main<ST>", "k1::va1_main<ST, CV>", "va_go kernel refs", 3)
    s = _rep(s, "ST == 213 || ST == 214)", "ST == 213 || ST == 214 || ST == 224 || ST == 234 || ST == 244 || ST == 254)", "stages_ok")
    s = _rep(s, "    if (stages % 10 == 3) va_go<3>(&p, grid, st, stages > 210);\n    else va_go<4>(&p, grid, st, stages > 210);",
             "    const int vmode = (int)((stages / 10) % 10), vcv = vmode >> 1;   // R-fac: + 20 * VA1_FAC (ST 4 only)\n"
             "    if (vcv == 1) va_go<4, 1>(&p, grid, st, vmode & 1);\n"
             "    else if (vcv == 2) va_go<4, 2>(&p, grid, st, vmode & 1);\n"
             "    else if (stages % 10 == 3) va_go<3>(&p, grid, st, vmode & 1);\n"
             "    else va_go<4>(&p, grid, st, vmode & 1);", "k1_forward VA dispatch")
    s = _rep(s, "VA1_PDL = True\n", f"VA1_PDL = True\nVA1_FAC = {fac}   # R-fac VA1 consumer: 0 ship, 1 TP, 2 TP + early release (tools/fac_patches.py)\n",
             "VA1_FAC flag")
    s = _rep(s, "    st += 10 if pdl else 0\n    ws = _VA1_WS[str(dev)]\n",
             "    st += 10 if pdl else 0\n    st += 20 * VA1_FAC if (VA1_FAC and st % 10 == 4) else 0\n    ws = _VA1_WS[str(dev)]\n",
             "_va1_run stage code")
    return s


def main():
    base = ROOT / ARGS.get("base", "kernels/muse_decode_layer/k3_gc_21.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_fac_1.py")
    s = patch(base.read_text(encoding="utf-8"))
    ast.parse(s)
    long_ = sum(1 for line in s.splitlines() if len(line) > 400)
    out.write_text(s, encoding="utf-8", newline="\n")
    print(f"[fac_patches] wrote {out.relative_to(ROOT)} ({len(s.encode())} bytes; VA1_FAC {ARGS.get('fac', '2')}; lines > 400: {long_})")


if __name__ == "__main__":
    main()

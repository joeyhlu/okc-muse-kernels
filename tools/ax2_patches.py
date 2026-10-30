"""ax2_patches.py: VA1 (decode full, q_len 1, B >= 2) variants from the public flat-q1 study (notes/lab.md R-ax2).

  python tools/ax2_patches.py --base=kernels/lab/k3_rider3_1.py --out=kernels/lab/k3_ax2_1.py [--vx=1] [--st=4]
  python tools/ax2_patches.py --base=kernels/muse_attention_decode/vfa_gc16as.py --out=kernels/muse_attention_decode/ax2_1.py \
         --vx=0 --lab=4:8,4:1,3:1,4:3,3:3,4:2,4:4,4:5

va1_main gains a template bit mask VX (va1_main<ST> == va1_main<ST, 0>, the ship code unchanged):
  bit 0 IX : page indices staged in a smem slot ring (AX_PS slots of one tile's 16-B aligned page-table window),
             filled by producer warp 4 with cp.async + cp.async.mbarrier.arrive, AX_PD tiles ahead of the K/V issue,
             across request boundaries (the cursor walks the CTA's segment list); readers release a slot through an
             mbarrier (128 arrivals). The ship loads each tile's indices with __ldg one tile ahead.
  bit 1 EP : epilogue: the two key-half warps of a head merge in smem, one red.add set per head row (half the
             atomics), no per-thread fence; thread 0 releases with one atom.acq_rel after the consumer barrier.
  bit 2 SK : (lab) gather skeleton: the consumer only waits and releases each stage (no math, no merge).
  bit 3    : (lab) no-op tag, a separately named copy of the ship code (in-check control).
  bit 4 TL : (lab) %globaltimer stamps per CTA (entry, first / last issue, tile 0, last tile, epilogue, finalize), printed once.
  bit 5 TX : (with IX, sm_90) the index window comes by one cp.async.bulk + expect_tx (async proxy), so the filler warp's
             K/V full-barrier arrivals no longer wait for its index copies (and vice versa).
  bit 6 EP2: the finalize's gate row is loaded before the epilogue; the finalize fence is fence.acquire.gpu.
Stage code 1000 + 100 * VX + 10 * pdl + ST selects a variant in k1_forward (ship codes 2xx unchanged); a variant that is not
built (or loses IX / TX to a misaligned or ragged page table) falls back to the ship kernel.
--vx / --st pick the live route (0 = ship). --lab=ST:VX,... adds research arms that run after the real call on the
B 16 / 32 page-1 decode-full shapes, each after a 256 MB L2 flush, AX2_LAB_REPS rounds (attention-board check only).
--slim=1 (default without --lab; required for the 1,000,000-byte layer file): live route only, no lab / SK / TL code,
our inserted CUDA without comments.
Measured: notes/lab.md R-ax2 (checks 936f48f8, d905369e, 3acf6db2).
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)


def _rep(src, old, new, what, n=1):
    c = src.count(old)
    if c != n:
        raise SystemExit(f"[ax2_patches] {what}: expected {n} match(es), got {c}")
    return src.replace(old, new)


CONSTS = r"""
// ---- ax2 (tools/ax2_patches.py): VA1 variants VX = IX 1 | EP 2 | SK 4 | lab tag 8 ----
constexpr int AX_PS = 8;                                  // IX: index slots
constexpr int AX_PD = 4;                                  // IX: tiles of index prefetch distance (<= AX_PS - 1)
constexpr int AX_SLOT = 80;                               // IX: ints per slot (17 x 16 B used)
constexpr int AX_CH = 17;                                 // IX: 16-B chunks per tile window (64 + 3 entries)
constexpr int AX_IXB = AX_PS * AX_SLOT * 4 + AX_PS * 16;  // IX: slots + full / free mbarriers
constexpr int AX_EPB = 2 * 32 * 18 * 4;                   // EP: key-half hand-off (2 warps x 32 lanes x 18 floats)
static_assert(AX_PD < AX_PS && AX_CH * 4 <= AX_SLOT, "ax2 index ring geometry");
__device__ __forceinline__ void ax_stamp(const Params& p, int k) {   // TL (lab): %globaltimer into p.inv_freq (u64 [CTA][8])
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  reinterpret_cast<unsigned long long*>(const_cast<float*>(p.inv_freq))[(blockIdx.y * gridDim.x + blockIdx.x) * 8 + k] = t;
}
"""

SMEM_OLD = "template <int ST> constexpr int va_smem() { return ST * STAGE_BYTES + NEW_BYTES + 2048 + 16 * ST + 4 * 1032; }"
SMEM_NEW = (CONSTS +
            "template <int ST, int VX = 0> constexpr int va_smem() {\n"
            "  return ST * STAGE_BYTES + NEW_BYTES + 2048 + 16 * ST + 4 * 1032 + ((VX & 1) ? AX_IXB : 0) + ((VX & 2) ? AX_EPB : 0)\n"
            "         + ((VX & 4) ? 40960 : 0);   // SK (lab): padded so it sits 2 CTAs/SM like the real kernel\n"
            "}")

KERN_OLD = "template <int ST>\n__global__ void __launch_bounds__(256, 2) va1_main(const Params p) {\n"
KERN_NEW = ("template <int ST, int VX = 0>\n__global__ void __launch_bounds__(256, 2) va1_main(const Params p) {\n"
            "  constexpr bool IX = (VX & 1) != 0, EP = (VX & 2) != 0, SK = (VX & 4) != 0;   // ax2\n"
            "  constexpr bool TX = (VX & 32) != 0, EP2 = (VX & 64) != 0;\n"
            "  constexpr int PFT = (VX & 256) ? 8 : ((VX & 128) ? 4 : 0);\n"
            "  constexpr int NWT = (VX & 512) ? 0 : ((VX & 1024) ? 1 : -1);\n"
            "  constexpr bool LT = (VX & 2048) != 0;\n"
            "  constexpr bool TL = (VX & 16) != 0;\n"
            "  if constexpr (TL) { if (threadIdx.x == 0) ax_stamp(p, 0); }\n")

INIT_OLD = (r"""    asm volatile("mbarrier.init.shared.b64 [%0], 4;\n" ::"r"(va_sa(empt + tid)) : "memory");
  }
""")
INIT_NEW = INIT_OLD + r"""  if constexpr (IX) {   // ax2 IX: slot full (32 filler lanes; TX: 1 + tx bytes) / slot free (128 producer threads)
    if (tid >= 32 && tid < 32 + AX_PS) {
      uint64_t* ib = reinterpret_cast<uint64_t*>(smem + va_smem<ST, 0>() + AX_PS * AX_SLOT * 4);
#if __CUDA_ARCH__ >= 900
      const int fc = TX ? 1 : 32;
#else
      const int fc = 32;
#endif
      asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(va_sa(ib + tid - 32)), "r"(fc) : "memory");
      asm volatile("mbarrier.init.shared.b64 [%0], 128;\n" ::"r"(va_sa(ib + AX_PS + tid - 32)) : "memory");
#if __CUDA_ARCH__ >= 900
      if constexpr (TX) asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
    }
  }
"""

PROD_OLD = r"""  if (warp >= 4) {
    const int q = tid - 128, ch = q & 7, r0 = q >> 3, psh = p.page_shift, pm = (1 << psh) - 1;
    const uint8_t* kc = p.k_cache + (size_t)hp * 128 + 16 * ch;
    const uint8_t* vc = p.v_cache + (size_t)hp * 128 + 16 * ch;
"""
PROD_NEW = PROD_OLD + r"""    if constexpr (IX) {   // ax2 IX: indices from the smem slot ring, filled AX_PD tiles ahead by warp 4
      int* ring = reinterpret_cast<int*>(smem + va_smem<ST, 0>());
      uint64_t* ifull = reinterpret_cast<uint64_t*>(ring + AX_PS * AX_SLOT);
      uint64_t* ifree = ifull + AX_PS;
      const bool filler = warp == 4;
      int cb = bf - 1, cx = 0, cnt = 0, ck0 = 0;   // filler cursor: segment (request) cb, its tile cx of cnt, first key ck0
      auto seg_next = [&]() {
        for (++cb; cb < B && sV[cb] < v1; ++cb) {
          const int a0 = max(v0 - sV[cb], 0), a1 = min(v1 - sV[cb], sN[cb]), ae = min(a1, sN[cb] - 1);
          if (a1 > a0 && ae > a0) { ck0 = a0; cnt = (ae - a0 + TILE - 1) / TILE; cx = 0; return; }
        }
        cb = B;
      };
      auto fill = [&](int m) {   // the cursor's tile (the CTA's m-th) -> slot m % AX_PS
        if (cb >= B) return;
        const int s = m % AX_PS;
        if (m >= AX_PS) va_wait(ifree + s, ((m / AX_PS) - 1) & 1);
        const long long c16 = (((long long)cb * p.pt_stride + ((ck0 + cx * TILE) >> psh)) >> 2) + lane;
        const long long rem = p.pt_bytes - c16 * 16;   // bytes of the table left from this chunk (no read past it)
#if __CUDA_ARCH__ >= 900
        if constexpr (TX) {   // TX: one bulk copy (async proxy, outside this warp's cp.async arrive tracking)
          if (lane == 0) {
            const long long w0 = (((long long)cb * p.pt_stride + ((ck0 + cx * TILE) >> psh)) >> 2) << 2;
            const long long rem = p.pt_bytes - w0 * 4;
            const int nb = (int)(rem >= AX_CH * 16 ? AX_CH * 16 : (rem > 0 ? (rem & ~15LL) : 0));
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(va_sa(ifull + s)), "r"(nb) : "memory");
            if (nb > 0)
              asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                           ::"r"(va_sa(ring + s * AX_SLOT)), "l"(p.page_table + w0), "r"(nb), "r"(va_sa(ifull + s)) : "memory");
          }
        } else
#endif
        {
        if (lane < AX_CH)
          cp_async16(ring + s * AX_SLOT + 4 * lane, rem > 0 ? (const void*)(p.page_table + 4 * c16) : (const void*)p.page_table,
                     rem >= 16 ? 16 : (rem > 0 ? (int)rem : 0));
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(va_sa(ifull + s)) : "memory");
        }
        if (++cx >= cnt) seg_next();
      };
      if (filler) {
        seg_next();
        for (int m = 0; m < AX_PD; ++m) fill(m);
      }
      if constexpr (PFT > 0) {   // ax2 PF: PDL launch only, pre-wait: L2 bulk prefetch of this CTA's tiles [ST, ST + PFT) (old keys)
        if (p.use_rope && ch == 0) {
          int pb = bf - 1, px = 0, pnt = 0, pk0 = 0, pce = 0;
          auto pnext = [&]() {
            for (++pb; pb < B && sV[pb] < v1; ++pb) {
              const int a0 = max(v0 - sV[pb], 0), a1 = min(v1 - sV[pb], sN[pb]), ae = min(a1, sN[pb] - 1);
              if (a1 > a0 && ae > a0) { pk0 = a0; pce = ae; pnt = (ae - a0 + TILE - 1) / TILE; px = 0; return; }
            }
            pb = B;
          };
          pnext();
          for (int t = 0; t < ST && pb < B; ++t) if (++px >= pnt) pnext();
          int pgv[PFT][4], psv[PFT][4];
#pragma unroll
          for (int t = 0; t < PFT; ++t) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const int pos = pk0 + px * TILE + r0 + 16 * k;
              psv[t][k] = pos;
              pgv[t][k] = (pb < B && pos < pce) ? __ldg(p.page_table + (size_t)pb * p.pt_stride + (pos >> psh)) : -1;
            }
            if (pb < B && ++px >= pnt) pnext();
          }
#pragma unroll
          for (int t = 0; t < PFT; ++t)
#pragma unroll
            for (int k = 0; k < 4; ++k)
              if (pgv[t][k] >= 0) {
                const size_t go = (size_t)((pgv[t][k] << psh) | (psv[t][k] & pm)) * rowB;
#if __CUDA_ARCH__ >= 900
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;\n" ::"l"(kc + go) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;\n" ::"l"(vc + go) : "memory");
#else
                asm volatile("prefetch.global.L2 [%0];\n" ::"l"(kc + go));
                asm volatile("prefetch.global.L2 [%0];\n" ::"l"(vc + go));
#endif
              }
        }
      }   // PF end
      for (int b = bf; b < B && sV[b] < v1; ++b) {
        const int k0 = max(v0 - sV[b], 0), k1 = min(v1 - sV[b], sN[b]), cend = min(k1, sN[b] - 1);
        if (k1 <= k0 || cend <= k0) continue;
        const int nt = (cend - k0 + TILE - 1) / TILE;
        const long long prb = (long long)b * p.pt_stride;
        for (int x = 0; x < nt; ++x, ++n) {
          const int s = n % AX_PS, pos0 = k0 + x * TILE, fp = pos0 >> psh;
          const int sb = s * AX_SLOT + (int)((prb + fp) & 3) - fp;   // slot entry of page e: sb + e
          va_wait(ifull + s, (n / AX_PS) & 1);
          int pg[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int pos = pos0 + r0 + 16 * k;
            pg[k] = (pos < cend) ? ring[sb + (pos >> psh)] : -1;
          }
          asm volatile("{\n.reg .b64 s;\nmbarrier.arrive.shared.b64 s, [%0];\n}\n" ::"r"(va_sa(ifree + s)) : "memory");
#if __CUDA_ARCH__ >= 900
          if constexpr (NWT >= 0) {   // NW / W1 (lab): K/V of tile NWT and later only after the PDL wait (pre-wait gather capped)
            if (n == NWT && p.use_rope) asm volatile("griddepcontrol.wait;\n" ::: "memory");
          }
#endif
          const int st = n % ST;
          if (n >= ST) va_wait(empt + st, ((n / ST) - 1) & 1);
          uint8_t* stg = smem + st * STAGE_BYTES;
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int r = r0 + 16 * k, pos = pos0 + r;
            const size_t go = pg[k] >= 0 ? (size_t)((pg[k] << psh) | (pos & pm)) * rowB : (size_t)0;
            cp_async16(stg + r * ROWB + 16 * kswz<1>(r, ch), kc + go, pg[k] >= 0 ? 16 : 0);
            cp_async16(stg + TILE * ROWB + r * ROWB + 16 * vswz<1>(r, ch), vc + go, pg[k] >= 0 ? 16 : 0);
          }
          asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(va_sa(full + st)) : "memory");
          if constexpr (TL) { if (q == 0) { if (n == 0) ax_stamp(p, 1); ax_stamp(p, 6); } }
          if (filler) fill(n + AX_PD);
        }
      }
#if __CUDA_ARCH__ >= 900
      if constexpr (LT) { if (p.use_rope) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }   // LT (lab): late trigger
#endif
      asm volatile("cp.async.wait_all;\n" ::: "memory");
      return;
    }
"""

# end of the producer block -> SK consumer (lab)
PEND_OLD = r"""    asm volatile("cp.async.wait_all;\n" ::: "memory");
    return;
  }
#if __CUDA_ARCH__ >= 900
"""
PEND_NEW = r"""    asm volatile("cp.async.wait_all;\n" ::: "memory");
    return;
  }
  if constexpr (SK) {   // ax2 SK (lab): gather skeleton, the consumer only waits and releases every stage
    for (int b = bf; b < B && sV[b] < v1; ++b) {
      const int k0 = max(v0 - sV[b], 0), k1 = min(v1 - sV[b], sN[b]);
      if (k1 <= k0) continue;
      const int cend = min(k1, sN[b] - 1);
      const int nt = (cend > k0) ? (cend - k0 + TILE - 1) / TILE : 0;
      for (int i = 0; i < nt; ++i, ++n) {
        const int st = n % ST;
        va_wait(full + st, (n / ST) & 1);
        __syncwarp();
        if (lane == 0) asm volatile("{\n.reg .b64 s;\nmbarrier.arrive.shared.b64 s, [%0];\n}\n" ::"r"(va_sa(empt + st)) : "memory");
      }
    }
    return;
  }
#if __CUDA_ARCH__ >= 900
"""

RED_OLD = r"""    const float f = (mA == neg_inf()) ? 0.f : ex2(fminf(mA - P_OFFSET, 126.f));   // to the absolute frame 2^0
    const size_t r0 = (size_t)(b * HP + hp) * nc * 16, rb = r0 + (blockIdx.x % nc) * 16;   // nc copies (W = nc)
    float* wa = p.part_o + (rb + h * 8 + g) * 64 + 16 * t;   // dims 16t + 8e + jn
    va_red4(wa, f, o[0][0], o[1][0], o[2][0], o[3][0]);
    va_red4(wa + 4, f, o[4][0], o[5][0], o[6][0], o[7][0]);
    va_red4(wa + 8, f, o[0][1], o[1][1], o[2][1], o[3][1]);
    va_red4(wa + 12, f, o[4][1], o[5][1], o[6][1], o[7][1]);
    if (t == 0) atomicAdd(p.part_lse + rb + h * 8 + g, lA * f);
"""
RED_NEW = r"""    uint4 ax_g = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (EP2) {   // ax2 EP2: the finalize's gate row, loaded before the epilogue (in flight during the merge)
      const int gr_ = tid >> 3, gu_ = tid & 7, gq_ = (2 * hp + (gr_ >> 3)) * 8 + (gr_ & 7);
      ax_g = __ldg(reinterpret_cast<const uint4*>(qrow + (size_t)(p.Hq + 2 * p.Hkv + gq_) * 64 + 8 * gu_));
    }
    if constexpr (EP) {   // ax2 EP: key half kw = 32 hands (m, l, o) to kw = 0 of the same head through smem
      float* xs = reinterpret_cast<float*>(smem + va_smem<ST, 0>() + (IX ? AX_IXB : 0)) + ((warp & 1) * 32 + lane) * 18;
      if (kw != 0) {
        xs[0] = mA; xs[1] = lA;
#pragma unroll
        for (int jn = 0; jn < 8; ++jn) { xs[2 + 2 * jn] = o[jn][0]; xs[3 + 2 * jn] = o[jn][1]; }
      }
      VA_CSYNC;
      if (kw == 0) {
        const float m1 = xs[0], l1 = xs[1], mm = fmaxf(mA, m1);
        const float f0 = (mA == neg_inf()) ? 0.f : ex2(mA - mm), f1 = (m1 == neg_inf()) ? 0.f : ex2(m1 - mm);
        lA = lA * f0 + l1 * f1;
#pragma unroll
        for (int jn = 0; jn < 8; ++jn) {
          o[jn][0] = o[jn][0] * f0 + xs[2 + 2 * jn] * f1;
          o[jn][1] = o[jn][1] * f0 + xs[3 + 2 * jn] * f1;
        }
        mA = mm;
      }
    }
    const float f = (mA == neg_inf()) ? 0.f : ex2(fminf(mA - P_OFFSET, 126.f));   // to the absolute frame 2^0
    const size_t r0 = (size_t)(b * HP + hp) * nc * 16, rb = r0 + (blockIdx.x % nc) * 16;   // nc copies (W = nc)
    float* wa = p.part_o + (rb + h * 8 + g) * 64 + 16 * t;   // dims 16t + 8e + jn
    if (!EP || kw == 0) {
      va_red4(wa, f, o[0][0], o[1][0], o[2][0], o[3][0]);
      va_red4(wa + 4, f, o[4][0], o[5][0], o[6][0], o[7][0]);
      va_red4(wa + 8, f, o[0][1], o[1][1], o[2][1], o[3][1]);
      va_red4(wa + 12, f, o[4][1], o[5][1], o[6][1], o[7][1]);
      if (t == 0) atomicAdd(p.part_lse + rb + h * 8 + g, lA * f);
    }
"""

CNT_OLD = r"""    VA_CSYNC;
    if (tid == 0) flag[0] = (atomicAdd(p.counters + b * HP + hp, k1 - k0) + k1 - k0 == sN[b]) ? 1 : 0;
    VA_CSYNC;
"""
CNT_NEW = r"""    VA_CSYNC;
    if (tid == 0) {
      int old;
      if constexpr (EP)   // ax2 EP: one release for the CTA's reds (ordered before it by the barrier)
        asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(p.counters + b * HP + hp), "r"(k1 - k0) : "memory");
      else
        old = atomicAdd(p.counters + b * HP + hp, k1 - k0);
      flag[0] = (old + k1 - k0 == sN[b]) ? 1 : 0;
    }
    VA_CSYNC;
    if constexpr (TL) { if (tid == 0) { ax_stamp(p, 4); reinterpret_cast<unsigned long long*>(const_cast<float*>(p.inv_freq))[(blockIdx.y * gridDim.x + blockIdx.x) * 8 + 7] += 1; } }
"""

# TL (lab, board file only): consumer per-tile and finalize stamps
TLC_OLD = "      tile_compute1<4>(Kb, Kb + TILE * ROWB, kw, h, g, t, qf, o, mA, lA, cl2, ts + TILE > cend, ts, cend, seq, 0);\n"
TLC_NEW = TLC_OLD + "      if constexpr (TL) { if (tid == 0) { if (n == 0) ax_stamp(p, 2); ax_stamp(p, 3); } }\n"
GATE_OLD = "      const uint4 gr = *reinterpret_cast<const uint4*>(qrow + (size_t)(p.Hq + 2 * p.Hkv + qh) * 64 + 8 * u);\n"
GATE_NEW = ("      const uint4 gr = EP2 ? ax_g : *reinterpret_cast<const uint4*>(qrow + (size_t)(p.Hq + 2 * p.Hkv + qh) * 64 + 8 * u);"
            "   // ax2 EP2\n")
TLF_OLD = "      if (tid == 0) p.counters[b * HP + hp] = 0;\n"
TLF_NEW = TLF_OLD + "      if constexpr (TL) { if (tid == 0) ax_stamp(p, 5); }\n"

PARAMS_OLD = "  int B, Hq, Hkv, W, page_shift, pt_stride, Gx, min_chunk, use_rope, fold;\n"
PARAMS_NEW = PARAMS_OLD + "  long long pt_bytes;         // ax2 IX: bytes of the page table's storage from page_table (no read past it)\n"

GO_OLD = ("template <int ST>\nstatic int va_go(const k1::Params* p, dim3 grid, cudaStream_t s, bool pdl = false) {")
GO_NEW = ("template <int ST, int VX = 0>\nstatic int va_go(const k1::Params* p, dim3 grid, cudaStream_t s, bool pdl = false) {")

STOK_OLD = "static bool stages_ok(int64_t L, int64_t ST) { return "
STOK_NEW = "static bool stages_ok(int64_t L, int64_t ST) { return (L == 1 && ST >= 1000) || "

OCC_OLD = "  if (ST > 200) return (int64_t)(ST % 10 == 3"
OCC_NEW = "  if (ST >= 1000) return (int64_t)ax2_dispatch(nullptr, dim3(1), 0, (int)((ST - 1000) / 100), (int)(ST % 10), false);\n" + OCC_OLD

FWD_OLD = "  p.fold = (int)fold;\n  p.qk_sm = (float)qk_sm;\n  const dim3 grid((unsigned)Gx, (unsigned)(Hkv / 2));\n"
FWD_NEW = (FWD_OLD +
           "  p.pt_bytes = (long long)page_table.storage().nbytes() - (long long)page_table.storage_offset() * 4;   // ax2\n"
           "  if (stages >= 1000) {   // ax2 variants: 1000 + 100 * VX + 10 * pdl + ST\n"
           "    TORCH_CHECK(L == 1 && fold == 1 && B <= 512, \"ax2: L 1, fold, B <= 512\");\n"
           "    int vx = (int)((stages - 1000) / 100);\n"
           "    if ((vx & 1) && (reinterpret_cast<uintptr_t>(page_table.data_ptr()) & 15)) vx &= ~1;   // IX needs a 16-B aligned table\n"
           "    if ((vx & 32) && (p.pt_bytes & 15)) vx &= ~32;   // TX bulk copies whole 16-B chunks only\n"
           "    TORCH_CHECK(ax2_dispatch(&p, grid, st, vx, (int)(stages % 10), ((stages / 10) % 10) != 0) >= 0, \"ax2: arm not built\");\n"
           "    C10_CUDA_KERNEL_LAUNCH_CHECK();\n"
           "    return;\n"
           "  }\n")


def dispatch_src(arms):
    lines = ["static int ax2_dispatch(const k1::Params* p, dim3 grid, cudaStream_t s, int vx, int st, bool pdl) {   // ax2 arms",
             "  switch (vx * 10 + st) {"]
    seen = set()
    for st, vx in arms:
        if (st, vx) in seen or vx == 0:
            continue
        seen.add((st, vx))
        lines.append(f"    case {vx * 10 + st}: return va_go<{st}, {vx}>(p, grid, s, pdl);")
    lines += ["    default: break;", "  }",
              "  return st == 3 ? va_go<3>(p, grid, s, pdl) : va_go<4>(p, grid, s, pdl);   // not built (or IX stripped): the ship",
              "}", ""]
    return "\n".join(lines)


def py_block(vx, st, lab, reps):
    return f'''
AX2_VX = {vx}   # ax2 live VA1 variant bits (IX 1 | EP 2), 0 = ship (tools/ax2_patches.py)
AX2_ST = {st}
AX2_LAB = {tuple(lab)!r}   # ((ST, VX), ...) research arms after the real call (B 16 / 32, page 1)
AX2_LAB_REPS = {reps}
_AX2 = {{}}


def _ax2_stcode(vx, st, pdl):
    if vx == 0:
        return 200 + st + (10 if pdl else 0)
    return 1000 + 100 * vx + (10 if pdl else 0) + st


def _ax2_code(st, data):
    vx = AX2_VX
    if not vx or not _AX2.get("ok") or (vx & 1 and data["page_table"].data_ptr() % 16):
        return st
    return _ax2_stcode(vx, AX2_ST, st >= 210)


def _ax2_lab(qkv, data, cfg, G, segk):
    B = cfg["batch"]
    if not AX2_LAB or B not in (16, 32) or cfg["page_size"] != 1 or not (_AX2.get("arms") or _AX2.get("tl")):
        return
    dev = qkv.device
    lab = _AX2.get("buf")
    if lab is None:
        return
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    ws = _VA1_WS[str(dev)]
    scratch = torch.empty((B, 1, hq * 64), dtype=torch.bfloat16, device=dev)
    run = lambda code, invf: _K1_MOD.k1_forward(
        qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
        data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], invf, ws[0], ws[1], scratch,
        ws[3], B, 1, hq, hkv, 1, cfg["page_size"].bit_length() - 1, G, segk, 0,
        float(cfg["qk_scale_factor"] * cfg["sm_scale"]), 1, 4, code)
    for _ in range(AX2_LAB_REPS):
        for code in _AX2["arms"]:
            torch.sum(lab[0], dim=0, out=lab[1])   # 256 MB read: cold L2 for the arm
            run(code, ws[2])
    for code in _AX2.get("tl", []):   # timeline arms: once per (arm, B), outside graph capture, host sync + log
        if (code, B) in _AX2["tl_done"] or torch.cuda.is_current_stream_capturing() or data["page_table"].shape[1] < 20000:
            continue
        _AX2["tl_done"].add((code, B))
        for rep in range(2):
            buf = torch.zeros(G * (hkv // 2) * 16, dtype=torch.float32, device=dev)
            torch.sum(lab[0], dim=0, out=lab[1])
            run(code, buf)
            torch.cuda.synchronize()
            s = buf.view(torch.int64).view(-1, 8).cpu()
            t0 = int(s[:, 0].min())
            names = ("entry", "prod first issue", "cons tile0 done", "cons last tile", "epilogue end", "finalize end",
                     "prod last issue")
            parts = []
            for k, nm in enumerate(names):
                v = s[:, k]
                v = v[v > 0]
                if v.numel() == 0:
                    continue
                r = (v - t0).double() / 1000.0
                q = torch.quantile(r, torch.tensor([0.0, 0.5, 0.9, 1.0], dtype=torch.float64)).tolist()
                parts.append(f"{{nm}} n{{v.numel()}} min {{q[0]:.1f}} p50 {{q[1]:.1f}} p90 {{q[2]:.1f}} max {{q[3]:.1f}}")
            end = torch.maximum(s[:, 4], s[:, 5])
            span = (float(end.max()) - t0) / 1000.0
            tail = (s[:, 4] - s[:, 3]).double() / 1000.0
            segs = torch.bincount(s[:, 7].clamp(0, 8)).tolist()
            print(f"[ax2] TL code {{code}} B={{B}} rep {{rep}} span {{span:.1f}} us; epi (last tile -> counter) p50 "
                  f"{{float(tail.median()):.2f}} max {{float(tail.max()):.2f}}; segs/CTA {{segs}} | " + " | ".join(parts), flush=True)


def _ax2_prepare(dev):
    try:
        _ax2_prepare_inner(dev)
    except Exception as ex:  # noqa: BLE001
        _AX2["ok"], _AX2["arms"] = False, []
        print(f"[ax2] OFF (ship VA1): {{ex!r}}", flush=True)


def _ax2_prepare_inner(dev):
    _AX2["ok"], _AX2["arms"] = False, []
    codes = []
    if AX2_VX:
        codes.append(_ax2_stcode(AX2_VX, AX2_ST, False))
    for (st, vx) in AX2_LAB:
        codes.append(_ax2_stcode(vx, st, False))
    occ = {{c: int(_K1_MOD.k1_occupancy(1, c)) for c in codes}}
    print(f"[ax2] occupancy (CTAs/SM) {{occ}}", flush=True)
    _AX2["ok"] = bool(AX2_VX) and occ.get(_ax2_stcode(AX2_VX, AX2_ST, False), 0) >= 2
    lab = [c for c in codes[(1 if AX2_VX else 0):] if occ.get(c, 0) >= 2]
    _AX2["tl"] = [c for c in lab if c >= 1000 and ((c - 1000) // 100) & 16]
    _AX2["arms"] = [c for c in lab if c not in _AX2["tl"]]
    _AX2["tl_done"] = set()
    if AX2_LAB:   # research arms vs the ship on self-test cases (outputs and a clean workspace)
        _AX2["buf"] = (torch.ones(64 << 20, dtype=torch.float32, device=dev), torch.zeros((), dtype=torch.float32, device=dev))
        ws = _VA1_WS[str(dev)]
        for (B, ctx, page, kvs, seed) in ((16, 3000, 1, 1.0, 401), (5, 9000, 1, 1.0 / 64, 402), (33, 700, 1, 1.0, 403), (3, 1500, 64, 1.0, 404)):
            outs = {{}}
            for code in [204] + _AX2["arms"]:
                if code >= 1000 and ((code - 1000) // 100) & 4:
                    continue   # SK arms produce no output
                d = _k1_mk_case(dev, B, ctx, 1, False, page, kvs, seed)
                hq, hkv = d["config"]["num_q_heads"], d["config"]["num_kv_heads"]
                G = max(B + 1, _k1_sms(dev) * 2 // (hkv // 2))
                o = torch.empty((B, 1, hq * 64), dtype=torch.bfloat16, device=dev)
                _K1_MOD.k1_forward(d["qkv"], d["seq_lens"], d["cache_loc"], d["page_table"], d["k_cache"].view(torch.uint8),
                                   d["v_cache"].view(torch.uint8), d["k_scale"], d["v_scale"], ws[2], ws[0], ws[1], o, ws[3],
                                   B, 1, hq, hkv, 1, page.bit_length() - 1, G, VA1_SEGK, 0,
                                   float(d["config"]["qk_scale_factor"] * d["config"]["sm_scale"]), 1, 4, code)
                outs[code] = o.float()
            clean = not (int(ws[3].abs().sum()) or float(ws[0].abs().sum()) or float(ws[1].abs().sum()))
            ref = outs.pop(204)
            msg = " ".join(f"{{c}}:{{float((v - ref).abs().max()) / max(1e-6, float(ref.abs().max())):.1e}}" for c, v in outs.items())
            print(f"[ax2] lab vs ship B={{B}} ctx={{ctx}} page={{page}}: {{msg}} ws {{'clean' if clean else 'DIRTY'}}", flush=True)
            if not clean:
                _AX2["arms"] = []
                print("[ax2] lab arms OFF (dirty workspace)", flush=True)
    print(f"[ax2] live vx {{AX2_VX if _AX2['ok'] else 0}} st {{AX2_ST}}, lab arms {{_AX2['arms']}}", flush=True)

'''


PY_SLIM = '''
AX2_VX = {vx}   # ax2 VA1 variant (IX 1 | EP 2), 0 = ship (tools/ax2_patches.py)
AX2_ST = {st}
_AX2 = {{}}


def _ax2_code(st, data):
    if not AX2_VX or not _AX2.get("ok") or (AX2_VX & 1 and data["page_table"].data_ptr() % 16):
        return st
    return 1000 + 100 * AX2_VX + (10 if st >= 210 else 0) + AX2_ST


def _ax2_prepare(dev):
    try:
        _AX2["ok"] = AX2_VX > 0 and int(_K1_MOD.k1_occupancy(1, 1000 + 100 * AX2_VX + AX2_ST)) >= 2
    except Exception as ex:  # noqa: BLE001
        _AX2["ok"] = False
        print(f"[ax2] OFF: {{ex!r}}", flush=True)
    print(f"[ax2] VA1 vx {{AX2_VX if _AX2['ok'] else 0}} st {{AX2_ST}}", flush=True)

'''


def _strip(block):
    """slim mode: drop // comments from our own inserted CUDA (none sit inside string literals)."""
    out = []
    for ln in block.split("\n"):
        i = ln.find("//")
        if i >= 0:
            ln = ln[:i].rstrip()
            if not ln.strip():
                continue
        out.append(ln)
    return "\n".join(out)


def _feature(block, keep, start, end):
    """slim mode: drop an `if constexpr (X) { ... }` feature block (start .. end markers) when its bit is off."""
    if keep:
        return block
    i0 = block.find(start)
    i1 = block.find(end, i0)
    if i0 < 0 or i1 < 0:
        raise SystemExit("[ax2_patches] feature markers")
    return block[:i0] + block[i1 + len(end):]


def _notl(block):
    return "\n".join(ln for ln in block.split("\n") if "(TL)" not in ln and "bool TL" not in ln)


def patch(src, vx, st, lab, reps, slim=False, extra=()):
    global CONSTS, SMEM_NEW, INIT_NEW, PROD_NEW, RED_NEW, CNT_NEW, PARAMS_NEW, FWD_NEW, KERN_NEW
    live_vx, vx = vx, vx
    for _st, _vx in extra:
        vx |= _vx
    if slim:
        PROD_NEW = _feature(PROD_NEW, vx & 384, "      if constexpr (PFT > 0) {", "      }   // PF end\n")
        if not vx & 384:
            KERN_NEW = KERN_NEW.replace("  constexpr int PFT = (VX & 256) ? 8 : ((VX & 128) ? 4 : 0);\n", "")
        KERN_NEW, PROD_NEW, CNT_NEW = _notl(KERN_NEW), _notl(PROD_NEW), _notl(CNT_NEW)
        i0, i1 = SMEM_NEW.find("__device__ __forceinline__ void ax_stamp"), SMEM_NEW.find("}\n", SMEM_NEW.find("= t;\n")) + 2
        SMEM_NEW = SMEM_NEW[:i0] + SMEM_NEW[i1:]
        SMEM_NEW, INIT_NEW, PROD_NEW, RED_NEW, CNT_NEW, PARAMS_NEW, FWD_NEW = (
            _strip(x) for x in (SMEM_NEW, INIT_NEW, PROD_NEW, RED_NEW, CNT_NEW, PARAMS_NEW, FWD_NEW))
        PROD_NEW = _feature(PROD_NEW, vx & 1, "    if constexpr (IX) {", "      return;\n    }\n")
        INIT_NEW = _feature(INIT_NEW, vx & 1, "  if constexpr (IX) {", "    }\n  }\n")
        RED_NEW = _feature(RED_NEW, vx & 2, "    if constexpr (EP) {", "        mA = mm;\n      }\n    }\n")
        if not vx & 32:   # no TX: the cp.async index fill only
            PROD_NEW = _feature(PROD_NEW, False, "#if __CUDA_ARCH__ >= 900\n        if constexpr (TX) {", "        } else\n#endif\n")
            fc_old = "#if __CUDA_ARCH__ >= 900\n      const int fc = TX ? 1 : 32;\n#else\n      const int fc = 32;\n#endif\n"
            if INIT_NEW.count(fc_old) != 1:
                raise SystemExit("[ax2_patches] fc block")
            INIT_NEW = INIT_NEW.replace(fc_old, "      const int fc = 32;\n")
            INIT_NEW = _feature(INIT_NEW, False, "#if __CUDA_ARCH__ >= 900\n      if constexpr (TX)", "#endif\n")
        if not vx & 1536:   # no NW / W1
            PROD_NEW = _feature(PROD_NEW, False, "#if __CUDA_ARCH__ >= 900\n          if constexpr (NWT >= 0) {", "#endif\n")
            KERN_NEW = KERN_NEW.replace("  constexpr int NWT = (VX & 512) ? 0 : ((VX & 1024) ? 1 : -1);\n", "")
        if not vx & 2048:   # no LT
            PROD_NEW = _feature(PROD_NEW, False, "#if __CUDA_ARCH__ >= 900\n      if constexpr (LT)", "#endif\n")
            KERN_NEW = KERN_NEW.replace("  constexpr bool LT = (VX & 2048) != 0;\n", "")
        if not vx & 64:   # no EP2: the ship gate load and fence
            RED_NEW = _feature(RED_NEW, False, "    uint4 ax_g = make_uint4(0u, 0u, 0u, 0u);\n", "8 * gu_));\n    }\n")
    s = src
    s = _rep(s, SMEM_OLD, SMEM_NEW, "va_smem")
    s = _rep(s, KERN_OLD, KERN_NEW.replace(", SK = (VX & 4) != 0", "") if slim else KERN_NEW, "va1_main template")
    s = _rep(s, INIT_OLD, INIT_NEW, "mbarrier init")
    s = _rep(s, PROD_OLD, PROD_NEW, "producer head")
    if not slim:
        s = _rep(s, PEND_OLD, PEND_NEW, "producer end / SK")
    s = _rep(s, RED_OLD, RED_NEW, "epilogue reds")
    # the fence before the counter: ship __threadfence(), layer FNC_WF() -> skipped by EP
    for fence in ("__threadfence();", "FNC_WF();"):
        old = "      if (t == 0) atomicAdd(p.part_lse + rb + h * 8 + g, lA * f);\n    }\n    " + fence + "\n" + CNT_OLD
        if s.count(old) == 1:
            s = s.replace(old, "      if (t == 0) atomicAdd(p.part_lse + rb + h * 8 + g, lA * f);\n    }\n    if constexpr (!EP) { " + fence
                          + " }\n" + CNT_NEW)
            break
    else:
        raise SystemExit("[ax2_patches] epilogue fence / counter anchor not found")
    if not slim or vx & 2048:   # LT: the top-of-kernel trigger moves to the producer's end
        s = _rep(s, '  if (p.use_rope) asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");   // use_rope = PDL: o proj may start\n',
                 '  if (!LT && p.use_rope) asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");   // use_rope = PDL: o proj may start\n',
                 "VA1 trigger")
    ep2 = not slim or bool(vx & 64)
    if ep2:
        s = _rep(s, GATE_OLD, GATE_NEW if not slim else _strip(GATE_NEW), "EP2 gate")
    flag_line = "    if (flag[0]) {                                   // last share of (b, hp): finalize, zero what was read\n"
    for fence in (("__threadfence();", "FNC_MF();") if ep2 else ()):
        old = flag_line + "      " + fence + "\n"
        if s.count(old) == 1:
            s = s.replace(old, flag_line + "#if __CUDA_ARCH__ >= 900\n      if constexpr (EP2) asm volatile(\"fence.acquire.gpu;\" ::: \"memory\"); else\n"
                          "#endif\n      " + fence + "\n")
            break
    else:
        if ep2:
            raise SystemExit("[ax2_patches] finalize fence anchor not found")
    if not slim:
        s = _rep(s, TLC_OLD, TLC_NEW, "TL consumer stamps")
        s = _rep(s, TLF_OLD, TLF_NEW, "TL finalize stamp")
    s = _rep(s, PARAMS_OLD, PARAMS_NEW, "Params")
    s = _rep(s, GO_OLD, GO_NEW, "va_go template")
    s = s.replace("cudaFuncSetAttribute(k1::va1_main<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, k1::va_smem<ST>())",
                  "cudaFuncSetAttribute(k1::va1_main<ST, VX>, cudaFuncAttributeMaxDynamicSharedMemorySize, k1::va_smem<ST, VX>())")
    s = _rep(s, "cfg.dynamicSmemBytes = k1::va_smem<ST>(); cfg.stream = s;", "cfg.dynamicSmemBytes = k1::va_smem<ST, VX>(); cfg.stream = s;",
             "va_go smem")
    s = _rep(s, "C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k1::va1_main<ST>, *p));", "C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k1::va1_main<ST, VX>, *p));",
             "va_go launch")
    s = _rep(s, "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k1::va1_main<ST>, 256, k1::va_smem<ST>())",
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k1::va1_main<ST, VX>, 256, k1::va_smem<ST, VX>())", "va_go occupancy")
    if s.count("k1::va1_main<ST>") or s.count("k1::va_smem<ST>()"):
        raise SystemExit("[ax2_patches] unpatched va_go reference left")
    arms = list(lab) + list(extra) + ([(st, live_vx)] if live_vx else [])
    # dispatch after va_go's definition (before k1_occupancy)
    s = _rep(s, "int64_t k1_occupancy(int64_t L, int64_t ST) {\n", dispatch_src(arms) + "int64_t k1_occupancy(int64_t L, int64_t ST) {\n",
             "dispatch before k1_occupancy")
    i0 = s.find(STOK_OLD)
    i1 = s.find("; }\n", i0)
    if i0 < 0 or s.count(STOK_OLD) != 1 or i1 < 0 or "\n" in s[i0:i1]:
        raise SystemExit("[ax2_patches] stages_ok: anchor")
    s = s[:i0] + STOK_NEW + "(" + s[i0 + len(STOK_OLD):i1] + ")" + s[i1:]
    s = _rep(s, OCC_OLD, OCC_NEW, "k1_occupancy")
    s = _rep(s, FWD_OLD, FWD_NEW, "k1_forward")
    # Python glue
    s = _rep(s, "    st += 10 if pdl else 0\n    ws = _VA1_WS[str(dev)]\n",
             "    st += 10 if pdl else 0\n    st = _ax2_code(st, data)   # ax2 (tools/ax2_patches.py)\n    ws = _VA1_WS[str(dev)]\n", "_va1_run code")
    old_ret = ("                       1 if pdl else 0, float(cfg[\"qk_scale_factor\"] * cfg[\"sm_scale\"]), 1, 4, st)\n    return out\n")
    if not slim:
        s = _rep(s, old_ret, old_ret.replace("    return out\n", "    if AX2_LAB:\n        _ax2_lab(qkv, data, cfg, G, max(1, int(VA1_SEGK if segk is None else segk)))\n    return out\n"),
                 "_va1_run lab hook")
    blk = PY_SLIM.format(vx=live_vx, st=st) if slim else py_block(live_vx, st, lab, reps)
    s = _rep(s, "\ndef _va1_try(qkv, data, cfg, state):\n", blk + "\ndef _va1_try(qkv, data, cfg, state):\n", "py block")
    ws_line = ("        _VA1_WS[str(dev)] = ws = (z(VA1_ROWS, 64), z(VA1_ROWS), _k1_inv_freq(0.0, dev),\n"
               "                                  torch.zeros(VA1_ROWS // 16, dtype=torch.int32, device=dev))\n")
    s = _rep(s, ws_line, ws_line + "        _ax2_prepare(dev)   # ax2: occupancy, live flag, lab arms (before the VA1 self-test)\n", "_va1_prepare hook")
    return s


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_rider3_1.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_ax2_1.py")
    vx, st = int(ARGS.get("vx", "3")), int(ARGS.get("st", "3"))
    lab = []
    if ARGS.get("lab"):
        for a in ARGS["lab"].split(","):
            ss, vv = a.split(":")
            lab.append((int(ss), int(vv)))
    reps = int(ARGS.get("reps", "2"))
    src = base.read_text(encoding="utf-8")
    slim = ARGS.get("slim", "0" if lab else "1") == "1"
    if slim and lab:
        raise SystemExit("[ax2_patches] --slim has no lab arms")
    extra = [tuple(int(x) for x in a.split(":")) for a in ARGS.get("extra", "").split(",") if a]   # ST:VX built, not live
    s = patch(src, vx, st, lab, reps, slim, extra)
    lines = s.split("\n")
    for i in range(min(3, len(lines))):
        if lines[i].startswith("# nonce"):
            lines[i] = f"# nonce ax2 vx{vx} st{st} lab{len(lab)} ({base.name} + tools/ax2_patches.py)"
            break
    s = "\n".join(lines)
    ast.parse(s)
    out.write_bytes(s.encode("utf-8"))
    print(f"[ax2_patches] {base.name} -> {out.relative_to(ROOT)} ({len(s.encode()) // 1024} KB) vx {vx} st {st} lab {lab}")


if __name__ == "__main__":
    main()

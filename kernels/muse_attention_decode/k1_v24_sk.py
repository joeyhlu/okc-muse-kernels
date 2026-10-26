"""K1 v24 patch texts (LAB source of truth; tools/k1_v24_patches.py exports them against k3_v23.py).

SK_*: stream-K split plan for L=1 fold calls (host passes fold == 2 when K1_STREAMK[(kind, L)]): CTA c takes the balanced
global range [floor(c T / Gx), floor((c + 1) T / Gx)) of the concatenated old-tile sequence (T = sum of every request's
old tiles), processed as one segment per request it touches (loop in k1_main). Segment (c, b) writes partial slot c + b
(the slots of request b are contiguous: [c_first(b) + b, c_last(b) + b]); the fold counter of (b, hp) expects
c_last - c_first + 1 arrivals; the segment holding b's last old tile attends / stores the new keys. Same math per tile;
only the split boundaries (hence FP32 merge grouping) change. Falls back in-kernel to the per-request plan when any
request has 0 old tiles or T < Gx (uniform per call).
PAIR: the L=1 per-tile CTA barrier -> a 64-thread named barrier per warp pair (warps 2 q4 and 2 q4 + 1 load exactly the
16 keys of both heads that they consume); bitwise.
"""

SK_FUNC_ANCHOR = ("// ---------------------------------------------------------------------------------------------\n"
                  "// Tile compute (k1_emulate.emulate_k1.process).\n")
SK_FUNC = r"""// v24 stream-K: segment of CTA c starting at global old tile x (warp-cooperative, all 32 lanes).
// res = {b, s0, s1 (tiles of b, [s0, s1)), S_b (CTAs touching b), slot base c_first(b) + b, next x, nt_b}
__device__ __forceinline__ int sk_cmax(int x, int Gx, int T) {   // last CTA c with floor(c T / Gx) <= x
  return min(Gx - 1, (int)((((long long)x + 1) * Gx - 1) / T));
}
__device__ void sk_seg(const int* __restrict__ seq_lens, int B, int W, int Gx, int T, int c, int x, int (&res)[7]) {
  const int lane = threadIdx.x & 31;
  const int xe = (int)(((long long)(c + 1) * T) / Gx);
  int carry = 0;
  res[0] = -1; res[1] = 0; res[2] = 0; res[3] = 1; res[4] = 0; res[5] = xe; res[6] = 0;
  for (int b0 = 0; b0 < B; b0 += 32) {
    const int bb = b0 + lane;
    const int nt = (bb < B) ? old_tiles(__ldg(seq_lens + bb), W) : 0;
    int incl = nt;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int v = __shfl_up_sync(FULL, incl, d);
      if (lane >= d) incl += v;
    }
    const int off = carry + incl - nt;
    const bool hit = (bb < B) && off <= x && x < off + nt;
    const unsigned m = __ballot_sync(FULL, hit);
    if (m) {   // warp-uniform
      const int src = __ffs(m) - 1;
      const int rb = __shfl_sync(FULL, bb, src), ro = __shfl_sync(FULL, off, src), rn = __shfl_sync(FULL, nt, src);
      const int se = min(xe, ro + rn);
      const int cf = sk_cmax(ro, Gx, T), cl = sk_cmax(ro + rn - 1, Gx, T);
      res[0] = rb; res[1] = x - ro; res[2] = se - ro; res[3] = cl - cf + 1; res[4] = cf + rb; res[5] = se; res[6] = rn;
    }
    carry += __shfl_sync(FULL, incl, 31);
  }
}

"""

SK_HEAD_OLD = """  if (warp == 0) {
    const int C = plan_C(p.seq_lens, p.B, p.W, p.Gx, p.min_chunk);
    int res[5];
    plan_find(p.seq_lens, p.B, p.W, C, c, true, res);
    if (lane == 0) {
      plan_s[0] = res[0]; plan_s[1] = res[1]; plan_s[2] = res[2]; plan_s[3] = res[4]; plan_s[4] = res[3];
    }
  }
  __syncthreads();
  K1P(1);
  const int b = plan_s[0];
  if (b < 0) return;                                  // idle CTA (uniform)
"""
SK_HEAD_NEW = """  // v24 stream-K (L=1, fold == 2): this CTA's balanced global old-tile range [skx0, skx1), one segment per request
  if (warp == 0) {
    int T = 0, mn = 1 << 30;
    if (L == 1 && p.fold == 2) {
      for (int b0 = 0; b0 < p.B; b0 += 32) {
        const int bb = b0 + lane;
        const int nt = (bb < p.B) ? old_tiles(__ldg(p.seq_lens + bb), p.W) : (1 << 30);
        if (bb < p.B) T += nt;
        mn = min(mn, nt);
      }
      T = warp_sum_i(T);
#pragma unroll
      for (int d = 16; d > 0; d >>= 1) mn = min(mn, __shfl_xor_sync(FULL, mn, d));
    }
    if (lane == 0) {
      const bool skv = (L == 1) && p.fold == 2 && mn >= 1 && T >= p.Gx;
      plan_s[9] = skv ? 1 : 0;
      plan_s[10] = T;
      plan_s[11] = skv ? (int)(((long long)c * T) / p.Gx) : 0;
      plan_s[12] = skv ? (int)(((long long)(c + 1) * T) / p.Gx) : 0;
    }
  }
  __syncthreads();
  const bool sk = plan_s[9] != 0;
  const int skT = plan_s[10], skx1 = plan_s[12];
  for (int skx = plan_s[11], sks = 0; sks == 0 || (sk && skx < skx1); ++sks) {
  if (sks > 0) __syncthreads();                       // the previous segment is done with plan_s and the ring
  if (warp == 0) {
    if (!sk) {
      const int C = plan_C(p.seq_lens, p.B, p.W, p.Gx, p.min_chunk);
      int res[5];
      plan_find(p.seq_lens, p.B, p.W, C, c, true, res);
      if (lane == 0) {
        plan_s[0] = res[0]; plan_s[1] = res[1]; plan_s[2] = res[2]; plan_s[3] = res[4]; plan_s[4] = res[3];
      }
    } else {
      int res[7];
      sk_seg(p.seq_lens, p.B, p.W, p.Gx, skT, c, skx, res);
      if (lane == 0) {
        plan_s[0] = res[0]; plan_s[1] = 0; plan_s[2] = res[3]; plan_s[3] = res[6]; plan_s[4] = res[4];
        plan_s[6] = res[1]; plan_s[7] = res[2]; plan_s[8] = res[5];
      }
    }
  }
  __syncthreads();
  K1P(1);
  const int b = plan_s[0];
  if (sk) skx = plan_s[8];                            // the next segment's first global tile
  if (b < 0) return;                                  // idle CTA (uniform)
"""

SK_SPLIT_OLD = """  const int s_idx = plan_s[1], S_b = plan_s[2], nt_b = plan_s[3], off_b = plan_s[4];
  // balanced split boundaries: tiles [s*nt/S, (s+1)*nt/S), each <= C tiles
  const int t0 = (int)(((long long)s_idx * nt_b) / S_b);
  const int ntl = (int)(((long long)(s_idx + 1) * nt_b) / S_b) - t0;
  const bool last = (s_idx == S_b - 1);
"""
SK_SPLIT_NEW = """  const int s_idx = plan_s[1], S_b = plan_s[2], nt_b = plan_s[3], off_b = plan_s[4];
  // balanced split boundaries: tiles [s*nt/S, (s+1)*nt/S), each <= C tiles (stream-K: the segment's tiles [s0, s1))
  const int t0 = sk ? plan_s[6] : (int)(((long long)s_idx * nt_b) / S_b);
  const int ntl = sk ? plan_s[7] - plan_s[6] : (int)(((long long)(s_idx + 1) * nt_b) / S_b) - t0;
  const bool last = sk ? (plan_s[7] == nt_b) : (s_idx == S_b - 1);
"""

SK_ONE_OLD = """    merge_finalize<L>(p, ost, lse_s, 1, b, hp, false);
    K1P(5);
    return;
  }
"""
SK_ONE_NEW = """    merge_finalize<L>(p, ost, lse_s, 1, b, hp, false);
    K1P(5);
    continue;                                         // v24: next segment (the loop ends for the per-request plan)
  }
"""

SK_PBASE_OLD = "    const size_t pbase = (size_t)hp * p.Gx + c;\n"
SK_PBASE_NEW = "    const size_t pbase = sk ? (size_t)hp * (p.Gx + p.B) + c + b : (size_t)hp * p.Gx + c;   // v24 slot c + b\n"

SK_BASE_OLD = "      const size_t base = (size_t)hp * p.Gx + off_b;\n"
SK_BASE_NEW = "      const size_t base = sk ? (size_t)hp * (p.Gx + p.B) + off_b : (size_t)hp * p.Gx + off_b;   // v24\n"

SK_END_OLD = ("  } else {\n    K1P(5);\n  }\n#endif\n}\n\n"
              "// ---------------------------------------------------------------------------------------------\n"
              "// k1_combine")
SK_END_NEW = ("  } else {\n    K1P(5);\n  }\n  }   // v24: segment loop\n#endif\n}\n\n"
              "// ---------------------------------------------------------------------------------------------\n"
              "// k1_combine")

SK_PATCHES = [
    ("sk func", SK_FUNC_ANCHOR, SK_FUNC + SK_FUNC_ANCHOR),
    ("sk head", SK_HEAD_OLD, SK_HEAD_NEW),
    ("sk split", SK_SPLIT_OLD, SK_SPLIT_NEW),
    ("sk single", SK_ONE_OLD, SK_ONE_NEW),
    ("sk pbase", SK_PBASE_OLD, SK_PBASE_NEW),
    ("sk base", SK_BASE_OLD, SK_BASE_NEW),
    ("sk end", SK_END_OLD, SK_END_NEW),
]

PAIR_PATCH = ("pair barrier",
              "      cp_async_wait<ST - 2>();\n      __syncthreads();\n      {\n        const int nx = i + ST - 1;",
              "      cp_async_wait<ST - 2>();\n"
              "      asm volatile(\"bar.sync %0, 64;\\n\" :: \"r\"(1 + q4) : \"memory\");   // v24: warp-pair barrier\n"
              "      {\n        const int nx = i + ST - 1;")

# host (k1_v7.py _k1 / k3_v23.py _k1_run): partial slots Gx + B per head pair, fold == 2 on K1_STREAMK routes
HOST_ALLOC_OLD = ("    part_o = torch.empty((hp, gx, 16 * L, 64), dtype=torch.float32, device=dev)\n"
                  "    part_lse = torch.empty((hp, gx, 16 * L), dtype=torch.float32, device=dev)\n")
HOST_ALLOC_NEW = ("    part_o = torch.empty((hp, gx + B, 16 * L, 64), dtype=torch.float32, device=dev)   # v24: stream-K slots\n"
                  "    part_lse = torch.empty((hp, gx + B, 16 * L), dtype=torch.float32, device=dev)\n")
HOST_FOLD_OLD = "                    1 if fold else 0, int(nsg), int(stages))\n"
HOST_FOLD_NEW = "                    ((2 if K1_STREAMK.get(key, False) else 1) if fold else 0), int(nsg), int(stages))\n"
HOST_TABLE_OLD = "K1_STAGES_T = "
HOST_TABLE_NEW = ("K1_STREAMK = {(\"full\", 1): True}   # v24: stream-K split plan on these fold routes (kernel fold == 2)\n"
                  "K1_STAGES_T = ")
HOST_PATCHES = [("host table", HOST_TABLE_OLD, HOST_TABLE_NEW), ("host alloc", HOST_ALLOC_OLD, HOST_ALLOC_NEW),
                ("host fold", HOST_FOLD_OLD, HOST_FOLD_NEW)]

# SECT: L=1 loader chunk map ch = part + 4 u instead of 2 part + u: each 16-B cp.async instruction of a warp then
# covers whole 32-B sectors (u = 0: bytes 0..63, u = 1: 64..127 of the key rows) instead of every other 16-B chunk
# of all four sectors; the same bytes land at the same swizzled smem positions (bitwise).
SECT_PATCH = ("sector loader",
              "        const int ch = 2 * part + u;\n        cp_async16(stg + lk * ROWB",
              "        const int ch = part + 4 * u;   // v24: sector-complete 16-B requests\n        cp_async16(stg + lk * ROWB")

"""va2_src.py: VA2 sources (verify full, q_len 8, B >= 2): flat warp-specialized FP8-wgmma attention inside K1's CUDA source
(imported by tools/va_patches.py; notes/study/va_attn.md s5-6).

One CTA = one KV pair x one equal share of the virtual key line (VA1's partition: n_b = seq_b + 1, SEGK per boundary),
2 CTAs/SM, 288 threads: warpgroup h (warps 4h .. 4h+3) = KV head h, warp 8 = producer. Per 64-key tile the producer copies
K into K1 L=8's wgmma K-major layout and raw V into K1's tid-indexed slots (whole 128-B rows, both heads), arrives on
full[st]; warpgroup h runs K1 L=8's numerics on its own head only: QK = 2 x wgmma m64n64k32 e4m3 SS, V^T of tile i+1 built
by its own 128 threads (K1's transpose for the virtual tids of head h) while QK(i) runs, softmax_pack7 (exact max, e4m3 P,
2^8 offset), PV = 2 x wgmma m64n72k32 RS (ones rows = row sums), arrive empty[st]. New tokens: K1 L=8's new K tile and new
V^T tile per warpgroup head, causal mask. Merge: o*2^(m-8) / l*2^(m-8) by red.add.v2 into the fp32 workspace, key ticket,
in-kernel finalize (rmsnorm * sigmoid(gate)) that zeroes what it read.
"""

VA2_CUDA = r'''// VA2 (tools/va2_src.py): verify full q_len 8; part_o / part_lse / counters = o / l workspace + key tickets.
#ifndef VA2_DIAG
#define VA2_DIAG 0   // lab timing probes (wrong results): 1 no V^T transpose, 2 no softmax math, 3 no PV, 4 no QK, 5 no consumer work
#endif
template <int ST, int NVT, int NB = 0> constexpr int va2_off_new() { return ST * STAGE_BYTES + NVT * VT_BYTES; }
template <int ST, int NVT, int NB = 0> constexpr int va2_off_bar() { return va2_off_new<ST, NVT>() + (NB ? 0 : NEW8_BYTES) + 8192; }
template <int ST, int NVT, int NB = 0> constexpr int va2_smem() { return va2_off_bar<ST, NVT, NB>() + 16 * ST + 4 * 264 + 2048; }   // B <= 128
__device__ __forceinline__ uint64_t desc_sw128(const void* ptr) {   // K-major, 128-B swizzle, SBO 1024 B
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(ptr);
  return (uint64_t)((a & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
// VT 1: V^T per head = 72 rows (64 dims + 8 rows of e4m3 ones) x 64 keys in P-register order, K-major, 64-B swizzle
// (8-row atoms of 512 B; logical byte c of row n at (n >> 3) * 512 + (n & 7) * 64 + ((((c >> 4) ^ (n >> 1)) & 3) << 4) + (c & 15)),
// 4608 B per head, buffers 9728 B apart (512-B aligned); built per tile with 2 ldmatrix.x4.trans + 8 prmt + 8 4-byte stores.
__device__ __forceinline__ uint64_t desc_sw64(const void* ptr) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(ptr);
  return (uint64_t)((a & 0x3FFFF) >> 4) | (1ull << 16) | (32ull << 32) | (2ull << 62);
}
__device__ __forceinline__ int vt_sw(int n, int c) { return (n >> 3) * 512 + (n & 7) * 64 + ((((c >> 4) ^ (n >> 1)) & 3) << 4) + (c & 15); }
__device__ __forceinline__ void pv_issue_sw(float (&O)[36], const uint32_t (&pa)[2][4], const uint8_t* vt) {
  wg_fence();
  const uint64_t d = desc_sw64(vt);
  wg_rs72(O, pa[0], d);
  wg_rs72(O, pa[1], d + 2);
  wg_commit();
}
__device__ __forceinline__ void qk_issue_sw(float (&S)[32], const uint64_t dq, const uint64_t dk) {
  wg_fence();
  wg_ss(S, dq, dk, 0);
  wg_ss(S, dq + 16, dk + 2, 1);
  wg_commit();
}
__device__ __forceinline__ void va_red2(float* a, float x, float y) {
#if __CUDA_ARCH__ >= 900
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n" ::"l"(a), "f"(x), "f"(y) : "memory");
#else
  atomicAdd(a, x); atomicAdd(a + 1, y);
#endif
}

// Lazy softmax (p.W & 2): P from the running offset with no per-tile max; the offset moves (exact softmax_pack7 on
// the unchanged S, with the O rescale) only when some packed e4m3 P byte exceeds 0x78 (P > 2^8; satfinite max 0x7E).
__device__ __forceinline__ bool va2_softmax_lazy(float (&S)[32], const float (&m)[2], uint32_t (&pa)[2][4], const int t,
                                                 const float cl2, const bool need, const int tile_abs, const int key_lim,
                                                 const int seq, const int j0) {
  if (m[0] == neg_inf() || m[1] == neg_inf()) return false;
  if (need) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int ak = tile_abs + 8 * j + 2 * t + e;
#pragma unroll
        for (int rb = 0; rb < 2; ++rb)
          if (!((ak < key_lim) && (ak <= seq + j0 + rb))) S[4 * j + 2 * rb + e] = neg_inf();
      }
  }
  const float b0 = m[0] - P_OFFSET, b1 = m[1] - P_OFFSET;
  uint32_t ov = 0u;
#pragma unroll
  for (int st = 0; st < 2; ++st) {
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      const int ja = 2 * (2 * st + h2), jb = ja + 1;
      pa[st][2 * h2 + 0] = pk4f(ex2(fmaf(S[4 * ja + 0], cl2, -b0)), ex2(fmaf(S[4 * ja + 1], cl2, -b0)),
                                ex2(fmaf(S[4 * jb + 0], cl2, -b0)), ex2(fmaf(S[4 * jb + 1], cl2, -b0)));
      pa[st][2 * h2 + 1] = pk4f(ex2(fmaf(S[4 * ja + 2], cl2, -b1)), ex2(fmaf(S[4 * ja + 3], cl2, -b1)),
                                ex2(fmaf(S[4 * jb + 2], cl2, -b1)), ex2(fmaf(S[4 * jb + 3], cl2, -b1)));
      ov |= ((pa[st][2 * h2] + 0x07070707u) | (pa[st][2 * h2 + 1] + 0x07070707u)) & 0x80808080u;
    }
  }
  return !__any_sync(FULL, ov != 0u);
}

// Registers: 2 CTAs/SM put 5-6 warps on an SM sub-partition (16 K regs each). PW 1: 288 threads, ptxas caps at 96 regs.
// PW 4: 384 threads (launch 80 regs), the producer warpgroup drops to 32 and the consumers rise to 104 (setmaxnreg;
// the pool must balance per sub-partition: 4 x (80 - 32) >= 8 x (104 - 80), else setmaxnreg.inc never returns).
template <int ST, int PW, int NVT, int KV = 0, int NB = 0, int DG = VA2_DIAG, int RP = 32, int PF = 0, int PP = 0, int VT = 0,
          int LZ = 0>
__global__ void __launch_bounds__(256 + 32 * PW, 2) va2_main(const Params p) {
  constexpr int VTB = VT ? 9728 : VT_BYTES;   // V^T buffer stride
#if K1_ARCH_OK
  constexpr int L = 8;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  uint8_t* smem = KV == 3 ? reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~(uintptr_t)1023) : smem_raw;
  uint8_t* vtb = smem + ST * STAGE_BYTES;
  uint8_t* newk = smem + va2_off_new<ST, NVT>();   // NB 1: no separate new tile (newk unused)
  uint8_t* qs = newk + (NB ? 0 : NEW8_BYTES);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + va2_off_bar<ST, NVT, NB>());
  uint64_t* empt = full + ST;
  int* flag = reinterpret_cast<int*>(empt + ST);
  int* sV = flag + 4;
  int* sN = sV + 129;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int hp = blockIdx.y, HP = gridDim.y, B = p.B, segk = p.min_chunk;
  constexpr int NPL = 32 * PW, MPT = 512 / NPL, XS = NPL / 8;   // producer lanes, (key, chunk) pairs per lane, key stride
  if (NB) {   // the V^T buffers double as new-token tiles: zero them once (no NaN bit patterns under P = 0)
    for (int i = tid; i < NVT * VT_BYTES / 16; i += 256 + NPL) reinterpret_cast<uint4*>(vtb)[i] = make_uint4(0u, 0u, 0u, 0u);
  } else {
    for (int i = tid; i < NEW8_BYTES / 16; i += 256 + NPL) reinterpret_cast<uint4*>(newk)[i] = make_uint4(0u, 0u, 0u, 0u);
  }
  if (tid < ST) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(va_sa(full + tid)), "n"(NPL) : "memory");
    asm volatile("mbarrier.init.shared.b64 [%0], 8;\n" ::"r"(va_sa(empt + tid)) : "memory");
  }
  if (warp == 8) {
    int carry = 0;
    for (int b0 = 0; b0 < B; b0 += 32) {
      const int b = b0 + lane;
      const int n = (b < B) ? __ldg(p.seq_lens + b) + 1 : 0, w = (b < B) ? n + segk : 0;
      int incl = w;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int v = __shfl_up_sync(FULL, incl, d);
        if (lane >= d) incl += v;
      }
      if (b < B) { sV[b] = carry + incl - w; sN[b] = n; }
      carry += __shfl_sync(FULL, incl, 31);
    }
    if (lane == 0) sV[B] = carry;
  }
  __syncthreads();
  if (VT && tid < 64 * NVT) {   // VT 1: rows 64..71 of both heads' V^T regions = e4m3 ones (row sums)
    const int buf = tid >> 6, hh = (tid >> 5) & 1, ch = tid & 31;
    *reinterpret_cast<uint4*>(vtb + buf * VTB + hh * 4608 + 4096 + ch * 16) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  }
  if (!VT && tid < 64 * (NVT + 1 - NB)) {   // e4m3 ones in row group 8 of every V^T region (NVT ring buffers + the new tile)
    const int buf = tid >> 6, rg = (tid >> 4) & 3, ch = tid & 15;
    uint8_t* base = (buf < NVT) ? (vtb + buf * VT_BYTES) : (newk + 8192);
    *reinterpret_cast<uint4*>(base + rg * VT_REGION + 8 * VT_SBO + (ch & 7) * 16 + (ch >> 3) * 128) =
        make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  }
#if __CUDA_ARCH__ >= 900
  if (p.use_rope) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
  const long long T = (long long)sV[B] - segk;
  const int v0 = (int)((T * blockIdx.x) / gridDim.x), v1 = (int)((T * (blockIdx.x + 1)) / gridDim.x);
  int bf = 0;
  for (int hi = B - 1; bf < hi;) {
    const int mid = (bf + hi) >> 1;
    if (sV[mid] + sN[mid] > v0) hi = mid;
    else bf = mid + 1;
  }
  const size_t rowB = (size_t)p.Hkv * 64;
  int n = 0;
  if (warp >= 8) {   // producers: (key x, chunk c) pairs x = (q >> 3) + XS m, c = q & 7
#if __CUDA_ARCH__ >= 900
    if (PW == 4) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(RP) : "memory");
#endif
    const int q = tid - 256, c = q & 7, x0 = q >> 3, psh = p.page_shift, pm = (1 << psh) - 1;
    const uint8_t* kc = p.k_cache + (size_t)hp * 128 + 16 * c;
    const uint8_t* vc = p.v_cache + (size_t)hp * 128 + 16 * c;
    // KV 1 (PW 4): K and V copied with a key-fastest lane map (8 lanes = 8 keys of one chunk: conflict-free in the
    // no-swizzle wgmma K layout and in the swizzled raw V rows). KV 0: chunk-fastest map (K stores 8-way conflicted).
    const int ck = (q >> 3) & 7, xk0 = (q & 7) + 8 * (q >> 6);
    const uint8_t* kck = p.k_cache + (size_t)hp * 128 + 16 * ck;
    if constexpr (KV == 2) {
      int* pix = flag + 264;   // [4][64] page indices of tiles n .. n + 3 (slot n & 3), after the segment table
      const uint8_t* vck = kck + (p.v_cache - p.k_cache);
      for (int b = bf; b < B && sV[b] < v1; ++b) {
        const int k0 = max(v0 - sV[b], 0), k1 = min(v1 - sV[b], sN[b]), cend = min(k1, sN[b] - 1);
        if (k1 <= k0 || cend <= k0) continue;
        const int nt = (cend - k0 + TILE - 1) / TILE;
        const int* pt = p.page_table + (size_t)b * p.pt_stride;
        auto ld = [&](int tt) { const int pos = k0 + tt * TILE + q; return (q < 64 && tt < nt && pos < cend) ? __ldg(pt + (pos >> psh)) : -1; };
        if (q < 64) { pix[(n & 3) * 64 + q] = ld(0); pix[((n + 1) & 3) * 64 + q] = ld(1); }
        int pend = ld(2);
        asm volatile("bar.sync 4, 128;\n" ::: "memory");
        for (int t = 0; t < nt; ++t, ++n) {
          if (q < 64) pix[((n + 2) & 3) * 64 + q] = pend;   // tile t + 2 (loaded one iteration ago)
          pend = ld(t + 3);
          const int st = n % ST;
          if (n >= ST) va_wait(empt + st, ((n / ST) - 1) & 1);
          uint8_t* stg = smem + st * STAGE_BYTES;
          const int* ix = pix + (n & 3) * 64;
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            const int xk = xk0 + 16 * m, posk = k0 + t * TILE + xk, pe = ix[xk];
            const uint32_t go = pe >= 0 ? (uint32_t)((pe << psh) | (posk & pm)) * (uint32_t)rowB : 0u;   // caches < 4 GB (host check)
            cp_async16(stg + (ck >> 2) * 4096 + (xk >> 3) * 512 + (ck & 3) * 128 + (xk & 7) * 16, kck + go, pe >= 0 ? 16 : 0);
            cp_async16(stg + 8192 + xk * 128 + ((ck ^ (xk & 7)) << 4), vck + go, pe >= 0 ? 16 : 0);
          }
          asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(va_sa(full + st)) : "memory");
          asm volatile("bar.sync 4, 128;\n" ::: "memory");
        }
      }
    } else
    for (int b = bf; b < B && sV[b] < v1; ++b) {
      const int k0 = max(v0 - sV[b], 0), k1 = min(v1 - sV[b], sN[b]), cend = min(k1, sN[b] - 1);
      if (k1 <= k0 || cend <= k0) continue;
      const int nt = (cend - k0 + TILE - 1) / TILE;
      const int* pt = p.page_table + (size_t)b * p.pt_stride;
      int pg[MPT];
#pragma unroll
      for (int m = 0; m < MPT; ++m) {
        const int x = (KV && KV != 3 && DG != 7) ? xk0 + 16 * m : x0 + XS * m;
        pg[m] = (k0 + x < cend) ? __ldg(pt + ((k0 + x) >> psh)) : -1;
      }
      for (int t = 0; t < nt; ++t, ++n) {
        const int st = n % ST;
        if (n >= ST) va_wait(empt + st, ((n / ST) - 1) & 1);
        uint8_t* stg = smem + st * STAGE_BYTES;
        if constexpr (DG == 7 || KV == 3) {   // whole rows (8 lanes = one 128-B row), chunk c at ((c ^ (r & 7)) << 4)
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            const int r = x0 + 16 * m, pos = k0 + t * TILE + r;
            const size_t go = pg[m] >= 0 ? (size_t)((pg[m] << psh) | (pos & pm)) * rowB : (size_t)0;
            cp_async16(stg + r * 128 + ((c ^ (r & 7)) << 4), kc + go, pg[m] >= 0 ? 16 : 0);
            cp_async16(stg + 8192 + r * 128 + ((c ^ (r & 7)) << 4), vc + go, pg[m] >= 0 ? 16 : 0);
            pg[m] = (pos + TILE < cend) ? __ldg(pt + ((pos + TILE) >> psh)) : -1;
          }
        } else if constexpr (KV) {
#pragma unroll
          for (int m = 0; m < MPT; ++m) {
            const int xk = xk0 + 16 * m, posk = k0 + t * TILE + xk;
            const size_t gk = pg[m] >= 0 ? (size_t)((pg[m] << psh) | (posk & pm)) * rowB : (size_t)0;
            cp_async16(stg + (ck >> 2) * 4096 + (xk >> 3) * 512 + (ck & 3) * 128 + (xk & 7) * 16, kck + gk, pg[m] >= 0 ? 16 : 0);
            cp_async16(stg + 8192 + xk * 128 + ((ck ^ (xk & 7)) << 4), kck + (p.v_cache - p.k_cache) + gk, pg[m] >= 0 ? 16 : 0);
            pg[m] = (posk + TILE < cend) ? __ldg(pt + ((posk + TILE) >> psh)) : -1;
          }
        } else {
#pragma unroll
        for (int m = 0; m < MPT; ++m) {
          const int x = x0 + XS * m, pos = k0 + t * TILE + x;
          const size_t go = pg[m] >= 0 ? (size_t)((pg[m] << psh) | (pos & pm)) * rowB : (size_t)0;
          const int vt = (x >> 4) * 64 + ((x >> 3) & 1) * 32 + ((x >> 1) & 3) * 8 + c;
          cp_async16(stg + (c >> 2) * 4096 + (x >> 3) * 512 + (c & 3) * 128 + (x & 7) * 16, kc + go, pg[m] >= 0 ? 16 : 0);
          cp_async16(stg + 8192 + (x & 1) * 4096 + vt * 16, vc + go, pg[m] >= 0 ? 16 : 0);
          const int nx = pos + TILE;
          pg[m] = (nx < cend) ? __ldg(pt + (nx >> psh)) : -1;
        }
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(va_sa(full + st)) : "memory");
      }
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    return;
  }
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  // consumers: warpgroup h = KV head h, wq = m-tile (tokens 2wq, 2wq + 1), ct = consumer tid
#if __CUDA_ARCH__ >= 900
  if (PW == 4) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(80 + (80 - RP) / 2) : "memory");
#endif
  const int h = warp >> 2, wq = warp & 3, wt = tid & 127, g = lane >> 2, t = lane & 3, j0 = 2 * wq;
  const int vtid = (wt >> 5) * 64 + ((wt >> 4) & 1) * 32 + ((wt >> 2) & 3) * 8 + 4 * h + (wt & 3);   // K1 tid of head h
  const int vt_own = vt_off(h, vtid >> 7, 16 * (wt & 3), 16 * ((vtid >> 6) & 1) + 4 * ((vtid >> 3) & 3) + 2 * ((vtid >> 5) & 1));
  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale), cl2 = p.qk_sm * ksc * LOG2E;
  const size_t stride = (size_t)(2 * p.Hq + 2 * p.Hkv) * 64;
  const uint64_t dq = desc_il(qs + h * 4096, 128, 512);
  auto transpose_v = [&](int nn) {
    const uint8_t* stg = smem + (nn % ST) * STAGE_BYTES + 8192;
    if constexpr (VT) {   // keys 16 wq .. 16 wq + 15 of head h, dims 32u .. 32u + 31 per ldmatrix.x4 (matrix m = 2 gB + b)
      const int mm = lane >> 3, key = 16 * wq + 8 * (mm & 1) + (lane & 7), vq = lane & 3, vp = lane >> 2;
      uint8_t* vt = vtb + (nn % NVT) * VTB + h * 4608;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        uint32_t r[4];
        ldsm_x4_t(r, stg + key * 128 + (((4 * h + 2 * u + (mm >> 1)) ^ (key & 7)) << 4));
#pragma unroll
        for (int gB = 0; gB < 2; ++gB) {
          const int n0 = 32 * u + 16 * gB + 2 * vp, c = 16 * wq + 4 * vq;
          *reinterpret_cast<uint32_t*>(vt + vt_sw(n0, c)) = __byte_perm(r[2 * gB], r[2 * gB + 1], 0x6420);
          *reinterpret_cast<uint32_t*>(vt + vt_sw(n0 + 1, c)) = __byte_perm(r[2 * gB], r[2 * gB + 1], 0x7531);
        }
      }
      return;
    }
    const int lk0 = 16 * (vtid >> 6) + 8 * ((vtid >> 5) & 1) + 2 * ((vtid >> 3) & 3), lc = vtid & 7;
    const uint4 va = *reinterpret_cast<const uint4*>(KV ? stg + lk0 * 128 + ((lc ^ (lk0 & 7)) << 4) : stg + vtid * 16);
    const uint4 vb = *reinterpret_cast<const uint4*>(KV ? stg + (lk0 + 1) * 128 + ((lc ^ ((lk0 + 1) & 7)) << 4)
                                                        : stg + 4096 + vtid * 16);
    uint8_t* dst = vtb + (nn % NVT) * VTB + vt_own;
    const uint32_t wa[4] = {va.x, va.y, va.z, va.w}, wb[4] = {vb.x, vb.y, vb.z, vb.w};
#pragma unroll
    for (int w4 = 0; w4 < 4; ++w4) {
      const uint32_t lo = __byte_perm(wa[w4], wb[w4], 0x5140), hi = __byte_perm(wa[w4], wb[w4], 0x7362);
      *reinterpret_cast<uint16_t*>(dst + ((4 * w4 + 0) & 7) * 16 + ((4 * w4 + 0) >> 3) * VT_SBO) = (uint16_t)(lo & 0xFFFFu);
      *reinterpret_cast<uint16_t*>(dst + ((4 * w4 + 1) & 7) * 16 + ((4 * w4 + 1) >> 3) * VT_SBO) = (uint16_t)(lo >> 16);
      *reinterpret_cast<uint16_t*>(dst + ((4 * w4 + 2) & 7) * 16 + ((4 * w4 + 2) >> 3) * VT_SBO) = (uint16_t)(hi & 0xFFFFu);
      *reinterpret_cast<uint16_t*>(dst + ((4 * w4 + 3) & 7) * 16 + ((4 * w4 + 3) >> 3) * VT_SBO) = (uint16_t)(hi >> 16);
    }
  };
  for (int b = bf; b < B && sV[b] < v1; ++b) {
    const int k0 = max(v0 - sV[b], 0), k1 = min(v1 - sV[b], sN[b]);
    if (k1 <= k0) continue;
    const int seq = sN[b] - 1, cend = min(k1, seq);
    const int nt = (cend > k0) ? (cend - k0 + TILE - 1) / TILE : 0;
    const bool has_new = (k1 == sN[b]);
    const __nv_bfloat16* qb = p.qkv + (size_t)b * L * stride;
    uint32_t qraw[16];   // QH (KV 3): every Q row load in flight before the first is used (one round trip per segment)
    if constexpr (KV == 3) {
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const int r = wq + 4 * k;
        qraw[k] = *reinterpret_cast<const uint32_t*>(qb + (size_t)(r >> 3) * stride + (size_t)((2 * hp + h) * 8 + (r & 7)) * 64 + 2 * lane);
      }
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) {                   // Q rows r = 8 token + gi of head h -> e4m3 wgmma A layout
      const int r = wq + 4 * k;
      float2 x = KV == 3 ? __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qraw[k]))
                         : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qb + (size_t)(r >> 3) * stride + (size_t)((2 * hp + h) * 8 + (r & 7)) * 64 + 2 * lane));
      const float rs = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f) + 1e-5f);
      x.x *= rs; x.y *= rs;
      *reinterpret_cast<uint16_t*>(qs + h * 4096 + il_off(r, 2 * lane, 128, 512)) =
          (uint16_t)__nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
    }
    if (has_new) {                                   // new K / V rows (kind, token j) of head h
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int R = wq + 4 * k, kind = R >> 3, j = R & 7, head = 2 * hp + h;
        float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qb + (size_t)j * stride + (size_t)(p.Hq + kind * p.Hkv + head) * 64 + 2 * lane));
        if (kind == 0) {
          const float rs = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f) + 1e-5f);
          x.x *= rs; x.y *= rs;
        }
        const float sc = kind ? vsc : ksc;
        const uint16_t q8 = (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(x.x / sc, x.y / sc), __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + (size_t)p.cache_loc[(size_t)b * L + j] * rowB + (size_t)head * 64 + 2 * lane) = q8;
        if (NB) {
        } else if (kind == 0) {
          *reinterpret_cast<uint16_t*>(newk + h * 4096 + il_off(j, 2 * lane, 128, 512)) = q8;
        } else {
          int s8, kl;
          key_to_sk(j, s8, kl);
          newk[8192 + vt_off(h, s8, 2 * lane, kl)] = (uint8_t)(q8 & 0xFF);
          newk[8192 + vt_off(h, s8, 2 * lane + 1, kl)] = (uint8_t)(q8 >> 8);
        }
      }
    }
    if (!PF && nt > 0) {
      va_wait(full + n % ST, (n / ST) & 1);
      if (DG != 1 && DG != 5) transpose_v(n);
    }
    fence_async_smem();
    asm volatile("bar.sync %0, 128;\n" ::"r"(1 + h) : "memory");
    float S[32], O[36];
#pragma unroll
    for (int i = 0; i < 36; ++i) O[i] = 0.f;
    float m[2] = {neg_inf(), neg_inf()};
    if (PP && h == 1 && nt > 0) asm volatile("bar.arrive 5, 256;\n" ::: "memory");   // WG 0 takes the first softmax turn
    uint32_t pa[2][4];
    for (int i = 0; i < nt; ++i, ++n) {
      const int ts = k0 + i * TILE;
      if constexpr (DG == 6 || DG == 7) {   // probes: the bare stage protocol
        va_wait(full + n % ST, (n / ST) & 1);
        __syncwarp();
        if (lane == 0) asm volatile("{\n.reg .b64 s;\nmbarrier.arrive.shared.b64 s, [%0];\n}\n" ::"r"(va_sa(empt + n % ST)) : "memory");
        m[0] = 0.f; m[1] = 0.f;
        continue;
      }
      if (PF) {
        va_wait(full + n % ST, (n / ST) & 1);
        fence_async_smem();                          // the producers' cp.async K rows -> the wgmma (async proxy)
      }
      if (DG != 4 && DG != 5) {
        if constexpr (KV == 3) qk_issue_sw(S, dq, desc_sw128(smem + (n % ST) * STAGE_BYTES + 64 * h));
        else qk_issue(S, dq, desc_il(smem + (n % ST) * STAGE_BYTES + h * 4096, 128, 512));
      }
      if (PF) {
        if (DG != 1 && DG != 5) transpose_v(n);
        fence_async_smem();
      }
      if (!PF && NVT == 2) wg_wait<1>();             // PV(i-1) done: V^T buffer (n + 1) % 2 is free (QK(i) may run)
      if (!PF && i + 1 < nt) {
        va_wait(full + (n + 1) % ST, ((n + 1) / ST) & 1);
        if (DG != 1 && DG != 5) transpose_v(n + 1);
        fence_async_smem();
      }
      wg_wait<0>();                                  // QK(i) and PV(i-1)
      __syncwarp();
      if (lane == 0) asm volatile("{\n.reg .b64 s;\nmbarrier.arrive.shared.b64 s, [%0];\n}\n" ::"r"(va_sa(empt + n % ST)) : "memory");
      asm volatile("bar.sync %0, 128;\n" ::"r"(1 + h) : "memory");   // V^T(i + 1) of every thread before PV(i + 1)
      if (DG == 2 || DG == 4 || DG == 5) {
#pragma unroll
        for (int q = 0; q < 8; ++q) pa[q >> 2][q & 3] = __float_as_uint(S[q]) & 0x37373737u;
        m[0] = 0.f; m[1] = 0.f;
      } else {
        if (PP) asm volatile("bar.sync %0, 256;\n" ::"r"(5 + h) : "memory");   // my softmax turn
        if (!((LZ || (p.W & 2)) && va2_softmax_lazy(S, m, pa, t, cl2, ts + TILE > cend, ts, cend, seq, j0)))
          softmax_pack7(S, O, m, pa, t, cl2, ts + TILE > cend, ts, cend, seq, 0, j0);
        if (PP && (h == 0 || i + 1 < nt)) asm volatile("bar.arrive %0, 256;\n" ::"r"(6 - h) : "memory");   // the other's turn
      }
      if (DG != 3 && DG != 5) {
        if constexpr (VT) pv_issue_sw(O, pa, vtb + (n % NVT) * VTB + h * 4608);
        else pv_issue7(O, pa, vtb + (n % NVT) * VT_BYTES, h);
      }
    }
    uint8_t* nK = KV == 3 ? vtb : NB ? vtb + ((n + 1) % NVT) * VT_BYTES : newk;   // KV 3: buffer 0 is 1024-B aligned
    uint8_t* nV = KV == 3 ? vtb + VTB : NB ? vtb + ((n + 2) % NVT) * VT_BYTES : newk + 8192;
    if (NB && has_new) {   // both warpgroups are past their loops before the V^T buffers take the new rows
      wg_wait<0>();
      asm volatile("bar.sync 3, 256;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int R = wq + 4 * k, kind = R >> 3, j = R & 7, head = 2 * hp + h;
        float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qb + (size_t)j * stride + (size_t)(p.Hq + kind * p.Hkv + head) * 64 + 2 * lane));
        if (kind == 0) {
          const float rs = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f) + 1e-5f);
          x.x *= rs; x.y *= rs;
        }
        const float sc = kind ? vsc : ksc;
        const uint16_t q8 = (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(x.x / sc, x.y / sc), __NV_SATFINITE, __NV_E4M3);
        if (kind == 0) {
          if constexpr (KV == 3) {
            const int bb = 64 * h + 2 * lane;
            *reinterpret_cast<uint16_t*>(nK + j * 128 + (((bb >> 4) ^ (j & 7)) << 4) + (bb & 15)) = q8;
          } else {
            *reinterpret_cast<uint16_t*>(nK + h * 4096 + il_off(j, 2 * lane, 128, 512)) = q8;
          }
        } else {
          int s8, kl;
          key_to_sk(j, s8, kl);
          if constexpr (VT) {
            nV[h * 4608 + vt_sw(2 * lane, 32 * s8 + kl)] = (uint8_t)(q8 & 0xFF);
            nV[h * 4608 + vt_sw(2 * lane + 1, 32 * s8 + kl)] = (uint8_t)(q8 >> 8);
          } else {
            nV[vt_off(h, s8, 2 * lane, kl)] = (uint8_t)(q8 & 0xFF);
            nV[vt_off(h, s8, 2 * lane + 1, kl)] = (uint8_t)(q8 >> 8);
          }
        }
      }
      fence_async_smem();
      asm volatile("bar.sync %0, 128;\n" ::"r"(1 + h) : "memory");
    }
    if (has_new) {
      if constexpr (KV == 3) qk_issue_sw(S, dq, desc_sw128(nK + 64 * h));
      else qk_issue(S, dq, desc_il(nK + h * 4096, 128, 512));
      wg_wait<0>();
      softmax_pack7(S, O, m, pa, t, cl2, true, seq, seq + L, seq, 0, j0);
      if constexpr (VT) pv_issue_sw(O, pa, nV + h * 4608);
      else pv_issue7(O, pa, nV, h);
    }
    wg_wait<0>();
    const size_t r0 = (size_t)(b * HP + hp) * 128;
#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
      const float f = (m[rb] == neg_inf()) ? 0.f : ex2(fminf(m[rb] - P_OFFSET, 126.f));
      float* wa = p.part_o + (r0 + h * 64 + 16 * wq + 8 * rb + g) * 64 + 2 * t;
#pragma unroll
      for (int j = 0; j < 8; ++j) va_red2(wa + 8 * j, O[4 * j + 2 * rb] * f, O[4 * j + 2 * rb + 1] * f);
      if (t == 0) atomicAdd(p.part_lse + r0 + h * 64 + 16 * wq + 8 * rb + g, O[32 + 2 * rb] * f);
    }
    __threadfence();
    asm volatile("bar.sync 3, 256;\n" ::: "memory");
    if (tid == 0) flag[0] = (atomicAdd(p.counters + b * HP + hp, k1 - k0) + k1 - k0 == sN[b]) ? 1 : 0;
    if (VT && NB && has_new && tid < 32)   // the new K tile covered head 0's ones rows of buffer 0
      *reinterpret_cast<uint4*>(nK + 4096 + tid * 16) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
    if (!VT && NB && has_new && tid < 64) {
      const int rg = tid >> 4, ch = tid & 15;
      *reinterpret_cast<uint4*>(nK + rg * VT_REGION + 8 * VT_SBO + (ch & 7) * 16 + (ch >> 3) * 128) =
          make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
    }
    asm volatile("bar.sync 3, 256;\n" ::: "memory");
    if (flag[0]) {
      __threadfence();
#pragma unroll 1
      for (int pass = 0; pass < 4; ++pass) {
        const int r = pass * 32 + (tid >> 3), u = tid & 7, hh = r >> 6, j = (r >> 3) & 7, qh = (2 * hp + hh) * 8 + (r & 7);
        float* ar = p.part_o + (r0 + r) * 64 + 8 * u;
        const float4 a0 = __ldcg(reinterpret_cast<const float4*>(ar)), a1 = __ldcg(reinterpret_cast<const float4*>(ar + 4));
        const float lt = __ldcg(p.part_lse + r0 + r), sc = (lt > 0.f) ? vsc / lt : 0.f;
        const uint4 gr = *reinterpret_cast<const uint4*>(qb + (size_t)j * stride + (size_t)(p.Hq + 2 * p.Hkv + qh) * 64 + 8 * u);
        const float y[8] = {a0.x * sc, a0.y * sc, a0.z * sc, a0.w * sc, a1.x * sc, a1.y * sc, a1.z * sc, a1.w * sc};
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) ss += y[e] * y[e];
        ss += __shfl_xor_sync(FULL, ss, 1);
        ss += __shfl_xor_sync(FULL, ss, 2);
        ss += __shfl_xor_sync(FULL, ss, 4);
        const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
        uint32_t ow[4] = {gr.x, gr.y, gr.z, gr.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ow[e]));
          const __nv_bfloat162 o2 = __floats2bfloat162_rn(y[2 * e] * rs * sigmoidf_(g2.x), y[2 * e + 1] * rs * sigmoidf_(g2.y));
          ow[e] = *reinterpret_cast<const uint32_t*>(&o2);
        }
        *reinterpret_cast<uint4*>(p.out + ((size_t)b * L + j) * p.Hq * 64 + (size_t)qh * 64 + 8 * u) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
        *reinterpret_cast<float4*>(ar) = make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(ar + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (u == 0) p.part_lse[r0 + r] = 0.f;
      }
      if (tid == 0) p.counters[b * HP + hp] = 0;
    }
  }
#endif
}

'''

VA2_HOST = r'''
template <int ST, int PW, int NVT, int KV = 0, int NB = 0, int DG = VA2_DIAG, int RP = 32, int PF = 0, int PP = 0, int VT = 0,
          int LZ = 0>
static int va2_go(const k1::Params* p, dim3 grid, cudaStream_t s, bool pdl) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k1::va2_main<ST, PW, NVT, KV, NB, DG, RP, PF, PP, VT, LZ>, cudaFuncAttributeMaxDynamicSharedMemorySize, k1::va2_smem<ST, NVT, NB>()));
    attr = true;
  }
  int nb = 0;
  if (p) {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = grid; cfg.blockDim = dim3(256 + 32 * PW); cfg.dynamicSmemBytes = k1::va2_smem<ST, NVT, NB>(); cfg.stream = s;
    cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k1::va2_main<ST, PW, NVT, KV, NB, DG, RP, PF, PP, VT, LZ>, *p));
  } else C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k1::va2_main<ST, PW, NVT, KV, NB, DG, RP, PF, PP, VT, LZ>, 256 + 32 * PW, k1::va2_smem<ST, NVT, NB>()));
  return nb;
}'''

VA2_PY = '''VA2_ENABLE = True   # VA2 (tools/va2_src.py): flat warp-specialized FP8-wgmma attention, verify full q_len 8, B >= 2
VA2_MIN_B = 2
VA2_SEGK = 600
VA2_PDL = False
VA2_PROBES = ()     # lab: VA2_DIAG probe codes (1 no V^T, 2 no softmax, 3 no PV, 5 no consumer work) of variant 348, run
                    # after the real verify-full attention on scratch outputs, timing only (distinct judge profile names)
VA2_LAZY = False    # lazy softmax offset (packed-byte overflow test), runtime flag W & 2
VA2_VAR = 347       # 347: producer warpgroup (setmaxnreg 32 / 104), KV map, 4 stages, 3 V^T; 348: same at 48 / 96; 345: 2 V^T; 303: 1 warp
VA2_CASES = ((2, 3000, 1, 1.0, 401), (3, 1500, 64, 1.0, 402), (16, 700, 1, 1.0 / 64, 403), (5, 9000, 1, 1.0, 404),
             (33, 300, 1, 1.0, 405), (32, 2500, 1, 1.0, 406), (2, 20, 64, 1.0, 407))


def _va2_run(qkv, data, cfg, segk=None, pdl=None):
    B, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    dev = qkv.device
    code = int(VA2_VAR)
    G = max(B + 1, _k1_sms(dev) * max(1, min(2, _K1_OCC.get((8, code, str(dev)), 1))) // (hkv // 2))
    pdl = VA2_PDL if pdl is None else pdl
    ws = _VA1_WS[str(dev)]
    out = torch.empty((B, 8, hq * 64), dtype=torch.bfloat16, device=dev)
    _K1_MOD.k1_forward(qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
                       data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], ws[2], ws[0], ws[1], out, ws[3],
                       B, 8, hq, hkv, 1 | (2 if VA2_LAZY else 0), cfg["page_size"].bit_length() - 1, G,
                       max(1, int(VA2_SEGK if segk is None else segk)),
                       1 if pdl else 0, float(cfg["qk_scale_factor"] * cfg["sm_scale"]), 1, 4, code + (10 if pdl else 0))
    return out


def _va2_probes(qkv, data, cfg):
    B, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    dev = qkv.device
    G = max(B + 1, _k1_sms(dev) * max(1, min(2, _K1_OCC.get((8, 353, str(dev)), 1))) // (hkv // 2))
    ws = _VA1_WS[str(dev)]
    for dg in VA2_PROBES:
        scratch = torch.empty((B, 8, hq * 64), dtype=torch.bfloat16, device=dev)
        _K1_MOD.k1_forward(qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
                           data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], ws[2], ws[0], ws[1], scratch,
                           ws[3], B, 8, hq, hkv, 1, cfg["page_size"].bit_length() - 1, G, max(1, int(VA2_SEGK)), 0,
                           float(cfg["qk_scale_factor"] * cfg["sm_scale"]), 1, 4, 1000 * int(dg) + 353)


def _va2_try(qkv, data, cfg, state):
    B = cfg["batch"]
    if (VA2_ENABLE and state.get("va2_ok") and cfg["q_len"] == 8 and not cfg["window"] and not cfg["rope_theta"]
            and VA2_MIN_B <= B <= 128 and B * cfg["num_kv_heads"] * 64 <= VA1_ROWS
            and str(data["k_cache"].device) in _VA1_WS and data["k_cache"].numel() < (1 << 32)
            and _k1_supported(data, dict(cfg, batch=1 << 20))):
        try:
            o = _va2_run(qkv, data, cfg).view(B * 8, cfg["num_q_heads"] * 64)
            if VA2_PROBES and state.get("va2_probes_ok"):
                _va2_probes(qkv, data, cfg)
            return o
        except Exception as ex:
            _k3_disable(state, "va2_ok", False, "VA2", ex)
    return None


def _va2_case(dev, B, ctx, page, kvs, seed):
    d = _k1_mk_case(dev, B, ctx, 8, False, page, kvs, seed)
    out = _va2_run(d["qkv"], d, d["config"])
    pos = d["seq_lens"].long()[:, None] + torch.arange(8, device=dev)
    slots = torch.gather(d["page_table"].long(), 1, pos // page) * page + pos % page
    kn, vn = (c.view(torch.uint8).view(-1, 2, 64)[slots] for c in (d["k_cache"], d["v_cache"]))
    tot, row = _k1_rel_rows(out, _k1_ref_attention(d, kn, vn, q_e4m3=True))
    kref, vref = _k1_ref_new_kv(d)
    kt, vt = _k1_rel_rows(kn.view(FP8).float(), kref)[0], _k1_rel_rows(vn.view(FP8).float(), vref)[0]
    return out, tot, row, kt, vt


def _va2_prepare(state):
    state["va2_ok"] = False
    dev = torch.empty(0, device="cuda").device
    try:
        if not state.get("va1_ok"):
            raise RuntimeError("VA1 (workspace, K1 module) not live")
        for code in (303, 345, 347, 348, 349, 350, 351, 353, 401, 402, 403, 404):
            _k1_occupancy(8, code, dev)
        ws = _VA1_WS[str(dev)]
        for (B, ctx, page, kvs, seed) in VA2_CASES:
            for rep in range(2):
                out, tot, row, kt, vt = _va2_case(dev, B, ctx, page, kvs, seed)
                clean = not (int(ws[3].abs().sum()) or float(ws[0].abs().sum()) or float(ws[1].abs().sum()))
                print(f"[va2] self-test B={B} ctx={ctx} page={page} kvs={kvs:g} #{rep}: rel {tot:.2e} row {row:.2e} "
                      f"new K {kt:.1e} V {vt:.1e} ws {'clean' if clean else 'DIRTY'}", flush=True)
                if not (bool(torch.isfinite(out.float()).all()) and clean and tot < 2e-2 and row < 6e-2
                        and kt < 3e-2 and vt < 3e-2):
                    raise RuntimeError(f"self-test mismatch B={B} ctx={ctx}")
        state["va2_ok"] = True
        if VA2_PROBES:
            d = _k1_mk_case(dev, 16, 3000, 8, False, 1, 1.0, 499)
            _va2_probes(d["qkv"], d, d["config"])
            torch.cuda.synchronize()
            state["va2_probes_ok"] = not (int(ws[3].abs().sum()) or float(ws[0].abs().sum()) or float(ws[1].abs().sum()))
            print(f"[va2] probes {VA2_PROBES}: workspace {'clean' if state['va2_probes_ok'] else 'DIRTY (probes off)'}", flush=True)
            for t_ in ws:
                t_.zero_()
        print(f"[va2] LIVE: verify full B >= {VA2_MIN_B}, segk {VA2_SEGK}, occupancy {[_K1_OCC.get((8, c, str(dev))) for c in (303, 345, 347, 348, 349, 350, 351, 353, 401, 402, 403, 404)]} (variant {VA2_VAR}), "
              f"enable {VA2_ENABLE}", flush=True)
    except Exception as ex:
        print(f"[va2] DISABLED, verify full stays on PR206: {ex!r}", flush=True)


'''


def va2_patches(text):
    """On a stripped layer text that carries VA1 with its riders (k3_va_3): add VA2."""
    K1_MAIN = "template <int L, int ST>\n__global__ void __launch_bounds__(NTHR, (L == 1) ? 2 : K1_L8_MINB) k1_main(const Params p) {"
    steps = (
        ("VA2 kernel", K1_MAIN, VA2_CUDA + K1_MAIN),
        ("stages_ok", ": (ST == 3 || ST == 4); }", ": (ST == 3 || ST == 4 || (ST > 1000 && ST % 1000 == 353 && ST < 8000) || ST == 303 || ST == 313 || ST == 345 || ST == 355 || "
         "ST == 347 || ST == 357 || ST == 348 || ST == 358 || ST == 349 || ST == 359 || ST == 350 || ST == 360 || ST == 351 || ST == 361 || "
         "ST == 353 || ST == 363 || (ST >= 401 && ST <= 404)); }"),
        ("VA2 host", "int64_t k1_occupancy(int64_t L, int64_t ST) {", VA2_HOST + "\n\nint64_t k1_occupancy(int64_t L, int64_t ST) {"),
        ("occupancy", "  if (ST > 200) return (int64_t)(ST % 10 == 3 ? va_go<3>",
         "  if (ST == 401) return (int64_t)va2_go<5, 4, 2, 3, 1, VA2_DIAG, 32, 1, 1, 0, 0>(nullptr, dim3(1), 0, false);\n"
         "  if (ST == 402) return (int64_t)va2_go<5, 4, 2, 3, 1, VA2_DIAG, 32, 1, 0, 1, 0>(nullptr, dim3(1), 0, false);\n"
         "  if (ST == 403) return (int64_t)va2_go<5, 4, 2, 3, 1, VA2_DIAG, 32, 1, 0, 0, 1>(nullptr, dim3(1), 0, false);\n"
         "  if (ST == 404) return (int64_t)va2_go<5, 4, 2, 3, 1, VA2_DIAG, 32, 1, 1, 1, 1>(nullptr, dim3(1), 0, false);\n"
         "  if (ST > 300) return (int64_t)(ST == 353 ? va2_go<5, 4, 2, 3, 1, VA2_DIAG, 32, 1>(nullptr, dim3(1), 0, false) : "
         "ST == 350 ? va2_go<4, 4, 2, 1, 1, VA2_DIAG, 32, 1>(nullptr, dim3(1), 0, false) : "
         "ST == 351 ? va2_go<5, 4, 2, 1, 1, VA2_DIAG, 32, 1>(nullptr, dim3(1), 0, false) : "
         "ST % 10 == 9 ? va2_go<4, 4, 3, 2, 1>(nullptr, dim3(1), 0, false) : "
         "ST % 10 == 8 ? va2_go<4, 4, 3, 1, 1, VA2_DIAG, 48>(nullptr, dim3(1), 0, false) : ST % 10 == 7 ? "
         "va2_go<4, 4, 3, 1, 1>(nullptr, dim3(1), 0, false) : ST % 10 == 5 ? va2_go<4, 4, 2, 1>(nullptr, dim3(1), 0, false) : "
         "va2_go<3, 1, 3>(nullptr, dim3(1), 0, false));\n  if (ST > 200) return (int64_t)(ST % 10 == 3 ? va_go<3>"),
        ("launch", "  if (stages > 200) {",
         "  if (stages > 300) {\n    TORCH_CHECK(L == 8 && fold == 1 && B <= 128, \"va2: L 8, fold, B <= 128\");\n"
         "    if (stages > 1000) {\n      const int dg = (int)(stages / 1000);\n"
         "      if (dg == 1) va2_go<5, 4, 2, 3, 1, 1, 32, 1>(&p, grid, st, false);\n      else if (dg == 2) va2_go<5, 4, 2, 3, 1, 2, 32, 1>(&p, grid, st, false);\n"
         "      else if (dg == 3) va2_go<5, 4, 2, 3, 1, 3, 32, 1>(&p, grid, st, false);\n"
         "      else if (dg == 4) va2_go<5, 4, 2, 3, 1, 4, 32, 1>(&p, grid, st, false);\n"
         "      else if (dg == 6) va2_go<5, 4, 2, 3, 1, 6, 32, 1>(&p, grid, st, false);\n"
         "      else if (dg == 7) va2_go<5, 4, 2, 3, 1, 7, 32, 1>(&p, grid, st, false);\n"
         "      else va2_go<5, 4, 2, 3, 1, 5, 32, 1>(&p, grid, st, false);\n    }\n"
         "    else if (stages == 401) va2_go<5, 4, 2, 3, 1, VA2_DIAG, 32, 1, 1, 0, 0>(&p, grid, st, false);\n"
         "    else if (stages == 402) va2_go<5, 4, 2, 3, 1, VA2_DIAG, 32, 1, 0, 1, 0>(&p, grid, st, false);\n"
         "    else if (stages == 403) va2_go<5, 4, 2, 3, 1, VA2_DIAG, 32, 1, 0, 0, 1>(&p, grid, st, false);\n"
         "    else if (stages == 404) va2_go<5, 4, 2, 3, 1, VA2_DIAG, 32, 1, 1, 1, 1>(&p, grid, st, false);\n"
         "    else if (stages == 353 || stages == 363) va2_go<5, 4, 2, 3, 1, VA2_DIAG, 32, 1>(&p, grid, st, stages > 355);\n"
         "    else if (stages == 350 || stages == 360) va2_go<4, 4, 2, 1, 1, VA2_DIAG, 32, 1>(&p, grid, st, stages > 355);\n"
         "    else if (stages == 351 || stages == 361) va2_go<5, 4, 2, 1, 1, VA2_DIAG, 32, 1>(&p, grid, st, stages > 355);\n"
         "    else if (stages % 10 == 9) va2_go<4, 4, 3, 2, 1>(&p, grid, st, stages > 350);\n"
         "    else if (stages % 10 == 8) va2_go<4, 4, 3, 1, 1, VA2_DIAG, 48>(&p, grid, st, stages > 350);\n"
         "    else if (stages % 10 == 7) va2_go<4, 4, 3, 1, 1>(&p, grid, st, stages > 350);\n"
         "    else if (stages % 10 == 5) va2_go<4, 4, 2, 1>(&p, grid, st, stages > 350);\n"
         "    else va2_go<3, 1, 3>(&p, grid, st, stages > 310);\n    C10_CUDA_KERNEL_LAUNCH_CHECK();\n    return;\n  }\n  if (stages > 200) {"),
        ("rows", "VA1_ROWS = 16 * 1024\n", "VA1_ROWS = 34 * 1024\n"),
        ("python", "def _va1_prepare(state):\n", VA2_PY + "def _va1_prepare(state):\n"),
        ("prepare", "    _va1_prepare(state)\n", "    _va1_prepare(state)\n    _va2_prepare(state)\n"),
        ("kernel_fn", "    if o is None and not _va1_b1(cfg, state) and _v18_fa4_wanted(qkv, data, cfg, state):\n",
         "    if o is None:\n        o = _va2_try(qkv, data, cfg, state)\n"
         "    if o is None and not _va1_b1(cfg, state) and _v18_fa4_wanted(qkv, data, cfg, state):\n"),
    )
    for what, old, new in steps:
        if text.count(old) != 1:
            raise ValueError(f"va2 anchor {what!r} found {text.count(old)} times")
        text = text.replace(old, new, 1)
    return text


P206_NS_ANCHOR = '    ns, _ = p206_plan(batch, hkv, hq // hkv, length, data["page_table"].shape[1], d, _p206_num_sms(device))\n'


def p206_patches(text):
    """PR206 tuning hooks (A/B arms via module globals): P206_NS = {batch: num_splits} overrides p206_plan's split count
    in p206_run (the route test still uses the plan)."""
    if text.count(P206_NS_ANCHOR) != 1:
        raise ValueError(f"p206 anchor found {text.count(P206_NS_ANCHOR)} times")
    text = text.replace(P206_NS_ANCHOR, P206_NS_ANCHOR + "    ns = int(P206_NS.get(batch, ns))\n")
    a = "def p206_run(data, pdl=True):\n"
    if text.count(a) != 1:
        raise ValueError("p206_run anchor")
    return text.replace(a, "P206_NS = {}   # PR206 split count per batch (tools/va2_src.py p206_patches; {} = p206_plan)\n\n\n" + a)

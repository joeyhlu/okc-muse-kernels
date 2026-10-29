"""fasttail_patches.py (R-ft-0 FAST-TAIL, notes/memory/gpu_requests.md): k3_gc_20 -> kernels/lab/k3_ft_1.py.

  python tools/fasttail_patches.py [--base=kernels/muse_decode_layer/k3_gc_20.py] [--out=kernels/lab/k3_ft_1.py]

Changes (V8 tail only; GD pooler / o / unpooler untouched):
  1. tail_kernel / tail_kernel2: every predecessor-data load (y, partials la / lb, lstart) is __ldcg (coherent LDG, never
     .nc), so ptxas cannot hoist it above griddepcontrol.wait (ACQBULK). Same arithmetic.
  2. tail_kernel3<Y32, NSMAX, RW>: tail_kernel2 with the routing chain spread out. RW routing warps after the 16 norm warps;
     thread r owns experts r*EPT .. +EPT-1 (EPT = 8 / RW): partial loads issued up front (coherent), the same fixed-order
     FP32 sum, sigmoid and bias as tail_kernel2, order key + s to smem; warp 16 (after a named barrier) presorts its lane's
     8 (key, j) (19-CAS, ties -> lower j = the scan's choice), runs 8 rounds key -> redux -> ballot -> shift, and lanes 0..7
     write one top-k slot each (the division / stores no longer serial on lane 0). Bitwise = tail_kernel2.
  3. Host: pdl bit 4 -> tail_kernel3 RW 2 (576 threads), bit 8 -> RW 8 (768 threads). Python: FT_TAIL3 {T bucket: 0/4/8},
     prepare check _ft_prepare (tail3 RW 2 and RW 8 vs tail_kernel2 bitwise on 7 shapes x 4 input variants x 4 value cases
     incl. exact ties) -> FT_OK; _v8_tail(..., ftf=) override for tests / benches.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)


def _rep(src, old, new, what, n=1):
    c = src.count(old)
    if c != n:
        raise SystemExit(f"[fasttail_patches] {what}: expected {n} match(es), got {c}")
    return src.replace(old, new)


HARDEN = [
    ("qa[q][0] = pq[0];\n        qa[q][1] = pq[1];", "qa[q][0] = __ldcg(pq);\n        qa[q][1] = __ldcg(pq + 1);", "pq"),
    ("qb0 = pb[0];\n      qb1 = pb[1];", "qb0 = __ldcg(pb);\n      qb1 = __ldcg(pb + 1);", "pb"),
    ("const float4 a = p[0], c = p[1];", "const float4 a = __ldcg(p), c = __ldcg(p + 1);", "y32"),
    ("const uint4 q = *(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(y) + (long)t * y_ld) + tx);",
     "const uint4 q = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(y) + (long)t * y_ld) + tx);",
     "y16"),
    ("const int ls = lstart[0];", "const int ls = __ldcg(lstart);", "lstart"),
]

TAIL3 = r"""// FAST-TAIL (R-ft-0): tail_kernel2 with the routing chain spread out (see tools/fasttail_patches.py). Bitwise = tail_kernel2.
template <int EPT, bool CG>
__device__ __forceinline__ void ft_ld(const float* p, float* o) {
  if constexpr (EPT == 4) {
    const float4 v = CG ? __ldcg(reinterpret_cast<const float4*>(p)) : *reinterpret_cast<const float4*>(p);
    o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
  } else if constexpr (EPT == 2) {
    const float2 v = CG ? __ldcg(reinterpret_cast<const float2*>(p)) : *reinterpret_cast<const float2*>(p);
    o[0] = v.x; o[1] = v.y;
  } else {
    o[0] = CG ? __ldcg(p) : *p;
  }
}
__device__ __forceinline__ bool ft_better(unsigned k1, int j1, unsigned k2, int j2) { return k1 > k2 || (k1 == k2 && j1 < j2); }

template <bool Y32, int NSMAX, int RW>
__global__ void __launch_bounds__(NT + 32 * RW) tail_kernel3(
    const void* __restrict__ y, long y_ld, const float* __restrict__ la, const float* __restrict__ lb, long l_ld,
    int ns, long sstride, const __nv_bfloat16* __restrict__ nw, const float* __restrict__ bias, const int* __restrict__ lstart,
    __nv_bfloat16* __restrict__ xn, float* __restrict__ tw, int* __restrict__ tid, int* __restrict__ lid,
    float eps, float mult, float rscale, float deps) {
  constexpr int EPT = NE / (32 * RW);
  const int t = blockIdx.x, tx = threadIdx.x, lane = tx & 31, warp = tx >> 5;
  __shared__ float red[NT / 32];
  __shared__ __align__(16) unsigned skey[NE];
  __shared__ __align__(16) float ssg[NE];
  if (warp >= NT / 32) {
    const int e0 = (tx - NT) * EPT;
    float bs[EPT];
    ft_ld<EPT, false>(bias + e0, bs);   // weights (prepare-time): pre-wait is fine
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    float qa[NSMAX][EPT], qb[EPT];
#pragma unroll
    for (int q = 0; q < NSMAX; ++q) {
      if (q < ns) {
        ft_ld<EPT, true>(la + q * sstride + (long)t * l_ld + e0, qa[q]);
      } else {
#pragma unroll
        for (int i = 0; i < EPT; ++i) qa[q][i] = 0.f;
      }
    }
    if (lb != nullptr) {
      ft_ld<EPT, true>(lb + (long)t * l_ld + e0, qb);
    } else {
#pragma unroll
      for (int i = 0; i < EPT; ++i) qb[i] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      float a = qa[0][i];
#pragma unroll
      for (int q = 1; q < NSMAX; ++q) a += qa[q][i];   // FP32 split-K partials summed in a fixed order (zeros past ns)
      a += qb[i];
      const float s = 1.f / (1.f + expf(-a * mult));
      const unsigned u = __float_as_uint((s + bs[i]) + 0.f);   // = tail_kernel2's key of b = s + bias
      skey[e0 + i] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
      ssg[e0 + i] = s;
    }
    if constexpr (RW > 1) {
      if (warp != NT / 32) {
        asm volatile("bar.arrive 2, %0;\n" ::"r"(32 * RW) : "memory");
        return;
      }
      asm volatile("bar.sync 2, %0;\n" ::"r"(32 * RW) : "memory");
    } else {
      __syncwarp();
    }
    const int ls = __ldcg(lstart);
    const uint4 k0 = *reinterpret_cast<const uint4*>(skey + lane * 8), k1 = *reinterpret_cast<const uint4*>(skey + lane * 8 + 4);
    const float4 s0 = *reinterpret_cast<const float4*>(ssg + lane * 8), s1 = *reinterpret_cast<const float4*>(ssg + lane * 8 + 4);
    unsigned hk[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
    float hs[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
    int hj[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    constexpr int NWK[19][2] = {{0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}, {0, 1}, {2, 3},
                                {4, 5}, {6, 7}, {2, 4}, {3, 5}, {1, 4}, {3, 6}, {1, 2}, {3, 4}, {5, 6}};
#pragma unroll
    for (int x = 0; x < 19; ++x) {   // descending by (key, then lower j): the head = tail_kernel2's local scan pick
      const int i = NWK[x][0], k = NWK[x][1];
      const bool sw = ft_better(hk[k], hj[k], hk[i], hj[i]);
      const unsigned tk = hk[i];
      const float ts = hs[i];
      const int tj = hj[i];
      hk[i] = sw ? hk[k] : tk; hs[i] = sw ? hs[k] : ts; hj[i] = sw ? hj[k] : tj;
      hk[k] = sw ? tk : hk[k]; hs[k] = sw ? ts : hs[k]; hj[k] = sw ? tj : hj[k];
    }
    int sel_id[TK];
    float sel_s[TK];
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      const unsigned mx = __reduce_max_sync(0xffffffffu, hk[0]);
      const int owner = __ffs(__ballot_sync(0xffffffffu, hk[0] == mx)) - 1;
      sel_id[k] = __shfl_sync(0xffffffffu, lane * 8 + hj[0], owner);
      sel_s[k] = __shfl_sync(0xffffffffu, hs[0], owner);
      const bool me = lane == owner;
#pragma unroll
      for (int j = 0; j < 7; ++j) {
        hk[j] = me ? hk[j + 1] : hk[j]; hs[j] = me ? hs[j + 1] : hs[j]; hj[j] = me ? hj[j + 1] : hj[j];
      }
      hk[7] = me ? 0u : hk[7];
    }
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < TK; ++i) sum += sel_s[i];
    const float inv = rscale / (sum + deps);
    if (lane < TK) {
      int mid = sel_id[0];
      float msv = sel_s[0];
#pragma unroll
      for (int k = 1; k < TK; ++k)
        if (lane == k) { mid = sel_id[k]; msv = sel_s[k]; }
      tw[t * TK + lane] = msv * inv;
      tid[t * TK + lane] = mid;
      const int rel = ((mid - ls) % NE + NE) % NE;
      lid[t * TK + lane] = rel < 32 ? rel : -1;
    }
    return;
  }
  const uint4 wq = *(reinterpret_cast<const uint4*>(nw) + tx);
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  float v[8];
  if (Y32) {
    const float4* p = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(y) + (long)t * y_ld) + tx * 2;
    const float4 a = __ldcg(p), c = __ldcg(p + 1);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = c.x; v[5] = c.y; v[6] = c.z; v[7] = c.w;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(__float2bfloat16(v[i]));
  } else {
    const uint4 q = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(y) + (long)t * y_ld) + tx);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(h[i]);
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) red[warp] = ss;
  asm volatile("bar.sync 1, %0;\n" ::"r"(NT) : "memory");   // the 16 norm warps only
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; ++i) tot += red[i];
  const float rstd = rsqrtf(tot / (float)D + eps);
  const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wq);
  __align__(16) __nv_bfloat16 o8[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16((v[i] * rstd) * __bfloat162float(w[i]));
  *(reinterpret_cast<uint4*>(xn + (long)t * D) + tx) = *reinterpret_cast<const uint4*>(o8);
}

"""

FT_PY = r'''
FT_TAIL3 = {1: 4, 8: 4, 16: 4, 32: 4, 128: 4, 256: 4}   # R-ft-0: T bucket -> 0 tail_kernel2 / 4 tail3 RW 2 / 8 tail3 RW 8
FT_OK = False


def _ft_flag(tokens):
    if not FT_OK:
        return 0
    for b in sorted(FT_TAIL3):
        if tokens <= b:
            return FT_TAIL3[b]
    return 0


def _ft_prepare(v8, dev):
    """tail_kernel3 (RW 2 and RW 8) vs tail_kernel2 bitwise (xn / tw / tid / lid) -> FT_OK."""
    global FT_OK
    FT_OK = False
    if not (TL1_TAIL2 and any(FT_TAIL3.values())):
        return
    try:
        gen = torch.Generator(device=dev).manual_seed(4545)
        same, n = True, 0
        for T, ls in ((1, 0), (8, 64), (16, 224), (32, 32), (33, 96), (128, 96), (256, 160)):
            d = {"local_start": torch.tensor([ls], dtype=torch.int32, device=dev)}
            for case in range(4):
                bias = v8["bias"]
                lg = torch.randn((T, N_EXPERTS), generator=gen, device=dev) * 3.0
                part = torch.randn((8, T, N_EXPERTS), generator=gen, device=dev) * 1.1
                if case == 1:   # exact ties in s and b (bias 0)
                    lg, bias = torch.round(lg), torch.zeros_like(bias)
                elif case == 2:   # ties through duplicated biases
                    lg, bias = torch.round(lg * 2) / 2, torch.round(bias * 4) / 4
                elif case == 3:   # partial sums built to collide (equal slices)
                    lg = torch.round(lg)
                if case:
                    part = (lg.unsqueeze(0).repeat(8, 1, 1) * 0.125).contiguous()
                pool = torch.randn((T, D_EXPERT), generator=gen, device=dev).to(torch.bfloat16)
                y32 = torch.randn((T, D_EXPERT + 2 * N_EXPERTS), generator=gen, device=dev)
                y32[:, D_EXPERT:D_EXPERT + N_EXPERTS] = lg
                y32[:, D_EXPERT + N_EXPERTS:] = 0.0 if case else 0.25 * lg
                vv = {"v8": dict(v8, bias=bias.contiguous())}
                for yy, ll in ((pool, part), (pool, lg.contiguous()), (y32, None), (y32, part)):
                    ref = [a.clone() for a in _v8_tail(yy, ll, vv, d, ftf=0)]
                    for f in (4, 8):
                        got = _v8_tail(yy, ll, vv, d, ftf=f)
                        torch.cuda.synchronize()
                        same = same and all(bool(torch.equal(a, b)) for a, b in zip(ref, got))
                        n += 1
        FT_OK = same
        print(f"[ft] tail_kernel3 == tail_kernel2 bitwise on {n} runs (7 shapes x 4 cases x 4 inputs x RW 2/8): {same}"
              f" -> {'FT_TAIL3 ' + str(FT_TAIL3) if same else 'tail_kernel2'}", flush=True)
    except Exception as ex:
        FT_OK = False
        print(f"[ft] tail_kernel3 check raised -> tail_kernel2: {ex!r}", flush=True)
'''


def main():
    base = ROOT / ARGS.get("base", "kernels/muse_decode_layer/k3_gc_20.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_ft_1.py")
    src = base.read_text(encoding="utf-8")
    a = src.index('V8_TAIL_CUDA_SRC = r"""')
    b = src.index('"""', a + 30)
    cu = src[a:b]
    for old, new, what in HARDEN:
        cu = _rep(cu, old, new, what, 2)
    cu = _rep(cu, "template <int MT>\n__global__ void __launch_bounds__(256) router32_kernel",
              TAIL3 + "template <int MT>\n__global__ void __launch_bounds__(256) router32_kernel", "tail3 insert")
    for yy in ("true", "false"):
        for nn in ("8", "1"):
            cu = _rep(cu, f"((pdl & 2) ? v8::tail_kernel2<{yy}, {nn}> : v8::tail_kernel<{yy}, {nn}>)", f"FT_K({yy}, {nn})",
                      f"launch {yy} {nn}")
    cu = _rep(cu, "lc.blockDim = dim3((pdl & 2) ? v8::NT + 32 : v8::NT);   // TL1",
              "lc.blockDim = dim3((pdl & 8) ? v8::NT + 256 : (pdl & 4) ? v8::NT + 64 : (pdl & 2) ? v8::NT + 32 : v8::NT);"
              "   // TL1 / FT", "blockDim")
    cu = _rep(cu, "  if (T == 0) return;\n",
              "  if (T == 0) return;\n#define FT_K(Y, N) ((pdl & 8) ? v8::tail_kernel3<Y, N, 8> : (pdl & 4) ? v8::tail_kernel3<Y, N, 2>"
              " : (pdl & 2) ? v8::tail_kernel2<Y, N> : v8::tail_kernel<Y, N>)\n", "FT_K macro")
    src = src[:a] + cu + src[b:]
    src = _rep(src, "def _v8_tail(y, logits, state, data):\n", "def _v8_tail(y, logits, state, data, ftf=None):\n", "_v8_tail sig")
    src = _rep(src, "(1 if V9_TAIL_PDL else 0) | (2 if TL1_TAIL2 else 0))",
               "(1 if V9_TAIL_PDL else 0) | (2 if TL1_TAIL2 else 0)\n                      | (_ft_flag(tokens) if ftf is None else ftf))",
               "_v8_tail flag")
    src = _rep(src, '    print(f"[v8] tail ready in', '    _ft_prepare(v8, dev)\n    print(f"[v8] tail ready in', "prepare hook")
    src = _rep(src, "\nTL1_TAIL2 = True\n", "\nTL1_TAIL2 = True\n" + FT_PY, "FT globals")
    out.write_text(src, encoding="utf-8", newline="\n")
    print(f"[fasttail_patches] {base.name} -> {out.relative_to(ROOT)} ({len(src.encode())} bytes)")


if __name__ == "__main__":
    main()

"""gen_k3_gt.py: k3_gt_<n> = a GC layer file (default k3_gc_2) + GT, the routing tail FUSED into the GC cat launch at T = 9..32.

Today (notes/study/lb_gc1.md): GC cat (18.6 us) -> v8::tail_kernel2 4.0 us on the critical path -> ~0.5 us idle -> vm_main (T = 32,
no PDL); at T = 16 VMS is PDL-resident but waits for the tail (4.0 us excl). GT removes the tail launch:
* every GC CTA, after storing its pooler tile and router K-slice partial: __threadfence, one atomicAdd ticket on a grid counter.
  The LAST T arrivers (tickets >= 256 - T; the grid is one wave, so every CTA is resident and the wait is bounded by the slowest
  CTA's epilogue) each spin (ld.acquire, __nanosleep, bounded by GD_CAT_TAIL_SPIN iterations; on timeout a flag is set and the
  CTA proceeds, never traps) until all 256 CTAs arrived, then do ONE token's tail: warp 0 = the 8 partial slices summed in the
  same fixed order as tail_kernel2 (q = 0..7, then + 0), sigmoid + bias, the same lane-parallel top-8 with the same key / ballot
  tie rule, tw / tid / lid; all 8 warps = the pre-expert RMS norm with the tail's exact 16 x 32 x 8 element layout (each warp
  plays two of the tail's 16 norm warps, same per-lane 8-element sums, same xor tree, same 16-term total), xn written bf16.
  The last tail worker resets both counters, so the kernel is replay-safe inside the CUDA graph.
* routing outputs are bitwise those of the shipped tail (asserted in prepare on T = 16 / 9 and 32 / 17 inputs against
  _v8_tail on the same y / partials; xn bitwise expected, <= 1 bf16 ulp tolerated with a printed note); any failure -> that
  bucket keeps GC + the separate tail (loud FALLBACK). GD_CAT_TAIL is read at call time (A/B arms).
* the GD module gets a TAIL template parameter: two more cat instances (TAIL 1), the TAIL 0 ones stay and remain the fallback.

Output goes through gen_k3_m32.finish (strip, compile, ASCII, nonce, checker scan, < 1 MB). Refuses to overwrite.

    python -I tools/gen_k3_gt.py [--base=k3_gc_2] [--n=1]
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32 as GM  # noqa: E402
sys.argv = _argv

N = ARGS.get("n", "1")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_2')}.py"
OUT = ROOT / f"kernels/muse_decode_layer/k3_gt_{N}.py"
LAB = ROOT / f"kernels/lab/k3_gt_{N}_ab.py"
TAG = f"k3-gt-{N}"
NL = chr(10)

GC_TAIL_DEVICE = r'''
// GT: the routing tail of one token, run by a whole 256-thread GC CTA; arithmetic and element layout = v8::tail_kernel2
__device__ __forceinline__ int gt_ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float4 gt_ldcg4(const float4* p) { return __ldcg(p); }
__device__ __forceinline__ uint4 gt_ldcg4u(const uint4* p) { return __ldcg(p); }

__device__ __noinline__ void gt_tail_run(const Args& A, const int tok, float* red16) {
  constexpr int TK = 8, NE = 256, D = 4096;
  const int tx = (int)threadIdx.x, lane = tx & 31, warp = tx >> 5;
  if (warp == 0) {   // the tail's logits warp
    const float4* pbias = reinterpret_cast<const float4*>(A.bias) + lane * 2;
    const float4 c0 = pbias[0], c1 = pbias[1];
    float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f), a1 = a0;
#pragma unroll
    for (int q = 0; q < 8; ++q) {   // FP32 split-K partials summed in the tail's fixed order (slice 0 first)
      const float4* pq = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(A.out2) + (size_t)q * A.sst2
                                                         + (size_t)tok * A.ldo2) + lane * 2;
      const float4 x0 = gt_ldcg4(pq), x1 = gt_ldcg4(pq + 1);
      if (q == 0) { a0 = x0; a1 = x1; }
      else {
        a0.x += x0.x; a0.y += x0.y; a0.z += x0.z; a0.w += x0.w;
        a1.x += x1.x; a1.y += x1.y; a1.z += x1.z; a1.w += x1.w;
      }
    }
    a0.x += 0.f; a0.y += 0.f; a0.z += 0.f; a0.w += 0.f; a1.x += 0.f; a1.y += 0.f; a1.z += 0.f; a1.w += 0.f;   // the tail's lb = 0
    const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    float s[8], b[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j] = 1.f / (1.f + expf(-lg[j] * A.mult));
      b[j] = s[j] + bs[j];
    }
    int sel_id[TK];
    float sel_s[TK];
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      float bv = b[0], sv = s[0];
      int bj = 0;
#pragma unroll
      for (int j = 1; j < 8; ++j)
        if (b[j] > bv) { bv = b[j]; sv = s[j]; bj = j; }
      const unsigned u = __float_as_uint(bv + 0.f);
      const unsigned key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
      const unsigned mx = __reduce_max_sync(0xffffffffu, key);
      const int owner = __ffs(__ballot_sync(0xffffffffu, key == mx)) - 1;
      sel_id[k] = __shfl_sync(0xffffffffu, lane * 8 + bj, owner);
      sel_s[k] = __shfl_sync(0xffffffffu, sv, owner);
      if (lane == owner) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (j == bj) b[j] = -INFINITY;
      }
    }
    if (lane == 0) {
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < TK; ++i) sum += sel_s[i];
      const float inv = A.rscale / (sum + A.deps);
      const int ls = A.lstart[0];
#pragma unroll
      for (int k = 0; k < TK; ++k) {
        A.tw[tok * TK + k] = sel_s[k] * inv;
        A.tid[tok * TK + k] = sel_id[k];
        const int rel = ((sel_id[k] - ls) % NE + NE) % NE;
        A.lid[tok * TK + k] = rel < 32 ? rel : -1;
      }
    }
  }
  // the tail's 16 norm warps (512 threads, 8 elements each): warp w plays tail warps w and w + 8
  const uint4* yrow = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(A.out) + (size_t)tok * A.ldo);
  float v[2][8];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int vw = warp + 8 * h;
    const uint4 q = gt_ldcg4u(yrow + vw * 32 + lane);
    const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[h][i] = __bfloat162float(hb[i]);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) ss += v[h][i] * v[h][i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) red16[vw] = ss;
  }
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) tot += red16[i];
  const float rstd = rsqrtf(tot / (float)D + A.eps);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int vw = warp + 8 * h;
    const uint4 wq = *(reinterpret_cast<const uint4*>(A.nw) + vw * 32 + lane);
    const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wq);
    __align__(16) __nv_bfloat16 o8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16((v[h][i] * rstd) * __bfloat162float(w[i]));
    *(reinterpret_cast<uint4*>(A.xn + (size_t)tok * D) + vw * 32 + lane) = *reinterpret_cast<const uint4*>(o8);
  }
}

'''

TAIL_BLOCK = r'''    if constexpr (TAIL > 0) {   // GT: fused routing tail (notes/study/gc_dense.md s8)
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) gt_tk = atomicAdd(A.ctr, 1);
      __syncthreads();
      const int gt_nb = (int)gridDim.x, gt_T = A.M, gt_tok = gt_tk - (gt_nb - gt_T);
      if (gt_tok >= 0) {
        if (threadIdx.x == 0) {
          int it = 0;
          while (gt_ld_acquire(A.ctr) < gt_nb) {
            if (++it > A.spin) { atomicExch(A.ctr + 2, 1); break; }   // bounded: flag, never trap
            __nanosleep(64);
          }
        }
        __syncthreads();
        gt_tail_run(A, gt_tok, gt_red);
        __syncthreads();
        if (threadIdx.x == 0 && atomicAdd(A.ctr + 1, 1) == gt_T - 1) { atomicExch(A.ctr + 1, 0); atomicExch(A.ctr, 0); }
      }
    }
'''

CUDA_HOST_ADD = r'''
int gdt_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, void* out2, void* xn, void* tw, void* tid,
                void* lid, const void* nw, const void* bias, const void* lstart, void* ctr, int M, int K, int ldo, int ldo2,
                uint32_t tb, uint32_t tbr, int nbx, int pdl, float eps, float mult, float rscale, float deps, int spin,
                cudaStream_t st) {
  Args A;
  A.w = reinterpret_cast<const uint4*>(w);
  A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.x = reinterpret_cast<const uint16_t*>(x);
  A.out = out;
  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;
  A.trig = (pdl >> 1) & 1;
  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;
  A.out2 = out2; A.ldo2 = ldo2; A.sst2 = (long long)M * ldo2; A.tbr = tbr;
  A.bias = reinterpret_cast<const float*>(bias); A.nw = reinterpret_cast<const __nv_bfloat16*>(nw);
  A.lstart = reinterpret_cast<const int*>(lstart); A.xn = reinterpret_cast<__nv_bfloat16*>(xn);
  A.tw = reinterpret_cast<float*>(tw); A.tid = reinterpret_cast<int*>(tid); A.lid = reinterpret_cast<int*>(lid);
  A.ctr = reinterpret_cast<int*>(ctr); A.eps = eps; A.mult = mult; A.rscale = rscale; A.deps = deps; A.spin = spin;
  const int nz = 1;
  switch (inst) {
@INST@
    default: return -1;
  }
}

'''

CPP_ADD = r'''int gdt_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, void* out2, void* xn, void* tw, void* tid,
                void* lid, const void* nw, const void* bias, const void* lstart, void* ctr, int M, int K, int ldo, int ldo2,
                uint32_t tb, uint32_t tbr, int nbx, int pdl, float eps, float mult, float rscale, float deps, int spin,
                cudaStream_t st);
void gdt_run(int64_t inst, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, torch::Tensor out2,
             torch::Tensor xn, torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, torch::Tensor nw, torch::Tensor bias,
             torch::Tensor lstart, torch::Tensor ctr, int64_t M, int64_t K, int64_t ldo, int64_t ldo2, int64_t tb, int64_t tbr,
             int64_t nbx, int64_t pdl, double eps, double mult, double rscale, double deps, int64_t spin) {
  TORCH_CHECK(out2.scalar_type() == at::kFloat && out2.is_contiguous() && out2.dim() == 3 && out2.size(0) == 8 &&
              out2.size(1) == M && out2.size(2) == ldo2, "gdt_run: router partials must be a contiguous [8, M, ldo2] f32 tensor");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == M && out.size(1) == ldo && ldo == 4096,
              "gdt_run: pooler output [M, 4096] bf16");
  TORCH_CHECK(xn.scalar_type() == at::kBFloat16 && xn.is_contiguous() && xn.size(0) == M && xn.size(1) == 4096, "gdt_run: xn");
  TORCH_CHECK(tw.scalar_type() == at::kFloat && tid.scalar_type() == at::kInt && lid.scalar_type() == at::kInt &&
              tw.is_contiguous() && tid.is_contiguous() && lid.is_contiguous() && tw.numel() == M * 8 && tid.numel() == M * 8 &&
              lid.numel() == M * 8, "gdt_run: tw / tid / lid [M, 8]");
  TORCH_CHECK(nw.scalar_type() == at::kBFloat16 && nw.numel() == 4096 && bias.scalar_type() == at::kFloat && bias.numel() == 256 &&
              lstart.scalar_type() == at::kInt && ctr.scalar_type() == at::kInt && ctr.numel() >= 4 && ldo2 == 256 && M <= nbx,
              "gdt_run: nw / bias / lstart / ctr");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(xn.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(nw.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(out2.data_ptr()) % 16 == 0, "gdt_run: alignment");
  const int e = gdt_run_raw(inst, w.data_ptr(), esc.data_ptr(), x.data_ptr(), out.data_ptr(), out2.data_ptr(), xn.data_ptr(),
                            tw.data_ptr(), tid.data_ptr(), lid.data_ptr(), nw.data_ptr(), bias.data_ptr(), lstart.data_ptr(),
                            ctr.data_ptr(), (int)M, (int)K, (int)ldo, (int)ldo2, (uint32_t)tb, (uint32_t)tbr, (int)nbx, (int)pdl,
                            (float)eps, (float)mult, (float)rscale, (float)deps, (int)spin, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gdt_run: inst ", inst, " CUDA error ", e, " ", cudaGetErrorString((cudaError_t)(e < 0 ? 1 : e)));
}
'''

CUDA_PATCHES = [
    ("uint32_t tbr = 0;",
     "uint32_t tbr = 0;\n"
     "  const float* bias = nullptr; const __nv_bfloat16* nw = nullptr; const int* lstart = nullptr; __nv_bfloat16* xn = nullptr;\n"
     "  float* tw = nullptr; int* tid = nullptr; int* lid = nullptr; int* ctr = nullptr;\n"
     "  float eps = 0.f, mult = 0.f, rscale = 0.f, deps = 0.f; int spin = 0;",
     "Args: GT fields"),
    ("#define GC_XP(S_)", GC_TAIL_DEVICE + "#define GC_XP(S_)", "gt_tail_run device function"),
    ("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0>\n"
     "__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(const __grid_constant__ Args A) {\n",
     "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0, int TAIL = 0>\n"
     "__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(const __grid_constant__ Args A) {\n"
     "  static_assert(TAIL == 0 || (CAT > 0 && WR * WK * 32 == 256), \"GT: the fused tail needs a cat route with 256 threads\");\n"
     "  __shared__ float gt_red[TAIL ? 16 : 1];\n"
     "  __shared__ int gt_tk;\n",
     "kernel template TAIL + smem"),
    ("          reinterpret_cast<float*>(A.out2)[(size_t)gc_ks * A.sst2 + (size_t)tok * A.ldo2 + gc_t * 8 + gg] = v;\n"
     "      }\n"
     "    }\n",
     "          reinterpret_cast<float*>(A.out2)[(size_t)gc_ks * A.sst2 + (size_t)tok * A.ldo2 + gc_t * 8 + gg] = v;\n"
     "      }\n"
     "    }\n" + TAIL_BLOCK,
     "fused tail after the router partial store"),
]
CUDA_REPLACE_ALL = [
    ("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0>\n",
     "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0, int TAIL = 0>\n", 2,
     "launch / info templates TAIL"),
    ("gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>", "gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT, TAIL>", 2,
     "launch / info instantiations TAIL"),
]

PY_GLOBALS = r'''GD_CAT_TAIL = True
GD_CAT_TAIL_SPIN = 200000   # bounded spin iterations (x __nanosleep 64 ns + loop >= ~13 ms) before the timeout flag
GD_CAT_TAIL_XN_ULP = 1      # xn is expected bitwise; this many bf16 ulps are tolerated with a printed note (routing stays bitwise)
'''

PY_FUNCS = r'''

def _gd_cat_tail_run(r, state, x, y, part, lstart, T, pdl):
    v8 = state["v8"]
    dev = x.device
    xn = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
    tw = torch.empty((T, TOP_K), dtype=torch.float32, device=dev)
    tid = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    lid = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    r["mod"].gdt_run(r["inst_t"], r["w"], r["esc"], x, y, part, xn, tw, tid, lid, v8["nw"], v8["bias"], lstart, state["gd_cat_ctr"],
                     T, HIDDEN, D_EXPERT, N_EXPERTS, r["tb"], r["tbr"], r["nbx"], pdl, RMS_EPS, 0.1767766952966369, 1.0, 1e-15,
                     GD_CAT_TAIL_SPIN)
    return xn, tw, tid, lid


def _gd_cat_tail_selftest(r, state, b):
    dev = state["v8"]["wcat"].device
    ctr = state["gd_cat_ctr"]
    parts = []
    for T in GD_CAT_SELF_TEST_T[b]:
        for ls in (64, 224):
            gen = torch.Generator(device=dev).manual_seed(4242 + 7 * T + ls)
            x = torch.randn((T, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
            lstart = torch.tensor([ls], dtype=torch.int32, device=dev)
            y = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
            part = torch.empty((8, T, N_EXPERTS), dtype=torch.float32, device=dev)
            _gd_cat_run(r, x, y, part, T, GD_CAT_PDL)
            xn0, tw0, tid0, lid0 = _v8_tail(y, part, state, {"local_start": lstart})
            outs = []
            for pdl in GD_SELF_TEST_PDL:
                outs.append(_gd_cat_tail_run(r, state, x, y, part, lstart, T, pdl))
            torch.cuda.synchronize()
            c = ctr.tolist()
            if c[0] != 0 or c[1] != 0 or c[2] != 0:
                return False, f"T={T} ls={ls}: counters {c} (not reset or timeout)"
            for xn, tw, tid, lid in outs:
                if not (torch.equal(tid, tid0) and torch.equal(lid, lid0) and torch.equal(tw, tw0)):
                    nd = int((tid != tid0).sum()) + int((lid != lid0).sum())
                    return False, f"T={T} ls={ls}: routing differs from the tail (tid/lid mismatches {nd}, tw equal {bool(torch.equal(tw, tw0))})"
                if not torch.equal(xn, xn0):
                    ulp = int((xn.view(torch.int16).to(torch.int32) - xn0.view(torch.int16).to(torch.int32)).abs().max())
                    if ulp > GD_CAT_TAIL_XN_ULP or not bool(torch.isfinite(xn.float()).all()):
                        return False, f"T={T} ls={ls}: xn differs from the tail by {ulp} bf16 ulp"
                    parts.append(f"T={T}ls{ls}:xn{ulp}ulp")
                    continue
                parts.append(f"T={T}ls{ls}:bitwise")
    return True, " ".join(parts)
'''

PY_PATCHES = [
    ("\n_GD_CAT_SEEN = set()\n", "\n" + PY_GLOBALS + "_GD_CAT_SEEN = set()\n", "GT globals"),
    ("def _gd_cat_keys():\n"
     "    keys = []\n"
     "    for b, rt in sorted(GD_CAT_ROUTE.items()):\n"
     "        key = _gd_cat_key(rt) if rt else None\n"
     "        if key is not None and key not in keys:\n"
     "            keys.append(key)\n"
     "    return keys\n",
     "def _gd_cat_keys():\n"
     "    keys = []\n"
     "    for b, rt in sorted(GD_CAT_ROUTE.items()):\n"
     "        key = _gd_cat_key(rt) if rt else None\n"
     "        if key is not None and key not in keys:\n"
     "            keys.append(key)\n"
     "        if key is not None and key + (1,) not in keys and GD_CAT_TAIL is not None:   # GT: the fused-tail twin\n"
     "            keys.append(key + (1,))\n"
     "    return keys\n",
     "_gd_cat_keys adds the TAIL instances"),
    ("    packs, routes, live = {}, {}, []\n    for b, rt in sorted(GD_CAT_ROUTE.items()):\n",
     "    packs, routes, live = {}, {}, []\n"
     "    state[\"gd_cat_ctr\"] = torch.zeros((4,), dtype=torch.int32, device=dev)   # GT: arrivals, done, timeout flag\n"
     "    for b, rt in sorted(GD_CAT_ROUTE.items()):\n",
     "GT counters in _gd_cat_prepare"),
    ("            else:\n"
     "                routes[b] = r\n"
     "                live.append(f\"cat/{b} {desc}\")\n",
     "            else:\n"
     "                r[\"inst_t\"] = ids.get(key + (1,))\n"
     "                try:\n"
     "                    tok_ok, tmsg = (False, \"no TAIL instance built\") if r[\"inst_t\"] is None else _gd_cat_tail_selftest(r, state, b)\n"
     "                except Exception as ex:\n"
     "                    tok_ok, tmsg = False, f\"EXC {ex!r}\"\n"
     "                if tok_ok:\n"
     "                    print(f\"[gt] self-test cat/{b} fused tail: ok ({tmsg})\", flush=True)\n"
     "                else:\n"
     "                    r[\"inst_t\"] = None\n"
     "                    state[\"gd_cat_ctr\"].zero_()\n"
     "                    print(f\"[gt] FALLBACK cat/{b}: {tmsg} -> GC + the separate tail\", flush=True)\n"
     "                routes[b] = r\n"
     "                live.append(f\"cat/{b} {desc}{' +tail' if r['inst_t'] is not None else ''}\")\n",
     "GT instance + self-test in _gd_cat_prepare"),
    ("        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)\n"
     "        if tokens not in _GD_CAT_SEEN:\n"
     "            _GD_CAT_SEEN.add(tokens)\n"
     "            print(f\"[gc] first call T={tokens}: {r['desc']} pdl {GD_CAT_PDL}\", flush=True)\n"
     "        return _v8_tail(y, part, state, data)\n",
     "        if GD_CAT_TAIL and r.get(\"inst_t\") is not None and state.get(\"gd_cat_tail_ok\", True):\n"
     "            try:\n"
     "                out = _gd_cat_tail_run(r, state, x, y, part, data[\"local_start\"], tokens, GD_CAT_PDL)\n"
     "                if (\"t\", tokens) not in _GD_CAT_SEEN:\n"
     "                    _GD_CAT_SEEN.add((\"t\", tokens))\n"
     "                    print(f\"[gt] first call T={tokens}: {r['desc']} + fused tail, pdl {GD_CAT_PDL}\", flush=True)\n"
     "                return out\n"
     "            except Exception as ex:\n"
     "                state[\"gd_cat_tail_ok\"] = False\n"
     "                print(f\"[gt] call raised ({ex!r}): GC + the separate tail from here on\", flush=True)\n"
     "        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)\n"
     "        if tokens not in _GD_CAT_SEEN:\n"
     "            _GD_CAT_SEEN.add(tokens)\n"
     "            print(f\"[gc] first call T={tokens}: {r['desc']} pdl {GD_CAT_PDL}\", flush=True)\n"
     "        return _v8_tail(y, part, state, data)\n",
     "_gd_cat: fused tail first, GC + tail as fallback"),
    ("\n\ndef _gd_cat_live(state, tokens):\n", PY_FUNCS + "\n\ndef _gd_cat_live(state, tokens):\n", "GT python functions"),
    ('functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat"]',
     'functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat", "gdt_run"]',
     "GD module exports gdt_run"),
]


def main():
    GM.check(not OUT.exists(), f"{OUT.name} exists; pick another --n")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    nonce = (f"# nonce {TAG} ({BASE.stem} + GT: the routing tail fused into the GC cat launch at T = 9..32 (last-T-arriver CTAs, "
             "bitwise the shipped tail), GD_CAT_TAIL read at call time, GC + separate tail as fallback)")
    t = b.replace(bn, nonce)
    for a, b2, what in CUDA_PATCHES:
        GM.check(t.count(a) == 1, f"anchor for {what}: found {t.count(a)}")
        t = t.replace(a, b2)
        rep.append(what)
    for a, b2, n, what in CUDA_REPLACE_ALL:
        GM.check(t.count(a) == n, f"anchor for {what}: found {t.count(a)}, expected {n}")
        t = t.replace(a, b2)
        rep.append(what)
    marker = "\nstatic constexpr int GDR_DS = 2 * 8 * 256 * 16;"
    GM.check(t.count(marker) == 1, f"GDR_DS marker: found {t.count(marker)}")
    t = t.replace(marker, CUDA_HOST_ADD + "static constexpr int GDR_DS = 2 * 8 * 256 * 16;")
    rep.append("gdt_run_raw")
    cpp_anchor = ("void gd_unpack_cat(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t wk, int64_t kup,\n")
    GM.check(t.count(cpp_anchor) == 1, f"gd_unpack_cat binding: found {t.count(cpp_anchor)}")
    t = t.replace(cpp_anchor, CPP_ADD + cpp_anchor)
    rep.append("CPP binding gdt_run")
    for a, b2, what in PY_PATCHES:
        GM.check(t.count(a) == 1, f"anchor for {what}: found {t.count(a)}")
        t = t.replace(a, b2)
        rep.append(what)
    GM.check(t.count("@INST@") == 4, "three @INST@ switch bodies + the Python replace call")
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_gt] OK  {r}")
    OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_gt] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, headroom "
          f"{1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""re_patches.py: L1 ROUTE-EARLY step 0 (lead_oct29.md s3 L1; moe-front16 builder) as anchored text patches on a layer file
(base k3_gc_20 = k3_sfx_2: SFX2 already in), plus the generator:

  python -I tools/re_patches.py [--base=k3_gc_20] [--out=k3_re_1] [--dry]

Step 0 = router-first pooler launch + parallel routing, marks only (tail2 still routes; the RE routing goes to scratch and is
compared bitwise with tail2's by tools/box/re_marks.py):
  - T = 1..8 (gd_kernel VF = 3, gdre mode 3): VF = 2's router32 rows (FP32, same arithmetic) run FIRST in every pooler CTA
    (right after the pooler's own x loads are issued), partials out, red.release counter; then the pooler K loop.
    T = 1 uses it too (ship: GD pooler + side-stream lean router32, same per-slice arithmetic -> same partial bits).
  - T = 9..16 (GC cat, VF = 3 + CAT = 2, mode 13): the two virtual router units run FIRST (unit order KUP, KUP+1, 0..KUP-1;
    escapes matched per step by ballot), the router partial store + counter release mid-loop, then the pooler units.
  - ceil(T / 8) routing CTAs at blockIdx.x 0.. (grid 256 + nrc <= 264 slots: one wave): pdl wait, bounded acquire spin on
    the counter (no trap), tail_kernel2's routing warp per token (ld.cg partials), scratch tw / tid / lid, counter reset by
    the last routing CTA, marks [ok, t_in, t_full, t_done] at rows 256 + c.
  - marks (VF_GDM = 1): every arm writes _RE["marks"] (rows 0..255 pooler CTAs [smid, entry, wait-return, end]); mode 11 =
    ship cat + marks; T = 1 ship marks via gdr vf 1.
Knobs: RE_ON (RE arm), VF_GDM (marks). Nothing else changes; VMS / VML / vm_fin untouched.
"""
import hashlib
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
NL = chr(10)


def _sub(s, old, new, n=1):
    if s.count(old) != n:
        raise SystemExit(f"[re_patches] anchor count {s.count(old)} != {n}: {old[:80]!r}")
    return s.replace(old, new)


def _sub_in(s, lo, hi, old, new, n=1):
    """replace inside the unique region [lo .. hi) only"""
    if s.count(lo) != 1:
        raise SystemExit(f"[re_patches] region anchor count {s.count(lo)} != 1: {lo[:80]!r}")
    i = s.index(lo)
    j = s.index(hi, i)
    return s[:i] + _sub(s[i:j], old, new, n) + s[j:]


RE_DEV = r'''
// RE (tools/re_patches.py, L1 ROUTE-EARLY step 0): routing CTAs inside the pooler launch (blockIdx.x < A.nrc)
__device__ __forceinline__ void re_release(int* p) { asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" :: "l"(p) : "memory"); }
__device__ __forceinline__ void re_tok(const Args& A, int t, const float* la, long long sst, int l_ld, int lane) {
  const float4* pbias = reinterpret_cast<const float4*>(A.rbias) + lane * 2;   // tail_kernel2's routing warp, verbatim order
  const float4 c0 = pbias[0], c1 = pbias[1];
  float4 qa[8][2];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const float4* pq = reinterpret_cast<const float4*>(la + q * sst + (long long)t * l_ld) + lane * 2;
    qa[q][0] = __ldcg(pq);
    qa[q][1] = __ldcg(pq + 1);
  }
  const float4 qb0 = make_float4(0.f, 0.f, 0.f, 0.f), qb1 = qb0;
  float s[8], b[8];
  float4 a0 = qa[0][0], a1 = qa[0][1];
#pragma unroll
  for (int q = 1; q < 8; ++q) {
    a0.x += qa[q][0].x; a0.y += qa[q][0].y; a0.z += qa[q][0].z; a0.w += qa[q][0].w;
    a1.x += qa[q][1].x; a1.y += qa[q][1].y; a1.z += qa[q][1].z; a1.w += qa[q][1].w;
  }
  a0.x += qb0.x; a0.y += qb0.y; a0.z += qb0.z; a0.w += qb0.w; a1.x += qb1.x; a1.y += qb1.y; a1.z += qb1.z; a1.w += qb1.w;
  const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
  const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    s[j] = 1.f / (1.f + expf(-lg[j] * A.rmult));
    b[j] = s[j] + bs[j];
  }
  int sel_id[8];
  float sel_s[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
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
        if (j == bj) b[j] = __int_as_float(0xff800000);
    }
  }
  if (lane == 0) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum += sel_s[i];
    const float inv = A.rrs / (sum + A.rdeps);
    const int ls = A.rls[0];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      A.rtw[t * 8 + k] = sel_s[k] * inv;
      A.rtid[t * 8 + k] = sel_id[k];
      const int rel = ((sel_id[k] - ls) % 256 + 256) % 256;
      A.rlid[t * 8 + k] = rel < 32 ? rel : -1;
    }
  }
}
__device__ __forceinline__ void re_cta(const Args& A, const float* la, long long sst, int l_ld, int npool, int nw, int lane,
                                       int warp) {
  __shared__ int re_ok;
  pdl_wait();
  const long long t_in = vf_gt();
  if (threadIdx.x == 0) {
    int v = 0;
    for (unsigned it = 0;; ++it) {   // bounded: a missing pooler CTA costs ~1 s and flags ok = 0, never a trap
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(A.rcnt) : "memory");
      if (v >= npool || it > (1u << 21)) break;
      __nanosleep(32);
    }
    __threadfence();
    re_ok = v >= npool ? 1 : 0;
  }
  __syncthreads();
  const long long t_full = vf_gt();
  for (int t = (int)blockIdx.x * nw + warp; t < A.M; t += A.nrc * nw) re_tok(A, t, la, sst, l_ld, lane);
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(A.rcnt + 1, 1) == A.nrc - 1) { atomicExch(A.rcnt, 0); atomicExch(A.rcnt + 1, 0); }
    if (A.marks) {
      long long* m = A.marks + (size_t)(npool + (int)blockIdx.x) * 4;
      m[0] = re_ok; m[1] = t_in; m[2] = t_full; m[3] = vf_gt();
    }
  }
}

#define GD_PS(S_) ((VF == 3 && CAT > 0) ? (((S_) < CAT) ? KUP + (S_) : (S_) - CAT) : (S_))   // RE cat: router units first
'''

RE_ROUTER_FIRST = r'''  if constexpr (VF == 3 && CAT == 0) {   // RE: router32 rows FIRST (VF = 2 arithmetic), partials out, counter release
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    if (warp < 8) {
      const float4* wsm = gd_vf_dsm + warp * 256;
      const float4* xs = gd_vf_dsm + 8 * 256;
      float racc[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) racc[j] = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int c = i * 32 + lane;
        const float4 wv = wsm[c];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (j < A.M) {
            const float4 xv = xs[j * 256 + c];
            racc[j] = fmaf(wv.x, xv.x, racc[j]);
            racc[j] = fmaf(wv.y, xv.y, racc[j]);
            racc[j] = fmaf(wv.z, xv.z, racc[j]);
            racc[j] = fmaf(wv.w, xv.w, racc[j]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j < A.M) {
          float a = racc[j];
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
          if (lane == 0) A.part[((long)vf_sl * A.M + j) * 256 + vf_bx * 8 + warp] = a;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) re_release(A.rcnt);
  }
'''

RE_CAT_STORE = r'''        if constexpr (VF == 3) {   // RE: router units done first: partials out (the GC store), counter release
          __syncthreads();
          const int gc_t = bxi >> 3, gc_ks = bxi & 7;
          for (int i = (int)threadIdx.x; i < NO / 2; i += NT) {
            const int l = i & 31, gg = l >> 2, tt = l & 3;
            const int c = (i >> 5) & 1, mt = i >> 6;
            float v = 0.f;
#pragma unroll
            for (int k2 = 0; k2 < WK; ++k2) v += redr[k2][mt * 4 + c][l] + redr[k2][mt * 4 + c + 2][l];
            const int tok = tok0 + mt * 8 + 2 * tt + c;
            if (tok < A.M)
              reinterpret_cast<float*>(A.out2)[(size_t)gc_ks * A.sst2 + (size_t)tok * A.ldo2 + gc_t * 8 + gg] = v;
          }
          __syncthreads();
          if (threadIdx.x == 0) re_release(A.rcnt);
        }
'''

RE_GLUE = r'''#define GDRE3 gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 3, 0>
#define GDRE11 gd_kernel<0, 2, 1, 8, 18, 2, 1, 2, 0, 1, 2>
#define GDRE13 gd_kernel<0, 2, 1, 8, 18, 2, 1, 2, 0, 3, 2>
int gdre_run_raw(int mode, const void* w, const void* esc, const void* x, void* out, void* out2, int M, int K, int ldo, int ldo2,
                 uint32_t tb, uint32_t tbr, int nbx, int pdl, const float* x32, const float* w32, long long* marks, int* cnt,
                 float* tw, int* tid, int* lid, const float* bias, const int* ls, float mult, float rscale, float deps,
                 cudaStream_t st) {
  Args A;
  A.w = reinterpret_cast<const uint4*>(w);
  A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.x = reinterpret_cast<const uint16_t*>(x);
  A.out = out;
  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;
  A.trig = (pdl >> 1) & 1;
  A.x32 = x32; A.w32 = w32; A.part = reinterpret_cast<float*>(out2); A.marks = marks;
  A.out2 = out2; A.ldo2 = ldo2; A.sst2 = (long long)M * ldo2; A.tbr = tbr;
  A.rcnt = cnt; A.rtw = tw; A.rtid = tid; A.rlid = lid; A.rbias = bias; A.rls = ls;
  A.rmult = mult; A.rrs = rscale; A.rdeps = deps;
  A.nrc = (mode == 3 || mode == 13) ? (M + 7) / 8 : 0;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(nbx + A.nrc), 1, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  unsigned na = 0;
  if (pdl & 1) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    na = 1;
  }
  cfg.attrs = attr;
  cfg.numAttrs = na;
  cfg.dynamicSmemBytes = mode == 3 ? GDR_DS : 0;
  if (mode == 3) {
    static bool set = false;
    if (!set) {
      const int e = (int)cudaFuncSetAttribute(GDRE3, cudaFuncAttributeMaxDynamicSharedMemorySize, GDR_DS);
      if (e) return e;
      set = true;
    }
    return (int)cudaLaunchKernelEx(&cfg, GDRE3, A);
  }
  if (mode == 11) return (int)cudaLaunchKernelEx(&cfg, GDRE11, A);
  if (mode == 13) return (int)cudaLaunchKernelEx(&cfg, GDRE13, A);
  return -1;
}

int gdre_info_raw(int mode, int64_t* res) {
  cudaFuncAttributes fa;
  auto fn = mode == 3 ? GDRE3 : mode == 11 ? GDRE11 : GDRE13;
  const int ds = mode == 3 ? GDR_DS : 0;
  int e = (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, ds);
  if (e) return e;
  e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 256, ds);
  res[0] = fa.numRegs; res[1] = (int64_t)fa.localSizeBytes; res[2] = occ; res[3] = (int64_t)fa.sharedSizeBytes;
  return e;
}

'''

RE_CPP = r'''int gdre_run_raw(int mode, const void* w, const void* esc, const void* x, void* out, void* out2, int M, int K, int ldo, int ldo2,
                 uint32_t tb, uint32_t tbr, int nbx, int pdl, const float* x32, const float* w32, long long* marks, int* cnt,
                 float* tw, int* tid, int* lid, const float* bias, const int* ls, float mult, float rscale, float deps,
                 cudaStream_t st);
int gdre_info_raw(int mode, int64_t* res);
void gdre_run(int64_t mode, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, torch::Tensor out2,
              int64_t M, int64_t K, int64_t ldo, int64_t ldo2, int64_t tb, int64_t tbr, int64_t nbx, int64_t pdl,
              torch::Tensor x32, torch::Tensor w32, torch::Tensor marks, torch::Tensor cnt, torch::Tensor tw, torch::Tensor tid,
              torch::Tensor lid, torch::Tensor bias, torch::Tensor ls, double mult, double rscale, double deps) {
  TORCH_CHECK(nbx == 256 && M >= 1 && M <= 16 && out2.scalar_type() == at::kFloat && out2.is_contiguous() &&
              out2.numel() == 8 * M * 256 && cnt.scalar_type() == at::kInt && cnt.numel() >= 4 && tw.numel() >= 8 * M &&
              tid.numel() >= 8 * M && lid.numel() >= 8 * M && bias.scalar_type() == at::kFloat && bias.numel() == 256 &&
              ls.scalar_type() == at::kInt, "gdre_run: args");
  if (mode == 3)
    TORCH_CHECK(M <= 8 && x32.scalar_type() == at::kFloat && x32.is_contiguous() && x32.numel() == M * 8192 &&
                w32.scalar_type() == at::kFloat && w32.numel() == 256 * 8192, "gdre_run: mode 3 router inputs");
  long long* mk = marks.numel() >= 4 * (nbx + 2) ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  const int e = gdre_run_raw((int)mode, w.data_ptr(), esc.data_ptr(), x.data_ptr(), out.data_ptr(), out2.data_ptr(), (int)M,
                             (int)K, (int)ldo, (int)ldo2, (uint32_t)tb, (uint32_t)tbr, (int)nbx, (int)pdl,
                             mode == 3 ? x32.data_ptr<float>() : nullptr, mode == 3 ? w32.data_ptr<float>() : nullptr, mk,
                             cnt.data_ptr<int>(), tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(),
                             bias.data_ptr<float>(), ls.data_ptr<int>(), (float)mult, (float)rscale, (float)deps,
                             at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gdre_run: mode ", mode, " CUDA error ", e, " ", cudaGetErrorString((cudaError_t)(e < 0 ? 1 : e)));
}
std::vector<int64_t> gdre_info(int64_t mode) {
  std::vector<int64_t> r(4, 0);
  const int e = gdre_info_raw((int)mode, r.data());
  TORCH_CHECK(e == 0, "gdre_info: mode ", mode, " error ", e);
  return r;
}
'''

RE_PY = r'''

RE_ON = False   # L1 ROUTE-EARLY step 0 (tools/re_patches.py): router-first pooler launch + routing CTAs (scratch; tail2 still routes)
RE_MULT = 0.1767766952966369
_RE = {"ok": False, "cnt": None, "marks": None, "nomk": None, "none": None, "out": None, "ref": {}}


def _re_alloc(dev):
    if _RE["cnt"] is None:
        _RE["cnt"] = torch.zeros((4,), dtype=torch.int32, device=dev)
        _RE["marks"] = torch.zeros((4 * 258,), dtype=torch.int64, device=dev)
        _RE["nomk"] = torch.empty((0,), dtype=torch.int64, device=dev)
        _RE["none"] = torch.empty((0,), dtype=torch.float32, device=dev)
        _RE["out"] = (torch.zeros((16, TOP_K), dtype=torch.float32, device=dev),
                      torch.zeros((16, TOP_K), dtype=torch.int32, device=dev), torch.zeros((16, TOP_K), dtype=torch.int32, device=dev))


def _re_run(state, mode, r, x, y, part, x32, lstart, marks, pdl=None):
    tw, tid, lid = _RE["out"]
    if pdl is None:
        pdl = ((1 if GD_PDL.get("pooler") else 0) | (2 if GD_TRIG.get("pooler") else 0)) if mode == 3 else GD_CAT_PDL
    none = _RE["none"]
    r["mod"].gdre_run(mode, r["w"], r["esc"], x, y, part, x.shape[0], HIDDEN, D_EXPERT, N_EXPERTS, r["tb"], r.get("tbr", 0),
                      r["nbx"], pdl, x32 if x32 is not None else none, state["router"] if mode == 3 else none, marks, _RE["cnt"],
                      tw, tid, lid, state["v8"]["bias"], lstart, RE_MULT, 1.0, 1e-15)


def _re_cat(state, r, x, y, part, T, data):
    if not ((RE_ON or VF_GDM) and _RE["ok"] and _gd_bucket(T) == 16 and r["nbx"] == 256):
        return False
    _re_run(state, 13 if RE_ON else 11, r, x, y, part, None, data["local_start"], _RE["marks"] if VF_GDM else _RE["nomk"])
    return True


def _re_prepare(state):
    _RE["ok"] = False
    try:
        gc = state.get("gd_cat")
        r8 = _gd_live(state, "pooler", 8)
        if not _VFR["ok"] or r8 is None or gc is None or gc.get(16) is None or state.get("v8") is None:
            print("[re] OFF: needs the fused VF pooler (T <= 8) and the GC cat (T = 16)", flush=True)
            return
        dev = state["router"].device
        _re_alloc(dev)
        info = {m: list(r8["mod"].gdre_info(m)) for m in (3, 11, 13)}
        print(f"[re] regs / local / occ / smem per mode: {info} (vf2 {list(r8['mod'].gdr_info(2))})", flush=True)
        if any(v[2] < 2 for v in info.values()):
            print("[re] OFF: below 2 CTAs / SM", flush=True)
            return
        gen = torch.Generator(device=dev).manual_seed(9393)
        ok = True
        for T, ls in ((1, 0), (5, 3), (8, 7), (9, 1), (16, 5)):
            lstart = torch.tensor([ls], dtype=torch.int32, device=dev)
            z = torch.randn((T, HIDDEN), generator=gen, device=dev)
            z = (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True))).contiguous()
            h = (z * 0.7).to(torch.bfloat16).contiguous()
            y0 = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
            y1 = torch.empty_like(y0)
            p0 = torch.full((8, T, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
            p1 = torch.full_like(p0, float("nan"))
            if T <= 8:
                r = _gd_live(state, "pooler", T)
                r["mod"].gdr_run(2, r["w"], r["esc"], h, y0, T, r["K"], r["N"], r["tb"], r["nbx"], 0, z, state["router"], p0,
                                 _VFR["nomk"])
                _re_run(state, 3, r, h, y1, p1, z, lstart, _RE["marks"], 0)
            else:
                r = gc[16]
                _gd_cat_run(r, h, y0, p0, T, 0)
                _re_run(state, 13, r, h, y1, p1, None, lstart, _RE["marks"], 0)
            _, tw, tid, lid = _v8_tail(y0, p0, state, {"local_start": lstart})
            torch.cuda.synchronize()
            rtw, rtid, rlid = (t[:T] for t in _RE["out"])
            nrc = (T + 7) // 8
            spin = bool((_RE["marks"].view(-1, 4)[256:256 + nrc, 0] == 1).all())
            e = (bool(torch.equal(y0, y1)), bool(torch.equal(p0, p1)), bool(torch.equal(tid, rtid)), bool(torch.equal(tw, rtw)),
                 bool(torch.equal(lid, rlid)), int(_RE["cnt"][:2].abs().sum()) == 0, spin)
            print(f"[re] self-test T={T} ls={ls}: pooler {e[0]} partials {e[1]} tid {e[2]} tw {e[3]} lid {e[4]} cnt-reset {e[5]} "
                  f"spin-ok {e[6]}", flush=True)
            ok &= all(e)
        _RE["ok"] = ok
    except Exception as ex:
        print(f"[re] OFF: {ex!r}", flush=True)
        _RE["ok"] = False
    print(f"[re] router-first pooler + routing CTAs {'READY' if _RE['ok'] else 'OFF'} (RE_ON {RE_ON})", flush=True)
'''


def patch(s):
    # --- GD CUDA: Args, device helpers, kernel ---
    s = _sub(s, "  void* out2 = nullptr; int ldo2 = 0; long long sst2 = 0; uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n",
             "  void* out2 = nullptr; int ldo2 = 0; long long sst2 = 0; uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n"
             "  int* rcnt = nullptr; float* rtw = nullptr; int* rtid = nullptr; int* rlid = nullptr; const float* rbias = nullptr;   // RE\n"
             "  const int* rls = nullptr; int nrc = 0; float rmult = 0.f, rrs = 1.f, rdeps = 0.f;   // RE routing CTAs (re_patches.py)\n")
    s = _sub(s, "\n#define GC_XP(S_) ", RE_DEV + "#define GC_XP(S_) ")
    s = _sub(s, "#undef GC_XP\n", "#undef GC_XP\n#undef GD_PS\n")
    k0, k1 = "gd_kernel(const __grid_constant__ Args A) {", "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0>"
    s = _sub_in(s, k0, k1, "  const int wk = warp % WK;\n",
                "  if constexpr (VF == 3) {   // RE: the routing CTAs\n"
                "    if ((int)blockIdx.x < A.nrc) {\n"
                "      re_cta(A, CAT ? reinterpret_cast<const float*>(A.out2) : A.part, CAT ? A.sst2 : (long long)A.M * 256,\n"
                "             CAT ? A.ldo2 : 256, (int)gridDim.x - A.nrc, NW, lane, warp);\n"
                "      return;\n"
                "    }\n"
                "  }\n"
                "  const int bxi = (int)blockIdx.x - (VF == 3 ? A.nrc : 0), gdx = (int)gridDim.x - (VF == 3 ? A.nrc : 0);\n"
                "  const int wk = warp % WK;\n")
    s = _sub_in(s, k0, k1, "const int G = (int)(gridDim.x * gridDim.y) * NW;", "const int G = gdx * (int)gridDim.y * NW;")
    s = _sub_in(s, k0, k1, "((int)blockIdx.y * (int)gridDim.x + (int)blockIdx.x) * NW + warp;", "((int)blockIdx.y * gdx + bxi) * NW + warp;")
    s = _sub_in(s, k0, k1, "const int vf_c = (int)blockIdx.x, ", "const int vf_c = bxi, ")
    s = _sub_in(s, k0, k1, "(size_t)(((int)blockIdx.x & 7) * 16 + 2 * wk)", "(size_t)((bxi & 7) * 16 + 2 * wk)")
    s = _sub_in(s, k0, k1, "const int row = ((int)blockIdx.x * WR + w2) * 16", "const int row = (bxi * WR + w2) * 16")
    s = _sub_in(s, k0, k1, "const int gc_t = (int)blockIdx.x >> 3, gc_ks = (int)blockIdx.x & 7;", "const int gc_t = bxi >> 3, gc_ks = bxi & 7;")
    s = _sub_in(s, k0, k1, "  if constexpr (VF == 2) {   // router32 CTA", "  if constexpr (VF == 2 || (VF == 3 && CAT == 0)) {   // router32 CTA")
    s = _sub_in(s, k0, k1, "  if constexpr (VF == 2) {   // the router input slice", "  if constexpr (VF == 2 || (VF == 3 && CAT == 0)) {   // the router input slice")
    s = _sub_in(s, k0, k1, "R[i][p] = ldw(wp + (size_t)i * wstep + p * 32);", "R[i][p] = ldw(wp + (size_t)GD_PS(i) * wstep + p * 32);")
    s = _sub_in(s, k0, k1, "    X[0][mt][0] = ldx(xb + xo[mt]);\n    X[0][mt][1] = ldx(xb + xo[mt] + 1);\n",
                "    X[0][mt][0] = ldx(GC_XP(GD_PS(0)) + xo[mt]);\n    X[0][mt][1] = ldx(GC_XP(GD_PS(0)) + xo[mt] + 1);\n")
    s = _sub_in(s, k0, k1, "  int ecur = 0;\n", RE_ROUTER_FIRST + "  int ecur = 0;\n")
    s = _sub_in(s, k0, k1, "ldx(GC_XP(s + 1) + xo[mt]);", "ldx(GC_XP(GD_PS(s + 1)) + xo[mt]);")
    s = _sub_in(s, k0, k1, "ldx(GC_XP(s + 1) + xo[mt] + 1);", "ldx(GC_XP(GD_PS(s + 1)) + xo[mt] + 1);")
    s = _sub_in(s, k0, k1, "ldx(GC_XP(s) + xo[mt]);", "ldx(GC_XP(GD_PS(s)) + xo[mt]);")
    s = _sub_in(s, k0, k1, "ldx(GC_XP(s) + xo[mt] + 1);", "ldx(GC_XP(GD_PS(s)) + xo[mt] + 1);")
    s = _sub_in(s, k0, k1, "decode(R[slot], (CAT && s >= KUP) ? A.tbr : A.tb, a);", "decode(R[slot], (CAT && GD_PS(s) >= KUP) ? A.tbr : A.tb, a);")
    s = _sub_in(s, k0, k1, "R[slot][p] = ldw(wp + (size_t)(s + ST) * wstep + p * 32);", "R[slot][p] = ldw(wp + (size_t)GD_PS(s + ST) * wstep + p * 32);")
    old_esc = ("    while ((int)(enx >> 20) == s) {\n      patch(a, enx, lane);\n      ++ecur;\n"
               "      enx = ecur < 32 ? __shfl_sync(0xffffffffu, em, ecur & 31) : 0xFFFFFFFFu;\n    }\n")
    new_esc = ("    if constexpr (VF == 3 && CAT > 0) {   // RE: units out of stream order; escapes matched per step\n"
               "      uint32_t bm = __ballot_sync(0xffffffffu, (int)(em >> 20) == GD_PS(s));\n"
               "      while (bm) {\n        patch(a, __shfl_sync(0xffffffffu, em, __ffs(bm) - 1), lane);\n        bm &= bm - 1;\n      }\n"
               "    } else {\n" + "".join("  " + ln + NL for ln in old_esc.rstrip(NL).split(NL)) + "    }\n")
    s = _sub_in(s, k0, k1, old_esc, new_esc)
    s = _sub_in(s, k0, k1, "      if (s == KUP - 1) {   // GC: pooler tile done;", "      if (s == (VF == 3 ? CAT - 1 : KUP - 1)) {   // GC: pooler tile done;")
    s = _sub_in(s, k0, k1, "            red[warp][mt * 4 + i][lane] = v;\n            acc[0][mt][i] = 0.f;\n",
                "            if constexpr (VF == 3) redr[warp][mt * 4 + i][lane] = v; else red[warp][mt * 4 + i][lane] = v;\n"
                "            acc[0][mt][i] = 0.f;\n")
    s = _sub_in(s, k0, k1, "            if constexpr (NC == 2) acc[1][mt][i] = 0.f;\n          }\n",
                "            if constexpr (NC == 2) acc[1][mt][i] = 0.f;\n          }\n" + RE_CAT_STORE)
    s = _sub_in(s, k0, k1, "      if constexpr (CAT > 0) redr[warp][mt * 4 + i][lane] = v; else red[warp][mt * 4 + i][lane] = v;",
                "      if constexpr (CAT > 0 && VF != 3) redr[warp][mt * 4 + i][lane] = v; else red[warp][mt * 4 + i][lane] = v;")
    s = _sub_in(s, k0, k1, "    if constexpr (CAT > 0) {   // GC: this K slice's router partial", "    if constexpr (CAT > 0 && VF != 3) {   // GC: this K slice's router partial")
    # --- glue ---
    s = _sub(s, "int gd_info_raw(int64_t inst, int64_t* res) {\n", RE_GLUE + "int gd_info_raw(int64_t inst, int64_t* res) {\n")
    s = _sub(s, "void gd_unpack_cat(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t wk, int64_t kup,\n",
             RE_CPP + "void gd_unpack_cat(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t wk, int64_t kup,\n")
    s = _sub(s, '"gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat"]', '"gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat", "gdre_run", "gdre_info"]')
    # --- python ---
    s = _sub(s, '_VFR = {"ok": False, "marks": None, "seen": set()}\n', '_VFR = {"ok": False, "marks": None, "seen": set()}\n' + RE_PY)
    s = _sub(s, "def _vfr_gdr(state, x, z, vf):\n    T = x.shape[0]\n    r = _gd_live(state, \"pooler\", T)\n    if r is None or _gd_bucket(T) != 8:\n",
             "def _vfr_gdr(state, x, z, vf, lstart=None):\n    T = x.shape[0]\n    r = _gd_live(state, \"pooler\", T)\n"
             "    if r is None or _gd_bucket(T) not in ((1, 8) if (vf == 3 or VF_GDM) else (8,)):\n")
    s = _sub(s, '    mk = _VFR["marks"] if VF_GDM else _VFR["nomk"]\n    r["mod"].gdr_run(vf,',
             '    mk = _RE["marks"] if VF_GDM else _VFR["nomk"]\n    if vf == 3:\n'
             '        _re_run(state, 3, r, x, y, part, z, lstart, mk if VF_GDM else _RE["nomk"])\n'
             '    else:\n      r["mod"].gdr_run(vf,')
    s = _sub(s, '                     z if vf == 2 else _VFR["nomk"], state["router"], part, mk)\n',
             '                       z if vf == 2 else _VFR["nomk"], state["router"], part, mk)\n')
    s = _sub(s, "def _vfr_fused(state, ffn_input, router_input):\n", "def _vfr_fused(state, ffn_input, router_input, data=None):\n")
    s = _sub(s, "    if not (2 <= ffn_input.shape[0] <= 8 and router_input.is_contiguous() and ffn_input.is_contiguous()):\n"
                "        return None\n    return _vfr_gdr(state, ffn_input, router_input, 2)\n",
             "    re_on = RE_ON and _RE[\"ok\"] and data is not None\n"
             "    if not ((1 if re_on else 2) <= ffn_input.shape[0] <= 8 and router_input.is_contiguous() and ffn_input.is_contiguous()):\n"
             "        return None\n"
             "    return _vfr_gdr(state, ffn_input, router_input, 3, data[\"local_start\"]) if re_on else _vfr_gdr(state, ffn_input, router_input, 2)\n")
    s = _sub(s, "    if not (VF_GDM and _VFR[\"ok\"]) or not 2 <= x.shape[0] <= 8 or not x.is_contiguous():\n",
             "    if not (VF_GDM and _VFR[\"ok\"]) or not 1 <= x.shape[0] <= 8 or not x.is_contiguous():\n")
    s = _sub(s, "vgr = _vfr_fused(state, ffn_input, router_input) if (tail is not None", "vgr = _vfr_fused(state, ffn_input, router_input, data) if (tail is not None")
    s = _sub(s, "        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)\n",
             "        if not _re_cat(state, r, x, y, part, tokens, data):\n            _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)\n")
    s = _sub(s, "                      RMS_EPS, 0.1767766952966369, 1.0, 1e-15, (1 if V9_TAIL_PDL else 0) | (2 if TL1_TAIL2 else 0))\n",
             "                      RMS_EPS, 0.1767766952966369, 1.0, 1e-15, (1 if V9_TAIL_PDL else 0) | (2 if TL1_TAIL2 else 0))\n"
             "    if RE_ON:\n        _RE[\"ref\"][tokens] = (tw, tid, lid)\n")
    s = _sub(s, "    _vfr_prepare(state)\n    _va1_prepare(state)\n", "    _vfr_prepare(state)\n    _re_prepare(state)\n    _va1_prepare(state)\n")
    return s


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    base = ROOT / f"kernels/muse_decode_layer/{args.get('base', 'k3_gc_20')}.py"
    outn = args.get("out", "k3_re_1")
    out = ROOT / f"kernels/lab/{outn}.py"
    if "--dry" not in sys.argv and out.exists():
        raise SystemExit(f"[re_patches] {out.name} exists")
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    assert bn.startswith("# nonce ")
    t = patch(b)
    nonce = f"# nonce {outn.replace('_', '-')} ({base.stem} + RE step 0: router-first pooler launch + routing CTAs; tools/re_patches.py)"
    t = t.replace(bn, nonce, 1)
    assert t.split(NL)[1] == nonce and t.isascii()
    assert len(t.encode()) < 1_000_000, len(t.encode())
    assert max(len(x) for x in t.split(NL)) <= max(len(x) for x in b.split(NL)), "line longer than the base's longest"
    assert len(re.findall(r"importlib|sys\.modules", t)) == len(re.findall(r"importlib|sys\.modules", b))
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[re_patches] OK; {len(t.encode())} B (base {len(b.encode())} B)")
    if "--dry" in sys.argv:
        return 0
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[re_patches] wrote {out.relative_to(ROOT)} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

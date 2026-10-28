"""vfr_patches.py: S2 (notes/study/system_gap_oct28.md) = the T = 8 FP32 router rows INSIDE the GD pooler grid, plus per-CTA
marks of the stage (step-0), on a layer file whose GD pooler route at bucket 8 is (1, 1, 8, 16, 2, 1, 1) = gd_kernel<0, 1, 1, 8,
16, 2, 1, 2, 1> (256 CTAs x 256 thr, 2 per SM, one wave). Default base kernels/muse_decode_layer/k3_slim_5.py.

gd_kernel gets a 10th template parameter VF (default 0: every existing instance compiles exactly as before):
  VF = 1  the stock pooler + per-CTA marks [smid, t_entry, t_wait_return, t_end] (%globaltimer) when A.marks != nullptr.
  VF = 2  pooler CTA c ALSO computes router32's CTA (bx, sl) = (c % 32, c / 32): its 8 router rows x one 1024-K slice of FP32
          weights are cp.async'd into 32 KB of dynamic smem BEFORE griddepcontrol.wait (weights only), the FP32 x slice of
          the <= 8 tokens (32 KB) right after it; after the pooler rows are stored, warps 0-7 run router32's per-lane FMAs
          (same order, same xor tree: bitwise equal partials) and write part[sl][t][row]. One launch, one wave, no CTA of
          any other kernel needs a pooler slot; the router's 8.4 MB move into the window before the wait. Marks as VF = 1.
Host: gdr_run(vf, ...) / gdr_info(vf) (64 KB dynamic smem for VF = 2). Python: VF_GDR (call time) routes the T = 2..8 f32 /
sep path through it (no side stream); VF_GDM marks the stock pooler (or the fused one). prepare() self-tests VF = 1 and 2:
pooler output bitwise vs the stock instance, router partials bitwise vs router32, at m = 8 / 5 / 2; fails -> loud [vfr] FALLBACK.

    python -I tools/vfr_patches.py --n=<new> [--base=k3_slim_5] [--on] [--lab-only] [--force]   (--on: VF_GDR = True)
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)


def sub(t, old, new, count=1, what=""):
    n = t.count(old)
    if n != count:
        raise SystemExit(f"[vfr] anchor {what or old[:60]!r}: found {n}, expected {count}")
    return t.replace(old, new)


GEO = "0, 1, 1, 8, 16, 2, 1, 2, 1"

DEV_HELPERS = r'''
extern __shared__ __align__(16) float4 gd_vf_dsm[];   // VF = 2: router weight rows [8][256] float4 + x slice [8][256] float4
__device__ __forceinline__ void vf_cp16(void* sdst, const void* gsrc) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(sdst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gsrc) : "memory");
}
__device__ __forceinline__ long long vf_gt() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ int vf_smid() {
  int s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

'''

PRE_WAIT = r'''  const int vf_c = (int)blockIdx.x, vf_bx = vf_c & 31, vf_sl = vf_c >> 5;
  if constexpr (VF >= 1) {
    if (A.marks && threadIdx.x == 0) { A.marks[vf_c * 4 + 0] = vf_smid(); A.marks[vf_c * 4 + 1] = vf_gt(); }
  }
  if constexpr (VF == 2) {   // router32 CTA (vf_bx, vf_sl): 8 rows x 1024 K of FP32 weights (no producer dependency)
    for (int idx = (int)threadIdx.x; idx < 8 * 256; idx += NT) {
      const int r = idx >> 8, c = idx & 255;
      vf_cp16(gd_vf_dsm + idx, reinterpret_cast<const float4*>(A.w32 + (size_t)(vf_bx * 8 + r) * 8192 + vf_sl * 1024) + c);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
'''

POST_WAIT = r'''  if constexpr (VF >= 1) {
    if (A.marks && threadIdx.x == 0) A.marks[vf_c * 4 + 2] = vf_gt();
  }
  if constexpr (VF == 2) {   // the router input slice (the boundary's FP32 output), tokens >= M zero
    float4* xs = gd_vf_dsm + 8 * 256;
    for (int idx = (int)threadIdx.x; idx < 8 * 256; idx += NT) {
      const int j = idx >> 8, c = idx & 255;
      if (j < A.M) vf_cp16(xs + idx, reinterpret_cast<const float4*>(A.x32 + (size_t)j * 8192 + vf_sl * 1024) + c);
      else xs[idx] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
'''

ROUTER_TAIL = r'''    if constexpr (VF == 2) {   // router32_kernel<8>'s arithmetic for row vf_bx * 8 + warp, K slice vf_sl
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
            const float4 xv = xs[j * 256 + c];
            racc[j] = fmaf(wv.x, xv.x, racc[j]);
            racc[j] = fmaf(wv.y, xv.y, racc[j]);
            racc[j] = fmaf(wv.z, xv.z, racc[j]);
            racc[j] = fmaf(wv.w, xv.w, racc[j]);
          }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float a = racc[j];
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
          if (lane == 0 && j < A.M) A.part[((long)vf_sl * A.M + j) * 256 + vf_bx * 8 + warp] = a;
        }
      }
    }
    if constexpr (VF >= 1) {
      if (A.marks) {
        __syncthreads();
        if (threadIdx.x == 0) A.marks[vf_c * 4 + 3] = vf_gt();
      }
    }
'''

HOST_CU = r'''
static constexpr int GDR_DS = 2 * 8 * 256 * 16;   // 64 KB dynamic smem for VF = 2

int gdr_run_raw(int vf, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo, uint32_t tb, int nbx,
                int pdl, const float* x32, const float* w32, float* part, long long* marks, cudaStream_t st) {
  Args A;
  A.w = reinterpret_cast<const uint4*>(w);
  A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.x = reinterpret_cast<const uint16_t*>(x);
  A.out = out;
  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;
  A.trig = (pdl >> 1) & 1;
  A.x32 = x32; A.w32 = w32; A.part = part; A.marks = marks;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)nbx, 1, 1);
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
  if (vf == 2) {
    static bool set = false;
    if (!set) {
      const int e = (int)cudaFuncSetAttribute(gd_kernel<@GEO@, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, GDR_DS);
      if (e) return e;
      set = true;
    }
    cfg.dynamicSmemBytes = GDR_DS;
    return (int)cudaLaunchKernelEx(&cfg, gd_kernel<@GEO@, 2>, A);
  }
  cfg.dynamicSmemBytes = 0;
  return (int)cudaLaunchKernelEx(&cfg, gd_kernel<@GEO@, 1>, A);
}

int gdr_info_raw(int vf, int64_t* res) {
  cudaFuncAttributes fa;
  auto fn = vf == 2 ? gd_kernel<@GEO@, 2> : gd_kernel<@GEO@, 1>;
  const int ds = vf == 2 ? GDR_DS : 0;
  int e = (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, ds > 0 ? ds : 0);
  if (e) return e;
  e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 256, ds);
  res[0] = fa.numRegs; res[1] = (int64_t)fa.localSizeBytes; res[2] = occ; res[3] = (int64_t)fa.sharedSizeBytes;
  return e;
}

int gd_info_raw(int64_t inst, int64_t* res) {
'''

HOST_CPP = r'''int gdr_run_raw(int vf, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo, uint32_t tb, int nbx,
                int pdl, const float* x32, const float* w32, float* part, long long* marks, cudaStream_t st);
int gdr_info_raw(int vf, int64_t* res);
void gdr_run(int64_t vf, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, int64_t M, int64_t K,
             int64_t ldo, int64_t tb, int64_t nbx, int64_t pdl, torch::Tensor x32, torch::Tensor w32, torch::Tensor part,
             torch::Tensor marks) {
  if (vf == 2)
    TORCH_CHECK(x32.scalar_type() == at::kFloat && x32.is_contiguous() && x32.size(0) == M && x32.size(1) == 8192 &&
                w32.scalar_type() == at::kFloat && w32.is_contiguous() && w32.size(0) == 256 && w32.size(1) == 8192 &&
                part.scalar_type() == at::kFloat && part.is_contiguous() && part.numel() == 8 * M * 256 && M >= 1 && M <= 8 &&
                reinterpret_cast<uintptr_t>(x32.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w32.data_ptr()) % 16 == 0,
                "gdr_run: x32 f32 [M <= 8, 8192], w32 f32 [256, 8192], part [8, M, 256]");
  TORCH_CHECK(nbx == 256, "gdr_run: the T = 8 pooler grid (256 row tiles)");
  long long* mk = marks.numel() >= 4 * nbx ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  const int e = gdr_run_raw((int)vf, w.data_ptr(), esc.data_ptr(), x.data_ptr(), out.data_ptr(), (int)M, (int)K, (int)ldo,
                            (uint32_t)tb, (int)nbx, (int)pdl, vf == 2 ? x32.data_ptr<float>() : nullptr,
                            vf == 2 ? w32.data_ptr<float>() : nullptr, vf == 2 ? part.data_ptr<float>() : nullptr, mk,
                            at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gdr_run: vf ", vf, " CUDA error ", e, " ", cudaGetErrorString((cudaError_t)(e < 0 ? 1 : e)));
}
std::vector<int64_t> gdr_info(int64_t vf) {
  std::vector<int64_t> r(4, 0);
  const int e = gdr_info_raw((int)vf, r.data());
  TORCH_CHECK(e == 0, "gdr_info: vf ", vf, " error ", e);
  return r;
}
'''

PY = r'''

VF_GDR = @ON@
VF_GDM = False
_VFR = {"ok": False, "marks": None, "seen": set()}


def _vfr_gdr(state, x, z, vf):
    T = x.shape[0]
    r = _gd_live(state, "pooler", T)
    if r is None or _gd_bucket(T) != 8:
        return None
    y = torch.empty((T, r["N"]), dtype=torch.bfloat16, device=x.device)
    part = torch.empty((8, T, N_EXPERTS), dtype=torch.float32, device=x.device)
    pdl = (1 if GD_PDL.get("pooler") else 0) | (2 if GD_TRIG.get("pooler") else 0)
    mk = _VFR["marks"] if VF_GDM else _VFR["nomk"]
    r["mod"].gdr_run(vf, r["w"], r["esc"], x, y, T, r["K"], r["N"], r["tb"], r["nbx"], pdl,
                     z if vf == 2 else _VFR["nomk"], state["router"], part, mk)
    if (vf, T) not in _VFR["seen"]:
        _VFR["seen"].add((vf, T))
        print(f"[vfr] first call vf {vf} T={T} pdl {pdl} marks {VF_GDM}", flush=True)
    return y, part


def _vfr_fused(state, ffn_input, router_input):
    if not (VF_GDR and _VFR["ok"]) or router_input is None or router_input.dtype != torch.float32:
        return None
    if not (2 <= ffn_input.shape[0] <= 8 and router_input.is_contiguous() and ffn_input.is_contiguous()):
        return None
    return _vfr_gdr(state, ffn_input, router_input, 2)


def _vfr_marked_pooler(state, x):
    if not (VF_GDM and _VFR["ok"]) or not 2 <= x.shape[0] <= 8 or not x.is_contiguous():
        return None
    out = _vfr_gdr(state, x, None, 1)
    return None if out is None else out[0]


def _vfr_prepare(state):
    _VFR["ok"] = False
    st = state.get("gd")
    r = st["route"].get(("pooler", 8)) if st is not None else None
    if r is None or not r["desc"].startswith("m1(1,8,16,1)s2x1") or state.get("v8") is None or _SBB1.get("mod") is None:
        print(f"[vfr] FALLBACK: GD pooler T=8 route {None if r is None else r['desc']} is not (1,8,16,1)s2x1 -> side-stream router32",
              flush=True)
        return
    dev = state["router"].device
    try:
        i1, i2 = r["mod"].gdr_info(1), r["mod"].gdr_info(2)
        print(f"[vfr] instances regs / local / occ / smem: vf1 {list(i1)} vf2 {list(i2)}", flush=True)
        if i2[2] < 2 or i2[1] > 64 or i1[2] < 2:
            print("[vfr] FALLBACK: below 2 CTAs/SM or spills -> side-stream router32", flush=True)
            return
        _VFR["marks"] = torch.zeros((4 * r["nbx"],), dtype=torch.int64, device=dev)
        _VFR["nomk"] = torch.empty((0,), dtype=torch.int64, device=dev)
        gen = torch.Generator(device=dev).manual_seed(8282)
        ok = True
        for m in (8, 5, 2):
            z = torch.randn((m, HIDDEN), generator=gen, device=dev)
            z = (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True))).contiguous()
            h = (z * 0.7).to(torch.bfloat16).contiguous()
            y0 = _gd(state, "pooler", h)
            p0 = torch.full((8, m, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
            state["v8"]["mod"].v8_router32(z, state["router"], p0)
            y1, p1 = _vfr_gdr(state, h, z, 2)
            y2, _ = _vfr_gdr(state, h, None, 1)
            torch.cuda.synchronize()
            e = (bool(torch.equal(y0, y1)), bool(torch.equal(p0, p1)), bool(torch.equal(y0, y2)))
            print(f"[vfr] self-test m={m}: fused pooler bitwise {e[0]}, router partials bitwise vs router32 {e[1]}, "
                  f"marked pooler bitwise {e[2]}", flush=True)
            ok &= all(e)
        _VFR["seen"].clear()
        _VFR["ok"] = ok
    except Exception as ex:
        print(f"[vfr] FALLBACK: {ex!r} -> side-stream router32", flush=True)
        _VFR["ok"] = False
    print(f"[vfr] fused T=8 pooler + router rows {'LIVE' if _VFR['ok'] else 'OFF'} (VF_GDR {VF_GDR})", flush=True)
'''


def apply(t, on):
    # CUDA: Args fields, template parameter, helpers, pre / post wait, router tail, host
    t = sub(t, "  int trig;                // 1: griddepcontrol.launch_dependents at entry (host: one-wave grids only)\n};\n",
            "  int trig;                // 1: griddepcontrol.launch_dependents at entry (host: one-wave grids only)\n"
            "  const float* x32; const float* w32; float* part; long long* marks;   // VF (vfr_patches): router rows / marks\n};\n",
            what="Args")
    head = ("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP>\n"
            "__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(const __grid_constant__ Args A) {\n")
    t = sub(t, head, DEV_HELPERS + head.replace("int XP>", "int XP, int VF = 0>"), what="kernel head")
    t = sub(t, "  pdl_wait();\n  const uint4* xb = reinterpret_cast<const uint4*>(A.x + (size_t)u0 * 64 + 16 * t);\n",
            PRE_WAIT + "  pdl_wait();\n" + POST_WAIT + "  const uint4* xb = reinterpret_cast<const uint4*>(A.x + (size_t)u0 * 64 + 16 * t);\n",
            what="pdl_wait")
    t = sub(t, "  if constexpr (S == 1) {\n    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n  } else {\n",
            "  if constexpr (S == 1) {\n    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n" + ROUTER_TAIL
            + "  } else {\n", what="store")
    t = sub(t, "  A.trig = (pdl >> 1) & 1;\n  switch (inst) {\n",
            "  A.trig = (pdl >> 1) & 1;\n  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;\n  switch (inst) {\n",
            what="gd_run_raw")
    t = sub(t, "int gd_info_raw(int64_t inst, int64_t* res) {\n", HOST_CU.replace("@GEO@", GEO), what="host")
    t = sub(t, "std::vector<int64_t> gd_info(int64_t inst) {\n", HOST_CPP + "std::vector<int64_t> gd_info(int64_t inst) {\n",
            what="cpp")
    t = sub(t, 'functions=["gd_run", "gd_info", "gd_unpack"],', 'functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info"],',
            what="gd functions")
    # Python: switches + helpers, prepare, T = 8 call site, marked stock pooler
    t = sub(t, "\n\ndef _gd_bucket(T):\n", PY.replace("@ON@", str(bool(on))) + "\n\ndef _gd_bucket(T):\n", what="py block")
    t = sub(t, "    _gd_prepare(state, t0)\n", "    _gd_prepare(state, t0)\n    _vfr_prepare(state)\n", what="prepare")
    t = sub(t, '        expert_x, topk_weights, topk_ids, local_ids = _v8_tail(y32, None, state, data)\n    else:\n        if place == "main":\n',
            '        expert_x, topk_weights, topk_ids, local_ids = _v8_tail(y32, None, state, data)\n    else:\n'
            '        vgr = _vfr_fused(state, ffn_input, router_input) if (tail is not None and mode == "f32") else None\n'
            '        if vgr is not None:\n            expert_input, logits = vgr\n        elif place == "main":\n', what="kernel_fn T=8")
    t = sub(t, 'def _v13_pooler(ffn_input, dense, state):\n    y = _gd(state, "pooler", ffn_input)\n',
            'def _v13_pooler(ffn_input, dense, state):\n    y = _vfr_marked_pooler(state, ffn_input) if VF_GDM else None\n'
            '    if y is None:\n        y = _gd(state, "pooler", ffn_input)\n', what="_v13_pooler")
    return t


def main():
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    if "n" not in ARGS:
        raise SystemExit("[vfr_patches] --n=<new number> is required")
    n = ARGS["n"]
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_slim_5')}.py"
    on = "--on" in sys.argv
    labonly = "--lab-only" in sys.argv
    out = ROOT / f"kernels/muse_decode_layer/k3_vf_{n}.py"
    lab = ROOT / f"kernels/lab/k3_vf_{n}_ab.py"
    for f in ([lab] if labonly else [out, lab]):
        if f.exists() and "--force" not in sys.argv:
            raise SystemExit(f"[vfr_patches] {f.relative_to(ROOT).as_posix()} exists: pick a new --n")
    tag = f"k3-vf-{n}"
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    GM.check(fb == b, f"{base.name} is not a fixed point of finish()")
    nonce = (f"# nonce {tag} ({base.stem} + VF S2: T = 8 router rows inside the GD pooler grid (one wave, bitwise router32)"
             f"{' on' if on else ', switch off'}; per-CTA marks; self-tested with fallback)")
    t = apply(b.replace(bn, nonce), on)
    s = GM.finish(t, nonce, tag, chk_base, rep)
    for r in rep:
        print(f"[vfr_patches] OK  {r}")
    if not labonly:
        out.write_text(s, encoding="utf-8", newline=NL)
    lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vfr_patches] wrote {'' if labonly else out.relative_to(ROOT).as_posix() + ' and '}{lab.relative_to(ROOT).as_posix()} "
          f"({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""K2A v5 fused small-T MoE lab, step 1 (one launch): route -> W13 phase -> in-kernel grid sync -> W2 phase -> grid sync ->
post-norm + weighted sum, all in one persistent kernel `fz_kernel<NCH, NST>` (grid = SMs x 2, all CTAs resident, 320 threads).
LAB FILE: box runs only (tools/box/fused_v5.py). Never submit it.

Reuses v3.7's HK3 code unchanged (kernels/muse_moe_w4a16/k2a_v37.py via kernels/lab/k2a_fb_lab.py's marked source): gemm37 =
static Stream-K segments + a dynamic ticket queue of D-stage chunks (P.probe dyn bits: K1 / K2 pool fractions f1 / f2 in 64ths,
chunk 2^log2D stages; D = 16 = whole items, so a full pool has almost no split items), NC1 cold items + hc_item<2..4> hot items,
mbarrier mailbox epilogue. W13 -> W2 and W2 -> post-norm dependencies: fz_sync = __syncthreads + thread 0 fence + atomicAdd on
a monotonic counter (P.sink[2], target = next multiple of the grid) + bounded ld.acquire poll (error flag err[0] / err[3], no
trap) + __syncthreads + fence.proxy.async (K1's generic stores -> W2's bulk copies). Ring / mailbox barriers are re-initialised
between the phases (setup_bars37 after a CTA-wide sync). The post-norm body is k3_kernel's loop (256 working threads of 320).
Run3 vids: 40 = fz<1, 3> (cold only: bs <= 8, hot_t 999), 45 = fz<4, 3> (HK3 NCH 4: bs 16 / 32).
"""
import hashlib
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hot_p1_lab as H  # noqa: E402
import k2a_fb_lab as FB  # noqa: E402

FZ = r"""
// ======== k2a_v5 fused (one launch) ========
__device__ __forceinline__ void fz_sync(const Params& P) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned t = atomicAdd(P.sink + 2, 1u);
    const unsigned target = (t / gridDim.x + 1u) * gridDim.x;
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(P.sink + 2) : "memory");
    if ((int)(v - target) < 0) {
      const long long t0 = clock64();
      while (true) {
        __nanosleep(32);
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(P.sink + 2) : "memory");
        if ((int)(v - target) >= 0) break;
        if (clock64() - t0 > WAIT_BUDGET) { atomicExch(P.err, 1); atomicExch(P.err + 3, 1); break; }
      }
    }
  }
  __syncthreads();
  asm volatile("fence.proxy.async;\n" ::: "memory");
}
// k3_kernel's post-norm + weighted sum loop for a 320-thread CTA (threads >= 256 only take part in the syncs)
__device__ __forceinline__ void fz_post(const Params& P, int* s_fc, float* s_fr) {
  const int tid = threadIdx.x;
  const int items = 4 * P.M;
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int tok = it >> 2;
    if (tid < 8) {
      const int c = __ldcg(P.meta + MH + tok * 8 + tid);
      const float wt = __ldg(P.wts + tok * 8 + tid);
      s_fc[tid] = c;
      s_fr[tid] = (c >= 0) ? rsqrtf(__ldcg(P.ss + c) * (1.f / 4096.f) + 1e-8f) * wt : 0.f;
    }
    __syncthreads();
    if (tid < 256) {
      const int col = (it & 3) * 1024 + 4 * tid;
      float4 v[8];
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int c = s_fc[s];
        v[s] = (c >= 0) ? __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)c * KDIM + col)) : z4;
      }
      const float4 w = __ldg(reinterpret_cast<const float4*>(P.nw + col));
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const float r = s_fr[s];
        a0 += bf16r(v[s].x) * w.x * r; a1 += bf16r(v[s].y) * w.y * r;
        a2 += bf16r(v[s].z) * w.z * r; a3 += bf16r(v[s].w) * w.w * r;
      }
      uint2 o;
      o.x = pack_bf2(a0, a1); o.y = pack_bf2(a2, a3);
      *reinterpret_cast<uint2*>(P.out + (size_t)tok * KDIM + col) = o;
    }
    __syncthreads();
  }
}
template <int NCH, int NST>
__global__ void __launch_bounds__(320, 2) fz_kernel(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  __shared__ int s_fc[8];
  __shared__ float s_fr[8];
  hp_mark(P, 0, 0, 2);
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * (SW + 8 * NCH * XROW));
  setup_bars37<NST>(bars);
  route<1, 8 * NCH, true>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);
  hp_mark(P, 0, 1, 2);
  gemm37<NCH, NST, 1>(P, S, s_list, smem, bars, false);
  hp_mark(P, 0, 2, 2);
  fz_sync(P);
  setup_bars37<NST>(bars);
  route<1, 8 * NCH, true, false>(P, S, nullptr, nullptr, 32, false);
  hp_mark(P, 1, 1, 2);
  gemm37<NCH, NST, 2>(P, S, nullptr, smem, bars, false);
  hp_mark(P, 1, 2, 2);
  fz_sync(P);
  if (threadIdx.x == 0) fb_ev(P, 2, 3);
  fz_post(P, s_fc, s_fr);
  if (threadIdx.x == 0) fb_ev(P, 2, 5);
}
"""

HOST = r"""
template <int NCH, int NST>
static void launchfz(const k3::Params& p, int G) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  constexpr int SM = smem_bytes37<NCH, NST>();
  k3::fz_kernel<NCH, NST><<<G, 320, SM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int NCH, int NST>
static std::vector<int64_t> attrsfz() {
  constexpr int SM = smem_bytes37<NCH, NST>();
  const void* f = (const void*)k3::fz_kernel<NCH, NST>;
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
  cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncAttributes fa;
  int occ = 0;
  if (cudaFuncGetAttributes(&fa, f) != cudaSuccess) return {-1};
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, 320, SM) != cudaSuccess) occ = -1;
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + SM), (int64_t)occ};
}
"""

VIDS = {0: (1, 3, 2, 0), 3: (1, 3, 2, 2, 4)}


FX_HELPERS = r"""
// v5 step 2 (FZX): per-(expert, pass) W13 readiness in P.xw (int view; HK3 never uses xw): [e * 64 + j] = W13 items done
// (target 64 row blocks), [2048] = consumer warps done zeroing y / ss (target 8 x grid). Zeroed by CTA 0 after the last sync.
__device__ __forceinline__ int fx_ld(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fx_poll(const int* p, int target, int* err) {
  if (fx_ld(p) >= target) return;
  const long long t0 = clock64();
  while (fx_ld(p) < target) {
    __nanosleep(64);
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 3, 1); return; }
  }
}
__device__ __forceinline__ void fx_ready(const Params& P, int key, int& rkey, bool& zok) {   // W2 producer lane 0
  const int* rdy = reinterpret_cast<const int*>(P.xw);
  if (!zok) { fx_poll(rdy + 2048, 8 * (int)gridDim.x, P.err); zok = true; }
  if (key != rkey) {
    fx_poll(rdy + key, 64, P.err);
    asm volatile("fence.proxy.async;\n" ::: "memory");
    rkey = key;
  }
}
"""

# (region, old, new) applied to the gemm37 source in every copy; gated by `if constexpr (FZX)` / FZX != 0
G37 = ("__device__ void gemm37(", "// K1 / K2 of the HK3 variants")
FX_PATCHES = [
    (None, "constexpr int HP_LAB = @HPLAB@;\n", "constexpr int HP_LAB = @HPLAB@;\nconstexpr int FZX = @FZX@;\n"),
    (None, "// v3.3 warm-expert item (all of its stages s0..s1)", FX_HELPERS + "// v3.3 warm-expert item (all of its stages s0..s1)"),
    (G37, "int* dq = P.meta + DQ37 + ((MODE == 1) ? 0 : 2);\n",
     "int* dq = P.meta + DQ37 + ((MODE == 1) ? 0 : 2);\n    int fx_rkey = -1;\n    bool fx_zok = false;\n"),
    (G37, "if (lane == 0) bulk_g2s(dst + SW, src, xbytes, fb);",
     "if (lane == 0) {\n                if constexpr (FZX != 0) fx_ready(P, e * 64 + j, fx_rkey, fx_zok);\n"
     "                bulk_g2s(dst + SW, src, xbytes, fb);\n              }"),
    (G37, "    for (int i = blockIdx.x * 256 + tid; i < S.R; i += gridDim.x * 256) P.ss[i] = 0.f;\n  }\n",
     "    for (int i = blockIdx.x * 256 + tid; i < S.R; i += gridDim.x * 256) P.ss[i] = 0.f;\n  }\n"
     "  if constexpr (FZX != 0) {\n    if (MODE == 1) {\n      __syncwarp();\n"
     "      if (lane == 0) { __threadfence(); atomicAdd(reinterpret_cast<int*>(P.xw) + 2048, 1); }\n    }\n  }\n"),
    (G37, "if (!(s0 == 0 && s1 == 16) || ho) post(e, rb, j, s1 - s0, ntok, c0);",
     "if (!(s0 == 0 && s1 == 16) || ho) post(e, rb, j, s1 - s0, ntok, c0);\n"
     "        else if (FZX != 0 && MODE == 1) post(e, rb, j, 32, ntok, c0);   // FZX: whole item -> readiness via the epilogue"),
    (G37, "if (!(t0 == 0 && t1 == 16) || ho2) post(e, rb2, j2, t1 - t0, nt2, S.off[e] + j2 * tp2);",
     "if (!(t0 == 0 && t1 == 16) || ho2) post(e, rb2, j2, t1 - t0, nt2, S.off[e] + j2 * tp2);\n"
     "        else if (FZX != 0 && MODE == 1) post(e, rb2, j2, 32, nt2, S.off[e] + j2 * tp2);"),
    (G37, "      if (e < 0) break;\n      __threadfence();   // cumulative over the consumers' h / y reds (ordered before us by the mailbox barrier)\n",
     "      if (e < 0) break;\n      __threadfence();   // cumulative over the consumers' h / y reds (ordered before us by the mailbox barrier)\n"
     "      if (FZX != 0 && MODE == 1 && ns == 32) {   // whole item flushed by all 8 consumer warps: W13 item done\n"
     "        if (lane == 0) atomicAdd(reinterpret_cast<int*>(P.xw) + e * 64 + j, 1);\n        continue;\n      }\n"),
    (G37, "              *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);\n            }\n          }\n        }\n      }\n",
     "              *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);\n            }\n          }\n        }\n      }\n"
     "      if (FZX != 0 && MODE == 1 && last) {   // split / hand-off item finalized: W13 item done\n"
     "        __syncwarp();\n        if (lane == 0) { __threadfence(); atomicAdd(reinterpret_cast<int*>(P.xw) + e * 64 + j, 1); }\n      }\n"),
    (None, "  fz_sync(P);\n  setup_bars37<NST>(bars);\n",
     "  if constexpr (FZX == 0) fz_sync(P);\n  else __syncthreads();   // FZX: no grid sync, W2 units gated per expert\n"
     "  setup_bars37<NST>(bars);\n"),
    (None, "  fz_sync(P);\n  if (threadIdx.x == 0) fb_ev(P, 2, 3);\n",
     "  fz_sync(P);\n  if constexpr (FZX != 0) {\n    if (blockIdx.x == 0)\n"
     "      for (int i = threadIdx.x; i <= 2048; i += blockDim.x) reinterpret_cast<int*>(P.xw)[i] = 0;\n  }\n"
     "  if (threadIdx.x == 0) fb_ev(P, 2, 3);\n"),
]


def lab_source():
    src = FB.patched_source()
    src = FB._sub_once(src, None, "}  // namespace k3\n", FZ + "}  // namespace k3\n", "fz kernel")
    src = FB._sub_once(src, None, "void run3(int64_t vid,", HOST + "\nvoid run3(int64_t vid,", "fz host")
    for i, (reg, old, new) in enumerate(FX_PATCHES):
        src = FB._sub_once(src, reg, old, new, f"fx patch {i}")
    k = src.index("namespace k3 {")
    head, body = src[:k], src[k:]
    cs = [FB._case(v) for v in (0, 3)]
    cases = "\n".join(c[0] for c in cs) + "\n    case 40: launchfz<1, 3>(p, (int)G); break;\n    case 45: launchfz<4, 3>(p, (int)G); break;"
    attrs = "\n".join(c[1] for c in cs) + "\n    case 40: return attrsfz<1, 3>();\n    case 45: return attrsfz<4, 3>();"
    parts = [head]
    for ns, fzx in (("fzL", 0), ("fxL", 1)):
        copy = (body.replace("@HPLAB@", "0").replace("@FZX@", str(fzx)).replace("@CASES3@", cases)
                .replace("@ATTRS3@", attrs))
        parts.append(f"\nnamespace {ns} {{\n{copy}\n}}  // namespace {ns}\n")
    cuda = ("".join(parts) + f"\nvoid fzrun({H.RUN_ARGS}) {{\n  fzL::run3({H.RUN_NAMES});\n}}\n"
            f"void fxrun({H.RUN_ARGS}) {{\n  fxL::run3({H.RUN_NAMES});\n}}\n"
            "std::vector<int64_t> fzattrs(int64_t vid) { return fzL::attrs3v(vid); }\n"
            "std::vector<int64_t> fxattrs(int64_t vid) { return fxL::attrs3v(vid); }\n")
    cpp = ("#include <torch/extension.h>\n#include <vector>\n"
           f"void fzrun({H.RUN_ARGS});\nvoid fxrun({H.RUN_ARGS});\nstd::vector<int64_t> fzattrs(int64_t vid);\n"
           "std::vector<int64_t> fxattrs(int64_t vid);\n")
    return cuda, cpp


def build(name="k2a_v5_fused_lab", verbose=False):
    from torch.utils.cpp_extension import load_inline
    cuda, cpp = lab_source()
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=cuda, functions=["fzrun", "fxrun", "fzattrs", "fxattrs"],
                      extra_cuda_cflags=H.CFLAGS, verbose=verbose)
    return mod, time.time() - t0


if __name__ == "__main__":
    cu, cp = lab_source()
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "k2a_v5_fused_gen.cu")
    out.write_text(cu, encoding="utf-8")
    print(f"wrote {out} ({len(cu)} chars)")

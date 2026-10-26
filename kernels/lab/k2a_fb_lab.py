"""Fused-MoE bound lab (phase 1): per-kernel timeline marks on K2A's small-T shipped paths (vid 0 = v3.3 NC1, bs 1/8; vid 3 =
v3.6 HK2 NCH 4, bs 16/32) plus a one-launch pure weight-stream probe. LAB FILE: box runs only (tools/box/fused_bound.py).

Source: hot_p1_lab's patched v3.7 (kernels/lab/hot_p1_lab.py: Params.hp / hpclk, hp_mark at K1 / K2 entry, gemm start / end for
the v3.3 kernels) with:
  * records widened to 16 longs per (kernel k, CTA): [0 smid, 1 role, 2 clk start, 3 gt gemm start, 4 clk end, 5 gt gemm end
    (thread 0 = consumer warp 0), 6 gt kernel entry, 7 gt first weight copy issued (producer lane 0), 8 gt producer back from
    griddepcontrol.wait (K2: the K1 -> K2 hand-off), ...]; k 0 = K1, 1 = K2, 2 = K3 (3 entry 6, after its wait 3, end 5).
  * the same marks in k1_kernel36 / k2_kernel36 (vid 3).
  * fbstream<NST>: one launch, grid G x 320 threads, streams the touched experts' stages expert-major (W13 then W2 of each
    expert, 1536 stages of 18,432 B each) through an NST-slot cp.async.bulk ring (L2 evict_first, as shipped), 8 null consumer
    warps (wait full, release empty): the byte floor of a one-launch fused MoE with the same ring. Bounded waits (R9).
"""
import hashlib
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hot_p1_lab as H  # noqa: E402

REC = 16
VIDS = {0: (1, 3, 2, 0), 3: (1, 3, 2, 2, 4)}

FB_HELPERS = r"""
// ======== fused-bound lab helpers (kernels/lab/k2a_fb_lab.py) ========
__device__ __forceinline__ void fb_ev(const Params& P, int k, int i) {
  if (P.hpclk == nullptr || blockIdx.x >= 1024) return;
  P.hpclk[((size_t)k * 1024 + blockIdx.x) * 16 + i] = hp_gt();
}
"""

STREAM = r"""
// one-launch pure weight stream: expert-major (W13 stages then W2 stages of each listed expert), null consumers
template <int NST>
__global__ void __launch_bounds__(320, 2) fbstream(const uint8_t* ws13, const uint8_t* ws2, const int* elist, int ne,
                                                    int* err, long long* marks) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * SW);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0 && marks != nullptr) marks[(size_t)blockIdx.x * 4] = hp_gt();
  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(sa(bars + i), 1); mbar_init(sa(bars + NST + i), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const long long S = (long long)ne * 1536;
  const int beg = (int)(S * blockIdx.x / gridDim.x), end = (int)(S * (blockIdx.x + 1) / gridDim.x);
  const uint32_t full_a = sa(bars), empty_a = sa(bars + NST), ring_a = sa(smem);
  if (warp == 8) {
    const uint64_t pol = policy_evict_first();
    int it = 0;
    for (int x = beg; x < end; ++x, ++it) {
      const int slot = it % NST;
      mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, err);
      if (lane == 0) {
        const int e = elist[x / 1536], l = x % 1536;
        const uint8_t* src = (l < 1024) ? ws13 + ((size_t)e * 1024 + l) * SW : ws2 + ((size_t)e * 512 + (l - 1024)) * SW;
        if (it == 0 && marks != nullptr) marks[(size_t)blockIdx.x * 4 + 1] = hp_gt();
        mbar_expect_tx(full_a + 8 * slot, SW);
        bulk_g2s_ef(ring_a + slot * SW, src, SW, full_a + 8 * slot, pol);
      }
      __syncwarp();
    }
  } else if (warp < 8) {
    int it = 0;
    for (int x = beg; x < end; ++x, ++it) {
      const int slot = it % NST;
      mbar_wait(full_a + 8 * slot, (it / NST) & 1, err);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * slot);
    }
    if (tid == 0 && marks != nullptr) marks[(size_t)blockIdx.x * 4 + 2] = hp_gt();
  }
}
"""

STREAM_HOST = r"""
void fbstream_run(torch::Tensor ws13, torch::Tensor ws2, torch::Tensor elist, int64_t ne, torch::Tensor err,
                  torch::Tensor marks, int64_t G, int64_t nst) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  long long* mk = marks.numel() >= G * 4 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  const uint8_t* a = reinterpret_cast<const uint8_t*>(ws13.data_ptr());
  const uint8_t* b = reinterpret_cast<const uint8_t*>(ws2.data_ptr());
  if (nst == 3) {
    constexpr int SM = 3 * k3::SW + 2 * 3 * 8;
    cudaFuncSetAttribute(fbL0::k3::fbstream<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    fbL0::k3::fbstream<3><<<(int)G, 320, SM, st>>>(a, b, elist.data_ptr<int>(), (int)ne, err.data_ptr<int>(), mk);
  } else {
    constexpr int SM = 5 * k3::SW + 2 * 5 * 8;
    cudaFuncSetAttribute(fbL0::k3::fbstream<5>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    fbL0::k3::fbstream<5><<<(int)G, 320, SM, st>>>(a, b, elist.data_ptr<int>(), (int)ne, err.data_ptr<int>(), mk);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _sub_once(src, region, old, new, name):
    a, b = H._region(src, *region) if region else (0, len(src))
    n = src[a:b].count(old)
    if n != 1:
        raise RuntimeError(f"k2a_fb {name}: anchor found {n} times")
    i = src.index(old, a)
    return src[:i] + new + src[i + len(old):]


def patched_source():
    src = H.patched_source()
    src = _sub_once(src, None, "long long* h = P.hpclk + ((size_t)k * 1024 + blockIdx.x) * 8;",
                    "long long* h = P.hpclk + ((size_t)k * 1024 + blockIdx.x) * 16;", "record stride")
    src = _sub_once(src, None, "constexpr int DQ37 = 120;\n", FB_HELPERS + "constexpr int DQ37 = 120;\n", "helpers")
    src = _sub_once(src, None, "// v3.3 warm-expert item (all of its stages s0..s1)",
                    "__device__ __forceinline__ void hp_mark(const Params& P, int k, int slot, int role);\n"
                    "__device__ __forceinline__ void fb_ev(const Params& P, int k, int i);\n"
                    "// v3.3 warm-expert item (all of its stages s0..s1)", "forward decls")
    for reg in (("__device__ void gemm(const Params& P, Route& S,", "__device__ __forceinline__ void setup_bars("),
                ("__device__ void gemm36(", "k1_kernel36(const Params P) {")):
        a, b = H._region(src, *reg)
        seg = src[a:b]
        seg2, n1 = re.subn(r"(\n(\s*))if \(hint\) bulk_g2s_ef\(dst, src, SW, fb, pol\);",
                           r"\1if (it == 0) fb_ev(P, MODE - 1, 7);\1if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);", seg)
        seg3, n2 = re.subn(r"(\n(\s*))pdl_wait\(\);\n", r"\1pdl_wait();\1if (lane == 0) fb_ev(P, MODE - 1, 8);\n", seg2)
        if n1 != 1 or n2 < 1:
            raise RuntimeError(f"k2a_fb producer marks in {reg[0]!r}: {n1} copy / {n2} wait anchors")
        src = src[:a] + seg3 + src[b:]
    # vid 3 kernels: entry / gemm start / gemm end marks (role 3 = HK2 mixed)
    for k, kern, nxt, call in ((0, "k1_kernel36(const Params P) {", "k2_kernel36(const Params P) {",
                                "  gemm36<NCH, NST, 1>(P, S, s_list, smem, bars, false, s_pr);\n"),
                               (1, "k2_kernel36(const Params P) {", "// ====",
                                "  gemm36<NCH, NST, 2>(P, S, s_list, smem, bars, true, s_pr);\n")):
        reg = (kern, nxt)
        src = _sub_once(src, reg, "  pdl_launch();\n", f"  pdl_launch();\n  hp_mark(P, {k}, 0, 3);\n", f"{kern} entry")
        src = _sub_once(src, reg, call, f"  hp_mark(P, {k}, 1, 3);\n" + call + f"  hp_mark(P, {k}, 2, 3);\n", f"{kern} gemm")
    # K3: entry, after its griddepcontrol.wait, end
    k3a = "k3_kernel(const Params P) {\n  __shared__ int s_fc[8];\n  __shared__ float s_fr[8];\n  pdl_wait();\n"
    src = _sub_once(src, None, k3a,
                    "k3_kernel(const Params P) {\n  __shared__ int s_fc[8];\n  __shared__ float s_fr[8];\n"
                    "  if (threadIdx.x == 0) fb_ev(P, 2, 6);\n  pdl_wait();\n  if (threadIdx.x == 0) fb_ev(P, 2, 3);\n", "k3 entry")
    k3e = "    *reinterpret_cast<uint2*>(P.out + (size_t)tok * KDIM + col) = o;\n    __syncthreads();\n  }\n}\n"
    src = _sub_once(src, None, k3e, k3e[:-2] + "  if (threadIdx.x == 0) fb_ev(P, 2, 5);\n}\n", "k3 end")
    src = _sub_once(src, None, "}  // namespace k3\n", STREAM + "}  // namespace k3\n", "stream kernel")
    return src


def _case(vid):
    nc, nst, minb, hk, nch = (tuple(VIDS[vid]) + (0,) * 5)[:5]
    if hk == 2:
        return (f"    case {vid}: launch36<{nch}, {nst}>(p, (int)G, (int)G3, (int)mask); break;",
                f"    case {vid}: return attrs36<{nch}, {nst}>();")
    return (f"    case {vid}: launch3<{nc}, {nst}, {minb}, {hk}>(p, (int)G, (int)G3, (int)mask); break;",
            f"    case {vid}: return attrs3<{nc}, {nst}, {minb}, {hk}>();")


def lab_source():
    src = patched_source()
    k = src.index("namespace k3 {")
    head, body = src[:k], src[k:]
    cs = [_case(v) for v in VIDS]
    copy = (body.replace("@HPLAB@", "0").replace("@CASES3@", "\n".join(c[0] for c in cs))
            .replace("@ATTRS3@", "\n".join(c[1] for c in cs)))
    cuda = (head + f"\nnamespace fbL0 {{\n{copy}\n}}  // namespace fbL0\n"
            f"\nvoid fbrun({H.RUN_ARGS}) {{\n  fbL0::run3({H.RUN_NAMES});\n}}\n"
            f"std::vector<int64_t> fbattrs(int64_t vid) {{ return fbL0::attrs3v(vid); }}\n"
            + STREAM_HOST.replace("k3::SW", "fbL0::k3::SW"))
    cpp = ("#include <torch/extension.h>\n#include <vector>\n"
           f"void fbrun({H.RUN_ARGS});\nstd::vector<int64_t> fbattrs(int64_t vid);\n"
           "void fbstream_run(torch::Tensor ws13, torch::Tensor ws2, torch::Tensor elist, int64_t ne, torch::Tensor err, "
           "torch::Tensor marks, int64_t G, int64_t nst);\n")
    return cuda, cpp


def build(name="k2a_fb_lab", verbose=False):
    from torch.utils.cpp_extension import load_inline
    cuda, cpp = lab_source()
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=cuda,
                      functions=["fbrun", "fbattrs", "fbstream_run"], extra_cuda_cflags=H.CFLAGS, verbose=verbose)
    return mod, time.time() - t0


if __name__ == "__main__":
    cu, cp = lab_source()
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "k2a_fb_lab_gen.cu")
    out.write_text(cu, encoding="utf-8")
    print(f"wrote {out} ({len(cu)} chars)")

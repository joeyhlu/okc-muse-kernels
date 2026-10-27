"""R1 lab (notes/memory/gpu_requests.md R1): how much of K2A's K1/K2 time above the pure weight stream is X staging?

LAB FILE. Box runs only (tools/box/r1_xfix.py). Never submit it, never import it from a scored or layer file: the X-fixed
arm computes wrong numbers by design.

Source: kernels/lab/k2a_fb_lab.py's patched K2A v3.7 (hot_p1 masks + per-CTA marks), one namespace copy `x1L0` with
HP_LAB = 0 and SMT = 0, so with hp = 0 it is fb4's "base" (shipped kernels + marks). Built for vid 3 (v3.6 HK2 NCH 4,
T = 9..32) and vid 5 (v3.7 HK3 NCH 4, T = 128) only.

New runtime bit HP_XFIX = 32 (Params.hp, run3's `hp` argument), producer + one fill only; weights, schedule (route, segs,
stream-K ranges, tickets, tail-first order), ring, consumers, flush, epilogue and PDL chain are unchanged:
  * K1 (MODE 1): no X row copies (the ntok per-lane 512 B bulk copies from expert_x); expect_tx = the weight stage only.
  * K2 (MODE 2): the per-stage X copy shrinks from ntok x 528 B of interS to 16 B from interS[0] (a fixed resident tile),
    so the deferral bookkeeping and the griddepcontrol.wait point of the shipped producer (take the wait when a deferred
    slot is about to be reused) are kept exactly: K2 consumers still cannot consume past NST stages before K1 is done.
  * Every ring slot's X region is filled once, before the CTA's gemm, with one fixed finite bf16 tile (2^-7 per element),
    so the consumers' X LDS, mma and flush run on sane values (no IEEE-division slow paths from stale smem garbage).
So the X-fixed arm removes X bytes (L2 -> smem), X copy issue and X-copy completion waits, and nothing else.

    python kernels/lab/k2a_x1_lab.py [out.cu]   # write the generated CUDA (+ .cpp) for an offline nvcc
"""
import hashlib
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hot_p1_lab as H  # noqa: E402
import k2a_fb_lab as L  # noqa: E402

HP_XFIX = 32
VIDS = (3, 5)
G36 = ("__device__ void gemm36(", "k1_kernel36(const Params P) {")
G37 = ("__device__ void gemm37(", "// K1 / K2 of the HK3 variants")

FILL = r"""
// R1 lab (HP_XFIX): one fixed finite bf16 tile (0x3C00 = 2^-7) in every ring slot's X region, written once before gemm
template <int STG, int NST>
__device__ __forceinline__ void xfix_fill(uint8_t* ring) {
  constexpr int XQ = (STG - SW) / 16;
  static_assert((STG - SW) % 16 == 0, "X region must be a multiple of 16 B");
  const uint4 v = make_uint4(0x3C003C00u, 0x3C003C00u, 0x3C003C00u, 0x3C003C00u);
  for (int i = threadIdx.x; i < NST * XQ; i += blockDim.x) {
    const int s = i / XQ, q = i - s * XQ;
    *reinterpret_cast<uint4*>(ring + (size_t)s * STG + SW + 16 * q) = v;
  }
}

"""

# (name, region or None, old, new): `old` must occur exactly once in its region
PATCHES = [
    ("constant", None,
     "constexpr int HP_XHOT = 1, HP_XCOLD = 2, HP_RPW = 4, HP_NOWAIT2 = 8, HP_NOP0 = 16;\n",
     "constexpr int HP_XHOT = 1, HP_XCOLD = 2, HP_RPW = 4, HP_NOWAIT2 = 8, HP_NOP0 = 16;\n"
     "constexpr int HP_XFIX = 32;   // R1 lab: no real X staging (kernels/lab/k2a_x1_lab.py)\n"),
    ("fill helper", None,
     "template <int NCH, int NST, int MODE>\n__device__ void gemm36(",
     FILL + "template <int NCH, int NST, int MODE>\n__device__ void gemm36("),
    # ---------------- v3.6 gemm36 producer (vid 3) ----------------
    ("36 xfix flag", G36,
     "    const bool hint = (P.probe & 16) != 0;\n",
     "    const bool hint = (P.probe & 16) != 0;\n"
     "    const bool xfix = (P.hp & HP_XFIX) != 0;   // R1 lab\n"),
    ("36 xbytes", G36,
     "          const uint32_t xbytes = (MODE == 1) ? ntok * 512 : ntok * XROW;\n",
     "          const uint32_t xbytes = xfix ? ((MODE == 1) ? 0u : 16u)\n"
     "                                       : (uint32_t)((MODE == 1) ? ntok * 512 : ntok * XROW);\n"),
    ("36 K1 X copy", G36,
     "              if (r < ntok) bulk_g2s(dst + SW + r * XROW, P.x + (size_t)tok * KDIM + s * 256, 512, fb);\n",
     "              if (!xfix && r < ntok) bulk_g2s(dst + SW + r * XROW, P.x + (size_t)tok * KDIM + s * 256, 512, fb);\n"),
    ("36 K2 X source", G36,
     "              const __nv_bfloat16* src = P.inter + ((size_t)s * (8 * M3) + c0) * XST;\n",
     "              const __nv_bfloat16* src = xfix ? P.inter : P.inter + ((size_t)s * (8 * M3) + c0) * XST;\n"),
    ("36 K1 fill", ("k1_kernel36(const Params P) {", "k2_kernel36(const Params P) {"),
     "  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr);\n  __syncthreads();\n",
     "  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr);\n"
     "  if ((P.hp & HP_XFIX) != 0) xfix_fill<STG, NST>(smem);   // R1 lab\n  __syncthreads();\n"),
    ("36 K2 fill", ("k2_kernel36(const Params P) {", "// ===="),
     "  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr);\n  __syncthreads();\n",
     "  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr);\n"
     "  if ((P.hp & HP_XFIX) != 0) xfix_fill<STG, NST>(smem);   // R1 lab\n  __syncthreads();\n"),
    # ---------------- v3.7 gemm37 producer (vid 5); hot_p1's HP_XHOT / HP_XCOLD expression kept underneath ----------------
    ("37 xbytes", G37,
     "        const uint32_t xbytes = (P.hp & (hk ? HP_XHOT : HP_XCOLD)) != 0\n",
     "        const uint32_t xbytes = (P.hp & HP_XFIX) != 0 ? ((MODE == 1) ? 0u : 16u)   // R1 lab\n"
     "            : (P.hp & (hk ? HP_XHOT : HP_XCOLD)) != 0\n"),
    ("37 K2 X source", G37,
     "            const __nv_bfloat16* src = P.inter + ((size_t)s * (8 * M3) + c0) * XST;\n",
     "            const __nv_bfloat16* src = (P.hp & HP_XFIX) != 0 ? P.inter\n"
     "                                       : P.inter + ((size_t)s * (8 * M3) + c0) * XST;   // R1 lab\n"),
    ("37 fill", G37,
     "  if (warp == 0) segs37<NCH>(P, S, (MODE == 1) ? ((P.probe >> 8) & 63) : ((P.probe >> 14) & 63), pr, sd);\n"
     "  __syncthreads();\n",
     "  if (warp == 0) segs37<NCH>(P, S, (MODE == 1) ? ((P.probe >> 8) & 63) : ((P.probe >> 14) & 63), pr, sd);\n"
     "  if ((P.hp & HP_XFIX) != 0) xfix_fill<STG, NST>(ring);   // R1 lab\n"
     "  __syncthreads();\n"),
]


def patched_source(r1=True):
    src = L.patched_source()
    if r1:
        for name, region, old, new in PATCHES:
            src = L._sub_once(src, region, old, new, "R1 " + name)
    return src


def lab_source(r1=True):
    src = patched_source(r1)
    k = src.index("namespace k3 {")
    head, body = src[:k], src[k:]
    cs = [L._case(v) for v in VIDS]
    copy = (body.replace("@HPLAB@", "0").replace("@SMT@", "0").replace("@CASES3@", "\n".join(c[0] for c in cs))
            .replace("@ATTRS3@", "\n".join(c[1] for c in cs)))
    cuda = (head + f"\nnamespace x1L0 {{\n{copy}\n}}  // namespace x1L0\n"
            + f"\nvoid x1run({H.RUN_ARGS}) {{\n  x1L0::run3({H.RUN_NAMES});\n}}\n"
            + "std::vector<int64_t> x1attrs(int64_t vid) { return x1L0::attrs3v(vid); }\n"
            + L.STREAM_HOST.replace("k3::SW", "x1L0::k3::SW").replace("fbL0::", "x1L0::"))
    cpp = ("#include <torch/extension.h>\n#include <vector>\n"
           f"void x1run({H.RUN_ARGS});\nstd::vector<int64_t> x1attrs(int64_t vid);\n"
           "void fbstream_run(torch::Tensor ws13, torch::Tensor ws2, torch::Tensor elist, int64_t ne, torch::Tensor err, "
           "torch::Tensor marks, int64_t G, int64_t nst);\n")
    return cuda, cpp


def build(name="k2a_x1_lab", verbose=False, r1=True, cflags=None):
    from torch.utils.cpp_extension import load_inline
    cuda, cpp = lab_source(r1)
    flags = list(H.CFLAGS if cflags is None else cflags)
    tag = hashlib.sha1((cpp + cuda + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=cuda,
                      functions=["x1run", "x1attrs", "fbstream_run"], extra_cuda_cflags=flags, verbose=verbose)
    return mod, time.time() - t0


if __name__ == "__main__":
    cu, cp = lab_source()
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "k2a_x1_lab_gen.cu")
    out.write_text(cu, encoding="utf-8")
    out.with_suffix(".cpp").write_text(cp, encoding="utf-8")
    print(f"wrote {out} ({len(cu)} chars) and {out.with_suffix('.cpp')}")

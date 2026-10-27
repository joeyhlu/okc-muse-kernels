"""vid 2 marks lab (R6): per-CTA timeline + work marks on K2A's shipped T=256 route (vid 2 = v3.3 HOTCAP, NC 2, NST 2,
2 CTAs/SM, with v4.0 LEAN + v4.2 FLUSH2 / WSK + v4.3 WSK72 = the k3_v35 K2A33 code at T=256). LAB FILE: box runs only
(tools/box/v2_marks.py); never submit it, never import it from a scored file.

Source: kernels/lab/k2a_v40_lab.py's patched v3.7 (hot_p1 marks + the v4.0 lab flags) as ONE copy `lpm` =
LEAN | FLUSH2 | WSK | WSKP (runtime WSK weights A + B rows from rdy[3000..3008); the driver passes the v4.3 weights
(72, 1, 36, 1, 72, 1, 36, 1), which is the lp_hh row v4.3 was derived from), vid 2 only. K2-only role weight: run3 hot_t =
16 + 1000 * w4(K1) + 100000 * w4(K2) (v40's `k2 own weight` patch; w4 >= 50 = fine weight (w4 - 50) / 32, ship 76).

Added here (thread 0 / lane 0 stores only; no new wait, no math change):
  * records widened to 16 longs per (kernel k, CTA), k 0 = K1, 1 = K2, 2 = K3:
      [0 smid, 1 role (1 hot, 0 cold), 2 clk gemm start, 3 gt gemm start, 4 clk gemm end, 5 gt gemm end (consumer warp 0,
       thread 0, after its last flush), 6 gt kernel entry, 7 gt first weight copy issued (producer lane 0),
       8 gt producer back from griddepcontrol.wait (K2: the K1 -> K2 hand-off; K1: P0), 9 hot stages, 10 cold stages,
       11 gt start of the last item, 12 items, 13 sum ntok x stages, 14 sum WSK weight x stages (v4.3 model: hot
       72 + rows rounded to 8, cold 36 + rows), 15 gt epilogue warp exit (split-item finalizes done)]
    K3 (k 2): [6] entry, [3] back from its griddepcontrol.wait (= K2 complete), [5] end.
Waits stay v3.7's bounded mbar_wait (R9).
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
import k2a_v40_lab as V  # noqa: E402

REC = 16
COPY = ("lpm", (0, 0, V.V40_LEAN | V.V40_FLUSH2 | V.V40_WSK | V.V40_WSKP, (2,)))
WSK43 = (72, 1, 36, 1, 72, 1, 36, 1)   # [K1 hot A, B, K1 cold A, B, K2 hot A, B, K2 cold A, B]: v4.3 as shipped

V2_HELPERS = r"""
// ======== vid 2 marks lab helpers (kernels/lab/k2a_v2m_lab.py) ========
__device__ __forceinline__ long long v2_gt() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void v2_ev(const Params& P, int k, int i) {
  if (P.hpclk == nullptr || blockIdx.x >= 1024) return;
  P.hpclk[((size_t)k * 1024 + blockIdx.x) * 16 + i] = v2_gt();
}
// per-CTA work (consumer thread 0, every item of the v3.3 gemm)
__device__ __forceinline__ void v2_item(const Params& P, int k, bool hk, int ntok, int ns) {
  if (P.hpclk == nullptr || blockIdx.x >= 1024 || threadIdx.x != 0) return;
  long long* h = P.hpclk + ((size_t)k * 1024 + blockIdx.x) * 16;
  h[hk ? 9 : 10] += ns;
  h[11] = v2_gt();
  h[12] += 1;
  h[13] += (long long)ntok * ns;
  h[14] += (long long)(hk ? 72 + ((ntok + 7) & ~7) : 36 + ntok) * ns;
}
"""


def _sub_once(src, region, old, new, name):
    a, b = H._region(src, *region) if region else (0, len(src))
    n = src[a:b].count(old)
    if n != 1:
        raise RuntimeError(f"k2a_v2m {name}: anchor found {n} times")
    i = src.index(old, a)
    return src[:i] + new + src[i + len(old):]


def patched_source():
    src = V.patched_source()
    src = _sub_once(src, None, "long long* h = P.hpclk + ((size_t)k * 1024 + blockIdx.x) * 8;",
                    "long long* h = P.hpclk + ((size_t)k * 1024 + blockIdx.x) * 16;", "record stride")
    src = _sub_once(src, None, "// v3.3 warm-expert item (all of its stages s0..s1)",
                    V2_HELPERS + "// v3.3 warm-expert item (all of its stages s0..s1)", "helpers")
    reg = V.GEMM_REGION
    a, b = H._region(src, *reg)
    seg = src[a:b]
    seg, n1 = re.subn(r"(\n(\s*))if \(hint\) bulk_g2s_ef\(dst, src, SW, fb, pol\);",
                      r"\1if (it == 0) v2_ev(P, MODE - 1, 7);\1if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);", seg)
    seg, n2 = re.subn(r"(\n(\s*))pdl_wait\(\);\n", r"\1pdl_wait();\1if (lane == 0) v2_ev(P, MODE - 1, 8);\n", seg)
    if n1 != 1 or n2 != 2:
        raise RuntimeError(f"k2a_v2m producer marks: {n1} copy / {n2} wait anchors (want 1 / 2)")
    src = src[:a] + seg + src[b:]
    src = _sub_once(src, reg, "    V40_ACC(1, vt);\n    return;\n  }\n\n  // ---------------- consumers",
                    "    V40_ACC(1, vt);\n    if (lane == 0) v2_ev(P, MODE - 1, 15);\n    return;\n  }\n\n"
                    "  // ---------------- consumers", "epilogue exit")
    src = _sub_once(src, reg, "    const int nw = ((ntok + 15) >> 4) << 4;   // hot: wgmma N in {16, 32, 48, 64}\n",
                    "    const int nw = ((ntok + 15) >> 4) << 4;   // hot: wgmma N in {16, 32, 48, 64}\n"
                    "    v2_item(P, MODE - 1, HOTCAP, ntok, s1 - s0);\n", "item")
    k3a = "k3_kernel(const Params P) {\n  __shared__ int s_fc[8];\n  __shared__ float s_fr[8];\n  pdl_wait();\n"
    src = _sub_once(src, None, k3a,
                    "k3_kernel(const Params P) {\n  __shared__ int s_fc[8];\n  __shared__ float s_fr[8];\n"
                    "  if (threadIdx.x == 0) v2_ev(P, 2, 6);\n  pdl_wait();\n  if (threadIdx.x == 0) v2_ev(P, 2, 3);\n", "k3 entry")
    k3e = "    *reinterpret_cast<uint2*>(P.out + (size_t)tok * KDIM + col) = o;\n    __syncthreads();\n  }\n}\n"
    src = _sub_once(src, None, k3e, k3e[:-2] + "  if (threadIdx.x == 0) v2_ev(P, 2, 5);\n}\n", "k3 end")
    return src


def lab_source():
    src = patched_source()
    k = src.index("namespace k3 {")
    head, body = src[:k], src[k:]
    name, (hl, prof, var, vids) = COPY
    cs = [H._case(v) for v in vids]
    copy = (body.replace("@HPLAB@", str(hl)).replace("@V40PROF@", str(prof)).replace("@V40VAR@", str(var))
            .replace("@CASES3@", "\n".join(c[0] for c in cs)).replace("@ATTRS3@", "\n".join(c[1] for c in cs)))
    cuda = (head + f"\nnamespace v2m_{name} {{\n{copy}\n}}  // namespace v2m_{name}\n"
            f"\nvoid v2mrun({V.RUN_ARGS}) {{\n  v2m_{name}::run3({V.RUN_NAMES});\n}}\n"
            f"std::vector<int64_t> v2mattrs(int64_t vid) {{ return v2m_{name}::attrs3v(vid); }}\n")
    cpp = ("#include <torch/extension.h>\n#include <vector>\n"
           f"void v2mrun({V.RUN_ARGS});\nstd::vector<int64_t> v2mattrs(int64_t vid);\n")
    return cuda, cpp


def build(name="k2a_v2m_lab", verbose=False):
    """torch extension with v2mrun / v2mattrs. Returns (module, seconds, extension name)."""
    from torch.utils.cpp_extension import load_inline
    cuda, cpp = lab_source()
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    ext = f"{name}_{tag}"
    mod = load_inline(name=ext, cpp_sources=cpp, cuda_sources=cuda, functions=["v2mrun", "v2mattrs"],
                      extra_cuda_cflags=H.CFLAGS, verbose=verbose)
    return mod, time.time() - t0, ext


if __name__ == "__main__":   # write the generated CUDA source for review / an offline nvcc
    cu, cp = lab_source()
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "k2a_v2m_lab_gen.cu")
    out.write_text(cu, encoding="utf-8", newline="\n")
    out.with_suffix(".cpp").write_text(cp, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(cu)} chars)")

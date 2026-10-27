"""k3p_patches.py: K3P = study-b's K2A K3 (MoE finalize) levers as shipping patches (R-study-b-3; lab generator
.claude/worktrees/study-b-06cd73/kernels/sb/gen_k3p_lab.py @ 4521413, kernel text copied verbatim, names de-labbed).

  * K3B (every bucket): k3_kernel_b issues each route's sum of squares together with the y rows (2 dependent L2 round trips
    per item instead of 3; same math, same order: bit-identical to k3_kernel locally for M 1 / 8 / 16 / 32 / 128 / 256 at
    grid 264 and 4M, kernels/sb/test_k3b_local.py). Selected per call by probe bit 31 in launch3 / launch36 / launch37 of
    K2A33_KERNEL_SRC; k3_kernel stays as the fallback (K3P_K3B[bucket] False).
  * K3 grid 4T at T >= 128 (buckets 128 / 256): one item per CTA instead of 2 x SMs CTAs with 2 / 4 items in series.
R-study-b-3 (box, 2 router seeds, vs k3_v35_pt1): grid at T >= 128 ver16 +1.6 / +2.5, ver32 +2.4 / +3.5, full +4.2 / +4.0 us
(6 of 6 positive); K3B +0.3..+0.9 us at T 16 / 32, +1.0..+2.1 at T = 256. Module globals are read at call time.

    python -I tools/k3p_patches.py in.py out.py
"""
import pathlib
import py_compile
import sys

K3B = r'''
// K3P (tools/k3p_patches.py, study-b R-study-b-3): K3 with two dependent L2 round trips per item. Each route's sum of squares
// is issued with the y rows instead of gating them; the arithmetic is k3_kernel's, term for term.
__global__ void __launch_bounds__(256) k3_kernel_b(const Params P) {
  __shared__ int s_fc[8];
  __shared__ float s_fw[8];
  __shared__ float s_fr[8];
  if ((P.probe >> 25) & 1) pdl_launch();   // SU1: the staged unpooler may launch now (its weights do not depend on K3)
  pdl_wait();
  const int tid = threadIdx.x;
  const int items = 4 * P.M;
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int tok = it >> 2;
    if (tid < 8) {
      s_fc[tid] = __ldcg(P.meta + MH + tok * 8 + tid);
      s_fw[tid] = __ldg(P.wts + tok * 8 + tid);
    }
    __syncthreads();
    const int col = (it & 3) * 1024 + 4 * tid;
    float ssv = 0.f;
    if (tid < 8 && s_fc[tid] >= 0) ssv = __ldcg(P.ss + s_fc[tid]);   // issued, consumed after the y loads
    float4 v[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int c = s_fc[s];
      v[s] = (c >= 0) ? __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)c * KDIM + col)) : z4;
    }
    const float4 w = __ldg(reinterpret_cast<const float4*>(P.nw + col));
    if (tid < 8) s_fr[tid] = (s_fc[tid] >= 0) ? rsqrtf(ssv * (1.f / 4096.f) + 1e-8f) * s_fw[tid] : 0.f;
    __syncthreads();
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
    __syncthreads();
  }
}

}  // namespace k3
'''

PY = '''

# K3P (tools/k3p_patches.py, study-b R-study-b-3): K2A K3 (MoE finalize) levers, read at call time
K3P_K3B = {1: True, 8: True, 16: True, 32: True, 128: True, 256: True}   # bucket rep -> K3B (probe bit 31)
K3P_K3_GRID = {128: True, 256: True}                                    # bucket rep -> K3 grid 4 T (one item per CTA)


def _k3p_bits(m):
    return (1 << 31) if K3P_K3B.get(_k2a_bucket_of(m)) else 0


def _k3p_g3(st, m, default):
    return max(4 * m, 1) if K3P_K3_GRID.get(_k2a_bucket_of(m)) else default

'''

REGION = ('K2A33_KERNEL_SRC = r"""', '"""\n')
LAUNCH = "C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel, p));"
CALL36 = '_su1_g3(st, m), probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG["k3trig"] else 0))\n'
CALL37 = 'st["sms"] * 2, pb | ((1 << 24) if K2A47_K1_PDL_T128 else 0))\n'


def _fail(msg):
    raise RuntimeError(f"k3p patch: {msg}")


def apply(src):
    a = src.index(REGION[0])
    b = src.index(REGION[1], a + len(REGION[0]))
    if src.count(REGION[0]) != 1:
        _fail("K2A33_KERNEL_SRC not unique")
    k3 = src.find("__global__ void __launch_bounds__(256) k3_kernel(const Params P) {", a, b)
    if k3 < 0:
        _fail("k3_kernel not in K2A33_KERNEL_SRC")
    end_k3 = src.find("}  // namespace k3\n", k3, b)
    if end_k3 < 0:
        _fail("namespace k3 end not found after k3_kernel")
    src = src[:end_k3] + K3B.lstrip("\n") + src[end_k3 + len("}  // namespace k3\n"):]
    a = src.index(REGION[0])
    b = src.index(REGION[1], a + len(REGION[0]))
    seg = src[a:b]
    if seg.count(LAUNCH) != 3:   # launch3 (v3.3), launch36, launch37
        _fail(f"K3 launches in K2A33_KERNEL_SRC: {seg.count(LAUNCH)} (want 3)")
    seg = seg.replace(LAUNCH, "if ((p.probe >> 31) & 1) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel_b, p));   "
                              "/* K3P */ else C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel, p));")
    src = src[:a] + seg + src[b:]
    if src.count(CALL36) != 2:
        _fail(f"_k2a33 / _k2a36 K3 grid call sites: {src.count(CALL36)} (want 2)")
    src = src.replace(CALL36, '_k3p_g3(st, m, _su1_g3(st, m)), probe | ((1 << 24) if K2A47_K1_PDL else 0) | '
                              '((1 << 25) if _SU1_FLAG["k3trig"] else 0) | _k3p_bits(m))\n')
    if src.count(CALL37) != 1:
        _fail(f"_k2a37 K3 grid call site: {src.count(CALL37)} (want 1)")
    src = src.replace(CALL37, '_k3p_g3(st, m, st["sms"] * 2), pb | ((1 << 24) if K2A47_K1_PDL_T128 else 0) | _k3p_bits(m))\n')
    if src.count("\ndef kernel_fn(data, state):\n") != 1:
        _fail("kernel_fn anchor")
    return src.replace("\ndef kernel_fn(data, state):\n", PY + "\ndef kernel_fn(data, state):\n")


if __name__ == "__main__":
    s, d = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    t = s.read_text(encoding="utf-8")
    out = apply(t)
    d.write_text(out, encoding="utf-8", newline="\n")
    py_compile.compile(str(d), doraise=True)
    print(f"k3p: applied to {s.name} -> {d} ({len(out) - len(t):+d} chars, compiles)")

"""fidx_local_test.py: local RTX 3070 check of the FIDX index ring (tools/fidx_patches.py) against the ship producers.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/fidx_local_test.py" [kernels/lab/k3_fidx_1.py]

VA and VFA run only on sm_89 / sm_90, so this tests the part FIDX changes: the producers' index walk. It takes the producer code
verbatim from the lab file (VA: the `if (warp >= 4)` block of va1_main; VFA: producer<>, with the FIDX helpers and pvt_on), and
stubs out the copy and barrier primitives. Every cp_async16 call records (source address, byte count) per thread, in order.
Ship (IX false) and FIDX (IX true) must record identical sequences. That means the same K / V rows, the same zero-fill, and
the same tile order, for random request lengths, splits, page sizes 1 / 64, segk and segment lists.
"""
import ast
import pathlib
import random
import sys

import torch
from torch.utils.cpp_extension import load_inline

ROOT = pathlib.Path(__file__).resolve().parents[1]


def src_of(tree, name):
    return next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name)


def between(s, a, b, inc_b=False):
    i = s.index(a)
    j = s.index(b, i)
    return s[i:j + (len(b) if inc_b else 0)]


STUBS = r"""
#include <stdint.h>
__device__ unsigned long long* g_out; __device__ int* g_cnt; __device__ int g_maxc;
__device__ __forceinline__ void rec(const void* g, int n) {
  const int t = threadIdx.x, i = g_cnt[t]++;
  if (i < g_maxc) g_out[(size_t)t * g_maxc + i] = ((unsigned long long)(uintptr_t)g << 8) | (unsigned)n;
}
"""

VA_HARNESS = r"""
namespace vt {
constexpr int TILE = 64, ROWB = 128, STAGE_BYTES = 2 * TILE * ROWB;
struct Params { const uint8_t* k_cache; const uint8_t* v_cache; const int* page_table; int page_shift, pt_stride, Hkv; };
template <int ST, bool IX = false> constexpr int va_smem() { return 0; }
__device__ __forceinline__ uint32_t va_sa(const void* x) { return (uint32_t)__cvta_generic_to_shared(x); }
__device__ __forceinline__ void va_wait(uint64_t*, int) {}
__device__ __forceinline__ void cp_async16(void*, const void* g, int n) { rec(g, n); }
template <int L> __device__ __forceinline__ int kswz(int x, int ch) { return ch; }
template <int L> __device__ __forceinline__ int vswz(int x, int ch) { return ch; }
@@HELPERS@@
template <int ST, bool IX>
__global__ void va_t(const Params p, const int* gV, const int* gN, int B, int segk, int bx, int G) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int sV[520], sN[520];
  __shared__ uint64_t full[8], empt[8];
  for (int i = threadIdx.x; i <= B; i += blockDim.x) { sV[i] = gV[i]; sN[i] = gN[i]; }
  __syncthreads();
  const int tid = threadIdx.x + 128, lane = tid & 31, warp = tid >> 5, hp = 0;
  const long long T = (long long)sV[B] - segk;
  const int v0 = (int)((T * bx) / G), v1 = (int)((T * (bx + 1)) / G);
  int bf = 0;
  for (int hi = B - 1; bf < hi;) {
    const int mid = (bf + hi) >> 1;
    if (sV[mid] + sN[mid] > v0) hi = mid;
    else bf = mid + 1;
  }
  const size_t rowB = (size_t)p.Hkv * 64;
  int n = 0;
@@PRODUCER@@
}
}  // namespace vt
"""

VFA_HARNESS = r"""
namespace vfa {
constexpr int TILE = 64, ROWB = 128, STAGE_BYTES = TILE * ROWB, VT_BYTES = 72 * 64, MAXSEG = 32;
struct Params { const uint8_t* k_cache; const uint8_t* v_cache; const int* page_table; int page_shift, pt_stride, Hkv; };
template <int ST> constexpr int off_v() { return ST * STAGE_BYTES; }
template <int ST> constexpr int off_vt() { return 2 * ST * STAGE_BYTES; }
template <int ST> constexpr int smem_bytes() { return 1024; }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_wait(uint64_t*, uint32_t) {}
__device__ __forceinline__ void mbar_arrive(uint64_t*) {}
__device__ __forceinline__ void fence_proxy_async() {}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t) { r[0] = r[1] = r[2] = r[3] = 0u; }
__device__ __forceinline__ void sts32(uint32_t, uint32_t) {}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t, uint32_t) { return a; }
__device__ __forceinline__ void cp_async16(uint32_t, const void* g, int n) { rec(g, n); }
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t*) {}
@@PRODUCER@@
template <bool IX>
__global__ void vfa_t(const Params p, const int* gseg, int nseg) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ uint64_t bars[64];
  __shared__ int seg[4 * MAXSEG];
  for (int i = threadIdx.x; i < 4 * nseg; i += blockDim.x) seg[i] = gseg[i];
  __syncthreads();
  producer<4, true, 0, IX>(p, smem, seg, nseg, bars, bars + 8, threadIdx.x, 0);
}
}  // namespace vfa
"""

LAUNCH = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void setrec(torch::Tensor out, torch::Tensor cnt, int64_t maxc) {
  unsigned long long* o = (unsigned long long*)out.data_ptr(); int* c = cnt.data_ptr<int>(); int m = (int)maxc;
  cudaMemcpyToSymbol(g_out, &o, sizeof(o)); cudaMemcpyToSymbol(g_cnt, &c, sizeof(c)); cudaMemcpyToSymbol(g_maxc, &m, sizeof(m));
}
void va(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor sv, torch::Tensor sn, int64_t B, int64_t segk, int64_t psh,
        int64_t G, int64_t ix) {
  vt::Params p{(const uint8_t*)kc.data_ptr(), (const uint8_t*)vc.data_ptr(), pt.data_ptr<int>(), (int)psh, (int)pt.size(1), 2};
  auto s = at::cuda::getCurrentCUDAStream();
  if (ix) vt::va_t<4, true><<<1, 128, 8192, s>>>(p, sv.data_ptr<int>(), sn.data_ptr<int>(), (int)B, (int)segk, (int)(G >> 16), (int)(G & 0xffff));
  else vt::va_t<4, false><<<1, 128, 8192, s>>>(p, sv.data_ptr<int>(), sn.data_ptr<int>(), (int)B, (int)segk, (int)(G >> 16), (int)(G & 0xffff));
}
void vfat(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seg, int64_t psh, int64_t ix) {
  vfa::Params p{(const uint8_t*)kc.data_ptr(), (const uint8_t*)vc.data_ptr(), pt.data_ptr<int>(), (int)psh, (int)pt.size(1), 2};
  auto s = at::cuda::getCurrentCUDAStream();
  if (ix) vfa::vfa_t<true><<<1, 128, 8192, s>>>(p, seg.data_ptr<int>(), (int)(seg.numel() / 4));
  else vfa::vfa_t<false><<<1, 128, 8192, s>>>(p, seg.data_ptr<int>(), (int)(seg.numel() / 4));
}
"""


def build(lab):
    tree = ast.parse(lab.read_text(encoding="utf-8"))
    k1 = next(v for v in (n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                          and isinstance(n.value.value, str)) if "va1_main" in v and "__global__" in v)
    helpers = between(k1, "#ifndef FIDX_TPC", "__device__ __forceinline__ void fx_sts", False)
    helpers += k1[k1.index("__device__ __forceinline__ void fx_sts"):].split("\n", 1)[0] + "\n"
    prod = between(k1, "  if (warp >= 4) {\n    const int q = tid - 128", '    asm volatile("cp.async.wait_all;\\n" ::: "memory");\n    return;\n  }', True)
    prod = prod.replace('asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\\n" ::"r"(va_sa(full + st)) : "memory");', "(void)full;")
    prod = prod.replace('    asm volatile("cp.async.wait_all;\\n" ::: "memory");\n    return;', "    return;")
    va_src = VA_HARNESS.replace("@@HELPERS@@", helpers).replace("@@PRODUCER@@", prod)
    vf = src_of(tree, "VFA_CUDA_SRC")
    vprod = between(vf, "#ifndef FIDX_TPC", "template <int ST, int DIAG>\n__device__ __forceinline__ void consumer(")
    vfa_src = VFA_HARNESS.replace("@@PRODUCER@@", vprod)
    cuda = STUBS + va_src + vfa_src + LAUNCH.split("#include <ATen/cuda/CUDAContext.h>\n", 1)[1].replace("#include <torch/extension.h>\n", "")
    cuda = "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + cuda
    decl = ("void setrec(torch::Tensor out, torch::Tensor cnt, int64_t maxc);\n"
            "void va(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor sv, torch::Tensor sn, int64_t B, int64_t segk, int64_t psh, int64_t G, int64_t ix);\n"
            "void vfat(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seg, int64_t psh, int64_t ix);\n")
    return load_inline("fidx_local_test", decl, cuda_sources=cuda, functions=["setrec", "va", "vfat"],
                       extra_cuda_cflags=["-O2", "--expt-relaxed-constexpr"], verbose=False)


def main():
    lab = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "kernels/lab/k3_fidx_1.py")
    m = build(lab)
    dev = torch.device("cuda")
    rnd = random.Random(7)
    maxc = 1 << 15
    out = torch.zeros((128, maxc), dtype=torch.int64, device=dev)
    cnt = torch.zeros(128, dtype=torch.int32, device=dev)
    m.setrec(out, cnt, maxc)
    kc = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)
    vc = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)

    def run(fn):
        res = []
        for ix in (0, 1):
            out.zero_(); cnt.zero_()
            fn(ix)
            torch.cuda.synchronize()
            c = cnt.clone()
            assert int(c.max()) <= maxc, "record overflow"
            res.append((c, out.clone()))
        (c0, o0), (c1, o1) = res
        mask = torch.arange(maxc, device=dev)[None, :] < c0[:, None]
        return bool(torch.equal(c0, c1) and torch.equal(o0[mask], o1[mask])), int(c0.sum())

    fails, calls, cases = 0, 0, 0
    for case in range(60):
        page = rnd.choice([1, 1, 64])
        psh = page.bit_length() - 1
        B = rnd.choice([1, 2, 3, 5, 16, 33])
        lens = [rnd.choice([1, 2, 63, 64, 65, 511, 512, 513, rnd.randint(1, 3000)]) for _ in range(B)]
        maxp = max((L + 8 + page - 1) // page for L in lens)
        pt = torch.tensor([rnd.sample(range(4000), maxp) for _ in range(B)], dtype=torch.int32, device=dev)
        segk = rnd.choice([0, 400])
        sn = [L + 1 for L in lens]
        sv = [0]
        for b in range(B):
            sv.append(sv[-1] + sn[b] + segk)
        svt = torch.tensor(sv, dtype=torch.int32, device=dev)
        snt = torch.tensor(sn + [0], dtype=torch.int32, device=dev)
        # VA: one CTA = block bx of a g-way flat split (g 1 -> the whole request list in one CTA)
        g = rnd.choice([1, 1, 2, 7, 33])
        G = (rnd.randrange(g) << 16) | g   # (block index << 16) | grid size
        ok, nc = run(lambda ix: m.va(kc, vc, pt, svt, snt, B, segk, psh, G, ix))
        cases += 1; calls += nc; fails += (not ok)
        # VFA: a random segment list (b, k0, k1) over these requests
        segs = []
        for _ in range(rnd.randint(1, 6)):
            b = rnd.randrange(B)
            k0 = rnd.randint(0, lens[b] - 1)
            k1 = rnd.randint(k0 + 1, lens[b] + 8)
            segs += [b, k0, k1, 0]
        st = torch.tensor(segs, dtype=torch.int32, device=dev)
        ok2, nc2 = run(lambda ix: m.vfat(kc, vc, pt, st, psh, ix))
        cases += 1; calls += nc2; fails += (not ok2)
        if not (ok and ok2):
            print(f"[fidx_local] case {case}: page {page} B {B} segk {segk} lens {lens[:6]} VA {'ok' if ok else 'MISMATCH'} "
                  f"VFA {'ok' if ok2 else 'MISMATCH'}")
    print(f"[fidx_local] {cases} producer runs, {calls} recorded copies each way, {fails} mismatches -> "
          f"{'PASS' if fails == 0 else 'FAIL'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

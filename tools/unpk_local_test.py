"""unpk_local_test.py: RTX 3070 (sm_86) bitwise check of the UNPK A+B unpack (tools/unpk_patches.py) vs the ship's VML
and VMS decode, regions verbatim from both sources (VM_TRICKS_DQ 0 as bf2 / w72). No torch build: plain nvcc + main().

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/unpk_local_test.py" [layer.py]
Checks: deq4 over all 2^32 code words; dec_half2 (ship) vs sgp_of + dec_half2s (B) over 2^28 random (w0, w8, sw2, par)
cases incl. every par; VMS r3 gather (ship unp_ok32 vs A unp_g1084) over all 2^32 words; plus a negative control.
"""
import ast
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import unpk_patches as P  # noqa: E402


def cut(s, a, b):
    i = s.index(a)
    return s[i:s.index(b, i)]


def regions(text):
    tree = ast.parse(text)
    get = lambda nm: next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == nm)
    v = get("_vml_CUDA_T")
    st = "__device__ __forceinline__ uint32_t unp_g1084" if "unp_g1084" in v else "__device__ __forceinline__ uint32_t unp_ok32"
    deq = cut(v, st, "__device__ __forceinline__ uint32_t bmul2(")
    dh = cut(v, "__device__ __forceinline__ void deq4_dq(", "#if VM_AD\nstruct Pend")
    vm = get("VMS_CUDA_SRC")
    g = "unp_g1084(qw & 0x001B001Bu)" if "unp_g1084(qw" in vm else "(qw & 0x001B001Bu) * unp_ok32(0x1084u)"
    return deq + dh, g


MAIN = r"""
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#define VM_TRICKS_DQ 0
namespace s { @S@ __device__ uint32_t vg(uint32_t qw) { return ((@SG@) & 0x81C081C0u); } }
namespace n { @N@ __device__ uint32_t vg(uint32_t qw) { return ((@NG@) & 0x81C081C0u); } }
__device__ __forceinline__ uint32_t hsh(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33; return (uint32_t)x; }
__global__ void kw(uint32_t hi, unsigned long long* bad, uint32_t neg) {   // all words (hi << 24) | tid
  const uint32_t w = (hi << 24) | (blockIdx.x * blockDim.x + threadIdx.x);
  uint32_t a[4], b[4]; s::deq4(w, a); n::deq4(w, b);
  b[3] ^= (neg && w == 0x12345678u);
  int e = 0;
  for (int j = 0; j < 4; ++j) e |= a[j] != b[j];
  e |= s::vg(w) != n::vg(w);
  if (e) atomicAdd(bad, 1ull);
}
__global__ void kd(uint32_t it, unsigned long long* bad) {
  const uint64_t id = (uint64_t)it * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t w0 = hsh(4 * id), w8 = hsh(4 * id + 1), sw2 = hsh(4 * id + 2), par = id & 1;
  uint32_t a[8], b[8];
  s::dec_half2(w0, w8, sw2, par, a, 16u, 64u, 4u);
  n::dec_half2s(w0, w8, n::sgp_of(sw2, par, 16u), b, 64u, 4u);
  int e = 0;
  for (int j = 0; j < 8; ++j) e |= a[j] != b[j];
  if (e) atomicAdd(bad, 1ull);
}
int main() {
  unsigned long long *d, h[3];
  cudaMalloc(&d, 24); cudaMemset(d, 0, 24);
  for (uint32_t hi = 0; hi < 256; ++hi) kw<<<65536, 256>>>(hi, d, 0);
  kw<<<65536, 256>>>(0x12, d + 2, 1);
  for (uint32_t it = 0; it < 16; ++it) kd<<<65536, 256>>>(it, d + 1);
  cudaMemcpy(h, d, 24, cudaMemcpyDeviceToHost);
  printf("[unpk-local] deq4 + VMS gather, 2^32 words: %llu mismatches\n", h[0]);
  printf("[unpk-local] dec_half2 vs sgp_of + dec_half2s, 2^28 cases: %llu mismatches\n", h[1]);
  printf("[unpk-local] negative control (1 flipped bit expected): %llu\n", h[2]);
  printf("[unpk-local] %s (%s)\n", (h[0] == 0 && h[1] == 0 && h[2] == 1) ? "PASS bitwise" : "FAIL", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
"""


def main():
    lay = sys.argv[1] if len(sys.argv) > 1 else "kernels/muse_decode_layer/k3_gc_21cg_s1.py"
    text = (ROOT / lay).read_text(encoding="utf-8")
    s, sg = regions(text)
    nn, ng = regions(P.patch(text, True, True))
    assert "dec_half2s" in nn and "unp_g1084" in nn and "unp_g1084" in ng
    src = MAIN.replace("@S@", s).replace("@SG@", sg).replace("@N@", nn).replace("@NG@", ng)
    with tempfile.TemporaryDirectory() as td:
        cu = pathlib.Path(td) / "t.cu"
        cu.write_text(src)
        exe = pathlib.Path(td) / "t"
        r = subprocess.run(["nvcc", "-O3", "-arch=sm_86", "-o", str(exe), str(cu)], capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-2000:])
            sys.exit(1)
        print(subprocess.run([str(exe)], capture_output=True, text=True).stdout, end="")


if __name__ == "__main__":
    main()

"""w128_gate.py ($0, R-w128-0): can the shipped VML RS-wgmma stage loop hold one-pass wide jobs (N 112 / 120 / 128) in a
512-thread CTA (4 consumer WGs, no producer WG -> launch_bounds(512, 1) = 128 registers)? E4 (lab.md:632) died on the
640-thread pool (consumers capped at 112). Compiles vcs_bench's vcs_seg_rs<N, 1 (nosync), 0, 0> (gc_18 stage<N> structure, real
unpack + RS wgmma, accumulators live across 16 stages) in a 512-thread kernel; KILL if any spill or C751x serialization.
    (WSL) python tools/w128_gate.py OUTDIR
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels/lab"))
import vcs_bench as V  # noqa: E402

KER = r"""
namespace vm {
template <int N>
__global__ void __launch_bounds__(512, 1) w128_kernel(const Args A, const PArgs P) {
extern __shared__ __align__(1024) uint8_t smem[];
const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, q = warp >> 2, wp = warp & 3;
const uint32_t bars = sa(smem + BAR_OFF);
int it = 0;
float d[N / 2];
vcs_seg_rs<N, 1, 0, 0>(A, smem, bars, it, d, q, wp, lane);
drp_store<N>(P, 0, q * 8 + wp, lane, d, 1.f);
}
template __global__ void w128_kernel<112>(const Args A, const PArgs P);
template __global__ void w128_kernel<120>(const Args A, const PArgs P);
template __global__ void w128_kernel<128>(const Args A, const PArgs P);
template __global__ void w128_kernel<64>(const Args A, const PArgs P);
}  // namespace vm
"""


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    dev, _, fl = V.module_src("w72")
    dev = V.strip_inc(dev) + KER
    cu = out / "w128.cu"
    cu.write_text(dev)
    flags = [f for f in fl if not f.startswith("-DVM_WG5") and not f.startswith("-DVM_CREG") and not f.startswith("-DVM_PREG")]
    r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *V.FLAGS, *flags, "-Xptxas=-v", "-o", str(out / "w128.cubin"), str(cu)],
                       capture_output=True, text=True)
    log = r.stdout + r.stderr
    (out / "ptxas.log").write_text(log)
    fn, rows = None, []
    for ln in log.splitlines():
        m = re.search(r"Compiling entry function '(\S+)'", ln)
        if m:
            fn = m.group(1)
        if fn and "w128_kernel" in fn and ("Used" in ln or "spill" in ln or re.search(r"C75\d\d|serializ", ln)):
            rows.append(f"{re.search(r'w128_kernelILi(\d+)E', fn).group(1):>4}: {ln.strip()[:160]}")
    warn = [ln.strip()[:200] for ln in log.splitlines() if re.search(r"C75\d\d|serializ", ln)]
    print(f"[w128] rc {r.returncode}; C75xx/serialized lines {len(warn)}")
    for w in warn[:8]:
        print("   " + w)
    for x in rows:
        print("   N" + x)


if __name__ == "__main__":
    main()

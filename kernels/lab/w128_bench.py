"""w128_bench.py (R-w128-1): stage rate of a producer-less 512-thread VML consumer (4 consumer WGs, no producer warps; the warp
that releases a ring slot LAST issues its refill) vs the shipped-structure w72 (640 threads, producer WG), on vcs_bench's lean
stage loops and dr_probe's unit plans (cold: distinct units after an L2 flush, warm: 8 L2-resident units).
  w72 : vcs_kernel (w72 defines), inc (gc_18 stage<N> / stage_m) + lean, N 8 / 32 / 64                    (incumbent)
  pl3 : vpl_kernel, NS = NX = 3, XSN 64 (w72's ring depth), lean, N 8 / 32 / 64      (cold-heavy jobs on producer-less CTAs)
  pl2 : vpl_kernel, NS = NX = 2, XSN 128 (N 128 fits 227 KB only with a 2-slot ring), lean, N 64 / 128  (one-pass hot jobs)
  w112: vcs_kernel, w72 + CREG 120 (PREG 32: the 64 K register file exactly), JN / XSN 112, NX 2, inc, N 64 / 112 (producer kept)
Per stage-row: us / stage (graph replay, median) and clk / stage (clock64 marks). An N 128 stage does the work of two N 64 stages.
  python3 -u kernels/lab/w128_bench.py gate OUTDIR        (local WSL sm_90a: ptxas regs / spills / C75xx of every kernel)
  python3 -u kernels/lab/w128_bench.py run [--rounds=9] [--units=3] [--builds=w72,pl3,pl2] [--out=box_out/w128/run.json]
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "tools", ROOT / "tools/judge", ROOT / "kernels/muse_moe_w4a16", ROOT / "kernels/lab"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import vcs_bench as V  # noqa: E402

NS_OLD = "constexpr int NS = 3, NX = VM_NX;"
NS_NEW = "#ifndef VM_NSW\n#define VM_NSW 3\n#endif\nconstexpr int NS = VM_NSW, NX = VM_NX;"
STRIP = ("-DVM_WG5", "-DVM_CREG", "-DVM_PREG", "-DVM_JN", "-DVM_XSN", "-DVM_NX")
PLDEF = {"pl3": ["-DVM_JN=64", "-DVM_XSN=64", "-DVM_NX=3", "-DVM_NSW=3"],
         "pl2": ["-DVM_JN=128", "-DVM_XSN=128", "-DVM_NX=2", "-DVM_NSW=2"]}
W112DEF = ["-DVM_CREG=120", "-DVM_JN=112", "-DVM_XSN=112", "-DVM_NX=2"]   # producer WG kept: 512 x 120 + 128 x 32 = 65536
KERNELS = {"w72": [(8, 0), (32, 0), (40, 0), (48, 0), (56, 0), (72, 0)],   # (N, V): V 0 inc, 1 lean
           "w112": [(8, 0), (32, 0), (56, 0), (80, 0), (96, 0), (112, 0)],
           "pl3": [(8, 1), (32, 1), (64, 1)], "pl2": [(64, 1), (128, 1)]}   # pl: R-w128-1 (KILL), kept for the record

VPL_DEV = r"""
namespace vm {
constexpr int PLT = 32 * NCW;   // 512 threads: four consumer WGs, no producer warps -> 128-register cap
constexpr int CNT_OFF = BAR_OFF + 8 * (2 * NS + 2 * NX);   // per-slot release counters (w: NS, x: NX)
constexpr int SMEM_PL = CNT_OFF + 4 * (NS + NX) + 64;
static_assert(SMEM_PL <= 232448, "vpl smem");
__device__ __forceinline__ void vpl_issue_w(const PArgs& P, uint32_t ring, uint32_t bars, int u0, int j) {
const int sl = j % NS;
const int4 un = P.unit[u0 + (j >> 5)];
const int h = (j >> 4) & 1, s = j & 15;
const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
const uint64_t pol = policy_evict_first();
mbar_expect_tx(fb, US);
const int blk = un.x * 64 + 4 * un.y + 2 * h;
bulk_g2s_ef(dst, P.ws + ((size_t)blk * 16 + s) * SW, SW, fb, pol);
bulk_g2s_ef(dst + SW, P.ws + ((size_t)(blk + 1) * 16 + s) * SW, SW, fb, pol);
}
__device__ __forceinline__ void vpl_issue_x(const PArgs& P, uint32_t xring, uint32_t bars, int u0, int j) {
const int sl = j % NX;
const uint32_t bytes = (uint32_t)P.npad * 512u;
const uint32_t fb = bars + 8 * (2 * NS + sl);
mbar_expect_tx(fb, bytes);
bulk_g2s(xring + sl * XS, P.xp + (size_t)P.unit[u0 + (j >> 5)].z + (size_t)(j & 15) * bytes, bytes, fb);
}
// lane 0 of a consumer warp releases stage j's weight slot; the 16th (last) releaser of the slot use waits the empty phase
// (acquire; normally complete already) and issues stage j + NS into it, as the producer warp would after its wait.
__device__ __forceinline__ void vpl_rel_w(const Args& A, const PArgs& P, uint8_t* smem, uint32_t bars, int u0, int nst, int j,
uint32_t dep) {
const int sl = j % NS;
mbar_arrive(bars + 8 * (NS + sl) + dep);
const unsigned old = atomicAdd(reinterpret_cast<unsigned*>(smem + CNT_OFF) + sl + dep, 1u);
if ((old & (NCW - 1)) == NCW - 1 && j + NS < nst) {
mbar_wait(bars + 8 * (NS + sl), (j / NS) & 1, A.err);
vpl_issue_w(P, sa(smem), bars, u0, j + NS);
}
}
__device__ __forceinline__ void vpl_rel_x(const Args& A, const PArgs& P, uint8_t* smem, uint32_t bars, int u0, int nst, int j) {
const int sl = j % NX;
mbar_arrive(bars + 8 * (2 * NS + NX + sl));
const unsigned old = atomicAdd(reinterpret_cast<unsigned*>(smem + CNT_OFF) + NS + sl, 1u);
if ((old & (NCW - 1)) == NCW - 1 && j + NX < nst) {
mbar_wait(bars + 8 * (2 * NS + NX + sl), (j / NX) & 1, A.err);
vpl_issue_x(P, sa(smem + NS * US), bars, u0, j + NX);
}
}
@SEGS@
template <int N, int V>
__global__ void __launch_bounds__(PLT, 1) vpl_kernel(const Args A, const PArgs P, int pf) {
extern __shared__ __align__(1024) uint8_t smem[];
const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
const uint32_t bars = sa(smem + BAR_OFF);
const int2 rg = P.cta[blockIdx.x];
const int nst = 32 * (rg.y - rg.x);
if (tid == 0) {
for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW); }
for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }
for (int i = 0; i < NS + NX; ++i) reinterpret_cast<unsigned*>(smem + CNT_OFF)[i] = 0u;
asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__syncthreads();
if (tid == 0) {   // prologue fill (the producers' first NS / NX iterations, no empty waits)
for (int j = 0; j < NS && j < nst; ++j) vpl_issue_w(P, sa(smem), bars, rg.x, j);
for (int j = 0; j < NX && j < nst; ++j) vpl_issue_x(P, sa(smem + NS * US), bars, rg.x, j);
}
(void)pf;
const int q = warp >> 2, wp = warp & 3;
long long* mk = P.marks ? P.marks + (size_t)blockIdx.x * 8 : nullptr;
if (mk && tid == 0) { mk[0] = gtime(); mk[5] = clock64(); }
int it = 0;
#pragma unroll 1
for (int u = rg.x; u < rg.y; ++u) {
const float gs = __ldg(P.gsc + P.unit[u].x);
#pragma unroll 1
for (int h = 0; h < 2; ++h) {
float d[N / 2];
if constexpr (N <= 16) vpl_seg_m<N, 0, 0>(A, smem, bars, it, d, q, wp, lane, P, rg.x, nst);
else vpl_seg_rs<N, 0, 0, 0>(A, smem, bars, it, d, q, wp, lane, P, rg.x, nst);
drp_store<N>(P, u, (2 * h + (q >> 1)) * 8 + (q & 1) * 4 + wp, lane, d, gs);
}
}
if (mk && wp == 0 && lane == 0) { mk[1 + q] = gtime(); if (q == 0) mk[6] = clock64(); }
}
}  // namespace vm
"""
VPL_CPP = r"""
#include <torch/extension.h>
void vpl_run(torch::Tensor ws, torch::Tensor gsc, torch::Tensor xp, torch::Tensor unit, torch::Tensor cta, torch::Tensor out,
             torch::Tensor err, torch::Tensor marks, int64_t npad, int64_t G, int64_t v, int64_t m, int64_t pf);
std::vector<int64_t> vpl_attrs(int64_t npad, int64_t v, int64_t m);
"""


def _seg(name_old, name_new, text):
    """copy of one vcs_bench lean loop as a producer-less loop: release sites -> vpl_rel_w / vpl_rel_x (last releaser refills)"""
    a = text.index(f"__device__ __forceinline__ void {name_old}(")
    a = text.rindex("template <", 0, a)
    b, depth = text.index("{", text.index("int lane)", a)), 0
    while True:   # brace match (the loop bodies are unindented: "\n}\n" occurs inside)
        depth += {"{": 1, "}": -1}.get(text[b], 0)
        b += 1
        if depth == 0:
            break
    s = (text[a:b] + "\n").replace(f"void {name_old}(", f"void {name_new}(")
    s = s.replace("float* d, int q, int wp, int lane) {", "float* d, int q, int wp, int lane, const PArgs& P, int u0, int nst) {", 1)
    reps = {"vpl_seg_m": [
        ("if (lane == 0) mbar_arrive(bars + 8 * (NS + r.ws) + ((c[1].x ^ c[2].x ^ c[3].x ^ sw1) & zom));",
         "if (lane == 0) vpl_rel_w(A, P, smem, bars, u0, nst, it + s, (c[1].x ^ c[2].x ^ c[3].x ^ sw1) & zom);"),
        ("mbar_arrive(bars + 8 * (2 * NS + NX + r.ws));\n}", "vpl_rel_x(A, P, smem, bars, u0, nst, it + s);\n}")],
        "vpl_seg_rs": [
        ("if (lane == 0 && prevx >= 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));\n}",
         "if (lane == 0 && prevx >= 0) vpl_rel_x(A, P, smem, bars, u0, nst, it + s - 1);\n}"),
        ("if (lane == 0) mbar_arrive(bars + 8 * (NS + r.ws) + ((c[1].x ^ c[2].x ^ c[3].x ^ sw1) & zo));",
         "if (lane == 0) vpl_rel_w(A, P, smem, bars, u0, nst, it + s, (c[1].x ^ c[2].x ^ c[3].x ^ sw1) & zo);"),
        ("if (SY && lane == 0 && prevx >= 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));\nit += 16;",
         "if (SY && lane == 0 && prevx >= 0) vpl_rel_x(A, P, smem, bars, u0, nst, it + 15);\nit += 16;")]}[name_new]
    for o, n in reps:
        assert s.count(o) == 1, f"{name_new}: anchor {o[:60]!r} found {s.count(o)} times"
        s = s.replace(o, n)
    assert "mbar_arrive(bars + 8 * (2 * NS + NX" not in s.replace("if (XD && lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + r.ws));", ""), name_new
    return s


def vpl_dev():
    segs = _seg("vcs_seg_m", "vpl_seg_m", V.VCS_DEV) + _seg("vcs_seg_rs", "vpl_seg_rs", V.VCS_DEV)
    return VPL_DEV.replace("@SEGS@", segs)


def vpl_host(build):
    ks = KERNELS[build]
    s = "\ntemplate <int N, int V>\nstatic void vpl_launch(const vm::Args& a, const vm::PArgs& p, int G, int pf) {\n"
    s += "  cudaFuncSetAttribute(vm::vpl_kernel<N, V>, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM_PL);\n"
    s += "  vm::vpl_kernel<N, V><<<G, vm::PLT, vm::SMEM_PL, at::cuda::getCurrentCUDAStream()>>>(a, p, pf);\n"
    s += "  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n"
    s += "template <int N, int V>\nstatic std::vector<int64_t> vpl_attr() {\n"
    s += "  cudaFuncSetAttribute(vm::vpl_kernel<N, V>, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM_PL);\n"
    s += "  cudaFuncAttributes fa;\n  cudaFuncGetAttributes(&fa, vm::vpl_kernel<N, V>);\n"
    s += "  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)vm::SMEM_PL, (int64_t)vm::PLT, vm::NS, vm::NX, vm::XS};\n}\n"
    s += "std::vector<int64_t> vpl_attrs(int64_t npad, int64_t v, int64_t m) {\n  switch (npad * 100 + v * 10 + m) {\n"
    s += "".join(f"    case {n * 100 + v * 10}: return vpl_attr<{n}, {v}>();\n" for n, v in ks)
    s += "    default: return {-1};\n  }\n}\n"
    s += ("void vpl_run(torch::Tensor ws, torch::Tensor gsc, torch::Tensor xp, torch::Tensor unit, torch::Tensor cta, torch::Tensor out,\n"
          "             torch::Tensor err, torch::Tensor marks, int64_t npad, int64_t G, int64_t v, int64_t m, int64_t pf) {\n"
          "  vm::Args a{};\n  a.err = err.data_ptr<int>();\n  a.flags = 193;\n  vm::PArgs p{};\n"
          "  p.ws = reinterpret_cast<const uint8_t*>(ws.data_ptr()); p.gsc = gsc.data_ptr<float>();\n"
          "  p.xp = reinterpret_cast<const uint8_t*>(xp.data_ptr()); p.unit = reinterpret_cast<const int4*>(unit.data_ptr());\n"
          "  p.cta = reinterpret_cast<const int2*>(cta.data_ptr()); p.out = out.data_ptr<float>();\n"
          "  p.marks = marks.numel() >= G * 8 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;\n"
          "  p.npad = (int)npad;\n  switch (npad * 100 + v * 10 + m) {\n")
    s += "".join(f"    case {n * 100 + v * 10}: vpl_launch<{n}, {v}>(a, p, (int)G, (int)pf); break;\n" for n, v in ks)
    s += '    default: TORCH_CHECK(false, "vpl: no such (N, V)");\n  }\n}\n'
    return s


def module_src(build):
    """(device + host CUDA, cpp decl, nvcc defines, entry names) of one build"""
    if build in ("w72", "w112"):
        V.kernels_of = lambda b: [(n, v, 0) for n, v in KERNELS[build]]
        dev, host, defs = V.module_src("w72")
        if build == "w112":   # inc kernels only (stage<N>, separate NS / NX rings): the lean loops' one-ring assert is moot
            a = 'static_assert(NS == NX, "vcs: one ring index for weight and x slots");'
            assert dev.count(a) == 1
            dev = dev.replace(a, 'static_assert(SMEM <= 232448, "w112 smem");')
            defs = [f for f in defs if not f.startswith(("-DVM_CREG", "-DVM_JN", "-DVM_XSN", "-DVM_NX"))] + W112DEF
        return dev, host, V.VCS_CPP, defs, ("vcs_run", "vcs_attrs")
    dev, _, defs = V.module_src("w72")
    assert dev.count(NS_OLD) == 1, "NS anchor"
    dev = dev.replace(NS_OLD, NS_NEW)
    i = dev.index("\nnamespace vm {\n__device__ void vcs_prod_w(")
    dev = dev[:i] + "\n" + V.VCS_DEV.split("// lean mma.sync loop (N <= 16)")[0] + "}  // namespace vm\n" + vpl_dev()
    defs = [f for f in defs if not f.startswith(STRIP)] + PLDEF[build]
    return dev, vpl_host(build), VPL_CPP, defs, ("vpl_run", "vpl_attrs")


def build_mod(build, src, verbose=False):
    import hashlib
    import time
    from torch.utils.cpp_extension import load_inline
    dev, host, cpp, defs, fns = src
    src = dev + host
    fl = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"] + defs
    tag = hashlib.sha1((cpp + src + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"w128_{build}_{tag}", cpp_sources=cpp, cuda_sources=src, functions=list(fns), extra_cuda_cflags=fl,
                      verbose=verbose)
    return mod, time.time() - t0


def gate_main(argv):
    out = pathlib.Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for b in (argv[1].split(",") if len(argv) > 1 else ("w72", "w112", "pl3", "pl2")):
        dev, host, _, defs, _ = module_src(b)
        dev = V.strip_inc(dev)
        if b in ("w72", "w112"):
            dev += "namespace vm {\n" + "".join(f"template __global__ void vcs_kernel<{n}, {v}, 0>(const Args A, const PArgs P, int pf);\n"
                                                for n, v in KERNELS[b]) + "}\n"
        else:
            dev += "namespace vm {\n" + "".join(f"template __global__ void vpl_kernel<{n}, {v}>(const Args A, const PArgs P, int pf);\n"
                                                for n, v in KERNELS[b]) + "}\n"
        cu = out / f"{b}.cu"
        cu.write_text(dev, encoding="utf-8")
        r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *V.FLAGS, *defs, "-Xptxas=-v", "-o", str(out / f"{b}.cubin"), str(cu)],
                           capture_output=True, text=True)
        log = r.stdout + r.stderr
        (out / f"{b}_ptxas.log").write_text(log, encoding="utf-8")
        if r.returncode:
            print(f"[w128] {b}: COMPILE FAILED\n" + log[-3000:])
            bad += 1
            continue
        fn = None
        for ln in log.splitlines():
            m = re.search(r"Compiling entry function '(\S+)'", ln)
            if m:
                fn = m.group(1)
            if fn and ("vpl_kernel" in fn or "vcs_kernel" in fn) and ("Used" in ln or "spill" in ln):
                k = re.search(r"(vpl|vcs)_kernelILi(\d+)ELi(\d+)E", fn)
                if "spill" in ln and not re.search(r"\b0 bytes spill stores, 0 bytes spill loads", ln):
                    bad += 1
                print(f"[w128] {b} N{k.group(2):>3} V{k.group(3)}: {ln.strip()[:150]}")
        warn = [ln.strip()[:200] for ln in log.splitlines() if re.search(r"C75\d\d|serializ", ln)]
        bad += len(warn)
        for w in warn[:6]:
            print("   " + w)
    print(f"[w128] gate {'PASS' if not bad else f'FAIL ({bad})'}")
    return 1 if bad else 0


def run_main(argv):
    import concurrent.futures as cf
    import json
    import statistics
    import time

    import torch
    import dr_probe as DRP
    import judge as J
    import k2a_v37 as D
    import unp_patches as UNP

    log = lambda s: print(s, flush=True)  # noqa: E731
    rounds, K, warm_set = int(DRP.arg(argv, "rounds", 9)), int(DRP.arg(argv, "units", 3)), int(DRP.arg(argv, "warm-set", 8))
    builds = str(DRP.arg(argv, "builds", "w72,w112")).split(",")
    only_n = {int(v) for v in str(DRP.arg(argv, "ntoks", "0")).split(",")} - {0}
    modes = tuple(str(DRP.arg(argv, "modes", "cold,warm")).split(","))
    one = bool(int(DRP.arg(argv, "one", 0)))   # 1: one call per arm + sync, no timing (compute-sanitizer runs)
    out_p = ROOT / str(DRP.arg(argv, "out", "box_out/w128/run.json"))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    srcs = {b: module_src(b) for b in builds}   # serial: module_src monkeypatches vcs_bench for w72
    mods = {}
    with cf.ThreadPoolExecutor(len(builds)) as ex:
        futs = {b: ex.submit(build_mod, b, srcs[b]) for b in builds}
        for b, f in futs.items():
            mods[b], dt = f.result()
    runs, attrs = {}, {}
    for b in builds:
        run_fn, attr_fn = (mods[b].vcs_run, mods[b].vcs_attrs) if b in ("w72", "w112") else (mods[b].vpl_run, mods[b].vpl_attrs)
        runs[b] = run_fn
        for n, v in KERNELS[b]:
            attrs[(b, n, v)] = list(attr_fn(n, v, 0))
        log(f"[w128] built {b}; (regs, local B, smem, thr, NS, NX, XS) " + " ".join(f"N{n}V{v} {attrs[(b, n, v)]}" for n, v in KERNELS[b]))
    del srcs
    if only_n and not any(n in only_n for b in builds for n, _ in KERNELS[b]):   # build-only pass (fills the extension cache)
        log(f"[w128] build-only pass done ({time.time() - t0:.0f} s)")
        return 0
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    prob = J.load_problem(ROOT / "problems/muse_moe_w4a16")
    ref = prob.ref
    w = ref.generate_weights(ref.WEIGHTS_SEED, device=dev)
    ws13 = torch.empty((32, 64, 16, D.SW3), dtype=torch.uint8, device=dev)
    for e in range(32):
        w13q, s13q = D.prepack_expert(w["w13"][e], w["w13_sf"][e], gate_up=True, bitpos=UNP.L2POS)
        ws13[e] = D.stage_repack(w13q, s13q)
    gsc13 = (w["w13_gs"][:32].float() * D.W_SHIFT).contiguous()
    perm = D.tile_row_perm(8192, True, device=dev)
    xsrc = torch.randn((512, 4096), generator=torch.Generator(device="cpu").manual_seed(5), dtype=torch.float32).to(torch.bfloat16).to(dev)
    timer = J.GpuTimer()
    ov = timer.replay_overhead_us()
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    nomark = torch.zeros((0,), dtype=torch.int64, device=dev)
    marks = torch.zeros((G * 8,), dtype=torch.int64, device=dev)
    nst = 32 * K
    log(f"[w128] setup {time.time() - t0:.0f} s, {G} SMs, overhead {ov:.2f} us, gate {timer.gate_us} us, {nst} stages / CTA, rounds {rounds}")

    def timed(g, flush, warm):
        with torch.cuda.stream(timer.stream):
            if warm:
                g.replay()
            if flush:
                timer.flush()
            timer.gate()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1e3

    res = {"G": G, "overhead_us": ov, "rounds": rounds, "stages_per_cta": nst, "attrs": {str(k): v for k, v in attrs.items()}, "rows": []}
    for N in sorted({n for b in builds for n, _ in KERNELS[b] if not only_n or n in only_n}):
        arms = [(b, v) for b in builds for n, v in KERNELS[b] if n == N]
        for mode in modes:
            P = DRP.plan(N, mode, G, K, warm_set, xsrc, dev)
            npad = P["npad"]
            outs = {a: torch.full((P["n"], 512, npad), float("nan"), dtype=torch.float32, device=dev) for a in arms}

            def fn(a, mk=nomark, P=P, outs=outs):
                b, v = a
                return lambda: runs[b](ws13, gsc13, P["xp"], P["unit"], P["cta"], outs[a], err, mk, npad, G, v, 0, 0)
            refs = {}
            for i in sorted({0, P["n"] // 2, P["n"] - 1}):
                e, U = P["units"][i]
                W = DRP.dequant_rows64(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e], perm[512 * U:512 * U + 512])
                refs[i] = W @ xsrc[torch.as_tensor(P["toks"][e], dtype=torch.long, device=dev)].double().T
            if one:
                for a in arms:
                    err.zero_()
                    runs[a[0]](ws13, gsc13, P["xp"], P["unit"], P["cta"], outs[a], err, nomark, npad, G, a[1], 0, 0)
                    torch.cuda.synchronize()
                    log(f"[w128] one N{N} {mode} {a}: err {int(err[0])}")
                continue
            chk = {}
            a0 = arms[0]
            for a in arms:
                err.zero_()
                fn(a)()
                torch.cuda.synchronize()
                o = outs[a]
                rel = max(float((o[i, :, :N].double() - r).norm() / r.norm()) for i, r in refs.items())
                o1 = o.clone()
                nrep = 0
                for _ in range(3):
                    fn(a)()
                    torch.cuda.synchronize()
                    nrep += int((o1 != outs[a]).sum())
                chk[a] = {"err": int(err[0]), "rel_fp64": rel, "repeat_equal": nrep == 0, "bitwise_first": bool((o1 == outs[a0]).all())}
                chk[a]["ok"] = chk[a]["err"] == 0 and rel < 1e-4 and nrep == 0
            gr = {}
            for a in arms:
                f = fn(a)
                with torch.cuda.stream(timer.stream):
                    f()
                    torch.cuda.synchronize()
                    g_ = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_, stream=timer.stream):
                        f()
                torch.cuda.synchronize()
                gr[a] = g_
            ts = {a: [] for a in arms}
            for rd in range(rounds):
                k = rd % len(arms)
                for a in arms[k:] + arms[:k]:
                    ts[a].append(max(timed(gr[a], mode == "cold", mode == "warm") - ov, 1e-3))
            us = {a: statistics.median(v) / nst for a, v in ts.items()}
            clk = {}
            for a in arms:
                vals = []
                for _ in range(3):
                    marks.zero_()
                    with torch.cuda.stream(timer.stream):
                        if mode == "cold":
                            timer.flush()
                        timer.gate()
                        fn(a, mk=marks)()
                    torch.cuda.synchronize()
                    mm = marks.view(G, 8).cpu().double()
                    vals.append(float((mm[:, 6] - mm[:, 5]).median()) / nst)
                clk[a] = statistics.median(vals)
            nm = lambda a: f"{a[0]}{'i' if a[0] in ('w72', 'w112') and a[1] == 0 else ''}"  # noqa: E731
            row = {"N": N, "mode": mode, "us_per_stage": {nm(a): us[a] for a in arms}, "clk_per_stage": {nm(a): clk[a] for a in arms},
                   "check": {nm(a): chk[a] for a in arms}, "us_all": {nm(a): ts[a] for a in arms}}
            res["rows"].append(row)
            log(f"[w128] N{N:3d} {mode}: us/stage " + " ".join(f"{nm(a)} {us[a]:.3f}" for a in arms) + " | clk/stage "
                + " ".join(f"{nm(a)} {clk[a]:.0f}" for a in arms) + " | ok " + " ".join(
                    f"{nm(a)} {chk[a]['ok']}{'' if chk[a]['bitwise_first'] else '(neq)'} rel {chk[a]['rel_fp64']:.1e}" for a in arms))
            out_p.write_text(json.dumps(res, indent=1, default=str))
            del gr, outs, P
            torch.cuda.empty_cache()
    for mode in ("cold", "warm"):
        log(f"[w128] SUMMARY {mode} ns / stage / token: " + " ".join(f"N{r['N']} " + ",".join(f"{k} {1e3 * v / r['N']:.1f}" for k, v in
            r["us_per_stage"].items()) for r in res["rows"] if r["mode"] == mode))
    for mode in modes:   # one w112 pass of n tokens vs the VML's even split into two w72 jobs of n / 2 (rounded up to 8)
        u = {(r["N"], k): v for r in res["rows"] if r["mode"] == mode for k, v in r["us_per_stage"].items()}
        log(f"[w128] SUMMARY {mode} us / stage: " + " ".join(
            f"n{n} w112i {u[(n, 'w112i')]:.3f} vs 2 x w72i N{h} {2 * u[(h, 'w72i')]:.3f} ({u[(n, 'w112i')] - 2 * u[(h, 'w72i')]:+.3f})"
            for n, h in ((80, 40), (96, 48), (112, 56)) if (n, 'w112i') in u and (h, 'w72i') in u) + " | same N w112i - w72i: " + " ".join(
            f"N{n} {u[(n, 'w112i')] - u[(n, 'w72i')]:+.3f}" for n in (8, 32, 56) if (n, 'w112i') in u and (n, 'w72i') in u))
    log("[w128] SUMMARY us / stage-equivalent (N 128 stage / 2 vs N 64):")
    for mode in ("cold", "warm"):
        r64 = next((r for r in res["rows"] if r["N"] == 64 and r["mode"] == mode), None)
        r128 = next((r for r in res["rows"] if r["N"] == 128 and r["mode"] == mode), None)
        if r64 and r128:
            u = r64["us_per_stage"]
            log(f"  {mode}: two N64 passes w72i {2 * u.get('w72i', float('nan')):.3f} w72 {2 * u.get('w72', float('nan')):.3f} "
                f"pl2 {2 * u.get('pl2', float('nan')):.3f} | one N128 pass pl2 {r128['us_per_stage']['pl2']:.3f} us")
    log(f"[w128] wrote {out_p} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    sys.exit(gate_main(sys.argv[2:]) if cmd == "gate" else run_main(sys.argv[2:]))   # run [--one=1] for sanitizer passes

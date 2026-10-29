"""dr_probe.py: one-command H200 stage probe for the T32 double-row VML variant (VM_DR, tools/dr_patches.py;
side_moe_20261028/RESEARCH.md "T32 double-row design", gate "compile-only first, then a stage probe").

    python3 -u kernels/lab/dr_probe.py [--arms=inc,ctl,dr,drp,incc,drc,drpc] [--ntoks=8,16,24,32] [--rounds=15]
                                       [--units=3] [--warm-set=8] [--full] [--out=box_out/dr/probe.json]

Compared (phase-A / W13 stages; the consumer code is the shipped one, compiled from k3_gc_6 + the VM_DR patches):
  ONE DR stage  = 512 weight rows x 256 K x N tokens: one 73,728 B weight slot (NS 2), one x-slice copy, every WG two
                  64-row tiles (seg_dr_acc = stage_mdr for N <= 16, stage_dr RS-wgmma for N 24 / 32);
  TWO incumbent = the same 512 rows as units 2U and 2U + 1 back to back: two 36,864 B slots (NS 3) and the same x slice
  stages           copied twice (stage_m / stage<N> exactly as bf2's seg<PB, N>).
Every CTA runs whole 16-stage units (K = 4096) of a fixed (expert, 512-row block, N tokens) list; all arms share the list,
the weights (MoE-board weights, experts 0..31, exact UNP-L2 bit layout + K2A stage layout, the exact unpack) and the
packed activations, and each writes FP32 sums x global scale per (row, token): outputs are comparable bit for bit.
Arms: inc  = bf2 (k3_gc_6's T <= 48 VML: 576 threads, 96 regs, NS 3 x 36 KB, NX 3 x 32 KB)
      ctl  = inc built again as a separate module (identical code): the arm-order / run-to-run noise floor
      dr   = VM_DR, K2A weight layout (4 bulk copies per stage);  drp = VM_DR + VM_DRP repack (one 72 KB copy per stage)
      incc / drc / drpc = the same + setmaxnreg split (WG5 2: producer WG 64, consumer WGs 104 regs; drc / drpc are the
      only spill-free DR builds, tools/dr_ptxas.py)
Modes: cold = each CTA its own units (3 x 132 = 396 distinct 1.18 MB blocks, ~470 MB), L2 flushed (256 MB read) before
       every replay -> weights from HBM;  warm = all CTAs on --warm-set blocks (8 -> 9.4 MB), an untimed replay right
       before every timed one -> weights from L2 (the consumer-bound regime).
Checks per arm and N: err flag, every output vs an FP64 reference (dequantized NVFP4 x bf16 tokens) on sampled units,
padded-token columns zero, and bitwise equality with inc on ALL units (DR keeps each tile's k16 order: expected equal).
Timing: CUDA graphs, rotating arm order, median of --rounds replays (minus the replay overhead); stage_us = per CTA per
512-row stage (= two incumbent stages); also the median CTA consumer span from in-kernel globaltimer marks and clock.
--full: the complete MoE kernels (bf2 vs dr / drc [/ drp / drpc]) on generate_input routings bs 25 / 32 / 48, normal +
skewed, seeds 1..5: FP32-reference check (rel < 1e-2, worst row < 5e-2), rel vs bf2 < 2e-3, err flag, two calls; median
cold replay time per variant.
Prints one [drp] line per (N, mode) and a decision summary; JSON with everything to --out.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "tools", ROOT / "tools/judge", ROOT / "kernels/muse_moe_w4a16", ROOT / "kernels/lab"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ARMS = ("inc", "ctl", "dr", "drp", "incc", "drc", "drpc")
_ARM_VAR = {"inc": "bf2", "ctl": "bf2", "dr": "dr", "drp": "drp", "incc": "bf2c", "drc": "drc", "drpc": "drpc"}
DRP_ARMS = ("drp", "drpc")
HOST_MARK = "// DRP_HOST"

PROBE_CU = r"""
namespace vm {
struct PArgs {
const uint8_t* ws;   // weights: K2A stage layout [E][64][16][SW], or the VM_DRP repack [E][16][16][4 x SW]
const float* gsc;    // per expert FP32 global scale (x W_SHIFT)
const uint8_t* xp;   // packed activations, per job [16 stages][npad x 512 B] (the xA / K2A xw layout)
const int4* unit;    // (expert, 512-row block U, job x byte offset, 0)
const int2* cta;     // [u0, u1) per CTA
float* out;          // [unit][512 prepacked rows][npad] = FP32 sum x gs
long long* marks;    // per CTA 8: gtime start, gtime end per WG (1..4), clock start / end (5, 6)
int npad;
};
#if VM_DR
constexpr int DRP_NH = 1;
#else
constexpr int DRP_NH = 2;
#endif
__device__ void drp_prod_w(const Args& A, const PArgs& P, uint32_t ring, uint32_t bars, int u0, int u1) {
const uint64_t pol = policy_evict_first();
int it = 0;
for (int u = u0; u < u1; ++u) {
const int4 un = P.unit[u];
for (int h = 0; h < DRP_NH; ++h)
for (int s = 0; s < 16; ++s, ++it) {
const int sl = it % NS;
if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
mbar_expect_tx(fb, US);
#if VM_DR && VM_DRP
bulk_g2s_ef(dst, P.ws + ((size_t)(un.x * 16 + un.y) * 16 + s) * US, US, fb, pol);
#elif VM_DR
#pragma unroll
for (int b = 0; b < 4; ++b) bulk_g2s_ef(dst + b * SW, P.ws + ((size_t)(un.x * 64 + 4 * un.y + b) * 16 + s) * SW, SW, fb, pol);
#else
const int blk = un.x * 64 + 4 * un.y + 2 * h;
bulk_g2s_ef(dst, P.ws + ((size_t)blk * 16 + s) * SW, SW, fb, pol);
bulk_g2s_ef(dst + SW, P.ws + ((size_t)(blk + 1) * 16 + s) * SW, SW, fb, pol);
#endif
}
}
}
__device__ void drp_prod_x(const Args& A, const PArgs& P, uint32_t xring, uint32_t bars, int u0, int u1) {
const uint32_t bytes = (uint32_t)P.npad * 512u;
int it = 0;
for (int u = u0; u < u1; ++u) {
const uint8_t* src = P.xp + (size_t)P.unit[u].z;
for (int h = 0; h < DRP_NH; ++h)
for (int s = 0; s < 16; ++s, ++it) {
const int sl = it % NX;
if (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * (2 * NS + sl);
mbar_expect_tx(fb, bytes);
bulk_g2s(xring + sl * XS, src + (size_t)s * bytes, bytes, fb);
}
}
}
// one warp-tile (16 rows: local warp-tile tl of the 512-row block) of FP32 sums x gs, the epilogue's accumulator layout
template <int N>
__device__ __forceinline__ void drp_store(const PArgs& P, int u, int tl, int lane, const float* d, float gs) {
const int g = lane >> 2, t = lane & 3;
float* o = P.out + ((size_t)u * 512 + 16 * tl + g) * P.npad;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
o[n] = d[4 * nt + hh] * gs;
o[(size_t)8 * P.npad + n] = d[4 * nt + 2 + hh] * gs;
}
}
#if !VM_DR
// bf2's seg<PB, N> accumulation part, verbatim order: stage_m chains for N <= 16 (VM_SYNC7), RS-wgmma stage<N> otherwise
template <int N>
__device__ __forceinline__ void drp_seg_inc(const Args& A, uint8_t* smem, uint32_t bars, int& it, float* d, int q, int wp,
int lane) {
Pend pd{-1};
if constexpr (N <= 16) {
float d2[2][N / 2];
#pragma unroll
for (int k = 0; k < N / 2; ++k) { d2[0][k] = 0.f; d2[1][k] = 0.f; }
stage_m<N>(A, smem, bars, it, d2, q, wp, lane);
#pragma unroll 1
for (int s = 1; s < 16; ++s) stage_m<N>(A, smem, bars, it, d2, q, wp, lane);
#pragma unroll
for (int k = 0; k < N / 2; ++k) d[k] = d2[0][k] + d2[1][k];
} else {
uint32_t hb[2][8];
int prevx = -1;
stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, pd);
#pragma unroll 1
for (int s = 1; s < 16; ++s) stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, pd);
wg_wait<0>();
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
}
}
#endif
template <int N>
__global__ void __launch_bounds__(NTH, 1) drp_kernel(const Args A, const PArgs P) {
extern __shared__ __align__(1024) uint8_t smem[];
const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
const uint32_t bars = sa(smem + BAR_OFF);
if (tid == 0) {
for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW); }
for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }
asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__syncthreads();
const int2 rg = P.cta[blockIdx.x];
if (warp >= NCW) {
#if VM_WG5 >= 2
asm volatile("setmaxnreg.dec.sync.aligned.u32 " VM_S(VM_PREG) ";\n" ::: "memory");
#endif
if (lane == 0) {
if (warp == NCW) drp_prod_w(A, P, sa(smem), bars, rg.x, rg.y);
else if (warp == NCW + 1) drp_prod_x(A, P, sa(smem + NS * US), bars, rg.x, rg.y);
}
return;
}
#if VM_WG5 >= 2
asm volatile("setmaxnreg.inc.sync.aligned.u32 " VM_S(VM_CREG) ";\n" ::: "memory");
#endif
const int q = warp >> 2, wp = warp & 3;
long long* mk = P.marks ? P.marks + (size_t)blockIdx.x * 8 : nullptr;
if (mk && tid == 0) { mk[0] = gtime(); mk[5] = clock64(); }
int it = 0;
#pragma unroll 1
for (int u = rg.x; u < rg.y; ++u) {
const float gs = __ldg(P.gsc + P.unit[u].x);
#if VM_DR
float d0[N / 2], d1[N / 2];
seg_dr_acc<N>(A, smem, bars, it, 0, 16, d0, d1, q, wp, lane);
drp_store<N>(P, u, q * 8 + wp, lane, d0, gs);
drp_store<N>(P, u, q * 8 + 4 + wp, lane, d1, gs);
#else
#pragma unroll 1
for (int h = 0; h < 2; ++h) {
float d[N / 2];
drp_seg_inc<N>(A, smem, bars, it, d, q, wp, lane);
drp_store<N>(P, u, (2 * h + (q >> 1)) * 8 + (q & 1) * 4 + wp, lane, d, gs);
}
#endif
}
if (mk && wp == 0 && lane == 0) { mk[1 + q] = gtime(); if (q == 0) mk[6] = clock64(); }
}
template __global__ void drp_kernel<8>(const Args A, const PArgs P);
template __global__ void drp_kernel<16>(const Args A, const PArgs P);
template __global__ void drp_kernel<24>(const Args A, const PArgs P);
template __global__ void drp_kernel<32>(const Args A, const PArgs P);
}  // namespace vm
""" + HOST_MARK + r"""
template <int N>
static void drp_launch(const vm::Args& a, const vm::PArgs& p, int G) {
  cudaFuncSetAttribute(vm::drp_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);
  vm::drp_kernel<N><<<G, vm::NTH, vm::SMEM, at::cuda::getCurrentCUDAStream()>>>(a, p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int N>
static std::vector<int64_t> drp_attr() {
  cudaFuncSetAttribute(vm::drp_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, vm::drp_kernel<N>);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, vm::drp_kernel<N>, vm::NTH, vm::SMEM);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)vm::SMEM, (int64_t)vm::NTH, occ, VM_DR, VM_DRP, vm::NS, vm::US, vm::XS};
}
std::vector<int64_t> drp_attrs(int64_t npad) {
  switch (npad) {
    case 8: return drp_attr<8>();
    case 16: return drp_attr<16>();
    case 24: return drp_attr<24>();
    case 32: return drp_attr<32>();
    default: return {-1};
  }
}
void drp_run(torch::Tensor ws, torch::Tensor gsc, torch::Tensor xp, torch::Tensor unit, torch::Tensor cta, torch::Tensor out,
             torch::Tensor err, torch::Tensor marks, int64_t npad, int64_t G) {
  vm::Args a{};
  a.err = err.data_ptr<int>();
  a.flags = 193;
  vm::PArgs p{};
  p.ws = reinterpret_cast<const uint8_t*>(ws.data_ptr()); p.gsc = gsc.data_ptr<float>();
  p.xp = reinterpret_cast<const uint8_t*>(xp.data_ptr()); p.unit = reinterpret_cast<const int4*>(unit.data_ptr());
  p.cta = reinterpret_cast<const int2*>(cta.data_ptr()); p.out = out.data_ptr<float>();
  p.marks = marks.numel() >= G * 8 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.npad = (int)npad;
  switch (npad) {
    case 8: drp_launch<8>(a, p, (int)G); break;
    case 16: drp_launch<16>(a, p, (int)G); break;
    case 24: drp_launch<24>(a, p, (int)G); break;
    case 32: drp_launch<32>(a, p, (int)G); break;
    default: TORCH_CHECK(false, "drp: npad must be 8 / 16 / 24 / 32");
  }
}
"""
PROBE_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> drp_attrs(int64_t npad);
void drp_run(torch::Tensor ws, torch::Tensor gsc, torch::Tensor xp, torch::Tensor unit, torch::Tensor cta, torch::Tensor out,
             torch::Tensor err, torch::Tensor marks, int64_t npad, int64_t G);
"""


# ---------------------------------------------------------------- sources (torch-free: tools/dr_ptxas.py imports these)
def vml_namespace(extra_globals=None):
    """k3_gc_6's VML Python block with the VM_DR patches applied, executed (needs torch only when its functions run)."""
    import dr_patches as DP  # noqa: PLC0415
    t = DP.patch_layer(DP.BASE.read_text(encoding="utf-8"))
    ns = dict(extra_globals or {})
    exec(compile("import hashlib, time\n" + DP.vml_block(t), "k3_gc_6+VM_DR:_vml_", "exec"), ns)   # noqa: S102
    cost = t[t.index("\nVML_COST = "):]
    ns["COST32"] = list(eval(cost[len("\nVML_COST = "):cost.index("\n", 1)])[32])   # noqa: S307 (our own literal)
    return ns


def probe_cuda(vml_cuda):
    """the VML CUDA up to its host entry point + the probe kernels + the probe's host functions."""
    return vml_cuda[:vml_cuda.index("void vml_run(")] + PROBE_CU


def arm_defines(arm):
    import dr_patches as DP  # noqa: PLC0415
    return tuple(DP.DEFS[_ARM_VAR[arm]]) + (("-DDRP_CTL=1",) if arm == "ctl" else ())


# ---------------------------------------------------------------- GPU harness
def arg(argv, k, d):
    for a in argv:
        if a.startswith(f"--{k}="):
            return a.split("=", 1)[1]
    return d


def build(arm, cuda, verbose=False):
    import hashlib
    import time
    from torch.utils.cpp_extension import load_inline
    fl = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"] + list(arm_defines(arm))
    src = probe_cuda(cuda)
    tag = hashlib.sha1((PROBE_CPP + src + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"dr_probe_{arm}_{tag}", cpp_sources=PROBE_CPP, cuda_sources=src,
                      functions=["drp_run", "drp_attrs"], extra_cuda_cflags=fl, verbose=verbose)
    return mod, time.time() - t0


def pack_x(x, toks, npad):
    """token rows (zero-padded to npad) in the xA / K2A xw wgmma-B layout per 256-K stage:
    [s][n8][i 4][q 8][r 8][t 4][b 2] for k = 256 s + 64 i + 16 t + 2 q + b -> uint8 [16 * npad * 512]."""
    import torch
    X = torch.zeros((npad, 4096), dtype=torch.bfloat16, device=x.device)
    X[:len(toks)] = x[torch.as_tensor(toks, dtype=torch.long, device=x.device)]
    X = X.view(npad // 8, 8, 16, 4, 4, 8, 2).permute(2, 0, 3, 5, 1, 4, 6).contiguous()
    return X.view(torch.uint8).reshape(-1)


def plan(N, mode, G, K, warm_set, x, dev, seed=11):
    """one job of N tokens per expert 0..31; units (e, U) of 512 rows; cold: K distinct units per CTA, warm: every CTA
    cycles through the first warm_set units of the shuffled list."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed + N)
    npad = -(-N // 8) * 8
    jb = 16 * npad * 512
    toks = [torch.randperm(x.shape[0], generator=g)[:N].tolist() for _ in range(32)]
    xp = torch.cat([pack_x(x, toks[e], npad) for e in range(32)]).contiguous()
    allu = [(e, U) for e in range(32) for U in range(16)]
    perm = torch.randperm(len(allu), generator=g).tolist()
    allu = [allu[i] for i in perm]
    n = G * K
    if mode == "cold":
        assert n <= len(allu), "cold plan needs distinct units: lower --units"
        us = allu[:n]
    else:
        us = [allu[i % warm_set] for i in range(n)]
    unit = torch.tensor([(e, U, e * jb, 0) for e, U in us], dtype=torch.int32).to(dev).contiguous()
    cta = torch.tensor([(c * K, (c + 1) * K) for c in range(G)], dtype=torch.int32).to(dev).contiguous()
    return {"N": N, "npad": npad, "toks": toks, "xp": xp, "unit": unit, "cta": cta, "units": us, "n": n}


def dequant_rows64(w, sf, gs, rows):
    """FP64 NVFP4 dequant of the given original rows: e2m1 x e4m3 block scale x global scale -> [len(rows), K]."""
    import torch
    grid = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float64, device=w.device)
    p = w[rows]
    codes = torch.stack((p & 15, p >> 4), dim=-1).flatten(-2)
    v = grid[(codes & 7).long()]
    v = torch.where((codes & 8) != 0, -v, v)
    k = v.shape[-1]
    s = sf[rows].to(torch.float64) * float(gs)
    return (v.view(-1, k // 16, 16) * s.unsqueeze(-1)).view(-1, k)


def main(argv=None):
    import concurrent.futures as cf
    import json
    import statistics
    import time

    import torch
    import judge as J
    import k2a_v37 as D
    import unp_patches as UNP

    argv = sys.argv[1:] if argv is None else argv
    log = lambda s: print(s, flush=True)  # noqa: E731
    arms = [a for a in str(arg(argv, "arms", ",".join(ARMS))).split(",") if a in ARMS]
    arms = ["inc"] + [a for a in arms if a != "inc"]   # inc first: every other arm is compared with its outputs
    ntoks = tuple(int(v) for v in str(arg(argv, "ntoks", "8,16,24,32")).split(","))
    assert all(n in (8, 16, 24, 32) for n in ntoks), "N must be 8 / 16 / 24 / 32"
    rounds, K, warm_set = int(arg(argv, "rounds", 15)), int(arg(argv, "units", 3)), int(arg(argv, "warm-set", 8))
    out_p = ROOT / str(arg(argv, "out", "box_out/dr/probe.json"))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ns = vml_namespace({"torch": torch})
    cuda = ns["_vml_CUDA"]
    mods = {}
    with cf.ThreadPoolExecutor(len(arms)) as ex:
        futs = {a: ex.submit(build, a, cuda) for a in arms}
        for a, f in futs.items():
            try:
                mods[a], dt = f.result()
                log(f"[drp] built {a} ({_ARM_VAR[a]}) in {dt:.0f} s; attrs N8 (regs, local B, smem, threads, occ, DR, DRP, NS, US, "
                    f"XS) {list(mods[a].drp_attrs(8))}, N32 {list(mods[a].drp_attrs(32))}")
            except Exception as e:  # noqa: BLE001
                log(f"[drp] build {a} FAILED: {e!r}"[:1500])
    if "inc" not in mods:
        raise SystemExit("[drp] the incumbent arm did not build")
    arms = [a for a in arms if a in mods]
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    prob = J.load_problem(ROOT / "problems/muse_moe_w4a16")
    ref = prob.ref
    w = ref.generate_weights(ref.WEIGHTS_SEED, device=dev)
    log(f"[drp] UNP L2 decode self-test: {UNP.selftest(20000)} words bit-exact vs BITPOS3")
    ws13 = torch.empty((32, 64, 16, D.SW3), dtype=torch.uint8, device=dev)
    for e in range(32):
        w13q, s13q = D.prepack_expert(w["w13"][e], w["w13_sf"][e], gate_up=True, bitpos=UNP.L2POS)
        ws13[e] = D.stage_repack(w13q, s13q)
    gsc13 = (w["w13_gs"][:32].float() * D.W_SHIFT).contiguous()
    ws13p = ns["_vml_dr_repack"](ws13) if any(a in DRP_ARMS for a in arms) else None
    if ws13p is not None:
        chk = bool(torch.equal(ws13p[3, 5, 7].view(4, D.SW3), ws13[3, 20:24, 7]))
        log(f"[drp] VM_DRP repack: {tuple(ws13p.shape)}, block (3, 5) stage 7 == K2A blocks 20..23 stage 7: {chk}")
    perm = D.tile_row_perm(8192, True, device=dev)
    xsrc = torch.randn((512, 4096), generator=torch.Generator(device="cpu").manual_seed(5), dtype=torch.float32).to(torch.bfloat16).to(dev)
    timer = J.GpuTimer()
    ov = timer.replay_overhead_us()
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    nomark = torch.zeros((0,), dtype=torch.int64, device=dev)
    marks = torch.zeros((G * 8,), dtype=torch.int64, device=dev)
    log(f"[drp] setup {time.time() - t0:.0f} s, {G} SMs, replay overhead {ov:.2f} us, arms {arms}, rounds {rounds}, "
        f"{K} units (= {16 * K} DR stages) per CTA")

    def timed(g, flush, warm):
        with torch.cuda.stream(timer.stream):
            if warm:
                g.replay()
            if flush:
                timer.flush()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1e3

    res = {"G": G, "overhead_us": ov, "arms": arms, "rounds": rounds, "units_per_cta": K, "warm_set": warm_set, "rows": []}
    for N in ntoks:
        for mode in ("cold", "warm"):
            P = plan(N, mode, G, K, warm_set, xsrc, dev)
            npad = P["npad"]
            outs = {a: torch.full((P["n"], 512, npad), float("nan"), dtype=torch.float32, device=dev) for a in arms}

            def fn(a, mk=nomark, P=P, outs=outs):
                wsa = ws13p if a in DRP_ARMS else ws13
                return lambda: mods[a].drp_run(wsa, gsc13, P["xp"], P["unit"], P["cta"], outs[a], err, mk, P["npad"], G)
            # correctness: err flag, FP64 reference on sampled units, zero padded columns, bitwise vs inc on all units
            chk = {}
            samp = sorted(set([0, P["n"] // 3, P["n"] // 2, (2 * P["n"]) // 3, P["n"] - 1]))
            refs = {}
            for i in samp:
                e, U = P["units"][i]
                rows = perm[512 * U:512 * U + 512]
                W = dequant_rows64(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e], rows)
                xt = xsrc[torch.as_tensor(P["toks"][e], dtype=torch.long, device=dev)].double()
                refs[i] = W @ xt.T
            for a in arms:
                err.zero_()
                fn(a)()
                torch.cuda.synchronize()
                o = outs[a]
                rel, mx = 0.0, 0.0
                for i, r in refs.items():
                    got = o[i, :, :N].double()
                    rel = max(rel, float((got - r).norm() / r.norm().clamp(min=1e-300)))
                    mx = max(mx, float((got - r).abs().max() / r.abs().max().clamp(min=1e-300)))
                padz = bool((o[:, :, N:] == 0).all()) if npad > N else True
                same = bool(torch.equal(o, outs["inc"])) if a != "inc" else True
                ndiff = int((o != outs["inc"]).sum()) if a != "inc" else 0
                chk[a] = {"err": int(err[0]), "rel_fp64": rel, "maxrel_fp64": mx, "pad_zero": padz, "finite": bool(torch.isfinite(o).all()),
                          "bitwise_inc": same, "ndiff_inc": ndiff,
                          "ok": int(err[0]) == 0 and rel < 1e-4 and padz and bool(torch.isfinite(o).all())}
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
            us = {a: statistics.median(v) for a, v in ts.items()}
            spread = {a: (min(v), max(v)) for a, v in ts.items()}
            span, ghz = {}, {}
            for a in arms:
                for _ in range(2):
                    marks.zero_()
                    with torch.cuda.stream(timer.stream):
                        if mode == "cold":
                            timer.flush()
                        fn(a, mk=marks)()
                    torch.cuda.synchronize()
                m = marks.view(G, 8).cpu().double()
                sp = (m[:, 1:5].max(dim=1).values - m[:, 0]) / 1e3
                span[a] = float(sp.median())
                cyc, nsx = m[:, 6] - m[:, 5], m[:, 1] - m[:, 0]
                okm = (cyc > 0) & (nsx > 0)
                ghz[a] = float((cyc[okm] / nsx[okm]).median()) if bool(okm.any()) else None
            stage = {a: us[a] / (16 * K) for a in arms}   # per CTA per 512-row stage (= two incumbent stages)
            wb = P["n"] * 16 * 4 * D.SW3
            row = {"N": N, "mode": mode, "us": us, "us_minmax": spread, "stage_us": stage, "span_us": span, "ghz": ghz,
                   "w_TBs": {a: wb / us[a] / 1e6 for a in arms}, "x_bytes_per_512row_stage": {"inc": 2 * npad * 512, "dr": npad * 512},
                   "check": chk}
            res["rows"].append(row)
            d_ctl = us.get("ctl", us["inc"]) - us["inc"]
            log(f"[drp] N={N:2d} {mode}: stage us " + " ".join(f"{a} {stage[a]:.3f}" for a in arms)
                + " | launch us " + " ".join(f"{a} {us[a]:.1f}" for a in arms)
                + f" | ctl-inc {d_ctl:+.2f} us | span us " + " ".join(f"{a} {span[a]:.1f}" for a in arms)
                + " | GHz " + " ".join(f"{a} {ghz[a]:.2f}" for a in arms if ghz[a])
                + " | rel64 " + " ".join(f"{a} {chk[a]['rel_fp64']:.1e}" for a in arms)
                + " | ==inc " + " ".join(f"{a} {chk[a]['bitwise_inc']}" for a in arms if a != "inc")
                + " | ok " + " ".join(f"{a} {chk[a]['ok']}" for a in arms))
            del gr, outs, P
            torch.cuda.empty_cache()
    # decision summary: a DR arm must beat inc by more than the identical-control gap, cold and warm, at every N
    log("[drp] ---- summary: per-512-row-stage delta vs inc (us, %), noise = |ctl - inc|")
    summ = {}
    for a in [x for x in arms if x not in ("inc", "ctl")]:
        base = "incc" if a in ("drc", "drpc") and "incc" in arms else "inc"
        ent = []
        for r in res["rows"]:
            dlt = r["stage_us"][a] - r["stage_us"][base]
            noise = abs(r["stage_us"].get("ctl", r["stage_us"]["inc"]) - r["stage_us"]["inc"])
            ent.append({"N": r["N"], "mode": r["mode"], "delta_us": dlt, "pct": 100 * dlt / r["stage_us"][base], "noise_us": noise,
                        "vs": base, "wins": dlt < -noise, "ok": r["check"][a]["ok"], "bitwise": r["check"][a]["bitwise_inc"]})
        summ[a] = ent
        log(f"[drp] {a} vs {base}: " + " ".join(f"N{x['N']}{x['mode'][0]} {x['delta_us']:+.3f} ({x['pct']:+.1f}%{'*' if x['wins'] else ''})"
                                        for x in ent)
            + f" | all checks ok {all(x['ok'] for x in ent)} bitwise {all(x['bitwise'] for x in ent)}")
    res["summary"] = summ
    if "--full" in argv:
        res["full"] = full_moe(argv, ns, w, ref, D, UNP, timer, ov, log)
    out_p.write_text(json.dumps(res, indent=1, default=str))
    log(f"[drp] wrote {out_p} ({time.time() - t0:.0f} s)")
    return 0


def full_moe(argv, ns, w, ref, D, UNP, timer, ov, log):
    """the complete MoE kernels on T = 25 / 32 / 48 routings: correctness vs FP32 / bf2 and cold replay time."""
    import concurrent.futures as cf
    import statistics

    import torch
    dev = "cuda"
    fvars = [v for v in str(arg(argv, "fullvars", "bf2,dr,drc,drp,drpc")).split(",") if v]
    if "bf2" not in fvars:
        fvars = ["bf2"] + fvars
    import dr_patches as DP
    mods = {}
    with cf.ThreadPoolExecutor(len(fvars)) as ex:
        futs = {v: ex.submit(ns["_vml_build"], False, DP.DEFS[v]) for v in fvars}
        for v, f in futs.items():
            try:
                mods[v], dt = f.result()
                log(f"[drp-full] built {v} in {dt:.0f} s cfg {list(mods[v].vml_cfg())}")
            except Exception as e:  # noqa: BLE001
                log(f"[drp-full] build {v} FAILED: {e!r}"[:1500])
    ws13 = torch.empty((32, 64, 16, D.SW3), dtype=torch.uint8, device=dev)
    ws2 = torch.empty((32, 32, 16, D.SW3), dtype=torch.uint8, device=dev)
    for e in range(32):
        a, b = D.prepack_expert(w["w13"][e], w["w13_sf"][e], gate_up=True, bitpos=UNP.L2POS)
        ws13[e] = D.stage_repack(a, b)
        a, b = D.prepack_expert(w["w2"][e], w["w2_sf"][e], gate_up=False, bitpos=UNP.L2POS)
        ws2[e] = D.stage_repack(a, b)
    k = {"ws13": ws13, "ws2": ws2, "gsc13": (w["w13_gs"][:32].float() * D.W_SHIFT).contiguous(),
         "gsc2": (w["w2_gs"][:32].float() * D.W_SHIFT).contiguous(),
         "nw": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous()}
    if any(v in DRP_ARMS for v in mods):
        k["dr_ws13"], k["dr_ws2"] = ns["_vml_dr_repack"](ws13), ns["_vml_dr_repack"](ws2)
    G = torch.cuda.get_device_properties(0).multi_processor_count
    sc = ns["_vml_alloc"](dev, G)
    cost = ns["COST32"]

    def run(v, d):
        kk = dict(k, ws13=k["dr_ws13"], ws2=k["dr_ws2"]) if v in DRP_ARMS else k
        return ns["_vml_call"](mods[v], kk, sc, d["hidden_states"], d["topk_ids"], d["topk_weights"], ls=0, flags=193, cost=cost)

    def rel(a, b):
        a, b = a.float(), b.float()
        tot = float((a - b).norm() / b.norm().clamp(min=1e-12))
        rn = b.norm(dim=-1)
        return tot, float(((a - b).norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-12)).max())
    rows = []
    rounds = int(arg(argv, "fullrounds", 9))
    for bs in (25, 32, 48):
        for dist in ("normal", "skewed"):
            for seed in range(1, 6):
                d = ref.generate_input(bs, "ep8", seed, dist, device=dev)
                d = dict(d, hidden_states=d["hidden_states"].contiguous(), topk_ids=d["topk_ids"].to(torch.int32).contiguous(),
                         topk_weights=d["topk_weights"].float().contiguous())
                r32 = ref.ref_kernel(d, w)
                ob, ent = None, {"bs": bs, "dist": dist, "seed": seed}
                cnt = torch.bincount(d["topk_ids"].flatten().long().clamp(min=0), minlength=256)[:32]
                ent["max_expert_tokens"] = int(cnt.max())
                for v in mods:
                    o2 = []
                    for _ in range(2):
                        sc["err"].zero_()
                        o2.append(run(v, d).clone())
                        torch.cuda.synchronize()
                    tot, row = rel(o2[1], r32)
                    rb = rel(o2[1], ob)[0] if ob is not None else 0.0
                    rep = rel(o2[1], o2[0])[0]
                    ok = (tot < 1e-2 and row < 5e-2 and rb < 2e-3 and not int(sc["err"][0]) and bool(torch.isfinite(o2[1].float()).all()))
                    if v == "bf2":
                        ob = o2[1]
                    g_ = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(timer.stream):
                        run(v, d)
                        torch.cuda.synchronize()
                        with torch.cuda.graph(g_, stream=timer.stream):
                            run(v, d)
                    torch.cuda.synchronize()
                    ent[v] = {"rel_fp32": tot, "row": row, "rel_bf2": rb, "replay_rel": rep, "err": int(sc["err"][0]), "ok": ok, "g": g_}
                tsv = {v: [] for v in mods}
                names = list(mods)
                for rd in range(rounds):
                    kk = rd % len(names)
                    for v in names[kk:] + names[:kk]:
                        tsv[v].append(max(timer.time_replay(ent[v]["g"]) - ov, 1e-3))
                for v in mods:
                    ent[v]["us"] = statistics.median(tsv[v])
                    del ent[v]["g"]
                rows.append(ent)
                log(f"[drp-full] bs={bs} {dist} seed {seed} maxE {ent['max_expert_tokens']}: us "
                    + " ".join(f"{v} {ent[v]['us']:.1f}" for v in mods)
                    + " | rel32 " + " ".join(f"{v} {ent[v]['rel_fp32']:.2e}" for v in mods)
                    + " | vs bf2 " + " ".join(f"{v} {ent[v]['rel_bf2']:.1e}" for v in mods if v != "bf2")
                    + " | ok " + " ".join(f"{v} {ent[v]['ok']}" for v in mods))
    for v in [x for x in mods if x != "bf2"]:
        dl = [r[v]["us"] - r["bf2"]["us"] for r in rows]
        log(f"[drp-full] {v} - bf2: median {statistics.median(dl):+.2f} us, worst {max(dl):+.2f}, best {min(dl):+.2f}, "
            f"faster {sum(x < 0 for x in dl)}/{len(dl)}, all ok {all(r[v]['ok'] for r in rows)}")
    return rows


if __name__ == "__main__":
    sys.exit(main())

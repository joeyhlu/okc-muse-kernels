# okc: problem=muse_bf16_proj
# LAB FILE — check runs only, never submit for score
# nonce stream-lab-1
"""Streaming-primitive lab (check runs only). kernel_fn and everything the judge times are SGLang's baseline.py,
byte for byte, so the run passes; prepare() runs the baseline's prepare first, then (untimed, < ~200 s) compiles
ONE load_inline extension and measures on the judge's H200 what is still unknown after lab M1-M5 (job
7516684e37bb45d1: plain LDG reaches 4.29-4.48 TB/s on a 1 GiB stream with >= 32 KiB/SM in flight; random 128 B rows
3.67 TB/s, 64 B rows 2.24):

  A  1 GiB contiguous stream: S1 plain LDG.128 (reference), S2 cp.async 16 B ring, S3 cp.async.bulk + mbarrier ring
  B  whole 9-64 MiB matrices under CUDA-graph replay with a cold L2 (expert W2/W13 x1-3, qkv, pooler)
  B2 2-3 experts of 18.9 MB over 132 SMs: N-split vs K-split (2, 4) and the split-K reduction (fp32 atomics fused
     into the stream kernel, or a second reduce kernel in the graph)
  C  random 64/128/256 B rows through an int32 index from a 1.34 GB table, deeper in flight, LDG vs cp.async vs bulk
  D  streaming WHILE consuming like an NVFP4 GEMM: Marlin LOP3 E2M1->BF16 dequant + BF16 scale + mma.sync
     m16n8k16 per 16 B, on 1 GiB and on 3 experts (W13x3), with regs/local per kernel
  E  S4 TMA 2-D tensor-map boxes (cuTensorMapEncodeTiled via cudaGetDriverEntryPointByVersion), last
How to read the output: kernels/lab/stream_lab.md.
"""

from typing import TypedDict

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

import hashlib
import time


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]; up to 8 tokens, only rows 0-1023 (this GPU's slice)
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [M, K]
    config: Config


# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    if LAB_ENABLED:
        t0 = time.time()
        try:
            _sl_run(t0)
        except Exception as e:  # the lab must never break the baseline run
            print("SL LAB ERR " + _sl_err(e), flush=True)
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception as e:
            print("SL cleanup ERR " + _sl_err(e), flush=True)
    return state


def kernel_fn(data, state):
    x, layer = data["x"], data["config"]["layer"]
    if layer != "router":
        # The unpooler's slice is its first N rows (all of them above 8 tokens).
        return _bf16_gemm_dispatch_impl(x, state[layer][: data["config"]["n"]], None)
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


# =====================================================================================================================
# LAB. Everything below runs only inside prepare() (untimed). Nothing touches the harness, the inputs or the weights.
# =====================================================================================================================
LAB_ENABLED = True
LAB_DEADLINE_S = 190.0        # no new lab work starts after this many seconds of prepare (prepare limit 300 s)
LAB_BUILD_RETRY_BY_S = 100.0  # a failed build is retried without S4 only before this many seconds
LAB_S4 = True                 # TMA 2-D tensor-map part (compiled in; runs last, after everything else has printed)
LAB_VERBOSE = False           # True: one extra line per measurement (output ~3x; the log tail keeps ~16 KB)
LAB_SPIN_BUDGET = 1 << 25     # clock64 cycles per mbarrier wait before the kernel aborts it (~17 ms at 1.98 GHz)
LAB_REPS_BIG = 3              # timed launches per config on the big streams (min kept)
LAB_REPS_1X = 25              # graph replays per small-matrix point, one launch per replay (median kept)
LAB_REPS_KX = 8               # graph replays per small-matrix point, K launches on K cold copies per replay

_SL_MIB = 1 << 20
_SL_PEAK = 4.8e12
_SL_DEVICE = "cuda"
_SL_BIG = 1280 * _SL_MIB      # one random-filled buffer: the A stream, the B matrix pool and the C gather table
_SL_A_BYTES = 1 << 30         # A/D: contiguous stream
_SL_FLUSH = 256 * _SL_MIB     # read before every timed launch/replay (L2 is 60 MiB on the judge's H200)
_SL_G_BYTES = 256 * _SL_MIB   # C: gathered row bytes per launch
_SL_G_REGION_SMALL = 320 * 1000 * 1000
# B: name, matrices, rows per matrix, row bytes. W13/W2 = one NVFP4 expert (4096 values = 2048 B + 256 B scales per
# row); qkv/pool = the BF16 qkv_gate (36 MiB) / pooler (64 MiB) weights as 4 KiB rows; ctl = a 256 MiB control.
_SL_SMALL = (("W2x1", 1, 4096, 2304), ("W13x1", 1, 8192, 2304), ("W13x2", 2, 8192, 2304),
             ("W13x3", 3, 8192, 2304), ("qkv36M", 1, 9216, 4096), ("pool64M", 1, 16384, 4096),
             ("ctl256M", 1, 65536, 4096))
_SL_KMAX = 8
_SL_INFL_BINS = (8, 16, 24, 32, 48, 64, 128, 256)

# ------------------------------------------------- CUDA source -------------------------------------------------------
_SL_BLOCK = r"""
// ===== BEGIN okc stream_block v1 =====
// sm_90 streaming ring: cp.async.bulk (1-D, no tensor map) or cp.async.bulk.tensor.2d into `stages` smem slots,
// full/empty mbarrier pair per slot, bounded warp-uniform waits. Requires sm_90 (PTX ISA >= 8.0); no sm_90a.
//
// Protocol (verified on CPU by the lab's emulate.py; every wait below is bounded):
//   full[s]  init count 1  : the producer's single arrive.expect_tx per fill, plus the fill's tx bytes
//   empty[s] init count NC : one arrive per consumer warp per use
//   chunk k -> slot s = k % stages, use n = k / stages
//   consumer: wait full[s]  for parity (n & 1)        == phase n of full[s] completed (fill n landed)
//   producer: wait empty[s] for parity ((n - 1) & 1)  == phase n-1 of empty[s] completed (use n-1 released), n >= 1
//             (n == 0: fresh slot, no wait)
//   The tested phase is never ambiguous: full[s] is in phase n or n+1 when a consumer tests it, empty[s] in phase
//   n-1 or n when the producer tests it (the next completion needs this warp's own progress).
//   expect_tx MUST equal the sum of the bytes of the copies that complete on that barrier for that fill, or the
//   phase never completes (the wait then times out, sets *flag and returns false).
//
// PTX ISA (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html):
//   mbarrier objects, phases, tx-count, init/arrive/expect_tx/try_wait:
//     #parallel-synchronization-and-communication-instructions-mbarrier            (anchor from memory)
//   fence.mbarrier_init:  #parallel-synchronization-and-communication-instructions-membar   (cited in repos/)
//   cp.async.bulk:        #data-movement-and-conversion-instructions-cp-async-bulk          (cited in repos/)
//   cp.async.bulk.tensor: #data-movement-and-conversion-instructions-cp-async-bulk-tensor   (cited in repos/)
// Instruction strings match CUTLASS (repos/cutlass): cutlass/arch/barrier.h:405 (init), :478 (try_wait.parity),
// :545 (arrive), :630 (arrive.expect_tx), :776 (fence.mbarrier_init); cute/arch/copy_sm90_tma.hpp:1448
// (bulk G2S); copy_sm90_tma.hpp:125 (tensor 2d, used here without the optional .L2::cache_hint operand).
#ifndef OKC_STREAM_BLOCK_V1
#define OKC_STREAM_BLOCK_V1
namespace okc_stream {

constexpr int kMaxStages = 8;

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------- PTX wrappers ----------------
// mbarrier.init: expected arrivals `count`, phase 0, tx-count 0.
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// Makes the mbarrier inits visible to the async proxy (bulk-copy complete_tx) before any copy targets them.
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
// One arrival, and tx-count += bytes (release, cta scope).
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
// One arrival (release, cta scope).
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
// 1 iff the phase with parity `parity` has completed (acquire, cta scope). May suspend for a HW-defined time
// limit, then returns 0; never blocks forever. "memory" keeps smem reads below it from being hoisted above it.
__device__ __forceinline__ uint32_t mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok;
}
// 1-D bulk copy global -> shared; completion = complete_tx(bytes) on `bar`. bytes % 16 == 0; src and dst 16 B
// aligned. A CTA is an implicit 1-CTA cluster, so its shared::cta address is valid as shared::cluster.
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// 2-D tensor-map copy of one box at element coords (c0 inner, c1 outer); tmap = address of a CUtensorMap in
// param/const/global space (e.g. &a __grid_constant__ kernel parameter). dst 128 B aligned. complete_tx counts the
// full box bytes (also for out-of-bounds boxes, which are zero-filled).
__device__ __forceinline__ void tma_g2s_2d(uint32_t dst, const void* tmap, uint32_t bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      :: "r"(dst), "l"(tmap), "r"(bar), "r"(c0), "r"(c1) : "memory");
}

// ---------------- bounded waits ----------------
struct Guard {
  int* flag;         // global int: 0 = ok; the first wait that times out sets it, every other wait then bails
  long long budget;  // clock64 cycles allowed per wait (1 << 25 ~ 17 ms at 1.98 GHz)
};

// Warp-collective (all 32 lanes, converged). Every lane polls (so every lane gets acquire ordering); returns true
// once all lanes saw the phase complete, false (warp-uniform) on timeout or when *flag was set elsewhere.
__device__ __forceinline__ bool wait_warp(uint32_t bar, uint32_t parity, const Guard& g) {
  const long long t0 = clock64();
  for (uint32_t it = 0;; ++it) {
    const uint32_t ok = mbar_try_wait(bar, parity);
    if (__all_sync(0xffffffffu, ok)) return true;
    if ((it & 7u) == 7u) {
      int stop = 0;
      if ((threadIdx.x & 31u) == 0u) {
        stop = (*reinterpret_cast<volatile int*>(g.flag) != 0) || (clock64() - t0 > g.budget);
        if (stop) atomicOr(g.flag, 1);
      }
      if (__shfl_sync(0xffffffffu, stop, 0)) return false;
    }
  }
}

// ---------------- ring ----------------
struct Ring {
  uint8_t* data;        // generic pointer to slot 0 (128 B aligned)
  uint32_t data_s;      // shared address of slot 0
  uint32_t full_s;      // shared address of full[0]  (full[s]  at full_s  + 8*s)
  uint32_t empty_s;     // shared address of empty[0] (empty[s] at empty_s + 8*s)
  uint32_t slot_bytes;  // bytes per slot (multiple of 16; of 128 for tensor copies)
  int stages;           // 1 .. kMaxStages
};

// bars: __shared__ uint64_t[2 * kMaxStages] (8 B aligned); data: stages * slot_bytes bytes of smem, 128 B aligned.
__device__ __forceinline__ Ring ring_make(uint8_t* data, uint64_t* bars, int stages, uint32_t slot_bytes) {
  Ring r;
  r.data = data;
  r.data_s = smem_addr(data);
  r.full_s = smem_addr(bars);
  r.empty_s = smem_addr(bars + kMaxStages);
  r.slot_bytes = slot_bytes;
  r.stages = stages;
  return r;
}
// Thread 0 only; then the whole CTA must __syncthreads() before any producer/consumer starts.
__device__ __forceinline__ void ring_init(const Ring& r, uint32_t consumer_warps) {
  for (int s = 0; s < r.stages; ++s) {
    mbar_init(r.full_s + 8u * (uint32_t)s, 1u);
    mbar_init(r.empty_s + 8u * (uint32_t)s, consumer_warps);
  }
  fence_mbar_init();
}
__device__ __forceinline__ uint32_t ring_slot(const Ring& r, int k) { return (uint32_t)(k % r.stages); }
__device__ __forceinline__ uint32_t ring_use(const Ring& r, int k) { return (uint32_t)(k / r.stages); }
__device__ __forceinline__ uint32_t ring_slot_s(const Ring& r, int k) { return r.data_s + ring_slot(r, k) * r.slot_bytes; }
__device__ __forceinline__ uint32_t ring_full_s(const Ring& r, int k) { return r.full_s + 8u * ring_slot(r, k); }

// Producer warp, warp-collective: true when chunk k's slot may be overwritten (consumers released chunk k-stages).
__device__ __forceinline__ bool producer_acquire(const Ring& r, int k, const Guard& g) {
  if (k < r.stages) return true;
  return wait_warp(r.empty_s + 8u * ring_slot(r, k), (ring_use(r, k) - 1u) & 1u, g);
}
// One lane: announce chunk k's total bytes (= the sum of the copies that follow for chunk k) and arrive once.
__device__ __forceinline__ void producer_expect(const Ring& r, int k, uint32_t total_bytes) {
  mbar_arrive_expect_tx(ring_full_s(r, k), total_bytes);
}
// Any lane(s), after producer_expect: copy `bytes` from global `src` to byte offset dst_off of chunk k's slot.
__device__ __forceinline__ void producer_copy(const Ring& r, int k, uint32_t dst_off, const void* src, uint32_t bytes) {
  bulk_g2s(ring_slot_s(r, k) + dst_off, src, bytes, ring_full_s(r, k));
}
// One lane: a whole contiguous chunk in one copy.
__device__ __forceinline__ void producer_put(const Ring& r, int k, const void* src, uint32_t bytes) {
  producer_expect(r, k, bytes);
  producer_copy(r, k, 0u, src, bytes);
}
// One lane: one 2-D tensor box as chunk k (box_bytes = the box size in bytes).
__device__ __forceinline__ void producer_put_tile(const Ring& r, int k, const void* tmap, int c0, int c1, uint32_t box_bytes) {
  producer_expect(r, k, box_bytes);
  tma_g2s_2d(ring_slot_s(r, k), tmap, ring_full_s(r, k), c0, c1);
}
// Consumer warp, warp-collective: wait for chunk k; returns its slot, or nullptr on timeout/abort (leave the loop).
__device__ __forceinline__ const uint8_t* consumer_wait(const Ring& r, int k, const Guard& g) {
  if (!wait_warp(r.full_s + 8u * ring_slot(r, k), ring_use(r, k) & 1u, g)) return nullptr;
  return r.data + (size_t)ring_slot(r, k) * r.slot_bytes;
}
// Consumer warp, warp-collective: this warp is done reading chunk k (orders the warp's reads before the arrive).
__device__ __forceinline__ void consumer_release(const Ring& r, int k) {
  __syncwarp();
  if ((threadIdx.x & 31u) == 0u) mbar_arrive(r.empty_s + 8u * ring_slot(r, k));
}

}  // namespace okc_stream
#endif  // OKC_STREAM_BLOCK_V1
// ===== END okc stream_block v1 =====
"""

_SL_KERNELS = r"""
namespace sl {

// Launch parameters shared by every scheme (passed by value).
struct Params {
  const uint8_t* mat0; const uint8_t* mat1; const uint8_t* mat2; const uint8_t* mat3;  // contig bases / gather table
  const int* idx;              // gather: int32 row indices into mat0
  unsigned long long* sink;    // [0..1023] checksum slots (per-warp sums, slot = global warp & 1023), [1024] dummy
  int* flag;                   // bounded-wait abort flag (okc_stream::Guard)
  float* red;                  // optional split-K epilogue model: one fp32 atomicAdd per thread into red[0..red_n)
  long long rows;              // contig/k-split: rows per matrix; gather: number of idx entries
  long long table_rows;        // gather: rows in the table (clamp only)
  long long budget;            // clock64 cycles per mbarrier wait
  int nmat, rb, chunk, interleave, stages, work, lanes, v, box_w, box_h, ksplit, red_n;
};

__device__ __forceinline__ const uint8_t* mat_ptr(const Params& P, int m) {
  return m == 0 ? P.mat0 : (m == 1 ? P.mat1 : (m == 2 ? P.mat2 : P.mat3));
}

// ---- contiguous work split (MODE 0): nmat matrices of rows x rb bytes, chunks of `chunk` bytes ----
//  interleave = 0 (N-split): CTAs [f(m), f(m+1)) own matrix m, f(m) = ceil(m*G/nmat); CTA j of that group owns rows
//                            [j*rows/gs, (j+1)*rows/gs) and streams them as one contiguous byte range.
//  interleave = 1          : chunk g of the concatenated chunk list (cpm chunks per matrix) goes to CTA g % G.
struct CIt {
  long long b0, b1, n, cpm, mat_bytes;
  int m, chunk, inter, c, G;
};
__device__ __forceinline__ CIt c_init(const Params& P, long long rows, int rb, int chunk) {
  CIt it;
  it.chunk = chunk; it.inter = P.interleave; it.c = (int)blockIdx.x; it.G = (int)gridDim.x;
  it.mat_bytes = rows * (long long)rb;
  it.cpm = (it.mat_bytes + chunk - 1) / chunk;
  it.m = 0; it.b0 = 0; it.b1 = 0;
  if (it.inter) {
    const long long total = it.cpm * P.nmat;
    it.n = (it.c < total) ? (total - 1 - it.c) / it.G + 1 : 0;
  } else {
    const int m = (int)(((long long)it.c * P.nmat) / it.G);
    const long long f0 = ((long long)m * it.G + P.nmat - 1) / P.nmat;
    const long long f1 = ((long long)(m + 1) * it.G + P.nmat - 1) / P.nmat;
    const long long gs = f1 - f0, j = it.c - f0;
    const long long r0 = j * rows / gs, r1 = (j + 1) * rows / gs;
    it.m = m; it.b0 = r0 * rb; it.b1 = r1 * rb;
    it.n = (it.b1 - it.b0 + chunk - 1) / chunk;
  }
  return it;
}
// chunk k of this CTA -> matrix m, byte offset `off` inside it, `bytes` (> 0, multiple of 16)
__device__ __forceinline__ void c_chunk(const CIt& it, long long k, int& m, long long& off, int& bytes) {
  long long rem;
  if (it.inter) {
    const long long g = (long long)it.c + k * it.G;
    m = (int)(g / it.cpm);
    off = (g - (long long)m * it.cpm) * it.chunk;
    rem = it.mat_bytes - off;
  } else {
    m = it.m;
    off = it.b0 + k * it.chunk;
    rem = it.b1 - off;
  }
  bytes = (int)(rem < it.chunk ? rem : (long long)it.chunk);
}

// ---- row-list work split, chunks of cr rows of `rowbytes` each ----
//  MODE 1 gather : rows = idx entries; row r lives at mat0 + idx[r]*rb; interleave 0: CTA c owns [c*N/G, (c+1)*N/G)
//                  (a KV split), 1: chunk g -> CTA g % G.
//  MODE 2 k-split: CTA c -> matrix m = c / gs (gs = G/nmat), j = c % gs, (rg, kp) = (j / ks, j % ks); it owns rows
//                  [rg*rows/ng, (rg+1)*rows/ng) (ng = gs/ks) of byte segment [kp*seg, (kp+1)*seg), seg = rb/ks.
//                  The host guarantees G % nmat == 0 and (G/nmat) % ks == 0.
struct RIt {
  const uint8_t* base;
  long long r0, r1, n, N;
  int cr, inter, c, G, rowbytes;
};
__device__ __forceinline__ RIt g_init(const Params& P, int cr) {
  RIt it;
  it.base = P.mat0; it.rowbytes = P.rb;
  it.cr = cr; it.inter = P.interleave; it.c = (int)blockIdx.x; it.G = (int)gridDim.x; it.N = P.rows;
  it.r0 = 0; it.r1 = 0;
  if (it.inter) {
    const long long total = (it.N + cr - 1) / cr;
    it.n = (it.c < total) ? (total - 1 - it.c) / it.G + 1 : 0;
  } else {
    it.r0 = (long long)it.c * it.N / it.G;
    it.r1 = (long long)(it.c + 1) * it.N / it.G;
    it.n = (it.r1 - it.r0 + cr - 1) / cr;
  }
  return it;
}
__device__ __forceinline__ RIt k_init(const Params& P, int cr) {
  RIt it;
  it.c = (int)blockIdx.x; it.G = (int)gridDim.x; it.cr = cr; it.inter = 0; it.N = P.rows;
  const int gs = it.G / P.nmat, m = it.c / gs, j = it.c - m * gs;
  const int ks = P.ksplit, ng = gs / ks, rg = j / ks, kp = j - rg * ks;
  it.rowbytes = P.rb / ks;
  it.base = mat_ptr(P, m) + (long long)kp * it.rowbytes;
  it.r0 = (long long)rg * P.rows / ng;
  it.r1 = (long long)(rg + 1) * P.rows / ng;
  it.n = (it.r1 - it.r0 + cr - 1) / cr;
  return it;
}
// chunk k -> first row (gather: idx position; k-split: matrix row), nrows
__device__ __forceinline__ long long r_chunk(const RIt& it, long long k, int& nrows) {
  long long first, end;
  if (it.inter) { first = ((long long)it.c + k * it.G) * it.cr; end = it.N; }
  else { first = it.r0 + k * it.cr; end = it.r1; }
  const long long rem = end - first;
  nrows = (int)(rem < it.cr ? rem : (long long)it.cr);
  return first;
}
// row value -> address. MODE 1: ri = table row (clamped; never expected to clamp, idx is a permutation prefix);
// MODE 2: ri = matrix row.
template <int MODE>
__device__ __forceinline__ const uint8_t* r_addr(const Params& P, const RIt& it, int ri) {
  if constexpr (MODE == 1) {
    const long long r = (long long)(unsigned)ri;
    return P.mat0 + (r < P.table_rows ? r : 0ll) * P.rb;
  } else {
    return it.base + (long long)ri * P.rb;
  }
}
// value of row j of a chunk starting at `first` (MODE 1: one idx load; MODE 2: arithmetic)
template <int MODE>
__device__ __forceinline__ int r_val(const Params& P, long long first, int j) {
  if constexpr (MODE == 1) return __ldg(P.idx + first + j);
  else return (int)(first + j);
}

// ---- loads ----
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld
// (form as sglang kernels/aot/csrc/elementwise/utils.cuh:54 and k2a_v1.py: ld.global.nc.L1::no_allocate.v*.s32)
__device__ __forceinline__ uint4 ldg_na(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// ---- consumer ----
// Every 16 B: checksum (sum of u32 words mod 2^32, compared with torch on the host). WORK kernels and work >= 1 add
// the NVFP4 GEMM inner loop: Marlin's E2M1 -> BF16 bit trick per 32-bit word (exactly lab M5's marlin-lop3,
// sglang marlin/dequant.h:393-407), one BF16x2 FMA by a group scale per BF16x2 (Marlin's scale HMUL2), and for
// work >= 2 four mma.sync m16n8k16 BF16 (32 weights x 8 tokens per thread-16B, FP32 accumulators).
// Every call site is warp-uniform (mma.sync.aligned).
struct Cons {
  uint32_t sum, dmy, b0, b1;
  float c[16];
};
__device__ __forceinline__ void cons_init(Cons& s) {
  s.sum = 0u; s.dmy = 0u;
  s.b0 = 0x3F803F80u ^ (threadIdx.x & 1u);   // B fragment (activations), not a compile-time constant
  s.b1 = 0x3F803F80u;
#pragma unroll
  for (int i = 0; i < 16; ++i) s.c[i] = 0.f;
}
__device__ __forceinline__ void dq_word(uint32_t q, uint32_t& o0, uint32_t& o1, uint32_t& o2, uint32_t& o3) {
  uint32_t x = q << 8;
  const uint32_t a1 = (x & 0x80008000u) | ((x & 0x70007000u) >> 6);
  x <<= 4;
  const uint32_t a2 = (x & 0x80008000u) | ((x & 0x70007000u) >> 6);
  x = q;
  const uint32_t b1 = (x & 0x80008000u) | ((x & 0x70007000u) >> 6);
  x <<= 4;
  const uint32_t b2 = (x & 0x80008000u) | ((x & 0x70007000u) >> 6);
  o0 = a2; o1 = a1; o2 = b2; o3 = b1;
}
// strings as CUTLASS cutlass/array.h:2304 (fma.rn.bf16x2) and cute/arch/mma_sm80.hpp:238 (SM80_16x8x16_F32BF16BF16F32_TN)
__device__ __forceinline__ uint32_t bf2_fma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <bool WORK>
__device__ __forceinline__ void consume(const uint4 v, int work, Cons& s) {
  s.sum += v.x + v.y + v.z + v.w;
  if constexpr (WORK) {
    if (work >= 1) {
      const uint32_t sc = 0x7E807E80u ^ (s.dmy & 0x00010001u);   // group scale 2^126 (undoes Marlin's 2^-126)
      uint32_t f[16];
      dq_word(v.x, f[0], f[1], f[2], f[3]);
      dq_word(v.y, f[4], f[5], f[6], f[7]);
      dq_word(v.z, f[8], f[9], f[10], f[11]);
      dq_word(v.w, f[12], f[13], f[14], f[15]);
#pragma unroll
      for (int i = 0; i < 16; ++i) f[i] = bf2_fma(f[i], sc, 0x80008000u);
      if (work >= 2) {
        mma_bf16(s.c + 0, f[0], f[1], f[2], f[3], s.b0, s.b1);
        mma_bf16(s.c + 4, f[4], f[5], f[6], f[7], s.b0, s.b1);
        mma_bf16(s.c + 8, f[8], f[9], f[10], f[11], s.b0, s.b1);
        mma_bf16(s.c + 12, f[12], f[13], f[14], f[15], s.b0, s.b1);
      } else {
        uint32_t x = 0u;
#pragma unroll
        for (int i = 0; i < 16; ++i) x ^= f[i];
        s.dmy ^= x;
      }
    }
  }
}
template <bool WORK>
__device__ __forceinline__ void finish(const Params& P, const Cons& s) {
  uint32_t d = s.dmy;
  if constexpr (WORK) {
    float f = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) f += s.c[i];
    d ^= __float_as_uint(f);
  }
  // Checksum: warp-reduce (every call site has the whole warp converged), then one atomic per warp spread over 1024
  // slots, so the epilogue costs ~4 atomics per address instead of one same-address atomic per thread.
  uint32_t ws = s.sum;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ws += __shfl_xor_sync(0xffffffffu, ws, o);
  if ((threadIdx.x & 31u) == 0u && ws != 0u) {
    const unsigned gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    atomicAdd(P.sink + (gw & 1023u), (unsigned long long)ws);
  }
  if (d == 0x12345679u) atomicAdd(P.sink + 1024, 1ull);
  if (P.red != nullptr) {   // split-K epilogue model: red_n fp32 atomics in total (one per output partial)
    const int per = P.red_n / (int)gridDim.x;
    const float val = __uint_as_float((d & 0x007FFFFFu) | 0x3F800000u);   // fp32 in [1, 2)
    for (int i = (int)threadIdx.x; i < per; i += (int)blockDim.x) atomicAdd(P.red + (size_t)blockIdx.x * per + i, val);
  }
}

template <int Z>
__global__ void sl_empty(const Params P) {
  if (P.work == 12345 && threadIdx.x == 0) P.sink[1024] = (unsigned long long)Z;
}

// ---------------- S1: plain LDG.128, U loads in flight per thread ----------------
template <int U>
__device__ __forceinline__ void s1_load(const Params& P, const CIt& it, long long k, int T, int t, uint4* v) {
  const uint4 z = make_uint4(0u, 0u, 0u, 0u);
  if (k >= it.n) {
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = z;
    return;
  }
  int m, bytes;
  long long off;
  c_chunk(it, k, m, off, bytes);
  const uint8_t* src = mat_ptr(P, m) + off;
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int o = (u * T + t) * 16;
    v[u] = (o < bytes) ? ldg_na(src + o) : z;
  }
}
template <int U, int MODE>
__device__ __forceinline__ void s1_rows(const Params& P, const RIt& it, long long k, const int* rj, int* ci) {
  int nr = 0;
  const long long first = (k < it.n) ? r_chunk(it, k, nr) : 0ll;
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int j = rj[u];
    ci[u] = (j >= 0 && j < nr) ? r_val<MODE>(P, first, j) : -1;
  }
}
// MODE 0 contig: chunk = T*16*U bytes; thread t loads 16 B at (u*T + t)*16. PIPE: the next chunk's loads are
//   issued before the current chunk is consumed (register double buffer, up to 2U loads in flight).
// MODE 1/2: rows of rowbytes = 16*tps; piece p = u*T + t of a chunk -> row p / tps, part p % tps (precomputed per
//   thread); cr = U*T/tps rows per chunk; the next chunk's row values (idx loads) are fetched behind the row loads.
template <int U, int MODE, bool PIPE, bool WORK>
__global__ void __launch_bounds__(512) s1_kernel(const Params P) {
  const int T = (int)blockDim.x, t = (int)threadIdx.x;
  Cons s;
  cons_init(s);
  if constexpr (MODE == 0) {
    const CIt it = c_init(P, P.rows, P.rb, T * 16 * U);
    if constexpr (!PIPE) {
      for (long long k = 0; k < it.n; ++k) {
        uint4 v[U];
        s1_load<U>(P, it, k, T, t, v);
#pragma unroll
        for (int u = 0; u < U; ++u) consume<WORK>(v[u], P.work, s);
      }
    } else {
      uint4 cur[U];
      s1_load<U>(P, it, 0, T, t, cur);
      for (long long k = 0; k < it.n; ++k) {
        uint4 nxt[U];
        s1_load<U>(P, it, k + 1, T, t, nxt);
#pragma unroll
        for (int u = 0; u < U; ++u) consume<WORK>(cur[u], P.work, s);
#pragma unroll
        for (int u = 0; u < U; ++u) cur[u] = nxt[u];
      }
    }
  } else {
    const int rowbytes = (MODE == 1) ? P.rb : P.rb / P.ksplit;
    const int tps = rowbytes >> 4;
    const int cr = (U * T) / tps;
    const RIt it = (MODE == 1) ? g_init(P, cr) : k_init(P, cr);
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    int rj[U], rs[U], ci[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int p = u * T + t;
      rj[u] = p / tps;
      rs[u] = p - rj[u] * tps;
      if (rj[u] >= cr) rj[u] = -1;
    }
    s1_rows<U, MODE>(P, it, 0, rj, ci);
    for (long long k = 0; k < it.n; ++k) {
      uint4 v[U];
#pragma unroll
      for (int u = 0; u < U; ++u) v[u] = (ci[u] >= 0) ? ldg_na(r_addr<MODE>(P, it, ci[u]) + rs[u] * 16) : z;
      s1_rows<U, MODE>(P, it, k + 1, rj, ci);
#pragma unroll
      for (int u = 0; u < U; ++u) consume<WORK>(v[u], P.work, s);
    }
  }
  finish<WORK>(P, s);
}

// ---------------- S2: cp.async.cg 16 B, STAGES-deep smem ring, __syncthreads per stage ----------------
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
// (strings as k1_v0.py's GPU-validated cp_async16 / commit / wait, and CUTLASS cute/arch/copy_sm80.hpp)
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

constexpr int kVMax = 8;

__device__ __forceinline__ void s2_idx(const Params& P, const RIt& it, long long s, int T, int t, int V, int lg,
                                       int* nid) {
  int nr = 0;
  const long long first = (s < it.n) ? r_chunk(it, s, nr) : 0ll;
#pragma unroll
  for (int v = 0; v < kVMax; ++v) {
    const int j = (v * T + t) >> lg;
    nid[v] = (v < V && j < nr) ? __ldg(P.idx + first + j) : -1;
  }
}
// Stage s (if s < n) into smem address dst, then one commit (always, so the group count stays exact).
template <bool GATHER>
__device__ __forceinline__ void s2_issue(const Params& P, const CIt& cit, const RIt& git, long long s, long long n,
                                         uint32_t dst, int T, int t, int V, int lg, int* nid) {
  if (s < n) {
    if constexpr (!GATHER) {
      int m, bytes;
      long long off;
      c_chunk(cit, s, m, off, bytes);
      const uint8_t* src = mat_ptr(P, m) + off;
#pragma unroll
      for (int v = 0; v < kVMax; ++v) {
        const int o = (v * T + t) * 16;
        if (v < V && o < bytes) cp_async16(dst + (uint32_t)o, src + o);
      }
    } else {
      const int tmask = (P.rb >> 4) - 1;
#pragma unroll
      for (int v = 0; v < kVMax; ++v) {
        const int p = v * T + t;   // piece p = row p>>lg, 16 B part p&tmask; smem offset p*16 = row*rb + part*16
        if (nid[v] >= 0) cp_async16(dst + (uint32_t)p * 16u, r_addr<1>(P, git, nid[v]) + (p & tmask) * 16);
      }
      s2_idx(P, git, s + 1, T, t, V, lg, nid);   // indices for the next stage, one stage ahead
    }
  }
  cp_async_commit();
}
template <int STAGES, bool GATHER, bool WORK>
__global__ void __launch_bounds__(512) s2_kernel(const Params P) {
  extern __shared__ __align__(128) uint8_t sl_dyn_smem[];
  const int T = (int)blockDim.x, t = (int)threadIdx.x, V = P.v;
  const int stage_bytes = T * 16 * V;
  const uint32_t sbase = okc_stream::smem_addr(sl_dyn_smem);
  Cons s;
  cons_init(s);
  CIt cit = {};
  RIt git = {};
  long long n;
  int lg = 0;
  if constexpr (!GATHER) {
    cit = c_init(P, P.rows, P.rb, stage_bytes);
    n = cit.n;
  } else {
    lg = __ffs(P.rb >> 4) - 1;
    git = g_init(P, stage_bytes / P.rb);
    n = git.n;
  }
  int nid[kVMax];
#pragma unroll
  for (int v = 0; v < kVMax; ++v) nid[v] = -1;
  if constexpr (GATHER) s2_idx(P, git, 0, T, t, V, lg, nid);
#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st)
    s2_issue<GATHER>(P, cit, git, st, n, sbase + (uint32_t)(st * stage_bytes), T, t, V, lg, nid);
  const int tt = t ^ 32;   // consume pieces another warp copied, as a real consumer would (T % 64 == 0)
  const uint4 z = make_uint4(0u, 0u, 0u, 0u);
  for (long long k = 0; k < n; ++k) {
    cp_async_wait<STAGES - 2>();   // this thread's copies of stage k landed
    __syncthreads();               // everyone's copies landed; everyone finished reading stage k-1's slot
    const long long nx = k + STAGES - 1;
    s2_issue<GATHER>(P, cit, git, nx, n, sbase + (uint32_t)((nx % STAGES) * stage_bytes), T, t, V, lg, nid);
    int bytes;
    if constexpr (!GATHER) {
      int m;
      long long off;
      c_chunk(cit, k, m, off, bytes);
    } else {
      int nr;
      r_chunk(git, k, nr);
      bytes = nr * P.rb;
    }
    const uint8_t* slot = sl_dyn_smem + (size_t)(k % STAGES) * (size_t)stage_bytes;
#pragma unroll
    for (int v = 0; v < kVMax; ++v) {
      if (v < V) {
        const int o = (v * T + tt) * 16;
        const uint4 d = (o < bytes) ? *reinterpret_cast<const uint4*>(slot + o) : z;
        consume<WORK>(d, P.work, s);
      }
    }
  }
  cp_async_wait<0>();
  finish<WORK>(P, s);
}

// ---------------- S3: okc_stream ring, 1 producer warp + NC consumer warps ----------------
template <int MODE>
__device__ __forceinline__ void s3_rows(const Params& P, const RIt& it, long long k, int lane, int* nid) {
  int nr = 0;
  const long long first = (k < it.n) ? r_chunk(it, k, nr) : 0ll;
#pragma unroll
  for (int g = 0; g < 16; ++g) {
    const int j = g * 32 + lane;
    nid[g] = (j < nr) ? r_val<MODE>(P, first, j) : 0;
  }
}
// Consumer warp loop over one landed chunk: warp-uniform trip count, lanes past `bytes` consume zeros.
template <bool WORK>
__device__ __forceinline__ void s3_consume(const Params& P, const uint8_t* slot, int bytes, int cw, int NC, int lane,
                                           Cons& s) {
  const uint4 z = make_uint4(0u, 0u, 0u, 0u);
  for (int base = cw * 512; base < bytes; base += NC * 512) {
    const int o = base + lane * 16;
    const uint4 d = (o < bytes) ? *reinterpret_cast<const uint4*>(slot + o) : z;
    consume<WORK>(d, P.work, s);
  }
}
// MODE 0 contig: lane 0 copies a whole chunk per bulk copy. MODE 1 gather / 2 k-split: one bulk copy per row
// (rowbytes); lanes = 32: lane l copies rows l, l+32, ...; lanes = 1: lane 0 issues every row (row values loaded by
// all lanes, handed over by shuffle).
template <int MODE, bool WORK>
__global__ void __launch_bounds__(512) s3_kernel(const Params P) {
  extern __shared__ __align__(128) uint8_t sl_dyn_smem[];
  __shared__ __align__(8) uint64_t bars[2 * okc_stream::kMaxStages];
  uint8_t* data = sl_dyn_smem + ((128u - (okc_stream::smem_addr(sl_dyn_smem) & 127u)) & 127u);
  const okc_stream::Ring R = okc_stream::ring_make(data, bars, P.stages, (uint32_t)P.chunk);
  okc_stream::Guard gd;
  gd.flag = P.flag;
  gd.budget = P.budget;
  const int warp = (int)(threadIdx.x >> 5), lane = (int)(threadIdx.x & 31u);
  const int NC = (int)(blockDim.x >> 5) - 1;
  if (threadIdx.x == 0) okc_stream::ring_init(R, (uint32_t)NC);
  __syncthreads();
  Cons s;
  cons_init(s);
  CIt cit = {};
  RIt rit = {};
  int n;
  if constexpr (MODE == 0) {
    cit = c_init(P, P.rows, P.rb, P.chunk);
    n = (int)cit.n;
  } else {
    const int rowbytes = (MODE == 1) ? P.rb : P.rb / P.ksplit;
    rit = (MODE == 1) ? g_init(P, P.chunk / rowbytes) : k_init(P, P.chunk / rowbytes);
    n = (int)rit.n;
  }
  if (warp == 0) {
    if constexpr (MODE == 0) {
      for (int k = 0; k < n; ++k) {
        if (!okc_stream::producer_acquire(R, k, gd)) break;
        if (lane == 0) {
          int m, bytes;
          long long off;
          c_chunk(cit, k, m, off, bytes);
          okc_stream::producer_put(R, k, mat_ptr(P, m) + off, (uint32_t)bytes);
        }
      }
    } else {
      const int rb = rit.rowbytes;
      int nid[16];
      s3_rows<MODE>(P, rit, 0, lane, nid);
      for (int k = 0; k < n; ++k) {
        if (!okc_stream::producer_acquire(R, k, gd)) break;
        int nr;
        r_chunk(rit, k, nr);
        if (lane == 0) okc_stream::producer_expect(R, k, (uint32_t)(nr * rb));   // == sum of the copies below
        __syncwarp();
        if (P.lanes >= 32) {
#pragma unroll
          for (int g = 0; g < 16; ++g) {
            const int j = g * 32 + lane;
            if (j < nr) okc_stream::producer_copy(R, k, (uint32_t)(j * rb), r_addr<MODE>(P, rit, nid[g]), (uint32_t)rb);
          }
        } else {
#pragma unroll
          for (int g = 0; g < 16; ++g) {
            if (g * 32 < nr) {
              for (int jj = 0; jj < 32; ++jj) {
                const int j = g * 32 + jj;
                if (j >= nr) break;   // warp-uniform
                const int r = __shfl_sync(0xffffffffu, nid[g], jj);
                if (lane == 0) okc_stream::producer_copy(R, k, (uint32_t)(j * rb), r_addr<MODE>(P, rit, r), (uint32_t)rb);
              }
            }
          }
        }
        s3_rows<MODE>(P, rit, k + 1, lane, nid);
      }
    }
  } else {
    const int cw = warp - 1;
    for (int k = 0; k < n; ++k) {
      const uint8_t* slot = okc_stream::consumer_wait(R, k, gd);
      if (slot == nullptr) break;
      int bytes;
      if constexpr (MODE == 0) {
        int m;
        long long off;
        c_chunk(cit, k, m, off, bytes);
      } else {
        int nr;
        r_chunk(rit, k, nr);
        bytes = nr * rit.rowbytes;
      }
      s3_consume<WORK>(P, slot, bytes, cw, NC, lane, s);
      okc_stream::consumer_release(R, k);
    }
  }
  finish<WORK>(P, s);
}

#ifndef SL_NO_S4
// ---------------- S4: TMA 2-D tensor-map boxes (UINT64 elements), same ring and consumers as S3 ----------------
// Matrix m is a [rows, rb/8] UINT64 tensor; box = [box_h rows, box_w/8 elements] = chunk bytes.
//  ksplit == 1: tiles in row-block-major order (tile = rbk * (rb/box_w) + cb), split over CTAs like contig chunks.
//  ksplit  > 1: box_w = rb/ks; CTA (m, rg, kp) as MODE 2 owns row blocks [rg*nrb/ng, (rg+1)*nrb/ng) of column kp.
struct TMaps {
  CUtensorMap m[4];
};
template <bool WORK>
__global__ void __launch_bounds__(512) s4_kernel(const __grid_constant__ TMaps tm, const Params P) {
  extern __shared__ __align__(128) uint8_t sl_dyn_smem[];
  __shared__ __align__(8) uint64_t bars[2 * okc_stream::kMaxStages];
  uint8_t* data = sl_dyn_smem + ((128u - (okc_stream::smem_addr(sl_dyn_smem) & 127u)) & 127u);
  const okc_stream::Ring R = okc_stream::ring_make(data, bars, P.stages, (uint32_t)P.chunk);
  okc_stream::Guard gd;
  gd.flag = P.flag;
  gd.budget = P.budget;
  const int warp = (int)(threadIdx.x >> 5), lane = (int)(threadIdx.x & 31u);
  const int NC = (int)(blockDim.x >> 5) - 1;
  if (threadIdx.x == 0) okc_stream::ring_init(R, (uint32_t)NC);
  __syncthreads();
  const int tpr = P.rb / P.box_w;
  const long long nrb = P.rows / P.box_h;
  CIt cit = {};
  int n = 0, km = 0, kp = 0;
  long long krb0 = 0;
  if (P.ksplit <= 1) {
    cit = c_init(P, nrb * tpr, P.chunk, P.chunk);
    n = (int)cit.n;
  } else {
    const int c = (int)blockIdx.x, G = (int)gridDim.x;
    const int gs = G / P.nmat, j = c - (c / gs) * gs, ks = P.ksplit, ng = gs / ks, rg = j / ks;
    km = c / gs;
    kp = j - rg * ks;
    krb0 = (long long)rg * nrb / ng;
    n = (int)((long long)(rg + 1) * nrb / ng - krb0);
  }
  Cons s;
  cons_init(s);
  if (warp == 0) {
    for (int k = 0; k < n; ++k) {
      if (!okc_stream::producer_acquire(R, k, gd)) break;
      if (lane == 0) {
        int m, c0, c1;
        if (P.ksplit <= 1) {
          int bytes;
          long long off;
          c_chunk(cit, k, m, off, bytes);
          const long long tile = off / P.chunk;
          const int rbk = (int)(tile / tpr), cb = (int)(tile - (long long)rbk * tpr);
          c0 = cb * (P.box_w / 8);
          c1 = rbk * P.box_h;
        } else {
          m = km;
          c0 = kp * (P.box_w / 8);
          c1 = (int)((krb0 + k) * P.box_h);
        }
        okc_stream::producer_put_tile(R, k, &tm.m[m], c0, c1, (uint32_t)P.chunk);
      }
    }
  } else {
    const int cw = warp - 1;
    for (int k = 0; k < n; ++k) {
      const uint8_t* slot = okc_stream::consumer_wait(R, k, gd);
      if (slot == nullptr) break;
      s3_consume<WORK>(P, slot, P.chunk, cw, NC, lane, s);
      okc_stream::consumer_release(R, k);
    }
  }
  finish<WORK>(P, s);
}
#endif

}  // namespace sl

// ---------------- host side ----------------
// kernel ids: S1 <U, MODE, PIPE, WORK>, S2 <STAGES, GATHER, WORK>, S3 <MODE, WORK>, S4 <WORK> (60/61), empty (1)
#define SL_LIST(X)                                                                                         \
  X(1, sl::sl_empty, 0)                                                                                    \
  X(10, sl::s1_kernel, 2, 0, false, false) X(11, sl::s1_kernel, 4, 0, false, false)                        \
  X(12, sl::s1_kernel, 8, 0, false, false)                                                                 \
  X(13, sl::s1_kernel, 2, 0, false, true) X(14, sl::s1_kernel, 4, 0, false, true)                          \
  X(15, sl::s1_kernel, 8, 0, false, true)                                                                  \
  X(16, sl::s1_kernel, 2, 0, true, true) X(17, sl::s1_kernel, 4, 0, true, true)                            \
  X(20, sl::s1_kernel, 2, 1, false, false) X(21, sl::s1_kernel, 4, 1, false, false)                        \
  X(22, sl::s1_kernel, 8, 1, false, false)                                                                 \
  X(25, sl::s1_kernel, 2, 2, false, false) X(26, sl::s1_kernel, 4, 2, false, false)                        \
  X(27, sl::s1_kernel, 8, 2, false, false)                                                                 \
  X(30, sl::s2_kernel, 3, false, false) X(31, sl::s2_kernel, 4, false, false)                              \
  X(32, sl::s2_kernel, 6, false, false) X(33, sl::s2_kernel, 8, false, false)                              \
  X(35, sl::s2_kernel, 3, false, true) X(36, sl::s2_kernel, 4, false, true)                                \
  X(37, sl::s2_kernel, 6, false, true) X(38, sl::s2_kernel, 8, false, true)                                \
  X(40, sl::s2_kernel, 3, true, false) X(41, sl::s2_kernel, 4, true, false)                                \
  X(42, sl::s2_kernel, 6, true, false) X(43, sl::s2_kernel, 8, true, false)                                \
  X(50, sl::s3_kernel, 0, false) X(51, sl::s3_kernel, 1, false) X(52, sl::s3_kernel, 2, false)             \
  X(53, sl::s3_kernel, 0, true)

static const void* sl_kptr(int64_t kid) {
#define SL_KPTR(ID, KER, ...) \
  case ID: return (const void*)KER<__VA_ARGS__>;
  switch (kid) {
    SL_LIST(SL_KPTR)
    default: break;
  }
#undef SL_KPTR
#ifndef SL_NO_S4
  if (kid == 60) return (const void*)sl::s4_kernel<false>;
  if (kid == 61) return (const void*)sl::s4_kernel<true>;
#endif
  return nullptr;
}

static int sl_scheme(int64_t kid) {
  if (kid >= 10 && kid < 30) return 1;
  if (kid >= 30 && kid < 50) return 2;
  if (kid >= 50 && kid < 60) return 3;
  if (kid == 60 || kid == 61) return 4;
  return 0;
}
static int sl_mode(int64_t kid) {   // 0 contig, 1 gather, 2 k-split
  if ((kid >= 20 && kid < 25) || (kid >= 40 && kid < 50) || kid == 51) return 1;
  if ((kid >= 25 && kid < 30) || kid == 52) return 2;
  return 0;
}
static bool sl_work_kernel(int64_t kid) {
  return (kid >= 13 && kid <= 17) || (kid >= 35 && kid <= 38) || kid == 53 || kid == 61;
}
static int sl_s2_stages(int64_t kid) {
  const int st[4] = {3, 4, 6, 8};
  const int i = (int)(kid % 5);
  return (i >= 0 && i < 4) ? st[i] : 0;
}
static int sl_optin_smem() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetAttribute(&v, (cudaDeviceAttr)97, dev) != cudaSuccess) {  // 97 = max smem/block opt-in
      cudaGetLastError();
      v = 232448;
    }
  }
  return v;
}

int64_t sl_smem(int64_t kid, int64_t threads, int64_t chunk, int64_t stages, int64_t v) {
  const int s = sl_scheme(kid);
  if (s == 2) return (int64_t)sl_s2_stages(kid) * threads * 16 * v;
  if (s == 3 || s == 4) return stages * chunk + 128;
  return 0;
}

// [regs, local bytes, static smem, max threads/block, occupancy (CTAs/SM) at (threads, smem), setattr rc, ptx, sass]
std::vector<int64_t> sl_attrs(int64_t kid, int64_t threads, int64_t smem) {
  std::vector<int64_t> r(8, -1);
  const void* f = sl_kptr(kid);
  if (f == nullptr) return r;
  cudaFuncAttributes a;
  if (cudaFuncGetAttributes(&a, f) != cudaSuccess) {
    cudaGetLastError();
    return r;
  }
  r[0] = a.numRegs; r[1] = (int64_t)a.localSizeBytes; r[2] = (int64_t)a.sharedSizeBytes;
  r[3] = a.maxThreadsPerBlock; r[6] = a.ptxVersion; r[7] = a.binaryVersion;
  const int want = sl_optin_smem() - (int)a.sharedSizeBytes;
  r[5] = (int64_t)cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
  if (r[5] != 0) cudaGetLastError();
  int nb = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, (int)threads, (size_t)smem) != cudaSuccess) {
    cudaGetLastError();
    nb = -1;
  }
  r[4] = nb;
  return r;
}

// [SM clock kHz, SMs, L2 bytes, smem opt-in/block, CUDART_VERSION, driver version, S4 compiled]
std::vector<int64_t> sl_dev() {
  std::vector<int64_t> r;
  int dev = 0;
  cudaGetDevice(&dev);
  const int ids[4] = {13, 16, 38, 97};
  for (int k = 0; k < 4; ++k) {
    int v = -1;
    if (cudaDeviceGetAttribute(&v, (cudaDeviceAttr)ids[k], dev) != cudaSuccess) {
      cudaGetLastError();
      v = -1;
    }
    r.push_back(v);
  }
  r.push_back((int64_t)CUDART_VERSION);
  int drv = -1;
  if (cudaDriverGetVersion(&drv) != cudaSuccess) {
    cudaGetLastError();
    drv = -1;
  }
  r.push_back(drv);
#ifndef SL_NO_S4
  r.push_back(1);
#else
  r.push_back(0);
#endif
  return r;
}

#ifndef SL_NO_S4
typedef CUresult (*sl_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                 const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                 CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
// Driver entry point without linking libcuda, as CUTLASS does (repos/cutlass/include/cutlass/cuda_host_adapter.hpp,
// the __CUDACC_VER_MAJOR__ > 12 branch: cudaGetDriverEntryPointByVersion(..., 12000, cudaEnableDefault, &status)).
static sl_encode_fn sl_get_encode(int64_t* status) {
  static sl_encode_fn fn = nullptr;
  static int64_t st = -1;
  if (st < 0) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
      cudaGetLastError();
      st = 1000 + 10 * (int64_t)e + (int64_t)q;
      fn = nullptr;
    } else {
      fn = reinterpret_cast<sl_encode_fn>(p);
      st = 0;
    }
  }
  *status = st;
  return fn;
}
#endif

int64_t sl_s4_init() {
#ifndef SL_NO_S4
  int64_t st = -1;
  sl_get_encode(&st);
  return st;
#else
  return -7;
#endif
}

// ptrs = [mat0, mat1, mat2, mat3, idx, sink, flag, red]
// a    = [grid, threads, rows, table_rows, budget, nmat, rb, chunk, interleave, stages, work, lanes, v, box_w, box_h,
//         ksplit, red_n]
// returns 0, a cudaError_t, -2 unknown kernel, -3 invalid config, -1000-x tensor-map encode error x
int64_t sl_launch(int64_t kid, std::vector<int64_t> ptrs, std::vector<int64_t> a) {
  if (ptrs.size() < 8 || a.size() < 17) return -3;
  sl::Params P;
  P.mat0 = (const uint8_t*)ptrs[0]; P.mat1 = (const uint8_t*)ptrs[1];
  P.mat2 = (const uint8_t*)ptrs[2]; P.mat3 = (const uint8_t*)ptrs[3];
  P.idx = (const int*)ptrs[4]; P.sink = (unsigned long long*)ptrs[5]; P.flag = (int*)ptrs[6];
  P.red = (float*)ptrs[7];
  const int grid = (int)a[0], threads = (int)a[1];
  P.rows = a[2]; P.table_rows = a[3]; P.budget = a[4];
  P.nmat = (int)a[5]; P.rb = (int)a[6]; P.chunk = (int)a[7]; P.interleave = (int)a[8]; P.stages = (int)a[9];
  P.work = (int)a[10]; P.lanes = (int)a[11]; P.v = (int)a[12]; P.box_w = (int)a[13]; P.box_h = (int)a[14];
  P.ksplit = (int)a[15]; P.red_n = (int)a[16];
  if (kid == 1) {   // empty kernel (graph-replay floor)
    sl::sl_empty<0><<<grid, threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(P);
    return (int64_t)cudaGetLastError();
  }
  const int s = sl_scheme(kid), mode = sl_mode(kid);
  // ---- validation: every kernel relies on these ----
  if (sl_kptr(kid) == nullptr) return -2;
  if (grid < 1 || threads < 32 || threads > 512 || (threads & 31) != 0) return -3;
  if (P.nmat < 1 || P.nmat > 4 || grid < P.nmat || P.rows < 1 || P.rb < 16 || (P.rb & 15) != 0) return -3;
  if (P.budget <= 0 || P.work < 0 || P.work > 2 || (P.work > 0 && !sl_work_kernel(kid))) return -3;
  if (P.red != nullptr && P.red_n < 1) return -3;
  if (P.ksplit < 1) P.ksplit = 1;
  if (mode == 1) {
    if (P.rb > 256 || (P.rb & (P.rb - 1)) != 0 || P.table_rows < 1 || P.idx == nullptr) return -3;
  }
  if (mode == 2 || (s == 4 && P.ksplit > 1)) {
    if ((grid % P.nmat) != 0 || ((grid / P.nmat) % P.ksplit) != 0 || (P.rb % (16 * P.ksplit)) != 0) return -3;
  }
  const int rowbytes = (mode == 1) ? P.rb : P.rb / P.ksplit;
  if (s == 1 && mode != 0 && (threads * 16 * (int)(kid % 5 == 0 ? 2 : kid % 5 == 1 ? 4 : 8)) < rowbytes) return -3;
  if (s == 2 && (threads % 64) != 0) return -3;
  if (s == 2 && (P.v < 1 || P.v > 8)) return -3;
  if (s == 2 && mode == 1 && ((threads * 16 * P.v) % P.rb) != 0) return -3;
  if (s == 3 || s == 4) {
    if (threads < 64 || P.stages < 1 || P.stages > okc_stream::kMaxStages) return -3;
    if (P.chunk < 16 || (P.chunk & 15) != 0) return -3;
  }
  if (s == 3 && mode != 0 && (P.chunk < rowbytes || (P.chunk % rowbytes) != 0 || P.chunk / rowbytes > 512)) return -3;
  if (s == 4) {
    if ((P.chunk & 127) != 0 || P.box_w < 16 || (P.box_w & 15) != 0 || P.box_w / 8 > 256 || P.box_h < 1 ||
        P.box_h > 256 || P.chunk != P.box_w * P.box_h || (P.rb % P.box_w) != 0 || (P.rows % P.box_h) != 0)
      return -3;
    if (P.ksplit > 1 && P.box_w * P.ksplit != P.rb) return -3;
  }
  const int64_t smem = sl_smem(kid, threads, P.chunk, P.stages, P.v);
  if (smem > sl_optin_smem()) return -3;
  cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
#ifndef SL_NO_S4
  if (s == 4) {
    int64_t est = -1;
    sl_encode_fn enc = sl_get_encode(&est);
    if (enc == nullptr) return -1000 - est;
    sl::TMaps tm;
    memset(&tm, 0, sizeof(tm));
    const uint8_t* bases[4] = {P.mat0, P.mat1, P.mat2, P.mat3};
    for (int i = 0; i < P.nmat; ++i) {
      cuuint64_t dims[2] = {(cuuint64_t)(P.rb / 8), (cuuint64_t)P.rows};
      cuuint64_t strides[1] = {(cuuint64_t)P.rb};
      cuuint32_t box[2] = {(cuuint32_t)(P.box_w / 8), (cuuint32_t)P.box_h};
      cuuint32_t es[2] = {1u, 1u};
      const CUresult r = enc(&tm.m[i], CU_TENSOR_MAP_DATA_TYPE_UINT64, 2, (void*)bases[i], dims, strides, box, es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (r != CUDA_SUCCESS) return -1000 - (int64_t)r;
    }
    if (kid == 61) sl::s4_kernel<true><<<grid, threads, (size_t)smem, st>>>(tm, P);
    else sl::s4_kernel<false><<<grid, threads, (size_t)smem, st>>>(tm, P);
    return (int64_t)cudaGetLastError();
  }
#endif
#define SL_LAUNCH(ID, KER, ...)                                  \
  case ID:                                                       \
    KER<__VA_ARGS__><<<grid, threads, (size_t)smem, st>>>(P);    \
    return (int64_t)cudaGetLastError();
  switch (kid) {
    SL_LIST(SL_LAUNCH)
    default: break;
  }
#undef SL_LAUNCH
  return -2;
}
"""

_SL_HEADERS = r"""
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#ifndef SL_NO_S4
#include <cuda.h>
#endif
"""

_SL_CUDA = _SL_HEADERS + _SL_BLOCK + _SL_KERNELS

_SL_CPP = r"""
#include <vector>
#include <cstdint>
std::vector<int64_t> sl_dev();
std::vector<int64_t> sl_attrs(int64_t kid, int64_t threads, int64_t smem);
int64_t sl_smem(int64_t kid, int64_t threads, int64_t chunk, int64_t stages, int64_t v);
int64_t sl_s4_init();
int64_t sl_launch(int64_t kid, std::vector<int64_t> ptrs, std::vector<int64_t> a);
"""
_SL_FUNCS = ["sl_dev", "sl_attrs", "sl_smem", "sl_s4_init", "sl_launch"]

# kernel ids (must match SL_LIST): (scheme, mode C/G/K, work kernel, unroll|stages, pipe) -> kid
_SL_U3 = (2, 4, 8)
_SL_ST2 = (3, 4, 6, 8)


def _sl_kid(c):
    s, mode, w = c["s"], c["mode"], bool(c["wk"])
    if s == "S1":
        i = _SL_U3.index(c["u"])
        if mode == "G":
            return 20 + i
        if mode == "K":
            return 25 + i
        if c["pipe"]:
            return {2: 16, 4: 17}[c["u"]]
        return (13 if w else 10) + i
    if s == "S2":
        i = _SL_ST2.index(c["st"])
        if mode == "G":
            return 40 + i
        return (35 if w else 30) + i
    if s == "S3":
        if mode == "G":
            return 51
        if mode == "K":
            return 52
        return 53 if w else 50
    return 61 if w else 60


# ------------------------------------------------- helpers -----------------------------------------------------------
def _sl_P(s):
    print(s, flush=True)


def _sl_err(e, n=600):
    s = "%s: %s" % (type(e).__name__, e)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:200] + " ... " + s[-(n - 200):]


def _sl_left(L):
    return L["deadline"] - time.time()


def _sl_build(L):
    """One load_inline build (S1-S4). If it fails and time allows, one retry without S4 (-DSL_NO_S4)."""
    from torch.utils import cpp_extension as ce
    P = L["P"]
    try:
        P("SL arch flags: %s" % " ".join(ce._get_cuda_arch_flags()))
    except Exception as e:
        P("SL arch flags n/a (%s)" % type(e).__name__)
    tag = hashlib.sha1((_SL_CPP + _SL_CUDA).encode()).hexdigest()[:10]
    variants = ([("okc_stream_lab_" + tag, ["-O3"], True)] if LAB_S4 else []) + \
        [("okc_stream_lab_nos4_" + tag, ["-O3", "-DSL_NO_S4"], False)]
    for name, flags, s4 in variants:
        if L.get("build_tried") and time.time() - L["t0"] > LAB_BUILD_RETRY_BY_S:
            P("SL build retry skipped (time)")
            break
        L["build_tried"] = True
        t = time.time()
        try:
            mod = ce.load_inline(name=name, cpp_sources=[_SL_CPP], cuda_sources=[_SL_CUDA], functions=_SL_FUNCS,
                                 extra_cflags=["-O2"], extra_cuda_cflags=flags, verbose=False)
            L["build_s"] = time.time() - t
            L["s4_built"] = s4
            P("SL build %s OK in %.1f s (S4 %s)" % (name, L["build_s"], "on" if s4 else "off"))
            return mod
        except Exception as e:
            msg = "%s: %s" % (type(e).__name__, e)
            P("SL BUILD FAILED %s after %.1f s (S4 %s): ...%s" % (name, time.time() - t, "on" if s4 else "off",
                                                                 msg[-2500:]))
            L["build_err"] = _sl_err(e, 300)
    return None


def _sl_flush(L):
    L["flush"].sum()


def _sl_sync(L):
    L["torch"].cuda.synchronize()


def _sl_infl(c):
    """Bytes in flight per SM (nominal): S1 CTAs*threads*16*unroll; S2 CTAs*(stages-1)*threads*16*v;
    S3/S4 CTAs*(stages-1)*chunk."""
    cps = c["ecps"]
    if c["s"] == "S1":
        return cps * c["T"] * 16 * c["u"]
    if c["s"] == "S2":
        return cps * (c["st"] - 1) * c["T"] * 16 * c["v"]
    return cps * (c["st"] - 1) * c["ck"]


def _sl_name(c):
    s = c["s"]
    if s == "S1":
        n = "S1 u%d t%d c%d" % (c["u"], c["T"], c["cps"]) + ("p" if c["pipe"] else "")
    elif s == "S2":
        n = "S2 s%d t%d v%d c%d" % (c["st"], c["T"], c["v"], c["cps"])
    elif s == "S3":
        n = "S3 s%d %dK n%d c%d" % (c["st"], c["ck0"] // 1024, c["nc"], c["cps"]) + ("" if c["ln"] >= 32 else " l1")
    else:
        n = "S4 s%d h%d n%d c%d" % (c["st"], c["bh"], c["nc"], c["cps"])
    if c["il"]:
        n += " il"
    if c["ks"] > 1:
        n += " k%d" % c["ks"]
    return n


_SL_CFG_KEYS = ("s", "mode", "wk", "u", "T", "cps", "pipe", "st", "v", "ck0", "nc", "ln", "il", "bh", "ks")


def _sl_cfg(s, **kw):
    """Config dict. mode: C contig, G gather, K k-split. wk: compile-time WORK kernel (the run-time work level is a
    measurement argument). ck0 = nominal S3 chunk bytes (k-split rounds it down to whole row segments)."""
    c = {"s": s, "mode": "C", "wk": False, "u": 4, "T": 256, "cps": 1, "pipe": False, "st": 4, "v": 1, "ck0": 0,
         "nc": 4, "ln": 32, "il": 0, "bh": 8, "ks": 1}
    c.update(kw)
    if s in ("S3", "S4"):
        c["T"] = 32 * (1 + c["nc"])
    c["kid"] = _sl_kid(c)
    return c


def _sl_variant(c, **kw):
    d = {k: c[k] for k in _SL_CFG_KEYS}
    d.update(kw)
    return _sl_cfg(d.pop("s"), **d)


def _sl_s4_box(rb, ks):
    if ks > 1:
        return rb // ks
    for bw in (2048, 1152, 1024, 512, 256, 128):
        if rb % bw == 0:
            return bw
    return 0


def _sl_prep(L, c, pat):
    """Derive chunk / grid / smem / occupancy / regs for config c on pattern pat (into c). False if it cannot run."""
    mod, sms = L["mod"], L["sms"]
    c["bw"] = 0
    if c["s"] == "S4":
        c["bw"] = _sl_s4_box(pat["rb"], c["ks"])
        c["ck"] = c["bw"] * c["bh"]
    elif c["s"] == "S3":
        rowb = pat["rb"] if c["mode"] == "G" else (pat["rb"] // c["ks"] if c["mode"] == "K" else 0)
        c["ck"] = c["ck0"] if not rowb else rowb * max(1, min(512, c["ck0"] // rowb))
    else:
        c["ck"] = 0
    smem = mod.sl_smem(c["kid"], c["T"], c["ck"], c["st"], c["v"])
    key = (c["kid"], c["T"], smem)
    if key not in L["attrs"]:
        L["attrs"][key] = list(mod.sl_attrs(c["kid"], c["T"], smem))
    at = L["attrs"][key]
    if at[4] is None or at[4] < 1:
        return False
    c["occ"], c["regs"], c["lmem"], c["smem"] = at[4], at[0], at[1], smem
    c["ecps"] = min(c["cps"], at[4])
    c["grid"] = c["ecps"] * sms
    c["infl"] = _sl_infl(c)
    return True


def _sl_args(L, c, pat, work=0, red=None):
    mats = list(pat["mats"]) + [pat["mats"][0]] * 4
    ptrs = mats[:4] + [pat.get("idx_ptr", 0), L["sink"].data_ptr(), L["flag"].data_ptr(),
                       red.data_ptr() if red is not None else 0]
    a = [c["grid"], c["T"], pat["rows"], pat.get("table_rows", 0), int(LAB_SPIN_BUDGET), pat.get("nmat", 1),
         pat["rb"], c["ck"], c["il"], c["st"], work, c["ln"], c["v"], c["bw"], c["bh"], c["ks"],
         red.numel() if red is not None else 0]
    return ptrs, a


def _sl_launcher(L, c, pat, work=0, red=None):
    mod = L["mod"]
    ptrs, a = _sl_args(L, c, pat, work, red)
    kid = c["kid"]

    def fn():
        rc = mod.sl_launch(kid, ptrs, a)
        if rc != 0:
            raise RuntimeError("sl_launch(%s) rc=%d" % (_sl_name(c), rc))
    return fn


def _sl_verify(L, c, pat, work=0):
    """One eager launch with a zeroed sink: checksum vs torch, abort flag. Returns 'ok' / 'BAD' / 'TIMEOUT'."""
    L["sink"].zero_()
    L["flag"].zero_()
    _sl_launcher(L, c, pat, work)()
    _sl_sync(L)
    if int(L["flag"].item()) != 0:
        return "TIMEOUT"
    got = int(L["sink"][:1024].sum().item()) & 0xFFFFFFFF
    return "ok" if got == pat["exp"] else "BAD"


def _sl_time_big(L, c, pat, reps, work=0):
    torch = L["torch"]
    fn = _sl_launcher(L, c, pat, work)
    e0, e1 = L["e0"], L["e1"]
    best = 1e30
    for _ in range(reps):
        _sl_flush(L)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1))
    if int(L["flag"].item()) != 0:
        return None
    return best


def _sl_graph_us(L, fn, reps):
    """Capture fn in a CUDA graph; replay `reps` times, each after an L2 flush, each timed alone. (median, min) us."""
    torch = L["torch"]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    e0, e1 = L["e0"], L["e1"]
    ts = []
    for _ in range(reps):
        _sl_flush(L)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    del g
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def _sl_note_err(L, c, e):
    L["n_err"] += 1
    L["consec_err"] += 1
    if len(L["errs"]) < 3:
        L["errs"].append("%s: %s" % (_sl_name(c) if c else "-", _sl_err(e, 160)))
        L["P"]("SL ERR %s" % L["errs"][-1])
    if L["consec_err"] >= 6:
        L["dead"] = True
        L["P"]("SL: 6 consecutive errors, stopping all measurements (CUDA context may be poisoned)")


def _sl_timeout(L, c, where):
    L["n_to"] += 1
    L["broken"].add(c["s"])
    L["flag"].zero_()
    L["P"]("SL TIMEOUT %s on %s: abort flag set, %s disabled for the rest of the lab" % (_sl_name(c), where, c["s"]))


def _sl_skip(L, c):
    return L["dead"] or c["s"] in L["broken"] or _sl_left(L) < 0


def _sl_measure_big(L, c, pat, tag, work=0):
    """Big stream: verify (checksum + abort flag), then min of LAB_REPS_BIG timed launches."""
    if _sl_skip(L, c):
        return None
    try:
        if not _sl_prep(L, c, pat):
            return None
        chk = _sl_verify(L, c, pat, work)
        if chk == "TIMEOUT":
            _sl_timeout(L, c, tag)
            return None
        if chk != "ok":
            L["n_bad"] += 1
        ms = _sl_time_big(L, c, pat, LAB_REPS_BIG, work)
        if ms is None:
            _sl_timeout(L, c, tag + " timed")
            return None
        tbs = pat["bytes"] / (ms * 1e-3) / 1e12
        L["consec_err"] = 0
        L["n_ok"] += 1
        r = {"c": c, "tbs": tbs, "infl": c["infl"], "chk": chk, "name": _sl_name(c)}
        if LAB_VERBOSE:
            L["P"]("SLV %s %-24s w%d %5.2f TB/s %4dK r%d l%d o%d %s" % (
                tag, r["name"], work, tbs, c["infl"] // 1024, c["regs"], c["lmem"], c["occ"], chk))
        return r
    except Exception as e:
        _sl_note_err(L, c, e)
        return None


def _sl_run_list(L, cfgs, pat, tag):
    out = []
    for c in cfgs:
        r = _sl_measure_big(L, c, pat, tag)
        if r is not None:
            out.append(r)
    return out


def _sl_curve(rows):
    """'X:T' = best TB/s among configs with <= X KiB/SM in flight (cumulative max; printed where it improves)."""
    cells, best = [], 0.0
    for b in _SL_INFL_BINS:
        cand = [r["tbs"] for r in rows if r["infl"] <= b * 1024]
        if cand:
            m = max(cand)
            if m > best + 0.005 or not cells:
                best = max(best, m)
                cells.append("%d:%.2f" % (b, best))
    return " ".join(cells)


def _sl_knee(rows, frac, ref=None):
    if not rows:
        return None
    ref = ref if ref is not None else max(r["tbs"] for r in rows)
    ok = [r for r in rows if r["tbs"] >= frac * ref]
    return min(ok, key=lambda r: (r["infl"], -r["tbs"])) if ok else None


def _sl_print_pattern(L, tag, desc, rows):
    """Per pattern: per scheme one cumulative curve and the top configs; then the 90/95% knees of the best."""
    P = L["P"]
    if not rows:
        P("%s %s: no results" % (tag, desc))
        return None
    best = max(rows, key=lambda r: r["tbs"])
    P("%s %s | best cfg TB/s %%4.8 KiB/SM regs occ chk | X:T = best TB/s with <= X KiB/SM" % (tag, desc))
    for s in ("S1", "S2", "S3", "S4"):
        rs = [r for r in rows if r["c"]["s"] == s]
        if not rs:
            continue
        r = max(rs, key=lambda q: q["tbs"])
        c = r["c"]
        P("%s %-21s %4.2f %2.0f%% %3dK r%d o%d %s | %s" % (tag, r["name"], r["tbs"], 100 * r["tbs"] * 1e12 / _SL_PEAK,
                                                       r["infl"] // 1024, c["regs"], c["occ"], r["chk"], _sl_curve(rs)))
    return best   # the 90/95% knees are in the SUM lines


def _sl_top(rows, s, n, key=None):
    rs = sorted([r for r in rows if r["c"]["s"] == s], key=lambda r: -r["tbs"])
    out, seen = [], set()
    for r in rs:
        k = key(r["c"]) if key else r["name"]
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
        if len(out) >= n:
            break
    return out


# ------------------------------------------------- config lists ------------------------------------------------------
def _sl_cfgs_s1_ref():
    """Reference points only: lab M3 already measured plain loads on 1 GiB (4.29-4.48 TB/s at >= 32 KiB/SM)."""
    return [_sl_cfg("S1", u=u, T=T, cps=cps) for u, T, cps in
            ((2, 256, 2), (4, 256, 2), (8, 256, 1), (2, 512, 2), (4, 256, 4), (8, 256, 2), (8, 128, 4), (4, 512, 2))]


def _sl_cfgs_s1_gather(rb):
    out = []
    for T in (256, 512):
        for u in _SL_U3:
            for cps in (1, 2, 4):
                infl = cps * T * 16 * u
                if T * cps < 1024 or T * cps > 2048 or infl > 256 * 1024 or (u * T * 16) < rb:
                    continue
                out.append(_sl_cfg("S1", mode="G", u=u, T=T, cps=cps))
    return out


def _sl_cfgs_s2(gather, rb=16):
    out = []
    for st in _SL_ST2:
        for T in (128, 256):
            for v in (1, 2, 4):
                for cps in (1, 2):
                    smem = st * T * 16 * v
                    infl = cps * (st - 1) * T * 16 * v
                    if smem * cps > 220 * 1024 or infl < 8192 or infl > 192 * 1024:
                        continue
                    if gather and (T != 256 or (T * 16 * v) % rb):
                        continue
                    out.append(_sl_cfg("S2", mode="G" if gather else "C", st=st, T=T, v=v, cps=cps))
    return out


def _sl_cfgs_s3(gather, rb=16):
    out = []
    for st in (2, 3, 4, 6, 8):
        for ck in (4096, 8192, 16384, 32768):
            for cps in (1, 2):
                smem = st * ck + 256
                infl = cps * (st - 1) * ck
                if smem * cps > 220 * 1024 or infl < 8192 or infl > 192 * 1024:
                    continue
                if gather and (st == 2 or ck > 16384 or ck // rb > 512):
                    continue
                out.append(_sl_cfg("S3", mode="G" if gather else "C", st=st, ck0=ck, cps=cps, nc=4))
    if gather:
        out += [_sl_cfg("S3", mode="G", st=st, ck0=8192, cps=4, nc=4) for st in (4, 6)]
    return out


def _sl_cfgs_s4():
    out = []
    for st in (2, 3, 4, 6):
        for bh in (4, 8, 16):
            for cps in (1, 2):
                if st * 2048 * bh * cps > 220 * 1024:
                    continue
                out.append(_sl_cfg("S4", st=st, bh=bh, cps=cps, nc=4))
    return out


# ------------------------------------------------- patterns ----------------------------------------------------------
def _sl_sum32(torch, t):
    return int(torch.sum(t.view(torch.int32), dtype=torch.int64).item()) & 0xFFFFFFFF


def _sl_pat_contig(L, name, offs, rows, rb, exp=True):
    torch, big = L["torch"], L["big"]
    mb = rows * rb
    e = 0
    if exp:
        for o in offs:
            e = (e + _sl_sum32(torch, big[o:o + mb])) & 0xFFFFFFFF
    return {"kind": "C", "name": name, "mats": [L["base"] + o for o in offs], "nmat": len(offs), "rows": rows,
            "rb": rb, "bytes": len(offs) * mb, "exp": e}


def _sl_pat_gather(L, rb, region_bytes, nrows):
    torch, big = L["torch"], L["big"]
    table_rows = region_bytes // rb
    perm = torch.randperm(table_rows, device=_SL_DEVICE)
    idx = perm[:nrows].to(torch.int32).contiguous()
    del perm
    tab = big[: table_rows * rb].view(torch.int32).view(table_rows, rb // 4)
    exp = int(tab.index_select(0, idx.long()).sum(dtype=torch.int64).item()) & 0xFFFFFFFF
    return {"kind": "G", "name": "g%d" % rb, "mats": [L["base"]], "nmat": 1, "rows": nrows, "rb": rb,
            "table_rows": table_rows, "idx": idx, "idx_ptr": idx.data_ptr(), "bytes": nrows * rb, "exp": exp}


def _sl_small_sets(L, nmat, rows, rb):
    """K disjoint copies (sets) of nmat matrices inside the big buffer, 1 MiB gaps; K launches = K cold sets."""
    mb = rows * rb
    mstride = ((mb + _SL_MIB - 1) // _SL_MIB + 1) * _SL_MIB
    sstride = nmat * mstride
    K = max(1, min(_SL_KMAX, _SL_BIG // sstride))
    return [[k * sstride + j * mstride for j in range(nmat)] for k in range(K)]


def _sl_size(name):
    for nm, nmat, rows, rb in _SL_SMALL:
        if nm == name:
            return nm, nmat, rows, rb
    return None


# ------------------------------------------------- small-matrix measurement ------------------------------------------
def _sl_measure_small(L, c, name, work=0, kx=True, extra=None):
    """Graph-replay timing of config c on one small size (cold L2). Returns dict: e1/ek eff TB/s (bytes over replay
    time minus the empty-graph floor), t1/tk raw us, chk. extra(): an op appended to the 1x graph (e.g. a reduce)."""
    if _sl_skip(L, c):
        return None
    try:
        nm, nmat, rows, rb = _sl_size(name)
        sets = _sl_small_sets(L, nmat, rows, rb)
        pat0 = _sl_pat_contig(L, nm, sets[0], rows, rb)
        if not _sl_prep(L, c, pat0):
            return None
        chk = _sl_verify(L, c, pat0, work)
        if chk == "TIMEOUT":
            _sl_timeout(L, c, nm)
            return None
        if chk != "ok":
            L["n_bad"] += 1
        f1 = _sl_launcher(L, c, pat0, work)
        if extra is not None:
            def f1x():
                f1()
                extra()
            t1, _ = _sl_graph_us(L, f1x, LAB_REPS_1X)
        else:
            t1, _ = _sl_graph_us(L, f1, LAB_REPS_1X)
        tk, K = None, 1
        if kx and len(sets) > 1:
            fns = [f1] + [_sl_launcher(L, c, _sl_pat_contig(L, nm, offs, rows, rb, exp=False), work)
                          for offs in sets[1:]]

            def fk():
                for f in fns:
                    f()
            tk, _ = _sl_graph_us(L, fk, LAB_REPS_KX)
            K = len(fns)
        if int(L["flag"].item()) != 0:
            _sl_timeout(L, c, nm + " graph")
            return None
        fl = L["floor1"]
        e1 = pat0["bytes"] / max(t1 - fl, 0.05) * 1e-6
        ek = pat0["bytes"] / max((tk - fl) / K, 0.05) * 1e-6 if tk is not None else None
        L["consec_err"] = 0
        L["n_ok"] += 1
        return {"e1": e1, "ek": ek, "t1": t1, "tk": tk, "K": K, "chk": chk, "bytes": pat0["bytes"]}
    except Exception as e:
        _sl_note_err(L, c, e)
        return None


def _sl_fit(pts):
    """Least squares t = t0 + bytes / BW over (bytes, t_us) points. Returns (t0 us, BW TB/s) or None."""
    if len(pts) < 3:
        return None
    n = float(len(pts))
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    if den <= 0:
        return None
    slope = (n * sxy - sx * sy) / den   # us per byte
    if slope <= 0:
        return None
    return (sy - slope * sx) / n, 1e-6 / slope


def _sl_fmt(x, w=4):
    return ("%.2f" % x) if x is not None else "-"


# ------------------------------------------------- parts -------------------------------------------------------------
def _sl_part_A(L):
    pat = _sl_pat_contig(L, "A", [0], _SL_A_BYTES // 4096, 4096)
    L["patA"] = pat
    rows = _sl_run_list(L, _sl_cfgs_s1_ref() + _sl_cfgs_s2(False) + _sl_cfgs_s3(False), pat, "A")
    extra = []
    for r in _sl_top(rows, "S3", 2):   # consumer-warp count
        for nc in (1, 2, 8):
            extra.append(_sl_variant(r["c"], nc=nc))
    for s in ("S1", "S3"):             # N-split (slice) vs chunk-interleaved split
        for r in _sl_top(rows, s, 1):
            extra.append(_sl_variant(r["c"], il=1))
    rows_x = _sl_run_list(L, extra, pat, "A")
    L["A"] = rows + [r for r in rows_x if not r["c"]["il"]]
    _sl_print_pattern(L, "A", "contig 1GiB, N-split (S1 = reference points; nN = consumer warps)", L["A"])
    cells = []
    for r in rows_x:
        if r["c"]["il"]:
            base = [q for q in rows if q["name"] == _sl_name(_sl_variant(r["c"], il=0))]
            cells.append("%s %.2f->%.2f" % (r["name"][:-3], base[0]["tbs"] if base else 0.0, r["tbs"]))
    if cells:
        L["P"]("A slice->interleave: " + " | ".join(cells))


def _sl_pick_small(L):
    """Configs for the small-matrix table: best 2 per scheme from A (distinct CTAs/SM) + interleaved twins."""
    picks = []
    for s in ("S1", "S2", "S3"):
        for r in _sl_top(L.get("A", []), s, 2, key=lambda c: c["cps"]):
            picks.append(_sl_variant(r["c"]))
    for s in ("S1", "S3"):
        t = _sl_top(L.get("A", []), s, 1)
        if t:
            picks.append(_sl_variant(t[0]["c"], il=1))
    return picks


def _sl_part_B(L, picks, tag="B"):
    P = L["P"]
    res = L.setdefault("B", {})
    sizes = [(nm, nmat, rows, rb, len(_sl_small_sets(L, nmat, rows, rb))) for nm, nmat, rows, rb in _SL_SMALL]
    for c in picks:
        for nm, nmat, rows, rb, K in sizes:
            if _sl_left(L) < 0 or L["dead"]:
                break
            r = _sl_measure_small(L, dict(c), nm)
            if r is not None:
                res.setdefault(_sl_name(c), {})[nm] = r
    hdr = " ".join("%-9s" % ("%s/%d" % (nm, K)) for nm, nmat, rows, rb, K in sizes)
    P("%s whole matrices, CUDA graph, cold L2; eff TB/s = bytes/(t - floor %.1f us); cell = 1 launch | K launches on K "
      "cold copies (name/K); fit over W2..pool64M: t1-floor = t0 + bytes/BW" % (tag, L["floor1"]))
    P("%s %-20s %s" % (tag, "config", hdr))
    for c in picks:
        name = _sl_name(c)
        rr = res.get(name, {})
        if not rr:
            continue
        cells, pts = [], []
        for nm, nmat, rows, rb, K in sizes:
            r = rr.get(nm)
            cells.append("%-9s" % (("%s|%s" % (_sl_fmt(r["e1"]), _sl_fmt(r["ek"]))) if r else "-"))
            if r and nm != "ctl256M":
                pts.append((r["bytes"], r["t1"] - L["floor1"]))
        fit = _sl_fit(pts)
        rr["_fit"] = fit
        P("%s %-20s %s %s" % (tag, name, " ".join(cells), ("t0 %.2fus BW %.2f" % fit) if fit else ""))
    best_us = []
    for nm, nmat, rows, rb, K in sizes:
        cand = [(v[nm]["t1"], k) for k, v in res.items() if nm in v]
        if cand:
            t, k = min(cand)
            best_us.append("%s %.1f" % (nm, t))
    if best_us:
        P("%s best raw t1 us incl. floor: %s" % (tag, " ".join(best_us)))


def _sl_part_B2(L):
    """MoE bs1: 1 and 3 experts (W13, 18.9 MB each) over all SMs. N-split vs K-split 2/4 (S1 LDG, S3 bulk), then the
    split-K reduction: fp32 atomics fused into the stream kernel vs a second reduce kernel in the same graph."""
    P = L["P"]
    torch = L["torch"]
    A = L.get("A", [])
    t1 = _sl_top(A, "S1", 1)
    t3 = _sl_top(A, "S3", 1)
    if not t1 and not t3:
        return
    fams = []
    if t1:
        c = t1[0]["c"]
        fams.append(("S1", [_sl_variant(c)] + [_sl_variant(c, mode="K", ks=ks) for ks in (2, 4)]))
    if t3:
        c = t3[0]["c"]
        fams.append(("S3", [_sl_variant(c)] + [_sl_variant(c, mode="K", ks=ks) for ks in (2, 4)]))
    L["B2"] = {}
    P("B2 MoE bs1 split (eff TB/s, 1 launch per graph, cold L2): k1 = N-split rows, kN = K-split into N row segments")
    for nm in ("W13x1", "W13x3"):
        cells = []
        for fam, cfgs in fams:
            vals = []
            for c in cfgs:
                r = _sl_measure_small(L, c, nm, kx=False)
                vals.append(r["e1"] if r else None)
                L["B2"][(nm, _sl_name(c))] = r
            cells.append("%s(%s) k1/k2/k4 %s" % (fam, _sl_name(cfgs[0])[3:], "/".join(_sl_fmt(v) for v in vals)))
            L.setdefault("B2rows", []).append((nm, fam, vals))
        P("B2 %-6s %s" % (nm, " | ".join(cells)))
    # split-K reduction cost on W13x3 with the S3 k2 / k4 configs (fallback S1)
    fam = fams[-1][1]
    nm, nmat, rows, rb = _sl_size("W13x3")
    cells = []
    for c in fam[1:]:
        if _sl_skip(L, c):
            break
        ks = c["ks"]
        try:
            base = L["B2"].get((nm, _sl_name(c)))
            if base is None:
                continue
            red = torch.zeros(nmat * rows * ks, dtype=torch.float32, device=_SL_DEVICE)
            part = torch.zeros(ks, nmat * rows, dtype=torch.float32, device=_SL_DEVICE)
            out = torch.zeros(nmat * rows, dtype=torch.float32, device=_SL_DEVICE)
            sets = _sl_small_sets(L, nmat, rows, rb)
            pat0 = _sl_pat_contig(L, nm, sets[0], rows, rb, exp=False)
            _sl_prep(L, c, pat0)
            fa = _sl_launcher(L, c, pat0, 0, red)
            ta, _ = _sl_graph_us(L, fa, LAB_REPS_1X)
            rk = _sl_measure_small(L, c, nm, kx=False, extra=lambda: torch.sum(part, 0, out=out))
            cells.append("k%d: fused atomics +%.2f us, 2nd reduce kernel +%.2f us (base %.1f us)" % (
                ks, ta - base["t1"], (rk["t1"] - base["t1"]) if rk else float("nan"), base["t1"]))
            L.setdefault("B2red_s", []).append("k%d atomics +%.2f 2nd-kernel +%s us" % (
                ks, ta - base["t1"], ("%.2f" % (rk["t1"] - base["t1"])) if rk else "-"))
            del red, part, out
        except Exception as e:
            _sl_note_err(L, c, e)
    L["B2red"] = cells
    if cells:
        P("B2 split-K reduce on W13x3 (%s): %s" % (_sl_name(fam[0]), " | ".join(cells)))


def _sl_part_C(L):
    P = L["P"]
    L["C"] = {}
    for rb in (64, 128, 256):
        if _sl_left(L) < 0 or L["dead"]:
            break
        try:
            pat = _sl_pat_gather(L, rb, _SL_BIG, _SL_G_BYTES // rb)
        except Exception as e:
            _sl_note_err(L, None, e)
            continue
        tag = "C%d" % rb
        rows = _sl_run_list(L, _sl_cfgs_s1_gather(rb) + _sl_cfgs_s2(True, rb) + _sl_cfgs_s3(True, rb), pat, tag)
        extra = []
        for r in _sl_top(rows, "S3", 2):
            extra.append(_sl_variant(r["c"], ln=1))
        t3 = _sl_top(rows, "S3", 1)
        if t3:
            extra.append(_sl_variant(t3[0]["c"], nc=8))
        rows += _sl_run_list(L, extra, pat, tag)
        L["C"][rb] = rows
        _sl_print_pattern(L, tag, "gather %dB rows, %.1fM idx over a %.2f GB table, +4 B idx/row (l1 = 1 issuing lane)"
                          % (rb, pat["rows"] / 1e6, pat["table_rows"] * rb / 1e9), rows)
        if rb == 128 and _sl_left(L) > 0:
            try:
                pat2 = _sl_pat_gather(L, rb, _SL_G_REGION_SMALL, _SL_G_BYTES // rb)
                cells = []
                for s in ("S1", "S2", "S3"):
                    t = _sl_top(rows, s, 1)
                    if t:
                        r2 = _sl_measure_big(L, _sl_variant(t[0]["c"]), pat2, "C128r")
                        if r2:
                            cells.append("%s %.2f->%.2f" % (s, t[0]["tbs"], r2["tbs"]))
                L["C128r"] = cells
                P("C128 table 1.34GB->0.32GB (best per scheme): " + " | ".join(cells))
                del pat2
            except Exception as e:
                _sl_note_err(L, None, e)
        del pat


def _sl_part_D(L):
    """Streaming while consuming like the NVFP4 GEMM: work 0 = checksum only, 1 = + Marlin dequant + scale FMA,
    2 = + 4 mma.sync m16n8k16 per thread-16B. 1 GiB stream (TB/s) and W13x3 (eff TB/s, 1 launch per graph)."""
    P = L["P"]
    pat = L.get("patA")
    A = L.get("A", [])
    if pat is None or not A:
        return
    cf = []
    t = _sl_top(A, "S1", 1)
    c1 = t[0]["c"] if t else _sl_cfg("S1", u=4, T=256, cps=2)
    cf.append(_sl_variant(c1, wk=True))
    cf.append(_sl_variant(c1, wk=True, pipe=True, u=c1["u"] if c1["u"] in (2, 4) else 4))
    cf.append(_sl_cfg("S1", wk=True, u=4, T=512, cps=1))
    t = _sl_top(A, "S2", 1)
    if t:
        cf.append(_sl_variant(t[0]["c"], wk=True))
    t = _sl_top(A, "S3", 1)
    c3 = t[0]["c"] if t else _sl_cfg("S3", st=4, ck0=16384, cps=1)
    for nc in (4, 8, 12):
        cf.append(_sl_variant(c3, wk=True, nc=nc))
    if c3["cps"] == 1 and c3["st"] * c3["ck0"] * 2 <= 200 * 1024:
        cf.append(_sl_variant(c3, wk=True, nc=8, cps=2))
    L["D"] = []
    P("D consumer work w0/w1/w2 (w1 = Marlin dequant + scale FMA, w2 = w1 + 4 mma.sync m16n8k16 per 16 B): "
      "1GiB TB/s | W13x3 eff TB/s | regs/local B/occ of the WORK kernel (w0 too: more regs than A's lean kernels)")
    for c in cf:
        if _sl_left(L) < 0 or L["dead"]:
            break
        big, small = [], []
        for w in (0, 1, 2):
            r = _sl_measure_big(L, c, pat, "D", work=w)
            big.append(r["tbs"] if r else None)
        for w in (0, 2):
            r = _sl_measure_small(L, c, "W13x3", work=w, kx=False)
            small.append(r["e1"] if r else None)
        regs = "r%s/l%s/o%s" % (c.get("regs", "?"), c.get("lmem", "?"), c.get("occ", "?"))
        L["D"].append((_sl_name(c), big, small, regs))
        P("D %-20s %s | W13x3 w0 %s w2 %s | %s" % (_sl_name(c), "/".join(_sl_fmt(v) for v in big),
                                                  _sl_fmt(small[0]), _sl_fmt(small[1]), regs))


def _sl_part_E(L):
    P = L["P"]
    if not L.get("s4_built"):
        L["E_msg"] = "skipped: S4 not compiled (%s)" % L.get("build_err", "LAB_S4 off")
        P("E S4 " + L["E_msg"])
        return
    st = int(L["mod"].sl_s4_init())
    if st != 0:
        L["E_msg"] = "skipped: cuTensorMapEncodeTiled entry point unavailable (status %d)" % st
        P("E S4 " + L["E_msg"])
        return
    pat = L.get("patA")
    rows = _sl_run_list(L, _sl_cfgs_s4(), pat, "E")
    L["E"] = rows
    _sl_print_pattern(L, "E", "TMA 2-D boxes [h rows x 2048 B] over the 1GiB stream", rows)
    if not rows:
        return
    best = _sl_top(rows, "S4", 1)[0]["c"]
    _sl_part_B(L, [_sl_variant(best)], tag="E")
    cells = []
    for ks in (2, 4):
        r = _sl_measure_small(L, _sl_variant(best, ks=ks), "W13x3", kx=False)
        cells.append("k%d %s" % (ks, _sl_fmt(r["e1"] if r else None)))
    vals = []
    for w in (0, 1, 2):
        r = _sl_measure_big(L, _sl_variant(best, wk=True, nc=8), pat, "E", work=w)
        vals.append(r["tbs"] if r else None)
    L["E_more"] = "W13x3 %s | work n8 w0/w1/w2 1GiB %s" % (" ".join(cells), "/".join(_sl_fmt(v) for v in vals))
    P("E %s: %s" % (_sl_name(best), L["E_more"]))


# ------------------------------------------------- summary -----------------------------------------------------------
def _sl_summary(L):
    out = ["===== STREAM SUMMARY stream_lab v1 (TB/s = 1e12 B/s; % of 4.8; KiB/SM = bytes in flight per SM) ====="]
    d = L.get("dev", {})
    out.append("SUM dev %s SMs=%s clk=%s MHz L2=%s MiB cudart=%s | build %s | floor graph 1x %.1f us, +%.2f us per "
               "extra launch in a graph" % (d.get("name", "?"), d.get("sms", "?"), d.get("clk", "?"), d.get("l2", "?"),
                                            d.get("cudart", "?"),
                                            ("%.0fs S4 %s" % (L["build_s"], "on" if L.get("s4_built") else "off"))
                                            if L.get("build_s") else "FAILED", L.get("floor1", 0.0),
                                            L.get("floor_add", 0.0)))

    def pat_line(tag, rows):
        if not rows:
            return "SUM %s: no results" % tag
        best = max(rows, key=lambda r: r["tbs"])
        per = []
        for s in ("S1", "S2", "S3", "S4"):
            rs = [r for r in rows if r["c"]["s"] == s]
            if rs:
                b = max(rs, key=lambda r: r["tbs"])
                k = _sl_knee(rs, 0.9)
                per.append("%s %.2f@%dK(90%%@%dK)" % (s, b["tbs"], b["infl"] // 1024, k["infl"] // 1024))
        k90, k95 = _sl_knee(rows, 0.9), _sl_knee(rows, 0.95)
        return "SUM %s best %s %.2f (%.0f%%) @%dK | 90%% @%dK %s, 95%% @%dK %s | %s" % (
            tag, best["name"], best["tbs"], 100 * best["tbs"] * 1e12 / _SL_PEAK, best["infl"] // 1024,
            k90["infl"] // 1024, k90["name"], k95["infl"] // 1024, k95["name"], " ".join(per))

    out.append(pat_line("A contig1GiB", L.get("A", []) + L.get("E", [])))
    B = L.get("B", {})
    if B:
        cells, fits = [], []
        for nm, nmat, rows, rb in _SL_SMALL:
            cand = [(v[nm]["e1"], k) for k, v in B.items() if nm in v]
            candk = [v[nm]["ek"] for k, v in B.items() if nm in v and v[nm]["ek"] is not None]
            if cand:
                e1, k = max(cand)
                cells.append("%s %.2f|%s" % (nm, e1, _sl_fmt(max(candk) if candk else None)))
        for k, v in B.items():
            if v.get("_fit"):
                fits.append((v["_fit"][1], v["_fit"][0], k))
        out.append("SUM B whole-matrix eff TB/s best 1x|Kx: " + " ".join(cells))
        if fits:
            fits.sort(reverse=True)
            out.append("SUM B fit t-floor = t0 + bytes/BW: " + " | ".join("%s t0 %.2fus BW %.2f" % (k, t0, bw)
                                                                          for bw, t0, k in fits[:2]))
    if L.get("B2rows"):
        out.append("SUM B2 MoE split eff TB/s k1/k2/k4: " + " | ".join(
            "%s %s %s" % (nm, fam, "/".join(_sl_fmt(v) for v in vals)) for nm, fam, vals in L["B2rows"]))
    if L.get("B2red_s"):
        out.append("SUM B2 split-K reduce W13x3: " + " | ".join(L["B2red_s"]))
    for rb, rows in sorted(L.get("C", {}).items()):
        out.append(pat_line("C gather%dB" % rb, rows))
        l1 = [r for r in rows if r["c"]["s"] == "S3" and r["c"]["ln"] < 32]
        if l1:
            out[-1] += " | S3 1-lane %.2f" % max(r["tbs"] for r in l1)
        s1 = [r for r in rows if r["c"]["s"] == "S1"]
        if s1:
            out[-1] += " | S1 deepest %dK (reg/occupancy-capped)" % (max(r["infl"] for r in s1) // 1024)
    if L.get("C128r"):
        out.append("SUM C128 table 1.34->0.32 GB: " + " | ".join(L["C128r"]))
    if L.get("D"):
        out.append("SUM D w0/w1/w2 1GiB|W13x3 w0,w2: " + " ; ".join(
            "%s %s|%s,%s %s" % (n.replace(" ", ""), "/".join(_sl_fmt(v) for v in big), _sl_fmt(sm[0]), _sl_fmt(sm[1]),
                                rg.split("/")[0]) for n, big, sm, rg in L["D"]))
    if L.get("E_msg"):
        out.append("SUM E S4: " + L["E_msg"])
    elif L.get("E"):
        b = max(L["E"], key=lambda r: r["tbs"])
        out.append("SUM E S4 best %s %.2f @%dK | %s" % (b["name"], b["tbs"], b["infl"] // 1024, L.get("E_more", "")))
    out.append("SUM health: %d measurements ok, checksum BAD %d, timeouts %d, errors %d%s%s, lab %.0f s" % (
        L["n_ok"], L["n_bad"], L["n_to"], L["n_err"], (" (first: %s)" % L["errs"][0]) if L["errs"] else "",
        (" broken: " + ",".join(sorted(L["broken"]))) if L["broken"] else "", time.time() - L["t0"]))
    v = []
    A = L.get("A", []) + L.get("E", [])
    for s in ("S1", "S2", "S3", "S4"):
        rs = [r for r in A if r["c"]["s"] == s]
        if rs:
            ok4 = [r for r in rs if r["tbs"] >= 4.0]
            v.append("%s %s" % (s, ("y@%dK" % (min(r["infl"] for r in ok4) // 1024)) if ok4 else
                                "n(%.2f)" % max(r["tbs"] for r in rs)))
    if v:
        out.append("SUM VERDICT >=4.0 TB/s on 1GiB (min KiB/SM): " + " ".join(v))
    if B:
        vv = []
        for nm in ("W13x1", "W13x3", "qkv36M", "pool64M"):
            cand = [(x[nm]["e1"], k) for k, x in B.items() if nm in x]
            if cand:
                e, k = max(cand)
                vv.append("%s %s(%.2f %s)" % (nm, "y" if e >= 4.0 else "n", e, k))
        out.append("SUM VERDICT >=4.0 whole matrix 1x: " + " ".join(vv))
    vv = []
    for rb, rows in sorted(L.get("C", {}).items()):
        if rows:
            b = max(rows, key=lambda r: r["tbs"])
            vv.append("%dB %s(%.2f %s)" % (rb, "y" if b["tbs"] >= 4.0 else "n", b["tbs"], b["name"]))
    if vv:
        out.append("SUM VERDICT >=4.0 gather: " + " ".join(vv))
    if L.get("D"):
        w2 = [(big[2], n) for n, big, sm, rg in L["D"] if big[2] is not None]
        if w2:
            e, n = max(w2)
            out.append("SUM VERDICT best streaming with dequant+mma (w2, 1GiB): %.2f TB/s %s" % (e, n))
    out.append("===== END STREAM SUMMARY =====")
    return out


# ------------------------------------------------- driver ------------------------------------------------------------
def _sl_run(t0):
    L = {"t0": t0, "deadline": t0 + LAB_DEADLINE_S, "P": _sl_P, "attrs": {}, "broken": set(), "errs": [],
         "n_ok": 0, "n_bad": 0, "n_to": 0, "n_err": 0, "consec_err": 0, "dead": False, "torch": torch}
    P = L["P"]
    P("SL stream_lab v1 (check only): S1 LDG.128 | S2 cp.async | S3 bulk+mbarrier ring | S4 TMA 2-D; deadline %.0f s"
      % LAB_DEADLINE_S)
    try:
        pr = torch.cuda.get_device_properties(torch.cuda.current_device())
        L["sms"] = int(pr.multi_processor_count)
        L["dev"] = {"name": pr.name.replace("NVIDIA ", ""), "sms": L["sms"],
                    "l2": int(getattr(pr, "L2_cache_size", 0)) // _SL_MIB}
    except Exception as e:
        P("SL dev ERR " + _sl_err(e))
        L["sms"] = 132
        L["dev"] = {}
    mod = _sl_build(L)
    if mod is None:
        for line in _sl_summary(L):
            P(line)
        return
    L["mod"] = mod
    try:
        dv = list(mod.sl_dev())
        L["dev"].update({"clk": dv[0] // 1000, "cudart": dv[4]})
        P("SL dev attrs clk_kHz=%d SMs=%d L2=%d optin_smem=%d cudart=%d driver=%d S4built=%d" % tuple(dv[:7]))
        if dv[1] > 0:
            L["sms"] = int(dv[1])
    except Exception as e:
        P("SL dev attrs ERR " + _sl_err(e))
    try:
        L["big"] = torch.empty(_SL_BIG, dtype=torch.uint8, device=_SL_DEVICE)
        L["big"].view(torch.int32).random_()
        L["base"] = L["big"].data_ptr()
        L["flush"] = torch.ones(_SL_FLUSH // 4, dtype=torch.float32, device=_SL_DEVICE)
        L["sink"] = torch.zeros(1025, dtype=torch.int64, device=_SL_DEVICE)
        L["flag"] = torch.zeros(1, dtype=torch.int32, device=_SL_DEVICE)
        L["e0"] = torch.cuda.Event(enable_timing=True)
        L["e1"] = torch.cuda.Event(enable_timing=True)
        _sl_sync(L)
    except Exception as e:
        P("SL alloc ERR " + _sl_err(e))
        for line in _sl_summary(L):
            P(line)
        return
    try:   # replay floor: 1 empty launch per graph, and the cost of each extra launch inside a graph
        ce = _sl_cfg("S1", u=2, T=32, cps=1)
        ce.update({"kid": 1, "grid": 1, "ecps": 1, "ck": 0, "bw": 0})
        pe = {"mats": [L["base"]], "rows": 1, "rb": 16}
        fe = _sl_launcher(L, ce, pe)
        L["floor1"], _ = _sl_graph_us(L, fe, 30)

        def fe8():
            for _ in range(8):
                fe()
        f8, _ = _sl_graph_us(L, fe8, 15)
        L["floor_add"] = (f8 - L["floor1"]) / 7.0
        P("SL floor: graph replay of 1 empty launch %.2f us; +%.2f us per extra launch in the same graph" % (
            L["floor1"], L["floor_add"]))
    except Exception as e:
        L["floor1"], L["floor_add"] = 4.0, 0.0
        _sl_note_err(L, None, e)
    parts = [("A", lambda: _sl_part_A(L)), ("B", lambda: _sl_part_B(L, _sl_pick_small(L))),
             ("B2", lambda: _sl_part_B2(L)), ("C", lambda: _sl_part_C(L)), ("D", lambda: _sl_part_D(L))]
    if LAB_S4:
        parts.append(("E", lambda: _sl_part_E(L)))
    for name, fn in parts:
        if _sl_left(L) < 0 or L["dead"]:
            P("SL part %s skipped (%s)" % (name, "dead context" if L["dead"] else "deadline"))
            continue
        t = time.time()
        try:
            fn()
        except Exception as e:
            _sl_note_err(L, None, e)
        try:
            _sl_sync(L)
        except Exception as e:
            _sl_note_err(L, None, e)
        P("SL part %s done in %.1f s (t=%.0f s)" % (name, time.time() - t, time.time() - t0))
    for k in ("big", "flush", "sink", "flag"):
        L.pop(k, None)
    for line in _sl_summary(L):
        P(line)

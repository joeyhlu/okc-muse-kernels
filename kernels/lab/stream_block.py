"""okc stream_block v1: a reusable sm_90 device-side streaming pipeline, as a CUDA source string.

What it is: a ring of `stages` shared-memory slots fed by 1-D bulk async copies (cp.async.bulk, "TMA without a
tensor map"), each slot guarded by a full/empty mbarrier pair, with BOUNDED waits (a hang becomes a flag, never a
timeout of the whole run). One producer warp issues copies (one lane for contiguous chunks, all 32 lanes for row
gathers), NC consumer warps read the slots. Optional 2-D tensor-map copy wrapper for TMA tiles.

Why smem and not registers: plain LDG.128 already streams ~4.48 TB/s on H200 (lab M3, job 7516684e37bb45d1) but
needs >= 32 KiB in flight per SM; holding that in registers costs 16-32 regs/thread that a dequant+MMA consumer
cannot spare. The ring keeps 32-128 KiB in flight in shared memory with one issuing lane.

Use: paste STREAM_BLOCK_CUH verbatim into a load_inline CUDA source after <cuda_runtime.h> and <stdint.h>
(load_inline already prepends <cuda.h> and <cuda_runtime.h>). API and protocol: kernels/lab/stream_lab.md.
The lab kernels/lab/stream_lab.py embeds the identical text (scheme S3/S4) and measures it on the judge.
Nothing here has run on a GPU yet; it was syntax-checked with clang (sm_90 device pass) and its phase/parity/tx
protocol was model-checked on CPU (scratch emulate.py, see stream_lab.md "Validation").
"""

STREAM_BLOCK_VERSION = "v1"

STREAM_BLOCK_CUH = r"""
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

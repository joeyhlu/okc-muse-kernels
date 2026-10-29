// conv2.cu - R-rev-11 (notes/study/independent_review_oct28.md s20): a valid step-0 for the one-launch T <= 8 layer
// ("conveyor"). R-rev-6's skeleton streamed at only 2.5-2.9 TB/s with zero compute, and its consumer ALU proxy ran
// ~6x slower than the real GD / VMS bodies, so its absolute times said nothing. This version:
//   1. proves the stream first: --mode stream = the same byte schedule, no dependencies, free consumers;
//   2. models consumers by their MEASURED rate (ns per KB per SM, a clock64 spin per ring slot; GD / VMS ~21 ns/KB
//      at T = 1, i.e. ~1.4x the 4.4 TB/s stream share of 30 ns/KB), not by an ALU proxy;
//   3. keeps the layer's real dependency chain as grid counters + flags: attention after qkv, o after attention,
//      sbb, router -> top-8 -> expert ids (the producer may not issue expert bytes before them), pooler -> pre-expert
//      norm -> W13, W13 -> W2 per expert, combine -> unpooler -> final norm. Latency-bound chains are a clock64 spin
//      on the last-arriving CTA (attention, norms, top-8, combine).
// One CTA per SM: 1 producer warp (lane 0 issues cp.async.bulk into an NS x SB ring) + NCW consumer warps.
// Every replay (CUDA graph): 256 MB L2 flush read, counter reset, the conveyor. Span = first CTA entry -> last exit
// (%globaltimer); per-phase completion times relative to the entry.
//
// Build: nvcc -O3 -arch=sm_90a -std=c++17 -o conv2 conv2.cu
// Run:   ./conv2 --sched t1s --mode dep --dcons 21 --vcons 21 --reps 30
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int MAXPH = 64;
constexpr int CTR0 = 0;      // sync[0..63]   phase counters
constexpr int FLG0 = 64;     // sync[64..95]  flags
constexpr int RTE0 = 96;     // sync[96..127] routed expert ids
constexpr int NSYNC = 128;
constexpr unsigned long long TIMEOUT_NS = 20000000ull;   // bounded waits, no trap

struct Phase {
  long long off, bytes;   // static: offset into w; routed: offset inside the expert slab, bytes per expert
  int route_slot;         // -1 static
  int cls;                // consumer class: 0 dense, 1 expert, 2 attention KV
  int wait_flag, wait_ctr, issue_flag;
  int done_flag, chain_ns, write_route;
};
struct Sched { int nph, nroute; Phase ph[MAXPH]; };
__constant__ Sched c_s;

struct Args {
  const unsigned char* w;
  const unsigned char* xw;
  long long xstride;
  unsigned* sync;
  unsigned long long* tph;     // [MAXPH] phase completion (last arriver) time
  unsigned long long* tspan;   // [0] min entry, [1] max exit
  unsigned* err;
  const int* route_src;
  int cons_cyc_per_kb[3];      // consumer spin, SM cycles per KB, per class
  int dep;                     // 0: pure stream (ignore dependencies), 1: honor them
  int ilv;                     // 1: chunk c of a phase goes to CTA c % G (a moving front), 0: contiguous shares
  float cyc_per_ns;
  unsigned long long* mk;      // R-rev-16 marks [G][mkmax][3]: issue time, land time, phase (null = off)
  int mkmax;
};

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned atom_add_acqrel(unsigned* p, unsigned v) {
  unsigned o; asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(o) : "l"(p), "r"(v) : "memory");
  return o;
}
__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t b, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, unsigned par) {
  unsigned ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t b, unsigned par, unsigned* err) {
  if (mbar_try(b, par)) return;
  const unsigned long long t0 = gtime();
  while (!mbar_try(b, par)) if (gtime() - t0 > TIMEOUT_NS) { atomicOr(err, 1u); return; }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, unsigned bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void wait_flag(const unsigned* p, unsigned* err) {
  const unsigned long long t0 = gtime();
  while (ld_acquire(p) == 0u) if (gtime() - t0 > TIMEOUT_NS) { atomicOr(err, 2u); return; }
}
__device__ __forceinline__ void wait_geq(const unsigned* p, unsigned v, unsigned* err) {
  const unsigned long long t0 = gtime();
  while (ld_acquire(p) < v) if (gtime() - t0 > TIMEOUT_NS) { atomicOr(err, 4u); return; }
}
__device__ __forceinline__ void spin_cyc(long long cyc) {
  const long long t0 = clock64();
  while (clock64() - t0 < cyc) { }
}
// This CTA's contiguous share [lo, hi) of `bytes` (shares 256-B aligned).
__device__ __forceinline__ void share_of(long long bytes, int cta, int G, long long& lo, long long& hi) {
  const long long s = ((bytes + G - 1) / G + 255) & ~255ll;
  lo = min((long long)cta * s, bytes);
  hi = min(lo + s, bytes);
}

template <int NS, int SB, int NCW>
__global__ void __launch_bounds__((NCW + 1) * 32, 1) conv_kernel(Args a) {
  extern __shared__ __align__(1024) unsigned char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + NS * SB);
  uint64_t* empty = full + NS;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, G = gridDim.x, cta = blockIdx.x;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(su32(&full[s]), 1); mbar_init(su32(&empty[s]), NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    atomicMin(&a.tspan[0], gtime());
  }
  __syncthreads();
  if (warp == NCW) {
    if (lane == 0) {                                   // producer: this CTA's share of every phase, in order
      int slot = 0; unsigned par = 0, kk = 0;
      for (int p = 0; p < c_s.nph; ++p) {
        const Phase& P = c_s.ph[p];
        if (P.route_slot >= c_s.nroute) continue;
        const unsigned char* base = a.w + P.off;
        if (P.route_slot >= 0) {
          if (a.dep) wait_flag(&a.sync[FLG0 + P.issue_flag], a.err);
          const unsigned e = a.dep ? ld_acquire(&a.sync[RTE0 + P.route_slot]) : (unsigned)a.route_src[P.route_slot];
          base = a.xw + (long long)e * a.xstride + P.off;
        } else if (a.dep && P.issue_flag >= 0) {
          wait_flag(&a.sync[FLG0 + P.issue_flag], a.err);
        }
        long long lo, hi, step = SB;
        if (a.ilv) { lo = (long long)cta * SB; hi = P.bytes; step = (long long)G * SB; } else share_of(P.bytes, cta, G, lo, hi);
        for (long long off = lo; off < hi; off += step) {
          const unsigned len = (unsigned)min((long long)SB, hi - off);
          mbar_wait(su32(&empty[slot]), par ^ 1u, a.err);
          if (a.mk && kk < (unsigned)a.mkmax) {
            unsigned long long* m = a.mk + ((long long)cta * a.mkmax + kk) * 3;
            m[0] = gtime(); m[2] = (unsigned long long)p;
          }
          ++kk;
          mbar_expect_tx(su32(&full[slot]), len);
          bulk_g2s(su32(smem + slot * SB), base + off, len, su32(&full[slot]));
          if (++slot == NS) { slot = 0; par ^= 1u; }
        }
      }
    }
  } else {
    int slot = 0; unsigned par = 0, kc = 0;            // consumers: NCW warps, one modelled body per slot
    for (int p = 0; p < c_s.nph; ++p) {
      const Phase& P = c_s.ph[p];
      if (P.route_slot >= c_s.nroute) continue;
      if (a.dep && (P.wait_flag >= 0 || P.wait_ctr >= 0)) {
        if (threadIdx.x == 0) {
          if (P.wait_flag >= 0) wait_flag(&a.sync[FLG0 + P.wait_flag], a.err);
          if (P.wait_ctr >= 0) wait_geq(&a.sync[CTR0 + P.wait_ctr], (unsigned)G, a.err);
        }
        asm volatile("bar.sync 1, %0;" :: "r"(NCW * 32) : "memory");
      }
      long long lo, hi, step = SB;
      if (a.ilv) { lo = (long long)cta * SB; hi = P.bytes; step = (long long)G * SB; } else share_of(P.bytes, cta, G, lo, hi);
      const long long cpk = a.cons_cyc_per_kb[P.cls];
      for (long long off = lo; off < hi; off += step) {
        const unsigned len = (unsigned)min((long long)SB, hi - off);
        if (lane == 0) mbar_wait(su32(&full[slot]), par, a.err);
        if (a.mk && threadIdx.x == 0 && kc < (unsigned)a.mkmax) a.mk[((long long)cta * a.mkmax + kc) * 3 + 1] = gtime();
        ++kc;
        __syncwarp();
        if (cpk > 0) spin_cyc(cpk * len / 1024);
        __syncwarp();
        if (lane == 0) mbar_arrive(su32(&empty[slot]));
        if (++slot == NS) { slot = 0; par ^= 1u; }
      }
      asm volatile("bar.sync 1, %0;" :: "r"(NCW * 32) : "memory");
      if (threadIdx.x == 0) {
        const unsigned old = atom_add_acqrel(&a.sync[CTR0 + p], 1u);
        if (old == (unsigned)G - 1u) {                 // last CTA: the dependent chain, then the flag
          a.tph[p] = gtime();
          if (a.dep && P.chain_ns > 0) spin_cyc((long long)(P.chain_ns * a.cyc_per_ns));   // R-rev-16: chains only with dependencies
          if (P.write_route) {
            for (int r = 0; r < c_s.nroute; ++r) a.sync[RTE0 + r] = (unsigned)a.route_src[r];
            __threadfence();
          }
          if (P.done_flag >= 0) st_release(&a.sync[FLG0 + P.done_flag], 1u);
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(&a.tspan[1], gtime());
}

__global__ void flush_kernel(const uint4* buf, long long n16, unsigned* sink) {
  unsigned x = 0;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n16; i += (long long)gridDim.x * blockDim.x) {
    uint4 v;   // normal-policy loads (like the judge's torch.sum flush); __ldcs (evict-first) left earlier lines in L2
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(buf + i));
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (x == 0x9e3779b9u) *sink = x;
}

// R-rev-16: one load per `step` bytes of a region (TLB pre-touch after the flush)
__global__ void touch_kernel(const unsigned char* base, long long bytes, long long step, unsigned* sink) {
  const long long n = (bytes + step - 1) / step;
  unsigned x = 0;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    unsigned v;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(base + i * step));
    x ^= v;
  }
  if (x == 0x9e3779b9u) *sink = x;
}

// ---------------------------------------------------------------- host
static const double MB = 1e6;
struct Build {
  Sched s; std::vector<int> route; long long wbytes; double total_mb;
};
// Byte schedule of one T <= 8 layer (P12 dense sizes from notes/study/01-challenge.md x 12/16; router BF16; expert
// 28.31 MB = W13 18.87 + W2 9.44). Flags: 0 att, 1 sbb, 2 route, 3 x_e, 4 combine, 5 end.
static Build make_sched(const std::string& name, int experts, int att_ns, int full_kv, int sbb_ns, int route_ns,
                        int pnorm_ns, int comb_ns, int norm_ns, int attcomb_ns) {
  Build b; memset(&b.s, 0, sizeof(b.s));
  long long off = 0;
  auto add = [&](double mb, int cls, int wf, int wc, int iflag, int dflag, int chain, int wr, int slot, long long xoff) {
    Phase& P = b.s.ph[b.s.nph];
    P.bytes = (long long)(mb * MB) & ~255ll; P.cls = cls; P.wait_flag = wf; P.wait_ctr = wc; P.issue_flag = iflag;
    P.done_flag = dflag; P.chain_ns = chain; P.write_route = wr; P.route_slot = slot;
    if (slot < 0) { P.off = off; off += (P.bytes + 4095) & ~4095ll; } else P.off = xoff;
    return b.s.nph++;
  };
  const double qkv = 37.75 * 0.75, o = 16.78 * 0.75, router = 8.39, pooler = 67.11 * 0.75, unp = 8.39 * 0.75;
  const double w13 = 18.87, w2 = 9.44;
  int p_qkv = add(qkv, 0, -1, -1, -1, full_kv ? -1 : 0, full_kv ? 0 : att_ns, 0, -1, 0);
  if (full_kv) add(20.48, 2, -1, p_qkv, -1, 0, attcomb_ns, 0, -1, 0);   // split-KV attention over 80K keys
  add(o, 0, 0, -1, -1, 1, sbb_ns, 0, -1, 0);
  add(router, 0, 1, -1, -1, 2, route_ns, 1, -1, 0);
  add(pooler, 0, 1, -1, -1, 3, pnorm_ns, 0, -1, 0);
  std::vector<int> pa;
  for (int e = 0; e < experts; ++e) pa.push_back(add(w13, 1, 3, -1, 2, -1, 0, 0, e, 0));
  for (int e = 0; e < experts; ++e) add(w2, 1, -1, pa[e], 2, e == experts - 1 ? 4 : -1, e == experts - 1 ? comb_ns : 0, 0, e,
                                        (long long)(w13 * MB) & ~255ll);
  add(unp, 0, 4, -1, -1, 5, norm_ns, 0, -1, 0);
  b.s.nroute = experts; b.wbytes = off;
  for (int e = 0; e < experts; ++e) b.route.push_back((e * 11 + 3) % 32);
  double t = 0; for (int i = 0; i < b.s.nph; ++i) t += b.s.ph[i].bytes / MB;
  b.total_mb = t;
  (void)name;
  return b;
}

static int argi(int argc, char** argv, const char* k, int d) {
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], k)) return atoi(argv[i + 1]);
  return d;
}
static std::string args(int argc, char** argv, const char* k, const char* d) {
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], k)) return argv[i + 1];
  return d;
}

struct Region { const unsigned char* p; long long bytes; };

template <int NS, int SB, int NCW>
static void run(const Build& b, Args a, int G, int reps, cudaStream_t st, const uint4* fl, long long fl16, unsigned* sink,
                unsigned* sync, unsigned long long* tph, unsigned long long* tspan, std::vector<double>& spans,
                std::vector<double>& ph_med, const std::vector<Region>& touch, long long tstep, const std::string& mkout) {
  const size_t smem = (size_t)NS * SB + 2 * NS * 8;
  CK(cudaFuncSetAttribute(conv_kernel<NS, SB, NCW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  CK(cudaMemsetAsync(sync, 0, NSYNC * 4, st));
  CK(cudaMemsetAsync(tph, 0, MAXPH * 8, st));
  CK(cudaMemsetAsync(tspan, 0xFF, 8, st));
  CK(cudaMemsetAsync(tspan + 1, 0, 8, st));
  flush_kernel<<<4 * G, 512, 0, st>>>(fl, fl16, sink);
  if (tstep > 0)
    for (const Region& r : touch) touch_kernel<<<16, 256, 0, st>>>(r.p, r.bytes, tstep, sink);
  conv_kernel<NS, SB, NCW><<<G, (NCW + 1) * 32, smem, st>>>(a);
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  std::vector<std::vector<double>> phs(b.s.nph);
  for (int r = 0; r < reps + 3; ++r) {
    CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
    unsigned long long hs[2], hp[MAXPH];
    CK(cudaMemcpy(hs, tspan, 16, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hp, tph, MAXPH * 8, cudaMemcpyDeviceToHost));
    if (r < 3) continue;
    spans.push_back((hs[1] - hs[0]) / 1e3);
    for (int i = 0; i < b.s.nph; ++i) phs[i].push_back(hp[i] ? (double)(long long)(hp[i] - hs[0]) / 1e3 : -1.0);
  }
  for (auto& v : phs) { std::sort(v.begin(), v.end()); ph_med.push_back(v[v.size() / 2]); }
  if (a.mk && !mkout.empty()) {   // one more replay; dump its per-CTA chunk marks relative to the first CTA entry
    CK(cudaMemset(a.mk, 0, (size_t)G * a.mkmax * 3 * 8));
    CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
    std::vector<unsigned long long> hm((size_t)G * a.mkmax * 3);
    unsigned long long hs[2];
    CK(cudaMemcpy(hm.data(), a.mk, hm.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs, tspan, 16, cudaMemcpyDeviceToHost));
    FILE* f = fopen(mkout.c_str(), "w");
    if (f) {
      fprintf(f, "{\"span_us\":%.3f,\"sb\":%d,\"ctas\":[", (hs[1] - hs[0]) / 1e3, SB);
      for (int c = 0; c < G; ++c) {
        fprintf(f, "%s[", c ? "," : "");
        bool first = true;
        for (int k = 0; k < a.mkmax; ++k) {
          const unsigned long long* m = &hm[((size_t)c * a.mkmax + k) * 3];
          if (!m[0]) break;
          fprintf(f, "%s[%.3f,%.3f,%llu]", first ? "" : ",", ((long long)(m[0] - hs[0])) / 1e3,
                  m[1] ? ((long long)(m[1] - hs[0])) / 1e3 : -1.0, m[2]);
          first = false;
        }
        fprintf(f, "]");
      }
      fprintf(f, "]}\n");
      fclose(f);
    }
  }
  CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
}

int main(int argc, char** argv) {
  const std::string sched = args(argc, argv, "--sched", "t1s"), mode = args(argc, argv, "--mode", "dep");
  const int reps = argi(argc, argv, "--reps", 30);
  const int experts = argi(argc, argv, "--experts", sched[1] == '1' ? 2 : 7);
  const int full_kv = sched.back() == 'f';
  const int att = argi(argc, argv, "--att", sched[1] == '1' ? 3500 : 5500);
  const int dcons = argi(argc, argv, "--dcons", 21), vcons = argi(argc, argv, "--vcons", 21), acons = argi(argc, argv, "--acons", 21);
  const int cfg = argi(argc, argv, "--cfg", 0);
  const int contig = argi(argc, argv, "--contig", 0), mkmax = argi(argc, argv, "--marks", 0);
  const long long tstep = (long long)argi(argc, argv, "--touch", 0);   // bytes between pre-touch loads (0 = off)
  const std::string mkout = args(argc, argv, "--mkout", "");
  Build b = make_sched(sched, experts, att, full_kv, argi(argc, argv, "--sbb", 1200), argi(argc, argv, "--route", 1500),
                       argi(argc, argv, "--pnorm", 1200), argi(argc, argv, "--comb", 1200), argi(argc, argv, "--norm", 1200),
                       argi(argc, argv, "--attcomb", 1200));
  const int flatmb = argi(argc, argv, "--flatmb", 0);   // > 0: one static phase of this many MB, no dependencies
  if (flatmb > 0) {
    memset(&b.s, 0, sizeof(b.s));
    b.s.nph = 1;
    Phase& P = b.s.ph[0];
    P.bytes = (long long)(flatmb * MB) & ~255ll; P.route_slot = -1; P.wait_flag = P.wait_ctr = P.issue_flag = P.done_flag = -1;
    b.wbytes = P.bytes; b.total_mb = P.bytes / MB; b.route.assign(1, 0);
  }
  int dev = 0; CK(cudaSetDevice(dev));
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
  const int G = pr.multiProcessorCount;
  int clk_khz = 0; CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
  const float cyc_per_ns = clk_khz / 1e6f;
  CK(cudaMemcpyToSymbol(c_s, &b.s, sizeof(Sched)));
  const long long xstride = (long long)(28.31 * MB + 4095) & ~4095ll;
  unsigned char *w, *xw; uint4* fl; unsigned *sync, *err, *sink; int* rs; unsigned long long *tph, *tspan;
  const long long FL = 256ll << 20;
  const long long walign = (b.wbytes + (2 << 20) - 1) & ~((2ll << 20) - 1);
  if (contig) {   // R-rev-16: the routed experts' slabs right behind the static weights, in one allocation
    CK(cudaMalloc(&w, walign + (long long)b.s.nroute * xstride + (1 << 20)));
    CK(cudaMemset(w, 1, walign + (long long)b.s.nroute * xstride + (1 << 20)));
    xw = w + walign;
    for (int e = 0; e < (int)b.route.size(); ++e) b.route[e] = e;
  } else {
    CK(cudaMalloc(&w, b.wbytes + (1 << 20))); CK(cudaMalloc(&xw, 32 * xstride));
    CK(cudaMemset(w, 1, b.wbytes + (1 << 20))); CK(cudaMemset(xw, 2, 32 * xstride));
  }
  CK(cudaMalloc(&fl, FL)); CK(cudaMemset(fl, 3, FL));
  CK(cudaMalloc(&sync, NSYNC * 4)); CK(cudaMalloc(&err, 4)); CK(cudaMalloc(&sink, 4)); CK(cudaMalloc(&rs, 64 * 4));
  CK(cudaMalloc(&tph, MAXPH * 8)); CK(cudaMalloc(&tspan, 16)); CK(cudaMemset(err, 0, 4));
  CK(cudaMemcpy(rs, b.route.data(), b.route.size() * 4, cudaMemcpyHostToDevice));
  Args a; memset(&a, 0, sizeof(a));
  a.w = w; a.xw = xw; a.xstride = xstride; a.sync = sync; a.tph = tph; a.tspan = tspan; a.err = err; a.route_src = rs;
  a.dep = mode == "dep"; a.cyc_per_ns = cyc_per_ns; a.ilv = argi(argc, argv, "--ilv", 0);   // modes: stream (free, no deps), cons (costs, no deps), dep
  const int free_c = mode == "stream";
  a.cons_cyc_per_kb[0] = free_c ? 0 : (int)(dcons * cyc_per_ns);
  a.cons_cyc_per_kb[1] = free_c ? 0 : (int)(vcons * cyc_per_ns);
  a.cons_cyc_per_kb[2] = free_c ? 0 : (int)(acons * cyc_per_ns);
  if (mkmax > 0) CK(cudaMalloc(&a.mk, (size_t)G * mkmax * 3 * 8));
  a.mkmax = mkmax;
  std::vector<Region> touch;
  touch.push_back({w, b.wbytes});
  for (int e = 0; e < b.s.nroute; ++e) touch.push_back({xw + (long long)b.route[e] * xstride, (long long)(28.31 * MB)});
  cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  std::vector<double> spans, ph;
  const char* cfgs[] = {"NS6xSB32K/4w", "NS8xSB24K/4w", "NS4xSB48K/4w", "NS12xSB16K/4w", "NS6xSB32K/8w", "NS6xSB32K/3w",
                        "NS12xSB16K/3w"};
  switch (cfg) {
    case 0: run<6, 32768, 4>(b, a, G, reps, st, fl, FL / 16, sink, sync, tph, tspan, spans, ph, touch, tstep, mkout); break;
    case 1: run<8, 24576, 4>(b, a, G, reps, st, fl, FL / 16, sink, sync, tph, tspan, spans, ph, touch, tstep, mkout); break;
    case 2: run<4, 49152, 4>(b, a, G, reps, st, fl, FL / 16, sink, sync, tph, tspan, spans, ph, touch, tstep, mkout); break;
    case 3: run<12, 16384, 4>(b, a, G, reps, st, fl, FL / 16, sink, sync, tph, tspan, spans, ph, touch, tstep, mkout); break;
    case 4: run<6, 32768, 8>(b, a, G, reps, st, fl, FL / 16, sink, sync, tph, tspan, spans, ph, touch, tstep, mkout); break;
    case 5: run<6, 32768, 3>(b, a, G, reps, st, fl, FL / 16, sink, sync, tph, tspan, spans, ph, touch, tstep, mkout); break;
    default: run<12, 16384, 3>(b, a, G, reps, st, fl, FL / 16, sink, sync, tph, tspan, spans, ph, touch, tstep, mkout); break;
  }
  unsigned herr = 0; CK(cudaMemcpy(&herr, err, 4, cudaMemcpyDeviceToHost));
  std::sort(spans.begin(), spans.end());
  const double med = spans[spans.size() / 2], p10 = spans[spans.size() / 10], p90 = spans[spans.size() * 9 / 10];
  printf("{\"touch\":%lld,\"contig\":%d,\"ilv\":%d,\"sched\":\"%s\",\"mode\":\"%s\",\"cfg\":\"%s\",\"experts\":%d,\"att_ns\":%d,\"dcons\":%d,\"vcons\":%d,\"acons\":%d,"
         "\"MB\":%.1f,\"span_us\":%.2f,\"p10\":%.2f,\"p90\":%.2f,\"TBps\":%.3f,\"at4p4_us\":%.2f,\"err\":%u,\"G\":%d,\"mhz\":%.0f,\"phase_us\":[",
         tstep, contig, a.ilv, sched.c_str(), mode.c_str(), cfgs[cfg], experts, att, dcons, vcons, acons, b.total_mb, med, p10, p90,
         b.total_mb / med, b.total_mb / 4.4, herr, G, cyc_per_ns * 1e3);   // MB / us = TB/s
  for (size_t i = 0; i < ph.size(); ++i) printf("%s%.2f", i ? "," : "", ph[i]);
  printf("]}\n");
  return herr ? 2 : 0;
}

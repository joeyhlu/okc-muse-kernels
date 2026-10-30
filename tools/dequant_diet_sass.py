"""dequant_diet_sass.py ($0, notes/study/dequant_diet.md): per-stage sm_90a SASS of a standalone replica of the shipped VML
consumer stage (k3_rider3_1 / k3_gc_21cg_s1 `stage<N>` RS path: VM_TRICKS 1, EA 0, NB 2, W 1; `stage_m<N>` mma.sync path for
N <= 16), with compile-time variants of the ring / descriptor / dequant code. Counts the steady stage loop by pipe.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/dequant_diet_sass.py" OUTDIR [variants=ship,uq,dhc,uq+dhc] [ns=16,32,64]

Variants (all bitwise-identical math; only control / descriptor code changes):
  ship   it % NS, (it / NS) & 1 on a signed int (as shipped)
  uq     unsigned ring: q = (uint32_t)it / 3 (no LEA.HI sign fix-up), ws = it - 3 q
  dhc    wgmma descriptor high word from a kernel parameter (uniform to ptxas) instead of a per-stage register value
  ring   divisionless ring: ws / phase carried in registers (VCS-style; for reference only, lab.md:530)
  nsc    timing-floor arm: no scale multiply (NOT exact; shows what the 64 HMUL2 / 4 PRMT / 4 IMAD cost in SASS)
Pipes: ALU = LOP3 SHF PRMT IADD3 LEA ISETP SEL MOV S2R CS2R FLO POPC; HVY = IMAD* (FMA-heavy); HM2 = HMUL2 / HFMA2 (BF16);
VI = VIADD / VIMNMX; UNI = R2UR / U*; TC = HGMMA / HMMA / WARPGROUP; LDS = LDS / LDSM; SYNC = SYNCS / WARPSYNC / BAR; CTL = BRA ...
"""
import collections
import pathlib
import re
import subprocess
import sys

OBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
HDR = r"""
#include <cuda_bf16.h>
#include <stdint.h>
namespace dd {
constexpr int NS = 3, NX = 3, SW = 18432, US = 2 * SW, XS = 64 * 512;
struct Args { const int* err; uint32_t flags; uint32_t dhi; float* out; int nst; };
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, const int* err) {
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  int n = 0;
  while (!mbar_try(b, par)) {
    if ((++n & 255) == 0) {
      if (*reinterpret_cast<const volatile int*>(err) != 0) return;
      if (clock64() - t0 > (1ll << 31)) { atomicExch(const_cast<int*>(err), 1); return; }
    }
  }
}
__device__ __forceinline__ uint64_t wg_desc_sa(uint32_t a, uint32_t sbo) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
template <int N> __device__ __forceinline__ void wgmma_acc(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
template <int N> __device__ __forceinline__ void wgmma_zero(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
@WGSPECS@
__device__ __forceinline__ uint32_t unp_ok32(uint32_t c) { uint32_t r; asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(c)); return r; }
__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // UNP L2 layout (ship)
  o[0] = w & 0x81C081C0u;
  o[1] = __funnelshift_l(w, w, 10) & 0x81C081C0u;
  o[2] = __funnelshift_l(w, w, 13) & 0x81C081C0u;
  o[3] = ((w & 0x001B001Bu) * unp_ok32(0x1084u)) & 0x81C081C0u;
}
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void dec_half2(uint32_t w0, uint32_t w8, uint32_t sw2, uint32_t par, uint32_t* hh, uint32_t k16) {
  deq4(w0, hh); deq4(w8, hh + 4);
#if DD_NSC
  (void)sw2; (void)par; (void)k16;
#else
  const uint32_t sb = 2u * par;
  const uint32_t sgp = __byte_perm(sw2, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16 + 0x75807580u;
#pragma unroll
  for (int j = 0; j < 4; ++j) { hh[j] = bmul2l(hh[j], sgp); hh[4 + j] = bmul2h(hh[4 + j], sgp); }
#endif
}
template <int N, bool Z>
__device__ __forceinline__ void mma2(float* d, const uint32_t* hh, uint32_t dlo, uint32_t dhi) {
  if constexpr (Z) wgmma_zero<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
  else wgmma_acc<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
  wgmma_acc<N>(d, hh[2], hh[6], hh[3], hh[7], dlo + 16, dhi);
}
// ring position of stage `it`: (slot, parity)
struct Ring { int ws; uint32_t ph; };
__device__ __forceinline__ Ring ring_of(int it) {
#if DD_UQ
  const uint32_t q = (uint32_t)it / 3u;
  return Ring{(int)((uint32_t)it - 3u * q), q & 1u};
#else
  return Ring{it % NS, (uint32_t)((it / NS) & 1)};
#endif
}
template <int N>
__device__ __forceinline__ void stage(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* d,
                                      uint32_t (&hb)[2][8], int q, int wp, int lane, bool first
#if DD_RING
                                      , int& rws, uint32_t& rph
#endif
                                      ) {
#if DD_RING
  const int ws = rws, xs = rws; const uint32_t ph = rph;
  rws = (rws == NS - 1) ? 0 : rws + 1; rph ^= (rws == 0) ? 1u : 0u;
#else
  const Ring r = ring_of(it);
  const int ws = r.ws, xs = r.ws; const uint32_t ph = r.ph;   // NS == NX (ship: separate % / / that NVVM CSEs)
#endif
  mbar_wait(bars + 8 * ws, ph, A.err);
  const uint8_t* st = smem + ws * US + (q >> 1) * SW;
  const int tile = (q & 1) * 4 + wp;
  const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
  const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
  uint4 c[4];
  c[0] = cp[0];
  const uint32_t sw0 = sp[0], sw1 = sp[32];
  mbar_wait(bars + 8 * (2 * NS + xs), ph, A.err);
  __syncwarp();
  const uint32_t zo = A.flags >> 30;
  const uint32_t k16 = 16u + zo;
  const uint64_t d0 = wg_desc_sa(sa(smem + NS * US), 4096u);
  const uint32_t dlo = (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4);
#if DD_DHC
  const uint32_t dhi = A.dhi;   // == (uint32_t)(d0 >> 32), uniform kernel parameter
#else
  const uint32_t dhi = (uint32_t)(d0 >> 32);
#endif
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
    if (i < 3) c[i + 1] = cp[32 * (i + 1)];
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      uint32_t* hh = hb[(2 * i + hf) % 2];
      dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16);
      if (i == 3 && hf == 0) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }
      wg_fence();
      if (first && i == 0 && hf == 0) mma2<N, true>(d, hh, dlo + 64 * i + 32 * hf, dhi);
      else mma2<N, false>(d, hh, dlo + 64 * i + 32 * hf, dhi);
      wg_commit();
      wg_wait<1>();
      if (2 * i + hf == 0 && prevx >= 0) {
        if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
        prevx = -1;
      }
    }
  }
  prevx = xs;
  ++it;
}
template <int N>
__global__ void __launch_bounds__(640, 1) dd_rs(const __grid_constant__ Args A) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, q = warp >> 2, wp = warp & 3;
  if (q >= 4) return;
  const uint32_t bars = sa(smem + NS * US + NX * XS);
  float d[N / 2];
  uint32_t hb[2][8];
  int it = reinterpret_cast<const volatile int*>(smem)[threadIdx.x & 7], prevx = -1;   // non-uniform to ptxas, as in vm_main
#if DD_RING
  int rws = it % NS; uint32_t rph = (uint32_t)((it / NS) & 1);
#define RARGS , rws, rph
#else
#define RARGS
#endif
  stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true RARGS);
#pragma unroll 1
  for (int s = 1; s < A.nst; ++s) stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false RARGS);
  wg_wait<0>();
  if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
#pragma unroll
  for (int k = 0; k < N / 2; ++k) A.out[(size_t)k * 512 + threadIdx.x] = d[k];
}
template __global__ void dd_rs<32>(const __grid_constant__ Args A);
template __global__ void dd_rs<64>(const __grid_constant__ Args A);
}  // namespace dd
"""


def wg_specs(ns):
    s = ""
    for n in ns:
        na = n // 2
        outs = ",".join(f"%{i}" for i in range(na))
        a = ",".join(f"%{na + i}" for i in range(4))
        plus = ", ".join(f'"+f"(d[{i}])' for i in range(na))
        eq = ", ".join(f'"=f"(d[{i}])' for i in range(na))
        body = ("{\\n.reg .b64 dd;\\nmov.b64 dd, {%%%d, %%%d};\\nwgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 "
                "{%s}, {%s}, dd, %d, 1, 1, 0;\\n}\\n")
        for nm, sd, cons in (("wgmma_acc", 1, plus), ("wgmma_zero", 0, eq)):
            s += ("template <> __device__ __forceinline__ void %s<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
                  "uint32_t a3, uint32_t lo, uint32_t hi) {\n  asm volatile(\"" % (nm, n)
                  + body % (na + 4, na + 5, n, outs, a, sd)
                  + "\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(lo), \"r\"(hi));\n}\n" % cons)
    return s


VARS = {"ship": [], "uq": ["-DDD_UQ=1"], "dhc": ["-DDD_DHC=1"], "uq+dhc": ["-DDD_UQ=1", "-DDD_DHC=1"],
        "ring": ["-DDD_RING=1"], "nsc": ["-DDD_NSC=1"]}
ALU = {"LOP3", "LOP", "SHF", "PRMT", "IADD3", "MOV", "SEL", "ISETP", "LEA", "FSEL", "IABS", "PLOP3", "P2R", "R2P", "FLO",
       "POPC", "BMSK", "SGXT", "CS2R", "S2R", "IMNMX"}


def pipe(op):
    b = op.split(".")[0]
    if b in ("HMUL2", "HFMA2", "HADD2"):
        return "HM2"
    if b.startswith("IMAD") or b in ("IMUL", "FFMA", "FMUL", "FADD"):
        return "HVY"
    if b in ALU:
        return "ALU"
    if b.startswith("VI"):
        return "VI"
    if b == "R2UR" or (b.startswith("U") and b not in ("UNKNOWN",)):
        return "UNI"
    if b in ("HGMMA", "HMMA", "WARPGROUP"):
        return "TC"
    if b in ("LDS", "LDSM"):
        return "LDS"
    if b in ("SYNCS", "WARPSYNC", "BAR", "NOP", "DEPBAR"):
        return "SYNC"
    return "CTL"


COLS = ("ALU", "HVY", "HM2", "VI", "UNI", "TC", "LDS", "SYNC", "CTL")


def stage_loop(sass_fn):
    """steady loop = the backward-branch range that holds 16 HGMMA (one stage); returns its opcodes."""
    ins = []
    for ln in sass_fn:
        m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?);", ln)
        if m:
            ins.append((int(m.group(1), 16), m.group(2).strip()))
    addr = {a: i for i, (a, _) in enumerate(ins)}
    best = None
    for i, (a, t) in enumerate(ins):
        m = re.search(r"BRA\s+.*?0x([0-9a-f]+)", t)
        if m and op_of(t) == "BRA":
            tgt = int(m.group(1), 16)
            if tgt < a and tgt in addr:
                body = ins[addr[tgt]:i + 1]
                nh = sum(1 for _, x in body if "HGMMA" in x)
                if nh == 16 and (best is None or len(body) > len(best)):
                    best = body
    return best


def op_of(t):
    t = re.sub(r"^@!?U?P\w+\s+", "", t)
    return t.split()[0]


def slow_path(body):
    """bounded-wait slow paths: after `SYNCS.PHASECHK...TRYWAIT P, ...` the forward `@P BRA tgt` skips them; mark (BRA, tgt)."""
    out = set()
    addr = {a: k for k, (a, _) in enumerate(body)}
    for k in range(len(body) - 1):
        if "PHASECHK" not in body[k][1]:
            continue
        a, t = body[k + 1]
        m = re.match(r"@P\d+\s+BRA\s+(?:`\(\S+\)\s*)?0x([0-9a-f]+)", t)
        if m:
            tgt = int(m.group(1), 16)
            if tgt > a and tgt in addr:
                out.update(range(k + 2, addr[tgt]))
    return out


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    vars_ = (sys.argv[2] if len(sys.argv) > 2 else "ship,uq,dhc,uq+dhc,ring,nsc").split(",")
    ns = [32, 64]
    src = HDR.replace("@WGSPECS@", wg_specs(ns))
    cu = out / "dd.cu"
    cu.write_text(src)
    rows = []
    for v in vars_:
        cub = out / f"dd_{v.replace('+', '_')}.cubin"
        r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", "-O3", "--expt-relaxed-constexpr", "-Xptxas=-v", *VARS[v],
                            "-o", str(cub), str(cu)], capture_output=True, text=True)
        log = r.stdout + r.stderr
        (out / f"ptxas_{v}.log").write_text(log)
        if r.returncode:
            print(v, "COMPILE FAIL\n", log[-3000:])
            continue
        regs = dict(re.findall(r"Compiling entry function '_ZN2dd5dd_rsILi(\d+)E\S*'.*?\n.*?Used (\d+) registers", log, re.S))
        sass = subprocess.run([OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
        (out / f"dd_{v.replace('+', '_')}.sass").write_text(sass)
        funcs, cur = collections.defaultdict(list), None
        for ln in sass.splitlines():
            m = re.search(r"Function : (\S+)", ln)
            if m:
                cur = m.group(1)
                continue
            if cur:
                funcs[cur].append(ln)
        for fn, lines in funcs.items():
            n = int(re.search(r"dd_rsILi(\d+)E", fn).group(1))
            body = stage_loop(lines)
            if body is None:
                print(v, n, "no stage loop found")
                continue
            sl = slow_path(body)
            cnt_all, cnt_rdy = collections.Counter(), collections.Counter()
            ops = collections.Counter()
            for k, (_, t) in enumerate(body):
                p = pipe(op_of(t))
                cnt_all[p] += 1
                if k not in sl:
                    cnt_rdy[p] += 1
                    ops[op_of(t).split(".")[0]] += 1
            rows.append((v, n, regs.get(str(n), "?"), cnt_rdy, sum(cnt_rdy.values()), cnt_all, ops))
    print(f"{'variant':8} {'N':>3} {'regs':>4} | " + " ".join(f"{c:>4}" for c in COLS) + " | ready  (all)")
    for v, n, rg, c, tot, ca, ops in rows:
        print(f"{v:8} {n:>3} {rg:>4} | " + " ".join(f"{c[k]:>4}" for k in COLS) + f" | {tot:>5}  ({sum(ca.values())})")
    for v, n, rg, c, tot, ca, ops in rows:
        print(f"  {v} N{n}: " + " ".join(f"{k}{ops[k]}" for k in sorted(ops, key=lambda x: -ops[x])))


if __name__ == "__main__":
    main()

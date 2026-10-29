"""vcs_bench.py: R-vcs (lean VML consumer stage loop) on k3_gc_18's VML code; notes/study/wgmma_bench.md s3 lead.

  python3 -u kernels/lab/vcs_bench.py sass OUTDIR [--builds=bf2,w72]      (local WSL: ptxas C75xx + per-stage SASS)
  python3 -u kernels/lab/vcs_bench.py map SASSFILE N[,N]                  (one char per stage-loop instruction)
  python3 -u kernels/lab/vcs_bench.py run [--ntoks=8,16,32,64] [--rounds=9] [--units=3] [--out=box_out/vcs/run.json]

sass: device-only cubins (no torch build) of dr_probe's drp_kernel<N> stage-loop harness on gc_18's VML head with the bf2 /
w72 defines (N 8 / 16 / 32 / 64), ptxas -v + C7510..C7520 warnings; the steady-state stage loop (the `#pragma unroll 1`
s-loop: smallest backward-branch range holding one stage's HGMMA / HMMA) is cut out of the SASS and its opcodes classed.
--ship also compiles the full vm_main (bf2 / w72) for the C75xx check only.
"""
from __future__ import annotations

import ast
import collections
import pathlib
import re
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "tools", ROOT / "tools/judge", ROOT / "kernels/muse_moe_w4a16", ROOT / "kernels/lab"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BASE = ROOT / "kernels/muse_decode_layer/k3_gc_18.py"
OBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
FLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills"]
INCS = ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>")
SASS_N = {"bf2": (8, 16, 32, 64), "w72": (32, 64)}


def gc18_vml():
    """k3_gc_18's VML CUDA (up to its host entry point) and its _VML_DEFS (bf2, w72, ...)."""
    t = BASE.read_text(encoding="utf-8")
    ns = {}
    exec(compile("import hashlib, time\n" + t[t.index("_vml_E_LOC, _vml_TOPK, _vml_KDIM = 32, 8, 4096\n"):t.index("\nVML_ENABLE = True\n")],
                 "k3_gc_18:_vml_", "exec"), ns)   # noqa: S102 (our own source)
    cuda = ns["_vml_CUDA"]
    i = t.index("\n_VML_DEFS = {")
    defs = ast.literal_eval(t[i + len("\n_VML_DEFS = "):t.index("}\n", i) + 1])
    return cuda[:cuda.index("void vml_run(")], defs


def strip_inc(s):
    for inc in INCS:
        s = s.replace(inc, "")
    return s


def probe_dev(ns_list):
    import dr_probe as DRP
    s = DRP.PROBE_CU[:DRP.PROBE_CU.index(DRP.HOST_MARK)]
    lines = [ln for ln in s.split("\n") if not ln.startswith("template __global__ void drp_kernel<")]
    s = "\n".join(lines)
    inst = "".join(f"template __global__ void drp_kernel<{n}>(const Args A, const PArgs P);\n" for n in ns_list)
    return s.replace("}  // namespace vm", inst + "}  // namespace vm")


# ---------------------------------------------------------------- SASS analysis
CLASSES = [
    ("mma", ("HGMMA", "HMMA")), ("ldsm", ("LDSM",)), ("lds", ("LDS",)), ("wgsync", ("WARPGROUP",)),
    ("mbar", ("SYNCS",)), ("bar", ("BAR", "WARPSYNC", "NOP", "DEPBAR")), ("unif", ("R2UR", "UMOV", "UIADD3", "ULOP3", "USHF", "UIMAD", "ULEA", "USEL", "UISETP", "UPRMT", "UFLO", "VOTEU", "S2UR", "R2UR")),
    ("bf16", ("HMUL2", "HFMA2", "HADD2")), ("alu", ("LOP3", "SHF", "PRMT", "IMAD", "IADD3", "LEA", "SEL", "ISETP", "MOV", "IABS", "VIADD", "FLO", "POPC", "VIMNMX", "IMNMX")),
    ("ctl", ("BRA", "BSSY", "BSYNC", "EXIT", "CALL", "RET", "YIELD", "PLOP3", "P2R", "R2P")), ("fp", ("FADD", "FMUL", "FFMA", "FSEL", "FSETP")),
    ("gmem", ("LDG", "STG", "RED", "ATOM", "LD", "ST", "UBLKCP", "UTMALDG")), ("sts", ("STS",)),
]


def opclass(op):
    base = op.split(".")[0]
    for nm, ops in CLASSES:
        if base in ops:
            return nm
    return "other:" + base


def parse_funcs(sass):
    funcs, cur = {}, None
    for ln in sass.splitlines():
        m = re.search(r"Function : (\S+)", ln)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is None:
            continue
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?);", ln)
        if m:
            txt = m.group(2).strip()
            txt = re.sub(r"^@!?U?P[T0-9]+\s+", "", txt)
            funcs[cur].append((int(m.group(1), 16), txt))
    return funcs


def loops(ins):
    """[(lo, hi)] index ranges of backward branches (BRA to an address <= its own)."""
    idx = {a: i for i, (a, _) in enumerate(ins)}
    out = []
    for i, (a, t) in enumerate(ins):
        if t.startswith("BRA"):
            m = re.search(r"0x([0-9a-f]+)", t)
            if m:
                tgt = int(m.group(1), 16)
                if tgt <= a and tgt in idx:
                    out.append((idx[tgt], i))
    return out


def stage_loop(ins, mma_op, n_mma):
    best = None
    for lo, hi in loops(ins):
        c = sum(1 for _, t in ins[lo:hi + 1] if t.startswith(mma_op))
        if c == n_mma and (best is None or hi - lo < best[1] - best[0]):
            best = (lo, hi)
    return best


def analyze(ins, lo, hi):
    body = ins[lo:hi + 1]
    inner = set()
    for a, b in loops(ins):
        if lo <= a and b <= hi and (a, b) != (lo, hi) and b - a < 40:   # spin loops (mbarrier try_wait)
            inner.update(range(a, b + 1))
    cls = collections.Counter()
    ops = collections.Counter()
    for k, (_, t) in enumerate(body, start=lo):
        op = t.split()[0]
        c = opclass(op)
        cls["spin" if k in inner else c] += 1
        ops[op.split(".")[0]] += 1
    return len(body), cls, ops


def kname_nvm(fn):
    m = re.search(r"vcs_kernelILi(\d+)ELi(\d+)ELi(\d+)E", fn)
    return tuple(int(v) for v in m.groups()) if m else None


def loop_spec(n, v, m):
    if m == 2:
        return "HMUL2", 64
    return ("HMMA", 2 * n) if n <= 16 else ("HGMMA", 16)


def sass_main(argv):
    out = pathlib.Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    builds = next((a.split("=", 1)[1] for a in argv if a.startswith("--builds=")), "bf2,w72").split(",")
    sem = threading.Semaphore(2)
    res = {}

    def job(tag, src, fl):
        with sem:
            cu = out / f"{tag}.cu"
            cu.write_text(src)
            cub = out / f"{tag}.cubin"
            r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *fl, "-Xptxas=-v", "-o", str(cub), str(cu)], capture_output=True, text=True)
            log = r.stdout + r.stderr
            (out / f"{tag}.ptxas.log").write_text(log)
            if r.returncode:
                res[tag] = ("FAIL", log[-2500:], None)
                return
            s = subprocess.run([OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
            (out / f"{tag}.sass").write_text(s)
            res[tag] = ("ok", log, s)

    jobs = []
    for b in builds:
        dev, _, fl = module_src(b)
        inst = "".join(f"template __global__ void vcs_kernel<{n}, {v}, {m}>(const Args A, const PArgs P, int pf);\n"
                       for n, v, m in kernels_of(b))
        dev = strip_inc(dev).rstrip()
        assert dev.endswith("}  // namespace vm")
        dev = dev[:-len("}  // namespace vm")] + inst + "}  // namespace vm\n"
        jobs.append((f"vcs_{b}", dev, FLAGS + fl))
    ths = [threading.Thread(target=job, args=j) for j in jobs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for tag, _, _ in jobs:
        st, log, s = res[tag]
        if st != "ok":
            print(f"[vcs-sass] {tag} FAIL\n{log}")
            continue
        warn = [ln.strip()[:200] for ln in log.splitlines() if re.search(r"C75\d\d|serializ", ln)]
        print(f"[vcs-sass] {tag}: C75xx/serialized lines {len(warn)}" + "".join("\n   " + w for w in warn[:6]))
        regs, fn, sp = {}, None, "?"
        for ln in log.splitlines():
            mm = re.search(r"Compiling entry function '(\S+)'", ln)
            if mm:
                fn = mm.group(1)
            mm = re.search(r"(\d+) bytes spill stores", ln)
            if mm:
                sp = mm.group(1)
            mm = re.search(r"Used (\d+) registers", ln)
            if mm and fn:
                regs[fn] = (mm.group(1), sp)
        funcs = parse_funcs(s)
        rows = []
        for fn, ins in funcs.items():
            key = kname_nvm(fn)
            if key is None:
                continue
            n, v, m = key
            op, cnt = loop_spec(n, v, m)
            lp = stage_loop(ins, op, cnt)
            rg, spl = regs.get(fn, ("?", "?"))
            if lp is None:
                rows.append((key, f"regs {rg} spill {spl} | no stage loop with {cnt} {op}"))
                continue
            ln_, cls, ops = analyze(ins, *lp)
            top = " ".join(f"{k} {c}" for k, c in sorted(cls.items(), key=lambda kv: -kv[1])[:9])
            rows.append((key, f"regs {rg} spill {spl} | loop {ln_}: {top}"))
        for key, txt in sorted(rows):
            print(f"   N{key[0]} V{key[1]} M{key[2]} {txt}")


# ---------------------------------------------------------------- the lean stage loops (R-vcs-1)
# V 0 = the incumbent (dr_probe's drp_seg_inc: gc_18's stage_m<N> / stage<N> verbatim), V 1 = lean, V 2 = lean + one commit
# group per k64 step (RS only). M 0 = ring-synced; M 1..3 = no barriers at all after setup (producers idle, weight slot 0 and
# x slot 0 reused; outputs garbage): 1 = full compute, 2 = dequant only (MMA + ldmatrix removed, A words kept live),
# 3 = MMA only (raw code words as A, no dequant). pflags bit 0 (producer): no weight stream after the first NS stages.
VCS_DEV = r"""
namespace vm {
__device__ void vcs_prod_w(const Args& A, const PArgs& P, uint32_t ring, uint32_t bars, int u0, int u1, int pf) {
const uint64_t pol = policy_evict_first();
int it = 0;
for (int u = u0; u < u1; ++u) {
const int4 un = P.unit[u];
for (int h = 0; h < 2; ++h)
for (int s = 0; s < 16; ++s, ++it) {
const int sl = it % NS;
if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
if ((pf & 1) && it >= NS) { mbar_arrive(fb); continue; }   // no weight stream: stale codes, same handshakes
mbar_expect_tx(fb, US);
const int blk = un.x * 64 + 4 * un.y + 2 * h;
bulk_g2s_ef(dst, P.ws + ((size_t)blk * 16 + s) * SW, SW, fb, pol);
bulk_g2s_ef(dst + SW, P.ws + ((size_t)(blk + 1) * 16 + s) * SW, SW, fb, pol);
}
}
}
__device__ void vcs_prod_x(const Args& A, const PArgs& P, uint32_t xring, uint32_t bars, int u0, int u1, int pf) {
const uint32_t bytes = (uint32_t)P.npad * 512u;
int it = 0;
for (int u = u0; u < u1; ++u) {
const uint8_t* src = P.xp + (size_t)P.unit[u].z;
for (int h = 0; h < 2; ++h)
for (int s = 0; s < 16; ++s, ++it) {
const int sl = it % NX;
if (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * (2 * NS + sl);
if ((pf & 2) && it >= NX) { mbar_arrive(fb); continue; }   // no x stream: stale x, same handshakes
mbar_expect_tx(fb, bytes);
bulk_g2s(xring + sl * XS, src + (size_t)s * bytes, bytes, fb);
}
}
}
__device__ __forceinline__ bool mbar_test(uint32_t b, uint32_t par) {
uint32_t ok;
asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
: "=r"(ok) : "r"(b), "r"(par) : "memory");
return ok != 0;
}
template <int F>
__device__ __forceinline__ void vwait(uint32_t b, uint32_t par, int* err) {   // F & 16: bounded test_wait poll (no suspend)
if constexpr ((F & 16) != 0) {
if (mbar_test(b, par)) return;
int n = 0;
while (!mbar_test(b, par)) {
if (++n > (1 << 26)) { atomicExch(err, 1); return; }
}
} else {
mbar_wait(b, par, err);
}
}
template <int F>
__device__ __forceinline__ bool vtest(uint32_t b, uint32_t par) { return ((F & 16) != 0) ? mbar_test(b, par) : mbar_try(b, par); }
// stage loop state: weight slot / parity and x slot / parity of the current stage, advanced without divisions
static_assert(NS == NX, "vcs: one ring index for weight and x slots");
struct VRing { int ws, wph; };   // x slot / parity == weight slot / parity (NS == NX)
__device__ __forceinline__ VRing vring_at(int it) { return VRing{it % NS, (it / NS) & 1}; }
__device__ __forceinline__ VRing vring_next(VRing r) {
r.ws += 1; if (r.ws == NS) { r.ws = 0; r.wph ^= 1; }
return r;
}
// one stage's codes / scales of this warp tile from weight slot ws (SY 0: slot 0)
#define VCS_CP(ws_) reinterpret_cast<const uint4*>(smem + (SY ? (ws_) * US : 0) + cofs)
#define VCS_SP(ws_) reinterpret_cast<const uint32_t*>(smem + (SY ? (ws_) * US : 0) + sofs)
// F flags: 1 = early barrier tests (stage s + 1's w test at the stage start, x test at k64 step 2: the try_wait latency under
// compute), 2 = FMA-pipe rotations in the unpack (rotl via one IMAD.WIDE.U32 instead of SHF; bitwise == deq4),
// 4 = debug: x slot released at the end of its own stage after wait_group 0 (RS), 8 = debug: no rolling refill (codes of
// stage s loaded at its start, as the incumbent).
__device__ __forceinline__ uint32_t vrot(uint32_t w, uint32_t m2k) {
const uint64_t t = (uint64_t)w * m2k;
return (uint32_t)t | (uint32_t)(t >> 32);
}
template <int F>
__device__ __forceinline__ void vdec(uint32_t w0, uint32_t w8, uint32_t sw2, uint32_t par, uint32_t* hh, uint32_t k16, uint32_t k64,
uint32_t k4, uint32_t m10, uint32_t m13) {
if constexpr ((F & 2) != 0) {
hh[0] = w0 & 0x81C081C0u; hh[1] = vrot(w0, m10) & 0x81C081C0u; hh[2] = vrot(w0, m13) & 0x81C081C0u;
hh[3] = ((w0 & 0x001B001Bu) * unp_ok32(0x1084u)) & 0x81C081C0u;
hh[4] = w8 & 0x81C081C0u; hh[5] = vrot(w8, m10) & 0x81C081C0u; hh[6] = vrot(w8, m13) & 0x81C081C0u;
hh[7] = ((w8 & 0x001B001Bu) * unp_ok32(0x1084u)) & 0x81C081C0u;
const uint32_t sb = 2u * par;
const uint32_t sgp = __byte_perm(sw2, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16 + 0x75807580u;
#pragma unroll
for (int j = 0; j < 4; ++j) { hh[j] = bmul2l(hh[j], sgp); hh[4 + j] = bmul2h(hh[4 + j], sgp); }
} else {
dec_half2(w0, w8, sw2, par, hh, k16, k64, k4);
}
}
// lean mma.sync loop (N <= 16): do-while over the 16 stages; stage s + 1's weight wait + code loads go into stage s's dead
// k64 registers (c[0], c[1], sw0 after step 1, c[2] after step 2, c[3], sw1 after step 3); same k16 / chain order as stage_m.
template <int N, int M, int F>
__device__ __forceinline__ void vcs_seg_m(const Args& A, uint8_t* smem, uint32_t bars, int& it, float* d, int q, int wp, int lane) {
constexpr bool SY = (M == 0), E = (F & 1) != 0, NR = (F & 8) != 0;
float d2[2][N / 2];
#pragma unroll
for (int k = 0; k < N / 2; ++k) { d2[0][k] = 0.f; d2[1][k] = 0.f; }
const int tile = (q & 1) * 4 + wp;
const uint32_t cofs = (uint32_t)((q >> 1) * SW + tile * 2048 + lane * 16), sofs = (uint32_t)((q >> 1) * SW + 16384 + tile * 256 + lane * 4);
const uint32_t zom = (uint32_t)A.flags >> 30;   // 0, opaque
const uint32_t k16q = 16u + zom;
const uint32_t m10 = unp_ok32(1u << 10), m13 = unp_ok32(1u << 13);
const uint32_t xb0 = sa(smem + NS * US) + (uint32_t)(((lane >> 3) * 8 + (lane & 7)) * 16);
VRing r = vring_at(it);
uint4 c[4];
uint32_t sw0, sw1, vsink = 0;
if (SY) vwait<F>(bars + 8 * r.ws, r.wph, A.err);
{
const uint4* cp = VCS_CP(r.ws);
const uint32_t* sp = VCS_SP(r.ws);
#pragma unroll
for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
sw0 = sp[0]; sw1 = sp[32];
}
if (SY) {
vwait<F>(bars + 8 * (2 * NS + r.ws), r.wph, A.err);
__syncwarp();
}
int s = 0;
#pragma unroll 1
do {
const bool more = s < 15;
const VRing r1 = vring_next(r);
const uint4* cp1 = VCS_CP(r1.ws);
const uint32_t* sp1 = VCS_SP(r1.ws);
const uint32_t xb = xb0 + (uint32_t)(SY ? r.ws * XS : 0);
if constexpr (NR) {
if (s > 0) {
if (SY) vwait<F>(bars + 8 * r.ws, r.wph, A.err);
const uint4* cp = VCS_CP(r.ws);
const uint32_t* sp = VCS_SP(r.ws);
#pragma unroll
for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
sw0 = sp[0]; sw1 = sp[32];
}
}
bool okw = false, okx = false;
#pragma unroll
for (int i = 0; i < 4; ++i) {
if constexpr (E && SY) {
if (more && i == 0) okw = vtest<F>(bars + 8 * r1.ws, r1.wph);
if (more && i == 2) okx = vtest<F>(bars + 8 * (2 * NS + r1.ws), r1.wph);
}
const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
uint32_t hh[8];
const uint32_t wa = hf ? c[i].y : c[i].x, wb = hf ? c[i].w : c[i].z;
if constexpr (M == 3) {   // raw words as A, quads = the LDS.128 registers of c[i] (no moves)
hh[0] = c[i].x; hh[4] = c[i].y; hh[1] = c[i].z; hh[5] = c[i].w; hh[2] = c[i].x; hh[6] = c[i].y; hh[3] = c[i].z; hh[7] = c[i].w;
(void)wa; (void)wb;
} else {
vdec<F>(wa, wb, sw2, par, hh, k16q, 64u, 4u, m10, m13);
}
if constexpr (M == 2) {
vsink ^= hh[0] ^ hh[1] ^ hh[2] ^ hh[3] ^ hh[4] ^ hh[5] ^ hh[6] ^ hh[7];   // keeps the unpack live (4 LOP3 per half)
} else {
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
uint32_t b0, b1, b2, b3;
ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);
mma16816(&d2[0][4 * nt], hh[0], hh[4], hh[1], hh[5], b0, b1);
mma16816(&d2[1][4 * nt], hh[2], hh[6], hh[3], hh[7], b2, b3);
}
}
}
if (SY && i == 0) {   // this stage's slot: every code word is in registers now (data dependency, see R-vcs-3)
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + r.ws) + ((c[1].x ^ c[2].x ^ c[3].x ^ sw1) & zom));
}
if (!NR && more) {   // rolling refill of the dead k64 registers with stage s + 1's codes
if (i == 1) { if (SY && !okw) vwait<F>(bars + 8 * r1.ws, r1.wph, A.err); c[0] = cp1[0]; c[1] = cp1[32]; sw0 = sp1[0]; }
if (i == 2) c[2] = cp1[64];
if (i == 3) { c[3] = cp1[96]; sw1 = sp1[32]; }
}
}
if (SY) {
__syncwarp();
if (lane == 0) {
mbar_arrive(bars + 8 * (2 * NS + NX + r.ws));
}
if (more && !okx) vwait<F>(bars + 8 * (2 * NS + r1.ws), r1.wph, A.err);
__syncwarp();
}
r = r1;
} while (++s < 16);
it += 16;
#pragma unroll
for (int k = 0; k < N / 2; ++k) d[k] = d2[0][k] + d2[1][k];
if constexpr (M == 2) d[0] = __uint_as_float(vsink);
}
// lean RS-wgmma loop (N >= 24): do-while, accumulators zeroed up front (no peeled first stage), the same rolling refill, the
// shipped group shape (G 0: 2 HGMMA per commit, wait_group 1, 2 A buffers) or G 1: one commit group per k64 step (4 HGMMA,
// wait_group 1, 4 A buffers). The previous stage's x slot is released after the first group of the next stage completes.
template <int N, int M, int G, int F>
__device__ __forceinline__ void vcs_seg_rs(const Args& A, uint8_t* smem, uint32_t bars, int& it, float* d, int q, int wp, int lane) {
constexpr bool SY = (M == 0), E = (F & 1) != 0, XD = (F & 4) != 0, NR = (F & 8) != 0;
constexpr int NB = G ? 4 : 2;
#pragma unroll
for (int k = 0; k < N / 2; ++k) d[k] = 0.f;
uint32_t hb[NB][8];
const int tile = (q & 1) * 4 + wp;
const uint32_t cofs = (uint32_t)((q >> 1) * SW + tile * 2048 + lane * 16), sofs = (uint32_t)((q >> 1) * SW + 16384 + tile * 256 + lane * 4);
const uint32_t zo = (uint32_t)A.flags >> 30;
const uint32_t k16 = 16u + zo, k64 = VM_TRICKS_DQ ? 64u + zo : 64u, k4 = VM_TRICKS_DQ ? 4u + zo : 4u;
const uint32_t m10 = unp_ok32(1u << 10), m13 = unp_ok32(1u << 13);
const uint64_t dd = wg_desc_sa(sa(smem + NS * US), 4096u);
const uint32_t dhi = (uint32_t)(dd >> 32);
VRing r = vring_at(it);
uint4 c[4];
uint32_t sw0, sw1, vsink = 0;
if (SY) vwait<F>(bars + 8 * r.ws, r.wph, A.err);
{
const uint4* cp = VCS_CP(r.ws);
const uint32_t* sp = VCS_SP(r.ws);
#pragma unroll
for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
sw0 = sp[0]; sw1 = sp[32];
}
if (SY) {
vwait<F>(bars + 8 * (2 * NS + r.ws), r.wph, A.err);
__syncwarp();
}
int prevx = -1;
int s = 0;
#pragma unroll 1
do {
const bool more = s < 15;
const VRing r1 = vring_next(r);
const uint4* cp1 = VCS_CP(r1.ws);
const uint32_t* sp1 = VCS_SP(r1.ws);
const uint32_t dlo = (uint32_t)dd + (uint32_t)(SY ? r.ws : 0) * (uint32_t)(XS >> 4);
if constexpr (NR) {
if (s > 0) {
if (SY) vwait<F>(bars + 8 * r.ws, r.wph, A.err);
const uint4* cp = VCS_CP(r.ws);
const uint32_t* sp = VCS_SP(r.ws);
#pragma unroll
for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
sw0 = sp[0]; sw1 = sp[32];
}
}
bool okw = false, okx = false;
#pragma unroll
for (int i = 0; i < 4; ++i) {
if constexpr (E && SY) {
if (more && i == 0) okw = vtest<F>(bars + 8 * r1.ws, r1.wph);
if (more && i == 2) okx = vtest<F>(bars + 8 * (2 * NS + r1.ws), r1.wph);
}
const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
uint32_t* hh = hb[G ? (2 * (i & 1) + hf) : hf];
const uint32_t wa = hf ? c[i].y : c[i].x, wb = hf ? c[i].w : c[i].z;
if constexpr (M == 3) {
hh[0] = c[i].x; hh[4] = c[i].y; hh[1] = c[i].z; hh[5] = c[i].w; hh[2] = c[i].x; hh[6] = c[i].y; hh[3] = c[i].z; hh[7] = c[i].w;
(void)wa; (void)wb;
} else {
vdec<F>(wa, wb, sw2, par, hh, k16, k64, k4, m10, m13);
}
if constexpr (M == 2) {
vsink ^= hh[0] ^ hh[1] ^ hh[2] ^ hh[3] ^ hh[4] ^ hh[5] ^ hh[6] ^ hh[7];
} else if (G == 0 || hf == 1) {
wg_fence();
if constexpr (G) mma2<N, false>(d, hb[2 * (i & 1)], dlo + 64 * i, dhi);
mma2<N, false>(d, hh, dlo + 64 * i + 32 * hf, dhi);
wg_commit();
wg_wait<1>();
if (!XD && SY && i == 0 && (G || hf == 0)) {   // the previous stage's last group is complete: its x slot is free
if (lane == 0 && prevx >= 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
}
}
if (SY && i == 0 && hf == 1) {   // this stage's slot: every code word is in registers now (data dependency, see R-vcs-3)
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + r.ws) + ((c[1].x ^ c[2].x ^ c[3].x ^ sw1) & zo));
}
if (!NR && more && hf == 1) {   // rolling refill: k64 step i's code words are dead after its second unpack (behind that group's commit)
if (i == 1) { if (SY && !okw) vwait<F>(bars + 8 * r1.ws, r1.wph, A.err); c[0] = cp1[0]; c[1] = cp1[32]; sw0 = sp1[0]; }
if (i == 2) c[2] = cp1[64];
if (i == 3) { c[3] = cp1[96]; sw1 = sp1[32]; }
}
}
}
if (SY) {
if constexpr (XD) { wg_wait<0>(); }
__syncwarp();
if (XD && lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + r.ws));
prevx = XD ? -1 : r.ws;
if (more && !okx) vwait<F>(bars + 8 * (2 * NS + r1.ws), r1.wph, A.err);
__syncwarp();
}
r = r1;
} while (++s < 16);
if constexpr (M != 2) wg_wait<0>();
if constexpr (M == 2) d[0] = __uint_as_float(vsink);
if (SY && lane == 0 && prevx >= 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
it += 16;
}
// V -> (G, F): 1 lean, 2 lean G1, 3 early tests (+ G1 on RS), 4 early + FMA rotations, 5 / 6 RS race debug (x sync / no roll)
template <int V> struct VF {
static constexpr int G = (V == 2 || V == 3 || V == 4 || V == 8) ? 1 : 0;
static constexpr int F = V == 3 ? 17 : V == 4 ? 3 : V == 5 ? 4 : V == 6 ? 8 : V == 8 ? 16 : 0;
};
template <int N, int V, int M>
__global__ void __launch_bounds__(NTH, 1) vcs_kernel(const Args A, const PArgs P, int pf) {
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
if (M == 0 && lane == 0) {
if (warp == NCW) vcs_prod_w(A, P, sa(smem), bars, rg.x, rg.y, pf);
else if (warp == NCW + 1) vcs_prod_x(A, P, sa(smem + NS * US), bars, rg.x, rg.y, pf);
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
#pragma unroll 1
for (int h = 0; h < 2; ++h) {
float d[N / 2];
if constexpr (V == 0) drp_seg_inc<N>(A, smem, bars, it, d, q, wp, lane);
else if constexpr (V == 7) {   // bisection: gc_18's stage<N> unchanged, the s-loop as a do-while
Pend pd{-1};
uint32_t hb[2][8];
int prevx = -1;
stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, pd);
int s = 1;
#pragma unroll 1
do { stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, pd); } while (++s < 16);
wg_wait<0>();
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
}
else if constexpr (N <= 16) vcs_seg_m<N, M, VF<V>::F>(A, smem, bars, it, d, q, wp, lane);
else vcs_seg_rs<N, M, VF<V>::G, VF<V>::F>(A, smem, bars, it, d, q, wp, lane);
drp_store<N>(P, u, (2 * h + (q >> 1)) * 8 + (q & 1) * 4 + wp, lane, d, gs);
}
}
if (mk && wp == 0 && lane == 0) { mk[1 + q] = gtime(); if (q == 0) mk[6] = clock64(); }
}
}  // namespace vm
"""
VCS_CPP = r"""
#include <torch/extension.h>
void vcs_run(torch::Tensor ws, torch::Tensor gsc, torch::Tensor xp, torch::Tensor unit, torch::Tensor cta, torch::Tensor out,
             torch::Tensor err, torch::Tensor marks, int64_t npad, int64_t G, int64_t v, int64_t m, int64_t pf);
std::vector<int64_t> vcs_attrs(int64_t npad, int64_t v, int64_t m);
"""
BUILDS = {"bf2": (8, 16, 32), "w72": (32, 64)}


def kernels_of(build):
    """(N, V, M) instantiations of one build: inc (V 0, M 0); lean V 1, M 0..3; RS-only V 2 (G 1), M 0 / 1."""
    out = []
    for n in BUILDS[build]:
        out.append((n, 0, 0))
        out += [(n, 1, m) for m in range(4)]
        if n > 16:
            out += [(n, 2, 0), (n, 2, 1)]
        out += [(n, 3, 0), (n, 3, 3), (n, 8, 0), (n, 8, 3)]
    return out


def vcs_host(build):
    ks = kernels_of(build)
    s = "\ntemplate <int N, int V, int M>\nstatic void vcs_launch(const vm::Args& a, const vm::PArgs& p, int G, int pf) {\n"
    s += "  cudaFuncSetAttribute(vm::vcs_kernel<N, V, M>, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);\n"
    s += "  vm::vcs_kernel<N, V, M><<<G, vm::NTH, vm::SMEM, at::cuda::getCurrentCUDAStream()>>>(a, p, pf);\n"
    s += "  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n"
    s += "template <int N, int V, int M>\nstatic std::vector<int64_t> vcs_attr() {\n"
    s += "  cudaFuncSetAttribute(vm::vcs_kernel<N, V, M>, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);\n"
    s += "  cudaFuncAttributes fa;\n  cudaFuncGetAttributes(&fa, vm::vcs_kernel<N, V, M>);\n"
    s += "  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)vm::SMEM, (int64_t)vm::NTH, vm::NS, vm::NX, vm::XS};\n}\n"
    s += "std::vector<int64_t> vcs_attrs(int64_t npad, int64_t v, int64_t m) {\n  switch (npad * 100 + v * 10 + m) {\n"
    s += "".join(f"    case {n * 100 + v * 10 + m}: return vcs_attr<{n}, {v}, {m}>();\n" for n, v, m in ks)
    s += "    default: return {-1};\n  }\n}\n"
    s += ("void vcs_run(torch::Tensor ws, torch::Tensor gsc, torch::Tensor xp, torch::Tensor unit, torch::Tensor cta, torch::Tensor out,\n"
          "             torch::Tensor err, torch::Tensor marks, int64_t npad, int64_t G, int64_t v, int64_t m, int64_t pf) {\n"
          "  vm::Args a{};\n  a.err = err.data_ptr<int>();\n  a.flags = 193;\n  vm::PArgs p{};\n"
          "  p.ws = reinterpret_cast<const uint8_t*>(ws.data_ptr()); p.gsc = gsc.data_ptr<float>();\n"
          "  p.xp = reinterpret_cast<const uint8_t*>(xp.data_ptr()); p.unit = reinterpret_cast<const int4*>(unit.data_ptr());\n"
          "  p.cta = reinterpret_cast<const int2*>(cta.data_ptr()); p.out = out.data_ptr<float>();\n"
          "  p.marks = marks.numel() >= G * 8 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;\n"
          "  p.npad = (int)npad;\n  switch (npad * 100 + v * 10 + m) {\n")
    s += "".join(f"    case {n * 100 + v * 10 + m}: vcs_launch<{n}, {v}, {m}>(a, p, (int)G, (int)pf); break;\n" for n, v, m in ks)
    s += '    default: TORCH_CHECK(false, "vcs: no such (N, V, M)");\n  }\n}\n'
    return s


def module_src(build):
    """gc_18 VML head (vm_main / vm_fin cut: never launched here) + dr_probe's harness + the lean loops + host code."""
    import dr_probe as DRP
    head, defs = gc18_vml()
    cut = head.find("__global__ void __launch_bounds__(NTH, 1) vm_main(")
    if cut > 0:
        head = head[:cut] + "}  // namespace vm (vm_main / vm_fin cut)" + chr(10)
    dev = DRP.PROBE_CU[:DRP.PROBE_CU.index(DRP.HOST_MARK)]
    dev = "\n".join(ln for ln in dev.split("\n") if not ln.startswith("template __global__ void drp_kernel<"))
    return head + dev + VCS_DEV, vcs_host(build), list(defs[build]) + ["-DVM_DR=0", "-DVM_DRP=0"]


def build_mod(build, verbose=False):
    import hashlib
    import time
    from torch.utils.cpp_extension import load_inline
    dev, host, defs = module_src(build)
    src = dev + host
    fl = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"] + defs
    tag = hashlib.sha1((VCS_CPP + src + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"vcs_{build}_{tag}", cpp_sources=VCS_CPP, cuda_sources=src, functions=["vcs_run", "vcs_attrs"],
                      extra_cuda_cflags=fl, verbose=verbose)
    return mod, time.time() - t0


# arm -> (V, M, pflags); stream arms run cold + warm, the others warm only (no weight bytes move)
ARMS = {"inc": (0, 0, 0), "lean": (1, 0, 0), "g2": (2, 0, 0), "inc_nw": (0, 0, 1), "lean_nw": (1, 0, 1), "g2_nw": (2, 0, 1),
        "inc_nx": (0, 0, 2), "lean_nx": (1, 0, 2), "g2_nx": (2, 0, 2), "inc_hs": (0, 0, 3), "lean_hs": (1, 0, 3), "g2_hs": (2, 0, 3),
        "early": (3, 0, 0), "spin": (8, 0, 0), "early_hs": (3, 0, 3), "spin_hs": (8, 0, 3),
        "lean_ns": (1, 1, 0), "g2_ns": (2, 1, 0), "dq_ns": (1, 2, 0), "mma_ns": (1, 3, 0)}
STREAM = ("inc", "lean", "g2", "early", "spin")
CHECKED = ("lean", "g2", "early", "spin", "early_hs", "spin_hs", "inc_nw", "lean_nw", "g2_nw", "inc_nx", "lean_nx", "g2_nx", "inc_hs", "lean_hs", "g2_hs")   # 4 runs each, bitwise vs inc (same producer flags)


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
    ntoks = tuple(int(v) for v in str(DRP.arg(argv, "ntoks", "8,16,32,64")).split(","))
    rounds, K, warm_set = int(DRP.arg(argv, "rounds", 9)), int(DRP.arg(argv, "units", 3)), int(DRP.arg(argv, "warm-set", 8))
    builds = [b for b in str(DRP.arg(argv, "builds", "bf2,w72")).split(",") if any(n in BUILDS[b] for n in ntoks)]
    out_p = ROOT / str(DRP.arg(argv, "out", "box_out/vcs/run.json"))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mods = {}
    with cf.ThreadPoolExecutor(len(builds)) as ex:
        futs = {b: ex.submit(build_mod, b) for b in builds}
        for b, f in futs.items():
            mods[b], dt = f.result()
            log(f"[vcs] built {b} in {dt:.0f} s; attrs (regs, local B, smem, thr, NS, NX, XS) " + " ".join(
                f"N{n}V{v}M{m} {list(mods[b].vcs_attrs(n, v, m))[:2]}" for n, v, m in kernels_of(b) if m == 0))
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
    log(f"[vcs] setup {time.time() - t0:.0f} s, {G} SMs, overhead {ov:.2f} us, gate {timer.gate_us} us, {nst} stages per CTA, rounds {rounds}")

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

    res = {"G": G, "overhead_us": ov, "rounds": rounds, "stages_per_cta": nst, "rows": []}
    for N in ntoks:
        for b in builds:
            if N not in BUILDS[b]:
                continue
            mod = mods[b]
            arms = [a for a, (v, m, pf) in ARMS.items() if (N, v, m) in kernels_of(b)]
            for mode in ("cold", "warm"):
                am = [a for a in arms if mode == "warm" or a in STREAM]
                P = DRP.plan(N, mode, G, K, warm_set, xsrc, dev)
                npad = P["npad"]
                outs = {a: torch.full((P["n"], 512, npad), float("nan"), dtype=torch.float32, device=dev) for a in am}

                def fn(a, mk=nomark, P=P, outs=outs):
                    v, m, pf = ARMS[a]
                    return lambda: mod.vcs_run(ws13, gsc13, P["xp"], P["unit"], P["cta"], outs[a], err, mk, npad, G, v, m, pf)
                chk = {}
                refs = {}
                for i in sorted({0, P["n"] // 2, P["n"] - 1}):
                    e, U = P["units"][i]
                    W = DRP.dequant_rows64(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e], perm[512 * U:512 * U + 512])
                    refs[i] = W @ xsrc[torch.as_tensor(P["toks"][e], dtype=torch.long, device=dev)].double().T
                for a in am:
                    err.zero_()
                    fn(a)()
                    torch.cuda.synchronize()
                    ref_a = {0: "inc", 1: "inc_nw", 2: "inc_nx", 3: "inc_hs"}[ARMS[a][2]]
                    if a not in ("inc",) + CHECKED or ref_a not in outs:
                        chk[a] = {"err": int(err[0])}
                        continue
                    o = outs[a]
                    rel = max(float((o[i, :, :N].double() - r).norm() / r.norm()) for i, r in refs.items())
                    o1 = o.clone()
                    nrep = 0
                    for _ in range(3):
                        fn(a)()
                        torch.cuda.synchronize()
                        nrep += int((o1 != outs[a]).sum())
                    ninc = int((o1 != outs[ref_a]).sum()) if a != ref_a else 0
                    where = None
                    if ninc:   # where: unit k within the CTA, seg h, 64-row WG tile q, token column
                        dm = (o1 != outs[ref_a]).view(G, K, 2, 4, 64, npad)
                        where = {"k": dm.sum((0, 2, 3, 4, 5)).tolist(), "h": dm.sum((0, 1, 3, 4, 5)).tolist(),
                                            "q": dm.sum((0, 1, 2, 4, 5)).tolist(), "n": dm.sum((0, 1, 2, 3, 4)).tolist()[:16],
                                            "ctas": int(dm.flatten(1).any(1).sum())}
                    chk[a] = {"where": where, "err": int(err[0]), "rel_fp64": rel, "repeat_equal": nrep == 0, "ndiff_rep": nrep, "ndiff_inc": ninc,
                              "bitwise_inc": ninc == 0,
                              "pad_zero": bool((o[:, :, N:] == 0).all()) if npad > N else True}
                    chk[a]["ok"] = chk[a]["err"] == 0 and (rel < 1e-4 or ARMS[a][2] != 0) and chk[a]["repeat_equal"] and chk[a]["pad_zero"]
                gr = {}
                for a in am:
                    f = fn(a)
                    with torch.cuda.stream(timer.stream):
                        f()
                        torch.cuda.synchronize()
                        g_ = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g_, stream=timer.stream):
                            f()
                    torch.cuda.synchronize()
                    gr[a] = g_
                ts = {a: [] for a in am}
                for rd in range(rounds):
                    k = rd % len(am)
                    for a in am[k:] + am[:k]:
                        ts[a].append(max(timed(gr[a], mode == "cold", mode == "warm") - ov, 1e-3))
                us = {a: statistics.median(v) / nst for a, v in ts.items()}
                clk, ghz = {}, {}
                for a in am:
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
                        cyc, nsx = mm[:, 6] - mm[:, 5], mm[:, 1:5].max(dim=1).values - mm[:, 0]
                        vals.append((float(cyc.median()) / nst, float((cyc / nsx.clamp(min=1)).median())))
                    clk[a] = statistics.median(v[0] for v in vals)
                    ghz[a] = statistics.median(v[1] for v in vals)
                row = {"N": N, "build": b, "mode": mode, "us_per_stage": us, "clk_per_stage": clk, "ghz": ghz, "check": chk,
                       "us_all": ts, "errs": {a: chk[a]["err"] for a in am}}
                res["rows"].append(row)
                base = clk.get("inc")
                log(f"[vcs] N{N} {b} {mode}: clk/stage " + " ".join(
                    f"{a} {clk[a]:.0f}" + (f" ({100 * (clk[a] / base - 1):+.1f}%)" if base and a != "inc" else "") for a in am)
                    + " | us/stage " + " ".join(f"{a} {us[a]:.3f}" for a in am) + " | GHz " + f"{statistics.median(ghz.values()):.2f}"
                    + " | ok " + " ".join(f"{a} {chk[a].get('ok', chk[a]['err'] == 0)}{'' if chk[a].get('bitwise_inc', True) else '(neq)'}"
                                         for a in am))
                del gr, outs, P
                torch.cuda.empty_cache()
    out_p.write_text(json.dumps(res, indent=1, default=str))
    log(f"[vcs] wrote {out_p} ({time.time() - t0:.0f} s)")


MAPC = {"mma": "M", "ldsm": "L", "lds": "s", "wgsync": "W", "mbar": "B", "bar": "y", "unif": "u", "bf16": "h", "alu": ".",
        "ctl": "|", "fp": "f", "gmem": "g", "sts": "t"}


def map_main(argv):
    """map SASSFILE N[,N]: one char per instruction of the stage loop (M mma, L ldsm, s lds, W warpgroup, B mbarrier, u uniform,
    h bf16 mul, . int alu, | control, y bar/warpsync, f fp32, g global, t sts)."""
    funcs = parse_funcs(pathlib.Path(argv[0]).read_text())
    want = {int(v) for v in argv[1].split(",")}
    for fn, ins in funcs.items():
        m = re.search(r"(?:drp|vcs)_kernelILi(\d+)E", fn)
        if not m or int(m.group(1)) not in want:
            continue
        n = int(m.group(1))
        op, cnt = ("HMMA", 2 * n) if (n <= 16 or "HGMMA" not in "".join(t for _, t in ins)) else ("HGMMA", 16)
        lp = stage_loop(ins, op, cnt)
        if lp is None:
            continue
        seq = "".join(MAPC.get(opclass(t.split()[0]), "?") for _, t in ins[lp[0]:lp[1] + 1])
        print(f"{fn[:50]} N{n} len {len(seq)}")
        for k in range(0, len(seq), 120):
            print("  " + seq[k:k + 120])


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "sass":
        sass_main(sys.argv[2:])
    elif len(sys.argv) > 3 and sys.argv[1] == "map":
        map_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        run_main(sys.argv[2:])
    else:
        print(__doc__)

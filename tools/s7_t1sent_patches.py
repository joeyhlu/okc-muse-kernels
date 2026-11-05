"""T1SENT: sentinel-validated `inter` for the T=1 HYBLO "tiny" MoE (TINYS bits 7 + -DOKC_TSENT=1).

Base  kernels/muse_decode_layer/k3_tinys7_c1.py (read as bytes, never written)
Out   box_out/s7/k3_t1sent_c1.py (LF, < 1,000,000 B)

What changes (ONLY the `tiny` key = HYBLO tokens <= 4; at HYBLO_T=(1,8) that is tokens == 1):
  producer  helper A epilogue: the two adjacent bf16 inter values of a lane pair are packed into one
            32-bit st.relaxed.gpu; a bf16 0xFFFF half (a NaN) is rewritten to 0x7FFF (still NaN);
            the doneA release publication is dropped (no fence on the helper).
  consumer  the first W2 unit of a job loads its x (inter) slice with ld.relaxed.gpu.v4 and re-loads
            only the 16-byte pieces that still contain the sentinel word 0xFFFFFFFF (bounded, sticky
            ctr[7] |= 8 on timeout). This replaces: tid-0 acquire poll on doneA + bar.sync 3 + __ldcg load.
  reset     after the grid barrier, CTAs >= M*FSEGP (non-combiners) write the sentinel back into rows
            [0, local routes) of the dedicated buffer (same place/ordering as the ship's doneA reset).
  buffer    a dedicated 64 x 4096 bf16 sentinel buffer (+guard) for `tiny` (scratch[2] of a separate
            scratch list); tiny2 (T=8) keeps the shared inter buffer and is compiled from identical
            preprocessed source (gate checks SASS identity).
Usage:  python tools/s7_t1sent_patches.py [--check]
"""
import hashlib
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUT = ROOT / "box_out/s7/k3_t1sent_c1.py"
LIMIT = 1_000_000

NONCE = ("# nonce t1sent-c1 (TINYS bits 7 on tiny / tiny2 + T1SENT sentinel-validated inter at tokens == 1 only; "
         "routing = exact top-8 of our own FP32 scores, ties to the lower expert index; no placement-dependent routing code)")

# ---------------------------------------------------------------- CUDA (inside _TINYS_CUDA only)
TSENT_HELPERS = r'''// T1SENT: sentinel-validated inter (tiny only). 0xFFFFFFFF = two bf16 0xFFFF halves; the producer
// rewrites any 0xFFFF half (a NaN) to 0x7FFF (NaN), so no published word can equal the sentinel.
#ifndef OKC_TSENT
#define OKC_TSENT 0
#endif
#if OKC_TSENT
static_assert(!(OKC_TINYS & 8), "T1SENT replaces the doneA gate; bit-8 staging needs doneA");
static_assert(OKC_L2H == 0 && OKC_TINY && OKC_SPLIT == 0, "T1SENT is the tiny (T=1) build only");
constexpr uint32_t TSENT_WORD = 0xFFFFFFFFu;
__device__ __forceinline__ uint32_t tsent_sanitize(uint32_t v) {
if ((v & 0xFFFFu) == 0xFFFFu) v = (v & 0xFFFF0000u) | 0x7FFFu;
if ((v >> 16) == 0xFFFFu) v = (v & 0x0000FFFFu) | 0x7FFF0000u;
return v;
}
__device__ __forceinline__ void tsent_st(__nv_bfloat16* a, uint32_t v) {
asm volatile("st.relaxed.gpu.global.b32 [%0], %1;\n" :: "l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ uint4 tsent_ld(const uint4* a) {
uint4 v;
asm volatile("ld.relaxed.gpu.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
             : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
return v;
}
__device__ __forceinline__ bool tsent_bad(const uint4& v) {
return v.x == TSENT_WORD || v.y == TSENT_WORD || v.z == TSENT_WORD || v.w == TSENT_WORD;
}
// Same addresses / register map as load_x_regs<true>; polls the data itself instead of doneA.
__device__ __forceinline__ void tsent_load_x(const Params& p, const Smem& S, int j, int lane, int warp,
                                             uint4 (&x)[2 * NCH]) {
const int g = lane >> 2, t = lane & 3;
const int cnt = S.jC[j], st = S.jS[j];
if (g < cnt) {
const uint4* src = reinterpret_cast<const uint4*>(p.inter + (size_t)(st + g) * ID + warp * (64 * NCH) + 16 * t);
unsigned miss = (1u << (2 * NCH)) - 1u;
for (unsigned spin = 0; miss && spin < TINYS_SPIN_LIMIT; ++spin) {
#pragma unroll
for (int c = 0; c < 2 * NCH; ++c)
if (miss & (1u << c)) x[c] = tsent_ld(src + (c >> 1) * 8 + (c & 1));
unsigned m2 = 0u;
#pragma unroll
for (int c = 0; c < 2 * NCH; ++c) m2 |= tsent_bad(x[c]) ? (1u << c) : 0u;
miss = m2;
}
if (miss) tinys_fail(p, 8);
} else {
#pragma unroll
for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);
}
__syncwarp();
}
#endif

constexpr int X1P = 512 + 8;
'''

CONSUMER_OLD = "#else\nif (PB) {\n#if OKC_TINYS & 1\nif (tid == 0) tinys_wait_flag(p, p.doneA + j, RB_A, 1);\n"
CONSUMER_NEW = ("#elif OKC_TSENT\n// T1SENT: no doneA poll / CTA barrier; each lane validates its own inter words.\n"
                "if (PB) tsent_load_x(p, S, j, lane, warp, x0);\nelse load_x_regs<PB>(p, S, j, lane, warp, x0);\n"
                + CONSUMER_OLD)

PRODUCER_OLD = r'''#pragma unroll
for (int h = 0; h < 2; ++h) {
const int tk = 2 * tt + h;
if (tk < cnt) {
const float gt = (h ? sm.y : sm.x) * gsc, up = (h ? sm.w : sm.z) * gsc;
st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
}
}
// TINYS_1: warp release publication (all helper stores precede sync).
if (q == nA - 1) {
'''
PRODUCER_NEW = r'''#if OKC_TSENT
// T1SENT: lanes gg / gg^1 (lane ^ 4, same tt) hold adjacent columns; the even lane stores both halves.
#pragma unroll
for (int h = 0; h < 2; ++h) {
const int tk = 2 * tt + h;
const float gt = (h ? sm.y : sm.x) * gsc, up = (h ? sm.w : sm.z) * gsc;
const __nv_bfloat16 hv = __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up));
const uint32_t mine = (uint32_t)*reinterpret_cast<const unsigned short*>(&hv);
const uint32_t odd = __shfl_xor_sync(0xffffffffu, mine, 4);
if (tk < cnt && !(gg & 1)) tsent_st(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), tsent_sanitize(mine | (odd << 16)));
}
#else
#pragma unroll
for (int h = 0; h < 2; ++h) {
const int tk = 2 * tt + h;
if (tk < cnt) {
const float gt = (h ? sm.y : sm.x) * gsc, up = (h ? sm.w : sm.z) * gsc;
st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
}
}
#endif
// TINYS_1: warp release publication (all helper stores precede sync).
if (!OKC_TSENT && q == nA - 1) {
'''

RESET_OLD = "for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;\n}\n"
RESET_NEW = RESET_OLD + r'''#if OKC_TSENT
// T1SENT: every consumer read of inter precedes its CTA's acq_rel barrier arrive, which this CTA has
// acquired; restore the sentinel in rows [0, local routes) for the next call (ship doneA-reset ordering).
{
const int jl = S.njF + S.njS - 1;
const int rows = S.njS > 0 ? S.jS[jl] + S.jC[jl] : 0;
const int c0 = (int)gridDim.x > p.M * FSEGP ? p.M * FSEGP : 0;
const int nw = rows * (ID / 8);
if ((int)blockIdx.x >= c0)
for (int i = ((int)blockIdx.x - c0) * NT + tid; i < nw; i += ((int)gridDim.x - c0) * NT)
reinterpret_cast<uint4*>(p.inter)[i] = make_uint4(TSENT_WORD, TSENT_WORD, TSENT_WORD, TSENT_WORD);
}
#endif
'''

# ---------------------------------------------------------------- Python
PY_EDITS = (
    ('fl=list(_HYBLO_FLAGS)+list(var)+["-DOKC_TINYS=7","-std=c++20"]  # TINYS bits 7 (tools/tinys_ship.py)',
     'fl=list(_HYBLO_FLAGS)+list(var)+["-DOKC_TINYS=7","-std=c++20"]+(["-DOKC_TSENT=1"]if k=="tiny"else[])  # TINYS bits 7 + T1SENT at tiny (tools/s7_t1sent_patches.py)'),
    (' _HYBLO["sc"],_HYBLO["full"]=_hyb_scratch(dev)\n',
     ' _HYBLO["sc"],_HYBLO["full"]=_hyb_scratch(dev)\n'
     '  _HYBLO["sent"]=torch.full((64*4096+_HYB_GUARD,),-1,dtype=torch.int16,device=dev) # T1SENT: bf16 0xFFFF rows for tiny + guard\n'
     '  _HYBLO["sc1"]=[_HYBLO["sc"][0],_HYBLO["sc"][1],_HYBLO["sent"][:64*4096].view(torch.bfloat16)]+list(_HYBLO["sc"][3:])\n'),
    ('  gok=_hyb_guards_ok()and all(bool((f[n:].view(vt)==sv).all())for f,n,vt,sv in _HYBLO["full"])\n',
     '  gok=_hyb_guards_ok()and all(bool((f[n:].view(vt)==sv).all())for f,n,vt,sv in _HYBLO["full"])\n'
     '  gok=gok and bool((_HYBLO["sent"]==-1).all()) # T1SENT: every self-test call restored the sentinel\n'),
    (' mod.okc_run(_HYBLO["w"],_HYBLO["sc"],x,ids,wts,out,0,32,_HYB["nsm"])\n',
     ' mod.okc_run(_HYBLO["w"],_HYBLO["sc1"]if k=="tiny"else _HYBLO["sc"],x,ids,wts,out,0,32,_HYB["nsm"])\n'),
)


def _sub(s, old, new, what):
    n = s.count(old)
    if n != 1:
        raise ValueError(f"T1SENT anchor {what}: count {n}")
    return s.replace(old, new)


def tinys_region(s):
    a = s.index('_TINYS_CUDA=r"""')
    b = s.index('"""', a + len('_TINYS_CUDA=r"""'))
    return a, b


def generate(base):
    lines = base.split("\n")
    assert lines[1].startswith("# nonce tinys7-c1 "), lines[1][:60]
    lines[1] = NONCE
    s = "\n".join(lines)
    a, b = tinys_region(s)
    cu = s[a:b]
    cu = _sub(cu, "constexpr int X1P = 512 + 8;\n", TSENT_HELPERS, "helpers")
    cu = _sub(cu, CONSUMER_OLD, CONSUMER_NEW, "consumer")
    cu = _sub(cu, PRODUCER_OLD, PRODUCER_NEW, "producer")
    cu = _sub(cu, RESET_OLD, RESET_NEW, "reset")
    s = s[:a] + cu + s[b:]
    for i, (old, new) in enumerate(PY_EDITS):
        s = _sub(s, old, new, f"py{i}")
    return s


def literal(src, name):
    import ast
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return ast.literal_eval(n.value)
    raise KeyError(name)


# ---------------------------------------------------------------- CPU checks
def _bf16_bits(f):
    import struct
    u = struct.unpack("<I", struct.pack("<f", f))[0]
    if (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF):
        return 0x7FFF  # cvt.rn.bf16.f32 NaN -> canonical NaN (sanitized below anyway)
    r = u + 0x7FFF + ((u >> 16) & 1)
    return (r >> 16) & 0xFFFF


def _san(v):
    if (v & 0xFFFF) == 0xFFFF:
        v = (v & 0xFFFF0000) | 0x7FFF
    if (v >> 16) == 0xFFFF:
        v = (v & 0xFFFF) | 0x7FFF0000
    return v


def _is_nan(h):
    return (h & 0x7F80) == 0x7F80 and (h & 0x7F) != 0


def cpu_check():
    out = {}
    # 1. Sentinel can never be a published word; sanitize is the identity except on the NaN 0xFFFF.
    for h in range(65536):
        s = _san(h | (0x1234 << 16)) & 0xFFFF
        assert s != 0xFFFF
        assert s == h or (h == 0xFFFF and s == 0x7FFF and _is_nan(h) and _is_nan(s))
    rnd = random.Random(7)
    for _ in range(200000):
        w = rnd.getrandbits(32)
        assert _san(w) != 0xFFFFFFFF
        lo, hi = w & 0xFFFF, w >> 16
        sw = _san(w)
        assert (sw & 0xFFFF) == (lo if lo != 0xFFFF else 0x7FFF) and (sw >> 16) == (hi if hi != 0xFFFF else 0x7FFF)
    for w in (0xFFFFFFFF, 0xFFFF0000, 0x0000FFFF, 0x7FFFFFFF, 0xFFFF7FFF):
        assert _san(w) != 0xFFFFFFFF
    out["sentinel"] = "PASS: 2^16 halves + 2e5 random words; 0xFFFF half -> 0x7FFF (NaN -> NaN), else identity"
    # 2. Producer pairing: for each unit, rows tk < cnt, cols rb*8 + 0..7 written exactly once, 4-B aligned.
    for cnt in range(1, 9):
        for rb in (0, 1, 255, 511):
            seen = {}
            for lane in range(32):
                gg, tt = lane >> 2, lane & 3
                for h in range(2):
                    tk = 2 * tt + h
                    if tk < cnt and not (gg & 1):
                        col = rb * 8 + gg
                        byte = (tk * 4096 + col) * 2
                        assert byte % 4 == 0
                        partner = lane ^ 4
                        assert (partner >> 2) == gg + 1 and (partner & 3) == tt
                        for cc in (col, col + 1):
                            assert (tk, cc) not in seen
                            seen[(tk, cc)] = lane
            assert set(seen) == {(tk, rb * 8 + c) for tk in range(cnt) for c in range(8)}
    out["producer_pairing"] = "PASS cnt 1..8: each (row, col) of a unit once, even lane stores (gg, gg+1), 4-B aligned"
    # 3. Consumer map identical to load_x_regs<true> (uint4 index c' = 2c + h  <-> src + c*8 + h).
    for warp in range(8):
        for lane in range(32):
            t = lane & 3
            base = warp * 512 + 16 * t  # bf16 elements
            for c in range(16):
                assert (c >> 1) * 8 + (c & 1) == {0: 0, 1: 1}[c & 1] + 8 * (c >> 1)
                elem = base + 8 * ((c >> 1) * 8 + (c & 1))
                assert elem % 8 == 0 and elem + 8 <= 4096
    out["consumer_map"] = "PASS: same 16 uint4 per lane as load_x_regs<true>"
    # 4. Reset coverage: rows used by producers/consumers == [0, rows); reset covers rows*512 uint4 once.
    FSEGP = 4096 // (256 * 4)
    trials = 0
    for M in (1, 2, 3, 4):
        for _ in range(400):
            ids = [rnd.sample(range(256), 8) for _ in range(M)]
            tot = [0] * 32
            for row in ids:
                for e in row:
                    if e < 32:
                        tot[e] += 1
            jobs, off = [], 0
            for e in range(32):  # route(): lane-ordered skinny jobs, jS = exclusive prefix, jC = tot
                if 0 < tot[e] <= 8:
                    jobs.append((e, off, tot[e]))
                off += tot[e]
            assert all(t <= 8 for t in tot)
            rows = jobs[-1][1] + jobs[-1][2] if jobs else 0
            used = {st + tk for _, st, c in jobs for tk in range(c)}
            assert used == set(range(rows)) and rows <= 64
            for G in (132, 128, 64, 33, 8, 4):
                c0 = M * FSEGP if G > M * FSEGP else 0
                nw = rows * 512
                cover = []
                for blk in range(G):
                    if blk < c0:
                        continue
                    for tid in range(256):
                        i = (blk - c0) * 256 + tid
                        while i < nw:
                            cover.append(i)
                            i += (G - c0) * 256
                assert sorted(cover) == list(range(nw)), (M, G, rows)
                assert all(i < 64 * 512 for i in cover)
            trials += 1
    out["reset_cover"] = f"PASS {trials} random routings x G in (132,128,64,33,8,4): exact once-cover of rows*512 uint4, <= 64 rows"
    # 5. Numeric identity of the published bits (finite values): packed word == the two ship bf16 stores.
    for _ in range(100000):
        a, b = rnd.uniform(-60, 60), rnd.uniform(-60, 60)
        lo, hi = _bf16_bits(a), _bf16_bits(b)
        w = _san(lo | (hi << 16))
        assert (w & 0xFFFF) == lo and (w >> 16) == hi
    out["bitwise_finite"] = "PASS: finite SwiGLU outputs publish the ship's exact bf16 bits"
    return out


def main():
    base_bytes = BASE.read_bytes()
    h0 = hashlib.sha256(base_bytes).hexdigest()
    base = base_bytes.decode("utf-8")
    assert "\r" not in base
    gen = generate(base)
    data = gen.encode("utf-8")
    assert b"\r" not in data and len(data) < LIMIT, len(data)
    compile(gen, str(OUT), "exec")
    res = cpu_check()
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == h0, "base modified"
    if "--check" in sys.argv:
        assert OUT.read_bytes() == data, "stale output"
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(data)
    res.update(out=str(OUT), bytes=len(data), base_sha256=h0, out_sha256=hashlib.sha256(data).hexdigest())
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

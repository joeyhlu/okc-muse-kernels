"""xks_patches.py: XKS (notes/study/fresh_eyes_oct29c.md mechanism 2) text patch on a layer's VML CUDA source (k3_gc_18 and later).

Every flag defaults to 0 = the shipped code (same SASS). Compile-time only; no runtime branch is added to the consumer.
  XKS_XNULL  step 0 ablation: stage_m (mma.sync path, N <= 16) takes its x fragments from lane registers instead of ldmatrix
             (no x smem reads; the MMAs stay live: their sums still reach the epilogue). Outputs are wrong by design.
  XKS_NOW    ablation: no weight stream after the first NS stages (the producer arrives without a copy; stale codes).
  XKS        K-split warp mapping in stage_m: the 4 warps of a WG own the WG's 4 tiles jointly, warp wp takes k64 step wp of
             all 4 tiles (x per warp per stage 4x fewer ldmatrix), one FP32 chain per tile; per unit the partial sums are
             combined through the unused tail of the unit's last x slot (held until then), fixed order w = 0, 1, 2, 3 per
             8-token chunk (deterministic; FP32 summation order differs from the shipped two-chain form, so not bitwise).
"""

PRE = """#ifndef XKS
#define XKS 0
#endif
#ifndef XKS_XNULL
#define XKS_XNULL 0
#endif
#ifndef XKS_NOW
#define XKS_NOW 0
#endif
"""

XN_OLD = "uint32_t b0, b1, b2, b3;\nldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);\n"
XN_NEW = ("uint32_t b0, b1, b2, b3;\n#if XKS_XNULL   // XKS step 0: no x smem reads, MMAs live\n"
          "b0 = xb ^ (0x3c003c00u + (uint32_t)((nt * 4 + i) * 2 + hf)); b1 = b0 ^ 0x00010001u; b2 = b0 ^ 0x00020002u; "
          "b3 = b0 ^ 0x00030003u;\n#else\n"
          "ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);\n#endif\n")

NW_OLD = "mbar_expect_tx(fb, US);\nbulk_g2s_ef(dst, src, SW, fb, pol);\nbulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);\n"
NW_NEW = ("if (XKS_NOW && it >= NS) { mbar_arrive(fb); } else {\n" + NW_OLD + "}\n")

FN_ANCHOR = "template <bool PB, int N>\n__device__ __forceinline__ void epilogue("
FN = r"""#if XKS
__device__ __forceinline__ void xks_fence_s() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
// XKS stage: warp wp = k64 step wp of the WG's 4 tiles; the previous stage's x slot is released here (one stage late:
// the unit's last slot carries the partial sums, released by xks_fin)
template <int N, bool FIRST>
__device__ __forceinline__ void stage_mx(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& px, float (&dx)[4][N / 2],
int q, int wp, int lane) {
const int ws = it % NS, xs = it % NX;
if constexpr (!FIRST) { if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + px)); }
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
const uint8_t* st = smem + ws * US + (q >> 1) * SW;
const uint4* cp = reinterpret_cast<const uint4*>(st + (q & 1) * 8192) + 32 * wp + lane;                  // tile t: + 128 t
const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + (q & 1) * 1024) + 32 * (wp >> 1) + lane;   // tile t: + 64 t
uint4 c[4];
uint32_t sw[4];
#pragma unroll
for (int t = 0; t < 4; ++t) { c[t] = cp[128 * t]; sw[t] = sp[64 * t]; }
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
const uint32_t xb = sa(smem + NS * US + xs * XS) + (uint32_t)(lane * 16 + wp * 1024);
const uint32_t par = (uint32_t)(wp & 1), k16q = 16u + ((uint32_t)A.flags >> 30);   // UNP SMQ: opaque 16, as stage_m
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
uint32_t bb[N / 8][4];
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
#if XKS_XNULL
bb[nt][0] = xb ^ (0x3c003c00u + (uint32_t)(nt * 2 + hf)); bb[nt][1] = bb[nt][0] ^ 0x00010001u; bb[nt][2] = bb[nt][0] ^ 0x00020002u;
bb[nt][3] = bb[nt][0] ^ 0x00030003u;
#else
ldsm4(xb + (uint32_t)((nt * 32 + 4 * hf) * 128), bb[nt][0], bb[nt][1], bb[nt][2], bb[nt][3]);
#endif
}
uint32_t hh[4][8];
#pragma unroll
for (int t = 0; t < 4; ++t) dec_half2(hf ? c[t].y : c[t].x, hf ? c[t].w : c[t].z, sw[t], par, hh[t], k16q, 64u, 4u);
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
#pragma unroll
for (int t = 0; t < 4; ++t) mma16816(&dx[t][4 * nt], hh[t][0], hh[t][4], hh[t][1], hh[t][5], bb[nt][0], bb[nt][1]);
}
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
#pragma unroll
for (int t = 0; t < 4; ++t) mma16816(&dx[t][4 * nt], hh[t][2], hh[t][6], hh[t][3], hh[t][7], bb[nt][2], bb[nt][3]);
}
}
px = xs;
++it;
}
// unit end: warp wp's tile = sum over the 4 warps' k64-step partials, order w = 0..3, per 8-token chunk through the unused
// tail [N * 512, XS) of the held x slot px (WG q: 12 slots (t, w != t) x 512 B at + q * 6144); then the slot is released
template <int N>
__device__ __forceinline__ void xks_fin(uint8_t* smem, uint32_t bars, int px, const float (&dx)[4][N / 2], float* d, int q,
int wp, int lane) {
static_assert(N > 16 || N * 512 + 4 * 6144 <= XS, "XKS sums do not fit the x slot tail");
float4* xr = reinterpret_cast<float4*>(smem + NS * US + px * XS + N * 512 + q * 6144) + lane;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
if (nt > 0) bar_sync(1 + q, 128);
float4 own = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
for (int t = 0; t < 4; ++t) {
const float4 v = make_float4(dx[t][4 * nt], dx[t][4 * nt + 1], dx[t][4 * nt + 2], dx[t][4 * nt + 3]);
if (t == wp) own = v;
else xr[32 * (3 * t + wp - (wp > t ? 1 : 0))] = v;
}
bar_sync(1 + q, 128);
float4 r[4];
#pragma unroll
for (int w = 0; w < 4; ++w) r[w] = (w == wp) ? own : xr[32 * (3 * wp + w - (w > wp ? 1 : 0))];
d[4 * nt] = ((r[0].x + r[1].x) + r[2].x) + r[3].x;
d[4 * nt + 1] = ((r[0].y + r[1].y) + r[2].y) + r[3].y;
d[4 * nt + 2] = ((r[0].z + r[1].z) + r[2].z) + r[3].z;
d[4 * nt + 3] = ((r[0].w + r[1].w) + r[2].w) + r[3].w;
}
xks_fence_s();   // generic writes into the slot before its next async-proxy (bulk copy) fill
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + px));
}
#endif
"""

SEG_OLD1 = "if (sync_path) {\nfloat d2[2][N / 2];\n"
SEG_NEW1 = """if (sync_path) {
#if XKS
float dx[4][N / 2];
#pragma unroll
for (int t = 0; t < 4; ++t)
#pragma unroll
for (int k = 0; k < N / 2; ++k) dx[t][k] = 0.f;
int px = 0;
stage_mx<N, true>(A, smem, bars, it, px, dx, q, wp, lane);
SEGMK(if (smk.on) smk.t1 = gtime();)
if constexpr (VM_DEFER) pend_flush(A, pd, q, wp, lane);
#pragma unroll 1
for (int s = s0 + 1; s < s1; ++s) stage_mx<N, false>(A, smem, bars, it, px, dx, q, wp, lane);
SEGMK(if (smk.on) smk.t2 = smk.t3 = gtime();)
xks_fin<N>(smem, bars, px, dx, d, q, wp, lane);
#else
float d2[2][N / 2];
"""
SEG_OLD2 = "for (int k = 0; k < N / 2; ++k) d[k] = d2[0][k] + d2[1][k];\n} else {\n"
SEG_NEW2 = "for (int k = 0; k < N / 2; ++k) d[k] = d2[0][k] + d2[1][k];\n#endif\n} else {\n"

SM_ANCHOR = "template <int N>\n__device__ __forceinline__ void stage_m("


def _sub(src, old, new, n=1):
    c = src.count(old)
    assert c == n, (c, n, old[:80])
    return src.replace(old, new)


def patch(src):
    s = PRE + src
    s = _sub(s, XN_OLD, XN_NEW)
    s = _sub(s, NW_OLD, NW_NEW, 2)
    s = _sub(s, FN_ANCHOR, FN + FN_ANCHOR)
    s = _sub(s, SEG_OLD1, SEG_NEW1)
    s = _sub(s, SEG_OLD2, SEG_NEW2)
    return s

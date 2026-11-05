"""S7 KS3 (sliding attention, decode B=16 / B=32: ks3_kernel<16,1> VSL and ks3_kernel<8,0>) latency patch.

Accounting (R-ks3g-0 marks, box, us after first entry, p50/max): d16 issue-end 2.82/3.58, landed 3.58/4.61, attention
5.12/5.89, partial 5.89/6.66, ticket 6.66/7.42, end 7.94/8.19; d32 issue-end 2.82/5.63, landed 5.12/7.17, attention
7.17/8.70, partial 8.19/9.73, ticket 8.70/10.50, end 10.75/11.52. Null consumer (same gather, no compute) ends at
3.58/4.10 (d16 NS16) and 5.89/6.66 (d32 NS8): the q path (positions -> cos_sin load issued AFTER the throttled
cp.async issue, then q loads at loaded DRAM latency) and the whole-tile wait_all put ~2 us of serial work behind the
last landed byte.

Changes (math identical: every key of the 2048 window, FP8 KV, f16 q / P exactly as before; the only rounding-order
change is that the tail CTA's warps 6/7 fold the new key into their online softmax first instead of last):
  1. positions / cache_loc are loaded together with cache_seqlens (one round trip), cos_sin with the page table.
  2. the K/V gather is committed as 2*NCH cp.async groups K(c0), V(c0), K(c1), V(c1): each warp runs QK of chunk c
     as soon as K(c) landed and PV as soon as V(c) landed (wait_group + barrier), so compute overlaps the landing.
  3. the tail CTA's extra one-key chunk (warps 6/7) runs before the old-key chunks (it needs no gathered data).
  4. q-first (QF, KS3_QF_MASK bit MG): PDL wait + q / new-k / gate loads before the page-table loads and q-prep before
     the gather issue, so the q path is unloaded and never queued behind the throttled cp.async issue. The
     launch_dependents point stays right after the gather issue. v1: QF only for MG = 0 (d32s; its predecessor sk2
     never triggers early). v2: QF for MG = 0 and 1 (d16s too; costs the pre-wait gather of CTAs that the qkv GD lets
     become resident early).

usage: python -B -I tools/s7_ks3_patches.py [--in kernels/muse_decode_layer/k3_tinys7_c1.py] [--outdir box_out/s7]
writes box_out/s7/k3_ks3_v1.py (mask 1) and box_out/s7/k3_ks3_v2.py (mask 3); --cu DIR also writes a standalone
compile unit per variant for the nvcc gate (device code only, explicit instantiations)."""
import argparse, pathlib, re, sys

ROW = 'kernels/muse_decode_layer/k3_tinys7_c1.py'
A0 = 'template <typename F>\n__device__ __forceinline__ void chunk32('
A1 = 'if (tail && q4 == 3) chunk32(kn, vn, 0, hw, qf, o, m, l, cl2, [&](int k) { return k == 0; });\n'
OLD_NONCE = '# nonce tinys7-c1 '

NEW = r'''__device__ __forceinline__ void ks3_cpwait(const int n) {
switch (n) {
case 0: asm volatile("cp.async.wait_group 0;\n" ::: "memory"); break;
case 1: asm volatile("cp.async.wait_group 1;\n" ::: "memory"); break;
case 2: asm volatile("cp.async.wait_group 2;\n" ::: "memory"); break;
case 3: asm volatile("cp.async.wait_group 3;\n" ::: "memory"); break;
case 4: asm volatile("cp.async.wait_group 4;\n" ::: "memory"); break;
case 5: asm volatile("cp.async.wait_group 5;\n" ::: "memory"); break;
case 6: asm volatile("cp.async.wait_group 6;\n" ::: "memory"); break;
default: asm volatile("cp.async.wait_group 7;\n" ::: "memory"); break;
}
}
template <typename F, typename W>
__device__ __forceinline__ void chunk32(const uint8_t* Ks, const uint8_t* Vs, const int x0, const int h, const uint32_t (&qf)[4][2],
float (&o)[8][4], float& m, float& l, const float cl2, F valid, W vwait) {
const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
float sc[4][2];
#pragma unroll
for (int j = 0; j < 4; ++j) {
const int key = x0 + 8 * j + g;
const uint4 kv = *reinterpret_cast<const uint4*>(Ks + key * 128 + 16 * kswz(key, 4 * h + t));
const uint32_t wv[4] = {kv.x, kv.y, kv.z, kv.w};
float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
for (int s = 0; s < 4; ++s) {
const uint32_t a[4] = {qf[s][0], 0u, qf[s][1], 0u};
mma16816(c, a, e4m3x2_to_h2(wv[s]), e4m3x2_to_h2(wv[s] >> 16));
}
sc[j][0] = c[0];
sc[j][1] = c[1];
}
float x = neg_inf();
#pragma unroll
for (int j = 0; j < 4; ++j) {
#pragma unroll
for (int e = 0; e < 2; ++e) {
sc[j][e] = valid(x0 + 8 * j + 2 * t + e) ? sc[j][e] * cl2 : neg_inf();
x = fmaxf(x, sc[j][e]);
}
}
x = fmaxf(x, __shfl_xor_sync(FULL, x, 1));
x = fmaxf(x, __shfl_xor_sync(FULL, x, 2));
const float nm = fmaxf(m, x);
const float u = (nm == neg_inf()) ? 0.f : nm;
const float al = ex2(m - u);
m = nm;
l *= al;
#pragma unroll
for (int n = 0; n < 8; ++n) { o[n][0] *= al; o[n][1] *= al; }
uint32_t pa[2][4];
#pragma unroll
for (int kk = 0; kk < 2; ++kk) {
const float p00 = ex2(sc[2 * kk][0] - u), p01 = ex2(sc[2 * kk][1] - u);
const float p10 = ex2(sc[2 * kk + 1][0] - u), p11 = ex2(sc[2 * kk + 1][1] - u);
l += (p00 + p01) + (p10 + p11);
pa[kk][0] = pack_h2(p00, p01);
pa[kk][1] = 0u;
pa[kk][2] = pack_h2(p10, p11);
pa[kk][3] = 0u;
}
vwait();
#pragma unroll
for (int kk = 0; kk < 2; ++kk) {
const int xa = x0 + 16 * kk + 2 * t;
const uint2 va = vld(Vs, xa, h, g), vb = vld(Vs, xa + 1, h, g), vc = vld(Vs, xa + 8, h, g), vd = vld(Vs, xa + 9, h, g);
const uint32_t w01[4] = {__byte_perm(va.x, vb.x, 0x5140), __byte_perm(va.x, vb.x, 0x7362),
__byte_perm(va.y, vb.y, 0x5140), __byte_perm(va.y, vb.y, 0x7362)};
const uint32_t w23[4] = {__byte_perm(vc.x, vd.x, 0x5140), __byte_perm(vc.x, vd.x, 0x7362),
__byte_perm(vc.y, vd.y, 0x5140), __byte_perm(vc.y, vd.y, 0x7362)};
#pragma unroll
for (int jn = 0; jn < 8; ++jn) {
const uint32_t s01 = w01[jn >> 1], s23 = w23[jn >> 1];
mma16816(o[jn], pa[kk], e4m3x2_to_h2((jn & 1) ? (s01 >> 16) : s01), e4m3x2_to_h2((jn & 1) ? (s23 >> 16) : s23));
}
}
}
#define KS3_QF_MASK @QFMASK@
template <int NS, int MG = 0>
__global__ void __launch_bounds__(NTHR, 2) ks3_kernel(const __grid_constant__ Params3 p) {
#if KS1_ARCH_OK
using G = G3<NS>;
constexpr int KMAX = G::KMAX, KPW = G::KPW, NCH = G::NCH;
constexpr bool QF = ((KS3_QF_MASK >> MG) & 1) != 0;
static_assert(2 * NCH <= 8, "KS3: at most 8 cp.async groups");
extern __shared__ __align__(128) uint8_t smem[];
uint8_t* Ks = smem;
uint8_t* Vs = smem + G::OFF_V;
uint8_t* kn = smem + G::OFF_KN;
uint8_t* vn = smem + G::OFF_VN;
uint8_t* qs = smem + G::OFF_Q;
float* mw = reinterpret_cast<float*>(smem + G::OFF_ML);
float* lw = mw + NWARP * 8;
int* misc = reinterpret_cast<int*>(smem + G::OFF_MISC);
const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
const int s = blockIdx.x, b = blockIdx.y;
const int cta = b * NS + s;
const bool tail = (s == NS - 1);
const int hw = warp & 1, q4 = warp >> 1;
if constexpr (MG == 2) vsl_carrive_relaxed();
if (p.trig == 2) pdl_trigger();
KS3P(0);
if (tail) {
reinterpret_cast<uint4*>(kn)[tid] = make_uint4(0u, 0u, 0u, 0u);
reinterpret_cast<uint4*>(kn)[tid + NTHR] = make_uint4(0u, 0u, 0u, 0u);
}
// independent inputs in one round trip (positions / cache_loc no longer wait behind the gather issue)
const int sraw = __ldg(p.cache_seqlens + b);
const long long pos = p.rope ? __ldg(p.positions + b) : 0ll;
const long long loc = tail ? __ldg(p.cache_loc + b) : 0ll;
const __nv_bfloat16* xr = p.qkv + (size_t)b * p.stride_qkv;
const int mr = tid >> 4, mu = tid & 15;
uint32_t qr0 = 0u, qr1 = 0u, nr = 0u;
uint2 graw = make_uint2(0u, 0u);
if constexpr (QF) {
KS3P(1);
pdl_wait();
KS3P(2);
qr0 = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)warp * 64) + lane);
qr1 = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(warp + 8) * 64) + lane);
if (tail && warp < 4) nr = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(HQ + warp) * 64) + lane);
graw = __ldcg(reinterpret_cast<const uint2*>(xr + (size_t)(HQ + 2 * HKV) * 64 + (size_t)mr * 64 + 4 * mu));
}
float2 cs = make_float2(1.f, 0.f);
if (p.rope) cs = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
const int seq = max(sraw - 1, 0);
const int n_old = min(seq, WIN - 1);
const int kstart = seq - n_old;
constexpr int PER = KMAX * 8 / NTHR;
static_assert(PER == 4 * NCH, "KS3: loader block j = quarter * NCH + chunk");
int rows[PER];
{
const int* pt = p.page_table + (size_t)b * p.pt_cols;
#pragma unroll
for (int j = 0; j < PER; ++j) {
const int key = (tid + NTHR * j) >> 3;
const int gs = s * KMAX + key;
rows[j] = (gs < n_old && kstart + gs < p.pt_cols) ? __ldg(pt + kstart + gs) : -1;
}
}
auto qprep = [&]() {
const float2 a = norm_rope(qr0, cs, p.rope, p.eps), c = norm_rope(qr1, cs, p.rope, p.eps);
reinterpret_cast<uint32_t*>(qs + warp * 128)[lane] = pack_h2(a.x, a.y);
reinterpret_cast<uint32_t*>(qs + (warp + 8) * 128)[lane] = pack_h2(c.x, c.y);
if (tail) __syncthreads();   // kn / vn zero fill done before the new key is written into them
if (tail && warp < 4) {
const int kind = warp >> 1, head = warp & 1;
float2 v = kind ? bf2f(nr) : norm_rope(nr, cs, p.rope, p.eps);
v.x = fminf(fmaxf(v.x, -FP8_MAX), FP8_MAX);
v.y = fminf(fmaxf(v.y, -FP8_MAX), FP8_MAX);
const uint32_t q8 = (uint32_t)__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
*reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + (size_t)loc * 128 + head * 64 + 2 * lane) = (uint16_t)q8;
const int bb = 64 * head + 2 * lane;
*reinterpret_cast<uint16_t*>((kind ? vn : kn) + 16 * (bb >> 4) + (bb & 15)) = (uint16_t)q8;
}
};
if constexpr (QF) qprep();
// gather, one cp.async group per (chunk, K|V): block j = q4 * NCH + c holds keys 32 j .. 32 j + 31 = chunk c of quarter q4
#pragma unroll
for (int c = 0; c < NCH; ++c) {
#pragma unroll
for (int hv = 0; hv < 2; ++hv) {
#pragma unroll
for (int qq = 0; qq < 4; ++qq) {
const int j = qq * NCH + c;
const int r = tid + NTHR * j, key = r >> 3, ch = r & 7;
const bool ok = rows[j] >= 0;
const size_t go = (size_t)(ok ? rows[j] : 0) * 128 + 16 * ch;
if (hv == 0) cp_async16(Ks + key * 128 + 16 * kswz(key, ch), p.k_cache + go, ok ? 16 : 0);
else cp_async16(Vs + key * 128 + 16 * vswz(key, ch), p.v_cache + go, ok ? 16 : 0);
}
asm volatile("cp.async.commit_group;\n" ::: "memory");
}
}
if constexpr (!QF) {
KS3P(1);
pdl_wait();
if (p.trig == 1) pdl_trigger();
KS3P(2);
qr0 = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)warp * 64) + lane);
qr1 = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(warp + 8) * 64) + lane);
if (tail && warp < 4) nr = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(HQ + warp) * 64) + lane);
graw = __ldcg(reinterpret_cast<const uint2*>(xr + (size_t)(HQ + 2 * HKV) * 64 + (size_t)mr * 64 + 4 * mu));
qprep();
} else {
if (p.trig == 1) pdl_trigger();
}
__syncthreads();   // qs, kn, vn visible
KS3P(3);
uint32_t qf[4][2];
{
const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs + (hw * 8 + g) * 128);
#pragma unroll
for (int k = 0; k < 4; ++k) { qf[k][0] = qa[8 * t + 2 * k]; qf[k][1] = qa[8 * t + 2 * k + 1]; }
}
const float cl2 = p.qk_sm * LOG2E;
float o[8][4];
#pragma unroll
for (int n = 0; n < 8; ++n) { o[n][0] = 0.f; o[n][1] = 0.f; o[n][2] = 0.f; o[n][3] = 0.f; }
float m = neg_inf(), l = 0.f;
const int base = s * KMAX;
// the new key needs no gathered data: fold it first, while the old keys land
if (tail && q4 == 3) chunk32(kn, vn, 0, hw, qf, o, m, l, cl2, [&](int k) { return k == 0; }, [] {});
#pragma unroll
for (int c = 0; c < NCH; ++c) {
ks3_cpwait(2 * (NCH - 1 - c) + 1);   // K(c) landed (own copies), barrier -> all copies
__syncthreads();
chunk32(Ks, Vs, q4 * KPW + 32 * c, hw, qf, o, m, l, cl2, [&](int k) { return base + k < n_old; },
[&] { ks3_cpwait(2 * (NCH - 1 - c)); __syncthreads(); });
}
'''


def build(row_text, mask, tag, desc):
    a = row_text.index(A0)
    assert row_text.count(A0) == 1 and row_text.count(A1) == 1
    e = row_text.index(A1) + len(A1)
    assert a < e
    out = row_text[:a] + NEW.replace('@QFMASK@', str(mask)) + row_text[e:]
    lines = out.split('\n')
    assert lines[1].startswith(OLD_NONCE), lines[1][:80]
    lines[1] = f'# nonce {tag} ({desc})'
    out = '\n'.join(lines)
    compile(out, tag, 'exec')
    return out


def cu_unit(text):
    """Device-only compile unit: the ks1 / ks3 namespaces of KS1_CUDA_SRC + explicit instantiations."""
    s = text.index('KS1_CUDA_SRC=r"""') + len('KS1_CUDA_SRC=r"""')
    cuda = text[s:text.index('"""', s)]
    a = cuda.index('namespace ks1 {')
    b = cuda.index('#define KS1_INSTS @INSTS@')
    head = ('#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n#include <cuda_fp8.h>\n#include <stdint.h>\n#include <vector>\n'
            '#include <cuda_runtime.h>\n#include <stdexcept>\n'
            '#define C10_CUDA_CHECK(x) do { if ((x) != cudaSuccess) throw std::runtime_error("cuda"); } while (0)\n'
            '#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)\n#define KS1_ARCH_OK 1\n#else\n#define KS1_ARCH_OK 0\n#endif\n')
    inst = ('template __global__ void ks3::ks3_kernel<16, 1>(const ks3::Params3);\n'
            'template __global__ void ks3::ks3_kernel<8, 0>(const ks3::Params3);\n'
            'template __global__ void ks3::ks3_kernel<16, 0>(const ks3::Params3);\n'
            'template __global__ void ks3::ks3_kernel<4, 0>(const ks3::Params3);\n')
    return head + cuda[a:b] + inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default=ROW)
    ap.add_argument('--outdir', default='box_out/s7')
    ap.add_argument('--cu', default=None)
    ap.add_argument('--base-cu', action='store_true', help='also write the unpatched row compile unit')
    a = ap.parse_args()
    raw = pathlib.Path(a.inp).read_bytes()
    assert b'\r' not in raw
    row = raw.decode('utf-8')
    outs = {
        'k3_ks3_v1.py': build(row, 1, 'ks3-v1', 'KS3 d16s/d32s: K/V cp.async groups consumed as they land, hoisted '
                              'positions/cache_loc/cos_sin, tail key first, q-first at B=32 only'),
        'k3_ks3_v2.py': build(row, 3, 'ks3-v2', 'KS3 d16s/d32s: K/V cp.async groups consumed as they land, hoisted '
                              'positions/cache_loc/cos_sin, tail key first, q-first at B=16 and B=32'),
    }
    od = pathlib.Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    for name, text in outs.items():
        data = text.encode('utf-8')
        assert len(data) < 1_000_000, (name, len(data))
        (od / name).write_bytes(data)
        print('wrote', od / name, len(data), 'bytes')
    if a.cu:
        cd = pathlib.Path(a.cu)
        cd.mkdir(parents=True, exist_ok=True)
        for name, text in outs.items():
            (cd / (name[:-3] + '.cu')).write_bytes(cu_unit(text).encode('utf-8'))
        if a.base_cu:
            (cd / 'k3_row.cu').write_bytes(cu_unit(row).encode('utf-8'))
        print('wrote compile units to', cd)


if __name__ == '__main__':
    main()

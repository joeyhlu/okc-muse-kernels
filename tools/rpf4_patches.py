"""RPF4 lab: four-warp rp_final for 33..128 tokens (rp_final4_kernel). T = 128 (v16s, HYA two-term query) still runs the one-CTA rp_final (256 threads, shared
atomics with heavy contention on the ~7 hot local experts, repeated-minimum elimination loop, 16-way select refill, 24 scalar stores per token).
rp_final4_kernel: one thread per token (up to 4 warps; blockDim = 32 * ceil(T / 32)), the rp_final2 algorithm per warp plus one cross-warp step:
  * per warp: token x local-expert masks (occ, must, tie, cmask) as rp_final2, 5-stage 32x32 bit transpose -> per-expert popc partial of the warp's tokens,
    REDUX.OR of must / tie masks, tie-token count by ballot; the partials go to shared memory (s_occ[warp][expert], s_must / s_tany / s_tcnt[warp]);
  * one __syncthreads; then the elimination of rp_final2 (occ summed over the warps, must / tie unions, rsum / dref / rh / dh, same rmin / dmin, rank-by-key
    prefix) = the drops of rp_final's repeated-minimum loop. RED = false (default): in warp 0, elim mask broadcast through shared memory behind a second
    __syncthreads. RED = true (host pdl bit 1, Python RPF4_RED): every warp computes it redundantly from the shared partials, no second barrier;
  * each thread refills its own token (r = 7..0, lowest candidate first), float4 / int4 stores (host checks contiguity / alignment).
Same floor flags as rpflags_patches.py (|rp_m| = margin + N: bit 0 tie window 0.008, bit 1 distinct cushion +1). Assumes distinct ids within a token's records.
Host: rp_final4 (same signature as rp_final, T <= 128), exported next to rp_final / rp_final2 in both load_inline lists. Python: RPF4_ON (default False)
swaps the call in _v8_tail when 32 < tokens <= 128 (else the existing rp_final2 / rp_final choice); RPF4_RED (default False) sets pdl bit 1.
Base kernels/muse_decode_layer/k3_fin5_c1.py -> kernels/lab/k3_rpf4_ab.py."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_fin5_c1.py"
OUT = ROOT / "kernels/lab/k3_rpf4_ab.py"

DECL2 = "void rp_final2(torch::Tensor rid, torch::Tensor rsv, torch::Tensor rky, torch::Tensor lstart, torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, double rscale, double deps, double rp_m, int64_t pdl);\n"
DECL4 = DECL2.replace("rp_final2", "rp_final4")

KERNEL = r"""// RPF4: rp_final for 33..128 tokens, one thread per token on up to 4 warps. Per warp: masks + 32x32 bit transpose -> expert count partials in shared memory;
// one barrier; rank-by-key elimination (as rp_final2) in warp 0 + broadcast (RED = false) or redundantly in every warp (RED = true); per-thread refill + stores.
template <bool RED>
__global__ void __launch_bounds__(128) rp_final4_kernel(const int* __restrict__ rid, const float* __restrict__ rsv, const int* __restrict__ rky,
const int* __restrict__ lstart, float* __restrict__ tw, int* __restrict__ tid, int* __restrict__ lid, int T, float rscale, float deps, float rp_m) {
const bool rzc = rp_m < 0.f;  // RZC: exact floors
const float am_ = fabsf(rp_m);
const int fl_ = (int)am_;  // rpflags: |rp_m| = margin + flags (the margin is < 1)
const float RP_KT = 0.030f - (am_ - (float)fl_);
const float RP_RT = 0.050f - (am_ - (float)fl_);
const float tiew_ = (fl_ & 1) ? 0.008f : 0.003f;  // bit 0: wider tie window
const int dcush_ = (fl_ >> 1) & 1;               // bit 1: one more distinct expert of cushion
__shared__ int s_occ[4][32];
__shared__ int s_tcnt[4];
__shared__ unsigned s_must[4], s_tany[4], s_elim;
const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
const int nw = (T + 31) >> 5;
asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
asm volatile("griddepcontrol.wait;\n" ::: "memory");
const int ls = __ldcg(lstart);
int id[16];
float sv[16];
unsigned occm = 0u, mustm = 0u, tiem = 0u, cmask = 0u;
if (t < T) {
float bv[16];
#pragma unroll
for (int q = 0; q < 4; ++q) {
const int4 a = __ldcg(reinterpret_cast<const int4*>(rid + t * 16) + q);
const float4 b = __ldcg(reinterpret_cast<const float4*>(rsv + t * 16) + q);
const int4 c = __ldcg(reinterpret_cast<const int4*>(rky + t * 16) + q);
id[4 * q] = a.x; id[4 * q + 1] = a.y; id[4 * q + 2] = a.z; id[4 * q + 3] = a.w;
sv[4 * q] = b.x; sv[4 * q + 1] = b.y; sv[4 * q + 2] = b.z; sv[4 * q + 3] = b.w;
const unsigned k0 = (unsigned)c.x, k1 = (unsigned)c.y, k2 = (unsigned)c.z, k3 = (unsigned)c.w;
bv[4 * q] = k0 == 0u ? -INFINITY : __uint_as_float((k0 & 0x80000000u) ? (k0 & 0x7fffffffu) : ~k0);
bv[4 * q + 1] = k1 == 0u ? -INFINITY : __uint_as_float((k1 & 0x80000000u) ? (k1 & 0x7fffffffu) : ~k1);
bv[4 * q + 2] = k2 == 0u ? -INFINITY : __uint_as_float((k2 & 0x80000000u) ? (k2 & 0x7fffffffu) : ~k2);
bv[4 * q + 3] = k3 == 0u ? -INFINITY : __uint_as_float((k3 & 0x80000000u) ? (k3 & 0x7fffffffu) : ~k3);
}
const float eight = bv[7], nine = bv[8];
#pragma unroll
for (int r = 0; r < 16; ++r) {
const int rel = (id[r] - ls) & (NE - 1);
const bool loc = rel < 32;
const unsigned bit = loc ? (1u << (rel & 31)) : 0u;
if (r < 8) {
occm |= bit;
if ((bv[r] - nine) > RP_KT) mustm |= bit;
} else if (!loc && (eight - bv[r]) <= RP_RT) {
cmask |= 1u << (r - 8);
}
if (rzc && r == 8 && (eight - bv[r]) <= tiew_) tiem |= bit;
}
}
// lane e <- column e of this warp's token x expert occupancy matrix (5-stage 32x32 bit transpose): tokens of the warp that picked local expert e
unsigned xt = occm;
#pragma unroll
for (int s = 16; s > 0; s >>= 1) {
const unsigned hi = s == 16 ? 0xffff0000u : s == 8 ? 0xff00ff00u : s == 4 ? 0xf0f0f0f0u : s == 2 ? 0xccccccccu : 0xaaaaaaaau;
const unsigned p = __shfl_xor_sync(0xffffffffu, xt, s);
xt = (lane & s) ? ((xt & hi) | ((p >> s) & ~hi)) : ((xt & ~hi) | ((p << s) & hi));
}
s_occ[warp][lane] = __popc(xt);
const unsigned must_w = __reduce_or_sync(0xffffffffu, mustm);
const unsigned tany_w = __reduce_or_sync(0xffffffffu, tiem);
const int tcnt_w = __popc(__ballot_sync(0xffffffffu, tiem != 0u));
if (lane == 0) { s_must[warp] = must_w; s_tany[warp] = tany_w; s_tcnt[warp] = tcnt_w; }
__syncthreads();
unsigned elim = 0u;
if (RED || warp == 0) {
int occ = 0, tcnt = 0;
unsigned must_any = 0u, tie_any = 0u;
#pragma unroll
for (int w = 0; w < 4; ++w)
if (w < nw) { occ += s_occ[w][lane]; must_any |= s_must[w]; tie_any |= s_tany[w]; tcnt += s_tcnt[w]; }
const unsigned occ_any = __ballot_sync(0xffffffffu, occ > 0);
const int rsum = (int)__reduce_add_sync(0xffffffffu, (unsigned)occ);
const int dref = __popc(occ_any);
int rmin, dmin;
if (rzc) {
const int rh = rsum + tcnt;
const int dh = __popc(occ_any | tie_any);
rmin = max(0, (int)ceilf(0.92f * (float)rh - 4.f + 1e-4f));
dmin = max(0, (int)ceilf(0.85f * (float)dh - 2.f + (float)dcush_ + 1e-4f));
} else {
rmin = max(0, (int)ceilf(0.92f * (float)rsum - 4.f + 1.f + 0.01f * (float)rsum));
dmin = max(dref > 0 ? 1 : 0, (int)ceilf(0.85f * (float)dref - 2.f + 1.f));
}
// eligible experts ordered by key occ * 64 + lane; expert of rank k (prefix occ sum P, k + 1 experts) is dropped iff the floors still hold after it
const bool elig = occ > 0 && !((must_any >> lane) & 1u);
const int key = elig ? occ * 64 + lane : 0x7fffffff;
int cnt = 0, pre = 0;
#pragma unroll
for (int j = 0; j < 32; ++j) {
const int kj = __shfl_sync(0xffffffffu, key, j);
if (kj <= key) { ++cnt; pre += kj >> 6; }
}
elim = __ballot_sync(0xffffffffu, elig && rsum - pre >= rmin && dref - cnt >= dmin);
if (!RED && lane == 0) s_elim = elim;
}
if (!RED) {
__syncthreads();
elim = s_elim;
}
if (t < T) {
unsigned avail = cmask;
float fs[8];
int fid[8];
#pragma unroll
for (int r = 7; r >= 0; --r) {
float vs = sv[r];
int vi = id[r];
const int rel = (id[r] - ls) & (NE - 1);
if (rel < 32 && avail != 0u && ((elim >> rel) & 1u)) {
const int c = __ffs(avail) - 1;
avail &= avail - 1u;
#pragma unroll
for (int q = 0; q < 8; ++q)
if (c == q) { vs = sv[8 + q]; vi = id[8 + q]; }
}
fs[r] = vs;
fid[r] = vi;
}
float psum = 0.f;
#pragma unroll
for (int i = 0; i < 8; ++i) psum += fs[i];
const float pinv = rscale / (psum + deps);
int lo[8];
#pragma unroll
for (int k = 0; k < 8; ++k) {
const int rel = (fid[k] - ls) & (NE - 1);
lo[k] = rel < 32 ? rel : -1;
}
float4* twp = reinterpret_cast<float4*>(tw + t * 8);
int4* tip = reinterpret_cast<int4*>(tid + t * 8);
int4* lip = reinterpret_cast<int4*>(lid + t * 8);
twp[0] = make_float4(fs[0] * pinv, fs[1] * pinv, fs[2] * pinv, fs[3] * pinv);
twp[1] = make_float4(fs[4] * pinv, fs[5] * pinv, fs[6] * pinv, fs[7] * pinv);
tip[0] = make_int4(fid[0], fid[1], fid[2], fid[3]);
tip[1] = make_int4(fid[4], fid[5], fid[6], fid[7]);
lip[0] = make_int4(lo[0], lo[1], lo[2], lo[3]);
lip[1] = make_int4(lo[4], lo[5], lo[6], lo[7]);
}
}
"""

HOST = r"""void rp_final4(torch::Tensor rid, torch::Tensor rsv, torch::Tensor rky, torch::Tensor lstart, torch::Tensor tw, torch::Tensor tid,
torch::Tensor lid, double rscale, double deps, double rp_m, int64_t pdl) {
const int T = (int)tw.size(0);
TORCH_CHECK(T >= 1 && T <= 128 && rid.numel() == 16L * T && rsv.numel() == 16L * T && rky.numel() == 16L * T, "rp_final4: shapes");
TORCH_CHECK(tw.is_contiguous() && tid.is_contiguous() && lid.is_contiguous() && tw.numel() == 8L * T && tid.numel() == 8L * T && lid.numel() == 8L * T, "rp_final4: outputs");
TORCH_CHECK(reinterpret_cast<uintptr_t>(tw.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(tid.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(lid.data_ptr()) % 16 == 0, "rp_final4: alignment");
cudaLaunchAttribute at[1];
at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
at[0].val.programmaticStreamSerializationAllowed = (pdl & 1) ? 1 : 0;
cudaLaunchConfig_t lc = {};
lc.gridDim = dim3(1);
lc.blockDim = dim3(32 * ((T + 31) / 32));
lc.dynamicSmemBytes = 0;
lc.stream = at::cuda::getCurrentCUDAStream();
lc.attrs = at;
lc.numAttrs = 1;
auto kern = (pdl & 2) ? &v8::rp_final4_kernel<true> : &v8::rp_final4_kernel<false>;
C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, (const int*)rid.data_ptr<int>(), (const float*)rsv.data_ptr<float>(),
(const int*)rky.data_ptr<int>(), (const int*)lstart.data_ptr<int>(), tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(),
T, (float)rscale, (float)deps, (float)rp_m));
C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:120])
    return s.replace(a, b)


def main():
    s = BASE.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert '"""' not in KERNEL and '"""' not in HOST and '"""' not in DECL4
    # 1. kernel (inside namespace v8, after rp_final2_kernel), declaration, host wrapper
    s = once(s, "}  // namespace v8\nvoid rp_final(torch::Tensor rid", KERNEL + "}  // namespace v8\nvoid rp_final(torch::Tensor rid")
    s = once(s, DECL2, DECL2 + DECL4)
    s = once(s, "void v8_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part) {\n", HOST + "void v8_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part) {\n")
    # 2. exports: both load_inline lists (standalone tail module and the combined k3_v10_all)
    s = once(s, 'functions=["v8_tail","v8_router32","rp_final","rp_final2"],extra_cuda_cflags=["-O3","-lineinfo"],verbose=False)',
             'functions=["v8_tail","v8_router32","rp_final","rp_final2","rp_final4"],extra_cuda_cflags=["-O3","-lineinfo"],verbose=False)')
    s = once(s, '  "v8_tail","v8_router32","rp_final","rp_final2"],\n  extra_cuda_cflags=', '  "v8_tail","v8_router32","rp_final","rp_final2","rp_final4"],\n  extra_cuda_cflags=')
    assert s.count('"rp_final"') == s.count('"rp_final2"') == s.count('"rp_final4"') == 2
    # 3. Python: switches + call (rp_final4 for 32 < tokens <= 128, otherwise the existing rp_final2 / rp_final choice; pdl bit 1 = redundant-elimination variant)
    flag = "RPF2_ON=True  # RPF2: one-warp rp_final2 for T <= 32 (tools/rpf2_patches.py)\n"
    s = once(s, flag, flag + "RPF4_ON=False  # RPF4: four-warp rp_final4 for 32 < T <= 128 (tools/rpf4_patches.py)\n"
             "RPF4_RED=False  # RPF4: True = elimination redundantly in every warp (no second barrier), False = warp 0 + shared broadcast\n")
    s = once(s, '(v8["mod"].rp_final2 if RPF2_ON and tokens<=32 else v8["mod"].rp_final)(',
             '(v8["mod"].rp_final4 if RPF4_ON and 32<tokens<=128 else v8["mod"].rp_final2 if RPF2_ON and tokens<=32 else v8["mod"].rp_final)(')
    s = once(s, "RP_FLAGS_T.get(tokens,0))*(-1.0 if RZC_ON else 1.0),(1 if V9_TAIL_PDL else 0))",
             "RP_FLAGS_T.get(tokens,0))*(-1.0 if RZC_ON else 1.0),(1 if V9_TAIL_PDL else 0)|(2 if RPF4_RED else 0))")
    compile(s, str(OUT), "exec")
    OUT.write_bytes(s.encode("ascii"))
    print("RPF4 lab", OUT, len(s.encode("ascii")), "B")


if __name__ == "__main__":
    main()

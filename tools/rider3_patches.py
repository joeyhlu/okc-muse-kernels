"""rider3_patches.py: three tiny bitwise-safe riders on the b23 base (notes/lab.md R-rider3-0; source: ext_codex_d32.md items 1, 3, 4).

  python -I tools/rider3_patches.py --base=kernels/lab/k3_b23_1.py --out=kernels/lab/k3_rider3_1.py [--r1] [--r2] [--r3]
      (no --r flag = all three; one flag = that rider alone, for per-rider A/B files; never overwrites --out)

R1  vm_fin padded-slot skip (VM_FTR path; the VM_FTR 0 text stays under #else).  vm_fin finalizes tokens in chunks of 4 slots,
    but nloc (valid local routes of the token, 0..8, CTA-uniform: s_n read after a __syncthreads) is usually 1-3: the invalid
    slots held z4 and still ran bf16r * nw, squares, a 5-step shuffle chain each.  The per-chunk work becomes fin_ss<NV> with
    NV = min(nloc - c0, 4) valid slots (a uniform 4-way branch; NV = 4 is the old code).  Every valid slot keeps its exact
    FP32 chain (m = 0..3, x y z w in order, FFMA ss += f * f; butterfly xor 16..1; the 8-warp sum and the `a` accumulation
    are untouched), the slots are independent, and a padded slot's ss was never read (the fr loop is guarded by
    c0 + jj < nloc): bitwise identical.  Barriers stay unconditional; the s_ss store of padded slots is skipped.
R2  sbb_kernel (SBB1_CU) output conversion: eight scalar __float2bfloat16_rn into ob[8] -> four __floats2bfloat162_rn after the
    loop (lo = nz[2i], hi = nz[2i + 1] = the same two cvt.rn.bf16.f32), same 16-byte store.  FP32 math untouched.
R3  vm_fin weight loads: the immutable nw (4 x float4) and the upstream wts (1 float / thread < 8) were `__ldg` (ld.global.nc)
    above griddepcontrol.wait in the source, which ptxas sank below the ACQBULK (tools/pdl_sass.py: PREFETCH-DEAD, 0 / 5
    pre-wait loads).  Now `__ldcg` (ld.global.cg): coherent, stays above the wait.  rmap / y stay __ldcg after the wait.
    wts safety: tail_kernel* writes tw (= topk_weights) in the stream before vm_main; vm_main is launched without the PDL
    attribute (VML_FLAGS bit 256 clear: cm.numAttrs = 0), so it starts after the tail kernel completed; with bit 256 set it
    waits (griddepcontrol.wait) before its only launch_dependents (VM_LD 1 / 2), and VM_LD 0 never triggers (implicit
    trigger = CTA exit).  vm_fin (PDL launch) therefore starts only after every vm_main CTA triggered post-wait or exited,
    i.e. after wts is final.  nw is a prepare()-time weight.  No launch_dependents is moved (T = 32 early trigger stays off).
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}


def check(c, msg):
    if not c:
        raise SystemExit(f"[rider3_patches] FAIL: {msg}")


def sub(t, old, new, what):
    n = t.count(old)
    check(n == 1, f"anchor {what}: found {n}, expected 1")
    return t.replace(old, new)


# ---- R1 --------------------------------------------------------------------------------------------------------------------
FIN_HEAD = "__global__ void __launch_bounds__(256) vm_fin(const Args A) {\n"

FIN_TEMPLATE = r"""#if VM_FTR
// R3-1 (tools/rider3_patches.py): the NV valid slots (jj < NV) of one 4-slot chunk: products once, ss chain, butterfly, lane-0 store of the
// warp sums. Per slot the FP32 operations and their order are the VM_FTR text of vm_fin verbatim; NV = 4 is that text.
template <int NV>
__device__ __forceinline__ void fin_ss(float4 (&v)[4][4], const float4 (&w4)[4], float* srow, int lane) {
#pragma unroll
for (int jj = 0; jj < NV; ++jj)
#pragma unroll
for (int m = 0; m < 4; ++m) {   // VTR: products once
v[jj][m].x = bf16r(v[jj][m].x) * w4[m].x; v[jj][m].y = bf16r(v[jj][m].y) * w4[m].y;
v[jj][m].z = bf16r(v[jj][m].z) * w4[m].z; v[jj][m].w = bf16r(v[jj][m].w) * w4[m].w;
}
float ss[NV];
#pragma unroll
for (int jj = 0; jj < NV; ++jj) {
ss[jj] = 0.f;
#pragma unroll
for (int m = 0; m < 4; ++m) {
float f;
f = v[jj][m].x; ss[jj] += f * f; f = v[jj][m].y; ss[jj] += f * f;
f = v[jj][m].z; ss[jj] += f * f; f = v[jj][m].w; ss[jj] += f * f;
}
#pragma unroll
for (int o = 16; o > 0; o >>= 1) ss[jj] += __shfl_xor_sync(0xffffffffu, ss[jj], o);
}
if (lane == 0) {
#pragma unroll
for (int jj = 0; jj < NV; ++jj) srow[jj] = ss[jj];
}
}
#endif

"""

FIN_OLD = """#if VM_FTR
#pragma unroll
for (int jj = 0; jj < 4; ++jj)
#pragma unroll
for (int m = 0; m < 4; ++m) {   // VTR: products once
v[jj][m].x = bf16r(v[jj][m].x) * w4[m].x; v[jj][m].y = bf16r(v[jj][m].y) * w4[m].y;
v[jj][m].z = bf16r(v[jj][m].z) * w4[m].z; v[jj][m].w = bf16r(v[jj][m].w) * w4[m].w;
}
#endif
float ss[4];
#pragma unroll
for (int jj = 0; jj < 4; ++jj) {
ss[jj] = 0.f;
#pragma unroll
for (int m = 0; m < 4; ++m) {
float f;
#if VM_FTR
f = v[jj][m].x; ss[jj] += f * f; f = v[jj][m].y; ss[jj] += f * f;
f = v[jj][m].z; ss[jj] += f * f; f = v[jj][m].w; ss[jj] += f * f;
#else
f = bf16r(v[jj][m].x) * w4[m].x; ss[jj] += f * f; f = bf16r(v[jj][m].y) * w4[m].y; ss[jj] += f * f;
f = bf16r(v[jj][m].z) * w4[m].z; ss[jj] += f * f; f = bf16r(v[jj][m].w) * w4[m].w; ss[jj] += f * f;
#endif
}
#pragma unroll
for (int o = 16; o > 0; o >>= 1) ss[jj] += __shfl_xor_sync(0xffffffffu, ss[jj], o);
}
if (lane == 0) {
#pragma unroll
for (int jj = 0; jj < 4; ++jj) s_ss[par][warp][jj] = ss[jj];
}
__syncthreads();
"""

FIN_NEW_FTR = """#if VM_FTR
{   // R3-1: nloc is CTA-uniform (s_n, read after a barrier), so this 4-way branch is uniform and the full-mask shuffles stay converged
const int nv = min(nloc - c0, 4);
if (nv >= 4) fin_ss<4>(v, w4, s_ss[par][warp], lane);
else if (nv == 3) fin_ss<3>(v, w4, s_ss[par][warp], lane);
else if (nv == 2) fin_ss<2>(v, w4, s_ss[par][warp], lane);
else fin_ss<1>(v, w4, s_ss[par][warp], lane);
}
__syncthreads();
#else
"""


def r1(t):
    t = sub(t, FIN_HEAD, FIN_TEMPLATE + FIN_HEAD, "vm_fin head")
    return sub(t, FIN_OLD, FIN_NEW_FTR + FIN_OLD + "#endif\n", "vm_fin products + ss block")


# ---- R2 --------------------------------------------------------------------------------------------------------------------
SBB_OLD = """  float ns[8], nz[8];
  __align__(16) __nv_bfloat16 ob[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float n = val[i] * post_rstd;
    if (p.post_w && !kInputScale) n *= pw[i];
    n = sb_rbf(n);
    const float u = fmaf(bt[i], n, res[i]);
    ns[i] = u;
    float z = u * pre_rstd;
    if (p.pre_w) z *= qw[i];
    nz[i] = z;
    ob[i] = __float2bfloat16_rn(z);
  }
  *reinterpret_cast<uint4*>(p.out + (size_t)row * SB_H + e0) = *reinterpret_cast<const uint4*>(ob);
"""
SBB_NEW = """  float ns[8], nz[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float n = val[i] * post_rstd;
    if (p.post_w && !kInputScale) n *= pw[i];
    n = sb_rbf(n);
    const float u = fmaf(bt[i], n, res[i]);
    ns[i] = u;
    float z = u * pre_rstd;
    if (p.pre_w) z *= qw[i];
    nz[i] = z;
  }
  {   // R3-2: four paired conversions (cvt.rn.bf16x2.f32: lo = nz[2i], hi = nz[2i + 1]; each half = __float2bfloat16_rn)
    const __nv_bfloat162 b0 = __floats2bfloat162_rn(nz[0], nz[1]), b1 = __floats2bfloat162_rn(nz[2], nz[3]);
    const __nv_bfloat162 b2 = __floats2bfloat162_rn(nz[4], nz[5]), b3 = __floats2bfloat162_rn(nz[6], nz[7]);
    uint4 q;
    q.x = *reinterpret_cast<const uint32_t*>(&b0); q.y = *reinterpret_cast<const uint32_t*>(&b1);
    q.z = *reinterpret_cast<const uint32_t*>(&b2); q.w = *reinterpret_cast<const uint32_t*>(&b3);
    *reinterpret_cast<uint4*>(p.out + (size_t)row * SB_H + e0) = q;
  }
"""


def r2(t):
    return sub(t, SBB_OLD, SBB_NEW, "sbb_kernel epilogue")


# ---- R3 --------------------------------------------------------------------------------------------------------------------
WTS_OLD = "const float wv = (tid < 8) ? __ldg(A.wts + t * 8 + tid) : 0.f;   // vm_main's input: ready before vm_main ran\n"
WTS_NEW = ("const float wv = (tid < 8) ? __ldcg(A.wts + t * 8 + tid) : 0.f;   // vm_main's input: final before vm_main's trigger; "
           "R3-3: ld.global.cg (coherent) so ptxas keeps it above the wait\n")
NW_OLD = "for (int m = 0; m < 4; ++m) w4[m] = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * m + 4 * tid));\nasm volatile(\"griddepcontrol.wait;"
NW_NEW = "for (int m = 0; m < 4; ++m) w4[m] = __ldcg(reinterpret_cast<const float4*>(A.nw + 1024 * m + 4 * tid));   // R3-3: immutable weight, coherent load above the wait\nasm volatile(\"griddepcontrol.wait;"


def r3(t):
    t = sub(t, WTS_OLD, WTS_NEW, "vm_fin wts load")
    return sub(t, NW_OLD, NW_NEW, "vm_fin nw loads")


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_b23_1.py")
    out = ROOT / ARGS["out"]
    sel = [r for r in ("r1", "r2", "r3") if r in FLAGS] or ["r1", "r2", "r3"]
    check(not out.exists(), f"{out} exists (never overwrite)")
    t = base.read_text(encoding="utf-8")
    check("\r" not in t, "CRLF in base")
    for r, fn in (("r1", r1), ("r2", r2), ("r3", r3)):
        if r in sel:
            t = fn(t)
    lines = t.split("\n")
    check(lines[1].startswith("# nonce "), "line 2 is not the nonce line")
    lines[1] = f"# nonce {out.stem.replace('_', '-')} ({base.stem} + rider3 {'+'.join(sel)}; tools/rider3_patches.py)"
    t = "\n".join(lines)
    check(t.isascii(), "non-ascii output")
    check(len(t.encode()) < 1_000_000, f"size {len(t.encode())}")
    compile(t, str(out), "exec")
    out.write_text(t, encoding="utf-8", newline="\n")
    shown = out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else out.as_posix()
    print(f"[rider3_patches] {base.name} -> {shown} ({'+'.join(sel)}), {len(t.encode())} B")


if __name__ == "__main__":
    main()

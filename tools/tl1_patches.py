"""tl1_patches.py: TL1 = study-b's v8 tail with a concurrent routing warp + redux.sync top-8 picks, as shipping patches
(R-study-b-4; lab generator .claude/worktrees/study-b-06cd73/kernels/sb/gen_tl_lab.py @ 82f34c1, kernel code verbatim).

tail_kernel2 runs the top-8 routing on a 17th warp concurrently with the 16-warp RMSNorm (named barrier 1 instead of
__syncthreads); each pick is one redux.sync.max over an orderable-float key + ballot (lowest lane = lowest expert id on ties,
as tail_kernel's shuffle reduction). Bit-identical to tail_kernel at T = 8 .. 256 including ties (study-b, sm_86); sm_90a
45-46 regs, 0 spill. Selected by bit 1 of the host's pdl argument (TL1_TAIL2, read at call time by _v8_tail); tail_kernel
stays as the fallback. R-study-b-4 (box, 2 router seeds, vs k3_v35_pt1): +0.454 % / +0.513 % six-load (net of control
+0.44 / +0.61 %), checks passed.
Prepare: the v8 tail self-tests (TIE1-tolerant) run through _v8_tail, so they exercise tail_kernel2 when TL1_TAIL2 is on;
TL1 then also requires tail_kernel2 == tail_kernel BITWISE (xn, weights, ids, local ids) on six fresh shapes of the 'cat'
path and turns TL1_TAIL2 off (tail_kernel) on any difference or error.

    python -I tools/tl1_patches.py in.py out.py
"""
import pathlib
import py_compile
import sys

TAIL2 = r'''
// TL1 (tools/tl1_patches.py, study-b R-study-b-4): tail_kernel with the routing on a 17th warp, concurrent with the 16-warp
// norm (named barrier 1), and each top-8 pick by one redux.sync (orderable-float key, lowest lane = lowest expert id on ties,
// as the shuffle reduction): the same outputs as tail_kernel.
template <bool Y32, int NSMAX>
__global__ void __launch_bounds__(NT + 32) tail_kernel2(
    const void* __restrict__ y, long y_ld, const float* __restrict__ la, const float* __restrict__ lb, long l_ld,
    int ns, long sstride, const __nv_bfloat16* __restrict__ nw, const float* __restrict__ bias, const int* __restrict__ lstart,
    __nv_bfloat16* __restrict__ xn, float* __restrict__ tw, int* __restrict__ tid, int* __restrict__ lid,
    float eps, float mult, float rscale, float deps) {
  const int t = blockIdx.x, tx = threadIdx.x, lane = tx & 31, warp = tx >> 5;
  __shared__ float red[NT / 32];
  if (warp == NT / 32) {
    // routing warp: lane l holds experts 8l .. 8l+7
    const float4* pbias = reinterpret_cast<const float4*>(bias) + lane * 2;
    const float4 c0 = pbias[0], c1 = pbias[1];
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");   // v4.7: K2A's K1 may launch now
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    float4 qa[NSMAX][2];
    float4 qb0 = make_float4(0.f, 0.f, 0.f, 0.f), qb1 = qb0;
#pragma unroll
    for (int q = 0; q < NSMAX; ++q) {
      if (q < ns) {
        const float4* pq = reinterpret_cast<const float4*>(la + q * sstride + (long)t * l_ld) + lane * 2;
        qa[q][0] = pq[0];
        qa[q][1] = pq[1];
      } else {
        qa[q][0] = make_float4(0.f, 0.f, 0.f, 0.f);
        qa[q][1] = qa[q][0];
      }
    }
    if (lb != nullptr) {
      const float4* pb = reinterpret_cast<const float4*>(lb + (long)t * l_ld) + lane * 2;
      qb0 = pb[0];
      qb1 = pb[1];
    }
    float s[8], b[8];
    float4 a0 = qa[0][0], a1 = qa[0][1];
#pragma unroll
    for (int q = 1; q < NSMAX; ++q) {   // FP32 split-K partials summed in a fixed order (zeros past ns)
      a0.x += qa[q][0].x; a0.y += qa[q][0].y; a0.z += qa[q][0].z; a0.w += qa[q][0].w;
      a1.x += qa[q][1].x; a1.y += qa[q][1].y; a1.z += qa[q][1].z; a1.w += qa[q][1].w;
    }
    a0.x += qb0.x; a0.y += qb0.y; a0.z += qb0.z; a0.w += qb0.w; a1.x += qb1.x; a1.y += qb1.y; a1.z += qb1.z; a1.w += qb1.w;
    const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j] = 1.f / (1.f + expf(-lg[j] * mult));
      b[j] = s[j] + bs[j];
    }
    int sel_id[TK];
    float sel_s[TK];
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      float bv = b[0], sv = s[0];
      int bj = 0;
#pragma unroll
      for (int j = 1; j < 8; ++j)
        if (b[j] > bv) { bv = b[j]; sv = s[j]; bj = j; }
      const unsigned u = __float_as_uint(bv + 0.f);   // -0 -> +0: equal floats get equal keys
      const unsigned key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
      const unsigned mx = __reduce_max_sync(0xffffffffu, key);
      const int owner = __ffs(__ballot_sync(0xffffffffu, key == mx)) - 1;
      sel_id[k] = __shfl_sync(0xffffffffu, lane * 8 + bj, owner);
      sel_s[k] = __shfl_sync(0xffffffffu, sv, owner);
      if (lane == owner) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (j == bj) b[j] = -INFINITY;
      }
    }
    if (lane == 0) {
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < TK; ++i) sum += sel_s[i];
      const float inv = rscale / (sum + deps);
      const int ls = lstart[0];
#pragma unroll
      for (int k = 0; k < TK; ++k) {
        tw[t * TK + k] = sel_s[k] * inv;
        tid[t * TK + k] = sel_id[k];
        const int rel = ((sel_id[k] - ls) % NE + NE) % NE;
        lid[t * TK + k] = rel < 32 ? rel : -1;
      }
    }
    return;
  }
  // norm warps 0..15 (8 elements of the row each)
  const uint4 wq = *(reinterpret_cast<const uint4*>(nw) + tx);
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  float v[8];
  if (Y32) {
    const float4* p = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(y) + (long)t * y_ld) + tx * 2;
    const float4 a = p[0], c = p[1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = c.x; v[5] = c.y; v[6] = c.z; v[7] = c.w;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(__float2bfloat16(v[i]));
  } else {
    const uint4 q = *(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(y) + (long)t * y_ld) + tx);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(h[i]);
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) red[warp] = ss;
  asm volatile("bar.sync 1, %0;\n" ::"r"(NT) : "memory");   // the 16 norm warps only
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; ++i) tot += red[i];
  const float rstd = rsqrtf(tot / (float)D + eps);
  const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wq);
  __align__(16) __nv_bfloat16 o8[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16((v[i] * rstd) * __bfloat162float(w[i]));
  *(reinterpret_cast<uint4*>(xn + (long)t * D) + tx) = *reinterpret_cast<const uint4*>(o8);
}
'''

PY = '''

# TL1 (tools/tl1_patches.py, study-b R-study-b-4): v8 tail with a concurrent routing warp + redux picks
TL1_TAIL2 = True   # tail_kernel2 (host pdl bit 1); read at call time by _v8_tail; prepare turns it off on a bitwise mismatch

'''

# inserted before the tail-ready print of _v8_tail_prepare (after its self-tests passed)
CHECK = '''    global TL1_TAIL2
    if TL1_TAIL2:
        try:
            gen2 = torch.Generator(device=dev).manual_seed(4343)
            same = True
            for T, ls in ((1, 0), (8, 64), (16, 224), (33, 32), (128, 96), (256, 160)):
                x = (torch.randn((T, HIDDEN), generator=gen2, device=dev) * 0.5).to(torch.bfloat16)
                d = {"local_start": torch.tensor([ls], dtype=torch.int32, device=dev)}
                y = torch.mm(x, v8["wcat"].t(), out_dtype=torch.float32)
                lg2 = (y[:, D_EXPERT:D_EXPERT + N_EXPERTS] + y[:, D_EXPERT + N_EXPERTS:]).contiguous()
                part2 = (lg2.unsqueeze(0).repeat(8, 1, 1) * 0.125).contiguous()
                pool2 = y[:, :D_EXPERT].to(torch.bfloat16).contiguous()
                for yy, ll in ((y, None), (y, part2), (pool2, lg2), (pool2, part2)):   # tail_kernel<Y32, 1 / 8> variants
                    outs = []
                    for flag in (True, False):
                        TL1_TAIL2 = flag
                        outs.append([a.clone() for a in _v8_tail(yy, ll, {"v8": v8}, d)])
                    torch.cuda.synchronize()
                    same = same and all(bool(torch.equal(a, b)) for a, b in zip(outs[0], outs[1]))
            TL1_TAIL2 = same
            print(f"[tl1] tail_kernel2 == tail_kernel bitwise on 6 shapes x 4 variants: {same} -> "
                  f"{'tail_kernel2' if same else 'tail_kernel'}", flush=True)
        except Exception as ex:
            TL1_TAIL2 = False
            print(f"[tl1] tail_kernel2 check raised -> tail_kernel: {ex!r}", flush=True)
'''

READY = '    print(f"[v8] tail ready in {time.time() - t0:.1f}s; route {V8_TAIL_ROUTE}", flush=True)\n'
TAIL_CALL = ("                      RMS_EPS, 0.1767766952966369, 1.0, 1e-15, 1 if V9_TAIL_PDL else 0)\n"
             "    return xn, tw, tid, lid\n")


def _fail(msg):
    raise RuntimeError(f"tl1 patch: {msg}")


def _cuda_region(src):
    a = src.index('V8_TAIL_CUDA_SRC = r"""')
    return a, src.index('"""', a + len('V8_TAIL_CUDA_SRC = r"""'))


def apply(src):
    if src.count('V8_TAIL_CUDA_SRC = r"""') != 1:
        _fail("V8_TAIL_CUDA_SRC not unique")
    a, b = _cuda_region(src)
    k0 = src.find("template <bool Y32, int NSMAX>   // NSMAX: logits partial slices held", a, b)
    k = src.find("__global__ void __launch_bounds__(NT) tail_kernel(", k0, b)
    if k0 < 0 or k < 0:
        _fail("tail_kernel not found in V8_TAIL_CUDA_SRC")
    k1 = src.index("\n}\n", k) + 3
    src = src[:k1] + TAIL2 + src[k1:]
    edits = [("at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;",
              "at[0].val.programmaticStreamSerializationAllowed = (pdl & 1) ? 1 : 0;"),
             ("lc.blockDim = dim3(v8::NT);", "lc.blockDim = dim3((pdl & 2) ? v8::NT + 32 : v8::NT);   // TL1")]
    for y32, nsm in (("true", 8), ("true", 1), ("false", 8), ("false", 1)):
        edits.append((f"cudaLaunchKernelEx(&lc, v8::tail_kernel<{y32}, {nsm}>,",
                      f"cudaLaunchKernelEx(&lc, ((pdl & 2) ? v8::tail_kernel2<{y32}, {nsm}> : v8::tail_kernel<{y32}, {nsm}>),"))
    for old, new in edits:
        a, b = _cuda_region(src)
        if src[a:b].count(old) != 1:
            _fail(f"host anchor x{src[a:b].count(old)}: {old!r}")
        src = src[:a] + src[a:b].replace(old, new) + src[b:]
    if src.count(TAIL_CALL) != 1:
        _fail("_v8_tail call anchor")
    src = src.replace(TAIL_CALL, "                      RMS_EPS, 0.1767766952966369, 1.0, 1e-15, (1 if V9_TAIL_PDL else 0) | "
                                 "(2 if TL1_TAIL2 else 0))\n    return xn, tw, tid, lid\n")
    r0 = src.index("def _v8_tail_prepare(weights, state):\n")
    r1 = src.index("\nV8_K2A3_ROUTE = ", r0)
    if src[r0:r1].count(READY) != 1:
        _fail("tail-ready print anchor")
    src = src[:r0] + src[r0:r1].replace(READY, CHECK + READY) + src[r1:]
    if src.count("\ndef kernel_fn(data, state):\n") != 1:
        _fail("kernel_fn anchor")
    return src.replace("\ndef kernel_fn(data, state):\n", PY + "\ndef kernel_fn(data, state):\n")


if __name__ == "__main__":
    s, d = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    t = s.read_text(encoding="utf-8")
    out = apply(t)
    d.write_text(out, encoding="utf-8", newline="\n")
    py_compile.compile(str(d), doraise=True)
    print(f"tl1: applied to {s.name} -> {d} ({len(out) - len(t):+d} chars, compiles)")

"""S7 DS-PDL experiment: make s6o4 combine_e a PDL-compatible kernel (no cooperative launch / grid barrier).
Every CTA derives the per-token scalars itself (warp w reduces token w, same slot order and fp32 reduction tree as the
validated version: bit-identical scalars), waits on the primary (extended_o) with griddepcontrol.wait and triggers its
dependents at once, so tail / MoE / unpooler can launch early like in the base chain.
usage: python -I tools/s7_combine_pdl.py <in.py> <out.py> <old-nonce> <new-nonce>"""
import ast, sys, pathlib

NEW_KERNEL = r'''__global__ void __launch_bounds__(256, 2) combine_e(const __grid_constant__ Combine A) {
  __shared__ float sc[8][2];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  // PDL: outputs of the primary (extended_o: moments, kg, a) are read only after the wait.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  // No grid barrier: every CTA reduces the scalars of all M <= 8 tokens (warp w -> token w). Per token the 128 slots are
  // reduced exactly like before (4 groups of 32 lanes, xor-shuffle tree, groups summed in order 0..3).
  if (warp < A.M) {
    float aa = 0.f, sa = 0.f, bb = 0.f, ss = 0.f;
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int s = g * 32 + lane;
      const float* p = A.mom + ((size_t)warp * 128 + s) * 3;
      float a0 = p[0], s0 = p[1], b0 = p[2], q0 = A.ss[warp * 128 + s];
      #pragma unroll
      for (int d = 16; d; d >>= 1) {
        a0 += __shfl_xor_sync(0xffffffffu, a0, d); s0 += __shfl_xor_sync(0xffffffffu, s0, d);
        b0 += __shfl_xor_sync(0xffffffffu, b0, d); q0 += __shfl_xor_sync(0xffffffffu, q0, d);
      }
      aa += a0; sa += s0; bb += b0; ss += q0;
    }
    if (lane == 0) {
      float ca = rsqrtf(__fadd_rn(aa / 8192.f, 1e-8f));
      float n = __fadd_rn(__fadd_rn(ss, __fmul_rn(2.f * ca, sa)), __fmul_rn(__fmul_rn(ca, ca), bb));
      // A negative/nonfinite norm fails numerics visibly; never silently clamp it.
      sc[warp][0] = ca; sc[warp][1] = rsqrtf(__fadd_rn(n / 8192.f, 1e-5f));
    }
  }
  __syncthreads();
  for (int idx = b * 256 + tid; idx < A.M * 8192; idx += gridDim.x * 256) {
    int t = idx / 8192, c = idx % 8192; float ca = sc[t][0];
    A.stream[idx] = __fadd_rn(A.as[idx], __fmul_rn(ca, __fmul_rn(A.beta[c], A.a[idx])));
  }
  for (int idx = b * 256 + tid; idx < A.M * 4352; idx += gridDim.x * 256) {
    int t = idx / 4352, c = idx % 4352; float ca = sc[t][0], c1 = sc[t][1], v;
    if (c < 4096) {
      v = __fmul_rn(c1, __fadd_rn(A.pu[t * 4096 + c], __fmul_rn(ca, A.kg[idx])));
      A.q[t * 4096 + c] = __float2bfloat16_rn(v);
    } else {
      float r = 0.f;
      #pragma unroll
      for (int k = 0; k < 8; ++k) r += A.ru[(k * A.M + t) * 256 + c - 4096];
      v = __fmul_rn(c1, __fadd_rn(r, __fmul_rn(ca, A.kg[idx])));
      A.logits[t * 256 + c - 4096] = v;
    }
  }
}
'''
OLD_LAUNCH = ("    void* args[]={&a};\n    int e=(int)cudaLaunchCooperativeKernel((void*)combine_e,dim3(32),dim3(256),args,0,st);\n"
              "    if(e) return e;\n")
NEW_LAUNCH = ("    cudaLaunchConfig_t cfg={}; cfg.gridDim=dim3(32); cfg.blockDim=dim3(256); cfg.stream=st;\n"
              "    cudaLaunchAttribute attr={}; attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;\n"
              "    attr.val.programmaticStreamSerializationAllowed=1; cfg.attrs=&attr; cfg.numAttrs=1;\n"
              "    int e=(int)cudaLaunchKernelEx(&cfg,combine_e,a);\n    if(e) return e;\n")

def main():
    src_p, dst_p, old_nonce, new_nonce = sys.argv[1:5]
    src = pathlib.Path(src_p).read_bytes().decode('utf-8')
    assert '\r' not in src
    lines = src.split('\n')
    idx = [i for i, l in enumerate(lines) if l.startswith('_S6O4_EXTRA=')]
    assert len(idx) == 1, idx
    cu = ast.literal_eval(lines[idx[0]][len('_S6O4_EXTRA='):])
    a = cu.index('__global__ void __launch_bounds__(256, 2) combine_e(')
    b = cu.index('} // namespace s6o4')
    cu2 = cu[:a] + NEW_KERNEL + cu[b:]
    assert OLD_LAUNCH in cu2, 'launch text not found'
    cu2 = cu2.replace(OLD_LAUNCH, NEW_LAUNCH)
    assert 'cudaLaunchCooperativeKernel' not in cu2 and 'this_grid().sync()' not in cu2
    lines[idx[0]] = '_S6O4_EXTRA=' + repr(cu2)
    assert old_nonce.encode() in lines[1].encode()
    lines[1] = lines[1].replace(old_nonce, new_nonce, 1)
    out = '\n'.join(lines)
    compile(out, 'patched', 'exec')
    pathlib.Path(dst_p).write_bytes(out.encode('utf-8'))
    print('wrote', dst_p, len(out.encode('utf-8')), 'bytes')

main()

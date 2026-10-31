"""rsp_patches.py (R-rsp-1): RSP step 1 = the FP32 router weights R (8 MB) prefetched into L2 on the side stream right after
the qkv projection, so the T <= 8 router (vf_router32 at T = 1, the fused GD pooler + router at T = 2..8) reads R from L2
instead of competing with the pooler for HBM. notes/study/indep_review_oct31.md s2.

  python -I tools/rsp_patches.py [--base=kernels/lab/k3_hh_3rr.py] [--out=kernels/lab/k3_rsp_1r.py]
  python -I tools/slim_patches.py --base=kernels/lab/k3_rsp_1r.py --out=kernels/lab/k3_rsp_1.py      (LAST)
  python tools/lead_gate.py file kernels/lab/k3_rsp_1.py

Defaults change nothing (RSP_PF = False): the new kernel is compiled into the SBB1 module but never launched. A/B arms set
RSP_PF (and RSP_PF_HINT: 0 = plain prefetch, 1 = L2::evict_last policy) through layer_ab2's module-global switches.
The prefetch kernel is fire-and-forget (cp.async.bulk.prefetch.L2, one thread per 64 KB chunk, no smem, no PDL); its node
depends on the qkv node only; the side stream is joined back to the main stream before kernel_fn returns.
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)


def sub(text, old, new, count=1):
    n = text.count(old)
    if n != count:
        sys.exit(f"rsp_patches: anchor found {n}x (want {count}): {old[:90]!r}")
    return text.replace(old, new)


KERNEL = r'''
// RSP step 1 (R-rsp-1): R -> L2 on the side stream after qkv. One thread per chunk; prefetch is a hint, nothing waits on it.
__global__ void rsp_pf_kernel(const uint8_t* __restrict__ p, unsigned chunk, unsigned long long total, int hint) {
  if (threadIdx.x != 0) return;
  const unsigned long long off = (unsigned long long)blockIdx.x * chunk;
  if (off >= total) return;
  const unsigned n = (unsigned)((total - off) < chunk ? (total - off) : chunk);
  if (hint) {
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;\n" :: "l"(p + off), "r"(n), "l"(pol) : "memory");
  } else {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(p + off), "r"(n) : "memory");
  }
}

void rsp_pf(torch::Tensor w, int64_t chunk, int64_t hint) {
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && chunk > 0 &&
              chunk % 16 == 0 && chunk <= (1 << 20), "rsp_pf: contiguous 16-B aligned tensor, chunk 16..1M, multiple of 16");
  const unsigned long long total = (unsigned long long)w.numel() * (unsigned long long)w.element_size();
  const int grid = (int)((total + (unsigned long long)chunk - 1) / (unsigned long long)chunk);
  rsp_pf_kernel<<<grid, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint8_t*>(w.data_ptr()),
                                                                     (unsigned)chunk, total, (int)hint);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "rsp_pf: launch failed");
}
'''

PY = '''
RSP_PF = False         # R-rsp-1 arm switch: prefetch the FP32 router weights into L2 after qkv (T <= RSP_PF_MAX_T)
RSP_PF_HINT = 0        # 0 = plain cp.async.bulk.prefetch.L2, 1 = with an L2::evict_last policy
RSP_PF_CHUNK = 65536
RSP_PF_MAX_T = 8


def _rsp_pf_launch(state, tokens):
    mod = _SBB1.get("mod")
    if not RSP_PF or tokens > RSP_PF_MAX_T or mod is None or not hasattr(mod, "rsp_pf"):
        return False
    w, side = state.get("router"), state.get("side_stream")
    if w is None or side is None or w.dtype != torch.float32:
        return False
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        mod.rsp_pf(w, RSP_PF_CHUNK, RSP_PF_HINT)
    return True


'''


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3rr.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_rsp_1r.py")
    s = base.read_bytes().decode("ascii")
    lines = s.split(NL)
    lines[1] = "# nonce k3-rsp-1r (k3_hh_3rr + R-rsp-1 router R -> L2 prefetch arm, default off; tools/rsp_patches.py)"
    s = NL.join(lines)
    end = 'TORCH_CHECK(cudaGetLastError() == cudaSuccess, "vf_router32: launch failed");' + NL + "}" + NL
    s = sub(s, end, end + KERNEL)
    s = sub(s, "int64_t kind);'" + NL, "int64_t kind); void rsp_pf(torch::Tensor w, int64_t chunk, int64_t hint);'" + NL)
    s = sub(s, 'functions=["sbb1_run", "vf_router32"]', 'functions=["sbb1_run", "vf_router32", "rsp_pf"]')
    s = sub(s, NL + "def kernel_fn(data, state):" + NL, NL + PY.rstrip(NL) + NL + NL + NL + "def kernel_fn(data, state):" + NL)
    s = sub(s, "    o = _hya_try(qkv, data, cfg, state, True)" + NL,
            "    rsp_side = _rsp_pf_launch(state, tokens)" + NL + "    o = _hya_try(qkv, data, cfg, state, True)" + NL)
    s = sub(s, "    return next_attn_input, stream, topk_ids" + NL,
            "    if rsp_side:" + NL + "        torch.cuda.current_stream().wait_stream(state[\"side_stream\"])" + NL
            + "    return next_attn_input, stream, topk_ids" + NL)
    ast.parse(s)
    out.write_bytes(s.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"rsp_patches: {base.name} -> {out.name} ({len(s):,} B)")


if __name__ == "__main__":
    main()

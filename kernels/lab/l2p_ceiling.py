"""l2p_ceiling.py: step-0 of the in-call L2 prefetch family (notes/study/l2p.md; system_gap_oct28.md S4; fable_review change 1).

For every weight consumer of the shipped layer file, time ONE call (CUDA graph replay, judge-style 256 MB L2 flush before)
with its weights cold vs L2-warm, four ways:
  cold  flush -> replay                                   (the judge's condition)
  rep   replay -> replay (timed)                          (everything the call touched is in L2: the warm ceiling)
  ldg   flush -> plain ld.global.nc sweep of W -> replay  (W warm by ordinary loads, evict_normal)
  pf    flush -> distributed cp.async.bulk.prefetch.L2 of W (264 CTAs x 32 lanes, 64 KB pieces) -> replay
  pff   bulk prefetch of W -> flush -> replay             (RULE CHECK: must equal cold; warm lines must not survive the flush)
Buffers above --cap MB warm only their first --cap MB (the GD stream is step-major, so that is its consumption-order head).
Consumers: every live GD route (o / unpooler / pooler / qkv at T = 1 / 8 / 16 / 32, the unpooler slice via _gd_unpool) and the
T = 128 / 256 cuBLAS GEMMs the layer calls (_v12_gemm for qkv / o / unpooler, torch.mm for cat). Also timed: the prefetch and
sweep kernels themselves (issue cost; the R-vms-9 trap was one CTA issuing 14 MB).
The in-layer bound of a GD consumer = (cold - warm) x (1 - ST / KU): the first ST register stages are already loaded before
griddepcontrol.wait in the layer, so only the post-wait bytes can gain.

Run on the box from /root/okc (one process, one prepare, ~2 min after prepare):
    python3 -u kernels/lab/l2p_ceiling.py kernels/muse_decode_layer/k3_slim_5.py --out=box_out/l2p1/ceiling.json [--rounds=15] [--cap=32]
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "judge"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "box"))
import judge as J  # noqa: E402
from layer_budget import FIELDS  # noqa: E402

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void l2p_sweep_kernel(const uint4* __restrict__ p, unsigned long long n16, unsigned* sink) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long st = (unsigned long long)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (; i < n16; i += st) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
    acc ^= v.x ^ v.w;
  }
  if (acc == 0xFFFFFFFFu) *sink = acc;
}

__global__ void l2p_pf_kernel(const unsigned char* p, unsigned long long bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const unsigned long long n = gridDim.x, c = blockIdx.x;
  const unsigned long long b0 = (bytes * c / n) & ~15ull;
  const unsigned long long b1 = (c + 1 == n) ? (bytes & ~15ull) : ((bytes * (c + 1) / n) & ~15ull);
  for (unsigned long long o = b0 + (unsigned long long)threadIdx.x * 65536ull; o < b1;
       o += (unsigned long long)blockDim.x * 65536ull) {
    const unsigned sz = (unsigned)((b1 - o) < 65536ull ? (b1 - o) : 65536ull);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(p + o), "r"(sz) : "memory");
  }
#endif
}

void l2p_sweep(torch::Tensor w, int64_t bytes) {
  const unsigned long long n16 = (unsigned long long)bytes / 16ull;
  static unsigned* sink = nullptr;
  if (!sink) cudaMalloc(&sink, 4);
  l2p_sweep_kernel<<<1056, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(w.data_ptr()), n16, sink);
}

void l2p_pf(torch::Tensor w, int64_t bytes, int64_t nblk, int64_t nthr) {
  l2p_pf_kernel<<<(unsigned)nblk, (unsigned)nthr, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const unsigned char*>(w.data_ptr()), (unsigned long long)bytes);
}
'''


def arg(name, default):
    for a in sys.argv[2:]:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


def nbytes(t):
    return t.numel() * t.element_size()


class Timer:
    def __init__(self):
        self.stream = torch.cuda.Stream()
        self.flush_buf = torch.ones(J.L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
        self.sink = torch.empty((), dtype=torch.float32, device="cuda")

    def flush(self):
        torch.sum(self.flush_buf, dim=0, out=self.sink)

    def capture(self, fn):
        with torch.cuda.stream(self.stream):
            fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=self.stream):
                fn()
        torch.cuda.synchronize()
        return g

    def timed(self, fn):
        """fn runs on self.stream between two events; returns us."""
        with torch.cuda.stream(self.stream):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1e3

    def overhead(self, n=100):
        tiny = torch.zeros(1, device="cuda")
        g = self.capture(lambda: tiny.add_(1))
        return statistics.median(self.timed(g.replay) for _ in range(n))


def main():
    path = Path(sys.argv[1])
    rounds, cap_mb, out = arg("rounds", 15), arg("cap", 32), arg("out", "")
    cap = cap_mb << 20
    prob = J.load_problem(Path("problems/muse_decode_layer"))
    ref = prob.ref
    for b in prob.benches:
        b.update(FIELDS[b["load"]])
    weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda", router_seed=8677534258059892491)
    mod = J.load_kernel(path, "okc_submission", prob)
    t0 = time.time()
    state = mod.prepare(J.clone_data(weights))
    torch.cuda.synchronize()
    print(f"[l2p] prepare took {time.time() - t0:.0f} s", flush=True)
    from torch.utils.cpp_extension import load_inline
    cpp = ("#include <torch/extension.h>\nvoid l2p_sweep(torch::Tensor w, int64_t bytes);\n"
           "void l2p_pf(torch::Tensor w, int64_t bytes, int64_t nblk, int64_t nthr);\n")
    ext = load_inline(name="l2p_ceiling_ext", cpp_sources=cpp, cuda_sources=CUDA_SRC, functions=["l2p_sweep", "l2p_pf"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    tm = Timer()
    ov = tm.overhead()
    print(f"[l2p] replay overhead {ov:.2f} us; rounds {rounds}; warm cap {cap_mb} MB", flush=True)
    dense = state["dense"]

    # ---- consumers: (name, T, call, [weight tensors], prewait_frac, desc)
    cons = []
    gd = state.get("gd") or {}
    for (proj, b), r in sorted((gd.get("route") or {}).items(), key=str):
        T = b
        K = r["K"]
        x = (torch.randn(T, K, device="cuda") * 0.05).to(torch.bfloat16)
        rt = mod.GD_ROUTE[(proj, b)]
        prewait = rt[4] / rt[3]
        if proj == "unpooler" and r["slice"]:
            outb = torch.zeros(T, mod.HIDDEN, dtype=torch.bfloat16, device="cuda")

            def call(x=x, outb=outb, T=T):
                y = mod._gd_unpool(state, x, T)
                if y is None:
                    y = mod._gd(state, "unpooler", x, outb, mod.HIDDEN)
                assert y is not None, "unpooler slice route declined"
                return y
        else:
            def call(x=x, proj=proj):
                y = mod._gd(state, proj, x)
                assert y is not None, f"{proj} route declined"
                return y
        cons.append((f"gd {proj}/{b}", T, call, [r["w"], r["esc"]], prewait, r["desc"]))
    for T in (128, 256):
        for proj, wname, tag in (("qkv", "qkv_gate", "qkv"), ("o", "o", "o"), ("unpooler", "unpooler", "unpool")):
            w = dense[wname]
            x = (torch.randn(T, w.shape[1], device="cuda") * 0.05).to(torch.bfloat16)
            cons.append((f"cublas {proj}/{T}", T, (lambda x=x, w=w, tag=tag, T=T: mod._v12_gemm(x, w, tag, T)), [w], 0.0,
                         f"{tuple(w.shape)} bf16"))
        wcat = state["v8"]["wcat"]
        xc = (torch.randn(T, wcat.shape[1], device="cuda") * 0.05).to(torch.bfloat16)
        cons.append((f"cublas cat/{T}", T, (lambda xc=xc, wcat=wcat: torch.mm(xc, wcat.t(), out_dtype=torch.float32)), [wcat],
                     0.0, f"{tuple(wcat.shape)} bf16 -> f32"))

    results = []
    for name, T, call, ws, prewait, desc in cons:
        try:
            g = tm.capture(call)
        except Exception as ex:
            print(f"[l2p] {name}: capture failed: {ex!r}", flush=True)
            continue
        wb = [(w, min(nbytes(w), cap) & ~15) for w in ws]
        tot = sum(nbytes(w) for w in ws)
        warmed = sum(b for _, b in wb)

        def sweep():
            for w, b in wb:
                ext.l2p_sweep(w, b)

        def pf():
            for w, b in wb:
                ext.l2p_pf(w, b, 264, 32)

        t = {k: [] for k in ("cold", "rep", "ldg", "pf", "pff", "t_pf", "t_ldg")}
        for _ in range(rounds):
            with torch.cuda.stream(tm.stream):
                tm.flush()
            t["cold"].append(tm.timed(g.replay) - ov)
            with torch.cuda.stream(tm.stream):
                g.replay()
            t["rep"].append(tm.timed(g.replay) - ov)
            with torch.cuda.stream(tm.stream):
                tm.flush()
            t["t_ldg"].append(tm.timed(sweep))
            t["ldg"].append(tm.timed(g.replay) - ov)
            with torch.cuda.stream(tm.stream):
                tm.flush()
            t["t_pf"].append(tm.timed(pf))
            torch.cuda.synchronize()
            t["pf"].append(tm.timed(g.replay) - ov)
            with torch.cuda.stream(tm.stream):
                pf()
                tm.flush()
            t["pff"].append(tm.timed(g.replay) - ov)
        m = {k: statistics.median(v) for k, v in t.items()}
        gain = m["cold"] - min(m["rep"], m["ldg"], m["pf"])
        row = dict(name=name, T=T, desc=desc, bytes=tot, warmed=warmed, prewait=prewait, med=m, raw=t,
                   gain_us=gain, gain_us_per_mb=gain / max(warmed / 2**20, 1e-9), inlayer_bound_us=gain * (1 - prewait))
        results.append(row)
        print(f"[l2p] {name:>18s} {desc:>24s} {tot / 2**20:6.1f} MB (warm {warmed / 2**20:5.1f}): cold {m['cold']:6.2f}  "
              f"rep {m['rep']:6.2f}  ldg {m['ldg']:6.2f}  pf {m['pf']:6.2f}  pff {m['pff']:6.2f} us | gain {gain:5.2f} us "
              f"= {row['gain_us_per_mb']:.3f} us/MB, prewait {prewait:.2f} -> in-layer bound {row['inlayer_bound_us']:5.2f} us "
              f"| issue: pf {m['t_pf']:5.1f} us, sweep {m['t_ldg']:5.1f} us", flush=True)
        del g
        torch.cuda.synchronize()
    if out:
        dst = Path(out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(dict(module=str(path), overhead_us=ov, rounds=rounds, cap_mb=cap_mb, results=results),
                                  indent=1, default=str))
        print(f"[l2p] wrote {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

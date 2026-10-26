"""T128-only LAB: incumbent HK3 K1 -> one mixed K2 -> incumbent K3.

prepare(weights, incumbent_module, incumbent_state); run(data, state) -> BF16.
Requires the exact v34 T128 route: vid5, hot=(8,0,12), dyn=(8,0,3).
No fallback. No GPU work on import. --source-audit runs without torch/CUDA.

K2 fixes each CTA's role at entry. Experts >=32 use the existing K2A WGMMA
consumer at token64/output128/K128 with a direct BF16 inter->shared-core pack.
Others keep the incumbent cold_item/hc_item math, 8/32-token pass handling and
mailbox. Both write the original FP32 y/ss; only incumbent K3 combines experts.
Hot weights are a lossless byte permutation of incumbent ws2. Its scaled
FP32 gsc2 (raw global *256) remains correct; no Humming numeric conversion.

Source generation reads/copies incumbent source; it never edits loaded modules.
The incumbent workspace must correspond to these weights and may not be used
concurrently. All routing, hot shared packing and finalization are timed.
GPU compile/numerics/resources must be gated by the caller; no speed claim.
"""

import ast
import hashlib
import json
import re
from pathlib import Path

T = 128
THRESHOLD = 32
HOT_SW = 9216
HOT_STG = HOT_SW + 64 * 128 * 2
HOT_NST = 3
COLD_SMEM = 3 * (18432 + 32 * 528) + (6 + 16) * 8
HOT_SMEM = HOT_NST * HOT_STG + 2 * HOT_NST * 8
SMEM = max(COLD_SMEM, HOT_SMEM)
_PRE_OUT = ("ws13", "ws2", "gsc13", "gsc2", "nw", "h", "inter", "y", "meta", "cnt", "err")
_POST_OUT = ("sink", "ss", "cnt2", "xw", "xw2")


def _once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("hotW2 source incompatibility: " + old[:100])
    return text.replace(old, new, 1)


def _closing(text, left):
    """Matching C++ brace, ignoring comments and quoted inline PTX strings."""
    depth, i, mode = 1, left + 1, None
    while i < len(text):
        c, pair = text[i], text[i:i + 2]
        if mode in ('"', "'"):
            if c == "\\":
                i += 2
                continue
            if c == mode:
                mode = None
        elif mode == "//":
            if c == "\n": mode = None
        elif mode == "/*":
            if pair == "*/": mode = None; i += 1
        elif pair in ("//", "/*"):
            mode = pair; i += 1
        elif c in ('"', "'"):
            mode = c
        elif c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0: return i
        i += 1
    raise RuntimeError("unterminated C++ source block")


def _function(source, marker, template=False):
    start = source.index(marker)
    if template:
        start = source.rfind("template <", 0, start)
    left = source.index("{", source.index(marker))
    return source[start:_closing(source, left) + 1]


def _base(module=None):
    if module is not None:
        return module.K2A33_KERNEL_SRC, module._k2a33_wg_specs()
    path = Path(__file__).resolve().parents[1] / "muse_decode_layer" / "k3_v34.py"
    tree, ns = ast.parse(path.read_text(encoding="utf-8")), {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "K2A33_KERNEL_SRC" for t in node.targets):
            ns["K2A33_KERNEL_SRC"] = ast.literal_eval(node.value)
        elif isinstance(node, ast.FunctionDef) and node.name == "_k2a33_wg_specs":
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    return ns["K2A33_KERNEL_SRC"], ns["_k2a33_wg_specs"]()


_PRODUCER = r"""
  if (warp == 8) {
    // All inter loads wait for the unchanged K1. Immutable weight copies need no
    // data-dependent host synchronization. This first port waits before both.
    pdl_wait();
    asm volatile("fence.proxy.async;\n" ::: "memory");
    int it = 0;
    const int mid = tail_first(S, beg, end);
    const uint64_t pol = policy_evict_first();
    for (int seg = 0; seg < 2; ++seg) {
      const int xb = seg ? beg : mid, xe = seg ? mid : end;
      int e = 0;
      for (int x = xb; x < xe;) {
        while (S.sb[e + 1] <= x) ++e;
        const int np = S.np[e], l = x - S.sb[e], item = l >> 4;
        const int s0 = l & 15, s1 = min(16, s0 + xe - x);
        const int rb = item / np, j = item - rb * np;
        const int ntok = min(64, S.cnt[e] - 64 * j), c0 = S.off[e] + 64 * j;
        const uint8_t* wb = P.ws2 + (size_t)(e * 32 + rb) * 32 * 9216;
        // Scheduling/counters keep sixteen logical K256 stages. Each emits two
        // physical K128 stages, so the existing split epilogue still totals16.
        for (int s = s0; s < s1; ++s) for (int hf = 0; hf < 2; ++hf, ++it) {
          const int slot = it % NST;
          mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, P.err);
          uint8_t* dstp = ring + slot * STG;
          const uint32_t dst = sa(dstp), fb = full_a + 8 * slot;
          if (lane == 0) {
            mbar_expect_tx(fb, 9216);
            const uint8_t* src = wb + (size_t)(2 * s + hf) * 9216;
            if (P.probe & 16) bulk_g2s_ef(dst, src, 9216, fb, pol);
            else bulk_g2s(dst, src, 9216, fb);
          }
          uint32_t* xp = reinterpret_cast<uint32_t*>(dstp + 9216);
          const int padded = (ntok + 7) & ~7;
          for (int piece = lane; piece < padded * 8; piece += 32) {
            const int n = piece >> 3, i = (piece >> 2) & 1, tt = piece & 3;
            uint4 a = make_uint4(0,0,0,0), b = a;
            if (n < ntok) {
              const __nv_bfloat16* src = P.inter + ((size_t)s * P3 + c0 + n) * XST
                                       + hf * 128 + i * 64 + tt * 16;
              const uint4* vp = reinterpret_cast<const uint4*>(src);
              a = __ldg(vp); b = __ldg(vp + 1);
            }
            const uint32_t v[8] = {a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
            const int u0 = (n >> 3) * 512 + i * 256 + 4 * (n & 7) + tt;
#pragma unroll
            for (int q = 0; q < 8; ++q) xp[u0 + 32 * q] = v[q];
          }
          // Generic shared stores -> WGMMA async proxy, then one full-barrier
          // arrival per producer lane. The bulk weight transaction supplies the
          // remaining arrive/expected-tx side: the original full count is33.
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
          mbar_arrive(fb);
          __syncwarp();
        }
        x += s1 - s0;
      }
    }
    return;
  }
"""

_MIXED = r"""
constexpr int HW_SM = @SMEM@;

__global__ void __launch_bounds__(320, 2) k2_hotw2(const Params P, const uint8_t* hot_ws) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ Route S;
  __shared__ int hot_ctas;
  pdl_launch();
  // Exactly the incumbent K2 count/offset calculation. LIST=false; K1 owns
  // slot->compact metadata. Every role retains the ORIGINAL compact offsets.
  route<1, 32, true, false>(P, S, nullptr, nullptr, 32, false);
  const int tid = threadIdx.x, lane = tid & 31;
  if (tid == 0) {
    long long wh = 0, wc = 0;
    for (int e = 0; e < 32; ++e) {
      const int c = S.cnt[e];
      const bool hot = c >= 32;
      const int tp = hot ? 64 : (c > P.hot_t ? 32 : 8);
      for (int n = 0; n < c; n += tp) {
        const int rows = min(tp, c - n);
        if (hot) wh += 36 + ((rows + 7) & ~7);
        else wc += 36 + rows;
      }
    }
    const int g = (int)gridDim.x;
    int gh = wh ? (wc ? (int)((wh * g + (wh + wc) / 2) / (wh + wc)) : g) : 0;
    if (wh) gh = max(1, min(gh, wc ? g - 1 : g));
    hot_ctas = gh;
  }
  __syncthreads();
  const bool fast = (int)blockIdx.x < hot_ctas;
  const int role_cta = fast ? (int)blockIdx.x : (int)blockIdx.x - hot_ctas;
  const int role_ncta = fast ? hot_ctas : (int)gridDim.x - hot_ctas;
  if (tid < 32) {
    const int c = S.cnt[lane];
    const int kept = ((c >= 32) == fast) ? c : 0;
    const bool old_hot = kept > P.hot_t;
    const int tp = fast ? 64 : (old_hot ? 32 : 8);
    const int np = (kept + tp - 1) / tp;
    int all = np, h = (fast || old_hot) ? np : 0, cold = (fast || old_hot) ? 0 : np;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int aa = __shfl_up_sync(FULL, all, d), hh = __shfl_up_sync(FULL, h, d), cc = __shfl_up_sync(FULL, cold, d);
      if (lane >= d) { all += aa; h += hh; cold += cc; }
    }
    S.cnt[lane] = kept; S.np[lane] = np;
    S.sb[lane] = 512 * (all - np);
    S.sbh[lane] = 512 * (h - ((fast || old_hot) ? np : 0));
    S.sbc[lane] = 512 * (cold - ((fast || old_hot) ? 0 : np));
    S.hp[lane] = h - ((fast || old_hot) ? np : 0);
    if (lane == 31) { S.sb[32] = 512 * all; S.sbh[32] = 512 * h; S.sbc[32] = 512 * cold; }
  }
  __syncthreads();
  // Uniform, fixed role for the whole CTA, outside either worker's item loop.
  if (fast) {
    constexpr int NST = 3, STG = 25600;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
    setup_bars<1, NST>(bars);
    __syncthreads();
    Params q = P; q.ws2 = hot_ws;
    hw_gemm<1, NST, 2, 1, STG>(q, S, nullptr, smem, bars, true, role_cta, role_ncta);
  } else {
    constexpr int NST = 3, STG = SW + 32 * XROW;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
    setup_bars37<NST>(bars);
    __syncthreads();
    hw_gemm37<4, NST, 2>(P, S, nullptr, smem, bars, true, role_cta, role_ncta);
  }
  __syncthreads();
}
"""

_HOST = r"""
static void launch_hw(const hotw2_lab::Params& p, const uint8_t* w, int G) {
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = at::cuda::getCurrentCUDAStream(); cfg.attrs = a; cfg.numAttrs = 1;
  cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = hotw2_lab::HW_SM;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, hotw2_lab::k2_hotw2, p, w));
}

std::vector<int64_t> hw_attrs() {
  auto f = hotw2_lab::k2_hotw2;
  C10_CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, hotw2_lab::HW_SM));
  C10_CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  cudaFuncAttributes a; C10_CUDA_CHECK(cudaFuncGetAttributes(&a, f));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, 320, hotw2_lab::HW_SM));
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes,
          hotw2_lab::HW_SM, occ, 320};
}
"""

_CPP = r"""
#include <torch/extension.h>
#include <vector>
void run_hw(int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13,
  torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h,
  torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err,
  torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor xw,
  torch::Tensor xw2, torch::Tensor hw, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe);
std::vector<int64_t> hw_attrs();
"""


def source(module=None):
    base, specs = _base(module)
    # Neither K1 nor K3 is defined/instantiated in this extension. Both are
    # launched from the already prepared incumbent binary.
    prefix = base[:base.index("// K1 / K2 of the HK3 variants")]
    hot = _function(base, "void hot_item(", True)
    hot = _once(hot, "void hot_item(", "void hw_hot_item(")
    hot = _once(hot, "for (int s = s0; s < s1; ++s, ++it)", "for (int s = 2 * s0; s < 2 * s1; ++s, ++it)")
    hot = _once(hot, "st + warp * 2048", "st + warp * 1024")
    hot = _once(hot, "st + 16384 + warp * 256", "st + 8192 + warp * 128")
    hot = _once(hot, "wg_desc_sa(sa(st + SW), 4096u)", "wg_desc_sa(sa(st + 9216), 2048u)")
    hot = _once(hot, "for (int i = 0; i < 4; ++i)", "for (int i = 0; i < 2; ++i)")
    gemm = _function(base, "void gemm(", True)
    gemm = _once(gemm, "void gemm(", "void hw_gemm(")
    producer = gemm.index("  if (warp == 8) {")
    left = gemm.index("{", producer)
    gemm = gemm[:producer] + _PRODUCER + gemm[_closing(gemm, left) + 1:]
    gemm = gemm.replace("hot_item<", "hw_hot_item<")
    segs = _function(base, "void segs37(", True)
    segs = _once(segs, "void segs37(", "void hw_segs37(")
    segs = _once(segs, "int* sg, int* sd)", "int* sg, int* sd, int role_cta, int role_ncta)")
    segs = segs.replace("gridDim.x", "role_ncta").replace("blockIdx.x", "role_cta")
    legacy = _function(base, "void gemm37(", True)
    legacy = _once(legacy, "void gemm37(", "void hw_gemm37(")
    legacy = _once(legacy, "uint64_t* bars, bool pdl)", "uint64_t* bars, bool pdl, int role_cta, int role_ncta)")
    legacy = _once(legacy, "segs37<NCH>", "hw_segs37<NCH>")
    legacy = _once(legacy, "pr, sd);", "pr, sd, role_cta, role_ncta);")
    legacy = legacy.replace("gridDim.x", "role_ncta").replace("blockIdx.x", "role_cta")
    entry = _function(base, "void run3(")
    entry = _once(entry, "void run3(", "void run_hw(")
    entry = _once(entry, "torch::Tensor xw2, int64_t hot_t", "torch::Tensor xw2, torch::Tensor hw, int64_t hot_t")
    pos = entry.index("  switch (vid) {")
    entry = entry[:pos] + r"""
  TORCH_CHECK(vid == 5 && mask == 2 && M == 128 && lo == 0, "hotW2: unsupported route/shape");
  TORCH_CHECK(p.hot_t == 8 && p.hot_mode == 0 && ((p.probe >> 14) & 63) == 0,
              "hotW2: requires incumbent T128 low/mid policy and static K2 tail");
  launch_hw(p, reinterpret_cast<const uint8_t*>(hw.data_ptr()), (int)G);
}
"""
    cuda = prefix + "\n" + hot + "\n" + gemm + "\n" + segs + "\n" + legacy
    cuda += "\n" + _MIXED.replace("@SMEM@", str(SMEM)) + "\n}\n" + _HOST + entry
    cuda = cuda.replace("@WGSPECS@", specs).replace("namespace k3 {", "namespace hotw2_lab {").replace("k3::", "hotw2_lab::")
    if re.search(r"@[A-Z][A-Z0-9_]*@", cuda):
        raise RuntimeError("unexpanded hotW2 source token")
    if len(cuda.encode()) >= 1_000_000:
        raise RuntimeError("hotW2 source exceeds 1 MB")
    return cuda, _CPP


def _repack(ws2):
    """Byte-only K256->two K128 stages, no dequantization or scale folding."""
    import torch
    if ws2.dtype != torch.uint8 or tuple(ws2.shape) != (32, 32, 16, 18432):
        raise TypeError("expected incumbent ws2 uint8 [32,32,16,18432]")
    w = ws2[..., :16384].reshape(32, 32, 16, 8, 2, 2, 32, 16)
    w = w.permute(0, 1, 2, 4, 3, 5, 6, 7).contiguous().reshape(32, 32, 32, 8192)
    sf = ws2[..., 16384:].reshape(32, 32, 16, 8, 2, 32, 4)
    sf = sf.permute(0, 1, 2, 4, 3, 5, 6).contiguous().reshape(32, 32, 32, 1024)
    return torch.cat((w, sf), dim=-1).contiguous()


def prepare(weights, incumbent_module, incumbent_state):
    import torch
    from torch.utils.cpp_extension import load_inline
    st = incumbent_state.get("st", incumbent_state)
    if st.get("r37", {}).get(128) != (5, (8, 0, 12), (8, 0, 3)):
        raise ValueError("hotW2 requires exact incumbent r37[128]=(5,(8,0,12),(8,0,3)); no fallback")
    if tuple(st["variants"][5]) != (1, 3, 2, 3, 4):
        raise ValueError("hotW2 requires incumbent vid5 HK3/NCH4/NST3")
    if getattr(incumbent_module, "K2A47_K1_PDL_T128", False):
        raise ValueError("hotW2 expects the default non-PDL T128 K1")
    if incumbent_module.K2A_W_SHIFT != 256.0:
        raise ValueError("hotW2 requires the incumbent's exact *256 dequant/global compensation")
    if tuple(weights["w2"].shape) != (32, 4096, 2048) or weights["w2"].dtype != torch.uint8:
        raise TypeError("hotW2 supports raw NVFP4 E32/D4096 weights")
    if weights["w2_sf"].dtype != torch.float8_e4m3fn or weights["w2_gs"].dtype != torch.float32:
        raise TypeError("hotW2 requires E4M3 group16 and FP32 global scales")
    if tuple(weights["w2_sf"].shape) != (32, 4096, 256) or tuple(weights["w2_gs"].shape) != (32,):
        raise ValueError("hotW2 scale shape mismatch")
    dev = st["ws2"].device
    if not st["ws2"].is_cuda or any(st[k].device != dev for k in _PRE_OUT + _POST_OUT):
        raise ValueError("incumbent buffers must share a CUDA device")
    if st["inter"].dtype != torch.bfloat16 or any(st[k].dtype != torch.float32 for k in ("gsc2", "y", "ss", "nw")):
        raise TypeError("hotW2 requires BF16 inter and FP32 globals/y/ss/norm weights")
    if torch.cuda.get_device_capability(dev)[0] != 9:
        raise ValueError("hotW2 requires SM90")
    grid = int(st["grid"][5])
    if grid < 2:
        raise ValueError("hotW2 mixed scheduler needs grid>=2")
    cuda, cpp = source(incumbent_module)
    digest = hashlib.sha256((cuda + cpp).encode()).hexdigest()
    with torch.cuda.device(dev), torch.no_grad():
        mod = load_inline(name="fresh_hotw2_" + digest[:16], cpp_sources=cpp, cuda_sources=cuda,
                          functions=["run_hw", "hw_attrs"], extra_cflags=["-O3"],
                          extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-lineinfo",
                              "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v", "-Xptxas=-warn-spills"],
                          verbose=True)
        attrs = list(map(int, mod.hw_attrs()))
        if grid == 264 and attrs[4] < 2:
            raise RuntimeError(
                f"hotW2 occupancy gate failed at grid264: {attrs}; requires >=2 CTAs/SM "
                "for the resident-grid/mailbox contract; this is not a performance ceiling"
            )
        hot_ws = _repack(st["ws2"])
        out = torch.empty((128, 4096), dtype=torch.bfloat16, device=dev)
    hot_code = incumbent_module._k2a36_hot_code(st["variants"][5], (8, 0, 12))
    probe = int(incumbent_module.K2A33_HINT) | incumbent_module._k2a37_dyn_code((8, 0, 3))
    return {"incumbent_state": st, "mod": mod, "hot_ws2": hot_ws, "out": out,
            "hot_code": hot_code, "probe": probe, "grid": grid, "grid3": int(st["sms"]) * 2,
            "metadata": {"source_sha256": digest, "cuda_bytes": len(cuda.encode()),
                "geometry": "token64/output128/K128; 320 threads, 8 math warps, 1 producer, 1 epilogue",
                "threshold": 32, "dynamic_smem": SMEM, "hot_ring_smem": HOT_SMEM,
                "cold_ring_smem": COLD_SMEM, "grid": grid,
                "occupancy_gate": "grid264 requires >=2 CTAs/SM; enforced before returning state",
                "error_flags": "incumbent_state['err']; inspect outside capture, all four must remain zero",
                "compiled": dict(zip(("registers", "local_bytes", "static_smem", "dynamic_smem", "occupancy", "threads"), attrs)),
                "single_y_ss": True, "K1_K3": "unchanged incumbent binary", "fallback": False,
                "gpu_numerics_validated": False}}


def run(data, state):
    import torch
    x, ids, wts = data["hidden_states"], data["topk_ids"], data["topk_weights"]
    if tuple(x.shape) != (128, 4096) or x.dtype != torch.bfloat16:
        raise ValueError("hotW2 only supports T128/D4096 BF16; no fallback")
    if tuple(ids.shape) != (128, 8) or ids.dtype != torch.int32:
        raise TypeError("hotW2 requires local int32 IDs [128,8]")
    if tuple(wts.shape) != (128, 8) or wts.dtype != torch.float32:
        raise TypeError("hotW2 requires FP32 routing weights [128,8]")
    if data.get("config", {}).get("local_start", 0) != 0:
        raise ValueError("hotW2 expects local IDs 0..31, remote -1 or >=32")
    if any(not v.is_contiguous() or v.device != state["out"].device for v in (x, ids, wts)):
        raise ValueError("hotW2 inputs must be contiguous on the prepared device")
    st = state["incumbent_state"]
    tensors = (x, ids, wts, *(st[k] for k in _PRE_OUT), state["out"], *(st[k] for k in _POST_OUT))
    tail = (state["hot_code"], 128, 0, state["grid"], state["grid3"], state["probe"])
    # Original K1 owns h/inter and zeroes y/ss; the mixed K2 waits before reading.
    st["mod"].run3(5, 1, *tensors, *tail)
    state["mod"].run_hw(5, 2, *tensors, state["hot_ws2"], *tail)
    st["mod"].run3(5, 4, *tensors, *tail)
    return state["out"]


def source_audit():
    cuda, cpp = source()
    base, _ = _base()
    assert HOT_SMEM == 76848 and COLD_SMEM == SMEM == 106160
    assert "@WGSPECS@" not in cuda and "__trap" not in cuda
    assert "namespace k3 {" not in cuda and "k3::" not in cuda
    assert "k1_kernel37" not in cuda and "void k3_kernel" not in cuda
    assert "role_ncta - 1" in cuda and "((p.probe >> 14) & 63) == 0" in cuda
    assert "fence.proxy.async.shared::cta" in cuda and "mbar_expect_tx(fb, 9216)" in cuda
    assert "for (int s = 2 * s0; s < 2 * s1; ++s, ++it)" in cuda
    for marker in ("void cold_item(", "void hc_item(", "void flush_item("):
        assert _function(base, marker, True) in cuda
    return {"status": "source_audit_only", "gpu_validated": False,
            "cuda_bytes": len(cuda.encode()), "source_sha256": hashlib.sha256((cuda + cpp).encode()).hexdigest(),
            "hot_smem": HOT_SMEM, "cold_smem": COLD_SMEM, "dynamic_smem": SMEM}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-audit", action="store_true")
    if not p.parse_args().source_audit: p.error("--source-audit only; parent owns the GPU harness")
    print(json.dumps(source_audit(), indent=2))

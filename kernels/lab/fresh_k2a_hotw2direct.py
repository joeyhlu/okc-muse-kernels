"""LAB T128: K1 writes hot BF16 SwiGLU directly to old K256 xw2 layout.

prepare(weights, incumbent_module, incumbent_state); run(data,state)->BF16.
One changed K1, one mixed K2, incumbent K3. Frozen adapters are untouched.
No GPU work on import. --source-audit uses only Python's standard library.
"""

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

HOT_SMEM = 2 * 51200 + 4 * 8
XW2_BYTES = 64 * 16 * 64 * 512


@lru_cache(maxsize=1)
def _base():
    path = Path(__file__).with_name("fresh_k2a_hotw2.py")
    spec = importlib.util.spec_from_file_location("fresh_hotw2direct_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STORES = r"""
// These pointers address this K1 CTA's immutable shared Route, not global
// metadata published by a different CTA. All CTAs reproduce the same offsets.
__device__ __forceinline__ void direct_store1(const Params& P, int e, int c, int col, __nv_bfloat16 v) {
  if (P.direct_cnt[e] >= 32) {
    const int r = c - P.direct_off[e], pid = 2 * e + (r >> 6), n = r & 63;
    reinterpret_cast<__nv_bfloat16*>(P.xw2)[xw_elem(pid, 64, n, col)] = v;
  } else {
    P.inter[((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)] = v;
  }
}
__device__ __forceinline__ void direct_store2(const Params& P, int e, int c, int col, uint32_t pk) {
  // Mailbox columns are even. xw_elem(col+1)==xw_elem(col)+1 for this pair.
  if (P.direct_cnt[e] >= 32) {
    const int r = c - P.direct_off[e], pid = 2 * e + (r >> 6), n = r & 63;
    *reinterpret_cast<uint32_t*>(P.xw2 + 2 * xw_elem(pid, 64, n, col)) = pk;
  } else {
    *reinterpret_cast<uint32_t*>(P.inter + ((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)) = pk;
  }
}
"""

_GUARD = r"""
  // Fixed two-pass/expert allocation supports counts<=128. Each warp checks
  // the same 32 shared counts, making this return uniform for the whole CTA.
  if (__ballot_sync(FULL, S.cnt[lane] > 128)) {
    if (tid == 0) { atomicExch(P.err, 1); atomicExch(P.err + 3, 1); }
    return;
  }
"""

_K1 = r"""
__global__ void __launch_bounds__(320, 2) k1_direct(const Params P) {
  constexpr int NCH = 4, NST = 3;
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = SW + 8 * NCH * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  setup_bars37<NST>(bars);
  pdl_wait();
  route<1, 8 * NCH, true>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);
  const int tid = threadIdx.x, lane = tid & 31;
@GUARD@
  Params q = P; q.direct_off = S.off; q.direct_cnt = S.cnt;
  gemm37<NCH, NST, 1>(q, S, s_list, smem, bars, false);
}
"""

_PRODUCER = r"""
  if (warp == 8) {
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
        const int ntok = min(64, S.cnt[e] - 64 * j), pid = 2 * e + j;
        const int xbytes = ((ntok + 7) >> 3) * 4096;
        const uint8_t* wb = P.ws2 + (size_t)(e * 32 + rb) * 16 * 18432;
        for (int s = s0; s < s1; ++s, ++it) {
          const int slot = it % NST;
          mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, P.err);
          const uint32_t dst = sa(ring + slot * STG), fb = full_a + 8 * slot;
          if (lane == 0) {
            mbar_expect_tx(fb, 18432 + xbytes);
            const uint8_t* src = wb + (size_t)s * 18432;
            if (P.probe & 16) bulk_g2s_ef(dst, src, 18432, fb, pol);
            else bulk_g2s(dst, src, 18432, fb);
            const uint8_t* xs = P.xw2 + (size_t)(pid * 16 + s) * (64 * 512);
            bulk_g2s(dst + 18432, xs, xbytes, fb);
          }
          mbar_arrive(fb); // one expect_tx +32 lane arrivals, both bulk completions
          __syncwarp();
        }
        x += s1 - s0;
      }
    }
    return;
  }
"""

_HOST_K1 = r"""
static void launch_direct_k1(const hotw2_lab::Params& p, int G) {
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a[0].val.programmaticStreamSerializationAllowed = (p.probe >> 24) & 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = at::cuda::getCurrentCUDAStream(); cfg.attrs = a; cfg.numAttrs = 1;
  cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = hotw2_lab::HW_SM;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, hotw2_lab::k1_direct, p));
}
"""


def source(module=None):
    h = _base()
    cuda, cpp = h.source(module)
    raw, _ = h._base(module)
    # Params is local to this extension. Host zero-initialization leaves these
    # unused pointers null in K2; K1 supplies its own CTA's shared arrays.
    cuda = h._once(cuda, "  int hot_t;                   //",
                  "  const int* direct_off; const int* direct_cnt;\n  int hot_t;                   //")
    xwe = h._function(cuda, "size_t xw_elem(")
    cuda = h._once(cuda, xwe, xwe + "\n" + _STORES)
    old = "ib[((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)] = __float2bfloat16_rn(swiglu(v0, v1));"
    cuda = h._once(cuda, old, "direct_store1(P, e, c, col, __float2bfloat16_rn(swiglu(v0, v1)));")
    pair = "*reinterpret_cast<uint32_t*>(P.inter + ((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)) = pk;"
    for marker in ("void gemm37(", "void hw_gemm37("):
        f = h._function(cuda, marker, True)
        cuda = h._once(cuda, f, h._once(f, pair, "direct_store2(P, e, c, col, pk);"))
    hot = h._function(cuda, "void hw_hot_item(", True)
    original_hot = h._function(raw, "void hot_item(", True)
    original_hot = h._once(original_hot, "void hot_item(", "void hw_hot_item(")
    cuda = h._once(cuda, hot, original_hot)
    cuda = h._once(cuda, h._PRODUCER, _PRODUCER)
    mixed = h._function(cuda, "__global__ void __launch_bounds__(320, 2) k2_hotw2(")
    new_mixed = h._once(mixed, "constexpr int NST = 3, STG = 25600;", "constexpr int NST = 2, STG = 51200;")
    new_mixed = h._once(new_mixed, "  const int tid = threadIdx.x, lane = tid & 31;",
                        "  const int tid = threadIdx.x, lane = tid & 31;\n" + _GUARD)
    cuda = h._once(cuda, mixed, _K1.replace("@GUARD@", _GUARD) + "\n" + new_mixed)
    # Host exports accept masks1/2, retaining the same torch Tensor signature.
    entry = h._function(cuda, "void run_hw(")
    new_entry = h._once(entry, "mask == 2", "(mask == 1 || mask == 2)")
    new_entry = h._once(new_entry, "Params p;", "Params p{};")
    new_entry = h._once(new_entry, "  launch_hw(p, reinterpret_cast<const uint8_t*>(hw.data_ptr()), (int)G);",
                        "  if (mask == 1) launch_direct_k1(p, (int)G);\n"
                        "  else launch_hw(p, reinterpret_cast<const uint8_t*>(hw.data_ptr()), (int)G);")
    cuda = h._once(cuda, entry, new_entry)
    attrs = h._function(cuda, "std::vector<int64_t> hw_attrs(")
    k1attrs = h._once(attrs, "hw_attrs()", "hw_k1_attrs()")
    k1attrs = h._once(k1attrs, "hotw2_lab::k2_hotw2", "hotw2_lab::k1_direct")
    cuda = h._once(cuda, attrs, _HOST_K1 + "\n" + k1attrs + "\n" + attrs)
    cpp += "\nstd::vector<int64_t> hw_k1_attrs();\n"
    return cuda.replace("hotw2_lab", "hotw2_direct_lab"), cpp


def prepare(weights, incumbent_module, incumbent_state):
    import torch
    from torch.utils.cpp_extension import load_inline
    h = _base()
    st = incumbent_state.get("st", incumbent_state)
    if st.get("r37", {}).get(128) != (5, (8, 0, 12), (8, 0, 3)):
        raise ValueError("direct hotW2 requires exact incumbent T128 route; no fallback")
    if tuple(st["variants"][5]) != (1, 3, 2, 3, 4):
        raise ValueError("direct hotW2 requires incumbent vid5 HK3/NCH4/NST3")
    if getattr(incumbent_module, "K2A47_K1_PDL_T128", False) or incumbent_module.K2A_W_SHIFT != 256.0:
        raise ValueError("direct hotW2 requires default non-PDL T128 K1 and *256 compensation")
    if tuple(weights["w2"].shape) != (32, 4096, 2048) or weights["w2"].dtype != torch.uint8:
        raise TypeError("direct hotW2 requires raw NVFP4 E32/D4096")
    if weights["w2_sf"].dtype != torch.float8_e4m3fn or weights["w2_gs"].dtype != torch.float32:
        raise TypeError("direct hotW2 requires E4M3 group16 and FP32 globals")
    if tuple(weights["w2_sf"].shape) != (32, 4096, 256) or tuple(weights["w2_gs"].shape) != (32,):
        raise ValueError("direct hotW2 scale shape mismatch")
    if tuple(st["ws2"].shape) != (32, 32, 16, 18432) or st["ws2"].dtype != torch.uint8:
        raise ValueError("direct hotW2 requires original incumbent K256 ws2")
    if st["xw2"].dtype != torch.uint8 or not st["xw2"].is_contiguous() or st["xw2"].numel() < XW2_BYTES:
        raise ValueError("direct hotW2 needs >=33554432 existing contiguous xw2 bytes")
    dev = st["ws2"].device
    if not st["ws2"].is_cuda or any(st[k].device != dev for k in h._PRE_OUT + h._POST_OUT):
        raise ValueError("incumbent buffers must share a CUDA device")
    if st["inter"].dtype != torch.bfloat16 or any(st[k].dtype != torch.float32 for k in ("h", "gsc13", "gsc2", "y", "ss", "nw")):
        raise TypeError("direct hotW2 requires BF16 inter and FP32 partials/globals/y/ss/norm")
    if torch.cuda.get_device_capability(dev)[0] != 9:
        raise ValueError("direct hotW2 requires SM90")
    grid = int(st["grid"][5])
    if grid < 2:
        raise ValueError("direct hotW2 needs grid>=2")
    cuda, cpp = source(incumbent_module)
    digest = hashlib.sha256((cuda + cpp).encode()).hexdigest()
    with torch.cuda.device(dev), torch.no_grad():
        mod = load_inline(name="fresh_hotw2direct_" + digest[:16], cpp_sources=cpp, cuda_sources=cuda,
                          functions=["run_hw", "hw_attrs", "hw_k1_attrs"], extra_cflags=["-O3"],
                          extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-lineinfo",
                              "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v", "-Xptxas=-warn-spills"],
                          verbose=True)
        attrs = list(map(int, mod.hw_attrs()))
        a1 = list(map(int, mod.hw_k1_attrs()))
        if grid == 264 and min(attrs[4], a1[4]) < 2:
            raise RuntimeError(f"direct hotW2 occupancy gate at grid264: K1={a1}, K2={attrs}; both require >=2 CTAs/SM")
        out = torch.empty((128, 4096), dtype=torch.bfloat16, device=dev)
    fields = ("registers", "local_bytes", "static_smem", "dynamic_smem", "occupancy", "threads")
    return {"incumbent_state": st, "mod": mod, "hot_ws2": st["ws2"], "out": out,
            "hot_code": incumbent_module._k2a36_hot_code(st["variants"][5], (8, 0, 12)),
            "probe": int(incumbent_module.K2A33_HINT) | incumbent_module._k2a37_dyn_code((8, 0, 3)),
            "grid": grid, "grid3": int(st["sms"]) * 2,
            "metadata": {"variant": "hotw2direct_K1_oldcore_bulkK2", "source_sha256": digest,
                "cuda_bytes": len(cuda.encode()), "threshold": 32, "route_count_limit": 128,
                "error_flags": "incumbent_state['err']; all4 must remain zero; >128 count sets err0+err3",
                "compiled": dict(zip(fields, attrs)), "compiled_K1": dict(zip(fields, a1)),
                "dynamic_smem": h.SMEM, "hot_ring_smem": HOT_SMEM, "xw2_required_bytes": XW2_BYTES,
                "K1": "original HK3 math, CTA-private counts/offsets, changed SwiGLU destination",
                "K3": "unchanged incumbent binary", "single_y_ss": True, "fallback": False,
                "hot_weight_repack": False, "gpu_numerics_validated": False}}


def run(data, state):
    import torch
    h = _base()
    x, ids, wts = data["hidden_states"], data["topk_ids"], data["topk_weights"]
    if tuple(x.shape) != (128, 4096) or x.dtype != torch.bfloat16:
        raise ValueError("direct hotW2 only supports T128/D4096 BF16")
    if tuple(ids.shape) != (128, 8) or ids.dtype != torch.int32:
        raise TypeError("direct hotW2 requires local int32 IDs [128,8]")
    if tuple(wts.shape) != (128, 8) or wts.dtype != torch.float32:
        raise TypeError("direct hotW2 requires FP32 routing weights [128,8]")
    if data.get("config", {}).get("local_start", 0) != 0:
        raise ValueError("direct hotW2 expects local IDs0..31, remote -1 or >=32")
    if any(not t.is_contiguous() or t.device != state["out"].device for t in (x, ids, wts)):
        raise ValueError("direct hotW2 inputs must be contiguous on prepared device")
    st = state["incumbent_state"]
    tensors = (x, ids, wts, *(st[k] for k in h._PRE_OUT), state["out"], *(st[k] for k in h._POST_OUT))
    tail = (state["hot_code"], 128, 0, state["grid"], state["grid3"], state["probe"])
    state["mod"].run_hw(5, 1, *tensors, state["hot_ws2"], *tail)
    state["mod"].run_hw(5, 2, *tensors, state["hot_ws2"], *tail)
    st["mod"].run3(5, 4, *tensors, *tail)
    return state["out"]


def source_audit():
    h = _base()
    baseline, cpp0 = h.source()
    assert hashlib.sha256((baseline + cpp0).encode()).hexdigest() == "aa04a93bc09f76ff069a17cefffd57d93b003e070e0db43da5ee720d77941c41"
    cuda, cpp = source()
    raw, _ = h._base()
    assert len(cuda.encode()) < 1_000_000 and HOT_SMEM == 102432 < h.SMEM == 106160
    assert h._function(cuda, "void hw_hot_item(", True).replace("void hw_hot_item(", "void hot_item(") == h._function(raw, "void hot_item(", True)
    assert h._function(cuda, "void cold_item(", True) == h._function(baseline, "void cold_item(", True)
    assert h._function(cuda, "void hc_item(", True) == h._function(baseline, "void hc_item(", True)
    assert "__shfl" not in _PRODUCER and "P.inter" not in _PRODUCER and "piece" not in _PRODUCER
    assert cuda.count(_GUARD) == 2 and "q.direct_off = S.off; q.direct_cnt = S.cnt" in cuda
    assert "mask == 1 || mask == 2" in cuda and "hw_k1_attrs" in cpp and "Params p{};" in cuda
    assert cuda.count("__global__ void __launch_bounds__(320, 2) k1_direct(") == 1
    assert cuda.count("__global__ void __launch_bounds__(320, 2) k2_hotw2(") == 1
    assert "__global__ \n__global__" not in cuda
    # Only destination stores in the instantiated K1 worker/finalizer change.
    for marker in ("void gemm37(", "void hw_gemm37("):
        f = h._function(cuda, marker, True)
        restored = h._once(f, "direct_store2(P, e, c, col, pk);",
            "*reinterpret_cast<uint32_t*>(P.inter + ((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)) = pk;")
        assert restored == h._function(baseline, marker, True)
    f = h._function(cuda, "void flush_item(", True)
    restored = h._once(f, "direct_store1(P, e, c, col, __float2bfloat16_rn(swiglu(v0, v1)));",
        "ib[((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)] = __float2bfloat16_rn(swiglu(v0, v1));")
    assert restored == h._function(baseline, "void flush_item(", True)
    # Extract the actual immutable layout function instead of assuming row pitch.
    def elem(pid, n, col):
        kk = col & 255
        return 2 * ((pid * 16 + (col >> 8)) * (64 * 128) + (n >> 3) * 1024 + 256 * (kk >> 6) + 32 * ((kk >> 1) & 7) + 4 * (n & 7) + ((kk >> 4) & 3)) + (kk & 1)
    assert "(n >> 3) * 1024 + 256 * i + 32 * q + 4 * (n & 7) + t" in cuda
    assert sorted(elem(0, n, k) for n in range(64) for k in range(256)) == list(range(64 * 256))
    # Every hot whole/pair store is unique, pair-aligned, and lies inside the
    # padded-prefix bulk transaction for its fixed pid and physical K256 stage.
    stores = 0
    for count in range(32, 129):
        for e, off in ((0, 0), (1, 3), (17, 57), (31, 897)):
            for rank in range(count):
                c = off + rank
                pid, n = 2 * e + ((c - off) // 64), (c - off) % 64
                ntok = min(64, count - 64 * (rank // 64))
                xbytes = ((ntok + 7) >> 3) * 4096
                # The local K256 address repeats at the explicitly checked
                # 16 stage offsets below; avoid re-enumerating it 16 times.
                for col in range(0, 256, 2):
                    a = elem(pid, n, col)
                    assert elem(pid, n, col + 1) == a + 1 and a % 2 == 0
                    stage_start = pid * 16 * 32768
                    assert stage_start <= 2 * a < 2 * (a + 1) + 1 < stage_start + xbytes
                    assert 2 * (a + 1) + 1 < XW2_BYTES
                    stores += 1
                for stage in range(16):
                    assert elem(pid, n, stage * 256) == elem(pid, n, 0) + stage * 16384
                    assert elem(pid, n, stage * 256 + 255) == elem(pid, n, 255) + stage * 16384
                    assert 2 * elem(pid, n, stage * 256 + 255) + 1 < XW2_BYTES
    # K1 passes32 may cross neither a fixed64 boundary nor an expert boundary.
    # Whole flush and last-contributor mailbox cover every valid (rank,col)
    # exactly once. Simulate every possible K split and whole/NOSWI paths.
    for count in range(129):
        tp = 32 if count > 8 else 8
        ranks = [r for j in range((count + tp - 1) // tp) for r in range(j * tp, min(count, (j + 1) * tp))]
        assert ranks == list(range(count))
        for ns in ([16], [1, 15], [8, 8], [3, 4, 9], [1] * 16):
            acc = finals = 0
            for part in ns:
                acc += part
                if acc == 16: finals += 1; acc = 0
            assert finals == 1 and acc == 0
    # Actual consumer-lane whole flush versus mailbox finalizer coverage for
    # every K1 remainder. Large whole items use NOSWI/mailbox; the same paired
    # finalizer handles last contributors to split items of every size.
    coverage = 0
    for ntok in range(1, 33):
        for rb in (0, 17, 63):
            for c0 in (0, 3, 64, 897):
                want = {(c0 + n, rb * 64 + col) for n in range(ntok) for col in range(64)}
                whole = []
                for warp in range(8):
                    for lane in range(32):
                        g, t = lane >> 2, lane & 3
                        for q in range((ntok + 7) // 8):
                            for hh in range(2):
                                n = 8 * q + 2 * t + hh
                                if n < ntok: whole.append((c0 + n, 8 * (rb * 8 + warp) + g))
                mailbox = [(c0 + n, rb * 64 + 2 * lane + h0)
                           for n in range(ntok) for lane in range(32) for h0 in range(2)]
                assert len(whole) == len(set(whole)) == len(want) and set(whole) == want
                assert len(mailbox) == len(set(mailbox)) == len(want) and set(mailbox) == want
                coverage += len(want)
    # A poisoned unused B column cannot enter another token's sum of squares:
    # reduction shuffles alter g (feature), preserving low lane bits t (token).
    for ntok in range(1, 65):
        nch = (ntok + 7) // 8
        for q in range(nch):
            for hh in range(2):
                for lane in range(32):
                    n = 8 * q + 2 * (lane & 3) + hh
                    for mask in (4, 8, 16):
                        assert 8 * q + 2 * ((lane ^ mask) & 3) + hh == n
    raw_hot = h._function(raw, "void hot_item(", True)
    assert "if (g == 0 && n < ntok) atomicAdd(P.ss + c0 + n, pz)" in raw_hot
    assert "if (q * 8 + 2 * t + hh < ntok)" in raw_hot
    # Independent no-swizzle descriptor reconstruction: each consumed label
    # equals the original scalar producer's RS K permutation in xw_elem.
    descriptor_reads = 0
    for pid in (0, 1, 62, 63):
        for s in range(16):
            for n in range(64):
                for i in range(4):
                    for q in range(4):
                        for k in range(16):
                            logical = 256 * s + 64 * i + 16 * ((k % 8) // 2) + 2 * (2 * q + k // 8) + (k & 1)
                            byte = (pid * 16 + s) * 32768 + 1024 * i + 256 * q + 4096 * (n // 8) + 16 * (n % 8) + 2 * (k % 8) + 128 * (k // 8)
                            assert 2 * elem(pid, n, logical) == byte
                            descriptor_reads += 1
    return {"status": "source_audit_only", "gpu_validated": False,
            "source_sha256": hashlib.sha256((cuda + cpp).encode()).hexdigest(),
            "cuda_bytes": len(cuda.encode()), "checked_pair_stores": stores,
            "whole_mailbox_coverage": coverage, "descriptor_label_comparisons": descriptor_reads,
            "hot_count_range": [32, 128], "xw2_required_bytes": XW2_BYTES,
            "dynamic_smem": h.SMEM, "hot_ring_smem": HOT_SMEM}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", action="store_true")
    if not parser.parse_args().source_audit:
        parser.error("--source-audit only; parent owns GPU validation")
    print(json.dumps(source_audit(), indent=2))

"""LAB hotW2v sibling: original K256 weights/consumer, ring2, SW128 B.

prepare(weights, incumbent_module, incumbent_state); run(data, state) -> BF16.
Frozen base/vector files are dependencies and are never edited or patched.
No GPU work on import; --source-audit uses only the standard library.
"""

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

HOT_STG = 51200
HOT_NST = 2
HOT_SMEM = HOT_NST * HOT_STG + 2 * HOT_NST * 8


@lru_cache(maxsize=1)
def _vector():
    path = Path(__file__).with_name("fresh_k2a_hotw2v.py")
    spec = importlib.util.spec_from_file_location("fresh_hotw2v256_vector", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _producer():
    v = _vector()
    h = v._base()
    p = h._once(h._PRODUCER, v._OLD_COPY, v._NEW_COPY)
    for old, new in (
        ("* 32 * 9216", "* 16 * 18432"),
        ("// Scheduling/counters keep sixteen logical K256 stages. Each emits two\n"
         "        // physical K128 stages, so the existing split epilogue still totals16.",
         "// One physical K256 stage per original logical stage; counters total16."),
        ("for (int s = s0; s < s1; ++s) for (int hf = 0; hf < 2; ++hf, ++it)",
         "for (int s = s0; s < s1; ++s, ++it)"),
        ("mbar_expect_tx(fb, 9216)", "mbar_expect_tx(fb, 18432)"),
        ("wb + (size_t)(2 * s + hf) * 9216", "wb + (size_t)s * 18432"),
        ("bulk_g2s_ef(dst, src, 9216, fb, pol)", "bulk_g2s_ef(dst, src, 18432, fb, pol)"),
        ("bulk_g2s(dst, src, 9216, fb)", "bulk_g2s(dst, src, 18432, fb)"),
        ("reinterpret_cast<uint16_t*>(dstp + 9216)", "reinterpret_cast<uint16_t*>(dstp + 18432)"),
        ("piece < padded * 8", "piece < padded * 16"),
        ("n = piece >> 3, i = (piece >> 2) & 1", "n = piece >> 4, i = (piece >> 2) & 3"),
        ("+ hf * 128 + i * 64 + tt * 16", "+ i * 64 + tt * 16"),
        ("(n >> 3) * 1024 + i * 512", "(n >> 3) * 2048 + i * 512"),
    ):
        p = h._once(p, old, new)
    return p


def source(module=None):
    v = _vector()
    h = v._base()
    cuda, cpp = v.source(module)
    hot = h._function(cuda, "void hw_hot_item(", True)
    new_hot = hot
    for old, new in (
        ("for (int s = 2 * s0; s < 2 * s1; ++s, ++it)", "for (int s = s0; s < s1; ++s, ++it)"),
        ("st + warp * 1024", "st + warp * 2048"),
        ("st + 8192 + warp * 128", "st + 16384 + warp * 256"),
        ("hwv_desc_sa(sa(st + 9216))", "hwv_desc_sa(sa(st + 18432))"),
        ("for (int i = 0; i < 2; ++i)", "for (int i = 0; i < 4; ++i)"),
    ):
        new_hot = h._once(new_hot, old, new)
    cuda = h._once(cuda, hot, new_hot)
    vector_producer = h._once(h._PRODUCER, v._OLD_COPY, v._NEW_COPY)
    cuda = h._once(cuda, vector_producer, _producer())
    cuda = h._once(cuda, "0x4000008000010000ull", "0x4000010000010000ull")
    cuda = h._once(cuda, "// B K-major SW128; LBO16 ignored, SBO2048, zero pattern base phase.",
                  "// B K-major SW128; LBO16 ignored, SBO4096, zero pattern base phase.")
    mixed = h._function(cuda, "void __launch_bounds__(320, 2) k2_hotw2(")
    new_mixed = h._once(mixed, "constexpr int NST = 3, STG = 25600;",
                        "constexpr int NST = 2, STG = 51200;")
    cuda = h._once(cuda, mixed, new_mixed)
    return cuda.replace("hotw2v_lab", "hotw2v256_lab"), cpp


def prepare(weights, incumbent_module, incumbent_state):
    """Same contract as frozen adapters; reuse original ws2 without a repack.

    Validation/compile setup is intentionally local: frozen base.prepare always
    allocates a K128 repack, which this K256 consumer neither needs nor reads.
    """
    import torch
    from torch.utils.cpp_extension import load_inline
    h = _vector()._base()
    st = incumbent_state.get("st", incumbent_state)
    if st.get("r37", {}).get(128) != (5, (8, 0, 12), (8, 0, 3)):
        raise ValueError("hotW2v256 requires exact incumbent T128 route; no fallback")
    if tuple(st["variants"][5]) != (1, 3, 2, 3, 4):
        raise ValueError("hotW2v256 requires incumbent vid5 HK3/NCH4/NST3")
    if getattr(incumbent_module, "K2A47_K1_PDL_T128", False):
        raise ValueError("hotW2v256 expects the default non-PDL T128 K1")
    if incumbent_module.K2A_W_SHIFT != 256.0:
        raise ValueError("hotW2v256 requires exact incumbent *256 compensation")
    if tuple(weights["w2"].shape) != (32, 4096, 2048) or weights["w2"].dtype != torch.uint8:
        raise TypeError("hotW2v256 requires raw NVFP4 E32/D4096")
    if weights["w2_sf"].dtype != torch.float8_e4m3fn or weights["w2_gs"].dtype != torch.float32:
        raise TypeError("hotW2v256 requires E4M3 group16 and FP32 global scales")
    if tuple(weights["w2_sf"].shape) != (32, 4096, 256) or tuple(weights["w2_gs"].shape) != (32,):
        raise ValueError("hotW2v256 scale shape mismatch")
    if tuple(st["ws2"].shape) != (32, 32, 16, 18432) or st["ws2"].dtype != torch.uint8:
        raise ValueError("hotW2v256 requires original incumbent K256 ws2")
    dev = st["ws2"].device
    if not st["ws2"].is_cuda or any(st[k].device != dev for k in h._PRE_OUT + h._POST_OUT):
        raise ValueError("incumbent buffers must share a CUDA device")
    if st["inter"].dtype != torch.bfloat16 or any(st[k].dtype != torch.float32 for k in ("gsc2", "y", "ss", "nw")):
        raise TypeError("hotW2v256 requires BF16 inter and FP32 globals/y/ss/norm weights")
    if torch.cuda.get_device_capability(dev)[0] != 9:
        raise ValueError("hotW2v256 requires SM90")
    grid = int(st["grid"][5])
    if grid < 2:
        raise ValueError("hotW2v256 mixed scheduler needs grid>=2")
    cuda, cpp = source(incumbent_module)
    digest = hashlib.sha256((cuda + cpp).encode()).hexdigest()
    with torch.cuda.device(dev), torch.no_grad():
        mod = load_inline(name="fresh_hotw2v256_" + digest[:16], cpp_sources=cpp, cuda_sources=cuda,
                          functions=["run_hw", "hw_attrs"], extra_cflags=["-O3"],
                          extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-lineinfo",
                              "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v", "-Xptxas=-warn-spills"],
                          verbose=True)
        attrs = list(map(int, mod.hw_attrs()))
        if grid == 264 and attrs[4] < 2:
            raise RuntimeError(f"hotW2v256 occupancy gate failed at grid264: {attrs}; requires >=2 CTAs/SM")
        out = torch.empty((128, 4096), dtype=torch.bfloat16, device=dev)
    hot_code = incumbent_module._k2a36_hot_code(st["variants"][5], (8, 0, 12))
    probe = int(incumbent_module.K2A33_HINT) | incumbent_module._k2a37_dyn_code((8, 0, 3))
    return {"incumbent_state": st, "mod": mod, "hot_ws2": st["ws2"], "out": out,
            "hot_code": hot_code, "probe": probe, "grid": grid, "grid3": int(st["sms"]) * 2,
            "metadata": {"variant": "hotw2v256_sw128_register_transpose",
                "source_sha256": digest, "cuda_bytes": len(cuda.encode()),
                "geometry": "token64/output128/K256; 320 threads, ring2 hot/ring3 cold",
                "threshold": 32, "dynamic_smem": h.SMEM, "hot_ring_smem": HOT_SMEM,
                "cold_ring_smem": h.COLD_SMEM, "grid": grid, "shared_alignment": 1024,
                "hot_weight_repack": False,
                "x_staging": "2 LDG128 + 8 shuffles + 2 STS128 per original K16 piece",
                "b_layout": "K-major SW128, SBO4096, internal RS K order preserved",
                "compiled": dict(zip(("registers", "local_bytes", "static_smem", "dynamic_smem", "occupancy", "threads"), attrs)),
                "occupancy_gate": "grid264 requires >=2 CTAs/SM; enforced in prepare",
                "error_flags": "incumbent_state['err']; inspect outside capture, all four must remain zero",
                "single_y_ss": True, "K1_K3": "unchanged incumbent binary", "fallback": False,
                "gpu_numerics_validated": False}}


def run(data, state):
    return _vector()._base().run(data, state)


def source_audit():
    v = _vector()
    h = v._base()
    prior = v.source_audit()
    assert prior["source_sha256"] == "56d44f47ce108dc859975f0f659eebebb212f189ab7611c9bbd013af3441eb8a"
    original, cpp0 = v.source()
    cuda, cpp = source()
    assert cpp == cpp0 and HOT_SMEM == 102432 < h.SMEM == 106160
    assert len(cuda.encode()) < 1_000_000 and "namespace hotw2v256_lab {" in cuda
    for marker in ("void cold_item(", "void hc_item(", "void flush_item(",
                   "void hw_gemm37(", "void hw_segs37("):
        assert h._function(original, marker, True) == h._function(cuda, marker, True)
    # Recover the original GPU-valid K256 hot consumer, changing only its name
    # and shared-B descriptor/layout increments; A decode/global math unchanged.
    raw, _ = h._base()
    expected = h._function(raw, "void hot_item(", True)
    expected = h._once(expected, "void hot_item(", "void hw_hot_item(")
    expected = h._once(expected, "wg_desc_sa(sa(st + SW), 4096u)", "hwv_desc_sa(sa(st + 18432))")
    for old, new in (("dlo + 64 * i + 16", "dlo + 64 * i + 2"),
                     ("dlo + 64 * i + 32", "dlo + 64 * i + 4"),
                     ("dlo + 64 * i + 48", "dlo + 64 * i + 6")):
        expected = h._once(expected, old, new)
    assert expected == h._function(cuda, "void hw_hot_item(", True)
    assert "hf" not in _producer() and "mbar_expect_tx(fb, 18432)" in _producer()
    assert "for (int s = s0; s < s1; ++s, ++it)" in _producer()
    assert "constexpr int NST = 2, STG = 51200;" in cuda
    # Reverse only this sibling's allowed changes, including namespace. This
    # recovers the entire frozen vector source, not just the cold worker bodies.
    restored = cuda.replace("hotw2v256_lab", "hotw2v_lab")
    restored = h._once(restored, h._function(restored, "void hw_hot_item(", True),
                       h._function(original, "void hw_hot_item(", True))
    restored = h._once(restored, _producer(), h._once(h._PRODUCER, v._OLD_COPY, v._NEW_COPY))
    restored = h._once(restored, "0x4000010000010000ull", "0x4000008000010000ull")
    restored = h._once(restored, "// B K-major SW128; LBO16 ignored, SBO4096, zero pattern base phase.",
                       "// B K-major SW128; LBO16 ignored, SBO2048, zero pattern base phase.")
    restored = h._once(restored, "constexpr int NST = 2, STG = 51200;",
                       "constexpr int NST = 3, STG = 25600;")
    assert restored == original

    def sw(u): return u ^ ((u & 0x1c0) >> 3)
    def layout(n, k): return sw(2048 * (n // 8) + 512 * (k // 64) + 64 * (n % 8) + k % 64)
    assert sorted(layout(n, k) for n in range(64) for k in range(256)) == list(range(16384))
    comparisons = 0
    for ntok in range(1, 65):
        padded = (ntok + 7) & ~7
        old, new = [None] * 16384, [None] * 16384
        for first in range(0, padded * 16, 32):
            assert first + 31 < padded * 16
            for quad in range(0, 32, 4):
                piece = first + quad
                n, i = piece >> 4, (piece >> 2) & 3
                lanes = []
                for t in range(4):
                    vals = [(n, 64 * i + 16 * t + b) if n < ntok else 0 for b in range(16)]
                    words = [tuple(vals[q:q + 2]) for q in range(0, 16, 2)]
                    lanes.append(words)
                    for q, word in enumerate(words):
                        u = 2 * ((n >> 3) * 1024 + i * 256 + 4 * (n & 7) + t + 32 * q)
                        old[u:u + 2] = word
                for half in range(2):
                    transposed = v._transpose4([x[4 * half:4 * half + 4] for x in lanes])
                    for t in range(4):
                        k = 64 * i + 8 * (4 * half + t)
                        dst = layout(n, k)
                        assert dst % 8 == 0
                        assert [layout(n, k + b) for b in range(8)] == list(range(dst, dst + 8))
                        assert new[dst:dst + 8] == [None] * 8
                        new[dst:dst + 8] = [x for pair in transposed[t] for x in pair]
        for base in (0, 1024, 16384):
            for slot in range(2):
                start = base + slot * HOT_STG + 18432
                assert start % 1024 == 0
                d0 = (start >> 4) | 0x4000010000010000
                for i in range(4):
                    for q in range(4):
                        desc = d0 + 64 * i + 2 * q
                        addr = (desc & 0x3fff) << 4
                        assert addr == start + 1024 * i + 32 * q
                        assert ((desc >> 16) & 0x3fff) == 1 and ((desc >> 32) & 0x3fff) == 256
                        assert ((desc >> 49) & 7) == 0 and (desc >> 62) == 1
                        for n in range(padded):
                            for k in range(16):
                                old_u = (1024 * i + 256 * q + 4096 * (n // 8) + 16 * (n % 8) + 2 * (k % 8) + 128 * (k // 8)) // 2
                                b = addr + 128 * (n % 8) + 4096 * (n // 8) + 2 * k
                                read = (b ^ ((b & 0x380) >> 3)) - start
                                assert read % 2 == 0 and new[read // 2] == old[old_u]
                                comparisons += 1
        assert sum(x is not None for x in new) == padded * 256
    # K256 source rows and both vector loads stay within the same padded row.
    for s in (0, 1, 15):
        for c0 in (0, 1, 127, 959):
            for n in range(64):
                for i in range(4):
                    for t in range(4):
                        p = ((s * 2560 + c0 + n) * 264 + i * 64 + t * 16)
                        assert 2 * p % 16 == 0 and (p + 15) // 264 == p // 264
    # Every logical split covers the same K256 stages and completion total16.
    for s0 in range(16):
        for s1 in range(s0 + 1, 17):
            assert [k for s in range(s0, s1) for k in range(256)] == [k for s in range(s0, s1) for hf in range(2) for k in range(hf * 128, (hf + 1) * 128)]
    return {"status": "source_audit_only", "gpu_validated": False,
            "source_sha256": hashlib.sha256((cuda + cpp).encode()).hexdigest(),
            "vector_source_sha256": prior["source_sha256"], "cuda_bytes": len(cuda.encode()),
            "descriptor_label_comparisons": comparisons, "token_remainders": 64,
            "dynamic_smem": h.SMEM, "hot_ring_smem": HOT_SMEM,
            "hot_weight_repack": False, "physical_stages_per_full_k": 16}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", action="store_true")
    if not parser.parse_args().source_audit:
        parser.error("--source-audit only; parent owns GPU validation")
    print(json.dumps(source_audit(), indent=2))

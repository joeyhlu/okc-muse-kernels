"""LAB hotW2 vector stores, preserving the existing RS K permutation.

prepare(weights, incumbent_module, incumbent_state); run(data, state) -> BF16.
Only hot B staging/layout and shared alignment change. K1/K3, lossless weight
pack, dequantization, scheduling, split counters and FP32 y/ss are inherited.
No GPU work on import. --source-audit needs only Python's standard library.
"""

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _base():
    # LAB file import, without sys.modules writes or edits to a loaded module.
    path = Path(__file__).with_name("fresh_k2a_hotw2.py")
    spec = importlib.util.spec_from_file_location("fresh_hotw2v_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HELPERS = r"""
// Four adjacent lanes own original inter K16 segments. Transpose their four
// u32 columns into each lane's contiguous internal-K8 vector. Applying this
// twice preserves the RS weight pack's existing within-K64 permutation.
__device__ __forceinline__ uint4 hwv_transpose4(uint4 v, int t) {
  uint32_t a = __shfl_xor_sync(FULL, (t & 1) ? v.x : v.y, 1, 4);
  uint32_t b = __shfl_xor_sync(FULL, (t & 1) ? v.z : v.w, 1, 4);
  if (t & 1) { v.x = a; v.z = b; }
  else       { v.y = a; v.w = b; }
  a = __shfl_xor_sync(FULL, (t & 2) ? v.x : v.z, 2, 4);
  b = __shfl_xor_sync(FULL, (t & 2) ? v.y : v.w, 2, 4);
  if (t & 2) { v.x = a; v.y = b; }
  else       { v.z = a; v.w = b; }
  return v;
}
__device__ __forceinline__ void hwv_store16(uint16_t* p, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
    :: "r"(sa(p)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint64_t hwv_desc_sa(uint32_t start) {
  // B K-major SW128; LBO16 ignored, SBO2048, zero pattern base phase.
  return uint64_t((start & 0x3ffffu) >> 4) | uint64_t(0x4000008000010000ull);
}
"""

_OLD_COPY = r"""          uint32_t* xp = reinterpret_cast<uint32_t*>(dstp + 9216);
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
"""

_NEW_COPY = r"""          uint16_t* xp = reinterpret_cast<uint16_t*>(dstp + 9216);
          const int padded = (ntok + 7) & ~7;
          // padded is a multiple of eight: all 32 lanes execute every shuffle.
          for (int piece = lane; piece < padded * 8; piece += 32) {
            const int n = piece >> 3, i = (piece >> 2) & 1, tt = piece & 3;
            uint4 a = make_uint4(0,0,0,0), b = a;
            if (n < ntok) {
              const __nv_bfloat16* src = P.inter + ((size_t)s * P3 + c0 + n) * XST
                                       + hf * 128 + i * 64 + tt * 16;
              const uint4* vp = reinterpret_cast<const uint4*>(src);
              a = __ldg(vp); b = __ldg(vp + 1);
            }
            a = hwv_transpose4(a, tt);
            b = hwv_transpose4(b, tt);
            // Here k is the WGMMA internal K, not the original inter column.
            const int u0 = (n >> 3) * 1024 + i * 512 + (n & 7) * 64 + 8 * tt;
            const int u1 = u0 + 32;
            const int h0 = u0 ^ ((u0 & 0x1c0) >> 3);
            const int h1 = u1 ^ ((u1 & 0x1c0) >> 3);
            hwv_store16(xp + h0, a);
            hwv_store16(xp + h1, b);
          }
"""


def source(module=None):
    h = _base()
    cuda, cpp = h.source(module)
    hot = h._function(cuda, "void hw_hot_item(", True)
    new_hot = h._once(hot, "wg_desc_sa(sa(st + 9216), 2048u)", "hwv_desc_sa(sa(st + 9216))")
    for old, new in (("dlo + 64 * i + 16", "dlo + 64 * i + 2"),
                     ("dlo + 64 * i + 32", "dlo + 64 * i + 4"),
                     ("dlo + 64 * i + 48", "dlo + 64 * i + 6")):
        new_hot = h._once(new_hot, old, new)
    cuda = h._once(cuda, hot, _HELPERS + "\n" + new_hot)
    cuda = h._once(cuda, _OLD_COPY, _NEW_COPY)
    mixed = h._function(cuda, "void __launch_bounds__(320, 2) k2_hotw2(")
    new_mixed = h._once(mixed, "extern __shared__ __align__(128) uint8_t smem[];",
                        "extern __shared__ __align__(1024) uint8_t smem[];")
    cuda = h._once(cuda, mixed, new_mixed)
    return cuda.replace("hotw2_lab", "hotw2v_lab"), cpp


def prepare(weights, incumbent_module, incumbent_state):
    state = _base().prepare(weights, incumbent_module, incumbent_state, source_factory=source)
    state["metadata"].update(
        variant="hotw2v_sw128_register_transpose",
        x_staging="2 LDG128 + 8 warp shuffles + 2 STS128 per original K16 piece",
        b_layout="K-major SW128, SBO2048, internal RS K order preserved",
        shared_alignment=1024,
    )
    return state


def run(data, state):
    return _base().run(data, state)


def _transpose4(rows):
    """CPU mirror of the two shuffle stages; each row is one lane's uint4."""
    v = [list(r) for r in rows]
    for mask, pairs in ((1, ((0, 1), (2, 3))), (2, ((0, 2), (1, 3)))):
        old = [r[:] for r in v]
        for low, high in pairs:
            sent = [old[t][low if t & mask else high] for t in range(4)]
            for t in range(4):
                v[t][low if t & mask else high] = sent[t ^ mask]
    return v


def source_audit():
    h = _base()
    original, cpp0 = h.source()
    cuda, cpp = source()
    old_hash = hashlib.sha256((original + cpp0).encode()).hexdigest()
    assert old_hash == "aa04a93bc09f76ff069a17cefffd57d93b003e070e0db43da5ee720d77941c41"
    assert cpp == cpp0 and len(cuda.encode()) < 1_000_000
    assert "namespace hotw2v_lab {" in cuda and "hotw2_lab" not in cuda
    assert _OLD_COPY not in cuda and _NEW_COPY in cuda
    assert h.SMEM == 106160 and h.HOT_STG == 25600
    for marker in ("void cold_item(", "void hc_item(", "void flush_item(",
                   "void hw_gemm37(", "void hw_segs37("):
        assert h._function(original, marker, True) == h._function(cuda, marker, True)
    old_mixed = h._function(original, "void __launch_bounds__(320, 2) k2_hotw2(")
    new_mixed = h._function(cuda, "void __launch_bounds__(320, 2) k2_hotw2(")
    assert old_mixed.replace("__align__(128)", "__align__(1024)") == new_mixed
    assert cuda.count('"st.shared.v4.u32 [%0], {%1,%2,%3,%4};') == 1
    assert cuda.count("hwv_transpose4(a, tt)") == cuda.count("hwv_transpose4(b, tt)") == 1
    # Reverse exactly the allowed changes: the entire remaining source must
    # equal the frozen source, including every wait, epilogue and host launch.
    restored = cuda.replace("hotw2v_lab", "hotw2_lab")
    restored = h._once(restored, _HELPERS + "\n", "")
    restored = h._once(restored, _NEW_COPY, _OLD_COPY)
    restored = h._once(restored, h._function(restored, "void hw_hot_item(", True),
                       h._function(original, "void hw_hot_item(", True))
    restored = h._once(restored, new_mixed, old_mixed)
    assert restored == original

    def sw(u): return u ^ ((u & 0x1c0) >> 3)
    def layout(n, k): return sw(1024 * (n // 8) + 512 * (k // 64) + 64 * (n % 8) + k % 64)
    assert sorted(layout(n, k) for n in range(64) for k in range(128)) == list(range(8192))
    matrix = [[4 * t + q for q in range(4)] for t in range(4)]
    assert _transpose4(matrix) == [list(col) for col in zip(*matrix)]
    for first in range(0, 65536, 16):
        # Every possible BF16 bit pattern is a raw half of one shuffled word.
        bits = [[(first + 4 * t + q) | ((65535 - first - 4 * t - q) << 16)
                 for q in range(4)] for t in range(4)]
        assert _transpose4(bits) == [list(col) for col in zip(*bits)]
    comparisons = 0
    for ntok in range(1, 65):
        padded = (ntok + 7) & ~7
        old, new = [None] * 8192, [None] * 8192
        for first in range(0, padded * 8, 32):
            # A full warp reaches every shuffle, including zero-filled rows.
            assert first + 31 < padded * 8
            for quad in range(0, 32, 4):
                piece = first + quad
                n, i = piece >> 3, (piece >> 2) & 1
                lanes = []
                for t in range(4):
                    vals = [(n, 64 * i + 16 * t + b) if n < ntok else 0 for b in range(16)]
                    words = [tuple(vals[q:q + 2]) for q in range(0, 16, 2)]
                    lanes.append(words)
                    for q, word in enumerate(words):
                        u = 2 * ((n >> 3) * 512 + i * 256 + 4 * (n & 7) + t + 32 * q)
                        old[u:u + 2] = word
                for half in range(2):
                    transposed = _transpose4([v[4 * half:4 * half + 4] for v in lanes])
                    for t in range(4):
                        k = 64 * i + 8 * (4 * half + t)
                        dst = layout(n, k)
                        assert dst % 8 == 0
                        assert [layout(n, k + b) for b in range(8)] == list(range(dst, dst + 8))
                        assert new[dst:dst + 8] == [None] * 8
                        new[dst:dst + 8] = [v for pair in transposed[t] for v in pair]
        # Independent reconstruction from descriptor fields for all eight K16
        # instructions. Compare exact input labels against the GPU-valid old RS
        # layout, preserving its K permutation and operand order.
        for base in (0, 1024, 16384):
            for slot in range(3):
                start = base + slot * 25600 + 9216
                assert start % 1024 == 0
                d0 = (start >> 4) | 0x4000008000010000
                for i in range(2):
                    for q in range(4):
                        desc = d0 + 64 * i + 2 * q
                        addr = (desc & 0x3fff) << 4
                        assert addr == start + 1024 * i + 32 * q
                        assert ((desc >> 16) & 0x3fff) == 1
                        assert ((desc >> 32) & 0x3fff) == 128
                        assert ((desc >> 49) & 7) == 0 and (desc >> 62) == 1
                        for n in range(padded):
                            for k in range(16):
                                old_u = (1024 * i + 256 * q + 2048 * (n // 8) + 16 * (n % 8) + 2 * (k % 8) + 128 * (k // 8)) // 2
                                b = addr + 128 * (n % 8) + 2048 * (n // 8) + 2 * k
                                read = (b ^ ((b & 0x380) >> 3)) - start
                                assert read % 2 == 0
                                assert new[read // 2] == old[old_u]
                                comparisons += 1
        assert sum(v is not None for v in new) == padded * 128
    # Global address arithmetic is unchanged and every uint4 is 16B aligned.
    for s in (0, 1, 15):
        for hf in range(2):
            for c0 in (0, 1, 127, 959):
                for n in range(64):
                    for i in range(2):
                        for t in range(4):
                            p = ((s * 2560 + c0 + n) * 264 + hf * 128 + i * 64 + t * 16)
                            assert 2 * p % 16 == 0 and (p + 15) // 264 == p // 264
    return {"status": "source_audit_only", "gpu_validated": False,
            "source_sha256": hashlib.sha256((cuda + cpp).encode()).hexdigest(),
            "base_source_sha256": old_hash, "cuda_bytes": len(cuda.encode()),
            "descriptor_label_comparisons": comparisons, "token_remainders": 64,
            "dynamic_smem": h.SMEM, "hot_ring_smem": h.HOT_SMEM,
            "copy": "2 LDG128 + 8 shuffles + 2 STS128; unchanged RS weights"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", action="store_true")
    if not parser.parse_args().source_audit:
        parser.error("--source-audit only; parent owns GPU validation")
    print(json.dumps(source_audit(), indent=2))

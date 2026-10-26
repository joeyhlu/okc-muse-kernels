"""Independent lossless BF16 block format feeding a dense Triton TensorCore GEMM.

Interface: prepare(w[N,K]), run(x[T,K], state, out=None) -> out[T,N].
T=1..256 (aim: 32/128/256). BF16 operands, FP32 accumulate/reduce; output
defaults to BF16, caller may supply BF16 or FP32. No GPU or cloud actions here.

Format: K-major blocks of one weight row x 64 K elements. Each block contains
either 64 sign/mantissa bytes + 32 packed exponent-delta bytes (96 bytes), or
64 low bytes + 64 high bytes (128 bytes). One uint32 descriptor encodes its
32-byte payload offset, 8-bit exponent base and raw flag. No exception list or
numeric approximation. Whole-matrix raw fallback if metadata/padding would
make storage >= the original. Packed state never retains the source weights.

One geometry: BM=64, BN=64, BK=64, four warps, three stages.
Reconstruct just the current BF16 MMA tile; tl.dot consumes dense blocks.
T<=64 uses four K splits and preallocated FP32 partials plus a reduction kernel.
T>64 uses full K and stores output directly, avoiding partial-buffer traffic.
No full BF16 expansion, scalar dot loop, custom register ring or inline PTX.

CPU-only standard-library format audit:
    python kernels/lab/fresh_packed_dense.py --smoke
No CUDA/Triton imports occur until prepare/run; no dependency on old P1/P1M.
"""

from array import array
from dataclasses import dataclass
import math
import random
import struct
import sys

BLOCK_K = 64
BLOCK_M = 64
BLOCK_N = 64
SPLIT_K = 4
SPLIT_MAX_TOKENS = 64
MAX_TOKENS = 256
_torch = None
_triton = None
_gemm = None
_reduce = None


@dataclass
class BitFormat:
    """CPU oracle storage, not a cached GEMM result or a runtime unpack buffer."""
    n: int
    k: int
    packed: bool
    data: bytes
    descriptors: array
    stats: dict


def _stats(n, k, packed, compact, raw_blocks, payload_bytes, metadata_bytes):
    original = 2 * n * k
    stored = payload_bytes + metadata_bytes
    return dict(
        format="sm8_exp4_block64" if packed else "raw_bf16",
        shape=(n, k), original_bytes=original, stored_bytes=stored,
        payload_bytes=payload_bytes, metadata_bytes=metadata_bytes,
        compression_ratio=stored / original, saved_fraction=1.0 - stored / original,
        compressed_blocks=compact if packed else 0,
        raw_blocks=raw_blocks if packed else n * ((k + 63) // 64),
        block_k=BLOCK_K, block_m=BLOCK_M, block_n=BLOCK_N,
        split_k={"T<=64": SPLIT_K, "T>64": 1},
        # Scratch is input-dependent FP32 partial sums, not a second weight copy.
        workspace_bytes=SPLIT_K * SPLIT_MAX_TOKENS * n * 4,
    )


def pack_bits(bits, n, k):
    """Pure Python reference packer for uint16 BF16 bits in row-major [N,K]."""
    if n < 1 or k < 1 or len(bits) != n * k:
        raise ValueError("bits must have exactly N*K elements, with positive N,K")
    if 2 * n * k >= 1 << 28:
        raise ValueError("bounded format requires raw weights <256 MiB")
    payload = bytearray()
    descriptors = array("I")
    compact = 0
    groups = (k + 63) // 64
    for kb in range(groups):
        count = min(64, k - kb * 64)
        for row in range(n):
            block = [int(v) for v in bits[row * k + kb * 64:row * k + kb * 64 + count]]
            if any(v < 0 or v > 65535 for v in block):
                raise ValueError("expected uint16 BF16 bit patterns")
            exp = [(v >> 7) & 255 for v in block]
            base = min(exp)
            is_raw = max(exp) - base > 15
            offset = len(payload)
            descriptors.append(((offset // 32) << 9) | (base << 1) | int(is_raw))
            block.extend([0] * (64 - count))
            if is_raw:
                payload.extend(v & 255 for v in block)
                payload.extend(v >> 8 for v in block)
            else:
                compact += 1
                payload.extend(((v >> 8) & 128) | (v & 127) for v in block)
                delta = [e - base for e in exp] + [0] * (64 - count)
                payload.extend(delta[j] | (delta[j + 1] << 4) for j in range(0, 64, 2))
    raw_blocks = groups * n - compact
    # One representation only. Global fallback drops all attempted packed state.
    if len(payload) + 4 * len(descriptors) >= 2 * n * k:
        raw = bytearray()
        for v in bits:
            if not 0 <= int(v) <= 65535:
                raise ValueError("expected uint16 BF16 bit patterns")
            raw.extend((int(v) & 255, int(v) >> 8))
        return BitFormat(n, k, False, bytes(raw), array("I"),
                         _stats(n, k, False, 0, raw_blocks, len(raw), 0))
    return BitFormat(n, k, True, bytes(payload), descriptors,
                     _stats(n, k, True, compact, raw_blocks, len(payload), 4 * len(descriptors)))


def unpack_bits(fmt):
    """CPU bit oracle; runtime GEMM never calls this or allocates an unpacked W."""
    if not fmt.packed:
        return [fmt.data[i] | (fmt.data[i + 1] << 8) for i in range(0, len(fmt.data), 2)]
    bits = [0] * (fmt.n * fmt.k)
    for kb in range((fmt.k + 63) // 64):
        for row in range(fmt.n):
            desc = fmt.descriptors[kb * fmt.n + row]
            offset, base, raw = (desc >> 9) * 32, (desc >> 1) & 255, desc & 1
            for j in range(min(64, fmt.k - kb * 64)):
                first = fmt.data[offset + j]
                second = fmt.data[offset + 64 + (j if raw else j // 2)]
                if raw:
                    value = first | (second << 8)
                else:
                    delta = (second >> ((j & 1) * 4)) & 15
                    value = ((first & 128) << 8) | ((base + delta) << 7) | (first & 127)
                bits[row * fmt.k + kb * 64 + j] = value
    return bits


def _dependencies():
    """Lazy ordinary imports; no installs, module patching, or source loading."""
    global _torch, _triton
    if _torch is None:
        import torch
        import triton
        _torch, _triton = torch, triton
    return _torch, _triton


def _kernels():
    global _gemm, _reduce, tl
    if _gemm is not None:
        return _gemm, _reduce
    import triton as tr
    # A module global gives Triton's source/dependency resolver an ordinary
    # namespace; the JIT functions do not close over a function-local module.
    import triton.language as tl

    @tr.jit
    def gemm(X, Storage, Meta, Partial,
             M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, PACKED: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             SK: tl.constexpr, MAX_M: tl.constexpr):
        mi, ni, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        rows = mi * BM + tl.arange(0, BM)
        cols = ni * BN + tl.arange(0, BN)
        lane_k = tl.arange(0, BK)
        iters: tl.constexpr = tl.cdiv(K, BK * SK)
        acc = tl.zeros((BM, BN), tl.float32)
        for step in range(iters):
            kb = split * iters + step
            ks = kb * BK + lane_k
            x = tl.load(X + rows[:, None] * K + ks[None, :],
                        (rows[:, None] < M) & (ks[None, :] < K), other=0)
            if PACKED:
                descriptor = tl.load(Meta + kb * N + cols,
                                     (cols < N) & (kb * BK < K), other=0).to(tl.uint32)
                # Payload is bounded below 256 MiB, so byte offsets fit int32.
                offset = (descriptor >> 9).to(tl.int32) * 32
                base = (descriptor >> 1) & 255
                raw = (descriptor & 1) != 0
                valid = (cols[:, None] < N) & (ks[None, :] < K)
                first = tl.load(Storage + offset[:, None] + lane_k[None, :], valid, other=0).to(tl.uint32)
                second_lane = tl.where(raw[:, None], lane_k[None, :], lane_k[None, :] // 2)
                second = tl.load(Storage + offset[:, None] + 64 + second_lane, valid, other=0).to(tl.uint32)
                delta = (second >> ((lane_k[None, :] & 1) * 4)) & 15
                compact_bits = ((first & 128) << 8) | ((base[:, None] + delta) << 7) | (first & 127)
                raw_bits = first | (second << 8)
                bits = tl.where(valid, tl.where(raw[:, None], raw_bits, compact_bits), 0).to(tl.uint16)
                weight = bits.to(tl.bfloat16, bitcast=True)
            else:
                weight = tl.load(Storage + cols[:, None] * K + ks[None, :],
                                 (cols[:, None] < N) & (ks[None, :] < K), other=0)
            acc = tl.dot(x, tl.trans(weight), acc, out_dtype=tl.float32)
        if SK == 1:
            dest = rows[:, None] * N + cols[None, :]
        else:
            dest = split * MAX_M * N + rows[:, None] * N + cols[None, :]
        tl.store(Partial + dest, acc, (rows[:, None] < M) & (cols[None, :] < N))

    @tr.jit
    def reduce(Partial, Out, ELEMENTS: tl.constexpr, STRIDE: tl.constexpr,
               SK: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        value = tl.full((B,), 0.0, tl.float32)
        for split in tl.static_range(SK):
            value += tl.load(Partial + split * STRIDE + i, i < ELEMENTS, other=0)
        tl.store(Out + i, value, i < ELEMENTS)

    _gemm, _reduce = gemm, reduce
    return _gemm, _reduce


def prepare(w):
    """Losslessly pack BF16 [N,K] weights and preallocate fixed maximum-T scratch.

    Works on CPU tensors for format inspection too; run() requires CUDA.
    Prepare may synchronize on compression metadata, never on input activations.
    All temporary raw/packed copies are released on return. Caller owns w's lifetime.
    """
    torch, _ = _dependencies()
    if w.ndim != 2 or w.dtype != torch.bfloat16 or not w.is_contiguous():
        raise ValueError("w must be contiguous BF16 [N,K]; FP32 router is not supported")
    n, k = map(int, w.shape)
    if not (1 <= n <= 16384 and 1 <= k <= 16384) or 2 * n * k >= 1 << 28:
        raise ValueError("N,K=1..16384 and raw weights <256 MiB required")
    with torch.no_grad():
        kg = (k + 63) // 64
        # Temporaries are integer bits only: no BF16 rounding or float dequantization.
        padded = torch.zeros((n, kg * 64), dtype=torch.int32, device=w.device)
        padded[:, :k] = w.view(torch.int16).to(torch.int32) & 65535
        bits = padded.view(n, kg, 64).permute(1, 0, 2).reshape(-1, 64)
        exponents = (bits >> 7) & 255
        lanes = torch.arange(64, device=w.device)
        block_k = torch.arange(kg, device=w.device).repeat_interleave(n)
        valid = block_k[:, None] * 64 + lanes[None, :] < k
        lo = torch.where(valid, exponents, 255).amin(1)
        hi = torch.where(valid, exponents, 0).amax(1)
        compact = hi - lo <= 15
        lengths = torch.where(compact, 96, 128).to(torch.int64)
        compact_count = int(compact.sum().item())
        payload_bytes = int(lengths.sum().item())
        metadata_bytes = 4 * n * kg
        use_packed = payload_bytes + metadata_bytes < 2 * n * k
        if use_packed:
            offsets = torch.cumsum(lengths, dim=0) - lengths
            descriptor = ((offsets // 32) << 9) | (lo.to(torch.int64) << 1) | (~compact).to(torch.int64)
            meta = descriptor.to(torch.int32).contiguous()
            sign_mantissa = ((bits >> 8) & 128) | (bits & 127)
            first = torch.where(compact[:, None], sign_mantissa, bits & 255)
            delta = torch.where(valid, exponents - lo[:, None], 0)
            pair = delta[:, 0::2] | (delta[:, 1::2] << 4)
            packed_second = torch.zeros_like(bits)
            packed_second[:, :32] = pair
            second = torch.where(compact[:, None], packed_second, bits >> 8)
            encoded = torch.cat((first, second), dim=1).to(torch.uint8)
            record_lanes = torch.arange(128, device=w.device)
            storage = encoded[record_lanes[None, :] < lengths[:, None]].contiguous()
            stats = _stats(n, k, True, compact_count, n * kg - compact_count,
                           payload_bytes, metadata_bytes)
        else:
            storage, meta = w.detach(), None  # Only raw state: no packed duplicate.
            stats = _stats(n, k, False, 0, n * kg, 2 * n * k, 0)
        partials = torch.empty((SPLIT_K, SPLIT_MAX_TOKENS, n), dtype=torch.float32, device=w.device)
    return dict(data=storage, meta=meta, partials=partials, n=n, k=k,
                packed=use_packed, stats=stats, compression_stats=stats)


def run(x, state, out=None):
    """Compute x @ W.T. With out supplied, launches only; no tensor allocation.

    Warm once outside graph capture for Triton compilation. Workspace is single
    stream/nonreentrant. Optional allocation when out is None is a convenience
    for eager lab use; graph callers must provide their output buffer.
    """
    torch, tr = _dependencies()
    n, k = state["n"], state["k"]
    if x.ndim != 2 or x.shape[1] != k or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("x must be contiguous BF16 [T,K] matching weights")
    m = int(x.shape[0])
    if not 1 <= m <= MAX_TOKENS:
        raise ValueError("this bounded lab supports T=1..256")
    if not x.is_cuda or x.device != state["data"].device:
        raise ValueError("x and prepared weights must share a CUDA device")
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if (tuple(out.shape) != (m, n) or out.dtype not in (torch.bfloat16, torch.float32)
            or out.device != x.device or not out.is_contiguous()):
        raise ValueError("out must be contiguous BF16/FP32 [T,N] on x.device")
    if out is x or out is state["data"] or out is state["partials"]:
        raise ValueError("out must not alias inputs or scratch")
    gemm, reduce = _kernels()
    meta = state["meta"] if state["packed"] else state["data"]
    splits = SPLIT_K if m <= SPLIT_MAX_TOKENS else 1
    dest = state["partials"] if splits > 1 else out
    with torch.cuda.device(x.device):
        gemm[(tr.cdiv(m, BLOCK_M), tr.cdiv(n, BLOCK_N), splits)](
            x, state["data"], meta, dest, m, n, k, state["packed"],
            BLOCK_M, BLOCK_N, BLOCK_K, splits, SPLIT_MAX_TOKENS, num_warps=4, num_stages=3,
        )
        if splits > 1:
            reduce[(tr.cdiv(m * n, 256),)](
                state["partials"], out, m * n, SPLIT_MAX_TOKENS * n, splits, 256, num_warps=4,
            )
    return out


def _bf16_bits(value):
    """Finite FP32 -> BF16 round-to-nearest-even for CPU sample generation."""
    u = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 65535


def smoke():
    """Exhaustive representation audit, ragged/mixed/raw cases and a CPU dot oracle."""
    import ast
    import json
    from pathlib import Path
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    rng = random.Random(5353)
    cases = [("all_bit_patterns", list(range(65536)), 256, 256)]
    for n, k in ((256, 512), (129, 259), (1, 1), (7, 63), (3, 65)):
        values = [_bf16_bits(rng.gauss(0, 1 / math.sqrt(k))) for _ in range(n * k)]
        cases.append((f"normal_{n}x{k}", values, n, k))
        mixed = values.copy()
        for i in range(0, len(mixed), 997):
            mixed[i] = _bf16_bits(2.**30)
        cases.append((f"outlier_{n}x{k}", mixed, n, k))
    uniform = [rng.randrange(65536) for _ in range(129 * 259)]
    cases.append(("uniform_bits_fallback", uniform, 129, 259))
    for n, k in ((256, 512), (129, 259)):
        values = [_bf16_bits(rng.gauss(0, .02)) for _ in range(n * k)]
        for row in range(0, n, 17):
            for col in range(0, k, 19):
                i = row * k + col
                v = struct.unpack("<f", struct.pack("<I", values[i] << 16))[0]
                values[i] = _bf16_bits(v * 2.**30)
        cases.append((f"parent_outlier_{n}x{k}", values, n, k))
    for k in (1024, 4096, 8192):
        values = [_bf16_bits(rng.gauss(0, 1 / math.sqrt(k))) for _ in range(16 * k)]
        cases.append((f"normal_specK_16x{k}", values, 16, k))
    special = [0x0000, 0x8000, 0x0001, 0x007F, 0x0080, 0x7F7F, 0x7F80, 0xFF80, 0x7FC1, 0xFFC1]
    cases.append(("special_bits", (special * 64)[:640], 10, 64))
    for name, bits, n, k in cases:
        fmt = pack_bits(bits, n, k)
        assert unpack_bits(fmt) == bits, name
        assert len(fmt.data) + 4 * len(fmt.descriptors) <= 2 * n * k, name
        assert fmt.stats["stored_bytes"] == len(fmt.data) + 4 * len(fmt.descriptors)
        if name == "uniform_bits_fallback":
            assert not fmt.packed and not fmt.descriptors
        if name.startswith(("normal_", "outlier_", "parent_outlier_")):
            print(name, json.dumps(fmt.stats), flush=True)
    # Every 4-bit exponent delta and every sign/mantissa code at two bases.
    for base in (0, 240):
        values = [((sm & 128) << 8) | ((base + delta) << 7) | (sm & 127)
                  for sm in range(256) for delta in range(16)]
        fmt = pack_bits(values, 64, 64)
        assert fmt.packed and unpack_bits(fmt) == values
    # Prove gather layout and all four split partitions cover each true K once.
    for splits in (1, SPLIT_K):
        for k in (1, 63, 64, 65, 259, 512, 1024, 4096, 8192):
            iters = (k + 64 * splits - 1) // (64 * splits)
            visited = [s * iters * 64 + step * 64 + lane
                       for s in range(splits) for step in range(iters) for lane in range(64)
                       if s * iters * 64 + step * 64 + lane < k]
            assert visited == list(range(k))
    n, k = 3, 259
    original = [_bf16_bits(rng.gauss(0, .1)) for _ in range(n * k)]
    original[67] = _bf16_bits(2.**30)
    restored = unpack_bits(pack_bits(original, n, k))
    x = [_bf16_bits(rng.gauss(0, 1)) for _ in range(k)]
    def to_float(u):
        return struct.unpack("<f", struct.pack("<I", u << 16))[0]
    for row in range(n):
        before = sum(to_float(x[j]) * to_float(original[row * k + j]) for j in range(k))
        after = sum(to_float(x[j]) * to_float(restored[row * k + j]) for j in range(k))
        assert before == after
    assert _torch is None and _triton is None, "CPU smoke imported GPU dependencies"
    print("PASS CPU: all 65536 BF16 codes, delta/sign/mantissa combinations, ragged/outliers/raw fallback, K partition, dot oracle")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.parse_args()
    smoke()

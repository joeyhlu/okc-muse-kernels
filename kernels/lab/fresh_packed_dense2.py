"""Source-only v2: contiguous 64x64 BF16 weight records feeding dense tl.dot.

Same prepare(w), run(x,state,out=None) interface, M64/N64/K64 geometry and
T<=64 four-way split policy as v1. Only the packed weight layout/load/decode
changes: one uniform raw/compact tag and offset per full tile, affine uint32
loads, four/eight-way register bit extraction, no per-row payload gathers.
No TMA/WGMMA claim: generated code and performance must be checked by parent.

Compact tile = 4096 sign/mantissa bytes + 2048 exponent-delta bytes. A tile
whose exponent range exceeds 15 stores 8192 raw bytes, uniformly. Descriptor
is (payload_offset_in_2048B_units << 9) | (base_exponent << 1) | raw_flag.
Whole-matrix raw BF16 fallback if payload+metadata >= original weight bytes.
State retains one weight representation only, plus input-dependent scratch.
Same-geometry raw control: prepare_raw(w), or prepare(w, force_raw=True),
then the identical run(x,state,out=buf). No extra raw kernel or tiling change.

CPU-only: python kernels/lab/fresh_packed_dense2.py --smoke
No dependencies imported or GPU commands run by this CLI.
"""

from array import array
from dataclasses import dataclass
import math
import random
import struct

BM = BN = BK = 64
SPLITS = 4
SPLIT_MAX_T = 64
MAX_T = 256
_torch = _triton = _gemm = _reduce = None


@dataclass
class TileFormat:
    n: int
    k: int
    packed: bool
    data: bytes
    descriptors: array
    stats: dict


def _stats(n, k, packed, compact, total, payload, metadata):
    raw = 2 * n * k
    return dict(
        format="sm8_exp4_tile64x64_u32" if packed else "raw_bf16",
        shape=(n, k), original_bytes=raw, stored_bytes=payload + metadata,
        payload_bytes=payload, metadata_bytes=metadata,
        compression_ratio=(payload + metadata) / raw,
        saved_fraction=1 - (payload + metadata) / raw,
        compressed_tiles=compact if packed else 0,
        raw_tiles=total - compact if packed else total,
        tile_count=total, block_m=BM, block_n=BN, block_k=BK,
        split_k={"T<=64": SPLITS, "T>64": 1},
        workspace_bytes=SPLITS * SPLIT_MAX_T * n * 4,
    )


def pack_bits(bits, n, k):
    """Standard-library bit oracle; accepts arbitrary uint16 BF16 patterns."""
    if n < 1 or k < 1 or len(bits) != n * k or 2 * n * k >= 1 << 28:
        raise ValueError("positive N,K, exactly N*K bits, raw size <256 MiB required")
    if any(not 0 <= int(v) <= 65535 for v in bits):
        raise ValueError("expected uint16 bit patterns")
    ng, kg = (n + 63) // 64, (k + 63) // 64
    payload, descriptors = bytearray(), array("I")
    compact_count = 0
    for kb in range(kg):
        for nb in range(ng):
            tile, valid = [0] * 4096, [False] * 4096
            exps = []
            for r in range(min(64, n - nb * 64)):
                for c in range(min(64, k - kb * 64)):
                    v = int(bits[(nb * 64 + r) * k + kb * 64 + c])
                    tile[r * 64 + c], valid[r * 64 + c] = v, True
                    exps.append((v >> 7) & 255)
            base = min(exps)
            raw = max(exps) - base > 15
            assert len(payload) % 2048 == 0
            descriptors.append(((len(payload) // 2048) << 9) | (base << 1) | int(raw))
            if raw:
                payload.extend(struct.pack("<4096H", *tile))
            else:
                compact_count += 1
                payload.extend(((v >> 8) & 128) | (v & 127) for v in tile)
                delta = [(((v >> 7) & 255) - base) if hit else 0 for v, hit in zip(tile, valid)]
                payload.extend(delta[i] | (delta[i + 1] << 4) for i in range(0, 4096, 2))
    total = ng * kg
    if len(payload) + 4 * total >= 2 * n * k:
        raw = bytearray()
        for v in bits:
            raw.extend((int(v) & 255, int(v) >> 8))
        return TileFormat(n, k, False, bytes(raw), array("I"),
                          _stats(n, k, False, 0, total, len(raw), 0))
    return TileFormat(n, k, True, bytes(payload), descriptors,
                      _stats(n, k, True, compact_count, total, len(payload), 4 * total))


def unpack_bits(fmt):
    if not fmt.packed:
        return [fmt.data[i] | (fmt.data[i + 1] << 8) for i in range(0, len(fmt.data), 2)]
    n, k = fmt.n, fmt.k
    ng, kg = (n + 63) // 64, (k + 63) // 64
    restored = [0] * (n * k)
    for kb in range(kg):
        for nb in range(ng):
            descriptor = fmt.descriptors[kb * ng + nb]
            offset, base, raw = (descriptor >> 9) * 2048, (descriptor >> 1) & 255, descriptor & 1
            for r in range(min(64, n - nb * 64)):
                for c in range(min(64, k - kb * 64)):
                    lane = r * 64 + c
                    if raw:
                        value = fmt.data[offset + 2 * lane] | (fmt.data[offset + 2 * lane + 1] << 8)
                    else:
                        sm = fmt.data[offset + lane]
                        exp_byte = fmt.data[offset + 4096 + lane // 2]
                        exp = base + ((exp_byte >> (4 * (lane & 1))) & 15)
                        value = ((sm & 128) << 8) | (exp << 7) | (sm & 127)
                    restored[(nb * 64 + r) * k + kb * 64 + c] = value
    return restored


def _deps():
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
    import triton.language as tl

    @tr.jit
    def gemm(X, Storage, Meta, Dest, M: tl.constexpr, N: tl.constexpr,
             K: tl.constexpr, PACKED: tl.constexpr, SK: tl.constexpr):
        mi, ni, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        local_n, local_k = tl.arange(0, 64), tl.arange(0, 64)
        rows, cols = mi * 64 + local_n, ni * 64 + local_n
        iters: tl.constexpr = tl.cdiv(K, 64 * SK)
        ng: tl.constexpr = tl.cdiv(N, 64)
        acc = tl.zeros((64, 64), tl.float32)
        for step in range(iters):
            kb = split * iters + step
            ks = kb * 64 + local_k
            x = tl.load(X + rows[:, None] * K + ks[None, :],
                        (rows[:, None] < M) & (ks[None, :] < K), other=0)
            if PACKED:
                # Scalar descriptor, uniform over CTA. Offset alignment is a
                # format invariant, not an assumption about data values.
                desc = tl.load(Meta + kb * ng + ni, kb * 64 < K, other=0).to(tl.uint32)
                offset = tl.multiple_of(((desc >> 9) * 512).to(tl.int32), 512)
                base = (desc >> 1) & 255
                if (desc & 1) != 0:
                    # Unique affine word loads: 32 words per row, two BF16/word.
                    words = tl.load(Storage + offset + local_n[:, None] * 32 + tl.arange(0, 32)[None, :]).to(tl.uint32)
                    halves = (words[:, :, None] >> (tl.arange(0, 2)[None, None, :] * 16)) & 65535
                    bits = tl.reshape(halves, (64, 64))
                else:
                    # No duplicated byte-addressed global loads. Unpack the
                    # 1024 SM words and 512 exponent words only after loading.
                    sm_words = tl.load(Storage + offset + local_n[:, None] * 16 + tl.arange(0, 16)[None, :]).to(tl.uint32)
                    exp_words = tl.load(Storage + offset + 1024 + local_n[:, None] * 8 + tl.arange(0, 8)[None, :]).to(tl.uint32)
                    sm = (sm_words[:, :, None] >> (tl.arange(0, 4)[None, None, :] * 8)) & 255
                    delta = (exp_words[:, :, None] >> (tl.arange(0, 8)[None, None, :] * 4)) & 15
                    sm, delta = tl.reshape(sm, (64, 64)), tl.reshape(delta, (64, 64))
                    bits = ((sm & 128) << 8) | ((base + delta) << 7) | (sm & 127)
                valid = (cols[:, None] < N) & (ks[None, :] < K)
                bits = tl.where(valid, bits, 0).to(tl.uint16)
                weight = bits.to(tl.bfloat16, bitcast=True)
            else:
                weight = tl.load(Storage + cols[:, None] * K + ks[None, :],
                                 (cols[:, None] < N) & (ks[None, :] < K), other=0)
            acc = tl.dot(x, tl.trans(weight), acc, out_dtype=tl.float32)
        offset_out = rows[:, None] * N + cols[None, :]
        if SK != 1:
            offset_out += split * 64 * N
        tl.store(Dest + offset_out, acc, (rows[:, None] < M) & (cols[None, :] < N))

    @tr.jit
    def reduce(Partial, Out, COUNT: tl.constexpr, N: tl.constexpr):
        i = tl.program_id(0) * 256 + tl.arange(0, 256)
        acc = tl.full((256,), 0, tl.float32)
        for split in tl.static_range(4):
            acc += tl.load(Partial + split * 64 * N + i, i < COUNT, other=0)
        tl.store(Out + i, acc, i < COUNT)
    _gemm, _reduce = gemm, reduce
    return gemm, reduce


def prepare(w, *, force_raw=False):
    """Weight-only packing. force_raw is a same-geometry diagnostic, not tuning."""
    torch, _ = _deps()
    if w.ndim != 2 or w.dtype != torch.bfloat16 or not w.is_contiguous():
        raise ValueError("expected contiguous BF16 W[N,K]")
    n, k = map(int, w.shape)
    if not (1 <= n <= 16384 and 1 <= k <= 16384) or 2 * n * k >= 1 << 28:
        raise ValueError("bounded N,K=1..16384; raw W <256 MiB")
    ng, kg = (n + 63) // 64, (k + 63) // 64
    total = ng * kg
    with torch.no_grad():
        use_packed = False
        if not force_raw:
            padded = torch.zeros((ng * 64, kg * 64), dtype=torch.int32, device=w.device)
            padded[:n, :k] = w.view(torch.int16).to(torch.int32) & 65535
            bits = padded.view(ng, 64, kg, 64).permute(2, 0, 1, 3).reshape(total, 4096)
            ni = torch.arange(ng, device=w.device).repeat(kg)
            ki = torch.arange(kg, device=w.device).repeat_interleave(ng)
            lane = torch.arange(4096, device=w.device)
            valid = ((ni[:, None] * 64 + lane[None, :] // 64 < n)
                     & (ki[:, None] * 64 + lane[None, :] % 64 < k))
            exps = (bits >> 7) & 255
            lo = torch.where(valid, exps, 255).amin(1)
            hi = torch.where(valid, exps, 0).amax(1)
            compact = hi - lo <= 15
            lengths = torch.where(compact, 1536, 2048).to(torch.int64)
            compact_count = int(compact.sum().item())
            payload_bytes = int(lengths.sum().item()) * 4
            use_packed = payload_bytes + 4 * total < 2 * n * k
            if use_packed:
                offset_words = torch.cumsum(lengths, 0) - lengths
                desc = ((offset_words // 512) << 9) | (lo.to(torch.int64) << 1) | (~compact).to(torch.int64)
                meta = desc.to(torch.int32).contiguous()
                sm = ((bits >> 8) & 128) | (bits & 127)
                sm = sm.reshape(total, 1024, 4)
                sm_words = sm[:, :, 0] | (sm[:, :, 1] << 8) | (sm[:, :, 2] << 16) | (sm[:, :, 3] << 24)
                delta = torch.where(valid, exps - lo[:, None], 0).reshape(total, 512, 8)
                exp_words = delta[:, :, 0].clone()
                for j in range(1, 8):
                    exp_words |= delta[:, :, j] << (4 * j)
                compact_words = torch.cat((sm_words, exp_words, torch.zeros_like(exp_words)), 1)
                raw_words = bits[:, 0::2] | (bits[:, 1::2] << 16)
                encoded = torch.where(compact[:, None], compact_words, raw_words)
                word_lane = torch.arange(2048, device=w.device)
                storage = encoded[word_lane[None, :] < lengths[:, None]].contiguous()
                stats = _stats(n, k, True, compact_count, total, payload_bytes, 4 * total)
        if not use_packed:
            storage, meta = w.detach(), None
            stats = _stats(n, k, False, 0, total, 2 * n * k, 0)
        partials = torch.empty((4, 64, n), dtype=torch.float32, device=w.device)
    return dict(data=storage, meta=meta, packed=use_packed, partials=partials,
                n=n, k=k, stats=stats, compression_stats=stats)


def prepare_raw(w):
    """Raw BF16 control with identical geometry, K splits and output handling."""
    return prepare(w, force_raw=True)


def run(x, state, out=None):
    """Return x@W.T; no tensor allocation when caller supplies nonaliasing out."""
    torch, tr = _deps()
    n, k = state["n"], state["k"]
    if (x.ndim != 2 or x.shape[1] != k or x.dtype != torch.bfloat16
            or not x.is_contiguous() or not x.is_cuda or x.device != state["data"].device):
        raise ValueError("x must be contiguous CUDA BF16 [T,K] on prepared device")
    m = int(x.shape[0])
    if not 1 <= m <= 256:
        raise ValueError("T=1..256 required")
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if (tuple(out.shape) != (m, n) or out.dtype not in (torch.bfloat16, torch.float32)
            or out.device != x.device or not out.is_contiguous()):
        raise ValueError("out must be contiguous BF16/FP32 [T,N] on x.device")
    if out is x or out is state["data"] or out is state["partials"]:
        raise ValueError("out must not alias input or state")
    gemm, reduce = _kernels()
    splits = 4 if m <= 64 else 1
    dest = state["partials"] if splits > 1 else out
    meta = state["meta"] if state["packed"] else state["data"]
    with torch.cuda.device(x.device):
        gemm[(tr.cdiv(m, 64), tr.cdiv(n, 64), splits)](
            x, state["data"], meta, dest, m, n, k, state["packed"], splits,
            num_warps=4, num_stages=3,
        )
        if splits > 1:
            reduce[(tr.cdiv(m * n, 256),)](state["partials"], out, m * n, n, num_warps=4)
    return out


def _bf16_bits(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 65535


def smoke():
    import ast
    import json
    from pathlib import Path
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    rng = random.Random(5353)
    cases = [("all_codes", list(range(65536)), 256, 256)]
    # Rearrange the same exhaustive set so every whole tile fits 16 exponents.
    # This exercises all BF16 codes through the compact uint32 decoder too.
    all_compact = [0] * 65536
    for kb in range(4):
        for nb in range(4):
            base = 16 * (kb * 4 + nb)
            for lane in range(4096):
                sm, exp = lane & 255, base + lane // 256
                value = ((sm & 128) << 8) | (exp << 7) | (sm & 127)
                all_compact[(nb * 64 + lane // 64) * 256 + kb * 64 + lane % 64] = value
    cases.append(("all_codes_compact", all_compact, 256, 256))
    cases.append(("packed_ragged", [_bf16_bits(.01)] * (127 * 257), 127, 257))
    # Every tile includes the full delta range, and all sign/mantissa values.
    for base in (0, 240):
        values = [((sm & 128) << 8) | ((base + d) << 7) | (sm & 127)
                  for sm in range(256) for d in range(16)]
        cases.append((f"all_delta_sm_{base}", values * 4, 128, 128))
    for n, k in ((256, 512), (129, 259), (7, 63), (3, 65), (1, 1), (256, 8192)):
        values = [_bf16_bits(rng.gauss(0, 1 / math.sqrt(k))) for _ in range(n * k)]
        cases.append((f"normal_{n}x{k}", values, n, k))
        if n <= 256 and k <= 512:
            mixed = values.copy()
            for row in range(0, n, 17):
                for col in range(0, k, 19):
                    mixed[row * k + col] = _bf16_bits(2.**30)
            cases.append((f"outlier_{n}x{k}", mixed, n, k))
    # Known mixed-mode case with one exponent outlier in exactly one of four tiles.
    mixed = [_bf16_bits(.01)] * (128 * 128)
    mixed[0] = _bf16_bits(2.**30)
    cases.append(("one_raw_three_compact", mixed, 128, 128))
    cases.append(("random_raw", [rng.randrange(65536) for _ in range(129 * 259)], 129, 259))
    for name, original, n, k in cases:
        fmt = pack_bits(original, n, k)
        assert unpack_bits(fmt) == original, name
        assert len(fmt.data) + 4 * len(fmt.descriptors) <= 2 * n * k, name
        if name == "one_raw_three_compact":
            assert fmt.packed and fmt.stats["raw_tiles"] == 1 and fmt.stats["compressed_tiles"] == 3
        if name in ("all_codes_compact", "packed_ragged"):
            assert fmt.packed, name
        if name == "all_codes_compact":
            assert sorted(original) == list(range(65536))
        if name == "random_raw":
            assert not fmt.packed and not fmt.descriptors
        if fmt.packed:
            # Validate the actual uint32-load extraction used in the kernel,
            # independently of the byte-based oracle above.
            for desc in fmt.descriptors:
                offset, raw, base = (desc >> 9) * 2048, desc & 1, (desc >> 1) & 255
                assert offset % 2048 == 0
                size = 8192 if raw else 6144
                words = struct.unpack_from("<" + "I" * (size // 4), fmt.data, offset)
                for lane in range(4096):
                    if raw:
                        got = (words[lane // 2] >> (16 * (lane & 1))) & 65535
                        expected = fmt.data[offset + lane * 2] | (fmt.data[offset + lane * 2 + 1] << 8)
                    else:
                        sm = (words[lane // 4] >> (8 * (lane & 3))) & 255
                        delta = (words[1024 + lane // 8] >> (4 * (lane & 7))) & 15
                        got = ((sm & 128) << 8) | ((base + delta) << 7) | (sm & 127)
                        first = fmt.data[offset + lane]
                        second = fmt.data[offset + 4096 + lane // 2]
                        expected = ((first & 128) << 8) | ((base + ((second >> (4 * (lane & 1))) & 15)) << 7) | (first & 127)
                    assert got == expected, (name, lane)
        print(name, json.dumps(fmt.stats), flush=True)
    assert _torch is None and _triton is None
    print("PASS CPU tile64x64: exhaustive BF16 codes, raw/compact/mixed, uint32 extraction, ragged, no expansion")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.parse_args()
    smoke()

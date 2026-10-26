# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lab-only SM90 NVFP4/BF16 cuTile MoE: prepare(weights, tokens), run(data,state).

Modified minimal source port of FlashInfer fda67b315244f10c29d845f0e049c690a3500a5b,
flashinfer/fused_moe/cutile/fp4.py: byte decoders, W4A16 loader, grouped mainloop
and launcher. Only Hopper, group-16 NVFP4, BF16 activations, FP32 accumulation
and one FP32 global multiplier per expert are retained. No FlashInfer import.
Raw weight codes/scales remain unchanged; globals multiply the accumulator.

Routing/SwiGLU/Muse norm/combine reuse fresh_humming_adapter's Triton helpers
(no Humming dependency is invoked). No input work or TMA descriptors in prepare;
ct.launch constructs its runtime descriptors. Warm run before graph capture.
One state is for one T, device and stream; no concurrent use of its workspace.

CPU-only, no torch, Triton or GPU needed:
  python kernels/lab/fresh_cutile_adapter.py --smoke
  python kernels/lab/fresh_cutile_adapter.py --compiler-info
Explicit GPU correctness/replay runner (caller owns provisioning/lock/budget):
  python kernels/lab/fresh_cutile_adapter.py --gpu-check --small --tokens 1,16
  python kernels/lab/fresh_cutile_adapter.py --gpu-check --tokens 16,32,128,256
"""

import sys

# The diagnostic CLI must also run on machines with no CUDA/PyTorch installed.
_CPU_ONLY = __name__ == "__main__" and "--gpu-check" not in sys.argv

if not _CPU_ONLY:
    import torch
    import triton as tr
    import triton.language as tl
    import cuda.tile as ct
    import fresh_humming_adapter as H

    ConstInt = ct.Constant[int]
    _E4M3_NAN = float("nan")

    @ct.function
    def _decode_e2m1_bytes_to_bf16(packed, tile_n: ConstInt, tile_k: ConstInt):
        low = ct.expand_dims(packed & 0xF, axis=2)
        high = ct.expand_dims(packed >> 4, axis=2)
        selector = ct.reshape(ct.arange(2, dtype=ct.int32), (1, 1, 2))
        codes = ct.where(selector == 0, low, high)
        magnitude_code = ct.astype(codes & 0x7, ct.int32)
        magnitude_bits = ct.where(
            magnitude_code == 0, 0,
            0x3EC0 + (magnitude_code << 6) + ct.where(magnitude_code > 1, 0x40, 0),
        )
        sign_bits = ct.astype(codes & 0x8, ct.int32) << 12
        values = ct.bitcast(
            ct.astype(magnitude_bits, ct.uint16) | ct.astype(sign_bits, ct.uint16),
            ct.bfloat16,
        )
        return ct.reshape(values, (tile_n, tile_k))

    @ct.function
    def _decode_e4m3_bytes(bits):
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        normal = (1.0 + ct.astype(mantissa, ct.float32) * 0.125) * ct.exp2(
            ct.astype(exponent, ct.float32) - 7.0
        )
        subnormal = ct.astype(mantissa, ct.float32) * (2.0**-9)
        magnitude = ct.where(exponent == 0, subnormal, normal)
        # E4M3FN has a NaN code (0x7f/0xff); preserve it as well.
        magnitude = ct.where((bits & 0x7F) == 0x7F, _E4M3_NAN, magnitude)
        return ct.where((bits & 0x80) != 0, -magnitude, magnitude)

    @ct.function
    def _load_w4a16_weight_tile(W, SF, expert, n_block, k_tile,
                                TILE_N: ConstInt, TILE_K: ConstInt):
        weight_bytes = ct.reshape(ct.load(
            W, index=(expert, n_block, k_tile), shape=(1, TILE_N, TILE_K // 2),
            padding_mode=ct.PaddingMode.ZERO, latency=3, allow_tma=True,
        ), (TILE_N, TILE_K // 2))
        weight = _decode_e2m1_bytes_to_bf16(weight_bytes, TILE_N, TILE_K)
        scale_bytes = ct.reshape(ct.load(
            SF, index=(expert, n_block, k_tile), shape=(1, TILE_N, TILE_K // 16),
            padding_mode=ct.PaddingMode.ZERO, latency=3, allow_tma=True,
        ), (TILE_N, TILE_K // 16))
        return weight, _decode_e4m3_bytes(scale_bytes)

    @ct.function
    def _grouped_gemm_w4a16_impl(
        X, W, SF, GS, SORTED, EXPERTS, PADDED, OUT, top_k, grid_m,
        K_IN: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt,
    ):
        initial_m = ct.bid(0)
        n_block = ct.bid(1)
        padded = ct.gather(PADDED, ct.zeros((1,), dtype=ct.int32), padding_value=0).item()
        live_blocks = (padded + TILE_M - 1) // TILE_M
        iterations = (live_blocks - initial_m + grid_m - 1) // grid_m
        k_offsets = ct.arange(TILE_K, dtype=ct.int32)
        n_offsets = n_block * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
        for iteration in range(iterations):
            m_block = initial_m + iteration * grid_m
            if m_block * TILE_M < padded:
                m_offsets = m_block * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
                slots = ct.gather(SORTED, (m_offsets,), padding_value=0)
                expert = ct.gather(
                    EXPERTS, ct.full((1,), m_block, dtype=ct.int32), padding_value=0
                ).item()
                alpha = ct.gather(
                    GS, ct.full((1,), expert, dtype=ct.int32), padding_value=1.0
                ).item()
                rows = slots // top_k
                acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
                for k_tile in range((K_IN + TILE_K - 1) // TILE_K):
                    indices = ct.reshape(rows, (TILE_M, 1)) * K_IN + ct.reshape(
                        k_tile * TILE_K + k_offsets, (1, TILE_K)
                    )
                    activation = ct.gather(
                        X, (indices,), padding_value=0, check_bounds=True, latency=3
                    )
                    weight, sf = _load_w4a16_weight_tile(
                        W, SF, expert, n_block, k_tile, TILE_N, TILE_K
                    )
                    weight = ct.reshape(weight, (TILE_N, TILE_K // 16, 16)) * ct.expand_dims(
                        ct.astype(sf, activation.dtype), axis=2
                    )
                    weight = ct.reshape(weight, (TILE_N, TILE_K))
                    acc = ct.mma(activation, ct.permute(weight, (1, 0)), acc)
                ct.scatter(
                    OUT, (ct.reshape(slots, (TILE_M, 1)), ct.reshape(n_offsets, (1, TILE_N))),
                    ct.astype(acc * alpha, OUT.dtype), check_bounds=True,
                )

    @ct.kernel
    def _grouped_gemm_w4a16(
        X, W, SF, GS, SORTED, EXPERTS, PADDED, OUT, top_k, grid_m,
        K_IN: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt,
    ):
        _grouped_gemm_w4a16_impl(
            X, W, SF, GS, SORTED, EXPERTS, PADDED, OUT, top_k, grid_m,
            K_IN, TILE_M, TILE_N, TILE_K,
        )

    @tr.jit
    def _local_ids(Ids, Local, START: tl.constexpr, E: tl.constexpr,
                   SLOTS: tl.constexpr, B: tl.constexpr):
        slot = tl.program_id(0) * B + tl.arange(0, B)
        expert = tl.load(Ids + slot, slot < SLOTS, other=-1) - START
        expert = tl.where((expert >= 0) & (expert < E), expert, -1)
        tl.store(Local + slot, expert, slot < SLOTS)


def prepare(weights, tokens, *, block_m=16, tile_n=128, tile_k=128, occupancy=2):
    """Keep native packed weights/scales; allocate workspace. No input work/JIT launches."""
    if not isinstance(tokens, int) or isinstance(tokens, bool) or not 1 <= tokens <= 256:
        raise ValueError("bounded lab supports integer tokens in [1,256]")
    if block_m not in (8, 16, 32) or tile_n not in (128, 256) or tile_k not in (64, 128, 256):
        raise ValueError("unsupported lab tile; BM=8/16/32, BN=128/256, BK=64/128/256")
    if occupancy not in (1, 2, 3, 4):
        raise ValueError("occupancy must be 1..4")
    w13 = weights["w13"]
    if w13.ndim != 3 or not w13.is_cuda:
        raise ValueError("expected CUDA packed W13 [E,2D,D/2]")
    e, _, kb = w13.shape
    d = kb * 2
    if not 1 <= e <= 32 or d % 128 or d < 128 or d > 4096 or d % tile_k:
        raise ValueError("bounded lab requires E=1..32, D=128..4096 divisible by 128 and BK")
    packed = {}
    with torch.cuda.device(w13.device), torch.no_grad():
        for name, rows in (("w13", 2 * d), ("w2", d)):
            w, sf, gs = (weights[name + suffix] for suffix in ("", "_sf", "_gs"))
            if w.dtype != torch.uint8 or tuple(w.shape) != (e, rows, d // 2):
                raise TypeError(f"{name} must be raw uint8 [E,N,D/2]")
            if sf.dtype != torch.float8_e4m3fn or tuple(sf.shape) != (e, rows, d // 16):
                raise TypeError(f"{name}_sf must be E4M3 [E,N,D/16]")
            if gs.dtype != torch.float32 or tuple(gs.shape) != (e,):
                raise TypeError(f"{name}_gs must be FP32 [E], multiply convention")
            if any(v.device != w13.device for v in (w, sf, gs)):
                raise ValueError("weights must share one device")
            # Byte view, not a quantizer or a dtype conversion. Gate/up order unchanged.
            packed[name] = (w.contiguous(), sf.contiguous().view(torch.uint8), gs.contiguous())
        nw = weights["post_norm_w"]
        if nw.device != w13.device or tuple(nw.shape) != (d,) or nw.dtype != torch.bfloat16:
            raise ValueError("post_norm_w must be BF16 [D] on the weights' device")
        slots = tokens * 8
        capacity = tr.cdiv(slots + e * (block_m - 1), block_m) * block_m
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=w13.device)
        return dict(
            tokens=tokens, d=d, experts=e, block_m=block_m, tile_n=tile_n, tile_k=tile_k,
            # Retain a single hinted object: replace_hints each call would lose JIT cache.
            kernel=_grouped_gemm_w4a16.replace_hints(occupancy=occupancy),
            w13=packed["w13"], w2=packed["w2"], post_norm_w=nw.contiguous(),
            counts=empty((e,), torch.int32), local_ids=empty((tokens, 8), torch.int32),
            sorted_ids=empty((capacity,), torch.int32),
            expert_ids=empty((capacity // block_m,), torch.int32),
            padded=empty((1,), torch.int32),
            h13=empty((slots, 2 * d), torch.bfloat16),
            inter=empty((slots, d), torch.bfloat16), y=empty((slots, d), torch.bfloat16),
            z=empty((slots, d), torch.float32), out=empty((tokens, d), torch.bfloat16),
        )


def _grouped_gemm_w4a16_launch(x, output, projection, state, top_k):
    w, sf_bytes, gs = projection
    bm, bn, bk = state["block_m"], state["tile_n"], state["tile_k"]
    grid_m = max(1, min(state["sorted_ids"].numel() // bm, output.shape[0]))
    ct.launch(
        torch.cuda.current_stream(x.device), (grid_m, tr.cdiv(output.shape[1], bn)),
        state["kernel"],
        (x.view(-1), w, sf_bytes, gs, state["sorted_ids"], state["expert_ids"],
         state["padded"], output, top_k, grid_m, x.shape[1], bm, bn, bk),
    )


def run(data, state):
    """Rebuild routing every call, return BF16 [T,D]; remote slots contribute zero."""
    x, ids, wt = (data[k] for k in ("hidden_states", "topk_ids", "topk_weights"))
    t, d, e = state["tokens"], state["d"], state["experts"]
    if tuple(x.shape) != (t, d) or x.dtype != torch.bfloat16:
        raise ValueError("hidden_states must be BF16 [T,D] matching state")
    if tuple(ids.shape) != (t, 8) or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be int32/int64 [T,8]")
    if tuple(wt.shape) != (t, 8) or wt.dtype != torch.float32:
        raise ValueError("topk_weights must be FP32 [T,8]")
    if any(not v.is_contiguous() or v.device != state["out"].device for v in (x, ids, wt)):
        raise ValueError("inputs must be contiguous on the prepared device")
    cfg = data.get("config", {})
    start = int(cfg.get("local_start", 0))
    if int(cfg.get("local_count", e)) != e:
        raise ValueError("config.local_count must match prepared weights")
    local = state["local_ids"]
    with torch.cuda.device(x.device), torch.no_grad():
        _local_ids[(tr.cdiv(t * 8, 256),)](ids, local, start, e, t * 8, 256)
        H._count_routes[(e,)](local, state["counts"], t * 8, tr.next_power_of_2(t * 8))
        H._routes(local, state, state)
        _grouped_gemm_w4a16_launch(x, state["h13"], state["w13"], state, 8)
        H._swiglu[(t * 8,)](state["h13"], local, state["inter"], d, e, tr.next_power_of_2(d))
        _grouped_gemm_w4a16_launch(state["inter"], state["y"], state["w2"], state, 1)
        H._norm_slots[(t * 8,)](
            state["y"], state["post_norm_w"], local, state["z"], d, e, tr.next_power_of_2(d)
        )
        H._combine_slots[(t, tr.cdiv(d, 256))](state["z"], wt, state["out"], d, 256)
    return state["out"]


def compiler_info():
    """CPU-only metadata + cuTile's actual compiler lookup; never initializes CUDA."""
    import importlib.metadata as metadata
    import json
    import os
    import re
    import shutil
    import subprocess
    from pathlib import Path
    result = {"packages": {}}
    for pkg in ("cuda-tile", "nvidia-cuda-tileiras", "nvidia-cuda-nvcc", "nvidia-nvvm",
                "nvidia-cuda-nvcc-cu13", "torch", "flashinfer-python"):
        try:
            result["packages"][pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            result["packages"][pkg] = None
    result["path_tileiras"] = shutil.which("tileiras")
    try:
        from cuda.tile._compile import _find_compiler_bin
        compiler = _find_compiler_bin()
        path = str(compiler.path)
        result["selected_tileiras"] = path
        env = os.environ.copy()
        # Preserve lookup's private search paths when present on this version.
        for attr, key in (("bin_path", "PATH"), ("ld_path", "LD_LIBRARY_PATH")):
            value = getattr(compiler, attr, None)
            if isinstance(value, str):
                env[key] = value
        if not getattr(compiler, "pass_cuda_home_var", True):
            env.pop("CUDA_HOME", None)
            env.pop("CUDA_PATH", None)
        for flag in ("--version", "--help"):
            p = subprocess.run([path, flag], capture_output=True, text=True, timeout=10, env=env)
            output = p.stdout + p.stderr
            result[flag] = {"returncode": p.returncode, "output": output[:24000]}
            if flag == "--help":
                result["sm90_advertised"] = bool(re.search(r"\bsm_90(?:a)?\b", output))
        sibling = Path(path).with_name("ptxas.exe" if os.name == "nt" else "ptxas")
        result["sibling_ptxas"] = str(sibling) if sibling.is_file() else None
    except Exception as exc:
        result["lookup_error"] = f"{type(exc).__name__}: {exc}"
    result["conclusion"] = "Metadata/help only; no kernel compilation or GPU correctness established"
    print(json.dumps(result, indent=2))


def smoke():
    """Standard-library-only layout/decode/routing audit; not a GPU correctness claim."""
    import ast
    import math
    from pathlib import Path
    import struct
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    magnitudes = (0., .5, 1., 1.5, 2., 3., 4., 6.)
    for byte in range(256):
        for code in (byte & 15, byte >> 4):
            m = code & 7
            bits = (0 if m == 0 else 0x3EC0 + (m << 6) + (0x40 if m > 1 else 0))
            bits |= (code & 8) << 12
            got = struct.unpack(">f", struct.pack(">I", bits << 16))[0]
            expected = math.copysign(magnitudes[m], -1 if code & 8 else 1)
            assert got == expected and math.copysign(1, got) == math.copysign(1, expected)
    for byte in range(256):
        exponent, mantissa = (byte >> 3) & 15, byte & 7
        scale = mantissa * 2.**-9 if exponent == 0 else (1 + mantissa / 8) * 2.**(exponent - 7)
        if (byte & 127) == 127:
            continue  # E4M3FN NaN (kernel preserves it).
        for magnitude in magnitudes:
            # Product is exact at the BF16 MMA boundary for all finite scales.
            bits = struct.unpack(">I", struct.pack(">f", scale * magnitude))[0]
            assert bits & 0xFFFF == 0
    for slots in (8, 16, 128, 256, 1024, 2048):
        for bm in (8, 16, 32):
            for e in (1, 7, 32):
                ids = [(i * 7 + i // 13) % (e + 5) for i in range(slots)]
                groups = [[i for i, expert in enumerate(ids) if expert == j] for j in range(e)]
                lengths = [((len(g) + bm - 1) // bm) * bm for g in groups]
                cap = ((slots + e * (bm - 1) + bm - 1) // bm) * bm
                assert sum(lengths) <= cap
                assert len({i for g in groups for i in g}) == sum(map(len, groups))
                assert max(lengths, default=0) <= ((slots + bm - 1) // bm) * bm
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "flashinfer" for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "flashinfer"
    print("PASS CPU smoke: syntax, all FP4 bytes, finite E4M3*FP4 exact BF16, routing capacities")


def _gpu_check(args):
    # Explicit opt-in only. Reuse the lab reference/error functions, never during run().
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tools/box"))
    import fresh_grouped_lab as lab
    torch.backends.cuda.matmul.allow_tf32 = False
    d = 256 if args.small else 4096
    with torch.no_grad():
        w = lab.weights(d)
        for t in map(int, args.tokens.split(",")):
            state = prepare(w, t)
            data = lab.inputs(t, d, 901, skew=True)
            for mode in ("initial", "changed", "remote", "offset"):
                if mode == "initial":
                    out = run(data, state)  # JIT warmup outside graph capture.
                    torch.cuda.synchronize()
                    graph, out = lab.J.GpuTimer().capture(lambda x: run(x, state), data)
                elif mode == "changed":
                    lab.J.copy_into(data, lab.inputs(t, d, 1923, skew=False))
                elif mode == "remote":
                    data["topk_ids"].fill_(255)
                else:
                    fresh = lab.inputs(t, d, 2073, skew=True)
                    fresh["topk_ids"].add_(32)
                    lab.J.copy_into(data, fresh)
                    data["config"]["local_start"] = 32
                    # Config is shape metadata: capture a new graph for this offset.
                    graph, out = lab.J.GpuTimer().capture(lambda x: run(x, state), data)
                graph.replay()
                torch.cuda.synchronize()
                ref_data = dict(data)
                ref_data["topk_ids"] = data["topk_ids"] - data["config"].get("local_start", 0)
                err = lab.error(out, lab.reference(ref_data, w))
                print(t, mode, err, flush=True)
                if not err["finite"] or err["rel"] > .015 or err["row"] > .075:
                    raise RuntimeError(f"correctness failed: T={t} {mode}: {err}")
                if mode == "remote" and bool(out.count_nonzero()):
                    raise RuntimeError("stale output on empty local routes")
    print("PASS GPU correctness and changed-input graph replay")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--compiler-info", action="store_true")
    mode.add_argument("--gpu-check", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--tokens", default="1,8,16,32,128,256")
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.compiler_info:
        compiler_info()
    else:
        _gpu_check(args)

# okc: problem=muse_attention_decode
# LAB FILE — check runs only, never submit for score
# rung 0b lab: SGLang's baseline.py with kernel_fn unchanged, plus printed H200 measurements (kernels/lab/README.md). nonce r0b-1
"""SGLang's attention for Muse Spark Open on H200 (FA3 backend, FP8 E4M3 KV cache), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: the kernels between the QKV and output projections
(MuseSpark1xAttention.gated_attention), in order:

  FusedQKNorm (Triton)                 per-head RMSNorm of q and k from the fused projection
                                       output, with interleaved RoPE fused in on sliding layers
  set_kv_buffer                        k and v cast to E4M3 (divided by the cache scales when
                                       set), stored at the step's cache rows (store_cache)
  q.to(float8_e4m3fn)                  as prepare_paged_mha_query does for an FP8 cache
  flash_attn_with_kvcache (FA3)        paged, causal, window (2047, 0) on sliding layers,
                                       5.473/8 softmax scale
  rms_norm_gate (JIT CUDA)             per-head RMSNorm x sigmoid(gate)

Differs from SGLang: the step's positions, cache rows, FA3 lengths and FA3 scheduler metadata
come in with the inputs (SGLang builds them once per step for all layers)."""

from typing import TypedDict

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: q (Hq*64), keys (Hkv*64),
                               # values (Hkv*64), the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
_NORMS = {}
_ROPE = {}
ROPE_POSITIONS = 1 << 21


def _qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _NORMS:
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, as SGLang's RotaryEmbedding builds it, cached per theta."""
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def kernel_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device

    if cfg["rope_theta"]:
        q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device))
    else:
        q, k = _qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]

    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)

    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        # The 5.473 query factor is folded into the softmax scale, as SGLang does.
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)

# ===== LAB helpers (shared by the r0b_* lab files; copied into each, since a submission is one file) =====
import time as _lab_time


def _lab_print(*a):
    print(*a, flush=True)


def _lab_err(e, n=600):
    s = "%s: %s" % (type(e).__name__, str(e).replace("\n", " | "))
    return s if len(s) <= n else s[: n - 120] + " ... " + s[-115:]


def _lab_flush_buf(device):
    """256 MiB (> 50 MB L2). Read before every timed replay, as the judge does."""
    return torch.ones(64 << 20, dtype=torch.int32, device=device)


def _lab_graph_us(fn, flush, reps):
    """(median, min) microseconds of CUDA-graph replays of fn(), each replay preceded by a full read of
    `flush` (cold L2). fn is warmed twice on a side stream before capture, like torch's graph recipe."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    g.replay()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in evs:
        flush.sum()
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) * 1e3 for a, b in evs)
    del g
    return ts[len(ts) // 2], ts[0]


def _lab_replay_floor(flush, reps, device):
    x = torch.zeros(1, device=device)
    return _lab_graph_us(lambda: x.add_(1.0), flush, reps)


_LAB_SUM = []


def _lab_sum(line):
    """Collect a compact line for the SUMMARY block printed last (the front of our output is what falls out of
    the 16,000-char tail first)."""
    _LAB_SUM.append(line)


def _lab_print_summary(tag):
    _lab_print("===== %s LAB SUMMARY (raw us; subtract the 1-kernel replay floor to compare with the judge) =====" % tag)
    for line in _LAB_SUM:
        _lab_print(line)
    _lab_print("===== END LAB SUMMARY =====")


def _lab_rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))

# ----- FA3 num_splits sweep on the scored attention shapes (page_size 1, ctx 80K) -----
import math as _lab_math

LAB_CTX = 80000
LAB_SPLITS = (1, 2, 4, 8, 16)
LAB_SM_SCALE = 64 ** -0.5 * 5.4730064863875  # sm_scale * qk_scale_factor, as the baseline passes it
LAB_AGENTIC = 0.075


def _lab_longest(ctx, spread):
    if spread == "fixed":
        return ctx
    if spread == "agentic":
        return _lab_math.ceil(ctx * (1 + LAB_AGENTIC))
    return 4 * ctx


def _lab_lengths(gen, batch, ctx, spread, device):
    if spread == "fixed":
        return torch.full((batch,), ctx, dtype=torch.int64, device=device)
    if spread == "agentic":
        u = torch.rand((batch,), generator=gen, device=device)
        return (ctx * (1 + LAB_AGENTIC * (2 * u - 1))).long().clamp(1, _lab_longest(ctx, spread))
    z = torch.randn((batch,), generator=gen, device=device)
    return (ctx * torch.exp(z - 0.5)).clamp(16, 4 * ctx).long()


def _lab_shared_cache(device):
    """One FP8 K and V pool big enough for every scored shape (batch 32 x (86000 + 8) rows of [2, 64]).
    Values are N(0,1) cast to E4M3: timing does not depend on them."""
    rows = 32 * (_lab_longest(LAB_CTX, "agentic") + 8)
    kc = torch.empty((rows, 1, 2, 64), dtype=torch.float8_e4m3fn, device=device)
    vc = torch.empty((rows, 1, 2, 64), dtype=torch.float8_e4m3fn, device=device)
    for dst in (kc, vc):
        flat = dst.view(-1, 2, 64)
        for at in range(0, rows, 1 << 20):
            n = min(1 << 20, rows - at)
            flat[at:at + n] = torch.randn((n, 2, 64), device=device).to(torch.float8_e4m3fn)
    return kc, vc


def _lab_attn_case(batch, ctx, layer, seed, q_len, spread, device, page=1):
    """The step metadata of the task's generate_input (copied, not imported): lengths, scattered page table,
    positions, FA3 lengths and, on full-attention decode, the scheduler metadata. q is a unit-RMS random
    query cast to E4M3 (what the baseline hands FA3); the KV pool comes from _lab_shared_cache."""
    from sgl_kernel.flash_attn import get_scheduler_metadata
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = batch * q_len
    seq_lens = _lab_lengths(gen, batch, ctx, spread, device)
    pages_per = (seq_lens + q_len + page - 1) // page
    longest = _lab_longest(ctx, spread)
    max_pages = (longest + q_len + page - 1) // page
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offsets[b])
        page_table[b, :n] = perm[o:o + n]
    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page + pos % page).reshape(-1)
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device=device)
    meta = None
    if layer == "full" and q_len == 1:
        meta = get_scheduler_metadata(
            batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()), num_heads=16, num_heads_k=2,
            headdim=64, cache_seqlens=cache_seqlens, qkv_dtype=torch.float8_e4m3fn, cu_seqlens_q=cu_seqlens_q,
            page_size=page, causal=True, num_splits=0)
    qf = torch.randn((tokens, 16, 64), device=device)
    q8 = (qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + 1e-5)).to(torch.float8_e4m3fn)
    return {"q8": q8, "pt": page_table, "cs": cache_seqlens, "cuq": cu_seqlens_q, "meta": meta, "L": q_len,
            "win": (2047, 0) if layer == "sliding" else (-1, -1), "batch": batch, "total": total,
            "seq_lens": seq_lens.to(torch.int32), "positions": positions, "cache_loc": cache_loc}


def _lab_fa3(c, kc, vc, num_splits, meta):
    out = flash_attn_with_kvcache(
        q=c["q8"], k_cache=kc, v_cache=vc, page_table=c["pt"], cache_seqlens=c["cs"], cu_seqlens_q=c["cuq"],
        max_seqlen_q=c["L"], softmax_scale=LAB_SM_SCALE, causal=True, window_size=c["win"],
        scheduler_metadata=meta, num_splits=num_splits)
    return out[0] if isinstance(out, tuple) else out


def _lab_fa3_sweep(shapes, reps, t_begin, deadline_s, tag):
    """shapes: (label, batch, q_len, layer, spread, seed). Per shape: 'base' = exactly the baseline's call
    (num_splits=0 + the given scheduler_metadata), 's0nm' = num_splits=0 without metadata (full decode only),
    'sN' = num_splits=N without metadata (FA3 builds a consistent schedule itself; forcing N together with
    metadata built for N=0 is NOT tried: its dynamic splits could exceed the N-sized accumulators)."""
    device = torch.device("cuda", torch.cuda.current_device())
    flush = _lab_flush_buf(device)
    floor = _lab_replay_floor(flush, reps, device)
    _lab_print("%s 1-kernel graph replay: median %.1f us, min %.1f us" % (tag, floor[0], floor[1]))
    _lab_sum("SUM %s replay floor %.1f us" % (tag, floor[0]))
    t = _lab_time.time()
    kc, vc = _lab_shared_cache(device)
    torch.cuda.synchronize()
    _lab_print("%s KV pool %d rows x 2 heads x 64 B (K and V), built in %.1f s" % (tag, kc.shape[0], _lab_time.time() - t))
    _lab_print("%s FA3 sweep: median us of %d graph replays, cold L2, raw; ! = rel-L2 vs base > 1e-2" % (tag, reps))
    for label, batch, q_len, layer, spread, seed in shapes:
        if _lab_time.time() - t_begin > deadline_s:
            _lab_print("%s %s skipped (deadline)" % (tag, label))
            continue
        try:
            c = _lab_attn_case(batch, LAB_CTX, layer, seed, q_len, spread, device)
            ref = _lab_fa3(c, kc, vc, 0, c["meta"]).float()
            torch.cuda.synchronize()
            configs = [("base", 0, c["meta"])]
            if c["meta"] is not None:
                configs.append(("s0nm", 0, None))
            configs += [("s%d" % n, n, None) for n in LAB_SPLITS]
            cells, best, base_us = [], None, None
            for name, n, meta in configs:
                try:
                    o = _lab_fa3(c, kc, vc, n, meta).float()
                    torch.cuda.synchronize()
                    rel = _lab_rel(o, ref)
                    us = _lab_graph_us(lambda n=n, meta=meta: _lab_fa3(c, kc, vc, n, meta), flush, reps)[0]
                    cells.append("%s %6.1f%s" % (name, us, "!" if rel > 1e-2 else ""))
                    if rel <= 1e-2 and (best is None or us < best[0]):
                        best = (us, name)
                    if name == "base":
                        base_us = us
                except Exception as e:
                    cells.append("%s ERR" % name)
                    _lab_print("%s %s %s ERR %s" % (tag, label, name, _lab_err(e)))
            extra = ""
            if c["meta"] is not None:
                extra = " | meta[:4]=%s" % c["meta"][:4].tolist()
            gain = (" best %s %.2fx" % (best[1], base_us / best[0])) if (best and base_us) else ""
            _lab_print("%s %-11s | %s |%s%s" % (tag, label, " ".join(cells), gain, extra))
            if best and base_us:
                _lab_sum("SUM %s %-11s base %6.1f | best %-4s %6.1f | %.2fx%s" % (
                    tag, label, base_us, best[1], best[0], base_us / best[0], extra))
            else:
                _lab_sum("SUM %s %-11s no valid result" % (tag, label))
        except Exception as e:
            _lab_print("%s %s ERR %s" % (tag, label, _lab_err(e, 1200)))
    del kc, vc, flush
    torch.cuda.empty_cache()

# ===== LAB (muse_attention_decode): probes run once at module import (this problem has no prepare) =====
# kernel_fn above is the baseline's, unchanged. Probes: (c) FusedQKNorm's own fused KV store (cheap, safe),
# (b) FA3 num_splits sweep on the 12 scored shapes, (a) FA3 append (k=/v=) and in-kernel RoPE into the FP8
# paged cache, page_size 1 -- last, because an unsupported FA3 path could fault and poison the context.
LAB_AT_IMPORT = True
LAB_PROBES = ("fused_store", "splits", "append")
LAB_DEADLINE_S = 150.0
LAB_REPS = 15
LAB_ATTN_SHAPES = (  # (label, batch, q_len, layer, spread, seed) as in meta.json; ctx 80000, page 1
    ("dec1 slide", 1, 1, "sliding", "fixed", 601), ("dec1 full", 1, 1, "full", "fixed", 602),
    ("dec16 slide", 16, 1, "sliding", "agentic", 603), ("dec16 full", 16, 1, "full", "agentic", 604),
    ("dec32 slide", 32, 1, "sliding", "agentic", 605), ("dec32 full", 32, 1, "full", "agentic", 606),
    ("ver1 slide", 1, 8, "sliding", "fixed", 607), ("ver1 full", 1, 8, "full", "fixed", 608),
    ("ver16 slide", 16, 8, "sliding", "agentic", 609), ("ver16 full", 16, 8, "full", "agentic", 610),
    ("ver32 slide", 32, 8, "sliding", "agentic", 611), ("ver32 full", 32, 8, "full", "agentic", 612),
)
LAB_ROPE_THETA = 500000.0


def _lab_small_input(batch, ctx, layer, seed, q_len, device, page=1):
    """The task's generate_input(batch, ctx, "tp8", layer, seed, q_len, page=page) for spread "fixed",
    dist "normal", kv_scale 1, copied (not imported)."""
    from sgl_kernel.flash_attn import get_scheduler_metadata
    hq, hkv, d = 16, 2, 64
    gen = torch.Generator(device=device).manual_seed(seed)
    seq_lens = _lab_lengths(gen, batch, ctx, "fixed", device)
    pages_per = (seq_lens + q_len + page - 1) // page
    max_pages = (_lab_longest(ctx, "fixed") + q_len + page - 1) // page
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offsets[b])
        page_table[b, :n] = perm[o:o + n]
    shape = (total, page, hkv, d)
    k_cache = torch.empty(shape, dtype=FP8, device=device)
    v_cache = torch.empty(shape, dtype=FP8, device=device)
    for dst in (k_cache, v_cache):
        flat = dst.view(-1, hkv, d)
        for at in range(0, flat.shape[0], 1 << 20):
            n = min(1 << 20, flat.shape[0] - at)
            flat[at:at + n] = torch.randn((n, hkv, d), generator=gen, device=device).float().to(FP8)
    q = (2 * torch.randn((batch, q_len, hq, d), generator=gen, device=device)).to(torch.bfloat16)
    k_new = (2 * torch.randn((batch, q_len, hkv, d), generator=gen, device=device)).to(torch.bfloat16)
    v_new = torch.randn((batch, q_len, hkv, d), generator=gen, device=device).to(torch.bfloat16)
    gate = torch.randn((batch, q_len, hq * d), generator=gen, device=device).to(torch.bfloat16)
    qkv = torch.cat([q.flatten(2), k_new.flatten(2), v_new.flatten(2), gate], -1)
    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page + pos % page).reshape(-1)
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device=device)
    meta = None
    if layer == "full" and q_len == 1:
        meta = get_scheduler_metadata(
            batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()), num_heads=hq, num_heads_k=hkv,
            headdim=d, cache_seqlens=cache_seqlens, qkv_dtype=FP8, cu_seqlens_q=cu_seqlens_q, page_size=page,
            causal=True, num_splits=0)
    return {
        "qkv": qkv, "positions": positions, "cache_loc": cache_loc, "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q, "scheduler_metadata": meta, "k_cache": k_cache, "v_cache": v_cache,
        "k_scale": torch.tensor(1.0, dtype=torch.float32, device=device),
        "v_scale": torch.tensor(1.0, dtype=torch.float32, device=device),
        "page_table": page_table, "seq_lens": seq_lens.to(torch.int32),
        "config": {"batch": batch, "q_len": q_len, "num_q_heads": hq, "num_kv_heads": hkv, "head_dim": d,
                   "page_size": page, "window": 2048 if layer == "sliding" else 0,
                   "rope_theta": LAB_ROPE_THETA if layer == "sliding" else 0.0, "qk_scale_factor": 5.4730064863875,
                   "sm_scale": d ** -0.5, "kv_scale": 1.0},
    }


def _lab_run_base(data):
    """The baseline kernel_fn on private copies of the caches; returns (output, base caches)."""
    base = dict(data)
    base["k_cache"], base["v_cache"] = data["k_cache"].clone(), data["v_cache"].clone()
    out = kernel_fn(base)
    torch.cuda.synchronize()
    return out, base


def _lab_probe_fused_store(device):
    """(c) FusedQKNorm(qkv, ..., kv_store=(k_cache, v_cache, cache_loc)): norm (+RoPE) + FP8 KV store + FP8 q in
    one Triton kernel, vs the baseline's qk_norm + store_cache + q cast. NB its kernel skips cache_loc == 0."""
    hq, hkv, d = 16, 2, 64
    for layer in ("full", "sliding"):
        try:
            data = _lab_small_input(2, 3000, layer, 4300 + (1 if layer == "sliding" else 0), 8, device)
            tokens = 16
            _, base = _lab_run_base(data)
            qkv = data["qkv"].view(tokens, -1)
            op = _qk_norm(hq, hkv, d)
            kc, vc = data["k_cache"].clone(), data["v_cache"].clone()
            store = (kc.view(-1, hkv * d), vc.view(-1, hkv * d), data["cache_loc"])
            if layer == "sliding":
                cs = _cos_sin(LAB_ROPE_THETA, d, device)
                q_f, _ = op(qkv, data["positions"], cs, kv_store=store)
                q_b, _ = op(qkv, data["positions"], cs)
            else:
                q_f, _ = op(qkv, kv_store=store)
                q_b, _ = op(qkv)
            torch.cuda.synchronize()
            q8b = q_b.reshape(tokens, hq * d).to(FP8)
            rows = data["cache_loc"]
            kf, kb = kc.view(-1, hkv * d)[rows], base["k_cache"].view(-1, hkv * d)[rows]
            vf, vb = vc.view(-1, hkv * d)[rows], base["v_cache"].view(-1, hkv * d)[rows]
            same = lambda a, b: float((a.view(torch.uint8) == b.view(torch.uint8)).float().mean())
            line = ("ATTN(c) fused kv_store %-7s q dtype %s | q bytes equal %.4f rel %.1e | K rows equal %.4f rel %.1e"
                    " | V rows equal %.4f rel %.1e | cache_loc==0 present: %d" % (
                        layer, q_f.dtype, same(q_f.reshape(tokens, hq * d), q8b),
                        _lab_rel(q_f.reshape(tokens, hq * d).float(), q8b.float()), same(kf, kb),
                        _lab_rel(kf.float(), kb.float()), same(vf, vb), _lab_rel(vf.float(), vb.float()),
                        int((rows == 0).sum().item())))
            _lab_print(line)
            _lab_sum("SUM " + line)
        except Exception as e:
            _lab_print("ATTN(c) fused kv_store %s ERR %s" % (layer, _lab_err(e, 800)))
            _lab_sum("SUM ATTN(c) %s ERR %s" % (layer, _lab_err(e, 150)))


def _lab_probe_append(device):
    """(a) FA3 with k=/v= (append into the paged FP8 cache, page_size 1) and optionally rotary_cos/sin, against
    the baseline kernel_fn on the same inputs: output rel-L2 and new-cache-row rel-L2 (K, V)."""
    hq, hkv, d = 16, 2, 64
    for layer in ("full", "sliding"):
        for L in (1, 8):
            try:
                data = _lab_small_input(2, 3000, layer, 4200 + L + (10 if layer == "sliding" else 0), L, device)
                tokens = 2 * L
                o_base, base = _lab_run_base(data)
            except Exception as e:
                _lab_print("ATTN(a) %s L=%d setup ERR %s" % (layer, L, _lab_err(e, 800)))
                continue
            rows = data["cache_loc"]
            kb = base["k_cache"].view(-1, hkv, d)[rows].float()
            vb = base["v_cache"].view(-1, hkv, d)[rows].float()
            qkv = data["qkv"].view(tokens, -1)
            v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
            gate = qkv[:, (hq + 2 * hkv) * d:]
            if layer == "full":
                variants = (("append", False, None),)
            else:
                variants = (("append, rope by qk_norm", True, None), ("append+FA3 rope, cos bf16", False, torch.bfloat16),
                            ("append+FA3 rope, cos fp8", False, FP8))
            for name, rope_outside, cos_dtype in variants:
                try:
                    if rope_outside:
                        qn, kn = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(LAB_ROPE_THETA, d, device))
                    else:
                        qn, kn = _qk_norm(hq, hkv, d)(qkv)
                    q8 = qn.reshape(tokens, hq, d).to(FP8)
                    k8 = kn.reshape(tokens, hkv, d).to(FP8)
                    v8 = v.reshape(tokens, hkv, d).to(FP8)
                    kc, vc = data["k_cache"].clone(), data["v_cache"].clone()
                    kw = {}
                    if cos_dtype is not None:
                        inv = 1.0 / (LAB_ROPE_THETA ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
                        ang = torch.arange(4096, dtype=torch.float, device=device)[:, None] * inv[None, :]
                        kw = dict(rotary_cos=ang.cos().to(cos_dtype).contiguous(),
                                  rotary_sin=ang.sin().to(cos_dtype).contiguous(), rotary_interleaved=True)
                    out = flash_attn_with_kvcache(
                        q=q8, k_cache=kc, v_cache=vc, k=k8, v=v8, page_table=data["page_table"],
                        cache_seqlens=data["seq_lens"], cu_seqlens_q=data["cu_seqlens_q"],
                        cu_seqlens_k_new=data["cu_seqlens_q"], max_seqlen_q=L, softmax_scale=LAB_SM_SCALE, causal=True,
                        window_size=(2047, 0) if layer == "sliding" else (-1, -1), **kw)
                    if isinstance(out, tuple):
                        out = out[0]
                    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d,
                                      eps=RMS_EPS, has_norm=True).view(2, L, hq * d)
                    torch.cuda.synchronize()
                    ko = kc.view(-1, hkv, d)[rows].float()
                    vo = vc.view(-1, hkv, d)[rows].float()
                    line = "ATTN(a) %-7s L=%d %-26s OK | out rel %.2e | new K rel %.2e | new V rel %.2e" % (
                        layer, L, name, _lab_rel(o, o_base), _lab_rel(ko, kb), _lab_rel(vo, vb))
                    _lab_print(line)
                    _lab_sum("SUM " + line)
                except Exception as e:
                    _lab_print("ATTN(a) %-7s L=%d %-26s REFUSED/ERR %s" % (layer, L, name, _lab_err(e, 500)))
                    _lab_sum("SUM ATTN(a) %-7s L=%d %-26s REFUSED/ERR %s" % (layer, L, name, _lab_err(e, 150)))


def _lab_attention_main():
    t_begin = _lab_time.time()
    device = torch.device("cuda", torch.cuda.current_device())
    _lab_print("LAB muse_attention_decode r0b: import-time probes %s on %s" % (LAB_PROBES, torch.cuda.get_device_name()))
    for probe in LAB_PROBES:
        if _lab_time.time() - t_begin > LAB_DEADLINE_S:
            _lab_print("LAB %s skipped (deadline)" % probe)
            continue
        try:
            if probe == "fused_store":
                _lab_probe_fused_store(device)
            elif probe == "splits":
                _lab_fa3_sweep(LAB_ATTN_SHAPES, LAB_REPS, t_begin, LAB_DEADLINE_S, "FA3A")
            elif probe == "append":
                _lab_probe_append(device)
            torch.cuda.synchronize()
        except Exception as e:
            _lab_print("LAB %s ERR %s" % (probe, _lab_err(e, 2000)))
        _lab_print("LAB %s finished at %.1f s" % (probe, _lab_time.time() - t_begin))
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        _lab_print("LAB empty_cache ERR %s" % _lab_err(e))
    _lab_sum("SUM LAB import-time probes total %.1f s" % (_lab_time.time() - t_begin))
    _lab_print_summary("muse_attention_decode")


if LAB_AT_IMPORT:
    try:
        if torch.cuda.is_available():
            _lab_attention_main()
        else:
            _lab_print("LAB: no CUDA at import, probes skipped")
    except Exception as _lab_e:
        _lab_print("LAB ERR " + _lab_err(_lab_e, 2000))

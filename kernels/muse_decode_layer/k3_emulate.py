"""K3 CPU emulation: runs kernels/muse_decode_layer/k3_v0*.py UNMODIFIED on CPU and checks the composition's
plumbing against a CPU copy of the layer reference (problems/muse_decode_layer/reference.py math, copied below,
FP8 caches held as uint8 storage because CPU torch 2.5.1 cannot index float8 tensors).

Run:  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -I k3_emulate.py [scenario ...]

How the GPU pieces are replaced (harness only; the submission file is exec'd as is):
  * torch.utils.cpp_extension.load_inline -> a fake module chosen by the `functions` list. It asserts the CUDA/C++
    source it was handed is byte-identical to the reviewed k1_v0.py / k2a_v0.py source, then:
      K1  k1_forward(...)  -> re-checks every TORCH_CHECK of the C++ wrapper, rebuilds the kernel's view of the call
                              ONLY from the arguments (B, L, heads, W, page_shift, Gx, use_rope + inv_freq, qk_sm,
                              uint8 cache views) and runs k1_emulate.emulate_k1 (CTA-level emulation of k1_main +
                              k1_combine), with per-row key-coverage check; writes `out` and the caches in place.
      K2A route/gemm/finalize -> k2a_emulate.emu_route / emu_gemm (+ simulate_cursor) / emu_final on exactly the
                              tensors k3 passes (kernel-written buffers poisoned with NaN first).
  * SGLang/sgl_kernel/triton imports -> CPU stand-ins with SGLang's rounding points (02-model-layer.md section 3):
    cuBLAS BF16 GEMM, FusedPostNormResidualPreNorm, FusedMuseSparkRouting, FusedQKNorm (+ FP8 KV-store epilogue),
    store_cache, FA3 (FP8 q/K/V, FP32 softmax), FusedRMSNormGate. Marlin is not emulated: scenarios that need the
    Marlin path replace k3's `_experts` (r1 code, unchanged) with a CPU W4A16 expert reference of the same signature.
  * torch.cuda.* -> stubs. While kernel_fn runs "under capture", Tensor.item/tolist/cpu/__bool__/__int__/__float__
    and torch.cuda.synchronize raise unless called from inside a stand-in (i.e. the k3 glue itself must not sync).
"""

import contextlib
import functools
import importlib.util
import math
import os
import pathlib
import sys
import time
import types

import torch
import torch.nn.functional as F
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "problems" / "muse_decode_layer"))
from checks import compare, query_precision_bound, relative_l2  # noqa: E402
from nvfp4 import dequantize, random_weights  # noqa: E402

torch.set_flush_denormal(False)
torch.set_num_threads(max(1, os.cpu_count() or 1))
FP8 = torch.float8_e4m3fn


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


K1E = _load("k1_emulate", ROOT / "kernels/muse_attention_decode/k1_emulate.py")
K2E = _load("k2a_emulate", ROOT / "kernels/muse_moe_w4a16/k2a_emulate.py")   # also loads k2a_v0.py (CPU-safe)


def _reviewed_sources():
    import ast
    out = {}
    for key, path in (("k1", ROOT / "kernels/muse_attention_decode/k1_v0.py"),
                      ("k2a", ROOT / "kernels/muse_moe_w4a16/k2a_v0.py")):
        tree = ast.parse(path.read_text())
        vals = {n.targets[0].id: n.value.value for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("CPP_SRC", "CUDA_SRC")}
        out[key] = (vals["CPP_SRC"], vals["CUDA_SRC"])
    return out


REVIEWED = _reviewed_sources()

# ------------------------------------------------------------------------------------------------
# Capture simulation: host syncs from the k3 glue raise while CAPTURE is on (stand-ins are exempt)
# ------------------------------------------------------------------------------------------------
CAPTURE = [False]
ALLOW = [0]
CALLS = {}


@contextlib.contextmanager
def allowed():
    ALLOW[0] += 1
    try:
        yield
    finally:
        ALLOW[0] -= 1


def standin(fn):
    @functools.wraps(fn)
    def wrap(*a, **k):
        CALLS[fn.__qualname__] = CALLS.get(fn.__qualname__, 0) + 1
        with allowed():
            return fn(*a, **k)
    return wrap


def _guard(name, orig):
    def g(self, *a, **k):
        if CAPTURE[0] and not ALLOW[0]:
            raise RuntimeError(f"host sync under (simulated) capture: Tensor.{name}")
        return orig(self, *a, **k)
    return g


for _n in ("item", "tolist", "cpu", "__bool__", "__int__", "__float__", "numpy"):
    setattr(torch.Tensor, _n, _guard(_n, getattr(torch.Tensor, _n)))


class _Stream:
    def wait_stream(self, other):
        pass


def _sync(*a, **k):
    if CAPTURE[0] and not ALLOW[0]:
        raise RuntimeError("torch.cuda.synchronize under (simulated) capture")


torch.cuda.Stream = lambda *a, **k: _Stream()
torch.cuda.current_stream = lambda *a, **k: _Stream()
torch.cuda.stream = lambda s: contextlib.nullcontext()
torch.cuda.is_current_stream_capturing = lambda: CAPTURE[0]
torch.cuda.synchronize = _sync
torch.cuda.get_device_properties = lambda dev: types.SimpleNamespace(multi_processor_count=132)
torch.Tensor.record_stream = lambda self, s: None
_MM = torch.mm


def _mm(a, b, *args, out_dtype=None, **kw):   # torch 2.5.1 CPU has no out_dtype (SGLang's image torch does)
    if out_dtype is None:
        return _MM(a, b, *args, **kw)
    return _MM(a.float(), b.float()).to(out_dtype)


torch.mm = _mm


# ------------------------------------------------------------------------------------------------
# Fake CUDA extensions (load_inline replacement)
# ------------------------------------------------------------------------------------------------
FAIL = {"k1_build": False, "k2a_build": False, "k1_corrupt": False, "k2a_corrupt": False, "k1_raise_at_call": False}
INV_FREQ_500K = (1.0 / (500000.0 ** (torch.arange(0, 64, 2).float() / 64)))


def _chk(cond, msg):
    if not cond:
        raise RuntimeError(f"TORCH_CHECK failed: {msg}")


class FakeK1:
    @standin
    def k1_occupancy(self, L):
        return 2

    @standin
    def k1_forward(self, qkv, seq_lens, cache_loc, page_table, k_cache, v_cache, k_scale, v_scale, inv_freq,
                   part_o, part_lse, out, B, L, Hq, Hkv, W, page_shift, Gx, min_chunk, use_rope, qk_sm):
        if FAIL["k1_raise_at_call"] and CAPTURE[0]:
            raise RuntimeError("TORCH_CHECK failed: injected call-time failure")
        # k1_forward's TORCH_CHECKs, verbatim conditions
        _chk(L in (1, 8), "k1: q_len must be 1 or 8")
        _chk(Hq == 8 * Hkv and Hkv % 2 == 0 and Hkv > 0, "k1: need Hq == 8*Hkv and even Hkv")
        _chk(Gx > B, "k1: Gx must exceed B")
        _chk(qkv.dtype == torch.bfloat16 and qkv.is_contiguous(), "k1: qkv")
        _chk(qkv.numel() == B * L * (2 * Hq + 2 * Hkv) * 64, "k1: qkv shape")
        _chk(seq_lens.dtype == torch.int32 and seq_lens.is_contiguous() and seq_lens.numel() == B, "k1: seq_lens")
        _chk(cache_loc.dtype == torch.int64 and cache_loc.is_contiguous() and cache_loc.numel() == B * L, "k1: cache_loc")
        _chk(page_table.dtype == torch.int32 and page_table.is_contiguous() and page_table.dim() == 2
             and page_table.size(0) == B, "k1: page_table")
        _chk(k_cache.dtype == torch.uint8 and k_cache.is_contiguous(), "k1: k_cache")
        _chk(v_cache.dtype == torch.uint8 and v_cache.is_contiguous(), "k1: v_cache")
        _chk(k_scale.dtype == torch.float32 and v_scale.dtype == torch.float32, "k1: scales")
        _chk(inv_freq.dtype == torch.float32 and inv_freq.numel() >= 32, "k1: inv_freq")
        _chk(part_o.dtype == torch.float32 and part_o.numel() >= (Hkv // 2) * Gx * 16 * L * 64, "k1: part_o")
        _chk(part_lse.dtype == torch.float32 and part_lse.numel() >= (Hkv // 2) * Gx * 16 * L, "k1: part_lse")
        _chk(out.dtype == torch.bfloat16 and out.is_contiguous() and out.numel() == B * L * Hq * 64, "k1: out")
        # harness-only consistency of the derived scalars
        if use_rope:
            assert torch.equal(inv_freq.cpu(), INV_FREQ_500K), "inv_freq is not theta=5e5's"
            theta = 500000.0
        else:
            assert float(inv_freq.abs().sum()) == 0.0
            theta = 0.0
        page = 1 << page_shift
        # the kernel's view of the call, built only from the arguments
        d = {"qkv": qkv, "seq_lens": seq_lens, "cache_loc": cache_loc, "page_table": page_table,
             "k_cache": k_cache, "v_cache": v_cache, "k_scale": k_scale, "v_scale": v_scale,
             "config": {"batch": B, "q_len": L, "num_q_heads": Hq, "num_kv_heads": Hkv, "head_dim": 64,
                        "page_size": page, "window": W, "rope_theta": theta,
                        "qk_scale_factor": qk_sm, "sm_scale": 1.0}}
        visits = {}
        res = K1E.emulate_k1(d, n_sm=Gx * (Hkv // 2), cps=1, min_chunk=min_chunk, visits=visits)
        K1E.check_coverage(d, visits)
        if FAIL["k1_corrupt"]:
            res = (res.float() * 1.5).to(torch.bfloat16)
        out.copy_(res.view_as(out))


class FakeK2A:
    def __init__(self):
        self.last = None

    @standin
    def route(self, ids, meta, M, lo, G):
        _chk(ids.dtype == torch.int32 and ids.is_contiguous() and ids.numel() == 8 * M, "route ids")
        _chk(meta.dtype == torch.int32 and meta.numel() >= 128 + 16 * M, "route meta")
        assert lo == 0, "k3 must pass lo=0 (ids already local)"
        m64 = K2E.emu_route(ids.view(M, 8), lo, G)
        meta.copy_(m64.to(torch.int32))
        self.last = dict(n_act=int(m64[0]), ks2=int(m64[1]), routes=int(m64[2]))

    @standin
    def gemm(self, mode, ng, wq, sq, gsc, x, meta, out, M, G):
        _chk(mode in (1, 2) and ng in (1, 2, 4), "mode/ng")
        _chk(x.dtype == torch.bfloat16 and x.is_contiguous() and x.size(1) == 4096 and x.data_ptr() % 16 == 0, "x")
        _chk(wq.is_contiguous() and sq.is_contiguous() and gsc.numel() >= 32, "weights")
        _chk(out.dtype == (torch.bfloat16 if mode == 1 else torch.float32), "out dtype")
        _chk(ng == K2E.K.ng_for_bs(M), "ng matches ng_for_bs(M)")
        m64 = meta.long()
        K2E.simulate_cursor(mode, m64, ng, G)       # asserts unit coverage + ring consistency
        # kernel-written buffers start poisoned; the emulator writes FP32 (MODE 1 values are BF16-exact)
        buf = torch.full(out.shape, float("nan"), dtype=torch.float32)
        K2E.emu_gemm(mode, wq, sq, gsc, x, m64, M, ng, G, buf)
        out.copy_(buf.to(out.dtype))

    @standin
    def finalize(self, y, meta, ids, wts, nw, out, M, lo):
        _chk(y.dtype == torch.float32 and wts.dtype == torch.float32 and nw.numel() == 4096, "finalize")
        _chk(ids.dtype == torch.int32 and ids.is_contiguous() and wts.is_contiguous(), "finalize ids/wts")
        assert lo == 0
        res = K2E.emu_final(y, meta.long(), ids.view(M, 8), wts.view(M, 8), nw, M, lo)
        if FAIL["k2a_corrupt"]:
            res = (res.float() * 1.3).to(torch.bfloat16)
        out.copy_(res)


def fake_load_inline(name, cpp_sources, cuda_sources, functions, extra_cuda_cflags=None, verbose=False, **kw):
    if functions == ["k1_forward", "k1_occupancy"]:
        assert (cpp_sources, cuda_sources) == REVIEWED["k1"], "K1 source differs from reviewed k1_v0.py"
        assert name.startswith("k1_v0_")
        if FAIL["k1_build"]:
            raise RuntimeError("Error building extension (injected)")
        return FakeK1()
    if functions == ["route", "gemm", "finalize"]:
        assert (cpp_sources, cuda_sources) == REVIEWED["k2a"], "K2A source differs from reviewed k2a_v0.py"
        assert name.startswith("k2a_v0_")
        if FAIL["k2a_build"]:
            raise RuntimeError("Error building extension (injected)")
        return FakeK2A()
    raise AssertionError(functions)


import torch.utils.cpp_extension as _cpp  # noqa: E402

_cpp.load_inline = fake_load_inline


# ------------------------------------------------------------------------------------------------
# SGLang stand-ins (CPU, SGLang's rounding points)
# ------------------------------------------------------------------------------------------------
def _rms(x, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class MuseSpark1xRMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6, has_weight=True, zero_centered_gamma=False, **kw):
        super().__init__()
        self.size, self.eps, self.zc = size, eps, zero_centered_gamma
        self.weight = nn.Parameter(torch.zeros(size, dtype=torch.bfloat16), requires_grad=False) if has_weight else None
        self._eff = None

    def precompute_effective_weight(self):
        self._eff = (self.weight.float() + 1.0) if self.zc else self.weight.float()

    def _get_effective_weight(self):
        return self._eff

    @standin
    def forward(self, x):
        y = _rms(x, self.eps)
        if self.weight is not None:
            y = y * self._eff
        return y.to(x.dtype)


class MuseSpark1xRMSNormWithInputScale(MuseSpark1xRMSNorm):
    pass


class MuseSpark1xNormPreservingResidualConnection(nn.Module):
    def __init__(self, hidden, temperature=0.3):
        super().__init__()
        self.temperature = temperature
        self.gate = None

    def precompute_coefficients(self):
        w = self.gate.float() / self.temperature
        self.beta = torch.sigmoid(w)
        self.alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + self.beta), min=1e-3))


class FusedPostNormResidualPreNorm:
    def __init__(self, post_norm, residual, pre_norm):
        self.post, self.res, self.pre = post_norm, residual, pre_norm

    @standin
    def __call__(self, stream, branch, branch_dtype=torch.bfloat16, router_dtype=None):
        b = branch.float()
        if isinstance(self.post, MuseSpark1xRMSNormWithInputScale) and self.post.weight is not None:
            b = (b * self.post._eff).to(torch.bfloat16).float()      # w*x rounded to BF16 before normalizing
        pn = _rms(b, self.post.eps).to(torch.bfloat16).float()      # post-norm output rounded to BF16
        s = self.res.alpha * stream.float() + self.res.beta * pn
        h = (_rms(s, self.pre.eps) * self.pre._eff).to(branch_dtype)
        return h, s, (h.float() if router_dtype is torch.float32 else None)


class FusedMuseSparkRouting:
    def __init__(self, expert_bias, route_scale, router_output_multiplier, router_score_division_eps):
        self.bias, self.scale, self.mult, self.eps = expert_bias, route_scale, router_output_multiplier, router_score_division_eps

    @standin
    def __call__(self, hidden, logits, top_k, renormalize):
        scores = torch.sigmoid(logits.float() * self.mult)
        ids = (scores + self.bias).topk(top_k, dim=-1).indices
        w = scores.gather(1, ids)
        w = w / (w.sum(-1, keepdim=True) + self.eps) * self.scale
        return w.float().contiguous(), ids.to(torch.int32).contiguous()


def _rope_table(x, pos, cos_sin):
    cs = cos_sin[pos.long()]
    cos, sin = cs[:, None, :32], cs[:, None, 32:]
    ev, od = x[..., 0::2], x[..., 1::2]
    return torch.stack([ev * cos - od * sin, ev * sin + od * cos], -1).flatten(-2)


class FusedQKNorm:
    def __init__(self, head_dim, num_q_heads, num_kv_heads, eps):
        self.d, self.hq, self.hkv, self.eps = head_dim, num_q_heads, num_kv_heads, eps

    def _qk(self, qkv, positions, cos_sin):
        t, d, hq, hkv = qkv.shape[0], self.d, self.hq, self.hkv
        q = _rms(qkv[:, :hq * d].reshape(t, hq, d), self.eps).to(torch.bfloat16).float()
        k = _rms(qkv[:, hq * d:(hq + hkv) * d].reshape(t, hkv, d), self.eps).to(torch.bfloat16).float()
        if positions is not None:
            q, k = _rope_table(q, positions, cos_sin), _rope_table(k, positions, cos_sin)
        return q.to(torch.bfloat16).reshape(t, hq * d), k.to(torch.bfloat16).reshape(t, hkv * d)

    @standin
    def __call__(self, qkv, positions=None, cos_sin=None):
        return self._qk(qkv, positions, cos_sin)

    @standin
    def forward_triton(self, qkv, positions, cos_sin, kv_store):
        q, k = self._qk(qkv, positions, cos_sin)
        k_stage, v_stage, loc = kv_store
        d, hq, hkv = self.d, self.hq, self.hkv
        v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
        assert int(loc.min()) >= 1, "kernel skips row 0"
        k_stage.view(torch.uint8)[loc] = k.float().clamp(-448, 448).to(FP8).view(torch.uint8)
        v_stage.view(torch.uint8)[loc] = v.float().clamp(-448, 448).to(FP8).view(torch.uint8)
        return q.float().to(FP8), None


@standin
def store_cache(k_u8, v_u8, kc_u8, vc_u8, loc, reserved_skip_index=0):
    assert reserved_skip_index == -1
    kc_u8[loc.long()] = k_u8
    vc_u8[loc.long()] = v_u8


@standin
def flash_attn_with_kvcache(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q,
                            softmax_scale, causal, window_size, scheduler_metadata=None, k_descale=None,
                            v_descale=None, num_splits=0):
    t, hq, d = q.shape
    page, hkv = k_cache.shape[1], k_cache.shape[2]
    B, L = page_table.shape[0], max_seqlen_q
    ks = float(k_descale.reshape(-1)[0]) if k_descale is not None else 1.0
    vs = float(v_descale.reshape(-1)[0]) if v_descale is not None else 1.0
    kc = k_cache.view(torch.uint8).view(-1, hkv, d)
    vc = v_cache.view(torch.uint8).view(-1, hkv, d)
    qf = q.float().view(B, L, hkv, hq // hkv, d)
    out = torch.empty((B, L, hq, d), dtype=torch.float32)
    for b in range(B):
        n = int(cache_seqlens[b])
        idx = torch.arange(n)
        rows = page_table[b].long()[idx // page] * page + idx % page
        keys = kc[rows].view(FP8).float() * ks
        vals = vc[rows].view(FP8).float() * vs
        for j in range(L):
            p = n - L + j
            vis = idx <= p
            if window_size[0] >= 0:
                vis &= idx >= p - window_size[0]
            att = torch.einsum("kgd,tkd->kgt", qf[b, j], keys) * softmax_scale
            att = att.masked_fill(~vis[None, None, :], float("-inf")).softmax(-1)
            out[b, j] = torch.einsum("kgt,tkd->kgd", att, vals).reshape(hq, d)
    return out.reshape(t, hq, d).to(torch.bfloat16)


class FusedRMSNormGate:
    def __init__(self, norm, flag, head_dim):
        self.norm, self.d = norm, head_dim

    @standin
    def __call__(self, o, gate):
        t = o.shape[0]
        y = _rms(o.float().view(t, -1, self.d), self.norm.eps) * torch.sigmoid(gate.float().reshape(t, -1, self.d))
        return y.reshape(t, -1).to(torch.bfloat16)


@standin
def _bf16_gemm_dispatch_impl(x, w, bias):
    return (x.float() @ w.float().t()).to(torch.bfloat16)


def _unsupported(*a, **k):
    raise NotImplementedError("Marlin/align/silu stand-ins are not emulated; the scenario must replace _experts")


def _install_stubs():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m
    mod("triton", next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
    mod("sgl_kernel")
    mod("sgl_kernel.flash_attn", flash_attn_with_kvcache=flash_attn_with_kvcache)
    mod("sgl_kernel.scalar_type", scalar_types=types.SimpleNamespace(float4_e2m1f="float4_e2m1f"))
    for p in ("sglang", "sglang.kernels", "sglang.kernels.ops", "sglang.kernels.ops.activation",
              "sglang.kernels.ops.kvcache", "sglang.kernels.ops.moe", "sglang.kernels.ops.muse_spark_v12",
              "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.moe", "sglang.srt.layers.quantization"):
        mod(p)
    mod("sglang.kernels.ops.activation.activation", silu_and_mul=_unsupported)
    mod("sglang.kernels.ops.kvcache.kvcache", store_cache=store_cache)
    mod("sglang.kernels.ops.moe.moe_wna16_marlin", moe_wna16_marlin_gemm=_unsupported)
    mod("sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing", FusedMuseSparkRouting=FusedMuseSparkRouting)
    mod("sglang.kernels.ops.muse_spark_v12.fused_norm_gate", FusedRMSNormGate=FusedRMSNormGate)
    mod("sglang.kernels.ops.muse_spark_v12.fused_norm_residual", FusedPostNormResidualPreNorm=FusedPostNormResidualPreNorm)
    mod("sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton",
        fused_post_expert_norm_weighted_sum=_unsupported)   # no _post_expert_norm_weighted_sum_kernel -> ImportError path
    mod("sglang.kernels.ops.muse_spark_v12.fused_qk_norm", FusedQKNorm=FusedQKNorm)
    mod("sglang.srt.layers.moe.fused_moe_triton", moe_align_block_size=_unsupported)
    mod("sglang.srt.layers.muse_spark_layers", MuseSpark1xNormPreservingResidualConnection=MuseSpark1xNormPreservingResidualConnection,
        MuseSpark1xRMSNorm=MuseSpark1xRMSNorm, MuseSpark1xRMSNormWithInputScale=MuseSpark1xRMSNormWithInputScale)
    mod("sglang.srt.layers.quantization.marlin_utils", marlin_make_workspace=lambda device, max_blocks_per_sm=1: torch.zeros(1))
    mod("sglang.srt.layers.quantization.marlin_utils_fp4", prepare_moe_nvfp4_layer_for_marlin=lambda layer: None)
    mod("sglang.srt.layers.quantization.unquant", _bf16_gemm_dispatch_impl=_bf16_gemm_dispatch_impl)


_install_stubs()


def make_cpu_experts(weights):
    """A CPU W4A16 expert path with r1 `_experts`' signature and SGLang's BF16 rounding points (h, silu*up,
    y in BF16; post-norm in FP32 with bf16(1+w); BF16 out). Stands in for Marlin, which is not emulated."""
    def deq(e, which):
        return _deq_ref(weights, e, which).to(torch.bfloat16).float()   # exact: e2m1 x e4m3 fits BF16

    @standin
    def experts(x, topk_weights, topk_ids, state, ids_are_local=False):
        ids = topk_ids.long()
        if not ids_are_local:
            ids = torch.where((ids >= 0) & (ids < 32), ids, torch.full_like(ids, -1))
        nw = (weights["post_expert_norm"].float() + 1.0).to(torch.bfloat16).float()
        out = torch.zeros(x.shape, dtype=torch.float32)
        for e in torch.unique(ids).tolist():
            if not 0 <= e < 32:
                continue
            tok, slot = (ids == e).nonzero(as_tuple=True)
            h = (x[tok].float() @ deq(e, "w13").t()).to(torch.bfloat16).float()
            inter = (F.silu(h[:, :4096]) * h[:, 4096:]).to(torch.bfloat16).float()
            y = (inter @ deq(e, "w2").t()).to(torch.bfloat16).float() * nw
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
            out.index_add_(0, tok, y * topk_weights[tok, slot].float()[:, None])
        return out.to(torch.bfloat16)
    return experts


# ------------------------------------------------------------------------------------------------
# Copy of the layer reference (problems/muse_decode_layer/reference.py), CPU, caches as uint8 storage
# ------------------------------------------------------------------------------------------------
HIDDEN = 8192
HEAD_DIM = 64
HQ, HKV = 16, 2
Q_SIZE, KV_SIZE = HQ * HEAD_DIM, HKV * HEAD_DIM
QKV_SIZE = 2 * Q_SIZE + 2 * KV_SIZE
SLIDING_WINDOW = 2048
ROPE_THETA = 500_000.0
QK_SCALE_FACTOR = 5.4730064863875
SM_SCALE = HEAD_DIM ** -0.5
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
PAGE = 64
D_EXPERT = 4096
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
ROUTER_MULTIPLIER = 0.1767766952966369
ROUTE_SCALE = 1.0
ROUTE_EPS = 1e-15
RESIDUAL_TEMPERATURE = 0.3
RESIDUAL_GATE_EPS = 1e-3
UNPOOL_SLICE = HIDDEN // 8
SLICED_MAX_TOKENS = 8
REL_L2_TOL = 0.15
FP8_QUERY_FACTOR = 2.5
FP8_QUERY_FLOOR = 0.03
CACHE_TOL = 8e-2
ROUTE_TOL = 5e-2
LOCAL_SLACK = (0.08, 4)
DISTINCT_SLACK = (0.15, 2)
KEEP_TOL = (0.03, 0.04)
KEEP_TOKENS = 256
STREAM_SEED = 5354
STREAM_MEAN = 0.7
STREAM_SHARED, STREAM_RANK = 0.9, 4
ROUTER_STD, ROUTER_ALIGN, ROUTER_BIAS_STD = 0.045, 0.07, 0.055
DRAFT_SHARE = 0.6
SOFT_C2 = {"sliding": (0.3, 0.75), "full": (0.5, 0.62)}
AGENTIC_SPREAD = 0.075
WEIGHTS_SEED = 5353


def generate_weights(seed):
    gen = torch.Generator().manual_seed(seed)

    def dense(n, k, dtype=torch.bfloat16):
        return (torch.randn((n, k), generator=gen) / math.sqrt(k)).to(dtype)

    def vec(n, std):
        return (std * torch.randn((n,), generator=gen)).to(torch.bfloat16)

    w13, w13_sf, gs13 = random_weights(2 * D_EXPERT, D_EXPERT, generator=gen, device="cpu", leading=(LOCAL_EXPERTS,))
    w2, w2_sf, gs2 = random_weights(D_EXPERT, D_EXPERT, generator=gen, device="cpu", leading=(LOCAL_EXPERTS,))
    gs13 = gs13 * torch.exp(torch.randn((LOCAL_EXPERTS,), generator=gen))
    gs2 = gs2 * torch.exp(torch.randn((LOCAL_EXPERTS,), generator=gen))
    o, attn_gate, pre_ffn = dense(HIDDEN, Q_SIZE), vec(HIDDEN, 0.3), vec(HIDDEN, 0.1)
    return {
        "qkv_gate": dense(QKV_SIZE, HIDDEN), "o": o, "attn_residual_gate": attn_gate, "pre_ffn_norm": pre_ffn,
        **_router(gen),
        "pooler": dense(D_EXPERT, HIDDEN), "pre_expert_norm": vec(D_EXPERT, 0.1),
        "w13": w13, "w13_sf": w13_sf, "w13_gs": gs13, "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2,
        "post_expert_norm": vec(D_EXPERT, 0.1), "unpooler": dense(HIDDEN, D_EXPERT),
        "post_ffn_norm": vec(HIDDEN, 0.1), "ffn_residual_gate": vec(HIDDEN, 0.3), "next_pre_attn_norm": vec(HIDDEN, 0.1),
    }


@functools.cache
def _stream_basis():
    gen = torch.Generator().manual_seed(STREAM_SEED)
    mean = torch.randn((HIDDEN,), generator=gen)
    shared = torch.randn((STREAM_RANK, HIDDEN), generator=gen)
    return mean, shared


def _router(gen):
    mean, _ = _stream_basis()
    w = ROUTER_STD * torch.randn((N_EXPERTS, HIDDEN), generator=gen)
    align = ROUTER_ALIGN * torch.randn((N_EXPERTS, 1), generator=gen)
    rows = w + align * mean[None, :] / math.sqrt(HIDDEN)
    bias = ROUTER_BIAS_STD * torch.randn((N_EXPERTS,), generator=gen)
    return {"router": rows.contiguous(), "router_bias": bias.contiguous()}


def _longest(ctx, spread):
    if spread == "fixed":
        return ctx
    if spread == "agentic":
        return math.ceil(ctx * (1 + AGENTIC_SPREAD))
    return 4 * ctx


def _lengths(gen, batch, ctx, spread):
    if spread == "fixed":
        return torch.full((batch,), ctx, dtype=torch.int64)
    if spread == "agentic":
        u = torch.rand((batch,), generator=gen)
        return (ctx * (1 + AGENTIC_SPREAD * (2 * u - 1))).long().clamp(1, _longest(ctx, spread))
    z = torch.randn((batch,), generator=gen)
    return (ctx * torch.exp(z - 0.5)).clamp(16, 4 * ctx).long()


def _quant(x, scale):
    return (x.float() / scale).to(FP8)


def _dequant(x, scale):
    return x.float() * scale


def _rms_r(x, eps=RMS_EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _fill_cache(gen, shape, scale, seed, dist, layer):
    hkv = shape[-2]
    k_u8 = torch.empty(shape, dtype=torch.uint8)
    v_u8 = torch.empty(shape, dtype=torch.uint8)
    mix = None
    if dist == "soft":
        soft = torch.Generator().manual_seed(seed + 0x50F7)
        u = torch.randn((hkv, HEAD_DIM), generator=soft)
        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))
        lo, hi = SOFT_C2[layer]
        c2 = lo + (hi - lo) * torch.rand((hkv, 1), generator=soft)
        mix = (c2.sqrt() * u, (1 - c2).sqrt())
    for dst, keys in ((k_u8, True), (v_u8, False)):
        flat = dst.view(-1, hkv, HEAD_DIM)
        x = torch.randn((flat.shape[0], hkv, HEAD_DIM), generator=gen)
        if keys and mix is not None:
            x = mix[0] + mix[1] * x
        flat[:] = _quant(x, scale).view(torch.uint8)
    return k_u8.view(FP8), v_u8.view(FP8)


def generate_input(batch, ctx, layer, seed, q_len=1, spread="fixed", dist="normal", page=PAGE, kv_scale=1.0,
                   local_start=0):
    gen = torch.Generator().manual_seed(seed)
    tokens = batch * q_len
    seq_lens = _lengths(gen, batch, ctx, spread)
    pages_per = (seq_lens + q_len + page - 1) // page
    longest = _longest(ctx, spread)
    max_pages = (longest + q_len + page - 1) // page
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offsets[b])
        page_table[b, :n] = perm[o:o + n]
    positions = (seq_lens[:, None] + torch.arange(q_len)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page + pos % page).reshape(-1)
    k_cache, v_cache = _fill_cache(gen, (total, page, HKV, HEAD_DIM), kv_scale, seed, dist, layer)
    mean, shared = _stream_basis()
    coef = torch.randn((tokens, STREAM_RANK), generator=gen)
    scale = torch.ones((batch, q_len, 1))
    if 1 < q_len <= 8:
        scale[:, 1:] = DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = (scale * (STREAM_MEAN * mean + math.sqrt(STREAM_SHARED / STREAM_RANK) * coef @ shared)
              + (1 - STREAM_SHARED * scale * scale).sqrt() * torch.randn((tokens, HIDDEN), generator=gen))
    if dist == "outliers":
        idx = torch.randperm(HIDDEN, generator=gen)[: HIDDEN // 100]
        stream[:, idx] *= 10
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, tokens + 1, q_len, dtype=torch.int32)
    return {
        "stream": stream, "attn_input": _rms_r(stream).to(torch.bfloat16), "positions": positions,
        "cache_loc": cache_loc, "cache_seqlens": cache_seqlens, "cu_seqlens_q": cu_seqlens_q,
        "scheduler_metadata": None,  # FA3's schedule needs sgl_kernel; K1 ignores it, the FA3 stand-in too
        "k_cache": k_cache, "v_cache": v_cache,
        "k_scale": torch.tensor(kv_scale, dtype=torch.float32), "v_scale": torch.tensor(kv_scale, dtype=torch.float32),
        "page_table": page_table, "seq_lens": seq_lens.to(torch.int32),
        "local_start": torch.tensor([local_start], dtype=torch.int32),
        "config": {"batch": batch, "q_len": q_len, "tokens": tokens, "hidden": HIDDEN, "num_q_heads": HQ,
                   "num_kv_heads": HKV, "head_dim": HEAD_DIM, "page_size": page,
                   "window": SLIDING_WINDOW if layer == "sliding" else 0,
                   "rope_theta": ROPE_THETA if layer == "sliding" else 0.0, "qk_scale_factor": QK_SCALE_FACTOR,
                   "sm_scale": SM_SCALE, "kv_scale": kv_scale, "n_experts": N_EXPERTS, "top_k": TOP_K,
                   "local_experts": LOCAL_EXPERTS},
    }


def _rope(x, pos, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    ang = pos.float()[:, None] * inv_freq
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)


def residual_coefficients(gate):
    w = gate.float() / RESIDUAL_TEMPERATURE
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=RESIDUAL_GATE_EPS))
    return alpha, beta


def new_slots(data):
    cfg = data["config"]
    pos = data["seq_lens"].long()[:, None] + torch.arange(cfg["q_len"])
    size = cfg["page_size"]
    page = torch.gather(data["page_table"].long(), 1, pos // size)
    return page * size + pos % size


def _attention(data, x, weights, query_dtype=torch.bfloat16):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    qkv = x @ weights["qkv_gate"].float().T
    q, k, v, gate = qkv.split([Q_SIZE, KV_SIZE, KV_SIZE, Q_SIZE], dim=-1)
    pos = (data["seq_lens"].long()[:, None] + torch.arange(length)).reshape(-1)
    q = _rms_r(q.view(-1, HQ, HEAD_DIM))
    k = _rms_r(k.view(-1, HKV, HEAD_DIM))
    if cfg["rope_theta"]:
        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])
    if query_dtype == torch.bfloat16:
        q = (q * cfg["qk_scale_factor"]).to(torch.bfloat16).float()
    else:
        q = q.to(query_dtype).float() * cfg["qk_scale_factor"]
    q = q.view(batch, length, HQ, HEAD_DIM)
    k_new = _quant(k, data["k_scale"]).view(batch, length, HKV, HEAD_DIM)
    v_new = _quant(v.reshape(-1, HKV, HEAD_DIM), data["v_scale"]).view(batch, length, HKV, HEAD_DIM)
    k_flat = data["k_cache"].view(torch.uint8).view(-1, HKV, HEAD_DIM)
    v_flat = data["v_cache"].view(torch.uint8).view(-1, HKV, HEAD_DIM)
    group, window = HQ // HKV, cfg["window"]
    out = torch.empty((batch, length, HQ, HEAD_DIM))
    steps = torch.arange(length)
    for b in range(batch):
        seq = int(data["seq_lens"][b])
        first = max(0, seq - window + 1) if window else 0
        past = torch.arange(first, seq)
        size = cfg["page_size"]
        rows = data["page_table"][b].long()[past // size] * size + past % size
        keys = torch.cat([_dequant(k_flat[rows].view(FP8), data["k_scale"]), _dequant(k_new[b], data["k_scale"])])
        vals = torch.cat([_dequant(v_flat[rows].view(FP8), data["v_scale"]), _dequant(v_new[b], data["v_scale"])])
        key_pos = torch.cat([past, seq + steps])
        query_pos = seq + steps
        visible = key_pos[None, :] <= query_pos[:, None]
        if window:
            visible &= key_pos[None, :] > query_pos[:, None] - window
        n = query_pos.numel()
        att = torch.einsum("lkgd,tkd->lkgt", q[b].reshape(n, HKV, group, HEAD_DIM), keys) * cfg["sm_scale"]
        att = att.masked_fill(~visible[:, None, None, :], float("-inf")).softmax(-1)
        out[b] = torch.einsum("lkgt,tkd->lkgd", att, vals).reshape(n, HQ, HEAD_DIM)
    out = _rms_r(out).reshape(-1, Q_SIZE) * torch.sigmoid(gate)
    return out, k_new, v_new


def route(h, weights):
    logits = h.double() @ weights["router"].double().T
    scores = torch.sigmoid(logits * ROUTER_MULTIPLIER).float()
    return scores, scores + weights["router_bias"].float()


def post_expert_norm(y, w):
    y = y.float() * (1.0 + w.float())
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)


_DEQ = {}
_DEQ_MAX = 12   # bounded (each entry is 67-134 MB of FP32; this Mac has 16 GB)


def _deq_ref(weights, e, which):
    key = (id(weights), e, which)
    if key in _DEQ:
        _DEQ[key] = _DEQ.pop(key)   # LRU refresh
    else:
        while len(_DEQ) >= _DEQ_MAX:
            _DEQ.pop(next(iter(_DEQ)))
        _DEQ[key] = dequantize(weights[which][e], weights[which + "_sf"][e], weights[which + "_gs"][e])
    return _DEQ[key]


def _experts(x, ids, topk_weights, weights):
    out = torch.zeros(x.shape, dtype=torch.float32)
    for e in torch.unique(ids).tolist():
        if not 0 <= e < LOCAL_EXPERTS:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ _deq_ref(weights, e, "w13").T
        inter = F.silu(h[:, :D_EXPERT]) * h[:, D_EXPERT:]
        y = inter @ _deq_ref(weights, e, "w2").T
        out.index_add_(0, tok, post_expert_norm(y, weights["post_expert_norm"]) * topk_weights[tok, slot, None])
    return out


def ref_front(data, weights, query_dtype=torch.bfloat16):
    o, k_new, v_new = _attention(data, data["attn_input"].float(), weights, query_dtype)
    a = o @ weights["o"].float().T
    alpha1, beta1 = residual_coefficients(weights["attn_residual_gate"])
    stream1 = alpha1 * data["stream"].float() + beta1 * _rms_r(a, POST_NORM_EPS)
    h = _rms_r(stream1) * (1.0 + weights["pre_ffn_norm"].float())
    scores, biased = route(h, weights)
    return {"stream1": stream1, "h": h, "scores": scores, "biased": biased, "k_new": k_new, "v_new": v_new,
            "local_start": int(data["local_start"])}


def ref_back(front, weights, ids):
    picked = front["scores"].gather(1, ids.long())
    topk_weights = picked / (picked.sum(-1, keepdim=True) + ROUTE_EPS) * ROUTE_SCALE
    p = front["h"] @ weights["pooler"].float().T
    p = _rms_r(p) * (1.0 + weights["pre_expert_norm"].float())
    local = (ids.long() - front["local_start"]) % N_EXPERTS
    moe = _experts(p, local, topk_weights, weights)
    if len(moe) <= SLICED_MAX_TOKENS:
        f = torch.zeros_like(front["stream1"])
        f[:, :UNPOOL_SLICE] = moe @ weights["unpooler"][:UNPOOL_SLICE].float().T
    else:
        f = moe @ weights["unpooler"].float().T
    f = f * (1.0 + weights["post_ffn_norm"].float())
    alpha2, beta2 = residual_coefficients(weights["ffn_residual_gate"])
    stream2 = alpha2 * front["stream1"] + beta2 * _rms_r(f, POST_NORM_EPS)
    hidden = _rms_r(stream2) * (1.0 + weights["next_pre_attn_norm"].float())
    return hidden.to(torch.bfloat16), stream2


def busiest_local_start(data, weights):
    """case_fields: the 32-expert range this input's reference routing sends the most pairs to."""
    ids = ref_front({**data, "local_start": torch.tensor([0], dtype=torch.int32)}, weights)["biased"].topk(TOP_K, -1).indices
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    return LOCAL_EXPERTS * int(pairs.argmax())


def _local_load(ids, local_start=0):
    local = ids[(ids - local_start) % N_EXPERTS < LOCAL_EXPERTS]
    return int(local.numel()), int(torch.unique(local).numel())


def check_routing(ids, biased, local_start=0):
    if ids.min() < 0 or ids.max() >= N_EXPERTS:
        return "expert id out of range"
    ranked = ids.long().sort(-1).values
    if (ranked[:, 1:] == ranked[:, :-1]).any():
        return "a token routed to the same expert twice"
    best = biased.topk(TOP_K, dim=-1)
    gap = best.values[:, -1:] - biased.gather(1, ids.long())
    if gap.max() > ROUTE_TOL:
        return f"routed to an expert {float(gap.max()):.2e} below its 8th best"
    ninth = biased.topk(TOP_K + 1, dim=-1).values[:, -1:]
    keep_tol = KEEP_TOL[0] if len(ids) <= KEEP_TOKENS else KEEP_TOL[1]
    kept = (best.indices[:, :, None] == ids.long()[:, None, :]).any(-1)
    local = (best.indices - local_start) % N_EXPERTS < LOCAL_EXPERTS
    lost = local & ~kept & (best.values - ninth > keep_tol)
    if lost.any():
        return "dropped a clear local expert"
    routes, distinct = _local_load(ids.long(), local_start)
    ref_routes, ref_distinct = _local_load(best.indices, local_start)
    if routes < ref_routes * (1 - LOCAL_SLACK[0]) - LOCAL_SLACK[1]:
        return f"{routes} local routes vs {ref_routes}"
    if distinct < ref_distinct * (1 - DISTINCT_SLACK[0]) - DISTINCT_SLACK[1]:
        return f"{distinct} local experts vs {ref_distinct}"
    return None


def check_implementation(data, hidden, stream, ids, k_u8, v_u8, weights):
    """reference.check_implementation on the step's outputs and the cache rows it wrote (uint8)."""
    front = ref_front(data, weights)
    error = check_routing(ids, front["biased"], front["local_start"])
    if error:
        return False, f"routing: {error}", {}
    ref_hidden, ref_stream = ref_back(front, weights, ids)
    alpha1, _ = residual_coefficients(weights["attn_residual_gate"])
    alpha2, _ = residual_coefficients(weights["ffn_residual_gate"])
    carried = alpha1 * alpha2 * data["stream"].float()
    e4m3_hidden, e4m3_stream = ref_back(ref_front(data, weights, torch.float8_e4m3fn), weights, ids)
    bound = lambda err: query_precision_bound(err, FP8_QUERY_FACTOR, FP8_QUERY_FLOOR, REL_L2_TOL)  # noqa: E731
    tol = bound(relative_l2(e4m3_stream - carried, ref_stream - carried))
    hidden_tol = bound(relative_l2(e4m3_hidden, ref_hidden))
    info = {"upd_err": relative_l2(stream - carried, ref_stream - carried), "upd_tol": tol,
            "hid_err": relative_l2(hidden, ref_hidden), "hid_tol": hidden_tol,
            "e4m3_upd": relative_l2(e4m3_stream - carried, ref_stream - carried)}
    ok, msg = compare(stream - carried, ref_stream - carried, tol)
    if not ok:
        return False, f"residual update: {msg} [bound {tol:.3f}]", info
    ok, hmsg = compare(hidden, ref_hidden, hidden_tol)
    if not ok:
        return False, f"next attention input: {hmsg} [bound {hidden_tol:.3f}]", info
    for name, got, want, scale in (("key", k_u8, front["k_new"], data["k_scale"]), ("value", v_u8, front["v_new"], data["v_scale"])):
        g = _dequant(got.view(FP8), scale)
        w = _dequant(want, scale)
        ok, kv_msg = compare(g, w, CACHE_TOL)
        info[f"{name}_err"] = relative_l2(g, w)
        if not ok:
            return False, f"new {name} cache entries: {kv_msg}", info
    return True, f"update {msg} [bound {tol:.3f}]; attention input {hmsg} [bound {hidden_tol:.3f}]", info


# ------------------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------------------
CASES = {  # name: generate_input kwargs (+ "force_local_start")
    "T1 sliding ctx10 page64": dict(batch=1, ctx=10, layer="sliding", seed=301),
    "T1 full ctx3000 page1 soft": dict(batch=1, ctx=3000, layer="full", seed=302, page=1, dist="soft"),
    "T8 sliding L8 ctx2500 page1 soft": dict(batch=1, ctx=2500, layer="sliding", seed=303, q_len=8, page=1, dist="soft"),
    "T16 sliding L8 page1 kv1/64 ls96": dict(batch=2, ctx=700, layer="sliding", seed=304, q_len=8, page=1,
                                             kv_scale=0.015625, force_local_start=96),
    "T24 full L8 page64 outliers": dict(batch=3, ctx=600, layer="full", seed=305, q_len=8, dist="outliers"),
    "T33 full L1 page1 mixed": dict(batch=33, ctx=300, layer="full", seed=306, page=1, spread="mixed"),
}

SCENARIOS = {
    # name: (file, cases, FAIL overrides, replace _experts with the CPU stand-in, expected paths, fast self-tests)
    "main": ("k3_v0.py", list(CASES), {}, False, ("K1", "K2A"), False),
    "k2a_only": ("k3_v0_k2a_only.py", ["T1 sliding ctx10 page64", "T16 sliding L8 page1 kv1/64 ls96"], {}, False,
                 ("FA3", "K2A"), True),
    "k1_only": ("k3_v0_k1_only.py", ["T1 full ctx3000 page1 soft", "T24 full L8 page64 outliers"], {}, True,
                ("K1", "Marlin"), True),
    "k1_build_fails": ("k3_v0.py", ["T8 sliding L8 ctx2500 page1 soft"], {"k1_build": True}, False, ("FA3", "K2A"), True),
    "k1_selftest_fails": ("k3_v0.py", ["T1 sliding ctx10 page64"], {"k1_corrupt": True}, False, ("FA3", "K2A"), True),
    "k2a_build_fails": ("k3_v0.py", ["T1 full ctx3000 page1 soft"], {"k2a_build": True}, True, ("K1", "Marlin"), True),
    "k2a_selftest_fails": ("k3_v0.py", ["T16 sliding L8 page1 kv1/64 ls96"], {"k2a_corrupt": True}, True,
                           ("K1", "Marlin"), True),
    # call 1 is eager (K1 runs), call 2 is "captured" and K1 raises there -> r1 FA3 from then on
    "k1_raises_at_call": ("k3_v0.py", ["T1 sliding ctx10 page64", "T8 sliding L8 ctx2500 page1 soft",
                                       "T1 full ctx3000 page1 soft"],
                          {"k1_raise_at_call": True}, False, [("K1", "K2A"), ("FA3", "K2A"), ("FA3", "K2A")], True),
}

_PREPACK = {}


def main(argv):
    t00 = time.time()
    names = argv or list(SCENARIOS)
    weights = generate_weights(WEIGHTS_SEED)
    print(f"[emu] layer weights generated ({time.time() - t00:.0f}s)", flush=True)
    inputs = {}
    all_ok = True
    for sname in names:
        fname, cases, fails, cpu_experts, expect, fast = SCENARIOS[sname]
        for k in FAIL:
            FAIL[k] = fails.get(k, False)
        CALLS.clear()
        t0 = time.time()
        print(f"\n===== scenario {sname}: {fname} FAIL={ {k: v for k, v in FAIL.items() if v} } =====", flush=True)
        CAPTURE[0] = False
        mod = _load(f"k3_{sname}", HERE / fname)
        orig_prepack = mod._k2a_prepack_expert

        def cached_prepack(w, sf, gate_up, _o=orig_prepack):   # harness-only: reuse the repack across scenarios
            key = (w.data_ptr(), gate_up)
            if key not in _PREPACK:
                _PREPACK[key] = _o(w, sf, gate_up)
            return _PREPACK[key]
        mod._k2a_prepack_expert = cached_prepack
        if fast:  # shorter self-tests for the bisect/fallback scenarios (the main scenario runs the real lists)
            mod.K2A_SELF_TEST_CASES = ((1, 11, False), (40, 13, False))
            mod.K1_SELF_TEST_CASES = mod.K1_SELF_TEST_CASES[:2] + mod.K1_SELF_TEST_CASES[6:7]
        mod._experts = make_cpu_experts(weights)   # Marlin stand-in; only reached if k3 dispatches to r1's experts
        state = mod.prepare(weights)
        print(f"[emu] prepare done ({time.time() - t0:.0f}s): k1_ok={state['k1_ok']} k2a={'on' if state['k2a'] else 'off'}",
              flush=True)
        for i, cname in enumerate(cases):
            kw = dict(CASES[cname])
            force = kw.pop("force_local_start", None)
            if cname not in inputs:
                d0 = generate_input(**kw)
                ls = force if force is not None else busiest_local_start(d0, weights)
                inputs[cname] = (kw, ls)
            kw, ls = inputs[cname]
            data = generate_input(**kw, local_start=ls)      # fresh caches every run
            k_before = data["k_cache"].view(torch.uint8).clone()
            n0 = dict(CALLS)
            CAPTURE[0] = i > 0       # first call eager (as the judge's first test), the rest under simulated capture
            t1 = time.time()
            hidden, stream, ids = mod.kernel_fn(data, state)
            CAPTURE[0] = False
            dt = time.time() - t1
            used = {q: CALLS.get(q, 0) - n0.get(q, 0) for q in CALLS}
            # FA3 ran => the r1 path produced o (a K1 call that raised before it is reported separately)
            path_attn = "FA3" if used.get("flash_attn_with_kvcache") else ("K1" if used.get("FakeK1.k1_forward") else "?")
            if path_attn == "FA3" and used.get("FakeK1.k1_forward"):
                print("[emu]   (K1 was called and raised; this call fell back to r1 FA3)", flush=True)
            path_moe = "K2A" if used.get("FakeK2A.route") else ("Marlin" if used.get("make_cpu_experts.<locals>.experts") else "?")
            slots = new_slots(data).reshape(-1)
            k_u8 = data["k_cache"].view(torch.uint8).view(-1, HKV, HEAD_DIM)[slots].view(kw["batch"], kw.get("q_len", 1), HKV, HEAD_DIM)
            v_u8 = data["v_cache"].view(torch.uint8).view(-1, HKV, HEAD_DIM)[slots].view(kw["batch"], kw.get("q_len", 1), HKV, HEAD_DIM)
            changed = (data["k_cache"].view(torch.uint8).reshape(-1, HKV * HEAD_DIM) != k_before.reshape(-1, HKV * HEAD_DIM)).any(-1)
            outside = sorted(set(changed.nonzero().reshape(-1).tolist()) - set(slots.tolist()))
            shapes_ok = (hidden.shape == (data["config"]["tokens"], HIDDEN) and hidden.dtype == torch.bfloat16
                         and stream.shape == (data["config"]["tokens"], HIDDEN) and stream.dtype == torch.float32
                         and ids.shape == (data["config"]["tokens"], TOP_K) and ids.dtype == torch.int32)
            ok, msg, info = check_implementation(data, hidden, stream, ids, k_u8, v_u8, weights)
            local = ((ids.long() - ls) % N_EXPERTS < LOCAL_EXPERTS)
            zero_tok = int((~local.any(-1)).sum())
            want = expect[i] if isinstance(expect, list) else expect
            path_ok = (path_attn, path_moe) == want
            good = ok and shapes_ok and not outside and path_ok
            all_ok &= good
            print(f"[{'PASS' if good else 'FAIL'}] {cname:34s} paths={path_attn}/{path_moe} (want {want[0]}/{want[1]})"
                  f" capture={'yes' if i > 0 else 'no'} local_start={ls} local_routes={int(local.sum())}"
                  f" tokens_without_local={zero_tok} | {msg} | upd {info.get('upd_err', float('nan')):.2e}"
                  f" (ref E4M3 {info.get('e4m3_upd', float('nan')):.2e}) hid {info.get('hid_err', float('nan')):.2e}"
                  f" K {info.get('key_err', float('nan')):.2e} V {info.get('value_err', float('nan')):.2e}"
                  f" shapes={'ok' if shapes_ok else 'BAD'} writes_outside_slots={len(outside)} ({dt:.0f}s run,"
                  f" {time.time() - t1:.0f}s with check)", flush=True)
        del state, mod
        print(f"[emu] scenario {sname} done in {time.time() - t0:.0f}s", flush=True)
    print(f"\n{'ALL PASS' if all_ok else 'SOME FAILED'} ({time.time() - t00:.0f}s)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

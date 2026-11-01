#!/usr/bin/env python3
"""RQ in-file lab generator. CPU only; never imports the layer or starts a GPU job.

python -B tools/rq_patches.py             # write only kernels/lab/k3_rq_ab.py
python -B tools/rq_patches.py --check     # verify output and run structural dry run
python -B tools/rq_patches.py --torch-cpu # additionally run real CPU packing/validation
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import time
import tokenize
import types

_rq_root = Path(__file__).resolve().parents[1]
_rq_base = _rq_root / "kernels/lab/k3_vmsx_ship_rt.py"
_rq_vendor = _rq_root / "vendor_ext/chenwainuo/submission.py"
_rq_out = _rq_root / "kernels/lab/k3_rq_ab.py"
_rq_anchors = []


def _rq_sub(text, old, new, what):
    # Exact-match discipline copied from tools/padz_patches.py:48-51.
    count = text.count(old)
    assert count == 1, f"anchor {what}: found {count}x, expected 1: {old[:100]!r}"
    _rq_anchors.append({"name": what, "line": text[:text.index(old)].count("\n") + 1,
                        "old": old, "new": new})
    return text.replace(old, new)


def _rq_node(text, name):
    nodes = [n for n in ast.parse(text).body
             if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))]
    assert len(nodes) == 1, (name, len(nodes))
    node = nodes[0]
    part = ast.get_source_segment(text, node)
    assert text.count(part) == 1, name
    return node, part


def _rq_rename(text, names):
    return tokenize.untokenize(t._replace(string=names.get(t.string, t.string))
                              if t.type == tokenize.NAME else t
                              for t in tokenize.generate_tokens(io.StringIO(text).readline))


_rq_glue = r'''
# RQ BEGIN: chenwainuo/okc RS128, Apache-2.0; local integration by team joeyhlu.
_rq_notice = """
RQ dense GEMM: derived from chenwainuo/okc, submission.py RQCU and lossless
12-bit BF16 pack/escape helpers, Apache License 2.0 (see license text above).
Upstream repository LICENSE also retains: Copyright 2023-2024 SGLang Team.
Integration reference: keell0renz/okc muse_decode_layer.py (_make_chen/_rq_gemm).
Modifications: rq-prefixed names, two fixed instances, selector and guarded
prepare, independent validation, no timeout traps, occupancy admission check.
CUDA compute, packing and split-K reduction algorithms retain upstream logic.
"""
THIRD_PARTY_NOTICES += _rq_notice
RQ_SEL = frozenset()
_rq_picks = {128: (0, "d2s5hzwP"), 256: (1, "n256s5m1hzmEH")}
_rq_ks = {"qkv": 7, "o": 2, "unpooler": 2, "pooler": 4}
_rq_tol = {2: 5e-3, 4: 6e-3, 7: 8e-3}
_rq_budget_s = 60.0
_rq_build_cache = {"mod": None, "error": None}
_rq_cpp = "int64_t rq_run(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);"


def _rq_build():
    if _rq_build_cache["error"] is not None:
        raise RuntimeError(_rq_build_cache["error"])
    if _rq_build_cache["mod"] is None:
        import hashlib as _rq_hashlib
        from torch.utils.cpp_extension import load_inline as _rq_load_inline
        rq_src = _rqsrc(_rq_variants)
        rq_tag = _rq_hashlib.sha256((rq_src + repr(_rq_flags)).encode()).hexdigest()[:12]
        try:
            _rq_build_cache["mod"] = _rq_load_inline(
                name="rq_layer_" + rq_tag, cpp_sources=_rq_cpp, cuda_sources=rq_src,
                functions=["rq_run"], extra_cuda_cflags=_rq_flags, verbose=False)
        except Exception as rq_error:
            _rq_build_cache["error"] = str(rq_error)
            raise
    return _rq_build_cache["mod"]


def _rq_weight(state, proj):
    # Exact base accessor: _dg1_weight, including the qkv_gate spelling.
    return state["dense"]["qkv_gate" if proj == "qkv" else proj]


def _rq_keys():
    # Invalid / unsupported arms never reach a pack or a launch.
    return tuple(sorted((rq_p, rq_t) for rq_p, rq_t in RQ_SEL
                        if rq_p in _rq_ks and rq_t in _rq_picks))


def _rq_call(rq_p, rq_x, rq_out, rq_info=0):
    return rq_p["rq_mod"].rq_run(
        rq_p["rq_i"], rq_p["rq_lay"], rq_x, rq_out, rq_p["rq_esc"], rq_p["rq_esc2"],
        rq_p["rq_cnt"], rq_p["rq_ks"], rq_p["rq_tab"], 0, rq_p["rq_prof"], rq_p["rq_sink"], rq_info)


def _rq_pack_weight(rq_w):
    rq_n, rq_k = rq_w.shape
    if rq_w.dtype != torch.bfloat16 or rq_n % 128 or rq_k % 64 or not rq_w.is_contiguous():
        raise ValueError("unsupported weight layout")
    rq_pk = _rqpack(rq_w)
    if rq_pk is None or not torch.equal(_rqunpack(*rq_pk, rq_n, rq_k), rq_w.view(torch.int16).int() & 0xffff):
        raise ValueError("lossless pack roundtrip failed")
    return rq_pk


def _rq_validate(rq_p, rq_w, rq_t, rq_gen):
    # No timing/autotuning. Random + outlier BF16 inputs, torch.mm reference,
    # per-row tolerance, poisoned output, repeated launches and counter reset.
    rq_tol = _rq_tol[rq_p["rq_ks"]]
    rq_out = torch.empty((rq_t, rq_w.shape[0]), dtype=torch.bfloat16, device=rq_w.device)
    for rq_scale in (1.0, 30.0):
        rq_x = torch.randn((rq_t, rq_w.shape[1]), generator=rq_gen, device=rq_w.device)
        if rq_scale != 1.0:
            rq_x[:, :max(1, rq_w.shape[1] // 100)] *= rq_scale
        rq_x = rq_x.to(torch.bfloat16)
        rq_ref = torch.mm(rq_x, rq_w.t()).float()
        for rq_repeat in range(2):
            rq_out.fill_(float("nan"))
            if _rq_call(rq_p, rq_x, rq_out) != 0:
                return False, "launch rejected"
            rq_delta = rq_out.float() - rq_ref
            rq_err = float(rq_delta.norm() / rq_ref.norm().clamp(min=1e-30))
            rq_row = float((rq_delta.norm(dim=-1) / rq_ref.norm(dim=-1).clamp(min=1e-30)).max())
            if not (bool(torch.isfinite(rq_out).all()) and rq_err <= rq_tol and rq_row <= 5 * rq_tol):
                return False, "random/outlier tolerance"
            if int(rq_p["rq_cnt"].abs().sum()) != 0:
                return False, "counters not reset"
    return True, "rel=%.3g row=%.3g tol=%.3g" % (rq_err, rq_row, rq_tol)


def _rq_prepare(state):
    rq_keys = _rq_keys()
    if not rq_keys:
        return  # Empty selector leaves the ship state and all build paths intact.
    import time as _rq_time
    rq_t0 = _rq_time.monotonic()
    rq_bank = state["rq_bank"] = {}
    try:
        rq_mod = _rq_build()
    except Exception as rq_error:
        for rq_key in rq_keys:
            print("[rq] %s fallback: build %s" % (rq_key, str(rq_error)[:180]), flush=True)
        return
    rq_packs = {}
    for rq_proj, rq_t in rq_keys:
        rq_key = (rq_proj, rq_t)
        try:
            if _rq_time.monotonic() - rq_t0 >= _rq_budget_s:
                raise RuntimeError("prepare time budget")
            rq_w = _rq_weight(state, rq_proj)
            rq_n, rq_k = rq_w.shape
            if rq_proj not in rq_packs:
                rq_packs[rq_proj] = _rq_pack_weight(rq_w)
            rq_lay, rq_tab, rq_esc6 = rq_packs[rq_proj]
            rq_ks = _rq_ks[rq_proj]
            rq_esc = _rqesc(rq_esc6, rq_n // 128, rq_k // 64, rq_ks, rq_w.device)[0]
            rq_i, rq_name = _rq_picks[rq_t]
            if _rq_variants[rq_i][7] & 1048576:
                rq_esc2, rq_max = _rqesc2(rq_esc6, rq_n // 128, rq_k // 64, rq_ks, rq_w.device)
                if rq_esc2 is None:
                    raise ValueError("escape cursor overflow (%d)" % rq_max)
            else:
                rq_esc2 = torch.empty(0, dtype=torch.int32, device=rq_w.device)
            rq_read_esc = rq_esc2 if _rq_variants[rq_i][7] & 1048576 else rq_esc
            rq_weight_bytes = rq_lay.numel() * rq_lay.element_size()
            rq_escape_bytes = rq_read_esc.numel() * rq_read_esc.element_size()
            if rq_weight_bytes + rq_escape_bytes >= rq_w.numel() * rq_w.element_size():
                raise ValueError("packed weight/escape stream does not shrink")
            rq_p = {"rq_mod": rq_mod, "rq_i": rq_i, "rq_name": rq_name, "rq_lay": rq_lay,
                    "rq_tab": rq_tab, "rq_esc": rq_esc, "rq_esc2": rq_esc2, "rq_ks": rq_ks,
                    "rq_cnt": torch.zeros(8 * (rq_n // 128), dtype=torch.int32, device=rq_w.device),
                    "rq_prof": torch.empty(0, dtype=torch.int64, device=rq_w.device),
                    "rq_sink": torch.zeros(4, dtype=torch.int32, device=rq_w.device),
                    "rq_n": rq_n, "rq_k": rq_k, "rq_device": rq_w.device,
                    "rq_weight_bytes": rq_weight_bytes, "rq_escape_bytes": rq_escape_bytes}
            # Self-zeroing polls require every split of a row block to be resident.
            # Conservative admission: the entire grid must fit simultaneously.
            rq_occ = _rq_call(rq_p, rq_w, rq_w, 4)
            rq_sms = torch.cuda.get_device_properties(rq_w.device).multi_processor_count
            if rq_occ < 1 or (rq_n // 128) * rq_ks > rq_sms * rq_occ:
                raise ValueError("self-zero grid exceeds residency")
            rq_gen = torch.Generator(device=rq_w.device).manual_seed(12828 + rq_t)
            rq_ok, rq_msg = _rq_validate(rq_p, rq_w, rq_t, rq_gen)
            if not rq_ok:
                raise ValueError(rq_msg)
            rq_bank[rq_key] = rq_p
            print("[rq] %s pick=%s KS=%d %s" % (rq_key, rq_name, rq_ks, rq_msg), flush=True)
        except Exception as rq_error:
            print("[rq] %s fallback: %s" % (rq_key, str(rq_error)[:180]), flush=True)
    # Pack temporaries are local; only validated picks retain packed weights.


def _rq_gemm(state, proj, x):
    rq_key = (proj, x.shape[0])
    if rq_key not in RQ_SEL:
        return None
    rq_p = state.get("rq_bank", {}).get(rq_key)
    if rq_p is None or x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != rq_p["rq_k"]:
        return None
    if x.device != rq_p["rq_device"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    try:
        rq_out = torch.empty((x.shape[0], rq_p["rq_n"]), dtype=torch.bfloat16, device=x.device)
        return rq_out if _rq_call(rq_p, x, rq_out) == 0 else None
    except Exception:
        return None  # No host sync, logging, lazy build, or state mutation in this path.


def _rq_pooler_tail(state, x, data, tail):
    if tail != "cat":
        return None
    rq_y = _rq_gemm(state, "pooler", x)
    if rq_y is None:
        return None
    # Existing stock hi/lo path: both mm results and the sum stay FP32.
    rq_logits = _v8_logits("stock", x, None, state)
    return _v8_tail(rq_y, rq_logits, state, data)
'''


def _rq_graft(vendor):
    rq_vs_node, _ = _rq_node(vendor, "RQV")
    rq_vlist = ast.literal_eval(rq_vs_node.value)
    rq_wanted = ("d2s5hzwP", "n256s5m1hzmEH")
    rq_vs = [next(v for v in rq_vlist if v[0] == name) for name in rq_wanted]
    rq_cuda_node, _ = _rq_node(vendor, "RQCU")
    rq_cuda = ast.literal_eval(rq_cuda_node.value)
    # No new early exit: every zero publisher and epilogue issuer still completes.
    rq_cuda = _rq_sub(rq_cuda,
        'for (int it = 0; it < (1 << 22); it++) {\n          asm volatile("ld.acquire.gpu.global.u32',
        'while ((int)c < KS) {\n          asm volatile("ld.acquire.gpu.global.u32', "rq zero wait")
    rq_cuda = _rq_sub(rq_cuda,
        '        if ((int)c < KS) asm volatile("trap;");  // not co-resident: fail loudly rather than reduce into unzeroed memory\n',
        '', "rq zero timeout trap")
    for rq_ending, rq_tag in [('        asm volatile("fence.proxy.async.global;\\n" ::: "memory");', 'EH'),
                               ('        if (!EFENCE) asm volatile("fence.proxy.async.global;\\n" ::: "memory");', 'normal')]:
        rq_old = ('      if (ZI) {  // the block\'s zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)\n'
                  '        uint32_t f = 0;\n'
                  '        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\\n" : "=r"(f) : "r"(zflag) : "memory");\n'
                  '        if (!f) asm volatile("trap;");\n' + rq_ending)
        rq_new = ('      if (ZI) {  // zero/epilogue coverage is preserved: do not exit before publication\n'
                  '        uint32_t f = 0;\n'
                  '        while (!f) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\\n" : "=r"(f) : "r"(zflag) : "memory");\n' + rq_ending)
        rq_cuda = _rq_sub(rq_cuda, rq_old, rq_new, "rq epilogue wait " + rq_tag)
    rq_cuda = _rq_sub(rq_cuda, "Bounded: a non-co-resident grid traps instead of hanging.",
                      "Prepare admits only a fully resident grid; no timeout exit.", "rq residency comment")
    for rq_old, rq_new in {"RqP": "rq_P", "RqTmap": "rq_Tmap", "RqC": "rq_C", "RqEnc": "rq_Enc",
                           "rq": "rq_kernel", "rqrun": "rq_run", "RQ_T0": "rq_T0", "RQ_T1": "rq_T1"}.items():
        rq_cuda = re.sub(r"\b" + rq_old + r"\b", rq_new, rq_cuda)
    assert 'trap;' not in rq_cuda
    rq_names = {"RQCU": "_rq_cuda", "RQMS": "_rq_ms", "_OS": "_rq_os", "_SS": "_rq_ss",
                "_pack_nat": "_rq_pack_nat", "_c12": "_rq_c12", "_unc12": "_rq_unc12"}
    rq_chunks = [_rq_glue, '_rq_cuda = r"""' + rq_cuda + '"""\n',
                 "_rq_variants = " + repr(rq_vs) + "\n_rq_ms = {}\n"]
    for rq_name in ("_OS", "_SS", "_c12", "_unc12", "_pack_nat", "_rqpack", "_rqunpack", "_rqesc", "_rqesc2",
                    "_rqwg", "_rqss", "_rqsrc"):
        _, rq_part = _rq_node(vendor, rq_name)
        if rq_name == "_rqesc2":
            # PyTorch cannot construct int32 from the positive Python 0xffffffff.
            rq_part = _rq_sub(rq_part,
                'torch.full((G * 32 * CAPT,), 0xffffffff, dtype=torch.int32, device=dev)',
                'torch.full((G * 32 * CAPT,), -1, dtype=torch.int32, device=dev)', "rq empty escape sentinel")
        rq_chunks.append(_rq_rename(rq_part, rq_names) + "\n")
    rq_flags_node, _ = _rq_node(vendor, "RQFLAGS")
    rq_chunks.append("_rq_flags = " + repr(ast.literal_eval(rq_flags_node.value)) + "\n")
    rq_chunks.append("# RQ END\n")
    rq_result = "\n".join(rq_chunks)
    # _rqsrc uses rq_go (kept) and includes renamed rq_run from _rq_cuda.
    return rq_result


def _rq_generate():
    _rq_anchors.clear()
    rq_ship = _rq_base.read_text(encoding="utf-8")
    rq_vendor_text = _rq_vendor.read_text(encoding="utf-8")
    rq_block = _rq_graft(rq_vendor_text)
    rq_text = rq_ship
    rq_changes = [
        ("    _gft_prepare(state, t0)\n", "    _gft_prepare(state, t0)\n    _rq_prepare(state)\n", "prepare hook"),
        ('    qkv = _sk2(state, "qkv", data["attn_input"]) if SK2_ON else None\n',
         '    qkv = _rq_gemm(state, "qkv", data["attn_input"])\n    if qkv is None:\n        qkv = _sk2(state, "qkv", data["attn_input"]) if SK2_ON else None\n', "qkv arm"),
        ('    attn_output = _gd(state, "o", o)\n',
         '    attn_output = _rq_gemm(state, "o", o)\n    if attn_output is None:\n        attn_output = _gd(state, "o", o)\n', "o arm"),
        ('    ffn_output = _sk2(state, "unpooler", expert_output) if SK2_ON else None\n',
         '    ffn_output = _rq_gemm(state, "unpooler", expert_output)\n    if ffn_output is None:\n        ffn_output = _sk2(state, "unpooler", expert_output) if SK2_ON else None\n', "unpooler arm"),
        ('    g2out = _g2_prt(ffn_input, router_input, state, data, tokens, mode, tail)\n',
         '    g2out = _rq_pooler_tail(state, ffn_input, data, tail)\n    if g2out is None:\n        g2out = _g2_prt(ffn_input, router_input, state, data, tokens, mode, tail)\n', "pooler cat arm"),
        ('def _v13_pooler(ffn_input, dense, state):\n',
         'def _v13_pooler(ffn_input, dense, state):\n    rq_y = _rq_gemm(state, "pooler", ffn_input)\n    if rq_y is not None:\n        return rq_y\n', "pooler separate fallback arm"),
    ]
    for rq_old, rq_new, rq_tag in rq_changes:
        rq_text = _rq_sub(rq_text, rq_old, rq_new, rq_tag)
    rq_text = _rq_sub(rq_text, "def kernel_fn(data, state):\n", rq_block + "\ndef kernel_fn(data, state):\n", "RQ block")
    ast.parse(rq_text)
    assert "\r" not in rq_text
    # Strong default-equivalence structural gate: remove ONLY our graft/hooks.
    rq_reverse = rq_text.replace(rq_block + "\n", "", 1)
    for rq_old, rq_new, _ in reversed(rq_changes):
        assert rq_reverse.count(rq_new) == 1
        rq_reverse = rq_reverse.replace(rq_new, rq_old)
    assert rq_reverse == rq_ship, "unrelated base modification"
    return rq_text, rq_block


def _rq_namespace(rq_text):
    rq_nodes = []
    for rq_node_ in ast.parse(rq_text).body:
        if isinstance(rq_node_, ast.FunctionDef) and rq_node_.name.startswith("_rq"):
            rq_nodes.append(rq_node_)
        elif isinstance(rq_node_, ast.Assign) and all(isinstance(t, ast.Name) and
                (t.id.startswith("_rq") or t.id == "RQ_SEL") for t in rq_node_.targets):
            rq_nodes.append(rq_node_)
    rq_ns = {}
    exec(compile(ast.Module(body=rq_nodes, type_ignores=[]), "rq-extracted", "exec"), rq_ns)
    return rq_ns


class _rq_FakeTensor:
    def __init__(self, shape, dtype="bf16", device="cpu", value=0):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype, self.device, self.value = dtype, device, value
    def contiguous(self): return self
    def is_contiguous(self): return True
    def dim(self): return len(self.shape)
    def data_ptr(self): return 16
    def numel(self):
        import math
        return math.prod(self.shape)
    def element_size(self): return {"bf16": 2, "i32": 4, "i64": 8, "u8": 1}[self.dtype]
    def abs(self): return self
    def sum(self): return self.value


def _rq_dry_run(rq_text):
    """Execute real new orchestration; GPU/compiler/numeric leaves are explicit doubles.

    The dense state comprehension is compiled directly from _r1_prepare in the
    REAL BASE, not a hand-invented rq fixture. Optional --torch-cpu tests numeric
    leaves with real tensors. No base module import or __pycache__ is needed.
    """
    rq_ns = _rq_namespace(rq_text)
    rq_base_ast = ast.parse(_rq_base.read_text(encoding="utf-8"))
    rq_accessor = next(n for n in rq_base_ast.body if isinstance(n, ast.FunctionDef) and n.name == "_dg1_weight").body[-1]
    rq_new_accessor = next(n for n in ast.parse(rq_text).body if isinstance(n, ast.FunctionDef) and n.name == "_rq_weight").body[-1]
    assert ast.dump(rq_accessor) == ast.dump(rq_new_accessor), "rq weight accessor differs from base"
    rq_base_prepare = next(n for n in rq_base_ast.body if isinstance(n, ast.FunctionDef) and n.name == "_r1_prepare")
    rq_state_dict = next(n.value for n in rq_base_prepare.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "state" for t in n.targets))
    rq_dense_expr = next(v for k, v in zip(rq_state_dict.keys, rq_state_dict.values) if isinstance(k, ast.Constant) and k.value == "dense")
    rq_shapes = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
    rq_weights = {k: _rq_FakeTensor(v) for k, v in rq_shapes.items()}
    rq_dense = eval(compile(ast.Expression(rq_dense_expr), "real-base-dense", "eval"), {"weights": rq_weights})
    rq_events = []
    rq_fake_mod = types.SimpleNamespace(rq_run=lambda *a: 1 if a[-1] == 4 else 0)
    rq_ns["torch"] = types.SimpleNamespace(
        bfloat16="bf16", int32="i32", int64="i64",
        empty=lambda shape, dtype, device: _rq_FakeTensor(shape, dtype, device),
        zeros=lambda shape, dtype, device: _rq_FakeTensor(shape, dtype, device),
        Generator=lambda **kw: types.SimpleNamespace(manual_seed=lambda s: s),
        cuda=types.SimpleNamespace(get_device_properties=lambda d: types.SimpleNamespace(multi_processor_count=132)))
    rq_ns["_rq_build"] = lambda: (rq_events.append("build") or rq_fake_mod)
    rq_ns["_rq_pack_weight"] = lambda w: (rq_events.append(("pack", w.shape)) or
        (_rq_FakeTensor((w.shape[0] // 128, w.shape[1] // 64, 8, 1536), "u8"), 42, []))
    rq_ns["_rqesc"] = lambda esc, nb, kc, ks, dev: (_rq_FakeTensor((ks * nb * 8 * 32 + 32,), "i32"), None)
    rq_ns["_rqesc2"] = lambda esc, nb, kc, ks, dev: (_rq_FakeTensor((ks * nb * 8 * 128,), "i32"), 1)
    rq_ns["_rq_validate"] = lambda p, w, t, g: (True, "dry-run numeric leaf mocked")
    rq_ns["_v8_logits"] = lambda mode, x, ri, st: ("fp32-hi+lo", mode, ri)
    rq_ns["_v8_tail"] = lambda y, lg, st, data: (y, lg)
    rq_keys = frozenset((p, t) for p in rq_ns["_rq_ks"] for t in (128, 256))
    rq_runs = []
    with contextlib.redirect_stdout(io.StringIO()) as rq_log:
        rq_state = {"dense": rq_dense}
        rq_ns["_rq_prepare"](rq_state)
        assert set(rq_state) == {"dense"} and not rq_events
        for rq_arm in [frozenset([k]) for k in sorted(rq_keys)] + [rq_keys]:
            rq_ns["RQ_SEL"] = rq_arm
            rq_state = {"dense": rq_dense}
            rq_events.clear()
            rq_ns["_rq_prepare"](rq_state)
            assert set(rq_state["rq_bank"]) == set(rq_arm)
            assert sum(e == "build" for e in rq_events) == 1
            assert sum(isinstance(e, tuple) for e in rq_events) == len({p for p, _ in rq_arm})
            for rq_proj in rq_ns["_rq_ks"]:
                for rq_t in (1, 8, 16, 32, 64, 128, 256, 257):
                    rq_x = _rq_FakeTensor((rq_t, rq_shapes["qkv_gate" if rq_proj == "qkv" else rq_proj][1]))
                    rq_got = rq_ns["_rq_gemm"](rq_state, rq_proj, rq_x)
                    assert (rq_got is not None) == ((rq_proj, rq_t) in rq_arm)
            rq_runs.append(sorted(rq_arm))
        rq_ns["RQ_SEL"] = frozenset()
        assert rq_ns["_rq_gemm"](rq_state, "qkv", _rq_FakeTensor((128, 8192))) is None
        rq_ns["RQ_SEL"] = rq_keys
        assert rq_ns["_rq_pooler_tail"](rq_state, _rq_FakeTensor((128, 8192)), {}, "cat")[1] == ("fp32-hi+lo", "stock", None)
        assert rq_ns["_rq_pooler_tail"](rq_state, _rq_FakeTensor((128, 8192)), {}, None) is None
        # Every fallback must leave the bank empty, including malformed REAL base layout.
        rq_faults = [
            ("_rq_build", lambda: (_ for _ in ()).throw(RuntimeError("compile fail"))),
            ("_rq_pack_weight", lambda w: (_ for _ in ()).throw(ValueError("pack fail"))),
            ("_rq_validate", lambda *a: (False, "validation fail")),
            ("_rq_validate", lambda *a: (_ for _ in ()).throw(RuntimeError("validation exception"))),
            ("_rq_call", lambda *a: -1),
            ("_rq_budget_s", 0),
        ]
        for rq_name, rq_fault in rq_faults:
            rq_saved = rq_ns[rq_name]; rq_ns[rq_name] = rq_fault
            rq_bad = {"dense": rq_dense}; rq_ns["_rq_prepare"](rq_bad)
            assert not rq_bad["rq_bank"], rq_name
            rq_ns[rq_name] = rq_saved
        rq_saved = rq_ns["_rqesc2"]; rq_ns["_rqesc2"] = lambda *a: (None, 5)
        rq_bad = {"dense": rq_dense}; rq_ns["_rq_prepare"](rq_bad)
        assert set(rq_bad["rq_bank"]) == {k for k in rq_keys if k[1] == 256}
        rq_ns["_rqesc2"] = rq_saved
        rq_saved = rq_ns["_rqesc"]
        rq_ns["_rqesc"] = lambda *a: (_rq_FakeTensor((1 << 30,), "i32"), None)
        rq_bad = {"dense": rq_dense}; rq_ns["_rq_prepare"](rq_bad)
        assert set(rq_bad["rq_bank"]) == {k for k in rq_keys if k[1] == 128}, "expanded stream accepted"
        rq_ns["_rqesc"] = rq_saved
        rq_saved = rq_ns["torch"].cuda.get_device_properties
        rq_ns["torch"].cuda.get_device_properties = lambda d: types.SimpleNamespace(multi_processor_count=1)
        rq_bad = {"dense": rq_dense}; rq_ns["_rq_prepare"](rq_bad)
        assert not rq_bad["rq_bank"], "nonresident grid accepted"
        rq_ns["torch"].cuda.get_device_properties = rq_saved
        rq_saved = rq_ns["_rq_call"]
        for rq_call in (lambda *a: -4, lambda *a: (_ for _ in ()).throw(RuntimeError("launch fail"))):
            rq_ns["_rq_call"] = rq_call
            assert rq_ns["_rq_gemm"](rq_state, "qkv", _rq_FakeTensor((128, 8192))) is None
        rq_ns["_rq_call"] = rq_saved
        for rq_x in (_rq_FakeTensor((128, 8192), "f32"), _rq_FakeTensor((128, 4096)),
                     _rq_FakeTensor((128, 8192), device="wrong-device")):
            assert rq_ns["_rq_gemm"](rq_state, "qkv", rq_x) is None
        rq_bad = {"dense": {}}; rq_ns["_rq_prepare"](rq_bad)
        assert not rq_bad["rq_bank"]
        rq_ns["RQ_SEL"] = frozenset({("router", 128), ("qkv", 127)})
        rq_bad = {"dense": rq_dense}; rq_ns["_rq_prepare"](rq_bad)
        assert "rq_bank" not in rq_bad
    return {"pass": True, "real_base_dense_line": rq_dense_expr.lineno, "arms": len(rq_runs),
            "shape_dispatch_checks": len(rq_runs) * 4 * 8, "fallbacks": len(rq_faults) + 10,
            "numeric_leaves": "vendor pack/CUDA mocked; new validation tested separately with numpy",
            "new_prepare_leaves": _rq_leaf_dry_run(rq_text)}


def _rq_leaf_dry_run(rq_text):
    """Execute new validation, pack wrapper and compile-cache paths without torch.

    NumPy provides the arithmetic double for the NEW validation code. The copied
    vendor pack is not reimplemented: its actual torch CPU gate lives below.
    """
    import numpy as rq_np
    rq_ns = _rq_namespace(rq_text)

    class _rq_Array:
        def __init__(self, data, dtype="f32"):
            self.a, self.dtype = rq_np.asarray(data), dtype
            self.device = "cpu"
        @property
        def shape(self): return self.a.shape
        def is_contiguous(self): return self.a.flags.c_contiguous
        def float(self): return _rq_Array(self.a.astype(rq_np.float32))
        def int(self): return _rq_Array(self.a.astype(rq_np.int32), "i32")
        def t(self): return _rq_Array(self.a.T, self.dtype)
        def view(self, dtype):
            assert self.dtype == "bf16" and dtype == "i16"
            return _rq_Array((self.a.astype(rq_np.float32).view(rq_np.uint32) >> 16).astype(rq_np.int16), "i16")
        def to(self, dtype):
            if dtype == "bf16":
                rq_u = self.a.astype(rq_np.float32).view(rq_np.uint32)
                rq_u = ((rq_u + 0x7fff + ((rq_u >> 16) & 1)) & 0xffff0000).astype(rq_np.uint32)
                return _rq_Array(rq_u.view(rq_np.float32), dtype)
            return _rq_Array(self.a, dtype)
        def norm(self, dim=None): return _rq_Array(rq_np.linalg.norm(self.a) if dim is None else rq_np.linalg.norm(self.a, axis=dim))
        def clamp(self, min): return _rq_Array(rq_np.maximum(self.a, min))
        def max(self): return _rq_Array(self.a.max())
        def all(self): return bool(self.a.all())
        def abs(self): return _rq_Array(rq_np.abs(self.a))
        def sum(self): return _rq_Array(self.a.sum())
        def fill_(self, value): self.a.fill(value); return self
        def copy_(self, other): self.a[...] = other.to(self.dtype).a; return self
        def __float__(self): return float(self.a)
        def __int__(self): return int(self.a)
        def __bool__(self): return bool(self.a)
        def __getitem__(self, key): return _rq_Array(self.a[key], self.dtype)
        def __setitem__(self, key, val): self.a[key] = val.a if isinstance(val, _rq_Array) else val
        def __imul__(self, val): self.a *= val; return self
        def __sub__(self, other): return _rq_Array(self.a - other.a)
        def __truediv__(self, other): return _rq_Array(self.a / other.a)
        def __and__(self, other): return _rq_Array(self.a & other, self.dtype)

    rq_gen = rq_np.random.default_rng(12828)
    rq_torch = types.SimpleNamespace(
        bfloat16="bf16", int16="i16", float32="f32",
        empty=lambda shape, dtype, device: _rq_Array(rq_np.empty(shape, dtype=rq_np.float32), dtype),
        randn=lambda shape, generator, device: _rq_Array(rq_gen.standard_normal(shape).astype(rq_np.float32)),
        mm=lambda a, b: _rq_Array(rq_np.matmul(a.a, b.a)),
        isfinite=lambda a: _rq_Array(rq_np.isfinite(a.a)),
        equal=lambda a, b: bool(rq_np.array_equal(a.a, b.a)))
    rq_ns["torch"] = rq_torch
    rq_w = _rq_Array(rq_gen.standard_normal((128, 64)).astype(rq_np.float32) * 0.01).to("bf16")
    rq_p = {"rq_ks": 2, "rq_cnt": _rq_Array(rq_np.zeros(8, dtype=rq_np.int32), "i32")}
    rq_count = 0
    for rq_t in (128, 256):
        rq_ns["_rq_call"] = lambda p, x, out: (out.copy_(rq_torch.mm(x.float(), rq_w.float().t())), 0)[1]
        assert rq_ns["_rq_validate"](rq_p, rq_w, rq_t, None)[0]
        rq_count += 1
        for rq_call in (lambda *a: -4, lambda *a: 0,
                        lambda p, x, out: (out.fill_(1234), 0)[1]):
            rq_ns["_rq_call"] = rq_call
            assert not rq_ns["_rq_validate"](rq_p, rq_w, rq_t, None)[0]
            rq_count += 1
        rq_p["rq_cnt"].fill_(1)
        rq_ns["_rq_call"] = lambda p, x, out: (out.copy_(rq_torch.mm(x.float(), rq_w.float().t())), 0)[1]
        assert not rq_ns["_rq_validate"](rq_p, rq_w, rq_t, None)[0]
        rq_p["rq_cnt"].fill_(0)
        rq_count += 1
    rq_pk = (object(), 42, [])
    rq_ns["_rqpack"] = lambda w: rq_pk
    rq_ns["_rqunpack"] = lambda *a: rq_w.view("i16").int() & 65535
    assert rq_ns["_rq_pack_weight"](rq_w) is rq_pk
    for rq_pack in (lambda w: None, lambda w: rq_pk):
        rq_ns["_rqpack"] = rq_pack
        rq_ns["_rqunpack"] = lambda *a: _rq_Array(rq_np.zeros(rq_w.shape, dtype=rq_np.int32), "i32")
        try: rq_ns["_rq_pack_weight"](rq_w)
        except ValueError: pass
        else: raise AssertionError("failed pack was accepted")
    try: rq_ns["_rq_pack_weight"](_rq_Array(rq_np.zeros((1, 1)), "f32"))
    except ValueError: pass
    else: raise AssertionError("bad weight layout was accepted")
    # Execute _rq_build itself with only the external load_inline import substituted.
    rq_build_ast = next(n for n in ast.parse(rq_text).body if isinstance(n, ast.FunctionDef) and n.name == "_rq_build")
    class _rq_Imports(ast.NodeTransformer):
        def visit_Import(self, node): return ast.copy_location(ast.Pass(), node)
        def visit_ImportFrom(self, node): return ast.copy_location(ast.Pass(), node)
    rq_build_ast = _rq_Imports().visit(rq_build_ast)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[rq_build_ast], type_ignores=[])), "rq-build-double", "exec"), rq_ns)
    rq_calls = []
    rq_ns["_rq_hashlib"] = hashlib
    rq_ns["_rq_load_inline"] = lambda **kw: (rq_calls.append(kw) or "compiled")
    assert rq_ns["_rq_build"]() == rq_ns["_rq_build"]() == "compiled"
    assert len(rq_calls) == 1 and rq_calls[0]["functions"] == ["rq_run"]
    assert rq_calls[0]["extra_cuda_cflags"] == rq_ns["_rq_flags"]
    rq_ns["_rq_build_cache"] = {"mod": None, "error": None}
    rq_ns["_rq_load_inline"] = lambda **kw: (_ for _ in ()).throw(RuntimeError("compile fail"))
    for _ in range(2):
        try: rq_ns["_rq_build"]()
        except RuntimeError: pass
        else: raise AssertionError("compile failure was accepted")
    return {"validation_cases": rq_count, "pack_wrapper_cases": 4, "build_cache_cases": 4,
            "pass": True, "actual_vendor_pack": "deferred to --torch-cpu"}


def _rq_torch_cpu(rq_text):
    import torch
    rq_ns = _rq_namespace(rq_text)
    rq_ns["torch"] = torch
    rq_gen = torch.Generator().manual_seed(12828)
    rq_rows = []
    # Small real BF16 weights, including the zero-escape branch and cursor overflow.
    for rq_kind in ("random", "constant", "wide"):
        rq_w = (torch.randn((128, 128), generator=rq_gen) * 0.01).to(torch.bfloat16)
        if rq_kind == "constant": rq_w.fill_(1)
        if rq_kind == "wide": rq_w[:, ::2] *= 1024
        rq_lay, rq_tab, rq_esc = rq_ns["_rq_pack_weight"](rq_w)
        for rq_ks in (2, 4, 7):
            rq_e1, _ = rq_ns["_rqesc"](rq_esc, 1, 2, rq_ks, "cpu")
            rq_e2, rq_mx = rq_ns["_rqesc2"](rq_esc, 1, 2, rq_ks, "cpu")
            assert rq_e1.dtype == torch.int32
            assert (rq_e2 is None) == (rq_mx > 4)
        rq_rows.append({"kind": rq_kind, "escapes": len(rq_esc)})
    rq_w = (torch.randn((128, 128), generator=rq_gen) * 0.01).to(torch.bfloat16)
    for rq_t in (128, 256):
        rq_p = {"rq_ks": 2, "rq_cnt": torch.zeros(8, dtype=torch.int32)}
        rq_ns["_rq_call"] = lambda p, x, out: (out.copy_(torch.mm(x.float(), rq_w.float().t())), 0)[1]
        assert rq_ns["_rq_validate"](rq_p, rq_w, rq_t, rq_gen)[0]
        rq_ns["_rq_call"] = lambda p, x, out: 0  # poisoned output must be rejected
        assert not rq_ns["_rq_validate"](rq_p, rq_w, rq_t, rq_gen)[0]
    return {"pass": True, "pack_cases": rq_rows, "validation_T": [128, 256], "gpu": "not used"}


def _rq_main():
    rq_parser = argparse.ArgumentParser(description=__doc__)
    rq_parser.add_argument("--check", action="store_true")
    rq_parser.add_argument("--torch-cpu", action="store_true")
    rq_args = rq_parser.parse_args()
    rq_text, rq_block = _rq_generate()
    rq_result = {"ast": "PASS", "default_reverse_gate": "PASS", "dry_run": _rq_dry_run(rq_text),
                 "bytes": len(rq_text.encode()), "graft_bytes": len(rq_text.encode()) - _rq_base.stat().st_size,
                 "sha256": hashlib.sha256(rq_text.encode()).hexdigest(),
                 "anchors": [{"name": a["name"], "line": a["line"]} for a in _rq_anchors]}
    rq_ns = _rq_namespace(rq_text)
    rq_cuda = rq_ns["_rqsrc"](rq_ns["_rq_variants"])
    assert "@@" not in rq_cuda and "trap;" not in rq_cuda
    assert rq_cuda.count("case ") == 2 and "rq_run(" in rq_cuda
    rq_result["cuda_bytes"] = len(rq_cuda.encode())
    rq_result["cuda_flags"] = rq_ns["_rq_flags"]
    if rq_args.torch_cpu:
        rq_result["torch_cpu"] = _rq_torch_cpu(rq_text)
    if rq_args.check:
        assert _rq_out.read_bytes() == rq_text.encode(), "generated output is stale"
    else:
        _rq_out.write_bytes(rq_text.encode())
    print(json.dumps(rq_result, indent=2))


if __name__ == "__main__":
    _rq_main()

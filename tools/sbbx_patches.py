"""SBBX: exact virtual-rank boundaries; only writes the named lab vehicle.

python -B tools/sbbx_patches.py             # generate + CPU gates + dry run
python -B tools/sbbx_patches.py --check     # read-only regeneration check
No imports of the full layer, CUDA calls, bytecode files, or GPU jobs.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_diet_rq128.py"
OUT = ROOT / "kernels/lab/k3_sbbx_ab.py"


def once(s, old, new):
    assert s.count(old) == 1, ("anchor count", s.count(old), old[:100])
    return s.replace(old, new, 1)


def literal(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, name
    return ast.literal_eval(nodes[0].value)


def between(s, a, b):
    assert s.count(a) == s.count(b) == 1, (a, b)
    return s[s.index(a):s.index(b)]


def cuda_source(base):
    """Copy arithmetic/load/store bodies verbatim; change only rank ownership."""
    cu = literal(base, "SBB1_CU")
    loads = between(cu, "float pw[8], qw[8], al[8], bt[8], st[8];\n",
                    '  asm volatile("griddepcontrol.wait;" ::: "memory");')
    loads = once(loads, "float pw[8], qw[8], al[8], bt[8], st[8];\n", "")
    # References remain identical inside each compile-time-unrolled virtual group.
    moments = between(cu, "float bsum = 0.f", "if (lane == 0)")
    finish = cu[cu.index("const float post_rstd ="):cu.rindex("\n}")]
    prelude = r'''
// SBBX keeps the base 8-element FMA chains, XOR tree and 0..15 tuple sum.
template <int NC, bool kPreStream, bool kRouter, bool kInputScale>
__global__ void __launch_bounds__(256) sbbx_kernel(const SbbParams p) {
static_assert(NC == 1 || NC == 4, "SBBX topology");
constexpr int NG = 4 / NC;
__shared__ __align__(16) float partials[16][4];
__shared__ float warp_red[NG * 8][4];
__shared__ __align__(8) uint64_t bar;
const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
if (p.trig_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
uint32_t rank = 0;
if constexpr (NC == 4) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
const int row = blockIdx.x / NC;
const uint32_t bar_a = (uint32_t)__cvta_generic_to_shared(&bar);
if constexpr (NC == 4) {
if (tid == 0) {
asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_a));
asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar_a), "r"(3 * 4 * 16));
asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
float all_pw[NG][8], all_qw[NG][8], all_al[NG][8], all_bt[NG][8], all_st[NG][8];
#pragma unroll
for (int g = 0; g < NG; ++g) {
const int e0 = (rank * 256 + g * 256 + tid) * 8;
float (&pw)[8] = all_pw[g], (&qw)[8] = all_qw[g], (&al)[8] = all_al[g];
float (&bt)[8] = all_bt[g], (&st)[8] = all_st[g];
'''
    middle = r'''
}
asm volatile("griddepcontrol.wait;" ::: "memory");
float all_br[NG][8];
#pragma unroll
for (int g = 0; g < NG; ++g) {
const int e0 = (rank * 256 + g * 256 + tid) * 8;
sb_ld8bf(p.branch + (size_t)row * SB_H + e0, all_br[g]);
if (!kPreStream) sb_ld8f(p.stream + (size_t)row * SB_H + e0, all_st[g]);
}
asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
float all_val[NG][8], all_res[NG][8];
#pragma unroll
for (int g = 0; g < NG; ++g) {
float (&pw)[8] = all_pw[g], (&al)[8] = all_al[g], (&bt)[8] = all_bt[g];
float (&st)[8] = all_st[g], (&br)[8] = all_br[g];
float (&val)[8] = all_val[g], (&res)[8] = all_res[g];
'''
    reduction = r'''
if (lane == 0) {
warp_red[g * 8 + warp][0] = bsum; warp_red[g * 8 + warp][1] = rsum;
warp_red[g * 8 + warp][2] = csum; warp_red[g * 8 + warp][3] = ssum;
}
}
__syncthreads();
if constexpr (NC == 4) asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
// 16 lanes: NC=1 one virtual rank per lane; NC=4 (local rank, destination).
if (tid < 16) {
const int local_rank = NC == 1 ? tid : tid / 4;
const int qrank = rank * 4 + local_rank;
float4 mine = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
for (int w = 0; w < 2; ++w) {
mine.x += warp_red[2 * local_rank + w][0]; mine.y += warp_red[2 * local_rank + w][1];
mine.z += warp_red[2 * local_rank + w][2]; mine.w += warp_red[2 * local_rank + w][3];
}
if constexpr (NC == 1) {
*reinterpret_cast<float4*>(&partials[qrank][0]) = mine;
} else {
const int dest = tid & 3;
if (dest == (int)rank) {
*reinterpret_cast<float4*>(&partials[qrank][0]) = mine;
} else {
const uint32_t dst_l = (uint32_t)__cvta_generic_to_shared(&partials[qrank][0]);
uint32_t dst, rbar;
asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dst) : "r"(dst_l), "r"(dest));
asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(bar_a), "r"(dest));
asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
:: "r"(dst), "f"(mine.x), "f"(mine.y), "f"(mine.z), "f"(mine.w), "r"(rbar) : "memory");
}
}
}
if constexpr (NC == 4) {
uint32_t done = 0;
while (!done)
asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
" selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(bar_a) : "memory");
}
__syncthreads();
float tb = 0.f, tr = 0.f, tc = 0.f, ts = 0.f;
#pragma unroll
for (int q = 0; q < SB_CS; ++q) { tb += partials[q][0]; tr += partials[q][1]; tc += partials[q][2]; ts += partials[q][3]; }
#pragma unroll
for (int g = 0; g < NG; ++g) {
const int e0 = (rank * 256 + g * 256 + tid) * 8;
float (&pw)[8] = all_pw[g], (&qw)[8] = all_qw[g], (&bt)[8] = all_bt[g];
float (&val)[8] = all_val[g], (&res)[8] = all_res[g];
'''
    return prelude + loads + middle + moments + reduction + finish + "\n}\n}\n"


def host_source(base):
    host = literal(base, "SBB1_HOST")
    run = between(host, "void sbb1_run(", "void vf_router32(")
    wrappers = []
    for nc in (1, 4):
        r = once(run, "void sbb1_run(", f"void sbbx{nc}_run(")
        r = once(r, "const int T = branch.size(0);", 'const int T = branch.size(0);\nTORCH_CHECK(T >= 1 && T <= 8, "SBBX: T outside 1..8");')
        assert r.count("sbb_go<") == 6
        r = r.replace("sbb_go<", f"sbbx_go<{nc}, ")
        r = r.rstrip()
        assert r.endswith("}")
        wrappers.append(r[:-1] + 'TORCH_CHECK(cudaGetLastError() == cudaSuccess, "SBBX: launch failed");\n}\n')
    go = r'''
template <int NC, bool A, bool B, bool C>
static void sbbx_go(const SbbParams& p, int T, cudaStream_t s) {
cudaLaunchConfig_t cfg = {};
cfg.gridDim = dim3(T * NC); cfg.blockDim = dim3(256); cfg.stream = s;
cudaLaunchAttribute at[2] = {};
at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
at[0].val.programmaticStreamSerializationAllowed = 1;
if constexpr (NC == 4) {
at[1].id = cudaLaunchAttributeClusterDimension;
at[1].val.clusterDim.x = 4; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;
}
cfg.attrs = at; cfg.numAttrs = NC == 4 ? 2 : 1;
cudaLaunchKernelEx(&cfg, sbbx_kernel<NC, A, B, C>, p);
}
'''
    return go + "\n".join(wrappers)


PYTHON = r'''
SBBX = 0  # call/capture-time selector: 0 base, 1 single CTA, 4 four-CTA cluster.
_SBBX = {"bank": {}, "ok": {1: False, 4: False}, "tests": [], "errors": {}}
SBBX_ARMS = [{"name": "lead", "set": {"SBBX": 0}},
             {"name": "sbb1cta", "set": {"SBBX": 1}},
             {"name": "mid1", "set": {"SBBX": 0}},
             {"name": "sbb4cta", "set": {"SBBX": 4}},
             {"name": "mid2", "set": {"SBBX": 0}}]

def _sbbx_run(st, tokens):
    if 1 <= tokens <= 8 and SBBX in (1, 4) and _SBBX["ok"].get(SBBX, False):
        return _SBBX["bank"][SBBX]
    return st["mod"].sbb1_run

def _sbbx_args(st, stream, branch, tokens, ffn):
    dev = branch.device
    out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
    ns = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev)
    rt = st["e"] if ffn else torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev)
    pw, qw, al, bt, isc, pe, qe = st["w"]
    flags = isc | (2 if ffn else (8 if LAT_SBB_TRIG.get(_k2a_bucket_of(tokens)) else 0))
    return (branch.contiguous(), stream.contiguous(), out, ns, rt, pw, qw, al, bt, pe, qe, flags)

def _sbbx_compare(ref, got, arm, which, T, case):
    errs, bits = [], []
    for a, b in zip(ref[2:5], got[2:5]):
        if not a.numel():
            errs.append(0.0)
            bits.append(True)
            continue
        errs.append(float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-30)))
        dt = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
        bits.append(bool(torch.equal(a.view(dt), b.view(dt))) and bool(torch.isfinite(b).all()))
    good = all(bits) and errs[0] < 1e-2 and errs[1] < 1e-4 and errs[2] < 1e-2
    _SBBX["tests"].append({"arm": arm, "boundary": which, "T": T, "case": case,
                           "rel": errs, "bitwise": bits, "ok": good})
    print(f"[sbbx] arm={arm} {which} T={T} {case}: out={errs[0]:.2e} stream={errs[1]:.2e} "
          f"router={errs[2]:.2e} bitwise={bits} -> {'ok' if good else 'FAIL'}", flush=True)
    return good

def _sbbx_prepare(state):
    # Same accessors as _sbb1/_vf_sbb2. No guessed MoE or layer-state handles.
    _SBBX.update(bank={}, ok={1: False, 4: False}, tests=[], errors={})
    states = (("attn", _SBB1["st"]), ("ffn", _VF["st2"]))
    mod = _SBB1["mod"]
    if mod is None or any(st is None for _, st in states):
        return
    for arm in (1, 4):
        try:
            run = mod.sbbx1_run if arm == 1 else mod.sbbx4_run
            _SBBX["bank"][arm] = run
            good = True
            dev = state["router"].device
            gen = torch.Generator(device=dev).manual_seed(82540)
            for which, st in states:
                ffn = which == "ffn"
                for T in (1, 2, 4, 5, 8):
                    good &= bool(st["ok"].get(_k2a_bucket_of(T)))
                    for case in ("normal", "outliers", "zeros"):
                        stream = torch.randn((T, HIDDEN), generator=gen, device=dev)
                        branch = torch.randn((T, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
                        if case == "outliers":
                            stream[:, ::31] *= 1024
                            stream[:, 1::31] *= 1 / 1024
                            branch[:, ::63] *= 256
                            branch[:, 1::63] *= 1 / 256
                        elif case == "zeros":
                            stream.zero_()
                            branch.zero_()
                        if ffn and T <= SLICED_UNPOOL_MAX_TOKENS:
                            branch[:, UNPOOL_SLICE:] = 0
                        ref = _sbbx_args(st, stream, branch, T, ffn)
                        got = _sbbx_args(st, stream, branch, T, ffn)
                        mod.sbb1_run(*ref)
                        run(*got)
                        torch.cuda.synchronize()
                        good &= _sbbx_compare(ref, got, arm, which, T, case)
                    # Capture the prepared callable; replay with changed live inputs.
                    # All output buffers exist before capture, no counters escape a CTA.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run(*got)
                    for replay in range(2):
                        stream.add_(0.25)
                        branch.add_(0.5)
                        if ffn and T <= SLICED_UNPOOL_MAX_TOKENS:
                            branch[:, UNPOOL_SLICE:] = 0
                        mod.sbb1_run(*ref)
                        graph.replay()
                        torch.cuda.synchronize()
                        good &= _sbbx_compare(ref, got, arm, which, T, f"graph{replay}")
            _SBBX["ok"][arm] = bool(good)
        except Exception as ex:
            _SBBX["ok"][arm] = False
            _SBBX["errors"][arm] = repr(ex)
            print(f"[sbbx] arm={arm} prepare failed -> disabled: {ex!r}", flush=True)
    print(f"[sbbx] ready {_SBBX['ok']} selector={SBBX} (default 0)", flush=True)
'''


def generate(base):
    cu, host = cuda_source(base), host_source(base)
    decl = literal(base, "SBB1_DECL").split(";", 1)[0] + ";"
    decl = " ".join(decl.replace("sbb1_run", f"sbbx{nc}_run") for nc in (1, 4))
    insert = "\nSBBX_CU = r'''" + cu + "'''\nSBBX_HOST = r'''" + host + "'''\nSBBX_DECL = " + repr(decl) + "\n" + PYTHON
    s = once(base, 'def _sbb1_build():\n', insert + '\ndef _sbb1_build():\n')
    tag = hashlib.sha256((cu + host + decl).encode()).hexdigest()[:12]
    s = once(s, '_SBB1["mod"]=load_inline("sbb1_k3t",cpp_sources=SBB1_DECL,cuda_sources=SBB1_CU+SBB1_HOST,functions=["sbb1_run","vf_router32"],',
             f'_SBB1["mod"]=load_inline("sbbx_{tag}",cpp_sources=SBB1_DECL+SBBX_DECL,cuda_sources=SBB1_CU+SBBX_CU+SBB1_HOST+SBBX_HOST,functions=["sbb1_run","vf_router32","sbbx1_run","sbbx4_run"],')
    s = once(s, ' st["mod"].sbb1_run(branch.contiguous(),stream.contiguous(),out,ns,st["e"],pw,qw,al,bt,pe,qe,isc|2)',
             ' _sbbx_run(st,tokens)(branch.contiguous(),stream.contiguous(),out,ns,st["e"],pw,qw,al,bt,pe,qe,isc|2)')
    s = once(s, ' st["mod"].sbb1_run(branch.contiguous(),stream.contiguous(),out,ns,rt,pw,qw,al,bt,pe,qe,',
             ' _sbbx_run(st,tokens)(branch.contiguous(),stream.contiguous(),out,ns,rt,pw,qw,al,bt,pe,qe,')
    # Disable during base tests; run new tests AFTER every guard-timed base prepare.
    s = once(s, ' _sbb1_prepare(state,t0)\n _vf_prepare(state,t0)\n',
             ' _SBBX.update(bank={},ok={1:False,4:False},tests=[],errors={})\n _sbb1_prepare(state,t0)\n _vf_prepare(state,t0)\n')
    s = once(s, ' _rq_prepare(state)\n', ' _rq_prepare(state)\n _sbbx_prepare(state)\n')
    nonces = [line for line in base.splitlines() if line.startswith("# nonce ")]
    assert len(nonces) == 1
    nonce = nonces[0]
    return once(s, nonce + "\n", "# nonce sbbx-ab: k3_diet_rq128 + exact boundaries SBBX=0/1/4\n")


def gates(base, new):
    compile(new, str(OUT), "exec")
    assert new.isascii() and "\r" not in new and len(new.encode()) < 1_000_000
    # Only these four existing Python functions and the nonce may change.
    a, b = ast.parse(base), ast.parse(new)
    allowed = {"prepare", "_sbb1_build", "_sbb1_call", "_vf_sbb2"}
    def key(n):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            return n.name
        if isinstance(n, ast.Assign):
            return ast.dump(ast.Tuple(n.targets, ast.Load()), include_attributes=False)
        return ast.dump(n, include_attributes=False)
    old = {key(n): n for n in a.body}
    now = {key(n): n for n in b.body}
    for k, n in old.items():
        assert k in now and (k in allowed or ast.dump(n) == ast.dump(now[k])), k
    for name in ("SBB1_CU", "SBB1_HOST", "SBB1_DECL"):
        assert literal(base, name) == literal(new, name), name
    cu = literal(new, "SBBX_CU")
    assert "__trap" not in cu and "return;" not in cu
    assert cu.count('griddepcontrol.wait;') == 1 and cu.count('griddepcontrol.launch_dependents;') == 2
    assert between(literal(base, "SBB1_CU"), "float bsum = 0.f", "if (lane == 0)") in cu
    return {"bytes": len(new.encode()), "unchanged_ast_nodes": len(old) - len(allowed),
            "base_cuda_identical": True, "default_arm": literal(new, "SBBX")}


def cpu_dry_run(base, new):
    """Run REAL base state expressions/prepares + new Python using CPU doubles.

    CUDA and tensor arithmetic are deliberately mocked, never called correctness
    evidence. Strict dictionaries/attributes expose wrong keys and module handles.
    A separate ownership check proves complete per-row output/tuple coverage.
    """
    import time
    class Tensor:
        def __init__(self, shape=(), dtype="f32", value=1.0):
            self.shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
            self.dtype, self.device, self.value = dtype, "cpu-dry-run", value
        def to(self, dtype): return Tensor(self.shape, dtype, self.value)
        def float(self): return self.to("f32")
        def contiguous(self): return self
        def view(self, dtype): return self.to(dtype)
        def norm(self): return Tensor((), value=abs(self.value))
        def clamp_min(self, x): return Tensor((), value=max(self.value, x))
        def numel(self):
            import math
            return math.prod(self.shape)
        def all(self): return True
        def zero_(self): self.value = 0; return self
        def add_(self, x): self.value += x; return self
        def __float__(self): return float(self.value)
        def __sub__(self, other): return Tensor(self.shape, value=self.value - other.value)
        def __truediv__(self, other): return Tensor((), value=self.value / other.value)
        def __imul__(self, x): return self
        def __getitem__(self, key): return self
        def __setitem__(self, key, value): pass
    class Generator:
        def __init__(self, **kw): pass
        def manual_seed(self, seed): return self
    class Graph:
        def replay(self): pass
    fake_torch = types.SimpleNamespace(bfloat16="bf16", float32="f32", int16="i16", int32="i32",
        Generator=Generator, empty=lambda shape, dtype="f32", **kw: Tensor(shape, dtype),
        randn=lambda shape, **kw: Tensor(shape), equal=lambda a, b: a.value == b.value,
        isfinite=lambda x: Tensor(), cuda=types.SimpleNamespace(synchronize=lambda: None,
            CUDAGraph=Graph, graph=lambda g: contextlib.nullcontext()))
    calls = []
    bad_arm = [None]
    def runner(arm):
        def run(*args):
            assert len(args) == 12
            T = args[0].shape[0]
            assert arm == 0 or 1 <= T <= 8
            calls.append((arm, T, args[-1]))
            if arm == bad_arm[0]:
                args[2].value = 2
        run.arm = arm
        return run
    mod = types.SimpleNamespace(sbb1_run=runner(0), sbbx1_run=runner(1), sbbx4_run=runner(4))
    class Norm:
        def __init__(self, weight, eps, input_scale=False):
            self.has_weight, self.eps, self.input_scale = weight is not None, eps, input_scale
    class Op:
        def __init__(self, post, gate, pre):
            self._post_norm, self._pre_norm = post, pre
            self._post_norm_is_input_scale = post.input_scale
            self._use_residual_gate = True
            self._cc_post_weight = self._cc_pre_weight = Tensor((8192,), "bf16")
            self._cc_alpha = self._cc_beta = Tensor((8192,))
        def _ensure_cached_weights(self, dtype): assert dtype == "bf16"
        def __call__(self, stream, branch, **kw):
            return (Tensor(branch.shape, "bf16"), Tensor(branch.shape),
                    Tensor(branch.shape) if kw.get("router_dtype") is not None else None)
    bn, nn = ast.parse(base), ast.parse(new)
    ns = {"torch": fake_torch, "time": time, "_pt1_wait": lambda x: None,
          "_SBB1": {"mod": mod, "st": None}, "_VF": {"r8": False, "prt": False, "st2": None, "seen": set()},
          "HIDDEN": 8192, "SBB1_PREPARE_BY_S": 250, "SBB1_ENABLE": True,
          "SBB1_ROUTE": {1: True, 8: True}, "VF_SBB2": {1: True, 8: True}, "VF_PRT": False,
          "VF_ROUTER": {1: "lean", 8: "r32"}, "LAT_SBB_TRIG": {1: True, 8: True},
          "SLICED_UNPOOL_MAX_TOKENS": 8, "UNPOOL_SLICE": 1024, "ROUTER_FP32_MAX_TOKENS": 8,
          "K2A_BUCKETS": ((1, 4, 1), (5, 12, 8)), "_sbb1_build": lambda: mod,
          "_k2a_bucket_of": lambda t: 1 if t <= 4 else 8, "_nb_sbb": lambda t: 1 if t <= 4 else 8}
    # Execute the exact construction expressions from _r1_prepare, not a guessed dict.
    prep = next(n for n in bn.body if isinstance(n, ast.FunctionDef) and n.name == "_r1_prepare")
    norm_names = {"pre_ffn", "next_pre_attn", "post_attn", "post_ffn"}
    ctor_ns = {"weights": {k: Tensor((8192,)) for k in ("pre_ffn_norm", "next_pre_attn_norm", "post_ffn_norm",
                "attn_residual_gate", "ffn_residual_gate")}, "HIDDEN": 8192, "RMS_EPS": 1e-6, "POST_NORM_EPS": 1e-6,
        "MuseSpark1xRMSNorm": "plain", "MuseSpark1xRMSNormWithInputScale": "isc",
        "_norm": lambda cls, w, h, eps, **kw: Norm(w, eps, cls == "isc"),
        "_residual": lambda x: x, "FusedPostNormResidualPreNorm": Op, "router": Tensor((256, 8192))}
    for node in prep.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in norm_names:
                exec(compile(ast.Module([node], []), str(BASE), "exec"), ctor_ns)
            if node.targets[0].id == "state":
                state = {ast.literal_eval(k): eval(compile(ast.Expression(v), str(BASE), "eval"), ctor_ns)
                         for k, v in zip(node.value.keys, node.value.values)
                         if ast.literal_eval(k) in ("router", "attn_boundary", "ffn_boundary")}
    assert set(state) == {"router", "attn_boundary", "ffn_boundary"}
    # The base prepare functions manufacture the actual st/w/e/ok dictionaries.
    for node in nn.body:
        if isinstance(node, ast.FunctionDef) and (node.name.startswith("_sbbx_") or node.name in
                ("_sbb1_prepare", "_vf_prepare", "_sbb1_call", "_sbb1", "_vf_sbb2")):
            exec(compile(ast.Module([node], []), str(OUT), "exec"), ns)
    ns.update(SBBX=0, _SBBX={"bank": {}, "ok": {1: False, 4: False}, "tests": [], "errors": {}})
    with contextlib.redirect_stdout(io.StringIO()):
        ns["_sbb1_prepare"](state, time.time())
        ns["_vf_prepare"](state, time.time())
        assert ns["_SBB1"]["st"] is not None and ns["_VF"]["st2"] is not None
        ns["_sbbx_prepare"](state)
        assert ns["_SBBX"]["ok"] == {1: True, 4: True}, ns["_SBBX"]
        normal_tests = len(ns["_SBBX"]["tests"])
        for arm in (0, 1, 4):
            ns["SBBX"] = arm
            for T in (1, 2, 4, 5, 8, 9, 12, 16, 32, 128, 256):
                expect = arm if T <= 8 else 0
                assert ns["_sbbx_run"](ns["_SBB1"]["st"], T).arm == expect
                if T <= 8:
                    st, br = Tensor((T, 8192)), Tensor((T, 8192), "bf16")
                    assert ns["_sbb1"](st, br, "f32", state, T)[2].dtype == "f32"
                    assert calls[-1] == (expect, T, 8), "attn pre-stream/router/no input scale/early trigger"
                    assert ns["_vf_sbb2"](st, br, T)[2] is None
                    assert calls[-1] == (expect, T, 3), "ffn post-stream/no router/input scale/no early trigger"
        # Failed numerics, exception, missing prerequisites, repeated prepare.
        bad_arm[0] = 1
        ns["_sbbx_prepare"](state)
        assert ns["_SBBX"]["ok"] == {1: False, 4: True}
        ns["SBBX"] = 1
        assert ns["_sbbx_run"](ns["_SBB1"]["st"], 1).arm == 0
        bad_arm[0] = None
        saved = mod.sbbx4_run
        def fail(*args): raise RuntimeError("injected CUDA wrapper failure")
        mod.sbbx4_run = fail
        ns["_sbbx_prepare"](state)
        assert ns["_SBBX"]["ok"] == {1: True, 4: False}
        mod.sbbx4_run = saved
        ns["_VF"]["st2"] = None
        ns["_sbbx_prepare"](state)
        assert ns["_SBBX"]["ok"] == {1: False, 4: False} and not ns["_SBBX"]["bank"]
    # All original 1024 eight-element tiles and all sixteen tuples exactly once.
    ownership = {}
    for nc in (1, 4):
        tiles = [(r * 256 + g * 256 + t) for r in range(nc) for g in range(4 // nc) for t in range(256)]
        assert sorted(tiles) == list(range(1024))
        receives = {d: [] for d in range(nc)}
        for r in range(nc):
            for t in range(16):
                q, d = (t, 0) if nc == 1 else (r * 4 + t // 4, t % 4)
                receives[d].append(q)
        assert all(sorted(q) == list(range(16)) for q in receives.values())
        assert all((tile // 64) * 2 == (tile // 32) // 2 * 2 for tile in tiles)
        ownership[nc] = {"tiles": len(tiles), "tuples_per_receiver": 16,
                         "remote_float4": 0 if nc == 1 else 48, "receive_bytes": 0 if nc == 1 else 192}
    return {"state_source": "real _r1_prepare AST + _sbb1_prepare/_vf_prepare execution",
            "numeric_backend": "mock only; CUDA correctness NOT claimed", "prepare_cases": normal_tests,
            "dispatch_cases": 33, "failure_paths": ["numeric", "exception", "missing state", "repeat prepare"],
            "ownership": ownership}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    base = BASE.read_bytes().decode("ascii").replace("\r\n", "\n")
    new = generate(base)
    result = {"gates": gates(base, new), "dry_run": cpu_dry_run(base, new)}
    if args.check:
        assert OUT.read_bytes() == new.encode("ascii"), "generated file stale"
    else:
        OUT.write_bytes(new.encode("ascii"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

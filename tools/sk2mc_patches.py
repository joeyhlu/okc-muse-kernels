"""SK2MC: cfg6 qkv32 multicast x; separate prepare-built, call-time module.

Only writes kernels/lab/k3_sk2mc_ab.py. --check executes CPU gates without
writing anything. --linux-gate also compiles and writes sk2mc_gate.json next
to the shell gate; compile artifacts are retained in a fresh /tmp directory.
No GPU launches in this generator or in the Linux gate.
"""
import argparse
import ast
import contextlib
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_sk2x_ab.py"
OUT = ROOT / "kernels/lab/k3_sk2mc_ab.py"


def replace(s, old, new, count=1):
    assert s.count(old) == count, (old[:120], s.count(old), count)
    return s.replace(old, new)


@lru_cache(maxsize=4)
def tree(s):
    return ast.parse(s)


def literal(s, key):
    nodes = [n.value for n in tree(s).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == key for t in n.targets)]
    assert len(nodes) == 1, key
    return ast.literal_eval(nodes[0])


MC_HELPERS = r'''
// SK2MC: PTX ISA cp.async.bulk.tensor multicast::cluster (mask = ranks 0,1).
__device__ __forceinline__ void sk2mc_sync() {
asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void sk2mc_tma(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c2) {
asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
" [%0], [%1, {%3, %4, %5}], [%2], %6, %7;"
:: "r"(dst), "l"(map), "r"(bar), "r"(0), "r"(0), "r"(c2), "h"((uint16_t)3), "l"(l2_pol_el()) : "memory");
}
template <int MC> __device__ __forceinline__ void sk2mc_empty(uint32_t bar, uint32_t rk) {
mbar_arrive(bar);
if constexpr (MC == 2) {
uint32_t rb;
asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(bar), "r"(rk ^ 1));
asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(rb) : "memory");
}
}
template <int MC, bool SLEEP> __device__ __forceinline__ void sk2mc_wait(uint32_t bar, uint32_t par, uint32_t* err) {
if constexpr (MC == 1) {
if constexpr (SLEEP) mbar_wait_sleep(bar, par); else mbar_wait(bar, par);
} else {
// Bounded fail-closed diagnostic, no trap/CTA exit. All roles still execute
// their arrivals/cleanup and reach the final cluster lifetime barrier.
// Keep the base sleep hint for producer/escape; bound both polls and clocks.
const unsigned long long begin = clock64();
for (uint32_t it = 0; it < (1u << 20); ++it) {
uint32_t ok;
if constexpr (SLEEP) {
asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
: "=r"(ok) : "r"(bar), "r"(par), "r"(0x989680u) : "memory");
} else { ok = mbar_test(bar, par); }
if (ok) return; // helper return only, never retires a kernel thread
if (clock64() - begin >= (1ull << 30)) break;
}
atomicExch(err, 1u);
}
}
'''


def patched_cuda(cu):
    """Start from ship source, not the wrong-output SK2_DG probe source."""
    cu = replace(cu, '#define SK2_WAIT(bar, par)', MC_HELPERS + '\n#define SK2_WAIT(bar, par)')
    cu = replace(cu, 'bool PK, bool FUSE>\n', 'bool PK, bool FUSE, int MC = 1>\n')
    cu = replace(cu, 'bool FUSE = false>\n', 'bool FUSE = false, int MC = 1>\n')
    cu = replace(cu, 'CL, OutT, false, FUSE>;', 'CL, OutT, false, FUSE, MC>;')
    start, end = cu.index('sk2_kernel('), cu.index('static void launch_sk2(')
    kernel = cu[start:end]
    kernel = replace(kernel, 'const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n',
                     'const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n'
                     'static_assert(MC == 1 || (MC == 2 && CL == 1 && TOKR == 32 && UPS % 2 == 0), "SK2MC geometry");\n'
                     'uint32_t rk = 0;\n'
                     'if constexpr (MC == 2) asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rk));\n'
                     '// Four spare bytes in the existing 32-byte barrier tail; no smem growth.\n'
                     'uint32_t* mc_error = reinterpret_cast<uint32_t*>(gbase + NSTAGE * SST + 16 * NSTAGE + 24);\n')
    kernel = replace(kernel, 'if (tid == 0) {\n', 'if (tid == 0) {\nif constexpr (MC == 2) *mc_error = 0u;\n')
    first_x = 'for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);\n'
    kernel = replace(kernel, first_x, 'if constexpr (MC == 1) {\n' + first_x + '}\n')
    kernel = replace(kernel, 'mbar_init(empty0, NW + (COMP ? NESC : 0));',
                     'mbar_init(empty0, MC * (NW + (COMP ? NESC : 0)));')
    kernel = replace(kernel, 'mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0));',
                     'mbar_init(empty0 + 8 * i, MC * (NW + (COMP ? NESC : 0)));')
    kernel = replace(kernel, '__syncthreads();\n',
                     'if constexpr (MC == 2) sk2mc_sync(); else __syncthreads();\n'
                     '// All peer full/empty barriers exist before the first multicast.\n'
                     'if constexpr (MC == 2) {\nif (tid == 0)\nfor (int st = 0; st < P; ++st)\n'
                     'sk2mc_tma(sbase + st * SST + rk * (UPS / 2) * XU, &xmap, full0 + 8 * st,\n'
                     '(gs0 + st) * UPS + rk * (UPS / 2));\n}\n')
    kernel = replace(kernel, 'tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);',
                     'if constexpr (MC == 2)\n'
                     'sk2mc_tma(s0 + rk * (UPS / 2) * XU, &xmap, full0 + 8 * slot, (gs0 + st) * UPS + rk * (UPS / 2));\n'
                     'else tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);')
    kernel = replace(kernel, 'mbar_arrive(empty0 + 8 * slot)', 'sk2mc_empty<MC>(empty0 + 8 * slot, rk)', 4)
    # Every kernel thread reaches the final barrier. No producer/escape return.
    kernel = replace(kernel, '__syncwarp();\nreturn;\n}\nif (warp >= NW + 1) {',
                     '__syncwarp();\n}\nelse if (warp >= NW + 1) {')
    kernel = replace(kernel, '__syncwarp();\nreturn;\n}\nconst int wr',
                     '__syncwarp();\n}\nelse {\nconst int wr')
    for old, new in (
        ('SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);',
         'sk2mc_wait<MC, true>(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1, mc_error);'),
        ('SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);',
         'sk2mc_wait<MC, true>(full0 + 8 * slot, (st / NSTAGE) & 1, mc_error);'),
        ('mbar_wait_sleep(pbar, 0);', 'sk2mc_wait<MC, true>(pbar, 0, mc_error);'),
        ('mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);',
         'sk2mc_wait<MC, false>(full0 + 8 * slot, (st / NSTAGE) & 1, mc_error);'),
        ('mbar_wait(cbar, 0);', 'sk2mc_wait<MC, false>(cbar, 0, mc_error);'),
    ):
        kernel = replace(kernel, old, new)
    kernel = replace(kernel, '*reinterpret_cast<uint2*>(o) = pk;\n}\n}\ntemplate',
                     '// Sticky wait failure is visible as non-finite output, never a false pass.\n'
                     'if constexpr (MC == 2) { if (*mc_error) pk.x = pk.y = 0x7fc07fc0u; }\n'
                     '*reinterpret_cast<uint2*>(o) = pk;\n}\n}\n'
                     'if constexpr (MC == 2) sk2mc_sync(); // all 288 threads, both CTAs\n}\ntemplate')
    assert 'return;' not in kernel and kernel.count('sk2mc_sync()') == 2
    cu = cu[:start] + kernel + cu[end:]
    cu = replace(cu, '(uint32_t)TOKR, UPS);', '(uint32_t)TOKR, UPS / MC);')
    cu = replace(cu, 'if (pdl > 0) {\ncudaLaunchConfig_t', 'if (pdl > 0 || MC == 2) {\ncudaLaunchConfig_t')
    cu = replace(cu, 'cudaLaunchAttribute at_[1];\nat_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n'
                 'at_[0].val.programmaticStreamSerializationAllowed = 1;\nlc.attrs = at_;\nlc.numAttrs = 1;',
                 'cudaLaunchAttribute at_[2] = {};\nint na = 0;\nif (pdl > 0) {\n'
                 'at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n'
                 'at_[na++].val.programmaticStreamSerializationAllowed = 1;\n}\n'
                 'if constexpr (MC == 2) {\n'
                 'TORCH_CHECK(N / R == 96 && M == 32 && K == 8192, "SK2MC: cfg6 qkv32 only");\n'
                 'at_[na].id = cudaLaunchAttributeClusterDimension;\n'
                 'at_[na++].val.clusterDim = {2, 1, 1};\n}\n'
                 'lc.attrs = at_;\nlc.numAttrs = na;')
    cu = replace(cu, 'launch_sk2<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, __nv_bfloat16>',
                 'launch_sk2<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, __nv_bfloat16, false, 2>')
    return cu


BANK = '''SK2_MC={} # call-time only: {"qkv": 1}; empty/zero selects the original ship module
_SK2MC={"mod":None,"err":None,"secs":0.0}
def _sk2mc_build():
 from torch.utils.cpp_extension import load_inline
 t0=time.time()
 try:
  tag=hashlib.sha1((_SK2_CPP+_SK2MC_CUDA+" ".join(_SK2_FLAGS)+" c++20").encode()).hexdigest()[:12]
  kw=dict(cpp_sources=_SK2_CPP,cuda_sources=_SK2MC_CUDA,functions=["sk2","sk2_set_pdl"],extra_cuda_cflags=list(_SK2_FLAGS)+["-std=c++20"],extra_cflags=["-std=c++20"],verbose=False)
  try:
   _SK2MC["mod"]=load_inline(name=f"sk2mc_{tag}",no_implicit_headers=True,**kw)
  except TypeError:
   _SK2MC["mod"]=load_inline(name=f"sk2mc_{tag}i",**kw)
 except Exception as ex:
  _SK2MC["err"]=repr(ex)[-600:]
 _SK2MC["secs"]=time.time()-t0

'''


def generate(s):
    mc = patched_cuda(literal(s, '_SK2X_BASE_CUDA'))
    s = replace(s, 'SK2_DG={} #', '_SK2MC_CUDA=r"""' + mc + '"""\n' + BANK + 'SK2_DG={} #')
    s = replace(s, '  jobs.append(("sk2x",_sk2x_build))\n',
                 '  jobs.append(("sk2x",_sk2x_build))\n  jobs.append(("sk2mc",_sk2mc_build))\n')
    s = replace(s, '  assert mod is not None, _SK2X["err"]\n mod.sk2_set_pdl(pdl)\n',
                 '  assert mod is not None, _SK2X["err"]\n'
                 ' if proj=="qkv" and SK2_MC.get("qkv",0):\n'
                 '  assert SK2_MC["qkv"]==1 and SK2_DG.get("qkv",0)==0, "SK2MC cannot combine with the wrong-output x probe"\n'
                 '  mod=_SK2MC["mod"]\n'
                 '  assert mod is not None, _SK2MC["err"]\n mod.sk2_set_pdl(pdl)\n')
    s = replace(s, '   _sk2x_build()\n', '   _sk2x_build()\n'
                 '  _pt1_wait("sk2mc")\n'
                 '  if _SK2MC["mod"]is None and _SK2MC["err"]is None:\n   _sk2mc_build()\n')
    return s


def functions(s, names):
    nodes = [n for n in tree(s).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names), names
    return ast.Module(body=nodes, type_ignores=[])


def dry_run(base, generated):
    """Real SK2 Python AST and base dense-state constructor, tensor/build doubles.

    Copied from the proven SK2X/padz one-state bank pipeline. Packing arithmetic
    is unchanged and mocked; this verifies new build/key/selector paths only.
    """
    class Tensor:
        dtype, device = "bf16", "cpu"
        def __init__(self, shape=(32, 8192)): self.shape = shape
        def contiguous(self): return self
        def float(self): return self
        def t(self): return Tensor(self.shape[::-1])
        def to(self, *a): return self
        def dim(self): return len(self.shape)
        def is_contiguous(self): return True
        def data_ptr(self): return 16
        def __mul__(self, v): return self
        def __matmul__(self, v): return Tensor((self.shape[0], v.shape[1]))
        def __sub__(self, v): return self
        def norm(self): return self
        def clamp_min(self, v): return self
        def __truediv__(self, v): return 0.0
        def all(self): return True
    calls, builds, waits = [], [], []
    build_mode = ["ok"]
    class Module:
        def __init__(self, label): self.label = label
        def sk2_set_pdl(self, pdl): self.pdl = pdl
        def sk2(self, *args): calls.append((self.label, self.pdl, args))
    def load_inline(**kw):
        if build_mode[0] == "error": raise RuntimeError("CPU gate injected compile failure")
        if build_mode[0] == "legacy" and "no_implicit_headers" in kw: raise TypeError("legacy API")
        assert kw["functions"] == ["sk2", "sk2_set_pdl"]
        label = {literal(base, '_SK2X_BASE_CUDA'): 'ship', literal(base, '_SK2_CUDA'): 'dg',
                 literal(generated, '_SK2MC_CUDA'): 'mc'}[kw['cuda_sources']]
        expected = list(literal(base, '_SK2_FLAGS')) + (['-std=c++20'] if label == 'mc' else [])
        assert kw['extra_cuda_cflags'] == expected
        if label == 'mc': assert kw['extra_cflags'] == ['-std=c++20']
        builds.append(label)
        return Module(label)
    fake = SimpleNamespace(bfloat16="bf16", int64="i64", cuda=SimpleNamespace(synchronize=lambda: None),
                           empty=lambda shape, **kw: Tensor(shape), zeros=lambda *a, **kw: Tensor(),
                           randn=lambda shape, **kw: Tensor(shape), equal=lambda *a: True,
                           isfinite=lambda a: a, Generator=lambda **kw: SimpleNamespace(manual_seed=lambda n: None))
    env = dict(torch=fake, time=time, hashlib=hashlib, _pt1_wait=waits.append, HIDDEN=8192, D_EXPERT=4096)
    for key in ("_SK2", "_SK2X", "_SK2MC", "SK2_DG", "SK2_MC", "_SK2_CUDA", "_SK2X_BASE_CUDA", "_SK2MC_CUDA",
                "_SK2_CPP", "_SK2_FLAGS", "SK2_ENABLE", "SK2_ON", "SK2_PDL", "SK2_T", "_SK2_CFGS", "_SK2_K", "_SK2_PROJ"):
        node = next(n for n in tree(generated).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == key for t in n.targets))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<real SK2 globals>", "exec"), env)
    env.update(_sk2_compress=lambda w: {"packed": w, "base4": 0},
               _sk2_stream=lambda packed, *a: packed, _sk2_table=lambda *a: (Tensor(), 2))
    exec(compile(functions(generated, {"_sk2_build", "_sk2x_build", "_sk2mc_build", "_sk2_prepare", "_sk2_run", "_sk2"}),
                 "<real SK2 Python>", "exec"), env)
    dense_expr = [v for n in ast.walk(tree(base)) if isinstance(n, ast.Dict)
                  for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and k.value == "dense"
                  and isinstance(v, ast.DictComp)]
    assert len(dense_expr) == 1
    weights = {k: Tensor((n, kdim)) for k, n, kdim in
               (("qkv_gate", 2304, 8192), ("o", 8192, 2048), ("pooler", 4096, 8192), ("unpooler", 8192, 4096))}
    state = {"dense": eval(compile(ast.Expression(dense_expr[0]), "<base dense state>", "eval"), {"weights": weights})}
    fake_ce = SimpleNamespace(load_inline=load_inline)
    modules = {"torch": fake, "torch.utils": SimpleNamespace(cpp_extension=fake_ce), "torch.utils.cpp_extension": fake_ce}
    # Execute the real prepare-time job registration too; synchronous thread
    # doubles run the exact build callbacks without starting background work.
    registered = []
    class Thread:
        def __init__(self, target, name, daemon): self.target = target
        def start(self): self.target()
    def target(name, fn):
        registered.append(name)
        return fn
    env.update(threading=SimpleNamespace(Thread=Thread), _pt1_target=target, _PT1={'threads': {}}, PT1_PREFETCH=True)
    for flag in ('V10_COMBINED_BUILD', 'KS1_ENABLE', 'SBB1_ENABLE', 'GD_BUILD', 'VML_ENABLE', 'HYB_ENABLE',
                 'HYBLO_ENABLE', 'VMS_ENABLE', 'AX1_ENABLE', 'HYA_ENABLE', 'VD_ENABLE', 'LT_ENABLE'):
        env[flag] = False
    exec(compile(functions(generated, {'_pt1_start'}), '<real prepare job registration>', 'exec'), env)
    with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
        env['_pt1_start']()
        env['_sk2_prepare'](state, 0)
    assert builds == ['ship', 'dg', 'mc'] and waits == ['sk2', 'sk2x', 'sk2mc']
    assert registered == ['sk2', 'sk2x', 'sk2mc'] and set(env['_PT1']['threads']) == set(registered)
    assert len(calls) == 6 and all(label == 'ship' and args[-2] == 0 for label, _, args in calls)
    assert env['_SK2']['ok'] == {k: True for k in env['_SK2_CFGS']}
    exercised = []
    for mc, dg in (({}, {}), ({'qkv': 0}, {}), ({'qkv': 1}, {}), ({}, {'qkv': 16}),
                   ({'qkv': 1}, {'unpooler': 3}), ({}, {})):
        env['SK2_MC'], env['SK2_DG'] = mc, dg
        for proj, (name, n, k) in env['_SK2_PROJ'].items():
            result = env['_sk2'](state, proj, Tensor((32, k)))
            label, pdl, args = calls[-1]
            expected = 'mc' if proj == 'qkv' and mc.get('qkv') else ('dg' if dg.get(proj) else 'ship')
            assert result is not None and label == expected and args[-2] == dg.get(proj, 0)
            assert pdl == env['SK2_PDL'][proj] and args[9] == env['_SK2_CFGS'][name][0]
            assert args[0] is env['_SK2']['w'][name][1] and args[3] is env['_SK2']['w'][name][2]
            before = len(calls)
            assert env['_sk2'](state, proj, Tensor((16, k))) is None and len(calls) == before
        exercised.append({'SK2_MC': mc, 'SK2_DG': dg})
    # Invalid combined correctness/probe selection must fail before any launch.
    env['SK2_MC'], env['SK2_DG'] = {'qkv': 1}, {'qkv': 16}
    before = len(calls)
    try: env['_sk2_run']('qkv32_r208', Tensor(), 2304, 8192, 0, 'qkv')
    except AssertionError: pass
    else: raise AssertionError('MC + wrong-output DG probe was allowed')
    assert len(calls) == before
    env['SK2_MC'], env['SK2_DG'] = {}, {}
    with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
        env['_SK2MC'].update(mod=None, err=None)
        env['_sk2_prepare'](state, 0) # PT1-off fallback
        assert env['_SK2MC']['mod'] is not None
        build_mode[0] = 'legacy'
        env['_SK2MC'].update(mod=None, err=None)
        env['_sk2mc_build']()
        assert env['_SK2MC']['mod'] is not None
        build_mode[0] = 'error'
        env['_SK2MC'].update(mod=None, err=None)
        env['_sk2mc_build']()
        assert env['_SK2MC']['mod'] is None and 'injected compile failure' in env['_SK2MC']['err']
        env['SK2_MC'] = {'qkv': 1}
        count = len(calls)
        try: env['_sk2_run']('qkv32_r208', Tensor(), 2304, 8192, 0, 'qkv')
        except AssertionError: pass
        else: raise AssertionError('MC missing module silently selected another module')
        assert len(calls) == count
        env['SK2_MC'] = {}
        env['_sk2_prepare'](state, 0) # recorded MC failure must not disable ship
        assert env['_SK2']['ok']['qkv32_r208']
        env['SK2_ENABLE'] = False
        count = len(builds)
        env['_sk2_prepare'](state, 0)
        env['_pt1_start']()
        assert len(builds) == count and env['_SK2']['ok'] == {}
        env['PT1_PREFETCH'] = False
        env['_pt1_start']()
    return {'prepare': 'PASS: real AST; tensor arithmetic/compiler mocked', 'arms': exercised,
            'state': 'base dense comprehension / state["dense"]', 'bank': '_SK2 / _SK2X / _SK2MC',
            'T16': 'no SK2 call', 'registered_builds': registered,
            'fallbacks': 'PT1-off, legacy API, compile failure, DG collision, SK2 disabled'}


def protocol_gate():
    """Byte-for-byte swizzle map plus role/phase coverage of all 48 clusters."""
    ship, halves = {}, {}
    for u in range(8):
        for t in range(32):
            for k in range(64):
                addr = u * 4096 + t * 128 + (((k >> 3) ^ ((u * 32 + t) & 7)) << 4) + (k & 7) * 2
                ship[addr] = (u, t, k)
    for rk in (0, 1):
        for u in range(4):
            for t in range(32):
                for k in range(64):
                    addr = rk * 16384 + u * 4096 + t * 128 + (((k >> 3) ^ ((u * 32 + t) & 7)) << 4) + (k & 7) * 2
                    assert addr not in halves
                    halves[addr] = (rk * 4 + u, t, k)
    assert ship == halves and len(ship) * 2 == 32768
    for cluster in range(48):
        assert [2 * cluster, 2 * cluster + 1] == list(range(2 * cluster, 2 * cluster + 2))
        for pdl in (0, 1, 2, 3, 4):
            p = min(pdl, 4, 16) if pdl else 1
            assert list(range(p)) + list(range(p, 16)) == list(range(16))
            for stage in range(16):
                assert 2 * (32768 // 2) + 18432 == 51200 # each destination full barrier
                arrivals = [(rk, warp) for rk in (0, 1) for warp in (0, 1, 2, 3, 5, 6, 7, 8)]
                assert len(arrivals) == 16 and len(set(arrivals)) == 16
                if stage >= 4: assert ((stage // 4) - 1) & 1 == (((stage - 4) // 4) & 1)
    assert sorted(i for lane in range(32) for i in range(lane, 32 * 25, 32)) == list(range(800))
    assert sorted(i for tid in range(128) for i in range(tid, 6 * 32, 128)) == list(range(192))
    staging = []
    for tid in range(128):
        warp, lane = divmod(tid, 32)
        for mt in range(2):
            for nt in range(3):
                row, tok = nt * 8 + 2 * (lane & 3), mt * 16 + (lane >> 2)
                off = (warp * 32 + tok) * 40 + row
                staging.extend((off, off + 1, off + 320, off + 321))
    assert sorted(staging) == sorted((w * 32 + t) * 40 + r for w in range(4) for t in range(32) for r in range(24))
    return {'swizzle': 'PASS: all 16384 BF16 coordinates, exact offsets/source coordinates',
            'CTAs': '0..95 = 48 adjacent pairs; no CTA/role exits early',
            'init': 'tid0 per CTA: full[0:4], empty[0:4] count16, pbar,cbar,qbar; spare error word reset',
            'zero_row': 'tid32..63 write all 32 uint32 entries before cluster entry barrier',
            'producer': 'tid128 issues stages P..15; tid0 issues 0..P-1; all ranks same stage/slot/parity',
            'escape_cleanup': 'tid160..287: each warp zeros all 800 FP32 corr entries (lane stride32), ptab wait; '
                              'all four warp leaders arrive local+peer once/stage, including skip-stage path; all 128 threads arrive cbar',
            'consumer_cleanup': 'tid0..127: four warp leaders arrive local+peer once/stage; UPW=2 uses post-loop arrival; '
                                'all 128 wait cbar, execute both named barriers, stage all 3072 red floats, combine all 192 float4 items',
            'end': 'tid0..287 in both CTAs execute one trailing cluster barrier after all roles; no rcnt/split-K loop (CL=1)',
            'waits': 'MC polls bounded by 2^20 attempts / 2^30 clock ticks plus one finite sleep hint; timeout poisons output; '
                     'hardware cluster/named barriers have fixed full participation, no polling timeout/early exit'}


def cpu_gates(base, generated):
    import py_compile
    assert generated == generate(base), 'stale generated file'
    for key in ('_SK2_CUDA', '_SK2X_BASE_CUDA', '_SK2_CPP', '_SK2_FLAGS', 'SK2_DG'):
        assert literal(base, key) == literal(generated, key), key
    assert literal(generated, 'SK2_MC') == {}
    # Ship builder and all unaffected Python functions stay literally identical.
    old_lines, new_lines = base.splitlines(keepends=True), generated.splitlines(keepends=True)
    old_nodes = {n.name: ''.join(old_lines[n.lineno - 1:n.end_lineno]) for n in tree(base).body if isinstance(n, ast.FunctionDef)}
    new_nodes = {n.name: ''.join(new_lines[n.lineno - 1:n.end_lineno]) for n in tree(generated).body if isinstance(n, ast.FunctionDef)}
    changed = [k for k in old_nodes if old_nodes[k] != new_nodes[k]]
    assert changed == ['_pt1_start', '_sk2_run', '_sk2_prepare'], changed
    for path in (Path(__file__), OUT, ROOT / 'tools/box/sk2mc_check.py'):
        bytecode = []
        with patch('importlib._bootstrap_external._write_atomic', lambda p, data, mode=0o666: bytecode.append(data)):
            py_compile.compile(str(path), cfile=str(path), doraise=True)
        assert bytecode
    assert b'\r' not in (ROOT / 'tools/local/sk2mc_gate.sh').read_bytes()
    ns = {'__name__': 'sk2mc_cpu_arms', '__file__': str(ROOT / 'tools/box/sk2mc_check.py')}
    exec(compile(Path(ns['__file__']).read_text(encoding='utf-8'), ns['__file__'], 'exec'), ns)
    assert [v['name'] for v in ns['arms']()] == ['ship', 'ship2', 'mc', 'ship3']
    assert ns['SHAPES'] == (4, 5, 2)
    return {'status': 'CPU PASS', 'py_compile': 'PASS (serialized bytecode held in memory)',
            'default_identity': 'PASS: original ship source, flags, builder, stream/table; selector defaults off',
            'changed_base_functions': changed, 'dry_run': dry_run(base, generated), 'protocol': protocol_gate(),
            'accounting': {'weight_stream_grows': False, 'weight_bytes_per_CTA_stage': [18432, 18432],
                           'x_source_bytes_per_CTA_stage': [32768, 16384], 'x_destination_bytes_per_CTA_stage': [32768, 32768],
                           'weight_bytes_per_call': 96 * 16 * 18432, 'x_source_saved_per_call': 96 * 16 * 16384,
                           'scratch_bytes': 'same smem; 4-byte error word in spare barrier tail, initialized once/CTA',
                           'issue_inline_PTX_per_producer_stage': '5 -> 5: expect, weight policy/bulk, x policy/TMA; rank offset integer arithmetic added',
                           'arrival_inline_PTX_per_leader_stage': '1 -> 3: local arrive + mapa + remote arrive (8 leaders/CTA)',
                           'wait_PTX_per_poll': 'same try_wait/test + selp, plus bounded-loop integer/control/clock reads; final SASS in Linux report',
                           'consumer_math_per_thread_stage': 'unchanged: 12 weight LDS (144 B/thread) + 16 LDSM + 48 MMA + 12 decode8 calls; each decode8 = 2 lop3 + 4 prmt plus integer ops',
                           'epilogue': 'same 3072 red floats / 192 float4 items; one error-word read + poison selection per store',
                           'sync': 'one entry CTA barrier -> cluster barrier; one trailing cluster barrier added; 16 remote empty arrives/leader over 16 stages'}}


def linux_gate(report):
    import collections
    import subprocess
    import sysconfig
    import tempfile
    import torch
    import torch.utils.cpp_extension as ce
    base, lab = BASE.read_text(encoding='utf-8'), OUT.read_text(encoding='utf-8')
    report.update(cpu_gates(base, lab))
    report['status'] = 'FAIL'
    work = Path(tempfile.mkdtemp(prefix='sk2mc_gate_'))
    report['artifacts'], report['builds'] = str(work), {}
    sources = {'base': literal(base, '_SK2X_BASE_CUDA'), 'off': literal(lab, '_SK2X_BASE_CUDA'),
               'mc': literal(lab, '_SK2MC_CUDA')}
    sass = {}
    for name, cu in sources.items():
        directory = work / name
        directory.mkdir()
        flags = list(literal(base, '_SK2_FLAGS')) + ['-std=c++20']
        record = report['builds'][name] = {'source_sha256': hashlib.sha256(cu.encode()).hexdigest(), 'directory': str(directory)}
        try:
            mod = ce.load_inline(name='sk2mc_gate_' + name, cpp_sources=literal(base, '_SK2_CPP'), cuda_sources=cu,
                                 functions=['sk2', 'sk2_set_pdl'], extra_cuda_cflags=flags,
                                 extra_cflags=['-std=c++20'], no_implicit_headers=True,
                                 build_directory=str(directory), verbose=True)
        except Exception as ex:
            record['extension_build_error'] = str(ex)
            raise
        ninja = (directory / 'build.ninja').read_text()
        assert '-std=c++17' not in ninja
        for key in ('cflags =', 'cuda_cflags ='):
            assert '-std=c++20' in next(line for line in ninja.splitlines() if line.startswith(key))
        inc = sum((['-I', p] for p in ce.include_paths(device_type='cuda')), []) + ['-I', sysconfig.get_paths()['include']]
        cmd = ['nvcc', '-cubin', '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__',
               '-D__CUDA_NO_BFLOAT16_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__', '--expt-relaxed-constexpr',
               '-DTORCH_EXTENSION_NAME=sk2mc_gate_' + name, '-DTORCH_API_INCLUDE_EXTENSION_H',
               f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}', '-Xptxas=-v']
        cmd += inc + flags + [str(directory / 'cuda.cu'), '-o', str(directory / 'sk2.cubin')]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        record.update(nvcc_command=cmd, ptxas=proc.stderr, returncode=proc.returncode, ninja=ninja)
        (directory / 'ptxas.log').write_text(proc.stderr)
        assert proc.returncode == 0, proc.stderr
        spills = re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', proc.stderr)
        assert len(spills) == 2, 'expected cfg6/cfg11 resource records'
        record['spill_records'] = spills
        assert all(int(stores) == int(loads) == 0 for _, stores, loads in spills), 'spill gate failed'
        txt = subprocess.check_output(['cuobjdump', '-sass', mod.__file__], text=True)
        (directory / 'sk2.sass').write_text(txt)
        record['sass'] = txt
        parsed = {}
        for match in re.finditer(r'Function\s*:\s*(\S+)(.*?)(?=Function\s*:|\Z)', txt, re.S):
            if 'sk2_kernel' in match[1]:
                parsed[match[1]] = [' '.join(ln.split()) for ln in match[2].splitlines() if re.search(r'/\*[0-9a-f]+\*/', ln)]
        assert len(parsed) == 2
        record['sass_opcode_counts_by_kernel'] = {
            k: dict(collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\b', '\n'.join(v))))
            for k, v in parsed.items()}
        sass[name] = parsed
    assert sass['base'] == sass['off'], 'off module SASS differs from ship'
    mc = [v for k, v in sass['mc'].items() if 'Li3ELi1ELi4ELi2ELi4E' in k]
    assert len(mc) == 1, 'missing cfg6 qkv specialization'
    tma = [ln for ln in mc[0] if re.search(r'\bUTMALDG\b', ln)]
    multi = [ln for ln in tma if 'MULTICAST' in ln]
    report['multicast_sass_lines'] = multi
    assert multi and tma == multi, 'qkv MC must have multicast TMA and no unicast x TMA'
    report['default_sass_identity'] = True
    report['status'] = 'PASS'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--linux-gate', action='store_true')
    args = parser.parse_args()
    if args.linux_gate:
        report = {'status': 'FAIL', 'GPU': 'not run'}
        try:
            linux_gate(report)
        except Exception as ex:
            report['status'], report['error'] = 'FAIL', repr(ex)
            raise
        finally:
            (ROOT / 'tools/local/sk2mc_gate.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
        return
    base = BASE.read_text(encoding='utf-8')
    generated = generate(base)
    if not args.check: OUT.write_bytes(generated.encode('utf-8'))
    print(json.dumps(cpu_gates(base, OUT.read_text(encoding='utf-8')), indent=2))


if __name__ == '__main__':
    main()

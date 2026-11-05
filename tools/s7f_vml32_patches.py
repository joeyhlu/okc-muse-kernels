"""S7 VML32: original-top-k-slot y addressing, with warp-local finalizer metadata.

Only writes the task's v1 file and its requested gate JSON. Never imports the layer.
Default-off VML32_ON is a capture-time selector; --activate emits the S7 active form.
CPU gates execute the actual allocation/build/prepare/route functions using tensors
and compiler calls replaced by strict CPU doubles. No CUDA context is created.

Kill reasons, first: B-epilogue shared indirection may erase the metadata saving;
the removed rmap dependency may be shorter than the standalone 0.5 us floor.
This is a correctness-complete experiment, not a measured performance claim.
"""
from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import difflib
import hashlib
import io
import json
from pathlib import Path
import random
import re
import runpy
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUT = ROOT / "box_out/s7/k3_vml32_v1.py"
RESULT = ROOT / "tools/local/s7f_vml32_gate.json"
SCRIPT = ROOT / "tools/local/s7f_vml32_gate.sh"


def sub(s, old, new, count=1):
    assert s.count(old) == count, (old[:140], s.count(old), count)
    return s.replace(old, new)


def cuda_patch(s):
    s = sub(s, "namespace vm {\n", """#ifndef VM_SLOT32
#define VM_SLOT32 0
#endif
#if VM_SLOT32 && (!VM_BRED || !VM_FTR || !VM_RT)
#error "VM_SLOT32 requires the shipped bf2 BRED/FTR/RT specialization"
#endif
namespace vm {
""")
    s = sub(s, "unsigned long long bperm[JMAX];\n};\n", """unsigned long long bperm[JMAX];
#if VM_SLOT32
uint16_t slot32[256];  // compact local row -> original token/top-k slot; M == 32 only
#endif
};
""")
    # Each live compact row has one writer in every CTA's route. VM_RSYNC already
    # publishes ent/Tab before any consumer can read either one.
    s = sub(s, "if (x >= 0) ent[r] = (uint16_t)e;", """if (x >= 0) {
ent[r] = (uint16_t)e;
#if VM_SLOT32
if (A.M == 32) T.slot32[r] = (uint16_t)e;
#endif
}""")
    # Also keep the inactive non-RT implementation structurally correct.
    s = sub(s, "if (x >= 0) { r = T.est[x] + hist[warp * 33 + x] + rank; ent[r] = (uint16_t)e; }", """if (x >= 0) { r = T.est[x] + hist[warp * 33 + x] + rank; ent[r] = (uint16_t)e;
#if VM_SLOT32
if (A.M == 32) T.slot32[r] = (uint16_t)e;
#endif
}""")
    # The two B epilogues are the ONLY writers of A.y. Preserve both store and
    # split red.global.add, including exactly the same operands and ordering.
    s = sub(s, "float* yb = A.y + (size_t)T.jS[j] * KDIM;", """#if VM_SLOT32
const bool slot32 = A.M == 32;
#endif
float* yb = A.y + (size_t)T.jS[j] * KDIM;""", 2)
    for statement in (
        "yb[(size_t)n * KDIM + h0] = d[4 * nt + hh] * gs;\nyb[(size_t)n * KDIM + h0 + 8] = d[4 * nt + 2 + hh] * gs;",
        'asm volatile("red.global.add.f32 [%0], %1;\\n" :: "l"(yb + (size_t)n * KDIM + h0), "f"(d[4 * nt + hh] * gs) : "memory");\nasm volatile("red.global.add.f32 [%0], %1;\\n" :: "l"(yb + (size_t)n * KDIM + h0 + 8), "f"(d[4 * nt + 2 + hh] * gs) : "memory");',
    ):
        replacement = """#if VM_SLOT32
float* yr = slot32 ? A.y + (size_t)T.slot32[T.jS[j] + n] * KDIM : yb + (size_t)n * KDIM;
#else
float* yr = yb + (size_t)n * KDIM;
#endif
""" + statement.replace("yb[(size_t)n * KDIM + h0", "yr[h0").replace("yb + (size_t)n * KDIM + h0", "yr + h0")
        s = sub(s, statement, replacement)
    # Only edit vm_fin after this point, not similarly spelled code in vm_main.
    a = s.index("__global__ void __launch_bounds__(256) vm_fin(")
    z = s.index("\nvoid vml_run(", a)
    fin = s[a:z]
    fin = sub(fin, "const float wv = (tid < 8) ? __ldcg(A.wts + t * 8 + tid) : 0.f;", """#if VM_SLOT32
const bool slot32 = A.M == 32;
const float wv = (slot32 ? lane < 8 : tid < 8) ? __ldcg(A.wts + t * 8 + lane) : 0.f;
const int slot_id = slot32 && lane < 8 ? __ldcg(A.ids + t * 8 + lane) : -1;
unsigned slot_code = 0, slot_mask = 0;
#else
const float wv = (tid < 8) ? __ldcg(A.wts + t * 8 + tid) : 0.f;
#endif""")
    nw_load = "for (int m = 0; m < 4; ++m) w4[m] = __ldcg(reinterpret_cast<const float4*>(A.nw + 1024 * m + 4 * tid));"
    fin = sub(fin, nw_load, nw_load + """
#if VM_SLOT32
if (slot32) {
const bool live = lane < 8 && (unsigned)(slot_id - A.ls) < (unsigned)E_LOC;
slot_mask = __ballot_sync(0xffffffffu, live);
const int rank = __popc(slot_mask & ((1u << lane) - 1u));
const unsigned part = live ? (unsigned)lane << (3 * rank) : 0u;
slot_code = __reduce_or_sync(0xffffffffu, part);
}
#endif""")
    old = """if (warp == 0) {
const int r = (lane < 8) ? __ldcg(A.rmap + t * 8 + lane) : -1;
const unsigned bm = __ballot_sync(0xffffffffu, r >= 0);
if (r >= 0) { const int pos = __popc(bm & ((1u << lane) - 1u)); s_r[pos] = r; s_w[pos] = wv; }
if (lane == 0) s_n = __popc(bm);
}
__syncthreads();"""
    fin = sub(fin, old, "#if VM_SLOT32\nif (!slot32) {\n#endif\n" + old + "\n#if VM_SLOT32\n}\n#endif")
    fin = sub(fin, "const int nloc = s_n;", """#if VM_SLOT32
const int nloc = slot32 ? __popc(slot_mask) : s_n;
#else
const int nloc = s_n;
#endif""")
    old = "const int r = (c0 + jj < nloc) ? s_r[c0 + jj] : -1;"
    fin = sub(fin, old, """#if VM_SLOT32
const int r = c0 + jj < nloc ? (slot32 ? t * 8 + ((slot_code >> (3 * (c0 + jj))) & 7u) : s_r[c0 + jj]) : -1;
#else
const int r = (c0 + jj < nloc) ? s_r[c0 + jj] : -1;
#endif""", 2)
    fin = sub(fin, "const float fr = rsqrtf(sm * (1.f / 4096.f) + 1e-8f) * s_w[c0 + jj];", """#if VM_SLOT32
const int sw_lane = (slot_code >> (3 * (c0 + jj))) & 7u;
const float sw = slot32 ? __shfl_sync(0xffffffffu, wv, sw_lane) : s_w[c0 + jj];
#else
const float sw = s_w[c0 + jj];
#endif
const float fr = rsqrtf(sm * (1.f / 4096.f) + 1e-8f) * sw;""")
    return s[:a] + fin + s[z:]


# Bank and hook follow tools/padz_patches.py:_vmlz_build/_vmlz_prepare/VROUTE_N.
# Their st argument is passed by _vml_prepare, itself called by
# _v11_k2a33_prepare. No invented lookup into the outer layer state.
HOST = '''VML32_ON=0
_VML32={"mod":None,"sc":None,"ok":False,"err":None}
def _vml32_build():
 _VML32["mod"]=None
 try:
  _VML32["mod"],_=_vml_build(defines=_VML_DEFS["bf2"]+("-DVM_SLOT32=1",))
 except Exception as ex:
  _VML32["err"]=repr(ex)[-600:]
def _vml32_prepare(st,w,dev):
 global VML32_ON
 saved=VML32_ON
 _VML32["ok"]=False
 if _VML32["mod"] is None or not _VML["ok"].get(32):
  return
 good=False
 try:
  sms=torch.cuda.get_device_properties(dev).multi_processor_count
  _VML32["sc"]=dict(_vml_alloc(dev,sms),err=_VML["sc"]["err"])
  torch.cuda.synchronize()
  good=True
  for bs,seed,dist in ((25,48,"normal"),(31,53,"normal"),(32,49,"skewed"),(32,54,"normal"),(33,55,"normal"),(48,50,"skewed"),(32,61,"normal")):
   d=_k2a_self_test_inputs(bs,seed,dev,dist)
   d={"hidden_states":d["hidden_states"].contiguous(),"topk_ids":d["topk_ids"].to(torch.int32).contiguous(),
      "topk_weights":d["topk_weights"].float().contiguous(),"config":{"local_start":0,"local_count":K2A_E_LOC}}
   ref=_k2a_ref_fp32(d,w)
   _VML["sc"]["err"].zero_()
   VML32_ON=0
   o0=_vml_route(st,d).clone()
   o1=_vml_route(st,d).clone()
   VML32_ON,_VML32["ok"]=1,True
   oa=_vml_route(st,d).clone()
   ob=_vml_route(st,d).clone()
   VML32_ON,_VML32["ok"]=0,False
   torch.cuda.synchronize()
   tot,row=_k2a_rel_err(oa,ref)
   ev=_k2a_rel_err(oa,o0)[0]
   rp=_k2a_rel_err(ob,oa)[0]
   errf=int(_VML["sc"]["err"][0])
   fin=bool(torch.isfinite(oa.float()).all() and torch.isfinite(ob.float()).all())
   clean=int(torch.count_nonzero(_VML32["sc"]["y"][:32*8*_vml_KDIM]))==0
   ok=fin and clean and not errf and tot<1e-2 and row<5e-2 and ev<2e-3 and rp<1e-3
   good &= ok
   print(f"[vml32] self-test bs={bs} {dist}: fp32={tot:.2e} row={row:.2e} base={ev:.2e} repeat={rp:.2e} clean={clean} -> {'ok' if ok else 'FAIL'}",flush=True)
 except Exception as ex:
  good=False
  _VML32["err"]=repr(ex)[-600:]
  print(f"[vml32] prepare raised {ex!r} -> OFF",flush=True)
 finally:
  _VML32["ok"]=bool(good)
  VML32_ON=saved
'''


def build(activate=False):
    raw = BASE.read_bytes()
    base = raw.decode("utf-8").replace("\r\n", "\n")
    a = base.index('_vml_CUDA_T=r"""')
    z = base.index('\n_vml_CUDA=', a)
    out = base[:a] + cuda_patch(base[a:z]) + base[z:]
    out = sub(out, "def _vml_build_job():\n", HOST + "def _vml_build_job():\n")
    out = sub(out, ' _VML["mod"]=_VML["mods"][VML_VAR]\n', ' _VML["mod"]=_VML["mods"][VML_VAR]\n _vml32_build()\n')
    out = sub(out, '  _m32e_vartest(st,w,dev)\n', '  _m32e_vartest(st,w,dev)\n  _vml32_prepare(st,w,dev)\n')
    out = sub(out, ' return _vml_call(_VML["mods"].get(var,_VML["mod"]),st,scx,x,data["topk_ids"].to(torch.int32).contiguous(),',
              ''' mz=_VML["mods"].get(var,_VML["mod"])
 if VML32_ON and x.shape[0]==32 and var=="bf2" and _VML32["ok"]:
  mz,scx=_VML32["mod"],_VML32["sc"]
 return _vml_call(mz,st,scx,x,data["topk_ids"].to(torch.int32).contiguous(),''')
    out = sub(out, "\nVML32_ON=0\n", f"\nVML32_ON={int(activate)}\n")
    lines = out.splitlines(keepends=True)
    lines[1] = f"# nonce vml32-v1 (slot-addressed y; VML32_ON={int(activate)}; tools/s7f_vml32_patches.py)\n"
    return raw, base, "".join(lines)


def select_nodes(source, names, ns):
    found = set()
    for node in ast.parse(source).body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<real-layer-ast>", "exec"), ns)
            found.add(name)
    assert found == set(names), set(names) - found
    return ns


def source_namespace(source):
    ns = {}
    names = {"_vml_JMAX", "_vml_FIXW", "_vml_SLOTS", "_vml_wg_spec", "_vml_wg_spec64", "_vml_CUDA_T", "_vml_CUDA", "_vml_CPP", "_vml_FLAGS", "_VML_DEFS"}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("_vml_"):
                    try:
                        ns[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    select_nodes(source, names, ns)
    return ns


class Tensor:
    """A deliberately strict shape-only tensor. It cannot execute CUDA math."""
    def __init__(self, shape=(), value=0, dtype=None, device="cpu-double"):
        self.shape, self.value, self.dtype, self.device = tuple(shape), value, dtype, device
    def contiguous(self): return self
    def to(self, dtype): return self
    def float(self): return self
    def clone(self): return Tensor(self.shape, self.value, self.dtype, self.device)
    def zero_(self): self.value = 0; return self
    def all(self): return bool(self.value)
    def __getitem__(self, key): return Tensor((), self.value)
    def __int__(self): return int(self.value)
    def numel(self):
        n = 1
        for v in self.shape: n *= v
        return n


def dry_run(source):
    cuda = types.SimpleNamespace(get_device_properties=lambda dev: types.SimpleNamespace(multi_processor_count=132), synchronize=lambda: None)
    torch = types.SimpleNamespace(int32="int32", int64="int64", bfloat16="bfloat16", float32="float32", cuda=cuda,
                                 zeros=lambda n, dtype=None, device=None: Tensor(n, dtype=dtype, device=device),
                                 empty=lambda n, dtype=None, device=None: Tensor(n, dtype=dtype, device=device),
                                 count_nonzero=lambda x: Tensor((), x.value), isfinite=lambda x: Tensor((), 1))
    torch.zeros_like = lambda x: Tensor(x.shape, dtype=x.dtype, device=x.device)
    ns = {"torch": torch, "K2A_M_MAX": 320, "K2A_E_LOC": 32, "_vml_TOPK": 8, "_vml_KDIM": 4096}
    # Execute real base declarations, allocation and route; avoid hand-written keys.
    names = {"K2A_BUCKETS", "_k2a_bucket_of", "NB_VML_REP", "_nb_vml_rep", "_vml_MMAX", "_vml_JMAX", "_vml_RMAX",
             "_vml_SLOTS", "_vml_FIXW", "_vml_COST", "_vml_mag", "_vml_alloc", "_vml_cfg", "_vml_cost_ext",
             "_vml_pool_ext", "_vml_seg_ext", "_vml_pdel_ext", "_vml_FLAGS_DEFAULT", "_vml_call", "VML_FLAGS",
             "VML_VAR", "VML_ABVARS", "VML_BVAR", "VML_POOL", "VML_DPFLAGS", "VML_SEG", "VML_SCFLAGS", "VML_PDEL",
             "VML_COST", "_VML_DEFS", "_VML", "VML32_ON", "_VML32", "_vml32_build", "_vml32_prepare", "_vml_build_job", "_vml_route"}
    select_nodes(source, names, ns)
    calls, builds = [], []

    class Module:
        def __init__(self, defs): self.defs = tuple(defs)
        def vml_cfg(self): return [64, 8, 3, 64, 32, 213424 if "-DVM_SLOT32=1" in self.defs else 212912, 0, 0, 1, 0, 0]
        def vml_run(self, ts, vs):
            assert len(ts) == 20 and len(vs) == 22
            assert vs[5] in (1,193) and not (vs[5] & 256)
            if vs[0] == 32: assert vs[5] == 193
            calls.append((self, ts, vs))
    def compiler(verbose=False, defines=()):
        m = Module(defines); builds.append(m); return m, 0.0
    ns["_vml_build"] = compiler
    ns["_vml_build_job"]()
    assert len(builds) == 3 and ns["_VML32"]["mod"] is not ns["_VML"]["mod"]
    sc = ns["_vml_alloc"]("cpu-double", 132)
    ns["_VML"].update(sc=sc, scb=dict(sc, y=torch.zeros_like(sc["y"])), scw=ns["_vml_alloc"]("cpu-double", 132), ok={32:True}, force=None)
    # Derive the actual state keys from the actual _vml_call AST. Every tensor
    # accessor executes below; a typo introduced in a prepare/route path fails.
    st_keys = set()
    call_node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_vml_call")
    for n in ast.walk(call_node):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "k":
            st_keys.add(ast.literal_eval(n.slice))
    st = {k: Tensor((1,)) for k in st_keys}
    assert st_keys == {"ws13", "ws2", "gsc13", "gsc2", "nw"}
    state = {"k2a33": {"st": st}}  # mirrors _v11_k2a33_prepare's return {st,ok}
    assert 'state["k2a33"]["st"]' in source
    def inputs(bs, seed, dev, dist):
        return {"hidden_states": Tensor((bs,4096)), "topk_ids": Tensor((bs,8)), "topk_weights": Tensor((bs,8))}
    ns["_k2a_self_test_inputs"] = inputs
    ns["_k2a_ref_fp32"] = lambda d,w: Tensor((d["hidden_states"].shape[0],4096))
    ns["_k2a_rel_err"] = lambda a,b: (0.0,0.0)
    saved = ns["VML32_ON"]
    with contextlib.redirect_stdout(io.StringIO()):
        ns["_vml32_prepare"](state["k2a33"]["st"], {}, "cpu-double")
    assert ns["_VML32"]["ok"] and ns["VML32_ON"] == saved, ns["_VML32"]
    assert len(calls) == 28
    assert ns["_VML32"]["sc"]["y"] is not sc["y"]
    assert ns["_VML32"]["sc"]["err"] is sc["err"]
    selected = []
    for arm in (0,1):
        ns["VML32_ON"] = arm
        for m in (1,8,16,25,31,32,33,48,128,256):
            d = inputs(m,0,"cpu-double","normal"); d["config"] = {"local_start": 64}
            out = ns["_vml_route"](state["k2a33"]["st"],d)
            mod, ts, vs = calls[-1]
            chosen = mod is ns["_VML32"]["mod"]
            assert chosen == bool(arm and m == 32)
            assert out is ts[16] and vs[1] == 64
            assert ts[10] is (ns["_VML32"]["sc"]["y"] if chosen else ns["_VML"]["scb"]["y"])
            selected.append([arm,m,chosen])
    # A failed gate and a forced non-bf2 module must never select the slot bank.
    d = inputs(32,0,"cpu-double","normal"); d["config"] = {"local_start":0}
    ns["_VML32"]["ok"] = False
    ns["_vml_route"](st,d)
    assert calls[-1][0] is ns["_VML"]["mods"]["bf2"]
    ns["_VML32"]["ok"] = True
    ns["_VML"]["force"] = "w72"
    ns["_vml_route"](st,d)
    assert calls[-1][0] is ns["_VML"]["mods"]["w72"]
    assert calls[-1][1][10] is ns["_VML"]["scw"]["y"]
    ns["_VML"]["force"] = None
    # Missing keys must fail (not silently turn into a timing-only fallback).
    d = inputs(32,0,"cpu-double","normal"); d["config"] = {"local_start":0}
    bad = dict(st); del bad["ws13"]
    try: ns["_vml_route"](bad,d)
    except KeyError: pass
    else: raise AssertionError("missing weight key did not fail")
    # Exercise all newly introduced prepare exits, including the exception path.
    good_mod = ns["_VML32"]["mod"]
    ns["_VML32"]["mod"] = None
    ns["_vml32_prepare"](st,{},"cpu-double")
    assert not ns["_VML32"]["ok"]
    ns["_VML32"]["mod"] = good_mod
    ns["_VML"]["ok"] = {}
    ns["_vml32_prepare"](st,{},"cpu-double")
    assert not ns["_VML32"]["ok"]
    ns["_VML"]["ok"] = {32:True}
    ns["_k2a_rel_err"] = lambda a,b: (1.0,1.0)
    with contextlib.redirect_stdout(io.StringIO()): ns["_vml32_prepare"](st,{},"cpu-double")
    assert not ns["_VML32"]["ok"], "failed numerical gate enabled the bank"
    ns["_k2a_self_test_inputs"] = lambda *a: (_ for _ in ()).throw(KeyError("injected input failure"))
    with contextlib.redirect_stdout(io.StringIO()): ns["_vml32_prepare"](st,{},"cpu-double")
    assert not ns["_VML32"]["ok"] and "injected input failure" in ns["_VML32"]["err"]
    ns["_vml_build"] = lambda **kw: (_ for _ in ()).throw(RuntimeError("injected build failure"))
    ns["_vml32_build"]()
    assert "injected build failure" in ns["_VML32"]["err"]
    assert ns["_VML32"]["mod"] is None
    return {"status":"PASS", "prepare_calls":28, "arm_matrix":selected,
            "state_keys":sorted(st_keys), "new_python_paths":"build, allocate, prepare success/failure/early exits, select/fallback",
            "limitation":"Shape-only doubles validate Python paths and keys, not CUDA numerics."}


def layout_gate():
    # Exhaust every finalizer mask: ballot -> packed slot code -> ascending order
    # and original slot weights, including empty/full and all four-slot boundaries.
    for mask in range(256):
        live = [k for k in range(8) if mask >> k & 1]
        code = 0
        for lane in live:
            rank = (mask & ((1 << lane) - 1)).bit_count()
            code |= lane << (3 * rank)
        assert [(code >> (3*j)) & 7 for j in range(len(live))] == live
    rng = random.Random(72032)
    layouts = 0
    for m in (1,8,16,25,31,32,33,48,128,256,320):
        for ls in (0,32,224):
            for dist in range(24):
                ids = [rng.sample(range(256),8) for _ in range(m)]
                if dist == 0: ids = [[ls+i for i in range(8)] for _ in range(m)]
                if dist == 1: ids = [[(ls+32+i)%256 for i in range(8)] for _ in range(m)]
                ent = [t*8+k for e in range(ls,ls+32) for t in range(m) for k in range(8) if ids[t][k] == e]
                assert len(ent) == len(set(ent))
                rmap = {slot:r for r,slot in enumerate(ent)}
                y = {}
                # Unique recognizable vector lanes stand in for 4096 columns;
                # three ordered split shares exercise both direct stores and adds.
                for r,slot in enumerate(ent):
                    key = slot if m == 32 else r
                    value = [0]*16
                    for part in range(3):
                        value = [v + ((r+1)*31 + h*7 + part) for h,v in enumerate(value)]
                    y[key] = value
                for t in range(m):
                    for k in range(8):
                        slot = t*8+k
                        if slot not in rmap: continue
                        r = rmap[slot]; key = slot if m == 32 else r
                        assert y[key] == [sum((r+1)*31+h*7+p for p in range(3)) for h in range(16)]
                        y[key] = [0]*16
                assert all(not any(v) for v in y.values())
                if m == 32: assert max(ent,default=0) < 256
                layouts += 1
    # Exact B epilogue ownership for every dispatched N. Remapping changes only
    # the row base: 4 WG x 4 warp x 32 lane still uniquely covers N x 256 columns.
    for n in range(8,65,8):
        covered = collections.Counter()
        for q in range(4):
            for wp in range(4):
                for lane in range(32):
                    tg = (q//2)*8 + (q&1)*4 + wp
                    h0 = 16*tg + lane//4
                    for nt in range(n//8):
                        for hh in range(2):
                            row = 8*nt + 2*(lane&3) + hh
                            for h in (h0,h0+8): covered[row,h] += 1
        assert len(covered) == n*256 and set(covered.values()) == {1}
    # vm_fin reset/combine ownership; no CTA exits are introduced.
    for end in (80,80*16,2*136*4):
        seen = [i for tid in range(256) for i in range(tid,end,256)]
        assert sorted(seen) == list(range(end))
    cols = [1024*m+4*tid+v for tid in range(256) for m in range(4) for v in range(4)]
    assert sorted(cols) == list(range(4096))
    return {"status":"PASS", "masks":256, "routing_layouts":layouts, "B_tile_coverage":"N8..64 x 256, exactly once",
            "cleanup":"CTA0 tid0..255 clears xready[80], doneA[1280], cnt[1088]; VM_DP=0. CTA t tid0..255 clears each live y row and writes all 4096 output columns.",
            "sync":"Existing route barriers publish slot32; all CTAs/threads retained. Only metadata barrier is bypassed at M32 because metadata is warp-private. RMS s_ss barriers remain unchanged."}


def cpu_gates(base, out):
    assert "\r" not in out and len(out.encode()) < 1_000_000
    compile(out, str(OUT), "exec")
    compile(Path(__file__).read_text(), __file__, "exec")
    before, after = source_namespace(base), source_namespace(out)
    assert before["_VML_DEFS"] == after["_VML_DEFS"]
    assert before["_vml_FLAGS"] == after["_vml_FLAGS"]
    assert before["_vml_CPP"] == after["_vml_CPP"]
    # Arithmetic, stage consumer, producer protocol and all self tests are kept.
    for start,end in (("__device__ void prod_w(","__global__ void __launch_bounds__(NTH, 1) vm_main("),):
        b = before["_vml_CUDA"]; a = after["_vml_CUDA"]
        # Exclude only the two explicitly changed B epilogues from this span.
        b = b[b.index(start):b.index(end)]
        a = a[a.index(start):a.index(end)]
        for marker in ("__device__ void prod_w(", "__device__ void prod_x(", "void stage_m("):
            assert b.count(marker) == a.count(marker)
    tree_b,tree_a = ast.parse(base),ast.parse(out)
    new_funcs = {"_vml32_build","_vml32_prepare"}
    changed_funcs = {"_vml_route","_vml_prepare","_vml_build_job"}
    funcs_b = {n.name:ast.dump(n,include_attributes=False) for n in tree_b.body if isinstance(n,ast.FunctionDef)}
    funcs_a = {n.name:ast.dump(n,include_attributes=False) for n in tree_a.body if isinstance(n,ast.FunctionDef)}
    assert set(funcs_a)-set(funcs_b) == new_funcs
    for n in funcs_b:
        if n not in changed_funcs: assert funcs_b[n] == funcs_a[n], n
    assert out.count("__trap()") == base.count("__trap()")
    # Enumerate actual modifications, including insertions, for the lead.
    changes = []
    bl,ol = base.splitlines(),out.splitlines()
    for tag,i,j,k,l in difflib.SequenceMatcher(a=bl,b=ol,autojunk=False).get_opcodes():
        if tag != "equal":
            changes.append({"bit":1,"base_lines":[i+1,j],"v1_lines":[k+1,l],"operation":tag,
                            "first_changed_line":ol[k] if k<l else "<removed>"})
    active = build(activate=True)[2]
    active_check = dry_run(active)
    assert active_check["status"] == "PASS"
    return {"status":"PASS", "bytes":len(out.encode()), "default_on":int(re.search(r'^VML32_ON=(\d+)$',out,re.M)[1]),
            "dry_run":dry_run(out), "active_emission_dry_run":active_check["status"], "layout":layout_gate(), "changes":changes,
            "weight_stage_bytes":{"before":36864,"after":36864},
            "accounting":{
                "B_epilogue_per_thread_N":{"before":"N/4 row bases; N/2 FP32 writes/reductions (2*N bytes)",
                                             "after":"same writes/reductions; N/4 LDS.U16 + N/4 row-base address operations; 512 B extra Tab/CTA"},
                "route_per_live_row":"one extra 2 B shared store per CTA; at most 512 B/CTA",
                "finalizer_per_token":"y read+reset: 32768*nloc B unchanged; nw reads 16384 B unchanged; output 8192 B unchanged",
                "metadata_per_token":"before 32 B rmap + 32 B wts; after 8 warps*(32 B ids+32 B wts)=512 B, +448 B logical L2 loads; identical cache-line footprint for each broadcast",
                "finalizer_instructions":"before one post-wait rmap LDG, ballot, rank/popcount, shared stores, CTA barrier + shared s_r/s_w reads; after pre-wait id LDG, ballot, rank/popcount, REDUX.OR; post-wait 3-bit field extracts + SHFL weight. Actual SASS reported by Linux gate; not a speed measurement.",
                "reset_ownership":"unchanged, no early returns; see layout.cleanup"},
            "limits":["CPU doubles do not prove CUDA numeric correctness", "no H200 timing performed", "estimated metadata delay is not measured in isolation"]}


def resources(log):
    result = {}
    pattern = r"ptxas info\s*: Compiling entry function '([^']+)'[^\n]*\n(.*?)(?=ptxas info\s*: Compiling entry function|\Z)"
    for symbol,body in re.findall(pattern,log,re.S):
        regs = re.search(r"Used (\d+) registers",body)
        spills = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",body)
        assert regs and spills, (symbol,body)
        result[symbol] = dict(registers=int(regs[1]),stack=int(spills[1]),spill_stores=int(spills[2]),spill_loads=int(spills[3]))
    assert result, "no ptxas resource entries"
    return result


def compile_gates(base, out, result):
    # Invoked only by the explicit Linux CPU/compile script. No module loading,
    # cuda device queries, inference, timing, or GPU context initialization.
    assert sys.platform == "linux", "Run the supplied script under cudaenv.sh on Linux"
    work = Path(tempfile.mkdtemp(prefix="s7f_vml32_"))  # retained, never deleted
    result["artifacts"] = str(work)
    result["commands"] = []
    def command(cmd,name):
        result["commands"].append(cmd)
        p = subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=900)
        (work/(name+".log")).write_bytes(p.stdout.encode())
        assert p.returncode == 0, f"{name}: {p.stdout[-5000:]}"
        return p.stdout
    nvcc,cobj,cxx = shutil.which("nvcc"),shutil.which("cuobjdump"),shutil.which("c++")
    assert nvcc and cobj and cxx, "nvcc/cuobjdump/c++ must be on PATH under cudaenv.sh"
    import torch
    from torch.utils.cpp_extension import include_paths
    common = ["-std=c++20", "-DTORCH_EXTENSION_NAME=s7f_vml32_gate", "-D_GLIBCXX_USE_CXX11_ABI="+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
              *["-I"+p for p in include_paths(device_type="cuda")], "-I"+sysconfig.get_paths()["include"]]
    bn,an = source_namespace(base),source_namespace(out)
    cpp = '#include <torch/extension.h>\n'+an["_vml_CPP"]+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("vml_run",&vml_run);m.def("vml_cfg",&vml_cfg);}\n'
    (work/"binding.cpp").write_bytes(cpp.encode())
    command([cxx,*common,"-O2","-fPIC","-c",str(work/"binding.cpp"),"-o",str(work/"binding.o")],"binding")
    result["cuda"] = {}
    for name,ns,extra in (("base_bf2",bn,()),("off_bf2",an,()),("slot_bf2",an,("-DVM_SLOT32=1",))):
        cu,obj = work/(name+".cu"),work/(name+".o")
        cu.write_bytes(ns["_vml_CUDA"].encode())
        log = command([nvcc,*common,*ns["_vml_FLAGS"],*ns["_VML_DEFS"]["bf2"],*extra,"-Xptxas","-v","-c",str(cu),"-o",str(obj)],name+"_compile")
        rr = resources(log)
        sass = command([cobj,"--dump-sass",str(obj)],name+"_sass")
        command([cobj,"--dump-resource-usage",str(obj)],name+"_resources")
        chunks = re.split(r"Function\s*:\s*(\S+)\s*\n",sass)
        funcs = {}
        for symbol,body in zip(chunks[1::2],chunks[2::2]):
            inst = [re.sub(r"/\*.*?\*/","",line).strip() for line in body.splitlines() if re.match(r"\s*/\*[0-9a-fA-F]+\*/\s+",line) and ";" in line]
            ops = collections.Counter(re.sub(r"^@!?P\d+\s+","",line).split()[0] for line in inst)
            funcs[symbol] = {"instructions":inst,"static_count":len(inst),"opcodes":dict(sorted(ops.items()))}
        result["cuda"][name] = {"resources":rr,"functions":funcs}
    bb,off,on = (result["cuda"][n] for n in ("base_bf2","off_bf2","slot_bf2"))
    assert bb == off, "default-off SASS/resources differ from the base"
    assert set(on["resources"]) == set(bb["resources"])
    for sym,res in on["resources"].items():
        assert res["spill_stores"] == res["spill_loads"] == 0, f"ZERO-SPILL GATE FAIL {sym}: {res}; baseline {bb['resources'][sym]}"
    main = next(s for s in on["functions"] if "vm_main" in s)
    fin = next(s for s in on["functions"] if "vm_fin" in s)
    def op_count(build,sym,prefix):
        return sum(v for k,v in build["functions"][sym]["opcodes"].items() if k.startswith(prefix))
    assert op_count(on,main,"HGMMA") == op_count(bb,main,"HGMMA")
    assert op_count(on,main,"HMMA") == op_count(bb,main,"HMMA")
    assert op_count(on,fin,"REDUX") > op_count(bb,fin,"REDUX"), "slot compaction missing in SASS"
    result["compile_status"] = "PASS"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activate",action="store_true",help="S7 active form; default emitted selector is off")
    ap.add_argument("--compile",action="store_true",help="Linux nvcc/SASS only, never device execution")
    args = ap.parse_args()
    result = {"status":"FAIL", "compile_status":"NOT_RUN", "gpu":"NOT_RUN", "version":1}
    try:
        raw,base,out = build(args.activate)
        result["base_sha256"] = hashlib.sha256(raw).hexdigest()
        result["output_sha256"] = hashlib.sha256(out.encode()).hexdigest()
        result["cpu"] = cpu_gates(base,out)
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_bytes(out.encode())
        assert BASE.read_bytes() == raw, "base changed during build"
        if args.compile:
            compile_gates(base,out,result)
            result["status"] = "CPU_COMPILE_SASS_PASS"
        else:
            result["status"] = "CPU_PASS_CUDA_PENDING"
    except Exception as ex:
        result["error"] = str(ex)
        result["traceback"] = traceback.format_exc()
    finally:
        RESULT.write_bytes((json.dumps(result,indent=2)+"\n").encode())
        print(json.dumps({k:result[k] for k in ("status","compile_status","gpu")},indent=2))
        print(f"result: {RESULT}\noutput: {OUT}")
        if "error" in result: print(result["traceback"],file=sys.stderr)
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

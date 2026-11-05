#!/usr/bin/env bash
""":"
# The task-named generator is also the self-contained Linux gate.
set -euo pipefail
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
exec python -I "$0" --linux-gate "$@"
":"""
from __future__ import annotations

"""S7-RT: three isolated route/instantiation probes, never a GPU runner.

Usage: python -I tools/s7_rt_patches.py --generate | --check | --map | --report
Linux: bash tools/local/cudaenv.sh bash tools/s7_rt_patches.py
Linux writes tools/s7_rt_patches.json and retains compiler products in /tmp.
Only --generate writes the task-named box_out/s7/k3_rt_v1..v3.py files.

Kills FIRST (these are requested test builds, NOT measured speedup claims):
v1: extra registers may spill / lose occupancy; the stream may already hide
    the load latency. Pooler/1 ST=4, keeping XP=7 and all math/packing intact.
v2: grouping can spill A fragments; current wgmma waits may already overlap.
    RQ128 GS=2 ST=6, MS=1, identical flags/splits/math. ST6 is required to
    avoid a ldr(G+1)-before-xrel(G-1) ring deadlock; ST5/GS2 is NOT emitted.
v3: the alternate Lt config can be unsupported; dirty writeback/tensor work
    can erase a layout win. Public TN/batch2 cat256 pin from
    notes/study/unpooler_lt.md:33. Keep FP32 cat and the real Lt self-tests.

No unsupported positive latency claim: expected us for v2/v3 is unknown.
v1's additional pre-wait bytes provide ONLY a gross bound of 1.356 us,
not a measured exposed saving. The judge must decide whether any clears
0.5 us/kernel. DNR R-rat-1, R-cbl-1, R-hybsync-1 remain contrary evidence.

The entire original executable AST, including every self-test and fallback,
is an unchanged prefix. An EOF selector is off in ARM_TEMPLATE; emitted
files explicitly opt into exactly one arm. No new prepare bank or accessor.
CPU tests execute the base's actual GD/RQ/Lt prepare and dispatch functions
against the base's actual dense/router/wcat state-construction expressions.
Only allocation, packing numerics, numerical comparisons, and CUDA calls
are mocked, explicitly. They do not certify GPU correctness or prepare time.

Helper code below is copied from the existing S7-DN CPU gates; no dependency
on that mutable work-in-progress file remains. Its Lt role mapping is fixed
here using the trace-backed notes/study/legit/astra_dense_t128_out.md:33.
Padz convention retained: default-off selector; existing validated bank;
no guessed k2a state handle or module monkeypatch in generated submissions.
"""

import argparse
import ast
import collections
import contextlib
import copy
import difflib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
BASE_SHA256 = "07eb81e0a91a5161e23e0f794abe1f536788ff4200326c209c484802092c6e10"
PROFILE = ROOT / "notes/study/evo/s7/kernel_baseline.md"
OUTS = {v: ROOT / f"box_out/s7/k3_rt_v{v}.py" for v in (1, 2, 3)}
MARKER = "# S7-RT begin: route-only, default-off template.\n"
ARM_TEMPLATE = '''
# S7-RT begin: route-only, default-off template.
S7_RT_VARIANT=0
if S7_RT_VARIANT not in (0,1,2,3):
 raise ValueError("S7_RT_VARIANT must be 0, 1, 2, or 3")
if S7_RT_VARIANT==1:
 GD_ROUTE[("pooler",1)]=(1,1,8,16,4,1,7) # bit 1: ST2 -> ST4
elif S7_RT_VARIANT==2:
 _rq_variants[0]=("rt128g2s6",128,6,2,0,0,0,42999939) # bit 2: GS1/ST5 -> GS2/ST6
 _rq_picks[128]=(0,"rt128g2s6") # bit 2: matching prepare name
elif S7_RT_VARIANT==3:
 LT_PIN[("cat",256)]=(66,183,35,1,0,0,0,0,2) # bit 4: public cat256 TN/batch2 pin
 ULP_LAYOUT[("cat",256)]=(0,2) # bit 4: output-feature batch; never K-split
# S7-RT end
'''
TARGETS = ((1, "pooler", 1),)


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def base_text():
    data = BASE.read_bytes()
    require(hashlib.sha256(data).hexdigest() == BASE_SHA256, "immutable base SHA256 mismatch")
    require(b"\r" not in data, "base is not LF")
    return data.decode("utf-8")


def build_text(src, variant):
    require(variant in (0,1,2,3) and MARKER not in src, "invalid/already patched source")
    require(src.count('\n  return None # No host sync, logging, lazy build, or state mutation in this path.\n') == 1,
            "RQ route anchor changed")
    require(src.count('GD_ROUTE={("qkv",1):') == 1 and src.count("ULP_LAYOUT={('qkv',128):") == 1,
            "route anchors changed")
    lines = src.splitlines(keepends=True)
    require(lines[1].startswith("# nonce "), "base nonce anchor")
    lines[1] = f"# nonce rt-v{variant} (S7-RT route-only arm {variant}; self-tests retained)\n"
    return "".join(lines).rstrip("\n") + "\n" + ARM_TEMPLATE.replace("S7_RT_VARIANT=0", f"S7_RT_VARIANT={variant}")

def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def node(tree, name):
    found = [n for n in tree.body if
             isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    require(len(found) == 1, f"definition count for {name}: {len(found)}")
    return found[0]


def segment(src, name):
    return ast.get_source_segment(src, node(ast.parse(src), name))


def literal(src, name):
    return ast.literal_eval(node(ast.parse(src), name).value)


NAMES = (
    "HIDDEN", "D_EXPERT", "N_EXPERTS", "UNPOOL_SLICE", "SLICED_UNPOOL_MAX_TOKENS",
    "GD_SMS", "GD_ROUTE", "GD_CAT_ROUTE", "GD_CAT_UNITS", "GD_DIMS", "GD_WEIGHT",
    "GD_ENABLE", "GD_BUILD", "GD_REQUIRE", "GD_PREPARE_BY_S", "GD_ONE_WAVE",
    "GD_CAT", "GD_CAT_SELF_TEST_T", "GD_SELF_TEST_T", "GD_SELF_TEST_PDL",
    "GD_CPP_SRC", "GD_CUDA_SRC", "_GD", "NB64_BUILD", "NB64_USE", "NB64_GD_B",
    "_gd_nk", "_gd_key", "_gd_cat_key", "_gd_cat_keys", "_gd_keys", "_gd_bucket",
    "_gd_prepare", "_gd_cat_prepare", "_gd_live", "_gd_cat_live", "_gd_require",
)


def cpp_key(key):
    return tuple(key[:9]) + (0, key[9] if len(key) == 10 else 0)


def target_key(ns, role, t):
    return (ns["_gd_cat_key"](ns["GD_CAT_ROUTE"][t]) if role == "cat" else
            ns["_gd_key"](*ns["_gd_nk"](role, t), ns["GD_ROUTE"][(role, t)]))


def capture_build(ns, src):
    """Execute the production builder twice, mocking ONLY its compiler import."""
    builder = copy.deepcopy(node(ast.parse(src), "_gd_build"))
    imports = [n for n in builder.body if isinstance(n, ast.ImportFrom)]
    require(len(imports) == 1 and imports[0].module == "torch.utils.cpp_extension" and
            [a.name for a in imports[0].names] == ["load_inline"], "builder import changed")
    builder.body = [n for n in builder.body if not isinstance(n, ast.ImportFrom)]
    captured = []
    module = object()
    def load_inline(**kw):
        captured.append(kw)
        return module
    ns.update(load_inline=load_inline, _pt1_wait=lambda _: None)
    exec(compile(ast.Module(body=[builder], type_ignores=[]), str(BASE), "exec"), ns)
    require(ns["_gd_build"]() is module, "builder result")
    require(ns["_gd_build"]() is module and len(captured) == 1, "builder cache")
    require("@INST@" not in captured[0]["cuda_sources"] and "@INFO@" not in captured[0]["cuda_sources"], "unexpanded source")
    return captured[0]


class TensorShape:
    """Strict shape fixture. It performs no numerical verification."""
    def __init__(self, shape, dtype="bf16", label="fixture"):
        self.shape, self.dtype, self.device, self.label = tuple(shape), dtype, "cpu", label
    def contiguous(self):
        return self
    def dim(self):
        return len(self.shape)
    def numel(self):
        return math.prod(self.shape)
    def __getitem__(self, item):
        require(isinstance(item, slice), f"unsupported fixture indexing {item}")
        start, stop, step = item.indices(self.shape[0])
        return TensorShape((len(range(start, stop, step)),) + self.shape[1:], self.dtype, self.label)
    def view(self, dtype):
        require(dtype == "i16", "unexpected view in dry-run")
        return TensorShape(self.shape, dtype, self.label)


def prepare_dry_run(src, ns):
    """Run both REAL prepare functions, including bank creation and live lookup.

    Tensor allocation, packing arithmetic, CUDA, occupancy and numerical tests
    are explicit stubs. No arbitrary attribute/key fallback is allowed. The
    dense/router/wcat fixture expressions come from the real state producers.
    A dropped route or caught exception is an unconditional CPU gate failure.
    """
    tree = ast.parse(src)
    state_node = next(n.value for n in node(tree, "_r1_prepare").body if
                      isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "state" for t in n.targets))
    state_expr = ast.Dict(keys=[], values=[])
    for key, value in zip(state_node.keys, state_node.values):
        if isinstance(key, ast.Constant) and key.value in ("dense", "router"):
            state_expr.keys.append(key)
            state_expr.values.append(value)
    require(len(state_expr.keys) == 2, "real base dense/router layout changed")
    shapes = {"qkv_gate": (2304,8192), "o": (8192,1024), "pooler": (4096,8192),
              "unpooler": (8192,4096), "router": (256,8192)}
    weights = {k: TensorShape(s, "f32" if k == "router" else "bf16", k) for k,s in shapes.items()}
    def cat(ts, dim=0):
        shape = list(ts[0].shape)
        require(all(len(t.shape) == len(shape) for t in ts), "cat dimensions")
        shape[dim] = sum(t.shape[dim] for t in ts)
        return TensorShape(shape, ts[0].dtype, "cat")
    cuda = SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(multi_processor_count=132),
                           synchronize=lambda: None, empty_cache=lambda: None)
    torch = SimpleNamespace(bfloat16="bf16", int16="i16", cuda=cuda, cat=cat,
                            empty_like=lambda t: TensorShape(t.shape,t.dtype,t.label),
                            equal=lambda a,b: a.shape == b.shape and a.dtype == b.dtype)
    state = eval(compile(ast.fix_missing_locations(ast.Expression(state_expr)), str(BASE), "eval"),
                 {"weights": weights, "router": weights["router"]})
    v8_node = next(n.value for n in node(tree, "_v8_tail_prepare").body if
                   isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "v8" for t in n.targets))
    wcat = next(v for k,v in zip(v8_node.keys,v8_node.values) if isinstance(k,ast.Constant) and k.value == "wcat")
    state["v8"] = {"wcat": eval(compile(ast.Expression(wcat), str(BASE), "eval"),
                                  {"torch":torch,"weights":weights,
                                   "hi":TensorShape((256,8192)),"lo":TensorShape((256,8192))})}
    keys = ns["_gd_keys"]()
    tests, packs = [], []
    class Module:
        def gd_info(self, i):
            key = keys[i]
            return (64, 0, key[7]) # MOCK: resource reality belongs to the Linux gate.
        def gd_unpack(self, *args):
            require(len(args) == 10, "gd_unpack ABI")
        def gd_unpack_cat(self, *args):
            require(len(args) == 9, "gd_unpack_cat ABI")
    mod = Module()
    def pack(w,wr,wk,ku,s,base):
        n,k = w.shape
        nbx, G = n//16//wr, n//16*s*wk
        require(k//64 == wk*ku*s, "pack geometry")
        packs.append((w.label,n,wr,wk,ku,s))
        return dict(w=TensorShape((n*k*3//8,),"i32"),esc=TensorShape((G*32,),"i32"),
                    tb=base*0x01010101,nbx=nbx,G=G,nesc=0)
    def cat_pack(wp,wv,wk,kup,bp,bh,bl):
        require(wp.shape == (4096,8192) and wv.shape == (512,8192), "real CAT layout")
        require(wk*kup == 128 and wk*ns["GD_CAT_UNITS"] == 16, "CAT pack geometry")
        packs.append(("cat",wk,kup))
        G = 256*wk
        return dict(w=TensorShape((4608*8192*3//8,),"i32"),esc=TensorShape((G*32,),"i32"),
                    tb=bp*0x01010101,tbr=bh|(bh<<8)|(bl<<16)|(bl<<24),nbx=256,G=G,nesc=0,ku=kup+2)
    def test_gd(r,w,b):
        require(r["mod"] is mod and r["N"] == w.shape[0] and r["K"] == w.shape[1], "GD route state")
        require(all(t <= r["T"] for t in ns["GD_SELF_TEST_T"][b]), "GD selftest coverage")
        tests.append(("gd",b,r["inst"]))
        return True,"MOCK numerical test; host layout only"
    def test_cat(r,wp,hi,lo,w32,b):
        require(r["mod"] is mod and wp.shape==(4096,8192) and hi.shape==lo.shape==w32.shape==(256,8192), "CAT route state")
        require(all(t <= r["T"] for t in ns["GD_CAT_SELF_TEST_T"][b]), "CAT selftest coverage")
        tests.append(("cat",b,r["inst"]))
        return True,"MOCK numerical test; host layout only"
    ns.update(torch=torch, _gd_build=lambda: mod, _gd_pack=pack, _gd_cat_pack=cat_pack,
              _gd_window_base=lambda w: 54, _gd_selftest=test_gd, _gd_cat_selftest=test_cat,
              _gd_cat_virtual=lambda hi,lo: cat((hi,lo)))
    ns["_GD"].update(mod=mod,keys=keys,build_s=0)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        ns["_gd_prepare"](state,time.time())
        ns["_gd_cat_prepare"](state,time.time())
    require("FALLBACK" not in captured.getvalue() and "failed" not in captured.getvalue().lower(), captured.getvalue())
    require(set(state["gd"]["route"]) == set(ns["GD_ROUTE"]), "missing GD route")
    require(set(state["gd_cat"]) == set(ns["GD_CAT_ROUTE"]), "missing CAT route")
    selected = []
    for t in range(1,65):
        for role in ("qkv","o","pooler","unpooler"):
            r = ns["_gd_live"](state,role,t)
            b = ns["_gd_bucket"](t)
            wanted = b != 64 and (role,b) in ns["GD_ROUTE"]
            require((r is not None) == wanted, f"unexpected live GD {role}/{t}")
        r = ns["_gd_cat_live"](state,t)
        require((r is not None) == (9<=t<=32), f"unexpected live CAT/{t}")
    for bit,role,t in TARGETS:
        r = ns["_gd_cat_live"](state,t) if role=="cat" else ns["_gd_live"](state,role,t)
        require(keys[r["inst"]] == target_key(ns,role,t), "wrong selected instantiation")
        selected.append(dict(bit=bit,role=role,T=t,key=cpp_key(keys[r["inst"]])))
    return dict(status="PASS", selected=selected, prepared_routes=len(tests),
                packs=len(packs), lookups=64*5, numerical_tests="NOT RUN (explicit CPU mocks)",
                state_layout_sources=["_r1_prepare:11628", "_v8_tail_prepare:1542"])


def ring_gate(key):
    """Symbolic ring execution proves every source weight is used exactly once."""
    ku, st = key[4], key[5]
    ring = list(range(st))
    loads, consumed = list(range(st)), []
    for s in range(ku):
        require(ring[s % st] == s, f"stale/uninitialized ring at {s}: {key}")
        consumed.append(ring[s % st]) # decode copies into a[16] before refill
        if s+st < ku:
            ring[s % st] = s+st
            loads.append(s+st)
    require(loads == consumed == list(range(ku)), "weight sequence changed")
    return dict(KU=ku,ST=st,weight_vector_loads_per_thread=3*ku,
                prologue_vector_loads=3*st,refill_vector_loads=3*(ku-st))


def baseline_rows():
    """Map profile medians to exact source routes; never split aggregated RQ time."""
    src=base_text()
    ns=route_namespace(src)
    tree=ast.parse(src)
    profile={}
    shape=None
    for line_no,line in enumerate(PROFILE.read_text(encoding="utf-8").splitlines(),1):
        h=re.match(r"## (\w+):",line)
        if h:
            shape=h[1]
            profile[shape]=[]
        m=re.match(r"- (.*): ([\d.]+) \[([\d.]+)-([\d.]+)\] calls (\d+)",line)
        if m:
            profile[shape].append(dict(label=m[1],measured_us=float(m[2]),p10=float(m[3]),p90=float(m[4]),
                                       calls=int(m[5]),measurement_source=f"{PROFILE.relative_to(ROOT).as_posix()}:{line_no}"))
    rows=[]
    for shape,items in profile.items():
        t=int(shape[1:-1])*(8 if shape[0]=="v" else 1)
        for p in items:
            label=p["label"]
            role,route,packed,esc,formula,code="",None,0,0,"",""
            if "gd::gd_kernel<" in label:
                key=tuple(map(int,re.search(r"gd_kernel<([^>]+)>",label)[1].split(",")))
                if key[10]:
                    role="pooler+router"
                    route=ns["GD_CAT_ROUTE"][t]
                    require(key==cpp_key(ns["_gd_cat_key"](route)),"profile CAT mismatch")
                    packed=(4096+2*256)*8192*3//2
                    esc=256*8*32*4
                    code=f"GD_CAT_ROUTE:{node(tree,'GD_CAT_ROUTE').lineno}; _gd_cat:16727; kernel_fn:22591"
                    formula="(4096+2*256)*8192*1.5 + (256*8)*128"
                elif key[9]:
                    require(t==8 and key[9]==2,"unexpected VF")
                    role="pooler+FP32 router"
                    route="gdr_run_raw VF=2 (GD pooler/8 descriptor)"
                    packed=4096*8192*3//2+256*8192*4
                    esc=256*8*128
                    formula="4096*8192*1.5 + 256*8192*4 + (256*8)*128"
                    code="gdr_run_raw:16133; _vfr_fused:16187; VF_GDR:16169"
                else:
                    candidates=[proj for proj in ("qkv","o","pooler","unpooler") if
                                (proj,t) in ns["GD_ROUTE"] and
                                cpp_key(ns["_gd_key"](*ns["_gd_nk"](proj,t),ns["GD_ROUTE"][(proj,t)]))==key]
                    require(len(candidates)==1,f"ambiguous GD role {shape} {key}")
                    role=candidates[0]
                    route=ns["GD_ROUTE"][(role,t)]
                    n,k=ns["_gd_nk"](role,t)
                    packed=n*k*3//2
                    esc=(n//16*route[2]*route[5])*128
                    formula=f"{n}*{k}*1.5 + ({n}//16*{route[2]}*{route[5]})*128"
                    code=f"GD_ROUTE:{node(tree,'GD_ROUTE').lineno}; _gd_live:16475; kernel_fn:22525"
                inst="gd_kernel<"+",".join(map(str,key))+">"
            elif "vf_router32_kernel<" in label:
                role,route,packed="router","VF_ROUTER[1]='lean', kind=0 / V8_ROUTER_PLACE[1]='side'",256*8192*4
                inst="vf_router32_kernel<1>"
                formula="256*8192*4"
                code="VF_ROUTER:15280; _v8_logits:925; vf_router32_launch:15273; kernel_fn:22525"
            elif "sk2_kernel<" in label:
                role="unpooler" if "2, 32, 4, 2, 2, 2, 3" in label else "qkv"
                name=eval(compile(ast.Expression(node(tree,"_SK2_PROJ").value),str(BASE),"eval"),ns)[role][0]
                cfg=literal(src,"_SK2_CFGS")[name]
                route=name
                n,k=ns["GD_DIMS"][role]
                packed=n*k*3//2
                esc=None # pmax is data-dependent, not in the aggregate profile.
                formula=f"{n}*{k}*1.5; escape table adds ({n}//{cfg[3]})*pmax*8"
                args="2,32,4,2,2,2,3" if role=="unpooler" else "2,32,3,1,4,2,4"
                inst=f"sk2_kernel<{args},true,false,1,1,__nv_bfloat16,false,false>"
                code=f"sk2_dispatch:{21100 if role=='unpooler' else 21099}; _SK2_CFGS:21114; _sk2:21209"
            elif "dg1::dg1_kernel<" in label:
                require(t==32,"unexpected DG1")
                role,route,packed="o",literal(src,"DG1_GEOMETRY")[("o",32)],8192*1024*2
                inst="dg1::dg1_kernel<32,1,1>"
                formula="8192*1024*2 (raw BF16; dg1_pack is permutation only)"
                code="DG1_GEOMETRY:13745; _dg1:14727; kernel_fn:22576"
            elif "rq_kernel<" in label:
                require(t==128 and p["calls"]==2,"unexpected RQ aggregate")
                role,route="qkv+unpooler (two calls, aggregate time)","RQ_SEL; d2s5hzwP; KS=7/2"
                packed=(2304*8192+8192*4096)*3//2
                esc=4*((7*18+2*64)*8*32*4)
                formula="(2304*8192+8192*4096)*1.5 + 4*((7*18+2*64)*8*32*4); selected CUR table, not the unused variable escape list"
                inst="rq_kernel<128,5,1,0,0,0,42999939,1>"
                code="RQ_SEL:21273; _rq_ks:21276; _rq_prepare:21353; kernel_fn:22533,22638"
            elif "vd::vd_kernel<" in label:
                role,route,packed="pooler+router",literal(src,"VD_INSTS")[0],4608*8192*2
                inst="vd::vd_kernel<2,128,5,3,1,1>"
                formula="(4096+2*256)*8192*2 (raw BF16, includes hi+lo router)"
                code="VD_ROUTE:14158; _vd:14705; _dg1:14732"
            elif "nvjet" in label or "cublasLt" in label:
                if t==128:
                    role="o"
                elif "tss_72x128" in label:
                    role="pooler+router"
                elif "coopA" in label:
                    role="unpooler"
                elif "tst_72x64" in label:
                    role="qkv"
                elif "2x1_v_bz_TNT" in label:
                    role="o"
                else:
                    raise AssertionError(f"unknown Lt kernel {shape} {label}")
                proj="cat" if role=="pooler+router" else role
                route=literal(src,"LT_PIN").get((proj,t),"torch.mm default (no o256 pin)")
                n,k=(4608,8192) if proj=="cat" else ns["GD_DIMS"][proj]
                packed=n*k*2
                formula=f"{n}*{k}*2 raw BF16; nominal unique payload, tile reloads/cache excluded"
                inst=label
                code="LT_PIN:22669; _dg1:14729; _lt:22865; kernel_fn:22525"
            else:
                continue
            weight_bytes=packed+(esc or 0)
            rows.append(dict(shape=shape,T=t,role=role,instance=inst,route=route,**p,
                             weight_bytes_EST=weight_bytes,byte_formula_EST=formula,
                             unknown_escape_bytes=esc is None,
                             effective_TB_s_EST=round(weight_bytes/p["measured_us"]/1e6,4),
                             bandwidth_note="payload / profile duration, NOT measured HBM throughput; includes PDL wait/overlap",
                             source_references=[f"{BASE.relative_to(ROOT).as_posix()}:{line} ({symbol})"
                                                for symbol,lines in re.findall(r"([A-Za-z_][A-Za-z_0-9]*):([\d,]+)",code)
                                                for line in lines.split(",")]))
    require(len(rows)==48 and sum(r["calls"] for r in rows)==50,
            f"dense mapping incomplete: {len(rows)} rows")
    return rows


def route_report():
    """Complete requested route inventory plus exact scored tail instances."""
    src=base_text(); tree=ast.parse(src); ns=route_namespace(src)
    names=("GD_ROUTE","GD_CAT_ROUTE","DG1_GEOMETRY","LT_PIN","ULP_LAYOUT",
           "RQ_SEL","_rq_picks","_rq_ks","_rq_variants","_rq_ms",
           "VD_INSTS","VD_ROUTE","VD_TT","VD_TT_PREP","VD_PDL","VD_TRIG","VD_PF","VD_ROT","VD_FLAGS",
           "_SK2_CFGS","_SK2_K","_SK2_PROJ","SK2_ON","SK2_PDL","SK2_T",
           "VF_GDR","FT_TAIL3","V8_TAIL_ROUTE")
    tables=[]
    for name in names:
        n=node(tree,name)
        value=eval(compile(ast.Expression(n.value),str(BASE),"eval"),ns)
        # Tuple keys are preserved as readable pairs, not silently stringified.
        if isinstance(value,dict): value=[dict(key=k,value=v) for k,v in value.items()]
        if isinstance(value,(set,frozenset)): value=sorted(value,key=repr)
        tables.append(dict(name=name,value=value,source=f"{BASE.relative_to(ROOT).as_posix()}:{n.lineno}"))
    tails=[]; shape=None
    ft=literal(src,"FT_TAIL3")
    for lineno,line in enumerate(PROFILE.read_text(encoding="utf-8").splitlines(),1):
        h=re.match(r"## (\w+):",line)
        if h: shape=h[1]
        m=re.match(r"- .*tail_kernel3<([^>]+)>.*: ([\d.]+) \[([\d.]+)-([\d.]+)\] calls (\d+)",line)
        if not m: continue
        t=int(shape[1:-1])*(8 if shape[0]=="v" else 1)
        args=tuple(v.strip() for v in m[1].split(","))
        require(args==(("true","1","8") if t>=128 else ("false","8","8")),"tail profile instance drift")
        require(ft[t]==8,"tail table/profile mismatch")
        tails.append(dict(shape=shape,T=t,role="pooler RMSNorm + exact router top8/weights/local IDs",
            instance="v8::tail_kernel3<"+",".join(args)+">",measured_us=float(m[2]),p10=float(m[3]),p90=float(m[4]),calls=int(m[5]),
            measurement_source=f"{PROFILE.relative_to(ROOT).as_posix()}:{lineno}",
            source_references=[f"{BASE.relative_to(ROOT).as_posix()}:{ln}" for ln in (1225,1432,1480,14879,14881)],
            route="FT_TAIL3=8; FT_OK self-test; v8_tail dtype selects true/false and ns selects 1/8",
            unchanged=True,closest_DNR="TAILW (notes/DO_NOT_REPEAT.md:183)" if t>=128 else "R-rat-1"))
    require(len(tails)==12,"incomplete tail map")
    return dict(dense=baseline_rows(),tails=tails,tables=tables,
        attribution_corrections=["d1s gd<0,1,1,4,8,2,4,5,1,0,0> is qkv, not unpooler",
            "v1s pooler includes FP32 router bytes; comparison with d1s is not the same work",
            "v16 rq_kernel duration aggregates TWO calls; individual role times are unavailable",
            "v32 tss72x128=cat, tst72x64=qkv, coopA=unpooler; older R-kb2 label mapping was stale"])


def ptxas_resources(stderr):
    rows={}
    current=None
    for line in stderr.splitlines():
        m=re.search(r"Compiling entry function '([^']+)'",line)
        if m:
            current=m[1]
            rows[current]={}
        m=re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",line)
        if m and current:
            rows[current].update(zip(("stack","spill_stores","spill_loads"),map(int,m.groups())))
        m=re.search(r"Used (\d+) registers",line)
        if m and current:
            rows[current]["registers"]=int(m[1])
    return rows


EXTRA_NAMES = (
    "RQ_SEL", "RQ_PACK", "_rq_picks", "_rq_ks", "_rq_tol", "_rq_budget_s", "_rq_build_cache",
    "_rq_cpp", "_rq_cuda", "_rq_variants", "_rq_ms", "_rq_flags", "_rq_keys", "_rq_weight",
    "_rq_call", "_rq_prepare", "_rq_gemm", "_rqwg", "_rqss", "_rqsrc",
    "LT_ENABLE", "LT_PIN", "ULP_LAYOUT", "LT_WS", "LT_CPP", "_LT", "DG1_TOL",
    "_dg1_weight", "_lt_prepare", "_lt_today", "_lt",
)


def route_namespace(src):
    tree = ast.parse(src)
    ns = dict(hashlib=hashlib, time=time, _nb64=lambda t: 33 <= t <= 64)
    exec(compile(ast.Module(body=[node(tree,n) for n in NAMES+EXTRA_NAMES],type_ignores=[]),str(BASE),"exec"),ns)
    if MARKER in src:
        exec(src[src.index(MARKER):],ns)
    return ns


class TensorShape(TensorShape):
    """Strict tensor metadata only; no numerical or CUDA result is implied."""
    def __init__(self,shape,dtype="bf16",label="fixture",finite=True,difference=False):
        super().__init__((shape,) if isinstance(shape,int) else shape,dtype,label)
        self.finite,self.difference=finite,difference
    def element_size(self):
        return {"bf16":2,"i16":2,"i32":4,"i64":8,"u8":1,"f32":4,"f64":8}[self.dtype]
    def is_contiguous(self): return True
    def data_ptr(self): return 65536
    def t(self):
        require(len(self.shape)==2,"transpose requires 2-D")
        return TensorShape(self.shape[::-1],self.dtype,self.label)
    def to(self,dtype): return TensorShape(self.shape,dtype,self.label,self.finite,self.difference)
    def double(self): return self.to("f64")
    def float(self): return self.to("f32")
    def __matmul__(self,other):
        require(len(self.shape)==len(other.shape)==2 and self.shape[1]==other.shape[0],"matmul shape")
        return TensorShape((self.shape[0],other.shape[1]),self.dtype,"MOCK matmul")
    def __sub__(self,other):
        require(self.shape==other.shape,"comparison shape")
        return TensorShape(self.shape,self.dtype,"MOCK difference",difference=True)
    def norm(self): return 0.0 if self.difference else 1.0


def shape_torch():
    def alloc(shape,dtype="f32",device="cpu",**kw):
        require(device=="cpu","CPU fixture attempted another device")
        return TensorShape(shape,dtype)
    def cat(ts,dim=0):
        s=list(ts[0].shape)
        require(all(t.shape[:dim]+t.shape[dim+1:]==ts[0].shape[:dim]+ts[0].shape[dim+1:] for t in ts),"cat shape")
        s[dim]=sum(t.shape[dim] for t in ts)
        return TensorShape(s,ts[0].dtype,"cat")
    class Generator:
        def __init__(self,device): require(device=="cpu","generator device")
        def manual_seed(self,seed): return self
    def full(shape,fill,**kw):
        t=alloc(shape,**kw); t.finite=not math.isnan(fill); return t
    return SimpleNamespace(bfloat16="bf16",float32="f32",int16="i16",int32="i32",int64="i64",uint8="u8",
        empty=alloc,zeros=alloc,randn=alloc,full=full,cat=cat,Generator=Generator,
        mm=lambda a,b,**kw:(a@b).to(kw.get("out_dtype",a.dtype)),
        isfinite=lambda t:SimpleNamespace(all=lambda:t.finite),
        equal=lambda a,b:a.shape==b.shape and a.dtype==b.dtype and a.finite and b.finite,
        cuda=SimpleNamespace(synchronize=lambda:None,get_device_properties=lambda _:SimpleNamespace(multi_processor_count=132)))


def state_fixture(src,torch):
    """Evaluate the REAL state constructors, never manufacture alternate keys."""
    tree=ast.parse(src)
    r1=node(tree,"_r1_prepare")
    state=next(n.value for n in r1.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="state" for t in n.targets))
    wanted=[(k,v) for k,v in zip(state.keys,state.values) if isinstance(k,ast.Constant) and k.value in ("dense","router")]
    require(len(wanted)==2,"base state schema")
    shapes={"qkv_gate":(2304,8192),"o":(8192,1024),"pooler":(4096,8192),"unpooler":(8192,4096),"router":(256,8192)}
    weights={k:TensorShape(s,"f32" if k=="router" else "bf16",k) for k,s in shapes.items()}
    expr=ast.Expression(ast.Dict(keys=[k for k,v in wanted],values=[v for k,v in wanted]))
    result=eval(compile(ast.fix_missing_locations(expr),str(BASE),"eval"),dict(weights=weights,router=weights["router"]))
    v8=next(n.value for n in node(tree,"_v8_tail_prepare").body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="v8" for t in n.targets))
    wcat=next(v for k,v in zip(v8.keys,v8.values) if isinstance(k,ast.Constant) and k.value=="wcat")
    result["v8"]={"wcat":eval(compile(ast.Expression(wcat),str(BASE),"eval"),
        dict(torch=torch,weights=weights,hi=TensorShape((256,8192)),lo=TensorShape((256,8192))))}
    return result


def rq_build_capture(src,ns):
    builder=copy.deepcopy(node(ast.parse(src),"_rq_build"))
    class ReplaceCompiler(ast.NodeTransformer):
        def visit_ImportFrom(self,n):
            if n.module=="torch.utils.cpp_extension":
                require([(a.name,a.asname) for a in n.names]==[("load_inline","_rq_load_inline")],"RQ compiler import")
                return None
            return n
    builder=ReplaceCompiler().visit(builder)
    specs=[]; mod=object()
    def load(**kw): specs.append(kw); return mod
    ns["_rq_load_inline"]=load
    exec(compile(ast.fix_missing_locations(ast.Module(body=[builder],type_ignores=[])),str(BASE),"exec"),ns)
    require(ns["_rq_build"]() is mod and ns["_rq_build"]() is mod and len(specs)==1,"RQ build/cache")
    require("@@" not in specs[0]["cuda_sources"],"unexpanded RQ source")
    return specs[0]


def rq_dry_run(src,ns):
    torch=shape_torch(); state=state_fixture(src,torch)
    calls=[]; tests=[]; packs=[]
    class Module:
        def rq_run(self,*a):
            require(len(a)==13,"RQ ABI")
            idx,lay,x,y,esc,esc2,cnt,ks,tab,alias,prof,sink,info=a
            require(idx in (0,1) and alias==0 and ks in (2,7),"RQ selected route")
            require(cnt.shape==(8*lay.shape[0],) and cnt.dtype=="i32","RQ counter allocation")
            require(lay.shape[2:]==(8,1536) and esc2.shape==(ks*lay.shape[0]*8*32*4,),"RQ cursor layout")
            calls.append((idx,ks,info))
            return 1 if info==4 else 0 # MOCK occupancy; actual gate is on H200.
    mod=Module()
    def pack(w):
        packs.append(w.label)
        n,k=w.shape
        return TensorShape((n//128,k//64,8,1536),"u8"),54*0x01010101,TensorShape((4,6),"i64")
    def validate(p,w,t,gen):
        require(p["rq_mod"] is mod and p["rq_n"]==w.shape[0] and p["rq_k"]==w.shape[1],"RQ bank layout")
        require(p["rq_name"]==ns["_rq_variants"][p["rq_i"]][0],"RQ name/index mismatch")
        tests.append((w.label,t))
        return True,"MOCK numerics; metadata only"
    ns.update(torch=torch,_rq_build=lambda:mod,_rq_pack_weight=pack,_rq_validate=validate,
        _rqesc=lambda e,n,k,s,d:(TensorShape((s*n*8*4+32,),"i32"),),
        _rqesc2=lambda e,n,k,s,d:(TensorShape((s*n*8*32*4,),"i32"),4))
    out=io.StringIO()
    with contextlib.redirect_stdout(out): ns["_rq_prepare"](state)
    require("fallback" not in out.getvalue().lower(),out.getvalue())
    require(set(state["rq_bank"])==set(ns["_rq_keys"]()),"RQ missing prepared arm")
    for role,t in ns["_rq_keys"]():
        p=state["rq_bank"][(role,t)]
        x=TensorShape((t,p["rq_k"]))
        y=ns["_rq_gemm"](state,role,x)
        require(y is not None and y.shape==(t,p["rq_n"]),"RQ selected output")
        require(ns["_rq_gemm"](state,role,TensorShape((t-1,p["rq_k"]))) is None,"RQ non-selected shape")
    # Missing real base keys must fail the dry run, not silently look valid.
    broken=state_fixture(src,torch); del broken["dense"]["qkv_gate"]
    with contextlib.redirect_stdout(io.StringIO()): ns["_rq_prepare"](broken)
    require(("qkv",128) not in broken["rq_bank"],"missing-key negative control did not fail")
    return dict(status="PASS",keys=sorted(state["rq_bank"]),calls=calls,tests=tests,packs=packs,
                numerical_tests="NOT RUN; explicit metadata mocks",missing_key_control="PASS")


def lt_dry_run(src,ns,enumerated=False,reject=False):
    torch=shape_torch(); state=state_fixture(src,torch); plans=[]; runs=[]
    class Module:
        def lt_plan_b(self,t,n,k,f32,nn,bt,ws,count):
            role="cat" if f32 else next(p for p in ("qkv","o","unpooler") if ns["GD_DIMS"][p]==(n,k))
            require(n%bt==0 and bt>=1 and ws==ns["LT_WS"],"Lt plan layout")
            plans.append(dict(key=(role,t),shape=(t,n,k),nn=nn,batch=bt,cfg=ns["LT_PIN"][(role,t)]))
            return ((len(plans)-1)<<8) | int(enumerated)
        def lt_cfg(self,pid,ai): require(ai in (0,1),"Lt algorithm index"); return plans[pid]["cfg"]
        def lt_make(self,pid,cfg,ws):
            require(tuple(cfg)==plans[pid]["cfg"],"Lt make configuration")
            return -1 if reject else 1
        def lt_run(self,pid,ai,w,x,y,beta,ws):
            p=plans[pid]; t,n,k=p["shape"]
            require(w.shape==((k,n) if p["nn"] else (n,k)) and x.shape==(t,k) and y.shape==(t,n),"Lt launch layout")
            require(y.dtype==("f32" if p["key"][0]=="cat" else "bf16") and beta==0,"Lt output precision")
            runs.append(p["key"]); y.finite=True
    mod=Module()
    ns.update(torch=torch,_lt_build=lambda:mod,_pt1_wait=lambda _:None,_gemm=lambda x,w:x@w.t())
    out=io.StringIO()
    with contextlib.redirect_stdout(out): ns["_lt_prepare"](state)
    if reject:
        require(state["lt"] is None and not runs,"rejected Lt config did not fall back")
        return dict(status="PASS",path="unsupported config fallback",plans=plans)
    require("FALLBACK" not in out.getvalue() and "FAIL:" not in out.getvalue(),out.getvalue())
    require(state["lt"] is not None and set(state["lt"]["pins"])==set(ns["LT_PIN"]),"missing Lt pins")
    with contextlib.redirect_stdout(io.StringIO()):
        for (role,t),p in state["lt"]["pins"].items():
            y=ns["_lt"](state,role,TensorShape((t,p[3])))
            require(y is not None and y.shape==(t,p[2]),"Lt selected dispatch")
    return dict(status="PASS",path="enumerated" if enumerated else "lt_make",plans=plans,runs=len(runs),
                numerical_tests="NOT RUN; explicit metadata mocks")


def rq_protocol_gate(gs,st):
    """Symbolically execute WM0's LDS-before-release schedule and all tails."""
    require(st>=3*gs,"WM0 prefetch needs three rounds of x slots")
    tested=[]
    for kc,ks,nb in ((128,7,18),(64,2,64)):
        for split in range(ks):
            count=(split+1)*kc//ks-split*kc//ks
            held={}; loaded=[]; released=[]; issued=[]
            rounds=math.ceil(count/gs)
            def ldr(g):
                for s in range(gs*g,min(gs*(g+1),count)):
                    slot=s%st
                    require(slot not in held,"RQ would wait on an unreleased x slot")
                    held[slot]=s; loaded.append(s)
            def rel(g):
                for s in range(gs*g,min(gs*(g+1),count)):
                    require(held.pop(s%st,None)==s,"RQ release generation mismatch")
                    released.append(s)
            ldr(0)
            for g in range(rounds):
                issued.extend(range(gs*g,min(gs*(g+1),count)))
                if g+1<rounds: ldr(g+1)
                if g>0: rel(g-1)
            rel(rounds-1)
            require(not held and loaded==issued==released==list(range(count)),"RQ stage/tail coverage")
            tested.append(count)
        # Actual zeroing: threads 320..383, contiguous floor-divided token ranges.
        tokens=[]
        for split in range(ks):
            lo,hi=split*128//ks,(split+1)*128//ks
            tokens.extend((lo+e//16,e%16) for tid in range(64) for e in range(tid,(hi-lo)*16,64))
        require(len(tokens)==len(set(tokens))==128*16,"RQ zeroing coverage")
        require(nb*ks<=132,"RQ self-zero grid residency bound")
    return dict(status="PASS",KCS_tested=tested,
        coverage="ALL CTAs kept: qkv (18,7), unpooler (64,2), threads 0..383",
        producer="t256 W, t288 X; slots 0..ST-1 refill; consumers warps0..7 lane0 release each used slot",
        zero="t320..383: e=tid-320+64*j < (r1-r0)*16; all 128x128 outputs zeroed once per row block",
        reset="t320 after warp sync: publishes ready then last of KS resetters clears cnt[bx] and cnt[NB+bx]",
        combine="t0..255 retain both WGs: each stages 128x64 and TMA-reduces its half; issuer warp0/lane0 per WG",
        early_exits="Only existing producer return after its role completes; no added early exits or reset/combine changes",
        final_drain="wait_group0, then release last GS round; every K64 stage loaded, issued, released once")


def accounting(variant):
    if variant==1:
        return dict(bit=1,weight_stream_grows=False,weight_B_CTA_stage_before_after=[12288,12288],
            weight_B_thread_stage_before_after=[48,48],weight_total_before_after=[50593792,50593792],
            named_ops_thread_stage_before_after=[45,45],ops="3 LDG128 + 36 decode + 2 x-LDG128 + 4 MMA; escape loop unchanged",
            loop_weight_loads_thread_before_after=[48,48],prologue_weight_loads_thread_before_after=[6,12],
            live_ring_registers_before_after=[24,48],scratch="unchanged red[8][4][32], no new scratch",
            gross_hidden_stream_bound_us_EST=(256*8*1536*(4-2))/4.64e6,
            arithmetic_EST="6,291,456 additional early bytes / 4,640,000 B/us = 1.356 us GROSS ONLY",
            measured_us={"d1s":14.0,"d1f":13.7},source="notes/study/evo/s7/kernel_baseline.md:6,18",
            critical_path="pooler output gates MoE input; profile includes PDL pre-wait, so exposed saving is UNESTABLISHED",
            expected_us="unknown; gross bound is not a prediction",closest_DNR="R-rat-1 / prefetch >2 pre-wait stages",
            kill_reasons=["extra 24 live weight registers may lose occupancy or spill (compiled total unknown)","extra lookahead may already be hidden by stream/PDL"])
    if variant==2:
        return dict(bit=2,weight_stream_grows=False,weight_B_CTA_K64_stage_before_after=[12288,12288],
            x_B_CTA_K64_stage_before_after=[16384,16384],weight_B_round_before_after=[12288,24576],
            named_ops_math_thread_K64_before_after=[43,43],ops="3 LDS128 + 20 high-byte decode + 16 PRMT + 4 WGMMA; cursor vote/patch loop unchanged",
            group_ops="WGMMA fence/commit/wait: 3 per round, rounds 18/19->9/10 (qkv), 32->16 (unpooler); per-stage wait/arrive counts unchanged",
            smem_B_before_after=[1040+5*(28672+24),1040+6*(28672+24)],
            weights="qkv 28,311,552 + unpooler 50,331,648 B; cursor tables 516,096 + 524,288 B unchanged",
            scratch="counters 8*NB int32, sink4 int32 unchanged; +28,696 B shared ring per CTA",
            measured_us={"v16s":30.8,"v16f":30.8},source="notes/study/evo/s7/kernel_baseline.md:98,108 (TWO CALLS AGGREGATED)",
            critical_path="both GEMMs feed serial consumers; deeper grouping can hide more operand work, but exposed wait/issue cost is UNMEASURED",
            expected_us="unknown; no positive latency forecast",closest_DNR="R-hybsync-1 / R-hovl gs1w2",
            kill_reasons=["larger live fragment arrays may spill or serialize WGMMA","group overhead may already overlap"])
    return dict(bit=4,weight_stream_grows="physical vendor reloads UNKNOWN; logical payload unchanged",
        logical_weight_B_before_after=[75497472,75497472],per_stage_B="vendor implementation unavailable; no edited hot-loop source",
        instructions_thread="vendor SASS unknown; no custom CUDA instantiation changed",output_B_before_after=[4718592,4718592],
        scratch="existing 32 MiB Lt workspace, unchanged; feature batches write disjoint output columns with ld=4608",
        measured_us={"v32s":32.6,"v32f":32.4},source="notes/study/evo/s7/kernel_baseline.md:116,127; roles: notes/study/legit/astra_dense_t128_out.md:33",
        pin_source="notes/study/unpooler_lt.md:33 (TN batch2 tile183 custom0 cluster2)",
        critical_path="cat gates routing; same FP32 hi/lo math and one launch, but changed library scheduling has no measured advantage",
        expected_us="unknown; no positive latency forecast",closest_DNR="R-cbl-1; batch family distinct from old batch1 repick",
        kill_reasons=["new pin may be rejected by this cuBLAS version","same dirty L2/tensor bound may absorb scheduling changes"])


def cpu_check(src,variant):
    base=base_text(); require(src==build_text(base,variant),"not exact anchored output")
    require("\r" not in src and len(src.encode())<1_000_000,"LF/size gate")
    require(src.splitlines()[1].startswith(f"# nonce rt-v{variant} ("),"nonce line")
    tree,bt=ast.parse(src),ast.parse(base); compile(tree,"<S7-RT>","exec")
    require(ast.dump(ast.Module(body=tree.body[:len(bt.body)],type_ignores=[]))==ast.dump(bt),"original executable AST/self-tests changed")
    bn,ns=route_namespace(base),route_namespace(src)
    expect=copy.deepcopy({n:bn[n] for n in ("GD_ROUTE","_rq_variants","_rq_picks","LT_PIN","ULP_LAYOUT")})
    if variant==1: expect["GD_ROUTE"][("pooler",1)]=(1,1,8,16,4,1,7)
    if variant==2:
        expect["_rq_variants"][0]=("rt128g2s6",128,6,2,0,0,0,42999939)
        expect["_rq_picks"][128]=(0,"rt128g2s6")
    if variant==3:
        expect["LT_PIN"][("cat",256)]=(66,183,35,1,0,0,0,0,2)
        expect["ULP_LAYOUT"][("cat",256)]=(0,2)
    require(all(ns[k]==v for k,v in expect.items()),"selector changed unexpected routes")
    require(len(bn["_gd_keys"]())==len(ns["_gd_keys"]()) and len(ns["_rq_variants"])==2,"increased build instance count")
    gd_spec=capture_build(ns,src); rq_spec=rq_build_capture(src,ns)
    dry={"gd":prepare_dry_run(src,ns),"rq":rq_dry_run(src,ns),
         "lt_make":lt_dry_run(src,ns),"lt_enumerated":lt_dry_run(src,ns,True),"lt_rejected":lt_dry_run(src,ns,False,True)}
    ring=ring_gate(cpp_key(target_key(ns,"pooler",1)))
    rq_proto=rq_protocol_gate(*((2,6) if variant==2 else (1,5)))
    changes=[]
    for tag,i,j,k,l in difflib.SequenceMatcher(a=base.splitlines(),b=src.splitlines(),autojunk=False).get_opcodes():
        if tag!="equal": changes.append(dict(kind=tag,base_lines=[i+1,j],output_lines=[k+1,l],text=src.splitlines()[k:l]))
    return dict(variant=variant,status="CPU_PASS",bytes=len(src.encode()),sha256=digest(src),dry_run=dry,
        GD_ring=ring,RQ_protocol=rq_proto,changed_lines=changes,original_AST_unchanged=True,
        prepare_budget="same counts: GD instances, RQ cases, packs, self-tests, Lt plans; real elapsed prepare <300s NOT VERIFIED",
        accounting=accounting(variant) if variant else None),dict(gd=gd_spec,rq=rq_spec,lt=ns["LT_CPP"])


def sass_rows(text):
    rows={}
    for m in re.finditer(r"Function\s*:\s*(\S+)(.*?)(?=Function\s*:|\Z)",text,re.S):
        name,body=m.groups()
        if not any(k in name for k in ("gd_kernel","rq_kernel")): continue
        ins=[s.strip() for s in re.findall(r"/\*[0-9a-fA-F]+\*/\s+([^;\n]+);",body)]
        require(ins,"empty SASS")
        rows[name]=dict(instructions=ins,static_instruction_sites=len(ins),
            opcode_counts=dict(collections.Counter(re.sub(r"^@!?P\d+\s+","",s).split()[0] for s in ins)))
    require(rows,"no target SASS")
    return rows


def compile_cuda(src,tag,work,torch_source=False):
    cu,cubin=work/f"{tag}.cu",work/f"{tag}.{'o' if torch_source else 'cubin'}"
    cu.write_bytes(src.encode())
    cmd=["nvcc","-c" if torch_source else "-cubin","-std=c++20","-arch=sm_90a","-O3","--expt-relaxed-constexpr","-DNDEBUG","-Xptxas=-v",str(cu),"-o",str(cubin)]
    if torch_source:
        import sysconfig
        import torch
        from torch.utils.cpp_extension import include_paths
        cmd += ["-DTORCH_EXTENSION_NAME=s7_rt_compile","-DTORCH_API_INCLUDE_EXTENSION_H",
                "-D_GLIBCXX_USE_CXX11_ABI=%d"%int(torch._C._GLIBCXX_USE_CXX11_ABI),"--compiler-options","-fPIC"]
        cmd += ["-I"+p for p in include_paths(device_type="cuda")+[sysconfig.get_path("include")]]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
    (work/f"{tag}.ptxas.log").write_bytes((p.stdout+p.stderr).encode())
    require(p.returncode==0,f"{tag}: nvcc failed: {p.stderr[-3000:]}")
    sass=subprocess.check_output(["cuobjdump","-sass",str(cubin)],text=True,timeout=60)
    res=subprocess.check_output(["cuobjdump","-res-usage",str(cubin)],text=True,timeout=60)
    (work/f"{tag}.sass").write_bytes(sass.encode()); (work/f"{tag}.resources.txt").write_bytes(res.encode())
    return dict(command=cmd,resources=ptxas_resources(p.stderr),resource_dump=res,
        compiler_warnings=[s for s in p.stderr.splitlines() if "warning" in s.lower()]),sass_rows(sass)


def linux_gate(cpu_only=False):
    os.environ["CUDA_VISIBLE_DEVICES"]="" # Also guard direct Python invocation.
    report=dict(status="FAIL",GPU_executed=False,base_sha256=BASE_SHA256); code=1
    try:
        metas,specs=[],[]
        for v,p in OUTS.items():
            m,s=cpu_check(p.read_bytes().decode(),v); metas.append(m); specs.append(s)
        off,offspec=cpu_check(build_text(base_text(),0),0)
        bn=route_namespace(base_text())
        require(offspec["gd"]["cuda_sources"]==capture_build(bn,base_text())["cuda_sources"],"off GD identity")
        require(offspec["rq"]["cuda_sources"]==bn["_rqsrc"](bn["_rq_variants"]),"off RQ identity")
        report.update(cpu="PASS",variants=metas,off=off,route_map=route_report())
        if cpu_only: report["status"]="CPU_PASS_CUDA_NOT_RUN"; return report,0
        require(os.name!="nt" and shutil.which("nvcc") and shutil.which("cuobjdump"),"Linux CUDA toolchain required")
        work=Path(tempfile.mkdtemp(prefix="s7_rt_gate_")); report["work_directory"]=str(work)
        report["nvcc_version"]=subprocess.check_output(["nvcc","--version"],text=True)
        failures=[]
        for kind,v,target in (("gd",1,(0,1,1,8,16,4,1,2,7,0,0)),("rq",2,(128,6,2,0,0,0,42999939,1))):
            try:
                ns=route_namespace(OUTS[v].read_bytes().decode())
                # Compile the complete production CUDA source, including rq_go
                # host dispatch; never substitute a toy/device-only kernel.
                old=offspec[kind]["cuda_sources"]
                new=specs[v-1][kind]["cuda_sources"]
                bc,bs=compile_cuda(old,kind+"_base",work,kind=="rq")
                nc,nss=compile_cuda(new,kind+"_v"+str(v),work,kind=="rq")
                candidates=[n for n in nss if tuple(map(int,re.findall(r"Li(\d+)E",n)))==target]
                require(len(candidates)==1,f"missing/ambiguous {kind} specialization")
                name=candidates[0]; resource=nc["resources"].get(name,{})
                require({"registers","spill_stores","spill_loads"}<=resource.keys(),"missing ptxas resources")
                require(resource["spill_stores"]==resource["spill_loads"]==0,f"{kind} target spills: {resource}")
                common=set(bs)&set(nss)
                require(all(bs[n]["instructions"]==nss[n]["instructions"] for n in common),"unchanged kernel SASS drift")
                base_target=(0,1,1,8,16,2,1,2,7,0,0) if kind=="gd" else (128,5,1,0,0,0,42999939,1)
                oldname=next(n for n in bs if tuple(map(int,re.findall(r"Li(\d+)E",n)))==base_target)
                report[kind+"_compile"]=dict(status="PASS",base=bc,variant=nc,target=name,
                    before=bs[oldname],after=nss[name],unchanged_SASS_instances=len(common))
            except Exception:
                failures.append(kind+": "+traceback.format_exc())
        # Lt routing changes use the SAME extension. Compile its host source,
        # but never create a handle or query/execute a device algorithm here.
        try:
            import sysconfig
            from torch.utils.cpp_extension import include_paths
            require(all(s["lt"]==offspec["lt"] for s in specs),"unexpected Lt code change")
            inc=include_paths()+[sysconfig.get_path("include"),str(Path(os.environ["CUDA_HOME"])/"include")]
            site=Path(sys.prefix)/"lib"
            inc+=list(dict.fromkeys(str(p.parent) for p in site.glob("python*/site-packages/nvidia/**/cublasLt.h")))
            cpp=work/"lt_unchanged.cpp"; cpp.write_bytes(offspec["lt"].encode())
            cmd=["g++","-fsyntax-only","-std=c++20","-DTORCH_EXTENSION_NAME=s7_rt_lt"]+["-I"+p for p in inc]+[str(cpp)]
            p=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
            (work/"lt_compile.log").write_bytes((p.stdout+p.stderr).encode())
            require(p.returncode==0,p.stderr[-3000:])
            report["lt_compile"]=dict(status="HOST_COMPILE_PASS",command=cmd,
                compatibility="NOT RUN; prepare self-test must accept pin on H200",new_CUDA_instances=0)
        except Exception: failures.append("lt: "+traceback.format_exc())
        report.update(status="COMPILE_FAIL" if failures else "COMPILE_PASS_GPU_NOT_RUN",failures=failures)
        code=int(bool(failures))
    except Exception: report["error"]=traceback.format_exc()
    return report,code


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    for name in ("generate","check","map","report","linux-gate"):
        mode.add_argument("--"+name,action="store_true")
    ap.add_argument("--cpu-only",action="store_true")
    args=ap.parse_args()
    if args.map:
        print(json.dumps(route_report(),indent=2)); return 0
    if args.linux_gate:
        report,code=linux_gate(args.cpu_only)
        if args.cpu_only: print(json.dumps(report,indent=2))
        else:
            path=Path(__file__).with_suffix(".json")
            path.write_bytes((json.dumps(report,indent=2)+"\n").encode())
            print(json.dumps(dict(status=report["status"],result=str(path),error=report.get("error"),failures=report.get("failures")),indent=2))
        return code
    results=[]
    base=base_text()
    for v,p in OUTS.items():
        src=build_text(base,v); meta,_=cpu_check(src,v)
        if args.generate:
            p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(src.encode())
        else: require(p.read_bytes()==src.encode(),f"stale output {p}")
        results.append(meta)
    off,_=cpu_check(build_text(base,0),0)
    require(base_text()==base,"base changed during gates")
    if args.report: print(json.dumps(dict(variants=results,default_off=off,route_map=route_report()),indent=2))
    else: print(json.dumps(dict(status="CPU_PASS",variants=[dict(variant=m["variant"],bytes=m["bytes"],sha256=m["sha256"],
        changed_lines=m["changed_lines"]) for m in results],default_off="PASS",CUDA="NOT RUN",GPU="NOT RUN"),indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
""":"
# This named file is also the Linux gate; invoke it through cudaenv.sh.
set -euo pipefail
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
exec python -I "$0" --linux-gate "$@"
":"""
from __future__ import annotations

"""S7-DN: anchored, opt-in GD register-lookahead route variants (no GPU calls).

  python -I tools/s7_dn_patches.py --generate
  python -I tools/s7_dn_patches.py --check
  python -I tools/s7_dn_patches.py --map
  python -I tools/s7_dn_patches.py --report

Linux compile / SASS gate (this file is a bash/Python polyglot):
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash \
    "<repo>/tools/local/cudaenv.sh" bash \
    "<repo>/tools/s7_dn_patches.py"
The lead's Linux invocation writes tools/s7_dn_patches.json; compiler products
are retained in /tmp, never deleted. Windows CPU commands only write the two
explicitly named variant files when --generate is supplied. No pycache files.

Kills first: (1) more live weight registers can spill or reduce occupancy;
(2) other warps / the HBM stream may already hide the load latency. These are
test variants requested by the task, not a claim of a measured speedup. GD-A
(combo91) added a cp.async/shared-memory producer; these variants only select
the existing register ring. They do not change the x loads (R-gdxh-0), decode
(R-gdd-1), XP/PDL (DPF/combo69), or known losing sk2 routes (R-sk2-1/2/3).

Both variants change three independently identifiable kernel/shape pairs:
bit 1: qkv/16 ST 1 -> 2 (v1), 3 (v2).
bit 2: fused pooler+router/16 ST 2 -> 3 (v1), 4 (v2).
bit 4: fused pooler+router/32 ST 1 -> 2 (v1), 3 (v2).
S7_DN_VARIANT=0 is the inert/default template; output v1/v2 explicitly opt in.
No added module, packed-weight bank, tensor, or GPU self-test. Existing checks
and prepare timeout guards are retained byte-for-byte. Changed instantiations
replace existing ones, so the instance count and packing count stay constant.

Accounting, per thread per K=64 unit (EST source operation count):
48 B weights = 3 vector loads; 32*MT B x = 2*MT vector loads;
36 decode scalar operations = 4*(2 AND + 2 ADD + 1 funnel + 4 permute);
4*MT mma.sync instructions; identical escape patch instructions/data.
Thus the named core is 39+6*MT ops/unit = 51 (MT2), 63 (MT4), unchanged.
Whole-loop weight loads = 3*KU per thread, unchanged. Only 3*ST loads move
to the prologue; ring storage grows by 12 registers/thread per extra stage.
Address/control and compiler-generated SASS are NOT priced by this estimate;
the Linux gate records their actual static instruction histograms and spills.
qkv16: 4 warps * 1536 = 6144 weight B/CTA/unit, KU8, 576 CTAs.
cat16/32: 8 warps * 1536 = 12288 weight B/CTA/unit, KU18, 256 CTAs.
Payloads + fixed GD escape tables are 28,606,464 / 56,885,248 B, unchanged.

There are no protocol changes. Coverage proof is executed by protocol_gate:
qkv16: CTAs (bx=0..143, split=0..3), threads 0..127; all publish red,
nonowners send i=tid+128*j<256 to cluster rank0 then exit, rank0 combines
all 3 received arrays and writes all 16*16 outputs per bx. CAT: bx=0..255,
threads 0..255; all publish pooler and router reductions; i=tid+256*j<NO
writes pooler, i<NO/2 writes router partials. NO=256/512 for T16/T32.
CAT has no exiting CTAs, no persistent counters, and no reset loop. Red/redr
are rewritten every call; ST does not occur in any cleanup/combine index.
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
OUTS = {v: ROOT / f"box_out/s7/k3_dn_v{v}.py" for v in (1, 2)}
MARKER = "# S7-DN begin: existing GD register ring, opt-in only.\n"
ARM_TEMPLATE = '''
# S7-DN begin: existing GD register ring, opt-in only.
S7_DN_VARIANT=0
if S7_DN_VARIANT:
 if S7_DN_VARIANT not in (1,2):
  raise ValueError("S7_DN_VARIANT must be 0, 1, or 2")
 _s7_dn_st=S7_DN_VARIANT+1
 GD_ROUTE[("qkv",16)]=(2,1,4,8,_s7_dn_st,4,0) # bit 1
 GD_CAT_ROUTE[16]=(2,1,8,18,_s7_dn_st+1,1,0) # bit 2
 GD_CAT_ROUTE[32]=(4,1,8,18,_s7_dn_st,1,0) # bit 4
# S7-DN end
'''
TARGETS = ((1, "qkv", 16), (2, "cat", 16), (4, "cat", 32))


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def base_text():
    raw = BASE.read_bytes()
    require(digest(raw) == BASE_SHA256, "row changed: re-audit anchors before building")
    require(len(raw) == 965632 and b"\r" not in raw, "row size / LF changed")
    return raw.decode("utf-8")


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


def build_text(src, variant):
    require(variant in (0, 1, 2), "invalid variant")
    require(MARKER not in src, "already patched")
    # Fail closed on the complete route lines, not a loose tuple replacement.
    require(segment(src, "GD_CAT_ROUTE") ==
            "GD_CAT_ROUTE={16:(2,1,8,18,2,1,0),32:(4,1,8,18,1,1,0)}", "CAT anchor")
    require(literal(src, "GD_ROUTE")[("qkv", 16)] == (2,1,4,8,1,4,0), "qkv anchor")
    require(src.endswith("  return None\n"), "EOF anchor")
    lines = src.splitlines(keepends=True)
    require(lines[1].startswith("# nonce tinys7-c1 "), "nonce anchor")
    lines[1] = f"# nonce dn-v{variant} (GD register lookahead; qkv16 + cat16 + cat32)\n"
    return "".join(lines) + ARM_TEMPLATE.replace("S7_DN_VARIANT=0", f"S7_DN_VARIANT={variant}", 1)


NAMES = (
    "HIDDEN", "D_EXPERT", "N_EXPERTS", "UNPOOL_SLICE", "SLICED_UNPOOL_MAX_TOKENS",
    "GD_SMS", "GD_ROUTE", "GD_CAT_ROUTE", "GD_CAT_UNITS", "GD_DIMS", "GD_WEIGHT",
    "GD_ENABLE", "GD_BUILD", "GD_REQUIRE", "GD_PREPARE_BY_S", "GD_ONE_WAVE",
    "GD_CAT", "GD_CAT_SELF_TEST_T", "GD_SELF_TEST_T", "GD_SELF_TEST_PDL",
    "GD_CPP_SRC", "GD_CUDA_SRC", "_GD", "NB64_BUILD", "NB64_USE", "NB64_GD_B",
    "_gd_nk", "_gd_key", "_gd_cat_key", "_gd_cat_keys", "_gd_keys", "_gd_bucket",
    "_gd_prepare", "_gd_cat_prepare", "_gd_live", "_gd_cat_live", "_gd_require",
)


def route_namespace(src):
    tree = ast.parse(src)
    ns = dict(hashlib=hashlib, time=time, _nb64=lambda t: 33 <= t <= 64)
    exec(compile(ast.Module(body=[node(tree, n) for n in NAMES], type_ignores=[]), str(BASE), "exec"), ns)
    if MARKER in src:
        exec(src[src.index(MARKER):], ns)
    return ns


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


def protocol_gate(key):
    _,mt,wr,wk,ku,st,s,mb,xp,vf,cat = key
    nt, no = wr*wk*32, wr*mt*128
    covered = [i for tid in range(nt) for i in range(tid,no,nt)]
    require(sorted(covered)==list(range(no)), "pooler/output coverage")
    if cat:
        require(s==wr==1 and vf==0 and cat==2, "CAT invariant")
        partials = [i for tid in range(nt) for i in range(tid,no//2,nt)]
        require(sorted(partials)==list(range(no//2)), "router partial coverage")
        # Exhaustively reconstruct the actual store coordinates for all CTAs.
        pool, router = set(), set()
        for bx in range(256):
            for i in covered:
                lane,c,r = i&31,(i>>5)&3,i>>7
                m = r%mt
                row=bx*16+(lane>>2)+8*(c>>1)
                tok=m*8+2*(lane&3)+(c&1)
                require((tok,row) not in pool, "duplicate pooler store")
                pool.add((tok,row))
            for i in partials:
                lane,c,m=i&31,(i>>5)&1,i>>6
                coord=(bx&7,m*8+2*(lane&3)+c,(bx>>3)*8+(lane>>2))
                require(coord not in router, "duplicate router partial")
                router.add(coord)
        require(len(pool)==mt*8*4096 and len(router)==8*mt*8*256, "missing output")
        return dict(CTAs=256,threads=[0,nt-1],pooler_indices=[0,no-1],router_indices=[0,no//2-1],
                    early_exits=0,persistent_resets=0,coverage="all stores exactly once; reductions unchanged")
    require(s==4 and wr==1 and wk==4 and mt==2, "qkv target changed")
    return dict(CTAs=[144,4,1],threads=[0,127],red_indices=[0,255],
                owner="split=0 combines 3*256 FP32 partials then stores 256 values per bx",
                exiting="split=1..3 each sends all 256 partials before return",persistent_resets=0)


def cpu_check(src, variant):
    base = base_text()
    require(src == build_text(base,variant), "output is not exact anchored generation")
    require("\r" not in src and len(src.encode()) < 1000000, "LF/size gate")
    tree = ast.parse(src)
    compile(tree,"<S7-DN>","exec")
    require(src.splitlines()[1].startswith(f"# nonce dn-v{variant} ("), "nonce line")
    # The original executable AST (including all self-tests and kernel bodies)
    # is an exact prefix. Only the EOF selector is new.
    bt = ast.parse(base)
    require(ast.dump(ast.Module(body=tree.body[:len(bt.body)],type_ignores=[]),include_attributes=False)==
            ast.dump(bt,include_attributes=False), "changed original code/self-tests")
    bn,ns=route_namespace(base),route_namespace(src)
    require(len(ns["_gd_keys"]())==len(bn["_gd_keys"]()), "added compilation instances")
    for bit,role,t in TARGETS:
        a,b=target_key(bn,role,t),target_key(ns,role,t)
        require(a[:5]+a[6:] == b[:5]+b[6:], "changed a parameter other than ST")
    spec=capture_build(ns,src)
    dry=prepare_dry_run(src,ns)
    rows=[]
    for bit,role,t in TARGETS:
        key=cpp_key(target_key(ns,role,t))
        rows.append(dict(bit=bit,role=role,T=t,key=key,ring=ring_gate(key),protocol=protocol_gate(key)))
    changes=[]
    for tag,i,j,k,l in difflib.SequenceMatcher(a=base.splitlines(),b=src.splitlines(),autojunk=False).get_opcodes():
        if tag!="equal":
            changes.append(dict(kind=tag,base_lines=[i+1,j],output_lines=[k+1,l],text=src.splitlines()[k:l]))
    return dict(variant=variant,status="CPU_PASS",bytes=len(src.encode()),sha256=digest(src),
                instance_count=len(ns["_gd_keys"]()),dry_run=dry,targets=rows,changed_lines=changes,
                CUDA_sources_identical=True,prepare_selftests_unchanged=True),spec


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
                esc=None
                formula="(2304*8192+8192*4096)*1.5; escape arrays add 4*(KS*(N/128)*8*CAP+32) per role"
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
                    role="unpooler"
                elif "coopA" in label:
                    role="qkv"
                elif "tst_72x64" in label:
                    role="pooler+router"
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


def instruction_accounting(meta):
    rows=[]
    for r in meta["targets"]:
        k=r["key"]
        old_st=2 if (r["role"],r["T"])==("cat",16) else 1
        mt,ku,st,warps=k[1],k[4],k[5],k[2]*k[3]
        rows.append(dict(bit=r["bit"],role=r["role"],T=r["T"],
                         weight_bytes_per_CTA_unit_before_after=[warps*1536]*2,
                         weight_bytes_per_thread_unit_before_after=[48]*2,
                         x_bytes_requested_per_thread_unit_before_after=[32*mt]*2,
                         named_core_source_ops_per_thread_unit_before_after=[39+6*mt]*2,
                         source_ops_breakdown=dict(weight_vector_loads=3,x_vector_loads=2*mt,decode_ops=36,mma=4*mt),
                         weight_vector_loads_per_thread_total_before_after=[3*ku]*2,
                         register_ring_words_per_thread_before_after=[12*old_st,12*st],
                         extra_weight_stream_bytes=0,extra_scratch_bytes=0,
                         caveat="Source counts exclude unchanged escape/address/control ops; SASS counts require Linux gate"))
    return rows


def performance_ledger(meta, dense_map):
    """Falsifiable targets, NOT a fabricated timing prediction.

    Added queue capacity is expressed in stream-equivalent microseconds only
    to size the experiment. It is neither a saving nor a bound on saving.
    No measured stall decomposition exists for these new instantiations.
    """
    result=[]
    for target in meta["targets"]:
        role,t=target["role"],target["T"]
        selected=[r for r in dense_map if r["T"]==t and r["role"]==
                  ("pooler+router" if role=="cat" else role)]
        require(len(selected)==2,"missing target profile pair")
        key=target["key"]
        ctas=576 if role=="qkv" else 256
        old_st=2 if (role,t)==("cat",16) else 1
        extra_inflight=ctas*key[2]*key[3]*1536*(key[5]-old_st)
        for r in selected:
            b=r["weight_bytes_EST"]
            result.append(dict(bit=target["bit"],shape=r["shape"],role=role,
                MEASURED_current_us=r["measured_us"],measurement_source=r["measurement_source"],
                EST_success_threshold_saving_us=0.5,
                EST_expected_us_CONDITIONAL_on_success=round(r["measured_us"]-0.5,2),
                estimate_status="test target only; speedup unknown and zero/regression plausible",
                kill_reasons=["ST adds 12 live words/thread/stage; register pressure can spill or lower occupancy",
                              "Other warps or the HBM stream may already hide the latency; PDL durations also include waits"],
                EST_stream_floor_us=round(1.9+b/4.64e6,6),
                EST_floor_arithmetic=f"1.9 + {b}/4,640,000; residual is NOT all exposed stall",
                EST_required_stall_cycles=f"0.5 * 1520 = 760 cycles/CTA, or 760/{key[4]} = {760/key[4]:.2f} cycles/unit",
                EST_extra_inflight_weight_capacity_B=extra_inflight,
                EST_capacity_stream_equivalent_us=round(extra_inflight/4.64e6,6),
                EST_capacity_arithmetic=f"{ctas} CTAs * {key[2]*key[3]} warps * 1536 B * {key[5]-old_st} stages / 4,640,000 B/us; NOT saved time",
                critical_path=("qkv is the first timed kernel, and its output gates attention; there is no preceding PDL work to hide a real duration reduction"
                               if role=="qkv" else
                               "Only a post-wait reduction counts: pooler and router partial completion gates tail/MoE. A shorter overlapped prologue alone is not a layer gain"),
                closest_DO_NOT_REPEAT="GD-A/combo91 (shared ring, different transport); R-gdxh-0; R-rat-1 (different routes)",
                exact_new_instance=key))
    return result


def read_sass(text):
    kernels={}
    for m in re.finditer(r"Function\s*:\s*(\S+)(.*?)(?=Function\s*:|\Z)",text,re.S):
        name,body=m.groups()
        if "gd_kernel" not in name:
            continue
        key=tuple(map(int,re.findall(r"Li(\d+)E",name)))
        require(len(key)==11 and key not in kernels,f"SASS symbol parse: {name}")
        ins=[(int(a,16),s.strip()) for a,s in re.findall(r"/\*([0-9a-fA-F]+)\*/\s+([^;\n]+);",body)]
        require(ins,"empty SASS")
        ops=collections.Counter(re.sub(r"^@!?P\d+\s+","",s).split()[0] for _,s in ins)
        kernels[key]=dict(symbol=name,instructions=ins,static_instruction_count=len(ins),opcodes=dict(ops))
    require(kernels,"no GD kernels in SASS")
    return kernels


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


def compile_gate(spec,tag,work):
    cu=work/f"{tag}.cu"
    cubin=work/f"{tag}.cubin"
    cu.write_bytes(spec["cuda_sources"].encode())
    cmd=["nvcc","-cubin","-std=c++20","-arch=sm_90a","-O3","-Xptxas=-v",str(cu),"-o",str(cubin)]
    p=subprocess.run(cmd,text=True,capture_output=True,timeout=240)
    (work/f"{tag}.ptxas.log").write_bytes((p.stdout+p.stderr).encode())
    require(p.returncode==0,f"{tag} nvcc failed: {p.stderr[-4000:]}")
    sass=subprocess.check_output(["cuobjdump","-sass",str(cubin)],text=True,timeout=60)
    resources=subprocess.check_output(["cuobjdump","-res-usage",str(cubin)],text=True,timeout=60)
    (work/f"{tag}.sass").write_bytes(sass.encode())
    (work/f"{tag}.resources.txt").write_bytes(resources.encode())
    return dict(command=cmd,ptxas=ptxas_resources(p.stderr),resources=resources),read_sass(sass)


def linux_gate(cpu_only=False):
    """All writes stay in /tmp except the adjacent result JSON. Never runs CUDA."""
    report=dict(status="FAIL",GPU_executed=False,base_sha256=BASE_SHA256)
    code=1
    try:
        base=base_text()
        metas,specs=[],[]
        for v,p in OUTS.items():
            meta,spec=cpu_check(p.read_bytes().decode(),v)
            meta["accounting"]=instruction_accounting(meta)
            metas.append(meta)
            specs.append(spec)
        off_meta,off_spec=cpu_check(build_text(base,0),0)
        bn=route_namespace(base)
        base_spec=capture_build(bn,base)
        require(base_spec["cuda_sources"]==off_spec["cuda_sources"],"off source identity")
        report.update(cpu="PASS",variants=metas,inert=off_meta,dense_map=baseline_rows())
        for meta in metas:
            meta["performance_ledger"]=performance_ledger(meta,report["dense_map"])
        if cpu_only:
            report.update(status="CPU_PASS_CUDA_NOT_RUN")
            return report,0
        require(os.name!="nt" and shutil.which("nvcc") and shutil.which("cuobjdump"),"Linux nvcc/cuobjdump required")
        work=Path(tempfile.mkdtemp(prefix="s7_dn_gate_"))
        report["work_directory"]=str(work)
        report["nvcc_version"]=subprocess.check_output(["nvcc","--version"],text=True)
        compiled,bsass=compile_gate(base_spec,"base",work)
        report["base_compile"]=compiled
        failures=[]
        for meta,spec in zip(metas,specs):
            try:
                c,sass=compile_gate(spec,f"v{meta['variant']}",work)
            except Exception:
                meta.update(status="COMPILE_FAIL",compile_error=traceback.format_exc())
                failures.append(f"v{meta['variant']} compile failed; see compile_error and retained /tmp logs")
                continue
            meta["compile"]=c
            meta["sass_targets"]=[]
            for r in meta["targets"]:
                key=tuple(r["key"])
                require(key in sass,f"uncompiled target {key}")
                sk=sass[key]
                resource=c["ptxas"].get(sk["symbol"],{})
                require(set(("spill_stores","spill_loads","registers"))<=set(resource),"missing ptxas resource record")
                before=cpp_key(target_key(bn,r["role"],r["T"]))
                waits=[pc for pc,s in sk["instructions"] if re.search(r"\bACQBULK\b",s)]
                require(len(waits)==1,"expected one PDL wait")
                # XP0 x loads lack NA; immutable weight loads have NA. No x is
                # permitted to cross the CAT dependency wait. qkv has no producer.
                pre_x=[(pc,s) for pc,s in sk["instructions"] if pc<waits[0] and
                       re.search(r"\bLDG\.E(?:\.[A-Z0-9]+)*\.128(?:\.|\s)",s) and ".NA." not in s]
                spills=resource["spill_stores"]+resource["spill_loads"]
                if spills:
                    failures.append(f"v{meta['variant']} bit{r['bit']}: nonzero spills {resource}")
                if r["role"]=="cat" and pre_x:
                    failures.append(f"v{meta['variant']} bit{r['bit']}: x load before dependency wait")
                old_mma=sum(n for op,n in bsass[before]["opcodes"].items() if op.startswith("HMMA"))
                new_mma=sum(n for op,n in sk["opcodes"].items() if op.startswith("HMMA"))
                require(new_mma==old_mma and new_mma>0,"math instruction count changed")
                meta["sass_targets"].append(dict(bit=r["bit"],key=key,resources=resource,
                    static_instruction_count=sk["static_instruction_count"],opcodes=sk["opcodes"],
                    baseline_static_instruction_count=bsass[before]["static_instruction_count"],
                    baseline_opcodes=bsass[before]["opcodes"],pre_wait_x_candidates=pre_x))
            shared=set(bsass)&set(sass)
            different=[k for k in shared if [s for _,s in bsass[k]["instructions"]]!=[s for _,s in sass[k]["instructions"]]]
            require(not different,f"untouched SASS changed: {different}")
            meta["unchanged_SASS_instances"]=len(shared)
            meta["status"]="COMPILE_PASS" if not any(f.startswith(f"v{meta['variant']} ") for f in failures) else "COMPILE_FAIL"
        report.update(status="COMPILE_PASS_GPU_NOT_RUN" if not failures else "COMPILE_FAIL",failures=failures)
        code=0 if not failures else 1
    except Exception:
        report["error"]=traceback.format_exc()
    return report,code


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate",action="store_true")
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--map",action="store_true")
    mode.add_argument("--report",action="store_true")
    mode.add_argument("--linux-gate",action="store_true")
    ap.add_argument("--cpu-only",action="store_true")
    args=ap.parse_args()
    if args.linux_gate:
        report,code=linux_gate(args.cpu_only)
        if args.cpu_only:
            print(json.dumps(report,indent=2))
        else:
            dest=Path(__file__).with_suffix(".json")
            dest.write_bytes((json.dumps(report,indent=2)+"\n").encode())
            print(json.dumps(dict(status=report["status"],result=str(dest),error=report.get("error")),indent=2))
        return code
    if args.map:
        print(json.dumps(baseline_rows(),indent=2))
        return 0
    base=base_text()
    result=[]
    for v,p in OUTS.items():
        src=build_text(base,v)
        meta,_=cpu_check(src,v)
        if args.generate:
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(src.encode())
        else:
            require(p.read_bytes()==src.encode(),f"stale generated output: {p}")
        # Required isolated AST gate, with no bytecode writes.
        subprocess.run([sys.executable,"-I","-c",
                        "import ast,sys;ast.parse(open(sys.argv[1],encoding='utf-8').read())",str(p)],check=True)
        meta["accounting"]=instruction_accounting(meta)
        result.append(meta)
    inert,_=cpu_check(build_text(base,0),0)
    require(route_namespace(build_text(base,0))["_gd_keys"]()==route_namespace(base)["_gd_keys"](),"inert route difference")
    require(digest(BASE.read_bytes())==BASE_SHA256,"base was modified")
    if args.report:
        dense_map=baseline_rows()
        for meta in result:
            meta["performance_ledger"]=performance_ledger(meta,dense_map)
        print(json.dumps(dict(variants=result,inert=inert,dense_map=dense_map),indent=2))
    else:
        print(json.dumps(dict(status="CPU_PASS_CUDA_NOT_RUN",base_unchanged=True,inert="PASS",
                              variants=[{k:m[k] for k in ("variant","bytes","sha256","instance_count","dry_run","changed_lines")}
                                        for m in result]),indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

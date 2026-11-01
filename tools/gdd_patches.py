"""R-gdd-1: compile-time GD decode-null diagnosis, never a submission.

Kill risks first: decode may already overlap the stream/PDL; changing operand
values/allocation may change downstream routing, power or placement. This probe
measures exposure, not a shippable saving. No new design is proposed.
EST source accounting per warp / 1024-weight stage: weights 3*32*16=1536 B,
x requested 32*MT*32=1024*MT B (unique 1024*MT B), unchanged; escape table
32*4=128 B per warp/call unchanged. Decode per thread 4*(2+3+4)=36 ops ->
16 three-input XORs (LOP3 0x96), plus one loop-invariant K&63; MMA 4*MT unchanged. CTA stage
bytes multiply by WR*WK; VF=2 router traffic and CAT's two router stages stay.
All 12 loaded words feed 16 distinct opaque outputs; all outputs feed patch/MMA.
The runtime mask is A.K&63, zero for every GD route, unknown to device compilation.
Physical-register/resource/SASS identity is REQUIRED by gdd_compile.py, not inferred.

Protocol proof (source suffix is byte-identical): every CTA (bx,split,bz),
tid=0..NT-1, NT=32*WR*WK, still waits PDL and reaches the same CTA barrier.
S=1 store: i=tid; i<NO; i+=NT, NO=128*WR*MT; CAT router combine i<NO/2
with the same stride, sums every k2=0..WK-1. VF=2 all NT wait cp.async and
join, warps0..7/lane0 store their two rows/four tokens; marks tid0 after join.
S>1 nonzero cluster ranks: all tid send i<NO stride NT, then original return;
rank0 all tid wait mbar then combine i=tid+j*NT<NO, all r=0..S-2, then store.
No added exits, counters, flags, resets, barriers or changed launch parameters.
"""
import ast
import contextlib
import copy
import functools
import hashlib
import io
import pathlib
import re
import sys
from types import SimpleNamespace as NS

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz_rt.py"
OUT = ROOT / "kernels/lab/k3_gdd_ab.py"
ARMS = ("twinA", "DQ_ALL", "DQ_POOL", "twinB")
DECODE_CALL = "decode(R[slot], (CAT && s >= KUP) ? A.tbr : A.tb, a);"
HELPER = r'''
// GDD: 16 opaque fragments; use every one of the 12 loaded weight words.
__device__ __forceinline__ void gdd_decode(const uint4 (&p)[3], uint32_t z, uint32_t (&a)[16]) {
const uint32_t L[8] = {p[0].x,p[0].y,p[0].z,p[0].w,p[1].x,p[1].y,p[1].z,p[1].w};
const uint32_t C[4] = {p[2].x,p[2].y,p[2].z,p[2].w};
#pragma unroll
for (int i = 0; i < 16; ++i) {
// All 16 (L,C) pairs differ. A.K is runtime data, so z is opaque to ptxas.
asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(a[i]) : "r"(L[i & 7]), "r"(C[i >> 2]), "r"(z));
}
}
'''
GLOBALS = '''GDD_ARM = "ship"
GDD_VARIANTS = ("twinA", "DQ_ALL", "DQ_POOL", "twinB")
_GDD = {"mods": {}, "calls": {}}
def _gdd_module(r):
    if GDD_ARM == "ship":
        return r["mod"]
    if GDD_ARM not in GDD_VARIANTS:
        raise ValueError("invalid GDD_ARM")
    return _GDD["mods"][GDD_ARM]
def _gdd_hit(path, T):
    key = f"{GDD_ARM}:{path}:T{T}"
    _GDD["calls"][key] = _GDD["calls"].get(key, 0) + 1
def _gdd_prepare(state):
    for arm in GDD_VARIANTS:
        _pt1_wait("gdd_" + arm)
        if arm not in _GDD["mods"]:
            _gdd_build_one(arm)
    # Exact state expressions from _gd_live / _gd_cat_live / _vfr_prepare.
    st = state.get("gd")
    if st is None or state.get("gd_cat") is None or not _VFR["ok"]:
        raise RuntimeError("GDD requires ship GD, cat and VF=2 routes")
    for proj, T in (("qkv",1),("o",1),("pooler",1),("unpooler",1),
                    ("qkv",8),("o",8),("pooler",8),("unpooler",8),
                    ("qkv",16),("o",16),("unpooler",16)):
        if (proj, T) not in st["route"]:
            raise RuntimeError(f"missing ship GD route {proj}/{T}")
    for T in (16,32):
        if T not in state["gd_cat"]:
            raise RuntimeError(f"missing ship cat route {T}")
'''


def sub(s, old, new, label):
    assert s.count(old) == 1, (label, s.count(old))
    return s.replace(old, new, 1)


@functools.lru_cache(maxsize=3)
def tree(s):
    return ast.parse(s)


def node(s, name):
    found = [n for n in tree(s).body if
             isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(found) == 1, (name, len(found))
    return found[0]


def literal(s, name):
    return ast.literal_eval(node(s, name).value)


def selected(s, names, ns=None):
    env = {} if ns is None else dict(ns)
    code = ast.Module(body=[node(s, name) for name in names], type_ignores=[])
    exec(compile(code, "<selected real layer>", "exec"), env)
    return env


def routes(base):
    names = ("HIDDEN", "D_EXPERT", "N_EXPERTS", "UNPOOL_SLICE", "SLICED_UNPOOL_MAX_TOKENS", "GD_SMS",
             "GD_ROUTE", "GD_CAT_ROUTE", "GD_CAT_UNITS", "GD_DIMS", "_gd_nk", "_gd_key", "_gd_cat_key", "_gd_cat_keys", "_gd_keys")
    return selected(base, names)


def patched_cuda(base):
    env = routes(base)
    pool = sorted({env["_gd_key"](*env["_gd_nk"]("pooler", t), rt)
                   for (p,t),rt in env["GD_ROUTE"].items() if p == "pooler" and rt})
    assert not set(pool).intersection(env["_gd_key"](*env["_gd_nk"](p,t),rt)
                                     for (p,t),rt in env["GD_ROUTE"].items() if p!="pooler" and rt)
    fields = ("OUT", "MT", "WR", "WK", "KU", "ST", "S", "MB", "XP")
    cond = " || ".join("(" + " && ".join(f"{f} == {v}" for f,v in zip(fields,k)) + ")" for k in pool)
    call = ("if constexpr (GDD_DQ == 1 || VF > 0 || CAT > 0 || " + cond + ") {\n"
            "gdd_decode(R[slot], (uint32_t)A.K & 63u, a);\n} else {\n" + DECODE_CALL + "\n}")
    cu = literal(base, "GD_CUDA_SRC")
    new = sub(cu, "#define GC_XP(S_)", HELPER + "#define GC_XP(S_)", "helper")
    new = sub(new, DECODE_CALL, call, "only gd_kernel decode call")
    # Every instruction outside this call and the new helper, including the
    # unpack self-tests, stream loads and entire post-wait cleanup, is verbatim.
    assert sub(sub(new, call, DECODE_CALL, "undo call"), HELPER, "", "undo helper") == cu
    assert new[new.index("__global__ void unpack_kernel"):] == cu[cu.index("__global__ void unpack_kernel"):]
    return new


def build(base, report=False):
    assert "GDD_ARM" not in base
    fn = ast.get_source_segment(base, node(base, "_gd_build"))
    changed = sub(fn, "def _gd_build():", "def _gdd_build_one(arm):", "builder name")
    changed = sub(changed, '    _pt1_wait("gd")\n', "", "no self wait")
    changed = sub(changed, '    if _GD["mod"] is not None and _GD["keys"] == keys:\n        return _GD["mod"]',
                  '    if arm in _GDD["mods"]:\n        return _GDD["mods"][arm]', "variant bank")
    changed = sub(changed, '    cuda = GD_CUDA_SRC.replace("@INST@", inst).replace("@INFO@", info)',
                  '    if arm not in GDD_VARIANTS:\n        raise ValueError("invalid GD variant")\n'
                  '    src = _GDD_CUDA if arm in ("DQ_ALL", "DQ_POOL") else GD_CUDA_SRC\n'
                  '    cuda = src.replace("@INST@", inst).replace("@INFO@", info)\n'
                  '    defines = ["-DGDD_DQ=" + ("1" if arm == "DQ_ALL" else "2")] if arm.startswith("DQ_") else []', "source")
    changed = sub(changed, 'name=f"gd_{tag}"', 'name=f"gdd_{arm}_{tag}"', "cache identity")
    changed = sub(changed, 'code=sm_90a"], verbose=False)', 'code=sm_90a"] + defines, verbose=False)', "defines")
    changed = sub(changed, '    _GD.update(mod=mod, keys=keys, build_s=time.time() - t1)',
                  '    _GDD["mods"][arm] = mod', "publish bank")
    edits = [("GD_ENABLE = True\n", GLOBALS + "GD_ENABLE = True\n", "globals / prepare / select"),
             (fn, fn + "\n" + changed, "separate compiled modules"),
             ('        jobs.append(("gd", _gd_build))', '        jobs.append(("gd", _gd_build))\n'
              '        for arm in GDD_VARIANTS:\n            jobs.append(("gdd_" + arm, lambda arm=arm: _gdd_build_one(arm)))', "PT1 builds"),
             ('    _vfr_prepare(state)\n', '    _vfr_prepare(state)\n    _gdd_prepare(state)\n', "prepare gates")]
    # The CAT self-test calls _gd_cat_run, which also selects timed arms. Force
    # the ENTIRE real prepare through ship even when the caller selected DQ.
    prepare_head = "def prepare(weights):"
    assert prepare_head in base
    prepare_wrapper = '''def prepare(weights):
    global GDD_ARM
    saved = GDD_ARM
    GDD_ARM = "ship"
    try:
        return _gdd_ship_prepare(weights)
    finally:
        GDD_ARM = saved

def _gdd_ship_prepare(weights):'''
    edits.append((prepare_head, prepare_wrapper, "ship-only prepare validation"))
    for name, method, hit in (("_gd", "gd_run", 'proj, T'), ("_vfr_gdr", "gdr_run", '"vf" + str(vf), T'),
                              ("_gd_cat_run", "gdc_run", '"cat", T')):
        old = ast.get_source_segment(base, node(base, name))
        new = sub(old, f'r["mod"].{method}(', f'_gdd_module(r).{method}(', name)
        lines = new.splitlines()
        # Add the counter AFTER the entire (possibly multiline) module call.
        tree = ast.parse(new)
        call = [n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == method]
        assert len(call) == 1
        lines.insert(call[0].end_lineno, " " * call[0].col_offset + f"_gdd_hit({hit})")
        edits.append((old, "\n".join(lines), name + " select/count"))
    s = base
    for old,new,label in edits:
        s = sub(s, old, new, label)
    tail = "\n# GDD null source; the ship literal above is unchanged.\n_GDD_CUDA = " + repr(patched_cuda(base)) + "\n"
    s += tail
    ast.parse(s)
    restored = s[:-len(tail)]
    for old,new,label in reversed(edits):
        restored = sub(restored, new, old, "undo " + label)
    assert restored == base
    if report:
        for old,new,label in edits:
            start = s[:s.index(new)].count("\n") + 1
            print(f"[gdd] {label}: {OUT.name}:{start}-{start+new.count(chr(10))}")
        print(f"[gdd] CUDA diff: +gdd_decode before GC_XP; -{DECODE_CALL}; +constexpr arm call")
    return s


def dry_run(base, patched):
    """Execute actual base prepare dictionaries and edited build/call paths on CPU."""
    class Tensor:
        def __init__(self, shape=(1,), dtype="bf16", **kw): self.shape,self.dtype,self.device=tuple(shape),dtype,"cpu"
        def dim(self): return len(self.shape)
        def is_contiguous(self): return True
        def contiguous(self): return self
        def view(self, *a): return self
        def data_ptr(self): return 16
        def numel(self): return 1
        def __getitem__(self, key):
            if isinstance(key, slice): return Tensor(((key.stop or self.shape[0])-(key.start or 0),)+self.shape[1:], self.dtype)
            return self
    seen, builds = [], []
    def module(arm):
        return NS(gd_info=lambda *a: (128,0,8,0),
                  gdr_info=lambda *a: (128,0,8,0),
                  gd_unpack=lambda *a: None, gd_unpack_cat=lambda *a: None,
                  gd_run=lambda *a: seen.append((arm,"gd",a)),
                  gdr_run=lambda *a: seen.append((arm,"vf",a)),
                  gdc_run=lambda *a: seen.append((arm,"cat",a)))
    def compiler(**kw):
        builds.append(kw)
        name=kw["name"]
        return module(name[4:].rsplit("_",1)[0] if name.startswith("gdd_") else "ship")
    torch = NS(bfloat16="bf16",float32="f32",int16="i16",int64="i64",empty=Tensor,
               empty_like=lambda t: Tensor(t.shape,t.dtype), equal=lambda *a: True,
               cat=lambda ts,dim=0: Tensor((sum(t.shape[0] for t in ts),ts[0].shape[1])),
               cuda=NS(synchronize=lambda: None,empty_cache=lambda: None,
                       get_device_properties=lambda d: NS(multi_processor_count=132)))
    env = routes(base)
    names = ("GD_BUILD", "GD_ENABLE", "GD_REQUIRE", "GD_PREPARE_BY_S", "GD_ONE_WAVE", "GD_WEIGHT", "GD_PDL", "GD_TRIG",
             "GD_CAT", "GD_CAT_PDL", "NB64_BUILD", "NB64_USE", "NB64_GD_B", "_GD", "_GD_SEEN", "_GD_CAT_SEEN",
             "GD_CPP_SRC", "GD_CUDA_SRC", "_GDD_CUDA", "GDD_ARM", "GDD_VARIANTS", "_GDD", "_gd_require", "_gd_bucket",
             "_gd_prepare", "_gd_cat_prepare", "_gd_live", "_gd_cat_live", "_gd", "_gd_cat_run", "_vfr_gdr",
             "_gdd_module", "_gdd_hit", "_gdd_prepare")
    env = selected(patched,names,dict(env,torch=torch,hashlib=hashlib,time=NS(time=lambda: 0),_pt1_wait=lambda *a: None,
                _VFR={"ok":True,"marks":Tensor(),"nomk":Tensor(),"seen":set()}, VF_GDM=False))
    for name in ("_gd_build", "_gdd_build_one"):
        fn = copy.deepcopy(node(patched,name))
        fn.body = [n for n in fn.body if not isinstance(n,ast.ImportFrom)]
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"<real builder CPU compiler>","exec"),env)
    env["load_inline"]=compiler
    env.update(_gd_pack=lambda w,wr,wk,ku,s,b: {"w":Tensor(),"esc":Tensor(),"tb":b,"G":1,"nbx":w.shape[0]//16,"nesc":0},
               _gd_cat_pack=lambda *a: {"w":Tensor(),"esc":Tensor(),"tb":0,"tbr":0,"G":1,"nbx":256,"nesc":0},
               _gd_window_base=lambda w: 32, _gd_cat_virtual=lambda hi,lo: Tensor((512,8192)),
               _gd_selftest=lambda *a: (True,"CPU"), _gd_cat_selftest=lambda *a: (True,"CPU"),
               _k3_disable=lambda *a: (_ for _ in ()).throw(AssertionError(a)))
    # Dense keys taken from GD_WEIGHT and dimensions, then real prepare constructs
    # every r and state['gd']/state['gd_cat']; no invented module-handle layout.
    state={"dense":{env["GD_WEIGHT"][p]:Tensor(d) for p,d in env["GD_DIMS"].items()},
           "v8":{"wcat":Tensor((4608,8192))},"router":Tensor((256,8192))}
    with contextlib.redirect_stdout(io.StringIO()):
        env["_gd_prepare"](state,0)
        env["_gd_cat_prepare"](state,0)
        env["_gdd_prepare"](state)
    assert state["gd"] and state["gd_cat"] and len(builds)==5
    assert set(env["_GDD"]["mods"])==set(ARMS)
    # Run the exact new PT1 registration branch, delaying lambda execution until
    # all arms are registered: catches late-binding of the arm variable.
    registration=[n for n in node(patched,"_pt1_start").body if isinstance(n,ast.If) and ast.unparse(n.test)=="GD_BUILD"]
    assert len(registration)==1
    env["_GDD"]["mods"].clear(); env["jobs"]=[]
    exec(compile(ast.Module(body=registration,type_ignores=[]),"<actual PT1 GD registration>","exec"),env)
    assert [name for name,_ in env["jobs"]]==["gd"]+["gdd_"+a for a in ARMS]
    for name,fn in env["jobs"]: fn()
    assert set(env["_GDD"]["mods"])==set(ARMS)
    env["_gdd_prepare"](state) # cached-bank prepare path
    stock=builds[0]
    for arm in ARMS:
        b=next(b for b in builds if b["name"].startswith("gdd_"+arm+"_"))
        if arm.startswith("twin"):
            assert b["cuda_sources"]==stock["cuda_sources"] and b["extra_cuda_cflags"]==stock["extra_cuda_cflags"]
        else: assert "-DGDD_DQ="+("1" if arm=="DQ_ALL" else "2") in b["extra_cuda_cflags"]
    # Run the new wrapper for every initial arm and both success/error paths.
    exec(compile(ast.Module(body=[node(patched,"prepare")],type_ignores=[]),"<real prepare wrapper>","exec"),env)
    def validating_prepare(weights):
        assert env["GDD_ARM"]=="ship"
        env["_gd_prepare"](weights,0); env["_gd_cat_prepare"](weights,0)
        env["_gdd_prepare"](weights)
        return weights
    env["_gdd_ship_prepare"]=validating_prepare
    for arm in ("ship",)+ARMS:
        env["GDD_ARM"]=arm
        with contextlib.redirect_stdout(io.StringIO()): assert env["prepare"](state) is state
        assert env["GDD_ARM"]==arm
    env["_gdd_ship_prepare"]=lambda w: (_ for _ in ()).throw(RuntimeError("prepare error"))
    try: env["prepare"](state)
    except RuntimeError: pass
    else: raise AssertionError("prepare exception swallowed")
    assert env["GDD_ARM"]=="twinB"
    assert len({(i & 7,i >> 2) for i in range(16)})==16
    for arm in ("ship",)+ARMS:
        env["GDD_ARM"]=arm
        for (proj,t),r in state["gd"]["route"].items():
            if t>32: continue
            with contextlib.redirect_stdout(io.StringIO()): env["_gd"](state,proj,Tensor((t,r["K"])))
            assert seen[-1][2][0]==r["inst"] and seen[-1][0]==arm
            assert r["K"]%64==0
        for t,r in state["gd_cat"].items():
            env["_gd_cat_run"](r,Tensor(),Tensor(),Tensor(),t,3)
            assert seen[-1][0]==arm
        with contextlib.redirect_stdout(io.StringIO()): env["_vfr_gdr"](state,Tensor((8,8192)),Tensor(),2)
        assert seen[-1][0]==arm
    for bad in ("typo",None,1):
        env["GDD_ARM"]=bad
        try: env["_gdd_module"]({"mod":None})
        except (ValueError,TypeError): pass
        else: raise AssertionError("invalid arm accepted")
    for arm in ARMS:
        del env["_GDD"]["mods"][arm]
        env["GDD_ARM"]=arm
        try: env["_gdd_module"]({"mod":None})
        except KeyError: pass
        else: raise AssertionError("missing module accepted")
    env["GDD_ARM"]="ship"
    env["_gdd_prepare"](state)
    for key in ("gd","gd_cat"):
        broken=dict(state); broken[key]=None
        try: env["_gdd_prepare"](broken)
        except RuntimeError: pass
        else: raise AssertionError("incomplete state accepted")
    saved=env["load_inline"]
    env["load_inline"]=lambda **kw: (_ for _ in ()).throw(RuntimeError("injected compile failure"))
    missing=env["_GDD"]["mods"].pop("DQ_ALL")
    try: env["_gdd_prepare"](state)
    except RuntimeError: pass
    else: raise AssertionError("build failure accepted")
    env["load_inline"]=saved; env["_GDD"]["mods"]["DQ_ALL"]=missing
    print("[gdd] CPU dry run PASS: real GD/cat prepare dictionaries, five build sources, PT1 closures/cache, all routes/arms/VF/CAT, invalid/missing modules/build failure")
    return env,state,Tensor


def main():
    assert sys.argv[1:]==[], "python -B -I tools/gdd_patches.py"
    base=BASE.read_text(encoding="utf-8")
    source=build(base,True)
    dry_run(base,source)
    OUT.write_bytes(source.encode())
    print(f"[gdd] {OUT.relative_to(ROOT)} {len(source.encode()):,} B; AST / exact reverse-edit gates PASS")


if __name__=="__main__": main()

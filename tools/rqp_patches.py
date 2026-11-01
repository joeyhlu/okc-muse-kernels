#!/usr/bin/env python3
""":"
set -euo pipefail
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
export PYTHONDONTWRITEBYTECODE=1
exec python -B "$0" --check --linux-gates "$@"
":"""
"""RQP: exact-anchor lab generator and CPU gates; also an LF bash entry point.

python -B tools/rqp_patches.py             # generate ONLY k3_rqp_ab.py
python -B tools/rqp_patches.py --check     # read-only CPU gates
bash tools/rqp_patches.py                 # under cudaenv.sh; CPU + offline CUDA/SASS

The Linux entry point never runs a GPU kernel. Its JSON sidecar and retained /tmp
compiler artifacts are created only when the lead explicitly runs that command.
On a prepared H200 module, call mod._rqp_self_test(state) separately; no timing,
pod, judge or submission is started by this generator.
"""
import argparse
import ast
import contextlib
import difflib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import types

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_rq32_ab.py"
OUT = ROOT / "kernels/lab/k3_rqp_ab.py"

HELPERS = '''
def _rqp_pooler_logits(state,x):
 # The caller is immediately after _v9_bnd; this event precedes the pooler.
 # Copy the T=1 side-stream accessor and capture-aware allocator convention.
 main_stream,side_stream=torch.cuda.current_stream(),state["side_stream"]
 side_stream.wait_stream(main_stream)
 rq_y,rq_logits=None,None
 try:
  rq_y=_rq_gemm(state,"pooler",x)
  if rq_y is not None:
   with torch.cuda.stream(side_stream):
    rq_logits=_v8_logits("stock",x,None,state)
 finally:
  # Also close the graph fork on a rejected launch or Python exception.
  main_stream.wait_stream(side_stream)
 if rq_y is None:
  return None
 if not torch.cuda.is_current_stream_capturing():
  x.record_stream(side_stream)
  rq_logits.record_stream(main_stream)
 return rq_y,rq_logits


def _rqp_self_test(state):
 """Explicit H200 correctness gate; never called from prepare/timed code.

 Uses the base tail self-test RNG sequence, plus outliers and changed-input
 graph replay. Returns observations, including exact logits and top-8 order
 and set equality against the *live* ship VD route. Raises on a correctness
 failure; no numerical claim is made until this function has run on H200.
 """
 global RQP128
 if ("pooler",128)not in state.get("rq_bank",{}):
  raise RuntimeError("RQP pooler bank is not live")
 if state.get("vd")is None or state["vd"]["route"].get(("cat",128))is None:
  raise RuntimeError("RQP comparison requires the live ship VD cat/128")
 saved=RQP128
 rows=[]
 try:
  RQP128=True
  dev=state["dense"]["pooler"].device
  for seed in(4242,4545,12828):
   gen=torch.Generator(device=dev).manual_seed(seed)
   # Same shapes/order/scale as _v8_tail_prepare, including RNG advancement.
   for T,ls in((1,0),(8,64),(16,224),(33,32),(128,96),(256,160)):
    x=(torch.randn((T,HIDDEN),generator=gen,device=dev)*0.5).to(torch.bfloat16)
    if T!=128:
     continue
    data={"local_start":torch.tensor([ls],dtype=torch.int32,device=dev)}
    for outlier in(False,True):
     if outlier:
      x[:,:max(1,HIDDEN//100)]*=30
     got=_rqp_pooler_logits(state,x)
     if got is None:
      raise RuntimeError("RQP fell back in self-test")
     y,lg=got
     serial=_v8_logits("stock",x,None,state)
     cat=_dg1(state,"cat",x)
     if cat is None:
      raise RuntimeError("ship cat fell back in self-test")
     ship_lg=(cat[:,D_EXPERT:D_EXPERT+N_EXPERTS]+cat[:,D_EXPERT+N_EXPERTS:]).contiguous()
     candidate=_v8_tail(y,lg,state,data)
     ship=_v8_tail(cat,None,state,data)
     row={"seed":seed,"outlier":outlier,
      "logits_serial_bitwise":torch.equal(lg,serial),
      "logits_ship_bitwise":torch.equal(lg,ship_lg),
      "logits_ship_max_abs":float((lg-ship_lg).abs().max()),
      "top8_order_equal":torch.equal(candidate[2],ship[2]),
      "top8_set_equal":torch.equal(candidate[2].sort(dim=1).values,ship[2].sort(dim=1).values),
      "local_ids_equal":torch.equal(candidate[3],ship[3]),
      "top8_weight_max_abs":float((candidate[1]-ship[1]).abs().max())}
     assert row["logits_serial_bitwise"],row
     assert row["top8_order_equal"]and row["top8_set_equal"]and row["local_ids_equal"],row
     assert bool(torch.isfinite(y).all())and bool(torch.isfinite(lg).all()),row
     # Warm both streams and cuBLAS before capture; each capture is fork/join closed.
     for _ in range(2):
      _rq_pooler_tail(state,x,data,"cat")
     torch.cuda.synchronize()
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):
      graph_pair=_rqp_pooler_logits(state,x)
      graph_tail=_v8_tail(*graph_pair,state,data)
     for replay in range(3):
      # New input each time detects missing boundary -> side or side -> tail edges.
      x.copy_((torch.randn((128,HIDDEN),generator=gen,device=dev)*0.5).to(torch.bfloat16))
      if outlier:
       x[:,:max(1,HIDDEN//100)]*=30
      graph.replay()
      serial=_v8_logits("stock",x,None,state)
      cat=_dg1(state,"cat",x)
      assert cat is not None
      ship=_v8_tail(cat,None,state,data)
      assert torch.equal(graph_pair[1],serial),(row,"graph logits",replay)
      assert torch.equal(graph_tail[2],ship[2]),(row,"graph top8",replay)
      assert torch.equal(graph_tail[3],ship[3]),(row,"graph local IDs",replay)
      assert int(state["rq_bank"][("pooler",128)]["rq_cnt"].abs().sum())==0
     row["changed_input_graph_replays"]=3
     rows.append(row)
  return {"status":"PASS","rows":rows,"logits_ship_bitwise_all":all(r["logits_ship_bitwise"]for r in rows)}
 finally:
  RQP128=saved

'''

EDITS = [
    ("RQP128 default", 'RQ_PACK=(("qkv",128),("unpooler",128),("qkv",32),("o",32),("unpooler",32))\n',
     'RQP128=False # RQP: only pooler/router scheduling at exactly T=128\n'
     'RQ_PACK=(("qkv",128),("unpooler",128),("qkv",32),("o",32),("unpooler",32),("pooler",128))\n'),
    ("RQP128 pooler selection", ' if rq_key not in frozenset(tuple(x)for x in RQ_SEL):\n',
     ' if rq_key not in frozenset(tuple(x)for x in RQ_SEL)and not(RQP128 and rq_key==("pooler",128)):\n'),
    ("RQP128 fork/join dispatch", 'def _rq_pooler_tail(state,x,data,tail):\n if tail!="cat":\n  return None\n',
     'def _rq_pooler_tail(state,x,data,tail):\n if tail!="cat":\n  return None\n'
     ' if RQP128 and x.shape[0]==128:\n'
     '  rq_pair=_rqp_pooler_logits(state,x)\n'
     '  return None if rq_pair is None else _v8_tail(*rq_pair,state,data)\n'),
    ("RQP128 helpers and explicit self-test", '\n_rq_cuda=r"""\n', HELPERS+'\n_rq_cuda=r"""\n'),
]


def node_named(text, name):
    return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)


def namespace(text):
    # AST extraction avoids importing SGLang, initializing CUDA, or writing pyc.
    nodes = []
    for n in ast.parse(text).body:
        if isinstance(n, ast.FunctionDef) and (n.name.startswith("_rq") or n.name == "_v8_logits"):
            nodes.append(n)
        elif isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and
                (t.id.startswith("_rq") or t.id in ("RQ_SEL", "RQ_PACK", "RQP128")) for t in n.targets):
            nodes.append(n)
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "rqp-extracted", "exec"), ns)
    return ns


def generate():
    base = BASE.read_text(encoding="utf-8")
    text = base
    # Exact count discipline copied from tools/padz_patches.py:48-51.
    for name, old, new in EDITS:
        n = text.count(old)
        assert n == 1, ("anchor", name, n)
        text = text.replace(old, new, 1)
    undo = text
    for name, old, new in reversed(EDITS):
        assert undo.count(new) == 1, ("reverse anchor", name)
        undo = undo.replace(new, old, 1)
    assert undo == base, "non-patch source drift"
    compile(text, str(OUT), "exec")
    assert text.isascii() and "\r" not in text
    old_ns, new_ns = namespace(base), namespace(text)
    assert new_ns["RQP128"] is False and old_ns["RQ_SEL"] == new_ns["RQ_SEL"]
    assert old_ns["_rqsrc"](old_ns["_rq_variants"]) == new_ns["_rqsrc"](new_ns["_rq_variants"])
    assert ast.dump(node_named(base, "kernel_fn")) == ast.dump(node_named(text, "kernel_fn"))
    # All existing CUDA strings, launchers, tail, boundary, and routing code are unchanged.
    old_nodes = {n.targets[0].id:n for n in ast.parse(base).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    new_nodes = {n.targets[0].id:n for n in ast.parse(text).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    strings = [k for k,n in old_nodes.items() if isinstance(n.value, ast.Constant) and isinstance(n.value.value,str)]
    assert all(ast.dump(old_nodes[k]) == ast.dump(new_nodes[k]) for k in strings)
    changes = []
    for tag,a,b,c,d in difflib.SequenceMatcher(None,base.splitlines(),text.splitlines(),autojunk=False).get_opcodes():
        if tag != "equal":
            changes.append({"base_lines":[a+1,b],"lab_lines":[c+1,d],"kind":tag})
    return text, changes


def prepare_dry_run(text):
    """Reuse the established rq32 bank test, changing only names and arms."""
    support = runpy.run_path(str(ROOT / "tools/rq32_patches.py"))
    src = (ROOT / "tools/rq32_patches.py").read_text(encoding="utf-8")
    fn = ast.get_source_segment(src,node_named(src,"dry_run"))
    old = '("pack", shapes["o"])]'
    assert fn.count(old) == 1
    fn = fn.replace(old,'("pack", shapes["o"]), ("pack", shapes["pooler"])]')
    assert fn.count('repeat=3') == 1
    fn = fn.replace('repeat=3','repeat=len(NEW_KEYS)')
    support.update(BASE=BASE, namespace=namespace,
                   NEW_KEYS=(("qkv",32),("o",32),("unpooler",32),("pooler",128)))
    coverage=ast.get_source_segment(src,node_named(src,"protocol_coverage"))
    assert coverage.count('ns["_rq_picks"][key][0]') == 1
    coverage=coverage.replace('ns["_rq_picks"][key][0]',
                              'ns["_rq_picks"].get(key,ns["_rq_picks"].get(key[1]))[0]')
    exec(compile(coverage,"rqp-protocol-coverage","exec"),support)
    exec(compile(fn,"rqp-real-prepare-dry-run","exec"),support)
    return support["dry_run"](text)


def scheduling_dry_run(text):
    """Run actual helper + exact base state expressions with stream/tensor doubles."""
    ns = namespace(text)
    support = runpy.run_path(str(ROOT / "tools/rq_patches.py"))
    BaseTensor = support["_rq_FakeTensor"]
    trace, active, capturing = [], ["main"], [False]
    class Tensor(BaseTensor):
        def record_stream(self, stream):
            assert not capturing[0]
            trace.append(("record",self.value,stream.name))
        def t(self):
            return Tensor(self.shape[::-1],self.dtype,self.device,self.value)
        def __iadd__(self, other):
            assert self.dtype == other.dtype == "f32"
            trace.append(("add_fp32",active[0]))
            return self
    class Stream:
        def __init__(self,name): self.name=name
        def wait_stream(self,other): trace.append(("wait",self.name,other.name))
    main, side = Stream("main"),Stream("side")
    @contextlib.contextmanager
    def stream(s):
        prev=active[0]; active[0]=s.name
        try: yield
        finally: active[0]=prev
    def mm(x,w,out_dtype):
        assert out_dtype == "f32"
        trace.append(("mm",w.value,active[0]))
        return Tensor((x.shape[0],w.shape[1]),"f32",value="logits")
    ns["torch"] = types.SimpleNamespace(bfloat16="bf16",float32="f32",mm=mm,
        empty=lambda shape,dtype,device:Tensor(shape,dtype,device,"pooler"),
        cuda=types.SimpleNamespace(current_stream=lambda:main,stream=stream,Stream=lambda dev:side,
                                  is_current_stream_capturing=lambda:capturing[0]))
    prep=node_named(BASE.read_text(encoding="utf-8"),"_r1_prepare")
    state_expr=next(n.value for n in prep.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="state" for t in n.targets))
    fields={k.value:v for k,v in zip(state_expr.keys,state_expr.values)}
    env={"torch":ns["torch"],"device":"cpu","hi":Tensor((256,8192),value="hi"),"lo":Tensor((256,8192),value="lo")}
    state={k:eval(compile(ast.Expression(fields[k]),"real-base-state","eval"),env)
           for k in ("side_stream","router_hi_t","router_lo_t")}
    p={"rq_k":8192,"rq_n":4096,"rq_device":"cpu"}
    state["rq_bank"]={("pooler",128):p}
    def launch(p,x,out,*args):
        trace.append(("pooler",active[0])); return 0
    ns["_rq_call"]=launch
    def tail(y,lg,st,data):
        assert active[0]=="main"
        trace.append(("tail",active[0])); return y,lg
    ns["_v8_tail"]=tail
    count=0
    for capture in (False,True):
        capturing[0]=capture
        for enabled in (False,True):
            ns["RQP128"]=enabled
            for t in (1,8,16,32,127,128,129,256):
                trace.clear()
                result=ns["_rq_pooler_tail"](state,Tensor((t,8192),value="input"),{},"cat")
                if enabled and t==128:
                    expected=[("wait","side","main"),("pooler","main"),("mm","hi","side"),
                              ("mm","lo","side"),("add_fp32","side"),("wait","main","side")]
                    if not capture: expected += [("record","input","side"),("record","logits","main")]
                    expected += [("tail","main")]
                    assert trace==expected,trace
                    assert result[0].dtype=="bf16" and result[1].dtype=="f32"
                else:
                    assert result is None and not trace,(enabled,t,trace)
                count+=1
        ns["RQP128"]=True
        # Missing bank, rejected launch and exceptions must still close graph fork.
        for failure in ("missing","reject","router"):
            trace.clear()
            saved_call,saved_logits=ns["_rq_call"],ns["_v8_logits"]
            if failure=="missing": state["rq_bank"]={}
            if failure=="reject": ns["_rq_call"]=lambda *a:-1
            if failure=="router": ns["_v8_logits"]=lambda *a:(_ for _ in ()).throw(ValueError("router"))
            try:
                out=ns["_rq_pooler_tail"](state,Tensor((128,8192)),{},"cat")
                assert out is None and failure!="router"
            except ValueError:
                assert failure=="router"
            finally:
                ns["_rq_call"],ns["_v8_logits"]=saved_call,saved_logits
                state["rq_bank"]={("pooler",128):p}
            assert trace[0]==("wait","side","main") and trace[-1]==("wait","main","side"),trace
        # Existing explicit serial RQ_SEL arm remains serial when RQP128=False.
        ns["RQP128"]=False; ns["RQ_SEL"]={("pooler",128)}
        trace.clear()
        assert ns["_rq_pooler_tail"](state,Tensor((128,8192)),{},"cat") is not None
        assert trace==[("pooler","main"),("mm","hi","main"),("mm","lo","main"),("add_fp32","main"),("tail","main")]
        ns["RQ_SEL"]=namespace(text)["RQ_SEL"]
        trace.clear()
        assert ns["_rq_pooler_tail"](state,Tensor((128,8192)),{},"sep") is None and not trace
    return {"status":"PASS","shape_capture_arm_cases":count,"fallback_and_exception_joins":"PASS",
            "real_base_state_fields":{k:fields[k].lineno for k in state if k in fields},
            "serial_hi_lo_expression":"executed base _v8_logits with FP32 doubles",
            "GPU_graph_validation":"NOT RUN; stream/event behavior modeled, not emulated"}


def leaf_gates(text,real_torch=False):
    support=runpy.run_path(str(ROOT / "tools/rq_patches.py"))
    src=(ROOT / "tools/rq_patches.py").read_text(encoding="utf-8")
    name="_rq_torch_cpu" if real_torch else "_rq_leaf_dry_run"
    support["_rq_namespace"]=namespace
    exec(compile(ast.Module(body=[node_named(src,name)],type_ignores=[]),"rqp-leaf-gates","exec"),support)
    return support[name](text)


def linux_gates(text,result):
    """Offline compilation only. Preserve build evidence, never delete anything."""
    import os
    import shutil
    import subprocess
    import sysconfig
    import tempfile
    import re
    import torch
    from torch.utils.cpp_extension import include_paths
    result["torch_cpu"]=leaf_gates(text,True)
    nvcc,cuobjdump=shutil.which("nvcc"),shutil.which("cuobjdump")
    assert nvcc and cuobjdump,"nvcc/cuobjdump missing; run the bash entry under cudaenv.sh"
    ns=namespace(text)
    work=Path(tempfile.mkdtemp(prefix="rqp-compile-"))
    source,obj=work/"rqp.cu",work/"rqp.o"
    cuda=ns["_rqsrc"](ns["_rq_variants"])
    source.write_bytes(cuda.encode("ascii"))
    includes=include_paths(device_type="cuda")+[sysconfig.get_path("include")]
    # Match the proven offline rq32 gate and cudaenv's current torch headers.
    command=[nvcc,*ns["_rq_flags"],"-std=c++20","-Xcompiler","-fPIC","-Xptxas=-v",
             "-DTORCH_EXTENSION_NAME=rqp_compile_gate","-DTORCH_API_INCLUDE_EXTENSION_H",
             "-D_GLIBCXX_USE_CXX11_ABI=%d" % int(torch._C._GLIBCXX_USE_CXX11_ABI)]
    command += ["-I"+p for p in includes]+["-c",str(source),"-o",str(obj)]
    built=subprocess.run(command,capture_output=True,text=True,timeout=600)
    result["compile"]={"returncode":built.returncode,"command":command,"log":built.stdout+built.stderr,"workdir":str(work)}
    assert built.returncode==0,"offline NVCC compile failed; see JSON log"
    sass=subprocess.run([cuobjdump,"--dump-sass",str(obj)],capture_output=True,text=True,timeout=120)
    names=[l.strip() for l in sass.stdout.splitlines() if "Function :" in l]
    assert sass.returncode==0 and names,"cuobjdump failed or empty SASS"
    support=runpy.run_path(str(ROOT / "tools/local/rq32_compile.py"))
    usage=subprocess.run([cuobjdump,"--dump-resource-usage",str(obj)],capture_output=True,text=True,timeout=120)
    assert usage.returncode==0,"cuobjdump resources failed"
    metrics=support["resources"](built.stdout+built.stderr)
    support["local_resources"](usage.stdout,metrics)
    result["resources"]=metrics
    assert len(metrics)==len(ns["_rq_variants"]),"missing compiled instances"
    assert all(m["SPILL_STORE"]==m["SPILL_LOAD"]==m["STACK"]==0 and m["LOCAL"] in (None,0)
               for m in metrics.values()),"local memory/spill gate failed"
    # CUDA 13.4 sometimes omits LOCAL; never infer zero without checking SASS too.
    assert not re.search(r"\bLDL\b|\bSTL\b",sass.stdout),"local-memory SASS"
    assert re.search(r"\bHGMMA\b|\bWGMMA\b",sass.stdout),"no Hopper WGMMA"
    assert not re.search(r"\bBPT\.TRAP\b|\bTRAP\b",sass.stdout),"TRAP instruction"
    result["sass"]={"status":"PASS","functions":names,"sha256":hashlib.sha256(sass.stdout.encode()).hexdigest(),
                    "CUDA_source_identical_to_base":True,"new_device_instructions_per_thread":0}
    result["GPU"]="NOT RUN (offline gates only)"


def main():
    parser=argparse.ArgumentParser(description="RQP generator, CPU gates and offline CUDA/SASS gate (no GPU execution)")
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--linux-gates",action="store_true")
    args=parser.parse_args()
    result={"status":"FAIL"}
    try:
        text,changes=generate()
        result.update(reverse_gate="PASS",all_CUDA_strings_unchanged="PASS",kernel_fn_unchanged="PASS",
                      prepare_dry_run=prepare_dry_run(text),scheduling_dry_run=scheduling_dry_run(text),
                      numpy_leaves=leaf_gates(text),changes=changes,bytes=len(text.encode()),
                      sha256=hashlib.sha256(text.encode()).hexdigest())
        if args.check:
            assert OUT.read_bytes()==text.encode("ascii"),"lab is stale; run generator"
        else:
            OUT.write_bytes(text.encode("ascii"))
        if args.linux_gates:
            linux_gates(text,result)
        result["status"]="PASS"
    except Exception as exc:
        result["error"]=repr(exc)
        raise
    finally:
        if args.linux_gates:
            # Created by the lead's explicitly invoked Linux gate, not by generation.
            Path(__file__).with_suffix(".json").write_bytes((json.dumps(result,indent=2)+"\n").encode())
        print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

"""hya_compile_gate.py (R-qfix, $0, local): compile a lab file's HYA attention module (HYA_CPP_SRC / HYA_CUDA_SRC) for sm_90a with ptxas -v and
print registers / spills / stack of every attn_kernel instantiation, so a patch (tools/hya_q2_patches.py) can be compared against its base.
usage (WSL): source tools/local/cudaenv.sh; python3 tools/hya_compile_gate.py [lab file] [more lab files ...]
"""
import ast
import contextlib
import io
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
files = sys.argv[1:] or ["kernels/lab/k3_q2_ab.py"]
import torch  # noqa: F401,E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

for f in files:
    lab = ROOT / f
    text = lab.read_text(encoding="utf-8")
    srcs = {}
    for n in ast.parse(text).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("HYA_CPP_SRC", "HYA_CUDA_SRC"):
            srcs[n.targets[0].id] = ast.literal_eval(n.value)
    buf = io.StringIO()
    ok = True
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            load_inline(name="hya_gate_" + str(abs(hash(text)) % 10**8), cpp_sources=srcs["HYA_CPP_SRC"], cuda_sources=srcs["HYA_CUDA_SRC"],
                        functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr", "-Xptxas", "-v"],
                        verbose=True)
    except Exception as ex:
        ok = False
        buf.write("\nEXC " + repr(ex)[:3000])
    log = buf.getvalue()
    print(f"== {f}: {'COMPILE_OK' if ok else 'COMPILE_FAIL'}")
    cur = None
    rows = {}
    for line in log.splitlines():
        m = re.search(r"Compiling entry function '(.*?)' for 'sm_90a'", line)
        if m:
            cur = m.group(1)
        m1 = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if cur and m1:
            rows.setdefault(cur, {}).update(stack=int(m1.group(1)), ss=int(m1.group(2)), sl=int(m1.group(3)))
        m2 = re.search(r"Used (\d+) registers", line)
        if cur and m2:
            rows.setdefault(cur, {}).update(regs=int(m2.group(1)))
    for name, r in sorted(rows.items()):
        if "attn_kernel" in name:
            mm = re.search(r"attn_kernelILi(\d+)ELb(\d)(?:ELi(\d+))?E", name)
            tag = f"L={mm.group(1)} flat={mm.group(2)} mode={mm.group(3) or '-'}" if mm else name[:60]
            print(f"   {tag:28s} regs {r.get('regs')} stack {r.get('stack')} spill {r.get('ss')}/{r.get('sl')}")
    if not ok:
        print(log[-3000:])
print("done")

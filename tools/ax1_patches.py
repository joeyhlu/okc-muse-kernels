"""ax1_patches.py: AX1 one-launch B = 1 full attention (tools/ax1_src.py) into a decode-layer file.

    python -I tools/ax1_patches.py --base=kernels/lab/k3_rider3_1.py --out=kernels/lab/k3_ax1_1.py \
        [--l1=ct5:st3:nc8] [--l8=ct5:st3:nc8] [--route=1,8] [--check]
Source-anchored, verbatim-checked insertions (each anchor must occur exactly once):
  * the slim AX1 block (CUDA without comments / indentation, the Python route without the board-only tune and with a
    shorter self-test) before `def _r2_page1(`;
  * its nvcc build in the PT1 pool (`_pt1_start`, beside VFA), `_ax1_prepare` after `_vfa_prepare` in `prepare()`;
  * the route first in `kernel_fn` (B = 1 full, page-1 view `adata`, as the FA4 chain it replaces; KS1 / VFA / FA4 /
    K1 / r1 routes unchanged, FA4 stays the fallback on any AX1 prepare or call failure).
Plans (tiles per split : ring stages : workspace copies) per q_len come from the board-check tune (kernels/results).
--check: $0 gates on the output: AST parse, ASCII, < 1,000,000 B, and the inserted CUDA compiled for sm_90a by the local
toolchain (regs / spills printed); correctness is the prepare() self-test (tools/local/ax1_local_test.py runs q_len 1 on
the RTX 3070).
"""
import argparse
import ast
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIMIT = 1_000_000
FUNCS = ("_ax1_build", "_ax1_ws", "_ax1_plan", "_ax1_call", "_ax1_wanted", "_ax1_ref", "_ax1_case", "_ax1_self_test",
         "_ax1_prepare")
SELF_TEST = ("((1, 3000, 701, False, 0), (1, 80000, 702, True, 0), (1, 1, 704, False, 0), (1, 127, 706, False, 37), "
             "(8, 3000, 711, False, 0), (8, 80000, 712, True, 0), (8, 1, 714, False, 0), (8, 60, 715, False, 0), "
             "(8, 127, 716, False, 37))")


def sub1(text, old, new, what):
    n = text.count(old)
    if n != 1:
        sys.exit(f"[ax1_patches] anchor {what!r}: found {n} times")
    return text.replace(old, new)


def slim_cuda(src):
    out = []
    for ln in src.split("\n"):
        i = ln.find("//")
        if i >= 0:
            ln = ln[:i]
        ln = ln.strip()
        if ln:
            out.append(ln)
    return "\n".join(out) + "\n"


def plan(spec, ntiles_hint):
    ct, st, nc = (int(x[2:]) for x in spec.split(":"))
    return ct, st, nc


def block(args):
    src = (ROOT / "tools" / "ax1_src.py").read_text()
    ns = {}
    exec(compile(src.split("\nAX1_ENABLE")[0], "ax1_src.py", "exec"), ns)
    cuda = slim_cuda(ns["AX1_CUDA_SRC"])
    assert '"' not in "".join(re.findall(r"//[^\n]*", ns["AX1_CUDA_SRC"])), "a // comment holds a quote"
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    route = sorted(int(x) for x in args.route.split(","))
    plans = {1: plan(args.l1, 0), 8: plan(args.l8, 0)}
    # judge widths 80001 / 80008 -> 1251 tiles: splits = ceil(1251 / ct) keeps the tuned tiles per split
    consts = [
        "AX1_ENABLE = True",
        f"AX1_ROUTE = {{1: {1 in route}, 8: {8 in route}}}",
        f"AX1_SPLITS = {{1: {-(-1251 // plans[1][0])}, 8: {-(-1251 // plans[8][0])}}}",
        f"AX1_ST = {{1: {plans[1][1]}, 8: {plans[8][1]}}}",
        f"AX1_NC = {{1: {plans[1][2]}, 8: {plans[8][2]}}}",
        "AX1_PDL = True",
        "AX1_TUNE = False",
        f"AX1_SELF_TEST = {SELF_TEST}",
        "AX1_TOL = {1: 1e-2, 8: 2e-2}",
        '_AX1 = {"mod": None, "ws": {}, "build_s": 0.0}',
    ]
    py = []
    for name in FUNCS:
        fn = fns[name]
        if name == "_ax1_prepare":   # (the tune is board-only: drop its branch)
            fn.body[-1].body = [s for s in fn.body[-1].body if not (isinstance(s, ast.If) and getattr(s.test, "id", "") == "AX1_TUNE")]
        py.append(ast.unparse(fn))
    text = ("AX1_CPP_SRC = r\"\"\"" + ns["AX1_CPP_SRC"] + "\"\"\"\n\nAX1_CUDA_SRC = r\"\"\"\n" + cuda + "\"\"\"\n\n"
            + "\n".join(consts) + "\n\n\n" + "\n\n\n".join(py) + "\n\n\n")
    assert "AX1_TUNE:" not in text and "_ax1_tune" not in text
    return text, cuda


def apply(base, args):
    blk, cuda = block(args)
    t = sub1(base, "\ndef _r2_page1(", "\n" + blk + "def _r2_page1(", "r2_page1")
    t = sub1(t, '    if VFA_ENABLE:\n        jobs.append(("vfa", _vfa_build))\n',
             '    if VFA_ENABLE:\n        jobs.append(("vfa", _vfa_build))\n    if AX1_ENABLE:\n        jobs.append(("ax1", _ax1_build))\n',
             "pt1 jobs")
    t = sub1(t, "    _vfa_prepare(state, t0)\n    torch.cuda.synchronize()\n    _pt1_join_all()\n",
             "    _vfa_prepare(state, t0)\n    _ax1_prepare(state, t0)\n    torch.cuda.synchronize()\n    _pt1_join_all()\n",
             "prepare")
    t = sub1(t, '    o = None\n    ks1_ran = False\n    if _ks1_wanted(qkv, adata, adata["config"], state):\n',
             '    o = None\n    ks1_ran = False\n'
             '    if _ax1_wanted(qkv, adata, adata["config"], state):\n'
             '        try:\n'
             '            o = _ax1_call(qkv, adata, adata["config"]).view(tokens, hq * d)\n'
             '        except Exception as ex:\n'
             '            _k3_disable(state, "ax1", False, "AX1", ex)\n'
             '            o = None\n'
             '    if o is None and _ks1_wanted(qkv, adata, adata["config"], state):\n', "route")
    return t, blk, cuda


def slim_layer(t):
    """--slim (run LAST, after any other builder's patcher, so their anchors still match): drop // comments and
    indentation from top-level CUDA source constants. A constant is left untouched if any string literal elsewhere in
    the file (a potential runtime .replace / .index / count anchor, e.g. _g2_stream_h's) changes its occurrence count
    in it, if a comment ends in a backslash, or if a quote precedes the // on the line."""
    tree = ast.parse(t)
    lits = set()
    cuda_nodes = []
    def const_str(x):   # a literal or a + / implicit concatenation of literals
        if isinstance(x, ast.Constant) and isinstance(x.value, str):
            return x.value
        if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Add):
            a, b = const_str(x.left), const_str(x.right)
            return a + b if a is not None and b is not None else None
        return None

    SURG = {"replace", "index", "rindex", "find", "rfind", "count", "split", "rsplit", "partition", "startswith", "endswith"}
    for n in ast.walk(tree):   # literals used in string surgery: method args, `in` operands, locals assigned in functions
        cands = []
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in SURG:
            cands += n.args
        elif isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops):
            cands.append(n.left)
        elif isinstance(n, ast.FunctionDef):
            cands += [s.value for s in ast.walk(n) if isinstance(s, ast.Assign)]
        for c in cands:
            s = const_str(c)
            if s is not None and 4 <= len(s) < 4000:
                lits.add(s)
    for n in tree.body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
                and ("__global__" in n.value.value or "__device__" in n.value.value)):
            cuda_nodes.append(n)
    lines = t.split("\n")
    edits, saved, skipped = [], 0, []
    for n in cuda_nodes:
        v = n.value.value
        seg_lines = lines[n.lineno - 1:n.end_lineno]
        seg = "\n".join(seg_lines)
        if not (seg.split("=", 1)[1].lstrip().startswith('r"""') and seg.rstrip().endswith('"""') and seg.count('"""') == 2):
            skipped.append((n.targets[0].id, "form"))
            continue
        out = []
        for ln in v.split("\n"):
            i = ln.find("//")
            while i >= 0 and (ln[:i].replace('\\"', "").count('"') % 2 or "'" in ln[:i]):
                i = ln.find("//", i + 2)   # (inside a string / char literal: not a comment)
            if i >= 0 and not ln.rstrip().endswith("\\"):
                ln = ln[:i]
            ln = ln.strip()
            if ln:
                out.append(ln)
        nv = "\n" + "\n".join(out) + "\n"
        moved = [lit for lit in lits if lit != v and lit.strip() and (lit in v or lit in nv) and v.count(lit) != nv.count(lit)]
        if moved:
            skipped.append((n.targets[0].id, "anchor " + repr(moved[0][:40])))
            continue
        name = n.targets[0].id
        edits.append((n.lineno, n.end_lineno, f'{name} = r"""{nv}"""'))
        saved += len(v.encode()) - len(nv.encode())
    for lo, hi, new in sorted(edits, reverse=True):
        lines[lo - 1:hi] = [new]
    t2 = "\n".join(lines)
    ast.parse(t2)
    print(f"[ax1_patches] slim: {len(edits)} CUDA constants stripped, -{saved} B; untouched {skipped}")
    return t2


def gate(t, cuda):
    ast.parse(t)
    n = len(t.encode())
    print(f"[ax1_patches] gates: ast ok, ascii {t.isascii()}, {n} B ({'OK' if n < LIMIT else 'OVER'} the {LIMIT} B cap by "
          f"{abs(LIMIT - n)} B)")
    tmp = ROOT / "box_out" / "ax1_patch_dev.cu"
    dev = cuda.split("void ax1_forward(")[0]
    for inc in ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>"):
        dev = dev.replace(inc, "")
    dev = dev.replace("C10_CUDA_CHECK(", "(").replace("TORCH_CHECK(", "(void)(")
    dev += "".join(f"\ntemplate __global__ void ax1::{k}<{s}>(const __grid_constant__ ax1::Params);"
                   for k, s in (("fb1_l1", 2), ("fb1_l1", 3), ("fb1_l1", 4), ("fb1_l8", 3), ("fb1_l8", 4))) + "\n"
    tmp.write_text(dev, newline="\n")
    wsl = "<repo>/" + tmp.relative_to(ROOT).as_posix()
    r = subprocess.run(["wsl", "-d", "Ubuntu", "-u", "root", "--", "bash", "<repo>/tools/local/cudaenv.sh",
                        "nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v", "-c", wsl,
                        "-o", "/tmp/ax1_patch_dev.o"], capture_output=True, text=True, env={"MSYS_NO_PATHCONV": "1", **__import__("os").environ})
    lines = [ln for ln in (r.stdout + r.stderr).splitlines() if "Used" in ln or "spill" in ln or "error" in ln.lower()]
    print("\n".join(lines))
    ok = r.returncode == 0 and n < LIMIT and t.isascii() and all(" 0 bytes spill" in ln for ln in lines if "spill" in ln)
    print(f"[ax1_patches] gate {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="kernels/lab/k3_rider3_1.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--l1", default="ct5:st3:nc8")
    ap.add_argument("--l8", default="ct5:st3:nc8")
    ap.add_argument("--route", default="1")   # q_len 8 is opt-in: board tune 0a7fa113 has it 0.35 us behind the FA4 chain
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--slim", type=int, default=1, help="1: strip CUDA comments / indentation last (1 MB cap)")
    a = ap.parse_args()
    base = (ROOT / a.base).read_text()
    t, blk, cuda = apply(base, a)
    if a.slim:
        t = slim_layer(t)
    out = ROOT / a.out
    out.write_text(t, newline="\n")
    print(f"[ax1_patches] {a.base} -> {a.out}: {len(t.encode())} B (+{len(t.encode()) - len(base.encode())} B, block "
          f"{len(blk.encode())} B, cuda {len(cuda.encode())} B)")
    if a.check:
        sys.exit(0 if gate(t, cuda) else 1)


if __name__ == "__main__":
    main()

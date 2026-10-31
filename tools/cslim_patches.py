"""cslim_patches.py (R-graft-1 step 1): targeted comment / indent strip of named CUDA source constants that slim_patches leaves
verbatim because generic `in`-operand anchor words ('keys', 'window', 'full', ...) occur in their comments.

  python -I tools/cslim_patches.py --base=<in.py> --out=<out.py> [--names=HYA_CUDA_SRC] [--force]

Only constants whose every use is provably text-insensitive are touched: each Load of the name must be a `cuda_sources=` /
`cpp_sources=` keyword of a call, or an operand of a `+` chain that ends in `.encode()` (the build-cache hash tag); no quoted
reference to the name may exist anywhere in the file (no globals()[...] dispatch). The strip itself is slim_patches.strip_cuda
(// comments, indentation, trailing blanks, empty lines) with only the license-notice lines protected (R-slim-1).
Gates (abort, nothing written): C token stream identical (slim_patches.c_tokens), every license block of the constant kept
byte-for-byte, the Python AST equals the base's with only the constants' values swapped, ascii, < 1,000,000 B, raw-string
form round-trips. The SASS gate is tools/local/cslim_sass.py (WSL): base vs slim cubin SASS byte-identical.
"""
import argparse
import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import slim_patches as sp  # noqa: E402

LIMIT = 1_000_000


def parents(tree):
    par = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            par[c] = n
    return par


def use_ok(node, par):
    """True when this Load of the constant cannot observe its text layout."""
    p = par.get(node)
    if isinstance(p, ast.keyword) and p.arg in ("cuda_sources", "cpp_sources"):
        return True
    while isinstance(p, ast.BinOp) and isinstance(p.op, ast.Add):
        p = par.get(p)
    if isinstance(p, ast.Attribute) and p.attr == "encode":
        q = par.get(p)
        return isinstance(q, ast.Call) and q.func is p and not q.args and not q.keywords
    return False


def build(t, names):
    tree = ast.parse(t)
    par = parents(tree)
    lines = t.split("\n")
    edits, vals, report = [], {}, []
    for name in names:
        nodes = [n for n in tree.body if sp.is_cuda_const(n) and n.targets[0].id == name]
        if len(nodes) != 1:
            raise SystemExit(f"[cslim] {name}: {len(nodes)} top-level CUDA constant(s), expected 1")
        n = nodes[0]
        if f'"{name}"' in t or f"'{name}'" in t:
            raise SystemExit(f"[cslim] {name}: quoted reference in the file (dynamic lookup possible)")
        stores = [x for x in ast.walk(tree) if isinstance(x, ast.Name) and x.id == name and not isinstance(x.ctx, ast.Load)]
        if len(stores) != 1:
            raise SystemExit(f"[cslim] {name}: {len(stores)} assignments, expected 1")
        loads = [x for x in ast.walk(tree) if isinstance(x, ast.Name) and x.id == name and isinstance(x.ctx, ast.Load)]
        bad = [x.lineno for x in loads if not use_ok(x, par)]
        if bad or not loads:
            raise SystemExit(f"[cslim] {name}: text-sensitive use(s) at lines {bad} (loads {len(loads)})")
        v = n.value.value
        seg = "\n".join(lines[n.lineno - 1:n.end_lineno])
        rhs = seg.split("=", 1)[1].lstrip()
        if not rhs.startswith('r"""') or not seg.endswith('"""') or seg.count('"""') != 2:
            raise SystemExit(f"[cslim] {name}: not a plain r\"\"\" constant")
        lic_blocks = sp.c_license_blocks(v)
        lic = {k for a, b, _ in lic_blocks for k in range(a, b + 1)}
        nv = sp.strip_cuda(v, set(lic))
        if sp.c_tokens(nv) != sp.c_tokens(v):
            raise SystemExit(f"[cslim] {name}: token gate FAIL")
        lost = sp.license_missing(lic_blocks, [nv])
        if lost:
            raise SystemExit(f"[cslim] {name}: license gate FAIL: {lost[0][:80]!r}")
        if '"""' in nv or nv.endswith("\\"):
            raise SystemExit(f"[cslim] {name}: stripped text cannot be a raw triple-quoted literal")
        vals[name] = nv
        edits.append((n.lineno, n.end_lineno, f'{name} = r"""{nv}"""'))
        report.append(f"{name}: {len(v.encode()):,d} -> {len(nv.encode()):,d} B (-{len(v.encode()) - len(nv.encode()):,d}); "
                      f"{len(lic_blocks)} notice block(s) kept; uses at lines {sorted(x.lineno for x in loads)}")
    for lo, hi, new in sorted(edits, reverse=True):
        lines[lo - 1:hi] = [new]
    t2 = "\n".join(lines)
    exp = ast.parse(t)
    for n in exp.body:
        if sp.is_cuda_const(n) and n.targets[0].id in vals:
            n.value.value = vals[n.targets[0].id]
    if ast.dump(exp) != ast.dump(ast.parse(t2)):
        raise SystemExit("[cslim] AST gate FAIL")
    if t.isascii() and not t2.isascii():
        raise SystemExit("[cslim] ascii gate FAIL")
    if len(t2.encode()) >= LIMIT:
        raise SystemExit(f"[cslim] size gate FAIL: {len(t2.encode()):,d} B")
    return t2, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="HYA_CUDA_SRC")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    src, dst = pathlib.Path(a.base), pathlib.Path(a.out)
    if dst.exists() and not a.force:
        raise SystemExit(f"[cslim] {dst} exists (use --force)")
    t = src.read_bytes().decode("utf-8")
    t2, report = build(t, [x for x in a.names.split(",") if x])
    for r in report:
        print("   " + r)
    dst.write_bytes(t2.encode("utf-8"))
    b0, b1 = len(t.encode()), len(t2.encode())
    print(f"[cslim] {src} -> {dst}: {b0:,d} -> {b1:,d} B (-{b0 - b1:,d}); gates ast / tokens / license / ascii PASS; "
          f"under {LIMIT:,d} B by {LIMIT - b1:,d}")


if __name__ == "__main__":
    main()

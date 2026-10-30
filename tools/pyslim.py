"""pyslim: drop Python comments (outside strings) from a generated layer file to stay under the 1 MB cap.

Usage: python -I tools/pyslim.py <in.py> <out.py> [--keep=N] [--inline]
Keeps lines 1..N (default 2: the okc header + nonce). --inline also cuts trailing `code  # comment` comments.
Gates: (1) ast.dump(in) == ast.dump(out) (comments are not in the AST, so code and every string constant, CUDA sources
included, are unchanged); (2) every decorated def re-parses from its own source text exactly as before, the way CuTe DSL /
Triton re-read it (inspect.getblock + textwrap.dedent): an AST-identical edit can still break source-reading decorators
(the removed --reindent left column-0 continuation lines inside @cute.kernel defs -> FA4 compile failed silently -> FA3
fallback at v1f / v16s / v32s, +15-20 us; b26 / b27, R-ax1-4). Refuses to overwrite an existing output.
License / provenance notices are never dropped (R-slim-1): every contiguous comment block with a line matching
slim_patches.LIC_RE (Copyright / License / SPDX / Apache / BSD / MIT / 'Redistribution and use' / 'derived from' /
'Permission is hereby granted' / NOTICE) is kept verbatim, and gate (3) checks that each such block of the input is present
in the output byte-for-byte (the FlashAttention-4 BSD-3-Clause notice was dropped before this).
"""
import ast
import inspect
import io
import pathlib
import sys
import textwrap
import tokenize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # python -I has no script dir on sys.path
from slim_patches import license_missing, py_license_blocks  # noqa: E402


def decorated_sources(t):
    """ast.dump of every decorator-started block, re-read from source text like inspect.getsource + dedent."""
    lines = t.splitlines(True)
    starts = sorted(min(d.lineno for d in n.decorator_list) - 1 for n in ast.walk(ast.parse(t))
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.decorator_list)
    out = []
    for i in starts:   # real decorators only (not `@p bra` lines inside PTX strings)
        try:
            out.append(ast.dump(ast.parse(textwrap.dedent("".join(inspect.getblock(lines[i:]))))))
        except (SyntaxError, IndentationError, tokenize.TokenError) as ex:
            out.append(f"UNPARSABLE {i + 1}: {ex!r}")
    return out


def main():
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    keep, inline = 2, False
    for a in sys.argv[3:]:
        if a.startswith("--keep="):
            keep = int(a.split("=", 1)[1])
        elif a == "--inline":   # tokenizer positions, so never inside strings
            inline = True
        else:
            sys.exit(f"[pyslim] unknown option {a} (--reindent was removed: it breaks CuTe / Triton source re-parsing)")
    if dst.exists():
        sys.exit(f"[pyslim] {dst} exists; refusing to overwrite")
    t = src.read_text(encoding="utf-8")
    lines = t.splitlines(keepends=True)
    drop, cut = set(), {}
    lic = py_license_blocks(t)
    lic_rows = {r for a, b, _ in lic for r in range(a, b + 1)}
    for tok in tokenize.generate_tokens(io.StringIO(t).readline):
        if tok.type != tokenize.COMMENT:
            continue
        r, c = tok.start
        if r <= keep or r in lic_rows:
            continue
        if lines[r - 1][:c].strip() == "":
            drop.add(r)
        elif inline:
            cut[r] = c
    for r, c in cut.items():
        ln = lines[r - 1]
        eol = ln[len(ln.rstrip("\r\n")):]
        lines[r - 1] = ln[:c].rstrip() + eol
    out = "".join(ln for i, ln in enumerate(lines, 1) if i not in drop)
    if ast.dump(ast.parse(t)) != ast.dump(ast.parse(out)):
        sys.exit("[pyslim] AST changed; abort")
    lost = license_missing(lic, [out])
    if lost:
        sys.exit(f"[pyslim] license gate FAIL: {len(lost)} notice block(s) not preserved byte-for-byte, first: {lost[0][:80]!r}; abort")
    d0, d1 = decorated_sources(t), decorated_sources(out)
    bad = [s for s in d1 if s.startswith("UNPARSABLE")]
    if d0 != d1 or bad:
        sys.exit(f"[pyslim] decorated-def sources changed ({len(d0)} -> {len(d1)}, unparsable {bad[:3]}); abort")
    dst.write_text(out, encoding="utf-8", newline="")
    print(f"[pyslim] {src} -> {dst}: {len(t.encode())} -> {len(out.encode())} B (-{len(t.encode()) - len(out.encode())}), "
          f"{len(drop)} comment lines dropped, {len(cut)} trailing comments cut, AST identical, "
          f"{len(d1)} decorated defs re-parse identically, {len(lic)} license notice blocks kept "
          f"({sum(len(x[2].encode()) + 1 for x in lic)} B verbatim)")


if __name__ == "__main__":
    main()

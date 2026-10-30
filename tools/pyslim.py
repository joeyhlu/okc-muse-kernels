"""pyslim: drop full-line Python comments (outside strings) from a generated layer file to stay under the 1 MB cap.

Usage: python -I tools/pyslim.py <in.py> <out.py> [--keep=N] [--inline] [--reindent]
Keeps lines 1..N (default 2: the okc header + nonce). Gate: ast.dump(in) == ast.dump(out) (comments are not in the AST,
so code and every string constant, CUDA sources included, are unchanged); refuses to overwrite an existing output.
"""
import ast
import io
import pathlib
import sys
import tokenize


def compact_indent(t, keep):
    """Rows inside or ending a multi-line string stay verbatim. Logical-line starts get rank(width) spaces (the distinct
    widths in use, ranked: injective and monotone, so Python's block structure is unchanged); continuation rows inside
    brackets or after a backslash lose their leading whitespace; blank rows outside strings are dropped."""
    lines = t.splitlines(keepends=True)
    frozen, starts, cont = set(), {}, set()
    depth, prev = 0, None
    for tok in tokenize.generate_tokens(io.StringIO(t).readline):
        (r0, c0), (r1, _) = tok.start, tok.end
        if tok.type == tokenize.STRING and r1 > r0:
            frozen.update(range(r0 + 1, r1 + 1))
        if tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER,
                        tokenize.ENCODING):
            prev = tok if tok.type in (tokenize.NL, tokenize.NEWLINE) else prev
            continue
        if prev is None or prev.end[0] < r0:   # first token on its row
            if r0 not in frozen:
                if depth == 0 and (prev is None or prev.type in (tokenize.NEWLINE, tokenize.NL)):
                    starts[r0] = c0
                else:
                    cont.add(r0)
        if tok.type == tokenize.OP and tok.string in "([{":
            depth += 1
        elif tok.type == tokenize.OP and tok.string in ")]}":
            depth -= 1
        prev = tok
    rank = {w: i for i, w in enumerate(sorted(set(starts.values())))}
    out = []
    for i, ln in enumerate(lines, 1):
        if i <= keep or i in frozen:
            out.append(ln)
        elif i in starts:
            out.append(" " * rank[starts[i]] + ln.lstrip(" "))
        elif i in cont:
            out.append(ln.lstrip(" "))
        elif ln.strip() == "":
            continue
        else:
            out.append(ln)
    return "".join(out)


def main():
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    keep, inline, reindent = 2, False, False
    for a in sys.argv[3:]:
        if a.startswith("--keep="):
            keep = int(a.split("=", 1)[1])
        elif a == "--inline":   # also cut trailing `code  # comment` comments (tokenizer positions, so never inside strings)
            inline = True
        elif a == "--reindent":   # block indents -> 1 space per level, continuation lines unindented, blank lines dropped
            reindent = True
    if dst.exists():
        sys.exit(f"[pyslim] {dst} exists; refusing to overwrite")
    t = src.read_text(encoding="utf-8")
    lines = t.splitlines(keepends=True)
    drop, cut = set(), {}
    for tok in tokenize.generate_tokens(io.StringIO(t).readline):
        if tok.type != tokenize.COMMENT:
            continue
        r, c = tok.start
        if r <= keep:
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
    if reindent:
        out = compact_indent(out, keep)
    if ast.dump(ast.parse(t)) != ast.dump(ast.parse(out)):
        sys.exit("[pyslim] AST changed; abort")
    dst.write_text(out, encoding="utf-8", newline="")
    print(f"[pyslim] {src} -> {dst}: {len(t.encode())} -> {len(out.encode())} B (-{len(t.encode()) - len(out.encode())}), "
          f"{len(drop)} comment lines dropped, {len(cut)} trailing comments cut, AST identical")


if __name__ == "__main__":
    main()

"""pyslim: drop full-line Python comments (outside strings) from a generated layer file to stay under the 1 MB cap.

Usage: python -I tools/pyslim.py <in.py> <out.py> [--keep=N]
Keeps lines 1..N (default 2: the okc header + nonce). Gate: ast.dump(in) == ast.dump(out) (comments are not in the AST,
so code and every string constant, CUDA sources included, are unchanged); refuses to overwrite an existing output.
"""
import ast
import io
import pathlib
import sys
import tokenize


def main():
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    keep = 2
    for a in sys.argv[3:]:
        if a.startswith("--keep="):
            keep = int(a.split("=", 1)[1])
    if dst.exists():
        sys.exit(f"[pyslim] {dst} exists; refusing to overwrite")
    t = src.read_text(encoding="utf-8")
    lines = t.splitlines(keepends=True)
    drop = set()
    for tok in tokenize.generate_tokens(io.StringIO(t).readline):
        if tok.type != tokenize.COMMENT:
            continue
        r, c = tok.start
        if r > keep and lines[r - 1][:c].strip() == "":
            drop.add(r)
    out = "".join(ln for i, ln in enumerate(lines, 1) if i not in drop)
    if ast.dump(ast.parse(t)) != ast.dump(ast.parse(out)):
        sys.exit("[pyslim] AST changed; abort")
    dst.write_text(out, encoding="utf-8", newline="")
    print(f"[pyslim] {src} -> {dst}: {len(t.encode())} -> {len(out.encode())} B (-{len(t.encode()) - len(out.encode())}), "
          f"{len(drop)} comment lines dropped, AST identical")


if __name__ == "__main__":
    main()

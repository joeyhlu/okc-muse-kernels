"""strip_cxx.py: drop full-line `//` comments inside the embedded C++ / CUDA source strings of a generated layer file.

Complements tools/strip_layer.py (which never touches string literals). Run it after strip_layer.

What is removed: an interior line of a multi-line, non-f, non-bytes string literal whose first non-blank characters are `//`.
What is kept, byte for byte:
  * lines 1-2 (`# okc: problem=...`, `# nonce ...`) and every token outside string literals;
  * the first and last line of every string literal;
  * a `//` line that ends with a backslash, or follows a line ending with a backslash (line splicing would change meaning);
  * a `//` line that mentions a license or copyright (strip_layer.LICENSE).
Check (`equivalent`): ast.dump of the output equals ast.dump of the original after applying the same line filter to every
plain string constant at the value level (f-string parts untouched), and the header lines are unchanged. Any escape sequence
that makes token lines and value lines disagree fails the check, so the tool refuses rather than guesses.

    python -I tools/strip_cxx.py in.py [out.py]
"""
import ast
import io
import re
import sys
import tokenize

sys.path.insert(0, __file__.rsplit("tools", 1)[0] + "tools")
from strip_layer import LICENSE, KEEP_LINES  # noqa: E402

CXX_COMMENT = re.compile(r"[ \t]*//")


def _kept(lines):
    """lines: the lines of one string (keepends). Returns the indices of the kept lines."""
    keep = []
    n = len(lines)
    for i, l in enumerate(lines):
        body = l.rstrip("\r\n")
        drop = (0 < i < n - 1 and CXX_COMMENT.match(body) and not body.endswith("\\")
                and not lines[i - 1].rstrip("\r\n").endswith("\\") and not LICENSE.search(body))
        if not drop:
            keep.append(i)
    return keep


def _filter(lines):
    return [lines[i] for i in _kept(lines)]


def strip(text):
    lines = text.splitlines(keepends=True)
    drop = set()
    for t in tokenize.generate_tokens(io.StringIO(text).readline):
        if t.type != tokenize.STRING or t.start[0] == t.end[0]:
            continue
        prefix = re.match(r"[A-Za-z]*", t.string).group(0).lower()
        if "f" in prefix or "b" in prefix:
            continue
        sr, er = t.start[0], t.end[0]
        seg = lines[sr - 1:er]
        kept_ids = set(_kept(seg))
        for k in range(len(seg)):
            if k not in kept_ids and sr + k > KEEP_LINES:
                drop.add(sr + k)
    return "".join(l for k, l in enumerate(lines, 1) if k not in drop)


def _reference(tree):
    in_fstr = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for v in ast.walk(node):
                in_fstr.add(id(v))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in in_fstr and "\n" in node.value:
            node.value = "".join(_filter(node.value.splitlines(keepends=True)))
    return tree


def equivalent(orig, out):
    a = _reference(ast.parse(orig))
    b = ast.parse(out)
    if ast.dump(a, include_attributes=False) != ast.dump(b, include_attributes=False):
        return False, "AST differs from the value-level reference filter"
    if out.split("\n")[:KEEP_LINES] != orig.split("\n")[:KEEP_LINES]:
        return False, "header lines changed"
    return True, "AST-equivalent to the value-level // line filter; header kept"


if __name__ == "__main__":
    src = open(sys.argv[1], encoding="utf-8").read()
    out = strip(src)
    ok, msg = equivalent(src, out)
    print(f"{len(src.encode())} -> {len(out.encode())} B ({len(src.encode()) - len(out.encode())} saved); {msg}")
    if not ok:
        sys.exit(1)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
            f.write(out)

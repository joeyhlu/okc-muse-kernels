"""strip_cxx.py: drop full-line `//` comments inside the embedded C++ / CUDA source strings of a generated layer file.

Complements tools/strip_layer.py (which never touches string literals). Run it after strip_layer.

What is removed: an interior line of a multi-line, non-f, non-bytes string literal whose first non-blank characters are `//`.
What is kept, byte for byte:
  * lines 1-2 (`# okc: problem=...`, `# nonce ...`) and every token outside string literals;
  * the first and last line of every string literal;
  * a `//` line that ends with a backslash, or follows a line ending with a backslash (line splicing would change meaning);
  * a `//` line that mentions a license or copyright (strip_layer.LICENSE);
  * a `//` line containing a string the Python code searches text for (first argument of str.index / find / split / replace /
    ... or the left side of an `in` test; e.g.
    the CUDA source for it, e.g. `p1_src.index("// D = 1: <= 64 registers")` in G2; dropping such a line broke G2 once).
Check (`equivalent`): ast.dump of the output equals ast.dump of the original after applying the same line filter to every
plain string constant at the value level (f-string parts untouched); the header lines are unchanged; and every single-line
string constant (>= 4 chars) occurs in the multi-line constants exactly as often as before. Any escape sequence that makes
token lines and value lines disagree fails the check, so the tool refuses rather than guesses.

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
NL = chr(10)
CR = chr(13)
SEP = chr(0)
BACKSLASH = chr(92)


SEARCH_METHODS = {"index", "rindex", "find", "rfind", "split", "rsplit", "partition", "rpartition", "replace", "count",
                  "startswith", "endswith"}


def _probes(text):
    """String constants the Python code searches text for: the first argument of a str search/split/replace method,
    or the left side of an `in` / `not in` test. A line holding one is never dropped, and its count must not change."""
    out = set()
    for n in ast.walk(ast.parse(text)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in SEARCH_METHODS and n.args
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
            out.add(n.args[0].value)
        elif (isinstance(n, ast.Compare) and isinstance(n.left, ast.Constant) and isinstance(n.left.value, str)
              and any(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops)):
            out.add(n.left.value)
    return {q for q in out if q.strip()}


def _kept(lines, probes=()):
    """lines: the lines of one string (keepends). Returns the indices of the kept lines."""
    keep = []
    n = len(lines)
    for i, l in enumerate(lines):
        body = l.rstrip(CR + NL)
        drop = (0 < i < n - 1 and CXX_COMMENT.match(body) and not body.endswith(BACKSLASH)
                and not lines[i - 1].rstrip(CR + NL).endswith(BACKSLASH) and not LICENSE.search(body)
                and not any(q in body for q in probes))
        if not drop:
            keep.append(i)
    return keep


def _filter(lines, probes=()):
    return [lines[i] for i in _kept(lines, probes)]


def strip(text):
    lines = text.splitlines(keepends=True)
    probes = sorted(_probes(text))
    drop = set()
    for t in tokenize.generate_tokens(io.StringIO(text).readline):
        if t.type != tokenize.STRING or t.start[0] == t.end[0]:
            continue
        prefix = re.match(r"[A-Za-z]*", t.string).group(0).lower()
        if "f" in prefix or "b" in prefix:
            continue
        sr, er = t.start[0], t.end[0]
        seg = lines[sr - 1:er]
        kept_ids = set(_kept(seg, probes))
        for k in range(len(seg)):
            if k not in kept_ids and sr + k > KEEP_LINES:
                drop.add(sr + k)
    return "".join(l for k, l in enumerate(lines, 1) if k not in drop)


def _reference(tree, probes=()):
    in_fstr = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for v in ast.walk(node):
                in_fstr.add(id(v))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in in_fstr and NL in node.value:
            node.value = "".join(_filter(node.value.splitlines(keepends=True), probes))
    return tree


def _multiline_text(text):
    return SEP.join(n.value for n in ast.walk(ast.parse(text))
                    if isinstance(n, ast.Constant) and isinstance(n.value, str) and NL in n.value)


def equivalent(orig, out):
    probes = _probes(orig)
    a = _reference(ast.parse(orig), sorted(probes))
    b = ast.parse(out)
    if ast.dump(a, include_attributes=False) != ast.dump(b, include_attributes=False):
        return False, "AST differs from the value-level reference filter"
    if out.split(NL)[:KEEP_LINES] != orig.split(NL)[:KEEP_LINES]:
        return False, "header lines changed"
    mo, mn = _multiline_text(orig), _multiline_text(out)
    lost = [q for q in probes if mo.count(q) != mn.count(q)]
    if lost:
        return False, f"{len(lost)} searched-for constant(s) changed count in the kernel sources, e.g. {lost[0]!r}"
    return True, "AST-equivalent to the value-level // line filter; header kept; every searched-for constant's count kept"


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

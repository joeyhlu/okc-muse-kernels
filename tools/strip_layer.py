"""strip_layer.py: AST-safe size reduction of a generated layer file (comments and docstrings only).

What is removed:
  * `#` comments. A comment line becomes empty and is dropped; an inline comment is cut together with the whitespace before it.
  * Docstrings: the first statement of a module / class / function body when it is a plain string expression. A body that held
    only its docstring gets `pass`.
What is kept, byte for byte:
  * lines 1-2 (`# okc: problem=...`, `# nonce ...`);
  * every comment block (a run of consecutive comment-only lines) and every docstring that mentions a license or copyright
    (the regex LICENSE);
  * every other token: string literals (the CUDA / C++ kernel sources) are never touched, so kernel sources stay byte-identical.
Check (`equivalent`): ast.dump of the stripped text equals ast.dump of the original with the same docstring removal applied in
the AST, and the multiset of string constants other than the removed docstrings is identical.

    python -I tools/strip_layer.py in.py out.py
"""
import ast
import io
import re
import sys
import tokenize

LICENSE = re.compile(r"(?i)copyright|licen[cs]e|spdx|redistribution|all rights reserved")
KEEP_LINES = 2


def _docstring_nodes(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                out.append((node, body[0]))
    return out


def _strip_ast(tree):
    """The AST-side reference transformation: drop non-license docstrings (pass for emptied bodies)."""
    for node, doc in _docstring_nodes(tree):
        if LICENSE.search(doc.value.value):
            continue
        node.body = node.body[1:] or ([] if isinstance(node, ast.Module) else [ast.Pass()])
    return tree


def strip(text):
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    # 1. docstrings: (start line, start col, end line, end col), 1-based lines
    edits = []   # (lineno, col, end_lineno, end_col, replacement)
    for node, doc in _docstring_nodes(tree):
        if LICENSE.search(doc.value.value):
            continue
        rep = "pass" if (len(node.body) == 1 and not isinstance(node, ast.Module)) else ""
        edits.append((doc.lineno, doc.col_offset, doc.end_lineno, doc.end_col_offset, rep))
    # 2. comments (tokenize never reports comments inside strings)
    toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    comment_only = {}
    for t in toks:
        if t.type == tokenize.COMMENT:
            row, col = t.start
            comment_only[row] = lines[row - 1][:col].strip() == ""
    keep_rows = set(range(1, KEEP_LINES + 1))
    rows = sorted(comment_only)
    i = 0
    while i < len(rows):   # comment blocks: consecutive comment-only rows; keep a block that mentions a license
        j = i
        if comment_only[rows[i]]:
            while j + 1 < len(rows) and rows[j + 1] == rows[j] + 1 and comment_only[rows[j + 1]]:
                j += 1
            block = rows[i:j + 1]
            if any(LICENSE.search(lines[r - 1]) for r in block):
                keep_rows.update(block)
        elif LICENSE.search(lines[rows[i] - 1]):
            keep_rows.add(rows[i])
        i = j + 1
    for t in toks:
        if t.type == tokenize.COMMENT and t.start[0] not in keep_rows:
            row, col = t.start
            edits.append((row, col, row, t.end[1], None))   # None = cut comment and the whitespace before it
    # apply edits from the end (they never overlap: comments are outside strings, docstrings are strings)
    edits.sort(key=lambda e: (e[0], e[1]), reverse=True)
    drop = set()
    for ln, col, eln, ecol, rep in edits:
        if rep is None:   # comment
            s = lines[ln - 1]
            body, nl = (s[:-1], "\n") if s.endswith("\n") else (s, "")
            head = body[:col].rstrip()
            if head == "":
                drop.add(ln)
            else:
                lines[ln - 1] = head + body[ecol:] + nl
        else:             # docstring (may span lines)
            first, last = lines[ln - 1], lines[eln - 1]
            prefix, suffix = first[:col], last[ecol:]
            new = prefix + rep + suffix
            for r in range(ln, eln + 1):
                lines[r - 1] = ""
            if new.strip() == "" or (rep == "" and suffix.strip() == "" and prefix.strip() == ""):
                drop.update(range(ln, eln + 1))
            else:
                lines[ln - 1] = new
    out = "".join(l for k, l in enumerate(lines, 1) if k not in drop)
    return out


def _consts(tree):
    return sorted(repr(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)))


def equivalent(orig, stripped):
    """(ok, message): stripped == original up to the documented removals."""
    a = _strip_ast(ast.parse(orig))
    b = ast.parse(stripped)
    if ast.dump(a, include_attributes=False) != ast.dump(b, include_attributes=False):
        return False, "AST differs beyond docstring removal"
    if _consts(a) != _consts(b):
        return False, "string constants differ"
    if stripped.split("\n")[:KEEP_LINES] != orig.split("\n")[:KEEP_LINES]:
        return False, "header lines changed"
    return True, "AST-equivalent (docstrings removed), every other string constant identical, header kept"


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

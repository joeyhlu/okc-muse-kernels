"""slim_patches.py (R-slim-0): deterministic size diet for a decode-layer bundle. Run it LAST, after every other patcher.

  python -I tools/slim_patches.py --base=<in.py> --out=<out.py> [--keep=2] [--no-cuda] [--no-dead] [--no-py] [--no-ifs]

Removes only text that cannot change what runs (no compression, plain top-level source, A19):
  d  CUDA constants: // comments, indentation, trailing blanks, empty lines (top-level str constants holding __global__ /
     __device__ / #include). Lines holding (or next to) an anchor literal whose occurrence count in the constant would
     change are kept verbatim. Anchors = literals >= 4 chars passed to str surgery (.replace / .index / .count / ...),
     `in` operands, and literals assigned to names used that way (_g2_stream_h's head / wait, '#if VFA_LAB', @INST@ ...).
  a  CUDA #if / #elif / #else blocks whose condition only uses macros with an #ifndef default that NO -D anywhere in the
     file (flags, route tables, define tuples, string surgery) ever overrides and that no other #define / #undef touches;
     the dead branch and the directive lines go, the live branch stays.
  b  top-level defs / pure constant assignments that nothing references (names, attributes, identifier-like strings
     < 300 chars, so getattr / globals().get dispatch counts), to a fixpoint from every non-def statement + the judge API.
  p  Python full-line comments, inline comments and blank lines outside strings (lines 1..keep untouched).
Gates (abort, nothing written, on any failure): 'd' leaves every CUDA constant's C token stream unchanged (comments
dropped, whitespace-insensitive except directive line ends); constants with __LINE__ / __FILE__ / assert( / R"( / #line
are not touched; the Python AST equals the base's with only the removed defs gone and the slimmed constants' values
swapped; every anchor keeps its count in every constant; ascii. The SASS gate (tools/local/slim_sass.py, WSL) captures
the sources the file hands to load_inline at runtime, base vs slim, and compiles them ('a' is proven there).
Measured on k3_b24_v (997,861 B): -70,064 B (d -63,220, a -205, b -751, p -5,888); idempotent.
"""
import argparse
import ast
import io
import pathlib
import re
import sys
import tokenize

LIMIT = 1_000_000
API = {"prepare", "kernel_fn", "Weights", "Config", "Inputs"}
IDRE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PUNCT = sorted("""<<= >>= ... -> ++ -- << >> <= >= == != && || :: ## += -= *= /= %= &= |= ^= .*
               ->*""".split(), key=len, reverse=True)
TOKRE = re.compile(r'(?:L|u8|u|U)?"(?:\\.|[^"\\\n])*"|(?:L|u8|u|U)?\'(?:\\.|[^\'\\\n])*\'|\.?\d(?:[eEpP][+-]|[\w.])*|'
                   r"[A-Za-z_]\w*|" + "|".join(re.escape(p) for p in PUNCT) + r"|\S")


# ---------------------------------------------------------------- C token streams (the SASS-neutrality proof on text)
def c_decomment(src):
    """Splice backslash-newlines, replace comments by a space (newlines inside block comments kept), keep literals."""
    s = src.replace("\\\n", "")
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in "\"'":
            j = i + 1
            while j < n and s[j] != c and s[j] != "\n":
                j += 2 if s[j] == "\\" else 1
            out.append(s[i:j + 1])
            i = j + 1
        elif s.startswith("//", i):
            j = s.find("\n", i)
            i = n if j < 0 else j
            out.append(" ")
        elif s.startswith("/*", i):
            j = s.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(" " + "\n" * s.count("\n", i, j))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def c_tokens(src):
    """Whitespace-insensitive token stream; a directive (# first on a line) is one ('#', tokens...) item."""
    toks = []
    for ln in c_decomment(src).split("\n"):
        t = TOKRE.findall(ln)
        if t and t[0] == "#":
            toks.append(("#",) + tuple(t[1:]))
        else:
            toks.extend(t)
    return toks


# ---------------------------------------------------------------- d: CUDA comments / whitespace
def strip_cuda(v, protect):
    """protect: set of 0-based line indices kept verbatim."""
    lines = v.split("\n")
    out = []
    for i, ln in enumerate(lines):
        if i in protect:
            out.append(ln)
            continue
        cont_in = i > 0 and lines[i - 1].endswith("\\")
        cont_out = ln.endswith("\\")
        if not cont_out:
            j = ln.find("//")
            while j >= 0 and (ln[:j].replace('\\"', "").count('"') % 2 or "'" in ln[:j] or "/*" in ln[:j]):
                j = ln.find("//", j + 2)   # inside a string / char literal or after a block comment opener: keep
            if j >= 0:
                ln = ln[:j]
            ln = ln.rstrip()
        s = ln.lstrip()
        if cont_in and s != ln:
            s = " " + s      # a continued line keeps one separator
        if s or cont_in or cont_out:
            out.append(s)
    nv = "\n".join(out)
    if v.startswith("\n") and not nv.startswith("\n"):
        nv = "\n" + nv
    if v.endswith("\n") and not nv.endswith("\n"):
        nv += "\n"
    return nv


def literal_counts_ok(v, nv, lits):
    return [lit for lit in lits if (lit in v or lit in nv) and v.count(lit) != nv.count(lit)]


def protect_lines(v, moved):
    prot = set()
    starts = [0]
    for ln in v.split("\n"):
        starts.append(starts[-1] + len(ln) + 1)
    import bisect
    for lit in moved:
        k = v.find(lit)
        while k >= 0:
            a = bisect.bisect_right(starts, k) - 1
            b = bisect.bisect_right(starts, k + max(len(lit) - 1, 0)) - 1
            prot.update(range(a - 1, b + 2))   # the neighbours too: a literal may start / end on a line break
            k = v.find(lit, k + 1)
    return prot


# ---------------------------------------------------------------- a: dead #if branches (fixed lab macros)
def _cond_eval(expr, fixed, at):
    """Value of an #if / #elif expression, or None. Only plain fixed macros whose default #define precedes `at`; any
    defined(), function-like macro, unknown name or character -> None (left alone)."""
    e = expr.split("//")[0].strip()
    if not e or "defined" in e or re.search(r"\w\s*\(", e):
        return None
    names = set(IDRE.findall(e))
    if any(n not in fixed or fixed[n][1] >= at for n in names):
        return None
    py = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: "(%d)" % fixed[m.group(0)][0], e)
    if not re.fullmatch(r"[\d\s()+\-*/%<>=!&|^~]*", py):
        return None
    py = py.replace("&&", " and ").replace("||", " or ")
    py = re.sub(r"!(?!=)", " not ", py).replace("/", "//")
    try:
        return bool(eval(py, {"__builtins__": {}}, {}))
    except Exception:
        return None


def fixed_macros(v, settable, all_cuda):
    """{X: (int default, line of its #define)} for `#ifndef X / #define X <int> / #endif` defaults that no -D / literal
    in the file can set (settable) and that no CUDA constant #defines / #undefs anywhere else."""
    fixed = {}
    pat = (r"^[ \t]*#[ \t]*ifndef[ \t]+(\w+)[ \t]*\n([ \t]*#[ \t]*define[ \t]+\1[ \t]+(-?\d+)[ \t]*(?://[^\n]*)?)\n"
           r"[ \t]*#[ \t]*endif")
    for m in re.finditer(pat, v, re.M):
        x = m.group(1)
        if x in settable:
            continue
        n = sum(len(re.findall(r"^[ \t]*#[ \t]*(?:define|undef)[ \t]+%s\b" % x, s, re.M)) for s in all_cuda)
        if n == 1:
            fixed[x] = (int(m.group(3)), v.count("\n", 0, m.start(2)))
    return fixed


def drop_dead_ifs(v, fixed):
    """Whole-line removal of resolved #if/#elif/#else/#endif directives and their dead branches.
    Returns (text, removed line count); (v, 0) if any directive is not understood."""
    lines = v.split("\n")
    keep = [True] * len(lines)
    stack = []   # {"res": resolved?, "cur": live branch? (None = unknown), "taken": a branch was live}
    dre = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")

    def dead(st):
        return any(s["cur"] is False for s in st)
    for i, ln in enumerate(lines):
        m = None if (i and lines[i - 1].endswith("\\")) else dre.match(ln)
        if not m:
            if dead(stack):
                keep[i] = False
            continue
        if ln.rstrip().endswith("\\"):
            return v, 0
        k, c = m.group(1), m.group(2)
        if k in ("if", "ifdef", "ifndef"):
            outer = dead(stack)
            r = _cond_eval(c, fixed, i) if (k == "if" and not outer) else None
            stack.append({"res": r is not None, "cur": r, "taken": bool(r)})
            keep[i] = not (outer or r is not None)
        elif k in ("elif", "else"):
            if not stack:
                return v, 0
            st, outer = stack[-1], dead(stack[:-1])
            if st["res"]:
                r = True if k == "else" else _cond_eval(c, fixed, i)
                if r is None:
                    return v, 0
                st["cur"] = (not st["taken"]) and r
                st["taken"] = st["taken"] or r
                keep[i] = False
            else:
                keep[i] = not outer
        else:
            if not stack:
                return v, 0
            st = stack.pop()
            keep[i] = not (dead(stack) or st["res"])
    if stack:
        return v, 0
    return "\n".join(ln for ln, kp in zip(lines, keep) if kp), keep.count(False)


# ---------------------------------------------------------------- b: unreferenced top-level defs
def _refs(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 300:
            out.update(IDRE.findall(n.value))
        elif isinstance(n, ast.alias):
            out.add(n.asname or n.name.split(".")[0])
    return out


def _pure(v):
    if isinstance(v, (ast.Constant, ast.Name)):
        return True
    if isinstance(v, ast.JoinedStr):
        return all(_pure(x.value) for x in v.values if isinstance(x, ast.FormattedValue))
    if isinstance(v, (ast.Tuple, ast.List, ast.Set)):
        return all(_pure(e) for e in v.elts)
    if isinstance(v, ast.Dict):
        return all(k is None or _pure(k) for k in v.keys) and all(_pure(x) for x in v.values)
    if isinstance(v, ast.BinOp):
        return _pure(v.left) and _pure(v.right)
    if isinstance(v, ast.UnaryOp):
        return _pure(v.operand)
    return False


def dead_defs(tree):
    cands = {}
    for i, n in enumerate(tree.body):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not n.decorator_list:
            cands[i] = n.name
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and _pure(n.value):
            cands[i] = n.targets[0].id
    by_name = {}
    for i, nm in cands.items():
        by_name.setdefault(nm, []).append(i)
    live, work = set(API), list(API)
    for i, n in enumerate(tree.body):
        if i not in cands:
            for r in _refs(n):
                if r not in live:
                    live.add(r)
                    work.append(r)
    while work:
        for i in by_name.get(work.pop(), []):
            n = tree.body[i]
            for r in _refs(n.value if isinstance(n, ast.Assign) else n):
                if r not in live:
                    live.add(r)
                    work.append(r)
    return sorted(i for i, nm in cands.items() if nm not in live)


# ---------------------------------------------------------------- p: Python comments / blank lines
def strip_python(t, keep):
    lines = t.splitlines(keepends=True)
    in_str = set()
    cut = {}   # row -> col of an inline comment
    code_rows = set()
    for tok in tokenize.generate_tokens(io.StringIO(t).readline):
        (r0, c0), (r1, c1) = tok.start, tok.end
        if tok.type == tokenize.STRING or tok.type in (getattr(tokenize, "FSTRING_MIDDLE", -1), getattr(tokenize, "FSTRING_START", -1),
                                                        getattr(tokenize, "FSTRING_END", -1)):
            in_str.update(range(r0 + 1, r1 + 1))
        if tok.type == tokenize.COMMENT:
            cut[r0] = c0
        elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            code_rows.update(range(r0, r1 + 1))
    out = []
    for r, ln in enumerate(lines, 1):
        if r <= keep or r in in_str:
            out.append(ln)
            continue
        if r in cut:
            body = ln[:cut[r]].rstrip()
            if not body:
                continue
            ln = body + "\n"
        if not ln.strip() and r not in code_rows:
            continue
        out.append(ln)
    return "".join(out)


# ---------------------------------------------------------------- driver
def is_cuda_const(n):
    if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)):
        return False
    v = n.value.value
    return len(v) >= 200 and ("__global__" in v or "__device__" in v or "#include" in v or "__forceinline__" in v)


SURG = {"replace", "index", "rindex", "find", "rfind", "count", "split", "rsplit", "partition", "rpartition", "startswith",
        "endswith", "removeprefix", "removesuffix"}


def python_literals(tree, cuda_ids, min_len=4):
    """Strings that can act as runtime anchors on a CUDA constant: every literal (or literal concatenation / f-string
    piece) of >= min_len chars that is an argument of a str-surgery method call or an operand of `in` / `not in`, plus the
    literal values assigned (anywhere) to a name that is itself used as such an argument (e.g. _g2_stream_h's head / wait)."""
    lits, names = set(), set()

    def const_strs(x):
        if isinstance(x, ast.Constant) and isinstance(x.value, str):
            return [] if id(x) in cuda_ids else [x.value]
        if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Add):
            a, b = const_strs(x.left), const_strs(x.right)
            return [p + q for p in a for q in b] + a + b if a and b else a + b
        if isinstance(x, ast.JoinedStr):
            return [v.value for v in x.values if isinstance(v, ast.Constant)]
        return []

    def add(x):
        lits.update(const_strs(x))
        names.update(y.id for y in ast.walk(x) if isinstance(y, ast.Name))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in SURG:
            for x in list(n.args) + [k.value for k in n.keywords]:
                add(x)
        elif isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops):
            for x in [n.left] + list(n.comparators):
                add(x)
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(g, ast.Name) and g.id in names for g in n.targets):
            for y in ast.walk(n.value):
                lits.update(const_strs(y))
    return {s for s in lits if len(s.strip()) >= min_len and len(s) < 4000}


def all_dash_d(tree, cuda_ids):
    """Every macro name the Python side could -D (any string mentioning -DNAME, incl. f-string parts)."""
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in cuda_ids:
            names.update(re.findall(r"-D\s*([A-Za-z_]\w*)", n.value))
            names.update(re.findall(r"#\s*define\s+([A-Za-z_]\w*)", n.value))
            names.update(re.findall(r"#\s*undef\s+([A-Za-z_]\w*)", n.value))
            # a literal that names a macro alone may be used to build -D flags / #if replacements: treat as settable
            if IDRE.fullmatch(n.value):
                names.add(n.value)
    return names


def build(t, opts, log=print):
    tree = ast.parse(t)
    lines = t.split("\n")
    cuda_nodes = [n for n in tree.body if is_cuda_const(n)]
    cuda_ids = {id(n.value) for n in cuda_nodes}
    lits = python_literals(tree, cuda_ids)
    dash_d = all_dash_d(tree, cuda_ids)
    all_cuda = [n.value.value for n in cuda_nodes]
    stats = {"d": 0, "a": 0, "b": 0, "p": 0}
    new_vals = {}
    report = []
    for n in cuda_nodes:
        name, v = n.targets[0].id, n.value.value
        seg = "\n".join(lines[n.lineno - 1:n.end_lineno])
        rhs = seg.split("=", 1)[1].lstrip()
        quote = next((q for q in ('r"""', "r'''", '"""', "'''") if rhs.startswith(q)), None)
        if quote is None or not seg.rstrip().endswith(quote[-3:]) or seg.count(quote[-3:]) != 2 or quote[-3:] in v:
            report.append((name, "skip: form"))
            continue
        if not quote.startswith("r") and "\\" in v:
            report.append((name, "skip: non-raw with backslashes"))
            continue
        if re.search(r"__LINE__|__FILE__|\bassert\s*\(|#\s*line\b|\bR\"", v):
            report.append((name, "skip: line-sensitive"))
            continue
        cur = v
        if opts.ifs:
            fixed = fixed_macros(cur, dash_d, all_cuda)
            if fixed:
                nv, nlines = drop_dead_ifs(cur, fixed)
                if nlines and not literal_counts_ok(cur, nv, lits):
                    stats["a"] += len(cur.encode()) - len(nv.encode())
                    used = sorted(x for x in fixed if re.search(r"^\s*#\s*(el)?if\b.*\b%s\b" % x, cur, re.M))
                    report.append((name, f"a: -{nlines} lines, -{len(cur.encode()) - len(nv.encode())} B; fixed {used}"))
                    cur = nv
        after_ifs = cur
        if opts.cuda:
            prot = set()
            for _ in range(4):
                nv = strip_cuda(cur, prot)
                moved = literal_counts_ok(cur, nv, lits)
                if not moved:
                    break
                prot |= protect_lines(cur, moved)
            else:
                report.append((name, "d skipped: anchors"))
                nv = cur
            if c_tokens(nv) != c_tokens(cur):
                raise SystemExit(f"[slim] {name}: token stream changed by the comment strip; abort")
            stats["d"] += len(cur.encode()) - len(nv.encode())
            if prot:
                report.append((name, f"d: {len(prot)} anchor lines kept verbatim"))
            cur = nv
        if cur != v:
            new_vals[name] = (n, quote, cur, after_ifs)
    # rewrite CUDA constants (bottom-up)
    edits = []
    for name, (n, quote, nv, _) in new_vals.items():
        edits.append((n.lineno, n.end_lineno, f"{name} = {quote}{nv}{quote[-3:]}"))
    dead_idx = dead_defs(tree) if opts.dead else []
    for i in dead_idx:
        n = tree.body[i]
        lo = min([d.lineno for d in getattr(n, "decorator_list", [])] + [n.lineno])
        stats["b"] += sum(len(x.encode()) + 1 for x in lines[lo - 1:n.end_lineno])
        edits.append((lo, n.end_lineno, None))
        report.append((getattr(n, "name", None) or n.targets[0].id, "b: unreferenced"))
    for lo, hi, new in sorted(edits, reverse=True):
        lines[lo - 1:hi] = [] if new is None else [new]
    t2 = "\n".join(lines)
    if opts.py:
        t3 = strip_python(t2, opts.keep)
        stats["p"] = len(t2.encode()) - len(t3.encode())
        t2 = t3
    # ---- gates
    exp = ast.parse(t)
    exp.body = [n for i, n in enumerate(exp.body) if i not in set(dead_idx)]
    for n in exp.body:
        if is_cuda_const(n) and n.targets[0].id in new_vals:
            n.value.value = new_vals[n.targets[0].id][2]
    got = ast.parse(t2)
    if ast.dump(exp) != ast.dump(got):
        raise SystemExit("[slim] AST gate FAIL")
    for n in got.body:
        if is_cuda_const(n) and n.targets[0].id in new_vals:
            ov = next(m.value.value for m in tree.body if is_cuda_const(m) and m.targets[0].id == n.targets[0].id)
            nv = n.value.value
            moved = literal_counts_ok(ov, nv, lits)
            if moved:
                raise SystemExit(f"[slim] literal gate FAIL in {n.targets[0].id}: {moved[0][:60]!r}")
            if c_tokens(nv) != c_tokens(new_vals[n.targets[0].id][3]):   # d is token-neutral; a is checked by SASS
                raise SystemExit(f"[slim] token gate FAIL in {n.targets[0].id}")
    if not t2.isascii() and t.isascii():
        raise SystemExit("[slim] ascii gate FAIL")
    return t2, stats, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=2, help="lines 1..N verbatim (okc header + nonce)")
    ap.add_argument("--no-cuda", dest="cuda", action="store_false")
    ap.add_argument("--no-ifs", dest="ifs", action="store_false")
    ap.add_argument("--no-dead", dest="dead", action="store_false")
    ap.add_argument("--no-py", dest="py", action="store_false")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    src, dst = pathlib.Path(a.base), pathlib.Path(a.out)
    if dst.exists() and not a.force:
        sys.exit(f"[slim] {dst} exists; refusing to overwrite (--force)")
    t = src.read_text(encoding="utf-8")
    t2, stats, report = build(t, a)
    if a.verbose:
        for r in report:
            print("  ", *r)
    n0, n1 = len(t.encode()), len(t2.encode())
    if n1 >= LIMIT:
        print(f"[slim] WARNING: {n1} B is not under the {LIMIT} B cap")
    dst.write_text(t2, encoding="utf-8", newline="")
    print(f"[slim] {src} -> {dst}: {n0} -> {n1} B (-{n0 - n1}); by category: CUDA comments/whitespace d -{stats['d']}, "
          f"dead #if a -{stats['a']}, dead defs b -{stats['b']}, Python comments/blank p -{stats['p']}; gates ast / tokens / "
          f"literals / ascii PASS; {'under' if n1 < LIMIT else 'OVER'} {LIMIT} B by {abs(LIMIT - n1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

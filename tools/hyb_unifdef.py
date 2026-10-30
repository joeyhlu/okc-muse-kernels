"""hyb_unifdef.py: partial C-preprocessor evaluation (unifdef-style) for one fixed set of -D defines.

Keeps the source text unchanged except that #if / #ifdef / #ifndef / #elif / #else / #endif groups whose conditions are
decidable from the given defines (plus the source's own #define / #undef seen in decided-active regions) are resolved:
only the taken branch's lines stay, the directives go. Undecidable groups stay verbatim (their nested decidable groups
are still resolved; any #define / #undef inside an undecided branch makes that macro unknown afterwards).
Used by tools/hyb_patches.py to cut the alygg77 MoE CUDA source to its OKC_WGS build; the gate there compares the
nvcc -E token stream of original and stripped source for that define set (must be identical).

    python -I tools/hyb_unifdef.py <in.cu> <out.cu> -DNAME=VAL ...
"""
import re
import sys

UNKNOWN = object()   # status unknown (maybe undefined)
FN = object()        # defined, but not usable as an #if value (function-like)
_TOK = re.compile(r"\s*(?:(0[xX][0-9a-fA-F]+|\d+)[uUlL]*|([A-Za-z_]\w*)|(&&|\|\||==|!=|<=|>=|<<|>>|[-+*/%<>!~&|^?:()]))")


class Unknown(Exception):
    pass


def _tokens(s):
    s = re.sub(r"/\*.*?\*/", " ", s)
    s = s.split("//")[0]
    out, i = [], 0
    while i < len(s):
        if s[i:].strip() == "":
            break
        m = _TOK.match(s, i)
        if not m:
            raise Unknown(s)
        if m.group(1) is not None:
            out.append(("n", int(m.group(1), 0)))
        elif m.group(2) is not None:
            out.append(("i", m.group(2)))
        else:
            out.append(("o", m.group(3)))
        i = m.end()
    return out


class Eval:
    """Recursive-descent #if evaluator with UNKNOWN propagation (logical ops short-circuit on known operands)."""

    def __init__(self, macros, depth=0):
        self.m, self.depth = macros, depth

    def value(self, text):
        self.t, self.p = _tokens(text), 0
        v = self.ternary()
        if self.p != len(self.t):
            raise Unknown(text)
        return v

    def peek(self):
        return self.t[self.p] if self.p < len(self.t) else (None, None)

    def eat(self, op=None):
        k, v = self.peek()
        if op is not None and v != op:
            raise Unknown(op)
        self.p += 1
        return k, v

    def ident(self, name):
        if name not in self.m:
            if name.startswith("__"):
                return UNKNOWN   # compiler / header macros (__CUDA_ARCH__, ...)
            return 0             # never defined: 0, as the preprocessor does
        body = self.m[name]
        if body is UNKNOWN or body is FN or self.depth > 20:
            return UNKNOWN
        if body.strip() == "":
            return UNKNOWN       # defined empty: not usable in #if arithmetic here
        try:
            return Eval(self.m, self.depth + 1).value(body)
        except Unknown:
            return UNKNOWN

    def primary(self):
        k, v = self.eat()
        if k == "n":
            return v
        if k == "i":
            if v == "defined":
                paren = self.peek()[1] == "("
                if paren:
                    self.eat("(")
                _, name = self.eat()
                if paren:
                    self.eat(")")
                if name in self.m:
                    return UNKNOWN if self.m[name] is UNKNOWN else 1
                if name.startswith("__"):
                    return UNKNOWN
                return 0
            if self.peek()[1] == "(":
                raise Unknown(v)   # function-like macro in #if
            return self.ident(v)
        if v == "(":
            x = self.ternary()
            self.eat(")")
            return x
        if v in ("!", "-", "+", "~"):
            x = self.primary()
            if x is UNKNOWN:
                return UNKNOWN
            return {"!": lambda a: int(not a), "-": lambda a: -a, "+": lambda a: a, "~": lambda a: ~a}[v](x)
        raise Unknown(v)

    BIN = [("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="), ("<", ">", "<=", ">="), ("<<", ">>"), ("+", "-"),
           ("*", "/", "%")]

    def binary(self, lvl):
        if lvl == len(self.BIN):
            return self.primary()
        a = self.binary(lvl + 1)
        while self.peek()[1] in self.BIN[lvl]:
            _, op = self.eat()
            b = self.binary(lvl + 1)
            ka, kb = a is not UNKNOWN, b is not UNKNOWN
            if op == "&&":
                a = 0 if (ka and not a) or (kb and not b) else (1 if ka and kb else UNKNOWN)
            elif op == "||":
                a = 1 if (ka and a) or (kb and b) else (0 if ka and kb else UNKNOWN)
            elif UNKNOWN in (a, b):
                a = UNKNOWN
            else:
                a = {"|": lambda x, y: x | y, "^": lambda x, y: x ^ y, "&": lambda x, y: x & y,
                     "==": lambda x, y: int(x == y), "!=": lambda x, y: int(x != y), "<": lambda x, y: int(x < y),
                     ">": lambda x, y: int(x > y), "<=": lambda x, y: int(x <= y), ">=": lambda x, y: int(x >= y),
                     "<<": lambda x, y: x << y, ">>": lambda x, y: x >> y, "+": lambda x, y: x + y,
                     "-": lambda x, y: x - y, "*": lambda x, y: x * y,
                     "/": lambda x, y: int(x / y) if y else UNKNOWN, "%": lambda x, y: x % y if y else UNKNOWN}[op](a, b)
        return a

    def ternary(self):
        c = self.binary(0)
        if self.peek()[1] == "?":
            self.eat("?")
            a = self.ternary()
            self.eat(":")
            b = self.ternary()
            if c is UNKNOWN:
                return a if (a == b and a is not UNKNOWN) else UNKNOWN
            return a if c else b
        return c


def _eval(expr, macros):
    try:
        v = Eval(macros).value(expr)
    except (Unknown, RecursionError):
        return UNKNOWN
    return v if v is UNKNOWN else int(bool(v))


def _logical_lines(src):
    """Yield (physical_lines, logical_text) with backslash continuations joined for directives."""
    lines = src.split("\n")
    i = 0
    while i < len(lines):
        j = i
        if lines[i].lstrip().startswith("#"):
            while lines[j].endswith("\\") and j + 1 < len(lines):
                j += 1
        yield lines[i:j + 1], " ".join(x[:-1] if x.endswith("\\") else x for x in lines[i:j + 1])
        i = j + 1


_DIR = re.compile(r"^\s*#\s*(\w+)\s*(.*)$")


def unifdef(src, defines):
    """defines: dict NAME -> value string. Returns the stripped source."""
    macros = {k: str(v) for k, v in defines.items()}
    out = []
    # stack frames: dict(mode='known'|'unknown', active=bool for known (this branch taken),
    #   taken=bool any branch taken so far (known mode), parent_emit=bool, parent_known=bool)
    st = []

    def emitting():
        return all(f["emit"] for f in st)

    def certain():   # every enclosing group decided -> #define here is certain
        return all(f["mode"] == "known" for f in st)

    for phys, text in _logical_lines(src):
        m = _DIR.match(text)
        d = m.group(1) if m else None
        rest = m.group(2).strip() if m else ""
        if d in ("if", "ifdef", "ifndef"):
            if d == "ifdef":
                name = rest.split()[0]
                c = (UNKNOWN if macros[name] is UNKNOWN else 1) if name in macros else (UNKNOWN if name.startswith("__") else 0)
            elif d == "ifndef":
                name = rest.split()[0]
                c = (UNKNOWN if macros[name] is UNKNOWN else 0) if name in macros else (UNKNOWN if name.startswith("__") else 1)
            else:
                c = _eval(rest, macros)
            if not emitting():
                st.append(dict(mode="dead", emit=False, taken=True))
                continue
            if c is UNKNOWN:
                st.append(dict(mode="unknown", emit=True, taken=False))
                out.extend(phys)
            else:
                st.append(dict(mode="known", emit=bool(c), taken=bool(c)))
            continue
        if d in ("elif", "else"):
            f = st[-1]
            if f["mode"] == "dead":
                continue
            if f["mode"] == "unknown":
                out.extend(phys)   # undecided chain: kept verbatim
                f["emit"] = True
                continue
            # known chain
            if f["taken"]:
                f["emit"] = False
                continue
            if d == "else":
                f["emit"] = f["taken"] = True
                continue
            c = _eval(rest, macros)
            if c is UNKNOWN:
                # a decided-false prefix then an undecidable elif: turn this elif into #if, keep the rest verbatim
                f["mode"] = "unknown"
                f["emit"] = True
                f["converted"] = True
                out.append(re.sub(r"#\s*elif", "#if", phys[0], count=1))
                out.extend(phys[1:])
                continue
            f["emit"] = f["taken"] = bool(c)
            continue
        if d == "endif":
            f = st.pop()
            if f["mode"] == "unknown":
                out.extend(phys)
            continue
        if not emitting():
            continue
        if d == "define":
            mm = re.match(r"([A-Za-z_]\w*)(\(?)(.*)$", rest)
            name, fn, body = mm.group(1), mm.group(2), mm.group(3)
            if not certain():
                macros[name] = UNKNOWN
            elif fn:
                macros[name] = FN
            else:
                macros[name] = body.split("//")[0].strip()
        elif d == "undef":
            name = rest.split()[0]
            if certain():
                macros.pop(name, None)
            else:
                macros[name] = UNKNOWN
        out.extend(phys)
    assert not st, "unbalanced #if"
    return "\n".join(out)


def main():
    a = sys.argv[1:]
    src = open(a[0], encoding="utf-8").read()
    defs = {}
    for x in a[2:]:
        assert x.startswith("-D")
        k, _, v = x[2:].partition("=")
        defs[k] = v if v else "1"
    out = unifdef(src, defs)
    open(a[1], "w", encoding="utf-8", newline="\n").write(out)
    print(f"[unifdef] {len(src.encode())} -> {len(out.encode())} B")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------------------------
# Unreferenced-function removal (after unifdef + comment strip): top-level (or namespace-level) function definitions
# whose name no kept item mentions are dropped, to a fixed point. Every non-function item (macros, structs, constexpr
# tables, the exported host functions in `roots`) is kept and seeds the reference set. Over-approximate: any
# identifier match (member, variable, overload) keeps every definition of that name. Gate: identical SASS.
_ID = re.compile(r"[A-Za-z_]\w*")
_KW = {"if", "for", "while", "switch", "return", "sizeof", "decltype", "static_assert", "alignof", "catch"}


def _items(src):
    """Split into (text, kind, name) items at brace depth 0 (namespace bodies are entered: depth counted inside)."""
    items, i, n, start = [], 0, len(src), 0
    depth, ns_depths = 0, []
    line_start = True

    def push(a, b):
        txt = src[a:b]
        if txt.strip():
            items.append(txt)

    while i < n:
        c = src[i]
        if line_start and c == "#":
            j = i
            while True:
                e = src.find("\n", j)
                if e < 0:
                    e = n
                if src[j:e].rstrip().endswith("\\"):
                    j = e + 1
                    continue
                break
            if depth == len(ns_depths):
                push(start, i)
                push(i, e + 1)
                start = e + 1
            i = e + 1
            line_start = True
            continue
        line_start = c == "\n"
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            head = src[start:i]
            if depth == len(ns_depths) and re.match(r"^\s*namespace\s+\w*\s*$", head):
                push(start, i + 1)
                ns_depths.append(depth)
                depth += 1
                start = i + 1
                i += 1
                continue
            depth += 1
        elif c == "}":
            if ns_depths and depth - 1 == ns_depths[-1] and depth == len(ns_depths):
                push(start, i)
                push(i, i + 1)
                ns_depths.pop()
                depth -= 1
                start = i + 1
                i += 1
                continue
            depth -= 1
            if depth == len(ns_depths):
                k = i + 1
                while k < n and src[k] in " \t":
                    k += 1
                if k < n and src[k] == ";":
                    k += 1
                push(start, k)
                start = k
                i = k
                continue
        elif c == ";" and depth == len(ns_depths):
            push(start, i + 1)
            start = i + 1
        i += 1
    push(start, n)
    return items


def _fn_name(item):
    """Name of a function DEFINITION item, else None."""
    s = item.strip()
    if s.startswith("#") or not s.endswith("}"):
        return None
    head = s[:s.find("{")]
    if re.match(r"^(template\s*<[^{]*>\s*)?(struct|class|union|enum|namespace)\b", head) or "=" in head.split("(")[0]:
        return None
    m = None
    for m in re.finditer(r"([A-Za-z_]\w*)\s*\(", head):
        if m.group(1) not in _KW and m.group(1) not in ("__launch_bounds__", "__align__", "alignas"):
            break
    else:
        return None
    return m.group(1) if m else None


def _struct_name(item):
    """Name of a struct / class template (specialization) DEFINITION item, else None."""
    m = re.match(r"^\s*template\s*<[^{;]*>\s*struct\s+([A-Za-z_]\w*)\s*(<[^{;]*>)?\s*(\{|;\s*$)", item)
    return m.group(1) if m and item.rstrip().endswith(";") else None   # (a bare declaration counts too)


def drop_unreferenced(src, roots=("okc_run", "okc_set_pdl")):
    """Drop unreferenced function definitions and struct-template definitions / specializations (to a fixpoint)."""
    items = _items(src)
    names = [_fn_name(t) or _struct_name(t) for t in items]
    keep = [nm is None or nm in roots for nm in names]
    while True:
        refs = set()
        for t, nm, k in zip(items, names, keep):
            if k:
                ids = set(_ID.findall(t))
                if nm is not None and nm not in roots:
                    ids.discard(nm)   # a definition does not keep itself (recursion / sibling specializations)
                refs |= ids
        changed = False
        for idx, (nm, k) in enumerate(zip(names, keep)):
            if not k and nm in refs:
                keep[idx] = True
                changed = True
        if not changed:
            break
    dropped = sorted({nm for nm, k in zip(names, keep) if not k})
    out = "".join(t if t.endswith("\n") else t + "\n" for t, k in zip(items, keep) if k)
    return out, dropped

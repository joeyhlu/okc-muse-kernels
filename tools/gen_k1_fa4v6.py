"""Generate kernels/muse_attention_decode/k1_fa4v6.py: the K1-FA4 v5 attention path (FA4 v2 core + fused prologue /
combine-norm-gate + route + self-test) as ONE plain Python source with no runtime patching (A19 REVISED).

  python tools/gen_k1_fa4v6.py            # writes k1_fa4v6.py and k1_fa4v6_route.py, prints size + scan

The generator does the following:
- reads the image copy of FA4's sm90 forward (box_out/fa4_image/flash_fwd_sm90.py, from
  sglang/kernels/ops/attention/flash_attn/cute/ in lmsysorg/sglang:dev-muse-spark-open);
- applies v5's patch table at generation time, each anchor exactly once;
- drops the three producer methods the subclass replaces (load, load_KV, load_Bias);
- renames every module-level name the vendored text binds to a k1fa4_ prefix. cutlass, cute, const_expr, Int32 and
  Float32 keep their names: the CuTe DSL preprocessor may match them by name, and any other binding of them in a
  host file is the same object;
- regenerates its imports as normal `from <unmodified module> import X as k1fa4_X` statements;
- prepends the FA4 BSD-3 license notice;
- appends tools/gen_k1_fa4v6_tail.py.in (the subclass, the host shim trimmed from interface._flash_attn_fwd, the
  Triton kernels, the route and the self-test).
interface.py is not vendored: the tail's k1fa4_fwd_partials holds the ~40 lines of its host path that this
configuration uses.
k1_fa4v6_route.py, the attention-problem submission used for the judge clone, is r1_glue.py + k1_fa4v6.py pasted at
top level + a kernel_fn that routes. Pasting at top level is the property the decode_layer file needs.
"""
import ast
import hashlib
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "box_out" / "fa4_image" / "flash_fwd_sm90.py"
LICENSE = ROOT / "repos" / "sglang" / "python" / "sglang" / "kernels" / "ops" / "attention" / "flash_attn" / "cute" / "LICENSE"
TAIL = ROOT / "tools" / "gen_k1_fa4v6_tail.py.in"
KDIR = ROOT / "kernels" / "muse_attention_decode"
OUT = KDIR / "k1_fa4v6.py"
ROUTE = KDIR / "k1_fa4v6_route.py"
R1 = KDIR / "r1_glue.py"

KEEP = {"cutlass", "cute", "const_expr", "Int32", "Float32"}
DROP_METHODS = {"load", "load_KV", "load_Bias"}
CLASS = "FlashAttentionForwardSm90"
PREFIX = "k1fa4_"

NL = "\n"
PATCHES = [  # v5's sm90 table (k1_fa4v5.py), knobs renamed k1fa4_K / k1fa4_skip_check
    ("min_blocks_per_mp=1,", 'min_blocks_per_mp=k1fa4_K["minb"], use_pdl=k1fa4_K["pdl"],'),
    ("1: (256, 56),", '1: k1fa4_K["regs1"],'),
    ("        super().__init__(*args, **kwargs)" + NL,
     '        kwargs["num_stages"] = k1fa4_K["stages"]' + NL + "        super().__init__(*args, **kwargs)" + NL),
    ("(mV.element_type, (self.tile_n, self.tile_hdimv), self.num_stages),",
     "(self.dtype, (self.tile_n, self.tile_hdimv), self.num_stages),"),
    ("        self._check_type(" + NL, "        k1fa4_skip_check(" + NL),
    ("        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(" + NL + "            self.dtype," + NL
     + "            self.dtype," + NL,
     "        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(" + NL + "            cutlass.Float8E4M3FN,"
     + NL + "            cutlass.Float8E4M3FN," + NL),
    ("        storage = smem.allocate(SharedStorage)" + NL,
     "        storage = smem.allocate(SharedStorage)" + NL + "        self._k1_storage = storage" + NL),
    ("            self.epilogue(" + NL + "                acc_O," + NL,
     "            if const_expr(k1fa4_K['trig'] == 'epi'):" + NL
     + "                cute.arch.griddepcontrol_launch_dependents()" + NL
     + "            self.epilogue(" + NL + "                acc_O," + NL),
]

FORBIDDEN = [r"\bimportlib\b", r"sys\.modules", r"(?<![\w.])exec\(", r"(?<![\w.])compile\(", r"__import__",
             r"(?<![\w.])setattr\(", r"globals\(\)\s*\[", r"(?<![\w.])open\(", r"sys\.path", r"os\.environ\[",
             r"os\.environ\.(setdefault|update|pop)", r"spec_from_file_location", r"module_from_spec", r"tempfile",
             r"__name__"]


def drop_methods(txt):
    tree = ast.parse(txt)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS)
    spans = []
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name in DROP_METHODS:
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            spans.append((start, n.end_lineno, n.name))
    assert {s[2] for s in spans} == DROP_METHODS, spans
    lines = txt.split(NL)
    for start, end, _ in sorted(spans, reverse=True):
        del lines[start - 1:end]
    return NL.join(lines)


def split_imports(txt):
    """(import nodes, body text from the class definition on)."""
    tree = ast.parse(txt)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    others = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(others) == 1 and isinstance(others[0], ast.ClassDef) and others[0].name == CLASS, others
    lines = txt.split(NL)
    return imports, NL.join(lines[others[0].lineno - 1:])


def import_block(imports):
    out, bound = [], {}
    for n in imports:
        if isinstance(n, ast.Import):
            for a in n.names:
                name = a.asname or a.name.split(".")[0]
                assert a.asname or "." not in a.name, a.name
                new = name if name in KEEP else PREFIX + name
                bound[name] = new
                out.append(f"import {a.name}" + ("" if new == a.name else f" as {new}"))
        else:
            assert n.level == 0, "relative import"
            for a in n.names:
                name = a.asname or a.name
                new = name if name in KEEP else PREFIX + name
                bound[name] = new
                out.append(f"from {n.module} import {a.name}" + ("" if new == a.name else f" as {new}"))
    return out, bound


def rename(body, mapping):
    """Rename NAME tokens per mapping, except attribute names (after '.') and keyword names (NAME '=' inside
    parentheses)."""
    toks = list(tokenize.generate_tokens(io.StringIO(body).readline))
    sig = [t for t in toks if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT,
                                             tokenize.DEDENT)]
    edits = []
    depth = 0
    for i, t in enumerate(sig):
        if t.type == tokenize.OP and t.string in "([{":
            depth += 1
        elif t.type == tokenize.OP and t.string in ")]}":
            depth -= 1
        if t.type != tokenize.NAME or t.string not in mapping:
            continue
        prev = sig[i - 1] if i else None
        nxt = sig[i + 1] if i + 1 < len(sig) else None
        if prev is not None and prev.type == tokenize.OP and prev.string == ".":
            continue
        if (depth > 0 and nxt is not None and nxt.type == tokenize.OP and nxt.string == "=" and prev is not None
                and prev.type == tokenize.OP and prev.string in ("(", ",")):
            continue  # keyword argument / parameter name (an annotation `p: T = d` is renamed: prev is ':')
        edits.append((t.start, t.end, mapping[t.string]))
    lines = body.split(NL)
    for (r0, c0), (r1, c1), new in sorted(edits, reverse=True):
        assert r0 == r1
        line = lines[r0 - 1]
        lines[r0 - 1] = line[:c0] + new + line[c1:]
    return NL.join(lines), len(edits)


def check_params(body, mapping):
    tree = ast.parse(body)
    for n in ast.walk(tree):
        if isinstance(n, ast.arg):
            assert n.arg not in mapping, ("parameter shadows a renamed global", n.arg)
        if isinstance(n, ast.keyword) and n.arg in mapping:
            raise AssertionError(("keyword arg equals a renamed global", n.arg))


def scan(txt):
    hits = []
    for pat in FORBIDDEN:
        for m in re.finditer(pat, txt):
            line = txt.count(NL, 0, m.start()) + 1
            hits.append((pat, line, txt.split(NL)[line - 1].strip()[:100]))
    return hits


def main():
    raw = SRC.read_bytes()
    txt = raw.decode("utf-8").replace("\r\n", NL)
    sha = hashlib.sha256(raw).hexdigest()
    for a, b in PATCHES:
        assert txt.count(a) == 1, ("anchor", a, txt.count(a))
        txt = txt.replace(a, b)
    txt = drop_methods(txt)
    imports, body = split_imports(txt)
    imp_lines, bound = import_block(imports)
    mapping = {k: v for k, v in bound.items() if v != k}
    mapping[CLASS] = PREFIX + CLASS
    check_params(body, mapping)
    body, n_edits = rename(body, mapping)
    assert "self.load_KV(" not in body and "self.load_Bias(" not in body and body.count("self.load(") == 1
    lic = LICENSE.read_text(encoding="utf-8").replace("\r\n", NL).strip().split(NL)
    head = [
        "# " + "=" * 116,
        "# K1-FA4 v6 attention (team joeyhlu). Generated by tools/gen_k1_fa4v6.py; do not edit by hand.",
        "# Part 1 vendors FlashAttention-4's CuTe-DSL Hopper forward (FlashAttentionForwardSm90) from",
        "#   sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm90.py (lmsysorg/sglang:dev-muse-spark-open)",
        f"#   sha256 {sha}",
        "# with these source changes, applied at generation time: launch min_blocks_per_mp / use_pdl and the",
        "# 1-warpgroup setmaxnreg pair from k1fa4_K; K/V pipeline depth from k1fa4_K; 16-bit V smem stages; no",
        "# 16-bit-only type check; E4M3 QK tiled MMA; the shared-storage handle kept on self for the producer; an",
        "# optional PDL trigger before the epilogue. The producer methods load / load_KV / load_Bias are removed",
        "# (k1fa4_Sm90Fwd.load replaces them). Module-level names carry a k1fa4_ prefix.",
        "# FlashAttention-4 license (BSD 3-Clause):",
    ] + ["#   " + ln if ln else "#" for ln in lic] + ["# " + "=" * 116]
    knobs = [
        'k1fa4_K = {"minb": 2, "regs1": (200, 56), "stages": 2, "pdl": True, "trig": False}',
        "",
        "",
        "def k1fa4_skip_check(*_a, **_k):",
        '    """FA4 checks for 16-bit Q/K/V; this build checks its own contract in k1fa4_fwd_partials."""',
        "    return None",
        "",
        "",
    ]
    tail = TAIL.read_text(encoding="utf-8").replace("\r\n", NL)
    out = NL.join(head + imp_lines + ["", ""] + knobs) + body.rstrip(NL) + NL + NL + tail.lstrip(NL)
    ast.parse(out)
    OUT.write_bytes(out.encode("utf-8"))

    r1 = R1.read_text(encoding="utf-8").replace("\r\n", NL)
    assert r1.count("\ndef kernel_fn(data):\n") == 1
    r1 = r1.replace("\ndef kernel_fn(data):\n", "\ndef _r1_kernel_fn(data):\n")
    route = (r1.rstrip(NL) + NL + NL + NL + out.rstrip(NL) + NL + NL + NL
             + "def kernel_fn(data):" + NL
             + '    """K1-FA4 v6 on its measured routes (k1fa4_route_ok); r1 (stock kernels) elsewhere or if the self-test'
             + NL + '    fails. The self-test runs once, on the first eager call of a routed shape."""' + NL
             + '    cfg = data["config"]' + NL
             + "    if k1fa4_route_ok(cfg):" + NL
             + '        if K1FA4_STATE["ready"] is None and not torch.cuda.is_current_stream_capturing():' + NL
             + '            k1fa4_self_test(data["qkv"].device)' + NL
             + '        if K1FA4_STATE["ready"]:' + NL
             + "            return k1fa4_run(data)" + NL
             + "    return _r1_kernel_fn(data)" + NL)
    ast.parse(route)
    ROUTE.write_bytes(route.encode("utf-8"))

    hits = scan(out)
    print(f"[gen] {OUT.name}: {len(out.encode())} bytes ({len(out.splitlines())} lines); vendored part "
          f"{len((NL.join(head + imp_lines + knobs) + body).encode())} bytes; {n_edits} renamed tokens; "
          f"{len(mapping)} renamed names")
    print(f"[gen] {ROUTE.name}: {len(route.encode())} bytes")
    print(f"[gen] scan {OUT.name}: {'CLEAN' if not hits else hits}")
    print(f"[gen] scan {ROUTE.name}: {scan(route) or 'CLEAN'}")
    return 0 if not hits else 1


if __name__ == "__main__":
    sys.exit(main())

"""gen_k3.py: generate kernels/muse_decode_layer/k3_v05.py from canonical sources (lessons.md rule R3).

    python3 tools/gen_k3.py            # writes k3_v05.py, prints a diff summary, asserts the invariants
    python3 tools/gen_k3.py --check    # same checks, compares with the file on disk, writes nothing

Canonical sources (read, never edited):
  kernels/muse_decode_layer/k3_v04.py        layer glue, r1 levers, Marlin bsm ranges, K2A code (USE_K2A = False),
                                             kernel_fn, prepare; the K1 v0 slice between its two banners is replaced
  kernels/muse_attention_decode/k1_v1.py     K1 v1 CUDA/C++ sources (byte-identical), host helpers, FP32 reference
                                             and self-test (fold, sep, fold per case + fold-counter reset check)
  kernels/muse_attention_decode/k1_v2.py     the measured K1 host tables (route / fold / min_chunk / cps); its
                                             kernel sources must equal k1_v1's (asserted)

Everything outside the K1 slice is emitted verbatim except a short, listed set of text patches (nonce, docstring,
switches block, switches print, prepare's summary print). Every patch must match exactly once. After writing, the
output is re-parsed and every top-level def/assignment outside the K1 slice is compared (ast.dump) with k3_v04.py;
K1 helpers are compared with k1_v1.py after the rename map. Only stdlib; no torch import.
"""

import ast
import difflib
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_LAYER = ROOT / "kernels/muse_decode_layer/k3_v04.py"
SRC_K1 = ROOT / "kernels/muse_attention_decode/k1_v1.py"
SRC_TABLES = ROOT / "kernels/muse_attention_decode/k1_v2.py"
OUT = ROOT / "kernels/muse_decode_layer/k3_v05.py"
NONCE = "k3-v05-1"

# ------------------------------------------------------------------------------------------------ intended values
WANT_TABLES = {
    "K1_ROUTE_MIN_BATCH": {("sliding", 1): 1, ("full", 1): 1},             # K1 for L=1 at every batch; L=8 -> r1
    "K1_FOLD_MIN_BATCH": {("sliding", 1): 1, ("sliding", 8): 1, ("full", 1): 8, ("full", 8): 8},
    "K1_MIN_CHUNK_T": {("sliding", 1): 2, ("sliding", 8): 4, ("full", 1): 2, ("full", 8): 2},
    "K1_CPS_T": {("sliding", 1): 1, ("sliding", 8): 2, ("full", 1): 2, ("full", 8): 2},  # sliding L=1: 1 CTA/SM
}
WANT_SWITCHES = {"USE_K1": True, "USE_K2A": False, "K1_SELF_TEST": True, "R1_QK_FUSED_FP8_STAGE": True}
# k3_v04 levers that must come through unchanged (compared by ast.dump of the value)
UNCHANGED_LEVERS = ("USE_K2A", "K2A_MIN_TOKENS", "K2A_MAX_TOKENS", "K1_SELF_TEST", "K2A_SELF_TEST",
                    "K2A_SELF_TEST_START_BY_S", "R1_QK_FUSED_FP8_STAGE", "R1_LOCAL_ID_TABLE", "R1_ALIGN_LOCAL_BUCKETS",
                    "R1_ALIGN_LOCAL_BUCKETS_MIN_TOKENS", "R1_POSTNORM_TRITON_ALL_T", "R1_UNPOOL_INTO_ZEROS",
                    "R1_ROUTER_SPLIT_ALL_T", "R1_HOPPER_GEMV_T1", "BLOCK_SIZE_M_OVERRIDE", "BLOCK_SIZE_M_RANGES",
                    "NUM_SPLITS_OVERRIDE", "STAGE_LOC_MAX", "K2A_CPP_SRC", "K2A_CUDA_SRC")

# k1_v1 name -> layer name (one regex pass, identifier boundaries; also applies inside f-strings)
RENAME = {
    "CPP_SRC": "K1_CPP_SRC", "CUDA_SRC": "K1_CUDA_SRC", "_build": "_k1_build", "_MOD": "_K1_MOD",
    "_STATE": "_K1_STATE", "_SMS": "_K1_SMS", "_OCC": "_K1_OCC", "_INV_FREQ": "_K1_INV_FREQ", "_CNT": "_K1_CNT",
    "_sms": "_k1_sms", "_occupancy": "_k1_occupancy", "_counters": "_k1_counters", "_inv_freq": "_k1_inv_freq",
    "_kind": "_k1_kind", "_k1": "_k1_run", "_nsg": "_k1_nsg", "_rms_t": "_k1_rms_t", "_rope_t": "_k1_rope_t",
    "_mk_case": "_k1_mk_case", "_ref_attention": "_k1_ref_attention", "_ref_new_kv": "_k1_ref_new_kv",
    "_rel_rows": "_k1_rel_rows", "_self_test": "_k1_self_test",
}
_RENAME_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(sorted(map(re.escape, RENAME), key=len, reverse=True))
                        + r")(?![A-Za-z0-9_])")

K1_VERBATIM_FNS = ("_nsg", "_build", "_sms", "_occupancy", "_inv_freq", "_kind", "_rms_t", "_rope_t", "_mk_case",
                   "_ref_attention", "_ref_new_kv", "_rel_rows")
K1_ADAPTED_FNS = ("_counters", "_k1", "_self_test")
LAYER_K1_FNS = ("_k1_supported", "_k1_ready", "_k1_prepare")      # v04 layer glue, adapted
LAYER_K1_DROPPED = ("_k1_ctas_per_sm", "K1_MIN_CHUNK", "K1_CPS", "K1_ROUTE")

BANNED_NAMES = {"os", "sys", "subprocess", "environ", "threading", "eval", "exec", "open", "__import__", "getenv"}
SYNC_ATTRS = {"tolist", "item", "synchronize", "cpu", "numpy"}


def die(msg):
    raise SystemExit(f"[gen_k3] INVARIANT FAILED: {msg}")


def check(cond, msg):
    if not cond:
        die(msg)


def sha(s):
    return hashlib.sha1(s.encode()).hexdigest()


def rename(text):
    return _RENAME_RE.sub(lambda m: RENAME[m.group(1)], text)


def patch(text, old, new, what, count=1):
    n = text.count(old)
    check(n == count, f"patch '{what}': expected {count} match(es), found {n}")
    return text.replace(old, new)


class Src:
    """A parsed source file: top-level nodes by name, exact line slices, literal values."""

    def __init__(self, path):
        self.path = path
        self.text = path.read_text()
        self.lines = self.text.splitlines(keepends=True)
        self.tree = ast.parse(self.text)
        self.nodes = {}
        for node in self.tree.body:
            for name in self._names(node):
                check(name not in self.nodes or name in ("_ex",), f"{path.name}: duplicate top-level name {name}")
                self.nodes[name] = node

    @staticmethod
    def _names(node):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            return [node.name]
        if isinstance(node, ast.Assign):
            return [t.id for t in node.targets if isinstance(t, ast.Name)]
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return [node.target.id]
        return []

    def seg(self, name):
        """Exact source lines of a top-level node (keeps trailing comments on its last line)."""
        node = self.nodes[name]
        return "".join(self.lines[node.lineno - 1:node.end_lineno])

    def value(self, name):
        node = self.nodes[name]
        return eval(compile(ast.Expression(node.value), str(self.path), "eval"), {"__builtins__": {}})  # noqa: S307

    def fn(self, name):
        node = self.nodes[name]
        check(isinstance(node, ast.FunctionDef), f"{self.path.name}: {name} is not a function")
        return node

    def find_line(self, needle):
        hits = [i for i, ln in enumerate(self.lines) if ln.rstrip("\n") == needle]
        check(len(hits) == 1, f"{self.path.name}: line {needle!r} found {len(hits)} times")
        return hits[0]


def dump(node):
    return ast.dump(node, include_attributes=False)


# ================================================================================================ build the K1 slice
def k1_section(lay, k1, tab):
    # --- the kernel sources: raw literal text, byte-identical; the tables' file must carry the same kernel
    for nm in ("CPP_SRC", "CUDA_SRC"):
        check(k1.value(nm) == tab.value(nm), f"k1_v2.py {nm} differs from k1_v1.py (tables measured on another kernel)")
        check('"""' not in k1.value(nm), f"{nm} contains triple quotes")
        check(k1.seg(nm).startswith(f'{nm} = r"""'), f"{nm} is not a raw triple-quoted literal")
    cpp = "K1_" + k1.seg("CPP_SRC")
    cuda = "K1_" + k1.seg("CUDA_SRC")

    # --- verbatim helpers (renamed)
    fns = {name: rename(k1.seg(name)) for name in K1_VERBATIM_FNS + K1_ADAPTED_FNS}

    # --- adapted: fold counters (the layer allocates them in prepare; a miss under capture is reported, not hidden)
    fns["_counters"] = patch(
        fns["_counters"],
        'print("[k1] fold counters first needed under graph capture: separate combine for this call", flush=True)',
        'print("[k3] K1 fold counters first needed under graph capture (prepare allocates them): r1 path", '
        'flush=True)', "counters message")

    # --- adapted: _k1 -> _k1_run(qkv, data, cfg, ...): the layer hands over its own projection output
    f = fns["_k1"]
    f = patch(f, "def _k1_run(data, fold=None, min_chunk=None, cps=None, nsg=None):\n",
              "def _k1_run(qkv, data, cfg, fold=None, min_chunk=None, cps=None, nsg=None):\n", "_k1_run signature")
    f = patch(f, '    """One K1 call. Overrides (lab / self-test): fold (bool), min_chunk, cps (target CTAs/SM), nsg."""\n',
              '    """k1_v1.py\'s _k1() on the layer\'s raw qkv projection [T, (2Hq+2Hkv)*64] BF16 (q|k|v|gate) -> [B, L, Hq*64]\n'
              '    BF16 = RMSNorm(attention) x sigmoid(gate); writes the block\'s K/V (E4M3) into the caches at cache_loc.\n'
              '    fold / min_chunk / cps / nsg / Gx come from the host tables and (kind, B, L) only (shape-only, graph-safe);\n'
              '    the fold counters are the persistent buffer prepare() allocated (the kernel resets every counter it\n'
              '    uses). Overrides (self-test): fold (bool), min_chunk, cps (target CTAs/SM), nsg."""\n',
              "_k1_run docstring")
    f = patch(f, '    cfg = data["config"]\n', "", "_k1_run cfg line")
    f = patch(f, '    qkv = data["qkv"]\n', "", "_k1_run qkv line")
    f = patch(f,
              "    cnt = _k1_counters(dev) if fold else None\n"
              "    if fold and (cnt is None or B * hp > cnt.numel()):\n"
              "        fold = False\n"
              "    if cnt is None:\n"
              "        cnt = _K1_CNT.get(str(dev))\n"
              "        if cnt is None:\n"
              "            cnt = torch.zeros(1, dtype=torch.int32, device=dev)   # unused when fold == 0\n",
              "    cnt = _k1_counters(dev)   # persistent zeroed int32 [K1_COUNTERS], allocated eagerly (prepare)\n"
              "    if cnt is None:\n"
              "        raise RuntimeError(\"k1: fold counters missing under graph capture\")\n"
              "    if fold and B * hp > cnt.numel():\n"
              "        fold = False\n",
              "_k1_run counters block")
    fns["_k1"] = f

    # --- adapted: self-test (cases hoisted to K1_SELF_TEST_CASES; layer entry point; [k3] failure line)
    st_node = k1.fn("_self_test")
    case_nodes = [n for n in st_node.body if isinstance(n, ast.Try) is False and isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "cases"]
    check(len(case_nodes) == 1, "k1_v1 _self_test: one `cases = (...)` assignment")
    cn = case_nodes[0]
    case_lines = "".join(k1.lines[cn.lineno - 1:cn.end_lineno])
    case_value = eval(compile(ast.Expression(cn.value), "cases", "eval"), {"__builtins__": {}})  # noqa: S307
    check(case_lines.startswith("    cases = ("), "cases block indentation")
    tuple_text = case_lines[len("    cases = "):]
    module_cases = "K1_SELF_TEST_CASES = " + tuple_text.replace("\n    ", "\n")
    check(eval(compile(module_cases.split("=", 1)[1].strip(), "cases", "eval"), {"__builtins__": {}}) == case_value,  # noqa: S307
          "hoisted self-test cases differ")
    s = fns["_self_test"]
    s = patch(s, case_lines, "    cases = K1_SELF_TEST_CASES\n", "self-test cases")
    s = patch(s, '                out = _k1_run(d, fold=(mode == "fold"))\n',
              '                out = _k1_run(d["qkv"], d, d["config"], fold=(mode == "fold"))\n', "self-test call")
    s = patch(s, '        print(f"[k1] self-test FAILED, using the fallback path: {ex!r}", flush=True)\n',
              '        print(f"[k3] K1 self-test FAILED: attention stays on the r1 FA3 path: {ex!r}", flush=True)\n',
              "self-test failure line")
    fns["_self_test"] = s

    # --- layer glue from k3_v04, adapted to v1's tables and counters
    sup = lay.seg("_k1_supported")
    sup = patch(sup, '"""K1\'s own shape/dtype requirements (k1_forward\'s TORCH_CHECKs), on the layer\'s inputs, plus K1_ROUTE."""',
                '"""K1\'s own shape/dtype requirements (k1_forward\'s TORCH_CHECKs), on the layer\'s inputs, plus the route\n'
                '    table K1_ROUTE_MIN_BATCH (a (kind, q_len) class without an entry takes the r1 path)."""',
                "_k1_supported docstring")
    sup = patch(sup, '(kind, cfg["q_len"]) in K1_ROUTE and ', "", "_k1_supported route")
    # v04 passed .contiguous() copies of seq_lens / cache_loc / page_table; v1 passes them as they are, so a
    # non-contiguous one (layout only, never values) routes that call to r1 instead of tripping a TORCH_CHECK.
    sup = patch(sup, 'and data["seq_lens"].dtype == torch.int32 and data["cache_loc"].dtype == torch.int64)',
                'and data["seq_lens"].dtype == torch.int32 and data["cache_loc"].dtype == torch.int64\n'
                '            and data["seq_lens"].is_contiguous() and data["cache_loc"].is_contiguous()\n'
                '            and data["page_table"].is_contiguous())', "_k1_supported contiguity")
    rdy = lay.seg("_k1_ready")
    rdy = patch(rdy, '"""The per-(L, theta, device) constants were filled eagerly in prepare(); a miss under graph capture',
                '"""The per-(L, theta, device) constants and the fold counters were filled eagerly in prepare(); a miss under graph capture',
                "_k1_ready docstring")
    rdy = patch(rdy, "           and str(dev) in _K1_SMS)\n", "           and str(dev) in _K1_SMS and str(dev) in _K1_CNT)\n",
                "_k1_ready counters")
    prep = lay.seg("_k1_prepare")
    prep = patch(prep, '"""Self-test K1, then make sure every constant a captured call looks up exists (both L, both thetas)."""',
                 '"""Self-test K1 (both merge modes + counter reset), then make sure every constant a captured call looks up\n'
                 '    exists (SM count, occupancy for both L, inv_freq for both thetas) and the persistent fold counters."""',
                 "_k1_prepare docstring")
    prep = patch(prep, "            _k1_ctas_per_sm(L, dev)\n", "            _k1_occupancy(L, dev)\n", "_k1_prepare occupancy")
    prep = patch(prep, "            _k1_inv_freq(theta, dev)\n",
                 "            _k1_inv_freq(theta, dev)\n"
                 "        if _k1_counters(dev) is None:   # eager here: allocates the zeroed buffer once (shape-only)\n"
                 "            raise RuntimeError(\"fold counters not allocated\")\n", "_k1_prepare counters")

    # --- the import-time build block, verbatim from k3_v04 (_K1_MOD = None ... the if USE_K1 block)
    b0 = lay.nodes["_K1_MOD"].lineno - 1
    b1 = lay.nodes["_K1_SMS"].lineno - 1
    build_block = "".join(lay.lines[b0:b1]).rstrip("\n") + "\n"
    check("if USE_K1:" in build_block and "_K1_MOD = _k1_build()" in build_block, "v04 build block shape")

    banner = "# " + "=" * 94 + "\n"
    parts = [
        banner,
        "# K1: fused attention v1 (kernels/muse_attention_decode/k1_v1.py, nonce k1-v1-1; judge jobs d857095aa064476f,\n"
        "# f7843f722b124e77). GENERATED by tools/gen_k3.py: CUDA/C++ sources byte-identical to k1_v1.py; host helpers,\n"
        "# FP32 reference and self-test copied from k1_v1.py with names prefixed _k1_/K1_; host tables (top of file)\n"
        "# copied from k1_v2.py. Python glue adapted to the layer (qkv comes from the layer's projection; constants and\n"
        "# the persistent fold counters are filled in prepare()).\n",
        banner, "\n",
        "".join(k1.seg(n) for n in ("K1_TILE", "K1_STAGES", "K1_NEWK", "K1_COUNTERS")), "\n\n",
        fns["_nsg"], "\n\n",
        "# (B, ctx, L, sliding, page, kv_scale, seed, heads): k1_v1.py's self-test cases; each runs fold, sep, fold and\n"
        "# checks that the fold counters are back at zero after every call. All of them gate K1 (L=8 included).\n",
        module_cases, "\n",
        cpp, "\n", cuda, "\n\n",
        fns["_build"], "\n\n",
        build_block, "\n",
        "_K1_STATE = {}\n_K1_SMS = {}\n_K1_OCC = {}\n_K1_INV_FREQ = {}\n"
        "_K1_CNT = {}  # device -> persistent int32 fold counters (zeroed once in prepare; the kernel resets them)\n"
        "_K3_WARNED = set()\n\n\n",
        fns["_sms"], "\n\n", fns["_occupancy"], "\n\n", fns["_counters"], "\n\n", fns["_inv_freq"], "\n\n",
        fns["_kind"], "\n\n", sup, "\n\n", rdy, "\n\n", fns["_k1"], "\n\n",
        fns["_rms_t"], "\n\n", fns["_rope_t"], "\n\n", fns["_mk_case"], "\n\n", fns["_ref_attention"], "\n\n",
        fns["_ref_new_kv"], "\n\n", fns["_rel_rows"], "\n\n", fns["_self_test"], "\n\n", prep, "\n\n",
    ]
    return "".join(parts), case_value


# ================================================================================================ splice
def generate():
    lay, k1, tab = Src(SRC_LAYER), Src(SRC_K1), Src(SRC_TABLES)
    tables = {nm: tab.value(nm) for nm in WANT_TABLES}
    for nm, want in WANT_TABLES.items():
        check(tables[nm] == want, f"k1_v2.py {nm} = {tables[nm]} != intended {want}")
    check(k1.value("K1_COUNTERS") == tab.value("K1_COUNTERS") == 1 << 16, "K1_COUNTERS")
    check(k1.seg("_nsg") == tab.seg("_nsg"), "_nsg differs between k1_v1 and k1_v2")

    i0 = lay.find_line("# K1: fused attention (kernels/muse_attention_decode/k1_v0.py, nonce k1-v0-1, review_k1_v0.md).") - 1
    i1 = lay.find_line("# K2a: custom NVFP4 W4A16 experts (kernels/muse_moe_w4a16/k2a_v0.py, nonce k2a-v0-1, review_k2a_v0.md).") - 1
    check(lay.lines[i0].startswith("# ====") and lay.lines[i1].startswith("# ===="), "K1 slice banners")
    head, tail = "".join(lay.lines[:i0]), "".join(lay.lines[i1:])
    section, case_value = k1_section(lay, k1, tab)

    # --- head patches (nonce, generated-by line, docstring, switches block, switches print)
    head = patch(head, "# nonce k3-v04-1 (k3_v03 + measured Marlin block_size_m ranges; K1 route table; K2A off)\n",
                 f"# nonce {NONCE} (k3_v04 + K1 v1: L=1 at every batch, fold/separate combine per k1_v2 tables, sliding L=1 1 CTA/SM;"
                 " L=8 -> r1 path with L2; K2A off)\n"
                 "# GENERATED by tools/gen_k3.py from k3_v04.py (layer), k1_v1.py (K1 sources/helpers/self-test) and k1_v2.py\n"
                 "# (K1 host tables). Do not hand-edit: change a canonical source or the generator and regenerate.\n",
                 "nonce")
    head = patch(head, "  USE_K1  (kernels/muse_attention_decode/k1_v0.py): ONE fused attention launch pair replaces r1's FusedQKNorm,\n",
                 "  USE_K1  (kernels/muse_attention_decode/k1_v1.py, host tables from k1_v2.py): ONE fused attention launch (plus\n"
                 "       a combine launch only on full layers below K1_FOLD_MIN_BATCH) replaces r1's FusedQKNorm,\n",
                 "docstring K1 line")
    head = patch(head, "prepare() self-tests each piece on synthetic inputs (K1: 9 synthetic cache cases vs an FP32 reference; K2a: 6\n",
                 "prepare() self-tests each piece on synthetic inputs (K1: 11 synthetic cache cases, each in fold, separate and\n"
                 "fold mode, vs an FP32 reference, plus the fold-counter reset check; K2a: 6\n",
                 "docstring self-test line")
    old_route = ('# (layer kind, q_len) classes that take K1 once it is up; drop an entry to send that class to FA3.\n'
                 + lay.seg("K1_ROUTE") + lay.seg("K1_ROUTE_MIN_BATCH"))
    new_route = ("# K1 v1 host tables, keyed by (layer kind, q_len), copied from kernels/muse_attention_decode/k1_v2.py by\n"
                 "# tools/gen_k3.py (measured: jobs d857095aa064476f k1_v1, f7843f722b124e77 lab).\n"
                 "#   route: smallest batch that takes K1; a class without an entry (L=8) takes the r1 path (L2 FP8 stage + FA3).\n"
                 + tab.seg("K1_ROUTE_MIN_BATCH")
                 + "#   fold (last-CTA merge, no combine launch) when batch >= this; else the separate k1_combine.\n"
                 + tab.seg("K1_FOLD_MIN_BATCH")
                 + "#   minimum tiles (64 keys) per split.\n"
                 + tab.seg("K1_MIN_CHUNK_T")
                 + "#   target CTAs per SM when sizing Gx (capped by the measured occupancy of k1_main<L>).\n"
                 + tab.seg("K1_CPS_T"))
    head = patch(head, old_route, new_route, "switches block")
    head = patch(head, "K1_ROUTE={sorted(K1_ROUTE)} K1_ROUTE_MIN_BATCH={K1_ROUTE_MIN_BATCH} K2A_TOKENS=",
                 "K1=v1 route={K1_ROUTE_MIN_BATCH} fold={K1_FOLD_MIN_BATCH} min_chunk={K1_MIN_CHUNK_T}"
                 " cps={K1_CPS_T} K2A_TOKENS=", "switches print")

    # --- tail patch: prepare's summary line (occupancy is raw now; cps per class comes from K1_CPS_T)
    tail = patch(tail, 'f" (USE_K1={USE_K1} USE_K2A={USE_K2A}); nvcc {builds}; k1 CTAs/SM {occ};"',
                 'f" (USE_K1={USE_K1} USE_K2A={USE_K2A}); nvcc {builds}; k1 v1 occupancy L1/L8 {occ} CTAs/SM,"\n'
                 '          f" route {K1_ROUTE_MIN_BATCH}, cps {K1_CPS_T}, fold counters {\'ok\' if _K1_CNT else \'none\'};"',
                 "prepare summary")
    return head + section + tail, (lay, k1, tab, i0, i1, case_value)


# ================================================================================================ verify
def verify(text, ctx):
    lay, k1, tab, i0, i1, case_value = ctx
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text)
    try:
        py_compile.compile(str(tmp), doraise=True)
        out = Src(tmp)
    finally:
        tmp.unlink()
    report = []

    # nonce
    check(text.splitlines()[1].startswith(f"# nonce {NONCE} "), "nonce line")
    check(text.count(NONCE) == 1, "nonce appears once")

    # K1 sources: parsed values from the EMITTED file == k1_v1's, sha1 printed
    for nm in ("CPP_SRC", "CUDA_SRC"):
        v_out, v_k1 = out.value("K1_" + nm), k1.value(nm)
        check(v_out == v_k1, f"K1_{nm} differs from k1_v1.py {nm}")
        check(out.seg("K1_" + nm) == "K1_" + k1.seg(nm), f"K1_{nm} raw literal text differs")
        report.append(f"K1_{nm}: byte-identical to k1_v1.py {nm} (sha1 {sha(v_out)[:12]}, {len(v_out)} chars)")
    v04_k1 = sha(lay.value("K1_CPP_SRC") + lay.value("K1_CUDA_SRC"))[:10]
    v05_k1 = sha(out.value("K1_CPP_SRC") + out.value("K1_CUDA_SRC"))[:10]
    check(v04_k1 != v05_k1, "K1 sources did not change")
    report.append(f"load_inline name: k1_v0_{v04_k1} (k3_v04) -> k1_v1_{v05_k1} (k3_v05, = k1_v1.py's)")
    for nm in ("K2A_CPP_SRC", "K2A_CUDA_SRC"):
        check(out.value(nm) == lay.value(nm), f"{nm} differs from k3_v04")
    report.append(f"K2A sources: identical to k3_v04 (sha1 {sha(out.value('K2A_CUDA_SRC'))[:12]})")

    # tables and switches: intended values
    for nm, want in WANT_TABLES.items():
        check(out.value(nm) == want, f"{nm} = {out.value(nm)}")
    for nm, want in WANT_SWITCHES.items():
        check(out.value(nm) is want, f"{nm} = {out.value(nm)}")
    check(out.value("K1_COUNTERS") == 1 << 16, "K1_COUNTERS")
    check(out.value("K1_SELF_TEST_CASES") == case_value, "K1_SELF_TEST_CASES")
    check({c[2] for c in case_value} == {1, 8} and len(case_value) == 11, "self-test covers L=1 and L=8, 11 cases")
    for nm in LAYER_K1_DROPPED:
        check(nm not in out.nodes, f"{nm} should be gone")
        check(re.search(r"(?<![A-Za-z0-9_])" + nm + r"(?![A-Za-z0-9_])", text) is None, f"{nm} still referenced")
    for nm in UNCHANGED_LEVERS:
        check(dump(out.nodes[nm].value) == dump(lay.nodes[nm].value), f"lever {nm} changed vs k3_v04")
    report.append("unchanged vs k3_v04: " + ", ".join(f"{n}={lay.seg(n).split('=', 1)[1].split('#')[0].strip()}"
                                                       for n in UNCHANGED_LEVERS if not n.endswith("_SRC")))
    report.append("K1 tables (from k1_v2.py): " + "; ".join(f"{n}={out.value(n)}" for n in WANT_TABLES))

    # top-level names: everything outside the K1 slice must be ast-identical to k3_v04
    slice_names_v04 = {n for n, node in lay.nodes.items() if i0 < node.lineno <= i1}
    patched = {"prepare"}           # prepare: summary print only
    same, changed = [], []
    for name, node in lay.nodes.items():
        if name in slice_names_v04 or name in ("K1_ROUTE", "K1_ROUTE_MIN_BATCH"):
            continue
        check(name in out.nodes, f"{name} lost")
        if dump(out.nodes[name]) == dump(node):
            same.append(name)
        else:
            changed.append(name)
    check(set(changed) == patched, f"unexpected changes outside the K1 slice: {sorted(set(changed) - patched)}")
    pv04, pv05 = ast.unparse(lay.nodes["prepare"]).splitlines(), ast.unparse(out.nodes["prepare"]).splitlines()
    pdiff = [ln for ln in difflib.unified_diff(pv04, pv05, lineterm="", n=0) if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    check(all("occupancy" in ln or "CTAs/SM" in ln or "route" in ln for ln in pdiff), f"prepare diff beyond summary: {pdiff}")
    report.append(f"outside the K1 slice: {len(same)} top-level defs/assignments ast-identical to k3_v04 (incl. kernel_fn,"
                  f" _r1_attention, _experts, _r1_prepare, K2A code); changed: prepare (summary print only)")
    # module-level statements that are not named (imports, prints, if/try blocks) must match too
    def unnamed(src, lo, hi):
        return [n for n in src.tree.body if not Src._names(n) and not (lo < n.lineno <= hi)
                and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    o0 = out.find_line("# K1: fused attention v1 (kernels/muse_attention_decode/k1_v1.py, nonce k1-v1-1; judge jobs d857095aa064476f,") - 1
    o1 = out.find_line("# K2a: custom NVFP4 W4A16 experts (kernels/muse_moe_w4a16/k2a_v0.py, nonce k2a-v0-1, review_k2a_v0.md).") - 1
    u04, u05 = unnamed(lay, i0, i1), unnamed(out, o0, o1)
    check(len(u04) == len(u05), "unnamed top-level statement count changed")
    ud = [b for a, b in zip(u04, u05) if dump(a) != dump(b)]
    check(len(ud) == 1 and "[k3] switches" in ast.unparse(ud[0]),
          f"unnamed statements changed: {[ast.unparse(n)[:80] for n in ud]}")   # exactly one: the [k3] switches print
    report.append("unnamed top-level statements (imports, prints, build blocks): all identical except the [k3] switches print")

    # K1 helpers vs k1_v1 (after the rename map)
    k1_same, k1_adapted = [], []
    for name in K1_VERBATIM_FNS + K1_ADAPTED_FNS:
        ref = ast.parse(rename(k1.seg(name))).body[0]
        got = out.nodes[RENAME[name]]
        if dump(ref) == dump(got):
            k1_same.append(RENAME[name])
        else:
            k1_adapted.append(RENAME[name])
    check(set(k1_same) == {RENAME[n] for n in K1_VERBATIM_FNS}, f"verbatim K1 helpers changed: {sorted(set(RENAME[n] for n in K1_VERBATIM_FNS) - set(k1_same))}")
    check(set(k1_adapted) == {RENAME[n] for n in K1_ADAPTED_FNS}, f"adapted set: {k1_adapted}")
    report.append(f"K1 helpers ast-identical to k1_v1.py after renames: {', '.join(k1_same)}")
    report.append(f"K1 helpers adapted (listed patches): {', '.join(k1_adapted)}; layer glue from k3_v04 adapted: "
                  f"{', '.join(LAYER_K1_FNS)}; dropped: {', '.join(LAYER_K1_DROPPED)}")
    # v04's verbatim build block and dicts survive
    check(dump(out.nodes["_K1_MOD"]) == dump(lay.nodes["_K1_MOD"]), "_K1_MOD")

    # static scan: banned names/imports; host syncs only in prepare-time functions
    for node in ast.walk(out.tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                check(m.split(".")[0] not in BANNED_NAMES, f"banned import {m}")
        if isinstance(node, ast.Name):
            check(node.id not in BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
        if isinstance(node, ast.Attribute):
            check(node.attr not in ("environ", "system", "popen"), f"banned attribute {node.attr}")

    def sync_fns(src):
        res = set()
        for node in src.tree.body:
            if isinstance(node, ast.FunctionDef):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and sub.attr in SYNC_ATTRS:
                        res.add(node.name)
        return res
    s04, s05 = sync_fns(lay), sync_fns(out)
    allowed_new = {"_k1_self_test", "_k1_ref_attention", "_k1_mk_case", "_k1_ref_new_kv", "_k1_rope_t"}
    check(s05 - s04 <= allowed_new, f"new functions with host syncs: {sorted(s05 - s04 - allowed_new)}")
    for hot in ("kernel_fn", "_k1_run", "_k1_supported", "_k1_ready", "_k1_counters", "_r1_attention", "_experts"):
        check(hot not in s05, f"{hot} contains a host-sync call")
    report.append(f"static: no os/sys/subprocess/environ/threading/eval/exec/open; host-sync calls only in "
                  f"{', '.join(sorted(s05))}")

    # the K1 call surface: 25 args to k1_forward in k1_v1's order
    calls = [n for n in ast.walk(out.nodes["_k1_run"]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "k1_forward"]
    check(len(calls) == 1 and len(calls[0].args) == 25, "k1_forward call has 25 args")
    check(ast.unparse(calls[0].args[12]) == "cnt", "counters passed after out, before B")
    report.append("k1_forward call: 25 args, counters (persistent buffer) after `out`, fold/nsg last (k1_v1 order)")
    return out, report


def diff_summary(new_text, ctx):
    """Hunks outside the K1 slice (line-level), and the K1 slice as one replaced block."""
    lay, i0, i1 = ctx[0], ctx[3], ctx[4]
    a, b = lay.text.splitlines(), new_text.splitlines()
    o0 = b.index("# K1: fused attention v1 (kernels/muse_attention_decode/k1_v1.py, nonce k1-v1-1; judge jobs d857095aa064476f,") - 1
    o1 = b.index("# K2a: custom NVFP4 W4A16 experts (kernels/muse_moe_w4a16/k2a_v0.py, nonce k2a-v0-1, review_k2a_v0.md).") - 1
    lines = []
    for (a0_, a1_, b0_, b1_) in ((0, i0, 0, o0), (i1, len(a), o1, len(b))):
        sm = difflib.SequenceMatcher(a=a[a0_:a1_], b=b[b0_:b1_], autojunk=False)
        for tag, x0, x1, y0, y1 in sm.get_opcodes():
            if tag == "equal":
                continue
            first = (b[b0_ + y0] if y1 > y0 else a[a0_ + x0]).strip()[:70]
            lines.append(f"  {tag:7s} k3_v04 l.{a0_ + x0 + 1}-{a0_ + x1} -> k3_v05 l.{b0_ + y0 + 1}-{b0_ + y1}: {first}")
        if a0_ == 0:
            lines.append(f"  replace k3_v04 l.{i0 + 1}-{i1} (K1 v0 slice, {i1 - i0} lines) -> k3_v05 l.{o0 + 1}-{o1}"
                         f" (K1 v1 slice, {o1 - o0} lines)")
    return lines, len(a), len(b)


def main(argv):
    text, ctx = generate()
    out, report = verify(text, ctx)
    hunks, na, nb = diff_summary(text, ctx)
    print(f"[gen_k3] {SRC_LAYER.relative_to(ROOT)} ({na} lines) + {SRC_K1.relative_to(ROOT)} + "
          f"{SRC_TABLES.relative_to(ROOT)} -> {OUT.relative_to(ROOT)} ({nb} lines), nonce {NONCE}")
    print(f"[gen_k3] diff vs k3_v04: {len(hunks)} hunks")
    for h in hunks:
        print(h)
    for r in report:
        print(f"[gen_k3] OK  {r}")
    if "--check" in argv:
        disk = OUT.read_text() if OUT.exists() else ""
        check(disk == text, f"{OUT.name} on disk differs from the generator's output (regenerate)")
        print(f"[gen_k3] --check: {OUT.name} on disk is up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[gen_k3] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}); all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

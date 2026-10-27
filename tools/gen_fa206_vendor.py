"""Generate kernels/muse_attention_decode/fa206_vendor.py: the SM90 native-FP8 forward of flash-attention PR #206 (pinned
commit 303f735573162381de2dc22dc202be6aa594c264), reduced to what the decode layer's ONE configuration executes, as plain
top-level source with every bound name prefixed p206_ (so it can be pasted at the top level of the layer file).

Configuration (tools/fa206_patches.py builds only this one and checks it on the compiled object): E4M3 Q/K/V, head dim
64 / 64, GQA 8 with pack-GQA, q_len 8 -> tile 64 x 128 with one MMA warpgroup, paged KV with page size 1 (cp.async, no TMA KV),
causal full attention, SplitKV (FP32 partials; the combine is the layer's own), seqused_k, two CTAs per SM; no cu_seqlens,
score / mask mod, block sparsity, learnable sink, descales, dynamic causal or output quant.

What the generator does (nothing is loaded, patched or written at run time):
- reads the pinned snapshot kernels/lab/fresh_fa4/cute/*.py (tools/build_fresh_fa4_snapshot.py; it equals the PR's files up
  to the fresh_fa4.cute namespace and CRLF) and the image copy box_out/fa4_image/*.py, both checked against PIN_* sha256;
- slices the SPEC symbols by AST (whole classes, or the listed methods; fields and nested classes always kept). Method
  bodies are NOT edited: every const_expr branch stays, so the DSL traces exactly the PR's code for this configuration. The
  methods left out are ones this configuration never calls (TMA K/V, fp8-KV dequant, intra-WG overlap, hdim 256, descales,
  score mod, SM100 / CLC paths); a wrong omission fails the prepare self-test (AttributeError), it cannot change results;
- every other flash_attn.cute symbol the slices still reference is imported from the judge image's installed copy
  (sglang.kernels.ops.attention.flash_attn.cute), only if its source there is AST-identical to the PR's and so is
  everything it references, or it is listed in EQUIV (reviewed: same behaviour here) or DEAD (bound, never executed);
- renames by scope-aware AST (locals, parameters, keyword and attribute names untouched); cutlass, cute, const_expr, Int32
  and Float32 keep their names (the CuTe DSL preprocessor may match them by name);
- emits ast.unparse of each symbol (docstrings and comments dropped) after the provenance / license header and imports.

    python tools/gen_fa206_vendor.py           # writes the file, prints size and the symbol table
    python tools/gen_fa206_vendor.py --check   # fails if the file on disk differs
"""
import ast
import builtins
import copy
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels" / "lab" / "fresh_fa4" / "cute"
IMG = ROOT / "box_out" / "fa4_image"
OUT = ROOT / "kernels" / "muse_attention_decode" / "fa206_vendor.py"
SHA = "303f735573162381de2dc22dc202be6aa594c264"
P = "p206_"
IMG_PKG = "sglang.kernels.ops.attention.flash_attn.cute"
NS = "fresh_fa4.cute"
KEEP = {"cutlass", "cute", "const_expr", "Int32", "Float32"}

PIN_PR = {  # kernels/lab/fresh_fa4/manifest.json lab_sha256
    "block_info": "dfdbefb8a51fbcff135e7060b1c44fb5729887c0697f25cf7ee56fe592e0a67a",
    "block_sparse_utils": "f73a9129165b2e2d5d3b127940f6137c839db2ebed4f3d5c96a8ab47fedc3c2f",
    "block_sparsity": "84a68074d1eda6264521015dc78fac65188e8535981a5f33b56ed392f6585bb5",
    "cute_dsl_utils": "ca7b233b8dff386f7b9408aed46dae1e469bb2e731b4fd7a8d6faf2bd3040e42",
    "fast_math": "eac26eb81e1aba103c747ac6343df1c82a2f7fc97af755de5c3925e32ae9132f",
    "flash_fwd": "e2053ec24e9790c96a5211925d864fac540f7382f5777ea444c4aafa767d4549",
    "flash_fwd_sm90": "d4d763ff21f02430f3b897278bca885d6c6f67fd3f3679958fa6bc63040aa825",
    "mask": "ca62f71a6061542316176f914241b037d8f2a9000484eca1b513b4c7df5f6f0d",
    "named_barrier": "1efca7c89ef6d61f470f09a68f854e648181d3c79d49acca97f3729c691bffa4",
    "pack_gqa": "e14d368381268bbb114911e0d6bb7893b219ebcec9d9dd4d289dbecb49af3d69",
    "paged_kv": "502a6717c5f6520d54d56c21a4c75e86dad2945501dc4064bfd896b649815d00",
    "pipeline": "ea90e05b6aab18617d3fc456fde778d8bac658372bb179d09c3ca9e68238f65c",
    "seqlen_info": "d292a4f6fbad59425fbab68afbda2533a378ba1c86ad0880c8269f1a672366fe",
    "softmax": "c0a16a2dc479eb27fbc556f76f3373e89bb2b36438aff0c65c233680306a5688",
    "tile_scheduler": "20b72dfd6c31bb262a7518232c10dc604b08137c43682de25ba76357a82bf7e3",
    "utils": "5370f6fcf0d7f4fdffa1cac768116c9ba324a8af94f97fce75369897ff502622",
}
PIN_IMG = {  # box_out/fa4_image (sglang/kernels/ops/attention/flash_attn/cute/ of lmsysorg/sglang:dev-muse-spark-open)
    "block_info": "607c304d0cd123595188537a9815d69282d3a7419cb457768d8c31993f25b359",
    "block_sparse_utils": "54abc93843b731ec2d114148c75798718221565fb09b0bc6923d70526fed7e90",
    "block_sparsity": "9ca21cf8b9ae5f7c44023df646f70b793667cdce2bbb09ba15f5f7a3a19a4c26",
    "cute_dsl_utils": "03520beefd65d44c5bb1d3420e18664f0202b3c7926ea9d5dd8c97a983ef2583",
    "fast_math": "7e5f822bf4d0ba36967558f3ae9a1408b410cb726c31222cd61703569d1be738",
    "flash_fwd": "97edf14f4be14d83176efc816f867c0bf751ae65b380942da9147932ce959ea4",
    "flash_fwd_sm90": "16815521b74bfc9e5fc16411e503da6ff5b4146a0d40e0563073bafcc90d61c1",
    "mask": "1f4a0980b684083ab4d8bd90dc617bc9f2b6ce8074532d9d5814e76230f413d9",
    "named_barrier": "729f4581800b31e1b6116bc44c0f9f5268b986cd7ed108d407d4bee4441ece46",
    "pack_gqa": "71b087e2f1d299a8c9e5181e6d083444e9a421f749b7cb51c5129abeef0aadf8",
    "paged_kv": "a80b0cafdca4a570722d6d4edaf9a35e5ec31d535360b14a931bbc393d51a359",
    "pipeline": "71f7416f99bd187758e940d51094e881a0e485c96d5fad5224830c7d5fa1f846",
    "seqlen_info": "e2444eb09131f12dddf4444e54985e60435b877012ea90e73ae4b8da6483e6d3",
    "softmax": "cca8b509587f2b433012b1f8691ecaf7128295d92249beda28824dfa792bfdc1",
    "tile_scheduler": "e6d1b4ebd9708f01cfb5e12f4aa54dc9f7d9b15d7a92c93a28faaf9407a7102d",
    "utils": "4d57a77e3eede5755dcb56cc1529652252c093a548d5cae4482025b53d231ba7",
}

SM90_KEEP = ["__init__", "_get_smem_layout_atom", "_get_tiled_mma", "_transpose_v_ldsm_native_fp8", "_stsm_v_native_fp8",
             "_get_shared_storage_cls", "__call__", "kernel", "load_native_fp8", "_load_kv_transpose_native_fp8",
             "_issue_kv_block_native_fp8", "_transpose_v_stage_native_fp8", "load_KV", "mma", "_mma_pv_tile_native_fp8",
             # mma() binds first / last_half_block_overlap with functools.partial (intra-WG overlap only; never called here)
             "first_half_block_overlap", "last_half_block_overlap",
             "mma_one_n_block", "cross_wg_p_barrier", "cross_wg_p_raw_barrier", "mma_init", "warp_scheduler_barrier_sync",
             "warp_scheduler_barrier_arrive"]

# (module, symbol, ("keep", methods) | ("drop", methods) | None = whole), in definition order
SPEC = [
    ("seqlen_info", "SeqlenInfoQK", None),
    ("block_info", "BlockInfo", ("keep", ["get_n_block_min_max", "get_n_block_min_causal_local_mask",
                                           "get_n_block_min_before_local_mask"])),
    ("mask", "AttentionMask", ("keep", ["seqlen_q", "seqlen_k", "apply_mask"])),
    ("utils", "permute_Cregs_fp8", None),
    ("utils", "convert_layout_acc_Aregs_fp8", None),
    ("softmax", "Softmax", None),
    ("paged_kv", "PagedKVManager", ("drop", ["_flatten_smem_sm100"])),
    ("pack_gqa", "PackGQA", ("keep", ["compute_ptr", "store_LSE", "store_O_splitkv"])),
    ("tile_scheduler", "TileSchedulerArguments", None),
    ("tile_scheduler", "SchedulerState", ("keep", ["consumer_wait", "consumer_release", "advance_consumer_state",
                                                   "producer_tail"])),
    ("tile_scheduler", "SingleTileLPTScheduler", ("drop", ["_clc_grid_shape", "clc_problem_shape", "clc_work_to_coords"])),
    ("flash_fwd", "FlashAttentionForwardBase", ("keep", ["__init__", "_setup_attributes", "epilogue"])),
    ("flash_fwd_sm90", "FlashAttentionForwardSm90", ("keep", SM90_KEEP)),
]

# imported from the image although the AST differs: reviewed, same behaviour on this configuration
EQUIV = {
    ("utils", "fmax"): "the image picks the nvvm.fmax signature by CUDA version, the PR always the new one (same op)",
    ("utils", "fmax_reduce"): "formatting only; calls utils.fmax",
    ("cute_dsl_utils", "assume_tensor_aligned"): "the image has an align=16 default; divisibility is 128 bits either way",
    ("cute_dsl_utils", "assume_strides_aligned"): "same",
    ("named_barrier", "NamedBarrierFwd"): "the PR appends members; Epilogue / WarpSchedulerWG1..3 keep their values",
}
# imported from the image only to bind the name: referenced on branches this configuration never executes
DEAD = {
    ("tile_scheduler", "SingleTileScheduler"): "non-causal or local",
    ("tile_scheduler", "SingleTileVarlenScheduler"): "cu_seqlens_q / seqused_q",
    ("block_sparse_utils", "produce_block_sparse_loads"): "block sparsity",
    ("block_sparse_utils", "consume_block_sparse_loads"): "block sparsity",
    ("softmax", "apply_score_mod_inner"): "score_mod",
}
# text edits on the unparsed output (each must hit once): message strings only, no executed behaviour changes
COSMETIC = [
    ("FlashAttentionForwardSm90", "f'Fused quant output not implemented for {type(self).__name__}'",
     "'Fused quant output not implemented for FlashAttentionForwardSm90'"),   # keeps `__name__` out of the file
]
# not in the image: a placeholder class keeps `TileScheduler is X` evaluable (False here)
SENTINEL = {("tile_scheduler", "SingleMBlockVarlenScheduler"): "direct (head, split, batch) grid needs cu_seqlens_q"}


def check(cond, msg):
    if not cond:
        raise SystemExit(f"[gen_fa206_vendor] FAIL: {msg}")


def norm_ns(s):
    s = s.replace("\r\n", "\n")
    for x in (NS, IMG_PKG, "flash_attn.cute"):
        s = s.replace(x, "X")
    return s


class Mod:
    def __init__(self, base, name, pin):
        self.name = name
        path = base / f"{name}.py"
        self.src = None
        self.defs, self.imports = {}, {}
        if not path.exists():
            return
        raw = path.read_bytes()
        check(name in pin and hashlib.sha256(raw).hexdigest() == pin[name], f"{path} sha256 not pinned / changed")
        self.src = raw.decode("utf-8").replace("\r\n", "\n")
        self.tree = ast.parse(self.src)
        for n in self.tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                self.defs[n.name] = n
            elif isinstance(n, ast.Import):
                for a in n.names:
                    self.imports[a.asname or a.name.split(".")[0]] = (a.name if a.asname else a.name.split(".")[0], None)
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    self.imports[a.asname or a.name] = (n.module, a.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            self.defs[x.id] = n

    def internal(self, local):
        """('sym', mod, name) | ('mod', mod) | ('ext', module, attr) | None (builtin / unbound)."""
        if local in self.defs:
            return ("sym", self.name, local)
        if local in self.imports:
            module, attr = self.imports[local]
            for ns in (NS, IMG_PKG):
                if module == ns and attr is not None:
                    return ("mod", attr)
                if module.startswith(ns + "."):
                    sub = module[len(ns) + 1:]
                    return ("mod", sub) if attr is None else ("sym", sub, attr)
            return ("ext", module, attr)
        return None


_mods = {}


def mod(side, name):
    key = (side, name)
    if key not in _mods:
        _mods[key] = Mod(SRC if side == "pr" else IMG, name, PIN_PR if side == "pr" else PIN_IMG)
    return _mods[key]


def resolve(side, modname, sym):
    """Follow `from X.a import b` re-exports to the defining module."""
    for _ in range(10):
        m = mod(side, modname)
        if sym in m.defs or sym not in m.imports:
            return modname, sym
        t = m.internal(sym)
        if not t or t[0] != "sym":
            return modname, sym
        modname, sym = t[1], t[2]
    return modname, sym


def node_refs(m, node):
    out = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name):
            t = m.internal(x.value.id)
            if t and t[0] == "mod":
                out.add(resolve("pr", t[1], x.attr))
        elif isinstance(x, ast.Name):
            t = m.internal(x.id)
            if t and t[0] == "sym":
                out.add(resolve("pr", t[1], t[2]))
    return out


def dump(n):
    return ast.dump(n, annotate_fields=False, include_attributes=False)


_deep = {}


def deep_identical(key, path=()):
    """PR symbol == image symbol (AST of the namespace-normalised source) and so is everything it references."""
    if key in _deep:
        return _deep[key]
    if key in path or key in EQUIV or key in DEAD:
        return True
    modname, sym = key
    a, b = mod("pr", modname), mod("img", modname)
    na, nb = a.defs.get(sym), (b.defs.get(sym) if b.src is not None else None)
    ok = (na is not None and nb is not None and dump(ast.parse(norm_ns(ast.get_source_segment(a.src, na))))
          == dump(ast.parse(norm_ns(ast.get_source_segment(b.src, nb)))))
    if ok:
        ok = all(deep_identical(r, path + (key,)) for r in node_refs(a, na) if r != key)
    _deep[key] = ok
    return ok


# ------------------------------------------------------------------------------------------------------------ slicing
def is_doc(stmt):
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def slice_symbol(m, sym, mode, report):
    node = copy.deepcopy(m.defs[sym])
    if isinstance(node, ast.ClassDef) and mode is not None:
        kind, names = mode
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        check(all(n in methods for n in names), f"{m.name}.{sym}: missing methods {[n for n in names if n not in methods]}")
        dropped = [n for n in methods if (n not in names if kind == "keep" else n in names)]
        node.body = [n for n in node.body if not (isinstance(n, ast.FunctionDef) and n.name in dropped)]
        report.append((m.name, sym, dropped))
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body and is_doc(n.body[0]):
            n.body = n.body[1:] or [ast.Pass()]
    return node


# ------------------------------------------------------------------------------------------------------------ renaming
class Scoper(ast.NodeVisitor):
    """Collects every Name / module Attribute that refers to a module-level binding (scope-aware)."""

    def __init__(self, m):
        self.m = m
        self.scopes = []
        self.hits = []
        self.ann_strings = []

    @staticmethod
    def stored(body_nodes):
        out = set()
        stack = list(body_nodes)
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
                continue
            if isinstance(n, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                out.add(n.id)
            if isinstance(n, ast.ExceptHandler) and n.name:
                out.add(n.name)
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                out |= {a.asname or a.name.split(".")[0] for a in n.names}
            check(not isinstance(n, (ast.Global, ast.Nonlocal)), "global / nonlocal in vendored code")
            stack.extend(ast.iter_child_nodes(n))
        return out

    def local(self, name):
        for i, (kind, locs) in enumerate(reversed(self.scopes)):
            if kind == "class" and i != 0:
                continue
            if name in locs:
                return True
        return False

    def visit_Name(self, node):
        if not self.local(node.id):
            self.hits.append(("name", node))

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and not self.local(node.value.id) and self.m is not None:
            t = self.m.internal(node.value.id)
            if t and t[0] == "mod":
                self.hits.append(("attr", node))
                return
        self.generic_visit(node)

    def _ann(self, a):
        if a is None:
            return
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            self.ann_strings.append(a)
        else:
            self.visit(a)

    def _args(self, args):
        for a in args.posonlyargs + args.args + args.kwonlyargs + [x for x in (args.vararg, args.kwarg) if x]:
            self._ann(a.annotation)
        for d in args.defaults + [d for d in args.kw_defaults if d is not None]:
            self.visit(d)

    @staticmethod
    def argnames(args):
        return {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs + [x for x in (args.vararg, args.kwarg) if x]}

    def visit_FunctionDef(self, node):
        for d in node.decorator_list:
            self.visit(d)
        self._args(node.args)
        self._ann(node.returns)
        self.scopes.append(("func", self.argnames(node.args) | self.stored(node.body)))
        for s in node.body:
            self.visit(s)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._args(node.args)
        self.scopes.append(("func", self.argnames(node.args)))
        self.visit(node.body)
        self.scopes.pop()

    def _comp(self, node, elts):
        targets = {x.id for g in node.generators for x in ast.walk(g.target) if isinstance(x, ast.Name)}
        self.visit(node.generators[0].iter)
        self.scopes.append(("func", targets))
        for g in node.generators:
            self.visit(g.target)
            if g is not node.generators[0]:
                self.visit(g.iter)
            for c in g.ifs:
                self.visit(c)
        for e in elts:
            self.visit(e)
        self.scopes.pop()

    def visit_ListComp(self, node):
        self._comp(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        self._comp(node, [node.key, node.value])

    def visit_ClassDef(self, node):
        for d in node.decorator_list:
            self.visit(d)
        for b in node.bases:
            self.visit(b)
        for k in node.keywords:
            self.visit(k.value)
        self.scopes.append(("class", self.stored(node.body)))
        for s in node.body:
            if isinstance(s, ast.AnnAssign):
                self._ann(s.annotation)
                if s.value is not None:
                    self.visit(s.value)
                self.visit(s.target)
            else:
                self.visit(s)
        self.scopes.pop()


class Namer:
    def __init__(self):
        self.vendored, self.img_syms, self.img_mods, self.ext, self.sentinels = {}, {}, {}, {}, {}

    def taken(self, new, key):
        return any(v == new and k != key for t in (self.vendored, self.img_syms) for k, v in t.items()) or new in self.ext

    @staticmethod
    def validate(key):
        check(deep_identical(key), f"{key[0]}.{key[1]} differs from the image (vendor it, or list it in EQUIV / DEAD)")
        check(mod("img", key[0]).src is not None and key[1] in mod("img", key[0]).defs, f"{key} missing in the image")

    def internal_sym(self, key):
        key = resolve("pr", *key)
        if key in self.vendored:
            return self.vendored[key]
        if key in SENTINEL:
            self.sentinels[key] = P + key[1]
            return P + key[1]
        if key not in self.img_syms:
            self.validate(key)
            check(not self.taken(P + key[1], key), f"name collision {P + key[1]}")
            self.img_syms[key] = P + key[1]
        return self.img_syms[key]

    def module_alias(self, sub):
        return self.img_mods.setdefault(sub, P + "m_" + sub)

    def external(self, local, module, attr):
        if local in KEEP:
            if attr is None:
                stmt = f"import {module}" + ("" if module == local else f" as {local}")
            else:
                stmt = f"from {module} import {attr}" + ("" if attr == local else f" as {local}")
            check(self.ext.setdefault(local, stmt) == stmt, f"KEEP name {local} bound two ways")
            return local
        new = P + local
        stmt = f"import {module} as {new}" if attr is None else f"from {module} import {attr} as {new}"
        check(self.ext.setdefault(new, stmt) == stmt, f"external {new} bound two ways: {self.ext[new]} vs {stmt}")
        check(new not in self.vendored.values() and new not in self.img_syms.values(), f"collision {new}")
        return new


class Apply(ast.NodeTransformer):
    def __init__(self, repl):
        self.repl = repl

    def visit_Attribute(self, node):
        if id(node) in self.repl:
            return self.repl[id(node)]
        self.generic_visit(node)
        return node


def rename(node, m, namer):
    sc = Scoper(m)
    sc.visit(node)
    node.name = namer.vendored[(m.name, node.name)]
    repl = {}
    for kind, n in sc.hits:
        if kind == "name":
            t = m.internal(n.id)
            if t is None:
                continue
            if t[0] == "sym":
                n.id = namer.internal_sym((t[1], t[2]))
            elif t[0] == "mod":
                n.id = namer.module_alias(t[1])
            else:
                n.id = namer.external(n.id, t[1], t[2])
        else:
            t = m.internal(n.value.id)
            key = resolve("pr", t[1], n.attr)
            if key in namer.vendored or key in SENTINEL:
                repl[id(n)] = ast.Name(namer.internal_sym(key), ast.Load())
            else:
                namer.validate(key)
                n.value = ast.Name(namer.module_alias(t[1]), ast.Load())
    vend = {k[1]: v for k, v in namer.vendored.items()}
    for s in sc.ann_strings:
        s.value = re.sub(r"\b(" + "|".join(map(re.escape, vend)) + r")\b", lambda mm: vend[mm.group(1)], s.value)
    node = Apply(repl).visit(node)
    return ast.fix_missing_locations(node)


def header():
    return f'''# ===================================================================================================================
# PR206 vendored (team joeyhlu): flash-attention PR #206, SM90 native-FP8 forward, pinned commit {SHA[:12]} (flash_attn/cute:
# flash_fwd_sm90.py, flash_fwd.py, softmax.py, tile_scheduler.py, mask.py, block_info.py, seqlen_info.py, paged_kv.py,
# pack_gqa.py, utils.py). Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao;
# BSD-3-Clause (the FA4 license notice above applies). GENERATED by tools/gen_fa206_vendor.py: AST slices (method bodies
# unedited, docstrings / comments dropped) of the symbols this layer's one configuration executes, names prefixed p206_;
# other flash_attn.cute names come from the image's installed copy (AST-identical to the PR's, see the generator).
# ==================================================================================================================='''


def generate():
    namer, report = Namer(), []
    for mname, sym, _ in SPEC:
        check(resolve("pr", mname, sym) == (mname, sym), f"{mname}.{sym} is a re-export")
        namer.vendored[(mname, sym)] = P + sym
    bodies = []
    for mname, sym, mode in SPEC:
        m = mod("pr", mname)
        node = rename(slice_symbol(m, sym, mode, report), m, namer)
        body = ast.unparse(node)
        check(dump(ast.parse(body).body[0]) == dump(node), f"{sym}: unparse round trip changed the AST")
        for csym, old, new in COSMETIC:
            if csym == sym:
                check(body.count(old) == 1, f"cosmetic edit {old!r} hits {body.count(old)} times in {sym}")
                body = body.replace(old, new)
        bodies.append(body)
    imports = sorted(set(namer.ext.values()))
    imports += [f"from {IMG_PKG} import {sub} as {alias}" for sub, alias in sorted(namer.img_mods.items())]
    imports += [f"from {IMG_PKG}.{k[0]} import {k[1]}" + ("" if v == k[1] else f" as {v}") for k, v in sorted(namer.img_syms.items())]
    sentinels = [f"class {v}:\n    pass" for k, v in sorted(namer.sentinels.items())]
    text = header() + "\n" + "\n".join(imports) + "\n\n\n" + "\n\n\n".join(sentinels + bodies) + "\n"
    return text, namer, report


def verify(text, namer):
    tree = ast.parse(text)
    bound = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.Import):
            bound |= {a.asname or a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            bound |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Assign):
            bound |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    bad = sorted(b for b in bound if not (b.startswith(P) or b in KEEP))
    check(not bad, f"unprefixed top-level names {bad}")
    missing = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            sc = Scoper(None)
            sc.visit(n)
            missing |= {x.id for kind, x in sc.hits if kind == "name" and x.id not in bound and not hasattr(builtins, x.id)}
    check(not missing, f"unbound names in the output: {sorted(missing)}")
    check(text.isascii(), "non-ASCII")
    for pat in (r"\bimportlib\b", r"sys\.modules", r"\bsubprocess\b", r"\bctypes\b", r"__trap", r"os\.environ", r"getenv",
                r"\bexec\(", r"\beval\(", r"\binspect\b", r"\bopen\(", r"__import__"):
        check(not re.search(pat, text), f"forbidden pattern {pat}")
    return [f"{len(text.encode())} bytes, {len(tree.body)} top-level statements",
            "vendored: " + ", ".join(k[0] + "." + k[1] for k in namer.vendored),
            "from the image: " + ", ".join(k[0] + "." + k[1] for k in namer.img_syms) + f"; modules {sorted(namer.img_mods)}"]


def main(argv):
    text, namer, report = generate()
    for r in verify(text, namer):
        print("[gen_fa206_vendor] OK ", r)
    for mname, sym, dropped in report:
        print(f"[gen_fa206_vendor] {mname}.{sym}: dropped (never called here) {dropped}")
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text(encoding="utf-8") == text, f"{OUT.name} differs from the generator output")
        print("[gen_fa206_vendor] --check: up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"[gen_fa206_vendor] wrote {OUT.relative_to(ROOT)} ({len(text.encode())} bytes, sha1 "
          f"{hashlib.sha1(text.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

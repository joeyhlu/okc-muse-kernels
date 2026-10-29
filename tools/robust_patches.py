"""robust_patches.py: robustness riders on the decode_layer ship (notes/study/robustness_gc21cg.md; 0-perf on the scored shapes).

  python tools/robust_patches.py [--r1] [--r2] [--n=1] [--base=k3_gc_21cg_s1] [--classes=sliding1,sliding8,full8] [--dry]
        -> kernels/lab/k3_rob_<n>.py
  python tools/robust_patches.py --test [--file=kernels/lab/k3_rob_1.py]   (local, no GPU: R1 forced-exception proof)

R1: `_k3_disable` was process-global (one call-time exception on a nearby shape turned that route off for every later shape, scored
    ones included). Now the disable is scoped to (state key, shape key = config values + k_cache shape) of the kernel_fn call that
    failed (found by walking the exception's traceback frames up to kernel_fn; exception path only). kernel_fn re-applies the
    per-shape disables at entry only once any exists (`if _K3_DIS_ON:`; one global load on the success path). A disable raised
    outside kernel_fn (prepare-time) stays process-wide, as before.
R2: page_size != 1 had no KS / FA4 route (sliding decode -> K1, sliding verify -> FA3, full verify B=1 -> FA3). At the top of
    kernel_fn (inside the timed call), for the (kind, q_len) classes in R2_CLASSES and kv_scale 1, one Triton kernel writes a
    token-level page table pt1[b, pos] = page_table[b, pos // ps] * ps + pos % ps (full: every pos < cache_seqlens[b]; sliding:
    the last window + q_len + 512 positions, 256-aligned) into a zero-initialised scratch buffer keyed by (B, cols, rows, device),
    and the KS1 / FA4 routes get the page-1 view (k/v cache viewed [-1, 1, Hkv, 64], same memory; cache_loc is already token-
    level). Built only when KS1 or FA4 is wanted on the view; anything else (and any exception) uses the original data.
    R2_PAGE1 = False is exactly the ship. page_size == 1: identical (adata is data).
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)


def check(c, msg):
    if not c:
        raise SystemExit(f"[robust_patches] FAIL: {msg}")


def sub(t, old, new, count=1, what=""):
    n = t.count(old)
    check(n == count, f"anchor {what or old[:70]!r}: found {n}, expected {count}")
    return t.replace(old, new)


OLD_DISABLE = '''def _k3_disable(state, key, value, piece, ex):
    state[key] = value
    if piece not in _K3_WARNED:
        _K3_WARNED.add(piece)
        print(f"[k3] {piece} raised at call time, r1 path from now on: {ex!r}", flush=True)
'''
NEW_DISABLE = '''_K3_DIS_ON = False   # R1 (tools/robust_patches.py): True once a call-time disable is scoped to one shape


def _k3_shape_key(data):
    try:
        return tuple(data["config"].values()) + tuple(data["k_cache"].shape)
    except Exception:
        return None


def _k3_disable(state, key, value, piece, ex):
    # R1: scope the disable to the shape of the kernel_fn call that raised; outside kernel_fn (prepare) it stays process-wide.
    global _K3_DIS_ON
    f = ex.__traceback__.tb_frame if getattr(ex, "__traceback__", None) is not None else None
    while f is not None and f.f_code.co_name != "kernel_fn":
        f = f.f_back
    sk = _k3_shape_key(f.f_locals.get("data")) if f is not None else None
    orig, dis = state.setdefault("_k3_orig", {}), state.setdefault("_k3_dis", {})
    if sk is None:
        if key in orig:
            orig[key] = value
    else:
        if key not in orig:
            orig[key] = state.get(key)
        dis.setdefault(sk, {})[key] = value
        _K3_DIS_ON = True
    state[key] = value
    w = piece if sk is None else (piece, sk)
    if w not in _K3_WARNED:
        _K3_WARNED.add(w)
        print(f"[k3] {piece} raised at call time, r1 path from now on {'(all shapes)' if sk is None else 'for this shape'}: "
              f"{ex!r}", flush=True)


def _k3_dis_apply(data, state):
    dis = state.get("_k3_dis", {}).get(_k3_shape_key(data), {})
    for k, v in state.get("_k3_orig", {}).items():
        state[k] = dis[k] if k in dis else v
'''

KF_HEAD = '''def kernel_fn(data, state):
    cfg = data["config"]
    tokens = cfg["tokens"]
'''
KF_HEAD_R1 = '''def kernel_fn(data, state):
    if _K3_DIS_ON:
        _k3_dis_apply(data, state)
    cfg = data["config"]
    tokens = cfg["tokens"]
'''
R2_LINE = '''    adata = _r2_try(data, cfg, state) if (R2_PAGE1 and cfg["page_size"] != 1) else data
'''

R2_CODE = '''R2_PAGE1 = True      # R2 (tools/robust_patches.py): page_size > 1 -> token-level page-1 view for the KS1 / FA4 routes
R2_CLASSES = {CLASSES}
R2_MARGIN = 512
_R2_BUF = {{}}


@triton.jit
def _r2_pt1_kernel(pt_ptr, len_ptr, out_ptr, pt_cols, out_cols, keep, SHIFT: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    n = tl.load(len_ptr + b)
    lo = tl.maximum(n - keep, 0) // 256 * 256
    pos = lo + tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = pos < n
    pg = tl.load(pt_ptr + b * pt_cols + (pos >> SHIFT), mask=m, other=0)
    tl.store(out_ptr + b * out_cols + pos, (pg << SHIFT) + (pos & ((1 << SHIFT) - 1)), mask=m)


def _r2_page1(data, cfg, state):
    kind = "sliding" if cfg["window"] else "full"
    ps = cfg["page_size"]
    pt, lens, kc, vc = data["page_table"], data["cache_seqlens"], data["k_cache"], data["v_cache"]
    if ((kind, cfg["q_len"]) not in R2_CLASSES or ps <= 1 or ps & (ps - 1) or cfg["kv_scale"] != 1.0
            or pt.dtype != torch.int32 or pt.dim() != 2 or not pt.is_contiguous() or lens.dtype != torch.int32
            or not lens.is_contiguous() or kc.dim() != 4 or kc.shape[1] != ps or not kc.is_contiguous() or not vc.is_contiguous()):
        return data
    cfg1 = dict(cfg, page_size=1)
    view = dict(data, config=cfg1, k_cache=kc.view(-1, 1, kc.shape[2], kc.shape[3]),
                v_cache=vc.view(-1, 1, vc.shape[2], vc.shape[3]), scheduler_metadata=None)
    x = data["attn_input"]   # the wanted() checks only look at qkv's dtype / dim / stride / rows, which x shares
    if kind == "full" and _vfa_wanted(x, data, cfg, state):   # VFA (page-aware) comes before FA4 in the chain
        return data
    if not (_ks1_wanted(x, view, cfg1, state) or (not _va1_b1(cfg1, state) and _v18_fa4_wanted(x, view, cfg1, state))):
        return data
    B, cols = pt.shape[0], pt.shape[1] * ps
    key = (B, cols, kc.shape[0] * ps, str(pt.device))
    buf = _R2_BUF.get(key)
    if buf is None:   # rows outside the written range keep 0 (or a same-key earlier row): always inside this cache
        buf = _R2_BUF[key] = torch.zeros((B, cols), dtype=torch.int32, device=pt.device)
    keep = cfg["window"] + cfg["q_len"] + R2_MARGIN if kind == "sliding" else (1 << 30)
    _r2_pt1_kernel[(B, triton.cdiv(min(cols, keep + 256), 1024))](pt, lens, buf, pt.shape[1], cols, keep,
                                                                   SHIFT=ps.bit_length() - 1, BLOCK=1024, num_warps=4)
    view["page_table"] = buf
    return view


def _r2_try(data, cfg, state):
    if state.get("r2_off"):
        return data
    try:
        return _r2_page1(data, cfg, state)
    except Exception as ex:
        _k3_disable(state, "r2_off", True, "R2", ex)
        return data


'''

ATT_OLD = '''    if _ks1_wanted(qkv, data, cfg, state):
        try:
            o = _ks1_run(qkv, data, state).view(tokens, hq * d)'''
ATT_NEW = '''    if _ks1_wanted(qkv, adata, adata["config"], state):
        try:
            o = _ks1_run(qkv, adata, state).view(tokens, hq * d)'''
FA4_OLD = '''    if o is None and not _va1_b1(cfg, state) and _v18_fa4_wanted(qkv, data, cfg, state):
        try:
            o = _v18_fa4_core(qkv, data).view(tokens, hq * d)'''
FA4_NEW = '''    if o is None and not _va1_b1(cfg, state) and _v18_fa4_wanted(qkv, adata, adata["config"], state):
        try:
            o = _v18_fa4_core(qkv, adata).view(tokens, hq * d)'''


def r1(t):
    t = sub(t, OLD_DISABLE, NEW_DISABLE, what="_k3_disable")
    return sub(t, KF_HEAD, KF_HEAD_R1, what="kernel_fn head")


def r2(t, classes):
    cl = "{" + ", ".join(f'("{k}", {q})' for k, q in classes) + "}"
    head = KF_HEAD_R1 if KF_HEAD_R1 in t else KF_HEAD
    t = sub(t, head, R2_CODE.format(CLASSES=cl) + head + R2_LINE, what="kernel_fn head (R2)")
    t = sub(t, ATT_OLD, ATT_NEW, what="KS1 site")
    return sub(t, FA4_OLD, FA4_NEW, what="FA4 site")


def parse_classes(s):
    out = []
    for c in s.split(","):
        m = re.fullmatch(r"(sliding|full)(\d+)", c.strip())
        check(m, f"bad class {c!r}")
        out.append((m.group(1), int(m.group(2))))
    return out


def build():
    check("r1" in FLAGS or "r2" in FLAGS, "need --r1 and/or --r2")
    n = ARGS.get("n", "1")
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_21cg_s1')}.py"
    out = ROOT / ARGS.get("out", f"kernels/lab/k3_rob_{n}.py")
    b = base.read_text(encoding="utf-8")
    t, what = b, []
    if "r1" in FLAGS:
        t = r1(t)
        what.append("R1 shape-scoped _k3_disable")
    if "r2" in FLAGS:
        cls = parse_classes(ARGS.get("classes", "sliding1,sliding8,full8"))
        t = r2(t, cls)
        what.append("R2 page-1 view at page>1 for " + "/".join(f"{k}{q}" for k, q in cls))
    bn = t.split(NL)[1]
    nonce = f"# nonce {out.stem.replace('_', '-')} ({base.stem} + {', '.join(what)}; tools/robust_patches.py)"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    check(max(len(x) for x in t.split(NL)) <= max(len(x) for x in b.split(NL)), "line longer than the base's longest")
    check(re.search(r"importlib|sys\.modules", t) is None or re.search(r"importlib|sys\.modules", b) is not None, "checker")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[robust_patches] OK {len(t.encode())} B (base {len(b.encode())} B, +{len(t.encode()) - len(b.encode())}); {'; '.join(what)}")
    if "dry" in FLAGS:
        return 0
    check(not out.exists() or "force" in FLAGS, f"{out.name} exists (never overwrite; --force)")
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[robust_patches] wrote {out.relative_to(ROOT).as_posix()} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


# ---------------------------------------------------------------- local R1 proof (pure Python, no torch / GPU)
MOCK = '''
RAISES = []


class _KC:
    def __init__(self, n):
        self.shape = (n, 1, 2, 64)


def _ks1_run(data, state):        # the KS1 route: raises on the "nearby" shape only
    if data["config"]["tokens"] == 24:
        RAISES.append(data["config"]["tokens"])
        raise RuntimeError("forced nearby-shape failure")
    return "KS1"


def _gd(state, data):             # a route whose try / except sits one frame below kernel_fn (like _gd / _dg1 / _v8_experts)
    if state.get("gd") is None:
        return None
    try:
        if data["config"]["tokens"] == 24:
            RAISES.append("gd")
            raise RuntimeError("forced gd failure")
        return "GD"
    except Exception as ex:
        _k3_disable(state, "gd", None, "GD", ex)
        return None


def kernel_fn(data, state):
{HEAD}
    g = _gd(state, data) or "cuBLAS"
    if state["ks1"]:
        try:
            return _ks1_run(data, state), g
        except Exception as ex:
            _k3_disable(state, "ks1", set(), "KS1", ex)
    return "FA3", g


def shape(tokens):
    return {{"config": {{"batch": tokens, "q_len": 1, "tokens": tokens, "page_size": 1, "window": 2048}}, "k_cache": _KC(4096 * tokens)}}
'''


def _extract(src, names):
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.Assign) and any(getattr(t_, "id", None) in names for t_ in node.targets):
            keep.append(ast.get_source_segment(src, node))
    return NL.join(keep)


def _run(src, label):
    names = {"_K3_DIS_ON", "_k3_shape_key", "_k3_disable", "_k3_dis_apply"}
    tree = ast.parse(src)
    kf = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_fn")
    kf_src = ast.get_source_segment(src, kf).split(NL)
    head = [ln for ln in kf_src[1:4] if "_K3_DIS_ON" in ln or "_k3_dis_apply" in ln]   # the real R1 entry guard, if present
    ns = {"_K3_WARNED": set(), "print": lambda *a, **k: None}
    exec(_extract(src, names), ns)
    exec(MOCK.format(HEAD=NL.join(head) if head else "    pass"), ns)
    st = {"ks1": {(1, 0, 0)}, "gd": {"route": 1}}
    seq = [16, 24, 16, 24, 32, 16]   # scored, nearby (raises), scored, nearby again, another, scored
    got = [ns["kernel_fn"](ns["shape"](t), st) for t in seq]
    outside = None
    try:
        raise RuntimeError("prepare-time failure")
    except Exception as ex:   # not under kernel_fn -> process-wide, as before
        st2 = {"ks1": {(1, 0, 0)}}
        ns["_k3_disable"](st2, "ks1", set(), "KS1", ex)
        outside = st2["ks1"]
    print(f"[r1-test] {label}: " + ", ".join(f"T{t}->{a}/{g}" for t, (a, g) in zip(seq, got))
          + f"; raises {ns['RAISES']}; prepare-time disable process-wide: {outside == set()}")
    return seq, got, ns["RAISES"], outside


def test():
    f = ROOT / ARGS.get("file", "kernels/lab/k3_rob_1.py")
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_21cg_s1')}.py"
    _run(base.read_text(encoding="utf-8"), "ship " + base.stem)
    seq, got, raises, outside = _run(f.read_text(encoding="utf-8"), "R1   " + f.stem)
    want = [("KS1", "GD"), ("FA3", "cuBLAS"), ("KS1", "GD"), ("FA3", "cuBLAS"), ("KS1", "GD"), ("KS1", "GD")]
    check(got == want, f"R1 routes {got} != {want}")
    check(raises == ["gd", 24], f"the failing shape must raise once per route, then stay off for itself: {raises}")
    check(outside == set(), "prepare-time disable must stay process-wide")
    print("[r1-test] PASS: the failing shape keeps its fallback (no re-raise), every other shape keeps KS1 + GD")
    return 0


if __name__ == "__main__":
    sys.exit(test() if "test" in FLAGS else build())

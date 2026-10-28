"""at_patches.py: AT = a prepare()-time route autotuner + an AT_DIAG research mode for the decode_layer file
(notes/study/autotune.md; the idea from vjeux's working method, notes/study/vjeux_path.md s(d); our own code).

What the patch does to a layer file (dispatch layer only, no kernel code is touched):
  * renames `def prepare(weights):` -> `_at_prepare0` and `def kernel_fn(data, state):` -> `_at_kernel_fn0` and appends
    new `prepare` / `kernel_fn` wrappers plus the AT section at the end of the file;
  * prepare() = the original prepare, then _at_tune: every registered variant (a set of module-global overrides, e.g.
    {"VMS_ENABLE": False}) whose `when` holds is timed as a whole-layer CUDA graph at the scored shapes of its token buckets
    on synthetic inputs (own random cache / stream; the stream's routing load is calibrated to the published per-load
    router targets), 256 MB L2 read before every replay, paired rotated rounds over 3 synthetic routings; a variant replaces
    the shipped route of its (op, bucket) only if it wins by > AT_MARGIN (weighted over the bucket's sliding / full shapes)
    and its output agrees with the shipped one; then every changed bucket is confirmed on fresh inputs (combination vs ship);
  * kernel_fn() looks up (MoE bucket, layer kind) -> the decided overrides, swaps them in for the call and restores them
    (host-side Python; under graph capture this picks the captured route). Nothing derived from inputs is kept.
  * AT_DIAG = True (research copies only): also times the research variants (ship=False), prints a per-shape table, the
    routing calibration and per-CTA timelines of kernels with marks (VMS, VML). AT_FORCE = {(op, bucket): name} forces
    decisions at call time (box A/B arms).

Registry API (other builders: one call per variant, then apply() on a STRIPPED base that is a finish() fixed point):

    import at_patches as AT
    AT.variant("moe", "vml-f41", (32,), {"VML_FLAGS": "{**VML_FLAGS, 32: 0x41}"}, when='_VML["ok"].get(32)', ship=False)
    text = AT.apply(stripped_base_text, diag=False)

  op       mutually exclusive group: variants of one op compete per bucket; different ops combine (dense: one op per proj)
  name     unique within (op, bucket)
  buckets  MoE token buckets (_k2a_bucket_of: 1, 8, 16, 32, 128, 256) the variant may route
  set      {MODULE_GLOBAL: python expression}, evaluated once after prepare (may read the current globals, `S` = state);
           the global must be a module-level assignment in the base; a name shared by several ops must be a dict
  when     python expression (S = state, after prepare): the variant is valid (its route is built and self-tested)
  kinds    ("sliding", "full") default, or one of them (attention variants)
  ship     False = research only (timed when AT_DIAG, never chosen)

    python -I tools/at_patches.py [--base=k3_va_3] [--n=1] [--extra=vml_at,va_at] [--lab-only]
        -> kernels/muse_decode_layer/k3_at_<n>.py (AT_DIAG False), k3_at_<n>r.py (research copy, AT_DIAG True)
           + byte-identical box copies kernels/lab/atn_k3_at_<n>_ab.py, kernels/lab/atn_k3_at_<n>r_ab.py
    --extra: builder modules tools/<name>.py defining `def register(AT): AT.variant(...)`, called after the built-ins.
"""
import ast
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

REGISTRY = []      # dicts: op, name, buckets, set, when, kinds, ship, note
PREPARE_SET = []   # dicts: name, expr, diag_only, note
TIMELINES = []     # dicts: name, buckets, fn (function name in the emitted code), src
KINDS = ("sliding", "full")
BUCKETS = (1, 8, 16, 32, 128, 256)
DENSE_SHIP = "--dense=research" not in sys.argv   # GD-vs-fallback units: ship (default) or research-only (saves ~0.9 KB)
# Output-agreement tolerance vs the shipped route (rel-L2 of the next attention input and of the residual update). Attention
# routes differ in their own error vs FP32 (FA4 / stock ~8-9 %, KS1 / KS3 / PR206 ~1-2 %; R-at-2), so an attention op gets
# the judge's upper bound 0.15 (garbage / NaN still fails); everything else 3e-2 (AT_TOL in the layer).
ATTN_TOL = 0.15


def variant(op, name, buckets, overrides, when="True", kinds=None, ship=True, note="", tol=None, margin=None):
    """Register one candidate route (see the module docstring); `overrides` = {MODULE_GLOBAL: python expression}."""
    kinds = tuple(kinds or KINDS)
    buckets = tuple(int(b) for b in buckets)
    if tol is None and op.startswith("attn"):
        tol = ATTN_TOL
    if not re.fullmatch(r"[a-z][a-z0-9_]*", op) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", name) or name == "ship":
        raise ValueError(f"bad op / name {op!r} {name!r}")
    if not buckets or any(b not in BUCKETS for b in buckets):
        raise ValueError(f"{op}/{name}: buckets {buckets} not in {BUCKETS}")
    if any(k not in KINDS for k in kinds):
        raise ValueError(f"{op}/{name}: kinds {kinds}")
    for v in REGISTRY:
        if v["op"] == op and v["name"] == name and set_overlap(v["buckets"], buckets):
            raise ValueError(f"{op}/{name} registered twice for buckets {buckets}")
        if v["op"] == op and v["kinds"] != kinds:
            raise ValueError(f"op {op}: every variant must share kinds ({v['kinds']} vs {kinds})")
    for k, e in overrides.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", k):
            raise ValueError(f"{op}/{name}: override target {k!r} is not a MODULE_GLOBAL name")
        ast.parse(e, mode="eval")
    ast.parse(when, mode="eval")
    REGISTRY.append(dict(op=op, name=name, buckets=buckets, set=dict(overrides), when=when, kinds=kinds, ship=bool(ship),
                         note=note, tol=None if tol is None else float(tol), margin=None if margin is None else float(margin)))


FAMILIES = []      # dicts: vars, rows, op, name, buckets, set, when, kinds, ship, tol (emitted as one loop)


def family(names, rows, op, name, buckets, overrides, when, kinds=None, ship=True, tol=None, note="", margin=None):
    """Many variants of one shape, emitted as ONE loop in the layer (bytes): `names` = loop variable names, `rows` = their
    literal values; `op`, `name`, `buckets` are python expressions over the loop variables (evaluated here, so each member
    is validated like a `variant`), `overrides` values and `when` are expressions emitted inside the loop."""
    fid = len(FAMILIES)
    FAMILIES.append(dict(vars=tuple(names), rows=[tuple(r) for r in rows], op=op, name=name, buckets=buckets,
                         set=dict(overrides), when=when, kinds=kinds, ship=bool(ship), tol=tol))
    for row in rows:
        env = dict(zip(names, row))
        variant(eval(op, {}, env), eval(name, {}, env), eval(buckets, {}, env), dict(overrides), when=when, kinds=kinds,
                ship=ship, note=note, tol=tol, margin=margin)
        REGISTRY[-1]["fam"] = fid


def set_overlap(a, b):
    return bool(frozenset(a) & frozenset(b))


def prepare_set(name, expr, diag_only=True, note="", unless=None):
    """A module-global override active only while the ORIGINAL prepare runs (e.g. self-test an extra bucket so a research
    variant becomes valid); restored before the autotune and the timed calls."""
    ast.parse(expr, mode="eval")
    PREPARE_SET.append(dict(name=name, expr=expr, diag_only=bool(diag_only), note=note, unless=unless))


def timeline(name, buckets, fn, src):
    """AT_DIAG only: `src` defines `fn(S, x, lid, tw)` -> None or (marks [G, n] int64 globaltimer ns, labels, slots); the
    autotuner runs it once per load of `buckets` on synthetic MoE inputs and prints med / max per mark over CTAs."""
    ast.parse(src.replace("@VMLFC@", ""))
    TIMELINES.append(dict(name=name, buckets=tuple(buckets), fn=fn, src=src))


# ---------------------------------------------------------------------------------------------------------------- built-ins
TL_VMS = '''
def _at_tl_vms(S, x, lid, tw):
    k = _at_k(S)
    vs = k.get("vms")
    if not (VMS_ENABLE and k.get("vms_ok") and vs is not None and x.shape[0] <= VMS_MAXT):
        return None
    G = int(_VMS["G"])
    mk = torch.zeros((G * 16,), dtype=torch.int64, device=x.device)
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    vs["mod"].run(x, lid, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"],
                  vs["ctr"], out, mk, m, K2A_POST_NORM_EPS, 1, 0)
    torch.cuda.synchronize()
    return (mk.view(G, 16), ("start", "routed", "first", "wB", "wB1", "cdone", "hA", "hdone", "bar", "end"),
            tuple(range(10)))
'''

TL_VML = '''
def _at_tl_vml(S, x, lid, tw):
    rep = _k2a_bucket_of(x.shape[0])
    if not (VML_ENABLE and _VML["ok"].get(rep)):
        return None
    G = int(_VML["sc"]["G"])
    mk = torch.zeros((G * 32,), dtype=torch.int64, device=x.device)
    _vml_call(_VML["mod"], _at_k(S), _VML["sc"], x, lid, tw, 0, marks=mk@VMLFC@)
    torch.cuda.synchronize()
    return mk.view(G, 32), ("start", "routed", "gathered", "Adone", "B0", "B1"), (0, 7, 5, 6, 1, 2)
'''


def builtin():
    """The default variant set (k3_va_3 routes). Every `when` names the self-test flag of the alternative route."""
    vml32 = 'VML_ENABLE and VML_BUCKETS.get(32) and _VML["ok"].get(32)'
    variant("moe", "k2a", (1, 8, 16), {"VMS_ENABLE": "False"},
            when='VMS_ENABLE and _at_k(S).get("vms_ok") and all((S.get("k2a33") or {}).get("ok", {}).get(b) for b in (1, 8, 16))',
            note="VMS one-launch (T <= 16) vs K2A v3.3 vid 0 (T <= 8) / v3.6 vid 3 (T 9-16)")
    variant("moe", "vid3", (32,), {"M32_PX": "False", "VML_BUCKETS": "{**VML_BUCKETS, 32: False}"},
            when='M32_PX and _at_k(S).get("px_ok") and 32 in (_at_k(S).get("r36") or {})',
            note="PX (finalize-free W2 queue) vs K2A v3.6 vid 3 at T = 25-48")
    variant("moe", "px", (32,), {"VML_BUCKETS": "{**VML_BUCKETS, 32: False}"},
            when=vml32 + ' and M32_PX and _at_k(S).get("px_ok")', note="VML at 32 shipped -> PX alternative")
    variant("moe", "k2a", (32,), {"VML_BUCKETS": "{**VML_BUCKETS, 32: False}"},
            when=vml32 + ' and not (M32_PX and _at_k(S).get("px_ok")) and 32 in (_at_k(S).get("r36") or {})',
            note="VML shipped at 32 without PX (slim bases) -> K2A v3.6 vid 3")
    variant("moe", "vml", (32,), {"VML_BUCKETS": "{**VML_BUCKETS, 32: True}"},
            when='VML_ENABLE and not VML_BUCKETS.get(32) and _VML["ok"].get(32)', ship=False,
            note="VML1 at T = 32 (research: needs the AT_DIAG prepare-time self-test of bucket 32)")
    prepare_set("VML_BUCKETS", "{**VML_BUCKETS, 32: True}", diag_only=True, note="self-test VML at bucket 32 (research)")
    for b in (128, 256):
        variant("moe", "k2a", (b,), {"VML_BUCKETS": "{**VML_BUCKETS, %d: False}" % b},
                when='VML_ENABLE and VML_BUCKETS.get(%d) and _VML["ok"].get(%d)' % (b, b),
                note="VML1 vs K2A (v3.7 at 128, SK1 / vid 2 at 256)")
    va1 = 'S.get("va1_ok") and VA1_ENABLE'
    for st in (3, 4):
        variant("attn_full", f"va1-st{st}", (16, 32), {"VA1_STAGES": str(st)}, kinds=("full",),
                when=va1 + f" and VA1_STAGES != {st}", note="VA1 ring stages (occupancy prepared for 3 and 4)")
    variant("attn_full", "k1", (16, 32), {"VA1_ENABLE": "False"}, kinds=("full",), when=va1, note="VA1 vs K1 (decode full)")
    for sk in (256, 640):
        variant("attn_full", f"va1-seg{sk}", (16, 32), {"VA1_SEGK": str(sk)}, kinds=("full",), when=va1, ship=False,
                note="VA1 segment restart cost SEGK")
    variant("attn_full", "va1-nopdl", (16, 32), {"VA1_PDL": "False"}, kinds=("full",), when=va1 + " and VA1_PDL", ship=False)
    variant("attn_vfull", "fa4v8", (128, 256), {"P206_ENABLE": "False"}, kinds=("full",),
            when='P206_ENABLE and P206_STATE["ready"] and S.get("fa4v6")', note="verify full B >= 16: PR206 vs FA4 v8")
    variant("attn_slide", "fa4", (1, 8), {"KS1_ROUTE": "{k: False for k in KS1_ROUTE}"}, kinds=("sliding",),
            when='KS1_ENABLE and S.get("ks1") and S.get("fa4v6")', ship=False, note="B = 1 sliding: KS1 vs FA4 v6")
    variant("attn_slide", "ks2", (16,), {"KS3_ROUTE": "{k: False for k in KS3_ROUTE}"}, kinds=("sliding",),
            when='KS3_ENABLE and S.get("ks3") and KS2_ROUTE.get((16, 1)) and S.get("ks2")', ship=False,
            note="decode 16 sliding: KS3 vs KS2")
    variant("attn_slide", "fa4", (16, 32), {"KS3_ROUTE": "{k: False for k in KS3_ROUTE}", "KS2_ROUTE": "{k: False for k in KS2_ROUTE}"},
            kinds=("sliding",), when='KS3_ENABLE and S.get("ks3") and S.get("fa4v6")', ship=False,
            note="decode 16 / 32 sliding: KS3 vs FA4 v6")
    for b in (128, 256):   # research rows for the VML builder (dropped on bases without VML_FLAGS / VML_COST)
        vml = 'VML_ENABLE and VML_BUCKETS.get(%d) and _VML["ok"].get(%d)' % (b, b)
        variant("moe", "vml-nob6", (b,), {"VML_FLAGS": "{**VML_FLAGS, %d: VML_FLAGS.get(%d, 1) & ~64}" % (b, b)},
                when=vml + " and VML_FLAGS.get(%d, 0) & 64" % b, ship=False, note="VML flags bit 6 off (self-fill / gather)")
        variant("moe", "vml-cost0", (b,), {"VML_COST": "{k: v for k, v in VML_COST.items() if k != %d}" % b},
                when=vml + " and %d in VML_COST" % b, ship=False, note="VML default cost table instead of the refit")
    variant("attn_slide", "vsl-off", (1, 8, 16), {"VSL_ENABLE": "False"}, kinds=("sliding",),
            when="VSL_ENABLE", ship=False, note="VSL red.add merge variants off (KS1 / KS3 base geometries)")
    rows = [(p, b) for p, bs in (("qkv", (1, 8, 16, 32)), ("o", (1, 8, 16, 32)), ("pooler", (8,)), ("unpooler", (1, 8, 16, 32)))
            for b in bs]
    family(("P", "B"), rows, "P", "P + '-nogd'", "(B,)", {"GD_ROUTE": "_at_gdrop(P, B)"}, "_at_gdok(S, P, B)",
           ship=DENSE_SHIP, note="GD GEMV vs the next route (P1M / P1 / DG1 / cuBLAS)")
    # VD bases (k3_vd_*): dense at T = 128 / 256 = VD (SS-wgmma swap-AB, VD_ROUTE) vs cuBLAS vs torchT (cuBLAS on a K-major
    # weight copy, VD_TT). torchT copies are built in prepare via the ship-mode prepare set (lossless repack, ~0.2 GB), then
    # VD_TT is restored, so the shipped routes are unchanged unless AT picks torchT. The judge's proj board ranks these within
    # 1-2 us of each other (0.3-0.6 % of the layer at T = 128), hence the 0.3 % unit margin (confirm needs half of it).
    vdp = ("qkv", "o", "cat", "unpooler")
    family(("P", "B"), [(p, 128) for p in vdp], "P", "'cublas'", "(B,)",
           {"VD_ROUTE": "{k: v for k, v in VD_ROUTE.items() if k != (P, B)}", "VD_TT": "{k: v for k, v in VD_TT.items() if k != (P, B)}"},
           'VD_ENABLE and S.get("vd") is not None and (P, B) in S["vd"]["route"] and VD_ROUTE.get((P, B)) is not None '
           "and not VD_TT.get((P, B))", margin=0.003, note="VD shipped -> cuBLAS")
    family(("P", "B"), [(p, b) for p in vdp for b in (128, 256)], "P", "'torcht'", "(B,)", {"VD_TT": "{**VD_TT, (P, B): True}"},
           'VD_ENABLE and S.get("vd") is not None and P in S["vd"]["wt"] and not VD_TT.get((P, B))', margin=0.003,
           note="cuBLAS on the K-major weight copy")
    prepare_set("VD_TT", "{**VD_TT, **{(p, b): True for p in ('qkv', 'o', 'cat', 'unpooler') for b in (128, 256)}}",
                diag_only=False, note="build + self-test the torchT weight copies (restored after prepare)", unless="VD_TT_PREP")
    timeline("vms", (1, 8, 16), "_at_tl_vms", TL_VMS)
    timeline("vml", (32, 128, 256), "_at_tl_vml", TL_VML)


# ---------------------------------------------------------------------------------------------------------------- emitted code
AT_SRC = r'''

# ---------------------------------------------------------------- AT: prepare()-time route autotuner (tools/at_patches.py)
# Configuration chosen in prepare() (notes/assumptions.md A23, notes/study/autotune.md): each registered alternative route
# of an (op, token bucket) is timed as a whole-layer CUDA graph on synthetic inputs of the scored shapes (own random cache
# and stream; the stream's routing load is calibrated to the published per-load router targets), cold L2 before every
# replay, paired rotated rounds over AT_SEEDS synthetic routings. It replaces the shipped route only if it wins by more
# than AT_MARGIN and its output agrees; every changed bucket is then confirmed on fresh inputs. Only the route choice is
# kept; any error keeps the shipped routes. AT_DIAG (research copies) also times research variants and prints a table.
AT_ENABLE = True
AT_DIAG = @DIAG@
AT_FORCE = {}
AT_TIMELINES = True
AT_BUDGET_S = 24.0
AT_DIAG_BUDGET_S = 60.0
AT_SKIP_AFTER_S = 230.0
AT_MARGIN = 0.015
AT_ROUNDS = 16
AT_SEEDS = 3
AT_CONFIRM = True
AT_TOL = 3e-2
AT_CTX = 80000
AT_SPREAD = 0.075
AT_LOADS = ((1, 1), (1, 8), (16, 1), (32, 1), (16, 8), (32, 8))
AT_WEIGHT = {"sliding": 46, "full": 16}
AT_ROUTER = {1: (2.57, 2.5), 8: (14.14, 7.5), 16: (30.9, 7.53), 32: (60.9, 9.0), 128: (205.5, 18.45), 256: (409.5, 21.35)}
AT_KAPPA = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.1, 1.5)
_AT = {"call": {}, "vars": [], "forced": None, "choice": {}, "secs": 0.0}


def _at_key(cfg):
    return (_k2a_bucket_of(cfg["tokens"]), "sliding" if cfg["window"] else "full")


def _at_k(S):
    return (S.get("k2a33") or {}).get("st") or {}


#@G+
def _at_gdrop(proj, b):
    return {k: v for k, v in GD_ROUTE.items() if k != (proj, b)}


def _at_gdok(S, proj, b):
    return GD_ENABLE and S.get("gd") is not None and (proj, b) in S["gd"]["route"]
#@G-


@SWAP@


@CUR@


@PREPSET@


@VARIANTS@


#@D+
@TLDEFS@
#@D-


def kernel_fn(data, state):
    tab = _AT["call"]
    if AT_FORCE:
        f = _AT["forced"]
        if f is None or f[0] is not AT_FORCE:
            f = _AT["forced"] = (AT_FORCE, _at_table(dict(AT_FORCE), _AT["vars"]))
        tab = f[1]
    ov = tab.get(_at_key(data["config"])) if AT_ENABLE else None
    if not ov:
        return _at_kernel_fn0(data, state)
    old = _at_swap(ov)
    try:
        return _at_kernel_fn0(data, state)
    finally:
        _at_swap(old)


def prepare(weights):
    t0 = time.time()
    base = None
    if AT_ENABLE:
        try:
            base = _at_base(weights)
        except Exception as ex:
            print(f"[at] routing basis failed ({ex!r}): autotune off, shipped routes", flush=True)
    old = _at_swap(_at_prepare_set() if AT_ENABLE else {})
    try:
        state = _at_prepare0(weights)
    finally:
        _at_swap(old)
    if AT_ENABLE and base is not None:
        _at_tune(state, weights, t0, base)
    return state


def _at_table(choice, vs):
    cur, out = _at_cur(), {}
    for (op, b), name in choice.items():
        for v in vs:
            if v[0] == op and v[1] == name and b in v[2] and v[5]:
                for kind in v[3]:
                    m = out.setdefault((b, kind), {})
                    for k, x in v[4].items():
                        c = cur.get(k)
                        if k in m and isinstance(x, dict) and isinstance(m[k], dict) and isinstance(c, dict):
                            x = {**{q: y for q, y in m[k].items() if q in x or q not in c},
                                 **{q: y for q, y in x.items() if q not in c or c[q] != y}}
                        m[k] = x
    return out


def _at_med(v):
    v = sorted(v)
    return 0.5 * (v[(len(v) - 1) // 2] + v[len(v) // 2]) if v else float("nan")


def _at_rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)


def _at_base(weights):
    r = weights["router"]
    g = torch.Generator(device=r.device).manual_seed(0x4154)
    v = torch.randn((r.shape[0],), generator=g, device=r.device) @ r.float()
    d = torch.randn((r.shape[1],), generator=g, device=r.device)
    return {"g": d * torch.rsqrt(d.pow(2).mean()), "r": v * torch.rsqrt(v.pow(2).mean()),
            "S": torch.randn((4, r.shape[1]), generator=g, device=r.device)}


def _at_stream(base, T, L, kap, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    d = base["g"] + kap * base["r"]
    d = d * torch.rsqrt(d.pow(2).mean())
    sc = torch.ones((T // L, L, 1), device=dev)
    if 1 < L <= 8:
        sc[:, 1:] = 0.6
    sc = sc.reshape(T, 1)
    sh = torch.randn((T, 4), generator=g, device=dev) @ base["S"]
    z = torch.randn((T, base["S"].shape[1]), generator=g, device=dev)
    return (sc * (0.7 * d + 0.4743 * sh) + (1 - 0.9 * sc * sc).sqrt() * z).contiguous()


def _at_input(B, L, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    T, I = B * L, torch.int32
    hi = int(AT_CTX * (1 + AT_SPREAD)) + 1 if B > 1 else AT_CTX
    seq = ((AT_CTX * (1 + AT_SPREAD * (2 * torch.rand((B,), generator=g, device=dev) - 1))).long().clamp(1, hi) if B > 1
           else torch.full((B,), AT_CTX, dtype=torch.long, device=dev))
    mp = hi + L
    perm = torch.randperm(B * mp, generator=g, device=dev).to(I)
    pt = torch.full((B, mp), -1, dtype=I, device=dev)
    off = 0
    for b, n in enumerate((seq + L).tolist()):
        pt[b, :n] = perm[off:off + n]
        off += n
    pos = seq[:, None] + torch.arange(L, device=dev)
    kv = [torch.randint(0, 256, (B * mp, 1, 2, 64), generator=g, device=dev, dtype=torch.uint8).bitwise_and_(0xBF).view(FP8)
          for _ in range(2)]
    one = torch.ones((), device=dev)
    return {"stream": torch.zeros((T, HIDDEN), device=dev), "attn_input": torch.zeros((T, HIDDEN), dtype=torch.bfloat16, device=dev),
            "positions": pos.reshape(-1), "cache_loc": torch.gather(pt.long(), 1, pos).reshape(-1), "cache_seqlens": (seq + L).to(I),
            "scheduler_metadata": None, "cu_seqlens_q": torch.arange(0, T + 1, L, dtype=I, device=dev), "k_cache": kv[0],
            "v_cache": kv[1], "k_scale": one, "v_scale": one.clone(), "page_table": pt, "seq_lens": seq.to(I),
            "local_start": torch.zeros((1,), dtype=I, device=dev)}


def _at_cfg(B, L, kind):
    sl = kind == "sliding"
    return {"batch": B, "q_len": L, "tokens": B * L, "hidden": HIDDEN, "num_q_heads": 16, "num_kv_heads": 2, "head_dim": 64,
            "page_size": 1, "window": 2048 if sl else 0, "rope_theta": ROPE_THETA if sl else 0.0,
            "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125, "kv_scale": 1.0, "n_experts": N_EXPERTS,
            "top_k": TOP_K, "local_experts": LOCAL_EXPERTS}


def _at_route(d, state):
    ids = _at_kernel_fn0(d, state)[2].long()
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    ls = LOCAL_EXPERTS * int(pairs.argmax())
    loc = (ids - ls) % N_EXPERTS
    m = loc < LOCAL_EXPERTS
    return ls, int(m.sum()), int(torch.unique(loc[m]).numel())


def _at_set_stream(d, s, state):
    d["stream"].copy_(s)
    d["attn_input"].copy_(_at_rms(s).to(torch.bfloat16))
    ls, R, D = _at_route(d, state)
    d["local_start"].fill_(ls)
    torch.cuda.synchronize()
    return R, D


def _at_calibrate(d0, state, base, B, L, dev):
    T = B * L
    tR, tD = AT_ROUTER.get(_k2a_bucket_of(T), (1.0, 1.0))
    d = dict(d0, config=_at_cfg(B, L, "sliding"))
    ns, best = (4 if T <= 16 else 2), None
    for kap in AT_KAPPA:
        R = D = 0.0
        for j in range(ns):
            r, dd = _at_set_stream(d, _at_stream(base, T, L, kap, 500 + 13 * j, dev), state)
            R, D = R + r / ns, D + dd / ns
        err = abs(R / tR - 1) + 0.5 * abs(D / tD - 1)
        if best is None or err < best[0]:
            best = (err, kap, R, D)
    return best[1:]


def _at_timer(dev):
    st = torch.cuda.Stream(dev)
    tm = [torch.ones((64 << 20,), dtype=torch.float32, device=dev), torch.empty((), dtype=torch.float32, device=dev), st,
          (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)), 0.0, None]
    tiny = torch.zeros(1, device=dev)
    with torch.cuda.stream(st):
        tiny.add_(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        tiny.add_(0)
    torch.cuda.synchronize()
    tm[4] = _at_med([_at_replay(g, tm) for _ in range(30)])
    return tm


def _at_replay(g, tm, d=None):
    with torch.cuda.stream(tm[2]):
        if d is not None:
            d["stream"].copy_(tm[5][0])
            d["attn_input"].copy_(tm[5][1])
        torch.sum(tm[0], dim=0, out=tm[1])
        tm[3][0].record()
        g.replay()
        tm[3][1].record()
    tm[3][1].synchronize()
    return tm[3][0].elapsed_time(tm[3][1]) * 1e3


def _at_capture(d, state, ov, st):
    old = _at_swap(ov)
    try:
        with torch.cuda.stream(st):
            _at_kernel_fn0(d, state)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=st):
                out = _at_kernel_fn0(d, state)
        torch.cuda.synchronize()
    finally:
        _at_swap(old)
    return g, out


def _at_measure(d, state, base, kap, cands, tm, seed, nseeds, dl):
    cfg = d["config"]
    gs, ids, bad, tols = [], [], {}, []
    for v in cands:
        try:
            gs.append(_at_capture(d, state, v[4] if v else {}, tm[2]))
            ids.append(v and (v[0], v[1]))
            tols.append(v[7] if v and len(v) > 7 and v[7] is not None else AT_TOL)
        except Exception as ex:
            if v is None:
                raise
            bad[(v[0], v[1])] = f"capture {ex!r}"[:120]
            torch.cuda.synchronize()
    K = len(gs)
    ts, rs = [[] for _ in gs], [[] for _ in gs]
    for si in range(nseeds):
        _at_set_stream(d, _at_stream(base, cfg["tokens"], cfg["q_len"], kap, seed + 7 * si, d["stream"].device), state)
        tm[5] = (d["stream"].clone(), d["attn_input"].clone())
        torch.cuda.synchronize()
        for r in range(AT_ROUNDS):
            t = {}
            for i in range(K):
                j = (i + si * AT_ROUNDS + r) % K
                t[j] = max(_at_replay(gs[j][0], tm, d) - tm[4], 1e-3)
            for i in range(K):
                ts[i].append(t[i])
                rs[i].append(t[0] / t[i])
        if time.time() > dl:
            break
    outs = []
    for g, out in gs:
        _at_replay(g, tm, d)
        outs.append([t.clone() for t in out])
    o0, s0, i0 = outs[0]
    for i in range(1, K):
        o, s, ii = outs[i]
        eo = float((o.float() - o0.float()).norm() / o0.float().norm().clamp(min=1e-30))
        es = float((s - s0).norm() / (s0 - tm[5][0]).norm().clamp(min=1e-30))
        ag = float((ii.sort(-1).values == i0.sort(-1).values).float().mean())
        if not (eo < tols[i] and es < tols[i] and ag > 0.9 and bool(o.float().isfinite().all())):
            bad[ids[i]] = f"out {eo:.1e} stream {es:.1e} ids {ag:.2f}"
    return {"ids": ids, "t": [_at_med(x) for x in ts], "r": [_at_med(x) for x in rs], "n": len(ts[0]), "bad": bad,
            "w": AT_WEIGHT["sliding" if cfg["window"] else "full"]}


def _at_gain(res, shp, vid):
    num = den = 0.0
    for key in shp:
        r = res[key]
        if vid not in r["ids"] or vid in r["bad"]:
            return None
        i = r["ids"].index(vid)
        num += r["w"] * r["t"][0]
        den += r["w"] * r["t"][0] / r["r"][i]
    return num / den if den > 0 else None


#@D+
def _at_moe_in(d, state):
    ids = _at_kernel_fn0(d, state)[2].long()
    loc = (ids - int(d["local_start"][0])) % N_EXPERTS
    lid = torch.where(loc < LOCAL_EXPERTS, loc, torch.full_like(loc, -1)).to(torch.int32).contiguous()
    g = torch.Generator(device=ids.device).manual_seed(77)
    x = torch.randn((ids.shape[0], K2A_D_HIDDEN), generator=g, device=ids.device).to(torch.bfloat16)
    return x, lid, torch.full((ids.shape[0], TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=ids.device)


def _at_timelines(d, state, tm, b):
    x, lid, tw = _at_moe_in(d, state)
    for name, bks, fn in _AT_TL:
        if b not in bks:
            continue
        try:
            torch.sum(tm[0], dim=0, out=tm[1])
            r = fn(state, x, lid, tw)
            if r is None:
                continue
            mk, labels, slots = r
            mk = mk.double()
            c0 = mk[:, slots[0]]
            t0 = c0[c0 > 0].min()
            parts = []
            for lab, s in zip(labels, slots):
                c = mk[:, s]
                c = ((c[c > 0] - t0) / 1000.0).sort().values
                if c.numel():
                    parts.append(f"{lab} {float(c[c.numel() // 2]):.1f}/{float(c[-1]):.1f}")
            print(f"[at] timeline {name} T={x.shape[0]} (us med/max over {mk.shape[0]} CTAs): {' '.join(parts)}", flush=True)
        except Exception as ex:
            print(f"[at] timeline {name} T={x.shape[0]} failed: {ex!r}"[:300], flush=True)
#@D-


def _at_tune(state, weights, t0, base):
    t1 = time.time()
    _AT.update(call={}, vars=[], forced=None, choice={})
    if t1 - t0 > AT_SKIP_AFTER_S:
        print(f"[at] skipped: prepare already took {t1 - t0:.0f}s: shipped routes", flush=True)
        return
    res, cal, lines, loads = {}, {}, [], {}
    try:
        dev = weights["qkv_gate"].device
        vs = _AT["vars"] = _at_variants(state)
        act = [v for v in vs if v[5] and (v[6] or AT_DIAG)]
        bud = AT_DIAG_BUDGET_S if AT_DIAG else AT_BUDGET_S
        dl, dl2 = t1 + 0.8 * bud, t1 + bud
        tm = _at_timer(dev)
        for B, L in AT_LOADS:
            T, b = B * L, _k2a_bucket_of(B * L)
            loads[b] = (B, L)
            want = {k: [v for v in act if b in v[2] and k in v[3]] for k in ("sliding", "full")}
            if not (want["sliding"] or want["full"] or AT_DIAG):
                continue
            if time.time() > dl:
                lines.append(f"[at] budget reached before T={T}: its buckets keep the shipped routes")
                break
            data = _at_input(B, L, 1000 + T, dev)
            cal[T] = _at_calibrate(data, state, base, B, L, dev)
            for kind in ("sliding", "full"):
                if (want[kind] or AT_DIAG) and time.time() < dl:
                    d = dict(data, config=_at_cfg(B, L, kind))
                    res[(b, kind)] = _at_measure(d, state, base, cal[T][0], [None] + want[kind], tm, 2000 + T, AT_SEEDS, dl)
#@D+
                    if AT_DIAG and AT_TIMELINES and kind == "sliding":
                        _at_timelines(d, state, tm, b)
#@D-
            del data   # no empty_cache(): freed blocks stay cached and are reused, as in the judge (R-at-7: the fault followed a fresh mapping)
        units, choice, rows = {}, {}, []
        for v in act:
            for b in (v[2] if v[6] else ()):
                units.setdefault((v[0], b), []).append(v)
        for (op, b), cand in sorted(units.items(), key=str):
            shp = [(b, k) for k in cand[0][3] if (b, k) in res]
            if shp:
                gs = [(v[1], _at_gain(res, shp, (v[0], v[1]))) for v in cand]
                mg = {v[1]: AT_MARGIN if v[8] is None else v[8] for v in cand}
                best = max(((g, n) for n, g in gs if g is not None and g > 1 + mg[n]), default=(0.0, None))
                if best[1] is not None:
                    choice[(op, b)] = best[1]
                bad = lambda n: "; ".join(res[k]["bad"][(op, n)] for k in shp if (op, n) in res[k]["bad"])
                rows.append((op, b, f"[at] {op}/{b} ({'+'.join(k[1][0] for k in shp)}): ship " + "/".join(
                    f"{res[k]['t'][0]:.1f}" for k in shp) + " us; " + ", ".join(
                    f"{n} " + (f"x{g:.3f}" if g is not None else f"BAD({bad(n)})") for n, g in gs) + " -> "))
        if AT_CONFIRM and choice and time.time() < dl2:
            tab = _at_table(choice, vs)
            for b in sorted({k[0] for k, ov in tab.items() if ov}):
                B, L = loads[b]
                T, num, den = B * L, 0.0, 0.0
                data = _at_input(B, L, 3000 + T, dev)
                ch = [v for v in vs if choice.get((v[0], b)) == v[1] and b in v[2]]
                tl = max([AT_TOL] + [v[7] for v in ch if v[7] is not None])
                mg = min(AT_MARGIN if v[8] is None else v[8] for v in ch) if ch else AT_MARGIN
                for kind in ("sliding", "full"):
                    if tab.get((b, kind)):
                        r = _at_measure(dict(data, config=_at_cfg(B, L, kind)), state, base, cal[T][0],
                                        [None, ("all", "combo", (b,), (kind,), tab[(b, kind)], True, True, tl, None)], tm,
                                        4000 + T, 2, dl2)
                        num += AT_WEIGHT[kind] * r["t"][0]
                        den += AT_WEIGHT[kind] * r["t"][0] / r["r"][1] if not r["bad"] and len(r["r"]) > 1 else 1e30
                g = num / den if den > 0 else 0.0
                lines.append(f"[at] confirm bucket {b}: combination x{g:.3f} vs ship -> "
                             + ("kept" if g > 1 + mg / 2 else "REVERTED to ship"))
                if g <= 1 + mg / 2:
                    choice = {u: n for u, n in choice.items() if u[1] != b}
                del data
        for op, b, txt in rows:
            print(txt + choice.get((op, b), "ship"), flush=True)
#@D+
        if AT_DIAG:
            for key, r in sorted(res.items(), key=str):
                cells = " | ".join(f"{i[0]}:{i[1]} {t:.1f} x{rr:.3f}{' BAD' if i in r['bad'] else ''}"
                                   for i, t, rr in zip(r["ids"][1:], r["t"][1:], r["r"][1:]))
                cells += "".join(f" | {i[0]}:{i[1]} BAD {m}" for i, m in r["bad"].items() if i not in r["ids"])
                print(f"[at] diag {key[0]}/{key[1]}: ship {r['t'][0]:.1f} us (n {r['n']}) | {cells}", flush=True)
#@D-
        _AT["choice"], _AT["call"] = choice, _at_table(choice, vs)
        for ln in lines:
            print(ln, flush=True)
        _AT["secs"] = time.time() - t1
        print(f"[at] autotune {_AT['secs']:.1f}s ({sum(len(r['ids']) for r in res.values())} graphs, {len(res)} shapes, "
              f"{AT_SEEDS}x{AT_ROUNDS} rounds, replay overhead {tm[4]:.1f} us, margin {AT_MARGIN:.3f}, diag {AT_DIAG}); routing "
              + ", ".join(f"T={t}: k {c[0]:.2f} R {c[1]:.1f}/{AT_ROUTER[_k2a_bucket_of(t)][0]} D {c[2]:.1f}/"
                          f"{AT_ROUTER[_k2a_bucket_of(t)][1]}" for t, c in sorted(cal.items()))
              + f"; decided {', '.join(f'{u[0]}/{u[1]} {n}' for u, n in sorted(choice.items(), key=str)) or 'none (shipped routes)'}; "
              f"prepare total {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:
        _AT["call"], _AT["choice"] = {}, {}
        print(f"[at] FAILED after {time.time() - t1:.1f}s ({ex!r})"[:400] + ": shipped routes everywhere", flush=True)
    finally:
        try:
            torch.cuda.synchronize()
        except Exception as ex:
            print(f"[at] cleanup raised {ex!r}"[:300], flush=True)
'''


def _emit(names, diag, vmlfc="", lean=False):
    """The generated pieces: _at_swap / _at_cur (explicit `global` statements: no globals()[...] / setattr), the prepare-time
    set, the variant list and the timeline table."""
    gl = ", ".join(names)
    swap = ["def _at_swap(ov):", f"    global {gl}", "    old = {}", "    if not ov:", "        return old"]
    for n in names:
        swap.append(f'    if "{n}" in ov: old["{n}"], {n} = {n}, ov["{n}"]')
    swap.append("    return old")
    cur = ["def _at_cur():", "    return {" + ", ".join(f'"{n}": {n}' for n in names) + "}"]
    ps = ["def _at_prepare_set():", "    ov = {}"]
    for p in PREPARE_SET:
        if lean and p["diag_only"]:
            continue
        ind = "    "
        if p["diag_only"]:
            ps.append("    if AT_DIAG:")
            ind = "        "
        ps += [f"{ind}try:", f"{ind}    ov[{p['name']!r}] = {p['expr']}", f"{ind}except Exception:", f"{ind}    pass"]
    ps.append("    return ov")
    va = ["def _at_variants(S):", "    out = []", "",
          "    def add(op, name, buckets, kinds, ship, fn, tol=None, margin=None):",
          "        try:",
          "            ov, ok = fn()",
          "            ok = bool(ok)",
          "        except Exception:",
          "            ov, ok = {}, False",
          "        out.append((op, name, buckets, kinds or ('sliding', 'full'), ov, ok, ship, tol, margin))", ""]
    done = set()
    for v in REGISTRY:
        if lean and not v["ship"]:
            continue
        ov = "{" + ", ".join(f'"{k}": {e}' for k, e in v["set"].items()) + "}"
        kd = None if v["kinds"] == KINDS else v["kinds"]
        tl = ("" if v["tol"] is None and v.get("margin") is None else f", {v['tol']!r}") + (
            "" if v.get("margin") is None else f", {v['margin']!r}")
        if "fam" in v:   # one loop per family, at its first surviving member
            if v["fam"] in done:
                continue
            done.add(v["fam"])
            f = FAMILIES[v["fam"]]
            rows = tuple(r for r in f["rows"] if any(m.get("fam") == v["fam"] and m["op"] == eval(f["op"], {}, dict(zip(f["vars"], r)))
                                                     and m["name"] == eval(f["name"], {}, dict(zip(f["vars"], r))) for m in REGISTRY))
            lv = ", ".join(f["vars"])
            dv = ", ".join(f"{x}={x}" for x in f["vars"])
            va.append(f"    for {lv} in {rows!r}:")
            va.append(f"        add({f['op']}, {f['name']}, {f['buckets']}, {kd!r}, {v['ship']!r}, lambda {dv}: ({ov}, {v['when']}){tl})")
            continue
        va.append(f"    add({v['op']!r}, {v['name']!r}, {v['buckets']!r}, {kd!r}, {v['ship']!r}, "
                  f"lambda: ({ov}, {v['when']}){tl})")
    va.append("    return out")
    tl = [t["src"].strip().replace("@VMLFC@", vmlfc) for t in TIMELINES]
    tl.append("_AT_TL = (" + "".join(f"({t['name']!r}, {t['buckets']!r}, {t['fn']}), " for t in TIMELINES) + ")")
    nl = chr(10)
    return {"@SWAP@": nl.join(swap), "@CUR@": nl.join(cur), "@PREPSET@": nl.join(ps), "@VARIANTS@": nl.join(va),
            "@TLDEFS@": (nl + nl + nl).join(tl), "@DIAG@": repr(bool(diag))}


ANCHORS = (("def prepare(weights):\n", "def _at_prepare0(weights):\n"),
           ("def kernel_fn(data, state):\n", "def _at_kernel_fn0(data, state):\n"))


def apply(text, diag=False, with_builtin=True, report=None, lean=None):
    """text: a layer file (stripped base). Returns text + AT. Variants whose override targets are not module-level
    assignments in `text` are dropped (and reported). lean (default: not diag) = the ship copy: only ship variants, no
    diag-only prepare sets, no timelines, no diag table (the research copy, diag=True, carries all of them)."""
    lean = (not diag) if lean is None else lean
    rep = report if report is not None else []
    if with_builtin and not REGISTRY:
        builtin()
    for a, b in ANCHORS:
        if text.count(a) != 1:
            raise ValueError(f"anchor {a.strip()!r} found {text.count(a)} times")
        text = text.replace(a, b)
    if re.search(r"^(_at_|AT_|_AT\b)", text, re.M):
        raise ValueError("base already defines _at_ / AT_ names (apply twice?)")
    glob = {m.group(1): m.group(2) for m in re.finditer(r"^([A-Z][A-Z0-9_]*) = (.)", text, re.M)}
    keep = []
    for v in REGISTRY:
        miss = [k for k in v["set"] if k not in glob]
        if miss:
            rep.append(f"dropped {v['op']}/{v['name']}: {miss} not module-level in the base")
            continue
        keep.append(v)
    REGISTRY[:] = keep
    for p in list(PREPARE_SET):
        if p.get("unless") and p["unless"] in glob:
            rep.append(f"dropped prepare_set {p['name']}: the base defines {p['unless']}")
            PREPARE_SET.remove(p)
        elif p["name"] not in glob:
            rep.append(f"dropped prepare_set {p['name']}: not module-level in the base")
            PREPARE_SET.remove(p)
    ops = {}
    for v in REGISTRY:
        if lean and not v["ship"]:
            continue
        for k in v["set"]:
            ops.setdefault(k, set()).add(v["op"])
    for k, o in ops.items():
        if len(o) > 1 and glob[k] != "{":
            raise ValueError(f"{k} is overridden by several ops {sorted(o)} but is not a dict (overrides cannot merge)")
    names = sorted(set(ops) | {p["name"] for p in PREPARE_SET if not (lean and p["diag_only"])})
    src = AT_SRC
    vmlfc = (", flags=VML_FLAGS.get(rep), cost=VML_COST.get(rep)"
             if "VML_FLAGS" in glob and "VML_COST" in glob and "flags=VML_FLAGS.get(rep)" in text else "")
    for key, val in _emit(names, diag, vmlfc, lean).items():
        src = src.replace(key, val)
    if lean:
        src = re.sub(r"(?ms)^#@D\+$.*?^#@D-$\n?", "", src)
    src = re.sub(r"(?m)^#@D[+-]$\n?", "", src)
    vtext = src[src.index("def _at_variants(S):"):]
    if "_at_gdrop(" not in vtext.split("return out", 1)[0]:   # GD helpers only where a variant uses them
        src = re.sub(r"(?ms)^#@G\+$.*?^#@G-$\n?", "", src)
    src = re.sub(r"(?m)^#@G[+-]$\n?", "", src)
    if lean and re.search(r"_at_timelines|_at_tl_|\[at\] diag", src):
        raise ValueError("lean copy still carries diag-only code")
    if re.search(r"@[A-Z]+@", src):
        raise ValueError("unfilled template slot")
    ast.parse(src)
    nv = sum(1 for v in REGISTRY if v["ship"] or not lean)
    rep.append(f"AT: {nv} variants ({sum(v['ship'] for v in REGISTRY)} ship), "
               f"{sum(1 for p in PREPARE_SET if not (lean and p['diag_only']))} prepare sets, "
               f"{0 if lean else len(TIMELINES)} timelines, {len(names)} swapped globals {names}, diag {diag}, lean {lean}")
    return text.rstrip(chr(10)) + chr(10) + src.rstrip(chr(10)) + chr(10)


def drop_gd(text, keys, rep):
    """Static 0-byte route change: remove (proj, bucket) entries from the base's GD_ROUTE literal (the next route in the chain
    then serves them, e.g. DG1 at o/32). Verified with ast.literal_eval: only those keys disappear, nothing else changes."""
    m = re.search(r"(?ms)^GD_ROUTE = (\{.*?\})$", text)
    if not m:
        raise ValueError("GD_ROUTE literal not found")
    old = ast.literal_eval(m.group(1))
    lit = m.group(1)
    for p, b in keys:
        if (p, b) not in old:
            raise ValueError(f"GD_ROUTE has no {(p, b)}")
        pat = re.escape(f'("{p}", {b}): ') + r"\([^()]*\)"
        n0 = len(lit)
        lit = re.sub(pat + r",\s*", "", lit, count=1)
        if len(lit) == n0:
            lit = re.sub(r",\s*" + pat, "", lit, count=1)
    new = ast.literal_eval(lit)
    want = {k: v for k, v in old.items() if k not in set(keys)}
    if new != want:
        raise ValueError("GD_ROUTE edit changed more than the dropped keys")
    rep.append(f"static: GD_ROUTE without {keys} ({len(m.group(1)) - len(lit)} B less)")
    return text[:m.start(1)] + lit + text[m.end(1):]


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    gdkeys = [(k.split(":")[0], int(k.split(":")[1])) for k in args.get("drop-gd", "").split(",") if k]
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    _, chk_base, _ = GM.G37.generate()
    n = args.get("n", "1")
    base = ROOT / f"kernels/muse_decode_layer/{args.get('base', 'k3_va_3')}.py"
    b = base.read_text(encoding="utf-8")
    bn = b.split(chr(10))[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{base.name} is not a finish() fixed point")
    lab_only = "--lab-only" in sys.argv
    builtin()
    import importlib
    for name in [x for x in args.get("extra", "").split(",") if x]:
        importlib.import_module(name).register(sys.modules[__name__])
        print(f"[at_patches] OK  extra variants from tools/{name}.py registered")
    outs = []
    for diag in (False, True):
        sfx = "r" if diag else ""
        tag = f"k3-at-{n}{sfx}"
        nonce = (f"# nonce {tag} ({base.stem} + AT: prepare()-time route autotuner over the registered variants, "
                 f"default kept unless > 1.5 % faster; {'research copy, AT_DIAG True' if diag else 'lean ship copy'}"
                 + (f"; static GD_ROUTE without {', '.join(f'{k[0]}/{k[1]}' for k in gdkeys)}" if gdkeys else "") + ")")
        rep = [f"base {base.name} {len(b.encode())} B is a finish() fixed point"]
        reg = [dict(v) for v in REGISTRY], [dict(p) for p in PREPARE_SET]
        t0 = b.replace(bn, nonce)
        if gdkeys:
            t0 = drop_gd(t0, gdkeys, rep)
        t = apply(t0, diag=diag, with_builtin=False, report=rep)
        kept = sum(1 for v in REGISTRY if v["ship"])
        REGISTRY[:], PREPARE_SET[:] = reg
        if lab_only:
            chk0 = GM.check
            GM.check = lambda ok, msg: chk0(ok or "file >= 1 MB" in msg, msg)
        try:
            s = GM.finish(t, nonce, tag, chk_base, rep)
        finally:
            if lab_only:
                GM.check = chk0
        GM.check(s.count("AT_DIAG = " + repr(diag)) == 1, f"{tag}: AT_DIAG line")
        rep.append(f"size: {len(b.encode())} -> {len(s.encode())} B (+{len(s.encode()) - len(b.encode())} B, headroom "
                   f"{1_000_000 - len(s.encode())} B)")
        for r in rep:
            print(f"[at_patches] OK  {r}")
        out = ROOT / f"kernels/muse_decode_layer/k3_at_{n}{sfx}.py"
        lab = ROOT / f"kernels/lab/atn_k3_at_{n}{sfx}_ab.py"
        if not lab_only:
            out.write_text(s, encoding="utf-8", newline=chr(10))
        lab.write_text(s, encoding="utf-8", newline=chr(10))
        outs.append(s)
        print(f"[at_patches] wrote {'' if lab_only else str(out.relative_to(ROOT)) + ' and '}{lab.relative_to(ROOT)} "
              f"({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    a, r = outs[0], outs[1]
    ia, ir = a.index(chr(10) + "AT_ENABLE = True" + chr(10)), r.index(chr(10) + "AT_ENABLE = True" + chr(10))
    pa, pr = a[:ia].split(chr(10)), r[:ir].split(chr(10))
    diff = [i for i, (x, y) in enumerate(zip(pa, pr)) if x != y]
    GM.check(len(pa) == len(pr) and diff == [1], f"ship / research copies differ before the AT section on lines {diff[:6]}")
    def ship_adds(t):   # the variant list without research lines (a research family's loop header goes with it)
        i = t.index("def _at_variants(S):")
        body = t[i:t.index(chr(10) + "    return out" + chr(10), i)].split(chr(10))
        keep = []
        for k, ln in enumerate(body):
            nxt = body[k + 1] if k + 1 < len(body) else ""
            if ", False, lambda" in ln or (ln.startswith("    for ") and ", False, lambda" in nxt):
                continue
            keep.append(ln)
        return keep
    GM.check(ship_adds(a) == ship_adds(r), "ship variants differ between the ship and the research copy")
    GM.check(", False, lambda" not in a and "_at_timelines" not in a, "ship copy carries research variants / timelines")
    nship = kept
    print(f"[at_patches] OK  ship copy (lean, {nship} ship variants) and research copy share the layer and the ship "
          f"variants; AT costs {len(a.encode()) - len(b.encode())} B in the ship copy, {len(r.encode()) - len(b.encode())} B in "
          f"the research copy")
    return 0


if __name__ == "__main__":
    sys.exit(main())

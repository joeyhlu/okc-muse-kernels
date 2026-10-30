"""rob2_patches.py (R-rob-2): robustness riders on b29 (notes/study/robustness_b29.md items 2 / 6); 0-perf on the scored shapes.

  python tools/rob2_patches.py [--base=kernels/lab/k3_b29_1.py] [--out=kernels/lab/k3_rob2_0.py] [--no-p2] [--dry] [--force]
  python -I tools/slim_patches.py --base=kernels/lab/k3_rob2_0.py --out=kernels/lab/k3_rob2_1.py      (LAST)
  Raw-base pipeline (lead rule): --base=kernels/lab/k3_b29_raw.py (or any later raw bundle) -> other riders -> slim_patches LAST;
  slim_patches(rob2(k3_b29_raw)) == rob2(k3_b29_1) == kernels/lab/k3_rob2_1.py from line 3, byte for byte.
  python tools/rob2_patches.py --test [--file=kernels/lab/k3_rob2_1.py] [--base=kernels/lab/k3_b29_1.py]   (local, no GPU)

P1: six call-time `except` sites still disabled a route for the whole process on one failure (R1 only covered the 17
    `_k3_disable(state, ...)` sites): VMS (`st["vms_ok"] = False`), VML (`_VML["ok"] = {}`, module-global), K2A v3.7 / v3.6
    (`st["r37"/"r36"].clear()`), GD cat (`state["gd_cat_ok"] = False`) and G2-A (`g2["prt"] = False`, T = 1). Each now goes
    through `_k3_disable` on its own container (the K2A `st`, `g2`, or the top state), so it is scoped to (container key,
    shape key) like R1. `_k3_disable` registers a sub-container (one that is not kernel_fn's `state`) in
    `state["_k3_subs"]` of the kernel_fn frame it finds, and `_k3_dis_apply` re-applies orig / per-shape values on the state
    and every registered sub-container. VML's process-global `_VML["ok"]` is no longer touched at call time: a per-state
    `st["vml_off"]` gate (one dict .get, CPU-only; the judge times graph replays) carries the disable. Prepare-time
    failures stay process-wide, as before (no kernel_fn frame).
P2: page_size > 1, B = 1 full decode ran K1 (+25 % layer at page 64): AX1's site already reads R2's `adata`, but
    ("full", 1) was not in R2_CLASSES and `_r2_page1` never asked `_ax1_wanted`. Now ("full", 1) at B = 1 builds the
    token-level page-1 view (one Triton launch, ~W int32 writes) when AX1 or FA4 wants it; B >= 2 full decode returns `data`
    before any check (VA1 is page-aware), exactly as today. Judge test 2 (B 1, ctx 3000, full, page 64) exercises it.
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
sys.path.insert(0, str(ROOT / "tools"))


def check(c, msg):
    if not c:
        raise SystemExit(f"[rob2_patches] FAIL: {msg}")


def sub(t, old, new, count=1, what=""):
    n = t.count(old)
    check(n == count, f"anchor {what or old[:70]!r}: found {n}, expected {count}")
    return t.replace(old, new)


# ---------------------------------------------------------------- P1: shape-scoped disables on sub-containers
DIS_OLD = '''        dis.setdefault(sk, {})[key] = value
        _K3_DIS_ON = True
    state[key] = value
'''
DIS_NEW = '''        dis.setdefault(sk, {})[key] = value
        _K3_DIS_ON = True
        top = f.f_locals.get("state")   # rob2 P1: a sub-container (K2A st, g2) is re-applied with kernel_fn's state
        if isinstance(top, dict) and top is not state and not any(c is state for c in top.setdefault("_k3_subs", [])):
            top["_k3_subs"].append(state)
    state[key] = value
'''
APPLY_OLD = '''def _k3_dis_apply(data, state):
    dis = state.get("_k3_dis", {}).get(_k3_shape_key(data), {})
    for k, v in state.get("_k3_orig", {}).items():
        state[k] = dis[k] if k in dis else v
'''
APPLY_NEW = '''def _k3_dis_apply(data, state):
    sk = _k3_shape_key(data)
    for c in (state, *state.get("_k3_subs", ())):
        dis = c.get("_k3_dis", {}).get(sk, {})
        for k, v in c.get("_k3_orig", {}).items():
            c[k] = dis[k] if k in dis else v
'''
SITES = [
    ("VMS", '''        except Exception as ex:
            print(f"[k3] VMS route raised ({ex!r}): K2A from here on", flush=True)
            st["vms_ok"] = False
''', '''        except Exception as ex:
            _k3_disable(st, "vms_ok", False, "VMS", ex)
'''),
    ("VML gate", '''    if VML_ENABLE and VML_BUCKETS.get(vrep) and _VML["ok"].get(vrep) and x.shape[0] <= _vml_MMAX:
''', '''    if VML_ENABLE and VML_BUCKETS.get(vrep) and _VML["ok"].get(vrep) and x.shape[0] <= _vml_MMAX and not st.get("vml_off"):
'''),
    ("VML", '''        except Exception as ex:
            print(f"[k3] VML1 route raised ({ex!r}): K2A from here on", flush=True)
            _VML["ok"] = {}
''', '''        except Exception as ex:
            _k3_disable(st, "vml_off", True, "VML1", ex)
'''),
    ("K2A v3.7", '''        except Exception as ex:
            print(f"[k3] K2A v3.7 route raised ({ex!r}): v3.6 / v3.3 routes from here on", flush=True)
            st["r37"].clear()
''', '''        except Exception as ex:
            _k3_disable(st, "r37", {}, "K2A v3.7", ex)
'''),
    ("K2A v3.6", '''        except Exception as ex:
            print(f"[k3] K2A v3.6 route raised ({ex!r}): v3.3's plan from here on", flush=True)
            st["r36"].clear()
''', '''        except Exception as ex:
            _k3_disable(st, "r36", {}, "K2A v3.6", ex)
'''),
    ("GD cat", '''    except Exception as ex:
        state["gd_cat_ok"] = False
        print(f"[gc] call raised ({ex!r}): today's cat from here on", flush=True)
        return None
''', '''    except Exception as ex:
        _k3_disable(state, "gd_cat_ok", False, "GD cat", ex)
        return None
'''),
    ("G2-A", '''    except Exception as ex:
        g2["prt"] = False
        print(f"[g2] G2-A raised at call time, k3_v18 path from now on: {ex!r}", flush=True)
        return None
''', '''    except Exception as ex:
        _k3_disable(g2, "prt", False, "G2-A", ex)
        return None
'''),
]


def p1(t):
    t = sub(t, DIS_OLD, DIS_NEW, what="_k3_disable body (R1)")
    t = sub(t, APPLY_OLD, APPLY_NEW, what="_k3_dis_apply (R1)")
    for what, old, new in SITES:
        t = sub(t, old, new, what=what)
    return t


# ---------------------------------------------------------------- P2: B = 1 full decode at page > 1 -> R2 page-1 view (AX1 / FA4)
CLS_OLD = '''R2_CLASSES = {("sliding", 1), ("sliding", 8), ("full", 8)}
'''
CLS_NEW = '''R2_CLASSES = {("sliding", 1), ("sliding", 8), ("full", 8), ("full", 1)}
'''
GATE_OLD = '''            or not lens.is_contiguous() or kc.dim() != 4 or kc.shape[1] != ps or not kc.is_contiguous() or not vc.is_contiguous()):
        return data
'''
GATE_NEW = '''            or not lens.is_contiguous() or kc.dim() != 4 or kc.shape[1] != ps or not kc.is_contiguous() or not vc.is_contiguous()):
        return data
    if kind == "full" and cfg["q_len"] == 1 and cfg["batch"] != 1:   # rob2 P2: full decode B >= 2 stays on VA1 (page-aware)
        return data
'''
WANT_OLD = '''    if not (_ks1_wanted(x, view, cfg1, state) or (not _va1_b1(cfg1, state) and _v18_fa4_wanted(x, view, cfg1, state))):
        return data
'''
WANT_NEW = '''    if not (_ks1_wanted(x, view, cfg1, state) or _ax1_wanted(x, view, cfg1, state)
            or (not _va1_b1(cfg1, state) and _v18_fa4_wanted(x, view, cfg1, state))):
        return data
'''
AX1_SITE = '''    if _ax1_wanted(qkv, adata, adata["config"], state):
        try:
            o = _ax1_call(qkv, adata, adata["config"]).view(tokens, hq * d)'''


def p2(t):
    check(t.count(AX1_SITE) == 1, "AX1 site must already read adata (b26+)")
    t = sub(t, CLS_OLD, CLS_NEW, what="R2_CLASSES")
    t = sub(t, GATE_OLD, GATE_NEW, what="_r2_page1 gate")
    return sub(t, WANT_OLD, WANT_NEW, what="_r2_page1 wanted")


# ---------------------------------------------------------------- gates shared by build and --check
def gates(t, base, label):
    from pyslim import decorated_sources
    check(t.isascii(), f"{label}: non-ASCII")
    n = len(t.encode())
    check(re.search(r"importlib|sys\.modules", t) is None or re.search(r"importlib|sys\.modules", base) is not None, "checker")
    check(max(len(x) for x in t.split(NL)) <= max(400, max(len(x) for x in base.split(NL))), "line length")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
        r = subprocess.run([sys.executable, "-m", "pyflakes", str(f)], capture_output=True, text=True)
        undef = [ln for ln in r.stdout.splitlines() if "undefined name" in ln or "unable to detect undefined" in ln]
    check(not undef, f"{label}: pyflakes undefined names {undef[:5]}")
    d0, d1 = decorated_sources(base), decorated_sources(t)
    bad = [s for s in d1 if s.startswith("UNPARSABLE")]
    check(not bad and len(d1) == len(d0), f"{label}: decorated defs {len(d0)} -> {len(d1)}, unparsable {bad[:2]}")
    check(d0 == d1, f"{label}: a decorated def changed")
    print(f"[rob2_patches] gates PASS ({label}): {n} B, ascii, compiles, pyflakes 0 undefined, {len(d1)} decorated defs "
          f"re-parse identically")
    return n


def build():
    base_p = ROOT / ARGS.get("base", "kernels/lab/k3_b29_1.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_rob2_0.py")
    b = base_p.read_text(encoding="utf-8")
    t, what = p1(b), ["P1 six call-time disables shape-scoped"]
    if "no-p2" not in FLAGS:
        t = p2(t)
        what.append("P2 R2 view for B=1 full decode (AX1 / FA4)")
    bn = t.split(NL)[1]
    nonce = f"# nonce {out.stem.replace('_', '-')} ({base_p.stem} + {', '.join(what)}; tools/rob2_patches.py)"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    n = gates(t, b, out.name)
    raw = len(b.encode()) >= 1_000_000   # the unslimmed raw base (k3_b29_raw.py) is over the cap by design
    check(n < 1_000_000 or raw, f"{n} B >= 1 MB")
    if raw:
        print("[rob2_patches] raw base: over 1 MB until slim_patches runs (LAST); the cap is checked there")
    print(f"[rob2_patches] {base_p.name} -> {out.name}: {n} B (+{n - len(b.encode())}); {'; '.join(what)}")
    if "dry" in FLAGS:
        return 0
    check(not out.exists() or "force" in FLAGS, f"{out.name} exists (never overwrite; --force)")
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[rob2_patches] wrote {out.relative_to(ROOT).as_posix()} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]}); "
          f"next: python -I tools/slim_patches.py --base={out.relative_to(ROOT).as_posix()} --out=<final>")
    return 0


# ---------------------------------------------------------------- local proof (pure Python, no torch / GPU)
# The real `_k3_*`, `_v11_k2a33_call`, `_gd_cat`, `_gd_cat_live`, `_g2_prt`, `_r2_page1` sources are exec'd from the file;
# only the kernels they launch are mocked. Each mock route raises on its "nearby" shape tags only.
MOCK = '''
import types
CUR = [None]
FAIL = {"vms": {"d12"}, "vml": {"d40"}, "r37": {"v12"}, "r36": {"v12", "d20"}, "gdcat": {"d24"}, "g2": {"d1c8k"}}
RAISES = []
def _hit(route):
    if CUR[0] in FAIL.get(route, ()):
        RAISES.append((route, CUR[0]))
        raise RuntimeError(f"forced {route} failure at {CUR[0]}")
class _X:
    def __init__(self, T, dt="bf16", K=8192):
        self.shape, self.dtype, self.device = (T, K), dt, "cuda"
    def contiguous(self): return self
    def clone(self): return self
    def data_ptr(self): return 0
    def dim(self): return 2
    def is_contiguous(self): return True
class _KC:
    def __init__(self, n): self.shape = (n, 1, 2, 64)
torch = types.SimpleNamespace(bfloat16="bf16", float32="f32", empty=lambda *a, **k: None)
K2A37_ENABLE = True                      # the shipped file has it off; on here so the v3.7 site is exercised
K2A36_BUCKETS = {16: True, 32: True, 128: True}   # + 128 so a v3.7 failure falls through to the v3.6 site
_VML = {"ok": {32: True, 256: True}}     # bucket 128 on K2A v3.7 / v3.6 so their sites are reachable
def _vms_call(st, x, ids, w): _hit("vms"); return "VMS"
def _vml_route(st, data): _hit("vml"); return "VML"
def _k2a37_call(st, data, *a): _hit("r37"); return "K37"
def _k2a36_call(st, data, *a): _hit("r36"); return "K36"
def _k2a33_call(st, data, vid, hot_t=None, hot_w4=None): return "K33"
def _gd_cat_run(r, x, y, part, tokens, pdl): _hit("gdcat")
def _v8_tail(y, part, state, data): return "GC"
_GD_CAT_SEEN = set()
_vf_t1_live = lambda state: False
_v13_p1_ok = lambda x, K: True
_VF = {"prt": False}
def _g2_prt_run(g2, P, x, r, state, data, flags): _hit("g2"); return "G2A"
def _v8_experts(x, state):               # one frame between kernel_fn and the MoE entry, like the real chain
    return _v11_k2a33_call(x, None, None, state["k2a33"]["st"])
def kernel_fn(data, state):
{HEAD}
    CUR[0] = data["tag"]
    T = data["config"]["tokens"]
    x = _X(T)
    if T == 1:
        return (_g2_prt(x, _X(T, "f32"), state, data, T, "f32", "sep") or "sep",)
    gc = _gd_cat(state, x, data, T) if 13 <= T <= 32 else None
    return (gc or "cat", _v8_experts(_X(T, K=4096), state))
SHAPES = {"d1": (1, 1, 80001), "d1c8k": (1, 1, 8001), "d16": (16, 16, 16 * 80001), "d12": (12, 12, 12 * 80001),
          "d20": (20, 20, 20 * 80001), "d24": (24, 24, 24 * 80001), "d32": (32, 32, 32 * 80001), "d40": (40, 40, 40 * 80001),
          "v16": (128, 16, 16 * 80008), "v12": (96, 12, 12 * 80008)}
def shape(tag):
    T, B, rows = SHAPES[tag]
    return {"tag": tag, "config": {"batch": B, "q_len": T // B, "tokens": T, "page_size": 1, "window": 2048},
            "k_cache": _KC(rows)}
def new_state():
    st = {"vms_ok": True, "r37": {128: (0, 0, 0)}, "r36": {16: (0, ()), 128: (0, ())}}
    return {"k2a33": {"st": st}, "gd": {}, "gd_cat": {16: {"T": 16, "desc": "m"}, 32: {"T": 32, "desc": "m"}},
            "g2": {"prt": True}, "v8": {}, "p1": {"proj": {"pooler": {}}, "route": {("pooler", 1): True}}}
'''
REAL = {"_K3_DIS_ON", "_k3_shape_key", "_k3_disable", "_k3_dis_apply", "_v11_k2a33_call", "_k2a_bucket_of", "K2A_M_MAX",
        "K2A_BUCKETS", "_nb_vml_rep", "NB_VML_REP", "_vml_MMAX", "VMS_ENABLE", "VMS_MAXT", "VML_ENABLE", "VML_BUCKETS",
        "K2A36_ENABLE", "K2A36_BUCKETS", "SK1_ENABLE", "SK1_VID", "K2A33_PLAN", "K2A33_HOT_T_BUCKET", "_gd_cat", "_gd_cat_live",
        "_gd_bucket", "_nb64", "NB64_LO", "NB64_HI", "NB64_USE", "GD_ENABLE", "GD_CAT", "GD_CAT_PDL", "HIDDEN", "D_EXPERT",
        "N_EXPERTS", "_g2_prt", "G2_ENABLE", "G2_PRT", "G2_PRT_PDL", "G2_PRT_MODE", "VF_PRT", "p1_bucket", "_K3_WARNED"}
SEQ = ["d1", "d16", "d32", "v16", "d1c8k", "d12", "d20", "d24", "d40", "v12", "d1", "d16", "d32", "v16",
       "d1c8k", "d12", "d20", "d24", "d40", "v12", "d32", "v16", "d1"]
WANT = {"d1": ("G2A",), "d16": ("GC", "VMS"), "d32": ("GC", "VML"), "v16": ("cat", "K37"), "d1c8k": ("sep",),
        "d12": ("cat", "K33"), "d20": ("GC", "K33"), "d24": ("cat", "K36"), "d40": ("cat", "K33"), "v12": ("cat", "K33")}
WANT_RAISES = [("g2", "d1c8k"), ("vms", "d12"), ("r36", "d20"), ("gdcat", "d24"), ("vml", "d40"), ("r37", "v12"), ("r36", "v12")]


def _names(node):
    ts = node.targets if isinstance(node, ast.Assign) else [node.target]
    out = set()
    for t_ in ts:
        out |= {getattr(e, "id", None) for e in (t_.elts if isinstance(t_, ast.Tuple) else [t_])}
    return out


def _extract(src, names):
    keep = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(ast.get_source_segment(src, node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and _names(node) & names:
            keep.append(ast.get_source_segment(src, node))
    return NL.join(keep)


def _run(src, label):
    tree = ast.parse(src)
    kf = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_fn")
    head = [ln for ln in ast.get_source_segment(src, kf).split(NL)[1:4] if "_K3_DIS_ON" in ln or "_k3_dis_apply" in ln]
    ns = {"print": lambda *a, **k: None}
    exec(_extract(src, REAL), ns)
    ns["_K3_WARNED"] = set()
    exec(MOCK.replace("{HEAD}", NL.join(head) if head else "    pass"), ns)
    st = ns["new_state"]()
    got = [ns["kernel_fn"](ns["shape"](tg), st) for tg in SEQ]
    try:   # a disable outside kernel_fn (prepare) stays process-wide
        raise RuntimeError("prepare-time failure")
    except Exception as ex:
        st2 = ns["new_state"]()
        ns["_k3_disable"](st2["k2a33"]["st"], "vms_ok", False, "VMS", ex)
        outside = st2["k2a33"]["st"]["vms_ok"] is False and ns["kernel_fn"](ns["shape"]("d16"), st2)[1] != "VMS"
    bad = [(i, tg, g, WANT[tg]) for i, (tg, g) in enumerate(zip(SEQ, got)) if g != WANT[tg]]
    print(f"[rob2-test] {label}: {len(bad)} / {len(SEQ)} calls off the expected route; raises {ns['RAISES']}; "
          f"prepare-time disable process-wide: {outside}")
    for i, tg, g, w in bad[:8]:
        print(f"    call {i} {tg}: got {g}, want {w}")
    return bad, ns["RAISES"], outside


R2_MOCK = '''
import types
class _T:
    def __init__(self, shape, dtype="i32"): self.shape, self.dtype = shape, dtype
    def dim(self): return len(self.shape)
    def is_contiguous(self): return True
    def view(self, *s): return _T(s, self.dtype)
    device = "cuda"
torch = types.SimpleNamespace(int32="i32", zeros=lambda s, dtype=None, device=None: _T(s, dtype))
class _Launch:
    def __getitem__(self, grid):
        def f(*a, **k): LAUNCH.append(grid)
        return f
LAUNCH = []
_r2_pt1_kernel = _Launch()
triton = types.SimpleNamespace(cdiv=lambda a, b: -(-a // b))
_R2_BUF = {}
R2_MARGIN = 512
WANT = {"ks1": False, "ax1": True, "fa4": True}
_ks1_wanted = lambda x, d, c, s: WANT["ks1"] and bool(c["window"])
_ax1_wanted = lambda x, d, c, s: WANT["ax1"] and c["batch"] == 1 and not c["window"] and c["page_size"] == 1
_v18_fa4_wanted = lambda x, d, c, s: WANT["fa4"] and c["page_size"] == 1 and (bool(c["window"]) or c["q_len"] == 8 or c["batch"] <= 1)
_va1_b1 = lambda c, s: False
_vfa_wanted = lambda x, d, c, s: c["q_len"] == 8 and not c["window"] and c["batch"] >= 2
def case(B, q, window, ps=64, rows=4096):
    cfg = {"batch": B, "q_len": q, "tokens": B * q, "page_size": ps, "window": window, "kv_scale": 1.0}
    return {"config": cfg, "page_table": _T((B, 1251)), "cache_seqlens": _T((B,)), "k_cache": _T((rows, ps, 2, 64), "fp8"),
            "v_cache": _T((rows, ps, 2, 64), "fp8"), "attn_input": _T((B * q, 8192), "bf16")}, cfg
'''


def _r2_run(src, label):
    ns = {}
    exec(_extract(src, {"_r2_page1", "R2_CLASSES"}), ns)
    exec(R2_MOCK, ns)
    res = {}
    for tag, (B, q, w), wa in (("d1f", (1, 1, 0), {}), ("d1f fa4-only", (1, 1, 0), {"ax1": False}),
                               ("d1f none", (1, 1, 0), {"ax1": False, "fa4": False}), ("d2f", (2, 1, 0), {}),
                               ("d16f", (16, 1, 0), {}), ("v1f", (1, 8, 0), {}), ("d1s", (1, 1, 2048), {"ks1": True})):
        ns["WANT"].update({"ks1": False, "ax1": True, "fa4": True}, **wa)
        n0 = len(ns["LAUNCH"])
        d, cfg = ns["case"](B, q, w)
        out = ns["_r2_page1"](d, cfg, {})
        res[tag] = ("view" if out is not d else "data", ns["LAUNCH"][n0:])
    print(f"[r2-test] {label}: " + ", ".join(f"{k}->{v[0]}{v[1] if v[1] else ''}" for k, v in res.items()))
    return {k: v[0] for k, v in res.items()}


def test():
    f = ROOT / ARGS.get("file", "kernels/lab/k3_rob2_1.py")
    base = ROOT / ARGS.get("base", "kernels/lab/k3_b29_1.py")
    bsrc, fsrc = base.read_text(encoding="utf-8"), f.read_text(encoding="utf-8")
    bb, br, _ = _run(bsrc, "ship " + base.stem)
    fb, fr, outside = _run(fsrc, "rob2 " + f.stem)
    check(bb, "the ship run should show the process-wide degrade (else the mock does not reach the sites)")
    check(not fb, f"rob2: {len(fb)} calls off route")
    check(fr == WANT_RAISES, f"each failing (route, nearby shape) must raise exactly once: {fr}")
    check(outside, "prepare-time disable must stay process-wide")
    print("[rob2-test] P1 PASS: every scored shape keeps VMS / VML / K2A v3.7 / GD cat / G2-A after nearby-shape failures; "
          "each nearby shape keeps its fallback without re-raising")
    rb, rf = _r2_run(bsrc, "ship " + base.stem), _r2_run(fsrc, "rob2 " + f.stem)
    want = {"d1f": "view", "d1f fa4-only": "view", "d1f none": "data", "d2f": "data", "d16f": "data", "v1f": "view", "d1s": "view"}
    check(rf == want, f"rob2 R2 routing {rf} != {want}")
    check(all(rb[k] == rf[k] for k in ("d2f", "d16f", "v1f", "d1s")) and rb["d1f"] == "data",
          f"ship R2 routing must differ only at B = 1 full decode: {rb}")
    print("[rob2-test] P2 PASS: B = 1 full decode at page 64 gets the page-1 view (AX1, or FA4 if AX1 is off); "
          "B >= 2 full decode and every other class unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(test() if "test" in FLAGS else build())

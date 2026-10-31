"""hyblo_patches.py (R-hyblo-1, HYB-small-T builder): alygg77's own small-M MoE builds and the in-file HYB (WGS) module as
in-layer arms ahead of VMS at T <= 16, on k3_hh_3tekv (lab file; ship behaviour unchanged with the defaults).

  python -I tools/hyblo_patches.py [--base=kernels/lab/k3_hh_3tekv.py] [--out=kernels/lab/k3_hyblo_ab.py] [--check]
  python -I tools/lead_gate.py file kernels/lab/k3_hyblo_ab.py
  ship form (R-hyblo-1 GO at T = 1): python -I tools/hyblo_patches.py --ship=1 --base=<final> --out=<new>
      only the builds the listed T need (T 1 -> tiny; 8 -> tiny2; 16 -> mid), HYBLO_T = those T, no HYB ("w") self-tests,
      self-tests at the builds' own M (+ an all-non-local nj = 0 case per build); same CUDA text (SASS-gated) and hunks.

Why: alygg77's MoE board rows bs1 / bs8 / bs16 (24.6 / 63.6 / 92.3 us vs ours 27.3 / 64.6 / 92.1) do NOT come from the WGS
build we carry as HYB. Their layer's final _moe(M, state) (their line ~13097) picks MOE_TINY_DEFINES at M <= 4,
MOE_TINY2_DEFINES at M <= 8 and MOE_MID_DEFINES at M <= 16, all compiled from _MA_ALT_SRCS["moe_helper0"] (their line 12246,
the "physical helper warp zero" variant of MOE_CUDA_SRC), with the _ma_prepare weight list w[0..13] (packed codes / scales,
fat order, unit order, WGS copies). This patcher vendors that source (Apache-2.0, notice kept), specialized with
tools/hyb_unifdef.py to the defines the three builds share (OKC_L2H / OKC_CPF / OKC_WSBLK / OKC_SPLIT left as #if and passed
explicitly per build; the SASS identity vs their exact flags is tools/local/hyblo_sass.py), plus their four repack helpers.

Arms (module globals, set per rev_ab3 variant): HYBLO_T = token counts routed to the lab engine ahead of VMS; HYBLO_K = "h"
(their build for that M) or "w" (the in-file HYB module). Defaults HYBLO_T = () -> ship dispatch, byte-identical file
outside these hunks: nonce line, lab block before _v11_k2a33_call, the 4-line route hook in _v11_k2a33_call, one prepare
call after _hyb_prepare, one PT1 job (the three builds run one after another in that job).
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from hyb_unifdef import drop_unreferenced, unifdef   # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000
TP = ROOT / "third_party" / "alygg77-okc-muse-spark-kernels-6cae49a"
PIN = "8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a"   # = tools/hyb_patches.py PIN
ARCH = {"OKC_NS": "3", "OKC_FNST": "4", "OKC_NSF": "3"}
COMMON = {"OKC_NW": "8", "OKC_TINY": "1", "OKC_EPI4": "0", "OKC_FAST1": "2", "OKC_NACC": "4", "OKC_UWARP": "0", "OKC_WS": "1",
          "OKC_WSB": "1", "OKC_WSBL": "2", "OKC_FINPF": "1", "OKC_PRMTSC": "1", "OKC_DQ7": "1"}
VAR = ("OKC_L2H", "OKC_CPF", "OKC_WSBLK", "OKC_SPLIT")
# per build: (key, max M, the four VAR values = their defines or the source defaults L2H 1 / CPF 1 / SPLIT 0)
BUILDS = (("tiny", 4, ("0", "1", "2", "0")), ("tiny2", 8, ("1", "0", "1", "0")), ("mid", 16, ("1", "0", "2", "1")))
# their exact load_inline flags per build (their _ma_ext: -O3, gencode, -lineinfo, NS / FNST / NSF, the build's defines without
# '@' entries, then the PRMT suffix and DQ7) -- for the SASS identity gate only
THEIR = {
    "tiny": ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0",
             "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1"),
    "tiny2": ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_CPF=0", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0",
              "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=1", "-DOKC_FINPF=1"),
    "mid": ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_CPF=0", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4",
            "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1")}
HEAD = ("-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-DOKC_NS=3", "-DOKC_FNST=4", "-DOKC_NSF=3")


def check(c, msg):
    if not c:
        raise SystemExit(f"[hyblo_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


def slim_cuda(src):   # = tools/hyb_patches.py slim_cuda
    out = []
    for ln in src.split("\n"):
        i = ln.find("//")
        while i >= 0 and (ln[:i].replace('\\"', "").count('"') % 2 or "'" in ln[:i]):
            i = ln.find("//", i + 2)
        if i >= 0 and not ln.rstrip().endswith("\\"):
            ln = ln[:i]
        ln = ln.strip()
        if ln:
            out.append(ln)
    return "\n".join(out) + "\n"


def upstream():
    p = TP / "kernels" / "muse_decode_layer.py"
    raw = p.read_bytes()
    check(hashlib.sha256(raw).hexdigest() == PIN, "alygg77 file sha256 differs from the pin")
    src = raw.decode("utf-8")
    lines = src.split("\n")
    ln = [x for x in lines if x.startswith('_MA_ALT_SRCS["moe_helper0"] = ')]
    check(len(ln) == 1, "moe_helper0 line")
    h0 = ast.literal_eval(ln[0].split(" = ", 1)[1])
    tree = ast.parse(src)
    fns = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in ("_pack_codes", "_pack_scales", "_fat_order", "_unit_order"):
            if isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
                n.body = n.body[1:]   # docstring dropped
            fns[n.name] = ast.unparse(n)
    check(len(fns) == 4, "repack helpers")
    return h0, fns


def strip_h0(h0):
    d = dict(ARCH, **COMMON)
    d.update({k: "@" for k in VAR})   # "@" = defined, value undecidable: the #if groups on VAR stay, their #ifndef defaults go
    t = slim_cuda(unifdef(h0, d))
    t, dropped = drop_unreferenced(t, roots=("okc_run", "okc_set_pdl"))
    check('"""' not in t and t.isascii(), "stripped CUDA embeddable as r-string")
    for k in VAR:
        check(f"#ifndef {k}" not in t, f"{k} default left")
    return t


def our_flags(vals):
    return HEAD + tuple(f"-D{k}={v}" for k, v in COMMON.items()) + tuple(f"-D{k}={v}" for k, v in zip(VAR, vals))


def their_flags(key):
    return HEAD + THEIR[key] + ("-DOKC_PRMTSC=1", "-DOKC_DQ7=1")


def helpers(fns):
    out = []
    for name in ("_pack_codes", "_pack_scales", "_fat_order", "_unit_order"):
        s = fns[name].replace(f"def {name}(", f"def _hyblo{name}(").replace("_REG_HJ", "_HYBLO_REG_HJ").replace("_POS[", "_HYB_POS[")
        out.append(s)
    return "_HYBLO_REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]\n" + "\n".join(out) + "\n"


LAB = r'''"""[HYBLO] R-hyblo-1 lab (tools/hyblo_patches.py): alygg77's small-M MoE builds (MOE_TINY M <= 4, MOE_TINY2 M <= 8, MOE_MID
M <= 16 of their _MA_ALT_SRCS["moe_helper0"]; Apache-2.0, _HYBLO_NOTICE) and the in-file HYB module as arms ahead of VMS:
HYBLO_T = routed token counts, HYBLO_K = "h" (their build for that M) / "w" (HYB). Defaults: ship dispatch."""
_HYBLO_NOTICE = r"""
==== alygg77/okc-muse-spark-kernels (Apache-2.0) -> the HYBLO lab block (search: _HYBLO_, _hyblo_) ====
Same source, license and notices as the HYB block above (commit 6cae49a, kernels/muse_decode_layer.py): its
_MA_ALT_SRCS["moe_helper0"] CUDA text (their MOE_TINY / MOE_TINY2 / MOE_MID builds), MOE_CPP_SRC (= _HYB_MOE_CPP),
_pack_codes / _pack_scales / _fat_order / _unit_order and the _ma_prepare weight list. Modified by team joeyhlu
(2026-10-31, lab only): the CUDA text is specialized to the defines the three builds share (tools/hyb_unifdef.py; comments,
indentation and unreferenced functions dropped; OKC_L2H / OKC_CPF / OKC_WSBLK / OKC_SPLIT passed explicitly per build;
cuobjdump -sass identical to their source with their flags, tools/local/hyblo_sass.py), the helpers are renamed _hyblo_*
without docstrings, and the prepare / self-test / routing glue is ours.
"""
THIRD_PARTY_NOTICES += _HYBLO_NOTICE
_HYBLO_CUDA = r"""@CUDA@"""
HYBLO_ENABLE = True
HYBLO_T = ()
HYBLO_K = "h"
HYBLO_PDL = True
_HYBLO_KEYS = @KEYS@
_HYBLO_FLAGS = @FLAGS@
HYBLO_SELF_TEST = ((1, 81, "normal"), (1, 82, "skewed"), (2, 83, "normal"), (4, 84, "skewed"), (5, 85, "normal"), (8, 86, "skewed"),
                   (8, 87, "normal"), (9, 88, "skewed"), (12, 89, "normal"), (16, 90, "skewed"), (16, 91, "normal"), (8, 92, "conc"),
                   (16, 93, "conc"))
_HYBLO = {"mods": {}, "ok": {}, "wok": False, "w": None, "sc": None, "full": None, "secs": {}, "pdl": {}, "err": {}, "calls": {}}
@HELPERS@def _hyblo_key(m):
    return next((k for k, mx, _ in _HYBLO_KEYS if m <= mx), None)
def _hyblo_build():
    from torch.utils.cpp_extension import load_inline
    for k, _, var in _HYBLO_KEYS:   # one nvcc at a time (R-hovl-1: parallel extra builds froze a container)
        fl = list(_HYBLO_FLAGS) + list(var)
        t0 = time.time()
        try:
            tag = hashlib.sha1((_HYB_MOE_CPP + _HYBLO_CUDA + " ".join(fl)).encode()).hexdigest()[:12]
            _HYBLO["mods"][k] = load_inline(name=f"hyblo_{k}_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=_HYBLO_CUDA,
                                            functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
        except Exception as ex:
            _HYBLO["err"][k] = repr(ex)[:300]
        _HYBLO["secs"][k] = time.time() - t0
def _hyblo_run(k, x, ids, wts):
    ids, wts = ids.to(torch.int32).contiguous(), wts.to(torch.float32).contiguous()
    if k == "w":
        return _hyb_call(x, ids, wts)
    mod = _HYBLO["mods"][k]
    out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYBLO_PDL else 0
    if _HYBLO["pdl"].get(k) != pdl:
        mod.okc_set_pdl(pdl)
        _HYBLO["pdl"][k] = pdl
    mod.okc_run(_HYBLO["w"], _HYBLO["sc"], x, ids, wts, out, 0, 32, _HYB["nsm"])
    return out
def _hyblo_route(x, ids, wts):
    m = x.shape[0]
    k = "w" if HYBLO_K == "w" else _hyblo_key(m)
    if not (_HYBLO["wok"] if k == "w" else _HYBLO["ok"].get(k)):
        return None
    _HYBLO["calls"][(k, m)] = _HYBLO["calls"].get((k, m), 0) + 1
    return _hyblo_run(k, x, ids, wts)
def _hyblo_inputs(bs, seed, dev, dist):
    d = _k2a_self_test_inputs(bs, seed, dev, "normal" if dist == "conc" else dist)
    if dist == "conc":   # every token on local experts 0..7: 8 experts x bs rows (MID splits, HYB gets fat jobs at bs > 8)
        d["topk_ids"] = torch.arange(8, dtype=torch.int32, device=dev).repeat(bs, 1).contiguous()
    elif dist == "none":   # no local route at all (nj = 0)
        d["topk_ids"] = torch.full((bs, 8), -1, dtype=torch.int32, device=dev)
    return d
def _hyblo_prepare(st, w, dev):
    _HYBLO["ok"], _HYBLO["wok"] = {}, False
    if not HYBLO_ENABLE:
        return
    t1 = time.time()
    try:
        _pt1_wait("hyblo")
        if not _HYBLO["mods"] and not _HYBLO["err"]:
            _hyblo_build()
        if not _HYB["ok"]:
            print("[hyblo] OFF: HYB is off (its WGS copies / PRMT check are shared)", flush=True)
            return
        I = H = 4096
        rb, g = torch.arange(I // 8, device=dev), torch.arange(8, device=dev)
        rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
        rowsB = torch.arange(H, device=dev)
        wA, sA, wB, sB = [], [], [], []
        for e in range(32):
            wA.append(_hyblo_pack_codes(w["w13"][e].view(torch.uint8), rowsA))
            sA.append(_hyblo_pack_scales(w["w13_sf"][e], rowsA))
            wB.append(_hyblo_pack_codes(w["w2"][e].view(torch.uint8), rowsB))
            sB.append(_hyblo_pack_scales(w["w2_sf"][e], rowsB))
        hw = _HYB["w"]
        _HYBLO["w"] = [torch.cat(wA), torch.cat(sA), hw[2], torch.cat(wB), torch.cat(sB), hw[5], hw[6], hw[7],
                       torch.cat([_hyblo_fat_order(a, b) for a, b in zip(wA, sA)]),
                       torch.cat([_hyblo_fat_order(a, b) for a, b in zip(wB, sB)]),
                       torch.cat([_hyblo_unit_order(a, b) for a, b in zip(wA, sA)]),
                       torch.cat([_hyblo_unit_order(a, b) for a, b in zip(wB, sB)]), hw[12], hw[13]]
        del wA, sA, wB, sB
        _HYBLO["sc"], _HYBLO["full"] = _hyb_scratch(dev)
        torch.cuda.synchronize()
        res = {}
        for bs, seed, dist in HYBLO_SELF_TEST:
            d = _hyblo_inputs(bs, seed, dev, dist)
            x, ids, wts = d["hidden_states"].contiguous(), d["topk_ids"], d["topk_weights"]
            ref = _k2a_ref_fp32(d, w)
            o_v = _v11_k2a33_call(x, ids, wts, st).clone()
            for k in (_hyblo_key(bs), "w"):
                if k != "w" and k not in _HYBLO["mods"]:
                    res[k] = False
                    continue
                prev = None
                for c in range(2):
                    o = _hyblo_run(k, x, ids, wts)
                    torch.cuda.synchronize()
                    ev, tot, row = _k2a_rel_err(o, o_v)[0], *_k2a_rel_err(o, ref)
                    rep = 0.0 if prev is None else _k2a_rel_err(o, prev)[0]
                    ok = tot < 1e-2 and row < 5e-2 and ev < 1e-2 and rep < 1e-3 and bool(torch.isfinite(o.float()).all())
                    print(f"[hyblo] self-test {k} bs={bs} {dist} call {c}: vs ship {ev:.2e} vs FP32 {tot:.3e} row {row:.3e} "
                          f"repeat {rep:.1e} -> {'ok' if ok else 'FAIL'}", flush=True)
                    res[k] = res.get(k, True) and ok
                    prev = o.clone()
        gok = _hyb_guards_ok() and all(bool((f[n:].view(vt) == sv).all()) for f, n, vt, sv in _HYBLO["full"])
        _HYBLO["ok"] = {k: bool(v and gok) for k, v in res.items() if k != "w"}
        _HYBLO["wok"] = bool(res.get("w") and gok)
        print(f"[hyblo] ok {_HYBLO['ok']} wok {_HYBLO['wok']} guards {gok} nvcc {_HYBLO['secs']} err {_HYBLO['err'] or 'none'} "
              f"prepare {time.time() - t1:.1f}s", flush=True)
    except Exception as ex:
        _HYBLO["ok"], _HYBLO["wok"] = {}, False
        print(f"[hyblo] prepare raised {ex!r}: lab arms off", flush=True)
'''
ROUTE_A = ('    data = {"hidden_states": x, "topk_ids": local_ids, "topk_weights": topk_weights, "config": {"local_start": 0}}\n'
           '    if VMS_ENABLE and st.get("vms_ok") and x.shape[0] <= VMS_MAXT:\n')
ROUTE_N = ('    data = {"hidden_states": x, "topk_ids": local_ids, "topk_weights": topk_weights, "config": {"local_start": 0}}\n'
           '    if HYBLO_T and x.shape[0] in HYBLO_T:\n'
           '        o = _hyblo_route(x, local_ids, topk_weights)\n'
           '        if o is not None:\n'
           '            return o\n'
           '    if VMS_ENABLE and st.get("vms_ok") and x.shape[0] <= VMS_MAXT:\n')
PREP_A = '    _hyb_prepare(st, w, dev)\n    print(f"[v8] K2A v3.3 buckets'
PREP_N = '    _hyb_prepare(st, w, dev)\n    _hyblo_prepare(st, w, dev)\n    print(f"[v8] K2A v3.3 buckets'
PT1_A = '    if HYB_ENABLE:\n        jobs.append(("hyb", _hyb_build))\n'
PT1_N = PT1_A + '    if HYBLO_ENABLE:\n        jobs.append(("hyblo", _hyblo_build))\n'
CALL_A = "def _v11_k2a33_call(x, local_ids, topk_weights, st):\n"


SHIP_TESTS = {"tiny": ((1, 81, "normal"), (1, 82, "skewed"), (2, 83, "normal"), (4, 84, "skewed"), (1, 94, "none"), (4, 95, "none")),
              "tiny2": ((5, 85, "normal"), (8, 86, "skewed"), (8, 87, "normal"), (8, 92, "conc"), (8, 96, "none")),
              "mid": ((9, 88, "skewed"), (12, 89, "normal"), (16, 90, "skewed"), (16, 91, "normal"), (16, 93, "conc"), (16, 97, "none"))}


def ship_builds(ts):
    need = []
    for t in ts:
        k = next((b for b in BUILDS if t <= b[1]), None)
        check(k is not None and t >= 1, f"--ship T {t} outside 1..16")
        if k not in need:
            need.append(k)
    return tuple(b for b in BUILDS if b in need)


def build_lab(h0s, fns, ship=None):
    builds = ship_builds(ship) if ship else BUILDS
    keys = tuple((k, mx, tuple(f"-D{n}={v}" for n, v in zip(VAR, vals))) for k, mx, vals in builds)
    fl = HEAD + tuple(f"-D{k}={v}" for k, v in COMMON.items())
    t = (LAB.replace("@CUDA@", h0s).replace("@KEYS@", repr(keys)).replace("@FLAGS@", repr(fl))
         .replace("@HELPERS@", helpers(fns)))
    if not ship:
        return t
    tests = tuple(c for k, _, _ in builds for c in SHIP_TESTS[k])
    for a, n in (("HYBLO_T = ()\n", f"HYBLO_T = {tuple(ship)!r}\n"),
                 ("            for k in (_hyblo_key(bs), \"w\"):\n", "            for k in (_hyblo_key(bs),):\n"),
                 ("(2026-10-31, lab only)", "(2026-10-31)"),
                 ("            o_v = _v11_k2a33_call(x, ids, wts, st).clone()\n",
                  "            o_v = ref if dist == \"none\" else _v11_k2a33_call(x, ids, wts, st).clone()\n"),
                 (" and the in-file HYB module as arms ahead of VMS:\nHYBLO_T = routed token counts, HYBLO_K = \"h\" (their build for that M) / \"w\" (HYB). Defaults: ship dispatch.",
                  ", routed ahead of VMS at\nHYBLO_T (R-hyblo-1: -1.9..-2.3 us in-layer at d1s / d1f); HYBLO_T = () restores VMS.")):
        check(t.count(a) == 1, f"ship edit anchor {a[:50]!r}")
        t = t.replace(a, n)
    i0 = t.index("HYBLO_SELF_TEST = (")
    i1 = t.index("\n_HYBLO = {", i0)
    return t[:i0] + f"HYBLO_SELF_TEST = {tests!r}" + t[i1:]


def apply(s, ship=None, base_name="k3_hh_3tekv"):
    h0, fns = upstream()
    h0s = strip_h0(h0)
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    if ship:
        lines[1] = (f"# nonce {base_name.replace('_', '-')}-hyblo (+ alygg77 helper0 small-M MoE build at T {list(ship)} ahead of VMS; "
                    f"tools/hyblo_patches.py --ship={','.join(map(str, ship))})")
    else:
        lines[1] = "# nonce k3-hyblo-ab (k3_hh_3tekv + R-hyblo lab arms: alygg77 helper0 TINY/TINY2/MID + HYB at T <= 16; tools/hyblo_patches.py)"
    s = NL.join(lines)
    s = sub(s, CALL_A, build_lab(h0s, fns, ship) + CALL_A, "lab block")
    s = sub(s, ROUTE_A, ROUTE_N, "route hook")
    s = sub(s, PREP_A, PREP_N, "prepare call")
    s = sub(s, PT1_A, PT1_N, "PT1 job")
    return s


def gates(path, s):
    import pyslim
    n = len(s.encode())
    check(n < LIMIT, f"size {n} B >= {LIMIT}")
    check(s.isascii(), "non-ASCII byte")
    ast.parse(s)
    d = pyslim.decorated_sources(s)
    bad = [x for x in d if x.startswith("UNPARSABLE")]
    check(not bad, f"decorated defs that do not re-parse: {bad[:2]}")
    for a in ("def _hyblo_prepare(", 'jobs.append(("hyblo", _hyblo_build))', "    _hyblo_prepare(st, w, dev)\n",
              "        o = _hyblo_route(x, local_ids, topk_weights)\n", "THIRD_PARTY_NOTICES += _HYBLO_NOTICE\n"):
        check(s.count(a) == 1, f"missing {a!r}")
    print(f"hyblo_patches: gates OK on {path.name}: {n:,} B ({LIMIT - n:,} B under the cap), {len(d)} decorated defs re-parse")


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3tekv.py")
    ship = tuple(int(x) for x in ARGS["ship"].split(",")) if ARGS.get("ship") else None
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hyblo_ab.py" if not ship else f"kernels/lab/{base.stem}_hyblo.py")
    new = apply(base.read_bytes().decode("ascii"), ship, base.stem)
    if "check" in FLAGS:
        check(out.exists(), f"{out} missing")
        check(out.read_bytes().decode("ascii") == new, f"{out.name} differs from apply({base.name})")
        gates(out, new)
        print(f"hyblo_patches: --check OK ({out.name} == apply({base.name}))")
        return
    out.write_bytes(new.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"hyblo_patches: {base.name} -> {out.name} ({len(new):,} B, +{len(new) - base.stat().st_size:,} B)")
    gates(out, new)


if __name__ == "__main__":
    main()

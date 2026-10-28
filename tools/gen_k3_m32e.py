"""gen_k3_m32e.py: k3_m32e_<n> = a layer file (default k3_vfa_7) with its VML block patched by tools/m32e_patches.py (VM_DP
dynamic A/B tail pools as compile-time variant "d" / "d2"; every patch is behind a switch, so the shipped variant 's'
compiles to the same code) and per-bucket variant routing:
  VML_BVAR  {bucket: variant} read at call time (e.g. {256: "d"}); a pair that failed its prepare self-test runs VML_VAR;
  VML_POOL  {bucket: (A pool, B pool)} in 256ths of each phase's units, for DP variants; VML_DPFLAGS (bit 13: jobs sorted
            hot-first) is OR'ed into VML_FLAGS for DP variants.
Prepare self-tests every (bucket, BVAR variant) pair against the base variant (< 2e-3) and FP32 (< 1e-2, row < 5e-2), two
calls, error flag, finite (_m32e_vartest). Base must be a finish() fixed point; output goes through gen_k3_m32.finish.
Refuses to overwrite.

    python -I tools/gen_k3_m32e.py --n=1 --lab-only --bvar=256:d,128:d,32:d --pool=256:32:32,128:32:32,32:32:32
    python -I tools/gen_k3_m32e.py --n=2 --bvar=256:d --pool=256:32:32      # kernels/muse_decode_layer/k3_m32e_2.py + box copy
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
LAB_ONLY = "--lab-only" in sys.argv
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32 as GM  # noqa: E402
sys.argv = _argv
import m32e_patches as P  # noqa: E402

N = ARGS.get("n", "1")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_vfa_7')}.py"
OUTN = ARGS.get("out", f"k3_m32e_{N}")   # --out=k3_gc_2: a ship name (its tag k3-gc-2)
OUT = ROOT / f"kernels/muse_decode_layer/{OUTN}.py"
LAB = ROOT / f"kernels/lab/{ARGS.get('lab', OUTN + '_ab')}.py"
TAG = OUTN.replace("_", "-")
NL = chr(10)
BVAR = {int(k): v for k, v in (kv.split(":") for kv in ARGS.get("bvar", "").split(",") if kv)}
POOL = {int(b): (int(a), int(c)) for b, a, c in (kv.split(":") for kv in ARGS.get("pool", "").split(",") if kv)}
SEGC = {int(b): int(v) for b, v in (kv.split(":") for kv in ARGS.get("seg", "").split(",") if kv)}
SCF = 16384 if "--snap" in sys.argv else 0
ABV = tuple(a for a in ARGS.get("abvars", "").split(",") if a)
PDEL = {int(b): int(v) for b, v in (kv.split(":") for kv in ARGS.get("pdel", "").split(",") if kv)}
VAR = ARGS.get("var", "s")

GLUE = [
    ('_VML_DEFS = {"": (), "o": ("-DVM_TRICKS=0",), "s": ("-DVM_TRICKS_DQ=0",)}\n',
     '_VML_DEFS = {"": (), "o": ("-DVM_TRICKS=0",), "s": ("-DVM_TRICKS_DQ=0",), "d": ("-DVM_TRICKS_DQ=0", "-DVM_DP=1"),\n'
     '             "d2": ("-DVM_TRICKS_DQ=0", "-DVM_DP=1", "-DVM_DPI=2"), "c": ("-DVM_TRICKS_DQ=0", "-DVM_SC=1"),\n'
     '             "b": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1"), "bd": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_DEFER2=1"),\n'
     '             "e": ("-DVM_TRICKS_DQ=0", "-DVM_DEFER2=1"), "p": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_PS=1"),\n'
     '             "bf": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1"), "bg": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_GSM=1", "-DVM_FSW=1"),\n'
     '             "bf2": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1"),\n'
     '             "bu": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", "-DVM_UD2=3", "-DVM_UD=1"),\n'
     '             "bv": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", "-DVM_UD2=4", "-DVM_UD=1")}\n'),
    ("VML_ABVARS = ()\n",
     "VML_ABVARS = @ABV@\nVML_BVAR = @BVAR@\nVML_POOL = @POOL@\nVML_DPFLAGS = 8192\nVML_SEG = @SEG@\nVML_SCFLAGS = @SCF@\n"
     "VML_PDEL = @PDEL@\n"),
    ('_VML = {"mod": None, "mods": {}, "sc": None, "ok": {}, "err": None}\n',
     '_VML = {"mod": None, "mods": {}, "sc": None, "scb": None, "ok": {}, "err": None, "okv": {}, "force": None}\n'),
    ('        _VML["sc"] = _vml_alloc(dev, sms)\n',
     '        _VML["sc"] = _vml_alloc(dev, sms)\n        _VML["scb"] = dict(_VML["sc"], y=torch.zeros_like(_VML["sc"]["y"]))\n'),
    ("    for v in (VML_VAR,) + tuple(a for a in VML_ABVARS if a != VML_VAR):\n",
     "    for v in (VML_VAR,) + tuple(dict.fromkeys(a for a in tuple(VML_ABVARS) + tuple(VML_BVAR.values()) if a != VML_VAR)):\n"),
    ('''    rep = _k2a_bucket_of(x.shape[0])
    return _vml_call(_VML["mods"].get(VML_VAR, _VML["mod"]), st, _VML["sc"], x, data["topk_ids"].to(torch.int32).contiguous(),
                     data["topk_weights"].to(torch.float32).contiguous(), int(data["config"]["local_start"]),
                     flags=VML_FLAGS.get(rep), cost=VML_COST.get(rep))
''',
     '''    rep = _k2a_bucket_of(x.shape[0])
    var = _VML["force"] if _VML["force"] is not None else VML_BVAR.get(rep, VML_VAR)
    if _VML["force"] is None and var != VML_VAR and not _VML["okv"].get((rep, var)):
        var = VML_VAR
    dp, sc = var in ("d", "d2"), var == "c"
    fl = VML_FLAGS.get(rep)
    fl = (_vml_FLAGS_DEFAULT if fl is None else fl) | (VML_DPFLAGS if dp else 0) | (VML_SCFLAGS if sc else 0)
    sg = int(VML_SEG.get(rep, 0)) if sc else 0
    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "bu", "bv") else _VML["sc"]
    pdl = int(VML_PDEL.get(rep, 0)) if var == "p" else 0
    return _vml_call(_VML["mods"].get(var, _VML["mod"]), st, scx, x, data["topk_ids"].to(torch.int32).contiguous(),
                     data["topk_weights"].to(torch.float32).contiguous(), int(data["config"]["local_start"]),
                     flags=fl, cost=VML_COST.get(rep), pool=tuple(VML_POOL.get(rep, (0, 0))) if dp else (0, 0), seg=(sg, sg), pdel=pdl)
'''),
    ('''    try:
        _pt1_wait("vml")
        if _VML["mod"] is None:
            _vml_build_job()
''',
     '''    try:
        _pt1_wait("vml")
        _VML["force"] = VML_VAR
        if _VML["mod"] is None:
            _vml_build_job()
'''),
    ('''        print(f"[vml] VML1 live buckets {[r for r, v in _VML['ok'].items() if v]} (self-test {_VML['ok']})", flush=True)
    except Exception as ex:
        _VML["ok"] = {}
''',
     '''        print(f"[vml] VML1 live buckets {[r for r, v in _VML['ok'].items() if v]} (self-test {_VML['ok']})", flush=True)
        _VML["force"] = None
        _m32e_vartest(st, w, dev)
    except Exception as ex:
        _VML["force"] = None
        _VML["ok"] = {}
'''),
    ("\nK2A36_ENABLE = True\n", '''

def _m32e_vartest(st, w, dev):
    """self-test of every (bucket, VML_BVAR variant) pair vs the base variant and FP32 (m32e); failures fall back to VML_VAR."""
    _VML["okv"] = {}
    cases = {128: ((100, 41, "normal"), (128, 42, "skewed"), (160, 43, "skewed")),
             256: ((200, 44, "skewed"), (256, 45, "normal"), (257, 46, "skewed"), (320, 47, "normal")),
             32: ((25, 48, "normal"), (32, 49, "skewed"), (48, 50, "skewed"))}
    pairs = [(rep, v) for rep in VML_BVAR for v in dict.fromkeys((VML_BVAR[rep],) + tuple(VML_ABVARS))]
    for rep, var in pairs:
        if var == VML_VAR or not _VML["ok"].get(rep) or var not in _VML["mods"]:
            continue
        good = True
        try:
            for bs, seed, dist in cases[rep]:
                d = _k2a_self_test_inputs(bs, seed, dev, dist)
                d = {"hidden_states": d["hidden_states"].contiguous(), "topk_ids": d["topk_ids"].to(torch.int32).contiguous(),
                     "topk_weights": d["topk_weights"].float().contiguous(), "config": {"local_start": 0, "local_count": K2A_E_LOC}}
                ref = _k2a_ref_fp32(d, w)
                _VML["force"] = VML_VAR
                o_b = _vml_route(st, d).clone()
                for k in range(2):
                    _VML["sc"]["err"].zero_()
                    _VML["force"] = var
                    o = _vml_route(st, d)
                    _VML["force"] = None
                    torch.cuda.synchronize()
                    eb, _ = _k2a_rel_err(o, o_b)
                    tot, row = _k2a_rel_err(o, ref)
                    errf = int(_VML["sc"]["err"][0])
                    ok = eb < 2e-3 and tot < 1e-2 and row < 5e-2 and not errf and bool(torch.isfinite(o.float()).all())
                    print(f"[m32e] self-test bucket {rep} var {var} bs={bs} {dist} call {k}: rel vs {VML_VAR} {eb:.2e} vs FP32 "
                          f"{tot:.3e} worst row {row:.3e} err={errf} -> {'ok' if ok else 'FAIL'}", flush=True)
                    good &= ok
        except Exception as ex:
            good = False
            print(f"[m32e] self-test bucket {rep} var {var} raised {ex!r}", flush=True)
        finally:
            _VML["force"] = None
        _VML["okv"][(rep, var)] = good
    print(f"[m32e] DP variants live {[k for k, v in _VML['okv'].items() if v]} (self-test {_VML['okv']})", flush=True)

K2A36_ENABLE = True
'''),
]


def main():
    GM.check("--dry" in sys.argv or (not LAB.exists() and (LAB_ONLY or not OUT.exists())),
             f"{OUT.name} or its lab copy exists; pick another --n")
    GM.check(all(v in ("d", "d2", "s", "c", "b", "bd", "e", "p", "bf", "bg", "bf2", "bu", "bv") for v in BVAR.values()) and set(POOL) <= set(BVAR) and set(SEGC) <= set(BVAR),
             "bad --bvar / --pool / --seg")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    t = b
    a0, b0 = P.cuda_span(t, "_vml_CUDA_T")
    t = t[:a0] + P.patch_cuda(t[a0:b0]) + t[b0:]
    rep.append(f"VML CUDA: {len(P.CUDA_PATCHES)} anchored m32e patches (wide / WG5 / DP switches, defaults unchanged)")
    t = P.patch_py(t, "_vml_")
    t = P.sub1(t, "\n\n_vml_FLAGS_DEFAULT = 1", P.COST_EXT.replace("@P@", "_vml_") + "\n\n_vml_FLAGS_DEFAULT = 1", "layer cost_ext")
    for i, (old, new) in enumerate(GLUE):
        t = P.sub1(t, old, new.replace("@BVAR@", repr(BVAR)).replace("@POOL@", repr(POOL)).replace("@SEG@", repr(SEGC))
                   .replace("@SCF@", str(SCF)).replace("@ABV@", repr(ABV)).replace("@PDEL@", repr(PDEL)), f"glue {i}")
    t = P.sub1(t, "VML_VAR = 's'\n", f"VML_VAR = {VAR!r}\n", "VML_VAR")
    rep.append(f"glue: VML_VAR {VAR!r}, VML_BVAR {BVAR}, VML_POOL {POOL}, per-pair self-test (_m32e_vartest)")
    nonce = (f"# nonce {TAG} ({BASE.stem} + m32e VML: VML_VAR {VAR!r} (b = VM_BRED B split shares red.add into y), VML_BVAR {BVAR}, "
             f"VML_ABVARS {ABV}, VML_POOL {POOL}, VML_SEG {SEGC}, snap "
             f"{bool(SCF)}; tools/gen_k3_m32e.py)")
    t = t.replace(bn, nonce)
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_m32e] OK  {r}")
    if "--dry" in sys.argv:
        print(f"[gen_k3_m32e] --dry: nothing written ({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B)")
        return 0
    if not LAB_ONLY:
        OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_m32e] wrote {'' if LAB_ONLY else str(OUT.relative_to(ROOT)) + ' and '}{LAB.relative_to(ROOT)} "
          f"({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

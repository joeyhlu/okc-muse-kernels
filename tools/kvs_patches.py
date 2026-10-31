"""kvs_patches.py (R-kvs-2): kv_scale != 1 at q_len 1 takes HYA (INDEPENDENT_REVIEW_TOP30 s7 exposure 3; notes/study/robustness_hh3_exposures.md s2).

  python -I tools/kvs_patches.py [--base=kernels/lab/k3_hh_3.py] [--out=kernels/lab/k3_hh_3k.py] [--check]
  python -I tools/lead_gate.py file kernels/lab/k3_hh_3k.py

Why: k3_hh_3 sends every kv_scale != 1 call to SGLang FA3 (r1). R-kvs-1 measured that +14..18 % (decode sliding B <= 16) and
+55..71 % (decode full B 16 / 32 / 129) slower than k3_b29_2's K1 / VA1. HYA's kernel already applies k_scale / v_scale (scores
x k_scale, output x v_scale / l, new K / V rows quantised by 1 / scale). q_len 8 stays on FA3 (A33: alygg77 measured the q_len 8
query path at 1.41x stock error on the 1/64 test), so the route is q_len 1 only.

Edits (each anchor must occur exactly once; every edit is a plain-text replacement of code lines, no comment anchors):
  1. nonce line (line 2).
  2. constants HYA_KVS / HYA_KVS_CASES / HYA_KVS_TOL after HYA_GRAPH_CASES, and "kvs": False in the _HYA state dict.
  3. _hya_wanted: the clause cfg["kv_scale"] == 1.0 becomes (cfg["kv_scale"] == 1.0 or (L == 1 and _HYA["kvs"])).
  4. _hya_self_test: after the graph checks, a kv_scale = 1/64 self-test (HYA_KVS_CASES: B16 full, B1 full, B4 sliding; q_len 1).
     The cache bytes are the unscaled case x 64 (a power of two: exact in e4m3), so the unscaled FP32 reference is the exact
     reference of the scaled case; HYA must stay within HYA_KVS_TOL (2e-2; unscaled HYA runs 5e-3) and leave its workspace clean.
     It is NOT strict: a failure (or any exception) sets _HYA["kvs"] False, prints, and kv_scale != 1 stays on FA3 while HYA stays
     live for kv_scale == 1.
  5. _hya_prepare LIVE line prints the kv_scale route state.
--check: no write; verifies that --out equals apply(--base) byte for byte, and the $0 gates on it (AST parse, ASCII, < 1,000,000 B,
every decorated def re-parses from its own source text via tools/pyslim.py, every inserted anchor present once).
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[kvs_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


NONCE = "# nonce k3-hh-3k (k3_hh_3 + R-kvs-2: kv_scale != 1 at q_len 1 takes HYA after a prepare self-test; tools/kvs_patches.py)"

A_STATE = '_HYA = {"mod": None, "ws": {}, "inv": {}, "sms": {}, "build_s": 0.0, "seen": set(), "mk": False}\n'
N_STATE = '_HYA = {"mod": None, "ws": {}, "inv": {}, "sms": {}, "build_s": 0.0, "seen": set(), "mk": False, "kvs": False}\n'

A_CONST = "HYA_GRAPH_CASES = ((4, 6000, 8, 2048, 1121), (1, 3000, 1, 0, 1122), (16, 1500, 1, 0, 1123))\n"
N_CONST = (A_CONST
           + "HYA_KVS = True\n"
           + "HYA_KVS_CASES = ((16, 1500, 1, 0, 1131), (1, 3000, 1, 0, 1132), (4, 2500, 1, 2048, 1133))\n"
           + "HYA_KVS_TOL = 2e-2\n")

A_WANT = 'and P > 0 and not (P & (P - 1)) and cfg["kv_scale"] == 1.0\n'
N_WANT = 'and P > 0 and not (P & (P - 1)) and (cfg["kv_scale"] == 1.0 or (L == 1 and _HYA["kvs"]))\n'

A_END = ('        del gr, og, o1, d\n'
         '    return worst\n'
         'def _hya_prepare(state, t0):\n')
N_END = '''        del gr, og, o1, d
    _hya_kvs_self_test(dev)
    return worst
def _hya_kvs_self_test(dev):
    _HYA["kvs"] = False
    if not HYA_KVS:
        return
    good = True
    try:
        for B, ctx, L, win, seed in HYA_KVS_CASES:
            ref = _k1fa4_reference(_k1fa4_case(B, ctx, L, win, seed, False, dev)).view(B * L, -1)
            dk = _k1fa4_case(B, ctx, L, win, seed, False, dev)
            for c in ("k_cache", "v_cache"):
                dk[c] = (dk[c].float() * 64.0).to(FP8)
            dk["k_scale"] = torch.tensor(1.0 / 64, dtype=torch.float32, device=dev)
            dk["v_scale"] = torch.tensor(1.0 / 64, dtype=torch.float32, device=dev)
            cfg = dict(dk["config"], tokens=B * L, kv_scale=1.0 / 64)
            oa = _hya_launch(dk["qkv"].view(B * L, -1).contiguous(), dk, cfg, True).float().view(B * L, -1)
            torch.cuda.synchronize(dev)
            ws = _HYA["ws"][str(dev)]
            clean = not (float(ws[0].abs().sum()) or float(ws[1].abs().sum()))
            ea = _hya_rel(oa, ref)
            ok = bool(torch.isfinite(oa).all()) and ea <= HYA_KVS_TOL and clean
            print(f"[hya] kv_scale 1/64 self-test B{B} ctx{ctx} L{L} w{win}: {'PASS' if ok else 'FAIL'} err {ea:.3e} "
                  f"(tol {HYA_KVS_TOL:.0e}) ws {'clean' if clean else 'DIRTY'}", flush=True)
            good &= ok
            del ref, dk, oa
    except Exception as ex:
        good = False
        print(f"[hya] kv_scale 1/64 self-test raised {ex!r}"[:600], flush=True)
    _HYA["kvs"] = good
    if not good:
        print("[hya] kv_scale != 1 stays on the r1 path (SGLang FA3); HYA stays live for kv_scale == 1", flush=True)
        for t in _HYA["ws"].get(str(dev), ()):
            t.zero_()
def _hya_prepare(state, t0):
'''

A_LIVE = ('        print(f"[hya] LIVE: route {sorted(k for k, v in HYA_ROUTE.items() if v)} + fallback {HYA_DEFAULT}, worst err {worst:.3e}, "\n')
N_LIVE = ('        print(f"[hya] LIVE: route {sorted(k for k, v in HYA_ROUTE.items() if v)} + fallback {HYA_DEFAULT}, worst err {worst:.3e}, "\n'
          '              f"kv_scale != 1 q_len 1 {\'ON\' if _HYA[\'kvs\'] else \'OFF (FA3)\'}, "\n')


def apply(s):
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] = NONCE
    s = NL.join(lines)
    s = sub(s, A_STATE, N_STATE, "_HYA state dict")
    s = sub(s, A_CONST, N_CONST, "HYA_GRAPH_CASES")
    s = sub(s, A_WANT, N_WANT, "_hya_wanted kv_scale clause")
    s = sub(s, A_END, N_END, "end of _hya_self_test")
    s = sub(s, A_LIVE, N_LIVE, "_hya_prepare LIVE print")
    return s


def gates(path, s):
    import pyslim
    n = len(s.encode())
    check(n < LIMIT, f"size {n} B >= {LIMIT}")
    check(s.isascii(), "non-ASCII byte")
    ast.parse(s)
    d = pyslim.decorated_sources(s)
    bad = [x for x in d if x.startswith("UNPARSABLE")]
    check(not bad, f"decorated defs that do not re-parse from their own text: {bad[:2]}")
    for a in ("_hya_kvs_self_test", '(L == 1 and _HYA["kvs"])', "HYA_KVS_CASES = ", '"mk": False, "kvs": False}'):
        check(s.count(a) >= 1, f"missing {a!r}")
    check(s.count("def _hya_kvs_self_test(dev):") == 1, "self-test defined once")
    print(f"kvs_patches: gates OK on {path.name}: {n:,} B ({LIMIT - n:,} B under the cap), {len(d)} decorated defs re-parse")


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hh_3k.py")
    new = apply(base.read_bytes().decode("ascii"))
    if "check" in FLAGS:
        check(out.exists(), f"{out} missing")
        check(out.read_bytes().decode("ascii") == new, f"{out.name} differs from apply({base.name})")
        gates(out, new)
        print(f"kvs_patches: --check OK ({out.name} == apply({base.name}))")
        return
    ast.parse(new)
    out.write_bytes(new.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"kvs_patches: {base.name} -> {out.name} ({len(new):,} B, +{len(new) - base.stat().st_size:,} B)")
    gates(out, new)


if __name__ == "__main__":
    main()

"""hyb193_patches.py (R-hyb193): HYB st7 (tools/hybst_patches.py bits 1 | 2 | 4) + hn (tools/hybk_patches.py arm hn: a hot expert's
non-final weight reads EVICT_NORMAL) as a SECOND HYB module used only at T >= HYB_T7 = 193 (v32); T < 193 keeps the ship module.

  python -I tools/hyb193_patches.py --base=<in.py> --out=<out.py> [--t7=193]
  python -I tools/lead_gate.py file <out.py>

R-rb-2: st7 + hn per rider v32s -1.57 / v32f -1.31 us but v16s +2.64 -> gate by T. The CUDA edits are hybst's (imported) plus hn
rewritten behind -DOKC_HN (hybk hard-wires it, which is why a plain hybst + hybk st0 build is not ship-identical): with OKC_HYBST = 0
and OKC_HN = 0 the HYB text preprocesses to the ship text, so _HYB["mod"] (HYB_FLAGS) is the ship SASS; _HYB["mod7"] = HYB_FLAGS +
HYB_F7. Both build in parallel inside _hyb_build; mod7 is checked against the ship module on every self-test case with T >= HYB_T7
(rel <= max(2 x ship repeat, 1e-6), finite) and dropped on any failure (T >= 193 then stays on the ship module). Arithmetic unchanged.
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)
LIMIT = 1_000_000
_argv, sys.argv = sys.argv, sys.argv[:1]
import hybst_patches as HS  # noqa: E402  (CUDA edit constants only)
sys.argv = _argv


def check(c, msg):
    if not c:
        raise SystemExit(f"[hyb193_patches] FAIL: {msg}")


HN0_A = "__device__ void wgs_ranges(const Params& p, Smem& S);\n"
HN0_N = r'''#ifndef OKC_HN
#define OKC_HN 0
#endif
#if OKC_HN
__device__ __forceinline__ unsigned long long hk_pol(const Smem& S, int j) {
const int e = S.jE[j], tot = S.cnt[e];
const int n = tot > SK_T ? (tot + FT_T - 1) / FT_T : 1;
const int q = n > 1 ? (S.jS[j] - S.off[e]) / FT_T : 0;
return q < n - 1 ? 0x1000000000000000ull : 0x12F0000000000000ull;
}
#define HN_ARG , "l"(hkpol)
#else
#define HN_ARG L2H_ARG
#endif
''' + HN0_A
HN1_A = "const int u = gs >> 4, st = gs & 15;\nconst int j = u >> ubl, ub = u & ((1 << ubl) - 1);\n"
HN1_N = HN1_A + "#if OKC_HN\nconst unsigned long long hkpol = hk_pol(S, j);\n#endif\n"
HN2_A = '::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n'
HN2_N = HN2_A.replace(" L2H_ARG ", " HN_ARG ")

NOTE_A = 'helpers.\n"""\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n'
NOTE_N = ('helpers.\nModified by team joeyhlu (2026-10-31, tools/hyb193_patches.py, -DOKC_HYBST / -DOKC_HN, second build at T >= 193 only):\n'
          'unit-end epilogue constants prefetched and computed once per unit, the phase-end split read-back issues all n-tiles\n'
          'at once, the final pass reuses its first-pass loads, and a hot expert\'s non-final weight reads use EVICT_NORMAL;\n'
          'HYB\'s arithmetic is unchanged.\n"""\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n')
STATE_A = ('_HYB = {"mod": None, "ok": False, "w": None, "sc": None, "full": None, "nsm": 0, "err": None, "build_s": 0.0, '
           '"pdl": -1}\n')
STATE_N = STATE_A[:-2] + ', "mod7": None, "pdl7": -1}\n'
BUILD_N = '''def _hyb_build():
    from torch.utils.cpp_extension import load_inline
    def one(fl):
        tag = hashlib.sha1((_HYB_MOE_CPP + _HYB_MOE_CUDA + " ".join(fl)).encode()).hexdigest()[:12]
        return load_inline(name=f"hyb_moe_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=_HYB_MOE_CUDA,
                           functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
    def b7():
        try:
            _HYB["mod7"] = one(list(HYB_FLAGS) + list(HYB_F7))
        except Exception as ex:
            _HYB["mod7"] = None
            print(f"[hyb] T>={HYB_T7} build failed {ex!r}"[:300], flush=True)
    t0 = time.time()
    th = threading.Thread(target=b7, name="hyb7", daemon=True)
    th.start()
    _HYB["mod"] = one(list(HYB_FLAGS))
    th.join()
    _HYB["build_s"] = time.time() - t0
    return _HYB["mod"]
'''
CALL_N = '''    md, k = (_HYB["mod7"], "pdl7") if (m >= HYB_T7 and _HYB["mod7"] is not None) else (_HYB["mod"], "pdl")
    if _HYB[k] != pdl:
        md.okc_set_pdl(pdl)
        _HYB[k] = pdl
    md.okc_run(_HYB["w"], _HYB["sc"], x, ids, wts, out, 0, 32, _HYB["nsm"])
'''
LOOP_A = "        good = True\n        for bs, seed, dist in HYB_SELF_TEST:\n"
LOOP_N = '        h7, _HYB["mod7"] = _HYB["mod7"], None\n' + LOOP_A
ST_A = "                prev = o.clone()\n        gok = _hyb_guards_ok()\n"
ST_N = '''                prev = o.clone()
            if h7 is not None and bs >= HYB_T7:
                try:
                    _HYB["mod7"] = h7
                    o7 = _hyb_call(d["hidden_states"], d["topk_ids"], d["topk_weights"])
                    torch.cuda.synchronize()
                    e7 = _k2a_rel_err(o7, prev)[0]
                    ok7 = bool(torch.isfinite(o7.float()).all()) and e7 <= max(2.0 * rep_d, 1e-6)
                except Exception as ex:
                    ok7, e7 = False, float("nan")
                    print(f"[hyb] T>={HYB_T7} build raised {ex!r}"[:300], flush=True)
                _HYB["mod7"] = None
                print(f"[hyb] T>={HYB_T7} build bs={bs}: rel vs ship {e7:.1e} (repeat {rep_d:.1e}) -> {'ok' if ok7 else 'FAIL: dropped'}", flush=True)
                if not ok7:
                    h7 = None
        _HYB["mod7"] = h7
        gok = _hyb_guards_ok()
'''


def apply(s, t7):
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] += f" + R-hyb193 HYB st7+hn build at T>={t7} [tools/hyb193_patches.py]"
    s = NL.join(lines)
    i0, i1 = s.index('_HYB_MOE_CUDA = r"""'), s.index('_HYB_MOE_CPP = r"""')
    cu = s[i0:i1]
    for a, n, w in ((HS.A_DEF, HS.N_DEF, "OKC_HYBST default"), (HS.A_SEGT, HS.N_SEGT, "wgs_seg template head"),
                    (HS.A_FIRST, HS.N_FIRST, "first stage call"), (HS.A_EPI, HS.N_EPI, "whole-unit epilogue"),
                    (HS.A_FIN, HS.N_FIN, "phase-end flush"), (HS.A_P1, HS.N_P1, "final pass 1"), (HS.A_P2, HS.N_P2, "final pass 2"),
                    (HN0_A, HN0_N, "hn policy fn"), (HN1_A, HN1_N, "hn policy select"), (HN2_A, HN2_N, "hn copy operand")):
        check(cu.count(a) == 1, f"CUDA anchor {w}: {cu.count(a)}x inside _HYB_MOE_CUDA")
        cu = cu.replace(a, n)
    s = s[:i0] + cu + s[i1:]
    a_flags = next(ln for ln in s.split(NL) if ln.startswith("HYB_FLAGS = ("))
    i_b0 = s.index("def _hyb_build():\n")
    i_b1 = s.index('    return _HYB["mod"]\n', i_b0) + len('    return _HYB["mod"]\n')
    i_c0 = s.index('    if _HYB["pdl"] != pdl:\n', s.index("def _hyb_call(x, ids, wts):\n"))
    i_c1 = s.index("    return out\n", i_c0)
    check(s[i_b0:i_b1].count("load_inline(") == 1 and s[i_c0:i_c1].count("okc_run(") == 1, "_hyb_build / _hyb_call shape")
    s = s[:i_b0] + BUILD_N + s[i_b1:i_c0] + CALL_N + s[i_c1:]
    for a, n, w in ((a_flags + NL, a_flags + NL + f"HYB_T7 = {t7}" + NL + 'HYB_F7 = ("-DOKC_HYBST=7", "-DOKC_HN=1")' + NL, "HYB_FLAGS"),
                    (NOTE_A, NOTE_N, "HYB notice end"), (STATE_A, STATE_N, "_HYB dict"), (LOOP_A, LOOP_N, "self-test loop head"),
                    (ST_A, ST_N, "self-test loop end"), (HS.A_LIVE, HS.A_LIVE.replace("pdl {int(HYB_PDL)}, ", "pdl {int(HYB_PDL)}, t7 {HYB_T7 if _HYB['mod7'] is not None else 'off'}, "), "LIVE print")):
        check(s.count(a) == 1, f"anchor {w}: {s.count(a)}x")
        s = s.replace(a, n)
    return s


def main():
    base, out = ROOT / ARGS["base"], ROOT / ARGS["out"]
    t7 = int(ARGS.get("t7", "193"))
    check(129 <= t7 <= 256, "--t7 in 129..256 (T=128 must stay on the ship module)")
    src = base.read_bytes().decode("ascii")
    new = apply(src, t7)
    ast.parse(new)
    check(new.isascii(), "non-ASCII")
    n = len(new.encode())
    out.write_bytes(new.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"hyb193_patches: {base.name} -> {out.name} ({n:,} B, +{n - len(src):,} B, {LIMIT - n:,} B under the cap; t7 {t7})")
    check(n < LIMIT or "lab" in out.name, f"size {n} >= {LIMIT}")


if __name__ == "__main__":
    main()

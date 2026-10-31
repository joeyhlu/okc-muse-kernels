"""hybpro_patches.py (R-hybpro, HYB-prologue builder): per-CTA prologue marks + the parked hybst riders as run-time arms on
k3_hh_3tkv, one prepare.

  python -I tools/hybpro_patches.py [--base=kernels/lab/k3_hh_3tkv.py] [--out=kernels/lab/k3_hybpro_ab.py] [--check]
  python -I tools/lead_gate.py file <out>

HYB = alygg77's WGS MoE engine (Apache-2.0, notice kept; T 49..320). The ship module (_HYB["mod"], unchanged source and flags)
stays the default. HYBP_ST selects an extra module built in prepare (PT1 job "hybp", parallel nvcc) from _HYB_MOE_CUDA with
the edits below applied at run time:
  7  = -DOKC_HYBST=7: tools/hybst_patches.py bits 1 / 2 / 4 (EPF unit-end constants + L1 prefetch, FIN phase-B-end read-back
       4 n-tiles per round trip, FP2 final pass keeps its y loads); same arithmetic as the ship.
  16 = -DOKC_HYB_MK=1: per-CTA marks into a 160 x 32 u64 buffer (_HYBP[16]["dbg"]): slot k = %globaltimer, slot 16 + k =
       clock64 at the same point (k = 0..14), slot 15 = stage / job info. Marks:
        0 pre-wait (tid 0, before griddepcontrol.wait)      1 wait done (tid 0)
        2 route done (tid 0, after route + bar.arrive 7)    3 gather done (tid 0, after wgs_gather incl. its xready release)
        4 WS_JOIN passed (tid 0)                            5 ranges done (x-producer lane 0, wgs_ranges, before WS_JOIN)
        6 first weight bulk copy issued (w-producer lane 0) 7 first x bulk copy issued (x-producer lane 0, after xready)
        8 first weight stage landed (lane 1 of the w-producer warp, otherwise idle, waits BW_F slot 0 parity 0 after
          WS_JOIN)                                          9 first x stage landed (the same lane, BX_F slot 0 parity 0)
       10 first weight slot freed = the w producer issues its 4th copy (all 16 consumer warps have read stage 0's codes at
          lg 3, i.e. the first stage's wgmma chain is issued); consumer-side marks spilled 28-172 B at the 96-register cap
       11 phase A end   12 phase B end   13 grid barrier passed
       14 kernel end (after the final pass)                15 = A stages | B stages << 16 | nj << 32 | gathered rows R << 48
     Arithmetic unchanged (marks are stores to a debug buffer only).
Each extra module has its own scratch, shares the ship weights, and is self-tested in prepare against the ship on every
HYB_SELF_TEST case + a concentrated input (3 local experts / 69 routes); a module that fails stays OFF (the call falls back to
the ship). --check: the output must equal apply(--base) byte for byte, plus AST / ASCII / size gates.
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import hybst_patches as HST  # noqa: E402  (the rider CUDA edits, OKC_HYBST bits 1 / 2 / 4)
sys.argv = _argv
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[hybpro_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


# ---------------- CUDA edits applied at run time to _HYB_MOE_CUDA (raw text: asm "\n" stays two characters) --------------
RIDERS = [(HST.A_DBG, HST.N_DBG), (HST.A_DEF, HST.N_DEF), (HST.A_SEGT, HST.N_SEGT), (HST.A_FIRST, HST.N_FIRST),
          (HST.A_EPI, HST.N_EPI), (HST.A_FIN, HST.N_FIN), (HST.A_P1, HST.N_P1), (HST.A_P2, HST.N_P2)]
MK_DEF_A = ("#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } "
            "while (0)\n")
MK_DEF_N = MK_DEF_A + ("#ifndef OKC_HYB_MK\n#define OKC_HYB_MK 0\n#endif\n"
                       "#define MK(k) do { if (OKC_HYB_MK && p.dbg) { const unsigned long long t_ = gtimer(); "
                       "p.dbg[blockIdx.x * 32 + (k)] = t_; p.dbg[blockIdx.x * 32 + 16 + (k)] = (unsigned long long)clock64(); } "
                       "} while (0)\n")
MK_WAIT_A = r'''DBG(0);
asm volatile("griddepcontrol.wait;\n" ::: "memory");
if (tid >= NT) {
'''
MK_WAIT_N = r'''DBG(0);
if (tid == 0) MK(0);
asm volatile("griddepcontrol.wait;\n" ::: "memory");
if (tid == 0) MK(1);
if (tid >= NT) {
'''
MK_ROUTE_A = "DBG(1);\n"
MK_ROUTE_N = "DBG(1);\nif (tid == 0) MK(2);\n"
MK_JOIN_A = "DBG(7);\nWS_JOIN();\nDBG(6);\n"
MK_JOIN_N = ("DBG(7);\nif (tid == 0) MK(3);\nWS_JOIN();\nDBG(6);\n"
             "if (tid == 0) { MK(4); if (OKC_HYB_MK && p.dbg) p.dbg[blockIdx.x * 32 + 15] = (unsigned long long)(S.wgr[1] - S.wgr[0]) "
             "| ((unsigned long long)(S.wgr[3] - S.wgr[2]) << 16) | ((unsigned long long)(S.njF + S.njS) << 32) "
             "| ((unsigned long long)S.jrow[S.njF + S.njS] << 48); }\n")
MK_END = [("DBG(2);\n", "DBG(2);\nif (tid == 0) MK(11);\n"), ("DBG(3);\n", "DBG(3);\nif (tid == 0) MK(12);\n"),
          ("DBG(4);\n", "DBG(4);\nif (tid == 0) MK(13);\n"), ("DBG(5);\n", "DBG(5);\nif (tid == 0) MK(14);\n")]
MK_RANGES_A = "if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);\n__syncwarp();\nWS_JOIN();\n"
MK_RANGES_N = "if ((threadIdx.x & 31) == 0) { wgs_ranges(p, S); MK(5); }\n__syncwarp();\nWS_JOIN();\n"
MK_XI_A = ('::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");\n'
           "if (++xs == WGS_NX) { xs = 0; xph ^= 1; }\n")
MK_XI_N = MK_XI_A.replace("if (++xs", "if (n == 0) MK(7);\nif (++xs")
MK_WI_A = ('::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n'
           "if (++ws == WGS_NS) { ws = 0; wph ^= 1; }\n")
MK_WI_N = MK_WI_A.replace("if (++ws", "if (n == 0) MK(6);\nelse if (n == WGS_NS) MK(10);\nif (++ws")
MK_WATCH_A = "__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {\nWS_JOIN();\nif ((threadIdx.x & 31) != 0) return;\n"
MK_WATCH_N = MK_WATCH_A.replace("WS_JOIN();\n", "WS_JOIN();\n#if OKC_HYB_MK\nif ((threadIdx.x & 31) == 1 && p.dbg && "
                                "(S.wgr[1] > S.wgr[0] || S.wgr[3] > S.wgr[2])) { mbar_wait(S.bars + BW_F, 0u); MK(8); "
                                "mbar_wait(S.bars + BX_F, 0u); MK(9); }\n#endif\n")
MARKS = ([(MK_DEF_A, MK_DEF_N), (MK_WAIT_A, MK_WAIT_N), (MK_ROUTE_A, MK_ROUTE_N), (MK_JOIN_A, MK_JOIN_N)] + MK_END
         + [(MK_RANGES_A, MK_RANGES_N), (MK_XI_A, MK_XI_N), (MK_WI_A, MK_WI_N), (MK_WATCH_A, MK_WATCH_N)])
EDITS = RIDERS + MARKS

# ---------------- Python glue ------------------------------------------------------------------------------------------------
HYB_CALL = '''def _hyb_call(x, ids, wts):
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYB_PDL else 0
    if _HYB["pdl"] != pdl:
        _HYB["mod"].okc_set_pdl(pdl)
        _HYB["pdl"] = pdl
    _HYB["mod"].okc_run(_HYB["w"], _HYB["sc"], x, ids, wts, out, 0, 32, _HYB["nsm"])
    return out
'''
HYB_CALL_NEW = '''def _hyb_call(x, ids, wts):
    H = _HYBP.get(HYBP_ST) if HYBP_ST else None
    return _hyb_run(H if (H is not None and H["ok"]) else _HYB, x, ids, wts)
def _hyb_run(H, x, ids, wts):
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYB_PDL else 0
    if H["pdl"] != pdl:
        H["mod"].okc_set_pdl(pdl)
        H["pdl"] = pdl
    sc = H["sc"] + [H["dbg"]] if H.get("dbg") is not None else H["sc"]
    H["mod"].okc_run(_HYB["w"], sc, x, ids, wts, out, 0, 32, _HYB["nsm"])
    return out
'''
HYB_LAB = '''HYBP_ST = 0
_HYBP_KEYS = (7, 16)
_HYBP = {k: {"mod": None, "ok": False, "sc": None, "full": None, "pdl": -1, "build_s": 0.0, "err": None, "dbg": None}
         for k in _HYBP_KEYS}
def _hybp_src(k):
    s = _HYB_MOE_CUDA
    for a, n in _HYBP_EDITS:
        assert s.count(a) == 1, a[:60]
        s = s.replace(a, n)
    return s, list(HYB_FLAGS) + [f"-DOKC_HYBST={k & 7}"] + (["-DOKC_HYB_MK=1"] if k & 16 else [])
def _hybp_build():
    import threading
    from torch.utils.cpp_extension import load_inline
    def one(k):
        H = _HYBP[k]
        try:
            src, fl = _hybp_src(k)
            tag = hashlib.sha1((_HYB_MOE_CPP + src + " ".join(fl)).encode()).hexdigest()[:12]
            t0 = time.time()
            H["mod"] = load_inline(name=f"hyb_moe_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=src,
                                   functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
            H["build_s"] = time.time() - t0
        except Exception as ex:
            H["err"] = repr(ex)[-600:]
    ths = [threading.Thread(target=one, args=(k,)) for k in _HYBP_KEYS]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
def _hybp_conc(T, dev, seed=777):
    ids = torch.full((T, 8), -1, dtype=torch.int32)
    for t in range(69):
        ids[t, 0] = (0, 7, 19)[t % 3]
    g = torch.Generator(device="cpu").manual_seed(4241 * seed + T)
    tw = torch.rand((T, 8), generator=g) + 0.05
    return (torch.randn((T, 4096), generator=g).to(torch.bfloat16).to(dev).contiguous(), ids.to(dev).contiguous(),
            (tw / tw.sum(-1, keepdim=True)).float().to(dev).contiguous())
def _hybp_prepare(st, w, dev):
    _pt1_wait("hybp")
    cases = []
    for bs, seed, dist in HYB_SELF_TEST:
        d = _k2a_self_test_inputs(bs, seed, dev, dist)
        cases.append((f"{bs}{dist[0]}", d["hidden_states"].contiguous(), d["topk_ids"].to(torch.int32).contiguous(),
                      d["topk_weights"].float().contiguous()))
    cases.append(("conc69",) + _hybp_conc(128, dev))
    refs = []
    for name, x, ids, tw in cases:
        d = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw, "config": {"local_start": 0, "local_count": K2A_E_LOC}}
        o0 = _hyb_run(_HYB, x, ids, tw).clone()
        o1 = _hyb_run(_HYB, x, ids, tw).clone()
        torch.cuda.synchronize()
        refs.append((_k2a_ref_fp32(d, w), o0, float((o1.float() - o0.float()).abs().max())))
    for k, H in _HYBP.items():
        H["ok"] = False
        if H["mod"] is None:
            print(f"[hybp] HYB st{k}: build failed {H['err']}", flush=True)
            continue
        try:
            H["sc"], H["full"] = _hyb_scratch(dev)
            H["dbg"] = torch.zeros(_HYB_SLOTS * 32, dtype=torch.int64, device=dev) if k & 16 else None
            torch.cuda.synchronize()
            good, worst, nbit = True, [0.0, 0.0, 0.0, 0.0], 0
            for (name, x, ids, tw), (ref, o3, rep3) in zip(cases, refs):
                oa = _hyb_run(H, x, ids, tw).clone()
                ob = _hyb_run(H, x, ids, tw).clone()
                torch.cuda.synchronize()
                tot, row = _k2a_rel_err(oa, ref)
                ev = _k2a_rel_err(oa, o3)[0]
                dd = float((oa.float() - o3.float()).abs().max())
                rp = _k2a_rel_err(ob, oa)[0]
                fin = bool(torch.isfinite(oa.float()).all() and torch.isfinite(ob.float()).all())
                ok = fin and tot < 1e-2 and row < 5e-2 and ev < 2e-3 and rp < 1e-3
                nbit += dd == 0.0
                worst = [max(worst[0], tot), max(worst[1], row), max(worst[2], ev), max(worst[3], rp)]
                if not ok or dd != 0.0:
                    print(f"[hybp] HYB st{k} self-test {name}: vs FP32 {tot:.2e} row {row:.2e} vs ship {ev:.2e} (max abs {dd:.2e}, "
                          f"ship repeat {rep3:.1e}) repeat {rp:.1e} finite {fin} -> {'ok' if ok else 'FAIL'}", flush=True)
                good &= ok
            gok = all(bool((f[n:].view(vt) == sv).all()) for f, n, vt, sv in H["full"])
            H["ok"] = bool(good and gok)
            print(f"[hybp] HYB st{k} {'LIVE' if H['ok'] else 'OFF'}: {len(cases)} cases ({nbit} bitwise == ship) worst vs FP32 "
                  f"{worst[0]:.2e} row {worst[1]:.2e} vs ship {worst[2]:.2e} repeat {worst[3]:.1e}, guards {gok}, nvcc {H['build_s']:.0f}s",
                  flush=True)
        except Exception as ex:
            H["ok"] = False
            print(f"[hybp] HYB st{k} prepare raised {ex!r} -> OFF", flush=True)
'''
HYB_PREP_END = ('              f"{len(THIRD_PARTY_NOTICES)} chars of third-party notices kept)", flush=True)\n'
                '    except Exception as ex:\n        _HYB["ok"] = False\n')
HYB_PREP_END_NEW = ('              f"{len(THIRD_PARTY_NOTICES)} chars of third-party notices kept)", flush=True)\n'
                    '        if _HYB["ok"]:\n            _hybp_prepare(st, w, dev)\n'
                    '    except Exception as ex:\n        _HYB["ok"] = False\n')
PT1_HYB = '    if HYB_ENABLE:\n        jobs.append(("hyb", _hyb_build))\n'
PT1_HYB_NEW = PT1_HYB + '        jobs.append(("hybp", _hybp_build))\n'
NOTE_NEW = ("helpers.\nModified by team joeyhlu (2026-10-31, tools/hybpro_patches.py, lab only): optional builds with per-CTA\n"
            "timing marks (-DOKC_HYB_MK) and the tools/hybst_patches.py unit-end / tail riders (-DOKC_HYBST); HYB's\n"
            "arithmetic is unchanged.\n\"\"\"\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n")


def apply(s):
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] = "# nonce k3-hybpro-ab (k3_hh_3tkv + R-hybpro lab arms: HYBP_ST 7 = hybst riders, 16 = prologue marks; tools/hybpro_patches.py)"
    s = NL.join(lines)
    i0, i1 = s.index('_HYB_MOE_CUDA = r"""'), s.index('_HYB_MOE_CPP = r"""')
    cu = s[i0:i1]
    for a, n in EDITS:   # dry run on a copy: the file keeps the ship CUDA text; _hybp_src applies EDITS at run time
        check(cu.count(a) == 1, f"CUDA anchor not unique: {a[:70]!r}")
        cu = cu.replace(a, n)
    check("#define OKC_HYB_MK 0" in cu and cu.count("MK(") >= 16, "marks inserted")
    s = sub(s, HYB_CALL, HYB_CALL_NEW, "_hyb_call")
    s = sub(s, "def _hyb_on(m):\n", HYB_LAB + f"_HYBP_EDITS = {EDITS!r}\n" + "def _hyb_on(m):\n", "HYBP lab block")
    s = sub(s, HYB_PREP_END, HYB_PREP_END_NEW, "_hyb_prepare end")
    s = sub(s, PT1_HYB, PT1_HYB_NEW, "PT1 hyb job")
    s = sub(s, HST.A_NOTE, NOTE_NEW, "HYB notice end")
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
    for a in ("def _hybp_prepare(", "_HYBP_EDITS = [", 'jobs.append(("hybp", _hybp_build))', "            _hybp_prepare(st, w, dev)" + NL,
              "def _hyb_run(H, x, ids, wts):"):
        check(s.count(a) == 1, f"missing {a!r}")
    print(f"hybpro_patches: gates OK on {path.name}: {n:,} B ({LIMIT - n:,} B under the cap), {len(d)} decorated defs re-parse")


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3tkv.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hybpro_ab.py")
    new = apply(base.read_bytes().decode("ascii"))
    if "check" in FLAGS:
        check(out.exists(), f"{out} missing")
        check(out.read_bytes().decode("ascii") == new, f"{out.name} differs from apply({base.name})")
        gates(out, new)
        print(f"hybpro_patches: --check OK ({out.name} == apply({base.name}))")
        return
    out.write_bytes(new.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"hybpro_patches: {base.name} -> {out.name} ({len(new):,} B, +{len(new) - base.stat().st_size:,} B)")
    gates(out, new)


if __name__ == "__main__":
    main()

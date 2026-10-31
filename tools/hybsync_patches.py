"""hybsync_patches.py (R-hybsync, HYB stage-sync builder): cut the per-stage mbarrier protocol cost of alygg77's WGS MoE engine (HYB,
T 49..320) measured by R-hovl-1b (two consumer waits = 190-330 clk per stage at EVERY nt class, i.e. protocol, not data lateness).

  python -I tools/hybsync_patches.py [--base=kernels/lab/k3_hh_5_sk2.py] [--out=kernels/lab/k3_hybsync_ab.py]      lab file
  python -I tools/hybsync_patches.py --ship --base=<final> --out=<new> [--hsy=3]                                    ship form
  python -I tools/lead_gate.py file <out>

-DOKC_HSY bits (the CUDA text preprocesses to the ship text with OKC_HSY 0 and OKC_HSY_MK 0):
  1 = MF (merged full barrier): BW_F[s] (count 2) carries both producers' arrive.expect_tx (weights UNIT_BYTES + x npad * 512 B);
      the consumer waits once per stage. Legal because WGS_NS == WGS_NX == 3 (same slot and parity for both rings; static_assert).
      Empty barriers unchanged (weight slot released at lg == 3, x slot after the next stage's first wgmma group), so the producers'
      issue times do not move. The BX_F barriers stay initialised and unused.
  2 = PK (peek): at lg == 3 of stage seq (after the BW_E arrive, before the dequant of the last two k16 groups) every lane issues a
      non-blocking mbarrier.test_wait.parity on the NEXT stage's full barrier(s) into pk (bit 0 = W / merged, bit 1 = X when not MF);
      the next stage's start waits only on the barriers whose bit is 0 (CUTLASS consumer_try_wait / consumer_wait(token) pattern).
      pk lives in wgs_phase (one register), is passed by reference into wgs_seg and carried across units; both phases start with
      pk = 0 (= wait). test_wait has acquire semantics, so a true peek orders the stage's smem loads like the wait would.
-DOKC_HSY_MK=1 (lab marks, hovl mk layout): clock() at the stage start and after the wait(s); per (CTA, WG, n-tile class) red.add
  into p.dbg[(cta * 4 + wg) * 64 + (ntt - 1) * 8 + k]: k = 0 stages, 1 wait clk (stage start -> waits done, = hovl wait_w + wait_x),
  2 stage clk, 3 stages whose peek said ready (PK only), 4 stage ns (globaltimer).
Lab keys (HYBP_ST; the ship module stays the default): bits 0-1 = OKC_HSY value, bit 2 (4) = HYB_F7 (st7 + hn, selected at
T >= HYB_T7 when the key exists), bit 4 (16) = marks. Default builds: 3 (MF + PK, ship flags), 7 (MF + PK + F7), 19 (MF + PK + marks).
Arithmetic unchanged: same bytes in the same smem slots, same wgmma order -> bitwise == ship on deterministic cases (self-tested in
prepare vs the matching ship module; conc69's atomics order is the only non-bitwise case).
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import hybpro_patches as HP  # noqa: E402  (check / sub + the lab-vehicle Python anchors)
sys.argv = _argv
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[hybsync_patches] FAIL: {msg}")


def sub(text, old, new, what, n_expected=1):
    n = text.count(old)
    check(n == n_expected, f"anchor {what}: found {n}x, expected {n_expected}: {old[:90]!r}")
    return text.replace(old, new)


# ---------------- CUDA edits (raw text: asm "\n" stays two characters) ------------------------------------------------------
DEF_A = 'static_assert(BX_E + WGS_NX <= 24, "WGS barriers");\n'
DEF_N = DEF_A + r'''#ifndef OKC_HSY
#define OKC_HSY 0
#endif
#ifndef OKC_HSY_MK
#define OKC_HSY_MK 0
#endif
#if OKC_HSY & 1
static_assert(WGS_NS == WGS_NX, "HSY merged full barrier needs equal ring depths");
#endif
#if OKC_HSY & 2
#define HSY_PK_PARAM , uint32_t& pk
#define HSY_PK_ARG , pk
__device__ __forceinline__ uint32_t mbar_test(uint64_t* bar, uint32_t parity) {
uint32_t r;
asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
: "=r"(r) : "r"(smem_u32(bar)), "r"(parity) : "memory");
return r;
}
#else
#define HSY_PK_PARAM
#define HSY_PK_ARG
#endif
'''
INIT_A = "for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);\n"
INIT_N = ("#if OKC_HSY & 1\nfor (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 2 : k < BX_F ? NW : k < BX_E ? 1 : NW);\n"
          "#else\n" + INIT_A + "#endif\n")
XBAR_A = "uint64_t* bar = S.bars + BX_F + xs;\nconst uint32_t bytes = (uint32_t)npad * 512u;\n"
XBAR_N = ("#if OKC_HSY & 1\nuint64_t* bar = S.bars + BW_F + xs;\n#else\nuint64_t* bar = S.bars + BX_F + xs;\n#endif\n"
          "const uint32_t bytes = (uint32_t)npad * 512u;\n")
WAIT_W = "wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);\n"
WAIT_X = "wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);\n"
WAIT_A = WAIT_W + "const unsigned char* pc = pc0 + ws * UNIT_BYTES;\n" + WAIT_X
WAIT_N = ("#if OKC_HSY_MK\nconst unsigned int mk0_ = clock(); const unsigned long long mg0_ = gtimer();\n"
          "#if OKC_HSY & 2\nconst uint32_t pk0_ = pk;\n#endif\n#endif\n"
          "#if OKC_HSY & 2\nif (!(pk & 1u)) " + WAIT_W + "#else\n" + WAIT_W + "#endif\n"
          "const unsigned char* pc = pc0 + ws * UNIT_BYTES;\n"
          "#if !(OKC_HSY & 1)\n#if OKC_HSY & 2\nif (!(pk & 2u)) " + WAIT_X + "#else\n" + WAIT_X + "#endif\n#endif\n"
          "#if OKC_HSY_MK\nconst unsigned int mk1_ = clock();\n#endif\n")
PEEK_A = "} else if (lane == 0) {\nmbar_arrive1(S.bars + BW_E + ws);\n}\n"
PEEK_N = ("#if OKC_HSY & 2\n} else {\nif (lane == 0) mbar_arrive1(S.bars + BW_E + ws);\n"
          "{\nconst int sq_ = seq + 1;\n"
          "#if OKC_HSY & 1\npk = mbar_test(S.bars + BW_F + sq_ % WGS_NS, (uint32_t)(sq_ / WGS_NS) & 1u);\n#else\n"
          "pk = mbar_test(S.bars + BW_F + sq_ % WGS_NS, (uint32_t)(sq_ / WGS_NS) & 1u)\n"
          "| (mbar_test(S.bars + BX_F + sq_ % WGS_NX, (uint32_t)(sq_ / WGS_NX) & 1u) << 1);\n#endif\n}\n}\n"
          "#else\n" + PEEK_A + "#endif\n")
END_A = "prev = xs;\n++seq;\n};\n"
END_N = ("prev = xs;\n++seq;\n#if OKC_HSY_MK\nif (p.dbg && wp == 0 && lane == 0) {\nconst unsigned int mk3_ = clock();\n"
         "unsigned long long* mkp_ = p.dbg + ((blockIdx.x * 4 + wg) * 64 + (NTT - 1) * 8);\n"
         "atomicAdd(mkp_, 1ull); atomicAdd(mkp_ + 1, (unsigned long long)(mk1_ - mk0_));\n"
         "atomicAdd(mkp_ + 2, (unsigned long long)(mk3_ - mk0_));\n"
         "#if OKC_HSY & 2\natomicAdd(mkp_ + 3, (unsigned long long)(pk0_ & 1u));\n#endif\n"
         "atomicAdd(mkp_ + 4, gtimer() - mg0_);\n}\n#endif\n};\n")
SIG_A = "int& seq, WgsPend& pd) {\n"
SIG_N = "int& seq, WgsPend& pd HSY_PK_PARAM) {\n"
PD_A = "WgsPend pd{0, 0, 0, 0};\n"
PD_N = PD_A + "#if OKC_HSY & 2\nuint32_t pk = 0;\n#endif\n"
CALLS = [(f"wgs_seg<PB, {n}>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n",
          f"wgs_seg<PB, {n}>(p, S, ring, u, s0, s1, slot, seq, pd HSY_PK_ARG); break;\n") for n in range(1, 9)]
EDITS = [(DEF_A, DEF_N), (INIT_A, INIT_N), (XBAR_A, XBAR_N), (WAIT_A, WAIT_N), (PEEK_A, PEEK_N), (END_A, END_N), (SIG_A, SIG_N),
         (PD_A, PD_N)] + CALLS


def apply_cuda(cu):
    for a, n in EDITS:
        cu = sub(cu, a, n, a[:60])
    check(cu.count("HSY_PK_ARG") == 10 and cu.count("HSY_PK_PARAM") == 3 and cu.count("mbar_test(") == 4, "HSY inserted")
    return cu


# ---------------- Python glue (lab vehicle; k3_hh_5* has the hyb193 structure: _HYB["mod"] + _HYB["mod7"] at T >= HYB_T7) ----------
CALL_A = '''def _hyb_call(x, ids, wts):
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
'''
CALL_N = '''def _hyb_call(x, ids, wts):
    m = x.shape[0]
    if HYBP_ST:
        H = _HYBP.get(HYBP_ST | 4) if m >= HYB_T7 else None
        if H is None or not H["ok"]:
            H = _HYBP.get(HYBP_ST)
        if H is not None and H["ok"]:
            return _hybp_run(H, x, ids, wts)
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
'''
HYB_LAB = '''HYBP_ST = 0
_HYBP_KEYS = (3, 7, 19)
_HYBP = {k: {"mod": None, "ok": False, "sc": None, "full": None, "pdl": -1, "build_s": 0.0, "err": None, "dbg": None} for k in _HYBP_KEYS}
_HYBP_CPP = _HYB_MOE_CPP
def _hybp_src(k):
    s = _HYB_MOE_CUDA
    for a, n in _HYBP_EDITS:
        assert s.count(a) == 1, a[:60]
        s = s.replace(a, n)
    fl = list(HYB_FLAGS) + (list(HYB_F7) if k & 4 else []) + [f"-DOKC_HSY={k & 3}"] + (["-DOKC_HSY_MK=1"] if k & 16 else [])
    return s, fl
def _hybp_build():
    from torch.utils.cpp_extension import load_inline
    def one(k):
        H = _HYBP[k]
        try:
            src, fl = _hybp_src(k)
            tag = hashlib.sha1((_HYBP_CPP + src + " ".join(fl)).encode()).hexdigest()[:12]
            t0 = time.time()
            H["mod"] = load_inline(name=f"hyb_moe_{tag}", cpp_sources=_HYBP_CPP, cuda_sources=src,
                                   functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
            H["build_s"] = time.time() - t0
        except Exception as ex:
            H["err"] = repr(ex)[-600:]
    ths = [threading.Thread(target=one, args=(k,)) for k in _HYBP_KEYS]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
def _hybp_run(H, x, ids, wts):
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYB_PDL else 0
    if H["pdl"] != pdl:
        H["mod"].okc_set_pdl(pdl)
        H["pdl"] = pdl
    sc = H["sc"] + [H["dbg"]] if H.get("dbg") is not None else H["sc"]
    H["mod"].okc_run(_HYB["w"], sc, x, ids, wts, out, 0, 32, _HYB["nsm"])
    return out
def _hybp_conc(T, dev, seed=777):
    ids = torch.full((T, 8), -1, dtype=torch.int32)
    for t in range(69):
        ids[t, 0] = (0, 7, 19)[t % 3]
    g = torch.Generator(device="cpu").manual_seed(4241 * seed + T)
    tw = torch.rand((T, 8), generator=g) + 0.05
    return (torch.randn((T, 4096), generator=g).to(torch.bfloat16).to(dev).contiguous(), ids.to(dev).contiguous(),
            (tw / tw.sum(-1, keepdim=True)).float().to(dev).contiguous())
def _hybp_ship(m, x, ids, tw, k):
    md = _HYB["mod7"] if ((k & 4) and _HYB["mod7"] is not None) else _HYB["mod"]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    md.okc_run(_HYB["w"], _HYB["sc"], x, ids, tw, out, 0, 32, _HYB["nsm"])
    return out
def _hybp_prepare(st, w, dev):
    _pt1_wait("hybp")
    cases = []
    for bs, seed, dist in HYB_SELF_TEST:
        d = _k2a_self_test_inputs(bs, seed, dev, dist)
        cases.append((f"{bs}{dist[0]}", d["hidden_states"].contiguous(), d["topk_ids"].to(torch.int32).contiguous(),
                      d["topk_weights"].float().contiguous()))
    cases.append(("conc69",) + _hybp_conc(128, dev))
    refs = {}
    for name, x, ids, tw in cases:
        d = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw, "config": {"local_start": 0, "local_count": K2A_E_LOC}}
        refs[name] = _k2a_ref_fp32(d, w)
    for k, H in _HYBP.items():
        H["ok"] = False
        if H["mod"] is None:
            print(f"[hybp] HYB st{k}: build failed {H['err']}", flush=True)
            continue
        try:
            H["sc"], H["full"] = _hyb_scratch(dev)
            H["dbg"] = torch.zeros(_HYB_SLOTS * 4 * 64, dtype=torch.int64, device=dev) if k & 16 else None
            torch.cuda.synchronize()
            good, worst, nbit, ncase = True, [0.0, 0.0, 0.0, 0.0], 0, 0
            for name, x, ids, tw in cases:
                m = x.shape[0]
                o3 = _hybp_ship(m, x, ids, tw, k).clone()
                o4 = _hybp_ship(m, x, ids, tw, k).clone()
                oa = _hybp_run(H, x, ids, tw).clone()
                ob = _hybp_run(H, x, ids, tw).clone()
                torch.cuda.synchronize()
                rep3 = float((o4.float() - o3.float()).abs().max())
                tot, row = _k2a_rel_err(oa, refs[name])
                ev = _k2a_rel_err(oa, o3)[0]
                dd = float((oa.float() - o3.float()).abs().max())
                rp = _k2a_rel_err(ob, oa)[0]
                fin = bool(torch.isfinite(oa.float()).all() and torch.isfinite(ob.float()).all())
                ok = fin and tot < 1e-2 and row < 5e-2 and ev < 2e-3 and rp < 1e-3
                nbit += dd == 0.0
                ncase += 1
                worst = [max(worst[0], tot), max(worst[1], row), max(worst[2], ev), max(worst[3], rp)]
                if not ok or dd != 0.0:
                    print(f"[hybp] HYB st{k} self-test {name}: vs FP32 {tot:.2e} row {row:.2e} vs ship {ev:.2e} (max abs {dd:.2e}, "
                          f"ship repeat {rep3:.1e}) repeat {rp:.1e} finite {fin} -> {'ok' if ok else 'FAIL'}", flush=True)
                good &= ok
            gok = all(bool((f[n:].view(vt) == sv).all()) for f, n, vt, sv in H["full"])
            H["ok"] = bool(good and gok)
            print(f"[hybp] HYB st{k} {'LIVE' if H['ok'] else 'OFF'}: {ncase} cases ({nbit} bitwise == ship) worst vs FP32 "
                  f"{worst[0]:.2e} row {worst[1]:.2e} vs ship {worst[2]:.2e} repeat {worst[3]:.1e}, guards {gok}, nvcc {H['build_s']:.0f}s",
                  flush=True)
        except Exception as ex:
            H["ok"] = False
            print(f"[hybp] HYB st{k} prepare raised {ex!r} -> OFF", flush=True)
'''
NOTE_A = "HYB's arithmetic is unchanged.\n\"\"\"\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n"
NOTE_LAB = ("HYB's arithmetic is unchanged.\nModified by team joeyhlu (2026-10-31, tools/hybsync_patches.py, lab only): optional builds with "
            "a merged per-slot full mbarrier\n(-DOKC_HSY bit 1), a non-blocking next-stage peek (bit 2) and per-stage timing marks "
            "(-DOKC_HSY_MK); HYB's arithmetic is unchanged.\n\"\"\"\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n")
NOTE_SHIP = ("HYB's arithmetic is unchanged.\nModified by team joeyhlu (2026-10-31, tools/hybsync_patches.py, -DOKC_HSY): one full "
             "mbarrier per ring slot carries both the\nweight and the x transaction (one consumer wait per stage) and the next stage's "
             "barrier is peeked non-blockingly during\nthe current stage; HYB's arithmetic, bytes and wgmma order are unchanged.\n\"\"\"\n"
             "THIRD_PARTY_NOTICES += _HYB_NOTICE\n")


def hyb_block(s):
    i0, i1 = s.index('_HYB_MOE_CUDA = r"""'), s.index('_HYB_MOE_CPP = r"""')
    check(i0 < i1 and s.count('_HYB_MOE_CUDA = r"""') == 1, "HYB CUDA block")
    return i0, i1


def apply_lab(s, base_name):
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] = (f"# nonce k3-hybsync-ab ({base_name} + R-hybsync lab arms: HYBP_ST 3 = MF + PK (7 = + F7 at T >= 193), 19 = + marks; "
                "tools/hybsync_patches.py)")
    s = NL.join(lines)
    i0, i1 = hyb_block(s)
    apply_cuda(s[i0:i1])   # dry run: the file keeps the ship CUDA text; _hybp_src applies _HYBP_EDITS at run time
    s = sub(s, CALL_A, CALL_N, "_hyb_call")
    s = sub(s, "def _hyb_on(m):\n", HYB_LAB + f"_HYBP_EDITS = {EDITS!r}\n" + "def _hyb_on(m):\n", "HYBP lab block")
    s = sub(s, HP.HYB_PREP_END, HP.HYB_PREP_END_NEW, "_hyb_prepare end")
    s = sub(s, HP.PT1_HYB, HP.PT1_HYB_NEW, "PT1 hyb job")
    s = sub(s, NOTE_A, NOTE_LAB, "HYB notice end")
    return s


def apply_ship(s, hsy):
    check(hsy in (1, 2, 3), "--hsy in 1..3")
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] += f" + R-hybsync HYB stage sync OKC_HSY={hsy} [tools/hybsync_patches.py]"
    s = NL.join(lines)
    i0, i1 = hyb_block(s)
    s = s[:i0] + apply_cuda(s[i0:i1]) + s[i1:]
    a_flags = next(ln for ln in s.split(NL) if ln.startswith("HYB_FLAGS = ("))
    check(a_flags.endswith(")") and "-DOKC_HSY" not in a_flags, "HYB_FLAGS line")
    s = sub(s, a_flags + NL, a_flags[:-1] + f", '-DOKC_HSY={hsy}')" + NL, "HYB_FLAGS")
    s = sub(s, NOTE_A, NOTE_SHIP, "HYB notice end")
    return s


def gates(path, s):
    ast.parse(s)
    check(s.isascii(), "non-ASCII output")
    check("\r" not in s, "CR in output")
    n = len(s.encode("ascii"))
    check(n < LIMIT or "_ab" in path.name or "lab" in path.name, f"size {n} B >= {LIMIT} (lab files may exceed the cap; a ship form may not)")
    path.write_bytes(s.encode("ascii"))
    py_compile.compile(str(path), doraise=True)
    return n


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_5_sk2.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hybsync_ab.py")
    src = base.read_bytes().decode("ascii")
    if "ship" in FLAGS:
        new = apply_ship(src, int(ARGS.get("hsy", "3")))
    else:
        new = apply_lab(src, base.stem)
    n = gates(out, new)
    print(f"hybsync_patches: {base.name} -> {out.name} ({n:,} B, {n - len(src):+,} B, {LIMIT - n:,} B under the cap; "
          f"{'ship form OKC_HSY=' + ARGS.get('hsy', '3') if 'ship' in FLAGS else 'lab vehicle keys 3 / 7 / 19'})")


if __name__ == "__main__":
    main()

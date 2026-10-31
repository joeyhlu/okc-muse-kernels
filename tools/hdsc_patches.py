"""hdsc_patches.py (R-hdsc, HYB post-barrier L2 discard; Fable outside view s5 S1): after HYB's final in-launch grid barrier no
CTA reads xA / xB any more, and each ybuf slot row is read once by its token's CTA in the finalize. Discard those lines from L2
(discard.global.L2, 128 B, no write-back) so the unpooler's 64 MB weight stream does not pay their dirty write-backs.

  python -I tools/hdsc_patches.py [--base=kernels/lab/k3_hh_5_sk2_pz.py] [--out=kernels/lab/k3_hdsc_ab.py]      lab file
  python -I tools/hdsc_patches.py --ship --base=<final> --out=<new> [--hdsc=3]                                 ship form
  python -I tools/lead_gate.py file <out>

-DOKC_HDSC bits (default 0 = the base text):
  1 = after the grid barrier's CTA_SYNC (DBG(4)), every CTA discards its share of the xA and xB lines [0, R * 8 KB),
      R = S.jrow[njF + njS] (padded rows; all streamed rows are rewritten every call by the pz base, so the lines are fully dead:
      the next call writes them before its xready / doneA releases). No async-proxy copy is in flight after the barrier
      (every bulk copy was waited on by its consumer stage before the phase loops ended).
  2 = in the finalize, after the per-token loop's closing CTA_SYNC (all 512 threads' reads of the token's 8 ybuf slot rows are
      done), thread t discards line (t & 127) of slot rows (t >> 7) and (t >> 7) + 4 (16 KB = 128 lines per row). Each slot
      row belongs to one (token, k) pair = one CTA; phase B rewrites every real row in full each call.
  The fix region is NOT discarded: the last arriver re-zeroes it for the next call's red.add (live lines).
Lab keys (HYBP_ST; 0 = the base module): 3 = OKC_HDSC 3 (7 = + HYB_F7 at T >= HYB_T7). Outputs bitwise == base (no arithmetic change).
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import hybpro_patches as HP  # noqa: E402  (the lab-vehicle Python anchors)
import padz_patches as PZ  # noqa: E402  (the HYB lab block: CALL / HYB_LAB templates)
sys.argv = _argv
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[hdsc_patches] FAIL: {msg}")


def sub(text, old, new, what, n_expected=1):
    n = text.count(old)
    check(n == n_expected, f"anchor {what}: found {n}x, expected {n_expected}: {old[:90]!r}")
    return text.replace(old, new)


DSC = 'asm volatile("discard.global.L2 [%0], 128;\\n" ::"l"('
D_DEF_A = "#ifndef OKC_PADZ\n#define OKC_PADZ 0\n#endif\n"
D_DEF_N = D_DEF_A + "#ifndef OKC_HDSC\n#define OKC_HDSC 0\n#endif\n"
D_BAR_A = "while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);\n}\nCTA_SYNC();\nDBG(4);\n"
D_BAR_N = D_BAR_A + ("#if OKC_HDSC & 1\n{\nconst int nl_ = S.jrow[S.njF + S.njS] * (HD * 2 / 128);\n"
                     "for (int li = blockIdx.x * NT + tid; li < nl_; li += gridDim.x * NT) {\n"
                     + DSC + 'reinterpret_cast<const char*>(p.xA) + (size_t)li * 128) : "memory");\n'
                     + DSC + 'reinterpret_cast<const char*>(p.xB) + (size_t)li * 128) : "memory");\n'
                     "}\n}\n#endif\n")
D_FIN_A = "*reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;\nCTA_SYNC();\n}\n"
D_FIN_N = ("*reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;\nCTA_SYNC();\n#if OKC_HDSC & 2\n#pragma unroll\n"
           "for (int h_ = 0; h_ < 2; ++h_) {\nconst int q_ = S.pos[tk * 8 + (tid >> 7) + 4 * h_];\n"
           "if (q_ >= 0) " + DSC + 'reinterpret_cast<const char*>(p.ybuf + (size_t)q_ * HD) + (tid & 127) * 128) : "memory");\n'
           "}\n#endif\n}\n")
EDITS = [(D_DEF_A, D_DEF_N), (D_BAR_A, D_BAR_N), (D_FIN_A, D_FIN_N)]


def apply_cuda(cu):
    for a, n in EDITS:
        cu = sub(cu, a, n, "HYB " + a[:50])
    check(cu.count("discard.global.L2") == 3 and cu.count("#if OKC_HDSC & ") == 2, "HDSC inserted")
    return cu


HYB_LAB = (PZ.HYB_LAB.replace("_HYBP_KEYS = (1, 5, 17)", "_HYBP_KEYS = (3, 7)")
           .replace('''    for a, n, c in _HYBP_EDITS + (_HYBP_MKEDITS if k & 16 else []):
        assert s.count(a) == c, a[:60]
        s = s.replace(a, n)
    pz = 2 if k & 16 else (1 if k & 1 else 0)
    fl = list(HYB_FLAGS) + (list(HYB_F7) if k & 4 else []) + [f"-DOKC_PADZ={pz}"] + (["-DOKC_HSY=0", "-DOKC_HSY_MK=1"] if k & 16 else [])
''', '''    for a, n in _HYBP_EDITS:
        assert s.count(a) == 1, a[:60]
        s = s.replace(a, n)
    fl = list(HYB_FLAGS) + (list(HYB_F7) if k & 4 else []) + [f"-DOKC_HDSC={k & 3}"]
''').replace("HYB pz{k}", "HYB hd{k}"))
NOTE_A = '"""\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n'
NOTE_LAB = ("Modified by team joeyhlu (2026-10-31, tools/hdsc_patches.py, lab only): optional builds that discard dead xA / xB / ybuf "
            "lines from L2\nafter the final grid barrier (-DOKC_HDSC); HYB's arithmetic and outputs are unchanged.\n" + NOTE_A)
NOTE_SHIP = ("Modified by team joeyhlu (2026-10-31, tools/hdsc_patches.py, -DOKC_HDSC): after the final grid barrier the dead xA / xB "
             "lines, and in the\nfinalize each ybuf slot row after its last read, are discarded from L2 (no write-back); HYB's "
             "arithmetic and outputs are unchanged.\n" + NOTE_A)


def hyb_block(s):
    i0, i1 = s.index('_HYB_MOE_CUDA = r"""'), s.index('_HYB_MOE_CPP = r"""')
    check(i0 < i1 and s.count('_HYB_MOE_CUDA = r"""') == 1, "HYB CUDA block")
    return i0, i1


def apply_lab(s, base_name):
    check("_HYBP_KEYS" not in s, "base already has a HYBP lab block")
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] = f"# nonce k3-hdsc-ab ({base_name} + R-hdsc lab arm: HYBP_ST 3 = HYB post-barrier L2 discard (7 = + F7 at T >= 193); tools/hdsc_patches.py)"
    s = NL.join(lines)
    i0, i1 = hyb_block(s)
    apply_cuda(s[i0:i1])
    s = sub(s, PZ.CALL_A, PZ.CALL_N, "_hyb_call")
    s = sub(s, "def _hyb_on(m):\n", HYB_LAB + f"_HYBP_EDITS = {EDITS!r}\n" + "def _hyb_on(m):\n", "HYBP lab block")
    s = sub(s, HP.HYB_PREP_END, HP.HYB_PREP_END_NEW, "_hyb_prepare end")
    s = sub(s, HP.PT1_HYB, HP.PT1_HYB_NEW, "PT1 hyb job")
    s = sub(s, NOTE_A, NOTE_LAB, "HYB notice end")
    return s


def apply_ship(s, hd):
    check(hd in (1, 2, 3), "--hdsc in 1..3")
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] += f" + R-hdsc HYB post-barrier L2 discard OKC_HDSC={hd} [tools/hdsc_patches.py]"
    s = NL.join(lines)
    i0, i1 = hyb_block(s)
    s = s[:i0] + apply_cuda(s[i0:i1]) + s[i1:]
    a_flags = next(ln for ln in s.split(NL) if ln.startswith("HYB_FLAGS = ("))
    check(a_flags.endswith(")") and "-DOKC_HDSC" not in a_flags, "HYB_FLAGS line")
    s = sub(s, a_flags + NL, a_flags[:-1] + f", '-DOKC_HDSC={hd}')" + NL, "HYB_FLAGS")
    s = sub(s, NOTE_A, NOTE_SHIP, "HYB notice end")
    return s


def gates(path, s):
    ast.parse(s)
    check(s.isascii(), "non-ASCII output")
    check("\r" not in s, "CR in output")
    n = len(s.encode("ascii"))
    check(n < LIMIT or "_ab" in path.name, f"size {n} B >= {LIMIT} (lab files may exceed the cap; a ship form may not)")
    path.write_bytes(s.encode("ascii"))
    py_compile.compile(str(path), doraise=True)
    return n


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_5_sk2_pz.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hdsc_ab.py")
    src = base.read_bytes().decode("ascii")
    new = apply_ship(src, int(ARGS.get("hdsc", "3"))) if "ship" in FLAGS else apply_lab(src, base.stem)
    n = gates(out, new)
    print(f"hdsc_patches: {base.name} -> {out.name} ({n:,} B, {n - len(src):+,} B, {LIMIT - n:,} B under the cap; "
          f"{'ship form OKC_HDSC=' + ARGS.get('hdsc', '3') if 'ship' in FLAGS else 'lab vehicle keys 3 / 7'})")


if __name__ == "__main__":
    main()

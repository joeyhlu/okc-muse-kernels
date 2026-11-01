"""hya_marks_patch.py (attention lane, 2026-11-01): kernels/muse_attention_decode/hya_t0.py + per-CTA phase timestamps -> kernels/lab/hya_marks.py.

Marks (thread 0 of each CTA writes %globaltimer and clock64 into p.dbg, 16 int64 per CTA: slot 2k = globaltimer ns, 2k + 1 = clock64):
  0 kernel entry   1 speculated page indices landed   2 first K/V tile landed   3 tile loop done (merge starts)
  4 partial red.add's issued + CTA barrier (ticket / barrier arrive next)
  5 ticket returned (last-CTA mode) or the cooperative barrier released   6 finalize done (last CTA / every CTA in coop mode)
Diagnostic only: the extension is renamed (okc_attn_v298mk) so it never shares a build with hya_t0. Run with tools/box/hya_marks_run.py.
    python -I tools/hya_marks_patch.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels" / "muse_attention_decode" / "hya_t0.py"
OUT = ROOT / "kernels" / "lab" / "hya_marks.py"

MACRO = r"""#define PROF_DUMP
#define HYA_MARK(k) do { if (p.dbg && threadIdx.x == 0) { long long* d_ = p.dbg + ((blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + 2 * (k)); long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_) :: "memory"); d_[0] = t_; d_[1] = (long long)clock64(); } } while (0)
"""

EDITS = [
    # (old, new, expected count)
    ("#define PROF_DUMP\n", MACRO, 1),
    ("  const int pair = blockIdx.z;\n  if (!FLAT) {\n", "  const int pair = blockIdx.z;\n  HYA_MARK(0);\n  if (!FLAT) {\n", 1),
    ("  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n", "  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n  HYA_MARK(1);\n", 1),
    ("      cp_wait<STAGES - 2>();\n      __syncthreads();\n", "      cp_wait<STAGES - 2>();\n      __syncthreads();\n      if (i == 0) HYA_MARK(2);\n", 1),
    ("      __syncthreads();       // tile i landed; previous tile's PV done (V16 free)\n",
     "      __syncthreads();       // tile i landed; previous tile's PV done (V16 free)\n      if (i == 0) HYA_MARK(2);\n", 1),
    ("  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----\n",
     "  HYA_MARK(3);\n  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----\n", 1),
    ("  if (!(mode1 || !mode1)) gpu_fence();\n  csync<WS, NT>();\n", "  if (!(mode1 || !mode1)) gpu_fence();\n  csync<WS, NT>();\n  HYA_MARK(4);\n", 1),
    # cooperative per-request mode: barrier released
    ("    csync<WS, NT>();\n    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);\n",
     "    csync<WS, NT>();\n    HYA_MARK(5);\n    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);\n", 1),
    # cooperative mode: slice finalized (end of the rbase loop, before the else branch of the last-CTA mode)
    ("      if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;\n    }\n    \n  } else {\n  if (tid == 0) {\n    const int mine = end - start;\n",
     "      if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;\n    }\n    HYA_MARK(6);\n  } else {\n  if (tid == 0) {\n    const int mine = end - start;\n", 1),
    # last-CTA mode: ticket returned
    ("    S.flag = (old + mine == n_keys);\n  }\n  csync<WS, NT>();\n  if (S.flag) {\n",
     "    S.flag = (old + mine == n_keys);\n  }\n  csync<WS, NT>();\n  HYA_MARK(5);\n  if (S.flag) {\n", 1),
    # last-CTA mode: finalize done (replaces the constant debug flag)
    ("  if (p.dbg && tid == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + 6] = 1;\n",
     "  HYA_MARK(6);\n", 1),
    ('name=f"okc_attn_v298{tag}_90a"', 'name=f"okc_attn_v298mk{tag}_90a"', 1),
]


def main():
    text = SRC.read_text(encoding="ascii")
    for old, new, n in EDITS:
        c = text.count(old)
        if c != n:
            sys.exit(f"anchor count {c} != {n} for: {old[:80]!r}")
        text = text.replace(old, new)
    lines = text.split("\n")
    lines[1] = "# nonce hya-marks (attention lane Nov 1, DIAGNOSTIC: per-CTA phase timestamps; tools/hya_marks_patch.py; never submitted)"
    OUT.write_text("\n".join(lines), encoding="ascii", newline="\n")
    print(OUT.relative_to(ROOT), len(OUT.read_bytes()), "B")


if __name__ == "__main__":
    main()

"""hya_clean_patch.py (attention lane, 2026-11-01): board-ready variants of hya_t0 WITHOUT the diagnostic marks / runtime switches of tools/hya_x_patch.py.

    python -I tools/hya_clean_patch.py            # writes kernels/lab/hya_u1.py (hoist only)

u1 = hya_t0 + HOIST: the finalize's gate-row load is issued before the ticket (last-CTA mode, L = 1: d1s / d16s / d32s) and before the cooperative barrier
(every other non-flat shape) instead of after it, so the DRAM-latency gate load no longer sits in the same round trip as the accumulator reads (R-att-4:
pfw & 8 vs control, in-process paired A/B, 100 reps: d1s +3.8 %, d16s +3.0 %, d32s +2.6 %, v16s +2.8 %, others within noise). Numerics unchanged.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels" / "muse_attention_decode" / "hya_t0.py"

GPRE_OLD = "  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----\n"
GPRE_NEW = r"""  // the last-CTA finalize's gate row (L = 1), loaded before the merge: its DRAM latency leaves the ticket -> finalize critical path
  uint4 gpre = make_uint4(0u, 0u, 0u, 0u);
  if (L == 1 && !WS && tid < NT)
    gpre = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + (tid >> 3)) * 64 + (tid & 7) * 8);
  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----
"""
FIN_SIG_OLD = "                                     const int dry) {\n  const int tid = threadIdx.x;\n"
FIN_SIG_NEW = ("                                     const int dry, const uint4 gpre = make_uint4(0u, 0u, 0u, 0u), const bool use_pre = false) {\n"
               "  const int tid = threadIdx.x;\n")
FIN_LOAD_OLD = ("      graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);\n"
                "    }\n#pragma unroll\n    for (int u = 0; u < GP; ++u) {\n      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);\n"
                "      const int j = rr >> 4, hh = rr & 15;\n      float* acc")
FIN_LOAD_NEW = ("      if (use_pre) graw[u] = gpre;\n"
                "      else graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);\n"
                "    }\n#pragma unroll\n    for (int u = 0; u < GP; ++u) {\n      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);\n"
                "      const int j = rr >> 4, hh = rr & 15;\n      float* acc")
COOP_OLD = '    asm volatile("" : "+r"(nact), "+r"(R));\n'
COOP_NEW = r"""    asm volatile("" : "+r"(nact), "+r"(R));
    const bool use_pre = R <= NT / 8;   // one pass of the slice loop: the gate row is loaded before the barrier instead of after it
    uint4 gate_pre = make_uint4(0u, 0u, 0u, 0u);
    if (use_pre) {
      const int rr = split_idx * R + (tid >> 3);
      if (rr < min(L * 16, split_idx * R + R)) {
        const int j = rr >> 4, hh = rr & 15;
        gate_pre = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + (tid & 7) * 8);
      }
    }
"""
COOP_LOAD_OLD = ("      const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)\n"
                 "                             : make_uint4(0, 0, 0, 0);\n")
COOP_LOAD_NEW = ("      const uint4 graw = use_pre ? gate_pre : (act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)\n"
                 "                             : make_uint4(0, 0, 0, 0));\n")

EDITS_U1 = [
    (GPRE_OLD, GPRE_NEW, 1),
    ("    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);\n", "    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0, gpre, L == 1);\n", 1),
    (FIN_SIG_OLD, FIN_SIG_NEW, 1),
    (FIN_LOAD_OLD, FIN_LOAD_NEW, 1),
    (COOP_OLD, COOP_NEW, 1),
    (COOP_LOAD_OLD, COOP_LOAD_NEW, 1),
]

NOTICE = ("# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41ee1c), licensed under the Apache\n"
          "# License 2.0 (full text: tools/vendor/vjeux_okc2026_f35a41ee/LICENSE). Modified by team joeyhlu, 2026-10-31 / 2026-11-01: HYA M1\n"
          "# (spin_fail sets a sticky flag and the poll loop breaks instead of __trap), the _TUNED entries for full B 16 / 32 and the finalize gate-row hoist.\n")


def build(name, edits):
    text = SRC.read_text(encoding="ascii")
    for old, new, n in edits:
        c = text.count(old)
        if c != n:
            sys.exit(f"anchor count {c} != {n} for: {old[:90]!r}")
        text = text.replace(old, new)
    assert text.count('name=f"okc_attn_v298{tag}_90a"') == 1
    text = text.replace('name=f"okc_attn_v298{tag}_90a"', 'name=f"okc_attn_v298' + name + '{tag}_90a"')   # its own extension build
    ls = text.split("\n")
    assert ls[0].startswith("# okc: problem=muse_attention_decode")
    ls[1] = f"# nonce hya-{name} (attention lane Nov 1: hya_t0 + finalize gate-row hoist; tools/hya_clean_patch.py)"
    out = ROOT / "kernels" / "lab" / f"hya_{name}.py"
    out.write_text("\n".join(ls), encoding="ascii", newline="\n")
    print(out.relative_to(ROOT), len(out.read_bytes()), "B")


if __name__ == "__main__":
    build("u1", EDITS_U1)

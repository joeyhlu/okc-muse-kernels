"""hya_x_patch.py (attention lane, 2026-11-01): hya_t0 + HYA_MARK timestamps (tools/hya_marks_patch.py) + a runtime-switchable L2 warm-up of the merge
workspace -> kernels/lab/hya_xa.py .. hya_xd.py (ONE CUDA source, so one JIT build; the arms differ only in Python-side `_TUNED` plan bits).

Hypothesis (R-att-3): the judge flushes L2 before every replay, so the merge workspace (ws_acc 4 KB per request-pair and copy, ws_l, the ticket /
barrier counters) is cold when the first red.add / the acq_rel ticket reaches L2: each starts from DRAM (~0.6 us). hya_t0 already warms the GATE rows
(split 0 only) but not the workspace. Switch bits (plan["tma"] bits 5..7 -> Params::pfw):
  pfw & 1  prefetch.global.L2 of this CTA's ws_acc lines (L * 32 x 128 B) and ws_l lines          (all non-flat CTAs, before pdl_wait)
  pfw & 2  prefetch.global.L2 of the ticket counter ws_cnt[bp] and the cooperative counters (+8192, +12288)
  pfw & 4  the gate-row prefetch (hya_t0: split 0 only) by every CTA, so the last CTA's gate loads hit L2 whichever CTA is last
Arms: xa = control (pfw 0, marks inert), xb = pfw 1, xc = pfw 3, xd = pfw 7. Numerics unchanged (prefetches only).
    python -I tools/hya_x_patch.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hya_marks_patch as M  # noqa: E402

ROOT = M.ROOT
SRC = M.SRC

PF_BLOCK = r"""  cp_commit();
  if (!WS && (p.pfw & 3)) {
    // L2 warm-up of the merge workspace (cold after the judge's L2 flush: the first red.add / the ticket would start from DRAM)
    if (p.pfw & 1) {
      const char* wa = reinterpret_cast<const char*>(p.ws_acc + (ix_t)((bp * p.nc + copy) * L) * (16 * 64));
      if (tid < L * 32) asm volatile("prefetch.global.L2 [%0];" ::"l"(wa + (ix_t)tid * 128));
      if (tid >= 64 && tid < 64 + L) asm volatile("prefetch.global.L2 [%0];" ::"l"(p.ws_l + (ix_t)((bp * p.nc + copy) * L + (tid - 64)) * 16));
    }
    if ((p.pfw & 2) && tid >= 96 && tid < 99)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(p.ws_cnt + (tid == 96 ? 0 : (tid == 97 ? 8192 : 12288)) + bp));
  }
  if (!WS && (MODE & 8)) {"""

EDITS = [
    ("  cp_commit();\n  if (!WS && (MODE & 8)) {", PF_BLOCK, 1),
    ("  if (split == 0 && (FLAT || !M_CL)) {  // warm L2 with the gate rows for whichever CTA finishes last",
     "  if ((split == 0 || (p.pfw & 4)) && (FLAT || !M_CL)) {  // warm L2 with the gate rows for whichever CTA finishes last", 1),
    ("  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)\n};",
     "  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)\n  int pfw;              // bits 0 / 1 / 2: L2 warm-up of the merge workspace / counters / gate rows by every CTA (R-att-3)\n};", 1),
    ("  p.coop = (int)((tma >> 1) & 1);\n", "  p.coop = (int)((tma >> 1) & 1);\n  p.pfw = (int)((tma >> 5) & 7);\n", 1),
]

# plan overrides per arm: pfw bits go in "tma" (bits 5..7). The base plan stays as in hya_t0 except for the two shapes hya_t0 already overrides.
BASE = {(32, 8, True): dict(n_splits=4, nc=1, coop=1), (1, 1, False): dict(nc=8, coop=1, n_splits=252)}
SHAPES = [(1, 1, True), (16, 1, True), (32, 1, True), (1, 8, True), (16, 8, True), (32, 8, True), (1, 1, False), (1, 8, False)]
ARMS = {"xa": 0, "xb": 1, "xc": 3, "xd": 7}
OLD_TUNED = """_TUNED = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
}
"""


def fmt(d):
    return "dict(" + ", ".join(f"{k}={v}" for k, v in d.items()) + ")"


def main():
    text = SRC.read_text(encoding="ascii")
    for old, new, n in M.EDITS + EDITS:
        c = text.count(old)
        if c != n:
            sys.exit(f"anchor count {c} != {n} for: {old[:90]!r}")
        text = text.replace(old, new)
    assert text.count("okc_attn_v298mk") == 1
    text = text.replace("okc_attn_v298mk", "okc_attn_v298x")   # its own extension build (never shared with hya_marks / hya_t0)
    assert text.count(OLD_TUNED) == 1
    for name, bits in ARMS.items():
        ent = ""
        for key in SHAPES:
            d = dict(BASE.get(key, {}))
            if bits:
                d["tma"] = bits << 5
            if d:
                ent += f"    ({key[0]}, {key[1]}, {key[2]}, 1): {fmt(d)},\n"
        t = text.replace(OLD_TUNED, "_TUNED = {\n" + ent + "}\n")
        ls = t.split("\n")
        ls[1] = f"# nonce hya-x-{name} (attention lane Nov 1: hya_t0 + HYA_MARK + merge-workspace L2 warm-up bits {bits}; tools/hya_x_patch.py; lab only)"
        out = ROOT / "kernels" / "lab" / f"hya_{name}.py"
        out.write_text("\n".join(ls), encoding="ascii", newline="\n")
        print(out.relative_to(ROOT), len(out.read_bytes()), "B", "pfw", bits)


if __name__ == "__main__":
    main()

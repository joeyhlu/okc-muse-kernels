"""wgs32_patches.py (R-wgs32-*): a FT_T = 32 instantiation of HYB (the public alygg77 WGS MoE engine, A32) for T 17..48.

  python -I tools/wgs32_patches.py [--base=kernels/lab/k3_hh_3k.py] [--out=kernels/lab/k3_wgs32_1.py] [--t=17,48] [--check]
  python -I tools/lead_gate.py file <out>

Idea (Madhavan113/kerneloptimizer, Apache-2.0, ideas only, no code copied: "WGS32 = MAIN with FT_T = 32 for 17 <= M <= 32",
-0.9..-3.8 us bs32 on their base; alygg77's NEGATIVE_RESULTS lists the same tile at +2.2..+3.0 us d32 on theirs). Adds a third
MoE module "32" (tools/hybx_common.py helper: own scratch, own PT1 build job "hyb32", ship HYB weight copies) built from
_HYB_MOE_CUDA with two text edits applied at import (_hyb32_src): OKC_FT_T 64 -> 32 (fat jobs <= 32 tokens, WGS_NT 4: accumulators,
x slots and fixup slots halve) and the n-tile switch capped at 4 (wgs_seg<PB, 5..8> would index d[WGS_NT] out of range).
HYB32_T (default --t) routes T in that range to it, ahead of VML (VMS keeps T <= 16; HYB keeps 49..320). Prepare self-test:
judge-like + skewed + concentrated routings (1..4 local experts, up to T tokens on one, incl. the (32, 72, skewed) input where
VML returned non-finite rows) vs FP32 and vs the ship HYB output. --t=0,-1 = compiled + self-tested but never routed (A/B file).
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hybx_common  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[wgs32_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


W32 = r'''HYB32_T = (@LO@, @HI@)
HYB32_TESTS = ((17, 70, "skewed"), (32, 71, "normal"), (32, 72, "skewed"), (48, 73, "normal"), (48, 74, "skewed"),
               (32, 75, "conc1"), (32, 76, "conc3"), (24, 77, "conc4"), (48, 78, "conc2"), (40, 79, "conc1"))
def _hyb32_src(s):
    seg = "(p, S, ring, u, s0, s1, slot, seq, pd); break;\n"
    old = "".join(f"case {k}: wgs_seg<PB, {k}>{seg}" for k in (5, 6, 7)) + f"default: wgs_seg<PB, 8>{seg}"
    for a, b in (("#define OKC_FT_T 64\n", "#define OKC_FT_T 32\n"), (old, f"default: wgs_seg<PB, 4>{seg}")):
        if s.count(a) != 1:
            return None
        s = s.replace(a, b)
    return s
_HYBX["32"] = {"src": _hyb32_src(_HYB_MOE_CUDA), "mod": None, "ok": False, "pdl": -1}
def _hyb32_prepare(st, w, dev):
    if _HYBX["32"]["src"] is None:
        print("[hyb32] OFF: a source anchor of the FT_T = 32 transform is missing", flush=True)
        return
    _hybx_prepare("32", w, dev, HYB32_TESTS)
'''

A_DISPATCH = "    if _hyb_on(x.shape[0]):\n"
N_DISPATCH = ("    if HYB32_T[0] <= x.shape[0] <= HYB32_T[1] and _HYBX[\"32\"][\"ok\"]:\n"
              "        try:\n"
              "            return _hybx_call(\"32\", x, local_ids.to(torch.int32).contiguous(), topk_weights.to(torch.float32).contiguous())\n"
              "        except Exception as ex:\n"
              "            if not _HYBX[\"32\"].get(\"warned\"):\n"
              "                _HYBX[\"32\"][\"warned\"] = True\n"
              "                print(f\"[hyb32] route raised ({ex!r}): this call on VML / K2A\", flush=True)\n"
              + A_DISPATCH)
A_PT1 = '        jobs.append(("hyb", _hyb_build))\n'
N_PT1 = A_PT1 + '        if _HYBX["32"]["src"] is not None:\n            jobs.append(("hyb32", lambda: _hybx_build("32")))\n'
A_PREP = "    _hyb_prepare(st, w, dev)\n"
N_PREP = A_PREP + "    _hyb32_prepare(st, w, dev)\n"


def apply(s, lo, hi, base_name):
    lines = s.split(NL)
    lines[1] = (f"# nonce k3-wgs32-1 ({base_name} + R-wgs32: FT_T = 32 HYB module at T {lo}..{hi} ahead of VML; "
                f"tools/wgs32_patches.py)")
    s = NL.join(lines)
    s = hybx_common.ensure(s, sub)
    s = sub(s, hybx_common.ANCHOR, W32.replace("@LO@", str(lo)).replace("@HI@", str(hi)) + hybx_common.ANCHOR, "w32 block")
    s = sub(s, A_DISPATCH, N_DISPATCH, "dispatch")
    s = sub(s, A_PT1, N_PT1, "pt1 job")
    s = sub(s, A_PREP, N_PREP, "prepare hook")
    return s


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3k.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_wgs32_1.py")
    lo, hi = (int(v) for v in ARGS.get("t", "17,48").split(","))
    s = apply(base.read_bytes().decode("ascii"), lo, hi, base.stem)
    ast.parse(s)
    check(s.isascii(), "non-ASCII output")
    check(len(s.encode()) < LIMIT, f"size {len(s.encode()):,} >= {LIMIT:,}")
    if "check" in FLAGS:
        check(out.read_bytes().decode("ascii") == s, f"{out.name} != apply({base.name})")
        print(f"[wgs32_patches] --check PASS: {out.name} == apply({base.name}) ({len(s):,} B)")
        return
    out.write_bytes(s.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"[wgs32_patches] {base.name} -> {out.name} ({len(s):,} B, +{len(s) - base.stat().st_size:,} B), HYB32_T ({lo}, {hi})")


if __name__ == "__main__":
    main()

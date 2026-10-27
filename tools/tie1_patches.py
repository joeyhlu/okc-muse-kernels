"""tie1_patches.py: TIE1 = tie-tolerant top-8 checks in the v8 tail's prepare-time self-tests ('cat' / 'sep' and router32).

Why (judge check cd12f3439b3f4496, -11 %): the judge draws a router per run (reference.weights_fields, secret seed). The
self-tests required the kernel's top-8 expert set to EQUAL the reference set on fixed random tokens. When a token's 8th and
9th biased scores are closer than the router's arithmetic error (bf16 hi + lo halves, FP32 accumulation), the sets may
legitimately differ; the self-test then raised, _v8_tail_prepare returned None and the whole run lost the tail (router32,
DG1 'cat', G2-A, the K2A PDL chain). tools/local/tail_tie_sim.py: 7 of 1000 router seeds (0.7 %) failed that way.

Check after TIE1 (_tie1_compare), per self-test case, with s = sigmoid(0.1767766952966369 * L64) + bias in FP64 from the FP64
logits L64:
  * the kernel's ids are 8 distinct in-range experts and a top-8 of s up to TIE1_GAP: every chosen score >= every unchosen
    score - TIE1_GAP (every token);
  * on DECIDABLE tokens (8th - 9th gap of s > TIE1_GAP) the set equals the reference routing's exactly;
  * the logits the kernel selected from move s by less than TIE1_GAP / 4 at every entry (measured here: 'cat' = the hi + lo
    GEMM's logits, 'sep' = the FP32 reference logits, router32 = its partials' sum). With |ds| <= d everywhere a top-8 set can
    change only where the exact gap is <= 2 d < TIE1_GAP / 2, so excluding gaps <= TIE1_GAP is sound, and the bound is
    enforced, not assumed;
  * topk weights (w maxerr < 1e-5, unchanged) and local ids are compared on the tokens whose sets match; local ids must
    also equal the local-id table of the kernel's own ids on every token.
TIE1_GAP = 2e-5: the router's score error is <= 0.0442 x the logit error (d sigmoid(m z) / dz <= m / 4, m = 0.17678); the
'cat' logits err rms ~7e-6 (rel 2.7-4.2e-6 of |logits| ~2); tail_tie_sim.py's worst |ds| over >= 2000 router seeds is
printed with the proof run (see notes/lab.md).
Only prepare-time self-test acceptance changes; no kernel, route or numeric of a timed call changes.

    python -I tools/tie1_patches.py in.py out.py
"""
import pathlib
import py_compile
import sys

HELPER = '''TIE1_GAP = 2e-5   # TIE1: biased-score tie tolerance of the top-8 self-tests (tools/tie1_patches.py)


def _tie1_compare(tid, tw, lid, ref_tid, ref_tw, ref_lid, l64, bias, lsel, table, ls):
    """TIE1: (ids_ok, werr, lid_ok) for a top-8 self-test case, tolerant of FP64 near-ties (tools/tie1_patches.py)."""
    s64 = torch.sigmoid(l64.double() * 0.1767766952966369) + bias.double()
    T, E = s64.shape
    tid, rt = tid.long(), ref_tid.long()
    if tid.shape != rt.shape or not bool(((tid >= 0) & (tid < E)).all()):
        return False, float("inf"), False
    chosen = torch.zeros((T, E), dtype=torch.bool, device=s64.device).scatter_(1, tid, True)
    inf = torch.full((), float("inf"), dtype=s64.dtype, device=s64.device)
    cmin = torch.where(chosen, s64, inf).min(1).values
    umax = torch.where(chosen, -inf, s64).max(1).values
    ok = bool((chosen.sum(1) == tid.shape[1]).all()) and bool((cmin >= umax - TIE1_GAP).all())
    v = s64.topk(tid.shape[1] + 1, dim=-1).values
    dec = (v[:, -2] - v[:, -1]) > TIE1_GAP
    same = (torch.sort(tid, 1).values == torch.sort(rt, 1).values).all(1)
    ok = ok and bool(same[dec].all())
    ds = (torch.sigmoid(lsel.double() * 0.1767766952966369) + bias.double() - s64).abs().max()
    ok = ok and float(ds) < TIE1_GAP / 4
    werr = 0.0
    if bool(same.any()):
        o_g, o_r = torch.argsort(tid[same], 1), torch.argsort(rt[same], 1)
        werr = float((tw[same].float().gather(1, o_g) - ref_tw[same].float().gather(1, o_r)).abs().max())
    lid_ok = True
    if lid is not None:
        own = table[(tid + N_EXPERTS * ls).reshape(-1)].view(T, -1)
        lid_ok = bool(torch.equal(torch.sort(lid.long(), 1).values, torch.sort(own.long(), 1).values))
        lid_ok = lid_ok and bool(torch.equal(torch.sort(lid[same].long(), 1).values,
                                             torch.sort(ref_lid[same].long(), 1).values))
    return ok, werr, lid_ok


'''

REGION = ("def _v8_tail_prepare(weights, state):\n", "\nV8_K2A3_ROUTE = ")
PATCHES = [
    ("helper", None, "def _v8_tail_prepare(weights, state):\n", HELPER + "def _v8_tail_prepare(weights, state):\n"),
    ("cat/sep fp64 logits", REGION,
     "            ref_logits = (x.double() @ router.double().T).float()\n",
     "            ref_l64 = x.double() @ router.double().T\n            ref_logits = ref_l64.float()\n"),
    ("cat/sep compare", REGION,
     "                ids_eq = bool(torch.equal(torch.sort(tid, 1).values, torch.sort(ref_tid.to(torch.int32), 1).values))\n"
     "                o_g, o_r = torch.argsort(tid, 1), torch.argsort(ref_tid.to(torch.int32), 1)\n"
     "                werr = float((tw.gather(1, o_g) - ref_tw.float().gather(1, o_r)).abs().max())\n"
     "                lid_eq = bool(torch.equal(torch.sort(lid, 1).values, torch.sort(ref_lid.to(torch.int32), 1).values))\n",
     "                ids_eq, werr, lid_eq = _tie1_compare(tid, tw, lid, ref_tid, ref_tw, ref_lid, ref_l64, v8[\"bias\"],\n"
     "                                                     lg if kind == \"cat\" else ref_logits,\n"
     "                                                     state[\"local_id_table\"], ls)\n"),
    ("router32 fp64 logits", REGION,
     "            ref_logits = (h.double() @ router.double().T).float()\n",
     "            ref_l64 = h.double() @ router.double().T\n            ref_logits = ref_l64.float()\n"),
    ("router32 compare", REGION,
     "            ids_eq = bool(torch.equal(torch.sort(got[2], 1).values, torch.sort(ref_tid.to(torch.int32), 1).values))\n"
     "            o_g, o_r = torch.argsort(got[2], 1), torch.argsort(ref_tid.to(torch.int32), 1)\n"
     "            werr = float((got[1].gather(1, o_g) - ref_tw.float().gather(1, o_r)).abs().max())\n",
     "            ids_eq, werr, _ = _tie1_compare(got[2], got[1], None, ref_tid, ref_tw, None, ref_l64, v8[\"bias\"],\n"
     "                                            part.sum(0), None, ls)\n"),
]


def apply(text, patches=PATCHES):
    for name, region, old, new in patches:
        a, b = 0, len(text)
        if region is not None:
            a = text.index(region[0])
            b = text.index(region[1], a)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"tie1 patch {name!r}: `old` occurs {n} times in its scope (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text


if __name__ == "__main__":
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    text = src.read_text(encoding="utf-8")
    out = apply(text)
    dst.write_text(out, encoding="utf-8", newline="\n")
    py_compile.compile(str(dst), doraise=True)
    print(f"tie1: {len(PATCHES)} patches applied once each: {src.name} -> {dst} ({len(out) - len(text):+d} chars, compiles)")

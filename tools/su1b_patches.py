"""su1b_patches.py: SU1 follow-ups on a layer that already carries SU1 (tools/su1_patches.py).

1. K3 grid = 4 T on calls whose K3 triggers early (SU1_K3_GRID). K2A launches K3 with 2 x SMs = 264 CTAs at every T although it
   has only 4 T items; PDL releases the dependent grid only once EVERY K3 CTA has executed launch_dependents, i.e. once all 264
   are resident, which waits for K2 to drain. With 4 T CTAs (4..32 at T <= 8) K3 is resident at once and the dependent (SU1 /
   P1M unpooler) launches during K2. Same item -> CTA map (one item per CTA), so K3's output is unchanged.
2. K3 early trigger also at the decode 16 / 32 buckets (K2A v3.6 calls), where the unpooler is P1M (PDL, weights before its
   wait): SU1_K3_TRIG_AT {bucket: "su1" | "p1m"}; "p1m" only when the P1M unpooler is routed, live and PDL-launched.

    python -I tools/su1b_patches.py in.py out.py
"""
import pathlib
import py_compile
import sys

HELPERS = r'''SU1_K3_GRID = True   # K3 grid = 4 T (its item count) on calls whose K3 triggers early, so all its CTAs are resident at once (read at call time)
SU1_K3_TRIG_AT = {1: "su1", 8: "su1", 16: "p1m", 32: "p1m"}   # bucket -> K3 early trigger when SU1 / the P1M unpooler (PDL) follows (read at call time)


def _su1_k3trig(state, tokens):
    """Does this call's K3 trigger its dependents at entry? Only when the kernel right after K3 is SU1 or the PDL-launched P1M
    unpooler (both issue only prepare-time weight loads before their wait). Shape / config only."""
    if not (SU1_ENABLE and SU1_K3_TRIG):   # SU1_ENABLE is the master switch of every explorer lever (False == k3_v34)
        return False
    rep = _k2a_bucket_of(tokens)
    mode = SU1_K3_TRIG_AT.get(rep)
    if mode == "su1":
        return _su1_live(state, "unpooler", tokens)
    if mode == "p1m":
        st = state.get("p1m")
        return bool(V20_P1M_ENABLE and V20_P1M_PDL and st is not None and ("unpooler", rep) in V20_P1M_ROUTE
                    and st["route"].get(("unpooler", rep)) is not None and 8 < tokens <= 32)
    if mode == "p1":   # the T <= 8 P1 unpooler, PDL-launched by G2-B (it issues only weight loads before its wait)
        return bool(1 < tokens <= 8 and not _su1_live(state, "unpooler", tokens) and P1_ENABLE and P1_PROJ.get("unpooler")
                    and P1_ROUTE.get(("unpooler", 8)) and _G2_LIVE.get("p1pdl") and G2_P1_PDL.get("unpooler"))
    return False


def _su1_g3(st, m):
    """K3's grid: 4 m (one item per CTA) when this call's K3 triggers early, else K2A's 2 x SMs."""
    return max(4 * m, 1) if (SU1_K3_GRID and _SU1_FLAG["k3trig"]) else st["sms"] * 2


'''

K2A33CALL = ("def _k2a33_call(st, data, vid, probe=K2A33_HINT, mask=7, hot_t=None, hot_w4=None):", "\ndef _k2a33_reset(st):")
K2A36CALL = ("def _k2a36_call(st, data, vid, hot=None, probe=K2A33_HINT, mask=7):", "\ndef _v14_self_test_inputs(")
FN = ("def kernel_fn(data, state):", "\n    return next_attn_input")
ANCHOR = "def _su1_prepare(state, t0):\n"


def patches():
    return [
        ("PY: helpers", (ANCHOR, None), ANCHOR, HELPERS + ANCHOR),
        ("K2A v3.3 call: K3 grid", K2A33CALL,
         'st["grid"][vid], st["sms"] * 2, probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG["k3trig"] else 0))',
         'st["grid"][vid], _su1_g3(st, m), probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG["k3trig"] else 0))'),
        ("K2A v3.6 call: K3 grid + probe bit 25", K2A36CALL,
         'st["grid"][vid], st["sms"] * 2, probe | ((1 << 24) if K2A47_K1_PDL else 0))',
         'st["grid"][vid], _su1_g3(st, m), probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG["k3trig"] else 0))'),
        ("FN: K3 trigger decision", FN,
         '    _SU1_FLAG["k3trig"] = bool(SU1_K3_TRIG and _su1_live(state, "unpooler", tokens))   # SU1: K3 triggers its dependents early\n',
         '    _SU1_FLAG["k3trig"] = _su1_k3trig(state, tokens)   # SU1 / P1M unpooler next: K3 triggers its dependents early\n'),
    ]


def apply(text):
    if "SU1_ENABLE" not in text or "SU1_K3_GRID" in text:
        raise RuntimeError("su1b: needs a layer with SU1 and without su1b")
    for what, scope, old, new in patches():
        if text.count(scope[0]) != 1:
            raise RuntimeError(f"su1b: scope start {scope[0]!r} occurs {text.count(scope[0])} times")
        a = text.index(scope[0])
        b = len(text) if scope[1] is None else text.find(scope[1], a + len(scope[0]))
        b = len(text) if b < 0 else b
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"su1b patch {what!r}: `old` occurs {n} times in its scope (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text


if __name__ == "__main__":
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    t = apply(src.read_text(encoding="utf-8"))
    out.write_bytes(t.encode("utf-8"))
    py_compile.compile(str(out), doraise=True)
    print(f"{out}: {len(t.encode())} B, all {len(patches())} patches hit once, compiles")

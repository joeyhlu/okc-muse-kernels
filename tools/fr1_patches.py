"""fr1_patches.py: call-time switches for the cheap levers of notes/study/fresh_review_v37.md (request R15), on k3_v37's chain.

  * FR1_P1_GEO {(proj, bucket): (warps, split, depth)}: overrides P1's prepare-time geometry at call time (p1_gemv), e.g.
    the unpooler at T = 8 as (16, 4, 1) or the qkv at T = 1 as (4, 8, 1) / (8, 4, 1). Empty = k3_v37 (shipped geometry).
    Every override must satisfy the kernel's own check (split * warps <= K / 64, split <= 8, depth 1, warps 4 / 8 / 16).
  * FR1_SU1_FIRST: prepare runs _su1_prepare before _p206_prepare (review #13: SU1 is worth more per second than PR206 and
    both are gated near the end of a cold prepare). False = k3_v37's order. Prepare-time only: timed code is unchanged.
Everything else (SU1_ROUTE for the unpooler at T = 8, K1FA4_EX2 for the FA4 poly-exp2) is already a call-time switch.

    python -I tools/gen_k3_v37.py --fr1   -> kernels/lab/k3_v37_fr1_ab.py (k3_v37 stays byte-identical)
"""

GEO_DEF_OLD = 'P1_ROWS_T8 = {"unpooler": 1024}\n'
GEO_DEF_NEW = ('P1_ROWS_T8 = {"unpooler": 1024}\n'
               'FR1_P1_GEO = {}   # FR1 (tools/fr1_patches.py): call-time P1 geometry overrides {(proj, bucket): (warps, split, depth)}\n')
GEO_USE_OLD = '        geo = st["geo"][(proj, p1_bucket(T))]\n'
GEO_USE_NEW = '        geo = FR1_P1_GEO.get((proj, p1_bucket(T))) or st["geo"][(proj, p1_bucket(T))]   # FR1\n'
P206_HEAD = "\n    _p206_prepare(state, t0)"
SU1_HEAD = "\n    _su1_prepare(state, t0)"
ORDER_FLAG = ("FR1_SU1_FIRST = True   # FR1 (tools/fr1_patches.py): prepare builds / self-tests SU1 before PR206 "
              "(False = k3_v37's order)\n")


def _fail(msg):
    raise RuntimeError(f"fr1 patch: {msg}")


def _line(text, head):
    if text.count(head) != 1:
        _fail(f"anchor {head.strip()!r} found {text.count(head)}x")
    i = text.index(head) + 1
    return i, text.index("\n", i) + 1


def apply(text):
    for old, new in ((GEO_DEF_OLD, GEO_DEF_NEW), (GEO_USE_OLD, GEO_USE_NEW)):
        if text.count(old) != 1:
            _fail(f"anchor {old.strip()!r} found {text.count(old)}x")
        text = text.replace(old, new)
    a0, a1 = _line(text, P206_HEAD)
    b0, b1 = _line(text, SU1_HEAD)
    if b0 != a1:
        _fail("_su1_prepare does not directly follow _p206_prepare")
    p206, su1 = text[a0:a1], text[b0:b1]
    block = "    if FR1_SU1_FIRST:\n    " + su1 + p206 + "    if not FR1_SU1_FIRST:\n    " + su1
    text = text[:a0] + block + text[b1:]
    anchor = "\ndef kernel_fn(data, state):\n"
    if text.count(anchor) != 1:
        _fail("kernel_fn anchor")
    return text.replace(anchor, "\n" + ORDER_FLAG + "\n" + anchor)

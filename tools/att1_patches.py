"""att1_patches.py: ATT1 = B = 1 full-attention handoff knobs on the FA4 chain (prologue -> FA4 split-KV -> combine) of a
layer file (k3_gc_7), plus an optional B1A copy with a lab-only phase-debug knob. Every knob defaults to the base behaviour.

Knobs (module level, read at call time; FA4 compile keys include them):
  ATT1_PRO_TF   True: the FA4 prologue fires griddepcontrol.launch_dependents BEFORE its griddepcontrol.wait (base: after),
                so the FA4 grid is launched while the qkv GEMV still runs (FA4's own griddepcontrol.wait still waits for
                the whole prologue, which waits for qkv: same data order).
  ATT1_PW       {(kind, batch): True}: FA4 producer issues page indices + first K / V tile before griddepcontrol.wait
                (K1FA4_PREWAIT, but scoped per shape); the CTA that owns the new keys waits first (base code path).
  ATT1_TRIG     {(kind, batch): "start" | "epi"}: FA4 launch_dependents placement per shape (base: K1FA4_TRIG[kind]).
  ATT1_COMB_NW  num_warps of the one-row combine at > 8 splits (0 = base: 4).
B1A (tools/b1a_patches.py, applied unchanged except): B1A_ROUTE default off, B1A_DBG (lab: 1 = exit after the merge,
2 = exit after compute, 3 = exit after the loads land; timing only, outputs garbage), carried in Params.sync bits 4+.
"""
NL = chr(10)

KNOBS = ("ATT1_PRO_TF = False" + NL + "ATT1_PW = {}" + NL + "ATT1_TRIG = {}" + NL + "ATT1_COMB_NW = 0" + NL)

EDITS = [
    # knobs next to the FA4 ones
    ('K1FA4_STATE = {"ready": None}' + NL,
     'K1FA4_STATE = {"ready": None}' + NL + KNOBS),
    # per-shape trigger override (compile time)
    ('        k1fa4_K["trig"] = K1FA4_TRIG[kind]' + NL,
     '        k1fa4_K["trig"] = k1fa4_K.get("trig_o") or K1FA4_TRIG[kind]' + NL),
    # per-shape prewait (compile time, producer load())
    ('        pre = k1fa4_K["pdl"] and K1FA4_PREWAIT' + NL,
     '        pre = k1fa4_K["pdl"] and (K1FA4_PREWAIT or k1fa4_K.get("pw", False))' + NL),
    ('    pv8 = bool(K1FA4_PV8.get((kind, q8.shape[1]), False))' + NL,
     '    pv8 = bool(K1FA4_PV8.get((kind, q8.shape[1]), False))' + NL +
     '    k1fa4_K["pw"] = bool(ATT1_PW.get((kind, B), False))' + NL +
     '    k1fa4_K["trig_o"] = ATT1_TRIG.get((kind, B))' + NL),
    ('tuple(sorted(K1FA4_L2PF.items())), K1FA4_PREWAIT)' + NL,
     'tuple(sorted(K1FA4_L2PF.items())), K1FA4_PREWAIT, k1fa4_K["pw"], k1fa4_K["trig_o"])' + NL),
    # prologue: trigger-first option
    ('                    BLOCK_M: tl.constexpr, HAS_ROPE: tl.constexpr, USE_PDL: tl.constexpr,' + NL +
     '                    FP8_MAX_C: tl.constexpr = 448.0):' + NL +
     '    if USE_PDL:' + NL +
     '        tl.extra.cuda.gdc_wait()' + NL +
     '        tl.extra.cuda.gdc_launch_dependents()' + NL,
     '                    BLOCK_M: tl.constexpr, HAS_ROPE: tl.constexpr, USE_PDL: tl.constexpr,' + NL +
     '                    FP8_MAX_C: tl.constexpr = 448.0, TF: tl.constexpr = False):' + NL +
     '    if USE_PDL:' + NL +
     '        if TF:' + NL +
     '            tl.extra.cuda.gdc_launch_dependents()' + NL +
     '        tl.extra.cuda.gdc_wait()' + NL +
     '        if not TF:' + NL +
     '            tl.extra.cuda.gdc_launch_dependents()' + NL),
    ('        HAS_ROPE=bool(theta), USE_PDL=pdl, num_warps=K1FA4_PRO_WARPS, **pdl_kw)' + NL +
     '    max_k = data["page_table"].shape[1]' + NL,
     '        HAS_ROPE=bool(theta), USE_PDL=pdl, num_warps=K1FA4_PRO_WARPS,' + NL +
     '        TF=bool(pdl and ATT1_PRO_TF and batch == 1 and not window), **pdl_kw)' + NL +
     '    max_k = data["page_table"].shape[1]' + NL),
    ('USE_PDL=pdl, num_warps=2 if ns <= 8 else 4, **pdl_kw)' + NL,
     'USE_PDL=pdl, num_warps=2 if ns <= 8 else (ATT1_COMB_NW or 4), **pdl_kw)' + NL),
]

B1A_EDITS = [
    ("B1A_ROUTE = {1: True, 8: True}" + NL, "B1A_ROUTE = {1: False, 8: False}" + NL),
    ("B1A_SYNC = 1" + NL, "B1A_SYNC = 1" + NL + "B1A_DBG = 0" + NL),
    ("int(B1A_SYNC), int(_B1A[\"nt\"])", "int(B1A_SYNC) | (int(B1A_DBG) << 4), int(_B1A[\"nt\"])"),
    ("  __syncthreads();   // the new-key unit is in shared memory" + NL,
     "  __syncthreads();   // the new-key unit is in shared memory" + NL +
     "  if ((p.sync >> 4) == 3) {   // lab: loads only" + NL +
     "    for (int i = 0; i < ntl; ++i) mbar_wait(bars + i, i < redo ? 1u : 0u);" + NL +
     "    return;" + NL +
     "  }" + NL),
    ("  // 4. merge: o 2^(m - 8)",
     "  if ((p.sync >> 4) == 2) {   // lab: loads + compute only (the store keeps the compute alive)" + NL +
     "    if (o[0][0] + la[0] == 1234.5f) p.out[0] = __float2bfloat16(0.f);" + NL +
     "    return;" + NL +
     "  }" + NL +
     "  // 4. merge: o 2^(m - 8)"),
    ("  // 5. barrier: every CTA arrives once",
     "  if ((p.sync >> 4) == 1) return;   // lab: no barrier / finalize" + NL +
     "  // 5. barrier: every CTA arrives once"),
]


def _rep(text, old, new, count=1):
    n = text.count(old)
    assert n == count, f"anchor found {n} times (want {count}): {old[:120]!r}"
    return text.replace(old, new)


def apply(text):
    assert "ATT1_PRO_TF = " not in text, "the base already carries ATT1"
    for old, new in EDITS:
        text = _rep(text, old, new)
    return text


def b1a_block_edits(block):
    for old, new in B1A_EDITS:
        block = _rep(block, old, new)
    assert block.count("if (p.sync == 0) __threadfence();") == 2
    return block.replace("if (p.sync == 0) __threadfence();", "if ((p.sync & 15) == 0) __threadfence();")


def set_defaults(text, sets):
    """sets: {NAME: python-literal-string}; each 'NAME = ...' module line must occur exactly once."""
    lines = text.split(NL)
    for k, v in sets.items():
        idx = [i for i, ln in enumerate(lines) if ln.startswith(f"{k} = ")]
        assert len(idx) == 1, f"{k}: {len(idx)} definitions"
        lines[idx[0]] = f"{k} = {v}"
    return NL.join(lines)

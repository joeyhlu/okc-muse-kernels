"""vfa2_patches.py: VFA2-PW, the PDL entry of the verify-full VFA kernel (B >= 2, q_len 8, full attention).

Before: _vfa_run launches the FA4 prologue (RoPE / q8 / new K-V rows) without trigger-first, and vfa_main waits
(griddepcontrol.wait) for the whole prologue before the role split, so VFA starts streaming only after the prologue ends.
VFA2-PW (VFA_PW, module global, read per call):
  * the prologue is launched with TF (launch_dependents before its gdc_wait), so VFA's CTAs land while it runs;
  * kernel mode p.pdl == 2: the producer streams tiles of strictly old keys (< seq_b, never written in this layer) before
    the wait and calls griddepcontrol.wait before the first tile that holds a key >= seq_b (the block's 8 new keys, written
    by the prologue): the owner of new keys always waits; page-table / seq_lens / segment reads are inputs;
  * the consumer waits at its entry, before reading q8 (and before every global write: partials, tickets, finalize).
Mode 1 (VFA_PW False) is the old entry wait. Same ATT1_PW pattern that won at B = 1.
"""
NL = chr(10)


def _one(t, old, new, what):
    assert t.count(old) == 1, f"vfa2_patches: anchor '{what}' found {t.count(old)}x"
    return t.replace(old, new)


def apply_pw(t, default=True):
    t = _one(t, "VFA_PDL = True" + NL,
             "VFA_PDL = True" + NL + f"VFA_PW = {bool(default)}   # VFA2-PW: prologue trigger-first + VFA producer pre-wait on old-key tiles" + NL,
             "VFA_PDL")
    t = _one(t, "  const uint32_t so = sa(smem) + r0 * ROWB + ((ch ^ (r0 & 7)) << 4);" + NL + "  int n = 0;" + NL,
             "  const uint32_t so = sa(smem) + r0 * ROWB + ((ch ^ (r0 & 7)) << 4);" + NL + "  int n = 0;" + NL
             + "  bool wpend = (p.pdl == 2);   // VFA2-PW: griddepcontrol.wait still owed (taken before the first new-key tile)" + NL,
             "producer n")
    t = _one(t, "    const int* pt = p.page_table + (long long)b * p.pt_stride;" + NL,
             "    const int* pt = p.page_table + (long long)b * p.pt_stride;" + NL
             + "    const int sq = __ldg(p.seq_lens + b);   // VFA2-PW: tiles x >= xw hold keys >= sq (the prologue's new rows)" + NL
             + "    const int xw = (k1 > sq) ? max(0, (sq - k0) / TILE) : nt;" + NL,
             "producer pt")
    t = _one(t, "      const int st = n % ST;" + NL + "      if (n >= ST) mbar_wait(empt + st,",
             "      if (wpend && x >= xw) { pdl_wait(); wpend = false; }   // VFA2-PW: the owner of new keys waits" + NL
             + "      const int st = n % ST;" + NL + "      if (n >= ST) mbar_wait(empt + st,",
             "producer st")
    t = _one(t, "  if (p.pdl) pdl_wait();   // q8 and the block's cache rows come from the FA4 prologue" + NL,
             "  if (p.pdl == 1) pdl_wait();   // q8 and the block's cache rows come from the FA4 prologue (VFA2-PW mode 2: per role)" + NL,
             "vfa_main wait")
    t = _one(t, "    consumer<ST, DIAG>(p, smem, seg, sN, nseg, full, empt, misc, tid, pair, pairs);" + NL,
             "    if (p.pdl == 2) pdl_wait();   // VFA2-PW: q8 and every global write after the prologue" + NL
             + "    consumer<ST, DIAG>(p, smem, seg, sN, nseg, full, empt, misc, tid, pair, pairs);" + NL,
             "consumer call")
    t = _one(t, 'num_warps=K1FA4_PRO_WARPS, **pdl_kw)' + NL + '    ws = _VFA["ws"]',
             'num_warps=K1FA4_PRO_WARPS, TF=bool(pdl and VFA_PW), **pdl_kw)' + NL + '    ws = _VFA["ws"]',
             "vfa prologue TF")
    t = _one(t, "int(VFA_SEGK if segk is None else segk), 1 if pdl else 0,",
             "int(VFA_SEGK if segk is None else segk), (2 if VFA_PW else 1) if pdl else 0,",
             "vfa pdl mode")
    return t

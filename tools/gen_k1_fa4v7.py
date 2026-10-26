"""Generate kernels/muse_attention_decode/k1_fa4v7.py (LAB copy): K1-FA4 v6 + K1-FA4 v4's FP8-PV core, selectable per
(layer, q_len) by K1FA4_PV8, + a lab knob K1FA4_EX2 for the softmax exp2 (0 = FA4's MUFU path, 1 = timing mask,
2 = partial software exp2). Plain source, no runtime patching (A19 REVISED), same build steps as tools/gen_k1_fa4v6.py.

  python tools/gen_k1_fa4v7.py            # writes k1_fa4v7.py, prints size + scan

Differences from gen_k1_fa4v6.py:
- vendored-text patches: v6's table with the V-stage patch replaced by k1fa4_sv_spec(self) (E4M3 Vt for the FP8 core,
  F16 V otherwise), sVt via k1fa4_vt_view, the PV tiled MMA's A/B types and B major mode via k1fa4_pv_dtype /
  k1fa4_pv_major_b, the two consumer P conversions via k1fa4_cvt_p, and the Softmax.create receiver via
  k1fa4_softmax_cls. Every helper reads k1fa4_K["pv8"] / k1fa4_K["ex2"] at trace time; with pv8 False and ex2 0 the
  kernel is v6's.
- tail: v6's tail (tools/gen_k1_fa4v6_tail.py.in) with K1FA4_PV8 / K1FA4_REGS1 / K1FA4_EX2 knobs, the class switch and
  the compile key in k1fa4_fwd_partials, plus tools/gen_k1_fa4v7_extra.py.in (helpers, the FP8 PV subclass
  k1fa4_Sm90FwdPV8 = K1-FA4 v4's producer + PDL wait, and the exp2 softmax subclass) after the v2 subclass.
"""
import ast
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_k1_fa4v6 as G6  # noqa: E402

NL = "\n"
OUT = G6.KDIR / "k1_fa4v7.py"
EXTRA = HERE / "gen_k1_fa4v7_extra.py.in"

V_STAGE = ("(mV.element_type, (self.tile_n, self.tile_hdimv), self.num_stages),",
           "(self.dtype, (self.tile_n, self.tile_hdimv), self.num_stages),")
PATCHES = [(a, ("k1fa4_sv_spec(self)," if (a, b) == V_STAGE else b), 1) for a, b in G6.PATCHES]
assert sum(1 for a, _, _ in PATCHES if a == V_STAGE[0]) == 1
PATCHES += [
    ("        sVt = layout_utils.transpose_view(sV)" + NL, "        sVt = k1fa4_vt_view(sV)" + NL, 1),
    ("        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(" + NL + "            self.dtype," + NL
     + "            self.dtype," + NL + "            warpgroup.OperandMajorMode.K," + NL
     + "            warpgroup.OperandMajorMode.MN," + NL,
     "        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(" + NL + "            k1fa4_pv_dtype(self.dtype),"
     + NL + "            k1fa4_pv_dtype(self.dtype)," + NL + "            warpgroup.OperandMajorMode.K," + NL
     + "            k1fa4_pv_major_b(warpgroup.OperandMajorMode.MN)," + NL, 1),
    ("utils.cvt_f16(tOrP_acc, tOrP_cur)", "k1fa4_cvt_p(acc_S, tOrP_acc, tOrP_cur)", 2),
    ("        softmax = Softmax.create(" + NL, "        softmax = k1fa4_softmax_cls(Softmax).create(" + NL, 1),
]

TAIL_PATCHES = [
    ('K1FA4_STAGES = {"full": 3, "sliding": 2}  # K/V pipeline depth per layer type' + NL,
     'K1FA4_STAGES = {"full": 3, "sliding": 2}  # K/V pipeline depth per layer type' + NL
     + '# (layer, q_len) -> FP8 PV core (k1fa4_Sm90FwdPV8, K1-FA4 v4) instead of F16 PV (k1fa4_Sm90Fwd, v2)' + NL
     + 'K1FA4_PV8 = {("full", 1): True, ("full", 8): True, ("sliding", 1): True, ("sliding", 8): True}' + NL
     + "K1FA4_REGS1 = {False: (200, 56), True: (184, 72)}  # setmaxnreg (consumer, producer) per core (pv8)" + NL
     + "K1FA4_P8_OFFSET = 8.0  # FP8 P max offset (stock FA3's Max_offset for FP8): P = exp2(s x - s m + 8); 0 = none (v4)" + NL
     + "# softmax exp2: mode 0 = FA4 (MUFU), 1 = timing mask (FFMA only; INVALID results, lab), 2 = degree-deg polynomial"
     + NL + "# on the FMA pipe for elements i % freq < res (FA4 Blackwell-style partial emulation), MUFU for the rest" + NL
     + 'K1FA4_EX2 = {"mode": 0, "freq": 8, "res": 1, "deg": 3}' + NL
     + "# lab: FP8-PV producer L2 hints: k / v = bulk L2 prefetch of the next tile's 128-B K / V rows (both KV heads)," + NL
     + "# vh = .L2::128B prefetch-size hint on the V gather" + NL
     + 'K1FA4_L2PF = {"k": False, "v": False, "vh": False}' + NL),
    ('    per SM, PDL wait in the producer."""' + NL + NL + "    def __init__(self, dtype, *args, **kwargs):" + NL
     + '        kind = "sliding" if kwargs.get("is_local") else "full"' + NL
     + '        k1fa4_K["stages"] = K1FA4_STAGES[kind]' + NL + '        k1fa4_K["trig"] = K1FA4_TRIG[kind]' + NL,
     '    per SM, PDL wait in the producer."""' + NL + "    k1fa4_pv8 = False" + NL + NL
     + "    def __init__(self, dtype, *args, **kwargs):" + NL
     + '        kind = "sliding" if kwargs.get("is_local") else "full"' + NL
     + '        k1fa4_K["stages"] = K1FA4_STAGES[kind]' + NL + '        k1fa4_K["trig"] = K1FA4_TRIG[kind]' + NL
     + '        k1fa4_K["pv8"] = self.k1fa4_pv8' + NL + '        k1fa4_K["regs1"] = K1FA4_REGS1[self.k1fa4_pv8]' + NL
     + '        k1fa4_K["ex2"] = K1FA4_EX2["mode"]' + NL),
    ('    key = (local, hq // hkv, d, K1FA4_STAGES[kind], K1FA4_TRIG[kind], k1fa4_K["pdl"], k1fa4_K["minb"],' + NL
     + '           k1fa4_K["regs1"])' + NL,
     "    pv8 = bool(K1FA4_PV8.get((kind, q8.shape[1]), False))" + NL
     + '    key = (local, hq // hkv, d, K1FA4_STAGES[kind], K1FA4_TRIG[kind], k1fa4_K["pdl"], k1fa4_K["minb"],' + NL
     + "           K1FA4_REGS1[pv8], pv8, K1FA4_P8_OFFSET, tuple(sorted(K1FA4_EX2.items())), tuple(sorted(K1FA4_L2PF.items())))"
     + NL),
    ("        fa = k1fa4_Sm90Fwd(cutlass.Float8E4M3FN, d, d, hq // hkv,",
     "        fa = (k1fa4_Sm90FwdPV8 if pv8 else k1fa4_Sm90Fwd)(cutlass.Float8E4M3FN, d, d, hq // hkv,"),
    ("# K1-FA4 v6 (team joeyhlu): FA4 v2 core (subclass of the vendored kernel above) + fused prologue / combine-norm-gate",
     "# K1-FA4 v7 (team joeyhlu): FA4 v2 / v4 cores (subclasses of the vendored kernel) + fused prologue / combine-norm-gate"),
]
EXTRA_ANCHOR = NL + NL + "def k1fa4_num_splits_heuristic("


def build(more_tail=(), post=(), more_extra="",
          title="K1-FA4 v7 attention (team joeyhlu, LAB). Generated by tools/gen_k1_fa4v7.py"):
    """The generated source text. more_tail: extra (anchor, new) patches on the v6 tail; more_extra: text appended to
    the extra block; post: (anchor, new) patches on the final text (each anchor exactly once)."""
    raw = G6.SRC.read_bytes()
    txt = raw.decode("utf-8").replace("\r\n", NL)
    sha = hashlib.sha256(raw).hexdigest()
    for a, b, n in PATCHES:
        assert txt.count(a) == n, ("anchor", a, txt.count(a), n)
        txt = txt.replace(a, b)
    txt = G6.drop_methods(txt)
    imports, body = G6.split_imports(txt)
    imp_lines, bound = G6.import_block(imports)
    mapping = {k: v for k, v in bound.items() if v != k}
    mapping[G6.CLASS] = G6.PREFIX + G6.CLASS
    G6.check_params(body, mapping)
    body, n_edits = G6.rename(body, mapping)
    assert "self.load_KV(" not in body and "self.load_Bias(" not in body and body.count("self.load(") == 1
    for name in ("k1fa4_sv_spec(self)", "k1fa4_vt_view(sV)", "k1fa4_pv_major_b(", "k1fa4_softmax_cls(k1fa4_Softmax)"):
        assert body.count(name) == 1, name
    assert body.count("k1fa4_cvt_p(acc_S, tOrP_acc, tOrP_cur)") == 2
    lic = G6.LICENSE.read_text(encoding="utf-8").replace("\r\n", NL).strip().split(NL)
    head = [
        "# " + "=" * 116,
        "# " + title + "; do not edit by hand.",
        "# Part 1 vendors FlashAttention-4's CuTe-DSL Hopper forward (FlashAttentionForwardSm90) from",
        "#   sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm90.py (lmsysorg/sglang:dev-muse-spark-open)",
        f"#   sha256 {sha}",
        "# with these source changes, applied at generation time: launch min_blocks_per_mp / use_pdl and the",
        "# 1-warpgroup setmaxnreg pair from k1fa4_K; K/V pipeline depth from k1fa4_K; V smem stage, Vt view, PV MMA",
        "# types, P conversion and softmax class through k1fa4_* helpers (FP8 PV core when k1fa4_K['pv8']); no 16-bit",
        "# type check; E4M3 QK tiled MMA; the shared-storage handle kept on self for the producer; an optional PDL",
        "# trigger before the epilogue. The producer methods load / load_KV / load_Bias are removed (the k1fa4_Sm90Fwd /",
        "# k1fa4_Sm90FwdPV8 load methods replace them). Module-level names carry a k1fa4_ prefix.",
        "# FlashAttention-4 license (BSD 3-Clause):",
    ] + ["#   " + ln if ln else "#" for ln in lic] + ["# " + "=" * 116]
    knobs = [
        'k1fa4_K = {"minb": 2, "regs1": (200, 56), "stages": 2, "pdl": True, "trig": False, "pv8": False, "ex2": 0}',
        "",
        "",
        "def k1fa4_skip_check(*_a, **_k):",
        '    """FA4 checks for 16-bit Q/K/V; this build checks its own contract in k1fa4_fwd_partials."""',
        "    return None",
        "",
        "",
    ]
    tail = G6.TAIL.read_text(encoding="utf-8").replace("\r\n", NL)
    for a, b in list(TAIL_PATCHES) + list(more_tail):
        assert tail.count(a) == 1, ("tail anchor", a[:80], tail.count(a))
        tail = tail.replace(a, b)
    extra = EXTRA.read_text(encoding="utf-8").replace("\r\n", NL) + more_extra
    assert tail.count(EXTRA_ANCHOR) == 1
    tail = tail.replace(EXTRA_ANCHOR, NL + extra.rstrip(NL) + EXTRA_ANCHOR)
    out = NL.join(head + imp_lines + ["", ""] + knobs) + body.rstrip(NL) + NL + NL + tail.lstrip(NL)
    for a, b in post:
        assert out.count(a) == 1, ("post anchor", a[:80], out.count(a))
        out = out.replace(a, b)
    ast.parse(out)
    return out, n_edits, mapping


def main():
    out, n_edits, mapping = build()
    OUT.write_bytes(out.encode("utf-8"))
    hits = G6.scan(out)
    long_lines = [i + 1 for i, ln in enumerate(out.split(NL)) if len(ln) > 400]
    print(f"[gen7] {OUT.name}: {len(out.encode())} bytes ({len(out.splitlines())} lines); {n_edits} renamed tokens; "
          f"{len(mapping)} renamed names; lines > 400 chars: {long_lines or 'none'}")
    print(f"[gen7] scan {OUT.name}: {'CLEAN' if not hits else hits}")
    return 0 if not hits else 1


if __name__ == "__main__":
    sys.exit(main())

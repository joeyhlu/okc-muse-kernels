"""ks2_patches.py: KS2 = KS1's one-launch sliding attention kernel on B > 1 (decode 16 / 32 sliding: T = 16 / 32, q_len 1; optionally
verify 16 / 32 sliding: q_len 8), replacing FA4 v8's prologue -> core -> combine chain there. Exact text patches against
kernels/muse_decode_layer/k3_v31.py (= v30 + tools/ks1_patches.py); every `old` occurs exactly once inside its scope.

  * kernel: grid z = B x NTG (request b = z / NTG); per-request cache_seqlens, page-table row, positions, cache_loc, qkv / out rows,
    tickets and partial slots (group = (b, KV head, token group)). B = 1 is the same arithmetic and launch as KS1 (b = 0).
  * host: B = cache_seqlens.numel(); shape checks scale with B; the same instantiations serve every batch (runtime grid), so the
    layer reuses KS1's build (KS1_INSTS gains the geometries KS2_GEOM names).
  * KS2_FOLD (B > 1): the new keys ride on split NS - 1 (its warp 7, second online-softmax segment) instead of a dedicated CTA,
    so a group is NS CTAs: decode 16 at NS 4 = 128 CTAs (one wave at 1 CTA/SM; the unfolded NS 8 = 288 CTAs needed two).
  * Python: KS2_ENABLE / KS2_ROUTE / KS2_GEOM / KS2_TRIG / KS2_FOLD keyed by (batch, q_len), read at call time (layer_ab2 arms);
    B = 1 keeps KS1's q_len-keyed tables. prepare runs _ks2_self_test (B in {3, 16, 32}, mixed lengths incl. < window, cache
    row 0): error <= 1.25 x r1's + 5e-3, new cache bytes == r1's, tickets back at zero, repeat with the other trigger /
    fence.sc protocol bit-identical, one graph replay identical; state["ks2"] = passing (q_len, NS, NTG). Failure: FA4 v8 stays.

Measured (box, layer_ab2 4 seeds x 4 pairs, check PASS): decode 16 sliding 148.8 -> 145.2 us (x1.026; NS 8 fold 145.9, unfolded 148.0);
decode 32 sliding 189.4 -> 200.0 (x0.94, not routed: B 32 needs >= 2 waves of single-shot CTAs -> a looping / 128-B-row design);
verify 16 / 32 sliding (unfolded) x0.93 / x0.87 (not routed); decode 1 / verify 1 unchanged (KS1).

    python -I tools/ks2_patches.py [k3_v31.py] [out.py]        # asserts every `old` occurs exactly once in its scope
    python3 tools/ks2_patches.py --build <patched layer>        # box: compile-only check of the patched KS1 source
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v31.py"

CU = ('KS1_CUDA_SRC = r"""', '"""')
PY = ("def _ks1_counters(dev):", "\ndef _ks1_prepare(")
PREP = ("def _ks1_prepare(state, t0):", "\n\n\n")
TABLES = ("KS1_ENABLE = True ", "\n_KS1 = {")

CALL_OLD = '''def _ks1_call(qkv, data, cos_sin, geom=None, trig=None, pdl=None, sync=None, phalf=None):
    """One KS1 launch on the layer's qkv [L, 2304] (B = 1): writes the new K / V cache rows, returns [L, 1024] bf16."""
    cfg = data["config"]
    L = cfg["q_len"]
    ns, ntg = geom if geom is not None else KS1_GEOM[L]
    trig = KS1_TRIG.get(L, 0) if trig is None else trig
'''
CALL_NEW = '''def _ks1_call(qkv, data, cos_sin, geom=None, trig=None, pdl=None, sync=None, phalf=None, fold=None):
    """One KS1 launch on the layer's qkv [B L, 2304]: writes the new K / V cache rows, returns [B L, 1024] bf16. B = 1: KS1's
    q_len-keyed tables; B > 1 (KS2): KS2_GEOM / KS2_TRIG keyed by (batch, q_len)."""
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    ns, ntg = geom if geom is not None else (KS1_GEOM[L] if B == 1 else KS2_GEOM[(B, L)])
    if trig is None:
        trig = KS1_TRIG.get(L, 0) if B == 1 else KS2_TRIG.get((B, L), 0)
    if fold is None:
        fold = False if B == 1 else KS2_FOLD.get((B, L), False)
'''
BUF_OLD = '''    po = torch.empty((2 * ntg * (ns + 1) * rows * 64,), dtype=torch.float32, device=dev)
    pml = torch.empty((2 * ntg * (ns + 1) * rows * 2,), dtype=torch.float32, device=dev)
    out = torch.empty((L, 16 * 64), dtype=torch.bfloat16, device=dev)
'''
BUF_NEW = '''    po = torch.empty((B * 2 * ntg * (ns + 1) * rows * 64,), dtype=torch.float32, device=dev)
    pml = torch.empty((B * 2 * ntg * (ns + 1) * rows * 2,), dtype=torch.float32, device=dev)
    out = torch.empty((B * L, 16 * 64), dtype=torch.bfloat16, device=dev)
'''
WANT_OLD = '''    L = cfg["q_len"]
    if not (cfg["batch"] == 1 and KS1_ROUTE.get(L) and cfg["window"] == 2048 and cfg["rope_theta"] == ROPE_THETA
            and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0 and cfg["head_dim"] == 64
            and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2 and (L, *KS1_GEOM.get(L, (0, 0))) in state["ks1"]):
        return False
'''
WANT_NEW = '''    B, L = cfg["batch"], cfg["q_len"]
    if B == 1:
        geo, passed = (KS1_GEOM.get(L) if KS1_ROUTE.get(L) else None), state.get("ks1") or set()
    else:   # KS2: (batch, q_len)-keyed route, geometries self-tested on B > 1
        geo, passed = (KS2_GEOM.get((B, L)) if (KS2_ENABLE and KS2_ROUTE.get((B, L))) else None), state.get("ks2") or set()
    if not (geo is not None and (L, *geo) in passed and cfg["window"] == 2048 and cfg["rope_theta"] == ROPE_THETA
            and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0 and cfg["head_dim"] == 64
            and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2):
        return False
'''

KS2_TABLES = '''
KS2_ENABLE = True                    # KS2: KS1's kernel on B > 1 sliding layers (read at call time)
KS2_ROUTE = {(16, 1): True, (32, 1): False, (16, 8): False, (32, 8): False}   # (batch, q_len) -> KS2 (box_out/ks2/ab2.log: dec16 x1.026,
#   dec32 x0.94 (2 waves), verify 16 / 32 x0.93 / x0.87 without fold; read at call time)
KS2_GEOM = {(16, 1): (4, 1), (32, 1): (4, 1), (16, 8): (16, 4), (32, 8): (16, 4)}   # (batch, q_len) -> (NS, NTG), in KS1_INSTS
KS2_TRIG = {(16, 1): 1, (32, 1): 1, (16, 8): 1, (32, 8): 1}   # (batch, q_len) -> launch_dependents mode (as KS1_TRIG)
KS2_FOLD = {(16, 1): True, (32, 1): True, (16, 8): True, (32, 8): True}   # new keys on split NS - 1 (NS CTAs per group, not NS + 1)
# (batch, ctx, q_len, seed, row0): lengths drawn in [ctx / 2, ctx], i.e. some requests shorter than the window
KS2_SELF_TEST_CASES = ((16, 3000, 1, 941, False), (32, 2600, 1, 942, True), (3, 2100, 1, 943, False))   # q_len 8 at B > 1: unrouted'''

KS2_TEST = '''def _ks2_self_test(cos_sin, device, geoms, verbose=False):
    """KS2 (B > 1): per KS2_SELF_TEST_CASES case and geometry of its q_len, the KS1 checks (rel L2 vs the FP32 reference <= 1.25 x
    the r1 path's + 5e-3, finite, new cache bytes == r1's, tickets back at zero, a call with the other trigger / fence.sc tickets
    bit-identical); then one graph replay per q_len (KS2_GEOM[(16, L)]) identical to the eager call. Returns passing
    (q_len, NS, NTG)."""
    cnt = _ks1_counters(device)
    bad, seen = set(), set()
    for batch, ctx, length, seed, row0 in KS2_SELF_TEST_CASES:
        mine = [g for g in geoms if g[0] == length and g not in bad]
        if not mine:
            continue
        T = batch * length
        ref = _k1fa4_reference(_k1fa4_case(batch, ctx, length, 2048, seed, row0, device)).view(T, -1)
        d1 = _k1fa4_case(batch, ctx, length, 2048, seed, row0, device)
        o1 = _k1fa4_r1_path(d1).float().view(T, -1)
        torch.cuda.synchronize(device)
        e1 = float((o1 - ref).norm() / ref.norm())
        rows = d1["cache_loc"]
        k1b = d1["k_cache"].view(-1, 128).view(torch.uint8)[rows]
        v1b = d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]
        for geo in mine:
            dk = _k1fa4_case(batch, ctx, length, 2048, seed, row0, device)
            qkv = dk["qkv"].view(T, -1)
            o_a = _ks1_call(qkv, dk, cos_sin, geo[1:], 0, KS1_PDL).clone()
            o_b = _ks1_call(qkv, dk, cos_sin, geo[1:], 1, KS1_PDL, 0)
            o_c = _ks1_call(qkv, dk, cos_sin, geo[1:], 0, KS1_PDL, fold=not KS2_FOLD.get((batch, length), False)).clone()
            torch.cuda.synchronize(device)
            ek = max(float((o_a.float() - ref).norm() / ref.norm()), float((o_c.float() - ref).norm() / ref.norm()))
            kb = dk["k_cache"].view(-1, 128).view(torch.uint8)[rows]
            vb = dk["v_cache"].view(-1, 128).view(torch.uint8)[rows]
            nbad = int((kb != k1b).sum()) + int((vb != v1b).sum())
            tick = int(cnt.abs().sum())
            same = bool(torch.equal(o_a, o_b))
            ok = bool(torch.isfinite(o_a).all()) and ek <= 1.25 * e1 + 5e-3 and nbad == 0 and tick == 0 and same
            seen.add(geo)
            if not ok:
                bad.add(geo)
            if verbose or not ok:
                print(f"[ks2] self-test B{batch} L{length} NS{geo[1]} NTG{geo[2]} ctx{ctx}{' row0' if row0 else ''}: "
                      f"{'PASS' if ok else 'FAIL'} err {ek:.4e} (r1 {e1:.4e}) cache bytes != r1 {nbad} tickets {tick} "
                      f"repeat identical {same}", flush=True)
            del dk, o_a, o_b, o_c
    ok_geo = {g for g in seen if g not in bad}
    for length in sorted({g[0] for g in ok_geo}):
        geo = KS2_GEOM.get((16, length))
        if geo is None or (length, *geo) not in ok_geo:
            continue
        d = _k1fa4_case(16, 3000, length, 2048, 950 + length, False, device)
        qkv = d["qkv"].view(16 * length, -1)
        o1 = _ks1_call(qkv, d, cos_sin, geo).clone()
        torch.cuda.synchronize(device)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            og = _ks1_call(qkv, d, cos_sin, geo)
        gr.replay()
        gr.replay()
        torch.cuda.synchronize(device)
        same = bool(torch.equal(og, o1)) and int(cnt.abs().sum()) == 0
        print(f"[ks2] graph check B16 L{length} NS{geo[0]} NTG{geo[1]}: replay identical {same}", flush=True)
        del gr, og
        if not same:
            ok_geo = {x for x in ok_geo if x[0] != length}
    print(f"[ks2] geometries ok on B > 1: {sorted(ok_geo)} (failed {sorted(bad)})", flush=True)
    return ok_geo


def _ks1_prepare('''

PATCHES = [
    # ---- CUDA kernel: request index b from grid z
    ("CU: Params.B", CU, "  int pt_cols;\n", "  int pt_cols;\n  int B;                      // KS2: requests (grid z = B x NTG)\n"),
    ("CU: block indices", CU,
     "  const int s = blockIdx.x, h = blockIdx.y, tg = blockIdx.z;\n  const int grp = h * NTG + tg;\n"
     "  const int cta = (tg * HKV + h) * (NS + 1) + s;\n",
     "  const int s = blockIdx.x, h = blockIdx.y, b = (int)blockIdx.z / NTG, tg = (int)blockIdx.z % NTG;   // KS2: z = request x group\n"
     "  const int grp = (b * HKV + h) * NTG + tg;\n"
     "  const int cta = ((int)blockIdx.z * HKV + h) * (NS + 1) + s;\n"
     "  const int tb = b * L;                                         // KS2: first token of request b\n"),
    ("CU: seq of request b", CU, "__ldg(p.cache_seqlens) - L", "__ldg(p.cache_seqlens + b) - L"),
    ("CU: page-table row b", CU, "__ldg(p.page_table + kstart + gs)", "__ldg(p.page_table + (size_t)b * p.pt_cols + kstart + gs)"),
    ("CU: q positions", CU, "__ldg(p.positions + tg * TPG + j)", "__ldg(p.positions + tb + tg * TPG + j)"),
    ("CU: new-key position", CU, "    const long long pos = __ldg(p.positions + warp);\n", "    const long long pos = __ldg(p.positions + tb + warp);\n"),
    ("CU: cache_loc", CU, "    loc = __ldg(p.cache_loc + warp);\n", "    loc = __ldg(p.cache_loc + tb + warp);\n"),
    ("CU: q rows", CU, "p.qkv + (size_t)(tg * TPG + j) * sq", "p.qkv + (size_t)(tb + tg * TPG + j) * sq"),
    ("CU: new k row", CU, "p.qkv + (size_t)warp * sq + (size_t)(HQ + h) * 64", "p.qkv + (size_t)(tb + warp) * sq + (size_t)(HQ + h) * 64"),
    ("CU: new v row", CU, "p.qkv + (size_t)warp * sq + (size_t)(HQ + HKV + h) * 64",
     "p.qkv + (size_t)(tb + warp) * sq + (size_t)(HQ + HKV + h) * 64"),
    ("CU: gate row", CU, "p.qkv + (size_t)(tg * TPG + (mr >> 3)) * sq", "p.qkv + (size_t)(tb + tg * TPG + (mr >> 3)) * sq"),
    ("CU: out row", CU, "p.out + (size_t)(tg * TPG + (mr >> 3)) * (HQ * 64)", "p.out + (size_t)(tb + tg * TPG + (mr >> 3)) * (HQ * 64)"),
    ("CU: grid z", CU, "cfg.gridDim = dim3((unsigned)(NS + 1), (unsigned)HKV, (unsigned)NTG);",
     "cfg.gridDim = dim3((unsigned)(NS + 1), (unsigned)HKV, (unsigned)(NTG * P.B));"),
    # ---- host checks scale with B = cache_seqlens.numel()
    ("CU: host B", CU, '  TORCH_CHECK(qkv.size(0) == L && qkv.size(1)',
     '  const int64_t B = cache_seqlens.numel();\n  TORCH_CHECK(B >= 1 && qkv.size(0) == B * L && qkv.size(1)'),
    ("CU: host seqlens", CU, 'cache_seqlens.numel() == 1, "ks1: B = 1 int32 cache_seqlens"', 'cache_seqlens.is_contiguous(), "ks1: int32 cache_seqlens"'),
    ("CU: host cache_loc", CU, "cache_loc.numel() == L &&", "cache_loc.numel() == B * L &&"),
    ("CU: host positions", CU, "positions.numel() == L &&", "positions.numel() == B * L &&"),
    ("CU: host page_table", CU, "page_table.size(0) == 1 &&", "page_table.size(0) == B &&"),
    ("CU: host part_o", CU, "part_o.numel() >= 2 * NTG", "part_o.numel() >= B * 2 * NTG"),
    ("CU: host part_ml", CU, "part_ml.numel() >= 2 * NTG", "part_ml.numel() >= B * 2 * NTG"),
    ("CU: host counters", CU, "counters.numel() >= 2 * NTG,", "counters.numel() >= B * 2 * NTG,"),
    ("CU: host out", CU, "out.numel() == L * ks1::HQ * 64", "out.numel() == B * L * ks1::HQ * 64"),
    ("CU: host prof", CU, "prof.numel() >= 16 * 2 * NTG * (NS + 1)", "prof.numel() >= 16 * B * 2 * NTG * (NS + 1)"),
    ("CU: host P.B", CU, "  P.pt_cols = (int)page_table.size(1);\n", "  P.pt_cols = (int)page_table.size(1);\n  P.B = (int)B;\n"),
    # ---- Python glue
    ("PY: KS2 tables + instantiations", TABLES,
     "KS1_INSTS = ((1, 16, 1), (8, 16, 4))   #",
     "KS1_INSTS = ((1, 16, 1), (8, 16, 4), (1, 4, 1))   # KS2: + (1, 4, 1) for decode 16 (lab: (1, 8, 1), (8, 8, 4));" + KS2_TABLES + "\n#"),
    ("PY: counters for B x 2 x NTG groups", PY, "torch.zeros(64, dtype=torch.int32, device=dev)", "torch.zeros(1024, dtype=torch.int32, device=dev)"),
    ("PY: _ks1_call batch", PY, CALL_OLD, CALL_NEW),
    ("PY: _ks1_call buffers", PY, BUF_OLD, BUF_NEW),
    ("PY: _ks1_wanted batch", PY, WANT_OLD, WANT_NEW),
    ("PY: _ks1_wanted shapes", PY, "and qkv.stride(1) == 1 and qkv.shape[0] == L\n", "and qkv.stride(1) == 1 and qkv.shape[0] == B * L\n"),
    ("PY: _ks1_wanted seqlens", PY, 'data["cache_seqlens"].numel() == 1\n', 'data["cache_seqlens"].numel() == B\n'),
    ("PY: KS1 tickets (all groups)", PY, "            tick = int(cnt[:16].abs().sum())\n", "            tick = int(cnt.abs().sum())\n"),
    ("PY: KS1 graph tickets", PY, "same = bool(torch.equal(og, o1)) and int(cnt[:16].abs().sum()) == 0",
     "same = bool(torch.equal(og, o1)) and int(cnt.abs().sum()) == 0"),
    ("PY: _ks2_self_test", PREP, "def _ks1_prepare(", KS2_TEST),
    # ---- KS2 fold: on B > 1 the new keys ride on split NS - 1 (warp 7), so a group has NS CTAs (one wave at NS 8 / B 16)
    ("CU: Params.fold", CU, "  int B;                      // KS2: requests (grid z = B x NTG)\n",
     "  int B;                      // KS2: requests (grid z = B x NTG)\n"
     "  int fold;                   // KS2: 1 = new keys on split NS - 1 (NS CTAs per group), 0 = dedicated new-key CTA s = NS\n"),
    ("CU: tail / NSP", CU, "  const bool tail = (s == NS);                                  // the new-key CTA (no old keys)\n",
     "  const bool tail = p.fold ? (s == NS - 1) : (s == NS);         // the CTA that attends (and stores) the new keys\n"
     "  const bool oldc = (s < NS);                                   // holds old-key slots\n"
     "  const int NSP = NS + (p.fold ? 0 : 1);                        // partials per group\n"),
    ("CU: old loads on old-key CTAs", CU, "      prow[kk] = (!tail && gs < n_old", "      prow[kk] = (oldc && gs < n_old"),
    ("CU: old tiles on old-key CTAs", CU, "  for (int kk = 0; kk < (tail ? 0 : G::KPT); ++kk) {", "  for (int kk = 0; kk < (oldc ? G::KPT : 0); ++kk) {"),
    ("CU: segments", CU, "  if (!tail) {\n    segment<G::NT>(kh, vh, warp * KPW,", "  if (oldc) {\n    segment<G::NT>(kh, vh, warp * KPW,"),
    ("CU: new segment warp", CU, "  } else if (warp == 0) {\n    segment<2>(kn, vn,", "  }\n  if (tail && warp == (p.fold ? NWARP - 1 : 0)) {\n    segment<2>(kn, vn,"),
    ("CU: fence on last ticket", CU, "      if (p.sync == 2 && tk == NS) fence_acqrel();", "      if (p.sync == 2 && tk == NSP - 1) fence_acqrel();"),
    ("CU: last ticket", CU, "    misc[0] = (tk == NS) ? 1 : 0;", "    misc[0] = (tk == NSP - 1) ? 1 : 0;"),
    ("CU: merge only live partials", CU,
     "    for (int q = 0; q <= NS; ++q) {\n      const size_t prow = ((size_t)grp * (NS + 1) + q) * ROWS + mr;\n      ml[q] = __ldcg(p.part_ml + prow);\n",
     "    for (int q = 0; q <= NS; ++q) {\n      const size_t prow = ((size_t)grp * (NS + 1) + q) * ROWS + mr;\n"
     "      if (q >= NSP) {                                            // KS2 fold: no slot NS\n"
     "        ml[q] = make_float2(neg_inf(), 0.f);\n"
     "#pragma unroll\n        for (int i = 0; i < DPT; ++i) ov[q][i] = 0.f;\n        continue;\n      }\n"
     "      ml[q] = __ldcg(p.part_ml + prow);\n"),
    ("CU: grid x", CU, "cfg.gridDim = dim3((unsigned)(NS + 1),", "cfg.gridDim = dim3((unsigned)(NS + (P.fold ? 0 : 1)),"),
    ("CPP: fold arg", ('KS1_CPP_SRC = r"""', '"""'), "int64_t pdl, int64_t sync, int64_t phalf);", "int64_t pdl, int64_t sync, int64_t phalf, int64_t fold);"),
    ("CU: fold arg", CU, "int64_t pdl, int64_t sync, int64_t phalf) {", "int64_t pdl, int64_t sync, int64_t phalf, int64_t fold) {"),
    ("CU: P.fold", CU, "  P.B = (int)B;\n", "  P.B = (int)B;\n  P.fold = (B > 1 && fold) ? 1 : 0;\n"),
    ("PY: fold to the kernel", PY, "KS1_EPS, int(trig), 1 if pdl else 0, int(sync), 1 if phalf else 0)",
     "KS1_EPS, int(trig), 1 if pdl else 0, int(sync), 1 if phalf else 0, 1 if fold else 0)"),
    ("PREP: state ks2", PREP, '    state["ks1"] = set()\n', '    state["ks1"] = set()\n    state["ks2"] = set()\n'),
    ("PREP: run ks2 self-test", PREP,
     '        ok_geo = _ks1_self_test(state["cos_sin"], device, KS1_INSTS, verbose=False)\n',
     '        ok_geo = _ks1_self_test(state["cos_sin"], device, KS1_INSTS, verbose=False)\n'
     '        if KS2_ENABLE:\n'
     '            state["ks2"] = _ks2_self_test(state["cos_sin"], device, [g for g in KS1_INSTS if g in ok_geo])\n'),
]


def _scope(text, scope):
    a = text.index(scope[0])
    if text.count(scope[0]) != 1:
        raise RuntimeError(f"ks2: scope start {scope[0]!r} occurs {text.count(scope[0])} times")
    b = text.find(scope[1], a + len(scope[0]))
    return a, (len(text) if b < 0 else b)


def apply(text, patches=PATCHES):
    for what, scope, old, new in patches:
        a, b = _scope(text, scope)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"ks2 patch {what!r}: `old` occurs {n} times in its scope (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text


def build(path):
    """Box: compile the patched file's KS1_CPP_SRC / KS1_CUDA_SRC with its KS1_INSTS (no kernel launch)."""
    import hashlib
    import time
    from torch.utils.cpp_extension import load_inline
    vals = {}
    for node in ast.parse(pathlib.Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in (
                "KS1_CPP_SRC", "KS1_CUDA_SRC", "KS1_INSTS"):
            vals[node.targets[0].id] = ast.literal_eval(node.value)
    insts = " ".join(f"KS1_X({l}, {n}, {g})" for l, n, g in vals["KS1_INSTS"])
    cuda = vals["KS1_CUDA_SRC"].replace("@INSTS@", insts)
    tag = hashlib.sha1((vals["KS1_CPP_SRC"] + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"ks1_{tag}", cpp_sources=vals["KS1_CPP_SRC"], cuda_sources=cuda, functions=["ks1_forward", "ks1_info"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"[ks2] build {time.time() - t0:.0f}s", flush=True)
    for l, n, g in vals["KS1_INSTS"]:
        print(f"[ks2] L{l} NS{n} NTG{g}: regs/local/smem/occ {list(mod.ks1_info(l, n, g))}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--build":
        build(sys.argv[2])
        sys.exit(0)
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_v31_ks2_check.py")
    text = src.read_text(encoding="utf-8")
    for what, scope, old, _ in PATCHES:
        a, b = _scope(text, scope)
        print(f"  {text[a:b].count(old)}x  {what}")
    patched = apply(text)
    out.write_bytes(patched.encode("utf-8"))   # LF as read
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(PATCHES)} patches hit once in scope; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

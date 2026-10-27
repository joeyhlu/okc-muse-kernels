"""t1_patches.py: T1 fix (R-t1-*) on a layer text that already carries SK1 (tools/sk1_patches.py): K2A33 vid 7 = vid 0
(the T <= 8 route, v3.3 NC 1) + route_fr (one batch of id loads per lane instead of two dependent trips) + the 8-token MODE 2
split-item finalize in K2. Same plan and arithmetic as vid 0.

  * launch3 / attrs3: HOTCAP template value bit 0 = hot (P0 + wgmma roles), value >= 4 = SK1 code (route_fr, SKF):
    5 = vid 6 (SK1 at T = 256, unchanged), 4 = vid 7 (cold + SK1). Every other instantiation is as before.
  * the cold K2 gemm passes SKF = SK1 (only vid 7 sets it).
  * K2A33_VARIANTS gains vid 7 = (1, 3, 2, 4); _k2a33_call's hot_t uses bit 0 of the variant's hot field.
  * T1FIX_ENABLE (read at call time): buckets 1 and 8 use vid 7 once _t1_prepare's self-test passed (vid 7 vs vid 0 and
    vs FP32 at bs 1 / 3 / 5 / 8, two calls each, err flag); otherwise vid 0 as before.

    text = apply_t1(text)   (after apply_sk1)
"""

L3_HEAD = "template <int NC, int NST, int MINB, int HOTCAP>\nstatic void launch3("
A3_HEAD = "template <int NC, int NST, int MINB, int HOTCAP>\nstatic std::vector<int64_t> attrs3() {"
K2_COLD_GEMM = "    gemm<NC, NST, 2, 0, STG>(P, S, s_list, smem, bars, true, blockIdx.x, gridDim.x);\n"
K2_TMPL = "template <int NC, int NST, int MINB, bool HOTCAP, bool SK1 = false>\n__global__ void __launch_bounds__(320, MINB) k2_kernel(const Params P) {"

PY = (
    ("variants", "(1, 3, 2, 3, 4), (2, 2, 2, 5)]\n", "(1, 3, 2, 3, 4), (2, 2, 2, 5), (1, 3, 2, 4)]\n"
     "T1FIX_ENABLE = True   # T1 fix (R-t1-*): buckets 1 / 8 on vid 7 = vid 0 + route_fr + 8-token K2 finalize, self-tested\n"
     "T1_VID = 7\n"),
    ("gate cases", '            "    case 6: launch3<2, 2, 2, 5>(p, (int)G, (int)G3, (int)mask); break;")',
     '            "    case 6: launch3<2, 2, 2, 5>(p, (int)G, (int)G3, (int)mask); break;\\n"\n'
     '            "    case 7: launch3<1, 3, 2, 4>(p, (int)G, (int)G3, (int)mask); break;")'),
    ("gate attrs", '            "    case 6: return attrs3<2, 2, 2, 5>();")',
     '            "    case 6: return attrs3<2, 2, 2, 5>();\\n"\n            "    case 7: return attrs3<1, 3, 2, 4>();")'),
    ("hot_t bit 0", "hot_t = (K2A33_HOT_T if hot_t is None else hot_t) if (len(v) >= 4 and v[3]) else 0",
     "hot_t = (K2A33_HOT_T if hot_t is None else hot_t) if (len(v) >= 4 and (v[3] & 1)) else 0"),
    ("call", "    vid = SK1_VID if (SK1_ENABLE and rep == 256 and st.get(\"sk1_ok\")) else K2A33_PLAN[rep]\n",
     "    vid = SK1_VID if (SK1_ENABLE and rep == 256 and st.get(\"sk1_ok\")) else K2A33_PLAN[rep]\n"
     "    if T1FIX_ENABLE and rep in (1, 8) and st.get(\"t1_ok\") and vid == 0:\n"
     "        vid = T1_VID\n"),
    ("prepare", "    _sk1_prepare(st, w, dev)\n", "    _sk1_prepare(st, w, dev)\n    _t1_prepare(st, w, dev)\n"),
    ("helper", "def _v11_k2a33_call(x, local_ids, topk_weights, st):\n",
     '''def _t1_prepare(st, w, dev):
    """T1 self-test: vid 7 vs vid 0 and vs the FP32 reference at bs 1 / 3 / 5 / 8 (two calls each, err flag); sets st["t1_ok"]."""
    st["t1_ok"] = False
    if not T1FIX_ENABLE:
        return
    try:
        if len(st["variants"]) <= T1_VID or st["grid"][T1_VID] != st["grid"][0]:
            print("[t1] vid 7 missing or its grid differs from vid 0: buckets 1 / 8 stay on vid 0", flush=True)
            return
        good = True
        for bs, seed, dist in ((1, 41, "normal"), (3, 42, "skewed"), (5, 12, "normal"), (8, 43, "skewed")):
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            ref = _k2a_ref_fp32(d, w)
            x = d["hidden_states"].contiguous()
            data = {"hidden_states": x, "topk_ids": d["topk_ids"], "topk_weights": d["topk_weights"], "config": {"local_start": 0}}
            st["err"].zero_()
            o0 = _k2a33_call(st, data, 0).clone()
            for k in range(2):
                o7 = _k2a33_call(st, data, T1_VID)
                torch.cuda.synchronize()
                e0, _ = _k2a_rel_err(o7, o0)
                tot, row = _k2a_rel_err(o7, ref)
                errf = int(st["err"][0])
                ok = e0 < 2e-3 and tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(o7.float()).all())
                print(f"[t1] self-test bs={bs} {dist} call {k}: rel vs vid 0 {e0:.2e} vs FP32 {tot:.3e} worst row {row:.3e} "
                      f"err={errf} -> {'ok' if ok else 'FAIL'}", flush=True)
                good &= ok
            _k2a33_reset(st)
        st["t1_ok"] = good
        print(f"[t1] T1 fix {'LIVE at buckets 1 / 8 (vid 7)' if good else 'OFF: buckets 1 / 8 stay on vid 0'}", flush=True)
    except Exception as ex:
        st["t1_ok"] = False
        print(f"[t1] self-test raised {ex!r}: buckets 1 / 8 stay on vid 0", flush=True)
        try:
            _k2a33_reset(st)
        except Exception:
            pass


def _v11_k2a33_call(x, local_ids, topk_weights, st):
'''),
)


def _one(text, old, new, what):
    n = text.count(old)
    if n != 1:
        raise ValueError(f"T1 anchor {what!r} found {n} times")
    return text.replace(old, new, 1)


def _region(text, head, end, edits, what):
    if text.count(head) != 1:
        raise ValueError(f"T1 region head {what!r} found {text.count(head)} times")
    a = text.index(head)
    b = text.index(end, a + len(head))
    seg = text[a:b]
    for old, new, n in edits:
        if seg.count(old) != n:
            raise ValueError(f"T1 {what}: {old!r} found {seg.count(old)} times (want {n})")
        seg = seg.replace(old, new)
    return text[:a] + seg + text[b:]


def apply_t1(text):
    if "route_fr<NC>" not in text:
        raise ValueError("T1: apply_sk1 first")
    text = _region(text, L3_HEAD, "\n}\n", (("(bool)HOTCAP", "(bool)(HOTCAP & 1)", 4), ("HOTCAP == 5", "(HOTCAP >= 4)", 4),
                                           ("    if (HOTCAP) {\n", "    if (HOTCAP & 1) {\n", 1)), "launch3")
    text = _region(text, A3_HEAD, "\n}\n", (("(bool)HOTCAP", "(bool)(HOTCAP & 1)", 3), ("HOTCAP == 5", "(HOTCAP >= 4)", 2)),
                   "attrs3")
    text = _region(text, K2_TMPL, "\n}\n", ((K2_COLD_GEMM, K2_COLD_GEMM.replace("gemm<NC, NST, 2, 0, STG>",
                                                                                "gemm<NC, NST, 2, 0, STG, SK1>"), 1),), "k2 cold gemm")
    for what, old, new in PY:
        text = _one(text, old, new, what)
    return text


# ---------------------------------------------------------------- T1P: W2 L2 prefetch from K1 (vid 7 only)
# After K1's producer has issued its last W13 copy, lane 0 issues cp.async.bulk.prefetch.L2 for the first T1_W2PF stages of
# the W2 range K2's CTA with the same blockIdx will stream (K2's Stream-K range over K2's stage prefix = K1's halved, in
# K2's tail-first order). K1's / K2's own copies carry L2::evict_first, so the prefetched W2 lines outlive the W13 stream.
# Same results; only when the bytes leave HBM changes. T1_W2PF (0..31, read at call time) -> probe bits 26-30 (vid 7 only).
K1_COLD_GEMM = "    gemm<NC, NST, 1, 0, STG>(P, S, s_list, smem, bars, false, blockIdx.x, gridDim.x);\n"
K1_TMPL = "template <int NC, int NST, int MINB, bool HOTCAP, bool SK1 = false>\n__global__ void __launch_bounds__(320, MINB) k1_kernel(const Params P) {"
GEMM_TMPL = "template <int NC, int NST, int MODE, int HM, int STG, bool SKF = false>\n__device__ void gemm("
PF_ANCHOR = "    if (waited && !pdl) pdl_wait();   // no-op without a PDL predecessor; orders completion in a PDL chain\n"
PF_CODE = r"""    if constexpr (SKF && MODE == 1 && !HOTCAP) {   // T1P: W2 L2 prefetch for K2's CTA of this blockIdx
      const int npf = (P.probe >> 26) & 31;
      if (npf > 0 && lane == 0) {
        int b2, e2;
        my_range2((long long)(S.sb[32] / 2), cta, ncta, b2, e2);
        if (e2 > b2) {
          int el = 0;
          while (S.sb[el + 1] / 2 <= e2 - 1) ++el;
          const int m2 = max(b2, S.sb[el] / 2 + ((e2 - 1 - S.sb[el] / 2) & ~15));
          int n = 0;
          for (int sg = 0; sg < 2 && n < npf; ++sg) {
            const int xb = sg ? b2 : m2, xe = sg ? m2 : e2;
            int e = 0;
            for (int x = xb; x < xe && n < npf; ++x, ++n) {
              while (S.sb[e + 1] / 2 <= x) ++e;
              const int l = x - S.sb[e] / 2;
              const int rb = (l >> 4) / S.np[e];
              const uint8_t* src = P.ws2 + ((size_t)(e * 32 + rb) * 16 + (l & 15)) * SW;
              asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(src), "r"((uint32_t)SW) : "memory");
            }
          }
        }
      }
    }
"""
PYP = (
    ("pf global", "T1_VID = 7\n", "T1_VID = 7\nT1_W2PF = 0   # T1P: W2 stages per K2 CTA prefetched into L2 by vid 7's K1 producer (0 = off; probe bits 26-30)\n"),
)


def apply_t1p(text):
    text = _region(text, K1_TMPL, "\n}\n", ((K1_COLD_GEMM, K1_COLD_GEMM.replace("gemm<NC, NST, 1, 0, STG>",
                                                                                "gemm<NC, NST, 1, 0, STG, SK1>"), 1),), "k1 cold gemm")
    text = _region(text, GEMM_TMPL, "\n}\n", ((PF_ANCHOR, PF_CODE + PF_ANCHOR, 1),), "gemm prefetch")
    text = _region(text, "def _k2a33_call(st, data, vid, probe=K2A33_HINT, mask=7, hot_t=None, hot_w4=None):\n", "\ndef ",
                   (("| _k3p_bits(m) | ((1 << 7) if RS1_ENABLE else 0))",
                     "| _k3p_bits(m) | ((1 << 7) if RS1_ENABLE else 0) | (((T1_W2PF & 31) << 26) if vid == T1_VID else 0))", 1),),
                   "k2a33 call probe")
    for what, old, new in PYP:
        text = _one(text, old, new, what)
    return text


# ---------------------------------------------------------------- T1P0: the W2 L2 prefetch on vid 0 itself (R-t1-2)
# R-t1-1 killed the vid 7 fix (route_fr + FF8 at T <= 8: +0.4..+1 us in the layer), so the prefetch goes on vid 0's K1 producer,
# gated at compile time by NC == 1 && !HOTCAP (vid 0 only: vid 1 / 2 are NC 2) and at run time by probe bits 26-30
# (T1_W2PF stages per K2 CTA, 0 = off). _k2a33_call sets the bits for vid 0 at M <= 8.
PF0_CODE = PF_CODE.replace("if constexpr (SKF && MODE == 1 && !HOTCAP) {   // T1P",
                           "if constexpr (NC == 1 && MODE == 1 && !HOTCAP) {   // T1P0")
PYP0 = (
    ("pf0 global", "SK1_VID = 6\n", "SK1_VID = 6\nT1_W2PF = 8   # T1P0 (R-t1-2): W2 stages per K2 CTA prefetched into L2 by vid 0's K1 producer at M <= 8 (0 = off)\n"),
)


def apply_t1p0(text):
    if "if constexpr (SK1) route_fr<NC>" not in text:
        raise ValueError("T1P0: apply_sk1 first")
    if PF0_CODE == PF_CODE:
        raise ValueError("T1P0: prefetch gate not replaced")
    text = _region(text, GEMM_TMPL, "\n}\n", ((PF_ANCHOR, PF0_CODE + PF_ANCHOR, 1),), "gemm prefetch")
    text = _region(text, "def _k2a33_call(st, data, vid, probe=K2A33_HINT, mask=7, hot_t=None, hot_w4=None):\n", "\ndef ",
                   (("| _k3p_bits(m) | ((1 << 7) if RS1_ENABLE else 0))",
                     "| _k3p_bits(m) | ((1 << 7) if RS1_ENABLE else 0) | (((T1_W2PF & 31) << 26) if (vid == 0 and m <= 8) else 0))", 1),),
                   "k2a33 call probe")
    for what, old, new in PYP0:
        text = _one(text, old, new, what)
    return text

"""sk1_patches.py: SK1 (notes/study/moe_t256_skeleton.md, R-sk-1 / R-sk-2) as an extra K2A33 variant on a generated layer.

SK1 = (a) route_fr: route() with a lane's routing ids loaded once as independent loads (registers, both passes) instead of
2 x T / 32 dependent L2 trips, used by the vid 2 K1 / K2 / P0 only; (b) the hot-role K2 split-item finalize (sum of squares
of the item's y rows) with 8 tokens per L2 trip instead of 4. Same plan, same per-token arithmetic and atomics.
Box (sk_lab arm sk, standalone vid 2, 10 layer routings): -5.9 us median at T = 256.

Shape of the change (so the A/B arm "SK1 off" is the shipped vid 2 codegen, not a runtime branch):
  * K2A33 source: k1_kernel / k2_kernel / p0_kernel get `bool SK1 = false` (route_fr when SK1), gemm gets `bool SKF = false`
    (FB 8 when SKF; only the hot K2 gemm of an SK1 kernel passes it); launch3 / attrs3 instantiate the kernels with
    SK1 = (HOTCAP == 5). Template defaults leave every other instantiation's code as it was.
  * K2A33_VARIANTS gains vid 6 = (2, 2, 2, 5) (= vid 2 + SK1); the gate case lists gain case 6.
  * SK1_ENABLE (read at call time): bucket 256 uses vid 6 once _sk1_prepare's self-test (vid 6 vs vid 2 and vs the FP32
    reference, bs 200 / 256 / 257 / 320, two calls each, err flag) passed; otherwise vid 2 (K2A33_PLAN) as before.

Every anchor is asserted unique: either in the whole text, or inside a region whose head is unique in the whole text.

    text = apply_sk1(text)
"""

K33_HEAD = 'K2A33_KERNEL_SRC = r"""'

ROUTE_HEAD = "template <int NC, int TPH = 64, bool HGE = false, bool LIST = true>\n__device__ void route("
FR_OLD1 = """  const int NP = 8 * P.M;
  for (int i = tid; i < 8 * 32; i += blockDim.x) (&S.wc[0][0])[i] = 0;
  __syncthreads();
  const int L = ((NP + 255) / 256) * 32;
  const int p0 = warp * L;
  if (warp < 8) {
    for (int base = 0; base < L; base += 32) {
      const int p = p0 + base + lane;
      int e = -1;
      if (p < NP) {
        const int v = P.ids[p] - P.lo;
        e = (v >= 0 && v < 32) ? v : -1;
      }
      if (e >= 0) atomicAdd(&S.wc[warp][e], 1);
    }
  }
  __syncthreads();
"""
FR_NEW1 = """  const int NP = 8 * P.M;
  const int L = ((NP + 255) / 256) * 32;
  const int p0 = warp * L;
  constexpr int RQ = (P3 + 255) / 256;   // SK1: a lane's ids, loaded once (independent loads) for both passes
  int ev[RQ];
#pragma unroll
  for (int q = 0; q < RQ; ++q) {
    const int p = p0 + 32 * q + lane;
    int e = -1;
    if (warp < 8 && 32 * q < L && p < NP) {
      const int v = __ldg(P.ids + p) - P.lo;
      e = (v >= 0 && v < 32) ? v : -1;
    }
    ev[q] = e;
  }
  for (int i = tid; i < 8 * 32; i += blockDim.x) (&S.wc[0][0])[i] = 0;
  __syncthreads();
  if (warp < 8) {
#pragma unroll
    for (int q = 0; q < RQ; ++q)
      if (ev[q] >= 0) atomicAdd(&S.wc[warp][ev[q]], 1);
  }
  __syncthreads();
"""
FR_OLD2 = """  if (warp < 8) {
    for (int base = 0; base < L; base += 32) {
      const int p = p0 + base + lane;
      int e = -1;
      if (p < NP) { const int v = __ldg(P.ids + p) - P.lo; e = (v >= 0 && v < 32) ? v : -1; }
      const unsigned m = __match_any_sync(FULL, e);
"""
FR_NEW2 = """  if (warp < 8) {
#pragma unroll
    for (int q = 0; q < RQ; ++q) {
      if (32 * q >= L) break;   // warp-uniform
      const int p = p0 + 32 * q + lane;
      const int e = ev[q];
      const unsigned m = __match_any_sync(FULL, e);
"""

K1_TMPL = "template <int NC, int NST, int MINB, bool HOTCAP>\n__global__ void __launch_bounds__(320, MINB) k1_kernel(const Params P) {"
K2_TMPL = "template <int NC, int NST, int MINB, bool HOTCAP>\n__global__ void __launch_bounds__(320, MINB) k2_kernel(const Params P) {"
P0_TMPL = "template <int NC>\n__global__ void __launch_bounds__(256) p0_kernel(const Params P) {"
GEMM_TMPL = "template <int NC, int NST, int MODE, int HM, int STG>\n__device__ void gemm("
CALL_K1 = "route<NC>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);"
CALL_K2 = "route<NC>(P, S, nullptr, s_list, 32, false);"
CALL_P0 = "route<NC>(P, S, nullptr, s_list, 64, false);"
K2_HOT_GEMM = "    if (hr) gemm<NC, NST, 2, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);\n"
FIN_OLD = """            for (int n0 = 0; n0 < ntok; n0 += 4) {
              float4 yv[4];
#pragma unroll
              for (int u = 0; u < 4; ++u)
                if (n0 + u < ntok) yv[u] = __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)(c0 + n0 + u) * KDIM + col));
#pragma unroll
              for (int u = 0; u < 4; ++u) {
"""
FIN_NEW = """            constexpr int FBX = SKF ? 8 : 4;   // SK1: hot split-item finalize, 8 tokens per L2 trip
            for (int n0 = 0; n0 < ntok; n0 += FBX) {
              float4 yv[FBX];
#pragma unroll
              for (int u = 0; u < FBX; ++u)
                if (n0 + u < ntok) yv[u] = __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)(c0 + n0 + u) * KDIM + col));
#pragma unroll
              for (int u = 0; u < FBX; ++u) {
"""
P0_LAUNCH = "k3::p0_kernel<NC><<<G3, 256, 0, st>>>(p);"
K1_INST = "k3::k1_kernel<NC, NST, MINB, (bool)HOTCAP>"
K2_INST = "k3::k2_kernel<NC, NST, MINB, (bool)HOTCAP>"

PY = (
    ("variants", "K2A33_VARIANTS = [(1, 3, 2, 0), (2, 3, 2, 0), (2, 2, 2, 1), (1, 3, 2, 2, 4), (1, 3, 2, 3, 1), (1, 3, 2, 3, 4)]\n",
     "K2A33_VARIANTS = [(1, 3, 2, 0), (2, 3, 2, 0), (2, 2, 2, 1), (1, 3, 2, 2, 4), (1, 3, 2, 3, 1), (1, 3, 2, 3, 4), (2, 2, 2, 5)]\n"
     "SK1_ENABLE = True   # SK1 (R-sk-1 / R-sk-2): bucket 256 on vid 6 = vid 2 + route_fr + 8-token hot K2 finalize, self-tested\n"
     "SK1_VID = 6\n"),
    ("gate cases", '            "    case 5: launch37<4, 3>(p, (int)G, (int)G3, (int)mask); break;")',
     '            "    case 5: launch37<4, 3>(p, (int)G, (int)G3, (int)mask); break;\\n"\n'
     '            "    case 6: launch3<2, 2, 2, 5>(p, (int)G, (int)G3, (int)mask); break;")'),
    ("gate attrs", '            "    case 5: return attrs37<4, 3>();")',
     '            "    case 5: return attrs37<4, 3>();\\n"\n            "    case 6: return attrs3<2, 2, 2, 5>();")'),
    ("call", "    return _k2a33_call(st, data, K2A33_PLAN[rep], hot_t=K2A33_HOT_T_BUCKET.get(rep),\n",
     "    vid = SK1_VID if (SK1_ENABLE and rep == 256 and st.get(\"sk1_ok\")) else K2A33_PLAN[rep]\n"
     "    return _k2a33_call(st, data, vid, hot_t=K2A33_HOT_T_BUCKET.get(rep),\n"),
    ("prepare", "    _m32px_prepare(st, w, dev)\n", "    _m32px_prepare(st, w, dev)\n    _sk1_prepare(st, w, dev)\n"),
    ("helper", "def _v11_k2a33_call(x, local_ids, topk_weights, st):\n",
     '''def _sk1_prepare(st, w, dev):
    """SK1 self-test: vid 6 vs vid 2 and vs the FP32 reference at bucket 256 (two calls each, err flag); sets st["sk1_ok"]."""
    st["sk1_ok"] = False
    if not SK1_ENABLE:
        return
    try:
        if len(st["variants"]) <= SK1_VID or st["grid"][SK1_VID] != st["grid"][K2A33_PLAN[256]]:
            print("[sk1] vid 6 missing or its grid differs from vid 2: bucket 256 stays on vid 2", flush=True)
            return
        good = True
        for bs, seed, dist in ((200, 31, "skewed"), (256, 32, "normal"), (257, 16, "normal"), (320, 18, "normal")):
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            ref = _k2a_ref_fp32(d, w)
            x = d["hidden_states"].contiguous()
            data = {"hidden_states": x, "topk_ids": d["topk_ids"], "topk_weights": d["topk_weights"], "config": {"local_start": 0}}
            st["err"].zero_()
            o2 = _k2a33_call(st, data, K2A33_PLAN[256], hot_t=K2A33_HOT_T_BUCKET.get(256), hot_w4=76).clone()
            for k in range(2):
                o6 = _k2a33_call(st, data, SK1_VID, hot_t=K2A33_HOT_T_BUCKET.get(256), hot_w4=76)
                torch.cuda.synchronize()
                e2, _ = _k2a_rel_err(o6, o2)
                tot, row = _k2a_rel_err(o6, ref)
                errf = int(st["err"][0])
                ok = e2 < 2e-3 and tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(o6.float()).all())
                print(f"[sk1] self-test bs={bs} {dist} call {k}: rel vs vid 2 {e2:.2e} vs FP32 {tot:.3e} worst row {row:.3e} "
                      f"err={errf} -> {'ok' if ok else 'FAIL'}", flush=True)
                good &= ok
            _k2a33_reset(st)
        st["sk1_ok"] = good
        print(f"[sk1] SK1 {'LIVE at bucket 256 (vid 6)' if good else 'OFF: bucket 256 stays on vid 2'}", flush=True)
    except Exception as ex:
        st["sk1_ok"] = False
        print(f"[sk1] self-test raised {ex!r}: bucket 256 stays on vid 2", flush=True)
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
        raise ValueError(f"SK1 anchor {what!r} found {n} times")
    return text.replace(old, new, 1)


def _in_region(text, head, end_marker, old, new, what, count=1):
    """Replace `old` (exactly `count` times) inside [head, next end_marker after head); head unique in the text."""
    if text.count(head) != 1:
        raise ValueError(f"SK1 region head for {what!r} found {text.count(head)} times")
    a = text.index(head)
    b = text.index(end_marker, a + len(head))
    seg = text[a:b]
    if seg.count(old) != count:
        raise ValueError(f"SK1 anchor {what!r} found {seg.count(old)} times in its region (want {count})")
    return text[:a] + seg.replace(old, new) + text[b:]


def apply_sk1(text):
    # region of the K2A33 CUDA source (the triple-quoted string after K33_HEAD)
    if text.count(K33_HEAD) != 1:
        raise ValueError("SK1: K2A33_KERNEL_SRC head not unique")
    k0 = text.index(K33_HEAD)
    k1 = text.index('"""', k0 + len(K33_HEAD))
    src = text[k0:k1]
    # route_fr = route() with the FR passes, inserted after route()
    if src.count(ROUTE_HEAD) != 1 or text.count(ROUTE_HEAD) != 1:
        raise ValueError("SK1: route() head not unique")
    a = src.index(ROUTE_HEAD)
    b = src.index("\n}\n", a) + 3
    fn = src[a:b]
    fr = _one(_one(fn, FR_OLD1, FR_NEW1, "route pass 1"), FR_OLD2, FR_NEW2, "route pass 2")
    fr = _one(fr, "__device__ void route(", "__device__ void route_fr(", "route_fr name")
    src = src[:b] + "// SK1: route() with one batch of id loads per lane (vid 6 = vid 2 + SK1 only)\n" + fr + src[b:]
    # kernel templates + route calls
    for tmpl, call, nm in ((K1_TMPL, CALL_K1, "k1"), (K2_TMPL, CALL_K2, "k2")):
        src = _one(src, tmpl, tmpl.replace("bool HOTCAP>", "bool HOTCAP, bool SK1 = false>"), f"{nm} template")
        new_head = tmpl.replace("bool HOTCAP>", "bool HOTCAP, bool SK1 = false>")
        src = _in_region(src, new_head, "\n}\n", call,
                         "if constexpr (SK1) " + call.replace("route<NC>", "route_fr<NC>") + " else " + call, f"{nm} route call")
    p0_new = P0_TMPL.replace("template <int NC>", "template <int NC, bool SK1 = false>")
    src = _one(src, P0_TMPL, p0_new, "p0 template")
    src = _in_region(src, p0_new, "\n}\n", CALL_P0,
                     "if constexpr (SK1) " + CALL_P0.replace("route<NC>", "route_fr<NC>") + " else " + CALL_P0, "p0 route call")
    # gemm: SKF template flag, FB 8 finalize for SKF, the SK1 K2's hot gemm passes SKF = SK1
    src = _one(src, GEMM_TMPL, GEMM_TMPL.replace("int STG>", "int STG, bool SKF = false>"), "gemm template")
    src = _in_region(src, GEMM_TMPL.replace("int STG>", "int STG, bool SKF = false>"), "if (MODE == 1 && last)", FIN_OLD,
                     FIN_NEW, "gemm MODE 2 finalize")
    src = _in_region(src, K2_TMPL.replace("bool HOTCAP>", "bool HOTCAP, bool SK1 = false>"), "\n}\n", K2_HOT_GEMM,
                     K2_HOT_GEMM.replace("gemm<NC, NST, 2, 1, STG>", "gemm<NC, NST, 2, 1, STG, SK1>"), "k2 hot gemm")
    # launch3 / attrs3 instantiate SK1 = (HOTCAP == 5)
    src = _one(src, P0_LAUNCH, "k3::p0_kernel<NC, HOTCAP == 5><<<G3, 256, 0, st>>>(p);", "p0 launch")
    if src.count(K1_INST) != 3 or src.count(K2_INST) != 2:
        raise ValueError(f"SK1: kernel instantiations {src.count(K1_INST)} / {src.count(K2_INST)} (want 3 / 2)")
    src = src.replace(K1_INST, "k3::k1_kernel<NC, NST, MINB, (bool)HOTCAP, HOTCAP == 5>")
    src = src.replace(K2_INST, "k3::k2_kernel<NC, NST, MINB, (bool)HOTCAP, HOTCAP == 5>")
    text = text[:k0] + src + text[k1:]
    for what, old, new in PY:
        text = _one(text, old, new, what)
    return text

"""vml_px_strip.py: remove the M32 PX route (k2_px / k3_px, their device helpers and host launchers, the K1 readiness hook,
_m32px_call / _m32px_prepare and the route hook) from a generated layer whose bucket 32 is served by VML1. Bucket 32's
fallback chain becomes VML1 -> r36 (K2A v3.6 vid 3, K2A36_BUCKETS) -> the v3.3 plan, as when M32_PX was False.

Kept on purpose: WSK (wsk_range lives in the shared NC 2 HOTCAP K1 route code that vid 2 runs at 128 / 256), the Params
fields rdy / pxmk / pxd and the PXB / PXW / PXV constants (run3 sets them; PXB is never set, so K1's y reset is
unchanged), M32_PX (now False) / M32_PX_D (other code may print them).
Every anchor is asserted unique; afterwards no PX symbol may remain.

    text = apply(text)
"""
NL = chr(10)


def _cut(text, a_marker, b_marker, what, keep_b=True):
    assert text.count(a_marker) == 1, f"{what}: start anchor x{text.count(a_marker)}"
    a = text.index(a_marker)
    b = text.index(b_marker, a)
    assert b > a, what
    return text[:a] + text[b if keep_b else b + len(b_marker):]


def apply(text):
    # ---- Python: route hook, prepare call, the two functions, the lever constant, load_inline export lists
    hook = ('    if M32_PX and rep == 32 and st.get("px_ok") and x.shape[0] <= 48:' + NL + '        try:' + NL
            + '            return _m32px_call(st, data)' + NL + '        except Exception as ex:' + NL
            + '            print(f"[k3] M32 PX route raised ({ex!r}): vid 3 from here on", flush=True)' + NL
            + '            st["px_ok"] = False' + NL)
    assert text.count(hook) == 1, "route hook"
    text = text.replace(hook, "")
    call = "    _m32px_prepare(st, w, dev)" + NL
    assert text.count(call) == 1, "prepare call"
    text = text.replace(call, "")
    text = _cut(text, "def _m32px_call(st, data):" + NL, "def _v17_k2a37_prepare(st, w, dev):", "_m32px_* functions")
    lever = "M32_PX = True" + NL
    assert text.count(lever) == 1, "M32_PX"
    text = text.replace(lever, "M32_PX = False   # PX removed (tools/vml_px_strip.py): VML1 serves bucket 32" + NL)
    for old, new in (('"v8_tail", "v8_router32", "pxrun", "pxattrs"]', '"v8_tail", "v8_router32"]'),
                     ('functions=["run3", "attrs3v", "pxrun", "pxattrs"]', 'functions=["run3", "attrs3v"]')):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    # ---- C++ declarations (K2A33_CPP_SRC)
    text = _cut(text, "std::vector<int64_t> pxattrs();" + NL + "void pxrun(", "int64_t G, int64_t probe);" + NL,
                "pxrun / pxattrs declarations", keep_b=False)
    # ---- CUDA: the K1 readiness hook in k1_kernel36
    k1hook = ('  if ((P.probe >> PXB) & 1) {   // PX: readiness signal once every warp of this CTA is done' + NL
              + '    long long* mk = px_rec(P, 0);' + NL + '    __syncthreads();' + NL + '    if (threadIdx.x == 0) {' + NL
              + '      if (mk != nullptr) { mk[0] = px_smid(); mk[2] = px_gt(); }' + NL
              + '      px_k1_signal(P, S, s_pr);' + NL + '      if (mk != nullptr) mk[3] = px_gt();' + NL + '    }' + NL
              + '  }' + NL)
    assert text.count(k1hook) == 1, "K1 hook"
    text = text.replace(k1hook, "")
    # ---- CUDA: device helpers, mbar_wait_ok, k2_px / k3_px, host launchers
    text = _cut(text, "__device__ __forceinline__ long long px_gt() {",
                "template <int NCH, int NST>" + NL + "__global__ void __launch_bounds__(320, 2) k1_kernel36(", "px helpers")
    text = _cut(text, "__device__ __forceinline__ bool mbar_wait_ok(uint32_t b, uint32_t par, int* err) {",
                "template <int NC, int NST, int MINB, bool HOTCAP, bool SK1 = false>", "mbar_wait_ok")
    text = _cut(text, "template <int NCH, int NST>" + NL + "__global__ void __launch_bounds__(320, 2) k2_px(",
                "}  // namespace k3" + NL + NL + "template <int NC, int NST, bool HOT = false>" + NL
                + "static constexpr int smem_bytes()", "k2_px / k3_px")
    text = _cut(text, "template <int NCH, int NST>" + NL + "static void launch_px(", "void run3(int64_t vid,", "launch_px / pxattrs / pxrun")
    cm = "   // M32 PX: y is self-reset by k3_px"
    assert text.count(cm) == 1, "K1 y-reset comment"
    text = text.replace(cm, "   // PXB is never set (PX removed)")
    for sym in ("k2_px", "k3_px", "pxrun", "pxattrs", "px_k1_signal", "px_decode", "px_gt", "launch_px", "_m32px", "mbar_wait_ok"):
        assert sym not in text, sym
    return text

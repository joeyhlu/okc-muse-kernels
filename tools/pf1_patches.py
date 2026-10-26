"""pf1_patches.py: PF1 = K2A's K2 tail stages the unpooler's weights in L2. After a K2 CTA finishes its last work item (end of
k2_kernel36 at T = 16 / 32, k2_kernel37 at T = 128, the generic / HOTCAP k2_kernel at T = 256), thread 0 issues
cp.async.bulk.prefetch.L2 for its slice (blockIdx.x of gridDim.x) of the first PF1[bucket] bytes of the unpooler's weights, in
the order the unpooler reads them:
  * T = 9..32 (P1M unpooler): the route's grid-interleaved stream-order buffer (state["p1m"]["il"][("unpooler", bucket)]): a
    plain prefix, 64 KB rows;
  * T > 32 (today's cuBLAS BF16 GEMM on dense["unpooler"], every row tile streams k = 0 -> K, no split-K): the first
    PF1 / 8192 bytes of every weight row.
Pointer / pitch / bytes / rows are run3 arguments read at call time (kernel_fn sets _PF1_CUR around the experts call; nothing
baked into prepare; self-tests run with rows = 0). A prefetch has no architectural side effect: outputs are bit-identical.
Exact text patches against kernels/muse_decode_layer/k3_v34.py (scoped as tools/k2a_v47_patches.py).

    python -I tools/pf1_patches.py [k3_v34.py] [out.py]
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v34.py"

# bucket rep -> bytes staged. KILLED 2026-10-26 (notes/lab.md PF1 row, box_out/pf1/): first sweep p16 looked like x1.005-1.008 on
# decode 16 / 32 and verify 32, but repeats (router seeds 12345 / public / 777) gave x1.0021 / x1.0012 / x1.0006 vs an off-vs-off
# control of x1.0016 / x1.0022; 32 / 48 MB lose (x1.0007 / x0.993). Nothing routed: the patch is inert as shipped.
ROUTE = {}

K2A = ('K2A33_KERNEL_SRC = r"""', '"""')
K2CPP = ('K2A33_CPP_SRC = r"""', '"""')
F33 = ("def _k2a33_call(", "\ndef ")
F36 = ("def _k2a36_call(", "\ndef ")
F37 = ("def _k2a37_call(", "\ndef ")
FN = ("def kernel_fn(data, state):", "\n    return next_attn_input")

SIG_OLD = "torch::Tensor xw2, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe"
SIG_NEW = (SIG_OLD + ",\n          torch::Tensor pf, int64_t pf_pitch, int64_t pf_len, int64_t pf_rows")

PF1_DEV = r"""__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
// PF1 (tools/pf1_patches.py): after this CTA's last work item, stage its slice of the next weight read (the unpooler) in L2.
__device__ __forceinline__ void pf1_tail(const Params& P) {
  if (P.pf_rows <= 0 || threadIdx.x != 0) return;
  const long long G = gridDim.x, b = blockIdx.x;
  const int r0 = (int)((P.pf_rows * b) / G), r1 = (int)((P.pf_rows * (b + 1)) / G);
  for (int r = r0; r < r1; ++r)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(P.pf + (size_t)r * (size_t)P.pf_pitch), "r"(P.pf_len) : "memory");
}
"""

PF1_PY = '''PF1 = {}   # bucket rep -> bytes of the unpooler's weights K2's finished CTAs stage in L2 (read at call time; absent / 0 = off)
PF1_CHUNK = 65536   # prefetch row for the P1M stream-order buffer (bytes)
_PF1_CUR = [None]   # (tensor, pitch, bytes per row, rows) for the K2 launch of the current kernel_fn call; None = no prefetch


def _pf1_target(state, tokens):
    """The unpooler's first PF1[bucket] bytes in its read order as (tensor, pitch, bytes per row, rows), or None. Shape-only."""
    b = _k2a_bucket_of(tokens)
    nbytes = int(PF1.get(b, 0))
    if nbytes <= 0:
        return None
    if SLICED_UNPOOL_MAX_TOKENS < tokens <= 32:   # P1M unpooler: the route's grid-interleaved (stream-order) buffer
        st = state.get("p1m")
        key = ("unpooler", b)
        if not V20_P1M_ENABLE or st is None or key not in V20_P1M_ROUTE or st["route"].get(key) is None:
            return None
        il = st["il"].get(key)
        if il is None:
            return None
        buf = il["buf"]
        n = min(nbytes, buf.numel() * buf.element_size()) // PF1_CHUNK
        return (buf, PF1_CHUNK, PF1_CHUNK, n) if n > 0 else None
    if tokens > 32:   # today's BF16 unpooler GEMM: every row tile streams k = 0 -> K, so the first bytes of every row
        w = state["dense"]["unpooler"]
        rowb = w.shape[1] * w.element_size()
        pre = min(rowb, nbytes // w.shape[0]) // 16 * 16
        return (w, rowb, pre, w.shape[0]) if pre > 0 else None
    return None


def _pf1_args(st):
    t = _PF1_CUR[0]
    return (st["sink"], 0, 0, 0) if t is None else t


'''


def _call(scope, tail):
    old = f"st[\"sms\"] * 2, {tail})\n    return out\n"
    new = f"st[\"sms\"] * 2, {tail}, *_pf1_args(st))\n    return out\n"
    return scope, old, new


PATCHES = [
    ("K2A: Params pf fields", K2A, "  int M, lo, probe;\n};\n",
     "  int M, lo, probe;\n  const uint8_t* pf; long long pf_pitch; int pf_len; long long pf_rows;   // PF1\n};\n"),
    ("K2A: pf1_tail", K2A, '__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\\n" ::: "memory"); }\n', PF1_DEV),
    ("K2A: k2_kernel36 tail", K2A, "  gemm36<NCH, NST, 2>(P, S, s_list, smem, bars, true, s_pr);\n}\n",
     "  gemm36<NCH, NST, 2>(P, S, s_list, smem, bars, true, s_pr);\n  pf1_tail(P);\n}\n"),
    ("K2A: k2_kernel37 tail", K2A, "  gemm37<NCH, NST, 2>(P, S, nullptr, smem, bars, true);\n}\n",
     "  gemm37<NCH, NST, 2>(P, S, nullptr, smem, bars, true);\n  pf1_tail(P);\n}\n"),
    ("K2A: k2_kernel tail", K2A, "    gemm<NC, NST, 2, 0, STG>(P, S, s_list, smem, bars, true, blockIdx.x, gridDim.x);\n  }\n}\n",
     "    gemm<NC, NST, 2, 0, STG>(P, S, s_list, smem, bars, true, blockIdx.x, gridDim.x);\n  }\n  pf1_tail(P);\n}\n"),
    ("K2A: run3 signature", K2A, SIG_OLD + ") {", SIG_NEW + ") {"),
    ("K2A: run3 pf params", K2A, "  p.M = (int)M; p.lo = (int)lo; p.probe = (int)probe;\n",
     "  p.M = (int)M; p.lo = (int)lo; p.probe = (int)probe;\n"
     "  p.pf = pf_rows > 0 ? reinterpret_cast<const uint8_t*>(pf.data_ptr()) : nullptr;   // PF1\n"
     "  p.pf_pitch = pf_pitch; p.pf_len = (int)pf_len; p.pf_rows = pf_rows > 0 ? pf_rows : 0;\n"),
    ("CPP: run3 signature", K2CPP, SIG_OLD + ");", SIG_NEW + ");"),
    ("PY: PF1 tables + helpers", ("def _k2a33_call(st, data, vid,", None), "def _k2a33_call(st, data, vid,",
     PF1_PY + "def _k2a33_call(st, data, vid,"),
    ("PY: _k2a33_call pf args", *_call(F33, "probe | ((1 << 24) if K2A47_K1_PDL else 0)")),
    ("PY: _k2a36_call pf args", *_call(F36, "probe | ((1 << 24) if K2A47_K1_PDL else 0)")),
    ("PY: _k2a37_call pf args", *_call(F37, "pb | ((1 << 24) if K2A47_K1_PDL_T128 else 0)")),
    ("FN: PF1 around the experts", FN,
     "    expert_output = _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens)\n",
     "    _PF1_CUR[0] = _pf1_target(state, tokens) if PF1 else None   # PF1: K2's finished CTAs stage the unpooler's first bytes in L2\n"
     "    try:\n"
     "        expert_output = _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens)\n"
     "    finally:\n"
     "        _PF1_CUR[0] = None\n"),
]


def _scope(text, scope):
    a = text.index(scope[0])
    if text.count(scope[0]) != 1:
        raise RuntimeError(f"pf1: scope start {scope[0]!r} occurs {text.count(scope[0])} times")
    if scope[1] is None:
        return a, len(text)
    b = text.find(scope[1], a + len(scope[0]))
    return a, (len(text) if b < 0 else b)


def apply(text, patches=PATCHES, route=None):
    if "def _pf1_target(" in text:
        raise RuntimeError("pf1: layer already carries PF1")
    for what, scope, old, new in patches:
        a, b = _scope(text, scope)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"pf1 patch {what!r}: `old` occurs {n} times in its scope (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    r = ROUTE if route is None else route
    old = "PF1 = {}   # bucket rep"
    assert text.count(old) == 1
    return text.replace(old, f"PF1 = {dict(sorted(r.items()))!r}   # bucket rep")


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_v34_pf1_check.py")
    text = src.read_text(encoding="utf-8")
    for what, scope, old, _ in PATCHES:
        a, b = _scope(text, scope)
        print(f"  {text[a:b].count(old)}x  {what}")
    patched = apply(text)
    out.write_bytes(patched.encode("utf-8"))   # LF as read
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(PATCHES)} patches hit once in scope; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

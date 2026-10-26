"""ks6_patches.py: KS6 = P1's qkv projection at T <= 8 launched from its OWN p1s_kernel instantiation (template flag CO = 1) whose
preferred shared-memory carveout is 100 %, so SMs running the qkv GEMV keep room for KS1's smem and KS1's CTAs (a PDL secondary)
become resident early; every other P1 launch keeps p1s_kernel<1, 0> with the default carveout (the all-P1 100 % carveout of
KS1_P1_CARVEOUT cost ~3 us elsewhere). Exact text patches against kernels/muse_decode_layer/k3_v34.py.

  * CUDA: p1s_kernel / launch gain `int CO = 0` (no code change: a distinct kernel symbol); launch<.., 1> sets the carveout once;
    p1s_gemv dispatches mode 0 to launch<1, 0, 1> when pdl bit 2 is set.
  * Python: p1_gemv sets pdl bit 2 for proj "qkv" when KS6_QKV_CARVEOUT (read at call time).
  * KS1_INSTS gains (1, 8, 1) / (8, 8, 4) so arms can pair the early residency with fewer, larger splits.

    python -I tools/ks6_patches.py [k3_v34.py] [out.py]
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v34.py"

P1CU = ('P1_CUDA_SRC = r"""', '"""')
P1GEMV = ("def p1_gemv(st, proj, x,", "\ndef ")
TABLES = ("KS1_ENABLE = True ", "\n_KS1 = {")

PATCHES = [
    ("P1 CU: kernel CO flag", P1CU,
     "template <int D, int MODE>\n__global__ void __launch_bounds__(512, Bounds<D, MODE>::minb) p1s_kernel(",
     "template <int D, int MODE, int CO = 0>   // KS6: CO = 1 is the qkv instantiation with its own (100 %) smem carveout\n"
     "__global__ void __launch_bounds__(512, Bounds<D, MODE>::minb) p1s_kernel("),
    ("P1 CU: launch CO", P1CU,
     "template <int D, int MODE>\nstatic void launch(const Args& A, int warps, int S, int nrt, int pdl) {\n",
     "template <int D, int MODE, int CO = 0>\nstatic void launch(const Args& A, int warps, int S, int nrt, int pdl) {\n"
     "  if constexpr (CO == 1) {\n"
     "    static bool co_set = false;\n"
     "    if (!co_set) {\n"
     "      C10_CUDA_CHECK(cudaFuncSetAttribute(p1s_kernel<D, MODE, 1>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));\n"
     "      co_set = true;\n"
     "    }\n"
     "  }\n"),
    ("P1 CU: launch kernel symbol", P1CU, "  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, p1s_kernel<D, MODE>, A));\n}\n",
     "  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, p1s_kernel<D, MODE, CO>, A));\n}\n"),
    ("P1 CU: dispatch bit 2", P1CU,
     "  const int w = (int)warps, S = (int)split, pd = (int)(pdl & 1);\n  P1S_DISPATCH(1, 0, (p1::launch<1, 0>(A, w, S, nrt, pd)); return)\n",
     "  const int w = (int)warps, S = (int)split, pd = (int)(pdl & 1);\n"
     "  if ((pdl >> 2) & 1) { P1S_DISPATCH(1, 0, (p1::launch<1, 0, 1>(A, w, S, nrt, pd)); return) }   // KS6: qkv instantiation\n"
     "  P1S_DISPATCH(1, 0, (p1::launch<1, 0>(A, w, S, nrt, pd)); return)\n"),
    ("PY: p1_gemv qkv bit", P1GEMV,
     "pdl=_g2_p1_pdl(proj) | _v20_p1_trig(st, proj, geo, mode), mode=mode)",
     "pdl=_g2_p1_pdl(proj) | _v20_p1_trig(st, proj, geo, mode) | (4 if (proj == \"qkv\" and KS6_QKV_CARVEOUT) else 0),\n"
     "           mode=mode)"),
    ("PY: KS6 switch + KS1 NS8 geometries", TABLES,
     "KS1_INSTS = ((1, 16, 1), (8, 16, 4), (1, 4, 1))   #",
     "KS6_QKV_CARVEOUT = False             # KS6: qkv P1 (T <= 8) from its own 100 %-carveout instantiation (read at call time)\n"
     "KS1_INSTS = ((1, 16, 1), (8, 16, 4), (1, 4, 1), (1, 8, 1), (8, 8, 4))   # KS6: + NS 8 for early-resident arms;"),
]


def _scope(text, scope):
    a = text.index(scope[0])
    if text.count(scope[0]) != 1:
        raise RuntimeError(f"ks6: scope start {scope[0]!r} occurs {text.count(scope[0])} times")
    b = text.find(scope[1], a + len(scope[0]))
    return a, (len(text) if b < 0 else b)


def apply(text, patches=PATCHES):
    for what, scope, old, new in patches:
        a, b = _scope(text, scope)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"ks6 patch {what!r}: `old` occurs {n} times in its scope (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_v34_ks6_check.py")
    text = src.read_text(encoding="utf-8")
    patched = apply(text)
    out.write_bytes(patched.encode("utf-8"))   # LF as read
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(PATCHES)} patches hit once in scope; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

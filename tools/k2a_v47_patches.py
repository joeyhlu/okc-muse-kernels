"""k2a_v47_patches.py: K2A v4.7 = K1 launched as a PDL secondary of the routing tail on the small / mid-T routes (vid 0 = v3.3
NC1, T <= 8; vid 3 = v3.6 HK2, T 9..32; vid 5 = v3.7 HK3, T = 128), so K1's launch, CTA rasterisation, smem carve-up and
mbarrier init overlap the v8 tail kernel; plus an early griddepcontrol.launch_dependents in the tail kernel. Exact text
patches against kernels/muse_decode_layer/k3_v28.py. The layer embeds four K2A sources (K2A_, K2A3_, K2A31_, K2A33_KERNEL_SRC)
with identical K1 text, so each patch names the source string it edits: `old` occurs exactly once inside that string (and
the string itself once in the file); the Python-side patches are scoped to their function.

  * K1 (all three kernels): griddepcontrol.wait right before route(), the first read of the tail's outputs (topk ids /
    weights; expert_x is read later by the producer's X copies). Nothing before it reads predecessor-written data; no
    global stores before it. HOTCAP's K1 (vid 2, PDL secondary of P0, lazy waits in its producer) is unchanged
    (`if constexpr (!HOTCAP)`).
  * launch3 (non-HOTCAP K1), launch36, launch37: K1 via cudaLaunchKernelEx with programmaticStreamSerializationAllowed =
    probe bit 24 (restored to 1 for K2 / K3). Without the attribute griddepcontrol.wait returns at once (old behaviour).
  * Python: K2A47_K1_PDL (default True) sets probe bit 24 in _k2a33_call / _k2a36_call / _k2a37_call (bit 24 is not decoded
    anywhere else: probe uses bits 0-4 and HK3's dyn bits 8-23).
  * v8 tail kernel: griddepcontrol.launch_dependents right before its own griddepcontrol.wait (its grid is one wave).

    python -I tools/k2a_v47_patches.py [k3_v28.py] [out.py]   # asserts every `old` occurs exactly once in its scope
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v28.py"

K2A = ('K2A33_KERNEL_SRC = r"""', '"""')
TAIL = ('V8_TAIL_CUDA_SRC = r"""', '"""')
F33 = ("def _k2a33_call(", "\ndef ")
F36 = ("def _k2a36_call(", "\ndef ")
F37 = ("def _k2a37_call(", "\ndef ")
WAIT = '  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n'
ROUTE = "route<1, 8 * NCH, true>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);\n"


def _k1_launch(kern, ind):
    old = f"{ind}k3::{kern}<<<G, 320, SM, st>>>(p);\n{ind}C10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    new = (f"{ind}cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;\n"
           f"{ind}at[0].val.programmaticStreamSerializationAllowed = (int)((p.probe >> 24) & 1);   // v4.7: K1 after the tail\n"
           f"{ind}C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::{kern.split('<')[0]}<{kern.split('<', 1)[1]}, p));\n"
           f"{ind}at[0].val.programmaticStreamSerializationAllowed = 1;\n")
    return old, new


# (what, scope, old, new): scope = (start marker, end marker) of the region `old` must occur once in
PATCHES = [
    ("TAIL: early launch_dependents", TAIL, WAIT + "  float4 qa[NSMAX][2]",
     '  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");   // v4.7: K2A\'s K1 may launch now\n'
     + WAIT + "  float4 qa[NSMAX][2]"),
    ("K1 v3.3: wait before route (non-HOTCAP)", K2A,
     "  route<NC>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);\n",
     "  if constexpr (!HOTCAP) pdl_wait();   // v4.7: K1 is a PDL secondary of the routing tail (vid 0 / 1)\n"
     "  route<NC>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);\n"),
    ("K1 v3.6: wait before route", K2A, "  setup_bars<1, NST>(bars);\n  " + ROUTE,
     "  setup_bars<1, NST>(bars);\n  pdl_wait();   // v4.7: K1 is a PDL secondary of the routing tail (vid 3)\n  " + ROUTE),
    ("K1 v3.7: wait before route", K2A, "  setup_bars37<NST>(bars);\n  " + ROUTE,
     "  setup_bars37<NST>(bars);\n  pdl_wait();   // v4.7: K1 is a PDL secondary of the routing tail (vid 5)\n  " + ROUTE),
    ("launch3: K1 (non-HOTCAP) PDL launch", K2A, *_k1_launch("k1_kernel<NC, NST, MINB, (bool)HOTCAP>", "      ")),
    ("launch36: K1 PDL launch", K2A, *_k1_launch("k1_kernel36<NCH, NST>", "    ")),
    ("launch37: K1 PDL launch", K2A, *_k1_launch("k1_kernel37<NCH, NST>", "    ")),
    ("PY: K2A47_K1_PDL switch", ("K2A33_HINT = 16", "\n"), "K2A33_HINT = 16",
     "K2A47_K1_PDL = True   # v4.7: K1 as a PDL secondary of the routing tail on vid 0 / 3 (probe bit 24)\n"
     "K2A47_K1_PDL_T128 = False   # v4.7: same for vid 5 (T = 128): measured -2..-4 % in-layer (K1 CTAs take the SMs early)\n"
     "K2A33_HINT = 16"),
    ("PY: _k2a33_call bit 24", F33, 'st["sms"] * 2, probe)\n',
     'st["sms"] * 2, probe | ((1 << 24) if K2A47_K1_PDL else 0))\n'),
    ("PY: _k2a36_call bit 24", F36, 'st["sms"] * 2, probe)\n',
     'st["sms"] * 2, probe | ((1 << 24) if K2A47_K1_PDL else 0))\n'),
    ("PY: _k2a37_call bit 24", F37, 'st["sms"] * 2, pb)\n', 'st["sms"] * 2, pb | ((1 << 24) if K2A47_K1_PDL_T128 else 0))\n'),
]


def _scope(text, scope):
    a = text.index(scope[0])
    if text.count(scope[0]) != 1:
        raise RuntimeError(f"k2a_v47: scope start {scope[0]!r} occurs {text.count(scope[0])} times")
    b = text.find(scope[1], a + len(scope[0]))
    return a, (len(text) if b < 0 else b)


def apply(text, patches=PATCHES):
    for what, scope, old, new in patches:
        a, b = _scope(text, scope)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"k2a_v47 patch {what!r}: `old` occurs {n} times in its scope (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_v28_k2a47_check.py")
    text = src.read_text(encoding="utf-8")
    for what, scope, old, _ in PATCHES:
        a, b = _scope(text, scope)
        print(f"  {text[a:b].count(old)}x  {what}")
    patched = apply(text)
    out.write_text(patched, encoding="utf-8")
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(PATCHES)} patches hit once in scope; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

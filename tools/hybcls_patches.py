"""hybcls_patches.py (R-hybcls-0, lab file only, never a ship patch): what does a big smem request cost the HYB kernel, and where?

  python -I tools/hybcls_patches.py [--base=kernels/lab/k3_hybsm_ab.py] [--out=kernels/lab/k3_hybcls_ab.py] [--check]
  python -I tools/lead_gate.py file <out>

R-hybsm-2: the same FT_T 32 HYB code with 47,616 B of unused dynamic smem (196 -> 228 KB carveout class) costs +7.85 us at d32s; the L1
capacity reading is rejected on paper (fresh_handoff_audit.md s11: the gather keeps <= 212 lines in flight, 52 CCTL.IVALL per kernel). This
file adds six timing/diagnostic modules next to the base's ship HYB, "32" (FT_T 32) and "nx2" (x ring 2, 198,400 B); each is HYB's CUDA
source after a text transform (tools/hybx_common.py helper: own PT1 build job, own scratch, ship weight copies, prepare self-test vs FP32
and vs the ship HYB; a failing module stays OFF), routed by its own window variable (default (0, -1) = never; arms set it):
  32p   "32" + 47,616 B of UNUSED dynamic smem (231,168 B, 228 KB class): the +7.85 us reference
  32c   "32" (183,552 B) with cudaFuncAttributePreferredSharedMemoryCarveout = 100 (MaxShared): the 228 KB carveout without one extra byte
  32n   "32" with every L1-allocating global load turned into ld.global.nc.L1::no_allocate (topk_ids, x gather, normw, gsA / gsB, topk_w)
  32pn  32p with the same L1 bypass
  nx2p  "nx2" + 32,768 B of unused smem (231,168 B, 228 KB class): same depth-2 x ring as nx2, same footprint as the ship HYB
  n64   ship HYB with the L1 bypass (production shapes)
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hybx_common  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[hybcls_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


LAB = r'''HYB32P_T = (0, -1)
HYB32C_T = (0, -1)
HYB32N_T = (0, -1)
HYB32PN_T = (0, -1)
HYBNX2P_T = (0, -1)
HYBN64_T = (0, -1)
HYB32P_PAD = 47616
HYBNX2P_PAD = 32768
HYB32P_TESTS = ((32, 72, "skewed"), (32, 75, "conc1"))
_CLS_PADA = "size_t sm = okc_smem_bytes();\n"
_CLS_CARVA = "(void)e0; (void)e1;\n"
_CLS_NOL1_H = "\n".join([
    '__device__ __forceinline__ unsigned ldna_u32(const unsigned* a) { unsigned v; asm volatile("ld.global.nc.L1::no_allocate.b32 %0, [%1];" : "=r"(v) : "l"(a)); return v; }',
    '__device__ __forceinline__ float ldna_f32(const float* a) { float v; asm volatile("ld.global.nc.L1::no_allocate.f32 %0, [%1];" : "=f"(v) : "l"(a)); return v; }',
    '__device__ __forceinline__ int4 ldna_i4(const int4* a) { int4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a)); return v; }',
    '__device__ __forceinline__ __nv_bfloat16 ldna_bf16(const __nv_bfloat16* a) { unsigned short v; asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(v) : "l"(a)); return __ushort_as_bfloat16(v); }',
    ""])
_CLS_NOL1_SUBS = (
    ("__ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4)", "ldna_i4(reinterpret_cast<const int4*>(p.topk_ids) + i4)"),
    ("make_uint4(__ldg(xs32), __ldg(xs32 + 32), __ldg(xs32 + 64), __ldg(xs32 + 96))",
     "make_uint4(ldna_u32(xs32), ldna_u32(xs32 + 32), ldna_u32(xs32 + 64), ldna_u32(xs32 + 96))"),
    ("__ldg(p.normw + n0 + 8)", "ldna_bf16(p.normw + n0 + 8)"),
    ("__ldg(p.normw + n0)", "ldna_bf16(p.normw + n0)"),
    ("p.gsA[e]", "ldna_f32(p.gsA + e)"),
    ("p.gsB[e]", "ldna_f32(p.gsB + e)"),
    ("p.topk_w[tk * 8 + sl]", "ldna_f32(p.topk_w + tk * 8 + sl)"))
_CLS_NOL1_A = "__device__ __forceinline__ void st_keep_bf16("
def _cls_pad(s, n):
    if s is None or s.count(_CLS_PADA) != 1:
        return None
    return "#define OKC_SMPAD %d\n" % n + s.replace(_CLS_PADA, "size_t sm = okc_smem_bytes() + (size_t)OKC_SMPAD;\n")
def _cls_carve(s):
    if s is None or s.count(_CLS_CARVA) != 1:
        return None
    pre = ("cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);\n"
           "cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);\n")
    return s.replace(_CLS_CARVA, pre + _CLS_CARVA)
def _cls_nol1(s):
    if s is None or s.count(_CLS_NOL1_A) != 1:
        return None
    for a, b in _CLS_NOL1_SUBS:
        if s.count(a) != 1:
            return None
        s = s.replace(a, b)
    if "__ldg(" in s:
        return None
    return s.replace(_CLS_NOL1_A, _CLS_NOL1_H + _CLS_NOL1_A)
_HYBX["32p"] = {"src": _cls_pad(_HYBX["32"]["src"], HYB32P_PAD), "mod": None, "ok": False, "pdl": -1}
_HYBX["32c"] = {"src": _cls_carve(_HYBX["32"]["src"]), "mod": None, "ok": False, "pdl": -1}
_HYBX["32n"] = {"src": _cls_nol1(_HYBX["32"]["src"]), "mod": None, "ok": False, "pdl": -1}
_HYBX["32pn"] = {"src": _cls_nol1(_cls_pad(_HYBX["32"]["src"], HYB32P_PAD)), "mod": None, "ok": False, "pdl": -1}
_HYBX["nx2p"] = {"src": _cls_pad(_HYBX["nx2"]["src"], HYBNX2P_PAD), "mod": None, "ok": False, "pdl": -1}
_HYBX["n64"] = {"src": _cls_nol1(_HYB_MOE_CUDA), "mod": None, "ok": False, "pdl": -1}
_CLS_MODS = (("32p", HYB32P_TESTS), ("32c", HYB32P_TESTS), ("32n", HYB32P_TESTS), ("32pn", HYB32P_TESTS),
             ("nx2p", HYBNX2_TESTS), ("n64", HYBNX2_TESTS))
def _hybcls_prepare(st, w, dev):
    for _hn, _tests in _CLS_MODS:
        if _HYBX[_hn]["src"] is None:
            print(f"[hyb{_hn}] OFF: a source anchor is missing", flush=True)
            continue
        _hybx_prepare(_hn, w, dev, _tests)
'''
A_DISPATCH = "    if _hyb_on(x.shape[0]):\n"
N_DISPATCH = ("    for _hn, _hr in (\"32p\", HYB32P_T), (\"32c\", HYB32C_T), (\"32n\", HYB32N_T), (\"32pn\", HYB32PN_T), (\"nx2p\", HYBNX2P_T), (\"n64\", HYBN64_T):\n"
              "        if _hr[0] <= x.shape[0] <= _hr[1] and _HYBX[_hn][\"ok\"]:\n"
              "            return _hybx_call(_hn, x, local_ids.to(torch.int32).contiguous(), topk_weights.to(torch.float32).contiguous())\n"
              + A_DISPATCH)
A_PT1 = '        jobs.append(("hyb", _hyb_build))\n'
N_PT1 = (A_PT1 + '        for _hn in ("32p", "32c", "32n", "32pn", "nx2p", "n64"):\n'
         '            if _HYBX[_hn]["src"] is not None:\n'
         '                jobs.append(("hyb" + _hn, (lambda _n=_hn: _hybx_build(_n))))\n')
A_PREP = "    _hyb_prepare(st, w, dev)\n"
N_PREP = A_PREP + "    _hybcls_prepare(st, w, dev)\n"


def apply(s, base_name):
    lines = s.split(NL)
    lines[1] = (f"# nonce k3-hybcls-ab ({base_name} + R-hybcls-0 lab: HYB modules 32p / 32c / 32n / 32pn / nx2p / n64 "
                f"(unused-smem pad, carveout attribute, L1 bypass); tools/hybcls_patches.py)")
    s = NL.join(lines)
    check("\ndef _hybx_build(name):\n" in s, "base lacks the hybx helper (use k3_hybsm_ab.py)")
    check('_HYBX["32"] = ' in s and '_HYBX["nx2"] = ' in s, 'base lacks modules "32" and "nx2" (use k3_hybsm_ab.py)')
    s = sub(s, hybx_common.ANCHOR, LAB + hybx_common.ANCHOR, "lab block")
    s = sub(s, A_DISPATCH, N_DISPATCH, "dispatch")
    s = sub(s, A_PT1, N_PT1, "pt1 jobs")
    s = sub(s, A_PREP, N_PREP, "prepare hook")
    return s


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hybsm_ab.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hybcls_ab.py")
    s = apply(base.read_bytes().decode("ascii"), base.stem)
    ast.parse(s)
    check(s.isascii(), "non-ASCII output")
    check(len(s.encode()) < LIMIT, f"size {len(s.encode()):,} >= {LIMIT:,}")
    if "check" in FLAGS:
        check(out.read_bytes().decode("ascii") == s, f"{out.name} != apply({base.name})")
        print(f"[hybcls_patches] --check PASS: {out.name} == apply({base.name}) ({len(s):,} B)")
        return
    out.write_bytes(s.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"[hybcls_patches] {base.name} -> {out.name} ({len(s):,} B, {len(s) - base.stat().st_size:+,} B)")


if __name__ == "__main__":
    main()

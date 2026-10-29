"""gen_k3_icx.py: ICX = VML (T >= 17 MoE) instruction-footprint variants on k3_gc_14, each a compile-time define
(defaults = the gc_14 code; tools/icx_sass.py checks SASS == ship for the no-define build).

  VM_ICXM=mask  kept N tile counts (bit k-1 = seg<PB, 8k> instantiated; VM_NTM always kept = default case). A segment with
                k tiles runs the smallest kept count >= k (N padded up: the padded B columns read stale smem / fix rows that
                the n < cn store guards drop; every kept column's accumulation is unchanged -> bitwise == ship).
  VM_ICXF=val   compile-time A.flags (the shipped 193 at T = 32 / 128 / 256): untaken flag paths compiled out. The kernel
                raises err 9 if called with other flags.

  python tools/gen_k3_icx.py lab              -> kernels/lab/k3_icx_0.py (flags present, all off)
  python tools/gen_k3_icx.py final ARMS N     -> kernels/lab/k3_icx_<N>.py, ARMS = bf2:<code>,w72:<code> baked in
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_gc_14.py"
CODES = {   # var -> code -> extra defines
    "bf2": {"a": ("-DVM_ICXM=0x8A",), "a4": ("-DVM_ICXM=0x8B",), "s": ("-DVM_ICXM=0x80",), "f": ("-DVM_ICXF=193",),
            "af": ("-DVM_ICXM=0x8A", "-DVM_ICXF=193"), "a4f": ("-DVM_ICXM=0x8B", "-DVM_ICXF=193"),
            "sf": ("-DVM_ICXM=0x80", "-DVM_ICXF=193"),
            "e4": ("-DVM_ICXM=0x8F",), "e4f": ("-DVM_ICXM=0x8F", "-DVM_ICXF=193"), "e5": ("-DVM_ICXM=0x9F",),
            "e6": ("-DVM_ICXM=0xBF",)},
    "w72": {"a": ("-DVM_ICXM=0x10A",), "a4": ("-DVM_ICXM=0x10B",), "a8": ("-DVM_ICXM=0x18A",), "s": ("-DVM_ICXM=0x100",),
            "f": ("-DVM_ICXF=193",), "af": ("-DVM_ICXM=0x10A", "-DVM_ICXF=193"), "a4f": ("-DVM_ICXM=0x10B", "-DVM_ICXF=193"),
            "sf": ("-DVM_ICXM=0x100", "-DVM_ICXF=193")},
}


def defs(var, code):
    return () if code in ("ship", "0") else CODES[var][code]


def _sub(s, old, new, n=1):
    c = s.count(old)
    assert c == n, (c, n, old[:90])
    return s.replace(old, new)


FLAGS = """namespace vm {
#ifndef VM_ICXM
#define VM_ICXM 0xFFFF
#endif
#ifdef VM_ICXF
#define ICXFL(a) (VM_ICXF)
#else
#define ICXFL(a) ((a).flags)
#endif
"""
SW_OLD = "switch ((T.jC[j] + 7) >> 3) {\n"
SW_NEW = """#if VM_ICXM != 0xFFFF
int icx_nt = (T.jC[j] + 7) >> 3;
icx_nt += __ffs((unsigned)((VM_ICXM | (1 << (VM_NTM - 1))) >> (icx_nt - 1))) - 1;
switch (icx_nt) {
#else
switch ((T.jC[j] + 7) >> 3) {
#endif
"""
CHK_OLD = "__global__ void __launch_bounds__(NTH, 1) vm_main(const Args A) {\n"
CHK_NEW = CHK_OLD + """#ifdef VM_ICXF
if (A.flags != VM_ICXF) { if (threadIdx.x == 0) atomicExch(A.err, 9); return; }
#endif
"""


def patch(src):
    a, b = src.index('_vml_CUDA_T = r"""'), src.index('_vml_CPP = r"""')
    s = src[a:b]
    s = _sub(s, "namespace vm {\n", FLAGS)
    s = _sub(s, SW_OLD, SW_NEW)
    for k in range(1, 16):
        line = f"case {k}: seg<PB, {8 * k}>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n"
        s = _sub(s, line, f"#if (VM_ICXM >> {k - 1}) & 1\n{line}#endif\n")
    n0 = len(re.findall(r"A\.flags & ", s))
    assert n0 == 18, n0
    s = re.sub(r"A\.flags & ", "ICXFL(A) & ", s)
    s = _sub(s, CHK_OLD, CHK_NEW)
    return src[:a] + s + src[b:]


def nonce(s, tag):
    lines = s.split("\n")
    assert lines[1].startswith("# nonce "), lines[1][:40]
    lines[1] = "# nonce " + tag
    return "\n".join(lines)


def main():
    mode = sys.argv[1]
    s = patch(BASE.read_text(encoding="utf-8"))
    out = ROOT / "kernels/lab"
    if mode == "lab":
        s = nonce(s, "k3-icx-0 (k3_gc_14 + ICX VML footprint flags, all off; tools/gen_k3_icx.py)")
        p = out / "k3_icx_0.py"
    elif mode == "ab":   # ab bf2:e4+e4f,w72:... -> kernels/lab/k3_icx_ab.py (+ VML variants <var>_<code>)
        extra = {}
        for a in sys.argv[2].split(","):
            var, codes = a.split(":")
            for c in codes.split("+"):
                extra[f"{var}_{c}"] = defs(var, c)
        line = "_VML_DEFS.update({" + ", ".join(f"{k!r}: _VML_DEFS[{k[:3]!r}] + {v!r}" for k, v in extra.items()) + "})\n"
        s = _sub(s, '_VML = {"mod": None, "mods": {}, ', line + '_VML = {"mod": None, "mods": {}, ')
        s = _sub(s, "VML_ABVARS = ()\nVML_BVAR = {256: 'w72', 128: 'w72'}\n",
                 f"VML_ABVARS = {tuple(extra)!r}\nVML_BVAR = {{256: 'w72', 128: 'w72', 32: 'bf2'}}\n")
        s = _sub(s, 'scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else',
                 'scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") or var[:4] in ("bf2_", "w72_") else')
        s = _sub(s, "pairs = [(rep, v) for rep in VML_BVAR for v in dict.fromkeys((VML_BVAR[rep],) + tuple(VML_ABVARS))]",
                 "pairs = [(rep, v) for rep in VML_BVAR for v in dict.fromkeys((VML_BVAR[rep],) + tuple(VML_ABVARS)) if (v[:3] == 'bf2') == (rep == 32)]")
        s = nonce(s, f"k3-icx-ab (k3_gc_14 + ICX A/B variants {sorted(extra)}; tools/gen_k3_icx.py)")
        p = out / "k3_icx_ab.py"
    else:
        arms, n = dict(a.split(":") for a in sys.argv[2].split(",")), sys.argv[3]
        for var, code in arms.items():
            m = re.search(r'\n             "%s": \((.*?)\)' % var, s)
            assert m, var
            new = m.group(1) + "".join(f', "{x}"' for x in defs(var, code) if x not in m.group(1))
            s = s[:m.start(1)] + new + s[m.end(1):]
        s = nonce(s, f"k3-icx-{n} (k3_gc_14 + ICX {sys.argv[2]}; tools/gen_k3_icx.py)")
        p = out / f"k3_icx_{n}.py"
    assert not p.exists(), f"refusing to overwrite {p}"
    p.write_text(s, encoding="utf-8", newline="\n")
    print(f"[gen_k3_icx] wrote {p.relative_to(ROOT)} ({len(s.encode()) / 1e6:.3f} MB)")


if __name__ == "__main__":
    main()

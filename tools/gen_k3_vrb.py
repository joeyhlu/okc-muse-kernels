"""gen_k3_vrb.py: VRB = small VML (T >= 17 MoE) riders on k3_gc_14, each a compile-time flag (default 0 = gc_14 code).

Riders (codes):
  c  VCP c1 on bf2 (T = 32): -DVM_DEFER=1 -DVCP_DPI=0 -DVCP_DPH=0 -DVCP_PK=1 (w72 has it since k3_gc_13)
  h  VRB_HO:  split-share readback: all nt __ldcg(fb) loads issued before the zero stores (seg + m32e_finish)
  v  VRB_CLR: vm_fin counter clears as int4 (xready / doneA / cnt: own torch allocations, 512-B aligned; sizes % 4 == 0)
  k  VRB_SK:  vm_fin skips the product / sum-of-squares arithmetic of unused slots (c0 + jj >= nloc, block-uniform)

  python tools/gen_k3_vrb.py lab                    -> kernels/lab/k3_vrb_0.py  (flags present, all off)
  python tools/gen_k3_vrb.py ab SET[,SET...]        -> kernels/lab/k3_vrb_ab.py (+ VML variants bf2_<SET>, w72_<SET>, bf2_0, w72_0)
  python tools/gen_k3_vrb.py final SET N            -> kernels/lab/k3_vrb_<N>.py (riders baked into bf2 / w72)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_gc_14.py"
C1 = ("-DVM_DEFER=1", "-DVCP_DPI=0", "-DVCP_DPH=0", "-DVCP_PK=1")
RIDER = {"h": ("-DVRB_HO=1",), "v": ("-DVRB_CLR=1",), "k": ("-DVRB_SK=1",)}


def defs(var, codes):
    out = []
    if var == "bf2" and "c" in codes:
        out += list(C1)
    for ch in codes:
        out += list(RIDER.get(ch, ()))
    return tuple(out)


def _sub(s, old, new, n=1):
    c = s.count(old)
    assert c == n, (c, n, old[:90])
    return s.replace(old, new)


FLAGS = """#ifndef VRB_HO
#define VRB_HO 0    // VRB: split readback: all nt loads before the zero stores
#endif
#ifndef VRB_CLR
#define VRB_CLR 0   // VRB: vm_fin counter clears as int4
#endif
#ifndef VRB_SK
#define VRB_SK 0    // VRB: vm_fin skips the arithmetic of unused slots
#endif
#ifndef VM_DEFER
#define VM_DEFER 0
#endif
"""

RB_OLD = """fence_acq_rel();
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
*reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;
}
"""
RB_NEW = """fence_acq_rel();
#if VRB_HO
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;
}
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) *reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
#else
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
*reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;
}
#endif
"""

CLR_OLD = """if (t == 0) {
for (int i = tid; i < JMAX; i += 256) A.xready[i] = 0;
for (int i = tid; i < JMAX * 16; i += 256) A.doneA[i] = 0;
for (int i = tid; i < 2 * SLOTSD * 4; i += 256) A.cnt[i] = 0;
"""
CLR_NEW = """if (t == 0) {
#if VRB_CLR
static_assert(JMAX % 4 == 0, "VRB_CLR: xready as int4");
const int4 zi = make_int4(0, 0, 0, 0);
for (int i = tid; i < JMAX / 4; i += 256) reinterpret_cast<int4*>(A.xready)[i] = zi;
for (int i = tid; i < JMAX * 4; i += 256) reinterpret_cast<int4*>(A.doneA)[i] = zi;
for (int i = tid; i < 2 * SLOTSD; i += 256) reinterpret_cast<int4*>(A.cnt)[i] = zi;
#else
for (int i = tid; i < JMAX; i += 256) A.xready[i] = 0;
for (int i = tid; i < JMAX * 16; i += 256) A.doneA[i] = 0;
for (int i = tid; i < 2 * SLOTSD * 4; i += 256) A.cnt[i] = 0;
#endif
"""

SK1_OLD = """#if VM_FTR
#pragma unroll
for (int jj = 0; jj < 4; ++jj)
#pragma unroll
for (int m = 0; m < 4; ++m) {   // VTR: products once
v[jj][m].x = bf16r(v[jj][m].x) * w4[m].x; v[jj][m].y = bf16r(v[jj][m].y) * w4[m].y;
v[jj][m].z = bf16r(v[jj][m].z) * w4[m].z; v[jj][m].w = bf16r(v[jj][m].w) * w4[m].w;
}
#endif
"""
SK1_NEW = """#if VM_FTR
#pragma unroll
for (int jj = 0; jj < 4; ++jj) {
#if VRB_SK
if (c0 + jj >= nloc) continue;   // VRB: block-uniform (s_n)
#endif
#pragma unroll
for (int m = 0; m < 4; ++m) {   // VTR: products once
v[jj][m].x = bf16r(v[jj][m].x) * w4[m].x; v[jj][m].y = bf16r(v[jj][m].y) * w4[m].y;
v[jj][m].z = bf16r(v[jj][m].z) * w4[m].z; v[jj][m].w = bf16r(v[jj][m].w) * w4[m].w;
}
}
#endif
"""
SK2_OLD = """for (int jj = 0; jj < 4; ++jj) {
ss[jj] = 0.f;
#pragma unroll
"""
SK2_NEW = """for (int jj = 0; jj < 4; ++jj) {
ss[jj] = 0.f;
#if VRB_SK
if (c0 + jj >= nloc) continue;   // VRB: whole block skips (shuffles stay warp-uniform); s_ss gets 0, never read
#endif
#pragma unroll
"""


def patch(src):
    s = _sub(src, "#ifndef VM_DEFER\n#define VM_DEFER 0\n#endif\n", FLAGS)
    s = _sub(s, RB_OLD, RB_NEW, 2)
    s = _sub(s, CLR_OLD, CLR_NEW)
    s = _sub(s, SK1_OLD, SK1_NEW)
    s = _sub(s, SK2_OLD, SK2_NEW)
    return s


def nonce(s, tag):
    lines = s.split("\n")
    assert lines[1].startswith("# nonce "), lines[1][:40]
    lines[1] = "# nonce " + tag
    return "\n".join(lines)


def main():
    mode = sys.argv[1]
    s = BASE.read_text(encoding="utf-8")
    if not (mode == "final" and set(sys.argv[2]) <= {"c"}):   # c alone is a defines-only change: no rider flags in the file
        s = patch(s)
    out = ROOT / "kernels/lab"
    if mode == "lab":
        s = nonce(s, "k3-vrb-0 (k3_gc_14 + VRB rider flags, all off; tools/gen_k3_vrb.py)")
        p = out / "k3_vrb_0.py"
    elif mode == "ab":
        sets = sys.argv[2].split(",")
        extra = {"bf2_0": (), "w72_0": ()}
        for st in sets:
            extra[f"bf2_{st}"] = defs("bf2", st)
            wc = st.replace("c", "")
            if wc:
                extra[f"w72_{wc}"] = defs("w72", wc)
        line = "_VML_DEFS.update({" + ", ".join(f"{k!r}: _VML_DEFS[{k[:3]!r}] + {v!r}" for k, v in extra.items()) + "})\n"
        s = _sub(s, '_VML = {"mod": None, "mods": {}, ', line + '_VML = {"mod": None, "mods": {}, ')
        s = _sub(s, "VML_ABVARS = ()\nVML_BVAR = {256: 'w72', 128: 'w72'}\n",
                 f"VML_ABVARS = {tuple(extra)!r}\nVML_BVAR = {{256: 'w72', 128: 'w72', 32: 'bf2'}}\n")
        s = _sub(s, 'scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else',
                 'scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") or var[:4] in ("bf2_", "w72_") else')
        s = _sub(s, "pairs = [(rep, v) for rep in VML_BVAR for v in dict.fromkeys((VML_BVAR[rep],) + tuple(VML_ABVARS))]",
                 "pairs = [(rep, v) for rep in VML_BVAR for v in dict.fromkeys((VML_BVAR[rep],) + tuple(VML_ABVARS)) if (v[:3] == 'bf2') == (rep == 32)]")
        s = nonce(s, f"k3-vrb-ab (k3_gc_14 + VRB A/B variants {sorted(extra)}; tools/gen_k3_vrb.py)")
        p = out / "k3_vrb_ab.py"
    else:
        st, n = sys.argv[2], sys.argv[3]
        for var in ("bf2", "w72"):
            add = defs(var, st if var == "bf2" else st.replace("c", ""))
            m = re.search(r'\n             "%s": \((.*?)\)' % var, s)
            assert m, var
            new = m.group(1) + "".join(f', "{a}"' for a in add if a not in m.group(1))
            s = s[:m.start(1)] + new + s[m.end(1):]
        s = nonce(s, f"k3-vrb-{n} (k3_gc_14 + VRB riders '{st}': {sorted(st)}; tools/gen_k3_vrb.py)")
        p = out / f"k3_vrb_{n}.py"
    p.write_text(s, encoding="utf-8", newline="\n")
    print(f"[gen_k3_vrb] wrote {p.relative_to(ROOT)} ({len(s.encode()) / 1e6:.3f} MB)")


if __name__ == "__main__":
    main()

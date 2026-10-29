"""gen_k3_v32.py: V32 = VML (T >= 17 MoE, bf2 at bucket 32) in-layer phase budget + one fix, on k3_gc_18.

  python tools/gen_k3_v32.py marks        -> kernels/lab/k3_v32_m.py  (VM_MK2 per-CTA phase marks on bf2, VML_MARKS hook)

VM_MK2 marks (row c = vm_main CTA, MKW 32 columns; the MKW-32 segment counts 8..23 and the lat probes are dropped):
  0 entry (after mbar init)  7 route end  5 fill end  6 phase A end (q0)  1..4 phase B end (q 0..3)
  8 first weight stage issued  9/10/11 stage 0/1/2 landed (idle lane 1 of the w-producer warp)  12 first compute (q0 wp0)
  13 q0 weight full-wait ns (A+B)  14 = 13 at phase A end  15 q0 x full-wait ns (A+B)  21 = 15 at phase A end
  16 producer empty-wait ns  17 last A issue  20 last B issue  18 / 19 A / B stages issued
  24..28 (ship) B doneA wait ns, B stages that waited, A xready wait ns, first / last B x acquire
  row G: 0 vm_fin first wait-return (atomicMin, host sets INT64_MAX), 1 vm_fin end (atomicMax), 3 vm_fin first entry (atomicMin)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_gc_18.py"


def _sub(s, old, new, n=1):
    c = s.count(old)
    assert c == n, (c, n, old[:90])
    return s.replace(old, new)


def marks(s, v2=False):
    """v2: ship MKW-32 columns untouched (segment counts kept); V32 columns in row G + 1 + c; no lane-1 landing waits.
    v2 row: 0 first issue, 1 first compute, 2 / 3 weight full-wait (sum / at A end), 4 / 5 x full-wait, 6 producer empty-wait,
    7 / 8 last A / B issue, 9 / 10 A / B stages issued."""
    R = (lambda c: f"A.marks + (size_t)(A.G0 + 1 + {c}) * MKW") if v2 else (lambda c: f"A.marks + (size_t){c} * MKW")
    C = (lambda k: {8: 0, 12: 1, 13: 2, 14: 3, 15: 4, 21: 5, 16: 6}[k]) if v2 else (lambda k: k)
    s = _sub(s, "#ifndef VM_MKW\n#define VM_MKW 32\n#endif\n",
             "#ifndef VM_MKW\n#define VM_MKW 32\n#endif\n#ifndef VM_MK2\n#define VM_MK2 0   // V32: in-layer phase marks (lab only)\n#endif\n")
    s = _sub(s, "m[MKP + 1] = lat_probe(A.ws13, A.flags);\nm[MKP + 2] = lat_probe(A.ws13 + 4096, A.flags);\n",
             "#if !VM_MK2\nm[MKP + 1] = lat_probe(A.ws13, A.flags);\nm[MKP + 2] = lat_probe(A.ws13 + 4096, A.flags);\n#endif\n")
    if not v2:
        s = _sub(s, "#else\nif (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * 32 + 8 + 8 * PB",
                 "#elif !VM_MK2\nif (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * 32 + 8 + 8 * PB")
    s = _sub(s, "#if VM_MKW == 64\nint G0;\n#endif\n};", "#if VM_MKW == 64 || VM_MK2\nint G0;\n#endif\n};")
    s = _sub(s, "#if VM_MKW == 64\na.G0 = (int)G;\n#endif", "#if VM_MKW == 64 || VM_MK2\na.G0 = (int)G;\n#endif")
    # producer: first issue, empty waits, stage counts, last issue (non-DP loop only; bf2 has VM_DP 0)
    s = _sub(s, "if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);\n#if VM_PS",
             "#if VM_MK2\nconst long long mk2t = A.marks ? gtime() : 0;\n#endif\n"
             "if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);\n#if VM_PS")
    tail = (" if (it == 0) mk[0] = t1; mk[9 + pb] += 1; mk[7 + pb] = t1; }" if v2 else
            " if (it == 0) mk[8] = t1; mk[18 + pb] += 1; mk[17 + 3 * pb] = t1; }")
    s = _sub(s, "#endif\nconst uint32_t fb = bars + 8 * sl, dst = ring + sl * US;\nmbar_expect_tx(fb, US);",
             "#endif\n#if VM_MK2\nif (A.marks) { long long* mk = " + R("c") + f"; const long long t1 = gtime(); mk[{C(16)}] += t1 - mk2t;"
             + tail + "\n#endif\nconst uint32_t fb = bars + 8 * sl, dst = ring + sl * US;\nmbar_expect_tx(fb, US);")
    if not v2:   # stage 0/1/2 landing: idle lane 1 of the weight-producer warp (perturbs the first issue: v1 only)
        s = _sub(s, "if (lane == 0) prod_w(A, T, sa(smem), bars);\n",
                 "if (lane == 0) prod_w(A, T, sa(smem), bars);\n#if VM_MK2\nelse if (lane == 1 && A.marks) {\n"
                 "const int c = blockIdx.x, ns = (T.bA[c + 1] - T.bA[c]) + (T.bB[c + 1] - T.bB[c]);\n"
                 "long long* mk = A.marks + (size_t)c * MKW;\n"
                 "for (int i = 0; i < NS && i < ns; ++i) { mbar_wait(bars + 8 * i, 0, A.err); mk[9 + i] = gtime(); }\n}\n#endif\n", n=2)
    # consumer full waits (weights / x) in stage + stage_m
    W = "mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);"
    s = _sub(s, W, "{\n#if VM_MK2\nconst bool mo = A.marks && q == 0 && wp == 0 && lane == 0; const long long tw0 = mo ? gtime() : 0;\n#endif\n"
             + W + "\n#if VM_MK2\nif (mo) { long long* mk = " + R("blockIdx.x") + "; const long long tw1 = gtime();"
             + f" if (it == 0) mk[{C(12)}] = tw1; mk[{C(13)}] += tw1 - tw0; }}" + "\n#endif\n}", n=2)
    X = "mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);"
    s = _sub(s, X, "{\n#if VM_MK2\nconst bool mo = A.marks && q == 0 && wp == 0 && lane == 0; const long long tw0 = mo ? gtime() : 0;\n#endif\n"
             + X + "\n#if VM_MK2\nif (mo) (" + R("blockIdx.x") + f")[{C(15)}] += gtime() - tw0;" + "\n#endif\n}", n=2)
    snap = ("{ long long* m2 = " + R("blockIdx.x") + "; m2[3] = m2[2]; m2[5] = m2[4]; }") if v2 else "mk[14] = mk[13]; mk[21] = mk[15];"
    s = _sub(s, "if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();",
             "if (wp == 0 && lane == 0 && A.marks && q == 0) { long long* mk = A.marks + (size_t)blockIdx.x * MKW; mk[6] = gtime();"
             "\n#if VM_MK2\n" + snap + "\n#endif\n}")
    # vm_fin entry / wait-return / end into row G
    s = _sub(s, "const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n#if VM_FT\n",
             "const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n#if VM_MK2\n"
             "if (tid == 0 && A.marks) atomicMin(reinterpret_cast<unsigned long long*>(A.marks + (size_t)A.G0 * MKW + 3), (unsigned long long)gtime());\n"
             "#endif\n#if VM_FT\n")
    s = _sub(s, "asm volatile(\"griddepcontrol.wait;\\n\" ::: \"memory\");\n#if VM_MKW == 64\nif (tid == 0 && A.marks) atomicMin(",
             "asm volatile(\"griddepcontrol.wait;\\n\" ::: \"memory\");\n#if VM_MKW == 64 || VM_MK2\nif (tid == 0 && A.marks) atomicMin(")
    s = _sub(s, "#if VM_MKW == 64\n__syncthreads();\nif (tid == 0 && A.marks) atomicMax(",
             "#if VM_MKW == 64 || VM_MK2\n__syncthreads();\nif (tid == 0 && A.marks) atomicMax(")
    # host: marks hook at bucket 32, MK2 on the bf2 build
    s = _sub(s, "VML_PDEL = {}\n", "VML_PDEL = {}\nVML_MARKS = None   # V32: int64 tensor ((G + 1) * 32) -> marks at bucket 32 (lab)\n")
    s = _sub(s, "flags=fl, cost=VML_COST.get(rep), pool=",
             "flags=fl, marks=(VML_MARKS if rep == 32 else None), cost=VML_COST.get(rep), pool=")
    s = _sub(s, '"bf2": ("-DVM_TRICKS_DQ=0",', '"bf2": ("-DVM_MK2=1", "-DVM_TRICKS_DQ=0",')
    return s


def s0p(s, var_defs):
    """VM_S0P (ns cap, 0 = off): the weight producer issues stage 0 alone; stages 1 .. NS-1 wait until stage 0 landed
    (full bar 0, phase 0) or VM_S0P ns after the stage-0 issue."""
    s = _sub(s, "#ifndef VM_MKW\n#define VM_MKW 32\n#endif\n",
             "#ifndef VM_MKW\n#define VM_MKW 32\n#endif\n#ifndef VM_S0P\n#define VM_S0P 0   // V32: stage-0 priority (ns cap)\n#endif\n")
    s = _sub(s, "__device__ void prod_w(const Args& A, const Tab& T, uint32_t ring, uint32_t bars) {\nconst uint64_t pol = policy_evict_first();\n",
             "__device__ void prod_w(const Args& A, const Tab& T, uint32_t ring, uint32_t bars) {\nconst uint64_t pol = policy_evict_first();\n"
             "#if VM_S0P\nlong long s0t = 0;\n#endif\n")
    s = _sub(s, "const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;\nmbar_expect_tx(fb, US);\nbulk_g2s_ef(dst, src, SW, fb, pol);\n"
             "bulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);\n}\n#if VM_DP",
             "#if VM_S0P\nif (it == 0) s0t = gtime();\nelse if (it < NS) { while (!mbar_try(bars, 0) && gtime() - s0t < VM_S0P) { } }\n#endif\n"
             "const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;\nmbar_expect_tx(fb, US);\nbulk_g2s_ef(dst, src, SW, fb, pol);\n"
             "bulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);\n}\n#if VM_DP")
    if var_defs:   # extra bf2 variants (validated through VML_BVAR[32] / VML_ABVARS)
        s = _sub(s, '_VML_DEFS = {"": (),', "_VML_DEFS = {" + "".join(f'"{v}": {d!r}, ' for v, d in var_defs.items()) + '"": (),')
    return s


def main():
    mode = sys.argv[1]
    s = BASE.read_text()
    if mode == "marks":
        out, s = ROOT / "kernels/lab/k3_v32_m.py", marks(s)
    elif mode == "marks2":
        out, s = ROOT / "kernels/lab/k3_v32_m2.py", marks(s, v2=True)
    elif mode.startswith("s0p"):   # s0p N NS1,NS2,..: lab file with bf2 variants bf2s<ns> (marks on when mode == s0pm)
        n, caps = sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
        if mode in ("s0pm", "s0pm2"):
            s = marks(s, v2=(mode == "s0pm2"))
        bf2 = eval(s.split('"bf2": ', 1)[1].split("),", 1)[0] + ")")   # the (patched) bf2 define tuple
        s = s0p(s, {f"bf2s{c}": bf2 + (f"-DVM_S0P={c}",) for c in caps})
        s = _sub(s, "VML_ABVARS = ()\n", f"VML_ABVARS = {tuple(f'bf2s{c}' for c in caps)!r}\n")
        s = _sub(s, "VML_BVAR = {256: 'w72', 128: 'w72'}\n", f"VML_BVAR = {{256: 'w72', 128: 'w72', 32: 'bf2s{caps[0]}'}}\n")
        s = _sub(s, 'scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]',
                 'scx = _VML["scb"] if (var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") or var.startswith("bf2s")) else _VML["sc"]')
        out = ROOT / f"kernels/lab/k3_v32_{n}.py"
    elif mode == "final":   # final N "-DVM_S0P=1500 ...": the fix baked into the bf2 build of k3_gc_18
        n, extra = sys.argv[2], tuple(sys.argv[3].split())
        s = s0p(s, {})
        s = _sub(s, '"bf2": ("-DVM_TRICKS_DQ=0",', '"bf2": (' + "".join(f'"{d}", ' for d in extra) + '"-DVM_TRICKS_DQ=0",')
        out = ROOT / f"kernels/lab/k3_v32_{n}.py"
    else:
        raise SystemExit(f"unknown mode {mode}")
    assert not out.exists(), f"{out} exists (never overwrite)"
    compile(s, str(out), "exec")
    out.write_text(s, newline="\n")
    print(out, len(s.encode()))


if __name__ == "__main__":
    main()

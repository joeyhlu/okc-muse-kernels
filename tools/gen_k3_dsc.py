"""R-dsc: discard dead VML intermediates from L2 (no write-back) + move the y / fix zeroing into the gather (gc_16 base).

  py tools/gen_k3_dsc.py <n>   -> kernels/lab/k3_dsc_<n>.py (ship: VML bf2d / w72d) + kernels/lab/k3_dsc_<n>_ab.py (A/B:
                                  also builds bf2 / w72 for the gc_16 arm, own y / fix scratch per arm)
CBL (R-cbl-0/1) showed the T >= 128 dense streams pay write-backs of DIRTY L2 lines the weight stream evicts. VML leaves
xA / xB (gather / swiglu scratch), y (Phase-B output, re-zeroed with __stcg for the next call's red.add) and fix (Phase-A split
partials, re-zeroed by the last arriver) dirty right before the unpooler. VM_DSC (compile-time, default 0):
  bit 1: vm_main block 0 publishes Rpad in err[2]; vm_fin (after griddepcontrol.wait) discards xA / xB rows [0, Rpad).
  bit 2: the gather zeroes the y tiles of THIS call's split Phase-B units before its xready release (every contributor
         acquires xready[j] first: A units directly, B units via doneA), vm_fin discards each y row after its read
         (instead of the __stcg re-zero).
  bit 4: the gather zeroes this call's split Phase-A fix slots the same way; the last arriver discards them after its read.
Lifetimes (writer / last reader / next writer) are in notes/study/dsc.md. Never overwrites a file.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels/muse_decode_layer/k3_gc_16.py"


def sub(text, old, new, count=1):
    assert text.count(old) == count, (text.count(old), old[:120])
    return text.replace(old, new)


def write_new(p, text):
    assert not p.exists(), f"{p} exists"
    assert len(text.encode()) < 1_000_000, len(text.encode())
    p.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {p} ({len(text.encode())} B)")


CU = [
    # defines
    ("#ifndef VM_DEFER2\n#define VM_DEFER2 0\n#endif\n",
     "#ifndef VM_DEFER2\n#define VM_DEFER2 0\n#endif\n"
     "#ifndef VM_DSC\n#define VM_DSC 0   // R-dsc bits: 1 xA / xB discard, 2 y zero-in-gather + discard, 4 fix zero-in-gather + discard\n#endif\n"),
    ('static_assert(!(VM_BRED && VM_DP), "m32e: VM_BRED and VM_DP use different split paths");\n',
     'static_assert(!(VM_BRED && VM_DP), "m32e: VM_BRED and VM_DP use different split paths");\n'
     'static_assert(!(VM_DSC & 6) || (VM_BRED && !VM_DP && !VM_AD), "dsc: zero-in-gather needs VM_BRED, no DP / AD");\n'),
    ('__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.global;\\n" ::: "memory"); }\n',
     '__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.global;\\n" ::: "memory"); }\n'
     '__device__ __forceinline__ void discard_l2(const void* p) { asm volatile("discard.global.L2 [%0], 128;\\n" :: "l"(p) : "memory"); }\n'),
    # zero-in-gather helper (before the first gather)
    ("__device__ void gather(const Args& A, const Tab& T, const uint16_t* ent) {\n",
     "#if VM_DSC & 6\n"
     "__device__ __forceinline__ int dsc_lb(const int* b, int lo, int hi, int v) {   // first i in [lo, hi) with b[i] >= v (b nondecreasing)\n"
     "while (lo < hi) { const int m = (lo + hi) >> 1; if (b[m] < v) lo = m + 1; else hi = m; }\n"
     "return lo;\n"
     "}\n"
     "// DSC: zero what THIS call's split units accumulate into, before this gatherer's xready release. Split units are the ones\n"
     "// with a CTA boundary bA / bB strictly inside (b & 15); Phase-B split units red.add into y tiles (cols 256 ub.., rows\n"
     "// jS[j] + n, n < cn), Phase-A split units red.add into fix slot cta_of(u << 4) (4 q x npad * 64 floats). Every contributor\n"
     "// of a unit of job j acquires xready[j] (== npad, all gatherers of job j) first: A units directly, B units via doneA.\n"
     "// Padded row r of job j zeroes chunk r (64 floats per q) of each split A slot and, for r < cn, y row jS[j] + r of each split\n"
     "// B tile: the rows partition each region exactly.\n"
     "__device__ __forceinline__ void dsc_zero(const Args& A, const Tab& T, int q0, int q1, int th, int nth) {\n"
     "const int G = gridDim.x;\n"
     "const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);\n"
     "int j = 0, jl = -1, a0 = 0, a1 = 0, b0 = 0, b1 = 0;\n"
     "for (int q = q0; q < q1; ++q) {\n"
     "while (T.jrow[j + 1] <= q) ++j;\n"
     "if (j != jl) {\n"
     "jl = j;\n"
     "a0 = dsc_lb(T.bA, 1, G, j << 9); a1 = dsc_lb(T.bA, a0, G, (j + 1) << 9);\n"
     "b0 = dsc_lb(T.bB, 1, G, j << 8); b1 = dsc_lb(T.bB, b0, G, (j + 1) << 8);\n"
     "}\n"
     "const int r = q - T.jrow[j];\n"
     "#if VM_DSC & 4\n"
     "for (int c = a0, lu = -1; c < a1; ++c) {\n"
     "const int b = T.bA[c];\n"
     "if ((b & 15) == 0 || (b >> 4) == lu) continue;\n"
     "lu = b >> 4;\n"
     "float* fb = A.fix + (size_t)cta_of(T, 0, lu << 4, G) * 4 * FIXW + r * 64;\n"
     "for (int k = th; k < 64; k += nth) *reinterpret_cast<float4*>(fb + (size_t)(k >> 4) * FIXW + 4 * (k & 15)) = z4;\n"
     "}\n"
     "#endif\n"
     "#if VM_DSC & 2\n"
     "if (r < T.jC[j]) {\n"
     "float* yr = A.y + (size_t)(T.jS[j] + r) * KDIM;\n"
     "for (int c = b0, lu = -1; c < b1; ++c) {\n"
     "const int b = T.bB[c];\n"
     "if ((b & 15) == 0 || (b >> 4) == lu) continue;\n"
     "lu = b >> 4;\n"
     "for (int k = th; k < 64; k += nth) *reinterpret_cast<float4*>(yr + (lu & 15) * 256 + 4 * k) = z4;\n"
     "}\n"
     "}\n"
     "#endif\n"
     "}\n"
     "}\n"
     "#endif\n"
     "__device__ void gather(const Args& A, const Tab& T, const uint16_t* ent) {\n"),
    # consumer gather: zero before the named barrier + release
    ("fence_proxy_async();\nbar_sync(5, 32 * NCW);\n",
     "#if VM_DSC & 6\ndsc_zero(A, T, q0, q1, tid, 32 * NCW);\n#endif\nfence_proxy_async();\nbar_sync(5, 32 * NCW);\n"),
    # Rpad for vm_fin's xA / xB discard
    ("route(A, T, hist, ent, reinterpret_cast<uint32_t*>(scr + 12288));\n",
     "route(A, T, hist, ent, reinterpret_cast<uint32_t*>(scr + 12288));\n"
     "#if VM_DSC & 1\nif (blockIdx.x == 0 && tid == 0) A.err[2] = T.Rpad;   // DSC: vm_fin discards xA / xB rows [0, Rpad)\n#endif\n"),
    # fix: last arriver discards instead of re-zeroing
    ("#pragma unroll\nfor (int nt = 0; nt < N / 8; ++nt) {\n"
     "const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));\n"
     "*reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);\n"
     "d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;\n}\n",
     "#pragma unroll\nfor (int nt = 0; nt < N / 8; ++nt) {\n"
     "const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));\n"
     "#if !(VM_DSC & 4)\n"
     "*reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);\n"
     "#endif\n"
     "d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;\n}\n"
     "#if VM_DSC & 4\n"
     "bar_sync(1 + q, 128);   // DSC: every read of this slot is performed; the next call's gather re-zeroes it\n"
     "for (int l = wt; l < N * 2; l += 128) discard_l2(fb - wt * 4 + 32 * l);\n"
     "#endif\n", 2),   # seg + m32e_finish (VM_AD only: excluded by the static_assert)
    # R-dsc-1 finding: self_fill (flags & 64) feeds each CTA's first <= NX-1 A stages (= the split units) straight from x,
    # without acquiring xready. Restore the chain: every doneA release and every split-A fix red.add acquires xready[j]
    # (complete by then except on self-filled units) before it.
    ("#else\nstruct Pend { int i; };\n#endif\n",
     "#elif VM_DSC & 6\nstruct Pend { int i; int np; };   // np: npad of job i >> 4 (DSC: xready target)\n"
     "#else\nstruct Pend { int i; };\n#endif\n"),
    ("if (((wp << 5) | lane) == 0) red_release_add(A.doneA + pd.i, 1);\n",
     "#if VM_DSC & 6\nif (((wp << 5) | lane) == 0) { wait_flag(A.xready + (pd.i >> 4), pd.np, A.err); red_release_add(A.doneA + pd.i, 1); }\n"
     "#else\nif (((wp << 5) | lane) == 0) red_release_add(A.doneA + pd.i, 1);\n#endif\n"),
    ("pd.i = j * 16 + (ub >> 1);\n",
     "pd.i = j * 16 + (ub >> 1);\n#if VM_DSC & 6\npd.np = npad;\n#endif\n"),
    ("if (wt == 0) red_release_add(A.doneA + j * 16 + (ub >> 1), 1);\n",
     "#if VM_DSC & 6\nif (wt == 0) { wait_flag(A.xready + j, npad, A.err); red_release_add(A.doneA + j * 16 + (ub >> 1), 1); }\n"
     "#else\nif (wt == 0) red_release_add(A.doneA + j * 16 + (ub >> 1), 1);\n#endif\n"),
    ("#pragma unroll\nfor (int nt = 0; nt < N / 8; ++nt) red_add_v4(fb + nt * 512, ",
     "#if VM_DSC & 4\nwait_flag(A.xready + j, (T.jC[j] + 7) & ~7, A.err);   // DSC: the gather's fix zero precedes this red\n#endif\n"
     "#pragma unroll\nfor (int nt = 0; nt < N / 8; ++nt) red_add_v4(fb + nt * 512, "),
    # vm_fin: y rows dropped after the read (the next call's gather zeroes its split tiles)
    ("#if VM_BRED\n#pragma unroll\nfor (int jj = 0; jj < 4; ++jj) {\n"
     "const int r = (c0 + jj < nloc) ? s_r[c0 + jj] : -1;\nif (r >= 0) {\n#pragma unroll\n"
     "for (int m = 0; m < 4; ++m) __stcg(reinterpret_cast<float4*>(A.y + (size_t)r * KDIM + 1024 * m + 4 * tid), z4);\n}\n}\n#endif\n",
     "#if VM_BRED && (VM_DSC & 2)\n"
     "if (tid < 128) {   // DSC: every read of these rows precedes the __syncthreads above\n"
     "#pragma unroll\nfor (int jj = 0; jj < 4; ++jj) {\n"
     "const int r = (c0 + jj < nloc) ? s_r[c0 + jj] : -1;\n"
     "if (r >= 0) discard_l2(A.y + (size_t)r * KDIM + 32 * tid);\n}\n}\n"
     "#elif VM_BRED\n#pragma unroll\nfor (int jj = 0; jj < 4; ++jj) {\n"
     "const int r = (c0 + jj < nloc) ? s_r[c0 + jj] : -1;\nif (r >= 0) {\n#pragma unroll\n"
     "for (int m = 0; m < 4; ++m) __stcg(reinterpret_cast<float4*>(A.y + (size_t)r * KDIM + 1024 * m + 4 * tid), z4);\n}\n}\n#endif\n"),
    # vm_fin: xA / xB discard after the counter resets (vm_main has completed: griddepcontrol.wait above)
    ("for (int i = tid; i < 2 * SLOTSD * 4; i += 256) A.cnt[i] = 0;\n#if VM_DP\nif (tid == 0) { A.gbar[0] = 0; A.gbar[1] = 0; }\n#endif\n}\n",
     "for (int i = tid; i < 2 * SLOTSD * 4; i += 256) A.cnt[i] = 0;\n#if VM_DP\nif (tid == 0) { A.gbar[0] = 0; A.gbar[1] = 0; }\n#endif\n}\n"
     "#if VM_DSC & 1\n"
     "{   // DSC: xA / xB rows [0, Rpad) are dead (next writer: the next call's gather / Phase A, before any read)\n"
     "const int nl = min(__ldcg(A.err + 2), XROWS) * 64;   // 128-B lines per buffer (8 KB rows)\n"
     "for (int i = t * 256 + tid; i < nl; i += (int)gridDim.x * 256) { discard_l2(A.xA + (size_t)i * 64); discard_l2(A.xB + (size_t)i * 64); }\n"
     "}\n"
     "#endif\n"),
]


FLAG_B = "A.cnt + (SLOTS + {z}) * 4 + 2"   # PB = 1 fix counters: unused under VM_BRED, reset by vm_fin every call
# v3 (k3_dsc_3+; R-dsc-1 attempt 3 showed the xready-chained gather zero costs vm_main +5..+11 us): bit 1 as before;
# bit 2 = y zero by ONE owner CTA per split Phase-B unit, done by its two idle gather warps (parts 1 / 2 of gather_w5,
# after their barrier-6 arrival, so prod_x is not delayed) + a release flag; every red.add of a split B unit acquires the
# owner's flag (== 2) first. Owner z of split unit u = cta_of(B, u << 4) + 1 = the CTA whose first B unit is u: it reaches
# Phase B long after its own gather warps finished, and the other contributor (z - 1) reds u as its LAST unit. fix (bit 4)
# is dropped: its early contributor (the self-filled first A unit) would wait on any zeroer.
CU3 = [CU[0],
       ('static_assert(!(VM_BRED && VM_DP), "m32e: VM_BRED and VM_DP use different split paths");\n',
        'static_assert(!(VM_BRED && VM_DP), "m32e: VM_BRED and VM_DP use different split paths");\n'
        'static_assert(!(VM_DSC & 6) || (VM_BRED && !VM_DP && !VM_AD && !(VM_DSC & 4)), '
        '"dsc v4: y owner-zero needs VM_BRED, no DP / AD; no fix bit");\n'),
       CU[2], CU[5],
       # v4 (k3_dsc_5+): the zero runs on the owner CTA's CONSUMER warps between its Phase A and Phase B (the producer
       # warps run under setmaxnreg 32: k3_dsc_4 failed at router 491), then a consumer barrier and one release.
       ("if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();\n",
        "if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();\n"
        "#if VM_DSC & 2\n"
        "{   // DSC v4: owner zero of the split Phase-B unit u this CTA starts in (its first B unit; see tools/gen_k3_dsc.py)\n"
        "const int c = blockIdx.x, b = T.bB[c], u = b >> 4;\n"
        "if (c > 0 && (b & 15) != 0 && T.bB[c - 1] <= (u << 4)) {\n"
        "const int j = u >> 4, cn = T.jC[j];\n"
        "float* yb = A.y + (size_t)T.jS[j] * KDIM + (u & 15) * 256;\n"
        "for (int k = tid; k < cn * 64; k += 32 * NCW) *reinterpret_cast<float4*>(yb + (size_t)(k >> 6) * KDIM + 4 * (k & 63)) = make_float4(0.f, 0.f, 0.f, 0.f);\n"
        "bar_sync(5, 32 * NCW);\n"
        "if (tid == 0) red_release_add(" + FLAG_B.format(z="c") + ", 1);\n"
        "}\n"
        "}\n"
        "#endif\n"),
       ("SEGMK(const long long t_r0 = (A.marks && wp == 0 && lane == 0 && q == 0) ? gtime() : 0;)\n",
        "SEGMK(const long long t_r0 = (A.marks && wp == 0 && lane == 0 && q == 0) ? gtime() : 0;)\n"
        "#if VM_DSC & 2\n"
        "{ const int z = cta_of(T, 1, (j * 16 + ub) << 4, gridDim.x) + 1; wait_flag(" + FLAG_B.format(z="z") + ", 1, A.err); }   // DSC: owner zeroed the tile\n"
        "#endif\n"),
       CU[12], CU[13]]


def gen(n, design="v1"):
    t = SRC.read_text(encoding="utf-8")
    if design == "v3":
        for e in CU3:
            t = sub(t, *e)
        t = sub(t, "constexpr int DPMAX = 384, SLOTSD = SLOTS + (VM_DP ? DPMAX : 0);\n",
                "constexpr int DPMAX = 384, SLOTSD = SLOTS + (VM_DP ? DPMAX : 0);\nconstexpr int XROWS = @XROWS@;   // DSC: xA / xB rows allocated (_vml_alloc)\n")
        t = sub(t, '.replace("@FIXW@", str(_vml_FIXW))\n', '.replace("@FIXW@", str(_vml_FIXW)).replace("@XROWS@", str(_vml_RMAX + 8 * _vml_JMAX))\n')
        return _host(n, t, dsc_bf2=1, dsc_w72=3)
    for e in CU:
        t = sub(t, *e)
    # gather_warp and gather_w5 share the same tail: zero before the warp's release
    t = sub(t, "fence_proxy_async();\n__syncwarp();\nif (lane == 0) {\nint jj = 0;\n",
            "#if VM_DSC & 6\ndsc_zero(A, T, q0, q1, lane, 32);\n#endif\nfence_proxy_async();\n__syncwarp();\nif (lane == 0) {\nint jj = 0;\n", 2)
    # XROWS for the vm_fin clamp
    t = sub(t, "constexpr int DPMAX = 384, SLOTSD = SLOTS + (VM_DP ? DPMAX : 0);\n",
            "constexpr int DPMAX = 384, SLOTSD = SLOTS + (VM_DP ? DPMAX : 0);\nconstexpr int XROWS = @XROWS@;   // DSC: xA / xB rows allocated (_vml_alloc)\n")
    t = sub(t, '.replace("@FIXW@", str(_vml_FIXW))\n', '.replace("@FIXW@", str(_vml_FIXW)).replace("@XROWS@", str(_vml_RMAX + 8 * _vml_JMAX))\n')
    return _host(n, t, dsc_bf2=7, dsc_w72=7)


def _host(n, t, dsc_bf2, dsc_w72):
    # host: DSC variants, own y / fix scratch (the gc_16 arms need y / fix zero at entry; DSC leaves them discarded)
    t = sub(t, '_VML = {"mod": None, "mods": {}, "sc": None, "scb": None,',
            f'_VML_DEFS["bf2d"] = _VML_DEFS["bf2"] + ("-DVM_DSC={dsc_bf2}",)\n_VML_DEFS["w72d"] = _VML_DEFS["w72"] + ("-DVM_DSC={dsc_w72}",)\n'
            # own y / fix only for the variants that leave y / fix discarded (a bit-1-only variant keeps scb's zeroed y)
            f'VML_DSC_VARS = {tuple(v for v, d in (("bf2d", dsc_bf2), ("w72d", dsc_w72)) if d & 6)!r}\n'
            '_VML = {"mod": None, "mods": {}, "sc": None, "scb": None, "scd": None,')
    t = sub(t, '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]\n',
            '    scx = (_VML["scd"] if var in VML_DSC_VARS else\n'
            '           _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72", "bf2d", "w72d") else _VML["sc"])\n')
    t = sub(t, '        _VML["scb"] = dict(_VML["sc"], y=torch.zeros_like(_VML["sc"]["y"]))\n',
            '        _VML["scb"] = dict(_VML["sc"], y=torch.zeros_like(_VML["sc"]["y"]))\n'
            '        _VML["scd"] = dict(_VML["sc"], y=torch.zeros_like(_VML["sc"]["y"]), fix=torch.zeros_like(_VML["sc"]["fix"]))\n')
    ship = sub(t, "VML_VAR = 'bf2'\n", "VML_VAR = 'bf2d'\n")
    ship = sub(ship, "VML_BVAR = {256: 'w72', 128: 'w72'}\n", "VML_BVAR = {256: 'w72d', 128: 'w72d'}\n")
    ab = sub(ship, "VML_ABVARS = ()\n", "VML_ABVARS = ('bf2', 'w72')\n")
    ab = sub(ab, "VML_BVAR = {256: 'w72d', 128: 'w72d'}\n", "VML_BVAR = {256: 'w72d', 128: 'w72d', 32: 'bf2d'}\n")
    write_new(ROOT / f"kernels/lab/k3_dsc_{n}.py", ship)
    write_new(ROOT / f"kernels/lab/k3_dsc_{n}_ab.py", ab)


if __name__ == "__main__":
    gen(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "v1")

"""vfn_patches.py: VFN (VMS doneA-credit fence vs in-flight bulk copies, R-dsm-0b) compile-time variants of the layer's VMS
source (VMS_CUDA_SRC of k3_gc_21cg_s1 = k3_gc_20). R-dsm-0b: a fence.acq_rel.gpu / __threadfence waits for the SM's own
in-flight bulk copies (0.54 / 0.70 / 2.2 / 4.6 us at 16 / 32 / 72 / 108 KB per SM). The VMS helper fences before the doneA
credit of its last A unit while the producer keeps up to NS x 36 KB of B prefetches in flight.
  VMS_DB 1 (+ VMS_DBK k, 0 <= k < NS): the producer (consumer tid 0) does not issue the CTA's B units nA + k .. nA + NS - 1
     (the ones it would issue before the last-A credit); the helper warp issues them right after that credit's atomic.
     At the credit fence at most k B units are in flight. Ship RO path only (VMS_RO=1, VMS_FX bit 4 off).
  VMS_FM 1: extra globaltimer marks (mrow 10 helper pre-fence at the last A unit, 11 first B unit landed, 13 consumers
     past the last A unit's bar.sync); only when marks are passed.
  (arm a = -DVMS_NS=2, arm c = -DVMS_FX=2: existing switches.)
At VMS_DB 0 / VMS_FM 0 the patched source compiles to the base SASS (tools/vfn_local_test.py). Arithmetic, job order and
the epilogue are unchanged; only the issue time (and issuing warp) of up to NS - k bulk copies per CTA moves.

  python tools/vfn_patches.py layer --arm=db1 [--base=kernels/muse_decode_layer/k3_gc_21cg_s1.py] --out=kernels/lab/k3_vfn_1.py
"""
NL = chr(10)

ARMS = {   # compile-time defines per arm (on top of the layer's -DVMS_NS=3 -DVMS_RO=1)
    "fm": "-DVMS_FM=1",
    "ns2": "-DVMS_NS=2 -DVMS_FM=1",
    "db0": "-DVMS_DB=1 -DVMS_DBK=0 -DVMS_FM=1",
    "db1": "-DVMS_DB=1 -DVMS_DBK=1 -DVMS_FM=1",
    "db2": "-DVMS_DB=1 -DVMS_DBK=2 -DVMS_FM=1",
    "fx2": "-DVMS_FX=2 -DVMS_FM=1",
    "fx2db1": "-DVMS_FX=2 -DVMS_DB=1 -DVMS_DBK=1 -DVMS_FM=1",
}


def _rep(src, old, new, what):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"[vfn_patches] {what}: expected 1 match, got {n}")
    return src.replace(old, new)


ISSUE_H = r"""#if VMS_DB
// deferred B issue by the helper warp (after its last-A credit); slot free: units <= nA - 1 passed the consumers' bar.sync
__device__ __forceinline__ void issue_h(const unsigned char* src, unsigned char* dst, uint64_t* bar, int lane) {
#if __CUDA_ARCH__ >= 900
  if (lane == 0) issue(src, dst, bar, 0);
#else   // sm_86 test path: full[] expects NT arrivals
  for (int i = lane; i < UB / 16; i += 32)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(su32(dst + i * 16)), "l"(src + i * 16) : "memory");
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncwarp();
  for (int k = 0; k < NT / 32; ++k) mbar_arrive(bar);
#endif
}
#endif
"""


def apply(src):
    s = src
    s = _rep(s, "#ifndef VMS_G" + NL,
             "#ifndef VMS_DB" + NL + "#define VMS_DB 0" + NL + "#endif" + NL + "#ifndef VMS_DBK" + NL + "#define VMS_DBK 0" + NL +
             "#endif" + NL + "#ifndef VMS_FM" + NL + "#define VMS_FM 0" + NL + "#endif" + NL +
             "#if VMS_DB && (!VMS_RO || (VMS_FX & 4) || VMS_MK2)" + NL + "#error \"VMS_DB needs VMS_RO=1, VMS_FX bit 4 off, VMS_MK2=0\"" + NL +
             "#endif" + NL + "#ifndef VMS_G" + NL, "define")
    s = _rep(s, "constexpr int WAIT_MAX = 1 << 22;" + NL,
             "constexpr int WAIT_MAX = 1 << 22;" + NL + "#if VMS_DB" + NL +
             "static_assert(VMS_DBK >= 0 && VMS_DBK < VMS_NS, \"VMS_DBK\");" + NL + "#endif" + NL, "dbk assert")
    old = "__device__ __forceinline__ uint32_t scd("
    s = _rep(s, old, ISSUE_H + NL + old, "issue_h")
    # prologue: skip deferred B units
    old = ("    for (int q = 0; q < NS && q < N; ++q) {" + NL + "#if __CUDA_ARCH__ >= 900" + NL + "      if (tid == 0)" + NL + "#endif" + NL +
           "      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);" + NL)
    new = ("    for (int q = 0; q < NS && q < N; ++q) {" + NL + "#if VMS_DB" + NL + "      if (nA > 0 && q >= nA + VMS_DBK) continue;" + NL +
           "#endif" + NL + "#if __CUDA_ARCH__ >= 900" + NL + "      if (tid == 0)" + NL + "#endif" + NL +
           "      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);" + NL)
    s = _rep(s, old, new, "prologue")
    # first B unit landed mark
    old = "      if (mk && tid == 0 && q == 0) mrow[2] = gt();" + NL
    s = _rep(s, old, old + "#if VMS_FM" + NL + "      if (mk && tid == 0 && q == nA && nA > 0) mrow[11] = gt();" + NL + "#endif" + NL,
             "mark 11")
    # RO loop issue: skip deferred B units (q + NS in [nA + DBK, nA + NS) <=> q in [nA + DBK - NS, nA))
    old = ("      if (q + NS < N) {" + NL + "#if __CUDA_ARCH__ >= 900" + NL + "        if (tid == 0)" + NL + "#endif" + NL +
           "        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);" + NL + "      }" + NL +
           "      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));" + NL)
    new = ("#if VMS_FM" + NL + "      if (mk && tid == 0 && q == nA - 1) mrow[13] = gt();" + NL + "#endif" + NL +
           "#if VMS_DB" + NL + "      if (q + NS < N && !(q < nA && q + NS >= nA + VMS_DBK)) {" + NL + "#else" + NL +
           "      if (q + NS < N) {" + NL + "#endif" + NL + "#if __CUDA_ARCH__ >= 900" + NL + "        if (tid == 0)" + NL + "#endif" + NL +
           "        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);" + NL + "      }" + NL +
           "      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));" + NL)
    s = _rep(s, old, new, "ro issue")
    # helper: pre-fence mark + deferred issue after the last-A credit
    old = ("        if (q == nA - 1 || ((v + 1) >> 9) != j) {     // this CTA's last unit of job j: credit its units" + NL +
           "#if VMS_FX & 2" + NL)
    s = _rep(s, old, old.replace("#if VMS_FX & 2" + NL, "#if VMS_FM" + NL + "          if (mk && lane == 0 && q == nA - 1) mrow[10] = gt();" + NL +
                                  "#endif" + NL + "#if VMS_FX & 2" + NL), "mark 10")
    old = "          if (mk && lane == 0 && q == nA - 1) mrow[6] = gt();" + NL
    s = _rep(s, old, old + "#if VMS_DB" + NL + "          if (q == nA - 1)" + NL +
             "            for (int u = nA + VMS_DBK; u < nA + NS && u < N; ++u)" + NL +
             "              issue_h(unit_src(p, S, u, nA, a0, b0), ring + (u % NS) * UB, &S.full[u % NS], lane);" + NL + "#endif" + NL,
             "deferred issue")
    return s


def arm_defs(arm):
    return ARMS[arm].split()


def layer(base, arm, out):
    """layer file = base with the patched VMS source and the arm's defines in _vms_build's flags"""
    import pathlib
    L = pathlib.Path(base).read_text(encoding="utf-8")
    a = L.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    e = L.index('"""', a)
    L = L[:a] + apply(L[a:e]) + L[e:]
    old = '"-DVMS_NS=3", "-DVMS_RO=1", '
    d = {"-DVMS_NS": "3", "-DVMS_RO": "1"}
    for x in arm_defs(arm):
        k, v = x.split("=")
        if k != "-DVMS_FM":   # marks code off in the layer file
            d[k] = v
    L = _rep(L, old, "".join(f'"{k}={v}", ' for k, v in d.items()), "layer flags")
    pathlib.Path(out).write_text(L, encoding="utf-8", newline="\n")
    return len(L.encode())


if __name__ == "__main__":
    import sys
    opt = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    if len(sys.argv) > 1 and sys.argv[1] == "layer":
        n = layer(opt.get("base", "kernels/muse_decode_layer/k3_gc_21cg_s1.py"), opt["arm"], opt["out"])
        print(f"[vfn_patches] wrote {opt['out']} ({n} B), arm {opt['arm']}: {ARMS[opt['arm']]}")
    else:
        print(__doc__)

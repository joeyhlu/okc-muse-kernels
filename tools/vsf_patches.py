"""vsf_patches.py: VMS_FX (compile-time bitmask) on the layer's VMS source (VMS_CUDA_SRC of k3_gc_7): cut the fixed
latency chains of VMS (T <= 16). At VMS_FX 0 the preprocessed source is the base's (every change sits in #if VMS_FX).
  1: phase-B wait: lane 0 polls doneA with ld.acquire (the base polls relaxed, then every lane issues one more ld.acquire:
     one extra L2 round trip between doneA and the inter loads on every CTA's A -> B switch).
  2: fences: the helper's two __threadfence (fence.sc.gpu) -> fence.acq_rel.gpu; the grid-barrier arrive
     (__threadfence + atomicAdd on tid 0) -> one red.release.gpu.
  4: front: params touched before griddepcontrol.wait (constant-cache warm); the first NS weight copies issued by warp 0
     lane 0 (= tid 0) as soon as the job table exists (before the ent / pos fill and the __syncthreads); sm_90 only.
Routing, top-k, job order and all arithmetic are unchanged: outputs are bitwise identical.
"""

NL = chr(10)


def _rep(src, old, new, what):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"[vsf_patches] {what}: expected 1 match, got {n}")
    return src.replace(old, new)


def apply_fx(src):
    s = src
    s = _rep(s, "#ifndef VMS_G" + NL, "#ifndef VMS_FX" + NL + "#define VMS_FX 0" + NL + "#endif" + NL + "#ifndef VMS_G" + NL,
             "define")
    # 4a: constant-cache warm before the PDL wait
    old = ("#if __CUDA_ARCH__ >= 900" + NL + "  asm volatile(\"griddepcontrol.wait;\\n\" ::: \"memory\");" + NL + "#endif" + NL +
           "  if (warp == 0) {" + NL + "    const int m8 = p.m * 8;" + NL)
    new = ("#if (VMS_FX & 4) && __CUDA_ARCH__ >= 900" + NL +
           "  if (warp == 0) asm volatile(\"\" :: \"l\"(p.ids), \"l\"(p.ctr), \"l\"(p.wA), \"l\"(p.wB), \"r\"(p.m), \"r\"(p.lo));" + NL +
           "#endif" + NL + old)
    s = _rep(s, old, new, "param touch")
    # 4b: first copies from warp 0 lane 0 right after the job table
    old = "    if (lane == 31) S.njobs = jinc;" + NL + "    __syncwarp();" + NL
    new = (old + "#if (VMS_FX & 4) && __CUDA_ARCH__ >= 900" + NL +
           "    {" + NL +
           "      const int nj0 = __shfl_sync(0xffffffffu, jinc, 31);" + NL +
           "      if (lane == 0) {" + NL +
           "        const int NA0 = nj0 * RBA, NN0 = nj0 * (RBA + RBB);" + NL +
           "        const int a00 = c * NA0 / G, nA0 = (c + 1) * NA0 / G - a00;" + NL +
           "        const int t00 = c * NN0 / G, N0 = (c + 1) * NN0 / G - t00;" + NL +
           "        for (int q = 0; q < NS && q < N0; ++q) issue(unit_src(p, S, q, nA0, a00, t00 - a00), ring + q * UB, &S.full[q], 0);" + NL +
           "      }" + NL +
           "    }" + NL +
           "#endif" + NL)
    s = _rep(s, old, new, "early issue")
    old = ("  if (warp < NW) {" + NL + "    for (int q = 0; q < NS && q < N; ++q) {" + NL + "#if __CUDA_ARCH__ >= 900" + NL +
           "      if (tid == 0)" + NL + "#endif" + NL +
           "      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);" + NL + "    }" + NL)
    new = ("  if (warp < NW) {" + NL + "#if !((VMS_FX & 4) && __CUDA_ARCH__ >= 900)" + NL +
           old[len("  if (warp < NW) {" + NL):] + "#endif" + NL)
    s = _rep(s, old, new, "base issue loop")
    # 1: acquire poll
    old = ("          if (lane == 0) {" + NL + "            int n = 0;" + NL +
           "            while (ld_rlx(p.ctr + 8 + jj) < RBA) {" + NL +
           "              if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }" + NL +
           "              __nanosleep(64);" + NL + "            }" + NL + "          }" + NL + "          __syncwarp();" + NL +
           "          (void)ld_acq(p.ctr + 8 + jj);" + NL)
    new = ("#if VMS_FX & 1" + NL + "          if (lane == 0) {" + NL + "            int n = 0;" + NL +
           "            while (ld_acq(p.ctr + 8 + jj) < RBA) {" + NL +
           "              if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }" + NL +
           "              __nanosleep(64);" + NL + "            }" + NL + "          }" + NL + "          __syncwarp();" + NL +
           "#else" + NL + old + "#endif" + NL)
    s = _rep(s, old, new, "acquire poll")
    # 2: fences
    FA = "asm volatile(\"fence.acq_rel.gpu;\\n\" ::: \"memory\");"
    old = ("        if (q == nA - 1 || ((v + 1) >> 9) != j) {     // this CTA's last unit of job j: credit its units" + NL +
           "          __threadfence();" + NL)
    new = (old.split(NL)[0] + NL + "#if VMS_FX & 2" + NL + "          " + FA + NL + "#else" + NL + "          __threadfence();" + NL +
           "#endif" + NL)
    s = _rep(s, old, new, "credit fence")
    old = "    __threadfence();" + NL + "    if (mk && lane == 0) mrow[7] = gt();" + NL
    new = ("#if VMS_FX & 2" + NL + "    " + FA + NL + "#else" + NL + "    __threadfence();" + NL + "#endif" + NL +
           "    if (mk && lane == 0) mrow[7] = gt();" + NL)
    s = _rep(s, old, new, "helper end fence")
    old = "  if (tid == 0) {" + NL + "    __threadfence();" + NL + "    atomicAdd(p.ctr + 1, 1);" + NL
    new = ("  if (tid == 0) {" + NL + "#if VMS_FX & 2" + NL +
           "    asm volatile(\"red.release.gpu.global.add.s32 [%0], 1;\\n\" :: \"l\"(p.ctr + 1) : \"memory\");" + NL +
           "#else" + NL + "    __threadfence();" + NL + "    atomicAdd(p.ctr + 1, 1);" + NL + "#endif" + NL)
    s = _rep(s, old, new, "barrier arrive")
    return s


def strip_fx0(src):
    """VMS_FX 0 view of a patched source (drop every #if VMS_FX / (VMS_FX & k) branch taken at 0) -> must equal the base."""
    out, stack = [], []
    for ln in src.split(NL):
        t = ln.strip()
        if t in ("#ifndef VMS_FX", "#define VMS_FX 0") and not stack:
            stack.append("def") if t == "#ifndef VMS_FX" else None
            continue
        if t.startswith("#if") and "VMS_FX" in t:
            neg = t.startswith("#if !(")
            stack.append("fxneg" if neg else "fx")
            continue
        if stack and t == "#else" and stack[-1] in ("fx", "fxneg"):
            stack[-1] = stack[-1] + "_else"
            continue
        if stack and t == "#endif" and stack[-1] in ("def", "fx", "fxneg", "fx_else", "fxneg_else"):
            stack.pop()
            continue
        if stack and stack[-1] in ("def", "fx", "fxneg_else"):
            continue
        if t.startswith("#if"):
            stack.append("other")
        elif t == "#endif" and stack:
            stack.pop()
        out.append(ln)
    return NL.join(out)


def check_fx0(base, patched):
    return strip_fx0(patched) == base

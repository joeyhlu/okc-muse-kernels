"""vms2_patches.py: VMS2 compile-time levers on the layer's VMS source (VMS_CUDA_SRC of k3_gc_14), on top of VMS_FX
(tools/vsf_patches.py). Every change sits in #if blocks; at the defaults the preprocessed source is the base's.
  VMS_MK2 1: lab marks (32 int64 per CTA instead of 16): clock64 stall sums of the compute leader (weight-landing wait
     q >= 1, pempty wait, compute bar.sync, mma body), epilogue pfull wait / busy, doneA wait, smid, last-A / last unit
     landed stamps. Lab only (marks fire only with p.marks != nullptr).
  VMS_RO 1: refill reorder: bar.sync (slot consumed) -> tid 0 issues unit q+NS -> per-warp pempty wait + red write ->
     lane 0 arrives pfull (pfull count NW). The base waits pempty before the bar.sync that gates the refill.
  VMS_NRB n: reduction buffers (power of two, base 4).
Arithmetic and job order are unchanged (the ss FP32 atomic order is not deterministic in the base either).
"""
import vsf_patches as F

NL = chr(10)
_rep = F._rep


def apply(src):
    s = F.apply_fx(src)
    s = _rep(s, "#ifndef VMS_FX" + NL, "#ifndef VMS_MK2" + NL + "#define VMS_MK2 0" + NL + "#endif" + NL + "#ifndef VMS_RO" + NL +
             "#define VMS_RO 0" + NL + "#endif" + NL + "#ifndef VMS_NRB" + NL + "#define VMS_NRB 4" + NL + "#endif" + NL +
             "#ifndef VMS_FX" + NL, "defines")
    s = _rep(s, "NTL = NT + 32, NRB = 4, NS", "NTL = NT + 32, NRB = VMS_NRB, NS", "nrb")
    # MK2: 32 marks per CTA, entry clock / smid
    old = "  long long* mrow = p.marks + (size_t)c * 16;" + NL
    new = ("#if VMS_MK2" + NL + "  long long* mrow = p.marks + (size_t)c * 32;" + NL +
           "  long long cF = 0, cE = 0, cS = 0, cC = 0, cW = 0, cP = 0, cU = 0, nE = 0, ck;" + NL +
           "  if (mk && tid == 0) { mrow[16] = clock64(); unsigned sm; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(sm)); mrow[21] = sm; }" + NL +
           "#else" + NL + old + "#endif" + NL)
    s = _rep(s, old, new, "mrow")
    # RO: pfull count NW
    old = "    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], 1); mbar_init(&S.pempty[i], 1); }" + NL
    new = ("#if VMS_RO" + NL + "    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], NW); mbar_init(&S.pempty[i], 1); }" + NL +
           "#else" + NL + old + "#endif" + NL)
    s = _rep(s, old, new, "pfull init")
    # MK2: doneA wait
    old = "          if (mk && tid == 0 && mrow[3] == 0) mrow[3] = gt();" + NL
    new = old + "#if VMS_MK2" + NL + "          ck = clock64();" + NL + "#endif" + NL
    s = _rep(s, old, new, "bwait0")
    old = "          if (mk && tid == 0 && mrow[4] == 0) mrow[4] = gt();" + NL
    new = old + "#if VMS_MK2" + NL + "          cW += clock64() - ck;" + NL + "#endif" + NL
    s = _rep(s, old, new, "bwait1")
    # MK2: weight landing wait + stamps
    old = ("      mbar_wait(&S.full[sl], par);" + NL + "      if (mk && tid == 0 && q == 0) mrow[2] = gt();" + NL)
    new = ("#if VMS_MK2" + NL + "      ck = clock64();" + NL + "#endif" + NL + old +
           "#if VMS_MK2" + NL + "      { const long long c2 = clock64(); if (q > 0) cF += c2 - ck; ck = c2; }" + NL +
           "      if (mk && tid == 0 && q == nA - 1) mrow[15] = gt();" + NL +
           "      if (mk && tid == 0 && q == N - 1) mrow[22] = gt();" + NL + "#endif" + NL)
    s = _rep(s, old, new, "full wait")
    # pempty / issue block
    old = ("      const int bq = q & (NRB - 1);" + NL +
           "      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));" + NL +
           "      S.red[bq][warp][lane] = o;" + NL +
           "      cta_sync_compute();" + NL +
           "      if (q + NS < N) {" + NL + "#if __CUDA_ARCH__ >= 900" + NL + "        if (tid == 0)" + NL + "#endif" + NL +
           "        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);" + NL + "      }" + NL +
           "      if (tid == 0) mbar_arrive(&S.pfull[bq]);" + NL)
    MKC = "#if VMS_MK2" + NL + "      { const long long c2 = clock64(); cC += c2 - ck; ck = c2; }" + NL + "#endif" + NL
    MKS = "#if VMS_MK2" + NL + "      { const long long c2 = clock64(); cS += c2 - ck; ck = c2; }" + NL + "#endif" + NL
    MKE = ("#if VMS_MK2" + NL + "      { const long long c2 = clock64(); cE += c2 - ck; if (c2 - ck > 200) ++nE; ck = c2; }" + NL +
           "#endif" + NL)
    L = old.split(NL)
    base_blk = (L[0] + NL + MKC + L[1] + NL + MKE + L[2] + NL + L[3] + NL + MKS + NL.join(L[4:]))
    ro_blk = (L[0] + NL + MKC + L[3] + NL + MKS + NL.join(L[4:10]) + NL + L[1] + NL + MKE + L[2] + NL + "      __syncwarp();" + NL +
              "      if (lane == 0) mbar_arrive(&S.pfull[bq]);" + NL)
    new = "#if VMS_RO" + NL + ro_blk + "#else" + NL + base_blk + "#endif" + NL
    s = _rep(s, old, new, "pempty block")
    old = "    if (mk && tid == 0) mrow[5] = gt();" + NL
    new = (old + "#if VMS_MK2" + NL + "    if (mk && tid == 0) {" + NL +
           "      mrow[10] = cF; mrow[11] = cE; mrow[12] = cS; mrow[19] = cC; mrow[18] = cW; mrow[20] = nE;" + NL +
           "      mrow[13] = (long long)N | ((long long)nA << 16) | ((long long)njobs << 32);" + NL + "    }" + NL + "#endif" + NL)
    s = _rep(s, old, new, "compute end")
    # epilogue warp: pfull wait / busy
    old = ("      mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));" + NL + "      float4 sm = S.red[bq][0][lane];" + NL)
    new = ("#if VMS_MK2" + NL + "      ck = clock64(); if (q > 0) cU += ck - cP;" + NL + "#endif" + NL +
           old.split(NL)[0] + NL + "#if VMS_MK2" + NL + "      { const long long c2 = clock64(); cF += c2 - ck; cP = c2; }" + NL +
           "#endif" + NL + old.split(NL)[1] + NL)
    s = _rep(s, old, new, "epi wait")
    old = "    if (mk && lane == 0) mrow[7] = gt();" + NL
    new = (old + "#if VMS_MK2" + NL + "    if (mk && lane == 0) { mrow[14] = cF; mrow[23] = cU + (clock64() - cP); }" + NL + "#endif" + NL)
    s = _rep(s, old, new, "epi end")
    old = "  if (mk && tid == 0) mrow[9] = gt();" + NL
    new = old + "#if VMS_MK2" + NL + "  if (mk && tid == 0) mrow[17] = clock64();" + NL + "#endif" + NL
    s = _rep(s, old, new, "end")
    return s


def strip0(src):
    """Default view: drop the #if VMS_MK2 / VMS_RO blocks taken at 0 and the new defines, restore NRB 4, then the FX 0 view."""
    out, stack = [], []
    for ln in src.split(NL):
        t = ln.strip()
        if stack and stack[-1] in ("def", "x", "skip"):
            if t.startswith("#if"):
                stack.append("skip")
                continue
            if t == "#endif" and stack[-1] == "skip":
                stack.pop()
                continue
            if t == "#else" and stack[-1] == "skip":
                continue
        if t in ("#ifndef VMS_MK2", "#ifndef VMS_RO", "#ifndef VMS_NRB") and not stack:
            stack.append("def")
            continue
        if t in ("#if VMS_MK2", "#if VMS_RO"):
            stack.append("x")
            continue
        if stack and t == "#else" and stack[-1] == "x":
            stack[-1] = "x_else"
            continue
        if stack and t == "#endif" and stack[-1] in ("def", "x", "x_else"):
            stack.pop()
            continue
        if stack and stack[-1] in ("def", "x", "skip"):
            continue
        if t.startswith("#if"):
            stack.append("other")
        elif t == "#endif" and stack and stack[-1] == "other":
            stack.pop()
        out.append(ln)
    return F.strip_fx0(NL.join(out).replace("NRB = VMS_NRB, NS", "NRB = 4, NS"))


def check0(base, patched):
    return strip0(patched) == base

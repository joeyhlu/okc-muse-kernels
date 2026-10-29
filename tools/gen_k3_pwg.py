"""gen_k3_pwg.py: PWG = pre-wait weight staging for GD (R-pwg-1).

GD's ST knob only issued ST register stages before griddepcontrol.wait (ST 1 at T 32 -> 1/16 of the unpooler stream,
1/18 of cat). PWG encodes ST = STR | PS << 4: PS extra stages are cp.async'd into the warp's own dynamic smem before
the wait (stages 0..PS-1), STR register stages follow (PS..PS+STR-1) as today's ring. Same decode / mma order -> bitwise.

  python tools/gen_k3_pwg.py kernels/muse_decode_layer/k3_gc_16.py kernels/lab/k3_pwg_ab.py --ab
  python tools/gen_k3_pwg.py kernels/muse_decode_layer/k3_gc_16.py kernels/lab/k3_pwg_1.py --unp32=0x41 --cat32=0x21
--ab: every candidate code below is built and checked bitwise against the route's ship inst in prepare; the A/B arm
selector GD_PWG {("unpooler", b) | ("cat", b): code} is read at call (capture) time.
"""
import ast
import pathlib
import sys

AB_UNP = {32: (0x22, 0x42), 16: (0x22, 0x42)}
AB_CAT = {32: (0x22, 0x41), 16: (0x22, 0x42)}


def rep(s, old, new, n=1):
    c = s.count(old)
    assert c == n, (c, n, old[:80])
    return s.replace(old, new)


def patch(s, ab, codes, nonce):
    lines = s.split("\n")
    assert lines[1].startswith("# nonce "), lines[1][:40]
    lines[1] = "# nonce " + nonce
    s = "\n".join(lines)
    # ---- kernel
    s = rep(s, """  constexpr int NW = WR * WK, NT = NW * 32, NP = 3;
  constexpr int KUP = KU - CAT;""", """  constexpr int NW = WR * WK, NT = NW * 32, NP = 3;
  constexpr int STR = ST & 15, PS = ST >> 4;   // PWG: STR register stages + PS smem stages, all issued before the wait
  static_assert(STR >= 1 && PS + STR <= KU && (PS == 0 || VF == 0), "PWG: ST = STR | PS << 4");
  constexpr int KUP = KU - CAT;""")
    s = rep(s, """  uint4 R[ST][NP];
#pragma unroll
  for (int i = 0; i < ST; ++i)
    if (i < KU) {
#pragma unroll
      for (int p = 0; p < NP; ++p) R[i][p] = ldw(wp + (size_t)i * wstep + p * 32);
    }
""", """  const uint32_t pwg_sb = smem_u32(gd_vf_dsm) + (uint32_t)(warp * (PS * NP * 32) + lane) * 16u;   // PWG: warp-private
  if constexpr (PS > 0) {   // PWG: weight stages 0..PS-1 -> own smem before griddepcontrol.wait (weights: no producer)
#pragma unroll
    for (int i = 0; i < PS; ++i) {
#pragma unroll
      for (int p = 0; p < NP; ++p)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(pwg_sb + (uint32_t)((i * NP + p) * 32) * 16u),
                     "l"(wp + (size_t)i * wstep + p * 32) : "memory");
      asm volatile("cp.async.commit_group;\\n" ::: "memory");
    }
  }
  uint4 R[STR][NP];
#pragma unroll
  for (int i = 0; i < STR; ++i)
    if (PS + i < KU) {
#pragma unroll
      for (int p = 0; p < NP; ++p) R[i][p] = ldw(wp + (size_t)(PS + i) * wstep + p * 32);
    }
""")
    s = rep(s, """    const int slot = s % ST;
    const int xs = XP ? (s & 1) : 0;""", """    const int slot = s < PS ? 0 : (s - PS) % STR;
    const int xs = XP ? (s & 1) : 0;""")
    s = rep(s, """    uint32_t a[16];
    decode(R[slot], (CAT && s >= KUP) ? A.tbr : A.tb, a);
    if (s + ST < KU) {
#pragma unroll
      for (int p = 0; p < NP; ++p) R[slot][p] = ldw(wp + (size_t)(s + ST) * wstep + p * 32);
    }
""", """    uint32_t a[16];
    if (s < PS) {   // PWG: smem stage s (own lane's 16 B per plane: no barrier, wait_group makes it visible)
      const int pend = PS - 1 - s;
      if (pend <= 0) asm volatile("cp.async.wait_group 0;\\n" ::: "memory");
      else if (pend == 1) asm volatile("cp.async.wait_group 1;\\n" ::: "memory");
      else if (pend == 2) asm volatile("cp.async.wait_group 2;\\n" ::: "memory");
      else if (pend == 3) asm volatile("cp.async.wait_group 3;\\n" ::: "memory");
      else if (pend == 4) asm volatile("cp.async.wait_group 4;\\n" ::: "memory");
      else if (pend == 5) asm volatile("cp.async.wait_group 5;\\n" ::: "memory");
      else asm volatile("cp.async.wait_group 6;\\n" ::: "memory");
      uint4 Rc[NP];
#pragma unroll
      for (int p = 0; p < NP; ++p)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\\n" : "=r"(Rc[p].x), "=r"(Rc[p].y), "=r"(Rc[p].z), "=r"(Rc[p].w)
                     : "r"(pwg_sb + (uint32_t)((s * NP + p) * 32) * 16u) : "memory");
      decode(Rc, (CAT && s >= KUP) ? A.tbr : A.tb, a);
    } else {
      decode(R[slot], (CAT && s >= KUP) ? A.tbr : A.tb, a);
      if (s + STR < KU) {
#pragma unroll
        for (int p = 0; p < NP; ++p) R[slot][p] = ldw(wp + (size_t)(s + STR) * wstep + p * 32);
      }
    }
""")
    s = rep(s, """  cfg.blockDim = dim3((unsigned)(WR * WK * 32), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = st;""", """  cfg.blockDim = dim3((unsigned)(WR * WK * 32), 1, 1);
  constexpr int PSD = (ST >> 4) * WR * WK * 3 * 32 * 16;   // PWG smem stages
  if constexpr (PSD > 0) {
    static bool pwg_attr = false;
    if (!pwg_attr) {
      const int e = (int)cudaFuncSetAttribute(gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, PSD);
      if (e) return e;
      pwg_attr = true;
    }
  }
  cfg.dynamicSmemBytes = PSD;
  cfg.stream = st;""")
    s = rep(s, """  auto fn = gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>;
  int e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, WR * WK * 32, 0);""", """  auto fn = gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>;
  constexpr int PSD = (ST >> 4) * WR * WK * 3 * 32 * 16;   // PWG smem stages
  int e = 0;
  if (PSD > 0) e = (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, PSD);
  if (e) return e;
  e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, WR * WK * 32, PSD);""")
    # ---- python: key checks accept the PWG code
    s = rep(s, "if n % 16 or k % 64 or nrt % wr or nu != wk * ku * s or st_ > ku or s not in",
            "if n % 16 or k % 64 or nrt % wr or nu != wk * ku * s or (st_ & 15) < 1 or (st_ & 15) + (st_ >> 4) > ku or s not in")
    s = rep(s, "if wr != 1 or s != 1 or mt not in (2, 4) or wk * (ku - GD_CAT_UNITS) != HIDDEN // 64 or st_ > ku or xp not in (0, 1):",
            "if wr != 1 or s != 1 or mt not in (2, 4) or wk * (ku - GD_CAT_UNITS) != HIDDEN // 64 or (st_ & 15) < 1 or (st_ & 15) + (st_ >> 4) > ku or xp not in (0, 1):")
    # ---- routes
    tree = ast.parse(s)
    def setroute(s, name, key, code):
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
                d = ast.literal_eval(node.value)
                rt = d[key]
                new = rt[:4] + (code,) + rt[5:]
                ks = f'("{key[0]}", {key[1]})' if isinstance(key, tuple) else repr(key)
                tup = lambda t: "(" + ", ".join(hex(v) if i == 4 and v > 15 else str(v) for i, v in enumerate(t)) + ")"
                old_t, new_t = ks + ": " + tup(rt), ks + ": " + tup(new)
                return rep(s, old_t, new_t)
        raise KeyError(name)
    for (kind, b), code in codes.items():
        s = setroute(s, "GD_ROUTE" if kind == "unpooler" else "GD_CAT_ROUTE", (kind, b) if kind == "unpooler" else b, code)
    if ab:
        s = patch_ab(s)
    return s


def patch_ab(s):
    s = rep(s, "_GD = {\"mod\": None, \"keys\": None, \"build_s\": 0.0}\n",
            "_GD = {\"mod\": None, \"keys\": None, \"build_s\": 0.0}\n"
            f"GD_PWG_AB = {dict((('unpooler', b), c) for b, c in AB_UNP.items())!r}   # PWG A/B: alt ST codes built + bitwise-checked\n"
            f"GD_PWG_CAT_AB = {AB_CAT!r}\n"
            "GD_PWG = {}   # PWG A/B arm selector, read at call (capture) time: {('unpooler', b) | ('cat', b): code}\n")
    s = rep(s, """    for key in _gd_cat_keys():
        if key not in keys:
            keys.append(key)
    return keys""", """    for key in _gd_cat_keys():
        if key not in keys:
            keys.append(key)
    for (proj, b), codes in sorted(GD_PWG_AB.items(), key=str):   # PWG A/B alternates
        rt = GD_ROUTE.get((proj, b))
        for c in (codes if rt else ()):
            key = _gd_key(*_gd_nk(proj, b), rt[:4] + (c,) + rt[5:])
            if key is not None and key not in keys:
                keys.append(key)
    for b, codes in sorted(GD_PWG_CAT_AB.items()):
        rt = GD_CAT_ROUTE.get(b)
        for c in (codes if rt else ()):
            key = _gd_cat_key(rt[:4] + (c,) + rt[5:])
            if key is not None and key not in keys:
                keys.append(key)
    return keys""")
    # GD route alternates: after the ship self-test passes
    s = rep(s, """            else:
                routes[(proj, b)] = r
                live.append(f"{proj}/{b} {desc}")""", """            else:
                routes[(proj, b)] = r
                r["alt"] = _pwg_alts(mod, ids, r, rt, (proj, b), lambda rr, x, o, T: rr["mod"].gd_run(
                    rr["inst"], rr["w"], rr["esc"], x, o, T, k, (HIDDEN if rr["slice"] else n), rr["tb"], rr["nbx"], 1, 0),
                    k, (HIDDEN if r["slice"] else n), lambda c: _gd_key(n, k, rt[:4] + (c,) + rt[5:]), sms, (n // 16 // rt[1]) * rt[5])
                live.append(f"{proj}/{b} {desc}")""")
    s = rep(s, """            else:
                routes[b] = r
                live.append(f"cat/{b} {desc}")""", """            else:
                routes[b] = r
                r["alt"] = _pwg_cat_alts(mod, ids, r, rt, b, sms)
                live.append(f"cat/{b} {desc}")""")
    s = rep(s, """        pdl = (1 if GD_PDL.get(proj) else 0) | (2 if GD_TRIG.get(proj) else 0)
        r["mod"].gd_run(r["inst"], """, """        pdl = (1 if GD_PDL.get(proj) else 0) | (2 if GD_TRIG.get(proj) else 0)
        r["mod"].gd_run(r.get("alt", {}).get(GD_PWG.get((proj, _gd_bucket(T))), r["inst"]), """)
    s = rep(s, """def _gd_cat_run(r, x, y, part, T, pdl):
    r["mod"].gdc_run(r["inst"], """, """def _gd_cat_run(r, x, y, part, T, pdl, inst=None):
    r["mod"].gdc_run(r["inst"] if inst is None else inst, """)
    s = rep(s, "        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)\n",
            "        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL, r.get(\"alt\", {}).get(GD_PWG.get((\"cat\", _gd_bucket(tokens)))))\n")
    s = rep(s, "\ndef _gd_cat_selftest(r, wp, hi, lo, w32, b):", '''
def _pwg_ok(mod, kid, sms, ctas, tag):
    info = mod.gd_info(kid)
    ok = not (info[1] > 64 or (GD_ONE_WAVE and info[2] * sms < ctas))
    if not ok:
        print(f"[pwg] {tag}: regs {info[0]} spill {info[1]} occ {info[2]} for {ctas} CTAs -> not live", flush=True)
    return ok, info


def _pwg_alts(mod, ids, r, rt, rk, run, k, ldo, keyf, sms, ctas):
    alts = {}
    dev = r["w"].device
    for c in GD_PWG_AB.get(rk, ()):
        key = keyf(c)
        tag = f"{rk[0]}/{rk[1]} code {c:#x}"
        if key is None or key not in ids:
            print(f"[pwg] {tag}: geometry invalid / not built", flush=True)
            continue
        ok, info = _pwg_ok(mod, ids[key], sms, ctas, tag)
        if not ok:
            continue
        ra = dict(r, inst=ids[key])
        same = True
        for T in GD_SELF_TEST_T[rk[1]]:
            for outl in (False, True):
                x, _ = _gd_test_input(T, k, 977 + 31 * T + int(outl), dev, outl)
                o0 = torch.full((T, ldo), float("nan"), dtype=torch.bfloat16, device=dev)
                o1 = torch.full((T, ldo), float("nan"), dtype=torch.bfloat16, device=dev)
                run(r, x, o0, T)
                for rep_ in range(2):
                    run(ra, x, o1, T)
                    torch.cuda.synchronize()
                    same &= bool(torch.equal(o0.view(torch.int16), o1.view(torch.int16)))
        print(f"[pwg] {tag}: regs {info[0]} spill {info[1]} occ {info[2]} smem {info[3]}: bitwise vs ship {same}", flush=True)
        if same:
            alts[c] = ids[key]
    return alts


def _pwg_cat_alts(mod, ids, r, rt, b, sms):
    alts = {}
    dev = r["w"].device
    for c in GD_PWG_CAT_AB.get(b, ()):
        key = _gd_cat_key(rt[:4] + (c,) + rt[5:])
        tag = f"cat/{b} code {c:#x}"
        if key is None or key not in ids:
            print(f"[pwg] {tag}: geometry invalid / not built", flush=True)
            continue
        ok, info = _pwg_ok(mod, ids[key], sms, D_EXPERT // 16, tag)
        if not ok:
            continue
        same = True
        for T in GD_CAT_SELF_TEST_T[b]:
            for outl in (False, True):
                x, _ = _gd_test_input(T, HIDDEN, 977 + 31 * T + int(outl), dev, outl)
                y0 = torch.full((T, D_EXPERT), float("nan"), dtype=torch.bfloat16, device=dev)
                p0 = torch.full((8, T, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
                _gd_cat_run(r, x, y0, p0, T, 0)
                for rep_ in range(2):
                    y1, p1 = torch.full_like(y0, float("nan")), torch.full_like(p0, float("nan"))
                    _gd_cat_run(r, x, y1, p1, T, 0, ids[key])
                    torch.cuda.synchronize()
                    same &= bool(torch.equal(y0.view(torch.int16), y1.view(torch.int16))) and bool(torch.equal(p0.view(torch.int32), p1.view(torch.int32)))
        print(f"[pwg] {tag}: regs {info[0]} spill {info[1]} occ {info[2]} smem {info[3]}: bitwise vs ship {same}", flush=True)
        if same:
            alts[c] = ids[key]
    return alts


def _gd_cat_selftest(r, wp, hi, lo, w32, b):''')
    return s


def main():
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    ab = "--ab" in sys.argv
    codes = {}
    for a in sys.argv[3:]:
        for tag, kk in (("--unp32=", ("unpooler", 32)), ("--unp16=", ("unpooler", 16)), ("--cat32=", ("cat", 32)), ("--cat16=", ("cat", 16))):
            if a.startswith(tag):
                codes[kk] = int(a[len(tag):], 0)
    desc = "A/B alternates " + repr({"unp": AB_UNP, "cat": AB_CAT}) if ab else repr({f"{k[0]}{k[1]}": hex(v) for k, v in codes.items()})
    nonce = f"{dst.stem.replace('_', '-')} ({src.stem} + PWG GD pre-wait smem weight staging {desc}; tools/gen_k3_pwg.py)"
    out = patch(src.read_text(encoding="utf-8"), ab, codes, nonce)
    ast.parse(out)
    dst.write_text(out, encoding="utf-8", newline="\n")
    print(f"wrote {dst} ({len(out.encode()) / 1e6:.3f} MB)")


if __name__ == "__main__":
    main()

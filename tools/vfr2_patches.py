"""vfr2_patches.py: S2 refinement on a layer file that carries tools/vfr_patches.py's fused T = 8 pooler + router rows (VF = 2;
default base kernels/muse_decode_layer/k3_vf_10.py). R-vf-7 marks: a fused CTA lives 16.5 us vs 11.9 for the stock pooler; the
router tail (warps 0-7, one row each: 72 LDS.128 per lane, ~288 KB of shared-memory reads per CTA, two CTAs per SM finishing
together) sits on every CTA's end. Variants, all with router32's per-row arithmetic (same per-lane FMA order, same xor tree:
bitwise equal partials), selected at call time by VF_GDR_V:
  2  shipped (k3_vf_10): router after the pooler rows, warps 0-7 x 1 row
  3  router after the pooler rows, warps 0-3 x 2 rows: the x slice is read once per 2 rows (-44 % shared-memory reads)
  5  router after the pooler rows, warps 0-1 x 4 rows (-67 %)
  4  router inside the K loop at step KU / 2, right after that step's weight loads are issued, all 8 warps x 1 row (CTA barrier
     there): its shared-memory / FMA work overlaps the in-flight pooler stages instead of extending the CTA
prepare() self-tests every variant (pooler bitwise vs the stock instance, partials bitwise vs router32, m = 8 / 5 / 2); a
failing variant is dropped (loud [vfr] line), and VF_GDR_V falls back to 2.

    python -I tools/vfr2_patches.py --n=<new> [--base=k3_vf_10] [--v=2] [--lab-only] [--force]   (--v: VF_GDR_V default)
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)
GEO = "0, 1, 1, 8, 16, 2, 1, 2, 1"
VARIANTS = (2, 3, 4, 5)


def sub(t, old, new, count=1, what=""):
    n = t.count(old)
    if n != count:
        raise SystemExit(f"[vfr2] anchor {what or old[:60]!r}: found {n}, expected {count}")
    return t.replace(old, new)


def router_rows(rows_per_warp, nwarps, indent):
    """router32_kernel<8>'s arithmetic for rows vf_bx * 8 + warp * R + rr (rr < R) of K slice vf_sl, on warps < nwarps."""
    R = rows_per_warp
    p = " " * indent
    return (f"{p}if (warp < {nwarps}) {{\n"
            f"{p}  const float4* xsr = gd_vf_dsm + 8 * 256;\n"
            f"{p}  float racc[{R}][8];\n"
            f"#pragma unroll\n{p}  for (int rr = 0; rr < {R}; ++rr)\n"
            f"#pragma unroll\n{p}    for (int j = 0; j < 8; ++j) racc[rr][j] = 0.f;\n"
            f"#pragma unroll\n{p}  for (int i = 0; i < 8; ++i) {{\n"
            f"{p}    const int c = i * 32 + lane;\n"
            f"{p}    float4 xv[8];\n"
            f"#pragma unroll\n{p}    for (int j = 0; j < 8; ++j) xv[j] = xsr[j * 256 + c];\n"
            f"#pragma unroll\n{p}    for (int rr = 0; rr < {R}; ++rr) {{\n"
            f"{p}      const float4 wv = gd_vf_dsm[(warp * {R} + rr) * 256 + c];\n"
            f"#pragma unroll\n{p}      for (int j = 0; j < 8; ++j) {{\n"
            f"{p}        racc[rr][j] = fmaf(wv.x, xv[j].x, racc[rr][j]);\n"
            f"{p}        racc[rr][j] = fmaf(wv.y, xv[j].y, racc[rr][j]);\n"
            f"{p}        racc[rr][j] = fmaf(wv.z, xv[j].z, racc[rr][j]);\n"
            f"{p}        racc[rr][j] = fmaf(wv.w, xv[j].w, racc[rr][j]);\n"
            f"{p}      }}\n{p}    }}\n{p}  }}\n"
            f"#pragma unroll\n{p}  for (int rr = 0; rr < {R}; ++rr)\n"
            f"#pragma unroll\n{p}    for (int j = 0; j < 8; ++j) {{\n"
            f"{p}      float a = racc[rr][j];\n"
            f"#pragma unroll\n{p}      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);\n"
            f"{p}      if (lane == 0 && j < A.M) A.part[((long)vf_sl * A.M + j) * 256 + vf_bx * 8 + warp * {R} + rr] = a;\n"
            f"{p}    }}\n{p}}}\n")


def apply(t, v):
    # copies for every fused variant
    t = sub(t, "  if constexpr (VF == 2) {   // router32 CTA (vf_bx, vf_sl): 8 rows x 1024 K of FP32 weights (no producer dependency)\n",
            "  if constexpr (VF >= 2) {   // router32 CTA (vf_bx, vf_sl): 8 rows x 1024 K of FP32 weights (no producer dependency)\n",
            what="pre-wait copies")
    t = sub(t, "  if constexpr (VF == 2) {   // the router input slice (the boundary's FP32 output), tokens >= M zero\n",
            "  if constexpr (VF >= 2) {   // the router input slice (the boundary's FP32 output), tokens >= M zero\n",
            what="post-wait copies")
    # tails for 3 / 5 (2 unchanged; 4 has none)
    tail2 = "    if constexpr (VF == 2) {   // router32_kernel<8>'s arithmetic for row vf_bx * 8 + warp, K slice vf_sl\n"
    tails = ("    if constexpr (VF == 3 || VF == 5) {   // vfr2: R rows per warp, the x slice read once per R rows\n"
             "      asm volatile(\"cp.async.wait_all;\\n\" ::: \"memory\");\n      __syncthreads();\n"
             "      if constexpr (VF == 3) {\n" + router_rows(2, 4, 8) + "      } else {\n" + router_rows(4, 2, 8) + "      }\n    }\n")
    t = sub(t, tail2, tails + tail2, what="tails")
    # in-loop router (VF 4) right after step KU / 2 issued its weight loads
    loop = ("    if (s + ST < KU) {\n#pragma unroll\n      for (int p = 0; p < NP; ++p) R[slot][p] = ldw(wp + (size_t)(s + ST) * wstep + p * 32);\n"
            "    }\n")
    hook = ("    if constexpr (VF == 4) {   // vfr2: the router row of this warp, overlapping the in-flight pooler stages\n"
            "      if (s == KU / 2) {\n        asm volatile(\"cp.async.wait_all;\\n\" ::: \"memory\");\n        __syncthreads();\n"
            + router_rows(1, 8, 8) + "      }\n    }\n")
    t = sub(t, loop, loop + hook, what="K loop")
    # host: dispatch / info for every variant
    old_launch = """  if (vf == 2) {
    static bool set = false;
    if (!set) {
      const int e = (int)cudaFuncSetAttribute(gd_kernel<@GEO@, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, GDR_DS);
      if (e) return e;
      set = true;
    }
    cfg.dynamicSmemBytes = GDR_DS;
    return (int)cudaLaunchKernelEx(&cfg, gd_kernel<@GEO@, 2>, A);
  }
""".replace("@GEO@", GEO)
    new_launch = """  if (vf >= 2 && vf <= 5) {
    static bool set[6] = {false, false, false, false, false, false};
    auto fn = vf == 2 ? gd_kernel<@GEO@, 2> : vf == 3 ? gd_kernel<@GEO@, 3> : vf == 4 ? gd_kernel<@GEO@, 4> : gd_kernel<@GEO@, 5>;
    if (!set[vf]) {
      const int e = (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, GDR_DS);
      if (e) return e;
      set[vf] = true;
    }
    cfg.dynamicSmemBytes = GDR_DS;
    return (int)cudaLaunchKernelEx(&cfg, fn, A);
  }
""".replace("@GEO@", GEO)
    t = sub(t, old_launch, new_launch, what="host launch")
    t = sub(t, "  auto fn = vf == 2 ? gd_kernel<@GEO@, 2> : gd_kernel<@GEO@, 1>;\n  const int ds = vf == 2 ? GDR_DS : 0;\n".replace("@GEO@", GEO),
            ("  auto fn = vf == 2 ? gd_kernel<@GEO@, 2> : vf == 3 ? gd_kernel<@GEO@, 3> : vf == 4 ? gd_kernel<@GEO@, 4>\n"
             "            : vf == 5 ? gd_kernel<@GEO@, 5> : gd_kernel<@GEO@, 1>;\n  const int ds = vf >= 2 ? GDR_DS : 0;\n").replace("@GEO@", GEO),
            what="host info")
    t = sub(t, "  if (vf == 2)\n    TORCH_CHECK(x32.scalar_type()", "  if (vf >= 2)\n    TORCH_CHECK(x32.scalar_type()", what="cpp check")
    t = sub(t, "vf == 2 ? x32.data_ptr<float>() : nullptr,", "vf >= 2 ? x32.data_ptr<float>() : nullptr,", what="cpp x32")
    t = sub(t, "vf == 2 ? w32.data_ptr<float>() : nullptr, vf == 2 ? part.data_ptr<float>() : nullptr, mk,",
            "vf >= 2 ? w32.data_ptr<float>() : nullptr, vf >= 2 ? part.data_ptr<float>() : nullptr, mk,", what="cpp ptrs")
    # Python
    t = sub(t, "VF_GDM = False\n", f"VF_GDM = False\nVF_GDR_V = {v}\n", what="VF_GDR_V")
    t = sub(t, '_VFR = {"ok": False, "marks": None, "seen": set()}', '_VFR = {"ok": False, "marks": None, "seen": set(), "vok": {}}',
            what="_VFR")
    t = sub(t, "                     z if vf == 2 else _VFR[\"nomk\"], state[\"router\"], part, mk)\n",
            "                     z if vf >= 2 else _VFR[\"nomk\"], state[\"router\"], part, mk)\n", what="py z")
    t = sub(t, "    return _vfr_gdr(state, ffn_input, router_input, 2)\n",
            "    v = VF_GDR_V if _VFR[\"vok\"].get(VF_GDR_V) else 2\n    return _vfr_gdr(state, ffn_input, router_input, v)\n",
            what="py fused")
    old_test = """            y1, p1 = _vfr_gdr(state, h, z, 2)
            y2, _ = _vfr_gdr(state, h, None, 1)
            torch.cuda.synchronize()
            e = (bool(torch.equal(y0, y1)), bool(torch.equal(p0, p1)), bool(torch.equal(y0, y2)))
            print(f"[vfr] self-test m={m}: fused pooler bitwise {e[0]}, router partials bitwise vs router32 {e[1]}, "
                  f"marked pooler bitwise {e[2]}", flush=True)
            ok &= all(e)
"""
    new_test = """            y1, p1 = _vfr_gdr(state, h, z, 2)
            y2, _ = _vfr_gdr(state, h, None, 1)
            torch.cuda.synchronize()
            e = (bool(torch.equal(y0, y1)), bool(torch.equal(p0, p1)), bool(torch.equal(y0, y2)))
            print(f"[vfr] self-test m={m}: fused pooler bitwise {e[0]}, router partials bitwise vs router32 {e[1]}, "
                  f"marked pooler bitwise {e[2]}", flush=True)
            ok &= all(e)
            for vv in (3, 4, 5):
                try:
                    info = r["mod"].gdr_info(vv)
                    yv, pv = _vfr_gdr(state, h, z, vv)
                    torch.cuda.synchronize()
                    good = (bool(torch.equal(y0, yv)) and bool(torch.equal(p0, pv)) and info[2] >= 2 and info[1] <= 128)
                except Exception as ex:
                    info, good = [ex], False
                _VFR["vok"][vv] = _VFR["vok"].get(vv, True) and good
                print(f"[vfr] self-test m={m} variant {vv}: {list(info)} bitwise / 2 CTAs per SM {good}", flush=True)
"""
    t = sub(t, old_test, new_test, what="py test")
    t = sub(t, "        _VFR[\"seen\"].clear()\n        _VFR[\"ok\"] = ok\n",
            "        _VFR[\"seen\"].clear()\n        _VFR[\"ok\"] = ok\n        _VFR[\"vok\"][2] = ok\n"
            "        print(f\"[vfr] variants live {sorted(k for k, x in _VFR['vok'].items() if x)} (VF_GDR_V {VF_GDR_V})\", flush=True)\n",
            what="py vok")
    return t


def main():
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    if "n" not in ARGS:
        raise SystemExit("[vfr2_patches] --n=<new number> is required")
    n = ARGS["n"]
    v = int(ARGS.get("v", "2"))
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_vf_10')}.py"
    labonly = "--lab-only" in sys.argv
    out = ROOT / f"kernels/muse_decode_layer/k3_vf_{n}.py"
    lab = ROOT / f"kernels/lab/k3_vf_{n}_ab.py"
    for f in ([lab] if labonly else [out, lab]):
        if f.exists() and "--force" not in sys.argv:
            raise SystemExit(f"[vfr2_patches] {f.relative_to(ROOT).as_posix()} exists: pick a new --n")
    tag = f"k3-vf-{n}"
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    GM.check(fb == b, f"{base.name} is not a fixed point of finish()")
    nonce = (f"# nonce {tag} ({base.stem} + VF S2 refinement: fused T = 8 router variants (2 / 3 / 4 / 5), default {v}; "
             f"self-tested with fallback)")
    t = apply(b.replace(bn, nonce), v)
    s = GM.finish(t, nonce, tag, chk_base, rep)
    for r in rep:
        print(f"[vfr2_patches] OK  {r}")
    if not labonly:
        out.write_text(s, encoding="utf-8", newline=NL)
    lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vfr2_patches] wrote {'' if labonly else out.relative_to(ROOT).as_posix() + ' and '}{lab.relative_to(ROOT).as_posix()} "
          f"({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

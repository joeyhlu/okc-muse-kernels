"""DK2: K1 v7 L=1 (decode) with QK on wgmma m64n8k32 e4m3, keys as M (notes/study/attn_full_fresh.md s6).

On top of K1-W (tools/at_k1w_patches.py; applied here first when absent):
  - loader: K rows land in the wgmma K-major INTERLEAVE layout (head h at +4096 h, 8-row groups 512 B, 16-B chunks
    128 B; K1 L=8's proven layout), V stays in K1's row layout;
  - Q: the normalized q also goes to smem as e4m3 [head][8 rows][64] (interleave) in K1's unused L=1 aux region
    (queries >= E4M3); the fp16 Q stays for the new-token tile;
  - roles: warpgroup = kv head (h = warp >> 2), warp = key quarter (q4 = warp & 3); per stage one CTA barrier (each
    warpgroup's wgmma reads all 64 keys) after cp.async.wait + fence.proxy.async;
  - per tile and warpgroup: S^T[64 keys x 8 q rows] = 2 x wgmma m64n8k32 SS (no e4m3 -> f16 converts of K, no padded
    rows); per-warp online softmax over the warp's 16 keys with K1's numerics (exact running max, P = 2^(s - m + 8)
    in fp16, fp32 row sums of the unrounded p); movmatrix.trans turns P^T into K1's PV A fragment; PV, the quarter
    merge, the new-token tile and the fold epilogue are K1's code (merge warp index 4 hh + q).
Usage: text = apply_dk2(text)  (K1-W applied first if missing). Fallback / self-test wiring: see DK2_* in the lab file
kernels/muse_attention_decode/at_3.py and the layer patch notes in notes/study/attn_full_fresh.md.
"""
import importlib.util
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("at_k1w_patches", _HERE / "at_k1w_patches.py")
_k1w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_k1w)

DK2_DEVICE = r'''
// ---- DK2 (tools/at_dk2_patches.py): QK on wgmma m64n8k32 e4m3 with keys as M ----
__device__ __forceinline__ void dk2_wg8(float (&d)[4], uint64_t da, uint64_t db, int scale_d) {
#if K1_ARCH_OK
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(da), "l"(db), "r"(scale_d));
#endif
}
__device__ __forceinline__ uint32_t dk2_movt(uint32_t x) {
  uint32_t y;
#if K1_ARCH_OK
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
#else
  y = x;
#endif
  return y;
}
// Kbuf: K tile (interleave, head h at +4096 h); qe: e4m3 Q [2][8][64] interleave (512 B per head).
// Thread (warp q4 of warpgroup h, lane g/t) holds S^T[key kw + g (+8)][q row 2t (+1)].
// m2 / ls: running max (log2 units) / fp32 row sum for q rows 2t, 2t+1 over this warp's keys (equal across g).
__device__ __forceinline__ void dk2_tile(const uint8_t* __restrict__ Kbuf, const uint8_t* __restrict__ Vbuf,
                                         const uint8_t* qe, const int kw, const int h, const int g, const int t,
                                         float (&o)[8][4], float (&m2)[2], float (&ls)[2], const float cl2,
                                         const bool need, const int tile_abs, const int key_lim, const int seq,
                                         const int W) {
  float S[4] = {0.f, 0.f, 0.f, 0.f};
  wg_fence();
  dk2_wg8(S, desc_il(Kbuf + h * 4096, 128, 512), desc_il(qe + h * 512, 128, 512), 0);
  dk2_wg8(S, desc_il(Kbuf + h * 4096 + 256, 128, 512), desc_il(qe + h * 512 + 256, 128, 512), 1);
  wg_commit();
  wg_wait0();
  if (need) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int ak = tile_abs + kw + g + 8 * e;
      const bool ok = (ak < key_lim) && (ak <= seq) && (W == 0 || ak > seq - W);
      if (!ok) { S[2 * e] = neg_inf(); S[2 * e + 1] = neg_inf(); }
    }
  }
  float x[2] = {fmaxf(S[0], S[2]), fmaxf(S[1], S[3])};
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    x[r] = fmaxf(x[r], __shfl_xor_sync(FULL, x[r], 4));
    x[r] = fmaxf(x[r], __shfl_xor_sync(FULL, x[r], 8));
    x[r] = fmaxf(x[r], __shfl_xor_sync(FULL, x[r], 16));
  }
  float a[2], b[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const float n = fmaxf(m2[r], x[r] * cl2);
    const float u = (n == neg_inf()) ? 0.f : n;
    a[r] = ex2(m2[r] - u);
    b[r] = u - P_OFFSET;
    m2[r] = n;
  }
  const float p00 = ex2(fmaf(S[0], cl2, -b[0])), p01 = ex2(fmaf(S[1], cl2, -b[1]));
  const float p10 = ex2(fmaf(S[2], cl2, -b[0])), p11 = ex2(fmaf(S[3], cl2, -b[1]));
  ls[0] = ls[0] * a[0] + (p00 + p10);
  ls[1] = ls[1] * a[1] + (p01 + p11);
  const float s0 = __shfl_sync(FULL, a[0], g >> 1), s1 = __shfl_sync(FULL, a[1], g >> 1);
  const float ag = (g & 1) ? s1 : s0;                // rescale of O row g
  if (__any_sync(FULL, ag != 1.f)) {
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) { o[jn][0] *= ag; o[jn][1] *= ag; }
  }
  const uint32_t pa0 = dk2_movt(pack_h2(p00, p01));   // -> row g, keys kw + 2t, 2t+1
  const uint32_t pa2 = dk2_movt(pack_h2(p10, p11));   // -> row g, keys kw + 8 + 2t, 2t+1
  uint32_t w01[4], w23[4];
  v_frags(Vbuf, kw + 2 * t, h, g, w01, w23);
#pragma unroll
  for (int jn = 0; jn < 8; ++jn) {
    const uint32_t s01 = w01[jn >> 1], s23 = w23[jn >> 1];
    const uint32_t b0 = e4m3x2_to_h2((jn & 1) ? (s01 >> 16) : s01);
    const uint32_t b1 = e4m3x2_to_h2((jn & 1) ? (s23 >> 16) : s23);
    mma16816(o[jn], pa0, 0u, pa2, 0u, b0, b1);
  }
}
// S^T-layout state (rows 2t, 2t+1) -> K1's layout (row g): mA everywhere, lA on t == 0 lanes (K1 quad-sums lA later)
__device__ __forceinline__ void dk2_state(const float (&m2)[2], const float (&ls)[2], const int g, const int t,
                                          float& mA, float& lA) {
  float l0 = ls[0], l1 = ls[1];
#pragma unroll
  for (int d = 4; d <= 16; d <<= 1) { l0 += __shfl_xor_sync(FULL, l0, d); l1 += __shfl_xor_sync(FULL, l1, d); }
  const float m0 = __shfl_sync(FULL, m2[0], g >> 1), m1 = __shfl_sync(FULL, m2[1], g >> 1);
  const float L0 = __shfl_sync(FULL, l0, g >> 1), L1 = __shfl_sync(FULL, l1, g >> 1);
  mA = (g & 1) ? m1 : m0;
  lA = (t == 0) ? ((g & 1) ? L1 : L0) : 0.f;
}

'''

DK2_PATCHES = (
    # device helpers, before k1_main
    ("#ifndef K1_L8_MINB\n#define K1_L8_MINB 2\n#endif\n",
     DK2_DEVICE + "#ifndef K1_L8_MINB\n#define K1_L8_MINB 2\n#endif\n"),
    # loader: K chunk -> wgmma K-major interleave layout
    ("        cp_async16(stg + x * ROWB + 16 * kswz<L>(x, wch), kc + goff + 16 * wch, nb);\n",
     "        cp_async16(stg + (wch >> 2) * 4096 + (x >> 3) * 512 + (wch & 3) * 128 + (x & 7) * 16, kc + goff + 16 * wch, nb);"
     "   // dk2\n"),
    # Q prep: e4m3 copy for wgmma (aux region, unused at L=1)
    ("        reinterpret_cast<uint32_t*>(qs)[(hh * R8 + r) * 32 + lane] = pack_h2(x.x, x.y);   // fp16\n",
     "        reinterpret_cast<uint32_t*>(qs)[(hh * R8 + r) * 32 + lane] = pack_h2(x.x, x.y);   // fp16\n"
     "        *reinterpret_cast<uint16_t*>(smem + off_aux<L, ST>() + hh * 512 + il_off(r, 2 * lane, 128, 512)) =\n"
     "            (uint16_t)__nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);   // dk2: e4m3 Q for wgmma\n"),
    ("  if constexpr (L == 8) fence_async_smem();          // Q, new K / V^T tiles -> async proxy (wgmma)\n",
     "  fence_async_smem();                                // Q (and L=8 new K / V^T tiles) -> async proxy (wgmma)\n"),
    # roles
    ("    const int h = warp & 1, q4 = warp >> 1, kw = 16 * q4;\n",
     "    const int h = warp >> 2, q4 = warp & 3, kw = 16 * q4;   // dk2: warpgroup = kv head, warp = key quarter\n"
     "    const uint8_t* qe = smem + off_aux<L, ST>();\n"),
    ("    float mA = neg_inf(), lA = 0.f;\n",
     "    float mA = neg_inf(), lA = 0.f;\n    float m2[2] = {neg_inf(), neg_inf()}, ls[2] = {0.f, 0.f};\n"),
    ('      asm volatile("bar.sync %0, 64;\\n" :: "r"(1 + q4) : "memory");   // v24: warp-pair barrier\n',
     "      fence_async_smem();\n      __syncthreads();   // dk2: each warpgroup's wgmma reads all 64 keys of the stage\n"),
    ("      tile_compute1<2>(Kb, Kb + TILE * ROWB, kw, h, g, t, qf, o, mA, lA, cl2, need, ts, seq, seq, W);\n",
     "      dk2_tile(Kb, Kb + TILE * ROWB, qe, kw, h, g, t, o, m2, ls, cl2, need, ts, seq, seq, W);\n"),
    ("    if (last && q4 == 0) tile_compute1<2>(newk, newv, 0, h, g, t, qf, o, mA, lA, cl2, true, seq, seq + 1, seq, W);\n",
     "    dk2_state(m2, ls, g, t, mA, lA);\n"
     "    if (last && q4 == 0) tile_compute1<2>(newk, newv, 0, h, g, t, qf, o, mA, lA, cl2, true, seq, seq + 1, seq, W);\n"),
    # quarter merge: warp index of (q, hh) is 4 hh + q
    ("      for (int q = 0; q < 4; ++q) M = fmaxf(M, mw[((q << 1) | hh) * 8 + r]);\n",
     "      for (int q = 0; q < 4; ++q) M = fmaxf(M, mw[((hh << 2) | q) * 8 + r]);   // dk2\n"),
    ("        const int w = (q << 1) | hh;\n", "        const int w = (hh << 2) | q;   // dk2\n"),
)


def patch_k1_source(text):
    """Patch the K1 CUDA source itself (lab use: DK2 replaces K1's L=1 path, no fallback)."""
    if "wk0 = 16 * (tid >> 6)" not in text:
        text = _k1w.apply_k1w(text)
    for old, new in DK2_PATCHES:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"DK2 anchor found {n} times: {old[:70]!r}")
        text = text.replace(old, new, 1)
    return text


L8_OFF = (
    ("  return (int64_t)(ST == 3 ? occ_of<8, 3>() : occ_of<8, 4>());\n", "  return 0;   // dk2 module: L=8 not built\n"),
    ("    if (stages == 3) launch_main<8, 3>(p, grid, st);\n    else launch_main<8, 4>(p, grid, st);\n",
     '    TORCH_CHECK(false, "dk2 module: L=8 not built");\n'),
)

_LAYER_CODE = r'''
# ---- DK2 (tools/at_dk2_patches.py, notes/study/attn_full_fresh.md s6): K1 L=1 with QK on wgmma m64n8k32 e4m3, keys as M.
# Own module built from K1_CUDA_SRC (K1-W included) + DK2_SRC_PATCHES, kernel k1d_main; _k1_run sends every L=1 call to it
# only after _dk2_self_test passes (e4m3-query reference) and its occupancy equals K1's; otherwise K1(-W) runs unchanged.
DK2_ENABLE = True
DK2_TEST_CASES = (   # (B, ctx, L, sliding, page, kv_scale, seed, heads): K1's L=1 self-test cases + judge-like batches
    (2, 3000, 1, True, 1, 1.0, 101, (16, 2)), (2, 2500, 1, False, 64, 1.0 / 64, 104, (16, 2)),
    (1, 9, 1, True, 64, 1.0, 107, (16, 2)), (3, 1500, 1, False, 64, 1.0, 109, (128, 16)),
    (1, 9000, 1, False, 1, 1.0, 110, (16, 2)), (16, 3000, 1, False, 1, 1.0, 201, (16, 2)),
    (32, 2000, 1, False, 1, 1.0 / 64, 202, (16, 2)), (16, 6000, 1, False, 1, 1.0, 203, (16, 2)))
_DK2 = {"mod": None, "ok": False}


def _dk2_build():
    from torch.utils.cpp_extension import load_inline
    src = K1_CUDA_SRC
    if "wk0 = 16 * (tid >> 6)" not in src:
        raise RuntimeError("dk2: K1-W loader missing from K1_CUDA_SRC")
    for old, new in DK2_SRC_PATCHES + DK2_L8_OFF:
        if src.count(old) != 1:
            raise RuntimeError(f"dk2: anchor found {src.count(old)} times: {old[:60]!r}")
        src = src.replace(old, new, 1)
    if src.count("namespace k1 {") != 1:
        raise RuntimeError("dk2: namespace anchor")
    src = src.replace("namespace k1 {", "#define k1_main k1d_main\nnamespace k1 {", 1)
    tag = hashlib.sha1((K1_CPP_SRC + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"k1_dk2_{tag}", cpp_sources=K1_CPP_SRC, cuda_sources=src,
                      functions=["k1_forward", "k1_occupancy"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _DK2["mod"] = mod
    print(f"[dk2] compiled in {time.time() - t0:.1f}s", flush=True)
    return mod


def _dk2_self_test(dev):
    cnt = _k1_counters(dev)
    worst = (0.0, 0.0)
    for (B, ctx, L, sliding, page, kvs, seed, heads) in DK2_TEST_CASES:
        d = _k1_mk_case(dev, B, ctx, L, sliding, page, kvs, seed, heads[0], heads[1])
        out = _k1_run(d["qkv"], d, d["config"])
        out2 = _k1_run(d["qkv"], d, d["config"])
        torch.cuda.synchronize(dev)
        hkv = heads[1]
        pos = d["seq_lens"].long()[:, None] + torch.arange(L, device=dev)
        slots = torch.gather(d["page_table"].long(), 1, pos // page) * page + pos % page
        k_new = d["k_cache"].view(torch.uint8).view(-1, hkv, 64)[slots]
        v_new = d["v_cache"].view(torch.uint8).view(-1, hkv, 64)[slots]
        ref = _k1_ref_attention(d, k_new, v_new, q_e4m3=True)
        tot, row = _k1_rel_rows(out, ref)
        kref, vref = _k1_ref_new_kv(d)
        kt, _ = _k1_rel_rows(k_new.view(FP8).float(), kref)
        vt, _ = _k1_rel_rows(v_new.view(FP8).float(), vref)
        ok = (bool(torch.isfinite(out.float()).all()) and torch.equal(out, out2) and tot < 4e-2 and row < 1e-1
              and kt < 3e-2 and vt < 3e-2 and (cnt is None or int(cnt.abs().sum()) == 0))
        worst = (max(worst[0], tot), max(worst[1], row))
        if not ok:
            print(f"[dk2] self-test FAIL B={B} ctx={ctx} {'sliding' if sliding else 'full'} page={page} kv_scale={kvs:g}: "
                  f"rel {tot:.3e} row {row:.3e} new K {kt:.3e} V {vt:.3e}", flush=True)
            return False
    print(f"[dk2] self-test passed on {len(DK2_TEST_CASES)} cases (worst rel {worst[0]:.3e} row {worst[1]:.3e})", flush=True)
    return True


def _dk2_prepare(dev):
    _DK2["ok"] = False
    if not DK2_ENABLE:
        return
    t0 = time.time()
    try:
        _pt1_wait("dk2")
        if _DK2["mod"] is None:
            _dk2_build()
        occ = int(_DK2["mod"].k1_occupancy(1, 4))
        if occ != _k1_occupancy(1, 4, dev):
            raise RuntimeError(f"occupancy {occ} != K1's {_k1_occupancy(1, 4, dev)}")
        _DK2["ok"] = True
        if not _dk2_self_test(dev):
            _DK2["ok"] = False
    except Exception as ex:  # noqa: BLE001
        _DK2["ok"] = False
        print(f"[dk2] FAILED, L=1 stays on K1: {ex!r}"[-2000:], flush=True)
    print(f"[dk2] {'route live (every K1 L=1 call)' if _DK2['ok'] else 'OFF: K1 L=1 unchanged'} ({time.time() - t0:.1f}s)",
          flush=True)

'''

LAYER_BLOCK = ("\nDK2_SRC_PATCHES = " + repr(DK2_PATCHES) + "\nDK2_L8_OFF = " + repr(L8_OFF) + "\n" + _LAYER_CODE)

LAYER_PATCHES = (
    ("\ndef _k1_rms_t(x, eps=RMS_EPS):\n", LAYER_BLOCK + "\ndef _k1_rms_t(x, eps=RMS_EPS):\n"),
    ('    _K1_MOD.k1_forward(qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),\n',
     '    (_DK2["mod"] if (L == 1 and _DK2["ok"]) else _K1_MOD).k1_forward(qkv, data["seq_lens"], data["cache_loc"],\n'
     '                    data["page_table"], data["k_cache"].view(torch.uint8),\n'),
    ('    if GD_BUILD:\n        jobs.append(("gd", _gd_build))\n',
     '    if GD_BUILD:\n        jobs.append(("gd", _gd_build))\n    if DK2_ENABLE and USE_K1:\n        jobs.append(("dk2", _dk2_build))\n'),
    ('            state["k1_ok"] = _k1_prepare(device)\n',
     '            state["k1_ok"] = _k1_prepare(device)\n            if state["k1_ok"]:\n                _dk2_prepare(device)\n'),
)


def apply_dk2(text):
    """Layer integration (k3_m32_2.py text; K1-W applied first when missing): DK2 module (PT1 prefetch build), prepare
    self-test, _k1_run switch; any failure (build, occupancy, self-test) leaves every L=1 call on K1(-W)."""
    if "wk0 = 16 * (tid >> 6)" not in text:
        text = _k1w.apply_k1w(text)
    for old, _ in DK2_PATCHES:        # the K1 source in the text must accept every DK2 patch (applied at build time)
        if text.count(old) != 1:
            raise ValueError(f"DK2 source anchor found {text.count(old)} times: {old[:70]!r}")
    for old, new in LAYER_PATCHES:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"DK2 layer anchor found {n} times: {old[:70]!r}")
        text = text.replace(old, new, 1)
    return text

"""k2a_v40_patches.py: K2A v4.0 "LEAN + FLUSH2 + fine role weight" as exact text patches against the K2A source embedded in
kernels/muse_decode_layer/k3_v21.py (K2A33_KERNEL_SRC = k2a_v37's KERNEL_SRC_V3) plus the bucket-256 call site.

Measured (tools/box/k2a_v40.py, lab copy `lf` at hot_w4 78 = row lf_f28; box_out/v40/run3.log, 5 layer routings x 30 paired rounds):
T=256 MoE 313.6 -> 299.6 us, median -11.7 us (x1.042; -2.5 .. -14.0 per routing); T=128 untouched (vid 5 never runs this code).
  1 LEAN    hot_item's wgmma descriptor = {lo + offset, hi} formed inside the asm (new wgmma_rs2<N>, N = 16/32/48/64): one 32-bit add
            per wgmma instead of a 64-bit add + register copies. Identical descriptor values (the 14-bit address field never carries:
            smem < 228 KB, offsets <= 0xF0). -31 SASS instructions per stage per warp (414 -> 383 at N=64).
  2 FLUSH2  hot_item's flush with affine addresses: one base pointer per lane and compile-time offsets per (chunk q, hh)
            (xw_elem(n) = xw_elem(2t) + 2048 (n >> 3) + 8 hh for n = 8q + 2t + hh). Same values, same stores / atomics (bitwise).
            K1 spill 232 -> 160 B, K2 136 -> 120 B; hot flush -16% in the per-warp profile.
  3 W4FINE  pick_role: hot_w4 >= 50 means a role-split weight of (hot_w4 - 50) / 32 (ship: 8 / 8 = 1.0); runtime only (no wgmma path).
  4 CALL256 bucket 256 (vid 2 HOTCAP) calls with hot_w4 = 78 (= 28 / 32): with 1-3 the hot role is faster, so fewer CTAs take it.
            Bucket 128's vid-2 fallback keeps the default weight.
Numerics: no math changes; the rebalanced split moves Stream-K cut points, so split-item FP32 atomics differ (vs shipped <= 3.4e-4,
same 3.75e-3 vs the FP32 reference as shipped).

    python -I tools/k2a_v40_patches.py [layer.py] [out.py]   # asserts every `old` occurs exactly once, applies all to a copy
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v21.py"


def _specs2():
    out = []
    for n in (16, 32, 48, 64):
        k = n // 2
        regs = ",".join("%%%d" % i for i in range(k))
        dl = ", ".join('"+f"(d[%d])' % i for i in range(k))
        out.append(
            "template <> __device__ __forceinline__ void wgmma_rs2<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
            "uint32_t a3, uint32_t lo, uint32_t hi) {\n"
            "  asm volatile(\"{\\n.reg .pred p;\\n.reg .b64 dd;\\nmov.b64 dd, {%%%d, %%%d};\\nsetp.ne.b32 p, 1, 0;\\n"
            "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%%%d,%%%d,%%%d,%%%d}, dd, p, 1, 1, 0;\\n}\\n\"\n"
            "    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(lo), \"r\"(hi));\n}"
            % (n, k + 4, k + 5, n, regs, k, k + 1, k + 2, k + 3, dl))
    return ("// v4.0 LEAN: wgmma m64nNk16 RS with the descriptor formed in the asm from two 32-bit halves\n"
            "template <int N> __device__ __forceinline__ void wgmma_rs2(float* d, uint32_t a0, uint32_t a1, uint32_t a2,"
            " uint32_t a3, uint32_t lo, uint32_t hi);\n" + "\n".join(out) + "\n")


FLUSH_OLD = r"""  // ---- flush (same as the cold path's, 8-token chunks q < ceil(ntok / 8)) ----
  const int nch = (ntok + 7) >> 3;
  const int tile = rb * 8 + warp;
  const float gs = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);
  const bool whole = (s0 == 0 && s1 == 16);
  const int pid = S.hp[e] + j;
#pragma unroll
  for (int q = 0; q < N / 8; ++q) {
    if (q < nch) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = q * 8 + 2 * t + hh;
        if (n < ntok) {
          const int c = c0 + n;
          const float v0 = hacc[4 * q + hh] * gs, v1 = hacc[4 * q + 2 + hh] * gs;
          if constexpr (MODE == 1) {
            const int col = 8 * tile + g;
            if (whole) {
              reinterpret_cast<__nv_bfloat16*>(P.xw2)[xw_elem(pid, 64, n, col)] = __float2bfloat16_rn(swiglu(v0, v1));
            } else {
              atomicAdd(P.h + (size_t)c * (2 * KDIM) + col, v0);
              atomicAdd(P.h + (size_t)c * (2 * KDIM) + KDIM + col, v1);
            }
          } else {
            float* yp = P.y + (size_t)c * KDIM + 16 * tile + g;
            if (whole) { yp[0] = v0; yp[8] = v1; }
            else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
          }
        }
      }
    }
  }
  if (MODE == 2 && whole) {
"""

FLUSH_NEW = r"""  // ---- flush (v4.0 FLUSH2: one base pointer per lane, compile-time offsets per (q, hh); same values, stores, atomics) ----
  const int nch = (ntok + 7) >> 3;
  const int tile = rb * 8 + warp;
  const float gs = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);
  const bool whole = (s0 == 0 && s1 == 16);
  if constexpr (MODE == 1) {
    const int col = 8 * tile + g;
    if (whole) {   // xw_elem(pid, 64, n, col) = xw_elem(pid, 64, 2t, col) + 2048 (n >> 3) + 8 hh for n = 8q + 2t + hh
      __nv_bfloat16* xb = reinterpret_cast<__nv_bfloat16*>(P.xw2) + xw_elem(S.hp[e] + j, 64, 2 * t, col);
#pragma unroll
      for (int q = 0; q < N / 8; ++q)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
          if (q * 8 + 2 * t + hh < ntok)
            xb[2048 * q + 8 * hh] = __float2bfloat16_rn(swiglu(hacc[4 * q + hh] * gs, hacc[4 * q + 2 + hh] * gs));
    } else {
      float* hb = P.h + (size_t)(c0 + 2 * t) * (2 * KDIM) + col;
#pragma unroll
      for (int q = 0; q < N / 8; ++q)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
          if (q * 8 + 2 * t + hh < ntok) {
            atomicAdd(hb + (8 * q + hh) * (2 * KDIM), hacc[4 * q + hh] * gs);
            atomicAdd(hb + (8 * q + hh) * (2 * KDIM) + KDIM, hacc[4 * q + 2 + hh] * gs);
          }
    }
  } else {
    float* yb = P.y + (size_t)(c0 + 2 * t) * KDIM + 16 * tile + g;
#pragma unroll
    for (int q = 0; q < N / 8; ++q)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
        if (q * 8 + 2 * t + hh < ntok) {
          const float v0 = hacc[4 * q + hh] * gs, v1 = hacc[4 * q + 2 + hh] * gs;
          float* yp = yb + (8 * q + hh) * KDIM;
          if (whole) { yp[0] = v0; yp[8] = v1; }
          else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
        }
  }
  if (MODE == 2 && whole) {
"""

# (what, old, new): every `old` occurs exactly once in kernels/muse_decode_layer/k3_v21.py
PATCHES = [
    ("LEAN: wgmma_rs2 specialisations after the shipped wgmma_rs ones",
     "@WGSPECS@\n#undef K3_WG_HEAD\n",
     "@WGSPECS@\n#undef K3_WG_HEAD\n" + _specs2()),
    ("LEAN: hot_item descriptor halves",
     "    const uint64_t d0 = wg_desc_sa(sa(st + SW), 4096u);\n    // per kb: decode into one of two 16-register A buffers",
     "    const uint64_t d0 = wg_desc_sa(sa(st + SW), 4096u);\n"
     "    const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);   // v4.0 LEAN: the low word's add never carries\n"
     "    // per kb: decode into one of two 16-register A buffers"),
    ("LEAN: hot_item wgmmas on wgmma_rs2",
     "      wgmma_rs<N>(hacc, h[0], h[8], h[1], h[9], d0 + 64 * i);\n"
     "      wgmma_rs<N>(hacc, h[2], h[10], h[3], h[11], d0 + 64 * i + 16);\n"
     "      wgmma_rs<N>(hacc, h[4], h[12], h[5], h[13], d0 + 64 * i + 32);\n"
     "      wgmma_rs<N>(hacc, h[6], h[14], h[7], h[15], d0 + 64 * i + 48);\n",
     "      wgmma_rs2<N>(hacc, h[0], h[8], h[1], h[9], dlo + 64 * i, dhi);\n"
     "      wgmma_rs2<N>(hacc, h[2], h[10], h[3], h[11], dlo + 64 * i + 16, dhi);\n"
     "      wgmma_rs2<N>(hacc, h[4], h[12], h[5], h[13], dlo + 64 * i + 32, dhi);\n"
     "      wgmma_rs2<N>(hacc, h[6], h[14], h[7], h[15], dlo + 64 * i + 48, dhi);\n"),
    ("FLUSH2: hot_item flush with affine addresses", FLUSH_OLD, FLUSH_NEW),
    ("W4FINE: pick_role fine role-split weight (hot_w4 >= 50 -> (hot_w4 - 50) / 32)",
     "    Wh *= P_hot_w4;\n    Wc *= 8;\n",
     "    if (P_hot_w4 >= 50) { Wh *= P_hot_w4 - 50; Wc *= 32; } else { Wh *= P_hot_w4; Wc *= 8; }   // v4.0 fine weight\n"),
    ("CALL256: bucket 256 (vid 2) with role-split weight 78 = 28 / 32",
     "    return _k2a33_call(st, data, K2A33_PLAN[rep], hot_t=K2A33_HOT_T_BUCKET.get(rep))\n",
     "    return _k2a33_call(st, data, K2A33_PLAN[rep], hot_t=K2A33_HOT_T_BUCKET.get(rep),\n"
     "                       hot_w4=(78 if rep == 256 else None))   # v4.0: LEAN + FLUSH2 hot role -> weight 28 / 32\n"),
]
KERNEL_PATCHES = PATCHES[:5]   # the CUDA-source ones (they also apply to k2a_v37.KERNEL_SRC_V3: used by the lab check)


def apply(text, patches=PATCHES):
    for what, old, new in patches:
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"k2a_v40 patch {what!r}: `old` occurs {n} times (want 1)")
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_v21_k2a40_check.py")
    text = src.read_text(encoding="utf-8")
    for what, old, _ in PATCHES:
        print(f"  {text.count(old)}x  {what}")
    patched = apply(text)
    out.write_text(patched, encoding="utf-8")
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(PATCHES)} patches hit once; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

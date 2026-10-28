"""REV probe (independent review, notes/study/independent_review_oct28.md; request R-rev-1): does the shipped VML RS-wgmma
consumer's tensor time collide with the weight stream, is the SM clock throttled under that load, and does staging the
next stages in L2 recover the excess?

Built on kernels/lab/vml_probe2.py (same kernel, plan, harness); base = the shipped VML v13 consumer (switches SC2 + DS + CL,
as kernels/lab/unp_probe.py). Compile-time switches added here (text patches of vml_probe2.HEAD / BODY):
  PG_TO   1 FT: weights and x streamed as in S; the consumer skips the code / scale LDS and the unpack (constant A
            fragments) and releases each weight slot right after its full-barrier wait; wgmma on the streamed x
          2 Z: FT without wgmma (stream + barrier traffic only)
          (both timing only: outputs wrong by design)
  PG_L2D  d > 0: the weight producer, right after issuing stage it, issues cp.async.bulk.prefetch.L2 for stage it + d of
            the CTA's own stage sequence (both 18 KB halves); stages NS .. d - 1 of the first unit are prefetched at entry.
            Outputs bitwise = S.
  PG_CTRL dummy define (a second, separately built control copy of S)
"""
import vml_probe2 as P2

_PF = r"""#ifndef PG_L2D
#define PG_L2D 0
#endif
__device__ __forceinline__ void l2pf(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ const uint8_t* stage_src(const Args& A, int u) {
  const int2 un = A.unit[u];
  return A.ws13 + ((size_t)A.job[un.x].x * 64 + 2 * un.y) * 16 * SW;
}
"""

_OLD_PW = """      bulk_g2s_ef(dst, wb + (size_t)s * SW, SW, fb, pol);
      bulk_g2s_ef(dst + SW, wb + (size_t)(16 + s) * SW, SW, fb, pol);
    }
  }
}"""
_NEW_PW = """      bulk_g2s_ef(dst, wb + (size_t)s * SW, SW, fb, pol);
      bulk_g2s_ef(dst + SW, wb + (size_t)(16 + s) * SW, SW, fb, pol);
#if PG_L2D
      {
        const int s2 = s + PG_L2D, u2 = u + (s2 >> 4), ss = s2 & 15;
        if (u2 < u1) {
          const uint8_t* w2 = stage_src(A, u2);
          l2pf(w2 + (size_t)ss * SW, SW);
          l2pf(w2 + (size_t)(16 + ss) * SW, SW);
        }
      }
#endif
    }
  }
}"""

_OLD_PW0 = """  const uint64_t pol = policy_evict_first();
  int it = 0;
  for (int u = u0; u < u1; ++u) {
    const int2 un = A.unit[u];"""
_NEW_PW0 = """  const uint64_t pol = policy_evict_first();
  int it = 0;
#if PG_L2D
  if (u0 < u1) {
    const uint8_t* w0 = stage_src(A, u0);
    for (int j = NS; j < PG_L2D && j < 16; ++j) { l2pf(w0 + (size_t)j * SW, SW); l2pf(w0 + (size_t)(16 + j) * SW, SW); }
  }
#endif
  for (int u = u0; u < u1; ++u) {
    const int2 un = A.unit[u];"""

_DEFS = r"""
#ifndef PG_TO
#define PG_TO 0
#endif
#ifndef PG_CTRL
#define PG_CTRL 0
#endif
"""

_OLD_LDS = """  if constexpr (RES < 2) {
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + (PG_WS ? 8192 + wp * 256 : 16384 + tile * 256)) + lane;"""
_NEW_LDS = """  if constexpr (RES < 2 && !PG_TO) {
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + (PG_WS ? 8192 + wp * 256 : 16384 + tile * 256)) + lane;"""

_OLD_REL = "    if (!PG_CL && lane == 0) mbar_arrive(bars + 8 * (PG_WS ? WSB + 4 * NS + q * NS + ws : NS + ws));"
_NEW_REL = "    if ((!PG_CL || PG_TO) && lane == 0) mbar_arrive(bars + 8 * (PG_WS ? WSB + 4 * NS + q * NS + ws : NS + ws));"

_OLD_TR = "    if constexpr (RES < 2 && TRICKS) {"
_NEW_TR = "    if constexpr (RES < 2 && TRICKS && !PG_TO) {"

_OLD_PL = """    } else if constexpr (RES < 2) {
      const int i = (GS == 2) ? (gi >> 1) : gi;"""
_NEW_PL = """    } else if constexpr (RES < 2 && !PG_TO) {
      const int i = (GS == 2) ? (gi >> 1) : gi;"""

_OLD_MMA = """    wg_fence();
#pragma unroll
    for (int h2 = 0; h2 < GS / 2; ++h2) {
      const uint32_t kk = (uint32_t)(gi * GS + 2 * h2);   // k16 step (even); SW128: k64 block kk >> 2, 32 B per step inside
      const uint32_t off = PG_XSW ? (kk >> 2) * (uint32_t)N * 8u + (kk & 3u) * 2u : 16u * kk;
      const uint32_t st2 = PG_XSW ? 2u : 16u;
      if (first && gi == 0 && h2 == 0) mma2<N, true>(d, hh + 8 * h2, dlo + off, dhi, st2);
      else mma2<N, false>(d, hh + 8 * h2, dlo + off, dhi, st2);
    }
    wg_commit();
    wg_wait<NPD>();
"""
_NEW_MMA = """    if constexpr (PG_TO == 2) {
    } else {
""" + _OLD_MMA + """    }
"""


def _patch(src, old, new):
    assert src.count(old) == 1, (src.count(old), old[:70])
    return src.replace(old, new)


HEAD = P2.HEAD
HEAD = _patch(HEAD, "__device__ __forceinline__ void prod_w(", _PF + "__device__ __forceinline__ void prod_w(")
HEAD = _patch(HEAD, _OLD_PW0, _NEW_PW0)
HEAD = _patch(HEAD, _OLD_PW, _NEW_PW)
BODY = P2.BODY
BODY = _patch(BODY, "// ---------------- consumer (probe 2) ----------------\n",
              "// ---------------- consumer (probe 2) ----------------\n" + _DEFS)
BODY = _patch(BODY, _OLD_LDS, _NEW_LDS)
BODY = _patch(BODY, _OLD_REL, _NEW_REL)
BODY = _patch(BODY, _OLD_TR, _NEW_TR)
BODY = _patch(BODY, _OLD_PL, _NEW_PL)
BODY = _patch(BODY, _OLD_MMA, _NEW_MMA)
CUDA = HEAD + BODY
CPP = P2.CPP
_S = {"PG_SC2": 1, "PG_DS": 1, "PG_CL": 1}   # shipped VML v13 consumer (VM_TRICKS 1, VM_TRICKS_DQ 0)
ARMS = {"S": dict(_S), "S2": dict(_S, PG_CTRL=1), "Z": dict(_S, PG_TO=2), "FT": dict(_S, PG_TO=1),
        "C": dict(_S, PG_RES=1), "T": dict(_S, PG_RES=2), "L3": dict(_S, PG_L2D=3), "L6": dict(_S, PG_L2D=6),
        "L10": dict(_S, PG_L2D=10)}
NOCHECK = ("PG_RES", "PG_XR", "PG_TO")    # timing-only arms
EXACT = ("S", "S2", "L3", "L6", "L10")      # streaming arms whose outputs must equal S bit for bit
to_sw128 = P2.to_sw128


def flags_of(arm):
    return P2.V1.FLAGS + [f"-D{k}={v}" for k, v in ARMS[arm].items()]


def build(arm, verbose=False):
    import hashlib
    import time
    from torch.utils.cpp_extension import load_inline
    fl = flags_of(arm)
    tag = hashlib.sha1((CPP + CUDA + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"rev_probe_{arm}_{tag}", cpp_sources=CPP, cuda_sources=CUDA, functions=["vp2_run", "vp2_attrs"],
                      extra_cuda_cflags=fl, verbose=verbose)
    return mod, time.time() - t0

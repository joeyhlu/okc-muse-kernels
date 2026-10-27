"""fin1_patches.py (study-a): FIN1 = all 9 warps finalize K2A vid 3's handed items once the consumers are done (T=16 / 32).

Why (R-study-a-5, box_out/sa5, per-warp exits): in a hot vid-3 CTA all consumer warps end together (skew 0.00 us), and
the epilogue warp exits 4.4-9.5 us after them (98-107 % of the hot role's completion excess on hot-last routings); the
completion lag after the last warp is 0.5-1.0 us. The tail grows with the finalized item's tokens (~2.5 us per 8-token
chunk; K2 hot: 1 chunk 1.6, 2: 5.5, 3: 7.3, 4: 10.6 us). EPI1 (R-study-a-4) batched that warp's loads and changed
nothing: the per-token work is single-warp compute latency (K1: SwiGLU with an IEEE divide per element, each behind a
predicated slow-path CALL that serialises the elements; K2: a 5-deep shuffle chain + an atomic per token), not L2 trips.
FIN1 spreads it over warps instead of batching it:
  * the finalize body (K1: h gate / up -> SwiGLU -> bf16 inter, zero h; K2: sum of squares of bf16(y) (1 + w) -> ss)
    becomes fin36_item<MODE>(rb, c0, ntok, w0, nw): tokens w0, w0 + nw, ... in groups of 4, each token exactly as
    shipped (same lane -> column map, same operation order, same shuffle tree, one atomic per token): bitwise identical;
  * after their item walk the consumer warps raise s_fin[0] and wait on named barrier 1 (288 threads = consumers + the
    epilogue warp); from then on the epilogue warp posts each remaining finalize (double-buffered parameters), and the
    9 warps take tokens w (consumers) / 8 (epilogue) mod 9; a final DONE post releases them;
  * runtime switch: probe bit 6 from FIN1_ENABLE; bit 5 from FIN1_K1_ONLY keeps K2 on the shipped finalize (bit 5 was
    EPI1's, unused in v36; RS1 in k3_v37 uses bit 7); a prepare-time self-test compares FIN1
    on / off at T = 16 / 32 and turns FIN1 off on any failure: inter / y / out within their off-vs-off envelope measured
    in the same prepare (bitwise where off is), h / counters back at zero, ss against an exact FP64 recomputation, output
    close. K2's ss (every input) and y (skewed routings) are FP32 atomics in arrival order, so they vary run to run even
    with FIN1 off: the all-bitwise test failed on ss (R-study-a-6), the fixed-bitwise inter / y test on y (R-study-a-7),
    a relative ss envelope on ss's heavy-tailed call-to-call distance (R-study-a-8; now dropped, the FP64 check stays).

    python -I tools/fin1_patches.py [k3_v36.py] [out.py]
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ('K2A33_KERNEL_SRC = r"""', '"""')
G36 = ("__device__ void gemm36(", "k1_kernel36(const Params P) {")

FIN_FN = r'''
// FIN1 (study-a): the finalize of one handed vid-3 item for tokens w0, w0 + nw, ... (each token exactly as the shipped
// epilogue warp does it: same lane -> column map, operation order and shuffle tree; one ss atomic per token in K2).
template <int MODE>
__device__ __forceinline__ void fin36_item(const Params& P, int rb, int c0, int ntok, int w0, int nw, bool acq) {
  const int lane = threadIdx.x & 31;
  if (acq) asm volatile("fence.acq_rel.gpu;\n" ::: "memory");   // helper warps: acquire after the epilogue's fence
  if (MODE == 2) {
    const int col = rb * 128 + 4 * lane;
    const float4 wv = __ldg(reinterpret_cast<const float4*>(P.nw + col));
    for (int i0 = w0; i0 < ntok; i0 += 4 * nw) {
      float4 yv[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int n = i0 + u * nw;
        if (n < ntok) yv[u] = __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)(c0 + n) * KDIM + col));
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int n = i0 + u * nw;
        float pz = 0.f;
        if (n < ntok) {
          float z;
          z = bf16r(yv[u].x) * wv.x; pz += z * z;  z = bf16r(yv[u].y) * wv.y; pz += z * z;
          z = bf16r(yv[u].z) * wv.z; pz += z * z;  z = bf16r(yv[u].w) * wv.w; pz += z * z;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) pz += __shfl_xor_sync(FULL, pz, o);
        if (lane == 0 && n < ntok) atomicAdd(P.ss + c0 + n, pz);
      }
    }
  } else {
    const int cl = 2 * lane;
    const int col = rb * 64 + cl;
    for (int i0 = w0; i0 < ntok; i0 += 4 * nw) {
      float2 gv[4], uv[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int n = i0 + u * nw;
        if (n < ntok) {
          const float* hp = P.h + (size_t)(c0 + n) * (2 * KDIM) + col;
          gv[u] = __ldcg(reinterpret_cast<const float2*>(hp));
          uv[u] = __ldcg(reinterpret_cast<const float2*>(hp + KDIM));
        }
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int n = i0 + u * nw;
        if (n < ntok) {
          const int c = c0 + n;
          float* hp = P.h + (size_t)c * (2 * KDIM) + col;
          const uint32_t pk = pack_bf2(swiglu(gv[u].x, uv[u].x), swiglu(gv[u].y, uv[u].y));
          *reinterpret_cast<uint32_t*>(P.inter + ((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)) = pk;
          *reinterpret_cast<float2*>(hp) = make_float2(0.f, 0.f);
          *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);
        }
      }
    }
  }
}

'''

# the epilogue warp's two finalize blocks (exact shipped text) -> one block with the FIN1 hand-out
OLD_FIN = r'''          if (MODE == 2 && last) {
            __threadfence();
            const int col = rb * 128 + 4 * lane;
            const float4 wv = __ldg(reinterpret_cast<const float4*>(P.nw + col));
            for (int n0 = 0; n0 < ntok; n0 += 4) {
              float4 yv[4];
#pragma unroll
              for (int u = 0; u < 4; ++u)
                if (n0 + u < ntok) yv[u] = __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)(c0 + n0 + u) * KDIM + col));
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                float pz = 0.f;
                if (n0 + u < ntok) {
                  float z;
                  z = bf16r(yv[u].x) * wv.x; pz += z * z;  z = bf16r(yv[u].y) * wv.y; pz += z * z;
                  z = bf16r(yv[u].z) * wv.z; pz += z * z;  z = bf16r(yv[u].w) * wv.w; pz += z * z;
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) pz += __shfl_xor_sync(FULL, pz, o);
                if (lane == 0 && n0 + u < ntok) atomicAdd(P.ss + c0 + n0 + u, pz);
              }
            }
          }
          if (MODE == 1 && last) {
            __threadfence();
            // ntok * 64 columns; lane handles column pair (2*(lane&31)) of 64 for tokens n = 0..ntok-1, batched
            const int cl = 2 * lane;
            const int col = rb * 64 + cl;
            for (int n0 = 0; n0 < ntok; n0 += 4) {
              float2 gv[4], uv[4];
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                if (n0 + u < ntok) {
                  const float* hp = P.h + (size_t)(c0 + n0 + u) * (2 * KDIM) + col;
                  gv[u] = __ldcg(reinterpret_cast<const float2*>(hp));
                  uv[u] = __ldcg(reinterpret_cast<const float2*>(hp + KDIM));
                }
              }
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                if (n0 + u < ntok) {
                  const int c = c0 + n0 + u;
                  float* hp = P.h + (size_t)c * (2 * KDIM) + col;
                  const uint32_t pk = pack_bf2(swiglu(gv[u].x, uv[u].x), swiglu(gv[u].y, uv[u].y));
                  *reinterpret_cast<uint32_t*>(P.inter + ((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)) = pk;
                  *reinterpret_cast<float2*>(hp) = make_float2(0.f, 0.f);
                  *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);
                }
              }
            }
          }
'''
NEW_FIN = r'''          if (last) {
            __threadfence();
            if (fin1 && *reinterpret_cast<volatile int*>(s_fin)) {
              // FIN1: the consumers are done; all 9 warps finalize this item (tokens mod 9, epilogue = slot 8)
              int* pp = s_fin + 2 + (kfin & 1) * 6;
              if (lane == 0) { pp[0] = 1; pp[1] = rb; pp[2] = c0; pp[3] = ntok; }
              __syncwarp();
              asm volatile("bar.sync 1, 288;\n" ::: "memory");
              ++kfin;
              fin36_item<MODE>(P, rb, c0, ntok, 8, 9, false);
            } else {
              fin36_item<MODE>(P, rb, c0, ntok, 0, 1, false);
            }
          }
'''
EPI_HEAD_OLD = "    // ---------------- epilogue warp: split-item counters; the last contributor finishes the item (v3.3) ----------------\n    int npart = 0;\n"
EPI_HEAD_NEW = ("    // ---------------- epilogue warp: split-item counters; the last contributor finishes the item (v3.3) ----------------\n"
                "    int npart = 0;\n    int kfin = 0;   // FIN1: cooperative finalizes posted so far\n")
EPI_END_OLD = "    return;\n  }\n\n  // ---------------- consumers (warps 0-7: warp w = tile w of the row block) ----------------"
EPI_END_NEW = ("    if (fin1) {   // FIN1: release the consumers' helper loop\n"
               "      int* pp = s_fin + 2 + (kfin & 1) * 6;\n"
               "      if (lane == 0) pp[0] = 2;\n"
               "      __syncwarp();\n"
               "      asm volatile(\"bar.sync 1, 288;\\n\" ::: \"memory\");\n"
               "    }\n" + EPI_END_OLD)
G36_HEAD_OLD = "  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const uint8_t* wsrc = (MODE == 1) ? P.ws13 : P.ws2;\n"
G36_HEAD_NEW = ("  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n"
                "  __shared__ int s_fin[14];   // FIN1: [0] consumers done, [2 + 6 k .. 7 + 6 k] posted finalize k & 1\n"
                "  const bool fin1 = ((P.probe >> 6) & 1) && (MODE == 1 || !((P.probe >> 5) & 1));   // bit 5: K1 only\n"
                "  if (tid == 0) s_fin[0] = 0;   // read by the epilogue only after a barrier with consumer warp 0\n"
                "  const uint8_t* wsrc = (MODE == 1) ? P.ws13 : P.ws2;\n")
CONS_END_OLD = ("        if (!(s0 == 0 && s1 == 16) || ho) ++npart;\n        x += s1 - s0;\n      }\n    }\n  }\n}\n\n"
                "// Ranges of this CTA")
CONS_END_NEW = ("        if (!(s0 == 0 && s1 == 16) || ho) ++npart;\n        x += s1 - s0;\n      }\n    }\n  }\n"
                "  if (fin1) {   // FIN1: help the epilogue warp finalize the handed items still ahead of it\n"
                "    if (tid == 0) *reinterpret_cast<volatile int*>(s_fin) = 1;\n"
                "    for (int k = 0;; ++k) {\n"
                "      asm volatile(\"bar.sync 1, 288;\\n\" ::: \"memory\");\n"
                "      const int* pp = s_fin + 2 + (k & 1) * 6;\n"
                "      if (pp[0] != 1) break;\n"
                "      fin36_item<MODE>(P, pp[1], pp[2], pp[3], warp, 9, true);\n"
                "    }\n"
                "  }\n"
                "}\n\n// Ranges of this CTA")
FN_ANCHOR = "// v3.6 gemm over four segments"

PY_CALL_OLD = "probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG[\"k3trig\"] else 0) | _k3p_bits(m))"
PY_CALL_NEW = ("probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG[\"k3trig\"] else 0) | _k3p_bits(m)"
               " | ((1 << 6) if FIN1_ENABLE else 0) | ((1 << 5) if FIN1_K1_ONLY else 0))")
PY_GLOBAL_ANCHOR = "def _k2a33_call(st, data, vid, probe=K2A33_HINT, mask=7, hot_t=None, hot_w4=None):\n"
PY_GLOBAL = r'''FIN1_ENABLE = True   # FIN1 (tools/fin1_patches.py): vid 3 handed-item finalize on all 9 warps once the consumers are done (probe bit 6, read at call time)


def _fin1_ulp(a, b):
    """Largest distance in bf16 ulps between two bf16 tensors (sign-magnitude bit patterns mapped to a monotone order)."""
    ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
    ka = torch.where(ia < 0, -32768 - ia, ia)
    kb = torch.where(ib < 0, -32768 - ib, ib)
    return int((ka - kb).abs().max()) if ia.numel() else 0


FIN1_FLOOR = {"inter": 2, "y": 1e-6, "oulp": 8, "ol2": 1e-4}   # absolute floors of the off-vs-off envelope (ss: hard check only;
#   out ulps floored at 8: cancellation in the weighted sum gave on 2 vs off 1 on the box, R-study-a-8)


def _fin1_metrics(a, b):
    """Distance of call a from call b per value: inter (bf16 ulps), y (max over routes of max |dy| / max |y|), ss (max
    rel), out (bf16 ulps, rel L2), plus elements of out differing and the bitwise flags of inter / y."""
    x, w = a["out"].float(), b["out"].float()
    yd = (a["y"] - b["y"]).abs()
    return {"inter": _fin1_ulp(a["inter"], b["inter"]),
            "y": float((yd.amax(1) / b["y"].abs().amax(1).clamp_min(1e-30)).max()) if yd.numel() else 0.0,
            "ss": float(((a["ss"] - b["ss"]).abs() / b["ss"].abs().clamp_min(1e-30)).max()) if a["ss"].numel() else 0.0,
            "oulp": _fin1_ulp(a["out"], b["out"]), "ol2": float((x - w).norm() / w.norm().clamp_min(1e-30)),
            "odiff": int((a["out"] != b["out"]).sum()),
            "inter_eq": bool(torch.equal(a["inter"], b["inter"])), "y_eq": bool(torch.equal(a["y"], b["y"]))}


def _fin1_selftest(st, dev):
    """FIN1 on vs off, vid 3 at T = 16 / 32, three inputs each: three calls with FIN1 off, then two with it on.

    FIN1 changes who runs the finalize code, not its arithmetic, but it moves timing, and K2A has shipped values whose
    FP32 atomic order follows timing: K2's per-route ss (one atomic per rb item and warp, every input) and K2's y on
    skewed routings (items cut across >= 3 CTAs; R-study-a-7: y differed between two FIN1-off calls on input 72). So
    each value is judged against its own off-vs-off envelope, measured here (A22): inter and y must be bitwise equal to
    the first off call if every off call was; otherwise inter / y and always out must stay within max(2 x the
    off-vs-off distance, FIN1_FLOOR). ss has no relative envelope (R-study-a-8: its call-to-call distance on the skewed
    input is heavy-tailed, 2.3e-7 .. 3.6e-6 with FIN1 off, because a reordered y can cross a bf16 rounding boundary and
    move one z^2 by ~0.8 %; 3 off calls under-sample that); its check is the exact one below, against the same call's y.
    Hard checks on every on call: no error flag; h / cnt / cnt2 back at zero whenever FIN1 off leaves them so (no lost
    or doubled finalize); every route's ss within 3e-5 of an exact FP64 recomputation from y (bound for <= 266 FP32
    roundings in any order; a lost / doubled item moves ss by ~3e-2); output rel L2 vs off <= 1e-3. Any failure turns
    FIN1 off."""
    global FIN1_ENABLE
    if not FIN1_ENABLE or not st.get("r36"):
        return
    try:
        bad = []
        for bs in (16, 32):
            r = st["r36"].get(_k2a_bucket_of(bs))
            if r is None:
                continue
            for seed, dist in ((71, "normal"), (72, "skewed"), (73, "normal")):
                d = _v14_self_test_inputs(bs, seed, dev, dist)
                data = {"hidden_states": d["hidden_states"], "topk_ids": d["topk_ids"], "topk_weights": d["topk_weights"],
                        "config": {"local_start": 0}}
                ids = d["topk_ids"]
                nr = int(((ids >= 0) & (ids < K2A_E_LOC)).sum())
                runs = []
                for on in (False, False, False, True, True):
                    FIN1_ENABLE = on
                    st["err"].zero_()
                    out = _k2a36_call(st, data, r[0], r[1]).clone()
                    torch.cuda.synchronize()
                    y = st["y"][:nr].clone()
                    ref = ((y.to(torch.bfloat16).float() * st["nw"]).double() ** 2).sum(1)
                    ss = st["ss"][:nr].double()
                    runs.append({"out": out, "inter": st["inter"].clone(), "y": y, "ss": ss,
                                 "ssx": float(((ss - ref).abs() / ref.clamp_min(1e-30)).max()) if nr else 0.0,
                                 "zero": not bool(st["h"].any() or st["cnt"].any() or st["cnt2"].any()),
                                 "err": int(st["err"][0])})
                offs, ons = runs[:3], runs[3:]
                mo = [_fin1_metrics(x, offs[0]) for x in offs[1:]]
                mn = [_fin1_metrics(x, offs[0]) for x in ons]
                env = {k: max(m[k] for m in mo) for k in FIN1_FLOOR}
                got = {k: max(m[k] for m in mn) for k in FIN1_FLOOR}
                why, rule = [], {}
                if any(x["err"] for x in runs):
                    why.append("err")
                if all(x["zero"] for x in offs) and not all(x["zero"] for x in ons):
                    why.append("zero")
                if max(x["ssx"] for x in ons) > 3e-5:
                    why.append("ss-exact")
                if got["ol2"] > 1e-3:
                    why.append("out-l2")
                for k in FIN1_FLOOR:
                    if k in ("inter", "y") and all(m[k + "_eq"] for m in mo):
                        rule[k] = "bitwise"
                        if not all(m[k + "_eq"] for m in mn):
                            why.append(k)
                    else:
                        lim = max(2 * env[k], FIN1_FLOOR[k])
                        rule[k] = f"<={lim:.2g}"
                        if got[k] > lim:
                            why.append(k)
                if why:
                    bad.append((bs, seed, "+".join(why)))
                print(f"[fin1] bs={bs} seed={seed} {dist} routes {nr}: "
                      + " | ".join(f"{k} off {env[k]:.2g} on {got[k]:.2g} ({rule[k]})" for k in FIN1_FLOOR)
                      + f" | ss off {max(m['ss'] for m in mo):.2g} on {max(m['ss'] for m in mn):.2g} (info)"
                      f" | out elems differing off {max(m['odiff'] for m in mo)} on {max(m['odiff'] for m in mn)}"
                      f" | zero off {''.join(str(int(x['zero'])) for x in offs)} on {''.join(str(int(x['zero'])) for x in ons)}"
                      f" | ss vs exact off {max(x['ssx'] for x in offs):.2e} on {max(x['ssx'] for x in ons):.2e}"
                      f" -> {'ok' if not why else '+'.join(why)}", flush=True)
        FIN1_ENABLE = not bad
        print(f"[fin1] self-test FIN1 on vs off (vid 3 T=16/32; inter / y / out vs off-vs-off envelope, bitwise where off "
              f"is; ss vs exact FP64, out rel L2): {'ok' if not bad else bad} -> FIN1_ENABLE={FIN1_ENABLE}", flush=True)
    except Exception as ex:  # noqa: BLE001
        FIN1_ENABLE = False
        print(f"[fin1] self-test raised {ex!r}: FIN1 off", flush=True)


'''
PY_GLOBAL = PY_GLOBAL + ("FIN1_K1_ONLY = False   # FIN1 in K1 only (probe bit 5; K2 keeps the shipped single-warp finalize), "
                         "read at call time\n")
PY_GLOBAL = PY_GLOBAL + ("KW3_W4 = {}   # KW3 (R-study-a-3): bucket rep -> vid 3 role-split hot_w4 override, read at call "
                         "time ({} = the plan)\n\n\n")
# KW3: the role-split weight of a v3.6 bucket, overridable at call time (st["r36"] is frozen at prepare)
PY_R36_OLD = "            return _k2a36_call(st, data, r36[0], r36[1])\n"
PY_R36_NEW = ("            hot36 = r36[1] if not KW3_W4.get(rep) else (r36[1][0], r36[1][1], int(KW3_W4[rep]))   # KW3\n"
              "            return _k2a36_call(st, data, r36[0], hot36)\n")
PY_PREP_OLD = "    _v14_k2a36_prepare(st, w, dev)\n    _v17_k2a37_prepare(st, w, dev)\n"
PY_PREP_NEW = "    _v14_k2a36_prepare(st, w, dev)\n    _v17_k2a37_prepare(st, w, dev)\n    _fin1_selftest(st, dev)\n"


def apply(text):
    a = text.index(SRC[0])
    b = text.index(SRC[1], a + len(SRC[0]))
    src = text[a:b]
    s = src.index(G36[0])
    e = src.index(G36[1], s)
    g = src[s:e]
    for name, old, new in (("finalize blocks", OLD_FIN, NEW_FIN), ("epilogue head", EPI_HEAD_OLD, EPI_HEAD_NEW),
                           ("epilogue end", EPI_END_OLD, EPI_END_NEW), ("gemm36 head", G36_HEAD_OLD, G36_HEAD_NEW),
                           ("consumer end", CONS_END_OLD, CONS_END_NEW)):
        if g.count(old) != 1:
            raise RuntimeError(f"fin1 {name}: anchor found {g.count(old)} times in gemm36")
        g = g.replace(old, new)
    src = src[:s] + g + src[e:]
    if src.count(FN_ANCHOR) != 1:
        raise RuntimeError("fin1 function anchor")
    src = src.replace(FN_ANCHOR, FIN_FN + FN_ANCHOR)
    text = text[:a] + src + text[b:]
    if text.count(PY_CALL_OLD) != 2:
        raise RuntimeError(f"fin1 call sites: {text.count(PY_CALL_OLD)}")
    text = text.replace(PY_CALL_OLD, PY_CALL_NEW)
    for name, old, new in (("global", PY_GLOBAL_ANCHOR, PY_GLOBAL + PY_GLOBAL_ANCHOR), ("prepare", PY_PREP_OLD, PY_PREP_NEW),
                           ("kw3 r36 call", PY_R36_OLD, PY_R36_NEW)):
        if text.count(old) != 1:
            raise RuntimeError(f"fin1 {name}: anchor found {text.count(old)} times")
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kernels/muse_decode_layer/k3_v36.py"
    out = apply(src.read_text(encoding="utf-8"))
    if len(sys.argv) > 2:
        pathlib.Path(sys.argv[2]).write_text(out, encoding="utf-8", newline="\n")
    print(f"fin1: patched {src.name} ({len(out)} chars)")

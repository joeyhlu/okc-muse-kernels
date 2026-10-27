"""pm8_patches.py: PM8 = persistent T <= 8 MoE (vid 0 = v3.3 NC1, cold items only): one launch replaces K1 + K2.

Plan and gates: notes/study/pm8_plan.md (R13: joint structural ceiling bs1 15.1 %, bs8 7.3 %).

Kernel pm8_kernel<NC, NST, MINB> (320 threads, MINB CTAs/SM, the v3.3 gemm reused verbatim per item):
  * whole items only: an item is one 128-row block x full K = 16 stages of one expert (a (rb, pass) pair; one pass per
    expert at T <= 8). A W13 item ends in the whole-item flush (SwiGLU -> inter, no h atomics, no finalize); a W2 item
    stores its y rows and adds its per-route sum of squares (no hand-off). gemm(cta = item, ncta = items) yields exactly
    that item's 16 stages (my_range2 on S.sb[32] = 16 x items).
  * the ring state carries across items: gemm takes it0 (stages this CTA consumed so far) so the mbarrier parities stay
    in step, and PM (template) turns off gemm's per-call MODE-1 zeroing of y / ss. Non-PM instantiations fold to the
    shipped expressions.
  * claiming: the producer warp's lane 0 takes item k (k = 0: position blockIdx.x when PM8_STATIC, else the ticket) and
    hands it to the consumers through a 4-deep smem queue (mbarrier per entry, count 1). The producer runs ahead across
    item boundaries, so the ring never drains between items.
  * list: per local expert (index order) W13 [64 x np items] and W2 [32 x np]; W2(e_k) follows W13(e_(k+L)), with
    L = PM8_LAG (0 = every W13 first, as K1 -> K2).
  * dependency: a W2 item's weight copies go out at once; its inter copies are deferred (the shipped K2 PDL deferral)
    until producer lane 0 sees cnt13[e] == 64 x np (ld.acquire.gpu, bounded by WAIT_BUDGET, err flag, no trap) followed by
    fence.proxy.async. W13 items publish with bar.sync among the 256 consumer threads + __threadfence + atomicAdd (the
    shipped split-item pattern). W13 item 0 of e zeroes e's ss rows before it publishes.
  * counters: unused cnt2 slots (j = 63: K2 never has 64 passes): ticket (0, 0), exit (0, 1), cnt13[e] = (e, 2). The last
    CTA out resets them, so every call starts from zero.
  * K3 unchanged (separate PDL-launched kernel). run3 mask bit 8 (launch3, NC 1 cold only): mask 12 = PM8 + K3.
Host: PM8_ENABLE / PM8_BUCKETS / PM8_OK (prepare self-test) read at call time; K2A's K1 + K2 path is the fallback; m > 8
never routes to PM8. Lag / static first wave travel in hot_t's upper digits (p.hot_mode, unused by vid 0's NC1 code).
Prepare self-test (_pm8_prepare): T 1 / 2 / 4 / 5 / 8, normal and skewed routings: PM8 within max(4 x K2A off-vs-off, 2e-3) of
K2A and within K2A's own FP32-reference bounds, finite, err 0, counters back at zero; register / spill / occupancy check;
fault injection (one expert's W13 counter preset to -1, i.e. a lost publish) must raise the err flag.

    python -I tools/pm8_patches.py in.py out.py     (on k3_v37's chain; k3_v37 itself stays byte-identical)
"""
import pathlib
import py_compile
import sys

REGION = ('K2A33_KERNEL_SRC = r"""', '"""\n')
GEMM_HEAD_OLD = ("template <int NC, int NST, int MODE, int HM, int STG>\n"
                 "__device__ void gemm(const Params& P, Route& S, const int16_t* s_list, uint8_t* ring, uint64_t* bars, bool pdl,\n"
                 "                     int cta, int ncta) {\n")
GEMM_HEAD_NEW = ("template <int NC, int NST, int MODE, int HM, int STG, int PMR = 0>   // PMR (PM8): 1 producer part only, 2 consumers only\n"
                 "__device__ void gemm(const Params& P, Route& S, const int16_t* s_list, uint8_t* ring, uint64_t* bars, bool pdl,\n"
                 "                     int cta, int ncta, int it0 = 0, const int* dep = nullptr, int need = 0) {   // PM8: see pm8_kernel\n")
GEMM_END = "__device__ __forceinline__ void setup_bars("
ZERO_OLD = "  if (MODE == 1 && ((P.probe & 15) == 0 || (P.probe & 15) >= 6)) {\n"
ZERO_NEW = "  if (!PMR && MODE == 1 && ((P.probe & 15) == 0 || (P.probe & 15) >= 6)) {   // PM8: whole items, ss zeroed per expert\n"
IT_OLD = "int it = 0, e = 0;"
IT_NEW = "int it = PMR ? it0 : 0, e = 0;"
DEFER_OLD = ("        if (!waited && nd > 0 && it >= NST) {   // a deferred slot is about to be reused: take the wait now\n"
             "          pdl_wait();\n")
DEFER_NEW = ("        if (!waited && nd > 0 && (PMR ? it - it0 : it) >= NST) {   // a deferred slot is about to be reused: take the wait now\n"
             "          if constexpr (PMR != 0) pm8_dep_wait(P, dep, need); else pdl_wait();\n")
END_OLD = ("    if (waited && !pdl) pdl_wait();   // no-op without a PDL predecessor; orders completion in a PDL chain\n"
           "    if (!waited) {\n"
           "      pdl_wait();\n")
END_NEW = ("    if (!PMR && waited && !pdl) pdl_wait();   // no-op without a PDL predecessor; orders completion in a PDL chain\n"
           "    if (!waited) {\n"
           "      if constexpr (PMR != 0) pm8_dep_wait(P, dep, need); else pdl_wait();\n")
ROLE8_OLD = "  if (warp == 8) {\n    // ---------------- producer ----------------\n"
ROLE8_NEW = "  if (PMR == 1 || (PMR == 0 && warp == 8)) {\n    // ---------------- producer ----------------\n"
ROLE9_OLD = "  if (warp == 9) {\n    // ---------------- epilogue warp"
ROLE9_NEW = "  if (PMR == 0 && warp == 9) {\n    // ---------------- epilogue warp"

HELPERS = r'''// PM8 (tools/pm8_patches.py): counters in unused cnt2 slots (j = 63), and the W2 dependency wait of producer lane 0.
__device__ __forceinline__ int* pm8_slot(const Params& P, int e, int k) { return P.cnt2 + (e * 32 + k) * 64 + 63; }
__device__ __forceinline__ int pm8_ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void pm8_dep_wait(const Params& P, const int* dep, int need) {
  if ((threadIdx.x & 31) == 0 && pm8_ld_acquire(dep) < need) {
    const long long t0 = clock64();
    for (;;) {
      __nanosleep(64);
      if (pm8_ld_acquire(dep) >= need) break;
      if (clock64() - t0 > WAIT_BUDGET) { atomicExch(P.err, 1); atomicExch(P.err + 2, 1); break; }
    }
  }
  __syncwarp();
}

'''

KERNEL_ANCHOR = "// P0 (v3.3): warm experts' x rows packed into wgmma's core-matrix layout"
PM8 = r'''// PM8 (tools/pm8_patches.py): persistent T <= 8 MoE, one launch for W13 + SwiGLU + W2 (cold NC1 items, whole items only).
__device__ __forceinline__ int pm8_find(const int* seg, int ns, int q) {
  int s = 0;
  while (s + 1 < ns && seg[s + 1] <= q) ++s;
  return s;
}
template <int NC, int NST, int MINB>
__global__ void __launch_bounds__(320, MINB) pm8_kernel(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = SW + 8 * NC * XROW;
  constexpr int QD = 4;
  __shared__ int16_t s_list[P3];
  __shared__ Route S13;
  __shared__ Route S2;
  __shared__ int s_seg[65], s_code[65], s_ord[32];   // list segments: first position, kind << 8 | expert
  __shared__ int s_ns, s_tot;
  __shared__ int s_qi[QD];
  __shared__ __align__(8) uint64_t s_qb[QD];
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  setup_bars<NC, NST>(bars);
  if (threadIdx.x == 0) {
    for (int i = 0; i < QD; ++i) mbar_init(sa(s_qb + i), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  pdl_wait();   // as K1 (v4.7): PDL secondary of the routing tail
  route<NC>(P, S13, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;   // (warp 9 = spare)
  if (tid < 33) S2.sb[tid] = S13.sb[tid] >> 1;   // = route(..., 32, ...)'s stage prefix (16 x nrb x passes)
  if (tid < 32) { S2.cnt[tid] = S13.cnt[tid]; S2.off[tid] = S13.off[tid]; S2.np[tid] = S13.np[tid]; S2.hp[tid] = S13.hp[tid]; }
  if (tid == 32) S2.R = S13.R;
  if (tid == 64) {
    const int L = (P.hot_mode % 100) > 0 ? (P.hot_mode % 100) : 64;
    int ne = 0;
    for (int e = 0; e < 32; ++e)
      if (S13.cnt[e] > 0) s_ord[ne++] = e;
    int pos = 0, ns = 0;
    for (int k = 0; k < ne; ++k) {
      const int e = s_ord[k];
      s_seg[ns] = pos; s_code[ns] = e; ++ns; pos += 64 * S13.np[e];
      if (k >= L) {
        const int e2 = s_ord[k - L];
        s_seg[ns] = pos; s_code[ns] = 256 | e2; ++ns; pos += 32 * S13.np[e2];
      }
    }
    for (int k = max(0, ne - L); k < ne; ++k) {
      const int e2 = s_ord[k];
      s_seg[ns] = pos; s_code[ns] = 256 | e2; ++ns; pos += 32 * S13.np[e2];
    }
    s_seg[ns] = pos;
    s_ns = ns; s_tot = pos;
  }
  __syncthreads();
  if (warp == 8) {
    // ---- producer: claim item k, queue it, issue its 16 stages (gemm's producer part only: PMR 1) ----
    int it0 = 0;
    for (int k = 0;; ++k) {
      int q = 0;
      if (lane == 0) {
        const bool stat = P.hot_mode >= 100;
        q = (k == 0 && stat) ? (int)blockIdx.x : (stat ? (int)gridDim.x : 0) + atomicAdd(pm8_slot(P, 0, 0), 1);
        s_qi[k & (QD - 1)] = q;
        mbar_arrive(sa(s_qb + (k & (QD - 1))));
      }
      q = __shfl_sync(FULL, q, 0);
      if (q >= s_tot) break;
      const int s = pm8_find(s_seg, s_ns, q), code = s_code[s], li = q - s_seg[s], e = code & 255;
      if (code < 256) gemm<NC, NST, 1, 0, STG, 1>(P, S13, s_list, smem, bars, false, (S13.sb[e] >> 4) + li, S13.sb[32] >> 4, it0);
      else gemm<NC, NST, 2, 0, STG, 1>(P, S2, s_list, smem, bars, true, (S2.sb[e] >> 4) + li, S2.sb[32] >> 4, it0,
                                       pm8_slot(P, e, 2), 64 * S13.np[e]);
      it0 += 16;
    }
  } else if (warp < 8) {
    // ---- consumers: item k from the queue, gemm's consumer part only (PMR 2), W13 publish ----
    // Only k and it0 live across gemm (register budget): the item is decoded again from the queue entry for the publish
    // (entry k & 3 is rewritten only for item k + 4, after these consumers have taken k + 1).
    int it0 = 0;
    for (int k = 0;; ++k) {
      mbar_wait(sa(s_qb + (k & (QD - 1))), (k / QD) & 1, P.err);
      {
        const int q = s_qi[k & (QD - 1)];
        if (q >= s_tot) break;
        const int s = pm8_find(s_seg, s_ns, q), code = s_code[s], li = q - s_seg[s], e = code & 255;
        if (code < 256) gemm<NC, NST, 1, 0, STG, 2>(P, S13, s_list, smem, bars, false, (S13.sb[e] >> 4) + li, S13.sb[32] >> 4, it0);
        else gemm<NC, NST, 2, 0, STG, 2>(P, S2, s_list, smem, bars, true, (S2.sb[e] >> 4) + li, S2.sb[32] >> 4, it0);
      }
      {
        const int q = s_qi[k & (QD - 1)];
        const int s = pm8_find(s_seg, s_ns, q), code = s_code[s], li = q - s_seg[s], e = code & 255;
        if (code < 256) {   // W13 publish: inter stores (and e's ss zeroing) -> barrier -> fence -> counter
          if (li == 0)
            for (int i = threadIdx.x; i < S13.cnt[e]; i += 256) P.ss[S13.off[e] + i] = 0.f;
          asm volatile("bar.sync 15, 256;\n" ::: "memory");
          if (threadIdx.x == 0) { __threadfence(); atomicAdd(pm8_slot(P, e, 2), 1); }
        }
      }
      it0 += 16;
    }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(pm8_slot(P, 0, 1), 1) == (int)gridDim.x - 1) {   // last CTA out: counters back to zero
      atomicExch(pm8_slot(P, 0, 0), 0);
      for (int e = 0; e < 32; ++e) atomicExch(pm8_slot(P, e, 2), 0);
      atomicExch(pm8_slot(P, 0, 1), 0);
    }
  }
}

'''

LAUNCH_ANCHOR = "  cfg.attrs = at;\n  cfg.numAttrs = 1;\n  if (mask & 1) {\n    if (HOTCAP) {\n"
LAUNCH_NEW = ("  cfg.attrs = at;\n  cfg.numAttrs = 1;\n"
              "  if (mask & 8) {   // PM8: one persistent launch replaces K1 + K2 (cold NC1 only); K3 below as usual\n"
              "    if constexpr (NC == 1 && !HOTCAP) {\n"
              "      cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;\n"
              "      at[0].val.programmaticStreamSerializationAllowed = (int)((p.probe >> 24) & 1);   // as K1 (v4.7)\n"
              "      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::pm8_kernel<NC, NST, MINB>, p));\n"
              "      at[0].val.programmaticStreamSerializationAllowed = 1;\n"
              "      mask &= 4;\n"
              "    } else {\n"
              "      TORCH_CHECK(false, \"k3: pm8 is built for the NC 1 cold variant only\");\n"
              "    }\n"
              "  }\n"
              "  if (mask & 1) {\n    if (HOTCAP) {\n")
ATTRS_OLD = ("  one((const void*)k3::k1_kernel<NC, NST, MINB, (bool)HOTCAP>, 320);\n"
             "  one((const void*)k3::k2_kernel<NC, NST, MINB, (bool)HOTCAP>, 320);\n"
             "  return r;\n")
ATTRS_NEW = ("  one((const void*)k3::k1_kernel<NC, NST, MINB, (bool)HOTCAP>, 320);\n"
             "  one((const void*)k3::k2_kernel<NC, NST, MINB, (bool)HOTCAP>, 320);\n"
             "  if constexpr (NC == 1 && !HOTCAP) one((const void*)k3::pm8_kernel<NC, NST, MINB>, 320);   // PM8: entries 8-11\n"
             "  return r;\n")

PY = '''

# PM8 (tools/pm8_patches.py): persistent T <= 8 MoE (vid 0), run3 mask bit 8; all switches read at call time
PM8_ENABLE = True
PM8_BUCKETS = {1: True, 8: True}
PM8_LAG = 0          # W2(e_k) after W13(e_(k+L)); 0 = every W13 first
PM8_STATIC = True    # first item of CTA b = list position b (needs every CTA co-resident: checked in prepare)
PM8_OK = {}          # bucket -> prepare self-test passed
_PM8_FORCE = None    # self-test / lab override: True = PM8 regardless of PM8_OK, False = K2A


def _pm8_on(m):
    if m > 8:
        return False
    rep = _k2a_bucket_of(m)
    if _PM8_FORCE is not None:
        return bool(_PM8_FORCE) and bool(PM8_BUCKETS.get(rep))
    return bool(PM8_ENABLE and PM8_BUCKETS.get(rep) and PM8_OK.get(rep))


def _pm8_hot(hot_t):
    return hot_t + 100000 * (int(PM8_LAG) % 100 + (100 if PM8_STATIC else 0))


def _pm8_slots(st):
    idx = [63, 64 + 63] + [(e * 32 + 2) * 64 + 63 for e in range(32)]
    return st["cnt2"][idx]


def _pm8_prepare(state, weights):
    global _PM8_FORCE
    PM8_OK.clear()
    k33 = state.get("k2a33")
    if not PM8_ENABLE or not PM8_BUCKETS or k33 is None:
        print("[pm8] off (PM8_ENABLE / PM8_BUCKETS / K2A v3.3 state)", flush=True)
        return
    st = k33["st"]
    dev = st["y"].device
    t0 = time.time()
    v0 = st["variants"][K2A33_PLAN[1]]
    a = st["mod"].attrs3v(K2A33_PLAN[1])
    if len(a) < 12 or K2A33_PLAN[1] != K2A33_PLAN[8]:
        print(f"[pm8] no pm8 attributes for vid {K2A33_PLAN[1]} ({len(a)} entries): off", flush=True)
        return
    occ_ok = a[9] == 0 and a[11] >= v0[2] and st["grid"][K2A33_PLAN[1]] <= st["sms"] * a[11]
    print(f"[pm8] pm8_kernel regs {a[8]} spill {a[9]} smem {a[10]} occ {a[11]} (grid {st['grid'][K2A33_PLAN[1]]}) -> "
          f"{'ok' if occ_ok else 'FAIL'}", flush=True)
    if not occ_ok:
        return
    w = {"w13": weights["w13"], "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"],
         "w2": weights["w2"], "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"],
         "post_norm_w": weights["post_expert_norm"]}
    ok = {}
    rel = lambda p, r: float((p - r).norm() / r.norm().clamp(min=1e-30))  # noqa: E731
    for bs, seed, dist in ((1, 41, "normal"), (2, 42, "normal"), (4, 43, "skewed"), (5, 44, "normal"), (8, 45, "normal"),
                           (8, 46, "skewed")):
        rep = _k2a_bucket_of(bs)
        if not PM8_BUCKETS.get(rep):
            continue
        good = False
        try:
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            ref = _k2a_ref_fp32(d, w)
            outs, errs, zero = [], 0, True
            for force in (False, False, True, True):
                _PM8_FORCE = force
                st["err"].zero_()
                o = _v11_k2a33_call(d["hidden_states"], d["topk_ids"], d["topk_weights"], st).clone()
                torch.cuda.synchronize()
                errs += int(st["err"][0])
                if force:
                    zero = zero and bool((_pm8_slots(st) == 0).all())
                outs.append(o.float())
            _PM8_FORCE = None
            env = rel(outs[1], outs[0])
            d_on = max(rel(outs[2], outs[0]), rel(outs[3], outs[0]))
            d_pp = rel(outs[3], outs[2])
            tot, row = _k2a_rel_err(outs[2], ref)
            fin = bool(torch.isfinite(outs[2]).all() and torch.isfinite(outs[3]).all())
            good = (errs == 0 and fin and zero and d_on <= max(4.0 * env, 2e-3) and tot < 1.0e-2 and row < 5.0e-2)
            print(f"[pm8] self-test T={bs} {dist}: pm8 vs K2A {d_on:.2e} (K2A off vs off {env:.2e}, pm8 vs pm8 {d_pp:.2e}), "
                  f"vs fp32 ref {tot:.2e} / row {row:.2e}, err {errs}, counters zero {zero}, finite {fin} -> "
                  f"{'ok' if good else 'FAIL'}", flush=True)
        except Exception as ex:
            print(f"[pm8] self-test T={bs} raised {ex!r}", flush=True)
            good = False
        finally:
            _PM8_FORCE = None
            _k2a33_reset(st)
        ok[rep] = ok.get(rep, True) and good
    # fault injection: one used expert's W13 counter starts at -1 (a lost publish): its W2 items must time out -> err
    caught = False
    try:
        d = _k2a_self_test_inputs(8, 47, dev)
        ids = d["topk_ids"]
        e0 = int(ids[(ids >= 0) & (ids < K2A_E_LOC)].min())
        st["err"].zero_()
        st["cnt2"][(e0 * 32 + 2) * 64 + 63] = -1
        _PM8_FORCE = True
        _v11_k2a33_call(d["hidden_states"], d["topk_ids"], d["topk_weights"], st)
        torch.cuda.synchronize()
        caught = int(st["err"][0]) != 0
        print(f"[pm8] fault injection (expert {e0}: one W13 publish lost) caught by the err flag: {caught}, counters "
              f"zero after {bool((_pm8_slots(st) == 0).all())}", flush=True)
    except Exception as ex:
        print(f"[pm8] fault injection raised {ex!r}", flush=True)
        caught = False
    finally:
        _PM8_FORCE = None
        st["err"].zero_()
        _k2a33_reset(st)
    if not caught:
        ok = {r: False for r in ok}
    PM8_OK.update(ok)
    print(f"[pm8] buckets {PM8_OK} (lag {PM8_LAG}, static {PM8_STATIC}) in {time.time() - t0:.1f}s", flush=True)

'''

CALL_OLD = "\n    hot_t += 1000 * (K2A33_HOT_W4 if hot_w4 is None else hot_w4)"
CALL_NEW = ("    if tuple(v)[:4] == (1, 3, 2, 0) and (mask & 3) == 3 and _pm8_on(data[\"hidden_states\"].shape[0]):   # PM8: K1+K2 -> pm8\n"
            "        mask, hot_t = (mask & 4) | 8, _pm8_hot(hot_t)\n")


def _fail(msg):
    raise RuntimeError(f"pm8 patch: {msg}")


def _sub(text, old, new, what):
    if text.count(old) != 1:
        _fail(f"{what}: anchor found {text.count(old)}x")
    return text.replace(old, new)


def _after_line(text, head, add, what):   # insert add after the (single) line that starts with head
    if text.count(head) != 1:
        _fail(f"{what}: anchor found {text.count(head)}x")
    j = text.index("\n", text.index(head) + 1) + 1
    return text[:j] + add + text[j:]


def apply(text):
    if text.count(REGION[0]) != 1:
        _fail("K2A33_KERNEL_SRC not unique")
    a = text.index(REGION[0])
    b = text.index(REGION[1], a + len(REGION[0]))
    src = text[a:b]
    if src.count(GEMM_HEAD_OLD) != 1:
        _fail("v3.3 gemm head")
    g0 = src.index(GEMM_HEAD_OLD)
    g1 = src.index(GEMM_END, g0)
    g = src[g0:g1]
    if g.count(IT_OLD) != 2:
        _fail(f"ring counters x{g.count(IT_OLD)}")
    g = _sub(g, GEMM_HEAD_OLD, GEMM_HEAD_NEW, "gemm head")
    g = _sub(g, ZERO_OLD, ZERO_NEW, "MODE-1 zeroing")
    g = _sub(g, DEFER_OLD, DEFER_NEW, "producer deferral")
    g = _sub(g, END_OLD, END_NEW, "producer end wait")
    g = _sub(g, ROLE8_OLD, ROLE8_NEW, "producer role")
    g = _sub(g, ROLE9_OLD, ROLE9_NEW, "epilogue role")
    g = g.replace(IT_OLD, IT_NEW)
    src = src[:g0] + HELPERS + g + src[g1:]
    src = _sub(src, KERNEL_ANCHOR, PM8 + KERNEL_ANCHOR, "kernel anchor")
    l0 = src.index("static void launch3(const k3::Params& p, int G, int G3, int mask) {")
    l1 = src.index("\n}\n", l0)
    src = src[:l0] + _sub(src[l0:l1], LAUNCH_ANCHOR, LAUNCH_NEW, "launch3") + src[l1:]
    r0 = src.index("static std::vector<int64_t> attrs3() {")
    r1 = src.index("\n}\n", r0) + 3
    src = src[:r0] + _sub(src[r0:r1], ATTRS_OLD, ATTRS_NEW, "attrs3") + src[r1:]
    text = text[:a] + src + text[b:]
    text = _after_line(text, CALL_OLD, CALL_NEW, "_k2a33_call")
    text = _sub(text, "\ndef kernel_fn(data, state):\n", PY + "\ndef kernel_fn(data, state):\n", "kernel_fn")
    return _after_line(text, "\n    _dg1_prepare(state, t0)",
                       "    _pm8_prepare(state, weights)   # PM8 (tools/pm8_patches.py): self-test\n", "_dg1_prepare")


if __name__ == "__main__":
    s, d = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    t = s.read_text(encoding="utf-8")
    out = apply(t)
    d.write_text(out, encoding="utf-8", newline="\n")
    py_compile.compile(str(d), doraise=True)
    print(f"pm8: applied {s.name} -> {d} ({len(out) - len(t):+d} chars, compiles)")

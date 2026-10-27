"""ss3_patches.py: SS3 = the routes' sums of squares (post-expert norm) computed in K3 instead of K2, at T <= 32.

Why (R11, box_out/nofin): K2's split-item finalize (epilogue warp: bar.sync, fences, counter atomic, then the last contributor's
dependent L2 trips computing ss) sits on the K2 -> K3 hop; removing it saves 1.2 / 1.3 / 1.4 / 0.4 us at bs 1 / 8 / 16 / 32.

What SS3 does when its probe bit 23 is set (SS3_ENABLE / SS3_BUCKETS, read at call time, and only for buckets whose prepare
self-test passed; bit 23 is vid 5's 'cold chunks only' and is read only by gemm37 / launch37, which SS3 does not touch;
SS3 also sets bit 5 = study-a's FIN1_K1_ONLY, so K2's epilogue never runs FIN1 while SS3 bypasses it; 6 = FIN1, 7 = RS1):
  * K2 (MODE 2), v3.3 gemm (vid 0) and gemm36 (vid 3): no ss work at all. Whole items skip their per-route sum-of-squares
    reduction and atomics; split items are not handed to the epilogue warp, and the epilogue warp returns at once (no
    counters, no finalize). y is unchanged (whole items stored, split items FP32 reds, as before).
  * K3 -> k3_kernel_ss, launched with a 4-CTA cluster per token (grid = 4 M, CTA rank = column quarter): each CTA loads
    its quarter of the token's 8 route rows (as K3B), sums (bf16(y) (1 + w))^2 over its 1024 columns per route (warp
    shuffles, then the 8 warps in order), publishes the 8 partials in shared memory, and after a cluster barrier adds the 4
    quarters' partials over DSMEM in rank order (the same total in all 4 CTAs), then normalizes and sums its quarter with
    k3_kernel's arithmetic; a second cluster barrier keeps the partials alive until every peer has read them.
Numerics: the same terms as K2's ss, summed in a fixed order (K2's were FP32 atomics in arbitrary order).
Prepare self-test (_ss3_prepare, envelope as study-a's: K2 y is already order-nondeterministic): per case T 1 / 5 / 8 / 12 /
16 / 24 / 32, SS3 off twice (the off-vs-off envelope) and on once; on must match an FP64 recomputation of K3 from that run's
own y / meta (rel < 5e-3), stay within max(4 x envelope, 2e-3) of off, be finite with no err flag, and the check must reject
an injected fault (ss from one quarter only, i.e. a missing DSMEM exchange: rel > 2e-2). A failing bucket stays on K2's ss.

    python -I tools/ss3_patches.py in.py out.py     (after k3p / rs1: the call-site anchors carry their bits)
"""
import pathlib
import py_compile
import sys

REGION = ('K2A33_KERNEL_SRC = r"""', '"""\n')
SS3 = "((P.probe >> 23) & 1)"

K3SS = r'''
// SS3 (tools/ss3_patches.py): K3 with the routes' sums of squares computed here. Grid = 4 M CTAs in 4-CTA clusters (one
// cluster per token, CTA rank = column quarter); the 4 quarter partials are exchanged over DSMEM and added in rank order.
__device__ __forceinline__ void ss3_cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ float ss3_peer(const float* p, int rank) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  uint32_t r;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(r) : "memory");
  return v;
}
__global__ void __launch_bounds__(256) k3_kernel_ss(const Params P) {
  __shared__ int s_fc[8];
  __shared__ float s_fw[8];
  __shared__ float s_wp[8][8];   // [warp][route]
  __shared__ float s_q[8];       // this quarter's partial per route (read by the peers)
  __shared__ float s_fr[8];
  if ((P.probe >> 25) & 1) pdl_launch();   // SU1: the staged unpooler may launch now (its weights do not depend on K3)
  pdl_wait();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int it = blockIdx.x;                 // one item per CTA: grid = 4 M
  const int tok = it >> 2;
  if (tid < 8) {
    s_fc[tid] = __ldcg(P.meta + MH + tok * 8 + tid);
    s_fw[tid] = __ldg(P.wts + tok * 8 + tid);
  }
  __syncthreads();
  const int col = (it & 3) * 1024 + 4 * tid;
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 v[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int c = s_fc[s];
    v[s] = (c >= 0) ? __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)c * KDIM + col)) : z4;
  }
  const float4 w = __ldg(reinterpret_cast<const float4*>(P.nw + col));
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    float z, q = 0.f;
    z = bf16r(v[s].x) * w.x; q += z * z;  z = bf16r(v[s].y) * w.y; q += z * z;
    z = bf16r(v[s].z) * w.z; q += z * z;  z = bf16r(v[s].w) * w.w; q += z * z;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    if (lane == 0) s_wp[warp][s] = q;
  }
  __syncthreads();
  if (tid < 8) {
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) q += s_wp[i][tid];
    s_q[tid] = q;
  }
  ss3_cluster_sync();                        // every quarter's partials are published
  if (tid < 8) {
    float ss = 0.f;
#pragma unroll
    for (int r = 0; r < 4; ++r) ss += ss3_peer(&s_q[tid], r);   // rank order: the same sum in all 4 CTAs
    s_fr[tid] = (s_fc[tid] >= 0) ? rsqrtf(ss * (1.f / 4096.f) + 1e-8f) * s_fw[tid] : 0.f;
  }
  __syncthreads();
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const float r = s_fr[s];
    a0 += bf16r(v[s].x) * w.x * r; a1 += bf16r(v[s].y) * w.y * r;
    a2 += bf16r(v[s].z) * w.z * r; a3 += bf16r(v[s].w) * w.w * r;
  }
  uint2 o;
  o.x = pack_bf2(a0, a1); o.y = pack_bf2(a2, a3);
  *reinterpret_cast<uint2*>(P.out + (size_t)tok * KDIM + col) = o;
  ss3_cluster_sync();                        // keep s_q alive until every peer has read it
}

'''

K3P_LAUNCH = ("if ((p.probe >> 31) & 1) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel_b, p));   /* K3P */ "
              "else C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel, p));")
SS3_LAUNCH = ("if ((p.probe >> 23) & 1) {   /* SS3: 4-CTA clusters, grid = 4 M */\n"
              "      cudaLaunchAttribute a2[2];\n"
              "      a2[0] = at[0];\n"
              "      a2[1].id = cudaLaunchAttributeClusterDimension;\n"
              "      a2[1].val.clusterDim.x = 4; a2[1].val.clusterDim.y = 1; a2[1].val.clusterDim.z = 1;\n"
              "      cudaLaunchConfig_t c2 = cfg;\n"
              "      c2.attrs = a2; c2.numAttrs = 2;\n"
              "      C10_CUDA_CHECK(cudaLaunchKernelEx(&c2, k3::k3_kernel_ss, p));\n"
              "    } else ")

CUDA = [
    ("v3.3 epilogue: SS3 return at MODE 2",
     "    // ---------------- epilogue warp: split-item counters; the last contributor finishes the item ----------------\n",
     "    // ---------------- epilogue warp: split-item counters; the last contributor finishes the item ----------------\n"
     f"    if (MODE == 2 && {SS3}) return;   // SS3: ss is computed in K3, nothing is handed to this warp\n", 1),
    ("gemm36 epilogue: SS3 return at MODE 2",
     "    // ---------------- epilogue warp: split-item counters; the last contributor finishes the item (v3.3) ----------------\n",
     "    // ---------------- epilogue warp: split-item counters; the last contributor finishes the item (v3.3) ----------------\n"
     f"    if (MODE == 2 && {SS3}) return;   // SS3: ss is computed in K3, nothing is handed to this warp\n", 1),
    ("v3.3 cold flush: no ss / hand-off under SS3",
     "        if (whole) {\n          // sum of squares of z = bf16(y) * (1 + w) over this item's 128 columns, per route (one red per warp)\n",
     f"        if (whole && !{SS3}) {{   // SS3: ss in K3\n"
     "          // sum of squares of z = bf16(y) * (1 + w) over this item's 128 columns, per route (one red per warp)\n", 1),
    ("v3.3 cold flush: no hand-off under SS3",
     "        } else if (fmode != 7) {\n          asm volatile(\"bar.arrive %0, 288;\\n\" :: \"r\"(2 + npart) : \"memory\");\n",
     f"        }} else if (fmode != 7 && !whole && !{SS3}) {{\n          asm volatile(\"bar.arrive %0, 288;\\n\" :: \"r\"(2 + npart) : \"memory\");\n", 1),
    ("v3.6 flush_item: no ss under SS3",
     "    if (whole) {\n      // sum of squares of z = bf16(y) * (1 + w) over this item's 128 columns, per route (one red per warp)\n",
     f"    if (whole && !{SS3}) {{   // SS3: ss in K3\n"
     "      // sum of squares of z = bf16(y) * (1 + w) over this item's 128 columns, per route (one red per warp)\n", 1),
    ("v3.6 flush_item: no hand-off under SS3",
     "    } else if (arrive) {\n",
     f"    }} else if (arrive && !whole && !{SS3}) {{\n", 1),
    ("k3_kernel_ss", "}  // namespace k3\n", K3SS + "}  // namespace k3\n", 1),
]

GRID_OLD = "_k3p_g3(st, m, _su1_g3(st, m))"
GRID_NEW = "_ss3_g3(m, _k3p_g3(st, m, _su1_g3(st, m)))"
BITS_OLD = "((1 << 7) if RS1_ENABLE else 0)"   # RS1's term, wherever FIN1 put its own bits
BITS_NEW = "((1 << 7) if RS1_ENABLE else 0) | _ss3_bits(m)"

PY = '''

# SS3 (tools/ss3_patches.py): ss computed in K3 (4-CTA clusters) at T <= 32, K2 with no ss work; probe bit 5, read at call time
SS3_ENABLE = True
SS3_BUCKETS = {1: True, 8: True, 16: True, 32: True}
SS3_OK = {}          # bucket -> prepare self-test passed
_SS3_FORCE = None    # self-test override


def _ss3_on(m):
    rep = _k2a_bucket_of(m)
    if _SS3_FORCE is not None:
        return bool(_SS3_FORCE) and bool(SS3_BUCKETS.get(rep))
    return bool(SS3_ENABLE and SS3_BUCKETS.get(rep) and SS3_OK.get(rep))


def _ss3_bits(m):
    return ((1 << 23) | (1 << 5)) if _ss3_on(m) else 0   # bit 5 = FIN1 K1-only: K2's epilogue must not run FIN1 under SS3


def _ss3_g3(m, g3):
    return 4 * m if _ss3_on(m) else g3


def _ss3_prepare(state):
    global _SS3_FORCE
    SS3_OK.clear()
    k33 = state.get("k2a33")
    if not SS3_ENABLE or not SS3_BUCKETS or k33 is None:
        print("[ss3] off (SS3_ENABLE / SS3_BUCKETS / K2A v3.3 state)", flush=True)
        return
    st = k33["st"]
    dev = st["y"].device
    t0 = time.time()
    ok = {}
    nw = st["nw"].double()
    for bs, seed in ((1, 31), (5, 32), (8, 33), (12, 34), (16, 35), (24, 36), (32, 37)):
        rep = _k2a_bucket_of(bs)
        if not SS3_BUCKETS.get(rep):
            continue
        good = False
        try:
            d = _k2a_self_test_inputs(bs, seed, dev)
            x, ids, tw = d["hidden_states"], d["topk_ids"], d["topk_weights"]
            outs, errs = [], 0
            for force in (False, False, True):
                _SS3_FORCE = force
                st["err"].zero_()
                o = _v11_k2a33_call(x, ids, tw, st).clone()
                torch.cuda.synchronize()
                errs += int(st["err"][0])
                outs.append(o.float())
                if force:
                    y = st["y"].clone()
                    meta = st["meta"][K2A_META_HDR:K2A_META_HDR + 8 * bs].clone()
            _SS3_FORCE = None
            rel = lambda a, b: float((a - b).norm() / b.norm().clamp(min=1e-30))  # noqa: E731
            env = rel(outs[1], outs[0])
            d_on = rel(outs[2], outs[0])
            c = meta.view(bs, 8).long()
            valid = (c >= 0).double()
            yb = y[c.clamp(min=0)].to(torch.bfloat16).double()
            z = yb * nw
            ss = (z * z).sum(-1)
            r = torch.rsqrt(ss / 4096.0 + 1e-8) * tw.double() * valid
            ref = (z * r.unsqueeze(-1)).sum(1)
            e_ref = rel(outs[2].double(), ref)
            sq = (z[..., :1024] * z[..., :1024]).sum(-1)          # injected fault: one quarter only (no DSMEM sum)
            rf = torch.rsqrt(sq / 4096.0 + 1e-8) * tw.double() * valid
            e_fault = rel((z * rf.unsqueeze(-1)).sum(1).to(torch.bfloat16).double(), ref)
            fin = bool(torch.isfinite(outs[2]).all())
            good = (errs == 0 and fin and e_ref < 5e-3 and d_on <= max(4.0 * env, 2e-3) and e_fault > 2e-2)
            print(f"[ss3] self-test T={bs}: on vs FP64 K3 of its own y {e_ref:.2e}, on vs off {d_on:.2e} (off vs off {env:.2e}), "
                  f"injected fault {e_fault:.2e}, err {errs}, finite {fin} -> {'ok' if good else 'FAIL'}", flush=True)
        except Exception as ex:
            print(f"[ss3] self-test T={bs} raised {ex!r}", flush=True)
            good = False
        finally:
            _SS3_FORCE = None
            _k2a33_reset(st)
        ok[rep] = ok.get(rep, True) and good
    SS3_OK.update(ok)
    print(f"[ss3] buckets {SS3_OK} in {time.time() - t0:.1f}s", flush=True)

'''


def apply(text):
    if text.count(REGION[0]) != 1:
        raise RuntimeError("ss3 patch: K2A33_KERNEL_SRC not unique")
    for name, old, new, count in CUDA:
        a = text.index(REGION[0])
        b = text.index(REGION[1], a + len(REGION[0]))
        n = text[a:b].count(old)
        if n != count:
            raise RuntimeError(f"ss3 patch {name!r}: `old` occurs {n} times in K2A33_KERNEL_SRC (want {count})")
        text = text[:a] + text[a:b].replace(old, new) + text[b:]
    for fn in ("static void launch3(", "static void launch36("):
        a = text.index(fn, text.index(REGION[0]))   # inside K2A33_KERNEL_SRC (older K2A sources have their own launch3)
        b = text.index(chr(10) + "}" + chr(10), a)
        if text[a:b].count(K3P_LAUNCH) != 1:
            raise RuntimeError(f"ss3 patch: K3 launch anchor x{text[a:b].count(K3P_LAUNCH)} in {fn}")
        text = text[:a] + text[a:b].replace(K3P_LAUNCH, SS3_LAUNCH + K3P_LAUNCH) + text[b:]
    for old, new in ((GRID_OLD, GRID_NEW), (BITS_OLD, BITS_NEW)):
        if text.count(old) != 2:
            raise RuntimeError(f"ss3 patch: call-site anchor x{text.count(old)} (want 2; apply after k3p + rs1): {old!r}")
        text = text.replace(old, new)
    anchor = "\ndef kernel_fn(data, state):\n"
    if text.count(anchor) != 1:
        raise RuntimeError("ss3 patch: kernel_fn anchor")
    text = text.replace(anchor, PY + anchor)
    i = text.index("\n    _dg1_prepare(state, t0)")
    j = text.index("\n", i + 1) + 1
    if text.count("\n    _dg1_prepare(state, t0)") != 1:
        raise RuntimeError("ss3 patch: _dg1_prepare anchor")
    return text[:j] + "    _ss3_prepare(state)   # SS3 (tools/ss3_patches.py): ss in K3 at T <= 32, self-test\n" + text[j:]


if __name__ == "__main__":
    s, d = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    t = s.read_text(encoding="utf-8")
    out = apply(t)
    d.write_text(out, encoding="utf-8", newline="\n")
    py_compile.compile(str(d), doraise=True)
    print(f"ss3: applied {s.name} -> {d} ({len(out) - len(t):+d} chars, compiles)")

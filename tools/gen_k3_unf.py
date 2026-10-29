"""UNF (fresh_eyes_oct29b s3 #2): fold the GD unpooler slice (rows [0, 1024) of W, K 4096, T <= 8) into VMS.

  py tools/gen_k3_unf.py <n> [at=1]   -> kernels/lab/k3_unf_<n>.py (base kernels/muse_decode_layer/k3_gc_18.py)

VMS_UNF (compile-time, default 0 = today's kernel): each CTA owns rows [c*N/G, (c+1)*N/G) (7-8 rows at N 1024, G 132), copies
them with cp.async.bulk into a 64.5-KB smem region (row stride 8256 B, conflict-free uint4 reads) at its first phase-B unit
(VMS_UNF_AT 1) or at entry (0); after the existing barrier + combine, the combine CTAs (c < 4m) publish their y columns with
one red.release on ctr[4] (CTA 0 adds 64 - 4m more, so every launch adds 64: target (gen + 1) * 64); every CTA polls it, reads
y (m x 8 KB, L2) and emits its f rows with mma.sync m16n8k16 (rows 0-7 = W rows, 8-15 zero; n = tokens; k permuted
consistently in A and B so both load as uint4), fp32 accumulate, 8-warp K split, one bf16 store into the zero buffer (ld 8192).
The UNF build is a second module (run_unf, own ctr / ss scratch), used only at m <= 8; the layer skips the GD unpooler when it
ran. Python switch UNF_ENABLE (module global, read at capture: rev_ab3 variants). Never overwrites a file.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels/muse_decode_layer/k3_gc_18.py"


def sub(text, old, new, count=1):
    assert text.count(old) == count, (text.count(old), old[:120])
    return text.replace(old, new)


def write_new(p, text):
    assert not p.exists(), f"{p} exists"
    assert len(text.encode()) < 1_000_000, len(text.encode())
    p.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {p} ({len(text.encode())} B)")


CU = [
    ("#ifndef VMS_G\n#define VMS_G 132\n#endif\n",
     "#ifndef VMS_G\n#define VMS_G 132\n#endif\n"
     "#ifndef VMS_UNF\n#define VMS_UNF 0   // UNF: unpooler slice rows [0, VMS_UNF_N) folded in (m <= 8), rows staged in smem\n#endif\n"
     "#ifndef VMS_UNF_N\n#define VMS_UNF_N 1024\n#endif\n"
     "#ifndef VMS_UNF_AT\n#define VMS_UNF_AT 1   // UNF weight copy issue: 0 = CTA entry (pre-wait), 1 = first phase-B unit\n#endif\n"),
    ("  __nv_bfloat16* out; long long* marks; int m; float eps; int lo;\n};\n",
     "  __nv_bfloat16* out; long long* marks; int m; float eps; int lo;\n"
     "#if VMS_UNF\n  const __nv_bfloat16* uw; __nv_bfloat16* uo; int uld;\n#endif\n};\n"),
    ("  float4 red[NRB][NW][32];\n};\n",
     "  float4 red[NRB][NW][32];\n#if VMS_UNF\n  uint64_t ufull;\n#endif\n};\n"),
    ("constexpr int SMEM_BYTES = RING_OFF + NS * UB;\n",
     "#if VMS_UNF\n"
     "constexpr int UROW = 8192 + 64, URMAX = (VMS_UNF_N + G - 1) / G, UOFF = RING_OFF + NS * UB;\n"
     "constexpr int SMEM_BYTES = UOFF + URMAX * UROW;\n"
     "static_assert(URMAX <= 8, \"unf: <= 8 rows per CTA (one m16 tile, rows 8-15 zero)\");\n"
     "#else\n"
     "constexpr int SMEM_BYTES = RING_OFF + NS * UB;\n"
     "#endif\n"),
    ("__device__ __forceinline__ uint32_t scd(uint32_t w, uint32_t sel, uint32_t k16) {\n",
     "#if VMS_UNF\n"
     "__device__ __forceinline__ void unf_issue(const Params& p, Smem& S, unsigned char* smem, int c) {   // tid 0 only\n"
     "  const int r0 = c * VMS_UNF_N / G, nr = (c + 1) * VMS_UNF_N / G - r0;\n"
     "#if __CUDA_ARCH__ >= 900\n"
     "  uint64_t pol;\n"
     "  asm volatile(\"createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\\n\" : \"=l\"(pol));\n"
     "  asm volatile(\"mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\\n\" :: \"r\"(su32(&S.ufull)), \"r\"(nr * 8192) : \"memory\");\n"
     "  for (int r = 0; r < nr; ++r)\n"
     "    asm volatile(\"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\\n\"\n"
     "                 :: \"r\"(su32(smem + UOFF + r * UROW)), \"l\"(p.uw + (size_t)(r0 + r) * HD), \"r\"(8192), \"r\"(su32(&S.ufull)),\n"
     "                    \"l\"(pol) : \"memory\");\n"
     "#else\n"
     "  (void)p; (void)smem; (void)r0; (void)nr;\n"
     "  mbar_arrive(&S.ufull);   // sm_86 test path: the rows are copied synchronously before the GEMV\n"
     "#endif\n"
     "}\n"
     "#endif\n"
     "__device__ __forceinline__ uint32_t scd(uint32_t w, uint32_t sel, uint32_t k16) {\n"),
    ("#endif\n#if __CUDA_ARCH__ >= 900\n    asm volatile(\"fence.mbarrier_init.release.cluster;\\n\" ::: \"memory\");\n#endif\n  }\n",
     "#endif\n#if VMS_UNF\n    mbar_init(&S.ufull, 1);\n#endif\n"
     "#if __CUDA_ARCH__ >= 900\n    asm volatile(\"fence.mbarrier_init.release.cluster;\\n\" ::: \"memory\");\n#endif\n"
     "#if VMS_UNF && VMS_UNF_AT == 0\n    unf_issue(p, S, smem, c);\n#endif\n  }\n"),
    ("    int curj = -1;\n", "    int curj = -1;\n#if VMS_UNF && VMS_UNF_AT == 1\n    bool uiss = false;\n#endif\n"),
    ("        if (PB) {\n          if (mk && tid == 0 && mrow[3] == 0) mrow[3] = gt();\n",
     "        if (PB) {\n#if VMS_UNF && VMS_UNF_AT == 1\n          if (tid == 0 && !uiss) { unf_issue(p, S, smem, c); uiss = true; }\n#endif\n"
     "          if (mk && tid == 0 && mrow[3] == 0) mrow[3] = gt();\n"),
    ("    if (mk && tid == 0) mrow[5] = gt();\n",
     "#if VMS_UNF && VMS_UNF_AT == 1\n    if (tid == 0 && !uiss) unf_issue(p, S, smem, c);   // no phase-B units (no local expert)\n#endif\n"
     "    if (mk && tid == 0) mrow[5] = gt();\n"),
    ("  if (mk && tid == 0) mrow[9] = gt();\n#if VMS_MK2\n  if (mk && tid == 0) mrow[17] = clock64();\n#endif\n}\n",
     "#if VMS_UNF\n"
     "  __syncthreads();\n"
     "  if (tid == 0) {\n"
     "    if (c < p.m * 4) {   // combine CTA (one item each: m * 4 <= 32 < G): publish its y columns\n"
     "      asm volatile(\"fence.acq_rel.gpu;\\n\" ::: \"memory\");\n"
     "      const int inc = 1 + (c == 0 ? 64 - p.m * 4 : 0);\n"
     "      asm volatile(\"red.release.gpu.global.add.s32 [%0], %1;\\n\" :: \"l\"(p.ctr + 4), \"r\"(inc) : \"memory\");\n"
     "    }\n"
     "    if (mk) mrow[9] = gt();\n"
     "    const int utarget = (gen + 1) * 64;\n"
     "    int n = 0;\n"
     "    while (ld_acq(p.ctr + 4) < utarget) {\n"
     "      if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 3); break; }\n"
     "      __nanosleep(20);\n"
     "    }\n"
     "    if (mk) mrow[10] = gt();\n"
     "  }\n"
     "  mbar_wait(&S.ufull, 0);\n"
     "  __syncthreads();\n"
     "  if (mk && tid == 0) mrow[11] = gt();\n"
     "  if (warp < NW) {\n"
     "    const int r0 = c * VMS_UNF_N / G, nr = (c + 1) * VMS_UNF_N / G - r0;\n"
     "    const int g = lane >> 2, t = lane & 3, kb = warp * (HD / NW);\n"
     "    unsigned char* ws = smem + UOFF;\n"
     "#if __CUDA_ARCH__ < 900\n"
     "    for (int i = tid; i < nr * 512; i += NT) {\n"
     "      const int r = i >> 9, cc = i & 511;\n"
     "      *reinterpret_cast<uint4*>(ws + r * UROW + cc * 16) = __ldg(reinterpret_cast<const uint4*>(p.uw + (size_t)(r0 + r) * HD) + cc);\n"
     "    }\n"
     "    cta_sync_compute();\n"
     "#endif\n"
     "    uint4 yv[16];\n"
     "    const uint4* ysrc = reinterpret_cast<const uint4*>(p.out + (size_t)g * HD + kb + 8 * t);\n"
     "#pragma unroll\n"
     "    for (int b = 0; b < 16; ++b) yv[b] = g < p.m ? __ldcg(ysrc + b * 4) : make_uint4(0u, 0u, 0u, 0u);\n"
     "    float acc[4][4];\n"
     "#pragma unroll\n"
     "    for (int s = 0; s < 4; ++s) { acc[s][0] = 0.f; acc[s][1] = 0.f; acc[s][2] = 0.f; acc[s][3] = 0.f; }\n"
     "    const unsigned char* wr = ws + g * UROW + (kb + 8 * t) * 2;\n"
     "    const bool rv = g < nr;\n"
     "#pragma unroll\n"
     "    for (int b = 0; b < 16; ++b) {   // lane (g, t): logical k {2t, 2t+1, 2t+8, 2t+9} = physical 8t + {0,1,2,3} | {4,5,6,7}\n"
     "      const uint4 w = rv ? *reinterpret_cast<const uint4*>(wr + b * 64) : make_uint4(0u, 0u, 0u, 0u);\n"
     "      mma(acc[(2 * b) & 3], w.x, 0u, w.y, 0u, yv[b].x, yv[b].y);\n"
     "      mma(acc[(2 * b + 1) & 3], w.z, 0u, w.w, 0u, yv[b].z, yv[b].w);\n"
     "    }\n"
     "    float2* rd = reinterpret_cast<float2*>(&S.red[0][0][0]);\n"
     "    rd[warp * 32 + lane] = make_float2((acc[0][0] + acc[1][0]) + (acc[2][0] + acc[3][0]),\n"
     "                                       (acc[0][1] + acc[1][1]) + (acc[2][1] + acc[3][1]));\n"
     "    cta_sync_compute();\n"
     "    if (tid < 64) {   // (row, token): C rows g = W rows, cols 2t / 2t + 1 = tokens\n"
     "      const int row = tid >> 3, tok = tid & 7, ln = row * 4 + (tok >> 1);\n"
     "      float s = 0.f;\n"
     "#pragma unroll\n"
     "      for (int w = 0; w < NW; ++w) { const float2 v = rd[w * 32 + ln]; s += (tok & 1) ? v.y : v.x; }\n"
     "      if (row < nr && tok < p.m) p.uo[(size_t)tok * p.uld + r0 + row] = __float2bfloat16(s);\n"
     "    }\n"
     "  }\n"
     "  if (mk && tid == 0) mrow[12] = gt();\n"
     "#else\n"
     "  if (mk && tid == 0) mrow[9] = gt();\n"
     "#endif\n"
     "#if VMS_MK2\n  if (mk && tid == 0) mrow[17] = clock64();\n#endif\n}\n"),
]

RUN_SIG = ("void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,\n"
           "         torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,\n"
           "         torch::Tensor out, torch::Tensor marks, int64_t m, double eps, int64_t pdl, int64_t lo) {\n")
UNF_SIG = RUN_SIG.replace("void run(", "void run_unf(").replace("int64_t lo) {\n",
                                                               "int64_t lo,\n             torch::Tensor uw, torch::Tensor uo, int64_t uld) {\n")
UNF_DECL = ("void run_unf(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,\n"
            "         torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,\n"
            "         torch::Tensor out, torch::Tensor marks, int64_t m, double eps, int64_t pdl, int64_t lo,\n"
            "         torch::Tensor uw, torch::Tensor uo, int64_t uld);\n")


def patch_cuda(cu):
    for old, new in CU:
        cu = sub(cu, old, new)
    a = cu.index(RUN_SIG)
    b = cu.index("std::vector<int64_t> attrs() {", a)
    body = cu[a:b]
    assert body.count('TORCH_CHECK(m >= 1 && m <= 16, "vms: 1 <= m <= 16");') == 1
    run = body.replace('TORCH_CHECK(m >= 1 && m <= 16, "vms: 1 <= m <= 16");',
                       'TORCH_CHECK(m >= 1 && m <= 16, "vms: 1 <= m <= 16");\n  TORCH_CHECK(!VMS_UNF, "vms: UNF build: use run_unf");')
    unf = body.replace(RUN_SIG, UNF_SIG)
    unf = unf.replace('TORCH_CHECK(m >= 1 && m <= 16, "vms: 1 <= m <= 16");',
                      'TORCH_CHECK(m >= 1 && m <= 8, "vms unf: 1 <= m <= 8");\n'
                      '  TORCH_CHECK(uw.is_contiguous() && uw.size(0) >= VMS_UNF_N && uw.size(1) == vms::HD && uo.size(0) >= m && uld >= VMS_UNF_N,\n'
                      '              "vms unf: uw [>= N, 4096] contiguous, uo [>= m, uld]");')
    assert unf.count("p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo;\n") == 1
    unf = unf.replace("p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo;\n",
                      "p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo;\n"
                      "  p.uw = reinterpret_cast<const __nv_bfloat16*>(uw.data_ptr()); p.uo = reinterpret_cast<__nv_bfloat16*>(uo.data_ptr());\n"
                      "  p.uld = (int)uld;\n")
    return cu[:a] + run + "#if VMS_UNF\n" + unf + "#endif\n\n" + cu[b:]


PY = [
    # build + prepare + call
    ("VMS_ENABLE = True\nVMS_MAXT = 16\n",
     "VMS_ENABLE = True\nVMS_MAXT = 16\n"
     "UNF_ENABLE = True   # UNF (tools/gen_k3_unf.py): VMS emits the T <= 8 unpooler slice itself (second build, run_unf)\n"
     "UNF_AT = __UNF_AT__\n"
     "UNF_SELF_TEST_CASES = ((1, 61, \"normal\"), (1, 62, \"skewed\"), (3, 63, \"normal\"), (8, 64, \"normal\"), (8, 65, \"skewed\"))\n"
     "UNF_CPP_SRC = VMS_CPP_SRC + r\"\"\"\n" + UNF_DECL + "\"\"\"\n"
     "_UNF = {\"mod\": None}\n"
     "_UNF_LAST = [None]\n"),
    ("def _vms_pack_words(cu):\n",
     "def _unf_build():\n"
     "    from torch.utils.cpp_extension import load_inline\n"
     "    G = torch.cuda.get_device_properties(0).multi_processor_count\n"
     "    flags = [\"-O3\", f\"-DVMS_G={G}\", \"-DVMS_NS=3\", \"-DVMS_RO=1\", \"-DVMS_UNF=1\", f\"-DVMS_UNF_N={UNPOOL_SLICE}\",\n"
     "             f\"-DVMS_UNF_AT={UNF_AT}\", \"-Xptxas=-warn-spills\", \"-gencode=arch=compute_90a,code=sm_90a\"]\n"
     "    tag = hashlib.sha1((VMS_CUDA_SRC + UNF_CPP_SRC + \" \".join(flags)).encode()).hexdigest()[:10]\n"
     "    t1 = time.time()\n"
     "    mod = load_inline(name=f\"k3_vmsu_{tag}\", cpp_sources=UNF_CPP_SRC, cuda_sources=VMS_CUDA_SRC,\n"
     "                      functions=[\"run\", \"run_unf\", \"attrs\"], extra_cuda_cflags=flags)\n"
     "    _UNF[\"mod\"] = mod\n"
     "    _K3_BUILD_S[\"vmsu\"] = time.time() - t1\n"
     "    return mod\n"
     "\n"
     "\n"
     "def _unf_prepare(state, t0):\n"
     "    if not (UNF_ENABLE and VMS_ENABLE):\n"
     "        return\n"
     "    vs = None\n"
     "    try:\n"
     "        _pt1_wait(\"vmsu\")\n"
     "        k33 = state.get(\"k2a33\")\n"
     "        st = k33[\"st\"] if k33 is not None else None\n"
     "        vs = st.get(\"vms\") if st is not None else None\n"
     "        zb = state.get(\"v12_zbuf\")\n"
     "        w = state[\"dense\"][\"unpooler\"]\n"
     "        if not (st is not None and st.get(\"vms_ok\") and vs is not None and zb is not None and V12_ZBUF and R1_UNPOOL_INTO_ZEROS\n"
     "                and zb.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[1] == D_EXPERT\n"
     "                and w.shape[0] >= UNPOOL_SLICE and w.is_contiguous() and zb.shape[1] == HIDDEN):\n"
     "            print(\"[unf] prerequisites missing (VMS / zero buffer / bf16 unpooler): UNF OFF\", flush=True)\n"
     "            return\n"
     "        mod = _UNF[\"mod\"] if _UNF[\"mod\"] is not None else _unf_build()\n"
     "        dev = zb.device\n"
     "        u = {\"mod\": mod, \"w\": w[:UNPOOL_SLICE], \"zb\": zb, \"ctr\": torch.zeros((72,), dtype=torch.int32, device=dev),\n"
     "             \"ss\": torch.zeros((256,), dtype=torch.float32, device=dev)}\n"
     "        good, worst, gdw = True, 0.0, -1.0\n"
     "        for bs, seed, dist in UNF_SELF_TEST_CASES:\n"
     "            d = _k2a_self_test_inputs(bs, seed, dev, dist)\n"
     "            x = d[\"hidden_states\"].contiguous()\n"
     "            vs.pop(\"unf\", None)\n"
     "            yb = _vms_call(st, x, d[\"topk_ids\"], d[\"topk_weights\"]).clone()\n"
     "            torch.cuda.synchronize()\n"
     "            vs[\"unf\"] = u\n"
     "            for k in range(2):\n"
     "                _UNF_LAST[0] = None\n"
     "                y = _vms_call(st, x, d[\"topk_ids\"], d[\"topk_weights\"])\n"
     "                torch.cuda.synchronize()\n"
     "                f = _UNF_LAST[0]\n"
     "                ry, _ = _k2a_rel_err(y, yb)   # the ss atomics make VMS itself non-bitwise run to run\n"
     "                ref = y.float() @ u[\"w\"].float().t()\n"
     "                ok = f is not None and tuple(f.shape) == (bs, HIDDEN) and ry < 1e-3\n"
     "                rel = float((f[:, :UNPOOL_SLICE].float() - ref).norm() / ref.norm().clamp_min(1e-30)) if ok else 1.0\n"
     "                errf, c0, c4 = int(u[\"ctr\"][2]), int(u[\"ctr\"][0]), int(u[\"ctr\"][4])\n"
     "                ok = ok and rel < 4e-3 and errf == 0 and c4 == 64 * c0 and bool(torch.isfinite(f.float()).all())\n"
     "                ok = ok and not bool(zb[:, UNPOOL_SLICE:].any())\n"
     "                if ok and k == 0 and _gd_live(state, \"unpooler\", bs) is not None:\n"
     "                    gz = torch.zeros((bs, HIDDEN), dtype=torch.bfloat16, device=dev)\n"
     "                    g = _gd(state, \"unpooler\", y, gz, HIDDEN)\n"
     "                    torch.cuda.synchronize()\n"
     "                    if g is not None:\n"
     "                        gdw = max(gdw, float((f[:, :UNPOOL_SLICE].float() - g[:, :UNPOOL_SLICE].float()).abs().max()))\n"
     "                worst = max(worst, rel)\n"
     "                if not ok or k == 0:\n"
     "                    print(f\"[unf] self-test bs={bs} {dist} call {k}: y vs VMS {ry:.1e} rel f vs FP32(y) \"\n"
     "                          f\"{rel:.2e} err={errf} ctr {c4}/{64 * c0} -> {'ok' if ok else 'FAIL'}\", flush=True)\n"
     "                good &= ok\n"
     "        _UNF_LAST[0] = None\n"
     "        if good:\n"
     "            vs[\"unf\"] = u\n"
     "        else:\n"
     "            vs.pop(\"unf\", None)\n"
     "        print(f\"[unf] UNF {'LIVE at T <= ' + str(SLICED_UNPOOL_MAX_TOKENS) if good else 'OFF: self-test failed'} (AT {UNF_AT}, \"\n"
     "              f\"regs/local/smem/G/NS {list(mod.attrs())}, worst rel {worst:.2e}, max |f - GD| {gdw:.3g}, \"\n"
     "              f\"{time.time() - t0:.1f}s into prepare)\", flush=True)\n"
     "    except Exception as ex:\n"
     "        if vs is not None:\n"
     "            vs.pop(\"unf\", None)\n"
     "        _UNF_LAST[0] = None\n"
     "        print(f\"[unf] prepare raised {ex!r}: UNF OFF\", flush=True)\n"
     "\n"
     "\n"
     "def _vms_pack_words(cu):\n"),
    ("    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)\n"
     "    vs[\"mod\"].run(x, ids, tw, vs[\"wA\"], vs[\"wB\"],",
     "    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)\n"
     "    u = vs.get(\"unf\")\n"
     "    if UNF_ENABLE and u is not None and m <= SLICED_UNPOOL_MAX_TOKENS:\n"
     "        u[\"mod\"].run_unf(x, ids, tw, vs[\"wA\"], vs[\"wB\"], vs[\"gsA\"], vs[\"gsB\"], vs[\"nw\"], vs[\"inter\"], vs[\"ybuf\"], u[\"ss\"],\n"
     "                         u[\"ctr\"], out, vs[\"nomarks\"], m, K2A_POST_NORM_EPS, 1, 0, u[\"w\"], u[\"zb\"], HIDDEN)\n"
     "        _UNF_LAST[0] = u[\"zb\"][:m]\n"
     "        return out\n"
     "    vs[\"mod\"].run(x, ids, tw, vs[\"wA\"], vs[\"wB\"],"),
    ("    if VMS_ENABLE:\n        jobs.append((\"vms\", _vms_build))\n",
     "    if VMS_ENABLE:\n        jobs.append((\"vms\", _vms_build))\n"
     "    if VMS_ENABLE and UNF_ENABLE:\n        jobs.append((\"vmsu\", _unf_build))\n"),
    ("    _vfa_prepare(state, t0)\n    torch.cuda.synchronize()\n    _pt1_join_all()\n",
     "    _vfa_prepare(state, t0)\n    _unf_prepare(state, t0)\n    torch.cuda.synchronize()\n    _pt1_join_all()\n"),
    # forward: take the folded unpooler output
    ("    _SU1_FLAG[\"k3trig\"] = _su1_k3trig(state, tokens)\n",
     "    _UNF_LAST[0] = None\n    _SU1_FLAG[\"k3trig\"] = _su1_k3trig(state, tokens)\n"),
    ("    ffn_output = _gd_unpool(state, expert_output, tokens)\n    if ffn_output is None:\n",
     "    ffn_output = _UNF_LAST[0] if (_UNF_LAST[0] is not None and _UNF_LAST[0].shape[0] == tokens) else None\n"
     "    _UNF_LAST[0] = None\n"
     "    if ffn_output is None:\n"
     "        ffn_output = _gd_unpool(state, expert_output, tokens)\n"
     "    if ffn_output is None:\n"),
]


def gen(at=1):
    text = SRC.read_text(encoding="utf-8")
    a = text.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    b = text.index('"""', a)
    text = text[:a] + patch_cuda(text[a:b]) + text[b:]
    for old, new in PY:
        text = sub(text, old, new.replace("__UNF_AT__", str(at)))
    ast.parse(text)
    return text


def main():
    n = int(sys.argv[1])
    at = int(dict(x.split("=", 1) for x in sys.argv[2:] if "=" in x).get("at", 1))
    write_new(ROOT / f"kernels/lab/k3_unf_{n}.py", gen(at))


if __name__ == "__main__":
    main()

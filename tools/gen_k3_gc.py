"""gen_k3_gc.py: k3_gc_<n> = a layer file (default k3_mb_1) + GC, the ONE-LAUNCH GD cat at T = 9..32.

Today the T = 16 cat (pooler + router hi / lo rows, FP32 out) is P1M (judge row ~20 us) and the T = 32 cat is DG1 (BF16 TMA
stream, judge row ~25.8 us, standalone 23-24), while the GD engine streams the P12 (12-bit lossless) pooler alone in 12.9 /
14.7 us net on the judge (proj board vd_1 log, m = 16 / 32). R-gd-3 (2026-10-27) tried GD pooler + a second GD router launch
and kept only 15-40 % of the standalone saving: the second launch's PDL handoff (~2 us) ate it. GC removes the second launch:

* one gd_kernel launch of the pooler geometry (MT 2 / 4, WR 1, WK 8, 256 CTAs, 2 per SM) whose warps stream KU = 16 + 2
  units: their 16 pooler units, then 2 units of ONE virtual router tile. Virtual router tile t (0..31) = rows 0-7 hi of experts
  8t..8t+7, rows 8-15 their lo rows (GD's per-byte window base `tb` already encodes two bases: bytes 0/1 rows g, 2/3 rows g+8).
  CTA c streams tile c / 8, K slice c % 8 (16 of the 128 k-units: warp wk takes 2). The pooler accumulators are parked in
  shared memory after unit 15 and reused for the router units, so registers do not grow.
* epilogue: the pooler tile is stored as today (bf16, SGLang's precision at T > 8); the router partial of this K slice is
  logit(8 t + gg) = acc rows gg (hi) + gg + 8 (lo), stored FP32 into part[8][T][256]. The routing tail (v8_tail, unchanged)
  already sums <= 8 FP32 partial slices in a fixed order (the T <= 8 FP32 router path) and takes the bf16 pooler output.
* bytes: pooler 50.3 MB + router 6.3 MB P12 (one stream, no second ramp, no FP32 pooler write-back of the cat).
* prepare: pack (per-element window bases), gd_unpack_cat bit-exact check against cat(pooler, virtual router), self-test
  vs FP64 (pooler bf16 as GD; logits vs FP64 of hi + lo and vs the FP32 router) with PDL 0 / 3 / 3 bitwise repeat; any
  failure prints FALLBACK and today's cat path runs. GD_CAT (per bucket) is read at call time (A/B arms).

Output goes through gen_k3_m32.finish (strip, compile, ASCII, nonce, checker scan, < 1 MB). Refuses to overwrite.

    python -I tools/gen_k3_gc.py [--base=k3_mb_1] [--n=1]
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32 as GM  # noqa: E402
sys.argv = _argv

N = ARGS.get("n", "1")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_mb_1')}.py"
OUT = ROOT / f"kernels/muse_decode_layer/k3_gc_{N}.py"
LAB = ROOT / f"kernels/lab/k3_gc_{N}_ab.py"
TAG = f"k3-gc-{N}"
NL = chr(10)

# ---------------------------------------------------------------- CUDA (inside GD_CUDA_SRC) -------------------------------
CUDA_PATCHES = [
    # Args: router partial output and the router window bases (default-initialised: gd_run_raw / gdr_run_raw leave them)
    ("  const float* x32; const float* w32; float* part; long long* marks;   // VF (vfr_patches): router rows / marks\n};",
     "  const float* x32; const float* w32; float* part; long long* marks;   // VF (vfr_patches): router rows / marks\n"
     "  void* out2 = nullptr; int ldo2 = 0; long long sst2 = 0; uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n"
     "};",
     "Args: GC fields"),
    # kernel template: CAT router units per warp (0 = today)
    ("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0>\n"
     "__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(const __grid_constant__ Args A) {\n"
     "  constexpr int NW = WR * WK, NT = NW * 32, NP = 3;\n",
     "#define GC_XP(S_) (((S_) < KUP) ? (xb + (S_) * 8) : (xbr + ((S_) - KUP) * 8))\n"
     "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0>\n"
     "__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(const __grid_constant__ Args A) {\n"
     "  constexpr int NW = WR * WK, NT = NW * 32, NP = 3;\n"
     "  constexpr int KUP = KU - CAT;   // GC: pooler units per warp; units KUP.. are the virtual router tile's\n"
     "  static_assert(CAT == 0 || (S == 1 && WR == 1 && OUT == 0 && CAT == 2), \"GC: cat routes are S 1, WR 1, bf16 pooler, 2 units\");\n",
     "kernel template CAT"),
    ("  __shared__ float red[NW][MT * 4][32];\n",
     "  __shared__ float red[NW][MT * 4][32];\n"
     "  __shared__ float redr[CAT ? NW : 1][CAT ? MT * 4 : 1][32];   // GC: the router partials of the CTA's warps\n",
     "redr smem"),
    ("  const int u0 = (split * WK + wk) * KU;\n", "  const int u0 = (split * WK + wk) * KUP;\n", "u0 uses KUP"),
    ("  const uint4* xb = reinterpret_cast<const uint4*>(A.x + (size_t)u0 * 64 + 16 * t);\n",
     "  const uint4* xb = reinterpret_cast<const uint4*>(A.x + (size_t)u0 * 64 + 16 * t);\n"
     "  const uint4* xbr = reinterpret_cast<const uint4*>(A.x + (size_t)(((int)blockIdx.x & 7) * 16 + 2 * wk) * 64 + 16 * t);   // GC\n",
     "xbr"),
    ("          X[(s + 1) & (XB - 1)][mt][0] = ldx(xb + xo[mt] + (s + 1) * 8);\n"
     "          X[(s + 1) & (XB - 1)][mt][1] = ldx(xb + xo[mt] + (s + 1) * 8 + 1);\n",
     "          X[(s + 1) & (XB - 1)][mt][0] = ldx(GC_XP(s + 1) + xo[mt]);\n"
     "          X[(s + 1) & (XB - 1)][mt][1] = ldx(GC_XP(s + 1) + xo[mt] + 1);\n",
     "x prefetch (XP) address"),
    ("        X[0][mt][0] = ldx(xb + xo[mt] + s * 8);\n"
     "        X[0][mt][1] = ldx(xb + xo[mt] + s * 8 + 1);\n",
     "        X[0][mt][0] = ldx(GC_XP(s) + xo[mt]);\n"
     "        X[0][mt][1] = ldx(GC_XP(s) + xo[mt] + 1);\n",
     "x load address"),
    ("    decode(R[slot], A.tb, a);\n", "    decode(R[slot], (CAT && s >= KUP) ? A.tbr : A.tb, a);\n", "decode base per step"),
    # end of the unit loop: park the pooler partials after unit KUP - 1; the final dump goes to redr for cat routes
    ("        mma16816(acc[q % NC][mt], a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], b0, b1);\n"
     "      }\n"
     "    }\n"
     "  }\n"
     "\n"
     "#pragma unroll\n"
     "  for (int mt = 0; mt < MT; ++mt)\n"
     "#pragma unroll\n"
     "    for (int i = 0; i < 4; ++i) {\n"
     "      float v = acc[0][mt][i];\n"
     "      if constexpr (NC == 2) v += acc[1][mt][i];\n"
     "      red[warp][mt * 4 + i][lane] = v;\n"
     "    }\n"
     "  __syncthreads();\n",
     "        mma16816(acc[q % NC][mt], a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], b0, b1);\n"
     "      }\n"
     "    }\n"
     "    if constexpr (CAT > 0) {\n"
     "      if (s == KUP - 1) {   // GC: pooler tile done; park its partials, reuse the accumulators for the router units\n"
     "#pragma unroll\n"
     "        for (int mt = 0; mt < MT; ++mt)\n"
     "#pragma unroll\n"
     "          for (int i = 0; i < 4; ++i) {\n"
     "            float v = acc[0][mt][i];\n"
     "            if constexpr (NC == 2) v += acc[1][mt][i];\n"
     "            red[warp][mt * 4 + i][lane] = v;\n"
     "            acc[0][mt][i] = 0.f;\n"
     "            if constexpr (NC == 2) acc[1][mt][i] = 0.f;\n"
     "          }\n"
     "      }\n"
     "    }\n"
     "  }\n"
     "\n"
     "#pragma unroll\n"
     "  for (int mt = 0; mt < MT; ++mt)\n"
     "#pragma unroll\n"
     "    for (int i = 0; i < 4; ++i) {\n"
     "      float v = acc[0][mt][i];\n"
     "      if constexpr (NC == 2) v += acc[1][mt][i];\n"
     "      if constexpr (CAT > 0) redr[warp][mt * 4 + i][lane] = v; else red[warp][mt * 4 + i][lane] = v;\n"
     "    }\n"
     "  __syncthreads();\n",
     "unit loop: park pooler partials, final dump"),
    ("    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n",
     "    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n"
     "    if constexpr (CAT > 0) {   // GC: this K slice's router partial: logit(8 t + gg) = tile rows gg (hi) + gg + 8 (lo)\n"
     "      const int gc_t = (int)blockIdx.x >> 3, gc_ks = (int)blockIdx.x & 7;\n"
     "      for (int i = (int)threadIdx.x; i < NO / 2; i += NT) {\n"
     "        const int l = i & 31, gg = l >> 2, tt = l & 3;\n"
     "        const int c = (i >> 5) & 1, mt = i >> 6;\n"
     "        float v = 0.f;\n"
     "#pragma unroll\n"
     "        for (int k2 = 0; k2 < WK; ++k2) v += redr[k2][mt * 4 + c][l] + redr[k2][mt * 4 + c + 2][l];\n"
     "        const int tok = tok0 + mt * 8 + 2 * tt + c;\n"
     "        if (tok < A.M)\n"
     "          reinterpret_cast<float*>(A.out2)[(size_t)gc_ks * A.sst2 + (size_t)tok * A.ldo2 + gc_t * 8 + gg] = v;\n"
     "      }\n"
     "    }\n",
     "router partial store"),
    # launch / info templates take CAT (VF stays 0 there)
    ("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP>\n"
     "static int launch(const Args& A, int nbx, int nz, int pdl, cudaStream_t st) {\n",
     "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0>\n"
     "static int launch(const Args& A, int nbx, int nz, int pdl, cudaStream_t st) {\n",
     "launch template CAT"),
    ("  return (int)cudaLaunchKernelEx(&cfg, gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP>, A);\n",
     "  return (int)cudaLaunchKernelEx(&cfg, gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>, A);\n",
     "launch instantiation CAT"),
    ("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP>\n"
     "static int info(int64_t* res) {\n",
     "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0>\n"
     "static int info(int64_t* res) {\n",
     "info template CAT"),
    ("  auto fn = gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP>;\n",
     "  auto fn = gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>;\n",
     "info instantiation CAT"),
    # the cat unpack kernel (bit-exact check of the packed stream against cat(pooler, virtual router))
    ("}  // namespace gd\n",
     "__global__ void unpack_cat_kernel(const uint4* __restrict__ w, const uint32_t* __restrict__ esc, uint32_t* __restrict__ out,\n"
     "                                  int K, int G, int WK, int KUP, uint32_t tb, uint32_t tbr) {\n"
     "  const int gw = (int)blockIdx.x, s = (int)blockIdx.y, lane = (int)threadIdx.x;\n"
     "  const int g = lane >> 2, t = lane & 3;\n"
     "  const int bx = gw / WK, wk = gw % WK;\n"
     "  int rt, u;\n"
     "  uint32_t tbs;\n"
     "  if (s < KUP) { rt = bx; u = wk * KUP + s; tbs = tb; }\n"
     "  else { rt = (G / WK) + (bx >> 3); u = (bx & 7) * 16 + 2 * wk + (s - KUP); tbs = tbr; }   // virtual router tile bx / 8\n"
     "  uint4 p[3];\n"
     "#pragma unroll\n"
     "  for (int i = 0; i < 3; ++i) p[i] = w[(((size_t)s * G + gw) * 3 + i) * 32 + lane];\n"
     "  uint32_t a[16];\n"
     "  decode(p, tbs, a);\n"
     "  for (int j = 0; j < 32; ++j) {\n"
     "    const uint32_t e = esc[(size_t)gw * 32 + j];\n"
     "    if ((int)(e >> 20) == s) patch(a, e, lane);\n"
     "  }\n"
     "#pragma unroll\n"
     "  for (int q = 0; q < 4; ++q)\n"
     "#pragma unroll\n"
     "    for (int i = 0; i < 4; ++i) {\n"
     "      const int row = rt * 16 + g + 8 * (i & 1);\n"
     "      const int k = u * 64 + 16 * t + 4 * q + 2 * (i >> 1);\n"
     "      out[((size_t)row * K + k) >> 1] = a[4 * q + i];\n"
     "    }\n"
     "}\n"
     "#undef GC_XP\n"
     "\n"
     "}  // namespace gd\n",
     "unpack_cat_kernel"),
    # variadic instance macros (9- and 10-tuple keys)
    ("#define GD_INST(ID, OUT, MT, WR, WK, KU, ST, S, MB, XP) \\\n"
     "  case ID: return launch<OUT, MT, WR, WK, KU, ST, S, MB, XP>(A, nbx, nz, pdl, st);\n"
     "#define GD_INFO(ID, OUT, MT, WR, WK, KU, ST, S, MB, XP) \\\n"
     "  case ID: return info<OUT, MT, WR, WK, KU, ST, S, MB, XP>(res);\n",
     "#define GD_INST(ID, ...) \\\n"
     "  case ID: return launch<__VA_ARGS__>(A, nbx, nz, pdl, st);\n"
     "#define GD_INFO(ID, ...) \\\n"
     "  case ID: return info<__VA_ARGS__>(res);\n",
     "variadic GD_INST / GD_INFO"),
]

# gdc_run_raw + gd_unpack_cat_raw (host side of GD_CUDA_SRC, after gd_run_raw's switch); anchored on gd_run_raw's start
CUDA_HOST_ADD = r'''
int gdc_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, void* out2, int M, int K, int ldo,
                int ldo2, uint32_t tb, uint32_t tbr, int nbx, int pdl, cudaStream_t st) {
  Args A;
  A.w = reinterpret_cast<const uint4*>(w);
  A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.x = reinterpret_cast<const uint16_t*>(x);
  A.out = out;
  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;
  A.trig = (pdl >> 1) & 1;
  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;
  A.out2 = out2; A.ldo2 = ldo2; A.sst2 = (long long)M * ldo2; A.tbr = tbr;
  const int nz = 1;
  switch (inst) {
@INST@
    default: return -1;
  }
}

int gd_unpack_cat_raw(const void* w, const void* esc, void* out, int K, int G, int wk, int kup, uint32_t tb, uint32_t tbr,
                      cudaStream_t st) {
  unpack_cat_kernel<<<dim3((unsigned)G, (unsigned)(kup + 2)), 32, 0, st>>>(
      reinterpret_cast<const uint4*>(w), reinterpret_cast<const uint32_t*>(esc), reinterpret_cast<uint32_t*>(out), K, G, wk,
      kup, tb, tbr);
  return (int)cudaGetLastError();
}

'''

CPP_ADD = r'''int gdc_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, void* out2, int M, int K, int ldo,
                int ldo2, uint32_t tb, uint32_t tbr, int nbx, int pdl, cudaStream_t st);
int gd_unpack_cat_raw(const void* w, const void* esc, void* out, int K, int G, int wk, int kup, uint32_t tb, uint32_t tbr,
                      cudaStream_t st);
void gdc_run(int64_t inst, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, torch::Tensor out2,
             int64_t M, int64_t K, int64_t ldo, int64_t ldo2, int64_t tb, int64_t tbr, int64_t nbx, int64_t pdl) {
  TORCH_CHECK(out2.scalar_type() == at::kFloat && out2.is_contiguous() && out2.dim() == 3 && out2.size(0) == 8 &&
              out2.size(1) == M && out2.size(2) == ldo2, "gdc_run: router partials must be a contiguous [8, M, ldo2] f32 tensor");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == M, "gdc_run: pooler output");
  const int e = gdc_run_raw(inst, w.data_ptr(), esc.data_ptr(), x.data_ptr(), out.data_ptr(), out2.data_ptr(), (int)M, (int)K,
                            (int)ldo, (int)ldo2, (uint32_t)tb, (uint32_t)tbr, (int)nbx, (int)pdl, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gdc_run: inst ", inst, " CUDA error ", e, " ", cudaGetErrorString((cudaError_t)(e < 0 ? 1 : e)));
}
void gd_unpack_cat(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t wk, int64_t kup,
                   int64_t tb, int64_t tbr) {
  const int e = gd_unpack_cat_raw(w.data_ptr(), esc.data_ptr(), out.data_ptr(), (int)K, (int)G, (int)wk, (int)kup, (uint32_t)tb,
                                  (uint32_t)tbr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gd_unpack_cat: CUDA error ", e);
}
'''

# ---------------------------------------------------------------- Python ------------------------------------------------
PY_BLOCK = r'''

# ======== GC: the one-launch GD cat at T = 9..32 (tools/gen_k3_gc.py). GD_CAT is read at call time (A/B arms); the routes are
# built and self-tested in prepare whatever GD_CAT says. Geometry (mt, wr, wk, ku, st, s, xp): ku = 16 pooler + 2 router units
# per warp (the pooler's judge-measured (1, 8, 16, 1) shapes at m = 16 / 32), 256 CTAs = 32 virtual router tiles x 8 K slices.
GD_CAT = {16: True, 32: True}
GD_CAT_ROUTE = {16: (2, 1, 8, 18, 2, 1, 0), 32: (4, 1, 8, 18, 1, 1, 0)}
GD_CAT_UNITS = 2
GD_CAT_PDL = 3
GD_CAT_SELF_TEST_T = {16: (16, 9), 32: (32, 17)}
GD_CAT_TOL = (1e-5, 1e-4)   # logits rel vs FP64 of (hi + lo) and vs the FP32 router; pooler bf16 uses GD_TOL
_GD_CAT_SEEN = set()


def _gd_cat_key(rt):
    mt, wr, wk, ku, st_, s, xp = rt
    if wr != 1 or s != 1 or mt not in (2, 4) or wk * (ku - GD_CAT_UNITS) != HIDDEN // 64 or st_ > ku or xp not in (0, 1):
        return None
    if (2 * N_EXPERTS) // 16 * 8 != D_EXPERT // 16 or wk * GD_CAT_UNITS != 16:
        return None   # 32 virtual router tiles x 8 K slices must equal the pooler's 256 row tiles; 16 router units per CTA
    nt = wk * 32
    ctas = D_EXPERT // 16
    return (0, mt, wr, wk, ku, st_, s, max(1, min(2048 // nt, -(-ctas // GD_SMS))), xp, GD_CAT_UNITS)


def _gd_cat_keys():
    keys = []
    for b, rt in sorted(GD_CAT_ROUTE.items()):
        key = _gd_cat_key(rt) if rt else None
        if key is not None and key not in keys:
            keys.append(key)
    return keys


def _gd_cat_virtual(hi, lo):
    """virtual router matrix [512, K]: tile t = rows 0-7 hi of experts 8t..8t+7, rows 8-15 their lo rows"""
    nt = N_EXPERTS // 8
    return torch.stack((hi.view(nt, 8, HIDDEN), lo.view(nt, 8, HIDDEN)), dim=1).reshape(2 * N_EXPERTS, HIDDEN).contiguous()


def _gd_cat_pack(wp, wv, wk, kup, base_p, bh, bl):
    """one stream [step][gw][plane][lane]: steps < kup = warp wk's pooler units (wk * kup + s) of row tile bx; steps kup, kup + 1
    = units (bx % 8) * 16 + 2 wk + {0, 1} of virtual router tile bx / 8 (window base bh for rows g, bl for rows g + 8)."""
    n, k = wp.shape
    nbx = n // 16
    G = nbx * wk
    dev = wp.device
    frp = _gd_frag(wp).view(nbx, wk, kup, 32, 4, 8).permute(2, 0, 1, 3, 4, 5)            # [kup][nbx][wk][lane][q][j]
    frv = _gd_frag(wv)                                                                   # [32 tiles][128 units][32][4][8]
    bxs = torch.arange(nbx, device=dev)
    tiles = (bxs // 8)[:, None].expand(nbx, wk)
    uu = (bxs % 8)[:, None] * 16 + 2 * torch.arange(wk, device=dev)[None, :]
    frr = torch.stack([frv[tiles, uu + j] for j in range(GD_CAT_UNITS)], dim=0)         # [2][nbx][wk][32][4][8]
    fr = torch.cat([frp, frr], dim=0).contiguous()                                       # [ku][nbx][wk][32][4][8]
    ku = kup + GD_CAT_UNITS
    half = ((torch.arange(8, device=dev) >> 1) & 1).view(1, 1, 1, 1, 1, 8)
    base = torch.full(fr.shape, base_p, dtype=torch.int32, device=dev)
    base[kup:] = torch.where(half.bool(), torch.full_like(half, bl), torch.full_like(half, bh)).to(torch.int32)
    lo = fr & 0xFF
    h7 = (fr >> 8) & 0x7F
    sg = (fr >> 15) & 1
    d = h7 - base
    isesc = (d < 0) | (d > 7)
    d = torch.where(isesc, torch.zeros_like(d), d)
    dA, dB, sA, sB = d[..., :4], d[..., 4:], sg[..., :4], sg[..., 4:]
    sBr = sB[..., [3, 0, 1, 2]]
    byte = (dA | (sBr << 3) | (dB << 4) | (sA << 7)).to(torch.int64)
    cw = _gd_i32(byte[..., 0] | (byte[..., 1] << 8) | (byte[..., 2] << 16) | (byte[..., 3] << 24))
    lo64 = lo.to(torch.int64)
    l0 = _gd_i32(lo64[..., 0] | (lo64[..., 1] << 8) | (lo64[..., 2] << 16) | (lo64[..., 3] << 24))
    l1 = _gd_i32(lo64[..., 4] | (lo64[..., 5] << 8) | (lo64[..., 6] << 16) | (lo64[..., 7] << 24))
    p0 = torch.stack([l0[..., 0], l1[..., 0], l0[..., 1], l1[..., 1]], dim=-1)
    p1 = torch.stack([l0[..., 2], l1[..., 2], l0[..., 3], l1[..., 3]], dim=-1)
    units = torch.stack([p0, p1, cw], dim=3)                                             # [ku][nbx][wk][3][32][4]
    pos = isesc.nonzero()
    nesc = int(pos.shape[0])
    esc = torch.full((G * 32,), -1, dtype=torch.int32, device=dev)
    if nesc:
        s_, bx, wi, ln, qq, jv = (pos[:, i] for i in range(6))
        gw = bx * wk + wi
        x7 = (h7[s_, bx, wi, ln, qq, jv] ^ base[s_, bx, wi, ln, qq, jv]).to(torch.int64)
        ent = (s_ << 20) | (ln << 15) | ((4 * qq + (jv >> 1)) << 11) | ((jv & 1) << 10) | x7
        order = torch.argsort((gw * 4096 + s_) * 16 + 4 * qq + (jv >> 1))
        gws, ents = gw[order], ent[order]
        cnt = torch.bincount(gws, minlength=G)
        if int(cnt.max()) > GD_ESC_CAP:
            raise ValueError(f"escape cap: a warp has {int(cnt.max())} > {GD_ESC_CAP} escapes")
        rank = torch.arange(nesc, device=dev) - (torch.cumsum(cnt, 0) - cnt)[gws]
        esc[gws * 32 + rank] = _gd_i32(ents)
    tbr = bh | (bh << 8) | (bl << 16) | (bl << 24)
    return {"w": units.contiguous().view(-1), "esc": esc, "tb": base_p * 0x01010101, "tbr": tbr, "G": G, "nbx": nbx,
            "nesc": nesc, "ku": ku}


def _gd_cat_run(r, x, y, part, T, pdl):
    r["mod"].gdc_run(r["inst"], r["w"], r["esc"], x, y, part, T, HIDDEN, D_EXPERT, N_EXPERTS, r["tb"], r["tbr"], r["nbx"], pdl)


def _gd_cat_selftest(r, wp, hi, lo, w32, b):
    dev = wp.device
    parts = []
    ref_w = hi.double() + lo.double()
    for T in GD_CAT_SELF_TEST_T[b]:
        for outl in (False, True):
            x, probes = _gd_test_input(T, HIDDEN, 977 + 31 * T + int(outl), dev, outl)
            outs = []
            for pdl in GD_SELF_TEST_PDL:
                y = torch.full((T, D_EXPERT), float("nan"), dtype=torch.bfloat16, device=dev)
                part = torch.full((8, T, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
                _gd_cat_run(r, x, y, part, T, pdl)
                outs.append((y, part))
            torch.cuda.synchronize()
            y, part = outs[0]
            if not (bool(torch.isfinite(y).all()) and bool(torch.isfinite(part).all())):
                return False, f"T={T} outl={outl}: non-finite"
            refp = (x.double() @ wp.double().T).to(torch.bfloat16).double()
            dd = y.double() - refp
            rn = refp.norm(dim=-1)
            tot = float(dd.norm() / refp.norm().clamp(min=1e-30))
            row = float((dd.norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-30)).max())
            pr = float((dd[probes].norm(dim=-1) / rn[probes].clamp(min=1e-30)).max()) if bool(probes.any()) else 0.0
            lg = part.double().sum(0)
            refl = x.double() @ ref_w.T
            ref32 = x.double() @ w32.double().T
            el = float((lg - refl).norm() / refl.norm().clamp(min=1e-30))
            e32 = float((lg - ref32).norm() / ref32.norm().clamp(min=1e-30))
            rep = all(bool(torch.equal(y, o[0])) and bool(torch.equal(part, o[1])) for o in outs[1:])
            if not (tot <= GD_TOL[0] and row <= GD_TOL[1] and pr == 0.0 and el <= GD_CAT_TOL[0] and e32 <= GD_CAT_TOL[1] and rep):
                return False, (f"T={T} outl={outl}: pooler rel {tot:.1e} row {row:.1e} probe {pr:.1e}; logits rel {el:.1e} "
                               f"(vs fp32 router {e32:.1e}) repeat {rep}")
            parts.append(f"T={T}{'o' if outl else ''} {tot:.1e}/{el:.1e}")
    return True, " ".join(parts)


def _gd_cat_prepare(state, t0):
    state["gd_cat"] = None
    st, v8 = state.get("gd"), state.get("v8")
    if st is None or v8 is None or not GD_CAT_ROUTE:
        print(f"[gc] FALLBACK: GD {'live' if st else 'not live'}, v8 {'live' if v8 else 'not live'} -> today's cat", flush=True)
        return None
    t1 = time.time()
    mod = st["mod"]
    ids = {k: i for i, k in enumerate(_GD["keys"])}
    dev = v8["wcat"].device
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    wp = v8["wcat"][:D_EXPERT]
    hi = v8["wcat"][D_EXPERT:D_EXPERT + N_EXPERTS]
    lo = v8["wcat"][D_EXPERT + N_EXPERTS:D_EXPERT + 2 * N_EXPERTS]
    w32 = state["router"]
    try:
        wv = _gd_cat_virtual(hi, lo)
        base_p, bh, bl = _gd_window_base(wp), _gd_window_base(hi), _gd_window_base(lo)
        ref = torch.cat((wp, wv), dim=0)
    except Exception as ex:
        print(f"[gc] FALLBACK: virtual router / bases failed: {ex!r} -> today's cat", flush=True)
        return None
    packs, routes, live = {}, {}, []
    for b, rt in sorted(GD_CAT_ROUTE.items()):
        why = None
        key = _gd_cat_key(rt) if rt else None
        if key is None or key not in ids:
            why = "geometry invalid / not built"
        if why is None:
            info = mod.gd_info(ids[key])
            ctas = D_EXPERT // 16
            if info[1] > 64 or (GD_ONE_WAVE and info[2] * sms < ctas):
                why = f"regs {info[0]} spill {info[1]} occ {info[2]} for {ctas} CTAs (> one wave or spills)"
        if why is None:
            kup = rt[3] - GD_CAT_UNITS
            pk_key = (rt[2], kup)
            if pk_key not in packs:
                try:
                    pk = _gd_cat_pack(wp, wv, rt[2], kup, base_p, bh, bl)
                    un = torch.empty_like(ref)
                    mod.gd_unpack_cat(pk["w"], pk["esc"], un, HIDDEN, pk["G"], rt[2], kup, pk["tb"], pk["tbr"])
                    torch.cuda.synchronize()
                    exact = bool(torch.equal(un.view(torch.int16), ref.view(torch.int16)))
                    print(f"[gc] pack cat ({rt[2]},{kup}+{GD_CAT_UNITS}): {pk['w'].numel() * 4 / 1e6:.1f} MB, bases {base_p}/{bh}/{bl}, "
                          f"{pk['nesc']} escapes, bit-exact unpack {exact}", flush=True)
                    packs[pk_key] = pk if exact else None
                    del un
                except Exception as ex:
                    print(f"[gc] pack cat ({rt[2]},{kup}) failed: {ex!r}", flush=True)
                    packs[pk_key] = None
            pk = packs[pk_key]
            if pk is None:
                why = "pack / bit-exact unpack failed"
        if why is None:
            desc = f"gc m{rt[0]}({rt[1]},{rt[2]},{rt[3]},{rt[5]})s{rt[4]}x{rt[6]}r{info[0]}"
            r = {"mod": mod, "inst": ids[key], "w": pk["w"], "esc": pk["esc"], "tb": pk["tb"], "tbr": pk["tbr"], "nbx": pk["nbx"],
                 "T": 8 * rt[0], "desc": desc}
            try:
                ok, msg = _gd_cat_selftest(r, wp, hi, lo, w32, b)
            except Exception as ex:
                ok, msg = False, f"EXC {ex!r}"
            if not ok:
                why = f"self-test: {msg}"
            else:
                routes[b] = r
                live.append(f"cat/{b} {desc}")
                print(f"[gc] self-test cat/{b} {desc}: ok ({msg})", flush=True)
        if why is not None:
            print(f"[gc] FALLBACK cat/{b}: {why} -> today's cat", flush=True)
    torch.cuda.empty_cache()
    state["gd_cat"] = routes or None
    print(f"[gc] live ({time.time() - t1:.1f}s): {'; '.join(live) or 'none'}; GD_CAT {GD_CAT}", flush=True)
    return None


def _gd_cat_live(state, tokens):
    st = state.get("gd_cat")
    if not GD_ENABLE or st is None or state.get("gd_cat_ok") is False:
        return None
    b = _gd_bucket(tokens)
    if b is None or not GD_CAT.get(b):
        return None
    r = st.get(b)
    return r if r is not None and tokens <= r["T"] else None


def _gd_cat(state, x, data, tokens):
    r = _gd_cat_live(state, tokens)
    if r is None or not (x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[0] == tokens and x.shape[1] == HIDDEN
                         and x.is_contiguous() and x.data_ptr() % 16 == 0):
        return None
    try:
        y = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=x.device)
        part = torch.empty((8, tokens, N_EXPERTS), dtype=torch.float32, device=x.device)
        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)
        if tokens not in _GD_CAT_SEEN:
            _GD_CAT_SEEN.add(tokens)
            print(f"[gc] first call T={tokens}: {r['desc']} pdl {GD_CAT_PDL}", flush=True)
        return _v8_tail(y, part, state, data)
    except Exception as ex:
        state["gd_cat_ok"] = False
        print(f"[gc] call raised ({ex!r}): today's cat from here on", flush=True)
        return None
'''

PY_PATCHES = [
    # instance list: the cat keys join the GD build (one module, one PT1 job)
    ("def _gd_keys():\n"
     "    keys = []\n"
     "    for (proj, b), rt in sorted(GD_ROUTE.items(), key=str):\n"
     "        key = _gd_key(*_gd_nk(proj, b), rt) if rt else None\n"
     "        if key is not None and key not in keys:\n"
     "            keys.append(key)\n"
     "    return keys\n",
     "def _gd_keys():\n"
     "    keys = []\n"
     "    for (proj, b), rt in sorted(GD_ROUTE.items(), key=str):\n"
     "        key = _gd_key(*_gd_nk(proj, b), rt) if rt else None\n"
     "        if key is not None and key not in keys:\n"
     "            keys.append(key)\n"
     "    for key in _gd_cat_keys():   # GC\n"
     "        if key not in keys:\n"
     "            keys.append(key)\n"
     "    return keys\n",
     "_gd_keys adds the cat instances"),
    ('functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info"]',
     'functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat"]',
     "GD module exports"),
    ("    _gd_prepare(state, t0)\n    _vfr_prepare(state)\n",
     "    _gd_prepare(state, t0)\n    _gd_cat_prepare(state, t0)\n    _vfr_prepare(state)\n",
     "prepare hook"),
    ('    elif tail == "cat":\n        y32 = _v20_p1m(state, "cat", ffn_input)\n',
     '    elif tail == "cat" and (gc := _gd_cat(state, ffn_input, data, tokens)) is not None:\n'
     '        expert_x, topk_weights, topk_ids, local_ids = gc\n'
     '    elif tail == "cat":\n        y32 = _v20_p1m(state, "cat", ffn_input)\n',
     "kernel_fn: GC before the P1M / DG1 / torch cat"),
    # the Python block goes right after GD's _gd_k3trig (end of the GD python section, before VFA)
    ("\n\n\nVFA_ENABLE = True\n", PY_BLOCK + "\n\n\nVFA_ENABLE = True\n", "GC python block"),
]


def main():
    GM.check(not OUT.exists(), f"{OUT.name} exists; pick another --n")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    nonce = (f"# nonce {TAG} ({BASE.stem} + GC: one-launch GD cat at T = 9..32, P12 pooler + virtual hi/lo router tiles in one "
             "gd_kernel launch, router K-slice partials summed by the routing tail; GD_CAT read at call time)")
    t = b.replace(bn, nonce)
    # CUDA patches live inside GD_CUDA_SRC: anchor on the GD block to keep them away from look-alikes (gv / p1m sources)
    for a, b2, what in CUDA_PATCHES:
        GM.check(t.count(a) == 1, f"anchor for {what}: found {t.count(a)}")
        t = t.replace(a, b2)
        rep.append(what)
    # host additions: gdc_run_raw after gd_run_raw's definition (second occurrence of the signature = the definition)
    sig = ("int gd_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo, uint32_t tb,\n"
           "               int nbx, int nz, int pdl, cudaStream_t st) {\n")
    GM.check(t.count(sig) == 1, f"gd_run_raw definition: found {t.count(sig)}")
    marker = "\n\nstatic constexpr int GDR_DS = 2 * 8 * 256 * 16;"
    GM.check(t.count(marker) == 1, f"GDR_DS marker: found {t.count(marker)}")
    t = t.replace(marker, "\n" + CUDA_HOST_ADD + "static constexpr int GDR_DS = 2 * 8 * 256 * 16;")
    rep.append("gdc_run_raw + gd_unpack_cat_raw")
    cpp_anchor = ("void gd_unpack(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t nbx, int64_t wr,\n"
                  "               int64_t wk, int64_t ku, int64_t tb) {\n")
    GM.check(t.count(cpp_anchor) == 1, f"gd_unpack binding: found {t.count(cpp_anchor)}")
    t = t.replace(cpp_anchor, CPP_ADD + cpp_anchor)
    rep.append("CPP bindings gdc_run / gd_unpack_cat")
    for a, b2, what in PY_PATCHES:
        GM.check(t.count(a) == 1, f"anchor for {what}: found {t.count(a)}")
        t = t.replace(a, b2)
        rep.append(what)
    GM.check(t.count("@INST@") == 3, "two @INST@ switch bodies (gd_run_raw, gdc_run_raw) + the Python replace call")
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_gc] OK  {r}")
    OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_gc] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, headroom "
          f"{1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

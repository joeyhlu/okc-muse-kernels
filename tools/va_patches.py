"""va_patches.py: VA1 = flat warp-specialized decode attention for full layers (q_len 1, 2 <= B <= 256), notes/study/va_attn.md.

Ideas from vjeux/okc-2026's flat q1 kernel (read, not copied): producer warps own every K/V copy (whole 128-B rows) into an
mbarrier ring, 4 consumer warps x 32 keys run K1's own tile_compute1 numerics, one equal-key partition over all requests
(SEGK virtual keys per request boundary), each split adds o*2^m / l*2^m into an fp32 workspace with red.add, a key-count
ticket elects the finalizer (rmsnorm * sigmoid(gate)), which zeroes what it read.

Size-lean integration (base headroom ~12 KB): VA1 lives in K1's CUDA source and host entry. k1_forward(stages = 200 + ST)
launches va1_main<ST> with K1's Params (part_o / part_lse / counters = the o / l workspace and key tickets, min_chunk = SEGK).
prepare() self-tests VA1 against the torch reference and K1 (twice: workspace reset) and falls back to K1 loudly;
VA1_ENABLE is read at call time (one file serves both A/B arms).

    python -I tools/va_patches.py [--n=1] [--check]   # base k3_vms_1k -> kernels/muse_decode_layer/k3_va_<n>.py + lab copy
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

K1_MAIN = "template <int L, int ST>\n__global__ void __launch_bounds__(NTHR, (L == 1) ? 2 : K1_L8_MINB) k1_main(const Params p) {"

VA1_CUDA = r'''// VA1 (tools/va_patches.py): part_o / part_lse / counters = o / l workspace + key tickets (zero between calls),
// min_chunk = SEGK, grid (G, Hkv / 2), warps 0-3 consume (KV head warp & 1, keys 32 (warp >> 1) ..), warps 4-7 load.
__device__ __forceinline__ uint32_t va_sa(const void* x) { return (uint32_t)__cvta_generic_to_shared(x); }
__device__ __forceinline__ void va_wait(uint64_t* b, int par) {
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra W;\n}\n" ::"r"(va_sa(b)), "r"(par) : "memory");
#else
  uint32_t ok = 0;
  do asm volatile("{\n.reg .pred P;\nmbarrier.test_wait.parity.shared.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}\n"
                  : "=r"(ok) : "r"(va_sa(b)), "r"(par) : "memory");
  while (!ok);
#endif
}
__device__ __forceinline__ void va_red4(float* a, float f, float x, float y, float z, float w) {
#if __CUDA_ARCH__ >= 900
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(a), "f"(x * f), "f"(y * f), "f"(z * f), "f"(w * f) : "memory");
#else
  atomicAdd(a, x * f); atomicAdd(a + 1, y * f); atomicAdd(a + 2, z * f); atomicAdd(a + 3, w * f);
#endif
}
#define VA_CSYNC asm volatile("bar.sync 1, 128;\n" ::: "memory")
template <int ST> constexpr int va_smem() { return ST * STAGE_BYTES + NEW_BYTES + 2048 + 16 * ST + 4 * 1032; }

// Request b holds virtual keys [V_b, V_b + n_b), n_b = seq_b + 1 (cached keys, then the new one), and a dead gap of
// SEGK keys after it (a segment restart's cost); CTA c owns [cT/G, (c+1)T/G). Both roles walk the same segment list.
template <int ST>
__global__ void __launch_bounds__(256, 2) va1_main(const Params p) {
#if K1_ARCH_OK
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* newk = smem + ST * STAGE_BYTES;
  uint8_t* newv = newk + NEWK * ROWB;
  uint8_t* qs = newk + NEW_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(qs + 2048);
  uint64_t* empt = full + ST;
  int* flag = reinterpret_cast<int*>(empt + ST);
  int* sV = flag + 4;
  int* sN = sV + 513;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int hp = blockIdx.y, HP = gridDim.y, B = p.B, segk = p.min_chunk;
  reinterpret_cast<uint4*>(newk)[tid] = make_uint4(0u, 0u, 0u, 0u);
  if (tid < ST) {
    asm volatile("mbarrier.init.shared.b64 [%0], 128;\n" ::"r"(va_sa(full + tid)) : "memory");
    asm volatile("mbarrier.init.shared.b64 [%0], 4;\n" ::"r"(va_sa(empt + tid)) : "memory");
  }
  if (warp == 0) {
    int carry = 0;
    for (int b0 = 0; b0 < B; b0 += 32) {
      const int b = b0 + lane;
      const int n = (b < B) ? __ldg(p.seq_lens + b) + 1 : 0, w = (b < B) ? n + segk : 0;
      int incl = w;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int v = __shfl_up_sync(FULL, incl, d);
        if (lane >= d) incl += v;
      }
      if (b < B) { sV[b] = carry + incl - w; sN[b] = n; }
      carry += __shfl_sync(FULL, incl, 31);
    }
    if (lane == 0) sV[B] = carry;
  }
  __syncthreads();
#if __CUDA_ARCH__ >= 900
  if (p.use_rope) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");   // use_rope = PDL: o proj may start
#endif
  const long long T = (long long)sV[B] - segk;
  const int v0 = (int)((T * blockIdx.x) / gridDim.x), v1 = (int)((T * (blockIdx.x + 1)) / gridDim.x);
  int bf = 0;
  for (int hi = B - 1; bf < hi;) {
    const int mid = (bf + hi) >> 1;
    if (sV[mid] + sN[mid] > v0) hi = mid;
    else bf = mid + 1;
  }
  const size_t rowB = (size_t)p.Hkv * 64;
  int n = 0;   // ring tile counter, continuous across segments
  if (warp >= 4) {
    const int q = tid - 128, ch = q & 7, r0 = q >> 3, psh = p.page_shift, pm = (1 << psh) - 1;
    const uint8_t* kc = p.k_cache + (size_t)hp * 128 + 16 * ch;
    const uint8_t* vc = p.v_cache + (size_t)hp * 128 + 16 * ch;
    for (int b = bf; b < B && sV[b] < v1; ++b) {
      const int k0 = max(v0 - sV[b], 0), k1 = min(v1 - sV[b], sN[b]), cend = min(k1, sN[b] - 1);
      if (k1 <= k0 || cend <= k0) continue;
      const int nt = (cend - k0 + TILE - 1) / TILE;
      const int* pt = p.page_table + (size_t)b * p.pt_stride;
      int pg[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) pg[k] = (k0 + r0 + 16 * k < cend) ? __ldg(pt + ((k0 + r0 + 16 * k) >> psh)) : -1;
      for (int x = 0; x < nt; ++x, ++n) {
        int pgn[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int pos = k0 + (x + 1) * TILE + r0 + 16 * k;
          pgn[k] = (pos < cend) ? __ldg(pt + (pos >> psh)) : -1;
        }
        const int st = n % ST;
        if (n >= ST) va_wait(empt + st, ((n / ST) - 1) & 1);
        uint8_t* stg = smem + st * STAGE_BYTES;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int r = r0 + 16 * k, pos = k0 + x * TILE + r;
          const size_t go = pg[k] >= 0 ? (size_t)((pg[k] << psh) | (pos & pm)) * rowB : (size_t)0;
          cp_async16(stg + r * ROWB + 16 * kswz<1>(r, ch), kc + go, pg[k] >= 0 ? 16 : 0);
          cp_async16(stg + TILE * ROWB + r * ROWB + 16 * vswz<1>(r, ch), vc + go, pg[k] >= 0 ? 16 : 0);
          pg[k] = pgn[k];
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(va_sa(full + st)) : "memory");
      }
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    return;
  }
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");   // PDL launches: qkv is written by the previous kernel
#endif
  const int h = warp & 1, kw = 32 * (warp >> 1), g = lane >> 2, t = lane & 3, nc = max(1, p.W);
  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale), cl2 = p.qk_sm * ksc * LOG2E;
  const size_t stride = (size_t)(2 * p.Hq + 2 * p.Hkv) * 64;
  for (int b = bf; b < B && sV[b] < v1; ++b) {
    const int k0 = max(v0 - sV[b], 0), k1 = min(v1 - sV[b], sN[b]);
    if (k1 <= k0) continue;
    const int seq = sN[b] - 1, cend = min(k1, seq);
    const int nt = (cend > k0) ? (cend - k0 + TILE - 1) / TILE : 0;
    const __nv_bfloat16* qrow = p.qkv + (size_t)b * stride;
#pragma unroll
    for (int k = 0; k < 4; ++k) {                    // Q row R = warp + 4k = KV head hh x query head gi, fp16
      const int R = warp + 4 * k;
      float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qrow + (size_t)((2 * hp + (R >> 3)) * 8 + (R & 7)) * 64 + 2 * lane));
      const float rs = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f) + 1e-5f);
      reinterpret_cast<uint32_t*>(qs)[R * 32 + lane] = pack_h2(x.x * rs, x.y * rs);
    }
    if (k1 == sN[b]) {                               // the new token: warp = kind (K, V) x KV head
      const int kind = warp >> 1, hh = warp & 1, head = 2 * hp + hh;
      float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qrow + (size_t)(p.Hq + kind * p.Hkv + head) * 64 + 2 * lane));
      if (kind == 0) {
        const float rs = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f) + 1e-5f);
        x.x *= rs; x.y *= rs;
      }
      const float sc = kind ? vsc : ksc;
      const uint16_t q8 = (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(x.x / sc, x.y / sc), __NV_SATFINITE, __NV_E4M3);
      *reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + (size_t)p.cache_loc[b] * rowB + (size_t)head * 64 + 2 * lane) = q8;
      const int bb = 64 * hh + 2 * lane;
      *reinterpret_cast<uint16_t*>((kind ? newv : newk) + 16 * (kind ? vswz<1>(0, bb >> 4) : kswz<1>(0, bb >> 4)) + (bb & 15)) = q8;
    }
    VA_CSYNC;
    uint32_t qf[4][4];
    const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs) + (h * 8 + g) * 32;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      qf[s][0] = qa[8 * t + 2 * s]; qf[s][2] = qa[8 * t + 2 * s + 1]; qf[s][1] = 0u; qf[s][3] = 0u;
    }
    float o[8][4];
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) { o[jn][0] = 0.f; o[jn][1] = 0.f; o[jn][2] = 0.f; o[jn][3] = 0.f; }
    float mA = neg_inf(), lA = 0.f;
    for (int i = 0; i < nt; ++i, ++n) {
      const int st = n % ST, ts = k0 + i * TILE;
      va_wait(full + st, (n / ST) & 1);
      const uint8_t* Kb = smem + st * STAGE_BYTES;
      tile_compute1<4>(Kb, Kb + TILE * ROWB, kw, h, g, t, qf, o, mA, lA, cl2, ts + TILE > cend, ts, cend, seq, 0);
      __syncwarp();
      if (lane == 0) asm volatile("{\n.reg .b64 s;\nmbarrier.arrive.shared.b64 s, [%0];\n}\n" ::"r"(va_sa(empt + st)) : "memory");
    }
    if (k1 == sN[b] && kw == 0) tile_compute1<2>(newk, newv, 0, h, g, t, qf, o, mA, lA, cl2, true, seq, seq + 1, seq, 0);
    lA += __shfl_xor_sync(FULL, lA, 1);
    lA += __shfl_xor_sync(FULL, lA, 2);
    const float f = (mA == neg_inf()) ? 0.f : ex2(fminf(mA - P_OFFSET, 126.f));   // to the absolute frame 2^0
    const size_t r0 = (size_t)(b * HP + hp) * nc * 16, rb = r0 + (blockIdx.x % nc) * 16;   // nc copies (W = nc)
    float* wa = p.part_o + (rb + h * 8 + g) * 64 + 16 * t;   // dims 16t + 8e + jn
    va_red4(wa, f, o[0][0], o[1][0], o[2][0], o[3][0]);
    va_red4(wa + 4, f, o[4][0], o[5][0], o[6][0], o[7][0]);
    va_red4(wa + 8, f, o[0][1], o[1][1], o[2][1], o[3][1]);
    va_red4(wa + 12, f, o[4][1], o[5][1], o[6][1], o[7][1]);
    if (t == 0) atomicAdd(p.part_lse + rb + h * 8 + g, lA * f);
    __threadfence();
    VA_CSYNC;
    if (tid == 0) flag[0] = (atomicAdd(p.counters + b * HP + hp, k1 - k0) + k1 - k0 == sN[b]) ? 1 : 0;
    VA_CSYNC;
    if (flag[0]) {                                   // last share of (b, hp): finalize, zero what was read
      __threadfence();
      const int r = tid >> 3, u = tid & 7, qh = (2 * hp + (r >> 3)) * 8 + (r & 7);
      float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f), a1 = a0;
      float lt = 0.f;
      for (int c = 0; c < nc; ++c) {
        float* ar = p.part_o + (r0 + 16 * c + r) * 64 + 8 * u;
        const float4 x0 = __ldcg(reinterpret_cast<const float4*>(ar)), x1 = __ldcg(reinterpret_cast<const float4*>(ar + 4));
        a0.x += x0.x; a0.y += x0.y; a0.z += x0.z; a0.w += x0.w; a1.x += x1.x; a1.y += x1.y; a1.z += x1.z; a1.w += x1.w;
        lt += __ldcg(p.part_lse + r0 + 16 * c + r);
        *reinterpret_cast<float4*>(ar) = make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(ar + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (u == 0) p.part_lse[r0 + 16 * c + r] = 0.f;
      }
      const float sc = (lt > 0.f) ? vsc / lt : 0.f;
      const uint4 gr = *reinterpret_cast<const uint4*>(qrow + (size_t)(p.Hq + 2 * p.Hkv + qh) * 64 + 8 * u);
      const float y[8] = {a0.x * sc, a0.y * sc, a0.z * sc, a0.w * sc, a1.x * sc, a1.y * sc, a1.z * sc, a1.w * sc};
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += y[e] * y[e];
      ss += __shfl_xor_sync(FULL, ss, 1);
      ss += __shfl_xor_sync(FULL, ss, 2);
      ss += __shfl_xor_sync(FULL, ss, 4);
      const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      uint32_t ow[4] = {gr.x, gr.y, gr.z, gr.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ow[e]));
        const __nv_bfloat162 o2 = __floats2bfloat162_rn(y[2 * e] * rs * sigmoidf_(g2.x), y[2 * e + 1] * rs * sigmoidf_(g2.y));
        ow[e] = *reinterpret_cast<const uint32_t*>(&o2);
      }
      *reinterpret_cast<uint4*>(p.out + (size_t)b * p.Hq * 64 + (size_t)qh * 64 + 8 * u) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
      if (tid == 0) p.counters[b * HP + hp] = 0;
    }
  }
#endif
}

'''

STAGES_OK = "static bool stages_ok(int64_t L, int64_t ST) { return (L == 1) ? (ST == 4 || ST == 6) : (ST == 3 || ST == 4); }"
VA_HOST = r'''static bool stages_ok(int64_t L, int64_t ST) { return (L == 1) ? (ST == 4 || ST == 6 || ST == 203 || ST == 204 || ST == 213 || ST == 214) : (ST == 3 || ST == 4); }

template <int ST>
static int va_go(const k1::Params* p, dim3 grid, cudaStream_t s, bool pdl = false) {   // VA1 (tools/va_patches.py): launch, or occupancy
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k1::va1_main<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, k1::va_smem<ST>()));
    attr = true;
  }
  int nb = 0;
  if (p) {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = grid; cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = k1::va_smem<ST>(); cfg.stream = s;
    cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k1::va1_main<ST>, *p));
  } else C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k1::va1_main<ST>, 256, k1::va_smem<ST>()));
  return nb;
}'''
OCC_OLD = "  TORCH_CHECK(stages_ok(L, ST), \"k1: (L, stages) not instantiated\");\n  if (L == 1) return"
OCC_NEW = ("  TORCH_CHECK(stages_ok(L, ST), \"k1: (L, stages) not instantiated\");\n"
           "  if (ST > 200) return (int64_t)(ST % 10 == 3 ? va_go<3>(nullptr, dim3(1), 0) : va_go<4>(nullptr, dim3(1), 0));\n"
           "  if (L == 1) return")
LAUNCH_OLD = "  const dim3 grid((unsigned)Gx, (unsigned)(Hkv / 2));\n  if (L == 1) {\n"
LAUNCH_NEW = ("  const dim3 grid((unsigned)Gx, (unsigned)(Hkv / 2));\n"
              "  if (stages > 200) {   // VA1: needs L 1, fold 1, B <= 512, part_o >= B*Hkv/2*16 rows of 64 (checked by Python)\n"
              "    TORCH_CHECK(L == 1 && fold == 1 && B <= 512, \"va1: L 1, fold, B <= 512\");\n"
              "    if (stages % 10 == 3) va_go<3>(&p, grid, st, stages > 210);\n    else va_go<4>(&p, grid, st, stages > 210);\n"
              "    C10_CUDA_KERNEL_LAUNCH_CHECK();\n    return;\n  }\n  if (L == 1) {\n")

PY_BLOCK = '''VA1_ENABLE = True   # VA1 (tools/va_patches.py): flat warp-specialized decode attention, full layers, q_len 1; read at call time
VA1_MIN_B = 2
VA1_STAGES = 4
VA1_SEGK = 400
VA1_PDL = False     # programmatic dependent launch (stages code + 10)
VA1_B1 = False      # also decode full B = 1 (ahead of the k1fa4 route), over VA1_NC1 workspace copies
VA1_NC1 = 8
VA1_ROWS = 16 * 1024
VA1_CASES = ((2, 3000, 1, 1.0, 301), (3, 1500, 64, 1.0, 302), (16, 700, 1, 1.0 / 64, 303), (5, 9000, 1, 1.0, 304),
             (33, 300, 1, 1.0, 305), (32, 2500, 1, 1.0, 307), (1, 9000, 1, 1.0, 308), (1, 2000, 64, 1.0 / 64, 309))
_VA1_WS = {}


def _va1_wanted(data, cfg, state):
    B = cfg["batch"]
    return (VA1_ENABLE and state.get("va1_ok") and cfg["q_len"] == 1 and not cfg["window"] and not cfg["rope_theta"]
            and (1 if VA1_B1 else VA1_MIN_B) <= B <= 256 and B * cfg["num_kv_heads"] * 8 * VA1_NC1 <= VA1_ROWS
            and str(data["k_cache"].device) in _VA1_WS
            and _k1_supported(data, cfg))


def _va1_b1(cfg, state):
    return VA1_B1 and VA1_ENABLE and state.get("va1_ok") and cfg["batch"] == 1 and cfg["q_len"] == 1 and not cfg["window"]


def _va1_run(qkv, data, cfg, stages=None, segk=None, pdl=None):
    B, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    dev = qkv.device
    st = 200 + int(stages or VA1_STAGES)
    G = max(B + 1, _k1_sms(dev) * max(1, min(2, _K1_OCC.get((1, st, str(dev)), 1))) // (hkv // 2))
    pdl = VA1_PDL if pdl is None else pdl
    st += 10 if pdl else 0
    ws = _VA1_WS[str(dev)]
    out = torch.empty((B, 1, hq * 64), dtype=torch.bfloat16, device=dev)
    _K1_MOD.k1_forward(qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
                       data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], ws[2], ws[0], ws[1], out, ws[3],
                       B, 1, hq, hkv, VA1_NC1 if B == 1 else 1, cfg["page_size"].bit_length() - 1, G, max(1, int(VA1_SEGK if segk is None else segk)),
                       1 if pdl else 0, float(cfg["qk_scale_factor"] * cfg["sm_scale"]), 1, 4, st)
    return out


def _va1_try(qkv, data, cfg, state):
    if _va1_wanted(data, cfg, state):
        try:
            return _va1_run(qkv, data, cfg)
        except Exception as ex:
            _k3_disable(state, "va1_ok", False, "VA1", ex)
    return None


def _va1_prepare(state):
    state["va1_ok"] = False
    dev = torch.empty(0, device="cuda").device
    try:
        if not state.get("k1_ok"):
            raise RuntimeError("K1 module not live")
        for st in (203, 204):
            _k1_occupancy(1, st, dev)
        z = lambda *s: torch.zeros(s, dtype=torch.float32, device=dev)  # noqa: E731
        _VA1_WS[str(dev)] = ws = (z(VA1_ROWS, 64), z(VA1_ROWS), _k1_inv_freq(0.0, dev),
                                  torch.zeros(VA1_ROWS // 16, dtype=torch.int32, device=dev))
        for (B, ctx, page, kvs, seed) in VA1_CASES:
            for rep in range(2):
                d = _k1_mk_case(dev, B, ctx, 1, False, page, kvs, seed)
                out = _va1_run(d["qkv"], d, d["config"], pdl=bool(rep))
                pos = d["seq_lens"].long()[:, None]
                slots = torch.gather(d["page_table"].long(), 1, pos // page) * page + pos % page
                kn, vn = (c.view(torch.uint8).view(-1, 2, 64)[slots] for c in (d["k_cache"], d["v_cache"]))
                tot, row = _k1_rel_rows(out, _k1_ref_attention(d, kn, vn))
                kref, vref = _k1_ref_new_kv(d)
                kt, vt = _k1_rel_rows(kn.view(FP8).float(), kref)[0], _k1_rel_rows(vn.view(FP8).float(), vref)[0]
                d2 = _k1_mk_case(dev, B, ctx, 1, False, page, kvs, seed)
                vk = _k1_rel_rows(out, _k1_run(d2["qkv"], d2, d2["config"]))[0]
                clean = not (int(ws[3].abs().sum()) or float(ws[0].abs().sum()) or float(ws[1].abs().sum()))
                print(f"[va1] self-test B={B} ctx={ctx} page={page} kvs={kvs:g} #{rep}: rel {tot:.2e} row {row:.2e} vs K1 "
                      f"{vk:.2e} new K {kt:.1e} V {vt:.1e} ws {'clean' if clean else 'DIRTY'}", flush=True)
                if not (bool(torch.isfinite(out.float()).all()) and clean and tot < 2e-2 and row < 6e-2 and vk < 1e-2
                        and kt < 3e-2 and vt < 3e-2):
                    raise RuntimeError(f"self-test mismatch B={B} ctx={ctx}")
        state["va1_ok"] = True
        print(f"[va1] LIVE: full q_len 1 B >= {VA1_MIN_B}, stages {VA1_STAGES}, segk {VA1_SEGK}, enable {VA1_ENABLE}", flush=True)
    except Exception as ex:
        print(f"[va1] DISABLED, full decode attention stays on K1: {ex!r}", flush=True)


'''

PATCHES = (
    ("k1_main (VA1 kernel)", K1_MAIN, VA1_CUDA + K1_MAIN),
    ("stages_ok (VA1 host)", STAGES_OK, VA_HOST),
    ("k1_occupancy (VA1)", OCC_OLD, OCC_NEW),
    ("k1_forward launch (VA1)", LAUNCH_OLD, LAUNCH_NEW),
    ("python block", "def _k1_rms_t(x, eps=RMS_EPS):\n", PY_BLOCK + "def _k1_rms_t(x, eps=RMS_EPS):\n"),
    ("prepare", "    _p206_prepare(state, t0)\n    torch.cuda.synchronize()\n",
     "    _p206_prepare(state, t0)\n    _va1_prepare(state)\n    torch.cuda.synchronize()\n"),
    ("kernel_fn B1", "    if o is None and _v18_fa4_wanted(qkv, data, cfg, state):\n",
     "    if o is None and not _va1_b1(cfg, state) and _v18_fa4_wanted(qkv, data, cfg, state):\n"),
    ("kernel_fn", "            o = _k1_run(qkv, data, cfg).view(tokens, hq * d)\n",
     "            o = _va1_try(qkv, data, cfg, state)\n"
     "            o = (o if o is not None else _k1_run(qkv, data, cfg)).view(tokens, hq * d)\n"),
)


def apply(text):
    for what, old, new in PATCHES:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"va patch anchor {what!r} found {n} times")
        text = text.replace(old, new, 1)
    return text


ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 1))
OUT = ROOT / f"kernels/muse_decode_layer/k3_va_{N}.py"
LAB = ROOT / f"kernels/lab/va_k3_va_{N}_ab.py"
TAG = f"k3-va-{N}"
NONCE = (f"# nonce {TAG} (k3_vms_1k + VA1: flat warp-specialized decode attention for full layers at q_len 1, B >= 2, "
         "self-tested against the reference and K1 with fallback; VA1_ENABLE)")
BASE = ROOT / "kernels/muse_decode_layer/k3_vms_1k.py"


def base_text():
    """k3_vms_1k pre-strip text (tools/gen_k3_vms.py vms1_text(kl1=True)); its strip is asserted equal to k3_vms_1k.py."""
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_vms as GV
    sys.argv = _argv
    t, base, n1 = GV.vms1_text(kl1=True)
    nonce = next(ln for ln in BASE.read_text(encoding="utf-8").split(chr(10))[:3] if ln.startswith("# nonce"))
    t = t.replace(n1, nonce)
    tag = nonce.split()[2]
    s0 = GV.GM.finish(t, nonce, tag, base, [])
    GV.GM.check(s0 == BASE.read_text(encoding="utf-8"), "the vms1_text(kl1=True) chain does not reproduce k3_vms_1k.py")
    return t, base, GV, nonce, s0


def main():
    t, base, GV, nonce0, s0 = base_text()
    rep = [f"base {BASE.name} reproduced byte-identical ({len(s0.encode())} B)"]
    s = GV.GM.finish(apply(t.replace(nonce0, NONCE)), NONCE, TAG, base, rep)
    rep.append(f"size: {len(s0.encode())} B -> {len(s.encode())} B (+{len(s.encode()) - len(s0.encode())} B, "
               f"headroom {1_000_000 - len(s.encode())} B)")
    for r in rep:
        print(f"[va_patches] OK  {r}")
    if "--check" in sys.argv:
        GV.GM.check(OUT.exists() and OUT.read_text(encoding="utf-8") == s, f"{OUT.name} on disk differs (regenerate)")
        print(f"[va_patches] --check: {OUT.name} up to date")
        return 0
    if len(s.encode()) < 1_000_000:
        OUT.write_text(s, encoding="utf-8", newline="\n")
    LAB.write_text(s, encoding="utf-8", newline="\n")
    print(f"[va_patches] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

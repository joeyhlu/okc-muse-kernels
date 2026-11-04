"""HYAP: default-off, shared in-order KV ring with two claiming consumer teams.

Only the named lab vehicle is generated. CPU checks never initialize CUDA.
Linux compile artifacts are retained in a temporary directory, never deleted.
"""
import argparse
import ast
import contextlib
import difflib
import hashlib
import io
import json
import math
import pathlib
import py_compile
import re
import runpy
import subprocess
import sys
import time
import types
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_pure_c1.py"
OUT = ROOT / "kernels/lab/k3_hyap_ab.py"

# CUDA added only to independently built arms. Ship and twin use the literal
# base source, including its launch, barriers and exact top-8 layer routing.
CUDA_SUPPORT = r'''
// HYAP: 0..255 / 256..511 consumers; 512..639 producer. Named barriers:
// 1 = finalize (512 consumers), 2 = producer, 3..6 = individual WGs,
// 7/8 = consumer teams. Barrier 0 is only the full-CTA prologue/epilogue.
template<int ROLE> __device__ __forceinline__ void hyap_sync() {
  if constexpr (ROLE == 1) asm volatile("bar.sync 2, 128;" ::: "memory");
  else asm volatile("bar.sync %0, 256;" :: "r"(7 + (int(threadIdx.x) >> 8)) : "memory");
}
__device__ __forceinline__ void hyap_wgsync(int wg) {
  asm volatile("bar.sync %0, 128;" :: "r"(3 + 2 * (int(threadIdx.x) >> 8) + wg) : "memory");
}
'''

CUDA_KERNEL = r'''
struct alignas(1024) HyapShared {
  alignas(1024) uint8_t k[4 * STAGE_BYTES], v[4 * STAGE_BYTES];
  uint64_t full[4], empty[4];
  int counter, sense, total, nseg;
  int cum[33], seg[32][3], tile[33];
  HyapView team[2];
};
static_assert(sizeof(HyapShared) <= 227 * 1024, "HYAP exceeds H200 smem");
static_assert(512 * 104 + 128 * 32 <= 65536, "HYAP role register budget");

template<int CHUNK>
__global__ void __launch_bounds__(640, 1) hyap_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t raw[];
  auto& A = *reinterpret_cast<HyapShared*>(raw);
  const int tid = threadIdx.x, pair = blockIdx.z;
  if (tid == 0) {
    A.counter = 0; A.cum[0] = 0;
    // The base reads seq_lens before its dependency wait as well.
    for (int b = 0; b < p.B; ++b) A.cum[b + 1] = A.cum[b] + __ldg(p.seq_lens + b) + 8;
    const int64_t T = int64_t(A.cum[p.B]) + (p.B - 1) * 600;
    // c/G == (2*c)/(2*G): exact union of adjacent ship ranges, same SEGK.
    auto vmap = [&](int v) {
      int lo = 0, hi = p.B - 1;
      while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (A.cum[mid] + mid * 600 <= v) lo = mid; else hi = mid - 1;
      }
      return min(v - lo * 600, A.cum[lo + 1]);
    };
    const int cs = vmap(int(int64_t(blockIdx.x) * T / gridDim.x));
    const int ce = vmap(int(int64_t(blockIdx.x + 1) * T / gridDim.x));
    int n = 0, tiles = 0;
    A.tile[0] = 0;
    for (int b = 0; b < p.B; ++b) {
      const int lo = max(cs, A.cum[b]) - A.cum[b];
      const int hi = min(ce, A.cum[b + 1]) - A.cum[b];
      if (lo < hi) {
        A.seg[n][0] = b; A.seg[n][1] = lo; A.seg[n][2] = hi;
        tiles += (hi - lo + 63) / 64;
        A.tile[++n] = tiles;
      }
    }
    A.nseg = n; A.total = tiles;
    for (int s = 0; s < 4; ++s) {
      mbar_init_n(&A.full[s], 128);
      mbar_init_n(&A.empty[s], 1); // exactly one elected arrival per owning team
    }
    for (int t = 0; t < 2; ++t) {
      auto& S = A.team[t];
      S.k = A.k; S.v = A.v; S.full = A.full; S.empty = A.empty;
      S.counter = &A.counter; S.ticket = 0; S.claim_end = 0; S.prepared_b = -1;
    }
    mbar_fence_init();
    cp_async4(smem_u32(&A.sense), p.ws_cnt + 30022);
    cp_commit(); cp_wait<0>();
  }
  __syncthreads();
  if (tid >= 512) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
    for (int s = 0; s < A.nseg; ++s) {
      const int b = A.seg[s][0], seq = __ldg(p.seq_lens + b);
      hyap_segment<8, true, 1, CHUNK>(p, A.team[0], b, pair, A.seg[s][1] == 0,
        A.seg[s][1], A.seg[s][2], seq, seq + 8, 0, 0, A.tile[s], 0, true);
    }
    cp_wait<0>(); // every producer async copy is complete before the CTA rendezvous
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 104;" ::: "memory");
    auto& S = A.team[tid >> 8];
    if ((tid & 255) == 0) {
      S.ticket = atomicAdd(S.counter, CHUNK);
      S.claim_end = S.ticket + CHUNK;
    }
    hyap_sync<2>();
    int s = 0;
    while (S.ticket < A.total) {
      while (s + 1 < A.nseg && S.ticket >= A.tile[s + 1]) ++s;
      const int b = A.seg[s][0], seq = __ldg(p.seq_lens + b);
      hyap_segment<8, true, 2, CHUNK>(p, S, b, pair, A.seg[s][1] == 0,
        A.seg[s][1], A.seg[s][2], seq, seq + 8, 0, 0, A.tile[s], 0, true);
    }
    wgmma_wait<0>();
  }
  // No CTA exits early, including an empty virtual range or an idle team.
  // All reductions from both teams precede the single ticket and reset loops.
  __syncthreads();
  if (tid < 512) flat_finalize<8, true, 512>(p, &A.sense);
}

void hyap_launch(Params p, dim3 grid, cudaStream_t stream, bool pdl) {
  TORCH_CHECK(p.flat && p.coop && p.nc == 1 && p.Hkv == 2 && !p.window && (p.B == 16 || p.B == 32));
  p.q2 = 0;
  static bool init = false;
  static int nsm = 0;
  if (!init) { // same first-eager-call initialization pattern as ship launch()
    int dev = 0, occ = 0;
    TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
    TORCH_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess);
    TORCH_CHECK(cudaFuncSetAttribute(hyap_kernel<HYAP_CHUNK>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(HyapShared)) == cudaSuccess);
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, hyap_kernel<HYAP_CHUNK>, 640, sizeof(HyapShared)) == cudaSuccess && occ >= 1);
    init = true;
  }
  grid.x = nsm; p.n_splits = nsm;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid; cfg.blockDim = dim3(640, 1, 1);
  cfg.dynamicSmemBytes = sizeof(HyapShared); cfg.stream = stream;
  cudaLaunchAttribute attr{};
  if (pdl) {
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &attr; cfg.numAttrs = 1;
  }
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, hyap_kernel<HYAP_CHUNK>, p) == cudaSuccess);
}
std::vector<int64_t> hyap_resources() {
  cudaFuncAttributes a{}; int occ = 0;
  TORCH_CHECK(cudaFuncSetAttribute(hyap_kernel<HYAP_CHUNK>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(HyapShared)) == cudaSuccess);
  TORCH_CHECK(cudaFuncGetAttributes(&a, hyap_kernel<HYAP_CHUNK>) == cudaSuccess);
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, hyap_kernel<HYAP_CHUNK>, 640, sizeof(HyapShared)) == cudaSuccess);
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes,
    sizeof(HyapShared), occ, 104, 32};
}
'''


def _hyap_cuda(ship, chunk):
    """Plain-source vendoring, no module handles; exact source anchors fail shut."""
    assert chunk in (2, 4, 8)
    def sub(s, old, new):
        assert s.count(old) == 1, (s.count(old), old[:100])
        return s.replace(old, new)
    a = ship.index('template <int L, bool F>\nstruct Smem {')
    b = ship.index('\ntemplate <int L, typename SM>', a)
    view = ship[a:b].replace('template <int L, bool F>\nstruct Smem {',
        'struct HyapView {\nstatic constexpr int L = 8; static constexpr bool F = true;\nint* counter; int ticket, claim_end, prepared_b;')
    view = sub(view, 'uint8_t k[Cfg<L, F>::STAGES * STAGE_BYTES];', 'uint8_t* k;')
    view = sub(view, 'uint8_t v[Cfg<L, F>::STAGES * STAGE_BYTES];', 'uint8_t* v;')
    view = sub(view, 'uint64_t full[8];', 'uint64_t* full;')
    view = sub(view, 'uint64_t empty[8];', 'uint64_t* empty;')
    a = ship.index('template <int L, bool FLAT, int ROLE, int MODE = 0>')
    b = ship.index('\ntemplate <int L, bool WS, int NT>\n__device__ __forceinline__ void flat_finalize', a)
    seg = ship[a:b]
    seg = sub(seg, 'template <int L, bool FLAT, int ROLE, int MODE = 0>',
        'template <int L, bool FLAT, int ROLE, int CHUNK, int MODE = 3>')
    seg = sub(seg, 'bool segment(const Params& p, Smem<L, FLAT>& S,', 'bool hyap_segment(const Params& p, HyapView& S,')
    seg = sub(seg, 'const int tid = threadIdx.x, lane = tid & 31,',
        'const int tid = ROLE == 1 ? int(threadIdx.x) - 256 : int(threadIdx.x) & 255, lane = tid & 31,')
    a = seg.index('if (!skip_init_) {'); b = seg.index('if constexpr (ROLE == 1)', a)
    seg = seg[:a] + '// Ring initialized once by CTA thread 0, phases span all requests.\n' + seg[b:]
    seg = seg.replace('__syncthreads();', 'hyap_sync<ROLE>();')
    seg = seg.replace('csync<WS, NT>();', 'hyap_sync<ROLE>();')
    seg = seg.replace('wgsync(wgi);', 'hyap_wgsync(wgi);')
    # Q, scales and new-KV scratch are private and remain valid for the team's
    # entire request. A discontinuous claim flushes accumulators, NOT queries.
    # kvq is reused by VT, so both the BF16 gather and its Q8 conversion must
    # be bypassed together on a repeated request; q8s itself is never reused.
    qa = seg.index('int tidL = tid, partL = part;')
    qb = seg.index('constexpr int MT = 1;',qa)
    query = seg[qa:qb]
    seg = seg[:qa] + '''const bool hyap_reload = S.prepared_b != b;
constexpr bool QDIR = L == 8 && !WS;
if (hyap_reload) {
''' + query + '''}
const float kscale = S.scl[0], vscale = S.scl[1];
const float sl2 = p.scale_log2 * kscale;
''' + seg[qb:]
    seg = sub(seg, 'uint32_t qa8[2][4];\nif constexpr (!QDIR)',
        'uint32_t qa8[2][4];\nif (hyap_reload) {\nif constexpr (!QDIR)')
    seg = sub(seg, 'float cr[2] = {0.f, 0.f};',
        'if (tid == 0) S.prepared_b = b;\n}\nfloat cr[2] = {0.f, 0.f};')
    # Cache append is done by the tile owner in ws_patch, including a newly
    # acquired contiguous chunk; query setup must never write unowned keys.
    seg = sub(seg, 'if (pos >= start && pos < end) {', 'if (false) { // HYAP: owner writes below at tile consumption')
    patch_call = 'ws_patch_ni<L, NT, false>(&S, i, st_c, start, seq, end, tid)'
    seg = sub(seg, patch_call, 'hyap_patch(p, S, i, st_c, start, seq, end, tid, pair)')
    seg = sub(seg, 'int st_c = tb % STAGES;\nuint32_t ph_c = (uint32_t)(tb / STAGES) & 1u;',
        'const int i_begin = S.ticket - tb;\nint st_c = (tb + i_begin) % STAGES;\nuint32_t ph_c = (uint32_t)((tb + i_begin) / STAGES) & 1u;')
    seg = sub(seg, 'const int nit = ntiles;', 'int nit = min(ntiles, S.claim_end - tb);')
    seg = sub(seg, 'for (int jj = 0; jj < nit; ++jj)', 'for (int jj = i_begin; jj < nit; ++jj)')
    seg = sub(seg, 'if (patched || i == 0)', 'if (patched || i == i_begin)')
    seg = sub(seg, 'if (jj == 0)', 'if (jj == i_begin)')
    seg = sub(seg, 'mbar_arrive(&S.empty[st]);',
        'hyap_sync<ROLE>(); // both WGs finished QK and V transpose\nif (tid == 0) mbar_arrive(&S.empty[st]);')
    # No accumulator reset for a contiguous next claim. At a gap/request end,
    # the original absolute-frame reduction and new-segment reset run verbatim.
    seg = sub(seg, 'vdesc_c ^= vdesc_x;\n}', '''vdesc_c ^= vdesc_x;
if (jj + 1 == nit) {
  hyap_sync<ROLE>();
  if (tid == 0) {
    const int next = tb + jj + 1;
    if (next == S.claim_end) {
      S.ticket = atomicAdd(S.counter, CHUNK);
      S.claim_end = S.ticket + CHUNK;
    } else S.ticket = next; // same claim straddles the request boundary
  }
  hyap_sync<ROLE>();
  if (S.ticket != tb + jj + 1 || jj + 1 == ntiles) break;
  nit = min(ntiles, S.claim_end - tb);
}
}''')
    # Keep dead L1/non-flat branches type-correct without instantiating their
    # Smem-only helpers for the HyapView specialization.
    seg = seg.replace('if (M_CL)', 'if constexpr (false)')
    seg = seg.replace('if (M_CL &&', 'if (false &&')
    extra = CUDA_SUPPORT + view + CUDA_PATCH + seg + CUDA_KERNEL
    anchor = 'template <int L>\nvoid launch(const Params& p0, dim3 grid, cudaStream_t stream, int cs, bool pdl) {'
    result = sub(ship, anchor, extra + '\n' + anchor + '''
if constexpr (L == 8) {
  if (p0.flat && !p0.window && (p0.B == 16 || p0.B == 32)) {
    hyap_launch(p0, grid, stream, pdl); return;
  }
}''')
    return '#define HYAP_CHUNK %d\n' % chunk + result


CUDA_PATCH = r'''
__device__ __forceinline__ bool hyap_patch(const Params& p, HyapView& S,
    int i, int st, int start, int seq, int end, int tid, int pair) {
  const int t0 = start + i * TILE;
  const int lo = max(t0, seq), hi = min(t0 + TILE, end);
  for (int e = tid; e < (hi - lo) * 8; e += 256) {
    const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
    const int sw = ((c ^ (r & 7)) << 4);
    const uint4 k = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
    const uint4 v = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
    *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = k;
    *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = v;
    const int64_t row = S.cloc[pos - seq];
    *reinterpret_cast<uint4*>(const_cast<uint8_t*>(cache_row(p.k_cache, p, row, pair * 128 + c * 16))) = k;
    *reinterpret_cast<uint4*>(const_cast<uint8_t*>(cache_row(p.v_cache, p, row, pair * 128 + c * 16))) = v;
  }
  if (lo < hi) fence_proxy_async();
  return lo < hi;
}
'''

RESOURCE_CPP = '\nstd::vector<int64_t> hyap_resources();\n'
LAB = '''
# HYAP BEGIN: only B16/B32 full q_len=8; independent padz-style arm banks.
HYAP = {}
HYAP_CHUNK = 4
HYAP_TWIN = False
HYAP_BUILD = False
_HYAP = {k: {"mod": None, "ws": {}, "build_s": 0.0, "err": None, "resources": None}
         for k in ("twin", 2, 4, 8)}

def _hyap_src(k):
    if k in (0, "twin"):
        return HYA_CUDA_SRC
    return _hyap_cuda(HYA_CUDA_SRC, k)

def _hyap_build(k):
    H = _HYAP[k]
    if H["mod"] is not None:
        return H["mod"]
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    try:
        src = _hyap_src(k)
        cpp = HYA_CPP_SRC + ("" if k == "twin" else _HYAP_RESOURCE_CPP)
        tag = hashlib.sha1((cpp + src).encode()).hexdigest()[:12]
        H["mod"] = load_inline(name=f"hyap_{k}_{tag}", cpp_sources=cpp, cuda_sources=src,
            functions=["run"] + ([] if k == "twin" else ["hyap_resources"]),
            extra_cflags=["-std=c++20"], extra_cuda_cflags=["-std=c++20", "-O3",
            "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr", "-Xptxas=-v,-warn-spills"], verbose=False)
    except Exception as ex:
        H["err"] = repr(ex)
        raise
    H["build_s"] = time.time() - t0
    return H["mod"]

def _hyap_prepare(state, t0):
    if not HYAP_BUILD:
        return
    if not state.get("hya"):
        raise RuntimeError("HYAP requires the real HYA prepare to succeed")
    dev = state["router"].device
    for k,H in _HYAP.items():
        mod = _hyap_build(k)
        if k != "twin":
            H["resources"] = list(mod.hyap_resources())
            reg, local, static, dynamic, occ, cr, pr = H["resources"]
            if reg > 104 or local or occ < 1 or cr != 104 or pr != 32 or static + dynamic > 227*1024:
                raise RuntimeError(f"HYAP {k} resource gate: {H['resources']}")
        if str(dev) not in H["ws"]:
            n = HYA_WS_ROWS
            H["ws"][str(dev)] = (torch.zeros(n*64, dtype=torch.float32, device=dev),
                torch.zeros(n, dtype=torch.float32, device=dev), torch.zeros(n, dtype=torch.int32, device=dev))

def _hyap_bank(cfg, plan, dev, ws):
    eligible = cfg["batch"] in (16,32) and cfg["q_len"] == 8 and not cfg["window"] and plan["flat"] == 1
    if not eligible:
        return _HYA["mod"], ws
    v = HYAP.get((cfg["batch"], "full"), 0)
    if type(v) is not int or v not in (0,1):
        raise ValueError(f"invalid HYAP selector {v}")
    if not v and not HYAP_TWIN:
        return _HYA["mod"], ws
    if v and HYAP_TWIN:
        raise ValueError("HYAP_TWIN and HYAP=1 are mutually exclusive")
    if type(HYAP_CHUNK) is not int or HYAP_CHUNK not in (2,4,8):
        raise ValueError(f"invalid HYAP_CHUNK {HYAP_CHUNK}")
    if v and (cfg["num_kv_heads"] != 2 or plan["nc"] != 1 or plan.get("coop", 1) != 1):
        raise ValueError("HYAP requires the scored Hkv=2, nc=1, coop=1 route")
    H = _HYAP["twin" if HYAP_TWIN else HYAP_CHUNK]
    own = H["ws"].get(str(dev))
    if H["mod"] is None or own is None or own[1].numel() < cfg["batch"]*(cfg["num_kv_heads"]//2)*plan["nc"]*8*16:
        raise RuntimeError("HYAP bank unavailable: set HYAP_BUILD=True before prepare")
    return H["mod"], own
# HYAP END
'''

PREP_A = ' _hya_prepare(state,t0)\n'
PREP_B = PREP_A + ' _hyap_prepare(state,t0)\n'
CALL_A = ' out=torch.empty((B,L,hq*64),dtype=torch.bfloat16,device=dev)\n _HYA["mod"].run('
CALL_B = ' hmod,ws=_hyap_bank(cfg,plan,dev,ws)\n out=torch.empty((B,L,hq*64),dtype=torch.bfloat16,device=dev)\n hmod.run('
END = '"HYA END (derived code above: HYA_CPP_SRC, HYA_CUDA_SRC, _hya_plan, _hya_launch; see _HYA_NOTICE)"\n'

def sub(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:80])
    return text.replace(old, new)

def literal(source, name):
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise KeyError(name)

def lab_text():
    import inspect
    return ('\nCUDA_SUPPORT = ' + repr(CUDA_SUPPORT) + '\nCUDA_PATCH = ' + repr(CUDA_PATCH)
        + '\nCUDA_KERNEL = ' + repr(CUDA_KERNEL) + '\n_HYAP_RESOURCE_CPP = ' + repr(RESOURCE_CPP)
        + '\n' + inspect.getsource(_hyap_cuda) + '\n' + LAB)

def build():
    base = BASE.read_bytes().decode('ascii')
    assert '\r' not in base
    return base, sub(sub(sub(base, PREP_A, PREP_B), CALL_A, CALL_B), END, END + lab_text())

def namespace(source):
    names = {'HYA_CPP_SRC','HYA_CUDA_SRC','_HYA','HYA_ENABLE','HYA_ROUTE','HYA_DEFAULT',
        'HYA_PDL','HYA_PDL_T','HYA_TUNED','HYA_PREPARE_BY_S','HYA_WS_ROWS','CUDA_SUPPORT','CUDA_PATCH','CUDA_KERNEL'}
    funcs = {'_hya_build','_hya_ws','_hya_inv_freq','_hya_sms','_hya_plan','_hya_launch','_hya_prepare'}
    nodes = [n for n in ast.parse(source).body if
        (isinstance(n,ast.FunctionDef) and (n.name in funcs or n.name.startswith('_hyap_'))) or
        (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and
        (t.id in names or t.id.startswith(('_HYAP','HYAP'))) for t in n.targets))]
    ns = dict(hashlib=hashlib,time=time)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OUT),'exec'),ns)
    return ns

def dry_run(source):
    # Same padz-style bank / real layer functions; only CUDA boundaries mocked.
    class Tensor:
        def __init__(self, shape, device="cpu-mock"):
            self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
            self.device = device
        def numel(self): return math.prod(self.shape)
        def float(self): return self
        def contiguous(self): return self
        def zero_(self): return self
        def __truediv__(self, _): return self
        def __rtruediv__(self, _): return self
        def __rpow__(self, _): return self

    class Module:
        def __init__(self, name): self.name, self.calls = name, []
        def run(self, *a): self.calls.append(a)
        def hyap_resources(self): return [96, 0, 0, 163840, 1, 104, 32]

    allocs, builds = [], []
    def tensor(shape, **kw):
        t = Tensor(shape, kw.get("device", "cpu-mock")); allocs.append(t); return t
    def loader(**kw):
        builds.append(kw)
        return Module(kw["name"])
    torch = types.ModuleType("torch")
    torch.zeros = torch.empty = tensor
    torch.arange = lambda *a, **kw: Tensor(32, kw.get("device", "cpu-mock"))
    torch.float32, torch.int32, torch.bfloat16 = "f32", "i32", "bf16"
    torch.cuda = types.SimpleNamespace(get_device_properties=lambda d: types.SimpleNamespace(multi_processor_count=132))
    ext = types.ModuleType("torch.utils.cpp_extension"); ext.load_inline = loader
    ns = namespace(source)
    ns.update(torch=torch, _pt1_wait=lambda _: None, _PT1={"errors": {}},
              ROPE_THETA=500000., THIRD_PARTY_NOTICES="dry", _HYA_NOTICE="dry", _hya_self_test=lambda _: 0.)
    with patch.dict(sys.modules, {"torch": torch, "torch.utils.cpp_extension": ext}), contextlib.redirect_stdout(io.StringIO()):
        state = {"router": Tensor((1,))}
        ns["_hya_prepare"](state, time.time())  # real base state constructor
        assert state["hya"]
        ns["_hyap_prepare"](state, time.time())
        assert len(builds) == 1, "default must not compile an experiment"
        ns["HYAP_BUILD"] = True
        ns["_hyap_prepare"](state, time.time())
        nalloc = len(allocs)
        ns["_hyap_prepare"](state, time.time())
        assert len(builds) == 5 and len(allocs) == nalloc
        banks = [ns["_HYA"]] + list(ns["_HYAP"].values())
        assert len({id(b["mod"]) for b in banks}) == 5
        assert len({id(t) for b in banks for t in b["ws"]["cpu-mock"]}) == 15
        cases = 0
        for B in (1, 8, 16, 32, 33):
            for L in (1, 8):
                for win in (0, 2048):
                    for flat in (0, 1):
                        for arm in (0, "twin", 2, 4, 8):
                            ns["HYAP"] = {(B, "full"): 0 if arm in (0, "twin") else 1}
                            ns["HYAP_TWIN"] = arm == "twin"
                            ns["HYAP_CHUNK"] = arm if type(arm) is int and arm else 4
                            cfg = dict(batch=B,q_len=L,window=win,page_size=1,num_kv_heads=2,num_q_heads=16,
                                       rope_theta=0.,sm_scale=.125,qk_scale_factor=5.473)
                            q = Tensor((B*L,2304))
                            d = {k: Tensor(1) for k in ("seq_lens","cache_loc","k_cache","v_cache","k_scale","v_scale")}
                            d["page_table"] = Tensor((B,80008))
                            before = [len(b["mod"].calls) for b in banks]
                            ns["_hya_launch"](q,d,cfg,True,{"flat":flat})
                            chosen = arm if B in (16,32) and L == 8 and not win and flat else 0
                            bank = ns["_HYA"] if chosen == 0 else ns["_HYAP"][chosen]
                            assert [len(b["mod"].calls)-n for b,n in zip(banks,before)] == [int(b is bank) for b in banks]
                            assert bank["mod"].calls[-1][10:13] == bank["ws"]["cpu-mock"]
                            cases += 1
        def fails(f, error):
            try: f()
            except error: return
            raise AssertionError("failure path silently accepted")
        cfg.update(batch=16,q_len=8,window=0)
        ns["HYAP_TWIN"] = False
        ws = ns["_HYA"]["ws"]["cpu-mock"]
        plan = dict(flat=1,nc=1)
        choose = lambda: ns["_hyap_bank"](cfg,plan,"cpu-mock",ws)
        for v in (-1,2,4,True,"1"):
            ns["HYAP"] = {(16,"full"):v}; fails(choose,ValueError)
        ns["HYAP"] = {(16,"full"):1}
        for v in (-1,0,1,3,True,'4'):
            ns['HYAP_CHUNK'] = v; fails(choose,ValueError)
        ns['HYAP_CHUNK'] = 4
        ns["HYAP"] = {(16,"full"):1}; ns["HYAP_CHUNK"] = 2; ns["HYAP_TWIN"] = True
        fails(choose,ValueError); ns["HYAP_TWIN"] = False
        H = ns["_HYAP"][2]
        saved = H["ws"]; H["ws"] = {}; fails(choose,RuntimeError); H["ws"] = saved
        fails(lambda: ns["_hyap_bank"](cfg,dict(flat=1,nc=100000),"cpu-mock",ws),ValueError)
        fails(lambda: ns["_hyap_prepare"]({"hya":False},time.time()),RuntimeError)
        H["mod"].hyap_resources = lambda: [96,0,0,163840,0,104,32]
        fails(lambda: ns["_hyap_prepare"](state,time.time()),RuntimeError)
        H["mod"] = None
        ext.load_inline = lambda **kw: (_ for _ in ()).throw(RuntimeError("dry compile failure"))
        fails(lambda: ns["_hyap_build"](2),RuntimeError)
        assert "dry compile failure" in H["err"]
    return dict(actual_launch_paths=cases, distinct_modules=5, independent_workspaces=5,
                real_base_prepare=True, default_no_build=True, failure_paths=True)


def protocol_check():
    """Adversarial producer/team interleavings, bounded ring and partial tails."""
    import random
    rng = random.Random(9417)
    cases = tiles_checked = 0
    for chunk in (2,4,8):
        for n in (0,1,2,3,4,5,7,8,9,15,16,17,31,63,129,257):
            for trial in range(16):
                slots = [None]*4; owners = [None]*n; seen = []
                cursor = produced = 0
                team = [dict(ticket=0,end=0),dict(ticket=0,end=0)]
                for t in rng.sample([0,1],2):
                    team[t] = dict(ticket=cursor,end=cursor+chunk); cursor += chunk
                for step in range(max(16,12*n+16)):
                    possible = []
                    if produced < n and slots[produced%4] is None: possible.append(2)
                    for t in (0,1):
                        x = team[t]['ticket']
                        if x < n and slots[x%4] == x: possible.append(t)
                    if len(seen) == n and produced == n: break
                    assert possible, ('ring deadlock',chunk,n,slots,team)
                    t = rng.choice(possible)
                    if t == 2:
                        slots[produced%4] = produced; produced += 1
                    else:
                        x = team[t]['ticket']; assert owners[x] is None
                        owners[x] = t; seen.append(x); slots[x%4] = None
                        team[t]['ticket'] += 1
                        if team[t]['ticket'] == team[t]['end']:
                            team[t] = dict(ticket=cursor,end=cursor+chunk); cursor += chunk
                else: raise AssertionError('scheduler failed to drain')
                assert sorted(seen) == list(range(n)) and slots == [None]*4
                cases += 1; tiles_checked += n
    # Exact union-of-two partition endpoints, SEGK virtual gaps and arbitrary
    # request boundaries (including chunks crossing a boundary).
    partitions = 0
    for B in (16,32):
        for seq in ([0]*B,[79992]*B,[rng.randrange(0,80000) for _ in range(B)]):
            cum = [0]
            for x in seq: cum.append(cum[-1]+x+8)
            total = cum[-1]+600*(B-1)
            def vmap(v):
                b = max(i for i in range(B) if cum[i]+600*i <= v)
                return min(v-600*b,cum[b+1])
            ranges = [(vmap(c*total//132),vmap((c+1)*total//132)) for c in range(132)]
            assert ranges[0][0] == 0 and ranges[-1][1] == cum[-1]
            assert all(a[1] == b[0] for a,b in zip(ranges,ranges[1:]))
            for c,(lo,hi) in enumerate(ranges):
                assert (lo,hi) == (vmap((2*c)*total//264),vmap((2*c+2)*total//264))
                spans = [(b,max(lo,cum[b])-cum[b],min(hi,cum[b+1])-cum[b]) for b in range(B)
                         if max(lo,cum[b]) < min(hi,cum[b+1])]
                assert sum(z-a for _,a,z in spans) == hi-lo
                prefix = [0]
                for _,a,z in spans: prefix.append(prefix[-1]+(z-a+63)//64)
                for chunk in (2,4,8):
                    owned = []
                    for ticket in range(0,prefix[-1],chunk):
                        end = min(prefix[-1],ticket+chunk)
                        while ticket < end:
                            s = max(i for i in range(len(spans)) if prefix[i] <= ticket)
                            stop = min(end,prefix[s+1])
                            owned.extend(range(ticket,stop)); ticket = stop
                    assert owned == list(range(prefix[-1]))
                partitions += 1
    # flat_finalize<8,true,512>, every CTA including empty ones: eight lanes
    # per row, complete workspace zeroing and output coverage with the new G.
    for B in (16,32):
        rows = B*8*16; G = 132; R = (rows+G-1)//G
        visited = set()
        for c in range(G):
            for rbase in range(c*R,min(rows,(c+1)*R),512//8):
                for tid in range(512):
                    row = rbase+(tid>>3)
                    if row < min(rows,(c+1)*R):
                        key = row,tid&7; assert key not in visited; visited.add(key)
        assert len(visited) == rows*8
    return dict(interleavings=cases,tiles=tiles_checked,partitions=partitions,
                exact_once_ownership=True,finalize_coverage=True,ring_slots=4)


def pycompile_without_writes(paths):
    captured = []
    with patch('importlib._bootstrap_external._write_atomic',side_effect=lambda *a,**k:captured.append(a[1])):
        for p in paths:
            py_compile.compile(str(p),cfile=str(ROOT/'hyap-memory.pyc'),doraise=True)
    assert len(captured) == len(paths)


def gates(base,source):
    assert OUT.read_bytes() == source.encode('ascii'), 'regenerate lab file'
    assert sub(sub(sub(source,lab_text(),''),PREP_B,PREP_A),CALL_B,CALL_A) == base
    assert literal(source,'HYAP') == {} and literal(source,'HYAP_BUILD') is False
    assert literal(source,'HYAP_TWIN') is False and literal(source,'HYAP_CHUNK') == 4
    for name in ('HYA_CPP_SRC','HYA_CUDA_SRC'):
        assert literal(source,name) == literal(base,name)
    ns = namespace(source); ship = ns['_hyap_src'](0)
    assert ship == ns['_hyap_src']('twin') == literal(base,'HYA_CUDA_SRC')
    # Audit the precise source fragments reused in the new hot loop/flush.
    hot_a = 'auto pack2 = [&]() {'; hot_b = '\n};\npack2();'
    pack = ship[ship.index(hot_a):ship.index(hot_b,ship.index(hot_a))+len(hot_b)]
    red_a = 'if (do_red) {\n'; red_b = '\nconst bool mode1 ='
    red = ship[ship.index(red_a):ship.index(red_b,ship.index(red_a))]
    for chunk in (2,4,8):
        cu = ns['_hyap_src'](chunk)
        assert cu.count(pack) == 2 and cu.count(red) == 2
        assert cu.count('atomicAdd(S.counter, CHUNK)') == 2
        assert 'flat_finalize<8, true, 512>(p, &A.sense)' in cu
        assert 'mbar_init_n(&A.empty[s], 1)' in cu
    pycompile_without_writes([pathlib.Path(__file__),OUT,ROOT/'tools/box/hyap_check.py'])
    assert b'\r' not in (ROOT/'tools/local/hyap_gate.sh').read_bytes()
    checker = runpy.run_path(str(ROOT/'tools/box/hyap_check.py'))['cpu_schema']()
    changes = [{'kind':tag,'base_lines':[i+1,j] if i<j else [],'insert_after':i if i==j else None,
                'lab_lines':[u+1,v]} for tag,i,j,u,v in difflib.SequenceMatcher(None,
                base.splitlines(),source.splitlines(),autojunk=False).get_opcodes() if tag!='equal']
    # Complete per-bit line inventory, relative to both the base layer and
    # each expanded CUDA translation unit (the lab stores plain source data).
    cuda_changes = {}
    for k in (2,4,8):
        cu = ns['_hyap_src'](k)
        cuda_changes[k] = [{'kind':tag,'ship_CUDA_lines':[i+1,j] if i<j else [],
            'base_layer_lines':[16719+i,16718+j] if i<j else [],
            'insert_after_base_line':16718+i if i==j else None,
            'variant_CUDA_lines':[u+1,v]} for tag,i,j,u,v in difflib.SequenceMatcher(None,
            ship.splitlines(),cu.splitlines(),autojunk=False).get_opcodes() if tag!='equal']
    return dict(default_cuda_identical=True,default_restore_byte_identical=True,
        py_compile=True,bash_LF=True,dry_run=dry_run(source),protocol=protocol_check(),checker=checker,
        changed_lines=changes,cuda_changed_lines_HYAP_1_by_CHUNK=cuda_changes,
        role_registers_EST={'consumer':104,'producer':32,'total':57344},
        smem_bytes_EST={'shared_ring':65536,'team_view_each':48128,'whole_CTA':162816},
        cleanup_coverage={
            'ring_init':'Every CTA, thread 0: all 4 full barriers (128 arrivals), all 4 empty barriers (1 arrival), counter and both team headers; all 640 synchronize',
            'new_KV_patch':'Owning team local threads 0..255, e=tid; e<(hi-lo)*8; e+=256, covers every appended K/V 16-byte vector exactly once',
            'stage_release':'Owning team 0..255 rendezvous after QK wait and V transpose; local thread 0 alone arrives empty, after which only private VT is read',
            'partial_flush':'Each team local threads 0..255: 2 rows x 4 red.v4; tig=0 threads also do 2 red.f32, followed by team sync',
            'producer_drain':'Threads 512..639 cp_wait<0>; consumers 0..511 wgmma_wait<0>; all 640 rendezvous even with no owned tiles',
            'grid_counter_reset':'Every CTA consumer thread 0 contributes exactly one acq_rel ticket; the last resets the alternate counter at 30020+((epoch&1)^1) and writes epoch+1 at 30022',
            'output_and_workspace_reset':'Every CTA consumers 0..511 execute unchanged flat_finalize<8,true,512>, 8 lanes/row, stride 64 rows. sum_copies clears both float4 accumulators per part and part=0 clears ws_l; rows cover [0,B*8*16). Producers exit only after the all-640 rendezvous and own no cleanup rows'},
        accounting_EST=dict(weight_bytes_per_tile=[0,0],KV_bytes_per_tile=[16384,16384],
            unique_page_index_bytes_per_tile=[256,256],producer_per_thread_per_tile={'cp_async16':[8,8],'index_ldg':[4,4]},
            consumer_per_thread_per_tile={'QK_wgmma':[2,2],'PV_wgmma':[2,2],'pack2_ex2':[32,32],
                'pack2_fma':[32,32],'pack2_cvt_pair':[16,16],'VT_ldmatrix_x4':[2,2],
                'VT_store_u32':[8,8],'stage_release_bar_sync':[0,1]},
            release_mbar_arrivals_per_tile=[256,1],claim_per_chunk='1 shared atomic (thread 0), 2 shared stores, 2 team barriers, ticket loads by 256 threads',
            claim_scratch_bytes=28,flush_per_thread='8 red.v4 (128 B), 2 red.f32 on tig=0 (8 B); 33280 B logical payload/team flush',
            query_load_per_team_request='16384 B BF16 Q + 136 B scale/frequency + 64 B cache locations; +4096 B new KV when request tail overlaps',
            query_reload_per_discontinuity=0,
            ring_bytes_per_SM=[2*4*16384,4*16384],instruction_note='source/inline-PTX counts; address, control and compiler instructions require Linux SASS gate'))


def sass_functions(text):
    result = {}
    for b in re.split(r'Function\s*:\s*',text)[1:]:
        name = b.splitlines()[0].strip()
        result[name] = [re.sub(r'/\*.*?\*/','',s).strip() for s in b.splitlines()
                       if re.match(r'\s*/\*[0-9a-f]+\*/',s)]
    assert result
    return result


def resources(text):
    result = {}
    for b in re.split(r'ptxas info\s*: Compiling entry function ',text)[1:]:
        name = b.split("'",2)[1]
        if 'attn_kernel' not in name and 'hyap_kernel' not in name: continue
        r = re.search(r'Used (\d+) registers',b)
        s = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',b)
        assert r and s, b
        result[name] = dict(registers=int(r[1]),stack=int(s[1]),spill_stores=int(s[2]),spill_loads=int(s[3]))
    assert result
    return result


def linux_gate(base,source,report):
    import os
    import shutil
    import sysconfig
    import tempfile
    import torch
    from torch.utils.cpp_extension import include_paths
    work = pathlib.Path(tempfile.mkdtemp(prefix='hyap-gate-'))
    report['build_dir'] = str(work)
    for i,p in enumerate((pathlib.Path(__file__),OUT,ROOT/'tools/box/hyap_check.py')):
        py_compile.compile(str(p),cfile=str(work/f'py{i}.pyc'),doraise=True)
    inc = ['-I'+p for p in include_paths(device_type='cuda')]+['-I'+sysconfig.get_path('include')]
    abi = f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}'
    def run(cmd,name):
        start = time.perf_counter()
        p = subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (work/(name+'.log')).write_bytes(p.stdout.encode())
        report.setdefault('commands',[]).append(dict(command=cmd,seconds_MEASURED=time.perf_counter()-start,rc=p.returncode))
        assert p.returncode == 0,(name,p.stdout[-16000:])
        return p.stdout
    nvcc,cobj,cxx = shutil.which('nvcc'),shutil.which('cuobjdump'),shutil.which(os.environ.get('CXX','c++'))
    assert nvcc and cobj and cxx,'nvcc, cuobjdump and C++20 compiler required'
    report['nvcc'] = run([nvcc,'--version'],'version')
    ns = namespace(source); report['builds'] = {}; assemblies = {}
    # A host executable reports actual C++ layout without loading a CUDA module.
    for k in (0,'twin',2,4,8):
        label = str(k); cuda = ns['_hyap_src'](k)
        cpp = '#include <torch/extension.h>\n'+literal(base,'HYA_CPP_SRC')
        funcs = ['run']
        if k not in (0,'twin'): cpp += RESOURCE_CPP; funcs.append('hyap_resources')
        cpp += '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n'+'\n'.join(f'm.def("{f}", &{f});' for f in funcs)+'\n}\n'
        cp,cu,co,obj = [work/(label+s) for s in ('.cpp','.cu','.cpp.o','.cu.o')]
        cp.write_bytes(cpp.encode()); cu.write_bytes(cuda.encode())
        run([cxx,'-std=c++20','-O3','-fPIC',abi,'-DTORCH_EXTENSION_NAME=hyap_compile_only',*inc,'-c',str(cp),'-o',str(co)],label+'_host')
        log = run([nvcc,'-std=c++20','-O3','-gencode=arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr',
            '-Xptxas=-v,-warn-spills','-Xcompiler=-fPIC',abi,*inc,'-c',str(cu),'-o',str(obj)],label+'_nvcc')
        res = resources(log)
        ass = sass_functions(run([cobj,'--dump-sass',str(obj)],label+'_sass'))
        usage = run([cobj,'--dump-resource-usage',str(obj)],label+'_resources')
        assemblies[label] = ass
        report['builds'][label] = dict(resources=res,resource_usage=usage)
        if k in (0,'twin'): continue
        target = next(n for n in res if 'hyap_kernel' in n)
        r = res[target]; ops = ass[target]
        assert r['spill_stores'] == r['spill_loads'] == r['stack'] == 0,(label,r)
        assert not any(re.search(r'\b(?:LDL|STL)\b',s) for s in ops),label
        # Runtime role limits must survive the optimizer; values also recorded
        # by the pod's cudaFuncGetAttributes/occupancy gate.
        assert any('SETMAXNREG' in s and '0x68' in s for s in ops),(label,'missing consumer 104')
        assert any('SETMAXNREG' in s and '0x20' in s for s in ops),(label,'missing producer 32')
        report['builds'][label]['roles'] = dict(consumer=104,producer=32,source_budget=57344)
        report['builds'][label]['static_instructions_MEASURED'] = len(ops)
        report['builds'][label]['shared_claim_SASS'] = [s for s in ops if 'ATOMS' in s]
        assert report['builds'][label]['shared_claim_SASS'], 'claim not shared memory'
        # Isolate just the POD layout and measure sizeof with the host C++20
        # compiler, including the real alignas rules from the base source.
        a = cuda.index('struct HyapView {'); b = cuda.index('\n__device__ __forceinline__ bool hyap_patch',a)
        view = cuda[a:b]
        view = view.replace('__device__','').replace('__forceinline__','inline')
        shared = CUDA_KERNEL[CUDA_KERNEL.index('struct alignas(1024) HyapShared'):CUDA_KERNEL.index('\ntemplate<int CHUNK>')]
        cfg = 'template<int L,bool F> struct Cfg { static constexpr int STAGES=4; };\n'
        host = '#include <cstdint>\n#include <cstdio>\nusing __nv_bfloat16=uint16_t;\nconstexpr int TILE=64, ROWB=128, STAGE_BYTES=8192, MAXB=512, MAXSEG=64;\n'+cfg+view+shared+'\nint main(){printf("%zu %zu\\n",sizeof(HyapView),sizeof(HyapShared));}\n'
        hp,exe = work/(label+'_layout.cpp'),work/(label+'_layout')
        hp.write_bytes(host.encode()); run([cxx,'-std=c++20',str(hp),'-o',str(exe)],label+'_layout_compile')
        sizes = list(map(int,run([str(exe)],label+'_layout_run').split()))
        assert sizes == [48128,162816],('layout accounting drift',sizes)
        report['builds'][label]['dynamic_smem_bytes_MEASURED'] = sizes[1]
    # Same source must produce exactly the same ship/twin instructions.
    for name,ops in assemblies['0'].items():
        if 'attn_kernel' in name: assert ops == assemblies['twin'][name],name
    report['default_twin_SASS_identical'] = True
    report['status'] = 'CPU_COMPILE_SASS_PASS'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu-gate',action='store_true')
    ap.add_argument('--linux-gate',action='store_true')
    ap.add_argument('--json',type=pathlib.Path)
    args = ap.parse_args()
    base,source = build()
    if not (args.cpu_gate or args.linux_gate):
        # Exercise all CUDA construction paths before writing the lab artifact.
        ns = namespace(source)
        for k in (0,'twin',2,4,8): ns['_hyap_src'](k)
        OUT.write_bytes(source.encode('ascii'))
        print(f'generated {OUT.relative_to(ROOT)}; HYAP=0; {len(source)} bytes')
        return
    artifacts = (BASE,OUT,pathlib.Path(__file__),ROOT/'tools/box/hyap_check.py',ROOT/'tools/local/hyap_gate.sh')
    result = dict(status='FAIL',gpu_launched=False,output_sha256=hashlib.sha256(source.encode()).hexdigest(),
        artifact_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    try:
        result['cpu'] = gates(base,source); result['status'] = 'CPU_PASS'
        if args.linux_gate: linux_gate(base,source,result)
    except Exception as ex:
        result['status'] = 'FAIL'; result['error'] = repr(ex); raise
    finally:
        if args.json: args.json.write_bytes((json.dumps(result,indent=2)+'\n').encode())
        print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()


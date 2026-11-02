"""FINO: default-off, in-file KS1 -> GD -> SBB lab. No GPU work in this generator.

Only writes kernels/lab/k3_fino_ab.py. python -B tools/fino_patches.py [--check]
The CPU gate lives in tools/local/fino_compile.py (also a bash/Python polyglot).
"""
import ast
import hashlib
import inspect
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexr_c1.py"
OUT = ROOT / "kernels/lab/k3_fino_ab.py"


def once(s, old, new):
    assert s.count(old) == 1, ("FINO anchor", s.count(old), old[:100])
    return s.replace(old, new, 1)


def span(s, start, end):
    assert s.count(start) == s.count(end) == 1, (start, end)
    return s[s.index(start):s.index(end)]


def constants(s):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(s).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant)}


def cuda_parts(c):
    # Keep the shipped device arithmetic / launch helpers, only instantiate d1s.
    ks = span(c["KS1_CUDA_SRC"], "#include <torch/extension.h>", "namespace ks3 {")
    host = span(c["KS1_CUDA_SRC"], "void ks1_forward(", "void ks3_forward(")
    ks += ("#define KS1_INSTS\n#define KS1_MINSTS KS1_XM(1, 16, 1, 1)\n" + host)
    ks = once(ks, "KS1P(0);\n  const int seq", """// FINO: each split owns a disjoint portion of the OTHER bank.
float* fino_o = p.part_o;
float* fino_l = reinterpret_cast<float*>(p.part_ml);
#if FINO_BUILD
static_assert(L == 1 && NS == 16 && NTG == 1 && MG == 1, "FINO d1s only");
unsigned fino_e;
asm volatile("ld.global.acquire.gpu.u32 %0, [%1];" : "=r"(fino_e) : "l"(p.counters + 1023) : "memory");
fino_o += (fino_e & 1u) * 1024;
fino_l += (fino_e & 1u) * 16;
// 17 splits * 256 threads cover the 512 accumulator and 8 denominator words per KV group.
for (int i = s * NTHR + tid; i < 8 * 65; i += NSP * NTHR) {
if (i < 512) __stcg(p.part_o + ((fino_e + 1u) & 1u) * 1024 + grp * 512 + i, 0.f);
else __stcg(reinterpret_cast<float*>(p.part_ml) + ((fino_e + 1u) & 1u) * 16 + grp * 8 + i - 512, 0.f);
}
#endif
KS1P(0);
  const int seq""")
    ks = once(ks, "float* wa = p.part_o + ((size_t)grp * ROWS + mr) * 64 + DPT * mu;",
              "float* wa = fino_o + ((size_t)grp * ROWS + mr) * 64 + DPT * mu;")
    ks = once(ks, "float* wl = reinterpret_cast<float*>(p.part_ml) + (size_t)grp * ROWS + mr;",
              "float* wl = fino_l + (size_t)grp * ROWS + mr;")
    ks = once(ks, "if (mu == 0) vsl_red1(wl, lt * f);\nKS1P(7);", """if (mu == 0) vsl_red1(wl, lt * f);
#if FINO_BUILD
// Kernel completion + GD's griddepcontrol.wait publish every red.add; no ticket/fence hop.
return;
#endif
KS1P(7);""")
    # Pull the exact shipped finalize into a register-returning helper. No regrouping.
    fin = span(c["KS1_CUDA_SRC"], "template <int DPT, int UPR>\n__device__ __forceinline__ void vsl_fin1",
               "__device__ __forceinline__ void vsl_fin3")
    fin = once(fin, "void vsl_fin1", "uint32_t fino_fin_word")
    fin = once(fin, ", __nv_bfloat16* op)", ")")
    tail = """if (op == nullptr) return;
if constexpr (DPT == 4) *reinterpret_cast<uint2*>(op) = make_uint2(ow2[0], ow2[DPT / 2 - 1]);
else *reinterpret_cast<uint32_t*>(op) = ow2[0];
}"""
    fin = once(fin, tail, "static_assert(DPT == 2 && UPR == 32, \"FINO exact lane tree\");\nreturn ow2[0];\n}")
    gd = span(c["GD_CUDA_SRC"], "#include <cuda_runtime.h>", "__global__ void unpack_kernel(") + "#undef GC_XP\n}\n"
    gd = once(gd, "struct Args {", """struct Args {
const float* fino_o = nullptr;
const float* fino_l = nullptr;
const unsigned* fino_epoch = nullptr;
const __nv_bfloat16* fino_gate = nullptr;
float fino_eps = 1e-5f;""")
    helper = "using ks1::bfr; using ks1::bf2f; using ks1::FULL;\n" + fin + r'''
__device__ __forceinline__ uint32_t fino_word(const float* po, const float* pl,
const __nv_bfloat16* gate, int head, float eps) {
const int lane = threadIdx.x & 31;
const float2 v = __ldcg(reinterpret_cast<const float2*>(po + head * 64) + lane);
float den = lane == 0 ? __ldcg(pl + head) : 0.f;
den = __shfl_sync(FULL, den, 0);
const float a[2] = {v.x, v.y};
const uint32_t gr[1] = {__ldcg(reinterpret_cast<const unsigned*>(gate + head * 64) + lane)};
return fino_fin_word<2, 32>(a, den, gr, eps);
}
__device__ __forceinline__ void fino_x(const Args& A, int head, uint4& lo, uint4& hi) {
// Coherent load AFTER GD's wait; sbb1 cannot advance this epoch until GD completes.
const unsigned e = __ldcg(A.fino_epoch) & 1u;
const uint32_t w = fino_word(A.fino_o + e * 1024, A.fino_l + e * 16, A.fino_gate, head, A.fino_eps);
// The shipped GD B fragment repeats the 64-element row across eight groups of four lanes.
const int src = 8 * (threadIdx.x & 3);
lo = make_uint4(__shfl_sync(FULL, w, src), __shfl_sync(FULL, w, src + 1),
               __shfl_sync(FULL, w, src + 2), __shfl_sync(FULL, w, src + 3));
hi = make_uint4(__shfl_sync(FULL, w, src + 4), __shfl_sync(FULL, w, src + 5),
               __shfl_sync(FULL, w, src + 6), __shfl_sync(FULL, w, src + 7));
}
'''
    gd = once(gd, "#define GC_XP(S_)", helper + "#define GC_XP(S_)")
    gd = once(gd, "  constexpr int NW = WR * WK, NT = NW * 32, NP = 3;", """  constexpr int NW = WR * WK, NT = NW * 32, NP = 3;
static_assert(!(XP & 8) || (OUT == 0 && MT == 1 && WR == 1 && WK == 8 && KU == 2 && ST == 2 && S == 1 && MB == 4 && VF == 0 && CAT == 0), "FINO o/1 only");""")
    old = "X[0][mt][0] = ldxk<XP>(xb + xo[mt]);\nX[0][mt][1] = ldxk<XP>(xb + xo[mt] + 1);"
    gd = once(gd, old, "if constexpr (XP & 8) fino_x(A, u0, X[0][mt][0], X[0][mt][1]);\nelse {\n" + old + "\n}")
    old = "X[0][mt][0] = ldxk<XP>(GC_XP(s) + xo[mt]);\nX[0][mt][1] = ldxk<XP>(GC_XP(s) + xo[mt] + 1);"
    gd = once(gd, old, "if constexpr (XP & 8) fino_x(A, u0 + s, X[0][mt][0], X[0][mt][1]);\nelse {\n" + old + "\n}")
    sb = c["SBB1_CU"] + c["SBB1_HOST"]
    sb = once(sb, "struct SbbParams {", "struct SbbParams {\nunsigned* fino_epoch;")
    sb = once(sb, '  asm volatile("griddepcontrol.wait;" ::: "memory");\n  float br[8];', '''  asm volatile("griddepcontrol.wait;" ::: "memory");
#if FINO_BUILD
// One writer, once per layer call, after ALL GD readers complete (not merely its early trigger).
if (blockIdx.x == 0 && tid == 0) {
unsigned e = __ldcg(p.fino_epoch);
asm volatile("st.global.release.gpu.u32 [%0], %1;" :: "l"(p.fino_epoch), "r"(e + 1u) : "memory");
}
#endif
  float br[8];''')
    sb = once(sb, "double pre_eps, int64_t input_scale) {", "double pre_eps, int64_t input_scale, torch::Tensor epoch) {")
    sb = once(sb, "SbbParams p = {};", "SbbParams p = {};\np.fino_epoch = reinterpret_cast<unsigned*>(epoch.data_ptr<int>() + 1023);")
    decl = once(c["SBB1_DECL"], "double pre_eps, int64_t input_scale);", "double pre_eps, int64_t input_scale, torch::Tensor epoch);")
    return ks, gd, sb, decl


GD_HOST = r'''
void fino_gd(torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out,
torch::Tensor po, torch::Tensor pl, torch::Tensor epoch, torch::Tensor qkv,
int64_t tb, int64_t nbx, int64_t pdl, double eps) {
TORCH_CHECK(x.numel() == 1024 && out.numel() == 8192 && nbx == 512, "FINO o/1 layout");
gd::Args A{};
A.w = reinterpret_cast<const uint4*>(w.data_ptr()); A.esc = reinterpret_cast<const unsigned*>(esc.data_ptr());
A.x = reinterpret_cast<const uint16_t*>(x.data_ptr()); A.out = out.data_ptr();
A.M = 1; A.K = 1024; A.ldo = 8192; A.tb = (uint32_t)tb; A.trig = (pdl >> 1) & 1;
A.fino_o = po.data_ptr<float>(); A.fino_l = pl.data_ptr<float>();
A.fino_epoch = reinterpret_cast<const unsigned*>(epoch.data_ptr<int>() + 1023);
A.fino_gate = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()) + 20 * 64;
A.fino_eps = (float)eps;
C10_CUDA_CHECK(static_cast<cudaError_t>(gd::launch<0, 1, 1, 8, 2, 2, 1, 4, (FINO_BUILD ? 14 : 6)>(A, 512, 1, (int)pdl, at::cuda::getCurrentCUDAStream())));
}
std::vector<int64_t> fino_info() {
int64_t v[4];
C10_CUDA_CHECK(static_cast<cudaError_t>(gd::info<0, 1, 1, 8, 2, 2, 1, 4, (FINO_BUILD ? 14 : 6)>(v)));
return {v[0], v[1], v[2], v[3]};
}
// Prepare-only diagnostic. Never inserted in the timed production graph.
__global__ void fino_probe_kernel(const float* po, const float* pl, const unsigned* epoch,
const __nv_bfloat16* gate, __nv_bfloat16* got, __nv_bfloat16* ref, float eps, int previous) {
const int head = blockIdx.x, lane = threadIdx.x;
const unsigned e = (__ldcg(epoch) - (unsigned)previous) & 1u;
po += e * 1024; pl += e * 16;
const uint32_t word = gd::fino_word(po, pl, gate, head, eps);
*reinterpret_cast<uint32_t*>(got + head * 64 + 2 * lane) = word;
const float2 v = __ldcg(reinterpret_cast<const float2*>(po + head * 64) + lane);
const float a[2] = {v.x, v.y};
float den = lane == 0 ? __ldcg(pl + head) : 0.f;
den = __shfl_sync(0xffffffffu, den, 0);
const uint32_t gr[1] = {__ldcg(reinterpret_cast<const unsigned*>(gate + head * 64) + lane)};
ks1::vsl_fin1<2, 32>(a, den, gr, eps, ref + head * 64 + 2 * lane);
}
void fino_probe(torch::Tensor po, torch::Tensor pl, torch::Tensor epoch, torch::Tensor qkv,
torch::Tensor got, torch::Tensor ref, double eps, int64_t previous) {
fino_probe_kernel<<<16, 32, 0, at::cuda::getCurrentCUDAStream()>>>(po.data_ptr<float>(), pl.data_ptr<float>(),
reinterpret_cast<const unsigned*>(epoch.data_ptr<int>() + 1023),
reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()) + 20 * 64,
reinterpret_cast<__nv_bfloat16*>(got.data_ptr()), reinterpret_cast<__nv_bfloat16*>(ref.data_ptr()), (float)eps, (int)previous);
C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

GD_DECL = '''void fino_gd(torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out,
torch::Tensor po, torch::Tensor pl, torch::Tensor epoch, torch::Tensor qkv,
int64_t tb, int64_t nbx, int64_t pdl, double eps);
std::vector<int64_t> fino_info();
void fino_probe(torch::Tensor po, torch::Tensor pl, torch::Tensor epoch, torch::Tensor qkv,
torch::Tensor got, torch::Tensor ref, double eps, int64_t previous);
'''

LAB = r'''
# FINO LAB BEGIN: 0 ship, 1 FINO, 2 compiled ship, 3 FINO twin, 4 compiled ship twin.
FINO = 0
FINO_REQUIRE_BITWISE = False  # MG1 red.add order is not repeat-bitwise even in ship.
_FINO_KEYS = (1, 2, 3, 4)
_FINO = {}
_FINO_VARIANTS = [{"name": n, "set": {"FINO": k}} for n, k in
                  (("ship", 0), ("fino", 1), ("mid", 2), ("fino2", 3), ("ship2", 4))]

def _fino_source(k):
    assert k in _FINO_KEYS
    # Namespace suffix forces distinct device symbols / cubins for the two twins.
    src = "#define FINO_BUILD %d\n" % (k & 1) + _FINO_KS + _FINO_GD + _FINO_SB + _FINO_HOST
    src = src.replace("namespace ks1 {", "namespace ks1_fino%d {" % k).replace("ks1::", "ks1_fino%d::" % k)
    src = src.replace("namespace gd {", "namespace gd_fino%d {" % k).replace("gd::", "gd_fino%d::" % k)
    src = src.replace("sbb_kernel", "sbb_kernel_fino%d" % k)
    return src

def _fino_load(k):
    from torch.utils.cpp_extension import load_inline
    src = _fino_source(k)
    tag = hashlib.sha1((_FINO_CPP + src).encode()).hexdigest()[:12]
    return load_inline(name=f"fino_{k}_{tag}", cpp_sources=_FINO_CPP, cuda_sources=src,
                       functions=["ks1_forward", "ks1_info", "fino_gd", "fino_info", "sbb1_run", "fino_probe"],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"],
                       verbose=False)

def _fino_build():
    # Same bank/build-error pattern as the proven PADZ vehicle; serial builds avoid CPU oversubscription.
    for k in _FINO_KEYS:
        H = {"mod": None, "ok": False, "err": None, "bit": k & 1}
        _FINO[k] = H
        try:
            H["mod"] = _fino_load(k)
        except Exception as ex:
            H["err"] = repr(ex)[-600:]

def _fino_ready(state):
    st = state.get("gd")
    r = st["route"].get(("o", 1)) if st is not None else None
    sb = _SBB1["st"]
    return (r is not None and r["K"] == 1024 and r["N"] == 8192 and r["nbx"] == 512
            and GD_ROUTE.get(("o", 1)) == (1, 1, 8, 2, 2, 1, 6)
            and _gd_key(8192, 1024, GD_ROUTE[("o", 1)]) == (0, 1, 1, 8, 2, 2, 1, 4, 6)
            and sb is not None and sb["ok"].get(1) and state.get("cos_sin") is not None
            and ("ks1", 1, 16, 1, 1) in (state.get("vsl") or ()))

def _fino_prepare(state):
    if not _fino_ready(state):
        print("[fino] OFF: shipped KS1/GD/SBB prerequisite missing", flush=True)
        for H in _FINO.values():
            H["ok"] = False
        return
    r = state["gd"]["route"][("o", 1)]
    sb = _SBB1["st"]
    dev = state["dense"]["qkv_gate"].device
    for k, H in _FINO.items():
        H["ok"] = False
        if H["mod"] is None:
            print(f"[fino] arm {k} OFF: build {H['err']}", flush=True)
            continue
        try:
            info = list(H["mod"].fino_info())
            H["info"] = info
            if not (0 < info[0] <= 64 and info[1] == 0 and info[2] >= 4):
                raise RuntimeError(f"GD register/local/occupancy gate {info}")
            # Reuse the REAL GD route and SBB weights; do not reconstruct a model state.
            H["r"], H["sb"] = r, dict(sb, mod=H["mod"])
            H["po"] = torch.zeros(2 * 17 * 8 * 64, dtype=torch.float32, device=dev)
            H["pl"] = torch.zeros(2 * 17 * 8 * 2, dtype=torch.float32, device=dev)
            H["cnt"] = torch.zeros(1024, dtype=torch.int32, device=dev)
            H["prof"] = torch.empty(0, dtype=torch.int64, device=dev)
            H["ok"] = bool(_fino_self_test(H, state, k, dev))
            print(f"[fino] arm {k} {'LIVE' if H['ok'] else 'OFF: self-test failed'} info={info}", flush=True)
        except Exception as ex:
            H["ok"] = False
            H["err"] = repr(ex)
            print(f"[fino] arm {k} OFF: prepare {ex!r}", flush=True)

def _fino_pick(qkv, data, state):
    # Inspect ORIGINAL config before R2 page-table adaptation: P64 never enters FINO.
    cfg = data["config"]
    H = _FINO.get(FINO)
    if H is None or not H["ok"] or cfg["batch"] != 1 or cfg["q_len"] != 1 or cfg["page_size"] != 1:
        return None
    # The non-PDL qkv call is a full stream edge from the preceding layer call.
    # This orders KS1's PRE-wait epoch read after the preceding sbb1 increment.
    if GD_PDL.get("qkv") or ("qkv", 1) in RQ_SEL or SK2_T == 1 or _gd_live(state, "qkv", 1) is None or not _fino_ready(state):
        return None
    if not (SBB1_ENABLE and SBB1_ROUTE.get(1) and not GFT_ROUTE.get(1)):
        return None
    if data["stream"].dtype != torch.float32:
        return None
    return H if _ks1_wanted(qkv, data, cfg, state) and _vsl_pick(cfg, state) == ("ks1", 1, 16, 1, 1) else None

def _fino_attn(H, qkv, data, state):
    # Call signature and input accessors copied from _vsl_call.
    cfg = data["config"]
    out = torch.empty((1, 16 * 64), dtype=torch.bfloat16, device=qkv.device)
    H["mod"].ks1_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                          data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), state["cos_sin"],
                          H["po"], H["pl"], H["cnt"], out, H["prof"], 1, 16, 1,
                          float(cfg["sm_scale"] * cfg["qk_scale_factor"]), KS1_EPS,
                          int(KS1_TRIG.get(1, 0)), 1 if KS1_PDL else 0, 3, 0, 0)
    return out

def _fino_o(H, qkv, o):
    r = H["r"]
    out = torch.empty((1, r["N"]), dtype=torch.bfloat16, device=qkv.device)
    pdl = (1 if GD_PDL.get("o") else 0) | (2 if GD_TRIG.get("o") else 0)
    H["mod"].fino_gd(r["w"], r["esc"], o, out, H["po"], H["pl"], H["cnt"], qkv,
                     r["tb"], r["nbx"], pdl, KS1_EPS)
    return out

def _fino_bnd(H, stream, branch, router_dtype):
    # _sbb1_call verbatim accesses, plus the tensor-owned epoch argument.
    st, tokens = H["sb"], 1
    dev = branch.device
    out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
    ns = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev)
    rt = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev) if router_dtype is not None else st["e"]
    pw, qw, al, bt, isc, pe, qe = st["w"]
    st["mod"].sbb1_run(branch.contiguous(), stream.contiguous(), out, ns, rt, pw, qw, al, bt, pe, qe,
                        isc | (8 if LAT_SBB_TRIG.get(_k2a_bucket_of(tokens)) else 0), H["cnt"])
    return out, ns, (rt if router_dtype is not None else None)

def _fino_rel(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return float(d.norm() / b.norm().clamp_min(1e-30)), float((d / b.abs().clamp_min(1e-6)).max())

def _fino_self_test(H, state, key, dev):
    # Test ship vs candidate, exact finalize with identical accumulator inputs, and real three-node graph.
    global FINO
    state = dict(state)  # A diagnostic fallback must not disable a shipped route in the caller's state.
    saved = FINO
    good, worst, maxrel, bits, count = True, 0.0, 0.0, 0, 0
    try:
        for ctx, seed in ((1, 6801), (63, 6802), (100, 6803), (2047, 6804), (3000, 6805), (4000, 6806)):
            d = _ks1_case(ctx, 1, seed, ctx == 4000, dev)
            d["config"]["tokens"] = 1
            qkv = d["qkv"].view(1, -1)
            d["attn_input"] = torch.empty((1, HIDDEN), dtype=torch.bfloat16, device=dev)  # R2's real guard accessor.
            stream = torch.zeros((1, HIDDEN), dtype=torch.float32, device=dev)
            d["stream"] = stream
            FINO, H["ok"] = key, True
            assert _fino_pick(qkv, d, state) is H
            # Real P64 case; even R2 page-table flattening in kernel_fn must not change this decision.
            paged = _hya_paged(d, 64, seed + 70)
            assert _fino_pick(qkv, paged, state) is None
            FINO = 0
            pship = _fino_attention_test(qkv, paged, state).clone()
            FINO = key
            pgot = _fino_attention_test(qkv, paged, state)
            torch.cuda.synchronize()
            assert _fino_rel(pgot, pship)[0] <= 5e-3, "P64 unchanged fallback failed"
            print(f"[fino] arm {key} ctx={ctx} page64 fallback bitwise={torch.equal(pgot, pship)} rel={_fino_rel(pgot, pship)[0]:.4e}", flush=True)
            H["ok"] = False
            ref = _k1fa4_reference(d).view(1, -1)
            r1 = _k1fa4_r1_path(d).view(1, -1)
            e1 = _fino_rel(r1, ref)[0]
            ship_o = _vsl_call(qkv, d, state["cos_sin"], ("ks1", 1, 16, 1, 1)).clone()
            ship_y = _gd(state, "o", ship_o).clone()
            ship_b = _sbb1_call(_SBB1["st"], stream, ship_y, torch.float32, 1)
            kb, vb = d["k_cache"].view(torch.uint8).clone(), d["v_cache"].view(torch.uint8).clone()
            H["po"].zero_(); H["pl"].zero_(); H["cnt"].zero_()
            def call():
                o = _fino_attn(H, qkv, d, state)
                y = _fino_o(H, qkv, o)
                b = _fino_bnd(H, stream, y, torch.float32)
                return o, y, b
            call()  # warm kernel attributes / allocator outside capture
            torch.cuda.synchronize()
            before = int(H["cnt"][1023])
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                o, y, b = call()
            # Capture itself does not advance e. Three consecutive replays exercise both banks and reuse.
            assert int(H["cnt"][1023]) == before
            for replay in range(3):
                gr.replay()
                torch.cuda.synchronize()
                epoch = int(H["cnt"][1023])
                assert epoch == before + (replay + 1) * H["bit"]
                xo = torch.empty_like(ship_o)
                exact = torch.empty_like(ship_o)
                if H["bit"]:
                    H["mod"].fino_probe(H["po"], H["pl"], H["cnt"], qkv, xo, exact, KS1_EPS, 1)
                    torch.cuda.synchronize()
                    assert torch.equal(xo, exact), "finalize 32-lane tree / BF16 rounding mismatch"
                    # The NEXT bank must have been completely cleared; current bank stays for GD.
                    bank = epoch & 1
                    assert int(torch.count_nonzero(H["po"][bank * 1024:(bank + 1) * 1024])) == 0
                    assert int(torch.count_nonzero(H["pl"][bank * 16:(bank + 1) * 16])) == 0
                else:
                    xo = o
                    assert int(torch.count_nonzero(H["po"])) == int(torch.count_nonzero(H["pl"])) == 0
                assert int(torch.count_nonzero(H["cnt"][:1023])) == 0
                # Isolate FINO's register remapping from MG1's atomic scheduling noise.
                mapped_y = _gd(state, "o", xo)
                mapped_b = _sbb1_call(_SBB1["st"], stream, y, torch.float32, 1)
                torch.cuda.synchronize()
                assert torch.equal(y, mapped_y), "GD register fragment differs from ship on SAME finalized x"
                assert all(torch.equal(a, z) for a, z in zip(b, mapped_b)), "SBB arithmetic changed"
                er, mr = _fino_rel(xo, ship_o)
                worst, maxrel = max(worst, er), max(maxrel, mr)
                eq = torch.equal(xo, ship_o) and torch.equal(y, ship_y) and all(torch.equal(a, z) for a, z in zip(b, ship_b))
                bits += int(eq); count += 1
                # Stock MG1 can reorder atomic additions. Preserve the stock KS1 correctness envelope.
                same = _fino_rel(xo, ref)[0] <= 1.25 * e1 + 5e-3
                same &= er <= 5e-3 and float((xo.float() - ship_o.float()).abs().max()) <= 0.02 * float(ship_o.float().abs().max()) + 1e-3
                same &= torch.equal(kb, d["k_cache"].view(torch.uint8)) and torch.equal(vb, d["v_cache"].view(torch.uint8))
                same &= bool(torch.isfinite(y).all()) and all(bool(torch.isfinite(z).all()) for z in b)
                same &= _fino_rel(y, ship_y)[0] <= 5e-3 and all(_fino_rel(a, z)[0] <= 5e-3 for a, z in zip(b, ship_b))
                same &= eq or not FINO_REQUIRE_BITWISE
                good &= bool(same)
                print(f"[fino] arm {key} ctx={ctx} replay={replay + 1} e={epoch} bitwise={bool(eq)} rel={er:.4e} max_rel={mr:.4e} {'ok' if same else 'FAIL -> OFF'}", flush=True)
        H["test"] = {"bitwise": bits, "cases": count, "max_norm_rel_ship": worst, "max_element_rel_ship": maxrel, "ok": good}
        return good
    finally:
        FINO = saved
        H["ok"] = False
        # CPU prepare-side reset only; timed calls NEVER reset the epoch or either active bank.
        H["po"].zero_(); H["pl"].zero_(); H["cnt"].zero_()
# FINO LAB END
'''


def generate(src):
    c = constants(src)
    ks, gd, sb, decl = cuda_parts(c)
    # Binding only the d1s entrypoints keeps compilation scoped; control instantiation is XP6.
    cpp = span(c["KS1_CPP_SRC"], "#include <torch/extension.h>", "void ks3_forward(") + decl + GD_DECL
    # Plain top-level source, not module creation; avoid duplicating ~70 KB of shipped CUDA in the lab.
    helpers = "\n\n".join(inspect.getsource(f) for f in (once, span, cuda_parts))
    for a, b in (("once(", "_fino_once("), ("span(", "_fino_span("), ("cuda_parts(", "_fino_cuda_parts(")):
        helpers = helpers.replace(a, b)
    payload = ("\n" + helpers + '\n_FINO_KS, _FINO_GD, _FINO_SB, _FINO_DECL = _fino_cuda_parts({'
               '"KS1_CUDA_SRC": KS1_CUDA_SRC, "GD_CUDA_SRC": GD_CUDA_SRC, "SBB1_CU": SBB1_CU, '
               '"SBB1_HOST": SBB1_HOST, "SBB1_DECL": SBB1_DECL})\n'
               + "\n_FINO_HOST = " + repr(GD_HOST) + "\n_FINO_CPP = " + repr(cpp) + "\n" + LAB)
    out = once(src, "def prepare(weights):\n", "def prepare(weights):\n")
    out = once(out, " _rq_prepare(state)\n builds=", " _rq_prepare(state)\n _fino_build()\n _fino_prepare(state)\n builds=")
    # Never fall back after FINO has red.added: swallowing an exception would use an unwritten attention tensor.
    out = once(out, ' o=_hya_try(qkv,data,cfg,state,True)\n ks1_ran=False',
               ' fino=None\n fino_candidate=_fino_pick(qkv,data,state)\n o=_hya_try(qkv,data,cfg,state,True)\n ks1_ran=False')
    old = ' if o is None and _ks1_wanted(qkv,adata,adata["config"],state):\n  try:'
    new = ''' if o is None and fino_candidate is not None:
  fino=fino_candidate
  o=_fino_attn(fino,qkv,data,state)
  ks1_ran=True
 if o is None and _ks1_wanted(qkv,adata,adata["config"],state):
  try:'''
    out = once(out, old, new)
    out = once(out, ' attn_output=_rq_gemm(state,"o",o)',
               ' attn_output=_fino_o(fino,qkv,o)if fino is not None else _rq_gemm(state,"o",o)')
    out = once(out, ' ffn_input,stream,router_input=_v9_bnd("attn",data["stream"],attn_output,\n None if split_router else torch.float32,state,tokens)',
               ''' if fino is not None:
  ffn_input,stream,router_input=_fino_bnd(fino,data["stream"],attn_output,None if split_router else torch.float32)
 else:
  ffn_input,stream,router_input=_v9_bnd("attn",data["stream"],attn_output,
  None if split_router else torch.float32,state,tokens)''')
    # Exercise the actual patched attention dispatch on P64 during prepare (no duplicated routing logic).
    dispatch = span(out, ' fino=None\n fino_candidate=', ' attn_output=_fino_o(')
    payload += ('\ndef _fino_attention_test(qkv,data,state):\n cfg=data["config"]\n tokens=cfg["tokens"]\n'
                ' hq,d=cfg["num_q_heads"],cfg["head_dim"]\n'
                ' adata=_r2_try(data,cfg,state)if(R2_PAGE1 and cfg["page_size"]!=1)else data\n'
                + dispatch + '\n assert fino is None, "P64 must stay on shipped fallback"\n return o\n')
    out += payload
    ast.parse(out)
    compile(out, str(OUT), "exec")  # no pycache or additional files
    assert len(out.encode("utf-8")) < 1_000_000, "FINO lab exceeds submission size limit"
    return out


def main():
    src = BASE.read_text(encoding="utf-8")
    out = generate(src)
    if "--check" in sys.argv:
        assert OUT.read_text(encoding="utf-8") == out, "generated lab stale"
    else:
        OUT.write_bytes(out.encode("utf-8"))
    print("FINO generator PASS bytes=%d sha256=%s default=0" % (len(out.encode()), hashlib.sha256(out.encode()).hexdigest()))


if __name__ == "__main__":
    main()

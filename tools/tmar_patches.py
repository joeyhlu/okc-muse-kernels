"""tmar_patches.py: VML split shares through shared memory + a dedicated writer warp's 1-D TMA bulk reduce-add (VM_TMAR).

Source technique: FlashAttention-3 backward, hopper/copy_sm90_bulk_reduce.hpp (SM90_BULK_REDUCE_ADD =
cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32) at its use site hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp
(store_dq): the MMA warpgroups write their fp32 dQ partials to smem, fence.proxy.async, arrive on a "full" barrier and go back
to their GEMMs; one extra warp waits "full", one lane issues the bulk reduce, waits bulk_group .read, arrives "empty".

VML today (k3_gc_1): at a split share (a unit whose 16 K stages are spread over CTAs) every consumer thread red.adds its
partial from registers, then bar_sync, thread 0 atom.acq_rel on the per-(slot, WG) counter (its release waits for the reds),
bar_sync, and the last arriver fences, reads the sum back and runs the epilogue (SwiGLU -> xB + doneA release in phase A, y
rows in phase B). The whole sequence stalls the WG's wgmma stream: 0.8-1.5 us per share (moe_t32plus.md s4).

VM_TMAR = 1 (variant "t", runtime bits A.flags 1 << 20 = phase A, 1 << 21 = phase B):
* split shares of jobs with <= 16 tokens (N <= 16, 2 n-tiles: 4 KB per WG, 16 KB per CTA of staging; N > 16 keeps the shipped
  path, so every share of one unit uses one path) are committed by the WG as: wait empty[q] (previous share read), st.shared of
  d[] in the fix layout, fence.proxy.async.shared::cta, __syncwarp, lane 0 arrives full[q] (4 warps). The WG returns at once.
* a 19th warp (warp NCW + 2; 608 threads keep the 96-register cap: still <= 5 warps per SMSP) enumerates the CTA's segments
  exactly as phase() does; per async share: wait full[0..3], lane 0 issues 4 bulk reduce-adds (one per WG slice) into the
  shipped fix slots, commit, wait_group.read 0 -> arrive empty[0..3], wait_group 0, fence.proxy.async.global, fence.acq_rel,
  4 relaxed atomic adds on the shipped counters (one round trip); for each last WG slice the warp fences (acquire), reads the
  sum back (ld.cg), zeroes the slot, resets the counter and runs that slice's epilogue (phase A: SwiGLU -> xB, fence.proxy.async,
  red.release doneA += slices; phase B: y rows). Same counters, slots, epilogue arithmetic and last-arriver rule as shipped.
* the 1-D bulk form needs no tensor map (nothing baked in prepare); all waits are the bounded mbar_wait (err flag, no trap).
VM_TMAR = 0 compiles to the shipped code (every change is behind #if VM_TMAR).
"""

NL = chr(10)

TM_DEFS = r"""#ifndef VM_TMAR
#define VM_TMAR 0   // ext (tools/tmar_patches.py): N <= 16 split shares -> smem staging + a writer warp's TMA bulk reduce-add
#endif
"""

TM_SMEM = r"""#if VM_TMAR
constexpr int TM_QB = 4096;                                              // ext: staging per WG, 2 n-tiles x 128 threads x 16 B
constexpr int TM_STG = (TAB_OFF + (int)sizeof(Tab) + 127) / 128 * 128;   // 4 x TM_QB fp32 partials in the fix layout
constexpr int TM_BAR = TM_STG + 4 * TM_QB;                               // mbarriers full[4] (4 warps each), empty[4] (writer)
constexpr int SMEM = TM_BAR + 64 + 64;                                   // + 16 per-warp async share counters
static_assert(SMEM <= 232448, "ext: VML smem above the 227 KB opt-in limit");
constexpr int TM_BITA = 1 << 20, TM_BITB = 1 << 21;                      // A.flags: async phase-A / phase-B split shares
#else
constexpr int SMEM = TAB_OFF + (int)sizeof(Tab);
#endif
"""

TM_FN = r"""#if VM_TMAR
__device__ __forceinline__ uint32_t tm_full(uint8_t* smem, int q) { return sa(smem + TM_BAR + 8 * q); }
__device__ __forceinline__ uint32_t tm_empty(uint8_t* smem, int q) { return sa(smem + TM_BAR + 32 + 8 * q); }
template <int N>
__device__ __forceinline__ void tm_commit(uint8_t* smem, int q, int wp, int lane, const float* d, int* err) {
int* tk = reinterpret_cast<int*>(smem + TM_BAR + 64) + (q * 4 + wp);
if (lane == 0) {
const int k = *tk;
if (k > 0) mbar_wait(tm_empty(smem, q), (uint32_t)((k - 1) & 1), err);
*tk = k + 1;
}
__syncwarp();
float* sg = reinterpret_cast<float*>(smem + TM_STG + q * TM_QB) + (((wp << 5) | lane) << 2);
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
*reinterpret_cast<float4*>(sg + nt * 512) = make_float4(d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]);
asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
__syncwarp();
if (lane == 0) mbar_arrive(tm_full(smem, q));
}
__device__ __forceinline__ int tm_atom_add(int* p, int v) {
int o;
asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(o) : "l"(p), "r"(v) : "memory");
return o;
}
__device__ void tm_writer(const Args& A, const Tab& T, uint8_t* smem, int lane) {
const int c = blockIdx.x, G = gridDim.x, g = lane >> 2, t = lane & 3;
uint32_t k = 0;
for (int pb = 0; pb < 2; ++pb) {
if (!(A.flags & (pb ? TM_BITB : TM_BITA))) continue;
const int g0 = pb ? T.bB[c] : T.bA[c], g1 = pb ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, pb, g0, g1);
#pragma unroll 1
for (int gs = sp, ge = g1; ;) {
if (gs >= ge) {
if (ge == g1 && sp > g0) { gs = g0; ge = sp; } else break;
}
const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
gs += s1 - s0;
const int j = pb ? (u >> 4) : (u >> 5), cn = T.jC[j];
if ((s0 == 0 && s1 == 16) || cn > 16) continue;
const int slot = cta_of(T, pb, u << 4, G), nn = (cn + 7) >> 3, ns = s1 - s0;
float* fb0 = A.fix + (size_t)((pb * SLOTS + slot) * 4) * FIXW;
int* fc0 = A.cnt + (pb * SLOTS + slot) * 4;
#pragma unroll 1
for (int q = 0; q < 4; ++q) mbar_wait(tm_full(smem, q), k & 1u, A.err);
int last = 0;
if (lane == 0) {
#pragma unroll 1
for (int q = 0; q < 4; ++q)
asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
:: "l"(fb0 + (size_t)q * FIXW), "r"(sa(smem + TM_STG + q * TM_QB)), "r"((uint32_t)nn * 2048u) : "memory");
asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
#pragma unroll 1
for (int q = 0; q < 4; ++q) mbar_arrive(tm_empty(smem, q));
asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
asm volatile("fence.proxy.async.global;\n" ::: "memory");
fence_acq_rel();
const int o0 = tm_atom_add(fc0, ns), o1 = tm_atom_add(fc0 + 1, ns), o2 = tm_atom_add(fc0 + 2, ns), o3 = tm_atom_add(fc0 + 3, ns);
last = (o0 + ns == 16 ? 1 : 0) | (o1 + ns == 16 ? 2 : 0) | (o2 + ns == 16 ? 4 : 0) | (o3 + ns == 16 ? 8 : 0);
}
++k;
last = __shfl_sync(0xffffffffu, last, 0);
if (last == 0) continue;
__syncwarp();
fence_acq_rel();
const int e = T.jE[j];
const float gsc = __ldg((pb ? A.gsc2 : A.gsc13) + e);
const int ub = pb ? (u & 15) : (u & 31), npad = (cn + 7) & ~7;
#pragma unroll 1
for (int q = 0; q < 4; ++q) {
if (!((last >> q) & 1)) continue;
float* fb = fb0 + (size_t)q * FIXW;
float4 v[4][2];
#pragma unroll
for (int wp = 0; wp < 4; ++wp)
#pragma unroll
for (int nt = 0; nt < 2; ++nt)
v[wp][nt] = (nt < nn) ? __ldcg(reinterpret_cast<const float4*>(fb + nt * 512 + (((wp << 5) | lane) << 2)))
: make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
for (int wp = 0; wp < 4; ++wp) {
#pragma unroll
for (int nt = 0; nt < 2; ++nt)
if (nt < nn) *reinterpret_cast<float4*>(fb + nt * 512 + (((wp << 5) | lane) << 2)) = make_float4(0.f, 0.f, 0.f, 0.f);
const int tg = (2 * ub + (q >> 1)) * 8 + (q & 1) * 4 + wp;
if (pb == 0) {
const int col = 8 * tg + g, s2 = col >> 8, kk = col & 255;
const int i = kk >> 6, tt = (kk >> 4) & 3, qq = (kk >> 1) & 7, b = kk & 1;
__nv_bfloat16* xb = A.xB + (size_t)T.jrow[j] * KDIM + (size_t)s2 * npad * 256;
#pragma unroll
for (int nt = 0; nt < 2; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
if (nt < nn && n < cn) {
const float gv = hh ? v[wp][nt].y : v[wp][nt].x, uv = hh ? v[wp][nt].w : v[wp][nt].z;
xb[((((nt * 4 + i) * 8 + qq) * 8 + (2 * t + hh)) * 4 + tt) * 2 + b] = __float2bfloat16_rn(swiglu(gv * gsc, uv * gsc));
}
}
} else {
const int h0 = 16 * tg + g;
float* yb = A.y + (size_t)T.jS[j] * KDIM;
#pragma unroll
for (int nt = 0; nt < 2; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
if (nt < nn && n < cn) {
yb[(size_t)n * KDIM + h0] = (hh ? v[wp][nt].y : v[wp][nt].x) * gsc;
yb[(size_t)n * KDIM + h0 + 8] = (hh ? v[wp][nt].w : v[wp][nt].z) * gsc;
}
}
}
}
if (lane == 0) fc0[q] = 0;
}
if (pb == 0) {
fence_proxy_async();
__syncwarp();
if (lane == 0) red_release_add(A.doneA + j * 16 + (ub >> 1), __popc(last));
}
}
}
}
#endif
"""

SEG_SIG = ("template <bool PB, int N>\n__device__ __forceinline__ void seg(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, "
           "int& it, int u, int s0, int s1,\n")

CUDA_PATCHES = [
    ("constexpr int NCW = 16;\nconstexpr int NTH = 32 * NCW + 64;\n",
     TM_DEFS + "constexpr int NCW = 16;\nconstexpr int NTH = 32 * NCW + 64 + (VM_TMAR ? 32 : 0);   // ext: + the writer warp\n",
     "NTH + VM_TMAR default"),
    ("constexpr int SMEM = TAB_OFF + (int)sizeof(Tab);\n", TM_SMEM, "SMEM: staging + barriers"),
    (SEG_SIG, TM_FN + SEG_SIG, "tm_commit / tm_writer before seg"),
    ("if (!(s0 == 0 && s1 == 16)) {\nconst int wt = (wp << 5) | lane;\n",
     "if (!(s0 == 0 && s1 == 16)) {\n#if VM_TMAR\nif constexpr (N <= 16) {\n"
     "if (A.flags & (PB ? TM_BITB : TM_BITA)) { tm_commit<N>(smem, q, wp, lane, d, A.err); return; }\n}\n#endif\n"
     "const int wt = (wp << 5) | lane;\n",
     "seg: async commit of N <= 16 split shares"),
    ("for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }\n",
     "for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }\n"
     "#if VM_TMAR\nfor (int i = 0; i < 4; ++i) { mbar_init(sa(smem + TM_BAR + 8 * i), 4); mbar_init(sa(smem + TM_BAR + 32 + 8 * i), 1); }\n"
     "for (int i = 0; i < 16; ++i) reinterpret_cast<int*>(smem + TM_BAR + 64)[i] = 0;\n#endif\n",
     "vm_main: TM barrier init"),
    ("if (warp >= NCW) {\nif (warp == NCW) {\n",
     "if (warp >= NCW) {\n#if VM_TMAR\nif (warp == NCW + 2) { tm_writer(A, T, smem, lane); return; }\n#endif\nif (warp == NCW) {\n",
     "vm_main: writer warp role"),
]

# ---------------------------------------------------------------- Python glue (layer file, whole text) -------------------
PREP_WRAP = '''

TMAR_SELF_TEST_FLAGS = (1 << 20, 0)   # ext: flag sets the "t" variant must also pass (phase-A-only arm, shipped protocol)


def _vml_tmar_good():
    return bool(_VML["ok"]) and all(_VML["ok"].get(r) for r, on in VML_BUCKETS.items() if on)


def _vml_prepare(st, w, dev):
    """ext (tools/tmar_patches.py): the "t" variant (VM_TMAR split shares) is self-tested under VML_FLAGS and under every
    TMAR_SELF_TEST_FLAGS set; any failure -> loud FALLBACK to the shipped variant "s" (which is self-tested again)."""
    global VML_VAR
    _VML["tm_ok"] = VML_VAR == "t"
    _vml_prepare_one(st, w, dev)
    if VML_VAR != "t" or not VML_ENABLE:
        return
    good = "t" in _VML["mods"] and _vml_tmar_good()
    if good:
        saved = dict(VML_FLAGS)
        try:
            for extra in TMAR_SELF_TEST_FLAGS:
                for r in VML_BUCKETS:
                    f0 = saved.get(r)
                    VML_FLAGS[r] = ((_vml_FLAGS_DEFAULT if f0 is None else f0) & ~(3 << 20)) | extra
                _vml_prepare_one(st, w, dev)
                good = _vml_tmar_good()
                print(f"[ext] TMAR self-test flags {VML_FLAGS}: {'ok' if good else 'FAIL'}", flush=True)
                if not good:
                    break
        finally:
            VML_FLAGS.clear()
            VML_FLAGS.update(saved)
    if good:
        print(f"[ext] TMAR (VML split shares via smem + writer-warp TMA bulk reduce-add) LIVE, VML_FLAGS {VML_FLAGS}", flush=True)
        return
    print("[ext] TMAR self-test FAILED or not built: FALLBACK to the shipped VML variant 's'", flush=True)
    _VML["tm_ok"] = False
    VML_VAR = "s"
    if "s" in _VML["mods"]:
        _VML["mod"] = _VML["mods"]["s"]
    _vml_prepare_one(st, w, dev)
'''

PY_PATCHES = [
    ('_VML_DEFS = {"": (), "o": ("-DVM_TRICKS=0",), "s": ("-DVM_TRICKS_DQ=0",)}\n',
     '_VML_DEFS = {"": (), "o": ("-DVM_TRICKS=0",), "s": ("-DVM_TRICKS_DQ=0",), "t": ("-DVM_TRICKS_DQ=0", "-DVM_TMAR=1")}\n',
     "_VML_DEFS: variant t"),
    ("VML_VAR = 's'\nVML_ABVARS = ()\n", "VML_VAR = 't'\nVML_ABVARS = ('s',)\n", "VML_VAR t, shipped s built too"),
    ("VML_FLAGS = {128: 193, 256: 193, 32: 193}\n", "VML_FLAGS = {128: 3145921, 256: 3145921, 32: 3145921}\n",
     "VML_FLAGS: 193 | phase A / B async bits"),
    ('_VML = {"mod": None, "mods": {}, "sc": None, "ok": {}, "err": None}\n',
     '_VML = {"mod": None, "mods": {}, "sc": None, "ok": {}, "err": None, "tm_ok": False}\n', "_VML tm_ok"),
    ('''def _vml_build_job():
    for v in (VML_VAR,) + tuple(a for a in VML_ABVARS if a != VML_VAR):
        _VML["mods"][v], _ = _vml_build(defines=_VML_DEFS[v])
    _VML["mod"] = _VML["mods"][VML_VAR]
''', '''def _vml_build_job():
    vs = (VML_VAR,) + tuple(a for a in VML_ABVARS if a != VML_VAR)
    errs = {}

    def one(v):
        try:
            _VML["mods"][v], _ = _vml_build(defines=_VML_DEFS[v])
        except Exception as ex:
            errs[v] = ex
    ths = [threading.Thread(target=one, args=(v,), name=f"vml-{v}", daemon=True) for v in vs]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    for v, ex in errs.items():
        print(f"[ext] VML variant {v!r} build FAILED ({ex!r})", flush=True)
    if VML_VAR not in _VML["mods"]:
        if len(errs) == len(vs):
            raise errs[VML_VAR]
        alt = next(v for v in vs if v in _VML["mods"])
        print(f"[ext] FALLBACK: VML variant {VML_VAR!r} not built -> {alt!r}", flush=True)
        _VML["mod"] = _VML["mods"][alt]
        return
    _VML["mod"] = _VML["mods"][VML_VAR]
''', "_vml_build_job: variants in parallel threads, fallback"),
    ('''    rep = _k2a_bucket_of(x.shape[0])
    return _vml_call(_VML["mods"].get(VML_VAR, _VML["mod"]), st, _VML["sc"], x, data["topk_ids"].to(torch.int32).contiguous(),
''', '''    rep = _k2a_bucket_of(x.shape[0])
    var = VML_VAR if (VML_VAR != "t" or _VML.get("tm_ok")) else "s"
    return _vml_call(_VML["mods"].get(var, _VML["mod"]), st, _VML["sc"], x, data["topk_ids"].to(torch.int32).contiguous(),
''', "_vml_route: t only while its self-test holds"),
    ('def _vml_prepare(st, w, dev):\n    _VML["ok"] = {}\n    if not VML_ENABLE:\n        return\n',
     'def _vml_prepare_one(st, w, dev):\n    _VML["ok"] = {}\n    if not VML_ENABLE:\n        return\n', "rename _vml_prepare_one"),
    ("\nK2A36_ENABLE = True\n", PREP_WRAP + "\nK2A36_ENABLE = True\n", "_vml_prepare wrapper (self-test + fallback)"),
]


def cuda_span(text, name="_vml_CUDA_T"):
    a = text.index(name + ' = r"""')
    b = text.index('"""', a + len(name) + 6)
    return a, b


def sub1(text, old, new, what):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"[tmar_patches] anchor for {what}: found {n}")
    return text.replace(old, new)


def apply(text):
    """VML CUDA patches inside the _vml_CUDA_T string only, then the Python glue patches; returns (text, report lines)."""
    rep = []
    a, b = cuda_span(text)
    cu = text[a:b]
    for old, new, what in CUDA_PATCHES:
        cu = sub1(cu, old, new, f"CUDA {what}")
        rep.append(f"CUDA {what}")
    text = text[:a] + cu + text[b:]
    for old, new, what in PY_PATCHES:
        text = sub1(text, old, new, f"PY {what}")
        rep.append(f"PY {what}")
    return text, rep

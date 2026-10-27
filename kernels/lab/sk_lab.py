"""SK lab (MoE skeleton at T = 128 / 256, notes/study/moe_t256_skeleton.md): the shipped K2A text of the layer
(kernels/muse_decode_layer/k3_gd_1.py: K2A33_KERNEL_SRC, i.e. v3.3 HOTCAP vid 2 at T = 256 and v3.7 HK3 vid 5 at T = 128)
with the per-CTA start-up and the K2 split-item finalize rewritten as the same arithmetic on a parallel schedule.
LAB FILE: box runs only (tools/box/sk_run.py); never submit it, never import it from a scored file.

Measured (box_out/v2m, R6 per-CTA marks, vid 2, 6 layer routings): K1 entry -> gemm start 6.8-7.1 us (route()), gemm start
-> first weight copy 8.4-9.0 us (pick_role + wsk_range, serial per-thread loops over 32 experts with 64-bit divisions);
P0 runs the same route() first, so the first weight byte is requested ~25 us after the routing tail ends. K2 CTAs that
land when K1's last CTAs exit pay 25-35 us of the same start-up and are K2's critical path; the hot epilogue warp exits
up to ~37 us after its consumers (16 serial L2 round trips per 64-token split item).

Patches (each must hit its anchor exactly once; the computed plan is bit-identical by construction):
  FR   route(): a lane's ids are loaded once (independent loads, kept in registers for both passes) instead of 2 x L/32
       dependent loop trips; counts / prefixes / s_list / meta identical (same per-warp iteration order).
  FP   pick_role(): lane = expert + exact integer butterfly (as RS1 did for v3.6 / v3.7); same Gh.
  FW   wsk_range(): lane = expert weights, shuffle prefix, ballot search; same beg / end (64-bit only where needed).
  FF   MODE 2 last-contributor finalize (sum of squares of the split item's y rows): SK_FB tokens per L2 trip instead
       of 4 (same per-token arithmetic and atomics); v3.3 and v3.7 copies.
  FP0  P0's per-piece expert search: warp ballot over the hot prefix (idx >> 8 is warp-uniform) instead of a 32-step loop.
"""
import ast
import hashlib
import pathlib
import re
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
LAYER = ROOT / "kernels/muse_decode_layer/k3_gd_1.py"
CFLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
CASES = ("    case 0: launch3<2, 2, 2, 1>(p, (int)G, (int)G3, (int)mask); break;\n"
         "    case 1: launch37<4, 3>(p, (int)G, (int)G3, (int)mask); break;")
ATTRS = "    case 0: return attrs3<2, 2, 2, 1>();\n    case 1: return attrs37<4, 3>();"
VIDS = {256: 0, 128: 1}   # module case index per T (layer vids 2 / 5)
SK_FB = 8


SNAP = ROOT / "kernels/lab/sk_k2a33_gd1.json"   # snapshot of the three strings (the box's fresh sync omits k3_gd_1.py)


def shipped():
    """(K2A33_KERNEL_SRC, K2A33_CPP_SRC, wgmma specs) of the layer file (or its snapshot), without importing it."""
    import json
    if not LAYER.exists():
        d = json.loads(SNAP.read_text(encoding="utf-8"))
        return d["src"], d["cpp"], d["wg"]
    tree = ast.parse(LAYER.read_text(encoding="utf-8"))
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("K2A33_KERNEL_SRC", "K2A33_CPP_SRC")
                                                for t in node.targets):
            exec(compile(ast.Module([node], []), str(LAYER), "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name == "_k2a33_wg_specs":
            exec(compile(ast.Module([node], []), str(LAYER), "exec"), ns)
    return ns["K2A33_KERNEL_SRC"], ns["K2A33_CPP_SRC"], ns["_k2a33_wg_specs"]()


# ---------------------------------------------------------------- FR: route()
FR_OLD1 = """  const int NP = 8 * P.M;
  for (int i = tid; i < 8 * 32; i += blockDim.x) (&S.wc[0][0])[i] = 0;
  __syncthreads();
  const int L = ((NP + 255) / 256) * 32;
  const int p0 = warp * L;
  if (warp < 8) {
    for (int base = 0; base < L; base += 32) {
      const int p = p0 + base + lane;
      int e = -1;
      if (p < NP) {
        const int v = P.ids[p] - P.lo;
        e = (v >= 0 && v < 32) ? v : -1;
      }
      if (e >= 0) atomicAdd(&S.wc[warp][e], 1);
    }
  }
  __syncthreads();
"""
FR_NEW1 = """  const int NP = 8 * P.M;
  const int L = ((NP + 255) / 256) * 32;
  const int p0 = warp * L;
  constexpr int RQ = (P3 + 255) / 256;   // SK FR: a lane's ids, loaded once (independent loads) for both passes
  int ev[RQ];
#pragma unroll
  for (int q = 0; q < RQ; ++q) {
    const int p = p0 + 32 * q + lane;
    int e = -1;
    if (warp < 8 && 32 * q < L && p < NP) {
      const int v = __ldg(P.ids + p) - P.lo;
      e = (v >= 0 && v < 32) ? v : -1;
    }
    ev[q] = e;
  }
  for (int i = tid; i < 8 * 32; i += blockDim.x) (&S.wc[0][0])[i] = 0;
  __syncthreads();
  if (warp < 8) {
#pragma unroll
    for (int q = 0; q < RQ; ++q)
      if (ev[q] >= 0) atomicAdd(&S.wc[warp][ev[q]], 1);
  }
  __syncthreads();
"""
FR_OLD2 = """  if (warp < 8) {
    for (int base = 0; base < L; base += 32) {
      const int p = p0 + base + lane;
      int e = -1;
      if (p < NP) { const int v = __ldg(P.ids + p) - P.lo; e = (v >= 0 && v < 32) ? v : -1; }
      const unsigned m = __match_any_sync(FULL, e);
"""
FR_NEW2 = """  if (warp < 8) {
#pragma unroll
    for (int q = 0; q < RQ; ++q) {
      if (32 * q >= L) break;   // warp-uniform
      const int p = p0 + 32 * q + lane;
      const int e = ev[q];
      const unsigned m = __match_any_sync(FULL, e);
"""

FR2_NEW2 = """  if (warp < 8) {
    int ev2[RQ];
#pragma unroll
    for (int q = 0; q < RQ; ++q) {
      const int p = p0 + 32 * q + lane;
      int e = -1;
      if (32 * q < L && p < NP) { const int v = __ldg(P.ids + p) - P.lo; e = (v >= 0 && v < 32) ? v : -1; }
      ev2[q] = e;
    }
#pragma unroll
    for (int q = 0; q < RQ; ++q) {
      if (32 * q >= L) break;   // warp-uniform
      const int p = p0 + 32 * q + lane;
      const int e = ev2[q];
      const unsigned m = __match_any_sync(FULL, e);
"""
FR2_NEW1 = FR_NEW1.replace("loaded once (independent loads) for both passes", "one batch of independent loads per pass (FR2)")


# ---------------------------------------------------------------- FP: pick_role()
FP_OLD = """    long long Wh = 0, Wc = 0;
    for (int e = 0; e < 32; ++e) {
      const int c = S.cnt[e];
      if (c == 0) continue;
      if (c > hot_t) {
        const int full = c >> 6, rem = c & 63;
        Wh += (long long)full * (36 + 64) + (rem ? 36 + ((rem + 7) & ~7) : 0);
      } else {
        const int tp = 8 * NC, full = c / tp, rem = c - full * tp;
        Wc += (long long)full * (36 + tp) + (rem ? 36 + rem : 0);
      }
    }
"""
FP_NEW = """    long long Wh = 0, Wc = 0;
    {   // SK FP: lane = expert, exact integer butterfly (every warp; same sums as the serial loop)
      const int c = S.cnt[threadIdx.x & 31];
      if (c > hot_t && c > 0) {
        const int full = c >> 6, rem = c & 63;
        Wh = (long long)full * (36 + 64) + (rem ? 36 + ((rem + 7) & ~7) : 0);
      } else if (c > 0) {
        const int tp = 8 * NC, full = c / tp, rem = c - full * tp;
        Wc = (long long)full * (36 + tp) + (rem ? 36 + rem : 0);
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) { Wh += __shfl_xor_sync(FULL, Wh, o); Wc += __shfl_xor_sync(FULL, Wc, o); }
    }
"""

# ---------------------------------------------------------------- FW: wsk_range()
FW_START = "template <bool HOT, int NC, int NRB>\n__device__ __forceinline__ int wsk_pos(const Route& S, long long T) {"
FW_END = "// v3.3 warm-expert item (all of its stages s0..s1)"
FW_NEW = """template <bool HOT, int NC, int NRB>
__device__ __forceinline__ int wsk_pos_w(const Route& S, long long T, long long incl, int wrb) {
  // SK FW: lane = expert; incl = inclusive prefix of NRB * wrb over the role's experts (0 without stages), so the first
  // lane with T < incl is the expert the serial scan stopped at; the in-expert walk is the serial code's.
  const unsigned m = __ballot_sync(FULL, T < incl);
  if (m == 0u) return S.sb[32];
  const int e = __ffs(m) - 1;
  const int we = __shfl_sync(FULL, wrb, e);
  const long long ie = __shfl_sync(FULL, incl, e);
  int Tl = (int)(T - (ie - (long long)NRB * we));
  const int np = S.np[e], c = S.cnt[e];
  const int rb = Tl / we;
  Tl -= rb * we;
  for (int jj = 0; jj < np; ++jj) {
    const int w = (int)wsk_wpass<HOT, NC>(c, jj);
    if (Tl >= 16 * w) { Tl -= 16 * w; continue; }
    return S.sb[e] + (rb * np + jj) * 16 + Tl / w;
  }
  return S.sb[e] + (rb + 1) * np * 16;
}
template <bool HOT, int NC, int NRB>
__device__ __forceinline__ void wsk_range(const Route& S, int cta, int ncta, int& beg, int& end) {
  const int lane = threadIdx.x & 31;
  int wrb = 0;
  if (S.sb[lane + 1] != S.sb[lane]) {
    const int np = S.np[lane], c = S.cnt[lane];
    for (int jj = 0; jj < np; ++jj) wrb += 16 * (int)wsk_wpass<HOT, NC>(c, jj);
  }
  long long incl = (long long)NRB * wrb;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) { const long long nb = __shfl_up_sync(FULL, incl, d); if (lane >= d) incl += nb; }
  const long long W = __shfl_sync(FULL, incl, 31);
  beg = wsk_pos_w<HOT, NC, NRB>(S, W * cta / ncta, incl, wrb);
  end = (cta + 1 == ncta) ? S.sb[32] : wsk_pos_w<HOT, NC, NRB>(S, W * (cta + 1) / ncta, incl, wrb);
}

"""

# ---------------------------------------------------------------- FP0: P0 expert search
FP0_OLD = """  int tot = 0;
  for (int e = 0; e < 32; ++e) if (S.cnt[e] > P.hot_t) tot += S.cnt[e];
  uint32_t* xw = reinterpret_cast<uint32_t*>(P.xw);
  for (int idx = blockIdx.x * 256 + threadIdx.x; idx < tot * 256; idx += gridDim.x * 256) {
    int r = idx >> 8;
    const int piece = idx & 255, s2 = piece >> 4, i = (piece >> 2) & 3, t = piece & 3;
    int e = 0;
    for (; e < 32; ++e) {
      if (S.cnt[e] <= P.hot_t) continue;
      if (r < S.cnt[e]) break;
      r -= S.cnt[e];
    }
"""
FP0_NEW = """  const int lane = threadIdx.x & 31;   // SK FP0: lane = expert, hot-count prefix; idx >> 8 is warp-uniform
  const int hc = (S.cnt[lane] > P.hot_t) ? S.cnt[lane] : 0;
  int incl = hc;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, incl, d); if (lane >= d) incl += nb; }
  const int tot = __shfl_sync(FULL, incl, 31);
  uint32_t* xw = reinterpret_cast<uint32_t*>(P.xw);
  for (int idx = blockIdx.x * 256 + threadIdx.x; idx < tot * 256; idx += gridDim.x * 256) {
    int r = idx >> 8;
    const int piece = idx & 255, s2 = piece >> 4, i = (piece >> 2) & 3, t = piece & 3;
    const int e = __ffs(__ballot_sync(FULL, r < incl)) - 1;
    r -= __shfl_sync(FULL, incl - hc, e);
"""


def _once(src, old, new, name):
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"sk patch {name}: anchor found {n} times")
    return src.replace(old, new)


def _ff(src, fb=SK_FB):
    """MODE 2 finalize loops (v3.3 / v3.6 / v3.7 copies): SK_FB tokens per trip."""
    out, pos, n = [], 0, 0
    pat = re.compile(r"for \(int n0 = 0; n0 < ntok; n0 \+= 4\) \{\n(\s*)float4 yv\[4\];")
    for m in pat.finditer(src):
        stop = src.index("atomicAdd(P.ss + c0 + n0 + u, pz);", m.end())
        seg = src[m.start():stop]
        if seg.count("u < 4") != 2:
            raise RuntimeError("sk patch FF: unexpected loop body")
        seg = seg.replace("n0 += 4", "n0 += SK_FB").replace("yv[4]", "yv[SK_FB]").replace("u < 4", "u < SK_FB")
        out.append(src[pos:m.start()])
        out.append(seg)
        pos = stop
        n += 1
    if n != 3:
        raise RuntimeError(f"sk patch FF: {n} MODE 2 finalize loops (want 3)")
    out.append(src[pos:])
    src = "".join(out)
    return _once(src, "constexpr int M3 = 320;\n", f"constexpr int M3 = 320;\nconstexpr int SK_FB = {fb};   // SK FF\n", "FF const")


def patch(src, flags):
    if "FR" in flags:
        src = _once(src, FR_OLD1, FR_NEW1, "FR pass 1")
        src = _once(src, FR_OLD2, FR_NEW2, "FR pass 2")
    if "FR2" in flags:
        src = _once(src, FR_OLD1, FR2_NEW1, "FR2 pass 1")
        src = _once(src, FR_OLD2, FR2_NEW2, "FR2 pass 2")
    if "FP" in flags:
        src = _once(src, FP_OLD, FP_NEW, "FP")
    if "FW" in flags:
        a = src.index(FW_START)
        b = src.index(FW_END, a)
        if src.count(FW_START) != 1:
            raise RuntimeError("sk patch FW: anchor count")
        src = src[:a] + FW_NEW + src[b:]
    if "FF" in flags:
        src = _ff(src)
    if "FF16" in flags:
        src = _ff(src, 16)
    if "KW" in flags:   # vid 2 K2-only fine hot weight from probe bits 26-30 (0 = hot_w4, as shipped; read by K2 only)
        a = src.index("__global__ void __launch_bounds__(320, MINB) k2_kernel(const Params P) {")
        b = src.index("// P0 (v3.3)", a)
        seg = _once(src[a:b], "const bool hr = pick_role<NC>(S, P.hot_w4, P.hot_t, cta, ncta);",
                    "const int kw2 = (P.probe >> 26) & 31;   // SK KW: K2-only fine weight (w - 50) / 32\n"
                    "    const bool hr = pick_role<NC>(S, kw2 ? 50 + kw2 : P.hot_w4, P.hot_t, cta, ncta);", "KW")
        src = src[:a] + seg + src[b:]
    if "FP0" in flags:
        src = _once(src, FP0_OLD, FP0_NEW, "FP0")
    return src


ARMS = {"ship": (), "sk": ("FR", "FP", "FW", "FF", "FP0"), "skr": ("FR", "FP0"), "sknf": ("FR", "FP", "FW", "FP0"),
        "sk2": ("FR2", "FP", "FW", "FF", "FP0"), "skw": ("FP", "FW", "FF", "FP0"),
        "sk16": ("FR", "FP", "FW", "FF16", "FP0", "KW"), "skk": ("FR", "FP", "FW", "FF", "FP0", "KW")}


def source(arm):
    src, cpp, wg = shipped()
    src = patch(src, ARMS[arm])
    return src.replace("@CASES3@", CASES).replace("@ATTRS3@", ATTRS).replace("@WGSPECS@", wg), cpp


def build(arm, verbose=False):
    from torch.utils.cpp_extension import load_inline
    cu, cpp = source(arm)
    tag = hashlib.sha1((cpp + cu).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"sk_{arm}_{tag}", cpp_sources=cpp, cuda_sources=cu, functions=["run3", "attrs3v"],
                      extra_cuda_cflags=CFLAGS, verbose=verbose)
    return mod, time.time() - t0


def build_all(arms, log=print):
    """Parallel builds (one nvcc per arm). Returns {arm: module}."""
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(len(arms)) as ex:
        futs = {a: ex.submit(build, a) for a in arms}
        out = {}
        for a, f in futs.items():
            out[a], dt = f.result()
            log(f"[sk] built {a} in {dt:.0f} s: attrs v2 {list(out[a].attrs3v(0))} v5 {list(out[a].attrs3v(1))}")
    return out


if __name__ == "__main__":   # write the generated sources for an offline sm_90a compile check (+ refresh the snapshot)
    import json
    import sys
    src0, cpp0, wg0 = shipped()
    SNAP.write_text(json.dumps({"layer": LAYER.name, "src": src0, "cpp": cpp0, "wg": wg0}), encoding="utf-8", newline=chr(10))
    d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for a in ARMS:
        cu, cpp = source(a)
        (d / f"sk_{a}.cu").write_text(cu, encoding="utf-8", newline="\n")
        print(f"wrote {d / f'sk_{a}.cu'} ({len(cu)} chars)")


# ---------------------------------------------------------------- SK1: the T = 256-only rider (R-sk-1 / R-sk-2)
# FRH: FR as a separate function route_fr, called only by the HOTCAP instantiations of the v3.3 k1 / k2 kernels and by P0
# (route() and every other vid's code stay as shipped). FFH: FF (8 tokens per trip) in the v3.3 gemm's MODE 2 finalize
# for HOTCAP only (cold v3.3 variants keep 4); v3.6 / v3.7 finalizes untouched. FP / FW / FP0 measured ~0: not included.
ROUTE_HEAD = "template <int NC, int TPH = 64, bool HGE = false, bool LIST = true>\n__device__ void route("
CALL_K1 = "route<NC>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);"
CALL_K2 = "route<NC>(P, S, nullptr, s_list, 32, false);"
CALL_P0 = "route<NC>(P, S, nullptr, s_list, 64, false);"


def _frh(src):
    a = src.index(ROUTE_HEAD)
    b = src.index("\n}\n", a) + 3
    fn = src[a:b]
    fr = _once(_once(fn, FR_OLD1, FR_NEW1, "FRH pass 1"), FR_OLD2, FR_NEW2, "FRH pass 2")
    fr = fr.replace("__device__ void route(", "__device__ void route_fr(", 1)
    src = src[:b] + "// SK1 FRH: route() with one batch of id loads (HOTCAP kernels only)\n" + fr + src[b:]
    src = _once(src, CALL_K1, "if constexpr (HOTCAP) route_fr<NC>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && "
                "(P.probe & 15) == 0); else " + CALL_K1, "FRH k1")
    src = _once(src, CALL_K2, "if constexpr (HOTCAP) route_fr<NC>(P, S, nullptr, s_list, 32, false); else " + CALL_K2, "FRH k2")
    return _once(src, CALL_P0, "route_fr<NC>(P, S, nullptr, s_list, 64, false);", "FRH p0")


def _ffh(src):
    a = src.index("constexpr bool HOTCAP = (HM == 1);")   # the v3.3 gemm; its MODE 2 finalize is the first loop after it
    m = re.compile(r"for \(int n0 = 0; n0 < ntok; n0 \+= 4\) \{\n(\s*)float4 yv\[4\];").search(src, a)
    stop = src.index("atomicAdd(P.ss + c0 + n0 + u, pz);", m.end())
    seg = src[m.start():stop]
    if seg.count("u < 4") != 2:
        raise RuntimeError("sk patch FFH: unexpected loop body")
    seg = ("constexpr int FBX = HOTCAP ? 8 : 4;   // SK1 FFH: hot split-item finalize, 8 tokens per L2 trip\n"
           + m.group(1)[:-2] + seg.replace("n0 += 4", "n0 += FBX").replace("yv[4]", "yv[FBX]").replace("u < 4", "u < FBX"))
    return src[:m.start()] + seg + src[stop:]


_patch0 = patch


def patch(src, flags):   # noqa: F811
    src = _patch0(src, flags)
    if "FRH" in flags:
        src = _frh(src)
    if "FFH" in flags:
        src = _ffh(src)
    return src


ARMS["sk1"] = ("FRH", "FFH")


ALL_CASES = "\n".join("    case %d: %s(p, (int)G, (int)G3, (int)mask); break;" % (i, f) for i, f in enumerate(
    ("launch3<1, 3, 2, 0>", "launch3<2, 3, 2, 0>", "launch3<2, 2, 2, 1>", "launch36<4, 3>", "launch37<1, 3>", "launch37<4, 3>")))
ALL_ATTRS = "\n".join("    case %d: return %s();" % (i, f) for i, f in enumerate(
    ("attrs3<1, 3, 2, 0>", "attrs3<2, 3, 2, 0>", "attrs3<2, 2, 2, 1>", "attrs36<4, 3>", "attrs37<1, 3>", "attrs37<4, 3>")))


def source_all(arm):
    """All six layer vids (k3_gd_1's K2A33_VARIANTS order): for the codegen check of the other vids."""
    src, cpp, wg = shipped()
    src = patch(src, ARMS[arm])
    return src.replace("@CASES3@", ALL_CASES).replace("@ATTRS3@", ALL_ATTRS).replace("@WGSPECS@", wg), cpp

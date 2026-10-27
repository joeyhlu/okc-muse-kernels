"""M32 WK lab (R-m32-2; box only via tools/box/m32_run2.py): K2A vid 3 (HK2 NCH 4, NST 3) with WSK36's chunk-weighted hot cut
re-tested at ~zero start-up cost. v44's WSK36 lost +4.4..+7.7 us on bs16 layer routings where its cut equals ship's (serial
64-bit cut search on thread 0 at both kernels' starts); here warp 0 computes it in parallel (lane = expert, 32-bit, one
inclusive scan + ballot per cut). Probe bit 27 = weighted hot cut (stage weight 16 + hot_w4 (nch - 1), ranges36's own Gh
model, averaged over passes); bit off = shipped my_range2 cut. segs36 runs on the whole of warp 0 in this copy (all lanes
compute the same values; identical smem writes), with or without the bit, so the lab copy without the bit is the control.
Role split (Gh) unchanged; hot_w4 comes from the hot code as in ship.
"""
import ast
import hashlib
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
K2A_FILE = ROOT / "kernels/muse_moe_w4a16/k2a_v37.py"
WKB = 27   # probe bit

ANCHOR = "// Ranges of this CTA (all threads compute the same from S). hot_mode bit 0 = 1: dual range, CTA c takes\n"
HELPER = r"""// M32 lab (R-m32-2): WSK36's weighted hot cut computed by a full warp (lane = expert). Stage weight of expert e:
// ws_e / np_e with ws_e = sum over its passes of 16 + hot_w4 (nch - 1); cost_e = span_e / np_e * ws_e. CTA c of G takes
// [pos(W c / G), pos(W (c + 1) / G)), pos = the stage at which the cost prefix reaches the target.
__device__ __forceinline__ int m32_pos(const Route& S, int T, int W, int lane, int incl, int excl, int ws, int np) {
  if (T >= W) return S.sbh[32];
  const unsigned bal = __ballot_sync(0xffffffffu, incl > T);
  const int e = __ffs(bal) - 1;
  const int p = (lane == e) ? S.sbh[lane] + (int)(((long long)(T - excl) * np) / ws) : 0;
  return __shfl_sync(0xffffffffu, p, e);
}
__device__ __forceinline__ void m32_cut(const Params& P, const Route& S, int c, int G, int tp, int& beg, int& end) {
  const int lane = threadIdx.x & 31;
  const int span = S.sbh[lane + 1] - S.sbh[lane];
  int ws = 16, np = 1, cost = 0;
  if (span > 0) {
    np = S.np[lane];
    const int n = S.cnt[lane];
    ws = 0;
    for (int j = 0; j < np; ++j) { const int nt = min(tp, n - j * tp); ws += 16 + P.hot_w4 * (((nt + 7) >> 3) - 1); }
    cost = span / np * ws;
  }
  int incl = cost;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }
  const int W = __shfl_sync(0xffffffffu, incl, 31), excl = incl - cost;
  beg = (c == 0) ? 0 : m32_pos(S, (int)((long long)W * c / G), W, lane, incl, excl, ws, np);
  end = (c + 1 == G) ? S.sbh[32] : m32_pos(S, (int)((long long)W * (c + 1) / G), W, lane, incl, excl, ws, np);
}

"""
PATCHES = [
    ("m32 cut helper (before ranges36)", ANCHOR, HELPER + ANCHOR, 1),
    ("ranges36 hot cut",
     "  if (hr) my_range2(Sh, c, Gh, b0, e0);\n",
     "  if (hr) {\n"
     "    if ((P.probe >> 27) & 1) m32_cut(P, S, c, Gh, 8 * NCH, b0, e0);   // M32 lab: weighted hot cut (warp 0)\n"
     "    else my_range2(Sh, c, Gh, b0, e0);\n"
     "  }\n", 1),
    ("segs36 on warp 0 (K1 and K2)",
     "  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr);\n",
     "  if (threadIdx.x < 32) segs36<NCH>(P, S, s_pr);   // M32 lab: full warp (m32_cut uses shuffles)\n", 2),
]


def _shipped():
    tree = ast.parse(K2A_FILE.read_text(encoding="utf-8"))
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("KERNEL_SRC_V3", "CPP_SRC_V3") for t in node.targets):
            exec(compile(ast.Module([node], []), str(K2A_FILE), "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name == "_wg_specs":
            exec(compile(ast.Module([node], []), str(K2A_FILE), "exec"), ns)
    return ns["KERNEL_SRC_V3"], ns["CPP_SRC_V3"], ns["_wg_specs"]()


def source():
    src, cpp, wg = _shipped()
    for what, old, new, cnt in PATCHES:
        n = src.count(old)
        if n != cnt:
            raise RuntimeError(f"m32 wk patch {what!r}: anchor found {n} times, want {cnt}")
        src = src.replace(old, new)
    src = (src.replace("@CASES3@", "    case 0: launch36<4, 3>(p, (int)G, (int)G3, (int)mask); break;")
              .replace("@ATTRS3@", "    case 0: return attrs36<4, 3>();").replace("@WGSPECS@", wg))
    return src, cpp


def build():
    from torch.utils.cpp_extension import load_inline
    src, cpp = source()
    tag = hashlib.sha1((cpp + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"m32_wk_{tag}", cpp_sources=cpp, cuda_sources=src, functions=["run3", "attrs3v"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills",
                                         "-gencode=arch=compute_90a,code=sm_90a"])
    return mod, time.time() - t0


def call(mod, st, d, hot, bits=0):
    import torch
    import k2a_v37 as D
    x = d["hidden_states"].contiguous()
    out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
    mod.run3(0, 7, x, d["topk_ids"].to(torch.int32).contiguous(), d["topk_weights"].float().contiguous(),
             st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], st["h"], st["inter"], st["y"], st["meta"],
             st["cnt"], st["err"], out, st["sink"], st["ss"], st["cnt2"], st["xw"], st["xw2"],
             D.hot_code(D.V3_VARIANTS[3], hot), x.shape[0], d["config"]["local_start"], st["grid"][0], st["sms"] * 2,
             D.V3_HINT | bits)
    return out

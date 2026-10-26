"""K2A v4.4 lab (box only; tools/box/k2a_v44.py --lab=...): vid 3 (HK2, bs 16 / 32) with the role split's hot range cut
at equal cost-model weight (WSK36). Source = k2a_v37.KERNEL_SRC_V3 (== the K2A33 source in k3_v28.py for gemm36 / ranges36 /
segs36) + PATCHES; one module per copy, run3 / attrs3v with the shipped host signature (vid 3 as case 0).

WSK36: in ranges36's role split (hot_mode bit 0 = 0), the hot CTAs' Stream-K ranges over the hot stage sequence S.sbh
were equal stage counts; a stage of a hot pass with nq 8-token chunks costs ~ 16 + hot_w4 (nq - 1) sixteenths of a cold
stage (ranges36's own Gh model: hc_item<NQ> issues NQ x the mma / X loads per decoded kb), so passes of 32-token experts
and of 9-16-token experts got the same number of stages per CTA. WSK36 cuts at equal weight under that model. Only the cut
points move (same items, same math; split-item FP32 atomics land elsewhere). Cold range, Gh, dual mode: unchanged.
"""
import ast
import hashlib
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
K2A_FILE = ROOT / "kernels/muse_moe_w4a16/k2a_v37.py"

W36_ANCHOR = "// Ranges of this CTA (all threads compute the same from S). hot_mode bit 0 = 1: dual range, CTA c takes\n"
W36_HELPERS = r"""// v4.4 WSK36: the HK2 role split's hot range cut at equal cost-model weight (a stage of a hot pass with nq 8-token chunks
// weighs 16 + hot_w4 (nq - 1), ranges36's own Gh model) instead of equal stage counts. Only the cut points move.
constexpr int V44_WSK36 = @V44WSK@;
__device__ __forceinline__ long long w36_pass(const Params& P, int c, int j, int tp) {
  const int nt = min(tp, c - j * tp);
  return 16 + (long long)P.hot_w4 * (((nt + 7) >> 3) - 1);
}
__device__ __forceinline__ int w36_pos(const Params& P, const Route& S, long long T, int tp) {
  for (int e = 0; e < 32; ++e) {
    const int span = S.sbh[e + 1] - S.sbh[e];
    if (span == 0) continue;
    const int np = S.np[e], c = S.cnt[e], nrb = span / (16 * np);
    long long wrb = 0;
    for (int j = 0; j < np; ++j) wrb += 16 * w36_pass(P, c, j, tp);
    if (T >= wrb * nrb) { T -= wrb * nrb; continue; }
    const int rb = (int)(T / wrb);
    T -= rb * wrb;
    for (int j = 0; j < np; ++j) {
      const long long w = w36_pass(P, c, j, tp);
      if (T >= 16 * w) { T -= 16 * w; continue; }
      return S.sbh[e] + (rb * np + j) * 16 + (int)(T / w);
    }
    return S.sbh[e] + (rb + 1) * np * 16;
  }
  return S.sbh[32];
}
__device__ __forceinline__ void w36_range(const Params& P, const Route& S, int c, int G, int tp, int& beg, int& end) {
  long long W = 0;
  for (int e = 0; e < 32; ++e) {
    const int span = S.sbh[e + 1] - S.sbh[e];
    if (span == 0) continue;
    const int np = S.np[e], nrb = span / (16 * np);
    for (int j = 0; j < np; ++j) W += 16LL * nrb * w36_pass(P, S.cnt[e], j, tp);
  }
  beg = (c == 0) ? 0 : w36_pos(P, S, W * c / G, tp);
  end = (c + 1 == G) ? S.sbh[32] : w36_pos(P, S, W * (c + 1) / G, tp);
}

"""
W36_RANGE_OLD = ("  if (hr) my_range2(Sh, c, Gh, b0, e0);\n"
                 "  else my_range2(Sc, c - Gh, G - Gh, b0, e0);\n")
W36_RANGE_NEW = ("  if (hr) {\n"
                 "    if constexpr (V44_WSK36 != 0) w36_range(P, S, c, Gh, 8 * NCH, b0, e0);   // v4.4: equal-weight hot cut\n"
                 "    else my_range2(Sh, c, Gh, b0, e0);\n"
                 "  } else {\n"
                 "    my_range2(Sc, c - Gh, G - Gh, b0, e0);\n"
                 "  }\n")


def patches(wsk="@V44WSK@"):
    return [("v4.4 WSK36 helpers (before ranges36)", W36_ANCHOR, W36_HELPERS.replace("@V44WSK@", str(wsk)) + W36_ANCHOR),
            ("v4.4 WSK36: ranges36 hot role cut by weight", W36_RANGE_OLD, W36_RANGE_NEW)]


COPIES = {"wsk": 1, "base": 0}


def _shipped():
    tree = ast.parse(K2A_FILE.read_text(encoding="utf-8"))
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("KERNEL_SRC_V3", "CPP_SRC_V3") for t in node.targets):
            exec(compile(ast.Module([node], []), str(K2A_FILE), "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name == "_wg_specs":
            exec(compile(ast.Module([node], []), str(K2A_FILE), "exec"), ns)
    return ns["KERNEL_SRC_V3"], ns["CPP_SRC_V3"], ns["_wg_specs"]()


def source(wsk):
    src, cpp, wg = _shipped()
    for what, old, new in patches(wsk):
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"k2a_v44 patch {what!r}: anchor found {n} times")
        src = src.replace(old, new)
    src = (src.replace("@CASES3@", "    case 0: launch36<4, 3>(p, (int)G, (int)G3, (int)mask); break;")
              .replace("@ATTRS3@", "    case 0: return attrs36<4, 3>();").replace("@WGSPECS@", wg))
    return src, cpp


def build(names):
    from torch.utils.cpp_extension import load_inline
    mods = {}
    for nm in names:
        src, cpp = source(COPIES[nm])
        tag = hashlib.sha1((cpp + src).encode()).hexdigest()[:10]
        t0 = time.time()
        mods[nm] = load_inline(name=f"k2a_v44_{nm}_{tag}", cpp_sources=cpp, cuda_sources=src, functions=["run3", "attrs3v"],
                               extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills",
                                                  "-gencode=arch=compute_90a,code=sm_90a"])
        print(f"[v44] built {nm} in {time.time() - t0:.0f}s attrs {list(mods[nm].attrs3v(0))}", flush=True)
    return mods, list(mods)


def call(mods, nm, st, d, hot):
    import torch
    import k2a_v37 as D
    x = d["hidden_states"].contiguous()
    out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
    mods[nm].run3(0, 7, x, d["topk_ids"].to(torch.int32).contiguous(), d["topk_weights"].float().contiguous(),
                  st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], st["h"], st["inter"], st["y"], st["meta"],
                  st["cnt"], st["err"], out, st["sink"], st["ss"], st["cnt2"], st["xw"], st["xw2"],
                  D.hot_code(D.V3_VARIANTS[3], hot), x.shape[0], d["config"]["local_start"], st["grid"][0], st["sms"] * 2,
                  D.V3_HINT)
    return out

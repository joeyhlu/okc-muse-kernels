"""gen_k3_v09.py: k3_v09.py = k3_v08.py (as gen_k3_v08.py generates it) + v09 layer pieces, each behind a per-bucket
route table that falls back to k3_v08's own path (rules R2/R3). Budget: notes/layer_budget.md, box_out/lb9_analysis.md.

    python3 -I tools/gen_k3_v09.py            # writes kernels/muse_decode_layer/k3_v09.py, asserts the invariants
    python3 -I tools/gen_k3_v09.py --check    # same checks, compares with the file on disk, writes nothing

What changes vs k3_v08 (everything else is asserted ast-identical to gen_k3_v08's output):
  * K2A v3.1 (kernels/muse_moe_w4a16/k2a_v31.py) as the expert path: sources byte-identical, host pieces ast-identical
    after a K2A31_/_k2a31_ rename (its bitpos-aware pack_words / prepack_expert renamed too; v2.2's stay). Chain in
    _v8_experts: v3.1 -> v3 -> v2.2 -> Marlin; prepare builds v3 / v2.2 only for buckets the one before failed.
  * Boundaries: SGLang's own JIT fused_norm_residual with a per-bucket launch config (cluster size / threads) instead
    of the dispatcher's default (which never picks the DSMEM cluster kernel below compute capability 10).
  * Post-pooler tail: every global load issued before any use (one DRAM round trip), the PDL wait placed after the
    weight / bias loads, launched with programmatic stream serialization (V9_TAIL_PDL).
"""

import ast
import importlib.util
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "kernels/muse_decode_layer/k3_v09.py"
SRC_K2A31 = ROOT / "kernels/muse_moe_w4a16/k2a_v31.py"
NONCE = "k3-v09-1"

_s8 = importlib.util.spec_from_file_location("gen_k3_v08", ROOT / "tools/gen_k3_v08.py")
G8 = importlib.util.module_from_spec(_s8)
_s8.loader.exec_module(G8)
G = G8.G
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha

# ------------------------------------------------------------------------------------------------ route tables
K2A31_ROUTE = {1: "v31", 8: "v31", 16: "v31", 32: "v31", 128: "v31", 256: "v31"}
# (boundary, bucket rep) -> (cluster_size, launch_config) for SGLang's fused_norm_residual; None = k3_v08 (module
# default dispatch). Standalone box bench (tools/box/bnd_bench.py, graph replay, cold L2), µs default -> pick:
#   attn T1 3.39 -> c16 1.44, T8 3.78 -> c16 1.50, T16 2.53 -> c8 1.70, T32 2.53 -> c16 2.08, T128 3.46 (default),
#   T256 7.89 -> 1x1024pf 6.64; ffn T1 2.45 -> c16 1.25, T8 2.69 -> c16 1.50, T16 2.85 -> c8 1.73,
#   T32 2.88 -> c8 2.43, T128 3.82 -> 1x1024pf 3.62, T256 6.96 (default).
BND_ROUTE = {("attn", 1): (16, None), ("attn", 8): (16, None), ("attn", 16): (8, None), ("attn", 32): (16, None),
             ("attn", 256): (1, (1024, True)),
             ("ffn", 1): (16, None), ("ffn", 8): (16, None), ("ffn", 16): (8, None), ("ffn", 32): (8, None),
             ("ffn", 128): (1, (1024, True))}
TAIL_PDL = True
# box_out/ab_v09a.log: with the FP32 router (v08-2), plain side-stream placement beats side_first at T = 1 (1.011/1.006)
ROUTER_PLACE = {1: "side", 8: "side", 16: "side", 32: "side", 128: "side", 256: "main"}

# ------------------------------------------------------------------------------------------------ K2A v3.1 slice
K2A31_NAMES = ("BITPOS3", "pack_words", "prepack_expert", "M3", "SW3", "V3_VARIANTS", "V3_PLAN", "stage_repack",
               "KERNEL_SRC_V3", "CPP_SRC_V3", "build_ext_v3", "v3_prepare", "V3_HINT", "v3_call", "v3_reset",
               "V3_SELF_TEST_CASES")
K2A31_RENAME = {
    "BITPOS3": "K2A31_BITPOS3", "pack_words": "_k2a31_pack_words", "prepack_expert": "_k2a31_prepack_expert",
    "M3": "K2A31_M3", "SW3": "K2A31_SW3", "V3_VARIANTS": "K2A31_VARIANTS", "V3_PLAN": "K2A31_PLAN",
    "stage_repack": "_k2a31_stage_repack", "KERNEL_SRC_V3": "K2A31_KERNEL_SRC", "CPP_SRC_V3": "K2A31_CPP_SRC",
    "build_ext_v3": "_k2a31_build_ext", "v3_prepare": "_k2a31_prepare_st", "V3_HINT": "K2A31_HINT",
    "v3_call": "_k2a31_call", "v3_reset": "_k2a31_reset", "V3_SELF_TEST_CASES": "K2A31_SELF_TEST_CASES",
    # shared with v2.2 (asserted ast-identical between k2a_v31.py and k2a_v22.py)
    "E_LOC": "K2A_E_LOC", "D_HIDDEN": "K2A_D_HIDDEN", "D_EXPERT": "K2A_D_EXPERT", "W_SHIFT": "K2A_W_SHIFT",
    "META_HDR": "K2A_META_HDR", "TOP_K": "K2A_TOP_K", "BITPOS": "K2A_BITPOS", "reencode_scales": "_k2a_reencode_scales",
    "tile_row_perm": "_k2a_tile_row_perm",
}
_K2A31_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(sorted(map(re.escape, K2A31_RENAME), key=len, reverse=True))
                       + r")(?![A-Za-z0-9_])")
K2A31_SHARED = ("E_LOC", "D_HIDDEN", "D_EXPERT", "TOP_K", "POST_NORM_EPS", "M_MAX", "META_HDR", "BITPOS", "W_SHIFT",
                "reencode_scales", "tile_row_perm", "bucket_of", "BUCKETS", "_dequant", "ref_fp32", "rel_err",
                "synth_inputs")


def r31(text):
    return _K2A31_RE.sub(lambda m: K2A31_RENAME[m.group(1)], text)


def k2a31_glue():
    """The v3 glue of gen_k3_v08 (prepare + self-test + layer call), renamed for v3.1."""
    g = G8.K2A3_GLUE
    a = g.index("def _v8_k2a3_prepare")
    b = g.index("def _v8_experts")
    g = g[a:b]
    for old, new in (("_v8_k2a3_", "_v9_k2a31_"), ("V8_K2A3_ROUTE", "V9_K2A31_ROUTE"), ("_k2a3_", "_k2a31_"),
                     ("K2A3_", "K2A31_"), ('"k2a3"', '"k2a31"'), ("[k2a3]", "[k2a31]"), ("K2A v3", "K2A v3.1"),
                     ('v == "v3"', 'v == "v31"'), ("experts stay on K2A v2.2 / Marlin", "experts stay on K2A v3 / v2.2 / Marlin")):
        check(old in g, f"glue rename {old}")
        g = g.replace(old, new)
    return "\n" + g


def k2a31_stage(text):
    k22, k31 = Src(ROOT / "kernels/muse_moe_w4a16/k2a_v22.py"), Src(SRC_K2A31)
    for nm in K2A31_SHARED:
        check(dump(k22.nodes[nm]) == dump(k31.nodes[nm]), f"k2a_v31 {nm} differs from k2a_v22's")
    for nm in ("KERNEL_SRC_V3", "CPP_SRC_V3"):
        check(k31.seg(nm).startswith(f'{nm} = r"""') and '"""' not in k31.value(nm), f"{nm} literal form")
    parts = ["# K2A v3.1 slice: kernels/muse_moe_w4a16/k2a_v31.py (nonce k2a-v31-1), sources byte-identical, host code\n"
             "# renamed K2A31_/_k2a31_ (its bitpos-aware packers too; names shared with v2.2 map to the layer's K2A_ ones).\n"
             "V9_K2A31_ROUTE = " + repr(K2A31_ROUTE) + "   # bucket rep -> \"v31\" | anything else = K2A v3 / v2.2 / Marlin\n"
             "V9_BUILD_V3_ALWAYS = False   # lab A/B only: build K2A v3 even when v3.1 covers every bucket\n"]
    for nm in K2A31_NAMES:
        if nm in ("KERNEL_SRC_V3", "CPP_SRC_V3"):
            parts.append(K2A31_RENAME[nm] + k31.seg(nm)[len(nm):] + "\n")
        else:
            parts.append(r31(k31.seg(nm)) + "\n")
    parts.append(k2a31_glue())
    banner = ("# ==============================================================================================\n"
              "# Rung 1 (kernels/muse_decode_layer/r1_glue.py, nonce r1-1): unchanged, and the fallback of both pieces.\n")
    text = patch(text, banner, "".join(parts) + "\n\n" + banner, "K2A v3.1 section")
    # expert chain: v3.1 first
    text = patch(text, '''    rep = _k2a_bucket_of(tokens)
    k3s = state.get("k2a3")
''', '''    rep = _k2a_bucket_of(tokens)
    k31 = state.get("k2a31")
    if (k31 is not None and rep is not None and V9_K2A31_ROUTE.get(rep) == "v31" and k31["ok"].get(rep)
            and tokens <= K2A31_M3 and cfg["local_experts"] == K2A_E_LOC and cfg["top_k"] == K2A_TOP_K
            and cfg["n_experts"] == N_EXPERTS):
        try:
            return _v9_k2a31_call(expert_x, local_ids, topk_weights, k31["st"])
        except Exception as ex:  # noqa: BLE001
            _k3_disable(state, "k2a31", None, "K2A v3.1", ex)
    k3s = state.get("k2a3")
''', "_v8_experts chain")
    text = patch(text, '''    K2A v3 where V8_K2A3_ROUTE says "v3" and its self-test passed for the bucket, else K2A v2.2 per K2A_ROUTE,''',
                 '''    K2A v3.1 (V9_K2A31_ROUTE), else K2A v3 where V8_K2A3_ROUTE says "v3" and its self-test passed for the bucket,
    else K2A v2.2 per K2A_ROUTE,''', "_v8_experts doc")
    text = patch(text, '''    state["k2a3"] = None
    if USE_K2A:
        state["k2a3"] = _v8_k2a3_prepare(weights, t0)
        k3ok = state["k2a3"]["ok"] if state["k2a3"] is not None else {}
''', '''    state["k2a3"] = None
    state["k2a31"] = None
    if USE_K2A:
        state["k2a31"] = _v9_k2a31_prepare(weights, t0)
        k31ok = state["k2a31"]["ok"] if state["k2a31"] is not None else {}
        need_v3 = [rep for rep in V8_K2A3_ROUTE if not (V9_K2A31_ROUTE.get(rep) == "v31" and k31ok.get(rep))]
        state["k2a3"] = _v8_k2a3_prepare(weights, t0) if (need_v3 or V9_BUILD_V3_ALWAYS) else None
        k3ok = dict(state["k2a3"]["ok"]) if state["k2a3"] is not None else {}
        for rep, ok in k31ok.items():
            if ok and V9_K2A31_ROUTE.get(rep) == "v31":
                k3ok[rep] = True   # covered by v3.1: v2.2 is not needed for it
''', "prepare chain")
    text = patch(text, '''experts={'K2A v3' if state.get('k2a3') is not None else''',
                 '''experts={'K2A v3.1' if state.get('k2a31') is not None else 'K2A v3' if state.get('k2a3') is not None else''',
                 "prepare summary")
    return text


# ------------------------------------------------------------------------------------------------ boundaries
BND_GLUE = '''

def _v9_bnd(which, stream, branch, router_dtype, state, tokens):
    """A layer boundary (post-norm + gated residual + pre-norm [+ FP32 router input]) on SGLang's own JIT
    fused_norm_residual with this bucket's measured launch config (V9_BND_ROUTE); k3_v08's module call otherwise."""
    op = state["attn_boundary" if which == "attn" else "ffn_boundary"]
    cfgb = V9_BND_ROUTE.get((which, _k2a_bucket_of(tokens))) if state.get("v9bnd") else None
    if cfgb is None or _V9_FNR is None:
        return op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=router_dtype)
    cs, lc = cfgb
    return _V9_FNR(branch, stream, op._post_norm.eps, op._pre_norm.eps,
                   post_norm_is_input_scale=op._post_norm_is_input_scale,
                   post_weight=op._cc_post_weight if op._post_norm.has_weight else None,
                   pre_weight=op._cc_pre_weight if op._pre_norm.has_weight else None,
                   alpha=op._cc_alpha if op._use_residual_gate else None,
                   beta=op._cc_beta if op._use_residual_gate else None,
                   want_router=router_dtype is not None, launch_config=lc, cluster_size=cs)


def _v9_bnd_prepare(state):
    """Compile every routed launch config now (never under capture) and check it against the module's own output at
    each bucket's T (and T = 1, 5, 33, 200 inside the buckets). Returns True when all routed configs pass."""
    global _V9_FNR
    try:
        from sglang.kernels.ops.muse_spark_v12.norm_residual import fused_norm_residual
        _V9_FNR = fused_norm_residual
    except Exception as ex:  # noqa: BLE001
        print(f"[v9] fused_norm_residual unavailable -> module boundaries: {ex!r}", flush=True)
        return False
    dev = state["router"].device
    gen = torch.Generator(device=dev).manual_seed(909)
    try:
        for name in ("attn_boundary", "ffn_boundary"):
            state[name]._ensure_cached_weights(torch.bfloat16)
        for (which, rep), cfgb in sorted(V9_BND_ROUTE.items(), key=str):
            lo, hi = [(a, b) for a, b, r in K2A_BUCKETS if r == rep][0]
            for T in sorted({lo, rep, hi if hi <= 320 else rep}):
                op = state["attn_boundary" if which == "attn" else "ffn_boundary"]
                stream = torch.randn((T, HIDDEN), generator=gen, device=dev)
                branch = torch.randn((T, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
                rd = torch.float32 if (which == "attn" and T <= ROUTER_FP32_MAX_TOKENS) else None
                ref = op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=rd)
                state["v9bnd"] = True
                got = _v9_bnd(which, stream, branch, rd, state, T)
                torch.cuda.synchronize()
                e0 = float((got[0].float() - ref[0].float()).norm() / ref[0].float().norm())
                e1 = float((got[1] - ref[1]).norm() / ref[1].norm())
                e2 = float((got[2] - ref[2]).norm() / ref[2].norm()) if rd is not None else 0.0
                ok = e0 < 1e-2 and e1 < 1e-4 and e2 < 1e-2 and bool(torch.isfinite(got[1]).all())
                print(f"[v9] boundary self-test {which} T={T} cfg {cfgb}: out {e0:.1e} stream {e1:.1e} router {e2:.1e}"
                      f" -> {'ok' if ok else 'FAIL'}", flush=True)
                if not ok:
                    raise RuntimeError(f"boundary {which} T={T}")
    except Exception as ex:  # noqa: BLE001
        print(f"[v9] boundary self-test FAILED -> module boundaries: {ex!r}", flush=True)
        state["v9bnd"] = False
        return False
    state["v9bnd"] = True
    print(f"[v9] boundaries routed: {V9_BND_ROUTE}", flush=True)
    return True
'''


def bnd_stage(text):
    head = ("# Boundaries on SGLang's JIT fused_norm_residual with a measured launch config per (boundary, bucket rep):\n"
            "# (cluster_size, launch_config); a missing key = the module's default dispatch (k3_v08).\n"
            "V9_BND_ROUTE = " + repr(BND_ROUTE) + "\n_V9_FNR = None\n")
    banner = ("# ==============================================================================================\n"
              "# Rung 1 (kernels/muse_decode_layer/r1_glue.py, nonce r1-1): unchanged, and the fallback of both pieces.\n")
    text = patch(text, banner, head + BND_GLUE + "\n\n" + banner, "boundary section")
    text = patch(text, '''    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
''', '''    ffn_input, stream, router_input = _v9_bnd("attn", data["stream"], attn_output,
                                              None if split_router else torch.float32, state, tokens)
''', "kernel_fn attn boundary")
    text = patch(text, '''    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
''', '''    next_attn_input, stream, _ = _v9_bnd("ffn", stream, ffn_output, None, state, tokens)
''', "kernel_fn ffn boundary")
    text = patch(text, '''    state["v8"] = _v8_tail_prepare(weights, state)
''', '''    state["v8"] = _v8_tail_prepare(weights, state)
    state["v9bnd"] = _v9_bnd_prepare(state)
''', "prepare boundaries")
    return text


# ------------------------------------------------------------------------------------------------ tail: loads first + PDL
def tail_stage(text):
    old_front = text[text.index("  // warp 0 owns the routing: lane l holds experts 8l .. 8l+7"):
                     text.index("  float ss = 0.f;\n#pragma unroll\n  for (int i = 0; i < 8; ++i) ss += v[i] * v[i];")]
    new_front = r'''  // v9: every global load is issued before any use (one DRAM round trip); the weight / bias loads, which do not
  // depend on the previous kernel, go before the PDL wait. Warp 0 owns the routing: lane l holds experts 8l .. 8l+7.
  const uint4 wq = *(reinterpret_cast<const uint4*>(nw) + tx);
  float4 c0 = make_float4(0.f, 0.f, 0.f, 0.f), c1 = c0;
  if (warp == 0) {
    const float4* pbias = reinterpret_cast<const float4*>(bias) + lane * 2;
    c0 = pbias[0];
    c1 = pbias[1];
  }
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  float4 qa[NSMAX][2];
  float4 qb0 = make_float4(0.f, 0.f, 0.f, 0.f), qb1 = qb0;
  if (warp == 0) {
#pragma unroll
    for (int q = 0; q < NSMAX; ++q) {
      if (q < ns) {
        const float4* pq = reinterpret_cast<const float4*>(la + q * sstride + (long)t * l_ld) + lane * 2;
        qa[q][0] = pq[0];
        qa[q][1] = pq[1];
      } else {
        qa[q][0] = make_float4(0.f, 0.f, 0.f, 0.f);
        qa[q][1] = qa[q][0];
      }
    }
    if (lb != nullptr) {
      const float4* pb = reinterpret_cast<const float4*>(lb + (long)t * l_ld) + lane * 2;
      qb0 = pb[0];
      qb1 = pb[1];
    }
  }
  float v[8];
  if (Y32) {
    const float4* p = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(y) + (long)t * y_ld) + tx * 2;
    const float4 a = p[0], c = p[1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = c.x; v[5] = c.y; v[6] = c.z; v[7] = c.w;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(__float2bfloat16(v[i]));
  } else {
    const uint4 q = *(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(y) + (long)t * y_ld) + tx);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(h[i]);
  }
  float s[8], b[8];
  if (warp == 0) {
    float4 a0 = qa[0][0], a1 = qa[0][1];
#pragma unroll
    for (int q = 1; q < NSMAX; ++q) {   // FP32 split-K partials summed in a fixed order (zeros past ns)
      a0.x += qa[q][0].x; a0.y += qa[q][0].y; a0.z += qa[q][0].z; a0.w += qa[q][0].w;
      a1.x += qa[q][1].x; a1.y += qa[q][1].y; a1.z += qa[q][1].z; a1.w += qa[q][1].w;
    }
    a0.x += qb0.x; a0.y += qb0.y; a0.z += qb0.z; a0.w += qb0.w; a1.x += qb1.x; a1.y += qb1.y; a1.z += qb1.z; a1.w += qb1.w;
    const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j] = 1.f / (1.f + expf(-lg[j] * mult));
      b[j] = s[j] + bs[j];
    }
  }
'''
    text = patch(text, old_front, new_front, "tail kernel front")
    text = patch(text, "template <bool Y32>\n__global__ void __launch_bounds__(NT) tail_kernel(",
                 "template <bool Y32, int NSMAX>   // NSMAX: logits partial slices held (8: T <= 8 FP32 router, else 1)\n"
                 "__global__ void __launch_bounds__(NT) tail_kernel(", "tail template")
    # the old y / wq loads that followed are now above
    old_y = text[text.index("  float ss = 0.f;\n#pragma unroll\n  for (int i = 0; i < 8; ++i) ss += v[i] * v[i];"):][:10]
    check(old_y.startswith("  float ss"), "tail: ss block")
    # launch with programmatic stream serialization when asked (graph-captured as a programmatic edge)
    for y32 in ("true", "false"):
        old = (f"    v8::tail_kernel<{y32}><<<T, v8::NT, 0, st>>>(y.data_ptr(), y_ld, la, lb, (long)l_ld, (int)ns, (long)sstride, wp, bias.data_ptr<float>(),\n"
               "        lstart.data_ptr<int>(), xp, tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(),\n"
               "        (float)eps, (float)mult, (float)rscale, (float)deps);\n")
        args = ("y.data_ptr(), y_ld, la, lb, (long)l_ld, (int)ns,\n"
                "        (long)sstride, wp, (const float*)bias.data_ptr<float>(), (const int*)lstart.data_ptr<int>(), xp,\n"
                "        tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(), (float)eps, (float)mult, (float)rscale,\n"
                "        (float)deps")
        new = (f"    {{ if (ns > 1) C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, v8::tail_kernel<{y32}, 8>, {args}));\n"
               f"      else C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, v8::tail_kernel<{y32}, 1>, {args})); }}\n")
        text = patch(text, old, new, f"tail launch {y32}")
    text = patch(text, "  TORCH_CHECK(l_ld % 4 == 0", "  TORCH_CHECK(ns <= 8, \"v8_tail: at most 8 partial slices\");\n  TORCH_CHECK(l_ld % 4 == 0",
                 "tail ns check")
    text = patch(text, '''  auto* xp = reinterpret_cast<__nv_bfloat16*>(xn.data_ptr());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(nw.data_ptr());
''', '''  auto* xp = reinterpret_cast<__nv_bfloat16*>(xn.data_ptr());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(nw.data_ptr());
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(T);
  lc.blockDim = dim3(v8::NT);
  lc.dynamicSmemBytes = 0;
  lc.stream = st;
  lc.attrs = at;
  lc.numAttrs = 1;
''', "tail launch config")
    text = patch(text, "torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, double eps, double mult, double rscale, double deps);\nvoid v8_router32",
                 "torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, double eps, double mult, double rscale, double deps,\n"
                 "             int64_t pdl);\nvoid v8_router32", "tail decl")
    text = patch(text, "torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, double eps, double mult, double rscale, double deps) {",
                 "torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, double eps, double mult, double rscale, double deps,\n"
                 "             int64_t pdl) {", "tail def")
    text = patch(text, '''                      RMS_EPS, 0.1767766952966369, 1.0, 1e-15)''',
                 '''                      RMS_EPS, 0.1767766952966369, 1.0, 1e-15, 1 if V9_TAIL_PDL else 0)''', "tail call")
    text = patch(text, '''V8_TAIL = True ''', '''V9_TAIL_PDL = ''' + repr(TAIL_PDL) + '''   # launch the tail with programmatic stream serialization (PDL)
V8_TAIL = True ''', "tail PDL lever")
    return text


def generate():
    text, lay, r1b = G8.generate()
    base = text
    lines = text.split("\n")
    check(lines[1].startswith("# nonce k3-v08-2 "), "k3_v08 nonce")
    text = patch(text, lines[1] + "\n",
                 f"# nonce {NONCE} (k3_v08 + K2A v3.1 experts, cluster boundaries, tail loads-first + PDL; k3_v09.md)\n",
                 "nonce")
    text = patch(text, "# GENERATED by tools/gen_k3_v08.py from k3_v07.py (itself generated by gen_k3_v07.py) and r1b_v2.py.\n",
                 "# GENERATED by tools/gen_k3_v09.py on top of gen_k3_v08.py's output (k3_v07.py + r1b_v2.py + k1_v4.py +\n"
                 "# k2a_v3.py) and k2a_v31.py.\n", "generated-by")
    old_pl = [ln for ln in text.split("\n") if ln.startswith("V8_ROUTER_PLACE = ")]
    check(len(old_pl) == 1, "V8_ROUTER_PLACE line")
    text = patch(text, old_pl[0] + "\n", "V8_ROUTER_PLACE = " + repr(ROUTER_PLACE) + "   # v09: box_out/ab_v09a.log\n",
                 "router place")
    text = k2a31_stage(text)
    text = bnd_stage(text)
    if "--old-tail" not in sys.argv:
        text = tail_stage(text)
    return text, base


def verify(text, base):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text)
    btmp = OUT.with_suffix(".py.gen_base")
    btmp.write_text(base)
    try:
        py_compile.compile(str(tmp), doraise=True)
        out, b8 = Src(tmp), Src(btmp)
    finally:
        tmp.unlink()
        btmp.unlink()
    rep = []
    check(text.splitlines()[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1, "nonce")
    k31 = Src(SRC_K2A31)
    for nm in K2A31_NAMES:
        if nm in ("KERNEL_SRC_V3", "CPP_SRC_V3"):
            check(out.value(K2A31_RENAME[nm]) == k31.value(nm), f"{nm} not byte-identical")
        else:
            check(dump(ast.parse(r31(k31.seg(nm))).body[0]) == dump(out.nodes[K2A31_RENAME[nm]]), f"k2a_v31 {nm} differs")
    rep.append(f"k2a_v31: sources byte-identical (sha1 {sha(k31.value('KERNEL_SRC_V3'))[:12]}), "
               f"{len(K2A31_NAMES) - 2} host pieces ast-identical after renames; shared pieces == k2a_v22's")
    allowed = {"kernel_fn", "prepare", "_v8_experts", "V8_ROUTER_PLACE", "V8_TAIL", "V8_TAIL_CUDA_SRC", "V8_TAIL_CPP_SRC", "_v8_tail"}
    changed = [n for n, node in b8.nodes.items() if n not in out.nodes or dump(out.nodes[n]) != dump(node)]
    check(set(changed) <= allowed, f"unexpected changes vs k3_v08: {sorted(set(changed) - allowed)}")
    rep.append(f"vs k3_v08 (gen_k3_v08 output): {len(b8.nodes) - len(changed)} top-level nodes ast-identical; "
               f"changed {sorted(changed)}")
    for node in ast.walk(out.tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                check(m.split(".")[0] not in G.BANNED_NAMES, f"banned import {m}")
        if isinstance(node, ast.Name):
            check(node.id not in G.BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
    hot = ("kernel_fn", "_v8_experts", "_v9_k2a31_call", "_v9_bnd", "_v8_tail", "_k2a31_call")
    for fn in hot:
        for sub in ast.walk(out.nodes[fn]):
            check(not (isinstance(sub, ast.Attribute) and sub.attr in G.SYNC_ATTRS), f"{fn} has a host sync")
    bad = G8.G7.undefined_globals(text, str(OUT))
    check(not bad, f"undefined globals: {sorted(bad)}")
    check(len(text.encode()) < 1_000_000, "file >= 1 MB")
    rep.append(f"static: no banned names, no host syncs in {', '.join(hot)}, no undefined globals; {len(text.encode())} bytes")
    return rep


def main(argv):
    text, base = generate()
    for r in verify(text, base):
        print(f"[gen_k3_v09] OK  {r}")
    if "--old-tail" in argv:   # A/B helper only (v08's tail kernel, no PDL), never shipped
        alt = ROOT / "kernels/muse_decode_layer/v08lab/k3_v09_oldtail.py"
        alt.write_text(text.replace(f"# nonce {NONCE} ", f"# nonce {NONCE}-oldtail-lab "))
        print(f"[gen_k3_v09] wrote {alt.relative_to(ROOT)} (lab A/B only)")
        return 0
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text() == text, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_v09] --check: up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text)
    print(f"[gen_k3_v09] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}, {len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

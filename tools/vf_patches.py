"""vf_patches.py: VF front / glue riders at T <= 16 on a stripped layer file (--base, e.g. k3_vml_6),
notes/study/vf_front.md. Composes like gen_k3_vml.py: the base must be a finish() fixed point; the patch anchors are code
lines of the stripped text; then gen_k3_m32.finish (strips, compiles, ASCII, nonce, no duplicate names, checker scan <= base,
< 1 MB). Slim since 2026-10-28: the killed riders (T = 8 lean routers, prt tail, FFN boundary on SBB, KS3-slot GD geometries;
R-vf-2..4) live only in git history (commit e4458a3).

Riders (module globals read at call time, so one file serves every A/B arm; prepare self-tests each, loud [vf] fallbacks):
  t1   (VF_T1, VF_ROUTER {1: "lean"}) T = 1 on the GD pooler (T = 8 geometry, route ("pooler", 1)) + the side-stream FP32
       router vf_router32 (router32's arithmetic, bitwise; 128 CTAs x 256 thr, no smem, 30 regs at MT 1: it fits beside two
       GD pooler CTAs) + tail_kernel2, instead of G2's prt; falls back to prt when that path is not live (R-vf-3 / R-vf-4).
  ppdl GD pooler launched with PDL.
  t16  (VF_T16) T = 9..16 on a GD pooler route ("pooler", 16) + the router hi / lo rows computed by P1M's own cat kernel on a
       row-tile slice of the cat pack (same geometry, same per-tile arithmetic: bitwise equal to the shipped cat columns),
       launched beside the pooler on the side stream (VF_T16_PLACE side_first / side) or before it (main); then tail_kernel2
       with the hi / lo logits as two column blocks (hi + lo summed as in the cat path). Instead of P1M's cat GEMV.

    python -I tools/vf_patches.py --n=<new> --base=<stem> [--on=t1,ppdl[,t16|t16lab]] [--lab-only] [--force]
      (refuses to overwrite an existing k3_vf_<n> / lab copy unless --force)
      -> kernels/muse_decode_layer/k3_vf_<n>.py (defaults from --on) + kernels/lab/k3_vf_<n>_ab.py
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)


def sub(t, old, new, count=1, what=""):
    n = t.count(old)
    if n != count:
        raise SystemExit(f"[vf] anchor {what or old[:60]!r}: found {n}, expected {count}")
    return t.replace(old, new)


# ----------------------------------------------------------------------------------------------------------------------
# 1. lean FP32 router (SBB1's module: small, built by PT1 in parallel)
# ----------------------------------------------------------------------------------------------------------------------
VF_ROUTER_CU = r'''
// VF (tools/vf_patches.py): FP32 router partials part[8][m][256], router32_kernel's arithmetic per (row, K slice of 1024)
// (lane-strided float4 FMAs in the same order, same xor tree: bitwise equal), 128 CTAs x 256 threads, no smem, <= 40 regs:
// one CTA fits beside two GD pooler CTAs of an SM. Warp w of CTA (g, s) handles rows 16 g + w and 16 g + 8 + w.
template <int MT>
__global__ void __launch_bounds__(256, 6) vf_router32_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                             float* __restrict__ part, int m) {
  constexpr int K = 8192, KS = 1024;
  const int sl = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int k0 = sl * KS;
#pragma unroll 1
  for (int h = 0; h < 2; ++h) {
    const int row = blockIdx.x * 16 + h * 8 + warp;
    const float4* wp = reinterpret_cast<const float4*>(w + (long)row * K + k0);
    float acc[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[j] = 0.f;
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
      float4 wv[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) wv[i] = __ldcs(wp + (hh * 2 + i) * 32 + lane);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int c = (hh * 2 + i) * 32 + lane;
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          if (j < m) {
            const float4 xv = __ldg(reinterpret_cast<const float4*>(x + (long)j * K + k0) + c);
            acc[j] = fmaf(wv[i].x, xv.x, acc[j]);
            acc[j] = fmaf(wv[i].y, xv.y, acc[j]);
            acc[j] = fmaf(wv[i].z, xv.z, acc[j]);
            acc[j] = fmaf(wv[i].w, xv.w, acc[j]);
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      float a = acc[j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0 && j < m) part[((long)sl * m + j) * 256 + row] = a;
    }
  }
}
'''

VF_ROUTER_HOST = r'''
void vf_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part) {
  const int m = (int)x.size(0);
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat && part.scalar_type() == at::kFloat &&
              x.is_contiguous() && w.is_contiguous() && part.is_contiguous() && x.size(1) == 8192 && w.size(0) == 256 &&
              w.size(1) == 8192 && m >= 1 && m <= 8 && part.numel() == 8L * m * 256 &&
              reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "vf_router32: x f32 [m <= 8, 8192], w f32 [256, 8192], part [8, m, 256]");
  auto s = c10::cuda::getCurrentCUDAStream();
  if (m == 1)
    vf_router32_kernel<1><<<dim3(16, 8), 256, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  else
    vf_router32_kernel<8><<<dim3(16, 8), 256, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "vf_router32: launch failed");
}
'''
VF_ROUTER_DECL = " void vf_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part);"


def patch_router(t):
    t = sub(t, "SBB1_HOST = r'''\n", "SBB1_HOST = r'''\n" + VF_ROUTER_CU, what="SBB1_HOST head")
    t = sub(t, "  else sbb_go<false, false>(p, T, s);\n}\n'''\n",
            "  else sbb_go<false, false>(p, T, s);\n}\n" + VF_ROUTER_HOST + "'''\n", what="sbb1_run end")
    t = sub(t, "double post_eps, double pre_eps, int64_t input_scale);'\n",
            "double post_eps, double pre_eps, int64_t input_scale);" + VF_ROUTER_DECL + "'\n", what="SBB1_DECL")
    t = sub(t, 'functions=["sbb1_run"],', 'functions=["sbb1_run", "vf_router32"],', what="sbb1 functions")
    t = sub(t, """    if mode == "f32":
        part = torch.empty((8, router_input.shape[0], N_EXPERTS), dtype=torch.float32, device=router_input.device)
        state["v8"]["mod"].v8_router32(router_input, state["router"], part)
        return part
""", """    if mode == "f32":
        part = torch.empty((8, router_input.shape[0], N_EXPERTS), dtype=torch.float32, device=router_input.device)
        if VF_ROUTER.get(1 if router_input.shape[0] == 1 else 8) == "lean" and _VF["r8"] and _SBB1["mod"] is not None:
            _SBB1["mod"].vf_router32(router_input, state["router"], part)
        else:
            state["v8"]["mod"].v8_router32(router_input, state["router"], part)
        return part
""", what="_v8_logits f32")
    # T = 1: GD pooler + side router + tail when live, else G2's prt
    t = sub(t, '    if not (G2_ENABLE and G2_PRT and g2 is not None and g2.get("prt")) or tokens != 1:\n        return None\n',
            '    if not (G2_ENABLE and G2_PRT and g2 is not None and g2.get("prt")) or tokens != 1:\n        return None\n'
            '    if _vf_t1_live(state):\n        return None\n', what="_g2_prt t1")
    return t


# ----------------------------------------------------------------------------------------------------------------------
# 2. T = 9..16: GD pooler + P1M router-row slice (bitwise = the cat columns) + tail
# ----------------------------------------------------------------------------------------------------------------------
def patch_t16(t):
    t = sub(t, '    elif tail == "cat":\n        y32 = _v20_p1m(state, "cat", ffn_input)\n',
            '    elif tail == "cat" and _vf_t16_live(state, tokens):\n'
            '        expert_x, topk_weights, topk_ids, local_ids = _vf_t16(state, ffn_input, data)\n'
            '    elif tail == "cat":\n        y32 = _v20_p1m(state, "cat", ffn_input)\n', what="kernel_fn cat")
    return t


VF_PY = r'''

VF_ROUTER = @ROUTER@
VF_T1 = @T1@
VF_T16 = @T16@
_VF = {"r8": False, "t16": None, "seen": set()}


def _vf_once(key, msg):
    if key not in _VF["seen"]:
        _VF["seen"].add(key)
        print(msg, flush=True)


def _vf_t1_live(state):
    if not VF_T1:
        return False
    ok = (VF_ROUTER.get(1) == "lean" and _VF["r8"] and _SBB1["mod"] is not None and state.get("v8") is not None
          and _gd_live(state, "pooler", 1) is not None)
    if not ok:
        _vf_once("t1off", "[vf] FALLBACK t1: lean router / GD pooler T=1 not live -> G2 prt at T = 1")
    return ok


def _vf_test_r8(state, mod):
    dev = state["router"].device
    ok = True
    for m, seed in ((1, 731), (1, 732)):
        gen = torch.Generator(device=dev).manual_seed(seed)
        z = torch.randn((m, HIDDEN), generator=gen, device=dev)
        z = (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True))).contiguous()
        ref = torch.full((8, m, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
        got = torch.full((8, m, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
        state["v8"]["mod"].v8_router32(z, state["router"], ref)
        mod.vf_router32(z, state["router"], got)
        torch.cuda.synchronize()
        eq = bool(torch.equal(ref, got))
        print(f"[vf] r8 self-test m={m}: router partials bitwise {eq}", flush=True)
        ok &= eq
    return ok


def _vf_prepare(state, t0):
    t1 = time.time()
    _VF["r8"] = False
    _VF["t16"] = None
    mod = _SBB1["mod"]
    try:
        if mod is not None and _SBB1["st"] is not None and state.get("v8") is not None and state.get("router") is not None:
            _VF["r8"] = _vf_test_r8(state, mod)
    except Exception as ex:
        print(f"[vf] r8 FAILED -> router32 stays: {ex!r}", flush=True)
        _VF["r8"] = False
    try:
        if state.get("v8") is not None and "_vf_prep_t16" in globals():
            _VF["t16"] = _vf_prep_t16(state)
    except Exception as ex:
        print(f"[vf] t16 FAILED -> P1M cat stays: {ex!r}", flush=True)
        _VF["t16"] = None
    print(f"[vf] riders router {'LIVE' if _VF['r8'] else 'OFF'} t16 slice {'LIVE' if _VF['t16'] else 'OFF'} (switches ROUTER "
          f"{VF_ROUTER} T1 {VF_T1} T16 {VF_T16} {globals().get('VF_T16_POOL')} {globals().get('VF_T16_PLACE')}; "
          f"{time.time() - t1:.1f}s, prepare so far "
          f"{time.time() - t0:.1f}s)", flush=True)
'''


VF_PY16 = r'''
VF_T16_POOL = "pooler"
VF_T16_PLACE = "side_first"


def _vf_t16_live(state, tokens):
    if not VF_T16 or not 9 <= tokens <= 16:
        return False
    ok = (_VF["t16"] is not None and state.get("v8") is not None and _gd_live(state, VF_T16_POOL, tokens) is not None
          and state.get("side_stream") is not None)
    if not ok:
        _vf_once("t16off", f"[vf] FALLBACK t16: router slice / GD {VF_T16_POOL} T=16 not live -> P1M cat")
    return ok


def _vf_router16(state, x, out, pdl):
    s = _VF["t16"]
    p1m_call(s["mod"], s["P"], x, out, s["geo"], pdl=pdl)


def _vf_tail16(y, lg, state, data):
    v8 = state["v8"]
    T = y.shape[0]
    dev = y.device
    xn = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
    tw = torch.empty((T, TOP_K), dtype=torch.float32, device=dev)
    tid = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    lid = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    v8["mod"].v8_tail(y, lg, 0, N_EXPERTS, lg.stride(0), 1, 0, v8["nw"], v8["bias"], data["local_start"], xn, tw, tid, lid,
                      RMS_EPS, 0.1767766952966369, 1.0, 1e-15, (1 if V9_TAIL_PDL else 0) | (2 if TL1_TAIL2 else 0))
    return xn, tw, tid, lid


def _vf_t16(state, x, data):
    T = x.shape[0]
    lg = torch.empty((T, 2 * N_EXPERTS), dtype=torch.float32, device=x.device)
    if VF_T16_PLACE == "main":
        _vf_router16(state, x, lg, 1)
        y = _gd(state, VF_T16_POOL, x)
    else:
        main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
        side_stream.wait_stream(main_stream)
        if VF_T16_PLACE == "side_first":
            with torch.cuda.stream(side_stream):
                _vf_router16(state, x, lg, 0)
            y = _gd(state, VF_T16_POOL, x)
        else:
            y = _gd(state, VF_T16_POOL, x)
            with torch.cuda.stream(side_stream):
                _vf_router16(state, x, lg, 0)
        main_stream.wait_stream(side_stream)
        if not torch.cuda.is_current_stream_capturing():
            lg.record_stream(main_stream)
    _vf_once(("t16", T), f"[vf] first call t16 T={T}: GD {VF_T16_POOL} + P1M router slice ({VF_T16_PLACE})")
    return _vf_tail16(y, lg, state, data)


def _vf_prep_t16(state):
    st = state.get("p1m")
    if st is None or "cat" not in st["proj"] or ("cat", 16) not in st["route"] or st["il"].get(("cat", 16)) is not None:
        print("[vf] t16: P1M cat route not live (or interleaved) -> off", flush=True)
        return None
    P = st["proj"]["cat"]
    geo = st["route"][("cat", 16)]
    if geo[1] != 1 or P["N"] != D_EXPERT + 2 * N_EXPERTS:
        print(f"[vf] t16: cat geometry {geo} / N {P['N']} unsupported -> off", flush=True)
        return None
    r0 = D_EXPERT // 16
    Ps = dict(P)
    Ps.update(N=2 * N_EXPERTS, toff=P["toff"][r0:], scale=P["scale"][r0:], eoff=P["eoff"][r0:], fmt=P["fmt"][r0:])
    s = {"mod": st["mod"], "P": Ps, "geo": geo[:5]}
    dev = P["buf"].device
    gen = torch.Generator(device=dev).manual_seed(7316)
    ok = True
    for T in (16, 9, 12):
        x = (torch.randn((T, HIDDEN), generator=gen, device=dev) * 0.5).to(torch.bfloat16).contiguous()
        full = torch.empty((T, P["N"]), dtype=torch.float32, device=dev)
        p1m_call(st["mod"], P, x, full, geo[:5], il=st["il"].get(("cat", 16)))
        got = torch.full((T, 2 * N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
        _VF["t16"] = s
        _vf_router16(state, x, got, 0)
        torch.cuda.synchronize()
        eq = bool(torch.equal(got, full[:, D_EXPERT:]))
        ref = x.double() @ state["v8"]["wcat"][D_EXPERT:].double().T
        rel = float((got.double() - ref).norm() / ref.norm())
        print(f"[vf] t16 router slice self-test T={T}: bitwise vs P1M cat columns {eq}, vs fp64 rel {rel:.1e}", flush=True)
        ok &= eq
    return s if ok else None


'''


def patch_py(t, on):
    blk = (VF_PY.replace("@ROUTER@", repr({1: "lean" if "t1" in on else "r32", 8: "r32"}))
           .replace("@T1@", str("t1" in on)).replace("@T16@", str("t16" in on)))
    if "t16" in on or "t16lab" in on:
        blk += VF_PY16
    t = sub(t, "\n\nSBB1_ENABLE = True\n", blk + "\n\nSBB1_ENABLE = True\n", what="SBB1_ENABLE")
    t = sub(t, "    _sbb1_prepare(state, t0)\n", "    _sbb1_prepare(state, t0)\n    _vf_prepare(state, t0)\n", what="prepare call")
    gd_old = '            ("pooler", 8): (1, 1, 8, 16, 2, 1, 1),\n'
    gd_new = '            ("pooler", 1): (1, 1, 8, 16, 2, 1, 1), ("pooler", 8): (1, 1, 8, 16, 2, 1, 1),\n'
    if "t16" in on or "t16lab" in on:
        gd_new += '            ("pooler", 16): (2, 1, 8, 16, 2, 1, 1),\n'
    if "t16lab" in on:   # A/B only: a second T = 16 pooler geometry under its own name (108 regs, no x prefetch)
        gd_new += '            ("pooler@s", 16): (2, 1, 8, 16, 2, 1, 0),\n'
        t = sub(t, 'GD_DIMS = {"qkv": (2304, HIDDEN),', 'GD_DIMS = {"pooler@s": (D_EXPERT, HIDDEN), "qkv": (2304, HIDDEN),',
                what="GD_DIMS")
        t = sub(t, 'GD_WEIGHT = {"qkv": "qkv_gate",', 'GD_WEIGHT = {"pooler@s": "pooler", "qkv": "qkv_gate",', what="GD_WEIGHT")
        t = sub(t, 'GD_TRIG = {"qkv": True, "o": True, "pooler": True, "unpooler": True}',
                'GD_TRIG = {"qkv": True, "o": True, "pooler": True, "unpooler": True, "pooler@s": True}', what="GD_TRIG")
    t = sub(t, gd_old, gd_new, what="GD pooler routes")
    if "ppdl" in on:   # GD pooler launched with PDL (T = 1 after the fork, T = 8, T = 16)
        t = sub(t, 'GD_PDL = {"qkv": False, "o": True, "pooler": False, "unpooler": True}',
                'GD_PDL = {"qkv": False, "o": True, "pooler": True, "unpooler": True'
                + (', "pooler@s": True' if "t16lab" in on else "") + '}', what="GD_PDL")
    return t


def apply(t, on=()):
    on = set(on)
    t = patch_router(t)
    if "t16" in on or "t16lab" in on:
        t = patch_t16(t)
    t = patch_py(t, on)
    return t


def main():
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    if "n" not in ARGS or "base" not in ARGS:
        raise SystemExit("[vf_patches] --n=<new number> and --base=<layer stem> are required")
    n = ARGS["n"]
    base = ROOT / f"kernels/muse_decode_layer/{ARGS['base']}.py"
    on = [x for x in ARGS.get("on", "t1,ppdl").split(",") if x]
    labonly = "--lab-only" in sys.argv
    out = ROOT / f"kernels/muse_decode_layer/k3_vf_{n}.py"
    lab = ROOT / f"kernels/lab/k3_vf_{n}_ab.py"
    for f in ([lab] if labonly else [out, lab]):   # never reuse an existing output name (parent rule 2026-10-28)
        if f.exists() and "--force" not in sys.argv:
            raise SystemExit(f"[vf_patches] {f.relative_to(ROOT).as_posix()} exists: pick a new --n (or --force for your own scratch file)")
    tag = f"k3-vf-{n}"
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    GM.check(fb == b, f"{base.name} is not a fixed point of finish()")
    rep.append(f"base {base.name} {len(b.encode())} B is a finish() fixed point")
    nonce = (f"# nonce {tag} ({base.stem} + VF front riders: T = 1 GD pooler + lean FP32 router beside it + tail, pooler PDL"
             f"{', T = 16 GD pooler + P1M router slice' if ('t16' in on or 't16lab' in on) else ''}; on: {','.join(on) or 'none'};"
             f" self-tested with fallback)")
    t = apply(b.replace(bn, nonce), on)
    s = GM.finish(t, nonce, tag, chk_base, rep)
    for r in rep:
        print(f"[vf_patches] OK  {r}")
    if not labonly:
        out.write_text(s, encoding="utf-8", newline=NL)
    lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vf_patches] wrote {'' if labonly else out.relative_to(ROOT).as_posix() + ' and '}{lab.relative_to(ROOT).as_posix()} "
          f"({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

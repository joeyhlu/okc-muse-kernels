"""vf8_patches.py: the T = 8 FP32 router "lean6" beside the GD pooler, on a layer file that already carries the VF T = 1 rider in
its k3_vf_5 form (vf_router32(x, w, part, kind) with kinds 0..2; default base kernels/muse_decode_layer/k3_vml_8.py).

lean6 = vf_router32d_kernel (kind 3): router32's arithmetic per (row, K slice of 1024) (same FMA order, same xor tree: bitwise
equal), 128 CTAs x 128 threads (<= 132: one per SM whatever the launch order), <= 80 regs, the 8-token FP32 x slice staged ONCE
per CTA in 32 KB of static shared memory (router32 does the same; the killed lean routers read x through L1 and had no smem, so
an SM configured by them may have had no shared-memory carveout for the GD pooler's 4 KB), weights by ld.global.nc.L1::no_allocate
with all 8 float4 of a row in flight, 4 rows per warp, rows 1-3 warmed in L2 at entry. Register file per SM beside two GD pooler
CTAs: 2 x 26,624 + 10,240 = 63,488. Self-tested bitwise vs router32 in prepare (its own flag _VF["r6"], so a failure leaves the
T = 1 rider alone); VF_ROUTER[8] = "lean6" selects it at call time.

    python -I tools/vf8_patches.py --n=<new> [--base=k3_vml_8] [--on] [--place-side] [--lab-only] [--force]
      --on: VF_ROUTER[8] = "lean6" by default (ship); without it the file behaves like the base (A/B lab copy)
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
        raise SystemExit(f"[vf8] anchor {what or old[:60]!r}: found {n}, expected {count}")
    return t.replace(old, new)


KERNEL = r'''template <int MT>
__global__ void __launch_bounds__(128, 6) vf_router32d_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                              float* __restrict__ part, int m) {
  constexpr int K = 8192, KS = 1024, KS4 = KS / 4;
  __shared__ float4 xs[MT * KS4];
  const int sl = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int k0 = sl * KS;
#pragma unroll
  for (int q = 1; q < 4; ++q) {
    const float* pl = w + (long)(blockIdx.x * 16 + q * 4 + warp) * K + k0 + lane * 32;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(pl));
  }
  float4 wv[8];
  {
    const float4* wp = reinterpret_cast<const float4*>(w + (long)(blockIdx.x * 16 + warp) * K + k0);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float4* p = wp + i * 32 + lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];"
                   : "=f"(wv[i].x), "=f"(wv[i].y), "=f"(wv[i].z), "=f"(wv[i].w) : "l"(p));
    }
  }
  for (int idx = threadIdx.x; idx < MT * KS4; idx += 128) {
    const int j = idx / KS4, c = idx - j * KS4;
    xs[idx] = (j < m) ? reinterpret_cast<const float4*>(x + (long)j * K + k0)[c] : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  __syncthreads();
#pragma unroll 1
  for (int q = 0; q < 4; ++q) {
    const int row = blockIdx.x * 16 + q * 4 + warp;
    if (q > 0) {
      const float4* wp = reinterpret_cast<const float4*>(w + (long)row * K + k0);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float4* p = wp + i * 32 + lane;
        asm volatile("ld.global.nc.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];"
                     : "=f"(wv[i].x), "=f"(wv[i].y), "=f"(wv[i].z), "=f"(wv[i].w) : "l"(p));
      }
    }
    float acc[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[j] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = i * 32 + lane;
#pragma unroll
      for (int j = 0; j < MT; ++j) {
        const float4 xv = xs[j * KS4 + c];
        acc[j] = fmaf(wv[i].x, xv.x, acc[j]);
        acc[j] = fmaf(wv[i].y, xv.y, acc[j]);
        acc[j] = fmaf(wv[i].z, xv.z, acc[j]);
        acc[j] = fmaf(wv[i].w, xv.w, acc[j]);
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

TEST = '''def _vf_test_r6(state, mod):
    dev = state["router"].device
    ok = True
    for m, seed in ((8, 741), (8, 742), (3, 743), (2, 744)):
        gen = torch.Generator(device=dev).manual_seed(seed)
        z = torch.randn((m, HIDDEN), generator=gen, device=dev)
        z = (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True))).contiguous()
        ref = torch.full((8, m, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
        got = torch.full((8, m, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
        state["v8"]["mod"].v8_router32(z, state["router"], ref)
        mod.vf_router32(z, state["router"], got, 3)
        torch.cuda.synchronize()
        eq = bool(torch.equal(ref, got))
        print(f"[vf] r6 (lean6, T = 8 smem-staged router) self-test m={m}: router partials bitwise {eq}", flush=True)
        ok &= eq
    return ok


def _vf_test_r8(state, mod):'''


def apply(t, on):
    t = sub(t, "template <int MT>\n__global__ void __launch_bounds__(128, 6) vf_router32c_kernel(",
            KERNEL + "template <int MT>\n__global__ void __launch_bounds__(128, 6) vf_router32c_kernel(", what="kernel")
    t = sub(t, "  if (kind == 2)\n    vf_router32c_kernel<8>",
            "  if (kind == 3)\n    vf_router32d_kernel<8><<<dim3(16, 8), 128, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), "
            "part.data_ptr<float>(), m);\n  else if (kind == 2)\n    vf_router32c_kernel<8>", what="host kind 3")
    t = sub(t, '        if kind != "r32" and _VF["r8"] and _SBB1["mod"] is not None:\n'
               '            _SBB1["mod"].vf_router32(router_input, state["router"], part, {"lean4": 1, "lean5": 2}.get(kind, 0))\n',
            '        if kind == "lean6" and not _VF.get("r6"):\n            kind = "r32"\n'
            '        if kind != "r32" and _VF["r8"] and _SBB1["mod"] is not None:\n'
            '            _SBB1["mod"].vf_router32(router_input, state["router"], part, {"lean4": 1, "lean5": 2, "lean6": 3}.get(kind, 0))\n',
            what="dispatch")
    t = sub(t, "def _vf_test_r8(state, mod):", TEST, what="test fn")
    t = sub(t, '            _VF["r8"] = _vf_test_r8(state, mod)\n',
            '            _VF["r8"] = _vf_test_r8(state, mod)\n'
            '            try:\n                _VF["r6"] = _vf_test_r6(state, mod)\n'
            '            except Exception as ex:\n                print(f"[vf] r6 FAILED -> router32 at T = 2..8: {ex!r}", flush=True)\n'
            '                _VF["r6"] = False\n', what="prepare r6")
    if on:
        t = sub(t, "VF_ROUTER = {1: 'lean', 8: 'r32'}\n", "VF_ROUTER = {1: 'lean', 8: 'lean6'}\n", what="VF_ROUTER default")
    if "--place-side" in sys.argv:   # T = 8: pooler captured before the router on the side stream
        t = sub(t, "V8_ROUTER_PLACE = {1: 'side', 8: 'side_first',", "V8_ROUTER_PLACE = {1: 'side', 8: 'side',", what="V8 place")
    return t


def main():
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    if "n" not in ARGS:
        raise SystemExit("[vf8_patches] --n=<new number> is required")
    n = ARGS["n"]
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_vml_8')}.py"
    on = "--on" in sys.argv
    labonly = "--lab-only" in sys.argv
    out = ROOT / f"kernels/muse_decode_layer/k3_vf_{n}.py"
    lab = ROOT / f"kernels/lab/k3_vf_{n}_ab.py"
    for f in ([lab] if labonly else [out, lab]):
        if f.exists() and "--force" not in sys.argv:
            raise SystemExit(f"[vf8_patches] {f.relative_to(ROOT).as_posix()} exists: pick a new --n")
    tag = f"k3-vf-{n}"
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    GM.check(fb == b, f"{base.name} is not a fixed point of finish()")
    nonce = (f"# nonce {tag} ({base.stem} + VF lean6: T = 8 FP32 router with smem-staged x beside the GD pooler"
             f"{' (on)' if on else ' (switch off)'}; self-tested with fallback)")
    t = apply(b.replace(bn, nonce), on)
    s = GM.finish(t, nonce, tag, chk_base, rep)
    for r in rep:
        print(f"[vf8_patches] OK  {r}")
    if not labonly:
        out.write_text(s, encoding="utf-8", newline=NL)
    lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vf8_patches] wrote {'' if labonly else out.relative_to(ROOT).as_posix() + ' and '}{lab.relative_to(ROOT).as_posix()} "
          f"({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

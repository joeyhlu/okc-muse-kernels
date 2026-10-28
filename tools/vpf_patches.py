"""vpf_patches.py: VPF = VMS first-unit L2 prefetch from tail_kernel2 (the parent's "VMS routing before its wait" lever,
in a form that needs no cross-kernel flag protocol: an L2 prefetch is only a hint, outputs cannot change).

At T <= VMS_MAXT the routing tail tail_kernel2 (one CTA per token; its routing warp writes the token's tw / tid / lid) is
launched with one extra (idle) warp, which is the per-call enable. Each CTA's routing warp then takes an atomic ticket after
its lid writes (fence); the last CTA reads all T x 8 lid, builds VMS's job table exactly as VMS does (experts in local-id
order, jobs of <= 8 tokens), and issues cp.async.bulk.prefetch.L2 for every VMS CTA's first 3 units (c * NA / G + 0..2,
36,864 B each: ~14 MB at T = 1..16), so VMS's own first copies after its PDL wait hit L2. The ticket self-resets.
v8_vms_setup(wA ptr, G) stores VMS's weight base and grid in device globals once at prepare (_vpf_prepare, loud print).
VPF_ENABLE is read at call time (A/B arms). Applied to a stripped layer file (finish() fixed point), default k3_vf_4.

    python -I tools/vpf_patches.py [--n=4] [--base=k3_vf_4] [--check]
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 4))
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_vf_4')}.py"
OUT = ROOT / f"kernels/muse_decode_layer/k3_vms_{N}.py"
LAB = ROOT / f"kernels/lab/k3_vms_{N}_ab.py"
TAG = f"k3-vms-{N}"
NL = chr(10)

GLOBALS = r'''namespace v8 {
__device__ const unsigned char* g_vwa = nullptr;   // VPF: VMS weight units base (36,864 B units, [32][512] W13 first)
__device__ int g_vG = 0;
__device__ int g_vtick = 0;
'''
ROUTE_END_OLD = "        lid[t * TK + k] = rel < 32 ? rel : -1;\n      }\n    }\n    return;\n  }\n"
ROUTE_END_NEW = r'''        lid[t * TK + k] = rel < 32 ? rel : -1;
      }
    }
#if __CUDA_ARCH__ >= 900
    if (blockDim.x > NT + 32 && g_vwa != nullptr && gridDim.x <= 16) {   // VPF (tools/vpf_patches.py)
      __shared__ int vcnt[32];
      __shared__ unsigned char vtab[64];
      int last = 0;
      __syncwarp();
      if (lane == 0) {
        __threadfence();
        last = atomicAdd(&g_vtick, 1) == (int)gridDim.x - 1;
      }
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        if (lane == 0) g_vtick = 0;
        __threadfence();
        vcnt[lane] = 0;
        __syncwarp();
        const int m8 = (int)gridDim.x * TK;
        for (int r = 0; r < 4; ++r) {
          const int i = r * 32 + lane;
          const int e = i < m8 ? __ldcg(lid + i) : -1;
          const unsigned mm = __match_any_sync(0xffffffffu, e);
          if (e >= 0 && lane == __ffs(mm) - 1) vcnt[e] += __popc(mm);
          __syncwarp();
        }
        const int nj = (vcnt[lane] + 7) >> 3;
        int js = nj;
        for (int d = 1; d < 32; d <<= 1) {
          const int a = __shfl_up_sync(0xffffffffu, js, d);
          if (lane >= d) js += a;
        }
        const int NJ = __shfl_sync(0xffffffffu, js, 31);
        for (int k = 0; k < nj; ++k) vtab[js - nj + k] = (unsigned char)lane;
        __syncwarp();
        const int G = g_vG, NA = NJ * 512;
        for (int pi = lane; pi < G * 3; pi += 32) {
          const int c = pi / 3, q = pi - c * 3;
          const int v = c * NA / G + q;
          if (v < NA) {
            const unsigned char* src = g_vwa + ((size_t)vtab[v >> 9] * 512 + (v & 511)) * 36864;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], 36864;\n" :: "l"(src) : "memory");
          }
        }
      }
    }
#endif
    return;
  }
'''
EXTRA_WARP_OLD = "  __shared__ float red[NT / 32];\n  if (warp == NT / 32) {"
EXTRA_WARP_NEW = "  __shared__ float red[NT / 32];\n  if (warp > NT / 32) return;   // VPF enable warp (idle)\n  if (warp == NT / 32) {"
SETUP_FN = r'''void v8_vms_setup(int64_t wa, int64_t G) {
  const unsigned char* p = reinterpret_cast<const unsigned char*>(wa);
  const int g = (int)G;
  C10_CUDA_CHECK(cudaMemcpyToSymbol(v8::g_vwa, &p, sizeof(p)));
  C10_CUDA_CHECK(cudaMemcpyToSymbol(v8::g_vG, &g, sizeof(g)));
}

'''
PY_BLOCK = r'''VPF_ENABLE = True


def _vpf_prepare(state):
    state["vpf_ok"] = False
    try:
        k33 = state.get("k2a33")
        st = k33["st"] if k33 is not None else None
        v8 = state.get("v8")
        if not (VPF_ENABLE and TL1_TAIL2 and st is not None and st.get("vms_ok") and v8 is not None
                and hasattr(v8["mod"], "v8_vms_setup")):
            print(f"[vpf] OFF (VPF_ENABLE={VPF_ENABLE}, TL1_TAIL2={TL1_TAIL2}, vms_ok={st.get('vms_ok') if st else None}, "
                  f"tail module={'yes' if v8 is not None else 'no'})", flush=True)
            return
        v8["mod"].v8_vms_setup(st["vms"]["wA"].data_ptr(), int(_VMS["G"]))
        torch.cuda.synchronize()
        state["vpf_ok"] = True
        print(f"[vpf] VMS first-unit L2 prefetch from tail_kernel2 LIVE at T <= {VMS_MAXT} (G {_VMS['G']})", flush=True)
    except Exception as ex:
        state["vpf_ok"] = False
        print(f"[vpf] prepare raised {ex!r}: VPF OFF", flush=True)


'''


def apply(text):
    reps = (
        ("globals", "namespace v8 {\n", GLOBALS),
        ("launch bounds", "__global__ void __launch_bounds__(NT + 32) tail_kernel2(",
         "__global__ void __launch_bounds__(NT + 64) tail_kernel2("),
        ("extra warp", EXTRA_WARP_OLD, EXTRA_WARP_NEW),
        ("routing end", ROUTE_END_OLD, ROUTE_END_NEW),
        ("blockDim", "lc.blockDim = dim3((pdl & 2) ? v8::NT + 32 : v8::NT);",
         "lc.blockDim = dim3((pdl & 2) ? v8::NT + 32 + ((pdl & 4) ? 32 : 0) : v8::NT);"),
        ("setup fn", "void v8_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part) {",
         SETUP_FN + "void v8_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part) {"),
        ("cpp decl", "void v8_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part);\n",
         "void v8_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part);\nvoid v8_vms_setup(int64_t wa, int64_t G);\n"),
        ("own build", 'functions=["v8_tail", "v8_router32"]', 'functions=["v8_tail", "v8_router32", "v8_vms_setup"]'),
        ("v10 build", '"v8_tail", "v8_router32", "pxrun", "pxattrs"]', '"v8_tail", "v8_router32", "v8_vms_setup", "pxrun", "pxattrs"]'),
        ("tail call", "(1 if V9_TAIL_PDL else 0) | (2 if TL1_TAIL2 else 0))",
         "(1 if V9_TAIL_PDL else 0) | (2 if TL1_TAIL2 else 0)\n                      | (4 if (VPF_ENABLE and TL1_TAIL2 and state.get(\"vpf_ok\") and tokens <= VMS_MAXT) else 0))"),
        ("prepare hook", "    _p206_prepare(state, t0)\n", "    _p206_prepare(state, t0)\n    _vpf_prepare(state)\n"),
        ("py block", "def _v8_tail(y, logits, state, data):\n", PY_BLOCK + "def _v8_tail(y, logits, state, data):\n"),
    )
    for what, old, new in reps:
        n = text.count(old)
        assert n == 1, f"vpf anchor {what!r} found {n} times"
        text = text.replace(old, new, 1)
    return text


def main():
    sys.path.insert(0, str(ROOT / "tools"))
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    GM.check(fb == b, f"{BASE.name} is not a fixed point of finish()")
    rep = [f"base {BASE.name} {len(b.encode())} B is a finish() fixed point"]
    nonce = (f"# nonce {TAG} ({BASE.stem} + VPF: tail_kernel2's last routing CTA L2-prefetches every VMS CTA's first 3 weight "
             "units at T <= 16, a hint only)")
    s = GM.finish(apply(b.replace(bn, nonce)), nonce, TAG, chk_base, rep)
    rep.append(f"size: {len(b.encode())} -> {len(s.encode())} B ({len(s.encode()) - len(b.encode()):+d} B, headroom "
               f"{1_000_000 - len(s.encode())} B)")
    for r in rep:
        print(f"[vpf] OK  {r}")
    if "--check" in sys.argv:
        GM.check(OUT.exists() and OUT.read_text(encoding="utf-8") == s, f"{OUT.name} on disk differs")
        print(f"[vpf] --check: {OUT.name} up to date")
        return 0
    OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vpf] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

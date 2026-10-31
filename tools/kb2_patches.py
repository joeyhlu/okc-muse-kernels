"""kb2_patches.py (R-kb2-1): qkv/256 on 2x the CTAs through cuBLASLt, in-layer arms on the current final (k3_hh_5_sk2_pz).

  python -I tools/kb2_patches.py [--base=kernels/lab/k3_hh_5_sk2_pz.py] [--out=kernels/lab/k3_kb2_ab.py] [--check]
  ship (after a GO): python -I tools/kb2_patches.py --ship=b|s [--base=...] [--out=...]

Why (ext_muse_qkv256.md, fable_outside_view.md D1): at T >= 128 the in-layer SM clock is ~1.52 GHz, so qkv/256 (9.66 GFLOP) is
tensor-bound in-layer; the shipped pin (algo 66 tile 124 = nvjet tst_128x128_64x6_1x2 coopA, 18 x 2 = 36 output tiles of 128x128,
<= 72 CTAs) fills well under the 132 SMs -> 24.2 / 25.4 us in-layer at v32s / v32f vs 16-17 standalone at 1.98 GHz.
$0 fill audit of the other T = 256 dense pins (kernel names in judge profile k3_gc_40_s1, output tiles = features / tile x tokens /
tile): o/256 tst_128x128 2x1 = 64 x 2 = 128 tiles; cat/256 tst_72x64 = 64 x 4 = 256; unpooler/256 (NN batch 2) tss_72x128 =
2 x 57 x 2 = 228 -> only qkv/256 is under-filled; no unpooler arm.
Arms (module globals; KB2_ON = None = ship):
  "b": ONE strided-batched cuBLASLt call, batch 2 over the two K halves (A / B offsets k/2, ld k; FP32 C0 / C1, offset m n), same
       pin config (lt_make), then kb2_sum (one tiny CUDA kernel: bf16(C0 + C1), PDL edge unless KB2_PDL = 0).
  "s": the pin config with SPLITK_NUM 2 (+ the first reduction scheme cuBLASLt accepts) in one call (cuBLASLt's own split-K).
Numerics: FP32-accumulate reassociation only (checked vs FP64 and vs the ship pin in prepare; the judge's eager checks on the box).
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[kb2_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


LT_ADD = r'''int64_t lt_plan_k(int64_t m, int64_t n, int64_t k, int64_t splits, int64_t ws_bytes, int64_t max_algos) {
  TORCH_CHECK(splits >= 2 && k % splits == 0, "lt: k splits");
  if (!g_h) TORCH_CHECK(cublasLtCreate(&g_h) == CUBLAS_STATUS_SUCCESS, "lt: create");
  LtPlan p;
  TORCH_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS, "lt: desc");
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  const int64_t kc = k / splits;
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, kc, n, k) == CUBLAS_STATUS_SUCCESS, "lt: la");
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, kc, m, k) == CUBLAS_STATUS_SUCCESS, "lt: lb");
  p.ct = CUDA_R_32F;
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.lc, p.ct, n, m, n) == CUBLAS_STATUS_SUCCESS, "lt: lc");
  const int32_t bc = (int32_t)splits;   // batch b = K slice b: A / B start kc * b elements in, C_b = FP32 [m, n] slice b
  cublasLtMatrixLayout_t ls[3] = {p.la, p.lb, p.lc};
  const int64_t so[3] = {kc, kc, n * m};
  for (int i = 0; i < 3; ++i) {
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)) == CUBLAS_STATUS_SUCCESS, "lt: bc");
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &so[i], sizeof(int64_t)) ==
                CUBLAS_STATUS_SUCCESS, "lt: bo");
  }
  cublasLtMatmulPreference_t pref;
  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS, "lt: pref");
  uint64_t ws = (uint64_t)ws_bytes;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  p.res.resize((size_t)max_algos);
  int got = 0;
  cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(g_h, p.op, p.la, p.lb, p.lc, p.lc, pref, (int)max_algos, p.res.data(), &got);
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "lt: heuristic status ", (int)s);
  p.res.resize((size_t)got);
  g_plans.push_back(p);
  return ((int64_t)(g_plans.size() - 1) << 8) | (int64_t)got;
}
'''
KB2_CUDA = r'''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
__global__ void kb2_sum_kernel(const float4* __restrict__ a, const float4* __restrict__ b, uint2* __restrict__ o, int n4, int pdl) {
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    const float4 x = a[i], y = b[i];
    __nv_bfloat162 lo = __floats2bfloat162_rn(x.x + y.x, x.y + y.y), hi = __floats2bfloat162_rn(x.z + y.z, x.w + y.w);
    uint2 r;
    r.x = *reinterpret_cast<uint32_t*>(&lo);
    r.y = *reinterpret_cast<uint32_t*>(&hi);
    o[i] = r;
  }
}
void kb2_sum(torch::Tensor c, torch::Tensor out, int64_t pdl) {
  const int64_t n = out.numel();
  TORCH_CHECK(c.scalar_type() == at::kFloat && out.scalar_type() == at::kBFloat16 && c.is_contiguous() && out.is_contiguous() &&
              c.numel() == 2 * n && n % 4 == 0, "kb2_sum: c f32 [2, n], out bf16 [n]");
  const int n4 = (int)(n / 4), th = 256;
  const float4* a = reinterpret_cast<const float4*>(c.data_ptr<float>());
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3((n4 + th - 1) / th);
  lc.blockDim = dim3(th);
  lc.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[1];
  at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at_[0].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = at_;
  lc.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kb2_sum_kernel, a, a + n4, reinterpret_cast<uint2*>(out.data_ptr()), n4, (int)pdl));
}
'''
LAB = r'''"""[KB2] R-kb2-1 (tools/kb2_patches.py): qkv/256 on 2x the CTAs: "b" = one strided-batched Lt call over the two K halves (pin
config, FP32 partials) + kb2_sum (bf16(C0 + C1), PDL); "s" = the pin with cuBLASLt split-K 2. KB2_ON = None: ship pin."""
KB2_ON = @ON@
KB2_PDL = 1
KB2_T = 256
_KB2_CUDA = r"""@CUDA@"""
_KB2 = {"mod": None, "err": None, "b": None, "s": None, "c": None, "calls": {}}
def _kb2_build():
    from torch.utils.cpp_extension import load_inline
    try:
        _KB2["mod"] = load_inline(name="kb2_" + hashlib.sha1(_KB2_CUDA.encode()).hexdigest()[:10],
                                  cpp_sources="void kb2_sum(torch::Tensor c, torch::Tensor out, int64_t pdl);",
                                  cuda_sources=_KB2_CUDA, functions=["kb2_sum"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    except Exception as ex:
        _KB2["err"] = repr(ex)[-400:]
def _kb2_prepare(state):
    _KB2["b"] = _KB2["s"] = None
    st = state.get("lt")
    p = st["pins"].get(("qkv", KB2_T)) if st is not None else None
    if p is None:
        print("[kb2] OFF: no qkv/256 pin", flush=True)
        return
    t1 = time.time()
    try:
        _pt1_wait("kb2")
        if _KB2["mod"] is None and _KB2["err"] is None:
            _kb2_build()
        mod, cfg = st["mod"], list(LT_PIN[("qkv", KB2_T)])
        w = _dg1_weight(state, "qkv")
        n, k = w.shape
        dev = w.device
        gen = torch.Generator(device=dev).manual_seed(4322)
        x = torch.randn((KB2_T, k), generator=gen, device=dev).to(torch.bfloat16)
        ref = x.double() @ w.double().t()
        ship = torch.empty((KB2_T, n), dtype=torch.bfloat16, device=dev)
        mod.lt_run(p[0], p[1], w, x, ship, 0.0, st["ws"])
        notes = []
        if _KB2["mod"] is not None:
            code = int(mod.lt_plan_k(KB2_T, n, k, 2, LT_WS, 32))
            pid, cnt = code >> 8, code & 255
            ai = next((i for i in range(cnt) if tuple(mod.lt_cfg(pid, i)) == tuple(cfg)), None)
            if ai is None:
                ai = int(mod.lt_make(pid, cfg, LT_WS))
            if ai >= 0 and tuple(mod.lt_cfg(pid, ai)) == tuple(cfg):
                _KB2["c"] = torch.empty((2, KB2_T, n), dtype=torch.float32, device=dev)
                outs = []
                for pdl in (0, 1, 1):
                    o = torch.full((KB2_T, n), float("nan"), dtype=torch.bfloat16, device=dev)
                    mod.lt_run(pid, ai, w, x, _KB2["c"], 0.0, st["ws"])
                    _KB2["mod"].kb2_sum(_KB2["c"], o, pdl)
                    outs.append(o)
                torch.cuda.synchronize()
                e64 = float((outs[0].double() - ref).norm() / ref.norm())
                es = float((outs[0].float() - ship.float()).norm() / ship.float().norm())
                ok = all(torch.equal(o, outs[0]) for o in outs) and bool(torch.isfinite(outs[0].float()).all()) and e64 < DG1_TOL[0] and es < DG1_TOL[2]
                notes.append(f"b: batch-2 K halves lt{ai}{'' if ai < cnt else ' (built)'} of {cnt} rel {e64:.1e} vs pin {es:.1e} -> {'ok' if ok else 'FAIL'}")
                if ok:
                    _KB2["b"] = (pid, ai, n, k)
            else:
                notes.append(f"b: pin config {tuple(cfg)} rejected for the batched K-split plan ({cnt} heuristic algos: "
                             f"{[tuple(mod.lt_cfg(pid, i))[:4] for i in range(min(cnt, 4))]})")
        else:
            notes.append(f"b: kb2_sum build failed {_KB2['err']}")
        pid = p[0]
        for red in (2, 1, 4):   # COMPUTE_TYPE workspace, INPLACE, OUTPUT_TYPE
            c2 = list(cfg)
            c2[3], c2[4] = 2, red
            ai = int(mod.lt_make(pid, c2, LT_WS))
            if ai < 0 or tuple(mod.lt_cfg(pid, ai)) != tuple(c2):
                notes.append(f"s: split-K 2 red {red} rejected")
                continue
            outs = []
            for _ in range(2):
                o = torch.full((KB2_T, n), float("nan"), dtype=torch.bfloat16, device=dev)
                mod.lt_run(pid, ai, w, x, o, 0.0, st["ws"])
                outs.append(o)
            torch.cuda.synchronize()
            e64 = float((outs[0].double() - ref).norm() / ref.norm())
            es = float((outs[0].float() - ship.float()).norm() / ship.float().norm())
            ok = torch.equal(outs[0], outs[1]) and bool(torch.isfinite(outs[0].float()).all()) and e64 < DG1_TOL[0] and es < DG1_TOL[2]
            notes.append(f"s: split-K 2 red {red} lt{ai} rel {e64:.1e} vs pin {es:.1e} -> {'ok' if ok else 'FAIL'}")
            if ok:
                _KB2["s"] = (pid, ai, n, k)
                break
        print(f"[kb2] ({time.time() - t1:.1f}s): " + "; ".join(notes), flush=True)
    except Exception as ex:
        _KB2["b"] = _KB2["s"] = None
        print(f"[kb2] prepare raised {ex!r}: ship pin", flush=True)
def _kb2(state, x):
    r = _KB2.get(KB2_ON)
    if r is None or x.shape[0] != KB2_T or x.shape[1] != r[3] or not x.is_contiguous() or x.data_ptr() % 256:
        return None
    try:
        st = state["lt"]
        w = _dg1_weight(state, "qkv")
        out = torch.empty((KB2_T, r[2]), dtype=torch.bfloat16, device=x.device)
        if KB2_ON == "b":
            c = torch.empty((2, KB2_T, r[2]), dtype=torch.float32, device=x.device)
            st["mod"].lt_run(r[0], r[1], w, x, c, 0.0, st["ws"])
            _KB2["mod"].kb2_sum(c, out, KB2_PDL)
        else:
            st["mod"].lt_run(r[0], r[1], w, x, out, 0.0, st["ws"])
        _KB2["calls"][KB2_ON] = _KB2["calls"].get(KB2_ON, 0) + 1
        return out
    except Exception as ex:
        _KB2[KB2_ON] = None
        print(f"[kb2] {KB2_ON} raised {ex!r}: ship pin", flush=True)
        return None
'''


def apply(s, on=None):
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header")
    if on:
        lines[1] += f" + R-kb2 qkv/256 {'batch-2 K-halves Lt + kb2_sum' if on == 'b' else 'Lt split-K 2'} [tools/kb2_patches.py --ship={on}]"
    else:
        lines[1] = "# nonce k3-kb2-ab (k3_hh_5_sk2_pz + R-kb2 lab arms: qkv/256 on 2x CTAs via cuBLASLt K split; tools/kb2_patches.py)"
    s = NL.join(lines)
    s = sub(s, "std::vector<int64_t> lt_cfg(int64_t pid, int64_t ai) {", LT_ADD + "std::vector<int64_t> lt_cfg(int64_t pid, int64_t ai) {", "lt_plan_k")
    s = sub(s, 'functions=["lt_plan", "lt_plan_b", "lt_cfg", "lt_make", "lt_run"]',
            'functions=["lt_plan", "lt_plan_b", "lt_plan_k", "lt_cfg", "lt_make", "lt_run"]', "lt functions")
    s = sub(s, "def _lt(state, proj, x):\n    st = state.get(\"lt\")\n",
            LAB.replace("@ON@", repr(on)).replace("@CUDA@", KB2_CUDA)
            + "def _lt(state, proj, x):\n    if KB2_ON and proj == \"qkv\" and x.shape[0] == KB2_T:\n"
              "        y = _kb2(state, x)\n        if y is not None:\n            return y\n    st = state.get(\"lt\")\n", "_lt hook")
    s = sub(s, "    _lt_prepare(state)\n", "    _lt_prepare(state)\n    _kb2_prepare(state)\n", "prepare call")
    s = sub(s, '        jobs.append(("lt", _lt_build))\n', '        jobs.append(("lt", _lt_build))\n        jobs.append(("kb2", _kb2_build))\n', "PT1 job")
    check('"""' not in KB2_CUDA and "trap" not in KB2_CUDA, "CUDA text")
    return s


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_5_sk2_pz.py")
    on = ARGS.get("ship")
    check(on in (None, "b", "s"), "--ship=b|s")
    out = ROOT / ARGS.get("out", f"kernels/lab/{base.stem}_kb2.py" if on else "kernels/lab/k3_kb2_ab.py")
    new = apply(base.read_bytes().decode("ascii"), on)
    n = len(new.encode())
    if "check" in FLAGS:
        check(out.read_bytes().decode("ascii") == new, f"{out.name} differs from apply({base.name})")
        print(f"kb2_patches: --check OK ({out.name}, {n:,} B)")
        return
    out.write_bytes(new.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    ast.parse(new)
    check(new.isascii(), "ascii")
    if on:
        check(n < LIMIT, f"size {n} >= {LIMIT}")
    print(f"kb2_patches: {base.name} -> {out.name} ({n:,} B, {n - base.stat().st_size:+,} B; {LIMIT - n:+,} vs the cap)")


if __name__ == "__main__":
    main()

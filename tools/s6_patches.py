"""s6_patches.py: S6 = L2 prefetch of the sliding window's old K / V rows under the qkv GEMV (notes/study/system_gap_oct28.md S6,
notes/study/s6_prefetch.md).

At T = 16 / 32 (KS3) and T = 128 / 256 (FA4 v6 chain) the qkv GEMV leaves HBM partly idle while KS3's late CTAs / FA4 later
gather the window's K / V from DRAM. kernel_fn forks a side stream at its start (the stream waits on the layer's first
event, so the prefetch kernel depends on nothing produced in the call), launches s6_kernel there (grid (2, B) x 256
threads, ~20 registers, no smem: page table -> `prefetch.global.L2::evict_normal` of each old K row and V row, 128 B each),
and joins the side stream back just before kernel_fn returns, so nothing on the critical path waits for it. A hint only:
outputs cannot change. Never evict_last; only rows of this call's window.
S6_ENABLE / S6_ROUTE {(batch, q_len): bool} read at call time (A/B arms), sliding layers with page size 1 only; prepare()
runs the kernel eagerly and inside a captured fork / join graph, any failure keeps S6 off.

    python -I tools/s6_patches.py [--n=1] [--base=k3_slim_5] [--off]   # -> kernels/muse_decode_layer/k3_s6_<n>.py + lab copy
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
NL = chr(10)

S6_CUDA = r'''
// ---- S6 (tools/s6_patches.py): L2 prefetch of a sliding window's old K / V rows (side stream, under the qkv GEMV) ----
namespace s6 {
__global__ void __launch_bounds__(256) s6_kernel(const int* __restrict__ seqlens, const int* __restrict__ pt, const int pt_cols,
                                                 const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc, const int L,
                                                 const int row_bytes, const int mode, const int pace_ns) {
#if KS1_ARCH_OK
  constexpr int PER = 4;                                        // keys per thread: 2 CTAs x 256 threads x 4 = 2048 slots
  const int b = blockIdx.y;
  const int seq = max(__ldg(seqlens + b) - L, 0);
  const int n_old = min(seq, ks1::WIN - 1);
  const int kstart = seq - n_old;
  const int* ptb = pt + (size_t)b * pt_cols;
  if (mode == 1) {                                              // page-table lines only (128 B = 32 entries each)
    const int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (n_old > 0 && 32 * i < n_old + 31) {
      const int k = min(kstart + 32 * i, kstart + n_old - 1);
      if (k < pt_cols) asm volatile("prefetch.global.L2::evict_normal [%0];\n" ::"l"(ptb + k));
    }
    return;
  }
  int rows[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int k = ((int)blockIdx.x * PER + j) * 256 + (int)threadIdx.x;
    rows[j] = (k < n_old && kstart + k < pt_cols) ? __ldg(ptb + kstart + k) : -1;
  }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    if (mode == 2 && j > 0) __nanosleep((unsigned)pace_ns);    // paced: 4 rounds spread over ~3 x pace
    if (rows[j] >= 0) {
      const size_t off = (size_t)rows[j] * (size_t)row_bytes;
      asm volatile("prefetch.global.L2::evict_normal [%0];\n" ::"l"(kc + off));
      asm volatile("prefetch.global.L2::evict_normal [%0];\n" ::"l"(vc + off));
    }
  }
#endif
}
}  // namespace s6

'''

S6_HOST = r'''
void s6_prefetch(torch::Tensor seqlens, torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, int64_t L,
                 int64_t mode, int64_t pace_ns) {
  const int64_t B = seqlens.numel();
  TORCH_CHECK(B >= 1 && seqlens.scalar_type() == at::kInt && seqlens.is_contiguous(), "s6: seqlens");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.size(0) == B && page_table.is_contiguous(),
              "s6: page_table");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && v_cache.scalar_type() == at::kByte && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && k_cache.numel() % (ks1::HKV * 64) == 0, "s6: caches");
  const at::cuda::CUDAGuard guard(seqlens.device());
  s6::s6_kernel<<<dim3(2u, (unsigned)B, 1u), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      seqlens.data_ptr<int>(), page_table.data_ptr<int>(), (int)page_table.size(1), k_cache.data_ptr<uint8_t>(),
      v_cache.data_ptr<uint8_t>(), (int)L, (int)(ks1::HKV * 64), (int)mode, (int)pace_ns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

S6_CPP = ("void s6_prefetch(torch::Tensor seqlens, torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, int64_t L,\n"
          "                 int64_t mode, int64_t pace_ns);\n")

S6_PY = r'''

def _s6_fork(data, cfg, state):
    if not (S6_ENABLE and state.get("s6") and S6_ROUTE.get((cfg["batch"], cfg["q_len"])) and cfg["window"] == 2048
            and cfg["page_size"] == 1 and _KS1["mod"] is not None):
        return None
    pt, sl, kc, vc = data["page_table"], data["cache_seqlens"], data["k_cache"], data["v_cache"]
    if not (pt.dtype == torch.int32 and pt.dim() == 2 and pt.is_contiguous() and pt.shape[0] == cfg["batch"]
            and sl.dtype == torch.int32 and sl.is_contiguous() and sl.numel() == cfg["batch"]
            and kc.dtype == FP8 and vc.dtype == FP8 and kc.is_contiguous() and vc.is_contiguous()):
        return None
    main, side = torch.cuda.current_stream(), state["s6_stream"]
    try:
        side.wait_stream(main)
        with torch.cuda.stream(side):
            _KS1["mod"].s6_prefetch(sl, pt, kc.view(torch.uint8), vc.view(torch.uint8), int(cfg["q_len"]), int(S6_MODE),
                                    int(S6_PACE_NS))
    except Exception as ex:
        main.wait_stream(side)
        state["s6"] = False
        print(f"[s6] prefetch raised at call time, S6 off from now on: {ex!r}", flush=True)
        return None
    if (cfg["batch"], cfg["q_len"]) not in _S6["seen"]:
        _S6["seen"].add((cfg["batch"], cfg["q_len"]))
        print(f"[s6] first prefetch B={cfg['batch']} L={cfg['q_len']}", flush=True)
    return side


def _s6_join(side):
    if side is not None:
        torch.cuda.current_stream().wait_stream(side)


def _s6_prepare(state, device):
    state["s6"] = False
    if not (S6_ENABLE and _KS1["mod"] is not None):
        return
    try:
        state["s6_stream"] = torch.cuda.Stream(device, priority=-1)
        d = _k1fa4_case(16, 3000, 8, 2048, 1301, False, device)
        cfg = dict(d["config"], tokens=16 * 8)
        state["s6"] = True
        before = (d["k_cache"].view(torch.uint8).clone(), d["v_cache"].view(torch.uint8).clone())
        side = _s6_fork(d, cfg, state)
        _s6_join(side)
        torch.cuda.synchronize(device)
        x = torch.ones(1024, device=device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            side = _s6_fork(d, cfg, state)
            y = x * 2.0
            _s6_join(side)
        g.replay()
        g.replay()
        torch.cuda.synchronize(device)
        same = (bool(torch.equal(before[0], d["k_cache"].view(torch.uint8))) and bool(torch.equal(before[1], d["v_cache"].view(torch.uint8)))
                and float(y.sum()) == 2048.0)
        state["s6"] = bool(same and state["s6"])
        del g, d, before, x, y
        print(f"[s6] prefetch kernel ok eager + graph fork / join: {state['s6']}; route {S6_ROUTE if state['s6'] else 'off'}",
              flush=True)
    except Exception as ex:
        state["s6"] = False
        print(f"[s6] prepare FAILED {repr(ex)[-800:]}: no prefetch", flush=True)
'''

PY_EDITS = [
    ("cpp decl", "std::vector<int64_t> ks3_info(int64_t NS);\n", "std::vector<int64_t> ks3_info(int64_t NS);\n" + S6_CPP),
    ("build fns", 'functions=["ks1_forward", "ks1_info", "ks3_forward", "ks3_info"] + _dg1_fold("fns"),',
     'functions=["ks1_forward", "ks1_info", "ks3_forward", "ks3_info", "s6_prefetch"] + _dg1_fold("fns"),'),
    ("consts", "VSL_PREPARE_BY_S = 240.0\n",
     "VSL_PREPARE_BY_S = 240.0\nS6_ENABLE = @ON@\nS6_ROUTE = {(16, 1): True, (32, 1): True, (16, 8): True, (32, 8): True}\n"
     "S6_MODE = 0\nS6_PACE_NS = 2000\n_S6 = {\"seen\": set()}\n"),
    ("prepare", "    _vsl_prepare(state, device, t0)\n", "    _vsl_prepare(state, device, t0)\n    _s6_prepare(state, device)\n"),
    ("funcs", "\n\ndef _ks1_case(ctx, length, seed, row0, device):\n", S6_PY + "\n\ndef _ks1_case(ctx, length, seed, row0, device):\n"),
    ("fork", '    dense = state["dense"]\n    qkv = _gd(state, "qkv", data["attn_input"])\n',
     '    dense = state["dense"]\n    s6side = _s6_fork(data, cfg, state)\n    qkv = _gd(state, "qkv", data["attn_input"])\n'),
    ("join", '    next_attn_input, stream, _ = _v9_bnd("ffn", stream, ffn_output, None, state, tokens)\n    return next_attn_input, stream, topk_ids\n',
     '    next_attn_input, stream, _ = _v9_bnd("ffn", stream, ffn_output, None, state, tokens)\n    _s6_join(s6side)\n'
     '    return next_attn_input, stream, topk_ids\n'),
]


def check(ok, msg):
    if not ok:
        raise SystemExit(f"[s6_patches] FAIL: {msg}")


def apply(text, on=True, log=None):
    i0 = text.index('KS1_CUDA_SRC = r"""')
    i1 = text.index('"""', i0 + len('KS1_CUDA_SRC = r"""'))
    cu = text[i0:i1]
    a = "}  // namespace ks3\n"
    check(cu.count(a) == 1, "anchor namespace ks3")
    cu = cu.replace(a, a + S6_CUDA)
    a = "std::vector<int64_t> ks3_info(int64_t NS) {\n"
    check(cu.count(a) == 1, "anchor ks3_info")
    cu = cu.replace(a, S6_HOST + "\n" + a)
    t = text[:i0] + cu + text[i1:]
    for what, old, new in PY_EDITS:
        new = new.replace("@ON@", "True" if on else "False")
        check(t.count(old) == 1, f"Python anchor {what!r} found {t.count(old)} times")
        t = t.replace(old, new)
    if log:
        log(f"S6 applied: +{len(t.encode()) - len(text.encode())} B, S6_ENABLE {on}")
    return t


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    n = args.get("n", "1")
    on = "--off" not in sys.argv
    base = ROOT / f"kernels/muse_decode_layer/{args.get('base', 'k3_slim_5')}.py"
    sfx = "" if on else "off"
    out = ROOT / f"kernels/muse_decode_layer/k3_s6_{n}{sfx}.py"
    lab = ROOT / f"kernels/lab/k3_s6_{n}{sfx}_ab.py"
    check(not out.exists() or "--force" in sys.argv, f"{out.name} exists (new names only)")
    tag = f"k3-s6-{n}{sfx}"
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    check(fb == b, f"{base.name} is not a finish() fixed point")
    nonce = (f"# nonce {tag} ({base.stem} + S6: side-stream L2 prefetch of the sliding window's K / V under the qkv GEMV at "
             f"T = 16 / 32 / 128 / 256, {'on' if on else 'off'}, self-tested with fallback)")
    t = apply(b.replace(bn, nonce), on, rep.append)
    s = GM.finish(t, nonce, tag, chk_base, rep)
    for r in rep:
        print(f"[s6_patches] OK  {r}")
    out.write_text(s, encoding="utf-8", newline=NL)
    if "--nolab" not in sys.argv:
        lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[s6_patches] wrote {out.relative_to(ROOT)} ({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

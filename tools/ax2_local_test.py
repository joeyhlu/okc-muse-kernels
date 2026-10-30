"""ax2 local test (RTX 3070, sm_86): va1_main ship (204) vs ax2 codes on synthetic paged FP8 caches. Run via tools/local/cudaenv.sh:
   python -I tools/ax2_local_test.py <file.py with K1_CUDA_SRC> [codes, e.g. 1303]
"""
import ast
import pathlib
import sys

import torch
from torch.utils.cpp_extension import load_inline

path = pathlib.Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding="utf-8"))
cu = None
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "K1_CUDA_SRC":
        cu = node.value.value
assert cu
cut = cu.index("template <int L, int ST>\n__global__ void __launch_bounds__(NTHR, (L == 1) ? 2 : K1_L8_MINB) k1_main")
body = cu[:cut].replace("(__CUDA_ARCH__ >= 890)", "(__CUDA_ARCH__ >= 800)").replace("fence.acquire.gpu;", "fence.acq_rel.gpu;").replace("fence.release.gpu;", "fence.acq_rel.gpu;")
h0 = cu.index("template <int ST, int VX = 0>\nstatic int va_go")
h1 = cu.index("int64_t k1_occupancy(int64_t L, int64_t ST) {")
host = cu[h0:h1]
src = body + "}  // namespace k1\n" + host + r"""
int64_t ax2t(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor page_table,
             torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale,
             torch::Tensor part_o, torch::Tensor part_lse, torch::Tensor out, torch::Tensor counters,
             int64_t B, int64_t Hq, int64_t Hkv, int64_t W, int64_t page_shift, int64_t Gx, int64_t segk, double qk_sm,
             int64_t code, int64_t occ_only) {
  k1::Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.page_table = page_table.data_ptr<int>();
  p.k_cache = k_cache.data_ptr<uint8_t>();
  p.v_cache = v_cache.data_ptr<uint8_t>();
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.inv_freq = k_scale.data_ptr<float>();
  p.part_o = part_o.data_ptr<float>();
  p.part_lse = part_lse.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.counters = counters.data_ptr<int>();
  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.W = (int)W; p.page_shift = (int)page_shift;
  p.pt_stride = (int)page_table.stride(0); p.Gx = (int)Gx; p.min_chunk = (int)segk; p.use_rope = 0;
  p.fold = 1;
  p.qk_sm = (float)qk_sm;
  p.pt_bytes = (long long)page_table.storage().nbytes() - (long long)page_table.storage_offset() * 4;
  const dim3 grid((unsigned)Gx, (unsigned)(Hkv / 2));
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (code < 1000) {
    if (occ_only) return code % 10 == 3 ? va_go<3>(nullptr, grid, st) : va_go<4>(nullptr, grid, st);
    if (code % 10 == 3) va_go<3>(&p, grid, st); else va_go<4>(&p, grid, st);
    return 0;
  }
  int vx = (int)((code - 1000) / 100);
  if ((vx & 1) && (reinterpret_cast<uintptr_t>(page_table.data_ptr()) & 15)) vx &= ~1;
  return ax2_dispatch(occ_only ? nullptr : &p, grid, st, vx, (int)(code % 10), false);
}
"""
cpp = """
#include <torch/extension.h>
int64_t ax2t(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor page_table,
             torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale,
             torch::Tensor part_o, torch::Tensor part_lse, torch::Tensor out, torch::Tensor counters,
             int64_t B, int64_t Hq, int64_t Hkv, int64_t W, int64_t page_shift, int64_t Gx, int64_t segk, double qk_sm,
             int64_t code, int64_t occ_only);
"""
mod = load_inline(name="ax2t_" + str(abs(hash(src)) % 10**8), cpp_sources=cpp, cuda_sources=src, functions=["ax2t"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86"], verbose=False)
dev = torch.device("cuda")
SMS = torch.cuda.get_device_properties(dev).multi_processor_count
FP8 = torch.float8_e4m3fn


def mk_case(B, ctx, page, kvs, seed, odd_stride=False, misalign=False):
    g = torch.Generator().manual_seed(seed)
    seq = [max(1, int(ctx * (0.6 + 0.8 * float(torch.rand(1, generator=g))))) for _ in range(B)]
    max_pages = (max(seq) + 1 + page - 1) // page + 1
    if odd_stride:
        max_pages += 1 - (max_pages % 2)   # odd, so B * stride entries is not a multiple of 4
    total = B * max_pages
    perm = torch.randperm(total, generator=g).to(torch.int32)
    if misalign:
        big = torch.empty(total + 1, dtype=torch.int32)
        big[1:] = perm
        pt = big.to(dev)[1:].view(B, max_pages)
    else:
        pt = perm.view(B, max_pages).to(dev)
    rows = total * page
    kc = (torch.randn(rows, 2, 64, generator=g) * 0.5).to(FP8).view(torch.uint8).to(dev)
    vc = (torch.randn(rows, 2, 64, generator=g)).to(FP8).view(torch.uint8).to(dev)
    qkv = (torch.randn(B, (16 + 2 + 2 + 16) * 64, generator=g)).to(torch.bfloat16).to(dev)
    sl = torch.tensor(seq, dtype=torch.int32, device=dev)
    loc = torch.tensor([int(pt[b, seq[b] // page]) * page + seq[b] % page for b in range(B)], dtype=torch.int64, device=dev)
    ks = torch.tensor([kvs], dtype=torch.float32, device=dev)
    vs = torch.tensor([kvs * 2], dtype=torch.float32, device=dev)
    return dict(qkv=qkv, sl=sl, loc=loc, pt=pt, kc=kc, vc=vc, ks=ks, vs=vs, B=B, page=page)


ws_o = torch.zeros(16 * 1024 * 64, dtype=torch.float32, device=dev)
ws_l = torch.zeros(16 * 1024, dtype=torch.float32, device=dev)
cnt = torch.zeros(1024, dtype=torch.int32, device=dev)


def run(d, code, W=None):
    B = d["B"]
    W = W if W is not None else (8 if B == 1 else 1)
    kc, vc = d["kc"].clone(), d["vc"].clone()
    out = torch.full((B, 16 * 64), float("nan"), dtype=torch.bfloat16, device=dev)
    G = max(B + 1, SMS * 2)
    mod.ax2t(d["qkv"], d["sl"], d["loc"], d["pt"], kc, vc, d["ks"], d["vs"], ws_o, ws_l, out, cnt,
             B, 16, 2, W, d["page"].bit_length() - 1, G, 400, 0.125, code, 0)
    torch.cuda.synchronize()
    clean = not (int(cnt.abs().sum()) or float(ws_o.abs().sum()) or float(ws_l.abs().sum()))
    return out.float(), kc, vc, clean


codes = [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1000 + 100 * vx + st for (st, vx) in [(4, 8), (4, 1), (3, 1), (4, 3), (3, 3), (4, 2), (4, 4), (4, 5)]]
cases = [(2, 3000, 1, 1.0, 1, {}), (16, 3000, 1, 1.0, 2, {}), (33, 700, 1, 1.0, 3, {}), (5, 9000, 1, 1.0 / 64, 4, {}),
         (3, 1500, 64, 1.0, 5, {}), (16, 2000, 16, 1.0, 6, {}), (32, 2500, 1, 1.0, 7, {}), (1, 9000, 1, 1.0, 8, {}),
         (7, 2600, 1, 1.0, 9, dict(odd_stride=True)), (6, 2600, 1, 1.0, 10, dict(misalign=True)),
         (9, 130, 1, 1.0, 11, {}), (4, 70, 4, 1.0, 12, dict(odd_stride=True))]
bad = 0
for (B, ctx, page, kvs, seed, kw) in cases:
    d = mk_case(B, ctx, page, kvs, seed, **kw)
    ref, rk, rv, rclean = run(d, 204)
    msg = []
    for c in codes:
        o, k, v, clean = run(d, c)
        vx = (c - 1000) // 100
        if vx & 4:
            ok = clean
            msg.append(f"{c}:skel{'' if clean else ' DIRTY'}")
        else:
            diff = float((o - ref).abs().max()) / max(1e-6, float(ref.abs().max()))
            same = bool(torch.equal(o, ref))
            kv_ok = bool(torch.equal(k, rk) and torch.equal(v, rv))
            ok = clean and kv_ok and bool(torch.isfinite(o).all()) and diff < (2e-2 if vx & 2 else 8e-3)
            msg.append(f"{c}:{'bit' if same else f'{diff:.1e}'}{'' if kv_ok else ' KV!'}{'' if clean else ' DIRTY'}")
        bad += 0 if ok else 1
    print(f"B={B} ctx={ctx} page={page} {kw} ship clean={rclean} finite={bool(torch.isfinite(ref).all())} | " + " ".join(msg), flush=True)
print("FAIL" if bad else "ALL OK", bad)

"""Local consumer-rate A/B on the RTX 3070 (sm_86): R-cr-0's predicated consumer (CR_CUDA as shipped in that burst,
kept in kernels/lab/cr_check_box_r0.py) vs the NG-templated consumer (kernels/lab/cr_k1_lab.py), weights L2-resident
(every unit points at expert 0 / block 0, 1.15 MB) so the consumer, not DRAM, sets the time. Arms: S (null), R (real),
for ng = 1 / 4 / 8 via token counts. Prints us and the R/S ratio per arm.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/kernels/lab/cr_local_ab.py"
"""
import ast
import hashlib
import pathlib
import statistics
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "problems/muse_moe_w4a16"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
import cr_k1_lab as L  # noqa: E402
import nvfp4  # noqa: E402


def old_cuda():
    src = (ROOT / "kernels/lab/cr_check_box_r0.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CR_CUDA":
            return node.value.value
    raise RuntimeError("CR_CUDA not found")


def build(cuda, tag):
    from torch.utils.cpp_extension import load_inline
    h = hashlib.sha1(cuda.encode()).hexdigest()[:8]
    return load_inline(name=f"cr_ab_{tag}_{h}", cpp_sources=L.CR_CPP, cuda_sources=cuda, functions=["cr_run", "cr_attrs"],
                       extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_86,code=sm_86",
                                          "-DCR_NST=1", "-DCR_NGMAX=8"])


def main():
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(5252)
    w13, w13_sf, gs13 = nvfp4.random_weights(8192, 4096, generator=g, device=dev, leading=(1,))
    ws, gsc = L.cr_prepack(w13, w13_sf, gs13.expand(1).contiguous())
    ws = ws.expand(32, -1, -1, -1).contiguous() if False else ws
    G = torch.cuda.get_device_properties(0).multi_processor_count
    mods = {"old": build(old_cuda(), "old"), "new": build(L.CR_CUDA, "new")}
    for ntok in (8, 32, 64):
        bs = ntok
        ids = torch.full((bs, 8), 100, dtype=torch.int32, device=dev)
        ids[:, 0] = 0
        ids[:, 1:] = (100 + torch.arange(7, device=dev)).int()
        d = {"hidden_states": torch.randn((bs, 4096), device=dev).to(torch.bfloat16), "topk_ids": ids,
             "topk_weights": torch.full((bs, 8), 0.125, device=dev), "config": {}}
        P = L.cr_plan(d, G)
        # replicate expert 0's 16 blocks x 24 times: 384 units, all reading 18.9 MB -> use block 0 only (1.15 MB, L2)
        u = P["units"][:1].repeat(384, 1).contiguous()
        u[:, 1] = 0
        P["units"] = u
        tot = 32 * 384
        P["cuts"] = torch.tensor([tot * i // G for i in range(G)] + [tot], dtype=torch.int32, device=dev)
        inter = torch.zeros((P["R"], 4096), dtype=torch.bfloat16, device=dev)
        res = {}
        for nm, mod in mods.items():
            st = L.CrState(mod, ws, gsc, G, dev)
            for mode, an in ((1, "S"), (0, "R")):
                f = st.fn(P, inter, mode)
                for _ in range(3):
                    f()
                torch.cuda.synchronize()
                ts = []
                for _ in range(15):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record(); f(); e.record(); e.synchronize()
                    ts.append(s.elapsed_time(e) * 1e3)
                res[f"{nm}-{an}"] = statistics.median(ts)
            assert int(st.err[0]) == 0, st.err.tolist()
        gb = tot * 36864 / 1e9
        print(f"ntok {ntok} (ng {P['units'][0, 3].item()}): " + " ".join(f"{k} {v:.0f}us" for k, v in res.items()) +
              f" | consumer GB/s old {gb / res['old-R'] * 1e6:.0f} new {gb / res['new-R'] * 1e6:.0f} | new/old R "
              f"{res['new-R'] / res['old-R']:.3f}")


if __name__ == "__main__":
    main()

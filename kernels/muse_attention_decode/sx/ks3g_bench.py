"""KS3G box bench (R-ks3g-0): judge-style RAW replay us (cold L2 + gate) on the board's decode 16 / 32 sliding inputs
(= the decode layer's d16s / d32s attention inputs) for KS3 as shipped in k3_gc_18 (d16: VSL NS 16 red, d32: NS 8 ticket),
KS3 NS 8 at d16, KS3G (NS, NP) variants and their null consumers (gather ceiling of the same access pattern);
rel err vs the shipped KS3 output; per-CTA %globaltimer marks (us after the first entry, median / max).
  python3 -u kernels/muse_attention_decode/sx/ks3g_bench.py [--reps 30]"""
import argparse
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "judge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge as J  # noqa: E402
import ks3g_src  # noqa: E402

GVARS = [(8, 2), (16, 2), (32, 2), (8, 4), (16, 4)]
NULLS = [(8, 2), (16, 2), (32, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--local", action="store_true")
    a = ap.parse_args()
    mod, m = (ks3g_src.build(ROOT, arch="86", name="ks3g_local_v1", local=True) if a.local
              else ks3g_src.build(ROOT, arch="90a", name="ks3g_box_v1"))
    for ns, np_ in GVARS:
        print(f"info KS3G NS{ns} NP{np_}: regs/local/smem/occ {mod.ks3g_info(ns, np_, 0)}", flush=True)
    print(f"info KS3 NS8 {mod.ks3_info(8)} NS16 {mod.ks3_info(16)}", flush=True)
    prob = J.load_problem(ROOT / "problems" / "muse_attention_decode")
    ref = prob.ref
    timer = J.GpuTimer()
    ovh = timer.replay_overhead_us()
    print(f"replay overhead (box timer) {ovh:.2f} us (judge ~4.1)", flush=True)
    dev = "cuda"
    cnt = torch.zeros(1024, dtype=torch.int32, device=dev)
    wso = torch.zeros(64 * 16 * 64, dtype=torch.float32, device=dev)
    wsl = torch.zeros(64 * 16, dtype=torch.float32, device=dev)
    vpo = torch.zeros(1 << 20, dtype=torch.float32, device=dev)
    vpml = torch.zeros(1 << 17, dtype=torch.float32, device=dev)
    noprof = torch.empty((0,), dtype=torch.int64, device=dev)
    benches = [c for c in prob.benches if c["layer"] == "sliding" and c["batch"] in (16, 32) and c.get("q_len", 1) == 1]
    for c in benches:
        d = J.make_input(ref, J.case_args(c), c["seed"], None, dev)
        cfg = d["config"]
        B = cfg["batch"]
        cos_sin = m["k1fa4_cos_sin"](cfg["rope_theta"] or 500000.0, 64, torch.device(dev))
        qk = float(cfg["sm_scale"] * cfg["qk_scale_factor"])
        rope = 1 if cfg["rope_theta"] else 0
        seqs = d["cache_seqlens"].tolist()
        nbytes = sum(min(max(s - 1, 0), 2047) for s in seqs) * 256
        print(f"== {c['load']} sliding B{B}: seq {min(seqs)}..{max(seqs)}, old-KV {nbytes / 1e6:.2f} MB", flush=True)

        def f_ks3(dd, ns, mg, prof=noprof):
            out = torch.empty((B, 1024), dtype=torch.bfloat16, device=dev)
            po, pml = (vpo, vpml) if mg else (torch.empty(B * ns * 16 * 64, dtype=torch.float32, device=dev),
                                             torch.empty(B * ns * 32, dtype=torch.float32, device=dev))
            mod.ks3_forward(dd["qkv"].view(B, -1), dd["cache_seqlens"], dd["cache_loc"], dd["positions"], dd["page_table"],
                            dd["k_cache"].view(torch.uint8), dd["v_cache"].view(torch.uint8), cos_sin, po, pml, cnt, out, prof,
                            ns + 100 * mg, qk, 1e-5, 1, 1, rope)
            return out

        def f_g(dd, ns, np_, nullc=0, prof=noprof):
            out = torch.empty((B, 1024), dtype=torch.bfloat16, device=dev)
            mod.ks3g_forward(dd["qkv"].view(B, -1), dd["cache_seqlens"], dd["cache_loc"], dd["positions"], dd["page_table"],
                             dd["k_cache"].view(torch.uint8), dd["v_cache"].view(torch.uint8), cos_sin, wso, wsl, cnt, out, prof,
                             ns, np_, nullc, qk, 1e-5, 1, 1, rope)
            return out

        ship = (16, 1) if B == 16 else (8, 0)
        arms = [(f"KS3 ship NS{ship[0]} MG{ship[1]}", lambda dd, pr=noprof: f_ks3(dd, ship[0], ship[1], pr), ship[0])]
        if B == 16:
            arms.append(("KS3 NS8 MG0", lambda dd, pr=noprof: f_ks3(dd, 8, 0, pr), 8))
        for ns, np_ in GVARS:
            arms.append((f"KS3G NS{ns} NP{np_}", lambda dd, pr=noprof, ns=ns, np_=np_: f_g(dd, ns, np_, 0, pr), ns))
        for ns, np_ in NULLS:
            arms.append((f"NULL NS{ns} NP{np_}", lambda dd, pr=noprof, ns=ns, np_=np_: f_g(dd, ns, np_, 1, pr), ns))
        base_out = None
        res = {}
        for name, fn, ns in arms:
            static = J.clone_data(d)
            g, out = timer.capture(lambda dd, fn=fn: fn(dd), static)
            raw = [timer.time_replay(g) for _ in range(a.reps)]
            med = statistics.median(raw)
            res[name] = med
            torch.cuda.synchronize()
            err = ""
            if name.startswith("KS3 ship"):
                base_out = out.float().clone()
            elif not name.startswith("NULL") and base_out is not None:
                err = f" err vs ship {float((out.float() - base_out).norm() / base_out.norm()):.2e}"
            clean = int(cnt.abs().sum()) == 0 and int(wso.view(torch.int32).abs().sum()) == 0
            gbw = f" = {nbytes / (med - 4.1) / 1e6:.2f} TB/s net of 4.1" if name.startswith("NULL") else ""
            print(f"  {name:18s}: raw {med:6.2f} us (min {min(raw):6.2f}){gbw}{err}{'' if clean else ' DIRTY'}", flush=True)
            # marks (separate graph with the prof buffer)
            prof = torch.zeros(8 * B * ns, dtype=torch.int64, device=dev)
            g2, _ = timer.capture(lambda dd, fn=fn: fn(dd, prof), J.clone_data(d))
            rows = []
            for _ in range(8):
                prof.zero_()
                timer.time_replay(g2)
                torch.cuda.synchronize()
                pm = prof.view(-1, 8).cpu().double()
                t0 = pm[:, 0].min()
                rel = torch.where(pm > 0, (pm - t0) / 1000.0, torch.full_like(pm, float("nan")))
                rows.append(rel)
            st = torch.stack(rows)                         # [rep, cta, 8]
            med_m = st.nanmedian(dim=1).values.median(dim=0).values
            mx = torch.stack([torch.tensor([float(r[:, k][~torch.isnan(r[:, k])].max()) if (~torch.isnan(r[:, k])).any() else float("nan")
                                            for k in range(8)]) for r in rows]).median(dim=0).values
            print("     marks " + " ".join(f"{k}:{med_m[k]:.2f}/{mx[k]:.2f}" for k in range(8)), flush=True)
        sh = res[arms[0][0]]
        best = min((v, k) for k, v in res.items() if k.startswith("KS3G"))
        print(f"  -> best KS3G {best[1]} {best[0]:.2f} vs ship {sh:.2f}: delta {best[0] - sh:+.2f} us", flush=True)


if __name__ == "__main__":
    main()

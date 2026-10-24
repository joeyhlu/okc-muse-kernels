"""K1 v4 CPU emulation: the L=8 layout of k1_v4.py on top of k1_emulate.py / k1_v3_emulate.py (both still pass).

  python3 -I k1_v4_emulate.py [--quick]

v4 changes checked here:
  1. K is converted like V: each loader thread converts its own two K chunks (L=8 K ring swizzle ch ^ (x&1); L=1 keeps ch ^ (4*(x&1))) and two
     V chunks (ch ^ (x&1)) into fp16 tiles Kh / Vh with physical chunk c16 ^ (x&7) ^ (2*head). QK B fragments come from
     ldmatrix.x4 (non-trans), lane address (key 8jn + (lane&7), chunk 4hs + (lane>>3)); r[0], r[1] = b0, b1 of k-step
     2hs and r[2], r[3] of k-step 2hs+1. The test rebuilds every lane's QK B fragments from the byte image with PTX
     ldmatrix semantics and compares them with K[key][dim]; it also checks the fp16 Q A-fragment indexing
     (u32 index 8s+t / +4, rows g / g+8) against Q directly, and bank conflicts of every ldmatrix phase and of the
     conv_kv LDS/STS phases.
  2. L=8 queries in fp16 (k1_emulate.L8_Q_FP16 = True), v3 warp roles; the end-to-end cases of k1_emulate are re-run
     (judge rule + tight check against the BF16-query reference + exact key coverage + fold vs separate merge).
     --quick runs the L=8 cases only (smaller CPU load).
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops the script dir
import k1_emulate as E  # noqa: E402
import k1_v3_emulate as V3  # noqa: E402

ROWB = 128


def v4_k_fragment_test(seed=3):
    g = torch.Generator().manual_seed(seed)
    K = torch.randint(0, 256, (64, 2, 64), generator=g, dtype=torch.uint8)
    K[(K & 0x7F) == 0x7F] = 0
    ring = torch.zeros(64 * ROWB, dtype=torch.uint8)
    for x in range(64):
        row = K[x].reshape(-1)
        for ch in range(8):
            o = x * ROWB + 16 * (ch ^ (x & 1))
            ring[o:o + 16] = row[16 * ch:16 * ch + 16]
    kh = torch.zeros(64 * 256, dtype=torch.uint8)
    f16 = E.dec(K).to(torch.float16)
    lds = {}
    sts = {}
    for tid in range(256):
        x, part = tid >> 2, tid & 3
        for u in range(2):
            ch = 2 * part + u
            o = x * ROWB + 16 * (ch ^ (x & 1))
            lds.setdefault((tid // 8, u), []).append(o)
            vals = E.dec(ring[o:o + 16]).to(torch.float16).view(torch.uint8)
            hh, c0 = ch >> 2, 2 * (ch & 3)
            for v in range(2):
                off = V3.vh_off(x, hh, c0 + v)
                kh[off:off + 16] = vals[16 * v:16 * v + 16]
                sts.setdefault((tid // 8, u, v), []).append(off)
    assert all(V3.conflict_free(a, 16) for a in lds.values()), "conv_kv K LDS.128 bank conflict"
    assert all(V3.conflict_free(a, 16) for a in sts.values()), "conv_kv K STS.128 bank conflict"
    kh16 = kh.view(torch.float16)
    n_ph = 0
    for hh in range(2):
        for jn in range(8):
            for hs in range(2):
                addr = [V3.vh_off(8 * jn + (lane & 7), hh, 4 * hs + (lane >> 3)) for lane in range(32)]
                for q in range(4):
                    assert V3.conflict_free(addr[8 * q:8 * q + 8], 16), "ldmatrix (K) phase bank conflict"
                    n_ph += 1
                for lane in range(32):
                    gq, t = lane >> 2, lane & 3
                    # non-trans: lane gets [row g][cols 2t, 2t+1] of each matrix (row i of matrix q at lane 8q+i)
                    regs = [(kh16[addr[8 * q + gq] // 2 + 2 * t], kh16[addr[8 * q + gq] // 2 + 2 * t + 1]) for q in range(4)]
                    for ks in range(2):
                        s4 = 2 * hs + ks
                        for kb, (lo, hi) in ((0, regs[2 * ks]), (8, regs[2 * ks + 1])):
                            d0 = 16 * s4 + kb + 2 * t                 # B[k = dim][n = key 8jn + g]
                            assert lo == f16[8 * jn + gq, hh, d0] and hi == f16[8 * jn + gq, hh, d0 + 1], "K fragment"
    # fp16 Q A-fragments: qs u32 [row][32] holds dims 2*lane, 2*lane+1
    Q = torch.randn((2, 64, 64), generator=g).half()
    qs32 = Q.view(torch.int32).view(2 * 64, 32)
    for hh in range(2):
        for mt in range(4):
            for lane in range(32):
                gq, t = lane >> 2, lane & 3
                base = hh * 64 + 16 * mt + gq
                for s4 in range(4):
                    regs = [qs32[base, 8 * s4 + t], qs32[base + 8, 8 * s4 + t], qs32[base, 8 * s4 + 4 + t],
                            qs32[base + 8, 8 * s4 + 4 + t]]
                    want = [(base, 16 * s4 + 2 * t), (base + 8, 16 * s4 + 2 * t), (base, 16 * s4 + 8 + 2 * t),
                            (base + 8, 16 * s4 + 8 + 2 * t)]
                    for rv, (row, d) in zip(regs, want):
                        pair = torch.tensor([int(rv)], dtype=torch.int32).view(torch.float16)
                        assert pair[0] == Q.view(128, 64)[row, d] and pair[1] == Q.view(128, 64)[row, d + 1], "Q frag"
    return n_ph


def main():
    t0 = time.time()
    n = V3.v3_fragment_test()
    print(f"[v4 frag] fp16 V tile + {n} ldmatrix.x4.trans phases (PV) OK")
    n = v4_k_fragment_test()
    print(f"[v4 frag] fp16 K tile (conv_kv), {n} ldmatrix.x4 phases conflict-free, every lane's QK B fragment == K;"
          f" fp16 Q A-fragments == Q")
    E.warp_roles = V3.warp_roles_v3
    E.L8_Q_FP16 = True
    if "--quick" in sys.argv:
        E.CASES = [c for c in E.CASES if c[1].get("q_len", 1) == 8]
    E.main()
    print(f"[v4] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

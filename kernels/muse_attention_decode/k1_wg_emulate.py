"""K1-WG (wgmma L=8) CPU emulation: layouts of k1_wg_design.md section 3 and the E4M3-P numerics (section 1).

  python3 -I k1_wg_emulate.py [--quick]

1. Layout tests (PTX/CUTLASS canonical K-major INTERLEAVE: byte(r, k) = start + (r%8)*16 + (r/8)*SBO + (k%16) +
   (k/16)*LBO, cute/arch/mma_sm90_desc.hpp):
   - K tile written by cp.async (thread -> key pair k0 = 16q+8b+2t', k0+1, chunk c) rebuilt per head and k32 step == K^T
   - Q tile rebuilt == Q
   - V^T tile written by the per-thread halfword transpose rebuilt == V[key(s, k)][n], key(s,k)=16q+8(i>>1)+2t+(i&1)
   - P packing: S-accumulator positions (rows 16w+g/+8, keys 8j+2t+e) -> A registers in the m16n8k32 8-bit A layout
     -> logical P_perm; P_perm @ B_V == P @ V
   - bank conflicts of the V^T STS.U16 phases and the own-V LDS.128 phases (raw V: chunk e of thread tid at
     e*4096 + tid*16)
2. Numerics: k1_emulate end-to-end cases with the v3/v4 L=8 warp roles, E4M3 queries and E4M3 P (k1_emulate.L8_P_E4M3);
   judge rule (own new entries, E4M3-query bound, row rule); reports error / bound per case. --quick: L=8 cases only.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops the script dir
import k1_emulate as E  # noqa: E402
import k1_v3_emulate as V3  # noqa: E402

FP8 = torch.float8_e4m3fn


def il_addr(r, k, lbo, sbo):
    """K-major INTERLEAVE byte offset of element (row r, k) of an 8-bit operand."""
    return (r % 8) * 16 + (r // 8) * sbo + (k % 16) + (k // 16) * lbo


# V^T tile: per (step s, head h) region of 8 row-groups x SBO; SBO = 272 and region stride 2192 (== 16 mod 128,
# index 2s+h) so that a warp's 32 halfword stores (c&3 -> +8 banks, head -> +4, t' -> +1) hit 32 distinct banks.
VT_SBO = 272
VT_REGION = 2192


def vt_base(h, s):
    return (2 * s + h) * VT_REGION


def thread_pair(tid):
    """loader thread -> (k0, chunk c, q, b, t')"""
    q, b, tp, c = tid >> 6, (tid >> 5) & 1, (tid >> 3) & 3, tid & 7
    return 16 * q + 8 * b + 2 * tp, c, q, b, tp


def key_of(s, k):
    """PV logical (step s, k in 0..31) -> tile key"""
    q = 2 * s + (k >= 16)
    t, i = (k % 16) // 4, k % 4
    return 16 * q + 8 * (i >> 1) + 2 * t + (i & 1)


def layout_tests(seed=11):
    g = torch.Generator().manual_seed(seed)
    Kb = torch.randint(0, 256, (64, 2, 64), generator=g, dtype=torch.uint8)    # [key][head][dim] e4m3 bytes
    Vb = torch.randint(0, 256, (64, 2, 64), generator=g, dtype=torch.uint8)
    Qb = torch.randint(0, 256, (2, 64, 64), generator=g, dtype=torch.uint8)    # [head][row][dim]
    for x in (Kb, Vb, Qb):
        x[(x & 0x7F) == 0x7F] = 0
    # --- cp.async: K tile in the wgmma layout; V into a per-thread raw slot
    ksm = torch.zeros(8192, dtype=torch.uint8)
    vraw = torch.zeros(8192, dtype=torch.uint8)
    cover = torch.zeros(64, 8, dtype=torch.int32)
    for tid in range(256):
        k0, c, q, b, tp = thread_pair(tid)
        for e in range(2):
            x = k0 + e
            cover[x, c] += 1
            h, cc = c >> 2, c & 3
            dst = h * 4096 + (x >> 3) * 512 + cc * 128 + (x & 7) * 16
            ksm[dst:dst + 16] = Kb[x].reshape(-1)[16 * c:16 * c + 16]
            vraw[e * 4096 + tid * 16: e * 4096 + tid * 16 + 16] = Vb[x].reshape(-1)[16 * c:16 * c + 16]
    assert bool((cover == 1).all()), "every (key, chunk) loaded exactly once"
    for h in range(2):
        for s in range(2):
            base = h * 4096 + 256 * s
            for n in range(64):            # B[k][n] = K[key n][dim 32s + k]
                for k in range(32):
                    assert ksm[base + il_addr(n, k, 128, 512)] == Kb[n, h, 32 * s + k], "K tile layout"
    # --- Q tile (A, rows m), same layout
    qsm = torch.zeros(8192, dtype=torch.uint8)
    for h in range(2):
        for m in range(64):
            for d in range(64):
                qsm[h * 4096 + il_addr(m, d, 128, 512)] = Qb[h, m, d]
    for h in range(2):
        for s in range(2):
            for m in range(64):
                for k in range(32):
                    assert qsm[h * 4096 + 256 * s + il_addr(m, k, 128, 512)] == Qb[h, m, 32 * s + k], "Q layout"
    # --- V^T via per-thread halfword transpose (logical k = 16(q&1) + 4t' + 2b, step q>>1)
    vt = torch.zeros(4 * VT_REGION, dtype=torch.uint8)
    sts = {}
    lds = {}
    for tid in range(256):
        k0, c, q, b, tp = thread_pair(tid)
        h, cc = c >> 2, c & 3
        lds.setdefault((tid // 8, 0), []).append(tid * 16)
        lds.setdefault((tid // 8, 1), []).append(4096 + tid * 16)
        s = q >> 1
        kl = 16 * (q & 1) + 4 * tp + 2 * b
        for d in range(16):
            n = 16 * cc + d
            a = vt_base(h, s) + il_addr(n, kl, 128, VT_SBO)
            vt[a] = vraw[tid * 16 + d]
            vt[a + 1] = vraw[4096 + tid * 16 + d]
            sts.setdefault((tid // 32, d), []).append(a)
    for (w, d), addrs in sts.items():     # a warp's 32 STS.U16 of one d: count distinct 4-B banks per word
        banks = {}
        for a in addrs:
            banks.setdefault((a // 4) % 32, set()).add(a // 4)
        assert max(len(v) for v in banks.values()) == 1, f"V^T STS.U16 bank conflict (warp {w} d {d})"
    nconf = 0
    assert all(V3.conflict_free(a, 16) for a in lds.values()), "own-V LDS.128 bank conflict"
    for h in range(2):
        for s in range(2):
            for n in range(64):
                for k in range(32):
                    assert vt[vt_base(h, s) + il_addr(n, k, 128, VT_SBO)] == Vb[key_of(s, k), h, n], "V^T layout"
    # --- P packing: accumulator -> A registers (m16n8k32 8-bit A layout per warp of the warpgroup)
    P = torch.rand((64, 64), generator=g) * 256
    P8 = P.to(FP8)
    Vf = E.dec(Vb[:, 0, :])                      # [key][dim], head 0
    Pperm = torch.zeros((2, 64, 32))
    for w in range(4):
        for lane in range(32):
            gq, t = lane >> 2, lane & 3
            acc = {}
            for j in range(8):
                for e in range(2):
                    acc[(0, j, e)] = P8[16 * w + gq, 8 * j + 2 * t + e]
                    acc[(1, j, e)] = P8[16 * w + gq + 8, 8 * j + 2 * t + e]
            for s in range(2):
                regs = []
                for rr, qq in ((0, 2 * s), (1, 2 * s), (0, 2 * s + 1), (1, 2 * s + 1)):   # a0, a1, a2, a3
                    regs.append([acc[(rr, 2 * qq + (i >> 1), i & 1)] for i in range(4)])
                for ri, (row_off, kbase) in enumerate(((0, 0), (8, 0), (0, 16), (8, 16))):
                    for i in range(4):
                        Pperm[s, 16 * w + gq + row_off, kbase + 4 * t + i] = regs[ri][i].float()
    for s in range(2):
        for m in range(64):
            for k in range(32):
                assert Pperm[s, m, k] == P8[m, key_of(s, k)].float(), "P register packing"
    Bv = torch.stack([torch.stack([Vf[key_of(s, k)] for k in range(32)]) for s in range(2)])  # [s][k][n]
    got = Pperm[0] @ Bv[0] + Pperm[1] @ Bv[1]
    want = P8.float() @ Vf
    rel = float((got - want).abs().max() / want.abs().max())
    assert rel < 1e-5, f"P_perm @ V^T_perm != P @ V (rel {rel:.2e})"
    return nconf


def main():
    t0 = time.time()
    nconf = layout_tests()
    print(f"[wg layout] K/Q tiles (INTERLEAVE, LBO 128 / SBO 512), V^T (LBO 128 / SBO 272, region 2192, permuted keys), P register"
          f" packing and P_perm.V^T == P.V: OK; V^T STS.U16 + own-V LDS.128 phases conflict-free")
    E.warp_roles = V3.warp_roles_v3
    E.L8_Q_FP16 = False
    E.L8_P_E4M3 = True
    E.TIGHT_TOL = 4.0e-2      # P in E4M3 adds ~1-2% vs the E4M3-query reference; the judge bound is the real check
    if "--quick" in sys.argv:
        E.CASES = [c for c in E.CASES if c[1].get("q_len", 1) == 8]
    E.main()
    print(f"[wg] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""K1 v3 CPU emulation: the layout changes of k1_v3.py on top of k1_emulate.py (which still covers L=1, the plan,
the K ring, the fp8 QK fragments and both merge paths, and still passes unchanged).

  python3 -I k1_v3_emulate.py

v3 changes checked here:
  1. L=8 V path: each loader thread converts its own two 16-B e4m3 V chunks (ring swizzle ch ^ (x&1)) into a CTA fp16
     V tile [key][head][8 chunks] with physical chunk c16 ^ (x&7) ^ (2*head); PV B fragments come from
     ldmatrix.x4.trans with lane address (key 16kk + (lane&7) + 8*((lane>>3)&1), chunk 2jp + (lane>>4)).
     The test rebuilds every lane's b0..b3 from the byte image with PTX .trans semantics and compares them with
     V[key][dim] directly; it also asserts that every ldmatrix phase (8 rows of 16 B) and every conv_v STS.128
     phase (8 lanes) and LDS.128 phase is bank-conflict free.
  2. L=8 warp roles: warp = (KV head w>>2, m-tile w&3 = rows 16mt..16mt+15), all 64 keys of each tile, every warp
     attends the new-token tile, no cross-warp merge. The end-to-end cases of k1_emulate (judge rule, exact key
     coverage per row, fold vs separate combine) are re-run with these roles.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops the script dir
import k1_emulate as E  # noqa: E402

ROWB = 128


def vh_off(x, hh, c16):
    return x * 256 + hh * 128 + 16 * (c16 ^ (x & 7) ^ (2 * hh))


def banks(addr, nbytes):
    """set of 4-B banks touched by an access."""
    return {((addr + i) // 4) % 32 for i in range(0, nbytes, 4)}


def conflict_free(addrs, nbytes):
    seen = set()
    for a in addrs:
        b = banks(a, nbytes)
        if seen & b:
            return False
        seen |= b
    return True


def v3_fragment_test(seed=0):
    g = torch.Generator().manual_seed(seed)
    V = torch.randint(0, 256, (64, 2, 64), generator=g, dtype=torch.uint8)   # e4m3 bytes [key][head][dim]
    V[(V & 0x7F) == 0x7F] = 0      # no NaN encodings (satfinite cache)
    # ring image (L=8 swizzle) as written by cp.async
    ring = torch.zeros(64 * ROWB, dtype=torch.uint8)
    for x in range(64):
        row = V[x].reshape(-1)                        # 128 B: head 0 dims 0..63, head 1 dims 0..63
        for ch in range(8):
            ring[x * ROWB + 16 * (ch ^ (x & 1)): x * ROWB + 16 * (ch ^ (x & 1)) + 16] = row[16 * ch: 16 * ch + 16]
    # conv_v: thread (x = tid>>2, part = tid&3) converts chunks 2part+u
    vh = torch.zeros(64 * 256, dtype=torch.uint8)
    f16 = E.dec(V).to(torch.float16)                 # exact
    sts_phases = {}
    lds_phases = {}
    for tid in range(256):
        x, part = tid >> 2, tid & 3
        for u in range(2):
            ch = 2 * part + u
            src = ring[x * ROWB + 16 * (ch ^ (x & 1)): x * ROWB + 16 * (ch ^ (x & 1)) + 16]
            lds_phases.setdefault((tid // 8, u), []).append(x * ROWB + 16 * (ch ^ (x & 1)))
            vals = E.dec(src).to(torch.float16).view(torch.uint8)  # 16 halves = 32 B
            hh, c0 = ch >> 2, 2 * (ch & 3)
            for v in range(2):
                off = vh_off(x, hh, c0 + v)
                vh[off: off + 16] = vals[16 * v: 16 * v + 16]
                sts_phases.setdefault((tid // 8, u, v), []).append(off)
    assert all(conflict_free(a, 16) for a in sts_phases.values()), "conv_v STS.128 bank conflict"
    assert all(conflict_free(a, 16) for a in lds_phases.values()), "conv_v LDS.128 bank conflict"
    # the image holds V as fp16 at the logical positions
    for x in range(64):
        for hh in range(2):
            for c16 in range(8):
                off = vh_off(x, hh, c16)
                got = vh[off: off + 16].view(torch.float16)
                assert torch.equal(got, f16[x, hh, 8 * c16: 8 * c16 + 8]), "fp16 V image mismatch"
    # ldmatrix.x4.trans: matrix q's row i comes from lane 8q+i's address; .trans: lane (g = l>>2, t = l&3) gets
    # elements [row 2t][col g] and [row 2t+1][col g] of each matrix (low half = row 2t)
    vh16 = vh.view(torch.float16)
    n_ph = 0
    for hh in range(2):
        for kk in range(4):
            for jp in range(4):
                addr = []
                for lane in range(32):
                    lx = (lane & 7) + 8 * ((lane >> 3) & 1)
                    lc = lane >> 4
                    addr.append(vh_off(16 * kk + lx, hh, 2 * jp + lc))
                for q in range(4):
                    assert conflict_free(addr[8 * q: 8 * q + 8], 16), "ldmatrix phase bank conflict"
                    n_ph += 1
                for lane in range(32):
                    gq, t = lane >> 2, lane & 3
                    regs = []
                    for q in range(4):
                        r0 = addr[8 * q + 2 * t] // 2 + gq       # half index: row 2t, col g
                        r1 = addr[8 * q + 2 * t + 1] // 2 + gq
                        regs.append((vh16[r0], vh16[r1]))
                    # expected B fragments: n-tile 2jp (dims 16jp..+7) b0 = keys 16kk+2t,+1 ; b1 = keys +8
                    for nt, (b0, b1) in ((0, (regs[0], regs[1])), (1, (regs[2], regs[3]))):
                        dim = 8 * (2 * jp + nt) + gq
                        for kb, (lo, hi) in ((0, b0), (8, b1)):
                            k0 = 16 * kk + kb + 2 * t
                            assert lo == f16[k0, hh, dim] and hi == f16[k0 + 1, hh, dim], "ldmatrix fragment"
    return n_ph


def warp_roles_v3(L, _orig=E.warp_roles):
    if L == 1:
        return _orig(L)
    roles = []
    for w in range(E.NWARP):
        h, mt = w >> 2, w & 3
        roles.append(dict(h=h, ko=0, kn=64, rows=torch.arange(16 * mt, 16 * mt + 16), new=True, grp=(h, mt), slot=0))
    return roles


def main():
    t0 = time.time()
    n = v3_fragment_test()
    print(f"[v3 frag] fp16 V tile image, {n} ldmatrix.x4.trans phases conflict-free, every lane's PV B fragment == V")
    E.warp_roles = warp_roles_v3
    E.main()
    print(f"[v3] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

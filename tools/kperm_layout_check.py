"""R-kperm-0 CPU check: does the shipped K2A31/K2A33 stage layout (the VML weights) already give each lane ONE 16-K scale
block per k64 step (= per 4 k16 steps), i.e. the KPERM3 property? Runs the file's own prepack + stage repack (AST-extracted)
on random NVFP4 codes / positive scales and checks, for every (stage, tile, lane, k64 step i): the 4 code words the lane
loads (c[i].x/.y = row g, c[i].z/.w = row g+8) hold exactly K = 64 kb + 16 t + 0..15 of their row (one scale block), and
the scale byte the kernel's dec_half2 picks (sw[i>>1] byte 2 (i&1) + h) is that block's scale.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \n      python -I "<repo>/tools/kperm_layout_check.py" <layer.py> [neg]
(neg = negative control: expects FAIL). R-kperm-0 on k3_rider3_1: PASS (2 x 524,288 checks), neg FAIL 64,965."""
import ast, sys, torch
src = open(sys.argv[1], encoding="utf-8").read()
tree = ast.parse(src)
want = {"_k2a_reencode_scales", "_k2a_tile_row_perm", "_k2a31_pack_words", "_k2a31_prepack_expert", "_k2a33_stage_repack",
        "K2A_BITPOS", "K2A31_BITPOS3"}
ns = {"torch": torch}
for n in tree.body:
    nm = n.name if isinstance(n, ast.FunctionDef) else (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None)
    if nm in want:
        exec(compile(ast.Module([n], []), nm, "exec"), ns)
        want.discard(nm)
assert not want, want
BP = ns["K2A31_BITPOS3"]
NEG = len(sys.argv) > 2  # negative control: claim the scale of the neighbour lane's block
bad = 0
for gate_up in (False, True):
    g = torch.Generator().manual_seed(7 + gate_up)
    n, k = 128, 4096
    codes = torch.randint(0, 16, (n, k), generator=g)
    w = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    ex = torch.randint(1, 15, (n, k // 16), generator=g); m = torch.randint(0, 8, (n, k // 16), generator=g)
    sf = ((ex << 3) | m).to(torch.uint8)
    wq, sq = ns["_k2a31_prepack_expert"](w, sf, gate_up, bitpos=BP)
    ws = ns["_k2a33_stage_repack"](wq, sq)          # [nrb=1, 16 stages, 18432]
    assert ws.shape == (1, 16, 18432), ws.shape
    bp_all, _, _, _ = ns["_k2a_reencode_scales"](sf)
    perm = ns["_k2a_tile_row_perm"](n, gate_up)
    W = ws[0].to(torch.int64)
    nchk = 0
    for s in range(16):
        for tl in range(8):
            for L in range(32):
                gg, t = L // 4, L % 4
                for i in range(4):
                    kb = 4 * s + i
                    sb = 16384 + tl * 256 + (i >> 1) * 128 + L * 4
                    for h in range(2):
                        r = int(perm[16 * tl + 8 * h + gg])
                        blk = 4 * kb + t
                        if int(W[s, sb + 2 * (i & 1) + h]) != int(bp_all[r, 4 * kb + ((t + 1) % 4 if NEG else t)]):
                            bad += 1
                        for a in range(2):
                            off = tl * 2048 + i * 512 + L * 16 + 4 * (2 * h + a)
                            word = sum(int(W[s, off + q]) << (8 * q) for q in range(4))
                            for b8, (p0, p1, p2, ps) in enumerate(BP):
                                nib = ((word >> p0) & 1) | (((word >> p1) & 1) << 1) | (((word >> p2) & 1) << 2) | (((word >> ps) & 1) << 3)
                                kk = 64 * kb + 16 * t + 8 * a + b8
                                assert kk // 16 == blk
                                if nib != int(codes[r, kk]):
                                    bad += 1
                                nchk += 1
    print(f"gate_up={gate_up}: {nchk} nibbles + scale bytes checked, mismatches so far {bad}")
# negative control: the natural mma K order (lane t holds 2t, 2t+1, 2t+8, 2t+9 of every k16) would put 4 blocks in a k64 step
nat = {(16 * st + c) // 16 for st in range(4) for c in (0, 1, 8, 9)}
print(f"natural-order blocks touched by one lane in a k64 step: {len(nat)} (KPERM-free engine) vs 1 here")
print("RESULT", "PASS" if bad == 0 else f"FAIL {bad}")

// k2b_v2_maps.cuh: K2b v2 index maps (RS wgmma, swap-AB: K2A lane items = A fragments in registers, tokens = N).
// Single source of truth: the v2 kernel pastes this text; k2b_v2_emulate.py parses every `#define K2B2_*` and
// evaluates the body as Python (non-negative ints; only + * << >> & ^ | and parentheses).
//
// k order. A K2A lane item (lane = 4g + t) holds rows g, g+8 at physical k = 64kb + 16t + 0..15 (words x/y = row g,
// z/w = row g+8; deq4 -> 4 bf16x2 = pairs 0-1, 2-3, 4-5, 6-7 of the word's 8 k). K2A v2.3's item order (the order
// tools/box/wgmma_unit.py mode 1 validated on the H200) feeds wgmma k16 step q with register slot s (a0..a3):
//   a0 = (row g,   logical k 2t, 2t+1)   = physical pair 2q     of row g   -> word x/y, deq4 output (2q) & 3
//   a1 = (row g+8, logical k 2t, 2t+1)   = physical pair 2q     of row g+8 -> word z/w
//   a2 = (row g,   logical k 2t+8, 2t+9) = physical pair 2q + 1 of row g
//   a3 = (row g+8, logical k 2t+8, 2t+9) = physical pair 2q + 1 of row g+8
// so logical column l = 16q + kappa of a 64-k stage (the k16 slice wgmma step q reads from the B operand) holds
// physical k PERM64(l). The activation operand (B, smem) must therefore be stored in logical order: MODE 1 x via a
// permute pre-pass (x_perm), MODE 2 inter written permuted by the MODE 1 epilogue (free).

// logical column l (0..63) of a k64 stage -> physical k (0..63); IPERM64 is its inverse
#define K2B2_PERM64(l) (((((l) >> 1) & 3) * 16) + (((l) >> 4) * 4) + ((((l) >> 3) & 1) * 2) + ((l) & 1))
#define K2B2_IPERM64(p) (((((p) >> 2) & 3) * 16) + ((((p) >> 1) & 1) * 8) + (((p) >> 4) * 2) + ((p) & 1))
// A fragment of k16 step q, register slot s (0..3 = a0..a3): K2A word (0..3 = x y z w) and deq4 output index
#define K2B2_AWORD(q, s) ((((s) & 1) * 2) + (((((q) * 2) + ((s) >> 1))) >> 2))
#define K2B2_AOUT(q, s) ((((q) * 2) + ((s) >> 1)) & 3)
// scale of register slot s: 0 = row g (byte hi = 0 of the SQ word), 1 = row g + 8 (hi = 1); selector as v1 K2B_SSEL
#define K2B2_AHI(s) ((s) & 1)
// consumer WG c (0..MW*... see MWG), warp w (0..3), m64 block mb (0..MW-1) -> K2A 16-row tile inside the expert
// for weight m-tile mt (tile rows = 2 * 64 * MW per CTA)
#define K2B2_WTILE(mt, mw, c, mb, w) (((mt) * 8 * (mw)) + ((c) * 4 * (mw)) + ((mb) * 4) + (w))
// token block of up to 256 rows: valid prefix v -> wgmma N (multiple of 8; 8..64 step 8, then 96..256 step 32)
#define K2B2_NSEL(v) ((((v) + 7) >> 3) * 8 + ((((v) + 7) >> 3) > 8) * (((32 - ((((((v) + 7) >> 3) * 8) & 31))) & 31)))
// accumulator (thread warp w, lane, reg v) of m64nN: weight row in the m64 block, token in the block (CLayout_64xN)
#define K2B2_ACC_WROW(w, lane, v) (((w) * 16) + ((lane) >> 2) + ((((v) >> 1) & 1) * 8))
#define K2B2_ACC_TOK(v, lane) (((((v) >> 2)) * 8) + (((lane) & 3) * 2) + ((v) & 1))
// MODE 1 SwiGLU: K2A W13 tile = gate rows rho 0..7 then up rows 8..15, so row g (gate) and row g+8 (up) of the
// warp's tile are registers 4j+e and 4j+2+e of the same thread (token 8j + 2t + e)
#define K2B2_GATE_REG(j, e) (((j) * 4) + (e))
#define K2B2_UP_REG(j, e) (((j) * 4) + 2 + (e))
// physical intermediate index of the thread's gate/up pair (tile = K2B2_WTILE(...)), and its logical (permuted)
// column in inter (the k order MODE 2's A fragments expect)
#define K2B2_INTER_PHYS(tile, lane) (((tile) * 8) + ((lane) >> 2))
#define K2B2_INTER_COL(i) ((((i) >> 6) * 64) + K2B2_IPERM64((i) & 63))
// MODE 2 output feature (natural order) of weight row r of m64 block mb, WG c, m-tile mt
#define K2B2_OUT_COL(mt, mw, c, mb, r) (((mt) * 128 * (mw)) + ((c) * 64 * (mw)) + ((mb) * 64) + (r))
// epilogue staging tile [token][64 * MW columns per WG] (bf16): byte offset of (token n, column col)
#define K2B2_STAGE_OFF(n, col, ncols) ((((n) * (ncols)) + (col)) * 2)

// K2A reading Marlin's layout measured 1.8x slower, see kernels/muse_moe_w4a16/k2a_mv1.md section 5.)

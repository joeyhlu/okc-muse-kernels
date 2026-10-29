"""p11_patches.py: P11 = an 11-bit lossless pack of the BF16 dense weights on the GD routes at T <= 8
(notes/study/fresh_eyes_nonmoe.md s1; format: dense_independent.md s3.1).

Format, per 16 x 64 unit (the P12 tile), 1408 B = 11 b/w: the P12 byte plane (e0 | mant7, 1024 B), a 2-bit plane n = h7 - base
(256 B, lane-interleaved u32 pair) and a sign plane (128 B). base = the per-matrix 4-value h7 window (8 exponents) holding the
most weights (judge recipe: per-tile and per-matrix windows tie, 1.36 % escapes). Escapes (h7 outside the window, ~1.4 %) are
stored with n = 0 (decoded w_d = their own sign / low byte at h7 = base) plus one u32 entry (unit 4 b | row 4 b | k 6 b | the
true bf16 16 b) in a per-warp list [G + 1 offsets | entries]. The kernel adds fma(w_t, x, -w_d * x) per escape and token into a
per-warp shared buffer (red.shared.add, one round of 32 entries per unit step, x loads one step ahead) and folds it into the
warp's partials before the CTA reduction. Weights reconstruct bit-exactly (prepare checks a full torch decode of the pack against
the matrix); only the FP32 summation order moves. P11 needs MT 1 (T <= 8); escape rounds past KU - 1 run after the unit loop.

Default: P11 on qkv at T = 1 and T = 8 only (coordinator 2026-10-29: o / pooler / unpooler belong to other builders).
--pool: also the pooler and the unpooler slice at T = 1 / 8 (opt-in section for the strategist to compose later).

    python -I tools/p11_patches.py in.py out.py [--pool]
"""
import pathlib
import py_compile
import sys

CU_HELPERS = r'''
__device__ __forceinline__ void ld11(const uint32_t* u, uint4 (&r)[3], int lane) {   // P11 unit: 2 byte planes, n pair, signs
  r[0] = ldw(reinterpret_cast<const uint4*>(u) + lane);
  r[1] = ldw(reinterpret_cast<const uint4*>(u) + 32 + lane);
  uint32_t n0, n1, sv;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(n0), "=r"(n1)
               : "l"(reinterpret_cast<const uint2*>(u + 256) + lane));
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(sv) : "l"(u + 320 + lane));
  r[2] = make_uint4(n0, n1, sv, 0u);
}

__device__ __forceinline__ void decode11(const uint4 (&p)[3], uint32_t tb, uint32_t (&a)[16]) {
  const uint32_t L[8] = {p[0].x, p[0].y, p[0].z, p[0].w, p[1].x, p[1].y, p[1].z, p[1].w};
  const uint32_t N[2] = {p[2].x, p[2].y}, S = p[2].z;
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int hs = 0; hs < 2; ++hs) {
      const int kk = 2 * q + hs;
      const uint32_t h = (((N[kk >> 2] >> (2 * (kk & 3))) & 0x03030303u) + tb) | ((S << kk) & 0x80808080u);
      a[4 * q + 2 * hs] = __byte_perm(L[2 * q + hs], h, 0x5140);
      a[4 * q + 2 * hs + 1] = __byte_perm(L[2 * q + hs], h, 0x7362);
    }
}

__device__ __forceinline__ uint32_t ldx16(const uint16_t* p) {
  unsigned short r;
  asm volatile("ld.global.nc.u16 %0, [%1];\n" : "=h"(r) : "l"(p));
  return (uint32_t)r;
}

template <int TM>
__device__ __forceinline__ void p11_ldx(uint32_t e, uint32_t (&xr)[TM], const uint16_t* x, int K, int u0, int tok0, int M) {
  const int kg = (u0 + (int)(e >> 26)) * 64 + (int)((e >> 16) & 63u);
#pragma unroll
  for (int j = 0; j < TM; ++j) xr[j] = (e != ~0u && tok0 + j < M) ? ldx16(x + (size_t)(tok0 + j) * K + kg) : 0u;
}

template <int TM>
__device__ __forceinline__ void p11_fin(uint32_t e, const uint32_t (&xr)[TM], uint32_t tb, int mrem, float* em) {
  if (e == ~0u) return;
  const float wt = __uint_as_float(e << 16);
  const float wd = __uint_as_float(((e & 0x80FFu) | ((tb & 0x7Fu) << 8)) << 16);
  float* rp = em + ((e >> 22) & 15u) * 8;
#pragma unroll
  for (int j = 0; j < TM; ++j)
    if (j < mrem) {
      const float xf = __uint_as_float(xr[j] << 16);
      atomicAdd(rp + j, __fmaf_rn(wt, xf, -__fmul_rn(wd, xf)));
    }
}

'''

CU = [  # (old, new, count) inside GD_CUDA_SRC
    ("__device__ __forceinline__ uint32_t smem_u32(const void* p)",
     CU_HELPERS.lstrip("\n") + "__device__ __forceinline__ uint32_t smem_u32(const void* p)", 1),
    ("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0>\n__global__",
     "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0, int P11 = 0>\n"
     "__global__", 1),
    ("  __shared__ __align__(8) unsigned long long mbar;\n",
     "  __shared__ __align__(8) unsigned long long mbar;\n"
     "  static_assert(P11 == 0 || ((P11 == 1 || P11 == 8) && MT == 1 && CAT == 0 && VF == 0 && KU >= 2), \"P11: T <= 8 GD routes\");\n"
     "  __shared__ __align__(16) float esm[P11 ? NW : 1][P11 ? 128 : 4];   // P11: escape corrections [row16][tok8] per warp\n", 1),
    ("""  uint4 R[ST][NP];
#pragma unroll
  for (int i = 0; i < ST; ++i)
    if (i < KU) {
#pragma unroll
      for (int p = 0; p < NP; ++p) R[i][p] = ldw(wp + (size_t)i * wstep + p * 32);
    }
  const uint32_t em = lde(A.esc + (size_t)gw * 32 + lane);
""", """  const uint32_t* wq = reinterpret_cast<const uint32_t*>(A.w) + (size_t)gw * 352;   // P11 unit = 352 u32
  const size_t wqs = (size_t)G * 352;
  uint4 R[ST][NP];
#pragma unroll
  for (int i = 0; i < ST; ++i)
    if (i < KU) {
      if constexpr (P11 > 0) ld11(wq + (size_t)i * wqs, R[i], lane);
      else {
#pragma unroll
        for (int p = 0; p < NP; ++p) R[i][p] = ldw(wp + (size_t)i * wstep + p * 32);
      }
    }
  uint32_t em = 0u, e_cur = ~0u;
  int p11n = 0, p11o = 0;
  if constexpr (P11 > 0) {   // [G + 1 offsets | entries]; entries of warp gw sorted by unit
    const uint32_t o0 = lde(A.esc + gw), o1 = lde(A.esc + gw + 1);
    p11n = (int)(o1 - o0);
    p11o = G + 1 + (int)o0;
    e_cur = lane < p11n ? lde(A.esc + p11o + lane) : ~0u;
    reinterpret_cast<float4*>(&esm[warp][0])[lane] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
  } else {
    em = lde(A.esc + (size_t)gw * 32 + lane);
  }
""", 1),
    ("    decode(R[slot], (CAT && s >= KUP) ? A.tbr : A.tb, a);\n",
     "    if constexpr (P11 > 0) decode11(R[slot], A.tb, a);\n    else decode(R[slot], (CAT && s >= KUP) ? A.tbr : A.tb, a);\n", 1),
    ("""    if (s + ST < KU) {
#pragma unroll
      for (int p = 0; p < NP; ++p) R[slot][p] = ldw(wp + (size_t)(s + ST) * wstep + p * 32);
    }
    while ((int)(enx >> 20) == s) {""", """    if (s + ST < KU) {
      if constexpr (P11 > 0) ld11(wq + (size_t)(s + ST) * wqs, R[slot], lane);
      else {
#pragma unroll
        for (int p = 0; p < NP; ++p) R[slot][p] = ldw(wp + (size_t)(s + ST) * wstep + p * 32);
      }
    }
    while (P11 == 0 && (int)(enx >> 20) == s) {""", 1),
    ("""    if constexpr (CAT > 0) {
      if (s == KUP - 1) {""", """    if constexpr (P11 > 0) {   // escape round s (its entries were loaded one step ahead)
      if (32 * s < p11n) {
        uint32_t xr[P11];
        p11_ldx<P11>(e_cur, xr, A.x, A.K, u0, tok0, A.M);
        p11_fin<P11>(e_cur, xr, A.tb, A.M - tok0, &esm[warp][0]);
        e_cur = 32 * (s + 1) + lane < p11n ? lde(A.esc + p11o + 32 * (s + 1) + lane) : ~0u;
      }
    }
    if constexpr (CAT > 0) {
      if (s == KUP - 1) {""", 1),
    ("""  }

#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float v = acc[0][mt][i];
      if constexpr (NC == 2) v += acc[1][mt][i];
      if constexpr (CAT > 0) redr""", """  }

  if constexpr (P11 > 0) {   // the rounds past the unit loop (rare: > 32 (KU - 1) escapes in this warp)
    for (int r = KU; 32 * r < p11n; ++r) {
      uint32_t xr[P11];
      const uint32_t e = 32 * r + lane < p11n ? lde(A.esc + p11o + 32 * r + lane) : ~0u;
      p11_ldx<P11>(e, xr, A.x, A.K, u0, tok0, A.M);
      p11_fin<P11>(e, xr, A.tb, A.M - tok0, &esm[warp][0]);
    }
    __syncwarp();
  }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float v = acc[0][mt][i];
      if constexpr (NC == 2) v += acc[1][mt][i];
      if constexpr (P11 > 0) v += esm[warp][(g + 8 * (i >> 1)) * 8 + 2 * t + (i & 1)];
      if constexpr (CAT > 0) redr""", 1),
    ("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0>\nstatic int launch(",
     "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0, int P11 = 0>\nstatic int launch(", 1),
    ("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0>\nstatic int info(",
     "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0, int P11 = 0>\nstatic int info(", 1),
    ("gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>", "gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT, P11>", 2),
]

PY_FUNCS = r'''
P11_ON = True
P11_ROUTES = {("qkv", 1), ("qkv", 8)}   # @P11_ROUTES@


def _p11_key(proj, b):
    rt = GD_ROUTE.get((proj, b))
    if not (P11_ON and (proj, b) in P11_ROUTES and b in (1, 8) and rt and rt[0] == 1):
        return None
    key = _gd_key(*_gd_nk(proj, b), rt)
    return None if key is None else key + (0, b)


def _p11_base(wv):
    h7 = (wv.contiguous().view(torch.int16).to(torch.int32) >> 8) & 0x7F
    cs = torch.cat([torch.zeros(1, dtype=torch.float64, device=wv.device),
                    torch.bincount(h7.reshape(-1), minlength=128).double().cumsum(0)])
    return int((cs[4:129] - cs[0:125]).argmax())


def _p11_rc(n, k, dev):   # unit coordinates of _gd_frag's [n/16, k/64, 32, 4, 8] layout
    lane = torch.arange(32, device=dev).view(32, 1, 1)
    q = torch.arange(4, device=dev).view(1, 4, 1)
    j = torch.arange(8, device=dev).view(1, 1, 8)
    r16 = ((lane >> 2) + 8 * ((j >> 1) & 1)).expand(32, 4, 8)
    k64 = (16 * (lane & 3) + 4 * q + (j & 1) + 2 * (j >> 2)).expand(32, 4, 8)
    rows = torch.arange(n // 16, device=dev).view(-1, 1, 1, 1, 1) * 16 + r16
    cols = torch.arange(k // 64, device=dev).view(1, -1, 1, 1, 1) * 64 + k64
    return rows, cols, r16, k64


def _p11_pack(wv, wr, wk, ku, s, base):
    n, k = wv.shape
    nrt, nu = n // 16, k // 64
    nbx = nrt // wr
    G = nbx * s * wr * wk
    dev = wv.device
    fr = _gd_frag(wv)
    lo64 = (fr & 0xFF).to(torch.int64)
    d = ((fr >> 8) & 0x7F) - base
    isesc = (d < 0) | (d > 3)
    d = torch.where(isesc, torch.zeros_like(d), d).to(torch.int64)
    sg = ((fr >> 15) & 1).to(torch.int64)
    l0 = _gd_i32(lo64[..., 0] | (lo64[..., 1] << 8) | (lo64[..., 2] << 16) | (lo64[..., 3] << 24))
    l1 = _gd_i32(lo64[..., 4] | (lo64[..., 5] << 8) | (lo64[..., 6] << 16) | (lo64[..., 7] << 24))
    p0 = torch.stack([l0[..., 0], l1[..., 0], l0[..., 1], l1[..., 1]], dim=-1).reshape(nrt, nu, 128)
    p1 = torch.stack([l0[..., 2], l1[..., 2], l0[..., 3], l1[..., 3]], dim=-1).reshape(nrt, nu, 128)
    kk = torch.arange(8, device=dev).view(8, 1)
    jj = torch.arange(4, device=dev).view(1, 4)
    nb = d.reshape(nrt, nu, 32, 8, 4) << (8 * jj + 2 * (kk & 3))
    nw = torch.stack([_gd_i32(nb[..., :4, :].sum((-1, -2))), _gd_i32(nb[..., 4:, :].sum((-1, -2)))], dim=-1).reshape(nrt, nu, 64)
    sw = _gd_i32((sg.reshape(nrt, nu, 32, 8, 4) << (8 * jj + 7 - kk)).sum((-1, -2)))
    units = torch.cat([p0, p1, nw, sw], dim=-1)
    pos = isesc.nonzero()
    nesc = int(pos.shape[0])
    cnt = torch.zeros(G, dtype=torch.int64, device=dev)
    ents = torch.zeros(0, dtype=torch.int32, device=dev)
    if nesc:
        rt, u, ln, qq, jv = (pos[:, i] for i in range(5))
        _, _, r16, k64 = _p11_rc(n, k, dev)
        bx, wri = rt // wr, rt % wr
        by, wki, st_ = u // (wk * ku), (u // ku) % wk, u % ku
        gw = ((by * nbx + bx) * wr + wri) * wk + wki
        rr, kc = r16[ln, qq, jv], k64[ln, qq, jv]
        ent = (st_ << 26) | (rr << 22) | (kc << 16) | fr[rt, u, ln, qq, jv].to(torch.int64)
        order = torch.argsort(((gw * 16 + st_) * 16 + rr) * 64 + kc)
        ents = _gd_i32(ent[order])
        cnt = torch.bincount(gw, minlength=G)
    off = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), cnt.cumsum(0)]).to(torch.int32)
    v = units.view(nbx, wr, s, wk, ku, 352).permute(4, 2, 0, 1, 3, 5).contiguous().view(-1)
    return {"w": v, "esc": torch.cat([off, ents]), "tb": base * 0x01010101, "G": G, "nbx": nbx, "nesc": nesc}


def _p11_unpack(pk, n, k, wr, wk, ku, s):   # torch decode of the pack (the kernel's bit layout) -> [n, k] int16 bits
    nrt, nu = n // 16, k // 64
    nbx, G, base, dev = pk["nbx"], pk["G"], pk["tb"] & 0x7F, pk["w"].device
    un = pk["w"].view(ku, s, nbx, wr, wk, 352).permute(2, 3, 1, 4, 0, 5).reshape(nrt, nu, 352).to(torch.int64) & 0xFFFFFFFF
    L = torch.cat([un[..., :128].reshape(nrt, nu, 32, 4), un[..., 128:256].reshape(nrt, nu, 32, 4)], dim=-1)
    N = un[..., 256:320].reshape(nrt, nu, 32, 2)
    S = un[..., 320:].reshape(nrt, nu, 32)
    out = torch.zeros(nrt, nu, 32, 4, 8, dtype=torch.int64, device=dev)
    for kk in range(8):
        q, hs = kk // 2, kk % 2
        h = ((((N[..., kk >> 2] >> (2 * (kk & 3))) & 0x03030303) + pk["tb"]) | ((S << kk) & 0x80808080)) & 0xFFFFFFFF
        lm = L[..., 2 * q + hs]
        for j4 in range(4):
            out[..., q, 4 * hs + j4] = (((h >> (8 * j4)) & 0xFF) << 8) | ((lm >> (8 * j4)) & 0xFF)
    rows, cols, _, _ = _p11_rc(n, k, dev)
    w = torch.zeros(n * k, dtype=torch.int64, device=dev)
    w[(rows * k + cols).reshape(-1)] = out.reshape(-1)
    esc = pk["esc"].to(torch.int64) & 0xFFFFFFFF
    off, e = esc[:G + 1], esc[G + 1:]
    if e.numel():
        gw = torch.repeat_interleave(torch.arange(G, device=dev), off[1:] - off[:-1])
        wki, wri, bx, by = gw % wk, (gw // wk) % wr, (gw // (wk * wr)) % nbx, gw // (wk * wr * nbx)
        row = (bx * wr + wri) * 16 + ((e >> 22) & 15)
        col = ((by * wk + wki) * ku + (e >> 26)) * 64 + ((e >> 16) & 63)
        w[row * k + col] = e & 0xFFFF
    return torch.where(w >= 32768, w - 65536, w).to(torch.int16).view(n, k)


'''

PY = [  # (old, new, count) inside the GD python section (def _gd_keys .. def _gd_live)
    ("    for key in _gd_cat_keys():\n",
     "    for pb in sorted(P11_ROUTES):\n"
     "        key = _p11_key(*pb)\n"
     "        if key is not None and key not in keys:\n"
     "            keys.append(key)\n"
     "    for key in _gd_cat_keys():\n", 1),
    ("def _gd_selftest(r, w, b):\n", PY_FUNCS.lstrip("\n") + "def _gd_selftest(r, w, b):\n", 1),
    ("        key = _gd_key(n, k, rt)\n",
     "        key = _gd_key(n, k, rt)\n"
     "        p11 = _p11_key(proj, b) is not None and key is not None and _p11_key(proj, b) in ids\n"
     "        if p11:\n"
     "            key = _p11_key(proj, b)\n", 1),
    ("            pk_key = (proj, n, rt[1], rt[2], rt[3], rt[5])\n",
     "            pk_key = (proj, n, rt[1], rt[2], rt[3], rt[5]) + ((\"p11\",) if p11 else ())\n", 1),
    ("""                    if (proj, n) not in bases:
                        bases[(proj, n)] = _gd_window_base(wv)
                    pk = _gd_pack(wv, rt[1], rt[2], rt[3], rt[5], bases[(proj, n)])
                    un = torch.empty_like(wv)
                    mod.gd_unpack(pk["w"], pk["esc"], un, k, pk["G"], pk["nbx"], rt[1], rt[2], rt[3], pk["tb"])
                    torch.cuda.synchronize()
                    exact = bool(torch.equal(un.view(torch.int16), wv.view(torch.int16)))
""", """                    if p11:
                        pk = _p11_pack(wv, rt[1], rt[2], rt[3], rt[5], _p11_base(wv))
                        un = _p11_unpack(pk, n, k, rt[1], rt[2], rt[3], rt[5])
                        exact = bool(torch.equal(un, wv.view(torch.int16)))
                    else:
                        if (proj, n) not in bases:
                            bases[(proj, n)] = _gd_window_base(wv)
                        pk = _gd_pack(wv, rt[1], rt[2], rt[3], rt[5], bases[(proj, n)])
                        un = torch.empty_like(wv)
                        mod.gd_unpack(pk["w"], pk["esc"], un, k, pk["G"], pk["nbx"], rt[1], rt[2], rt[3], pk["tb"])
                        torch.cuda.synchronize()
                        exact = bool(torch.equal(un.view(torch.int16), wv.view(torch.int16)))
""", 1),
    ("f\"base {bases[(proj, n)]}, {pk['nesc']} escapes", "f\"{'P11 ' if p11 else ''}base {pk['tb'] & 0x7F}, {pk['nesc']} escapes", 1),
    ("""            desc = f"m{rt[0]}({rt[1]},{rt[2]},{rt[3]},{rt[5]})s{rt[4]}x{rt[6]}r{info[0]}"\n""",
     """            desc = f"m{rt[0]}({rt[1]},{rt[2]},{rt[3]},{rt[5]})s{rt[4]}x{rt[6]}r{info[0]}" + ("p11" if p11 else "")\n""", 1),
]

POOL_ROUTES = '{("qkv", 1), ("qkv", 8), ("pooler", 1), ("pooler", 8), ("unpooler", 1), ("unpooler", 8)}'


def sub(text, old, new, count, tag):
    c = text.count(old)
    if c != count:
        raise SystemExit(f"[p11] {tag}: expected {count} occurrence(s), found {c}: {old[:70]!r}")
    return text.replace(old, new)


def region(s, start, end):
    a = s.index(start)
    b = s.index(end, a + len(start))
    return a, b


def apply(s, pool=False):
    if "P11_ROUTES" in s:
        raise SystemExit("[p11] already applied")
    a, b = region(s, "GD_CUDA_SRC = r'''", "'''")
    cu = s[a:b]
    for i, (o, n, c) in enumerate(CU):
        cu = sub(cu, o, n, c, f"cu{i}")
    s = s[:a] + cu + s[b:]
    a, b = region(s, "def _gd_keys():", "def _gd_live(")
    py = s[a:b]
    for i, (o, n, c) in enumerate(PY):
        py = sub(py, o, n, c, f"py{i}")
    if pool:   # opt-in: pooler + unpooler slice at T = 1 / 8 (left to the strategist's FRONT / BACK composition)
        py = sub(py, '{("qkv", 1), ("qkv", 8)}   # @P11_ROUTES@', POOL_ROUTES + "   # @P11_ROUTES@ (--pool)", 1, "pool")
    return s[:a] + py + s[b:]


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    src, dst = pathlib.Path(args[0]), pathlib.Path(args[1])
    s = apply(src.read_text(encoding="utf-8"), pool="--pool" in sys.argv)
    nb = len(s.encode())
    if nb >= 1_000_000:
        raise SystemExit(f"[p11] {nb} B >= 1 MB")
    if any(len(line) > 400 for line in s.splitlines()):
        print("[p11] warning: a line > 400 chars")
    dst.write_text(s, encoding="utf-8", newline="\n")
    py_compile.compile(str(dst), doraise=True)
    print(f"[p11] wrote {dst} ({nb} B, headroom {1_000_000 - nb} B)")


if __name__ == "__main__":
    main()

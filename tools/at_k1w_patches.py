"""K1-W: K1 v7 (L=1) loader with the whole-row map, for the layer generator (notes/study/attn_full_fresh.md).

Each warp instruction copies 4 complete 128-B K (or V) rows (8 lanes x 16 B) instead of 64-B halves of 8 rows; the warp
pair of key quarter q4 still loads keys 16 q4 .. 16 q4 + 15, so the v24 warp-pair barrier stays valid. Same smem bytes,
same math: bitwise equal to K1 on 4 cases (judge prepare, job 565404af). Judge profile (cold L2): decode16 104.7 -> 102.7,
decode32 193.9 -> 189.1 us. Anchors match kernels/muse_decode_layer/k3_gd_1.py exactly once each.

Usage: text = apply_k1w(text)   (text = a generated layer / attention file containing the K1 v7 CUDA source)
"""

K1W_PATCHES = (
    ('  const int lk = tid >> 2, part = tid & 3;\n',
     '  const int lk = tid >> 2, part = tid & 3;\n  const int wk0 = 16 * (tid >> 6) + ((tid >> 3) & 7), wch = tid & 7;   // at_1 K1-W: whole-row map\n'),
    ('      const int absk = kstart + (t0 + i) * TILE + lk;\n      if (i < ntl && absk < seq) pe.x = __ldg(pt + (absk >> pshift));\n    } else {\n',
     '      const int absk = kstart + (t0 + i) * TILE + wk0;\n      if (i < ntl && absk < seq) pe.x = __ldg(pt + (absk >> pshift));\n      if (i < ntl && absk + 8 < seq) pe.y = __ldg(pt + ((absk + 8) >> pshift));\n    } else {\n'),
    ('      const int absk = kstart + (t0 + i) * TILE + lk;\n      const int pe = pe2.x;\n      const bool valid = pe >= 0;\n      const size_t goff = valid ? (size_t)((pe << pshift) | (absk & pmask)) * rowB : (size_t)0;\n      const int nb = valid ? 16 : 0;\n#pragma unroll\n      for (int u = 0; u < 2; ++u) {\n        const int ch = part + 4 * u;   // v24: sector-complete 16-B requests\n        cp_async16(stg + lk * ROWB + 16 * kswz<L>(lk, ch), kc + goff + 16 * ch, nb);\n        cp_async16(stg + TILE * ROWB + lk * ROWB + 16 * vswz<L>(lk, ch), vc + goff + 16 * ch, nb);\n      }\n',
     '#pragma unroll\n      for (int u = 0; u < 2; ++u) {\n        const int x = wk0 + 8 * u;\n        const int absk = kstart + (t0 + i) * TILE + x;\n        const int pe = u ? pe2.y : pe2.x;\n        const bool valid = pe >= 0;\n        const size_t goff = valid ? (size_t)((pe << pshift) | (absk & pmask)) * rowB : (size_t)0;\n        const int nb = valid ? 16 : 0;\n        cp_async16(stg + x * ROWB + 16 * kswz<L>(x, wch), kc + goff + 16 * wch, nb);\n        cp_async16(stg + TILE * ROWB + x * ROWB + 16 * vswz<L>(x, wch), vc + goff + 16 * wch, nb);\n      }\n'),
)


def apply_k1w(text):
    for old, new in K1W_PATCHES:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"K1-W anchor found {n} times: {old[:60]!r}")
        text = text.replace(old, new, 1)
    return text

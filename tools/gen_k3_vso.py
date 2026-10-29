"""R-vso: VML stage-overlap levers at small N (gc_16 base): weight-ring depth NS, byte-ring x, register split.

  py tools/gen_k3_vso.py <n> [bvar128 bvar256 [var32]]   -> kernels/lab/k3_vso_<n>.py (never overwrites)

Compile-time defines (all default off = the gc_16 code):
  VM_NS   weight-ring depth (default 3).
  VM_XB   x ring as ONE byte ring of VM_XB bytes with VM_NX barrier entries. Entry it holds npad * 512 B at the next free
          offset (wrap to 0 when it does not fit); the x producer lane waits for the free of the oldest in-flight entry while
          it overlaps (frees are in order, so the oldest is the first one a new entry can hit) or while NX entries are in
          flight. Offsets go through a per-entry smem table xo[] (producer / self-fill tid 0 write before the full-barrier
          arrive, consumers read after the full wait). Route / gather scratch (12.8 KB) sits at the ring end; self-fill
          only uses entries that end below it. Needs VM_XB >= 2 x the largest entry (stage<N> frees x one stage late).
Variants (_VML_DEFS): w72n4x2 (NS 4, NX 2 fixed 36 KB slots), w72n4b (NS 4 + 78 KB byte ring, 4 entries), w72n3b (NS 3 +
114 KB byte ring, 6 entries), w72p64 (setmaxnreg producer 64 / consumer 104), w72n4bp (n4b + p64), bf2n4b (bf2 + NS 4 +
78 KB byte ring). The DSC agent's VM_DSC flags are untouched (separate defines).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels/muse_decode_layer/k3_gc_16.py"


def sub(text, old, new, count=1):
    assert text.count(old) == count, (text.count(old), old[:120])
    return text.replace(old, new)


def write_new(p, text):
    assert not p.exists(), f"{p} exists"
    assert len(text.encode()) < 1_000_000, len(text.encode())
    p.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {p} ({len(text.encode())} B)")


CU = [
    ("constexpr int NS = 3, NX = VM_NX;\n",
     "#ifndef VM_NS\n#define VM_NS 3\n#endif\n#ifndef VM_XB\n#define VM_XB 0   // R-vso: x byte ring bytes (0 = NX fixed XS slots)\n#endif\n"
     "constexpr int NS = VM_NS, NX = VM_NX;\n"),
    ("constexpr int BAR_OFF = NS * US + NX * XS;\nconstexpr int TAB_OFF = BAR_OFF + 8 * 2 * (NS + NX);\n",
     "constexpr int XR = VM_XB ? VM_XB : NX * XS;   // x region bytes\n"
     "constexpr int SCRB = 12800;                   // route hist 4 KB + ent 8 KB + fin 512 B\n"
     "constexpr int SCR_OFF = VM_XB ? NS * US + XR - SCRB : NS * US + (NX - 1) * XS;\n"
     "constexpr int BAR_OFF = NS * US + XR;\n"
     "constexpr int XO_OFF = BAR_OFF + 8 * 2 * (NS + NX);   // VM_XB: xo / xps / xpe [NX] (u32)\n"
     "constexpr int TAB_OFF = XO_OFF + (VM_XB ? ((12 * NX + 15) & ~15) : 0);\n"
     "static_assert(!VM_XB || (VM_XB >= 2 * XS && VM_XB % 16 == 0 && NX >= 2), \"vso: byte ring needs 2 largest entries\");\n"
     "static_assert(!VM_XB || !VM_DP, \"vso: byte ring not wired into the DP path\");\n"
     "#if VM_XB\n#define VSO_XOFF(xs) (reinterpret_cast<const uint32_t*>(smem + XO_OFF)[xs])\n"
     "#define VSO_XOFF4(xs) (VSO_XOFF(xs) >> 4)\n"
     "#else\n#define VSO_XOFF(xs) ((xs) * XS)\n#define VSO_XOFF4(xs) ((uint32_t)(xs) * (uint32_t)(XS >> 4))\n#endif\n"),
    ("constexpr int SMEM = TAB_OFF + (int)sizeof(Tab);\n",
     "constexpr int SMEM = TAB_OFF + (int)sizeof(Tab);\nstatic_assert(SMEM <= 232448, \"vso: dynamic smem\");\n"),
    # self-fill entry count: byte ring = the first entries that end below the scratch
    ("__device__ __forceinline__ int nfill_of(const Args& A, const Tab& T) {\n"
     "return (A.flags & 64) ? min(NX - 1, T.bA[blockIdx.x + 1] - T.bA[blockIdx.x]) : 0;\n}\n",
     "__device__ __forceinline__ int nfill_of(const Args& A, const Tab& T) {\n"
     "#if VM_XB\n"
     "if (!(A.flags & 64)) return 0;\n"
     "const int c = blockIdx.x, g0 = T.bA[c], g1 = T.bA[c + 1], sp = split_point(A, 0, g0, g1), m = min(NX - 1, g1 - g0);\n"
     "int nf = 0;\nuint32_t e = 0;\n"
     "for (; nf < m; ++nf) {\n"
     "const int gs = (sp + nf < g1) ? sp + nf : sp + nf - (g1 - g0);\n"
     "const uint32_t b = (uint32_t)((T.jC[gs >> 9] + 7) & ~7) * 512u;\n"
     "if (e + b > (uint32_t)(XR - SCRB)) break;\ne += b;\n}\n"
     "return nf;\n"
     "#else\n"
     "return (A.flags & 64) ? min(NX - 1, T.bA[blockIdx.x + 1] - T.bA[blockIdx.x]) : 0;\n"
     "#endif\n}\n"),
    ("const int g0 = T.bA[c], g1 = T.bA[c + 1], sp = split_point(A, 0, g0, g1);\nfor (int f = 0; f < nf; ++f) {\n"
     "const int gs = (sp + f < g1) ? sp + f : sp + f - (g1 - g0);\nconst int j = gs >> 9, st = gs & 15, cn = T.jC[j];\n"
     "uint32_t* xs = reinterpret_cast<uint32_t*>(smem + NS * US + f * XS);\n",
     "const int g0 = T.bA[c], g1 = T.bA[c + 1], sp = split_point(A, 0, g0, g1);\nuint32_t xoff = 0;\n"
     "for (int f = 0; f < nf; ++f) {\n"
     "const int gs = (sp + f < g1) ? sp + f : sp + f - (g1 - g0);\nconst int j = gs >> 9, st = gs & 15, cn = T.jC[j];\n"
     "#if VM_XB\n"
     "uint32_t* xs = reinterpret_cast<uint32_t*>(smem + NS * US + xoff);\n"
     "if (tid == 0) reinterpret_cast<uint32_t*>(smem + XO_OFF)[f] = xoff;   // consumers read it after the full wait\n"
     "xoff += (uint32_t)((cn + 7) & ~7) * 512u;\n"
     "#else\n"
     "uint32_t* xs = reinterpret_cast<uint32_t*>(smem + NS * US + f * XS);\n"
     "#endif\n"),
    # x producer: byte-ring state, self-filled entries, issue
    ("__device__ void prod_x(const Args& A, const Tab& T, uint32_t xring, uint32_t bars) {\n"
     "const int c = blockIdx.x, nf = nfill_of(A, T);\nint it = 0;\n",
     "__device__ void prod_x(const Args& A, const Tab& T, uint32_t xring, uint32_t bars) {\n"
     "const int c = blockIdx.x, nf = nfill_of(A, T);\nint it = 0;\n"
     "#if VM_XB\n"
     "uint32_t* xo = reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&T)) - TAB_OFF + XO_OFF);\n"
     "uint32_t* xps = xo + NX;\nuint32_t* xpe = xo + 2 * NX;\n"
     "uint32_t hd = 0;   // next free byte\nint tq = 0;         // oldest entry not yet seen freed\n"
     "#endif\n"),
    ("if (!pb && i < nf) continue;\n",
     "#if VM_XB\n"
     "if (!pb && i < nf) { xps[it % NX] = hd; hd += (uint32_t)npad * 512u; xpe[it % NX] = hd; continue; }\n"
     "#else\n"
     "if (!pb && i < nf) continue;\n"
     "#endif\n"),
    ("const int sl = it % NX;\nif (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);\n"
     "const uint32_t fb = bars + 8 * (2 * NS + sl);\nmbar_expect_tx(fb, bytes);\nbulk_g2s(xring + sl * XS, src, bytes, fb);\n}\n#if VM_DP\n",
     "const int sl = it % NX;\n"
     "#if VM_XB\n"
     "const uint32_t off = (hd + bytes <= (uint32_t)XR) ? hd : 0u;\n"
     "while (tq < it && (it - tq >= NX || (off < xpe[tq % NX] && xps[tq % NX] < off + bytes))) {\n"
     "mbar_wait(bars + 8 * (2 * NS + NX + tq % NX), (tq / NX) & 1, A.err);\n++tq;\n}\n"
     "xps[sl] = off; xpe[sl] = off + bytes; hd = off + bytes;\nxo[sl] = off;\n"
     "const uint32_t fb = bars + 8 * (2 * NS + sl);\nmbar_expect_tx(fb, bytes);\nbulk_g2s(xring + off, src, bytes, fb);\n"
     "#else\n"
     "if (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);\n"
     "const uint32_t fb = bars + 8 * (2 * NS + sl);\nmbar_expect_tx(fb, bytes);\nbulk_g2s(xring + sl * XS, src, bytes, fb);\n"
     "#endif\n}\n#if VM_DP\n"),
]


def gen_text():
    t = SRC.read_text(encoding="utf-8")
    for e in CU:
        t = sub(t, *e)
    # consumers: x tile offset from the table (byte ring) or xs * XS (slots)
    t = sub(t, "(uint32_t)xs * (uint32_t)(XS >> 4)", "VSO_XOFF4(xs)", 2)
    t = sub(t, "smem + NS * US + xs * XS", "smem + NS * US + VSO_XOFF(xs)", 2)
    t = sub(t, "smem + NS * US + (NX - 1) * XS", "smem + SCR_OFF", 2)
    # host: variants (BRED scratch scb like w72 / bf2)
    w = '"-DVM_PREG=32", "-DVM_CREG=112"'
    t = sub(t, '_VML = {"mod": None, "mods": {}, "sc": None, "scb": None,',
            '_VSO_N4B = ("-DVM_NS=4", "-DVM_NX=4", "-DVM_XB=79872")\n'
            '_VML_DEFS["w72n4x2"] = _VML_DEFS["w72"] + ("-DVM_NS=4", "-DVM_NX=2")\n'
            '_VML_DEFS["w72n4b"] = _VML_DEFS["w72"] + _VSO_N4B\n'
            '_VML_DEFS["w72n3b"] = _VML_DEFS["w72"] + ("-DVM_NX=6", "-DVM_XB=116736")\n'
            '_VML_DEFS["w72p64"] = tuple("-DVM_PREG=64" if f == "-DVM_PREG=32" else "-DVM_CREG=104" if f == "-DVM_CREG=112" else f for f in _VML_DEFS["w72"])\n'
            '_VML_DEFS["w72n4bp"] = _VML_DEFS["w72p64"] + _VSO_N4B\n'
            '_VML_DEFS["bf2n4b"] = _VML_DEFS["bf2"] + _VSO_N4B\n'
            'VML_VSO_VARS = ("w72n4x2", "w72n4b", "w72n3b", "w72p64", "w72n4bp", "bf2n4b")\n'
            '_VML = {"mod": None, "mods": {}, "sc": None, "scb": None,')
    t = sub(t, '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]\n',
            '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") + VML_VSO_VARS else _VML["sc"]\n')
    assert w in t
    return t


def gen(n, b128=None, b256=None, v32=None):
    t = gen_text()
    if b128:
        bv = {256: b256, 128: b128}
        t = sub(t, "VML_BVAR = {256: 'w72', 128: 'w72'}\n", f"VML_BVAR = {bv!r}\n")
    if v32:
        t = sub(t, "VML_VAR = 'bf2'\n", f"VML_VAR = {v32!r}\n")
    write_new(ROOT / f"kernels/lab/k3_vso_{n}.py", t)


if __name__ == "__main__":
    a = sys.argv[1:]
    gen(int(a[0]), *(a[1:] + [None] * 3)[:3])

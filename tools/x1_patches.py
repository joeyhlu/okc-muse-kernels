"""x1_patches.py: moe-x1 -- lossless 5-bit re-encoding of the VML (bf2 / w72) NVFP4 E4M3 block scales.

  python tools/x1_patches.py {x1|x2} [--base=kernels/muse_decode_layer/k3_gc_20.py] [--out=kernels/lab/k3_x1_{1|2}.py]

Layout (per stage row group = one 18432-B ws13/ws2 unit of 128 rows x 256 K): codes 16384 B unchanged, then per tile t
(8 tiles x 256 scales) 160 B: 32 nibble words (lane l: byte b = off(sw0 byte b) & 15 | (off(sw1 byte b) & 15) << 4) +
32 hi bytes (bit 3 - b: bit 4 of off(sw0 byte b), bit 7 - b: bit 4 of off(sw1 byte b)), then a u32 at +1280 = 0x75807580 +
base * 0x00100010 (the dec_half2 IMAD addend with the unit base folded in; VM_X1 1), a u32 at +1284 = base in
every byte (VM_X1 2: byte add, one live register fewer), 8 B pad: 17680 B (-4.08 % bytes / stage).
Decoded IMAD sums equal ship's exactly (byte = base + off, no half-word carry), so outputs are bitwise ship.
prepare(): every unit must span <= 31 codes and round-trip exactly, else the ship 8-bit variants (bf2 / w72) are built and used.
Ship 8-bit ws13 / ws2 stay (K2A fallback + self-test reference); the X1 copies are ws13x / ws2x (~0.87 GB).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_gc_20.py"


def _sub(s, old, new, n=1):
    c = s.count(old)
    assert c == n, (c, n, old[:90])
    return s.replace(old, new)


CU_DEFS = r"""constexpr int US = 2 * SW;
#ifndef VM_X1
#define VM_X1 0   // moe-x1: 5-bit block scales (tools/x1_patches.py)
#endif
constexpr int SWG = VM_X1 ? 17680 : SW;   // global bytes per stage row group (smem slot stride stays SW)
#if VM_X1 == 2
#define X1_SW \
const uint8_t* x1p = st + 16384 + tile * 160; \
const uint32_t x1n = reinterpret_cast<const uint32_t*>(x1p)[lane], x1h = x1p[128 + lane]; \
const uint32_t x1b = *reinterpret_cast<const uint32_t*>(st + 16384 + 1284); \
const uint32_t sw0 = ((x1n & 0x0F0F0F0Fu) | ((x1h * 0x10080402u) & 0x10101010u)) + x1b; \
const uint32_t sw1 = (((x1n >> 4) & 0x0F0F0F0Fu) | (((x1h >> 4) * 0x10080402u) & 0x10101010u)) + x1b; \
const uint32_t x1c = 0x75807580u; (void)sp;
#define X1_DEP
#elif VM_X1
#define X1_SW \
const uint8_t* x1p = st + 16384 + tile * 160; \
const uint32_t x1n = reinterpret_cast<const uint32_t*>(x1p)[lane], x1h = x1p[128 + lane]; \
const uint32_t x1c = *reinterpret_cast<const uint32_t*>(st + 16384 + 1280); \
const uint32_t sw0 = (x1n & 0x0F0F0F0Fu) | ((x1h * 0x10080402u) & 0x10101010u); \
const uint32_t sw1 = ((x1n >> 4) & 0x0F0F0F0Fu) | (((x1h >> 4) * 0x10080402u) & 0x10101010u); \
(void)sp;
#define X1_DEP ^ x1c
#else
#define X1_SW const uint32_t sw0 = sp[0], sw1 = sp[32]; const uint32_t x1c = 0x75807580u;
#define X1_DEP
#endif
"""

PY_X1 = r'''

_X1_SWG = 17680
_X1 = {"on": False, "why": None}


def _x1_encode_ws(ws):
    """ws [E, R, 16, 18432] uint8 (ship stage layout) -> [E, R, 16, 17680] X1 layout, or None if a unit spans > 31 codes."""
    E, R, S, _ = ws.shape
    out = torch.zeros((E, R, S, _X1_SWG), dtype=torch.uint8, device=ws.device)
    sh = torch.tensor([3, 2, 1, 0], dtype=torch.int32, device=ws.device)
    for e in range(E):
        w = ws[e]
        sc = w[..., 16384:].to(torch.int32)
        base = sc.amin(-1, keepdim=True)
        off = sc - base
        if int(off.amax()) > 31:
            return None
        o = off.view(R, S, 8, 2, 32, 4)
        nib = (o[:, :, :, 0] & 15) | ((o[:, :, :, 1] & 15) << 4)
        hb = (o >> 4) & 1
        h = (hb[:, :, :, 0] << sh).sum(-1) | (hb[:, :, :, 1] << (sh + 4)).sum(-1)
        blk = torch.cat((nib.reshape(R, S, 8, 128), h.reshape(R, S, 8, 32)), -1).reshape(R, S, 1280)
        cad = base.to(torch.int64) * 0x00100010 + 0x75807580
        cadb = torch.stack([(cad >> (8 * i)) & 255 for i in range(4)], -1).reshape(R, S, 4)
        out[e, ..., :16384] = w[..., :16384]
        out[e, ..., 16384:17664] = blk.to(torch.uint8)
        out[e, ..., 17664:17668] = cadb.to(torch.uint8)
        out[e, ..., 17668:17672] = base.to(torch.uint8).expand(R, S, 4)   # VM_X1 2: base in every byte (byte add)
        # round trip with the kernel's arithmetic (u32 words, little endian)
        x = out[e, ..., 16384:].to(torch.int64)
        xp = x[..., :1280].reshape(R, S, 8, 160)
        nw = xp[..., :128].reshape(R, S, 8, 32, 4)
        n = nw[..., 0] | (nw[..., 1] << 8) | (nw[..., 2] << 16) | (nw[..., 3] << 24)
        hh = xp[..., 128:]
        M = 0xFFFFFFFF
        s0 = (n & 0x0F0F0F0F) | (((hh * 0x10080402) & M) & 0x10101010)
        s1 = ((n >> 4) & 0x0F0F0F0F) | ((((hh >> 4) * 0x10080402) & M) & 0x10101010)
        c = x[..., 1280] | (x[..., 1281] << 8) | (x[..., 1282] << 16) | (x[..., 1283] << 24)
        bse = ((c - 0x75807580) >> 4) & 255
        ok = bool(((c - 0x75807580) == bse * 0x00100010).all()) and bool((x[..., 1284:1288] == bse.unsqueeze(-1)).all())
        dec = torch.stack([torch.stack([(s >> (8 * i)) & 255 for i in range(4)], -1) for s in (s0, s1)], 3)  # [R,S,8,2,32,4]
        dec = dec + bse.view(R, S, 1, 1, 1, 1)
        if not ok or not bool((dec.reshape(R, S, 2048) == sc.to(torch.int64)).all()):
            raise RuntimeError(f"X1 round trip mismatch (expert {e})")
    return out


def _x1_prepare(st):
    global VML_VAR, VML_BVAR
    t0 = time.time()
    try:
        a = _x1_encode_ws(st["ws13"])
        b = _x1_encode_ws(st["ws2"]) if a is not None else None
        why = None if b is not None else "a stage row group spans > 31 scale codes"
    except Exception as ex:
        a = b = None
        why = repr(ex)[:200]
    if why is None:
        st["ws13x"], st["ws2x"] = a, b
        _X1["on"] = True
        torch.cuda.synchronize()
        print(f"[x1] 5-bit block scales live ({a.numel() + b.numel()} B, -{1 - _X1_SWG / 18432:.2%}) in {time.time() - t0:.1f}s",
              flush=True)
        return True
    _X1["on"], _X1["why"] = False, why
    st.pop("ws13x", None)
    st.pop("ws2x", None)
    VML_VAR, VML_BVAR = "bf2", {256: "w72", 128: "w72"}
    _VML["mod"] = None
    print(f"[x1] OFF ({why}): ship 8-bit VML variants", flush=True)
    return False


def _x1_w(mod, k):
    if getattr(mod, "x1", False):
        return k["ws13x"], k["ws2x"]
    return k["ws13"], k["ws2"]
'''


def x1(s):
    # --- CUDA (VML source only; anchors asserted unique) ---
    s = _sub(s, "A.ws2 + ((size_t)(e * 32 + 2 * ub) * 16 + s) * SW\n", "A.ws2 + ((size_t)(e * 32 + 2 * ub) * 16 + s) * SWG\n", n=2)
    s = _sub(s, "A.ws13 + ((size_t)(e * 64 + 2 * ub) * 16 + s) * SW;\n", "A.ws13 + ((size_t)(e * 64 + 2 * ub) * 16 + s) * SWG;\n", n=2)
    s = _sub(s, "mbar_expect_tx(fb, US);\nbulk_g2s_ef(dst, src, SW, fb, pol);\nbulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);\n",
             "mbar_expect_tx(fb, 2 * SWG);\nbulk_g2s_ef(dst, src, SWG, fb, pol);\nbulk_g2s_ef(dst + SW, src + (size_t)16 * SWG, SWG, fb, pol);\n",
             n=2)
    s = _sub(s, "uint32_t* hh, uint32_t k16,\nuint32_t k64, uint32_t k4) {\n#if VM_TRICKS_DQ\ndeq4_dq(w0",
             "uint32_t* hh, uint32_t k16,\nuint32_t k64, uint32_t k4, uint32_t cad = 0x75807580u) {\n#if VM_TRICKS_DQ\ndeq4_dq(w0")
    s = _sub(s, "* k16 + 0x75807580u;", "* k16 + cad;")
    s = _sub(s, "const uint32_t sw0 = sp[0], sw1 = sp[32];", "X1_SW", n=3)
    s = _sub(s, ", k16, k64, k4);", ", k16, k64, k4, x1c);", n=3)
    s = _sub(s, "sw2, par, hh, k16q, 64u, 4u);", "sw2, par, hh, k16q, 64u, 4u, x1c);")
    s = _sub(s, "^ sw0 ^ sw1) & zom))", "^ sw0 ^ sw1 X1_DEP) & zom))")
    s = _sub(s, "constexpr int US = 2 * SW;\n", CU_DEFS)
    # --- Python ---
    s = _sub(s, "VML_VAR = 'bf2'\n", "VML_VAR = 'bf2x'\n")
    s = _sub(s, "VML_BVAR = {256: 'w72', 128: 'w72'}\n", "VML_BVAR = {256: 'w72x', 128: 'w72x'}\n")
    a = '_VML = {"mod": None, "mods": {}, "sc": None, "scb": None, "ok": {}, "err": None, "okv": {}, "force": None}\n'
    s = _sub(s, a, a + '_VML_DEFS["bf2x"] = _VML_DEFS["bf2"] + ("-DVM_X1=1",)\n_VML_DEFS["w72x"] = _VML_DEFS["w72"] + ("-DVM_X1=1",)\n'
             '_VML_DEFS["bf2y"] = _VML_DEFS["bf2"] + ("-DVM_X1=2",)\n_VML_DEFS["w72y"] = _VML_DEFS["w72"] + ("-DVM_X1=2",)\n'
             + PY_X1)
    s = _sub(s, '            _VML["mods"][v], _ = _vml_build(defines=_VML_DEFS[v])\n',
             '            _VML["mods"][v], _ = _vml_build(defines=_VML_DEFS[v])\n            _VML["mods"][v].x1 = any(d.startswith("-DVM_X1=") for d in _VML_DEFS[v])\n')
    s = _sub(s, '("b", "bd", "p", "bf", "bg", "bf2", "w72")', '("b", "bd", "p", "bf", "bg", "bf2", "w72", "bf2x", "w72x", "bf2y", "w72y")')
    s = _sub(s, 'mod.vml_run([x, ids, wts, k["ws13"], k["ws2"], k["gsc13"]', 'mod.vml_run([x, ids, wts, *_x1_w(mod, k), k["gsc13"]')
    s = _sub(s, '        _pt1_wait("vml")\n        _VML["force"] = VML_VAR\n',
             '        _pt1_wait("vml")\n        _x1_prepare(st)\n        _VML["force"] = VML_VAR\n')
    return s


def nonce(s, tag):
    lines = s.split("\n", 2)
    assert lines[1].startswith("# nonce "), lines[1][:60]
    lines[1] = f"# nonce {tag}"
    return "\n".join(lines)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "x1"
    base_p = next((ROOT / a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--base=")), BASE)
    out_p = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--out=")), f"kernels/lab/k3_x1_{1 if mode == 'x1' else 2}.py")
    s = base_p.read_text(encoding="utf-8")
    if mode in ("x1", "x2"):   # x1: VM_X1 1 (base folded into the IMAD addend); x2: VM_X1 2 (byte add)
        s = x1(s)
        if mode == "x2":
            s = _sub(s, "VML_VAR = 'bf2x'\n", "VML_VAR = 'bf2y'\n")
            s = _sub(s, "VML_BVAR = {256: 'w72x', 128: 'w72x'}\n", "VML_BVAR = {256: 'w72y', 128: 'w72y'}\n")
        if "--keep-nonce" not in sys.argv:
            s = nonce(s, f"k3-{mode}-1 ({base_p.stem} + X1 5-bit VML block scales, VM_X1 {mode[1]}; tools/x1_patches.py)")
    else:
        raise SystemExit(f"unknown mode {mode}")
    Path(ROOT / out_p).write_text(s, encoding="utf-8", newline="\n")
    print(f"wrote {out_p} ({mode} on {base_p.name}, {len(s)} B)")


if __name__ == "__main__":
    main()

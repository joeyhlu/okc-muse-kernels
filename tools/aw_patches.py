"""aw_patches.py (moe-aw, R-aw): VM_AW "A whole" planner-only variant of the VML static plan (VM_RT route()).

Every CTA's phase-A range starts and ends on an A-unit edge (16 stages = one output tile over full K): the cut is the
unit edge nearest the ideal cost cut, so no A unit is shared by two CTAs (no A split shares, no red.add -> counter ->
last-arriver read-back chain in phase A). The phase-B cut of CTA boundary c then takes the even A+B cost target:
xB_c = clamp(totB c / G + (xA_c - cA_c), 0, totB), prefix-maxed over c (warps 5-9, one named barrier 11) so B ranges
stay contiguous and non-negative; B stays stage-granular. Shipped VML_COST tables; consumer / producer / epilogue code
untouched (they already handle any [g0, g1) range). The B warps recompute the A cut of their own c (no A->B handoff).

  python tools/aw_patches.py arms [--base=kernels/muse_decode_layer/k3_gc_21.py]
      -> kernels/lab/k3_aw_1.py (base + bf2w / w72w variants; default route bf2w / w72w; ship bf2 / w72 still built
         when named in VML_VAR / VML_BVAR, so the file holds every arm)
  python tools/aw_patches.py rider --out=kernels/...py [--base=...]   (same edit, written to --out)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ae_patches as AE  # noqa: E402  (_sub, defs, nonce: anchored helpers)

BASE = ROOT / "kernels/muse_decode_layer/k3_gc_21.py"
_sub, defs, nonce = AE._sub, AE.defs, AE.nonce

BR_HEAD = "} else if (warp < 10) {   // stage_at(pb, c) for c = 0..G: warps 0-4 phase A, 5-9 phase B\n"
BR_END = "if (c <= G) { if (pb) T.bB[c] = res; else T.bA[c] = res; }\n}\n}\nif (tid == 0) T.bready = 0;\n"

AW = r"""#if VM_AW
} else if (warp < 10) {   // VM_AW: A cuts on A-unit edges (no A split shares); B cuts carry the even A+B cost target
bar_sync(10, 32 * 11);
const int G = gridDim.x, pb = (warp >= 5) ? 1 : 0, c = tid - 160 * pb;
const int nj = pl[64 + lane], base = pl[96 + lane], tq2 = pl[128 + lane], jst = pl[160 + lane];
const int njT = pl[192];
const uint32_t totA = (uint32_t)pl[193], totB = (uint32_t)pl[194];
const int lastne = 31 - __clz(__ballot_sync(0xffffffffu, nj > 0));   // the last expert with jobs (-1: none)
const uint32_t xa = mdiv(totA * (uint32_t)c, A.magG);
int jA, uA;
uint32_t cA;
{   // the A cut of boundary c (both warp halves: B needs its cost), snapped to the nearest unit edge
const uint32_t cs = (uint32_t)pl[lane];
int lo = 0;
#pragma unroll
for (int s = 16; s >= 1; s >>= 1) { const uint32_t v = __shfl_sync(0xffffffffu, cs, (lo + s) & 31); if (lo + s <= lastne && v <= xa) lo += s; }
const int njE = __shfl_sync(0xffffffffu, nj, lo), bE = __shfl_sync(0xffffffffu, base, lo), tE = __shfl_sync(0xffffffffu, tq2, lo);
const int jE0 = __shfl_sync(0xffffffffu, jst, lo);
uint32_t cp = __shfl_sync(0xffffffffu, cs, lo);
int k = 0, sz = min(bE, tE);
while (k + 1 < njE) {
const uint32_t w = 512u * scost[max(1, (sz + 7) >> 3) - 1];
if (cp + w > xa) break;
cp += w;
++k;
sz = min(bE, tE - k * bE);
}
const int nt = max(1, (sz + 7) >> 3);
const uint32_t sst = scost[nt - 1];
uint32_t u = (xa > cp) ? (mdiv(xa - cp + 8u * sst, smag[nt - 1]) >> 4) : 0u;
if (u > 32u) u = 32u;
jA = jE0 + k; uA = (int)u; cA = cp + 16u * u * sst;
}
int res;
if (pb == 0) {
res = (njT == 0) ? 0 : (c >= G) ? (njT << 9) : ((jA << 9) + 16 * uA);
} else {
const uint32_t xb0 = mdiv(totB * (uint32_t)c, A.magG);
int xb = (int)xb0 + (int)xa - (int)cA;
xb = min(max(xb, 0), (int)totB);
#pragma unroll
for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, xb, o); if (lane >= o) xb = max(xb, n); }
int* wm = hist + 728;   // free words between the plan handoff (528..722) and scost (768)
if (lane == 31) wm[warp - 5] = xb;
bar_sync(11, 160);
for (int w = 5; w < warp; ++w) xb = max(xb, wm[w - 5]);
const uint32_t x = (uint32_t)xb, cs = (uint32_t)pl[32 + lane];
int lo = 0;
#pragma unroll
for (int s = 16; s >= 1; s >>= 1) { const uint32_t v = __shfl_sync(0xffffffffu, cs, (lo + s) & 31); if (lo + s <= lastne && v <= x) lo += s; }
const int njE = __shfl_sync(0xffffffffu, nj, lo), bE = __shfl_sync(0xffffffffu, base, lo), tE = __shfl_sync(0xffffffffu, tq2, lo);
const int jE0 = __shfl_sync(0xffffffffu, jst, lo);
uint32_t cp = __shfl_sync(0xffffffffu, cs, lo);
int k = 0, sz = min(bE, tE);
while (k + 1 < njE) {
const uint32_t w = 256u * scost[VM_NTM + max(1, (sz + 7) >> 3) - 1];
if (cp + w > x) break;
cp += w;
++k;
sz = min(bE, tE - k * bE);
}
const int nt = max(1, (sz + 7) >> 3);
uint32_t st = (x > cp) ? mdiv(x - cp, smag[VM_NTM + nt - 1]) : 0u;
if (st > 255u) st = 255u;
res = (njT == 0) ? 0 : (c >= G) ? (njT << 8) : (((jE0 + k) << 8) + (int)st);
}
if (c <= G) { if (pb) T.bB[c] = res; else T.bA[c] = res; }
}
#else
"""


def aw(s):
    s = _sub(s, "#ifndef VM_RT\n#define VM_RT 0\n#endif\n",
             "#ifndef VM_RT\n#define VM_RT 0\n#endif\n#ifndef VM_AW\n#define VM_AW 0   // R-aw: A cuts on unit edges (tools/aw_patches.py)\n#endif\n"
             "#if VM_AW && !VM_RT\n#error \"VM_AW needs the VM_RT static plan\"\n#endif\n")
    s = _sub(s, BR_HEAD, AW + BR_HEAD)
    s = _sub(s, BR_END, BR_END.replace("}\n}\nif (tid", "}\n#endif   // VM_AW\n}\nif (tid", 1))
    s = defs(s, {"bf2w": ("bf2", ("-DVM_AW=1",)), "w72w": ("w72", ("-DVM_AW=1",))})
    s = _sub(s, 'scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]',
             'scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72", "bf2w", "w72w") else _VML["sc"]')
    return s


def route(s, var="bf2w", bvar="w72w"):
    return AE.route(s, var, bvar)


def main():
    mode = sys.argv[1]
    base_p = next((ROOT / a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--base=")), BASE)
    out_p = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--out=")), None)
    s0 = base_p.read_text(encoding="utf-8")
    if mode == "arms":
        AE.build("k3_aw_1.py", nonce(route(aw(s0)), f"k3-aw-1 ({base_p.name} + VM_AW unit-edge A cuts: bf2w / w72w; tools/aw_patches.py)"))
    elif mode == "rider":
        assert out_p, "--out=kernels/...py required"
        Path(ROOT / out_p).write_text(route(aw(s0)), encoding="utf-8", newline="\n")
        print(f"wrote {out_p} (AW rider on {base_p.name})")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()

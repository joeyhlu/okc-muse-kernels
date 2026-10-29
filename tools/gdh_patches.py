"""gdh_patches.py: GDH (dense GD handoff at T = 16 / 32) lab files on a layer file.

  python -I tools/gdh_patches.py --n=<n> [--base=k3_gc_14] [--clean=<arms>] [--dry]   -> kernels/lab/k3_gdh_<n>.py

Lab form (default): base + (all inert unless armed / selected)
- GD per-CTA marks (tools/gdd_patches.py marks()), sbb1 per-CTA marks + stamps (lat_patches.sbb_marks, anchors adapted
  to the shipped trig_early field), KS1 smid -> tools/box/lat_marks.py works on it;
- GD alternates LAT_GD_ALT {(proj, b, tag): route} / LAT_GD_ON {(proj, b): tag} and GC alternates LAT_CAT_ALT {(b, tag): route}
  / LAT_CAT_ON {b: tag} (built + self-tested at prepare, picked at capture time);
- XP 2 GD template: the next unit's x half 0 is loaded right after q = 1 (its last use), half 1 after q = 3: x one half-unit
  ahead with the XP 0 register count (XP 1 doubles X and spills at MT 4);
- VF_SBB2 self-tests 16 / 32 (dispatch stays off: VF_SBB2 16 / 32 False until a variant sets them).
Clean form (--clean=trig,sbb2[,x2]): base + only the named switches on at T = 16 / 32, no marks.
"""
import hashlib
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from gdd_patches import check, sub, marks as gd_marks  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)
X2_GD = {("qkv", 16, "x2"): (2, 1, 4, 8, 1, 4, 2), ("qkv", 32, "x2"): (4, 1, 4, 8, 1, 4, 2),
         ("o", 16, "x2"): (2, 2, 4, 4, 2, 1, 2),
         ("unpooler", 16, "x2"): (2, 2, 4, 16, 2, 1, 2), ("unpooler", 32, "x2"): (4, 2, 4, 16, 1, 1, 2)}
X2_CAT = {(16, "x2"): (2, 1, 8, 18, 2, 1, 2), (32, "x2"): (4, 1, 8, 18, 1, 1, 2)}


def sbb_marks(t):
    a0 = t.index("SBB1_CU = r'''")
    a1 = t.index("SBB1_DECL = ", a0)
    c = t[a0:a1]
    c = sub(c, "  int trig_early;   // LAT: launch_dependents at entry\n};",
            "  int trig_early;   // LAT: launch_dependents at entry\n  long long* marks;\n};", what="params marks")
    c = sub(c, "template <bool kPreStream, bool kRouter, bool kInputScale>\n__global__",
            """__device__ __forceinline__ long long sb_gt() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ long long sb_smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return (long long)s; }
#define SBM(k) do { if (p.marks != nullptr && threadIdx.x == 0) p.marks[blockIdx.x * 8 + (k)] = sb_gt(); } while (0)
template <bool kPreStream, bool kRouter, bool kInputScale>
__global__""", what="sbm defs")
    c = sub(c, '  if (p.trig_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n  uint32_t rank;',
            '  if (p.trig_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n'
            "  if (p.marks != nullptr && tid == 0) p.marks[blockIdx.x * 8 + 0] = sb_smid();\n  SBM(1);\n  uint32_t rank;",
            what="sbm entry")
    c = sub(c, '  asm volatile("griddepcontrol.wait;" ::: "memory");\n  float br[8];',
            '  asm volatile("griddepcontrol.wait;" ::: "memory");\n  SBM(2);\n  float br[8];', what="sbm wait")
    c = sub(c, "  __syncthreads();\n  float4 mine", "  __syncthreads();\n  SBM(3);\n  float4 mine", what="sbm red")
    c = sub(c, "  __syncthreads();\n  float tb = 0.f", "  __syncthreads();\n  SBM(4);\n  float tb = 0.f", what="sbm cluster")
    c = sub(c, "  if (kRouter) sb_st8f(p.router + (size_t)row * SB_H + e0, nz);\n}",
            "  if (kRouter) sb_st8f(p.router + (size_t)row * SB_H + e0, nz);\n  SBM(5);\n}", what="sbm end")
    c = sub(c, "void sbb1_run(torch::Tensor branch,", """static long long* g_sb_mk = nullptr;
static int g_sb_mk_n = 0;
void sbb1_marks(torch::Tensor buf) {
  g_sb_mk = buf.numel() >= 8 * 4096 ? reinterpret_cast<long long*>(buf.data_ptr()) : nullptr;
  g_sb_mk_n = 0;
}
int64_t sbb1_marks_n() { return g_sb_mk_n; }
__global__ void sb_stamp_kernel(long long* q) { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); *q = t; }
void sbb1_stamp(torch::Tensor buf, int64_t i) {
  sb_stamp_kernel<<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<long long*>(buf.data_ptr()) + i);
}
void sbb1_run(torch::Tensor branch,""", what="sbb host")
    c = sub(c, "  p.trig_early = (input_scale & 8) ? 1 : 0;\n",
            "  p.trig_early = (input_scale & 8) ? 1 : 0;\n"
            "  p.marks = (g_sb_mk && g_sb_mk_n < 8) ? g_sb_mk + (size_t)(g_sb_mk_n++) * 4096 : nullptr;\n", what="sbb take")
    t = t[:a0] + c + t[a1:]
    t = sub(t, "SBB1_DECL = 'void sbb1_run(", "SBB1_DECL = 'void sbb1_marks(torch::Tensor buf); int64_t sbb1_marks_n(); "
            "void sbb1_stamp(torch::Tensor buf, int64_t i); void sbb1_run(", what="decl")
    t = sub(t, 'load_inline("sbb1_k3t", cpp_sources=SBB1_DECL, cuda_sources=SBB1_CU + SBB1_HOST, functions=["sbb1_run", "vf_router32"],',
            'load_inline("sbb1_gdh", cpp_sources=SBB1_DECL, cuda_sources=SBB1_CU + SBB1_HOST, functions=["sbb1_run", "vf_router32", '
            '"sbb1_marks", "sbb1_marks_n", "sbb1_stamp"],', what="load_inline")
    return t


def xp2(t):
    """XP 2: x one half-unit ahead without a second X buffer (XP 0 / 1 code paths unchanged)."""
    a0 = t.index("GD_CUDA_SRC = r'''")
    a1 = t.index(NL + "'''" + NL, a0)
    c = t[a0:a1]
    c = sub(c, "  constexpr int XB = XP ? 2 : 1;\n", "  constexpr int XB = XP == 1 ? 2 : 1;   // GDH: XP 2 = half-unit x pipeline, one buffer\n",
            what="xb")
    c = sub(c, "    const int xs = XP ? (s & 1) : 0;\n    if (XP) {", "    const int xs = XP == 1 ? (s & 1) : 0;\n    if (XP == 1) {",
            what="xs")
    c = sub(c, "    } else if (s > 0) {\n#pragma unroll\n      for (int mt = 0; mt < MT; ++mt) {\n        X[0][mt][0] = ldx(GC_XP(s) + xo[mt]);",
            "    } else if (XP == 0 && s > 0) {\n#pragma unroll\n      for (int mt = 0; mt < MT; ++mt) {\n        X[0][mt][0] = ldx(GC_XP(s) + xo[mt]);",
            what="xp0 only")
    c = sub(c, """        mma16816(acc[q % NC][mt], a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], b0, b1);
      }
    }
""", """        mma16816(acc[q % NC][mt], a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], b0, b1);
      }
      if constexpr (XP == 2) {   // GDH: half (q >> 1) of unit s is dead after q = 1 / 3 -> load unit s + 1's half now
        if ((q & 1) && s + 1 < KU) {
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) X[0][mt][q >> 1] = ldx(GC_XP(s + 1) + xo[mt] + (q >> 1));
        }
      }
    }
""", what="xp2 loads")
    t = t[:a0] + c + t[a1:]
    t = sub(t, "st_ > ku or xp not in (0, 1):\n        return None\n    if (2 * N_EXPERTS)",
            "st_ > ku or xp not in (0, 1, 2):\n        return None\n    if (2 * N_EXPERTS)", what="cat key xp")
    return t


def alt_routes(t):
    t = sub(t, "GD_ONE_WAVE = True\n", "GD_ONE_WAVE = True\n" + f"LAT_GD_ALT = {X2_GD!r}" + "\nLAT_GD_ON = {}\n"
            + f"LAT_CAT_ALT = {X2_CAT!r}" + "\nLAT_CAT_ON = {}\n\n\n"
            "def _lat_items():\n"
            "    return ([((p_, b_, None), r_) for (p_, b_), r_ in sorted(GD_ROUTE.items(), key=str)]\n"
            "            + [(k_, r_) for k_, r_ in sorted(LAT_GD_ALT.items(), key=str)])\n", what="alt globals")
    t = sub(t, """    keys = []
    for (proj, b), rt in sorted(GD_ROUTE.items(), key=str):
        key = _gd_key(*_gd_nk(proj, b), rt) if rt else None""", """    keys = []
    for (proj, b, *_), rt in sorted(GD_ROUTE.items(), key=str) + sorted(LAT_GD_ALT.items(), key=str):
        key = _gd_key(*_gd_nk(proj, b), rt) if rt else None""", what="alt keys")
    t = sub(t, """    for (proj, b), rt in sorted(GD_ROUTE.items(), key=str):
        if not rt:
            continue
        n, k = _gd_nk(proj, b)""", """    for (proj, b, alt), rt in _lat_items():
        if not rt:
            continue
        n, k = _gd_nk(proj, b)""", what="alt prepare loop")
    t = sub(t, """                routes[(proj, b)] = r
                live.append(f"{proj}/{b} {desc}")""", """                routes[(proj, b, alt) if alt else (proj, b)] = r
                live.append(f"{proj}/{b}{' ALT ' + alt if alt else ''} {desc}")""", what="alt store")
    t = sub(t, """    if b is None or not GD_ROUTE.get((proj, b)):
        return None
    r = st["route"].get((proj, b))""", """    if b is None or not GD_ROUTE.get((proj, b)):
        return None
    r = (st["route"].get((proj, b, LAT_GD_ON[(proj, b)])) if LAT_GD_ON.get((proj, b)) else None) or st["route"].get((proj, b))""",
            what="alt live")
    # GC cat alternates
    t = sub(t, """    keys = []
    for b, rt in sorted(GD_CAT_ROUTE.items()):
        key = _gd_cat_key(rt) if rt else None""", """    keys = []
    for b, rt in sorted(GD_CAT_ROUTE.items()) + sorted(LAT_CAT_ALT.items(), key=str):
        key = _gd_cat_key(rt) if rt else None""", what="cat alt keys")
    t = sub(t, """    for b, rt in sorted(GD_CAT_ROUTE.items()):
        why = None
        key = _gd_cat_key(rt) if rt else None""", """    for bk, rt in sorted(GD_CAT_ROUTE.items()) + sorted(LAT_CAT_ALT.items(), key=str):
        b, alt = (bk, None) if isinstance(bk, int) else bk
        why = None
        key = _gd_cat_key(rt) if rt else None""", what="cat alt loop")
    t = sub(t, """                routes[b] = r
                live.append(f"cat/{b} {desc}")""", """                routes[(b, alt) if alt else b] = r
                live.append(f"cat/{b}{' ALT ' + alt if alt else ''} {desc}")""", what="cat alt store")
    t = sub(t, """    r = st.get(b)
    return r if r is not None and tokens <= r["T"] else None""", """    r = (st.get((b, LAT_CAT_ON[b])) if LAT_CAT_ON.get(b) else None) or st.get(b)
    return r if r is not None and tokens <= r["T"] else None""", what="cat alt live")
    t = sub(t, "VF_SBB2 = {1: True, 8: True, 16: False}\n", "VF_SBB2 = {1: True, 8: True, 16: False, 32: False}   # GDH: 16 / 32 self-tested\n",
            what="sbb2 keys")
    return t


def clean(t, arms):
    if "trig" in arms:
        t = sub(t, "LAT_SBB_TRIG = {1: True, 8: True}\n",
                "LAT_SBB_TRIG = {1: True, 8: True, 16: True, 32: True}   # GDH: pooler/cat resident before the boundary's wait\n",
                what="trig 16/32")
    if "sbb2" in arms:
        t = sub(t, "VF_SBB2 = {1: True, 8: True, 16: False}\n",
                "VF_SBB2 = {1: True, 8: True, 16: True, 32: True}   # GDH: FFN boundary on sbb1's kernel at 16 / 32\n", what="sbb2 16/32")
    if "x2" in arms:
        for (p, b, _), rt in X2_GD.items():
            old = GD_OLD[(p, b)]
            t = sub(t, f'("{p}", {b}): {old!r}', f'("{p}", {b}): {rt!r}', what=f"x2 {p}/{b}")
        t = xp2(t)
        t = sub(t, "GD_CAT_ROUTE = {16: (2, 1, 8, 18, 2, 1, 0), 32: (4, 1, 8, 18, 1, 1, 0)}",
                "GD_CAT_ROUTE = {16: (2, 1, 8, 18, 2, 1, 2), 32: (4, 1, 8, 18, 1, 1, 2)}", what="x2 cat")
    t = sub(t, 'load_inline("sbb1_k3t",', 'load_inline("sbb1_gdh",', what="module name")
    return t


GD_OLD = {("qkv", 16): (2, 1, 4, 8, 1, 4, 0), ("qkv", 32): (4, 1, 4, 8, 1, 4, 0), ("o", 16): (2, 2, 4, 4, 2, 1, 0),
          ("unpooler", 16): (2, 2, 4, 16, 2, 1, 0), ("unpooler", 32): (4, 2, 4, 16, 1, 1, 0)}


def main():
    n = ARGS.get("n")
    check(n, "--n=<n> required")
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_14')}.py"
    out = ROOT / f"kernels/lab/k3_gdh_{n}.py"
    check(not out.exists() or "--force" in sys.argv, f"{out.name} exists (never overwrite)")
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    if ARGS.get("clean"):
        arms = ARGS["clean"].split(",")
        t = clean(b, arms)
        what = "clean: " + "+".join(arms) + " at T = 16 / 32"
    else:
        t = alt_routes(xp2(sbb_marks(gd_marks(b))))
        t = sub(t, "  KS1P(0);\n", "  KS1P(0);\n  if (p.prof != nullptr && threadIdx.x == 0) { unsigned sm_; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(sm_)); "
                "p.prof[cta * 16 + 15] = (long long)sm_; }\n", what="ks1 smid")
        what = "GD / sbb1 marks + XP 2 + GD / GC alternates + SBB2 16/32 self-tests (inert unless armed)"
    nonce = f"# nonce k3-gdh-{n} ({base.stem} + {what}; tools/gdh_patches.py)"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    check(re.search(r"importlib|sys\.modules", t) is None or re.search(r"importlib|sys\.modules", b) is not None, "checker")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[gdh_patches] OK {len(t.encode())} B (base {len(b.encode())} B)")
    if "--dry" in sys.argv:
        return 0
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[gdh_patches] wrote {out.relative_to(ROOT).as_posix()} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""lat_patches.py: latency-phase step-0 instrumentation (T <= 8: KS1, sbb1, final norm, GD placement) on a layer file.

  python -I tools/lat_patches.py --n=<n> [--base=k3_gc_11] [--dry]   -> kernels/lab/k3_lat_<n>.py

Adds (all inert unless armed):
- GD per-CTA marks (tools/gdd_patches.py marks(): smid, entry, post-wait, end; gd_marks(buf) arms a 4096-long slot per launch);
- sbb1 per-CTA marks: marks[c*8 + {0 smid, 1 entry, 2 post-wait, 3 branch reduced (block), 4 cluster moments in,
  5 stores issued}] (globaltimer ns); sbb1_marks(buf int64[8*4096]) arms one 4096-long slot per sbb1_run (attn and the
  VF_SBB2 ffn path); sbb1_marks_n() = slots used;
- sbb1_stamp(buf, i): a 1-thread kernel (no PDL attribute) writing %globaltimer into buf[i] (layer start / end anchors).
KS1 already has KS1P marks (KS1_PROF_BUF).
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


def sbb_marks(t):
    a0 = t.index("SBB1_CU = r'''")
    a1 = t.index("SBB1_DECL = ", a0)
    c = t[a0:a1]
    c = sub(c, "  int gate_channel_wise;\n};", "  int gate_channel_wise;\n  long long* marks;\n};", what="params marks")
    c = sub(c, "template <bool kPreStream, bool kRouter, bool kInputScale>\n__global__",
            """__device__ __forceinline__ long long sb_gt() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ long long sb_smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return (long long)s; }
#define SBM(k) do { if (p.marks != nullptr && threadIdx.x == 0) p.marks[blockIdx.x * 8 + (k)] = sb_gt(); } while (0)
template <bool kPreStream, bool kRouter, bool kInputScale>
__global__""", what="sbm defs")
    c = sub(c, "  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  uint32_t rank;",
            "  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n"
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
    c = sub(c, "  p.post_eps = (float)post_eps; p.pre_eps = (float)pre_eps;\n",
            "  p.post_eps = (float)post_eps; p.pre_eps = (float)pre_eps;\n"
            "  p.marks = (g_sb_mk && g_sb_mk_n < 8) ? g_sb_mk + (size_t)(g_sb_mk_n++) * 4096 : nullptr;\n", what="sbb take")
    t = t[:a0] + c + t[a1:]
    t = sub(t, "SBB1_DECL = 'void sbb1_run(", "SBB1_DECL = 'void sbb1_marks(torch::Tensor buf); int64_t sbb1_marks_n(); "
            "void sbb1_stamp(torch::Tensor buf, int64_t i); void sbb1_run(", what="decl")
    t = sub(t, 'load_inline("sbb1_k3", cpp_sources=SBB1_DECL, cuda_sources=SBB1_CU + SBB1_HOST, functions=["sbb1_run", "vf_router32"],',
            'load_inline("sbb1_lat", cpp_sources=SBB1_DECL, cuda_sources=SBB1_CU + SBB1_HOST, functions=["sbb1_run", "vf_router32", '
            '"sbb1_marks", "sbb1_marks_n", "sbb1_stamp"],', what="load_inline")
    return t


def alt_routes(t):
    """LAT_GD_ALT {(proj, b, tag): route}: extra GD instances built + packed + self-tested at prepare (shared pack when
    only ST differs); LAT_GD_ON {(proj, b): tag} picks one at call (capture) time. LAT_GD_ON empty = the base file's routes."""
    t = sub(t, "GD_ONE_WAVE = True\n", "GD_ONE_WAVE = True\n" + ARGS.get("altdef", "LAT_GD_ALT = {}") + "\nLAT_GD_ON = {}\n\n\n"
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
    # sbb1 early trigger: LAT_SBB_TRIG {bucket: True} -> griddepcontrol.launch_dependents at entry (input_scale bit 8)
    t = sub(t, "  long long* marks;\n};", "  long long* marks;\n  int trig_early;\n};", what="sbb trig field")
    t = sub(t, "  if (p.marks != nullptr && tid == 0) p.marks[blockIdx.x * 8 + 0] = sb_smid();\n",
            "  if (p.trig_early) asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n"
            "  if (p.marks != nullptr && tid == 0) p.marks[blockIdx.x * 8 + 0] = sb_smid();\n", what="sbb trig entry")
    t = sub(t, "  p.post_eps = (float)post_eps; p.pre_eps = (float)pre_eps;\n",
            "  p.post_eps = (float)post_eps; p.pre_eps = (float)pre_eps;\n  p.trig_early = (input_scale & 8) ? 1 : 0;\n",
            what="sbb trig host")
    t = sub(t, "    st[\"mod\"].sbb1_run(branch.contiguous(), stream.contiguous(), out, ns, rt, pw, qw, al, bt, pe, qe, isc)\n",
            "    st[\"mod\"].sbb1_run(branch.contiguous(), stream.contiguous(), out, ns, rt, pw, qw, al, bt, pe, qe,\n"
            "                       isc | (8 if LAT_SBB_TRIG.get(_k2a_bucket_of(tokens)) else 0))\n", what="sbb trig call")
    t = sub(t, "SBB1_ENABLE = True\n", "SBB1_ENABLE = True\nLAT_SBB_TRIG = {}\n", what="sbb trig global")
    return t


def trig_only(t):
    """Ship form (--clean=1): sbb1 early trigger default on at T = 1 / 8 (R-lat-1/2) + VF_SBB2 final norm at T = 1 / 8,
    no marks, no alternates. The trigger only moves griddepcontrol.launch_dependents: attention-boundary outputs bitwise."""
    t = sub(t, "  int gate_channel_wise;\n};", "  int gate_channel_wise;\n  int trig_early;   // LAT: launch_dependents at entry\n};",
            what="sbb trig field")
    t = sub(t, "  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  uint32_t rank;",
            "  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n"
            "  if (p.trig_early) asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n  uint32_t rank;",
            what="sbb trig entry")
    t = sub(t, "  p.post_eps = (float)post_eps; p.pre_eps = (float)pre_eps;\n",
            "  p.post_eps = (float)post_eps; p.pre_eps = (float)pre_eps;\n  p.trig_early = (input_scale & 8) ? 1 : 0;\n",
            what="sbb trig host")
    t = sub(t, "    st[\"mod\"].sbb1_run(branch.contiguous(), stream.contiguous(), out, ns, rt, pw, qw, al, bt, pe, qe, isc)\n",
            "    st[\"mod\"].sbb1_run(branch.contiguous(), stream.contiguous(), out, ns, rt, pw, qw, al, bt, pe, qe,\n"
            "                       isc | (8 if LAT_SBB_TRIG.get(_k2a_bucket_of(tokens)) else 0))\n", what="sbb trig call")
    t = sub(t, "SBB1_ENABLE = True\n", "SBB1_ENABLE = True\nLAT_SBB_TRIG = {1: True, 8: True}   # R-lat-1/2: pooler resident ~2 us earlier\n",
            what="sbb trig global")
    t = sub(t, "VF_SBB2 = {1: False, 8: False, 16: False}\n", "VF_SBB2 = {1: True, 8: True, 16: False}   # R-lat-2 rider (stsb)\n",
            what="sbb2 on")
    t = sub(t, 'load_inline("sbb1_k3",', 'load_inline("sbb1_k3t",', what="module name")
    return t


def main():
    n = ARGS.get("n")
    check(n, "--n=<n> required")
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_11')}.py"
    out = ROOT / f"kernels/lab/k3_lat_{n}.py"
    check(not out.exists(), f"{out.name} exists (never overwrite)")
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    if ARGS.get("clean") == "1":
        t = trig_only(b)
    else:
        t = sbb_marks(gd_marks(b))
    if ARGS.get('alt', '1') == '1' and ARGS.get("clean") != "1":
        t = alt_routes(t)
    if ARGS.get("clean") != "1":
        t = sub(t, "  KS1P(0);\n", "  KS1P(0);\n  if (p.prof != nullptr && threadIdx.x == 0) { unsigned sm_; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(sm_)); "
            "p.prof[cta * 16 + 15] = (long long)sm_; }\n", what="ks1 smid")
    what = ("sbb1 trigger at entry + sbb2 final norm at T <= 8" if ARGS.get("clean") == "1"
            else "GD / sbb1 per-CTA marks + stamps (inert unless armed)")
    nonce = f"# nonce k3-lat-{n} ({base.stem} + {what}; tools/lat_patches.py)"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    check(re.search(r"importlib|sys\.modules", t) is None or re.search(r"importlib|sys\.modules", b) is not None, "checker")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[lat_patches] OK {len(t.encode())} B (base {len(b.encode())} B)")
    if "--dry" in sys.argv:
        return 0
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[lat_patches] wrote {out.relative_to(ROOT).as_posix()} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

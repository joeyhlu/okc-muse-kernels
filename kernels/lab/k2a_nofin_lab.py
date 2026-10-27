"""NOFIN2 lab (R11 kill test): kernels/lab/k2a_fb_lab.py + one timing-only probe. LAB FILE: box runs only.

HP bit 32 (P.hp, run3's hp argument): the K2 (MODE 2) epilogue warp of the v3.3 gemm (vids 0-2) and of gemm36 (vid 3)
returns at once, i.e. K2 does no split-item counters and no split-item finalize (sum of squares -> ss). The consumers'
non-blocking bar.arrive hand-offs then go unmatched (named barriers reset at CTA exit). ss misses the split items' share,
so the output is wrong: timing only. It is the ceiling of moving K2's ss out of K2 (ss computed in K3), the K2 -> K3 hop
lever. K1 (MODE 1) is untouched.
"""
import hashlib
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hot_p1_lab as H  # noqa: E402
import k2a_fb_lab as F  # noqa: E402

REC = F.REC
_ORIG = F.patched_source   # captured before lab_source() swaps it
PROBE = ("    if (MODE == 2 && (P.hp & 32) != 0) return;   // NOFIN2 lab probe: no K2 split-item finalize (timing only)\n"
         "    if (MODE == 1 && (P.hp & 64) != 0) return;   // NOFIN1 lab probe (R13): no K1 split-item finalize (timing only)\n")
ANCHORS = [
    ("__device__ void gemm(const Params& P, Route& S,", "__device__ __forceinline__ void setup_bars(",
     "  if (warp == 9) {\n    // ---------------- epilogue warp: split-item counters; the last contributor finishes the item ----------------\n"),
    ("__device__ void gemm36(", "k1_kernel36(const Params P) {",
     "  if (warp == 9) {\n    // ---------------- epilogue warp: split-item counters; the last contributor finishes the item (v3.3) ----------------\n"),
]


def patched_source():
    src = _ORIG()
    for start, end, old in ANCHORS:
        a, b = H._region(src, start, end)
        if src[a:b].count(old) != 1:
            raise RuntimeError(f"k2a_nofin: epilogue anchor x{src[a:b].count(old)} in {start!r}")
        i = src.index(old, a)
        src = src[:i] + old + PROBE + src[i + len(old):]
    return src


def lab_source():
    F.patched_source = patched_source
    try:
        return F.lab_source()
    finally:
        F.patched_source = _ORIG


def build(name="k2a_nofin_lab", verbose=False):
    from torch.utils.cpp_extension import load_inline
    cuda, cpp = lab_source()
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=cuda,
                      functions=["fbrun", "fbrun1", "fbattrs", "fbattrs1", "fbstream_run"], extra_cuda_cflags=H.CFLAGS, verbose=verbose)
    return mod, time.time() - t0


if __name__ == "__main__":
    cu, cp = lab_source()
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "k2a_nofin_lab_gen.cu")
    out.write_text(cu, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(cu)} chars, NOFIN2 probes: {cu.count('NOFIN2 lab probe')})")

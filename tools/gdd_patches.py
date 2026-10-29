"""gdd_patches.py: GD (dense GEMV, T <= 32) per-CTA end marks (+ later: DYN chunk claiming) on a layer file.

  python -I tools/gdd_patches.py --n=<n> [--base=k3_gc_10] [--dry]   -> kernels/lab/k3_gdd_<n>.py

Marks (always in, inert unless armed): gd_kernel writes marks[c*4 + {0 smid, 1 entry, 2 after griddepcontrol.wait,
3 end}] (globaltimer ns) for every instance (VF 0 too; c = linear CTA index) when Args.marks is set. Host side:
gd_marks(buf int64[64*4096]) arms it; every following gd_run / gdc_run / gdr_run (no own marks) takes the next 4096-long
slot and logs (kind 1 gd / 2 gdc / 3 gdr, inst|vf, M, K, ldo, nbx, nz); gd_marks_log() returns the log; gd_marks(empty)
disarms. Arm only around a CUDA-graph capture (tools/box/gdd_marks.py): slots are baked into the captured launches.
"""
import hashlib
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)


def check(c, msg):
    if not c:
        raise SystemExit(f"[gdd_patches] FAIL: {msg}")


def sub(t, old, new, count=1, what=""):
    n = t.count(old)
    check(n == count, f"anchor {what or old[:70]!r}: found {n}, expected {count}")
    return t.replace(old, new)


def marks(t):
    a0 = t.index("GD_CUDA_SRC = r'''")
    a1 = t.index(NL + "'''" + NL, a0)
    c = t[a0:a1]
    c = sub(c, """  if constexpr (VF >= 1) {
    if (A.marks && threadIdx.x == 0) { A.marks[vf_c * 4 + 0] = vf_smid(); A.marks[vf_c * 4 + 1] = vf_gt(); }
  }
""", """  const int mk_c = VF >= 1 ? vf_c : (int)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));   // GDD marks
  if (A.marks && threadIdx.x == 0) { A.marks[mk_c * 4 + 0] = vf_smid(); A.marks[mk_c * 4 + 1] = vf_gt(); }
""", what="mark entry")
    c = sub(c, """  pdl_wait();
  if constexpr (VF >= 1) {
    if (A.marks && threadIdx.x == 0) A.marks[vf_c * 4 + 2] = vf_gt();
  }
""", """  pdl_wait();
  if (A.marks && threadIdx.x == 0) A.marks[mk_c * 4 + 2] = vf_gt();
""", what="mark wait")
    c = sub(c, """    if constexpr (VF >= 1) {
      if (A.marks) {
        __syncthreads();
        if (threadIdx.x == 0) A.marks[vf_c * 4 + 3] = vf_gt();
      }
    }
""", """    if (A.marks) {
      __syncthreads();
      if (threadIdx.x == 0) A.marks[mk_c * 4 + 3] = vf_gt();
    }
""", what="mark end S1")
    c = sub(c, """st_async(rb + 4u * (uint32_t)i, cta_value(i), rm);
      return;""", """st_async(rb + 4u * (uint32_t)i, cta_value(i), rm);
      if (A.marks && threadIdx.x == 0) A.marks[mk_c * 4 + 3] = vf_gt();
      return;""", what="mark end rank")
    c = sub(c, """        store(i, ok ? v : __int_as_float(0x7fc00000));
      }
    }
#endif""", """        store(i, ok ? v : __int_as_float(0x7fc00000));
      }
    }
    if (A.marks) {
      __syncthreads();
      if (threadIdx.x == 0) A.marks[mk_c * 4 + 3] = vf_gt();
    }
#endif""", what="mark end leader")
    c = sub(c, "using namespace gd;\n", """using namespace gd;

static long long* g_gd_mk = nullptr;   // GDD marks: armed by gd_marks(), one 4096-long slot per launch
static int g_gd_mk_n = 0;
static long long g_gd_mk_log[64 * 8];
static long long* gd_mk_take(int kind, long long inst, int M, int K, int ldo, int nbx, int nz) {
  if (!g_gd_mk || g_gd_mk_n >= 64) return nullptr;
  long long* L = g_gd_mk_log + g_gd_mk_n * 8;
  L[0] = kind; L[1] = inst; L[2] = M; L[3] = K; L[4] = ldo; L[5] = nbx; L[6] = nz; L[7] = 0;
  return g_gd_mk + (size_t)(g_gd_mk_n++) * 4096;
}
void gd_marks_raw(long long* p) { g_gd_mk = p; g_gd_mk_n = 0; }
int gd_marks_log_raw(long long* out) {
  for (int i = 0; i < g_gd_mk_n * 8; ++i) out[i] = g_gd_mk_log[i];
  return g_gd_mk_n;
}
""", what="mark host")
    c = sub(c, "A.marks = nullptr;\n  switch (inst) {", "A.marks = gd_mk_take(1, inst, M, K, ldo, nbx, nz);\n  switch (inst) {",
            what="gd_run_raw marks")
    c = sub(c, "A.marks = nullptr;\n  A.out2 = out2;", "A.marks = gd_mk_take(2, inst, M, K, ldo, nbx, 1);\n  A.out2 = out2;",
            what="gdc_run_raw marks")
    c = sub(c, "A.part = part; A.marks = marks;", "A.part = part; A.marks = marks ? marks : gd_mk_take(3, vf, M, K, ldo, nbx, 1);",
            what="gdr_run_raw marks")
    t = t[:a0] + c + t[a1:]
    t = sub(t, "std::vector<int64_t> gdr_info(int64_t vf) {", """void gd_marks_raw(long long* p);
int gd_marks_log_raw(long long* out);
void gd_marks(torch::Tensor buf) {
  gd_marks_raw(buf.numel() >= 64 * 4096 ? reinterpret_cast<long long*>(buf.data_ptr()) : nullptr);
}
std::vector<int64_t> gd_marks_log() {
  std::vector<int64_t> r(64 * 8, 0);
  const int n = gd_marks_log_raw(reinterpret_cast<long long*>(r.data()));
  r.resize((size_t)n * 8);
  return r;
}
std::vector<int64_t> gdr_info(int64_t vf) {""", what="cpp marks")
    t = sub(t, '"gdc_run", "gd_unpack_cat"]', '"gdc_run", "gd_unpack_cat", "gd_marks", "gd_marks_log"]', what="functions")
    return t


def main():
    n = ARGS.get("n")
    check(n, "--n=<n> required")
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_10')}.py"
    out = ROOT / f"kernels/lab/k3_gdd_{n}.py"
    check(not out.exists() or "--force" in sys.argv, f"{out.name} exists")
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    t = marks(b)
    nonce = f"# nonce k3-gdd-{n} ({base.stem} + GDD per-CTA marks (inert unless armed); tools/gdd_patches.py)"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    check(re.search(r"importlib|sys\.modules", t) is None or re.search(r"importlib|sys\.modules", b) is not None, "checker")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[gdd_patches] OK {len(t.encode())} B (base {len(b.encode())} B)")
    if "--dry" in sys.argv:
        return 0
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[gdd_patches] wrote {out.relative_to(ROOT).as_posix()} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

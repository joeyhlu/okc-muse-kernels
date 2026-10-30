"""ulp_patches.py: ULP = unpooler at T >= 128 through a pinned NN cuBLASLt plan with a batched layout (notes/study/unpooler_lt.md, R-ulp-0).

  python tools/ulp_patches.py --base=kernels/lab/k3_rider3_1.py --out=kernels/lab/k3_ulp_1.py [--t256=1] [--t128=0]
         [--batch=2] [--custom256=7] [--custom128=3] [--cluster=5] [--check]

What it changes (every anchor must match exactly once, else SystemExit):
  * LT_CPP: lt_plan becomes a wrapper of lt_plan_impl(m, n, k, out_f32, nn, batch, ws, max_algos); new lt_plan_b exposes it.
    nn = 1: A is the weight stored TRANSPOSED ([K, N] row-major, i.e. column-major [N, K] with ld N) and TRANSA = N;
    batch = B > 1: the three layouts get BATCH_COUNT = B, A / C batch offsets N / B rows (output features split in B slices,
    the same bytes, no split-K), x batch offset 0. batch = 1 and nn = 0 reproduce the old plan exactly.
  * _lt_build exports lt_plan_b; _lt_prepare builds pins whose ULP_LAYOUT entry says (nn, batch); the transposed weight
    (lossless repack, 64 MB, one copy per projection) is made in prepare from state["dense"], never from raw pointers baked in.
    Every pin still goes through the existing self-test (finite, bitwise repeat, e64 < DG1_TOL[0], vs today < DG1_TOL[2]) and
    falls back to today's path (torchT at 128, the default plan at 256) when anything fails.
  * _lt: an NN pin passes its own transposed weight and needs x 16-B aligned (the TN pins keep their 256-B rule).
  * LT_PIN gains ('unpooler', 256): (66, 20, 35, 1, 0, 0, CUSTOM256, 0, CLUSTER) = tile 128x128, 35 stages, no split-K,
    custom option 7, cluster id 5 (the config decoded from the public layer's pin words) and, with --t128=1,
    ('unpooler', 128): (66, 18, 35, 1, 0, 0, CUSTOM128, 0, CLUSTER) (today's torchT already runs this config at batch 1).
Local gates only (no GPU): ast parse, ASCII, size < 1,000,000 B, anchors; --check also prints the diff size.
"""
import argparse
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _rep(src, old, new, what, n=1):
    c = src.count(old)
    if c != n:
        raise SystemExit(f"[ulp_patches] {what}: expected {n} match(es), got {c}")
    return src.replace(old, new)


LIMIT = 1_000_000

# ---- LT_CPP edits -------------------------------------------------------------------------------------------------------
CPP_HEAD_OLD = "int64_t lt_plan(int64_t m, int64_t n, int64_t k, int64_t out_f32, int64_t ws_bytes, int64_t max_algos) {\n"
CPP_HEAD_NEW = ("static int64_t lt_plan_impl(int64_t m, int64_t n, int64_t k, int64_t out_f32, int64_t nn, int64_t batch,\n"
                "                              int64_t ws_bytes, int64_t max_algos) {\n"
                "  TORCH_CHECK(batch >= 1 && n % batch == 0, \"lt: batch\");\n"
                "  const int64_t nb = n / batch;\n")
CPP_TA_OLD = "  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;\n"
CPP_TA_NEW = "  cublasOperation_t ta = nn ? CUBLAS_OP_N : CUBLAS_OP_T, tb = CUBLAS_OP_N;\n"
CPP_LA_OLD = '  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, k, n, k) == CUBLAS_STATUS_SUCCESS, "lt: la");\n'
CPP_LA_NEW = ('  TORCH_CHECK((nn ? cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, nb, k, n)\n'
              '                  : cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, k, nb, k)) == CUBLAS_STATUS_SUCCESS, "lt: la");\n')
CPP_LC_OLD = '  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.lc, p.ct, n, m, n) == CUBLAS_STATUS_SUCCESS, "lt: lc");\n'
CPP_LC_NEW = ('  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.lc, p.ct, nb, m, n) == CUBLAS_STATUS_SUCCESS, "lt: lc");\n'
              '  if (batch > 1) {   // ULP: B slices of the output features; x is shared (batch offset 0)\n'
              '    const int32_t bc = (int32_t)batch;\n'
              '    cublasLtMatrixLayout_t ls[3] = {p.la, p.lb, p.lc};\n'
              '    const int64_t so[3] = {nn ? nb : k * nb, 0, nb};\n'
              '    for (int i = 0; i < 3; ++i) {\n'
              '      TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)) ==\n'
              '                      CUBLAS_STATUS_SUCCESS, "lt: batch count");\n'
              '      TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &so[i],\n'
              '                                                   sizeof(int64_t)) == CUBLAS_STATUS_SUCCESS, "lt: batch offset");\n'
              '    }\n'
              '  }\n')
CPP_TAIL_OLD = "}\nstd::vector<int64_t> lt_cfg(int64_t pid, int64_t ai) {\n"
CPP_TAIL_NEW = ("}\n"
                "int64_t lt_plan(int64_t m, int64_t n, int64_t k, int64_t out_f32, int64_t ws_bytes, int64_t max_algos) {\n"
                "  return lt_plan_impl(m, n, k, out_f32, 0, 1, ws_bytes, max_algos);\n"
                "}\n"
                "int64_t lt_plan_b(int64_t m, int64_t n, int64_t k, int64_t out_f32, int64_t nn, int64_t batch, int64_t ws_bytes,\n"
                "                  int64_t max_algos) {\n"
                "  return lt_plan_impl(m, n, k, out_f32, nn, batch, ws_bytes, max_algos);\n"
                "}\n"
                "std::vector<int64_t> lt_cfg(int64_t pid, int64_t ai) {\n")

# ---- Python edits ---------------------------------------------------------------------------------------------------------
BUILD_OLD = 'functions=["lt_plan", "lt_cfg", "lt_make", "lt_run"]'
BUILD_NEW = 'functions=["lt_plan", "lt_plan_b", "lt_cfg", "lt_make", "lt_run"]'

PREP_W_OLD = ("            w = _dg1_weight(state, proj)\n"
              "            n, k = w.shape\n"
              "            f32 = proj == \"cat\"\n"
              "            code = int(mod.lt_plan(T, n, k, int(f32), LT_WS, 32))\n")
PREP_W_NEW = ("            w = _dg1_weight(state, proj)\n"
              "            n, k = w.shape\n"
              "            f32 = proj == \"cat\"\n"
              "            nn, bt = ULP_LAYOUT.get((proj, T), (0, 1))\n"
              "            wl = w\n"
              "            if nn:\n"
              "                if proj not in wts:\n"
              "                    wts[proj] = w.t().contiguous()\n"
              "                wl = wts[proj]\n"
              "                assert wl.is_contiguous() and wl.data_ptr() % 256 == 0 and wl.shape == (k, n)\n"
              "            code = int(mod.lt_plan_b(T, n, k, int(f32), nn, bt, LT_WS, 32))\n")
PREP_PINS_OLD = "    pins, notes = {}, []\n    for (proj, T), cfg in sorted(LT_PIN.items()):\n"
PREP_PINS_NEW = "    pins, notes, wts = {}, [], {}\n    for (proj, T), cfg in sorted(LT_PIN.items()):\n"
PREP_RUN_OLD = "                mod.lt_run(pid, ai, w, x, o, 0.0, ws)\n"
PREP_RUN_NEW = "                mod.lt_run(pid, ai, wl, x, o, 0.0, ws)\n"
PREP_PIN_OLD = "                pins[(proj, T)] = (pid, ai, n, k, f32)\n"
PREP_PIN_NEW = "                pins[(proj, T)] = (pid, ai, n, k, f32, wl if nn else None)\n"
PREP_NOTE_OLD = ("            notes.append(f\"{proj}/{T}: lt{ai}{'' if ai < cnt else ' (built)'} rel {e64:.1e}, vs today {et:.1e} -> \"\n")
PREP_NOTE_NEW = ("            notes.append(f\"{proj}/{T}: lt{ai}{'' if ai < cnt else ' (built)'}{' NN b%d' % bt if nn else ''} rel {e64:.1e}, \"\n"
                "                         f\"vs today {et:.1e} -> \"\n")

RUN_ALIGN_OLD = ("    ok = (p is not None and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == p[3] and x.is_contiguous()\n"
                 "          and x.data_ptr() % 256 == 0)\n")
RUN_ALIGN_NEW = ("    ok = (p is not None and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == p[3] and x.is_contiguous()\n"
                 "          and x.data_ptr() % (16 if p[5] is not None else 256) == 0)\n")
RUN_W_OLD = "        st[\"mod\"].lt_run(p[0], p[1], _dg1_weight(state, proj), x, out, 0.0, st[\"ws\"])\n"
RUN_W_NEW = ("        st[\"mod\"].lt_run(p[0], p[1], p[5] if p[5] is not None else _dg1_weight(state, proj), x, out, 0.0, st[\"ws\"])\n")

PIN_TAIL_OLD = "('qkv', 256): (66, 134, 35, 1, 0, 0, 2, 0, 9)}\nLT_WS = 32 << 20\n"


def apply(base, a):
    t = base
    # C++ (inside LT_CPP; each anchor is unique in the file)
    for old, new, what in ((CPP_HEAD_OLD, CPP_HEAD_NEW, "lt_plan head"), (CPP_TA_OLD, CPP_TA_NEW, "lt_plan TRANSA"),
                           (CPP_LA_OLD, CPP_LA_NEW, "lt_plan layout A"), (CPP_LC_OLD, CPP_LC_NEW, "lt_plan layout C"),
                           (CPP_TAIL_OLD, CPP_TAIL_NEW, "lt_plan tail")):
        t = _rep(t, old, new, what)
    # Python
    for old, new, what in ((BUILD_OLD, BUILD_NEW, "_lt_build functions"), (PREP_PINS_OLD, PREP_PINS_NEW, "prepare pins init"),
                           (PREP_W_OLD, PREP_W_NEW, "prepare plan"), (PREP_RUN_OLD, PREP_RUN_NEW, "prepare self-test run"),
                           (PREP_PIN_OLD, PREP_PIN_NEW, "prepare pin tuple"), (PREP_NOTE_OLD, PREP_NOTE_NEW, "prepare note"),
                           (RUN_ALIGN_OLD, RUN_ALIGN_NEW, "_lt alignment"), (RUN_W_OLD, RUN_W_NEW, "_lt weight")):
        t = _rep(t, old, new, what)
    pins, layout = [], {}
    if int(a.t256):
        pins.append(f"('unpooler', 256): (66, 20, 35, 1, 0, 0, {int(a.custom256)}, 0, {int(a.cluster)})")
        layout[("unpooler", 256)] = (1, int(a.batch))
    if int(a.t128):
        pins.append(f"('unpooler', 128): (66, 18, 35, 1, 0, 0, {int(a.custom128)}, 0, {int(a.cluster)})")
        layout[("unpooler", 128)] = (1, int(a.batch))
    new_tail = ("('qkv', 256): (66, 134, 35, 1, 0, 0, 2, 0, 9)" + "".join(", " + p for p in pins) + "}\n"
                f"ULP_LAYOUT = {layout!r}   # (proj, T) -> (nn, batch); tools/ulp_patches.py\n"
                "LT_WS = 32 << 20\n")
    t = _rep(t, PIN_TAIL_OLD, new_tail, "LT_PIN tail")
    return t


def gate(t):
    ast.parse(t)
    n = len(t.encode())
    ok = t.isascii() and n < LIMIT
    print(f"[ulp_patches] gates: ast ok, ascii {t.isascii()}, {n} B ({'OK' if n < LIMIT else 'OVER'} the {LIMIT} B cap by "
          f"{abs(LIMIT - n)} B)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="kernels/lab/k3_rider3_1.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t256", default="1")
    ap.add_argument("--t128", default="0")
    ap.add_argument("--batch", default="2")
    ap.add_argument("--custom256", default="7")
    ap.add_argument("--custom128", default="3")
    ap.add_argument("--cluster", default="5")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    base = (ROOT / a.base).read_text(encoding="utf-8")
    t = apply(base, a)
    out = ROOT / a.out
    out.write_text(t, encoding="utf-8", newline="\n")
    print(f"[ulp_patches] {a.base} -> {a.out}: {len(t.encode())} B (+{len(t.encode()) - len(base.encode())} B)")
    ok = gate(t)
    if a.check:
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

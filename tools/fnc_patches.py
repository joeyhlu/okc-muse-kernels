"""fnc_patches.py (lever FNC, R-fnc-2): split-ticket merges without the SM-wide fence tax.

R-fnc-1 (box_out/fnc0/run.json): with a co-resident CTA streaming 2x36K bulk per SM, fence.sc.gpu (= __threadfence)
p50 2048 ns, fence.release.gpu 608, fence.acquire.gpu below the 32 ns timer (SASS: CCTL.IVALL only). The non-MoE
threadFenceReduction tickets use __threadfence on both sides:
  writer: partial stores; __threadfence (every thread, or tid 0 after a CTA barrier); barrier; tid 0 relaxed atomicAdd
  merger: atomicAdd result says "last"; barrier; __threadfence; read the partials
Macros FNC_WF() (writer) / FNC_MF() (merger) replace those fences in VA1 / K1 (K1_CUDA_SRC), gemv (R1B), P1S (P1_CUDA_SRC),
P1M / P1K (P1M_CUDA_ADD), PRT (G2_CUDA_SRC, built together with P1) and VFA:
  FNC_MODE 0  ship (both __threadfence)
  FNC_MODE 1  merger fence.acquire.gpu                    (-DFNC_ACQ)
  FNC_MODE 2  merger fence.acquire.gpu + writer fence.release.gpu   (-DFNC_ACQ -DFNC_REL)
PTX memory-model argument (notes/study/fnc.md): the writer half of the ship's fence.sc that the protocol uses is its
release part (stores -> fence -> CTA barrier -> strong atomic = release pattern, cumulative through the barrier), the
merger half is its acquire part (strong atomic read of the ticket -> CTA barrier -> fence = acquire pattern). fence.sc =
fence.acq_rel + a global order of sc fences that a two-party ticket never uses, so release-only / acquire-only keep the
synchronizes-with edge. The merger's later stores (zeroing, counter reset) are read only by the next launch (kernel boundary).
Lab file: every patched module is compiled 3 times (_fnc_li) behind a proxy that dispatches on FNC_MODE at call time,
so rev_ab3 can switch it per variant ({"set": {"FNC_MODE": 1}}). Setup/query functions (occupancy / info / carveout) run on all 3.
--clean=<mode>: no proxy, the macros fixed to that mode by #define (candidate file).

  python -I tools/fnc_patches.py --base=kernels/muse_decode_layer/k3_gc_21cg_s1.py --out=kernels/lab/k3_fnc_1.py [--clean=1|2]
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)

MACROS = r"""#ifndef FNC_MF
#if defined(FNC_ACQ)
#define FNC_MF() asm volatile("fence.acquire.gpu;" ::: "memory")
#else
#define FNC_MF() __threadfence()
#endif
#if defined(FNC_REL)
#define FNC_WF() asm volatile("fence.release.gpu;" ::: "memory")
#else
#define FNC_WF() __threadfence()
#endif
#endif
"""

# (source variable, old, new): each old must occur exactly once in the file
SITES = [
    ("K1_CUDA_SRC", "    __threadfence();\n    VA_CSYNC;\n    if (tid == 0) flag[0]", "    FNC_WF();\n    VA_CSYNC;\n    if (tid == 0) flag[0]"),
    ("K1_CUDA_SRC", "      __threadfence();\n      const int r = tid >> 3, u = tid & 7, qh", "      FNC_MF();\n      const int r = tid >> 3, u = tid & 7, qh"),
    ("K1_CUDA_SRC", "    __threadfence();                                  // this thread's partial stores visible device-wide",
     "    FNC_WF();                                         // this thread's partial stores visible device-wide"),
    ("K1_CUDA_SRC", "      __threadfence();\n      K1P(7);", "      FNC_MF();\n      K1P(7);"),
    ("R1B_CUDA_SRC", "  __threadfence();\n  __syncthreads();\n  const int tile =", "  FNC_WF();\n  __syncthreads();\n  const int tile ="),
    ("R1B_CUDA_SRC", "  if (!s_last) return;\n  __threadfence();\n  for (int i = tid; i < rows_cta * MT;",
     "  if (!s_last) return;\n  FNC_MF();\n  for (int i = tid; i < rows_cta * MT;"),
    ("P1_CUDA_SRC", "  __threadfence();\n  __syncthreads();\n  if (tid == 0) s_last = (atomicAdd(A.cnt + rt, 1)",
     "  FNC_WF();\n  __syncthreads();\n  if (tid == 0) s_last = (atomicAdd(A.cnt + rt, 1)"),
    ("P1_CUDA_SRC", "  if (!s_last) return;\n  __threadfence();\n  if (mine) {", "  if (!s_last) return;\n  FNC_MF();\n  if (mine) {"),
    ("G2_CUDA_SRC", "  __threadfence();\n  __syncthreads();\n  if (tid == 0) s_last = (atomicAdd(G.tcnt",
     "  FNC_WF();\n  __syncthreads();\n  if (tid == 0) s_last = (atomicAdd(G.tcnt"),
    ("G2_CUDA_SRC", "  if (!s_last) return;\n  __threadfence();\n  if (tid == 0) atomicExch(G.tcnt, 0);",
     "  if (!s_last) return;\n  FNC_MF();\n  if (tid == 0) atomicExch(G.tcnt, 0);"),
    ("P1M_CUDA_ADD", "  __threadfence();\n  __syncthreads();\n  if (tid == 0) s_last = (atomicAdd(A.cnt + blockIdx.x",
     "  FNC_WF();\n  __syncthreads();\n  if (tid == 0) s_last = (atomicAdd(A.cnt + blockIdx.x"),
    ("P1M_CUDA_ADD", "  if (!s_last) return;\n  __threadfence();\n  for (int idx = tid; idx < nout;",
     "  if (!s_last) return;\n  FNC_MF();\n  for (int idx = tid; idx < nout;"),
    ("P1M_CUDA_ADD", "      __threadfence();\n      __syncwarp();\n      int last = 0;", "      FNC_WF();\n      __syncwarp();\n      int last = 0;"),
    ("P1M_CUDA_ADD", "      if (last) {\n        __threadfence();\n        const float* base = Kp.ws",
     "      if (last) {\n        FNC_MF();\n        const float* base = Kp.ws"),
    ("VFA_CUDA_SRC", "      __threadfence();\n      const int mine = k1 - k0;", "      FNC_WF();\n      const int mine = k1 - k0;"),
    ("VFA_CUDA_SRC", "zero what was read\n      __threadfence();", "zero what was read\n      FNC_MF();"),
]
SRC_VARS = ["K1_CUDA_SRC", "R1B_CUDA_SRC", "P1_CUDA_SRC", "G2_CUDA_SRC", "P1M_CUDA_ADD", "VFA_CUDA_SRC"]
# builder call sites (load_inline -> _fnc_li); each anchor must be unique
BUILDERS = ["    mod = load_inline(\n        name=f\"k1_v7_{tag}\",", "    mod = load_inline(\n        name=f\"r1b_gemv_{tag}\",",
            "    _P1_MOD = load_inline(\n        name=f\"p1s_mod_{tag}\",", "    mod = load_inline(name=f\"k3_g2_{tag}\",",
            "    _VFA[\"mod\"] = load_inline(\n        name=f\"vfa_{tag}\","]

LAB_HELPER = r'''

# ---- FNC (tools/fnc_patches.py): ticket-merge fence variants, rev_ab3-selectable --------------------------------------
FNC_MODE = 0   # 0 ship __threadfence both sides | 1 merger fence.acquire.gpu | 2 + writer fence.release.gpu
_FNC_FLAGS = ([], ["-DFNC_ACQ"], ["-DFNC_ACQ", "-DFNC_REL"])


class _FncSw:
    """Proxy over the 3 builds of one extension; dispatches on FNC_MODE at call time."""

    def __init__(self, mods):
        self._fnc_mods = mods

    def __getattr__(self, n):
        if n.startswith("__") or n == "_fnc_mods":
            raise AttributeError(n)
        if any(s in n for s in ("occupancy", "info", "carveout")):
            fs = [getattr(m, n) for m in self._fnc_mods]

            def _all(*a, **k):
                rs = [f(*a, **k) for f in fs]
                return rs[FNC_MODE]
            return _all
        return getattr(self._fnc_mods[FNC_MODE], n)


def _fnc_li(**kw):
    from torch.utils.cpp_extension import load_inline as _li
    mods = []
    for i, fl in enumerate(_FNC_FLAGS):
        k = dict(kw)
        if i:
            k["name"] = kw["name"] + f"_fnc{i}"
            k["extra_cuda_cflags"] = list(kw.get("extra_cuda_cflags", [])) + fl
        mods.append(_li(**k))
    return _FncSw(mods)
'''


def main():
    base = ROOT / ARGS.get("base", "kernels/muse_decode_layer/k3_gc_21cg_s1.py")
    out = ROOT / ARGS["out"]
    clean = ARGS.get("clean")
    assert not out.exists(), f"{out} exists (never overwrite)"
    t = base.read_text(encoding="utf-8")
    for var, old, new in SITES:
        assert t.count(old) == 1, f"site not unique / missing in {var}: {old[:60]!r}"
        t = t.replace(old, new, 1)
    head = MACROS
    if clean is not None:
        head = {"0": "", "1": "#define FNC_ACQ 1\n", "2": "#define FNC_ACQ 1\n#define FNC_REL 1\n"}[clean] + MACROS
    for var in SRC_VARS:
        for q in ('r"""', "r'''"):
            a = f"\n{var} = {q}"
            if t.count(a) == 1:
                t = t.replace(a, a + head, 1)
                break
        else:
            raise SystemExit(f"source anchor {var} missing")
    if clean is None:
        for b in BUILDERS:
            assert t.count(b) == 1, f"builder anchor not unique: {b[:50]!r}"
            t = t.replace(b, b.replace("load_inline(", "_fnc_li(", 1), 1)
        i = t.index("\nimport torch\n")
        j = t.index("\n\n", t.index("from torch import nn"))
        t = t[:j] + LAB_HELPER + t[j:]
    assert t.count("__threadfence();\n      VA_CSYNC") == 0
    lines = t.split("\n")
    mode = f"clean {clean}" if clean is not None else "lab FNC_MODE 0/1/2"
    lines[1] = f"# nonce {out.stem.replace('_', '-')} ({base.stem} + FNC ticket fences, {mode}; tools/fnc_patches.py)"
    t = "\n".join(lines)
    assert t.isascii() and len(t.encode()) < 1_000_000, len(t.encode())
    compile(t, str(out), "exec")
    out.write_text(t, encoding="utf-8", newline="\n")
    print(f"[fnc_patches] {base.name} -> {out.relative_to(ROOT).as_posix()} ({mode}), {len(SITES)} sites, {len(t.encode())} B")


if __name__ == "__main__":
    main()

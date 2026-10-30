"""fncm_patches.py (moe lane, R-fnc-m): cheaper fences in the VML split-share resolve (notes/study/fnc.md, assumptions A20).

R-fnc-1: under a co-resident 3 x 36 KB bulk ring, fence.acq_rel / fence.sc ~3.7 us, fence.release ~1.8, fence.acquire ~0,
atom.acquire ~0.35. VM_FNCM 1 changes both split resolves (inline A segment end and the VM_AD m32e_resolve / m32e_finish):
  writers' arrival count  atom.acq_rel.gpu  -> atom.release.gpu (their red.add partials need release ordering only)
  last arriver            all-thread fence.acq_rel.gpu after the barrier -> fence.acquire.gpu by tid 0 right after it
                          sees the final count, BEFORE the bar.sync that publishes wlast (strict PTX acquire pattern:
                          strong read + fence.acquire in one thread's program order; the other 127 threads' __ldcg
                          read-backs follow through bar.sync, as ship's acquire does today).
VM_FNCM 0 = ship source. Variants bf2f / w72f = bf2 / w72 + -DVM_FNCM=1.

  python tools/fncm_patches.py arms [--base=kernels/muse_decode_layer/k3_gc_21cg_s1.py]
      -> kernels/lab/k3_fncm_1.py (base + bf2f / w72f; routes bf2f / w72f; ship bf2 / w72 built by name)
  python tools/fncm_patches.py rider --out=kernels/...py [--base=...] [--var=bf2f] [--bvar=w72f]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ae_patches as AE  # noqa: E402  (_sub, defs, nonce: anchored helpers)

BASE = ROOT / "kernels/muse_decode_layer/k3_gc_21cg_s1.py"
_sub, defs, nonce = AE._sub, AE.defs, AE.nonce
SCX = 'scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]'
HELP = "__device__ __forceinline__ int atom_acqrel_add(int* p, int v) {\nint o;\n"   # unique: the VML helper
W_AD = "if (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, pd.scnt) + pd.scnt == 16);\n"
W_IN = "if (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, s1 - s0) + (s1 - s0) == 16);\n"
F_AD = "float d[N / 2];\nfence_acq_rel();\n"
F_IN = "SEGMK(const long long t_rb = smk.on ? gtime() : 0;)\nfence_acq_rel();\n"


def fncm(s):
    s = _sub(s, "#ifndef VM_RT\n#define VM_RT 0\n#endif\n",
             "#ifndef VM_RT\n#define VM_RT 0\n#endif\n#ifndef VM_FNCM\n#define VM_FNCM 0   // R-fnc-m: release count + "
             "acquire-only last arriver (tools/fncm_patches.py)\n#endif\n")
    s = _sub(s, HELP, "__device__ __forceinline__ int atom_rel_add(int* p, int v) {\nint o;\n"
             "asm volatile(\"atom.release.gpu.global.add.s32 %0, [%1], %2;\\n\" : \"=r\"(o) : \"l\"(p), \"r\"(v) : \"memory\");\n"
             "return o;\n}\n__device__ __forceinline__ void fence_acquire() { asm volatile(\"fence.acquire.gpu;\\n\" ::: \"memory\"); }\n" + HELP)
    for w, cnt in ((W_AD, "pd.scnt"), (W_IN, "(s1 - s0)")):
        s = _sub(s, w, "#if VM_FNCM\nif (wt == 0) { const bool l_ = (atom_rel_add(fc, " + cnt.strip("()") + ") + " + cnt
                 + " == 16); if (l_) fence_acquire(); T.wlast[q] = l_; }\n#else\n" + w + "#endif\n")
    s = _sub(s, F_AD, "float d[N / 2];\n#if !VM_FNCM\nfence_acq_rel();\n#endif\n")
    s = _sub(s, F_IN, "SEGMK(const long long t_rb = smk.on ? gtime() : 0;)\n#if !VM_FNCM\nfence_acq_rel();\n#endif\n")
    s = defs(s, {"bf2f": ("bf2", ("-DVM_FNCM=1",)), "w72f": ("w72", ("-DVM_FNCM=1",))})
    s = _sub(s, SCX, SCX.replace('"w72")', '"w72", "bf2f", "w72f")'))
    return s


def main():
    mode = sys.argv[1]
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith(f"--{k}=")), d)  # noqa: E731
    base_p = ROOT / arg("base", str(BASE.relative_to(ROOT)))
    s0 = base_p.read_text(encoding="utf-8")
    var, bvar = arg("var", "bf2f"), arg("bvar", "w72f")
    if mode == "arms":
        AE.build("k3_fncm_1.py", nonce(AE.route(fncm(s0), var, bvar),
                                       f"k3-fncm-1 ({base_p.name} + VM_FNCM release count / acquire-only resolve: bf2f / w72f; tools/fncm_patches.py)"))
    elif mode == "rider":
        out_p = arg("out")
        assert out_p, "--out=kernels/...py required"
        Path(ROOT / out_p).write_text(AE.route(fncm(s0), var, bvar), encoding="utf-8", newline="\n")
        print(f"wrote {out_p} (VM_FNCM rider on {base_p.name}: var {var} bvar {bvar})")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()

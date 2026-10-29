"""lat32_patches.py: moe-lat32 (VML latency taxes at T >= 32, in-layer) lab files on k3_sfx_2 (= k3_gc_19 + SFX2 = k3_gc_20).

  python tools/lat32_patches.py marks   -> kernels/lab/k3_lat32_m.py   VM_MK2 v2 marks (tools/gen_k3_v32.py) at buckets
                                           32 / 128 / 256 (bf2 + w72) + LSEG segment marks in the v2 row (below)
  python tools/lat32_patches.py rider --base=kernels/muse_decode_layer/k3_gc_20.py --out=<file>   E1 rider on any base
  python tools/lat32_patches.py arms    -> kernels/lab/k3_lat32_0.py   control (generator defaults, VM_NWF 0)
                                           kernels/lab/k3_lat32_1.py   E1 NOWFENCE: writer fence.proxy.async removed (VM_NWF 1)
                                           kernels/lab/k3_lat32_1m.py  = _1 + marks
The SFX2 ordering fix is in the base (k3_sfx_2), so every file here carries it.

v2 row (A.marks + (G0 + 1 + c) * 32), gen_k3_v32 cols 0..10, LSEG cols (q0 wp0 lane0, per seg<PB, N>, PB 0 -> 11.., PB 1 -> 17..):
  11 + 6 PB: first stage ns (t1 - t0)   12 + 6 PB: steady stages ns   13 + 6 PB: drain (wg_wait 0) ns   14 + 6 PB: end work ns
  15 + 6 PB: segments                   16 + 6 PB: steady stages (s1 - s0 - 1)
  23 split atomic ns  24 split read-back ns  25 A epilogue ns  26 B epilogue / bred ns  27 inline fence + release ns
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gen_k3_v32 as V32  # noqa: E402

BASE = ROOT / "kernels/lab/k3_sfx_2.py"
_sub0 = V32._sub


def _sub(s, old, new, n=1):
    if n == 2 and old.startswith("mbar_wait(bars + 8 * (2 * NS + xs)") and s.count(old) == 3:
        n = 3   # stage_m has the SFX and the non-SFX copy of the x wait (one compiled): mark both
    return _sub0(s, old, new, n)


V32._sub = _sub


def lseg(s):
    """Turn the VM_MKW == 64 SEGMK marks on under VM_MK2 too, written into the v2 row's free columns."""
    s = _sub(s, "#if VM_MKW == 64\n#define SEGMK(...) __VA_ARGS__\n#else\n#define SEGMK(...)\n#endif\n",
             "#if VM_MKW == 64 || VM_MK2\n#define SEGMK(...) __VA_ARGS__\n#else\n#define SEGMK(...)\n#endif\n"
             "#if VM_MK2\n#define LSROW(c) (A.marks + (size_t)(A.G0 + 1 + (c)) * MKW)\n#define LSC(k) ((k) == 52 ? 31 : (k) == 53 ? 25 : "
             "(k) == 54 ? 26 : (k) == 55 ? 24 : (k) == 62 ? 23 : (k) == 63 ? 27 : 30)\n#else\n#define LSROW(c) (A.marks + (size_t)(c) * MKW)\n"
             "#define LSC(k) (k)\n#endif\n")
    s = _sub(s, "#if VM_MKW == 64\nstruct SegMk { long long t0, t1, t2, t3; bool on; };",
             "#if VM_MKW == 64 || VM_MK2\nstruct SegMk { long long t0, t1, t2, t3; bool on; };")
    s = _sub(s, "long long* mk = A.marks + (size_t)blockIdx.x * MKW;\n"
             "mk[56] += m.t1 - m.t0; mk[57] += m.t2 - m.t1; mk[58] += m.t3 - m.t2; mk[59] += t4 - m.t3; mk[60] += 1; mk[61] += s1 - s0 - 1;",
             "#if VM_MK2\nlong long* mk = LSROW(blockIdx.x) + 11 + 6 * PB - 56;\n#else\nlong long* mk = A.marks + (size_t)blockIdx.x * MKW;\n#endif\n"
             "mk[56] += m.t1 - m.t0; mk[57] += m.t2 - m.t1; mk[58] += m.t3 - m.t2; mk[59] += t4 - m.t3; mk[60] += 1; mk[61] += s1 - s0 - 1;")
    s = _sub(s, "__device__ __forceinline__ void segmk_done(const Args& A, SegMk& m, int s0, int s1) {",
             "template <bool PB>\n__device__ __forceinline__ void segmk_done(const Args& A, SegMk& m, int s0, int s1) {")
    s = _sub(s, "segmk_done(A, smk, s0, s1);", "segmk_done<PB>(A, smk, s0, s1);", n=s.count("segmk_done(A, smk, s0, s1);"))
    for k in (52, 53, 54, 55, 62, 63):
        old = f"A.marks[(size_t)blockIdx.x * MKW + {k}]"
        s = s.replace(old, f"LSROW(blockIdx.x)[LSC({k})]")
    return s


def nwf(s):
    """E1 NOWFENCE (VM_NWF): drop the writer-side fence.proxy.async.global in front of the phase-A doneA release (deferred
    pend_flush and the inline path). The reader keeps its fence (prod_x: acquire doneA -> fence.proxy.async.global -> bulk
    copy of xB), so every generic-write -> bulk-read chain still has exactly one proxy fence (notes/assumptions.md A25)."""
    s = _sub(s, "#ifndef VM_DEFER\n#define VM_DEFER 0\n#endif\n",
             "#ifndef VM_DEFER\n#define VM_DEFER 0\n#endif\n#ifndef VM_NWF\n#define VM_NWF 0   // lat32 E1: no writer proxy fence\n#endif\n")
    s = _sub(s, "if (pd.i >= 0) {\nfence_proxy_async();\nbar_sync(1 + q, 128);",
             "if (pd.i >= 0) {\nif constexpr (!VM_NWF) fence_proxy_async();\nbar_sync(1 + q, 128);")
    return s


def cfl(s):
    """E0 (a) D2: VM_CFLAGS (compile-time A.flags in the VML region; 0 = runtime as shipped). Lab / SASS only."""
    k = "#ifndef VM_MKW\n#define VM_MKW 32\n#endif\n"
    a = s.index(k)
    b = s.index("std::vector<int64_t> vml_cfg()")
    r = s[a:b]
    n = r.count("A.flags & ") + r.count("(a).flags & 512")
    r = r.replace("A.flags & ", "VFL(A) & ").replace("(a).flags & 512", "VFL(a) & 512")
    r = r.replace(k, k + "#ifndef VM_CFLAGS\n#define VM_CFLAGS 0\n#endif\n#define VFL(a) (VM_CFLAGS ? VM_CFLAGS : (a).flags)\n", 1)
    print(f"cfl: {n} flag tests")
    return s[:a] + r + s[b:]


def ldcg(s):
    """R-pdl-0 fix: predecessor outputs (topk ids, x) in VML's route() / gathers via __ldcg (coherent: ptxas keeps them below
    griddepcontrol.wait; LDG.E.CONSTANT from __ldg was hoisted above ACQBULK in bf2 = the R15 failure)."""
    a = s.index("#ifndef VM_MKW@n#define VM_MKW 32@n#endif@n".replace("@n", chr(10)))
    b = s.index("std::vector<int64_t> vml_cfg()")
    r = s[a:b]
    n1, n2 = r.count("__ldg(A.ids"), r.count("__ldg(reinterpret_cast<const uint4*>(A.x")
    assert n1 == 2 and n2 == 2, (n1, n2)
    r = r.replace("__ldg(A.ids", "__ldcg(A.ids").replace("__ldg(reinterpret_cast<const uint4*>(A.x", "__ldcg(reinterpret_cast<const uint4*>(A.x")
    return s[:a] + r + s[b:]


def pdl8(s, buckets):
    import ast
    i = s.index(chr(10) + "VML_FLAGS = {") + 1
    j = s.index(chr(10), i)
    d = ast.literal_eval(s[i + len("VML_FLAGS = "):j])
    for bk in buckets:
        d[bk] = d[bk] | 256
    return s[:i] + "VML_FLAGS = " + repr(d) + s[j:]


def hook_all(s):
    n = s.count("((T.jC[j] + 7) >> 3) - 1] += s1 - s0;")
    s = _sub(s, "((T.jC[j] + 7) >> 3) - 1] += s1 - s0;", "min((T.jC[j] + 7) >> 3, 8) - 1] += s1 - s0;", n=n)   # w72 N 65..72
    s = _sub(s, "marks=(VML_MARKS if rep == 32 else None)", "marks=(VML_MARKS if rep in (32, 128, 256) else None)")
    s = _sub(s, '"w72": (', '"w72": ("-DVM_MK2=1", ')
    return s


def set_defs(s, extra):
    for v in ("bf2", "w72"):
        s = _sub(s, f'"{v}": (', f'"{v}": (' + "".join(f'"{d}", ' for d in extra))
    return s


def nonce(s, tag):
    lines = s.split("\n", 2)
    assert lines[1].startswith("# nonce "), lines[1][:60]
    lines[1] = f"# nonce {tag}"
    return "\n".join(lines)


def build(name, s):
    p = ROOT / "kernels/lab" / name
    p.write_text(s, encoding="utf-8", newline="\n")
    print(f"wrote {p.relative_to(ROOT)} ({len(s)} B)")


def main():
    mode = sys.argv[1]
    base_p = next((ROOT / a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--base=")), BASE)
    out_p = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--out=")), None)
    s0 = base_p.read_text(encoding="utf-8")
    base = nwf(s0)
    if mode == "marks":
        s = hook_all(lseg(V32.marks(base, v2=True)))
        build("k3_lat32_m.py", nonce(s, "k3-lat32-m (k3_sfx_2 + MK2 v2 marks + LSEG at buckets 32/128/256; lab)"))
    elif mode == "arms":
        build("k3_lat32_0.py", nonce(base, "k3-lat32-0 (k3_sfx_2 via tools/lat32_patches.py, VM_NWF 0: control)"))
        build("k3_lat32_1.py", nonce(set_defs(base, ["-DVM_NWF=1"]), "k3-lat32-1 (k3_sfx_2 + E1 NOWFENCE VM_NWF 1)"))
        s = set_defs(hook_all(lseg(V32.marks(base, v2=True))), ["-DVM_NWF=1"])
        build("k3_lat32_1m.py", nonce(s, "k3-lat32-1m (k3_lat32_1 + MK2 v2 marks + LSEG; lab)"))
    elif mode == "rider":   # adoption: VM_NWF 1 on bf2 + w72 of --base (asserted unique anchors), written to --out
        assert out_p, "--out=kernels/...py required"
        s = set_defs(base, ["-DVM_NWF=1"])
        Path(ROOT / out_p).write_text(s, encoding="utf-8", newline=chr(10))
        print(f"wrote {out_p} (rider VM_NWF on {base_p.name})")
    elif mode in ("ldcg", "pdl8", "nwfpdl8", "ldcgnwf"):   # R-lat32-4 riders on --base: ldcg only / + bit 8 at buckets / + VM_NWF
        assert out_p, "--out required"
        bks = [int(x) for x in next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--buckets=")), "32").split(",")]
        s = ldcg(s0)
        if mode in ("pdl8", "nwfpdl8"):
            s = pdl8(s, bks)
        if mode in ("nwfpdl8", "ldcgnwf"):
            s = set_defs(nwf(s), ["-DVM_NWF=1"])
        Path(ROOT / out_p).write_text(s, encoding="utf-8", newline=chr(10))
        print(f"wrote {out_p} ({mode} buckets {bks} on {base_p.name})")
    elif mode == "cfl":
        build("k3_lat32_c.py", nonce(cfl(base), "k3-lat32-c (k3_sfx_2 + VM_CFLAGS hook; SASS audit only)"))
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()

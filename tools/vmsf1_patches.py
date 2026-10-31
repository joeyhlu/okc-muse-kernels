"""vmsf1_patches.py (R-vmsf1): alygg77 TINY's two latency mechanisms (ext_muse_tiny_mechanism.md s3 P1) in the VMS T=16 hi build,
behind -DVMS_F1 bits (dual-tile <= 16-token jobs kept; no expert splitting). Arithmetic per unit unchanged.

  python -I tools/vmsf1_patches.py --base=<in.py> --out=<out.py> [--bits=3]      ship: the hi build (9..16 tokens) gets -DVMS_F1=<bits>
  python -I tools/vmsf1_patches.py --base=<in.py> --out=<out.py> --lab=1,2,3,5   lab: hi build unchanged + one extra hi build per bit set,
                                                                                  VMS_F1_ON = 0 (ship hi) or a bit set; prepare self-test
  bit 1 FAST1: the epilogue warp (warp NW) counts the routes itself (match_any + smem atomics, the route filter), runs route's weighted
        t16 split when every expert has <= 16 tokens (one job per touched expert, in expert order), writes S.pl, inits full[] and issues
        the CTA's first NS units from the touched-expert mask before route() / __syncthreads; route skips its split then (the same
        predicate on the same counts: one source of S.pl either way), compute thread 0 issues only units S.f1n..NS-1.
  bit 2 WS: compute warps release a ring slot with an mbarrier arrive (count NW) instead of the CTA-wide bar.sync + thread-0 refill;
        the epilogue warp waits that barrier at the top of its unit loop and issues unit q + NS (alygg77 ws_helper order).
  bit 4 (lab, exclusive with 2): last-arriving compute warp issues the refill (smem counter, no bar.sync).
Every edit is anchored once inside VMS_CUDA_SRC and is under #if VMS_F1: the VMS_F1=0 build preprocesses to the base text.
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[vmsf1_patches] FAIL: {msg}")


DEF_A = "#ifndef VMS_ER\n#define VMS_ER 0\n#endif\n"
DEF_N = DEF_A + "#ifndef VMS_F1\n#define VMS_F1 0\n#endif\n"
SM_A = "#if VMS_LPAD > 0\nunsigned char lpad[VMS_LPAD];\n#endif\n};\n"
SM_N = "#if VMS_LPAD > 0\nunsigned char lpad[VMS_LPAD];\n#endif\n#if VMS_F1\nuint64_t f1e[NS];\nint f1c[NE], f1n, f1k[NS];\n#endif\n};\n"
IN_A = "#if __CUDA_ARCH__ >= 900\n    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], 1);\n#else\n"
IN_N = "#if __CUDA_ARCH__ >= 900\n#if !(VMS_F1 & 1)\n    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], 1);\n#endif\n#else\n"
RO_A = "    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], NW); mbar_init(&S.pempty[i], 1); }\n"
RO_N = RO_A + "#if VMS_F1 & 2\nfor (int i = 0; i < NS; ++i) mbar_init(&S.f1e[i], NW);\n#endif\n#if VMS_F1 & 4\nfor (int i = 0; i < NS; ++i) S.f1k[i] = 0;\n#endif\n"
W8_A = "if (warp == 0) {\nconst int m8 = p.m * 8;\n"
W8_N = r'''#if VMS_F1 & 1
if (warp == NW) {
const int m8 = p.m * 8;
int fv[4];
#pragma unroll
for (int r = 0; r < 4; ++r) { const int i = r * 32 + lane; fv[r] = i < m8 ? ((__ldcg(p.ids + i) - p.lo) & 255) : 255; if (fv[r] >= NE) fv[r] = -1; }
S.f1c[lane] = 0;
if (lane == 0) {
for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], 1);
asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__syncwarp();
#pragma unroll
for (int r = 0; r < 4; ++r) {
const unsigned mm = __match_any_sync(0xffffffffu, fv[r]);
if (fv[r] >= 0 && lane == __ffs(mm) - 1) atomicAdd(&S.f1c[fv[r]], __popc(mm));
}
__syncwarp();
const int tot = S.f1c[lane];
int n = 0;
if (!__any_sync(0xffffffffu, tot > 16)) {
const unsigned msk = __ballot_sync(0xffffffffu, tot > 0);
const int nJ = __popc(msk), cj = __shfl_sync(0xffffffffu, tot, lane < nJ ? (int)__fns(msk, 0, lane + 1) : 0);
const int w0 = lane < nJ ? (cj > 8 ? VMS_WT2 : VMS_WT1) : 0, w1 = 0;
int s0 = w0, s1 = w1;
#pragma unroll
for (int d = 1; d < 32; d <<= 1) {
const int u0 = __shfl_up_sync(0xffffffffu, s0, d), u1 = __shfl_up_sync(0xffffffffu, s1, d);
if (lane >= d) { s0 += u0; s1 += u1; }
}
s1 += __shfl_sync(0xffffffffu, s0, 31);
const int WJ = __shfl_sync(0xffffffffu, s1, 31);
const int pa0 = t16_split(s0, s1, w0, w1, nJ, RBA, RBA * WJ * c / G), pa1 = t16_split(s0, s1, w0, w1, nJ, RBA, RBA * WJ * (c + 1) / G);
const int T0 = (RBA + RBB) * WJ * c / G, T1 = (RBA + RBB) * WJ * (c + 1) / G;
const int pb0 = t16_split(s0, s1, w0, w1, nJ, RBB, T0 - t16_wpre(s0, s1, w0, w1, RBA, pa0));
const int pb1 = t16_split(s0, s1, w0, w1, nJ, RBB, T1 - t16_wpre(s0, s1, w0, w1, RBA, pa1));
const int fA = pa1 - pa0;
n = max(0, min(NS, fA + pb1 - pb0));
if (lane == 0) {
S.pl[0] = pa0; S.pl[1] = pa1; S.pl[2] = pb0; S.pl[3] = pb1;
for (int q = 0; q < n; ++q) {
const bool B = q >= fA;
const int v = B ? pb0 + (q - fA) : pa0 + q, e = (int)__fns(msk, 0, (B ? v >> 8 : v >> 9) + 1);
issue(B ? p.wB + ((size_t)e * RBB + (v & 255)) * UB : p.wA + ((size_t)e * RBA + (v & 511)) * UB, ring + q * UB, &S.full[q], 0);
}
}
}
if (lane == 0) S.f1n = n;
}
#endif
''' + W8_A
RT_A = "#if VMS_T16 && !(VMS_T16X & 4)\n{\nconst int nJ = __shfl_sync(0xffffffffu, jinc, 31);\n"
RT_N = "#if VMS_T16 && !(VMS_T16X & 4)\n#if VMS_F1 & 1\nif (__any_sync(0xffffffffu, tot > 16))\n#endif\n{\nconst int nJ = __shfl_sync(0xffffffffu, jinc, 31);\n"
I0_A = "#if !((VMS_FX & 4) && __CUDA_ARCH__ >= 900)\nfor (int q = 0; q < NS && q < N; ++q) {\n"
I0_N = ("#if !((VMS_FX & 4) && __CUDA_ARCH__ >= 900)\n#if VMS_F1 & 1\nfor (int q = S.f1n; q < NS && q < N; ++q) {\n#else\n"
        "for (int q = 0; q < NS && q < N; ++q) {\n#endif\n")
ER_A = ("for (int cc = 0; cc < 8; ++cc) wv[cc] = *reinterpret_cast<const uint4*>(slot + (warp * 8 + cc) * 512 + lane * 16);\n"
        "cta_sync_compute();\nif (q + NS < N) {\n#if __CUDA_ARCH__ >= 900\nif (tid == 0)\n#endif\n"
        "        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);\n      }\n#endif\n")
ER_W = "#if VMS_F1 & 2\n__syncwarp();\nif (lane == 0) mbar_arrive(&S.f1e[sl]);\n"
ER_L = ("#elif VMS_F1 & 4\n__syncwarp();\nif (lane == 0) {\n__threadfence_block();\n"
        "if (atomicAdd(&S.f1k[sl], 1) % NW == NW - 1 && q + NS < N) { __threadfence_block(); "
        "issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], 0); }\n}\n")
HP_A = "float* ssc = p.ss + (S.gen & 1) * MAXP;\nfor (int q = 0; q < N; ++q) {\nconst int bq = q & (NRB - 1);\n"
HP_N = HP_A + ("#if VMS_F1 & 2\nif (q + NS < N) {\nmbar_wait(&S.f1e[q % NS], (uint32_t)((q / NS) & 1));\n"
               "if (lane == 0) issue(unit_src(p, S, q + NS, nA, a0, b0), ring + (q % NS) * UB, &S.full[q % NS], 0);\n}\n#endif\n")


def er_new(lab_l):
    head, rest = ER_A.split("cta_sync_compute();\n", 1)
    return head + ER_W + (ER_L if lab_l else "") + "#else\ncta_sync_compute();\n" + rest[:-len("#endif\n")] + "#endif\n#endif\n"


def cuda_edits(cu, lab_l):
    for a, n, w in ((DEF_A, DEF_N, "VMS_ER default"), (SM_A, SM_N, "Smem end"), (IN_A, IN_N, "full[] init"), (RO_A, RO_N, "RO init"),
                    (W8_A, W8_N, "route warp head"), (RT_A, RT_N, "route t16 split"), (I0_A, I0_N, "first issue"),
                    (ER_A, er_new(lab_l), "ER refill"), (HP_A, HP_N, "epilogue loop head")):
        check(cu.count(a) == 1, f"CUDA anchor {w}: {cu.count(a)}x inside VMS_CUDA_SRC")
        cu = cu.replace(a, n)
    return cu


FL_A = '"-DVMS_ER=1"] + flags[4:]\n'
LAB_STATE_A = '_VMS = {"mod": None, "G": None, "mod_hi": None}\nVMS_HI_ON = True\n'
LAB_BUILD_A = "    ths.append(threading.Thread(target=_hi_build, daemon=True))\n    ths[-1].start()\n"
LAB_BUILD_N = LAB_BUILD_A + '''    for b in VMS_F1_BUILDS:
        def _f1_build(b=b):
            try:
                _VMS["mod_f1"][b] = load_inline(name=f"k3_vmsf1_{b}_{tag_hi}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC,
                                                functions=["run", "attrs"], extra_cuda_cflags=fl_hi + [f"-DVMS_F1={b}"])
            except Exception as ex:
                print(f"[vms] F1 build {b} failed {ex!r}"[:300], flush=True)
        ths.append(threading.Thread(target=_f1_build, daemon=True))
        ths[-1].start()
'''
LAB_CALL_A = '    md = vs["mod_hi"] if (VMS_HI_ON and m > 8 and vs.get("mod_hi") is not None) else vs["mod"]\n'
LAB_CALL_N = LAB_CALL_A + '    if VMS_F1_ON and md is not None and md is vs.get("mod_hi"):\n        md = _VMS["mod_f1"].get(VMS_F1_ON, md)\n'
LAB_ST_A = '                print("[vms] T16 module (9-16 tokens) self-test ok", flush=True)\n'
LAB_ST_N = LAB_ST_A + "            _vms_f1_check(st, w, dev)\n"
LAB_FN_A = "def _vms_prepare(st, w, dev):\n"
LAB_FN_N = '''def _vms_f1_check(st, w, dev):
    global VMS_F1_ON
    for b in sorted(_VMS["mod_f1"]):
        ok = True
        for bs, seed, dist in ((12, 61, "skewed"), (16, 62, "normal"), (16, 63, "hot8"), (16, 64, "skewed"), (9, 65, "skewed"),
                               (16, 66, "normal"), (16, 67, "skewed"), (16, 68, "hot8")):
            d = _k2a36_synth_inputs(bs, seed, dev, "hot8") if dist == "hot8" else _k2a_self_test_inputs(bs, seed, dev, dist)
            if dist == "hot8":
                d["topk_ids"] = torch.where(d["topk_ids"] < K2A_E_LOC, d["topk_ids"], torch.full_like(d["topk_ids"], -1))
            ref, x = _k2a_ref_fp32(d, w), d["hidden_states"].contiguous()
            try:
                VMS_F1_ON = 0
                o0 = _vms_call(st, x, d["topk_ids"], d["topk_weights"]).clone()
                VMS_F1_ON = b
                o = _vms_call(st, x, d["topk_ids"], d["topk_weights"])
                torch.cuda.synchronize()
                tot, row = _k2a_rel_err(o, ref)
                e0, errf = _k2a_rel_err(o, o0)[0], int(st["vms"]["ctr"][2])
                ok = tot < 1.0e-2 and row < 5.0e-2 and e0 < 1.0e-3 and not errf and bool(torch.isfinite(o.float()).all())
                msg = f"vs FP32 {tot:.3e} row {row:.3e} vs hi {e0:.1e} err={errf}"
            except Exception as ex:
                ok, msg = False, f"raised {ex!r}"[:200]
            finally:
                VMS_F1_ON = 0
            print(f"[vms] F1 build {b} self-test bs={bs} {dist}: {msg} -> {'ok' if ok else 'FAIL: dropped'}", flush=True)
            if not ok:
                _VMS["mod_f1"].pop(b, None)
                break
''' + LAB_FN_A


def apply(s, bits, lab):
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] += (f" + R-vmsf1 lab F1 builds {','.join(map(str, lab))} [tools/vmsf1_patches.py]" if lab
                 else f" + R-vmsf1 VMS T16 FAST1/WS bits {bits} [tools/vmsf1_patches.py]")
    s = NL.join(lines)
    i0 = s.index('VMS_CUDA_SRC = r"""')
    i1 = s.index('"""', i0 + 20)
    s = s[:i0] + cuda_edits(s[i0:i1], bool(lab) or bool(bits & 4)) + s[i1:]
    if lab:
        check(s.count(LAB_STATE_A) == 1, "_VMS state")
        s = s.replace(LAB_STATE_A, '_VMS = {"mod": None, "G": None, "mod_hi": None, "mod_f1": {}}\nVMS_HI_ON = True\n'
                      f"VMS_F1_ON = 0\nVMS_F1_BUILDS = {tuple(lab)!r}\n")
        for a, n, w in ((LAB_BUILD_A, LAB_BUILD_N, "hi build thread"), (LAB_CALL_A, LAB_CALL_N, "_vms_call md"),
                        (LAB_ST_A, LAB_ST_N, "hi self-test ok"), (LAB_FN_A, LAB_FN_N, "_vms_prepare def")):
            check(s.count(a) == 1, f"anchor {w}: {s.count(a)}x")
            s = s.replace(a, n)
    else:
        check(s.count(FL_A) == 1, "fl_hi flags")
        s = s.replace(FL_A, f'"-DVMS_ER=1"] + flags[4:] + ["-DVMS_F1={bits}"]\n')
    return s


def main():
    base, out = ROOT / ARGS["base"], ROOT / ARGS["out"]
    lab = tuple(int(b) for b in ARGS["lab"].split(",")) if "lab" in ARGS else ()
    bits = int(ARGS.get("bits", "3"))
    check(1 <= bits <= 7 and not (bits & 2 and bits & 4), "--bits: 1..7, bits 2 and 4 exclusive")
    check(all(1 <= b <= 7 and not (b & 2 and b & 4) for b in lab), "--lab bit sets")
    src = base.read_bytes().decode("ascii")
    new = apply(src, bits, lab)
    ast.parse(new)
    check(new.isascii(), "non-ASCII")
    n = len(new.encode())
    check(lab or n < LIMIT, f"size {n} >= {LIMIT}")
    out.write_bytes(new.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"vmsf1_patches: {base.name} -> {out.name} ({n:,} B, +{n - len(src):,} B; {'lab ' + str(lab) if lab else 'ship bits ' + str(bits)}"
          f"{'' if lab else f', {LIMIT - n:,} B under the cap'})")


if __name__ == "__main__":
    main()

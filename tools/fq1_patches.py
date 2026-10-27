"""fq1_patches.py: FQ1 = K2A consumer dequant shifts and scale words moved from the INT (ALU) pipe to the FMA pipe,
bitwise-identical (notes/study/k2a_consumer_audit.md lever 1). Scoped text patches against K2A33_KERNEL_SRC of
kernels/muse_decode_layer/k3_v34.py; each `old` must occur the stated number of times inside that source string.

Why: per 18,432 B stage each consumer warp issues ~290 instructions, ~164 of them on the ALU pipe (LOP3 / SHF / PRMT / LEA;
16 lanes per SMSP = 0.5 warp-instr/clk), which needs ~69-73 % of the ALU pipe at the 4.45 TB/s stream rate at the measured
1.64-1.74 GHz load clocks. The FMA pipe (IMAD / HMUL2, 1 warp-instr/clk) is ~20 % busy.
  * deq4 (BITPOS3): the funnel rotates and shifts become mul.lo / mul.hi by powers of two (IMAD / IMAD.HI, FMA pipe):
      o1: rotl(w, 6) & M == (w << 6) & M         (the 6 bits the rotate brings around land on mask-0 bits 0-5)
      o2: rotl(w, 12) == (w << 12) | (w >> 20)    (one LOP3 does (a | b) & M)
      o3: (w >> 6) via mul.hi(w, 2^26), (w << 2) via mul.lo(w, 4)
    Every output word is bit-identical to the rotate form.
  * scale words: byte_perm(...) * 16 + 0x75807580 (LEA, ALU) -> mad.lo.u32 (IMAD, FMA pipe), identical value.
Compile-time switch K3_FMA_DEQ (1 = FMA-pipe forms; the layer defines it from FQ1_FMA_DEQ at build time).

    python -I tools/fq1_patches.py [k3_v34.py] [out.py]
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
K2A33 = ('K2A33_KERNEL_SRC = r"""', '"""')

DEQ_OLD = ("__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // BITPOS3 layout\n"
           "  o[0] = w & 0x81C081C0u;\n"
           "  o[1] = __funnelshift_l(w, w, 6) & 0x81C081C0u;\n"
           "  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;\n"
           "  o[3] = ((w >> 6) & 0x01000100u) | ((w << 2) & 0x80C080C0u);\n"
           "}\n")
DEQ_NEW = ("#ifndef K3_FMA_DEQ\n#define K3_FMA_DEQ 0\n#endif\n"
           "// FQ1: shifts by powers of two as mul.lo / mul.hi (IMAD / IMAD.HI on the FMA pipe instead of SHF / LEA on the ALU pipe)\n"
           "__device__ __forceinline__ uint32_t fq_mul(uint32_t a, uint32_t b) {\n"
           "  uint32_t d; asm(\"mul.lo.u32 %0, %1, %2;\" : \"=r\"(d) : \"r\"(a), \"r\"(b)); return d;\n}\n"
           "__device__ __forceinline__ uint32_t fq_mulhi(uint32_t a, uint32_t b) {\n"
           "  uint32_t d; asm(\"mul.hi.u32 %0, %1, %2;\" : \"=r\"(d) : \"r\"(a), \"r\"(b)); return d;\n}\n"
           "__device__ __forceinline__ uint32_t fq_scl(uint32_t b) {   // b * 16 + 0x75807580\n"
           "#if K3_FMA_DEQ\n"
           "  uint32_t d; asm(\"mad.lo.u32 %0, %1, 16, %2;\" : \"=r\"(d) : \"r\"(b), \"r\"(0x75807580u)); return d;\n"
           "#else\n"
           "  return b * 16u + 0x75807580u;\n"
           "#endif\n}\n"
           "__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // BITPOS3 layout\n"
           "#if K3_FMA_DEQ\n"
           "  o[0] = w & 0x81C081C0u;\n"
           "  o[1] = fq_mul(w, 64u) & 0x81C081C0u;                                   // == rotl(w, 6) & M\n"
           "  o[2] = (fq_mul(w, 4096u) | fq_mulhi(w, 4096u)) & 0x81C081C0u;           // rotl(w, 12)\n"
           "  o[3] = (fq_mulhi(w, 1u << 26) & 0x01000100u) | (fq_mul(w, 4u) & 0x80C080C0u);   // (w >> 6), (w << 2)\n"
           "#else\n"
           "  o[0] = w & 0x81C081C0u;\n"
           "  o[1] = __funnelshift_l(w, w, 6) & 0x81C081C0u;\n"
           "  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;\n"
           "  o[3] = ((w >> 6) & 0x01000100u) | ((w << 2) & 0x80C080C0u);\n"
           "#endif\n"
           "}\n")
SCALE_RE = re.compile(r"(__byte_perm\(sw2, 0u, 0x4[01]4[01]u \+ 0x0202u \* par\)) \* 16u \+ 0x75807580u")


def patch_src(src):
    """K2A33 source text -> FQ1 source text (K3_FMA_DEQ defaults to 0 = shipped instructions)."""
    if src.count(DEQ_OLD) != 1:
        raise RuntimeError(f"fq1: deq4 anchor found {src.count(DEQ_OLD)} times")
    src = src.replace(DEQ_OLD, DEQ_NEW)
    src, n = SCALE_RE.subn(r"fq_scl(\1)", src)
    if n != 8:
        raise RuntimeError(f"fq1: {n} scale-word sites (expected 8)")
    return src


HOST_ADD = '''
FQ1_FMA_DEQ = True    # FQ1 (tools/fq1_patches.py): K2A consumer shifts / scale words on the FMA pipe (bitwise forms); read at call time
FQ1_BUILD = True      # build the FQ1 K2A module in prepare (lab A/B arms toggle FQ1_FMA_DEQ)


def _fq1_build_mod(variants):
    """K2A33 alone with K3_FMA_DEQ 1: own load_inline, same variants / cases as the combined module."""
    from torch.utils.cpp_extension import load_inline
    cases, attrs = _k2a36_variant_cases(variants)
    src = ("#define K3_FMA_DEQ 1" + chr(10) + K2A33_KERNEL_SRC.replace("@CASES3@", cases).replace("@ATTRS3@", attrs)
           .replace("@WGSPECS@", _k2a33_wg_specs()))
    tag = hashlib.sha1((K2A33_CPP_SRC + src).encode()).hexdigest()[:10]
    return load_inline(name=f"k2a_fq1_{tag}", cpp_sources=K2A33_CPP_SRC, cuda_sources=src, functions=["run3", "attrs3v"],
                       extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills",
                                          "-gencode=arch=compute_90a,code=sm_90a"])


def _fq1_mod(st):
    m = st.get("mod_fq1")
    return m if (FQ1_FMA_DEQ and m is not None) else st["mod"]


def _fq1_prepare(st, w, dev):
    """FQ1 module + self-test through the layer's K2A entry point (v3.3 / v3.6 / v3.7 routes) on K2A33_SELF_TEST_CASES:
    occupancy equal to the shipped module's, outputs within the shipped module's own run-to-run difference (split items use
    FP32 reds) and no worse against the FP32 reference. Any failure or exception keeps the shipped module."""
    global FQ1_FMA_DEQ
    st["mod_fq1"] = None
    if not FQ1_BUILD:
        return
    keep, t0 = FQ1_FMA_DEQ, time.time()
    try:
        m = _fq1_build_mod(st["variants"])
        for i in range(len(st["variants"])):
            a, b = list(st["mod"].attrs3v(i)), list(m.attrs3v(i))
            if (a[3], a[7]) != (b[3], b[7]) or b[1] or b[5]:
                raise RuntimeError(f"variant {i}: occupancy / spill {b} vs shipped {a}")
        st["mod_fq1"] = m
        for bs, seed, dist in K2A33_SELF_TEST_CASES:
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            ref = _k2a_ref_fp32(d, w)
            outs = []
            for on in (False, False, True):
                FQ1_FMA_DEQ = on
                _k2a33_reset(st)
                st["err"].zero_()
                outs.append(_v11_k2a33_call(d["hidden_states"], d["topk_ids"], d["topk_weights"], st).clone())
                torch.cuda.synchronize()
                if int(st["err"][0]) or not bool(torch.isfinite(outs[-1].float()).all()):
                    raise RuntimeError(f"bs={bs}: error flag or non-finite output (fq1 {on})")
            self_d = _k2a_rel_err(outs[1], outs[0])[0]
            fq_d = _k2a_rel_err(outs[2], outs[0])[0]
            e_ship, e_fq = _k2a_rel_err(outs[0], ref)[0], _k2a_rel_err(outs[2], ref)[0]
            ok = fq_d <= max(4 * self_d, 1e-5) and e_fq <= e_ship * 1.02 + 1e-6
            print(f"[fq1] self-test bs={bs} {dist}: fq1 vs ship {fq_d:.2e} (ship vs ship {self_d:.2e}), vs ref {e_fq:.3e} / {e_ship:.3e}"
                  f" -> {'ok' if ok else 'FAIL'}", flush=True)
            if not ok:
                raise RuntimeError(f"bs={bs} mismatch")
        print(f"[fq1] FMA-pipe dequant module ready in {time.time() - t0:.1f}s (FQ1_FMA_DEQ={keep})", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[fq1] disabled, shipped K2A module kept: {ex!r}", flush=True)
        st["mod_fq1"] = None
    finally:
        FQ1_FMA_DEQ = keep
        _k2a33_reset(st)
'''
CALL_OLD = '    st["mod"].run3(vid, mask, x, ids, wts, st["ws13"]'
CALL_NEW = '    _fq1_mod(st).run3(vid, mask, x, ids, wts, st["ws13"]'


def _sub_scoped(text, start, end, old, new, what):
    a = text.index(start)
    b = text.index(end, a + len(start))
    n = text[a:b].count(old)
    if n != 1:
        raise RuntimeError(f"fq1: {what}: anchor found {n} times")
    i = text.index(old, a)
    return text[:i] + new + text[i + len(old):]


def apply(text):
    """Patch a layer file text: K2A33_KERNEL_SRC (K3_FMA_DEQ machinery, default 0 = shipped instructions), the FQ1 module,
    its prepare self-test after the v3.6 / v3.7 prepares, and the three K2A33-family call sites."""
    a = text.index(K2A33[0])
    b = text.index(K2A33[1], a + len(K2A33[0]))
    text = text[:a] + patch_src(text[a:b]) + text[b:]
    anchor = "_K2A33_OWN_BUILD = _k2a33_build_ext\n"
    if text.count(anchor) != 1:
        raise RuntimeError("fq1: build anchor")
    text = text.replace(anchor, anchor + HOST_ADD)
    for fn in ("def _k2a33_call(", "def _k2a36_call(", "def _k2a37_call("):
        text = _sub_scoped(text, fn, "\ndef ", CALL_OLD, CALL_NEW, fn)
    prep = "    _v14_k2a36_prepare(st, w, dev)\n    _v17_k2a37_prepare(st, w, dev)\n"
    if text.count(prep) != 1:
        raise RuntimeError("fq1: prepare anchor")
    text = text.replace(prep, prep + "    _fq1_prepare(st, w, dev)   # FQ1: after the v3.6 / v3.7 routes exist, so it tests vids 3 / 5 too\n")
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kernels/muse_decode_layer/k3_v34.py"
    out = apply(src.read_text(encoding="utf-8"))
    if len(sys.argv) > 2:
        pathlib.Path(sys.argv[2]).write_text(out, encoding="utf-8", newline="\n")
    print(f"fq1: patched {src.name} ({len(out)} chars)")

"""gr1_patches.py: GR1 = the front reorder of notes/study/glue_fusion_independent.md s3, as lab patches on k3_v37 (R-glue-1).

At T <= 8, B = 1, sliding (the KS1 shapes) the front becomes, in stream order (one stream, PDL edges):
  1. P1 qkv_A: rows 0-1279 (q, k, v) of the shipped 12-bit qkv pack (meta tensors as views); first kernel, early trigger;
     every CTA's exit is counted into F_A (flags[0]).
  2. KS1: arithmetic unchanged; trigger at entry; the group-last CTA loads the gate rows only after a bounded acquire spin on
     F_gate (flags[1] == CTAs of qkv_B) and releases F_attn (flags[2]) after writing the o-input.
  3. P1 qkv_B: rows 1280-2303 (gate) of the same pack (meta views at tile 80, its own split-K slots / counters), no PDL wait
     (its input is the layer input; waiting on KS1 would deadlock), streams once F_A >= theta x n_A, counts its exits into
     F_gate, early trigger.
  4. SU1-o, the shipped staged o-projection (128 CTAs x 131 KB, raw BF16): staging starts once F_A (or F_gate) reaches its
     target; the mma waits on F_attn (KS1 is not its PDL primary) after the usual griddepcontrol.wait.
  5. SBB1 (attention boundary) zeroes flags[0..2] right after its PDL wait: every reader and writer of them is done then.
Flags live at _ks1_counters()[1000:1008] (+3 = err). Every spin is bounded (2^20 polls), sets err and never traps (R9).
GR1_ENABLE = False (the default) is k3_v37's behaviour: the new kernel paths sit behind arguments whose defaults are the
shipped values (P1 Args fields default to "off", KS1 / SU1 GR1 bits unset, SBB1 reset pointer null).

    python -I tools/gr1_patches.py [in.py] [out.py]
    (default kernels/muse_decode_layer/k3_v37.py -> kernels/lab/k3_v37_gr1.py)
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kernels/muse_decode_layer/k3_v37.py"
DST = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "kernels/lab/k3_v37_gr1.py"
NONCE = ("# nonce k3-v37-gr1-lab (k3_v37 + GR1 front reorder: qkv_A -> KS1 || qkv_B -> SU1-o, tools/gr1_patches.py; "
         "lab A/B file for R-glue-1, GR1_ENABLE off by default)")


def sub(text, old, new, start=None, end=None, count=1):
    """Replace old by new inside text[start-marker : end-marker] (whole text if None), asserting the hit count."""
    i = 0 if start is None else text.index(start)
    j = len(text) if end is None else text.index(end, i)
    seg = text[i:j]
    n = seg.count(old)
    assert n == count, f"anchor {old[:70]!r}: {n} hits (want {count}) in region {str(start)[:40]!r}"
    return text[:i] + seg.replace(old, new) + text[j:]


def region(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


P1_REG = ('P1_CUDA_SRC = r"""', '\n"""\n\n_P1_MOD = None')
P1_CPP_REG = ('P1_CPP_SRC = r"""', '"""\n\nP1_CUDA_SRC')
KS1_REG = ('KS1_CUDA_SRC = r"""', '\nKS1_ENABLE = True')
KS1_KER = ('ks1_kernel(const __grid_constant__ Params p) {', 'static void launch(const Params& P, bool pdl, cudaStream_t st) {')
SU1_REG = ('SU1_CUDA_SRC = r"""', '_SU1_FLAG = {"k3trig": False}')
SU1_CPP_REG = ('SU1_CPP_SRC = r"""', 'SU1_CUDA_SRC = r"""')
SBB_CU_REG = ("SBB1_CU = r'''", "SBB1_HOST = r'''")
SBB_HOST_REG = ("SBB1_HOST = r'''", "SBB1_DECL = ")

SPIN = r'''__device__ __forceinline__ void gr_spin_ge(const int* f, int target, int* err) {   // GR1: bounded acquire spin, err on timeout
  for (int it = 0; it < (1 << 20); ++it) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(f) : "memory");
    if (v >= target) return;
    __nanosleep(64);
  }
  if (err != nullptr) atomicOr(err, 1);
}
'''


def patch_p1(t):
    t = sub(t, "  int trig;                                            // V20: 1 = griddepcontrol.launch_dependents at entry "
               "(one-wave grids)\n};\n",
            "  int trig;                                            // V20: 1 = griddepcontrol.launch_dependents at entry "
            "(one-wave grids)\n"
            "  // GR1 (tools/gr1_patches.py, lab): nowait = 1 skips the PDL wait (the input is a layer input); spin_cnt holds the\n"
            "  // CTA until *spin_cnt >= spin_target; done_cnt counts this CTA's exit after its outputs; err receives timeouts.\n"
            "  int nowait = 0;\n  int* spin_cnt = nullptr;\n  int spin_target = 0;\n  int* done_cnt = nullptr;\n  int* err = nullptr;\n};\n",
            *P1_REG)
    t = sub(t, "__device__ __forceinline__ uint32_t rotr(uint32_t x, int r) { return __funnelshift_r(x, x, r); }\n",
            SPIN + "__device__ __forceinline__ void gr_done(const Args& A) {   // GR1: count this CTA's exit (outputs fenced)\n"
            "  if (A.done_cnt == nullptr) return;\n  __syncthreads();\n"
            "  if (threadIdx.x == 0) { __threadfence(); atomicAdd(A.done_cnt, 1); }\n}\n"
            "__device__ __forceinline__ uint32_t rotr(uint32_t x, int r) { return __funnelshift_r(x, x, r); }\n", *P1_REG)
    wait = '  asm volatile("griddepcontrol.wait;" ::: "memory");\n'
    t = sub(t, "  // 3. PDL: x (and the output) belong to the predecessor kernel. A no-op without programmatic serialization.\n" + wait,
            "  // 3. PDL: x (and the output) belong to the predecessor kernel. A no-op without programmatic serialization.\n"
            "  if (A.nowait == 0) {\n" + wait + "  }\n"
            "  if (A.spin_cnt != nullptr) {   // GR1: hold this CTA's stream until the flag reaches its target\n"
            "    if (threadIdx.x == 0) gr_spin_ge(A.spin_cnt, A.spin_target, A.err);\n    __syncthreads();\n  }\n", *P1_REG)
    assert region(t, *P1_REG).count(wait) == 1 and region(t, *P1_REG).count("griddepcontrol.wait") == 1, "G2 stream_h needs 1 wait"
    t = sub(t, "  if (S == 1) return;\n", "  if (S == 1) { gr_done(A); return; }\n", *P1_REG)
    t = sub(t, "  if (!s_last) return;\n  __threadfence();\n  if (mine) {",
            "  if (!s_last) { gr_done(A); return; }\n  __threadfence();\n  if (mine) {", *P1_REG)
    t = sub(t, "  if (tid == 0) A.cnt[rt] = 0;\n}\n", "  if (tid == 0) A.cnt[rt] = 0;\n  gr_done(A);\n}\n", *P1_REG)
    # p1s_gemv_gr: p1s_gemv with a flags tensor and the GR1 fields (deterministic split-K only)
    p1 = region(t, *P1_REG)
    a, b = p1.index("void p1s_gemv("), p1.index("std::vector<int64_t> p1s_info(int64_t depth, int64_t mode, int64_t warps) {")
    body = p1[a:b]
    sig_old = ("void p1s_gemv(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,\n"
               "              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,\n"
               "              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split, int64_t depth, int64_t mode, int64_t pdl,\n"
               "              int64_t red_mode) {\n")
    sig_new = ("void p1s_gemv_gr(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,\n"
               "                 torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,\n"
               "                 torch::Tensor flags, int64_t M, int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split,\n"
               "                 int64_t depth, int64_t mode, int64_t pdl, int64_t gr, int64_t spin_idx, int64_t spin_target,\n"
               "                 int64_t done_idx) {\n"
               "  // GR1 (tools/gr1_patches.py): gr bit 0 = no PDL wait, bit 1 = spin on flags[spin_idx] >= spin_target, bit 2 = count\n"
               "  // exits into flags[done_idx]; flags[3] = err. Deterministic split-K only.\n"
               "  TORCH_CHECK(flags.is_cuda() && flags.scalar_type() == at::kInt && flags.is_contiguous() && flags.numel() >= 4 &&\n"
               "              spin_idx >= 0 && spin_idx < 3 && done_idx >= 0 && done_idx < 3, \"p1s_gr: flags int32 [>= 4], slots 0..2\");\n"
               "  const int64_t red_mode = 0;\n")
    assert body.count(sig_old) == 1
    body = body.replace(sig_old, sig_new)
    disp = "  const int w = (int)warps, S = (int)split, pd = (int)(pdl & 1);\n"
    assert body.count(disp) == 1
    body = body.replace(disp, "  int* const fp = flags.data_ptr<int>();\n  A.nowait = (gr & 1) ? 1 : 0;\n"
                              "  A.spin_cnt = (gr & 2) ? fp + spin_idx : nullptr;\n  A.spin_target = (int)spin_target;\n"
                              "  A.done_cnt = (gr & 4) ? fp + done_idx : nullptr;\n  A.err = fp + 3;\n" + disp)
    t = sub(t, "std::vector<int64_t> p1s_info(int64_t depth, int64_t mode, int64_t warps) {",
            body + "std::vector<int64_t> p1s_info(int64_t depth, int64_t mode, int64_t warps) {", *P1_REG)
    t = sub(t, "std::vector<int64_t> p1s_info(int64_t depth, int64_t mode, int64_t warps);\n",
            "void p1s_gemv_gr(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,\n"
            "                 torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,\n"
            "                 torch::Tensor flags, int64_t M, int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split,\n"
            "                 int64_t depth, int64_t mode, int64_t pdl, int64_t gr, int64_t spin_idx, int64_t spin_target,\n"
            "                 int64_t done_idx);\n"
            "std::vector<int64_t> p1s_info(int64_t depth, int64_t mode, int64_t warps);\n", *P1_CPP_REG)
    t = sub(t, '    cpp, cuda, fns = P1_CPP_SRC, P1_CUDA_SRC, ["p1s_gemv", "p1s_info", "p1_unpack", "p1s_carveout"]\n',
            '    cpp, cuda, fns = P1_CPP_SRC, P1_CUDA_SRC, ["p1s_gemv", "p1s_info", "p1_unpack", "p1s_carveout", "p1s_gemv_gr"]\n')
    return t


def patch_ks1(t):
    t = sub(t, '__device__ __forceinline__ void fence_acqrel() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }\n',
            '__device__ __forceinline__ void fence_acqrel() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }\n' + SPIN, *KS1_REG)
    k = region(t, *KS1_KER)
    assert k.count("p.trig") == 2, "KS1: trig is read in exactly two places"
    t = sub(t, "  if (p.trig == 2) pdl_trigger();\n  KS1P(0);\n",
            "  if ((p.trig & 3) == 2) pdl_trigger();\n  KS1P(0);\n"
            "  const bool gr = (p.trig & 0x100) != 0;   // GR1 (tools/gr1_patches.py): gate rows from a later kernel (qkv_B)\n"
            "  const int gtarget = p.trig >> 12;         // GR1: qkv_B CTAs to wait for (F_gate)\n"
            "  int* const grf = p.counters + 1000;      // GR1 flags: +1 F_gate, +2 F_attn, +3 err\n", *KS1_KER)
    t = sub(t, "  if (p.trig == 1) pdl_trigger();\n", "  if ((p.trig & 3) == 1) pdl_trigger();\n", *KS1_KER)
    k = region(t, *KS1_KER)
    a = k.index("  uint32_t graw[DPT / 2];\n  {\n")
    b = k.index("  }\n  // old K / V rows -> fp16 tiles\n", a)
    inner = k[a + len("  uint32_t graw[DPT / 2];\n  {\n"):b]
    assert inner.startswith("    const __nv_bfloat16* gp = p.qkv") and inner.count("graw[0]") == 2
    t = sub(t, "  uint32_t graw[DPT / 2];\n  {\n    const __nv_bfloat16* gp", "  uint32_t graw[DPT / 2];\n  if (!gr) {\n    const __nv_bfloat16* gp",
            *KS1_KER)
    t = sub(t, "  if (misc[0] == 0) return;\n",
            "  if (misc[0] == 0) return;\n"
            "  if (gr) {   // GR1: qkv_B, a later kernel in the stream, writes the gate rows: wait for all of its CTAs, then load\n"
            "    if (tid == 0) gr_spin_ge(grf + 1, gtarget, grf + 3);\n    __syncthreads();\n" + inner + "  }\n", *KS1_KER)
    t = sub(t, "  if (tid == 0) p.counters[grp] = 0;                             // self-reset (every split of the group has arrived)\n",
            "  if (tid == 0) p.counters[grp] = 0;                             // self-reset (every split of the group has arrived)\n"
            "  if (gr) {   // GR1: this group's o-input rows are written: release them to SU1-o (F_attn)\n"
            "    __syncthreads();\n    if (tid == 0) { __threadfence(); atomicAdd(grf + 2, 1); }\n  }\n", *KS1_KER)
    return t


def patch_su1(t):
    t = sub(t, "__device__ __forceinline__ uint32_t sm_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n",
            "__device__ __forceinline__ uint32_t sm_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n" + SPIN,
            *SU1_REG)
    su = region(t, *SU1_REG)
    a = su.index("  const bool late = (trig >> 4) & 1;")
    line = su[a:su.index("\n", a) + 1]
    t = sub(t, line, line + "  const bool gr = ((trig >> 8) & 1) != 0;   // GR1 (tools/gr1_patches.py, su1_run_gr): flags at err - 3\n"
                            "  int* const grf = gr ? (err - 3) : nullptr;\n", *SU1_REG)
    t = sub(t, "    if (!late) {\n#pragma unroll 1\n      for (int o = 0; o < BYTES; o += CH)",
            "    if (!late) {\n      if (gr) gr_spin_ge(grf + ((trig >> 9) & 1), (trig >> 16) & 0x7fff, err);   // GR1: stage after F_A / F_gate\n"
            "#pragma unroll 1\n      for (int o = 0; o < BYTES; o += CH)", *SU1_REG)
    wl = '  asm volatile("griddepcontrol.wait;\\n" ::: "memory");   // x (the predecessor\'s output) is visible from here on\n'
    t = sub(t, wl, wl + "  if (gr) {   // GR1: x = the o-input from KS1, which is not this kernel's PDL primary: wait for its release (F_attn)\n"
                        "    if (tid == 0) gr_spin_ge(grf + 2, (trig >> 10) & 63, err);\n    __syncthreads();\n  }\n", *SU1_REG)
    su = region(t, *SU1_REG)
    a, b = su.index("void su1_run(torch::Tensor wp"), su.index("std::vector<int64_t> su1_info(int64_t K, int64_t R, int64_t T) {")
    body = su[a:b]
    for old, new in (("void su1_run(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor err,",
                      "void su1_run_gr(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor flags,"),
                     ("err.scalar_type() == at::kInt", "flags.scalar_type() == at::kInt && flags.is_contiguous() && flags.numel() >= 4"),
                     ("  int* ep = err.data_ptr<int>();\n", "  int* ep = flags.data_ptr<int>() + 3;   // GR1: the kernel finds the flags at err - 3\n")):
        assert body.count(old) == 1, old
        body = body.replace(old, new)
    t = sub(t, "std::vector<int64_t> su1_info(int64_t K, int64_t R, int64_t T) {",
            body + "std::vector<int64_t> su1_info(int64_t K, int64_t R, int64_t T) {", *SU1_REG)
    t = sub(t, "std::vector<int64_t> su1_info(int64_t K, int64_t R, int64_t T);\n",
            "void su1_run_gr(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor flags, int64_t T, int64_t K, int64_t R,\n"
            "                int64_t ldo, int64_t pdl, int64_t trig, torch::Tensor prof, int64_t smem);\n"
            "std::vector<int64_t> su1_info(int64_t K, int64_t R, int64_t T);\n", *SU1_CPP_REG)
    t = sub(t, 'functions=["su1_run", "su1_info"]', 'functions=["su1_run", "su1_info", "su1_run_gr"]')
    return t


def patch_sbb1(t):
    t = sub(t, "  float post_eps, pre_eps;\n  int gate_channel_wise;\n};",
            "  float post_eps, pre_eps;\n  int gate_channel_wise;\n"
            "  int* reset;                    // GR1 (tools/gr1_patches.py): flags zeroed after the PDL wait, or nullptr\n"
            "  int nreset;\n};", *SBB_CU_REG)
    t = sub(t, '  asm volatile("griddepcontrol.wait;" ::: "memory");\n  float br[8];',
            '  asm volatile("griddepcontrol.wait;" ::: "memory");\n'
            "  if (p.reset != nullptr && blockIdx.x == 0 && tid < p.nreset) p.reset[tid] = 0;   // GR1: all front-flag users are done\n"
            "  float br[8];", *SBB_CU_REG)
    t = sub(t, "              double pre_eps, int64_t input_scale) {",
            "              double pre_eps, int64_t input_scale, torch::Tensor reset) {", *SBB_HOST_REG)
    t = sub(t, "  p.post_eps = (float)post_eps; p.pre_eps = (float)pre_eps;\n",
            "  p.post_eps = (float)post_eps; p.pre_eps = (float)pre_eps;\n"
            "  p.reset = reset.numel() ? reset.data_ptr<int>() : nullptr; p.nreset = (int)reset.numel();\n", *SBB_HOST_REG)
    t = sub(t, "double post_eps, double pre_eps, int64_t input_scale);'",
            "double post_eps, double pre_eps, int64_t input_scale, torch::Tensor reset);'")
    t = sub(t, '    st["mod"].sbb1_run(branch.contiguous(), stream.contiguous(), out, ns, rt, pw, qw, al, bt, pe, qe, isc)\n',
            '    st["mod"].sbb1_run(branch.contiguous(), stream.contiguous(), out, ns, rt, pw, qw, al, bt, pe, qe, isc,\n'
            '                       _gr1_take_reset(st["e"]))\n')
    return t


PY_GR1 = r'''
# ============================================================================================ GR1 (tools/gr1_patches.py, lab)
GR1_ENABLE = False    # lab arms set True (read at call time); False = k3_v37's front
GR1_BUILD = True
GR1_REQUIRE = True    # lab file: prepare raises if a routed GR1 bucket fails its self-test (an A/B with dead arms is invalid)
GR1_ROUTE = {1: True, 8: True}                                         # q_len -> GR1 front on B = 1 sliding (the KS1 shapes)
GR1_GEO = {1: ((16, 2, 1), (16, 2, 1)), 8: ((4, 8, 1), (4, 8, 1))}     # q_len -> P1 (warps, split, depth) of qkv_A, qkv_B
GR1_THETA = {1: 1.0, 8: 1.0}                                           # qkv_B streams once this share of qkv_A's CTAs exited
GR1_WO = {1: ("A", 1.0), 8: ("A", 1.0)}                                # SU1-o staging starts at F_A ("A") / F_gate ("G") share
GR1_OFF = 1000                                                         # flags in _ks1_counters: +0 F_A +1 F_gate +2 F_attn +3 err
GR1_PREPARE_BY_S = 260.0
_GR1 = {"ok": set(), "reset": None, "seen": set()}
_GR1_QA, _GR1_QB = 80, 64                                              # 16-row tiles: q, k, v = 1280 rows; gate = 1024 rows


def _gr1_flags(dev):
    return _ks1_counters(dev)[GR1_OFF:GR1_OFF + 8]


def _gr1_take_reset(default):
    r = _GR1["reset"]
    _GR1["reset"] = None
    return default if r is None else r


def _gr1_run(qkv, x, data, state, cfg, geo=None, theta=None, wo=None, gate_extra=0):
    """GR1 front: P1 qkv_A -> KS1 || P1 qkv_B -> SU1-o. Returns (attn_output [T, 8192], qkv, o-input); arithmetic = shipped."""
    T, L = cfg["tokens"], cfg["q_len"]
    p1 = state["p1"]
    P, mod = p1["proj"]["qkv"], p1["mod"]
    ga, gb = geo if geo is not None else GR1_GEO[L]
    th = GR1_THETA[L] if theta is None else theta
    wsel, wfrac = GR1_WO[L] if wo is None else wo
    fl = _gr1_flags(x.device)
    na, nb = _GR1_QA * int(ga[1]), _GR1_QB * int(gb[1])
    groups = 2 * KS1_GEOM[L][1]
    K, ra = P["K"], 16 * _GR1_QA
    mod.p1s_gemv_gr(x, P["buf"], P["toff"], P["scale"], P["eoff"], P["esc"], P["fmt"], qkv, P["ws"], P["cnt"], fl,
                    T, ra, K, qkv.stride(0), int(ga[0]), int(ga[1]), int(ga[2]), 0, 2, 4, 0, 0, 0)
    o = _ks1_call(qkv, data, state["cos_sin"], trig=2 | 0x100 | ((nb + gate_extra) << 12))
    tgt = max(1, min(na, int(round(th * na))))
    mod.p1s_gemv_gr(x, P["buf"], P["toff"][_GR1_QA:], P["scale"][_GR1_QA:], P["eoff"][_GR1_QA:], P["esc"],
                    P["fmt"][_GR1_QA:], qkv[:, ra:], P["ws"][72 * ra:], P["cnt"][_GR1_QA:], fl, T, 16 * _GR1_QB, K,
                    qkv.stride(0), int(gb[0]), int(gb[1]), int(gb[2]), 0, 3, 7, 0, tgt, 1)
    su = state["su1"]
    ko, r, rows = SU1_GEOM["o"]
    out = torch.empty((T, rows), dtype=torch.bfloat16, device=x.device)
    sel = 0 if wsel == "A" else 1
    stgt = max(1, min(na if sel == 0 else nb, int(round(float(wfrac) * (na if sel == 0 else nb)))))
    trig = 2 | 0x100 | (sel << 9) | (groups << 10) | (stgt << 16)
    su["mod"].su1_run_gr(su["w"]["o"], o, out, fl, T, ko, r, HIDDEN, 1, trig, su["noprof"], int(SU1_SMEM.get("o", 0)))
    return out, qkv, o


def _gr1_front(data, state, cfg):
    if not GR1_ENABLE:
        return None
    T, L, B = cfg["tokens"], cfg["q_len"], cfg["batch"]
    if B != 1 or not GR1_ROUTE.get(L) or L not in _GR1["ok"]:
        return None
    rep = _k2a_bucket_of(T)
    sb = _SBB1["st"]
    if not (SBB1_ENABLE and SBB1_ROUTE.get(rep) and sb is not None and sb["ok"].get(rep)):
        return None   # SBB1 resets the flags: no SBB1, no GR1
    x = data["attn_input"]
    p1 = state.get("p1")
    if p1 is None or "qkv" not in p1["proj"] or not _v13_p1_ok(x, p1["proj"]["qkv"]["K"]):
        return None
    su = state.get("su1")
    if not (SU1_ENABLE and su is not None and "o" in su["w"] and T <= SU1_MAX_T["o"]):
        return None
    qkv = torch.empty((T, 16 * (_GR1_QA + _GR1_QB)), dtype=torch.bfloat16, device=x.device)
    if not _ks1_wanted(qkv, data, cfg, state):
        return None
    try:
        out = _gr1_run(qkv, x, data, state, cfg)[0]
    except Exception as ex:
        _GR1["ok"].clear()
        _k3_disable(state, "gr1_off", True, "GR1", ex)
        return None
    _GR1["reset"] = _gr1_flags(x.device)[:3]
    if L not in _GR1["seen"]:
        _GR1["seen"].add(L)
        print(f"[gr1] first call T={T}: qkv_A {GR1_GEO[L][0]} -> KS1 || qkv_B {GR1_GEO[L][1]} (theta {GR1_THETA[L]}) -> "
              f"SU1-o (staging {GR1_WO[L]}); flags reset by SBB1", flush=True)
    return out


def _gr1_case(L, seed, dev):
    d = _ks1_case(3000, L, seed, False, dev)
    g = torch.Generator(device=dev).manual_seed(seed + 77)
    d["attn_input"] = torch.randn((L, HIDDEN), generator=g, device=dev).to(torch.bfloat16)
    d["config"] = dict(d["config"], tokens=L)
    return d


def _gr1_rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _gr1_kv(d):
    rows = d["cache_loc"]
    return (d["k_cache"].view(-1, 128).view(torch.uint8)[rows].clone(), d["v_cache"].view(-1, 128).view(torch.uint8)[rows].clone())


def _gr1_selftest(state, L, dev):
    """GR1 vs the shipped kernels in the same order of operations (P1 qkv -> KS1 -> SU1-o): bitwise at the shipped P1 geometry,
    within P1's split-K envelope at the routed one; flag counts; repeatability; a forced timeout must set err, not hang."""
    T, fl = L, _gr1_flags(dev)
    groups = 2 * KS1_GEOM[L][1]
    su = state["su1"]
    ko, r, rows = SU1_GEOM["o"]
    seed = 9300 + L
    dr = _gr1_case(L, seed, dev)
    qkv_r = _v13_p1(state, "qkv", dr["attn_input"])
    if qkv_r is None:
        return False, ["shipped P1 qkv route unavailable"]
    o_r = _ks1_call(qkv_r, dr, state["cos_sin"])
    a_r = torch.empty((T, rows), dtype=torch.bfloat16, device=dev)
    su["mod"].su1_run(su["w"]["o"], o_r, a_r, su["err"], T, ko, r, HIDDEN, 1, 1, su["noprof"], 0)
    torch.cuda.synchronize()
    kv_r = _gr1_kv(dr)
    ship = tuple(state["p1"]["geo"][("qkv", p1_bucket(T))])
    ok, parts = True, []
    for name, geo in (("ship", (ship, ship)), ("route", tuple(tuple(g) for g in GR1_GEO[L]))):
        res = []
        for _ in range(2):
            dg = _gr1_case(L, seed, dev)
            qg = torch.empty((T, 16 * (_GR1_QA + _GR1_QB)), dtype=torch.bfloat16, device=dev)
            ag, _, og = _gr1_run(qg, dg["attn_input"], dg, state, dg["config"], geo=geo)
            torch.cuda.synchronize()
            fv = [int(v) for v in fl[:4].tolist()]
            fl[:4].zero_()
            res.append((ag, qg, og, _gr1_kv(dg), fv))
        ag, qg, og, kvg, fv = res[0]
        want = [_GR1_QA * geo[0][1], _GR1_QB * geo[1][1], groups, 0]
        rep = all(torch.equal(a, b) for a, b in zip(res[0][:3], res[1][:3]))
        flags = all(x[4] == want for x in res)
        fin = bool(torch.isfinite(ag.float()).all())
        if geo == (ship, ship):
            same = (torch.equal(qg, qkv_r) and torch.equal(og, o_r) and torch.equal(ag, a_r) and torch.equal(kvg[0], kv_r[0])
                    and torch.equal(kvg[1], kv_r[1]))
            good = same and rep and flags and fin
            parts.append(f"{name} {geo[0]}: bitwise {same} rep {rep} flags {fv} (want {want}) fin {fin}")
        else:
            eq, ea = _gr1_rel(qg, qkv_r), _gr1_rel(ag, a_r)
            ek = _gr1_rel(kvg[0].view(FP8).float(), kv_r[0].view(FP8).float())
            good = eq < 4e-3 and ea < 1e-2 and ek < 2e-2 and rep and flags and fin
            parts.append(f"{name} {geo}: qkv {eq:.1e} a {ea:.1e} k {ek:.1e} rep {rep} flags {fv} (want {want}) fin {fin}")
        ok &= good
    dn = _gr1_case(L, seed, dev)
    qn = torch.empty((T, 16 * (_GR1_QA + _GR1_QB)), dtype=torch.bfloat16, device=dev)
    t1 = time.time()
    _gr1_run(qn, dn["attn_input"], dn, state, dn["config"], gate_extra=1)
    torch.cuda.synchronize()
    errv = int(fl[3])
    fl[:4].zero_()
    parts.append(f"forced timeout err {errv} ({time.time() - t1:.2f}s, no hang)")
    ok &= errv != 0
    return ok, parts


def _gr1_prepare(state, t0):
    _GR1["ok"] = set()
    want = sorted(L for L, on in GR1_ROUTE.items() if on)
    if not (GR1_BUILD and want):
        return
    if time.time() - t0 > GR1_PREPARE_BY_S:
        print(f"[gr1] skipped: prepare already at {time.time() - t0:.0f}s", flush=True)
    else:
        dev = state["router"].device
        try:
            p1, su, sb = state.get("p1"), state.get("su1"), _SBB1["st"]
            if not (p1 is not None and "qkv" in p1["proj"] and hasattr(p1["mod"], "p1s_gemv_gr") and su is not None
                    and "o" in su["w"] and hasattr(su["mod"], "su1_run_gr") and _KS1["mod"] is not None and sb is not None):
                raise RuntimeError("P1 qkv / SU1 o / KS1 / SBB1 not live")
            if any(p1["proj"]["qkv"]["fmt_cpu"]):
                raise RuntimeError("qkv pack has RAW row tiles; the layer's P1 build has MODE 0 only")
            _gr1_flags(dev).zero_()
            for L in want:
                good, parts = _gr1_selftest(state, L, dev)
                print(f"[gr1] self-test L={L}: {'; '.join(parts)} -> {'ok' if good else 'FAIL'}", flush=True)
                if good:
                    _GR1["ok"].add(L)
        except Exception as ex:
            print(f"[gr1] FAILED -> today's front: {ex!r}", flush=True)
            _GR1["ok"] = set()
        _gr1_flags(dev).zero_()
        torch.cuda.synchronize()
    print(f"[gr1] live q_len {sorted(_GR1['ok'])} geo {GR1_GEO} theta {GR1_THETA} staging {GR1_WO} (prepare so far "
          f"{time.time() - t0:.1f}s)", flush=True)
    missing = [L for L in want if L not in _GR1["ok"]]
    if GR1_REQUIRE and missing:
        raise RuntimeError(f"GR1_REQUIRE: GR1 not live for q_len {missing}")

'''


def patch_front(t):
    a = t.index("\ndef kernel_fn(data, state):\n")
    k0 = t.index('    qkv = _v20_p1m(state, "qkv", data["attn_input"])\n', a)
    last = '        attn_output = _v12_gemm(o, dense["o"], "o", tokens)\n'
    k1 = t.index(last, k0) + len(last)
    block = t[k0:k1]
    head = ('    cfg = data["config"]\n    tokens = cfg["tokens"]\n    hq, d = cfg["num_q_heads"], cfg["head_dim"]\n'
            '    dense = state["dense"]\n')
    assert t[a:k0].endswith(head), "kernel_fn head changed"
    front = ("\ndef _front_r(data, state, cfg):\n    \"\"\"k3_v37's attention front (qkv -> attention -> o), verbatim.\"\"\"\n"
             '    tokens = cfg["tokens"]\n    hq, d = cfg["num_q_heads"], cfg["head_dim"]\n    dense = state["dense"]\n'
             + block + "    return attn_output\n\n")
    t = t[:k0] + ("    attn_output = _gr1_front(data, state, cfg)\n    if attn_output is None:\n"
                  "        attn_output = _front_r(data, state, cfg)\n") + t[k1:]
    t = t[:a] + "\n" + PY_GR1 + front + t[a:]
    t = sub(t, "    _su1_prepare(state, t0)\n    torch.cuda.synchronize()\n    _pt1_join_all()\n",
            "    _su1_prepare(state, t0)\n    _gr1_prepare(state, t0)\n    torch.cuda.synchronize()\n    _pt1_join_all()\n")
    return t


def main():
    t = SRC.read_text(encoding="utf-8")
    lines = t.split("\n")
    assert lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "line 1 / nonce"
    lines[1] = NONCE
    t = "\n".join(lines)
    for f in (patch_p1, patch_ks1, patch_su1, patch_sbb1, patch_front):
        t = f(t)
    assert t.isascii(), "non-ASCII"
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(t, encoding="utf-8", newline="\n")
    py_compile.compile(str(DST), doraise=True)
    print(f"gr1: {SRC.name} -> {DST} ({len(t) - len(SRC.read_text(encoding='utf-8')):+d} chars, {len(t.encode())} B), compiles")


if __name__ == "__main__":
    main()

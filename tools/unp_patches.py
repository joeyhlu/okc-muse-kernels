"""UNP (R-unp-1/2): the L2 in-word bit layout for every BITPOS3 / VMS NVFP4 consumer of the decode layer.

L2 = one fixed bit permutation of each 32-bit code word (lossless repack in prepare). Decode per word (8 codes):
  o0 = w & M, o1 = rotl(w, 10) & M, o2 = rotl(w, 13) & M, o3 = ((w & 0x001B001B) * 0x1084) & M   (M = 0x81C081C0)
o3 is a carry-free three-way bit gather by one IMAD (the masked word has bits {0,1,3,4,16,17,19,20}; the copies at
+2 / +7 / +12 never overlap). 8 ops per 8 codes instead of 9 (BITPOS3: 5 LOP3 + 3 SHF + 1 IMAD.SHL; VMS: 5 LOP3 + 2 SHF
+ 2 IMAD), and the products are bit-identical (same BF16 fields). R-unp-1 probe: -0.046..-0.064 us per VML stage.
Only the unpack function bodies and the packing tables change: composes with the m32e VML scheduling patches.

Two modes (tools/gen_k3_unp.py):
  ship: the whole file on L2 (K2A31_BITPOS3 table, K2A v3.3 deq4, VML deq4 / deq4_dq, VMS_POS + VMS decode).
  ab:   K2A stays on BITPOS3; VML gets variant "bf2l" (= bf2 + -DVM_L2=1) reading its own L2 weight copy built in
        prepare (VML_BVAR selects it per bucket, self-tested against bf2 by _m32e_vartest); VMS gets a second module
        (-DVMS_L2=1) + L2 weight copy, selected by the global VMS_L2 at capture time, self-tested against the base VMS.
"""

# (m, e0, e1, sign) word bit of each code slot; slots 2j, 2j + 1 = output j low / high half (BF16 fields 6, 7, 8, 15)
BITPOS3 = ((6, 7, 8, 15), (22, 23, 24, 31), (0, 1, 2, 9), (16, 17, 18, 25),
           (26, 27, 28, 3), (10, 11, 12, 19), (4, 5, 14, 13), (20, 21, 30, 29))
L2POS = ((6, 7, 8, 15), (22, 23, 24, 31), (28, 29, 30, 5), (12, 13, 14, 21),
         (25, 26, 27, 2), (9, 10, 11, 18), (4, 0, 1, 3), (20, 16, 17, 19))
VMS_POS_OLD = (((6, 7, 8, 15), (22, 23, 24, 31)), ((2, 3, 4, 11), (18, 19, 20, 27)),
               ((28, 29, 30, 5), (12, 13, 14, 21)), ((0, 1, 26, 9), (16, 17, 10, 25)))
VMS_POS_L2 = tuple((L2POS[2 * r], L2POS[2 * r + 1]) for r in range(4))

M = 0x81C081C0


def _rot(x, r):
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


def dec_bitpos3(w):
    return [w & M, _rot(w, 6) & M, _rot(w, 12) & M, ((w >> 6) & 0x01000100) | ((w << 2) & 0x80C080C0)]


def dec_vms_old(w):
    return [w & M, (w << 4) & M, _rot(w, 10) & M, ((w << 6) & 0x80C080C0) | (_rot(w, 14) & 0x01000100)]


def dec_l2(w):
    return [w & M, _rot(w, 10) & M, _rot(w, 13) & M, (((w & 0x001B001B) * 0x1084) & 0xFFFFFFFF) & M]


def _pack(codes, pos):   # codes: 8 nibbles (slot order), pos: 8 x (m, e0, e1, s)
    w = 0
    for c, p in zip(codes, pos):
        for b in range(4):
            w |= ((c >> b) & 1) << p[b]
    return w


def selftest(n=50000, seed=7):
    """every code nibble in every slot, both tables: L2 decode of the L2 word == the old decode of the old word."""
    import random
    assert sorted(b for p in L2POS for b in p) == list(range(32))
    rng = random.Random(seed)
    cases = [[(v if i == s else 0) for i in range(8)] for s in range(8) for v in range(16)]
    cases += [[rng.randrange(16) for _ in range(8)] for _ in range(n)]
    vms_old = [p for pair in VMS_POS_OLD for p in pair]
    for cs in cases:
        assert dec_l2(_pack(cs, L2POS)) == dec_bitpos3(_pack(cs, BITPOS3)), cs
        assert dec_l2(_pack(cs, L2POS)) == dec_vms_old(_pack(cs, vms_old)), cs
    return len(cases)


# ---------------------------------------------------------------- text patches (anchor, replacement, expected count)
OK32 = ('__device__ __forceinline__ uint32_t unp_ok32(uint32_t c) {   // c, opaque to NVVM: the 3-way gather stays one IMAD\n'
        '  uint32_t r;\n  asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(c));\n  return r;\n}\n')
L2_BODY = ('o[0] = w & 0x81C081C0u;\n'
           'o[1] = __funnelshift_l(w, w, 10) & 0x81C081C0u;\n'
           'o[2] = __funnelshift_l(w, w, 13) & 0x81C081C0u;\n'
           'o[3] = ((w & 0x001B001Bu) * unp_ok32(0x1084u)) & 0x81C081C0u;   // UNP L2: carry-free bit gather\n')

K2A_DEQ_OLD = """__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // BITPOS3 layout
  o[0] = w & 0x81C081C0u;
  o[1] = __funnelshift_l(w, w, 6) & 0x81C081C0u;
  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
  o[3] = ((w >> 6) & 0x01000100u) | ((w << 2) & 0x80C080C0u);
}
"""
K2A_DEQ_NEW = OK32 + "__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // UNP L2 layout (tools/unp_patches.py)\n" \
    + "".join("  " + ln + "\n" for ln in L2_BODY.splitlines()) + "}\n"

VML_DEQ_OLD = """__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {
o[0] = w & 0x81C081C0u;
o[1] = __funnelshift_l(w, w, 6) & 0x81C081C0u;
o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
o[3] = ((w >> 6) & 0x01000100u) | ((w << 2) & 0x80C080C0u);
}
"""
VML_DQ_OLD = """__device__ __forceinline__ void deq4_dq(uint32_t w, uint32_t* o, uint32_t k64, uint32_t k4) {   // == deq4 (same bits)
o[0] = w & 0x81C081C0u;
o[1] = (w * k64) & 0x81C081C0u;                                  // rotl 6: the mask keeps no wrapped bit
o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
o[3] = ((w >> 6) & 0x01000100u) | ((w * k4) & 0x80C080C0u);
}
"""
VML_DQ_L2 = """__device__ __forceinline__ void deq4_dq(uint32_t w, uint32_t* o, uint32_t k64, uint32_t k4) {   // UNP L2: == deq4
(void)k64; (void)k4;
deq4(w, o);
}
"""

VMS_DEC_OLD = """          const uint32_t r1 = (qw * 16u) & M;
          const uint32_t r2 = __funnelshift_l(qw, qw, 10) & M;
          const uint32_t r3 = ((qw * 64u) & 0x80C080C0u) | (__funnelshift_l(qw, qw, 14) & 0x01000100u);
"""
VMS_DEC_L2 = """          const uint32_t r1 = __funnelshift_l(qw, qw, 10) & M;   // UNP L2 layout (tools/unp_patches.py)
          const uint32_t r2 = __funnelshift_l(qw, qw, 13) & M;
          const uint32_t r3 = ((qw & 0x001B001Bu) * unp_ok32(0x1084u)) & M;
"""
VMS_HEAD_OLD = """VMS_CUDA_SRC = r\"\"\"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
"""
VMS_HEAD_NEW = VMS_HEAD_OLD + OK32

BITPOS3_SRC = """K2A31_BITPOS3 = ((6, 7, 8, 15), (22, 23, 24, 31), (0, 1, 2, 9), (16, 17, 18, 25),
           (26, 27, 28, 3), (10, 11, 12, 19), (4, 5, 14, 13), (20, 21, 30, 29))
"""
L2POS_SRC = """K2A31_BITPOS3 = ((6, 7, 8, 15), (22, 23, 24, 31), (28, 29, 30, 5), (12, 13, 14, 21),
           (25, 26, 27, 2), (9, 10, 11, 18), (4, 0, 1, 3), (20, 16, 17, 19))   # UNP L2 (tools/unp_patches.py)
"""
VMS_POS_SRC = """VMS_POS = (((6, 7, 8, 15), (22, 23, 24, 31)), ((2, 3, 4, 11), (18, 19, 20, 27)),
           ((28, 29, 30, 5), (12, 13, 14, 21)), ((0, 1, 26, 9), (16, 17, 10, 25)))
"""
VMS_POS_L2_SRC = """VMS_POS = (((6, 7, 8, 15), (22, 23, 24, 31)), ((28, 29, 30, 5), (12, 13, 14, 21)),
           ((25, 26, 27, 2), (9, 10, 11, 18)), ((4, 0, 1, 3), (20, 16, 17, 19)))   # UNP L2 (tools/unp_patches.py)
"""

SHIP = [
    (BITPOS3_SRC, L2POS_SRC, 1),
    (K2A_DEQ_OLD, K2A_DEQ_NEW, 1),
    (VML_DEQ_OLD, OK32 + VML_DEQ_OLD.replace("{\n", "{   // UNP L2 layout (tools/unp_patches.py)\n", 1)
     .replace(VML_DEQ_OLD[VML_DEQ_OLD.index("o[0]"):VML_DEQ_OLD.index("}\n")], L2_BODY), 1),
    (VML_DQ_OLD, VML_DQ_L2, 1),
    (VMS_HEAD_OLD, VMS_HEAD_NEW, 1),
    (VMS_DEC_OLD, VMS_DEC_L2, 1),
    (VMS_POS_SRC, VMS_POS_L2_SRC, 1),
]

# ---------------------------------------------------------------- A/B mode (dual layouts for VML and VMS)
VML_DEQ_AB = ("#ifndef VM_L2\n#define VM_L2 0   // UNP: 1 = L2 layout decode (variant bf2l reads st['unp_ws13'] / ['unp_ws2'])\n#endif\n"
              + OK32 + "__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {\n#if VM_L2\n" + L2_BODY + "#else\n"
              + VML_DEQ_OLD[VML_DEQ_OLD.index("o[0]"):VML_DEQ_OLD.index("}\n")] + "#endif\n}\n")
VML_DQ_AB = VML_DQ_OLD.replace("{   // == deq4 (same bits)\n", "{   // == deq4 (same bits)\n#if VM_L2\n(void)k64; (void)k4; deq4(w, o);\n#else\n", 1) \
    .replace("}\n", "#endif\n}\n")
VMS_DEC_AB = "#if VMS_L2\n" + VMS_DEC_L2 + "#else\n" + VMS_DEC_OLD + "#endif\n"
VMS_HEAD_AB = VMS_HEAD_OLD + "#ifndef VMS_L2\n#define VMS_L2 0\n#endif\n" + OK32

VML_DEFS_OLD = '''             "bf2": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1")}'''
VML_DEFS_AB = '''             "bf2": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1"),
             "bf2l": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", "-DVM_L2=1")}
UNP_L2VARS = ("bf2l",)'''
SCB_OLD = '''    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2") else _VML["sc"]'''
SCB_AB = '''    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "bf2l") else _VML["sc"]
    if var in UNP_L2VARS:
        st = dict(st, ws13=st["unp_ws13"], ws2=st["unp_ws2"])'''
VMLPREP_OLD = '''        _VML["sc"] = _vml_alloc(dev, sms)
'''
VMLPREP_AB = '''        _VML["sc"] = _vml_alloc(dev, sms)
        if any(v in UNP_L2VARS for v in (VML_VAR,) + tuple(VML_ABVARS) + tuple(VML_BVAR.values())) and "unp_ws13" not in st:
            t_u = time.time()
            st["unp_ws13"], st["unp_ws2"] = torch.empty_like(st["ws13"]), torch.empty_like(st["ws2"])
            for e in range(K2A_E_LOC):
                w13q, s13q = _k2a31_prepack_expert(w["w13"][e], w["w13_sf"][e], gate_up=True, bitpos=UNP_L2POS)
                w2q, s2q = _k2a31_prepack_expert(w["w2"][e], w["w2_sf"][e], gate_up=False, bitpos=UNP_L2POS)
                st["unp_ws13"][e] = _k2a33_stage_repack(w13q, s13q)
                st["unp_ws2"][e] = _k2a33_stage_repack(w2q, s2q)
            torch.cuda.synchronize()
            print(f"[unp] VML L2 weight copy in {time.time() - t_u:.1f}s", flush=True)
'''
L2POS_AB = BITPOS3_SRC + L2POS_SRC.replace("K2A31_BITPOS3 = ", "UNP_L2POS = ").replace("           (25", "       (25")

VMS_GLOBAL_OLD = "VMS_ENABLE = True\n"
VMS_GLOBAL_AB = ("VMS_ENABLE = True\nVMS_L2 = False      # UNP A/B: capture-time switch to the L2 module + weights\n"
                 "VMS_L2_AB = True    # UNP A/B: prepare the L2 VMS module + weights (self-tested against the base VMS)\n")
VMS_PACK_OLD = '''def _vms_pack_words(cu):
'''
VMS_PACK_AB = '''def _vms_pack_words(cu, pos=None):
    pos = VMS_POS if pos is None else pos
'''
VMS_PACK_USE_OLD = '''                word |= ((cd >> b) & 1) << VMS_POS[r][h][b]
'''
VMS_PACK_USE_AB = '''                word |= ((cd >> b) & 1) << pos[r][h][b]
'''
VMS_PREX_OLD = '''def _vms_prepack_expert(w, sf, gate_up):
'''
VMS_PREX_AB = '''def _vms_prepack_expert(w, sf, gate_up, pos=None):
'''
VMS_PREX_USE_OLD = '''    words = _vms_pack_words(cu).view(torch.uint8).view(n // 16, 32768)
'''
VMS_PREX_USE_AB = '''    words = _vms_pack_words(cu, pos).view(torch.uint8).view(n // 16, 32768)
'''
VMS_BUILD_OLD = '''def _vms_build():
    from torch.utils.cpp_extension import load_inline
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
'''
VMS_BUILD_AB = '''def _vms_build(l2=False):
    from torch.utils.cpp_extension import load_inline
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
    if l2:
        flags.append("-DVMS_L2=1")
'''
VMS_BUILD_TAIL_OLD = '''    _VMS["G"] = G
    _VMS["mod"] = mod
    _K3_BUILD_S["vms"] = time.time() - t1
    return mod
'''
VMS_BUILD_TAIL_AB = '''    if l2:
        return mod
    _VMS["G"] = G
    _VMS["mod"] = mod
    _K3_BUILD_S["vms"] = time.time() - t1
    return mod
'''
VMS_CALL_OLD = '''    vs["mod"].run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],
                  out, vs["nomarks"], m, K2A_POST_NORM_EPS, 1, 0)
'''
VMS_CALL_AB = '''    l2 = VMS_L2 and vs.get("mod_l2") is not None
    (vs["mod_l2"] if l2 else vs["mod"]).run(x, ids, tw, vs["wA_l2"] if l2 else vs["wA"], vs["wB_l2"] if l2 else vs["wB"],
                  vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],
                  out, vs["nomarks"], m, K2A_POST_NORM_EPS, 1, 0)
'''
VMS_LIVE_OLD = '''        st["vms_ok"] = good
        print(f"[vms] VMS {'LIVE at T <= ' + str(VMS_MAXT) if good else 'OFF: self-test failed, K2A at every T'} "
'''
VMS_LIVE_AB = '''        st["vms_ok"] = good
        if good and VMS_L2_AB:
            _unp_vms_l2_prepare(st, w, dev, E)
        print(f"[vms] VMS {'LIVE at T <= ' + str(VMS_MAXT) if good else 'OFF: self-test failed, K2A at every T'} "
'''
VMS_L2PREP_FN = '''

def _unp_vms_l2_prepare(st, w, dev, E):
    """UNP A/B: second VMS module on the L2 layout + its weights; live only if every self-test case is bitwise the base VMS."""
    global VMS_L2
    vs = st["vms"]
    vs["mod_l2"] = None
    try:
        t_u = time.time()
        mod2 = _vms_build(l2=True)
        wA2 = torch.empty_like(vs["wA"])
        wB2 = torch.empty_like(vs["wB"])
        for e in range(E):
            wA2[e] = _vms_prepack_expert(w["w13"][e], w["w13_sf"][e], True, pos=UNP_VMS_POS_L2)
            wB2[e] = _vms_prepack_expert(w["w2"][e], w["w2_sf"][e], False, pos=UNP_VMS_POS_L2)
        vs["wA_l2"], vs["wB_l2"] = wA2, wB2
        same = True
        for bs, seed, dist in VMS_SELF_TEST_CASES:
            if bs > VMS_MAXT or dist == "hot8":
                continue
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            x = d["hidden_states"].contiguous()
            VMS_L2 = False
            vs["mod_l2"] = None
            o0 = _vms_call(st, x, d["topk_ids"], d["topk_weights"]).clone()
            vs["mod_l2"] = mod2
            VMS_L2 = True
            o1 = _vms_call(st, x, d["topk_ids"], d["topk_weights"]).clone()
            VMS_L2 = False
            torch.cuda.synchronize()
            eq = bool(torch.equal(o0, o1)) and not int(vs["ctr"][2])
            print(f"[unp] VMS L2 self-test bs={bs} {dist}: bitwise vs base {eq}", flush=True)
            same &= eq
        vs["mod_l2"] = mod2 if same else None
        print(f"[unp] VMS L2 {'LIVE' if same else 'OFF'} ({time.time() - t_u:.1f}s)", flush=True)
    except Exception as ex:
        VMS_L2 = False
        vs["mod_l2"] = None
        print(f"[unp] VMS L2 prepare raised {ex!r}: OFF", flush=True)
'''
VMS_POS_AB = VMS_POS_SRC + VMS_POS_L2_SRC.replace("VMS_POS = ", "UNP_VMS_POS_L2 = ").replace("           ((25", "                  ((25")

VML_GLOB_OLD = """VML_VAR = 'bf2'
VML_ABVARS = ()
VML_BVAR = {}
"""
VML_GLOB_AB = """VML_VAR = 'bf2'
VML_ABVARS = ('bf2l',)   # UNP A/B: build + self-test the L2 variant (arms switch VML_BVAR)
VML_BVAR = {256: 'bf2', 128: 'bf2', 32: 'bf2'}   # == VML_VAR (ship); keys make _m32e_vartest run bf2l
"""

AB = [
    (BITPOS3_SRC, L2POS_AB, 1),
    (VML_GLOB_OLD, VML_GLOB_AB, 1),
    (VML_DEQ_OLD, VML_DEQ_AB, 1),
    (VML_DQ_OLD, VML_DQ_AB, 1),
    (VML_DEFS_OLD, VML_DEFS_AB, 1),
    (SCB_OLD, SCB_AB, 1),
    (VMLPREP_OLD, VMLPREP_AB, 1),
    (VMS_HEAD_OLD, VMS_HEAD_AB, 1),
    (VMS_DEC_OLD, VMS_DEC_AB, 1),
    (VMS_POS_SRC, VMS_POS_AB, 1),
    (VMS_GLOBAL_OLD, VMS_GLOBAL_AB, 1),
    (VMS_BUILD_OLD, VMS_BUILD_AB, 1),
    (VMS_BUILD_TAIL_OLD, VMS_BUILD_TAIL_AB, 1),
    (VMS_PACK_OLD, VMS_PACK_AB, 1),
    (VMS_PACK_USE_OLD, VMS_PACK_USE_AB, 1),
    (VMS_PREX_OLD, VMS_PREX_AB, 1),
    (VMS_PREX_USE_OLD, VMS_PREX_USE_AB, 1),
    (VMS_CALL_OLD, VMS_CALL_AB, 1),
    (VMS_LIVE_OLD, VMS_LIVE_AB, 1),
]
VMS_L2PREP_ANCHOR = "\nVMS_SELF_TEST_CASES = "

# ---------------------------------------------------------------- R-unp-3: stage_m (mma.sync path) scale decode = dec_half2
# stage_m still built two scale words (2 x PRMT + IMAD) per 4-step group and called dec_half (8 x fma.rn.bf16x2 with a -0
# addend); stage<N> uses dec_half2 (one PRMT + IMAD for both rows, HMUL2 .H0_H0 / .H1_H1). Same products bit for bit
# (round(a * b) == round(a * b + -0)). Code loads, slot release and the mma.sync loop are unchanged (m32e's domain).
SMQ_OLD = """const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
uint32_t hh[8];
dec_half(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, Sg, S8, hh);
"""
SMQ_SHIP = """const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
const uint32_t k16q = 16u + ((uint32_t)A.flags >> 30);   // UNP SMQ: opaque 16 (flags < 2^30), as stage<N>
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
uint32_t hh[8];
dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16q, 64u, 4u);   // UNP SMQ (tools/unp_patches.py)
"""
SMQ_AB = """const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
#if VM_SMQ
const uint32_t k16q = 16u + ((uint32_t)A.flags >> 30);   // UNP SMQ: opaque 16 (flags < 2^30), as stage<N>
#else
const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#endif
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
uint32_t hh[8];
#if VM_SMQ
dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16q, 64u, 4u);   // UNP SMQ (tools/unp_patches.py)
#else
dec_half(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, Sg, S8, hh);
#endif
"""
SHIP_SMQ = [(SMQ_OLD, SMQ_SHIP, 1)]

VML_DEFS_AB3 = VML_DEFS_AB.replace('''"-DVM_NOB9=1", "-DVM_L2=1")}
UNP_L2VARS = ("bf2l",)''', '''"-DVM_NOB9=1", "-DVM_L2=1"),
             "bf2lq": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", "-DVM_L2=1", "-DVM_SMQ=1")}
UNP_L2VARS = ("bf2l", "bf2lq")''')
SCB_AB3 = SCB_AB.replace('"bf2", "bf2l")', '"bf2", "bf2l", "bf2lq")')
VML_GLOB_AB3 = VML_GLOB_AB.replace("VML_ABVARS = ('bf2l',)", "VML_ABVARS = ('bf2l', 'bf2lq')")
VML_DEQ_AB3 = VML_DEQ_AB.replace("#ifndef VM_L2\n", "#ifndef VM_SMQ\n#define VM_SMQ 0   // UNP: 1 = stage_m scale decode via dec_half2\n#endif\n#ifndef VM_L2\n", 1)
# VMS L2 gate: the base VMS is not bitwise run-to-run (R-unp-2: base-vs-L2 equal in some cases only), so gate on tolerance
VMS_GATE_OLD = '''            eq = bool(torch.equal(o0, o1)) and not int(vs["ctr"][2])
            print(f"[unp] VMS L2 self-test bs={bs} {dist}: bitwise vs base {eq}", flush=True)'''
VMS_GATE_NEW = '''            eb, _ = _k2a_rel_err(o1, o0)
            tot, row = _k2a_rel_err(o1, _k2a_ref_fp32(d, w))
            eq = eb < 2e-3 and tot < 1e-2 and row < 5e-2 and not int(vs["ctr"][2]) and bool(torch.isfinite(o1.float()).all())
            print(f"[unp] VMS L2 self-test bs={bs} {dist}: bitwise vs base {bool(torch.equal(o0, o1))}, rel vs base {eb:.2e} "
                  f"vs FP32 {tot:.3e} worst row {row:.3e} -> {'ok' if eq else 'FAIL'}", flush=True)'''

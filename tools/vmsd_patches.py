"""R-vmsd-1: VMS T16 class-null probe; generator + CPU dry run, no GPU.

Kill risks first: issue/MM/port work can overlap another class or the stream;
register allocation/clock changes can contaminate a null probe. No speed claim.
MEASURED R-vmsn-1 (notes/lab.md:1179): d16s/d16f whole-consumer pool
8.49/9.78 us, H=9.13 us. Per-class saving is UNMEASURED; this probe resolves it.

Real route: _v11_k2a33_call -> _vms_call -> vs["mod_hi"].run at m=16.
Hi flags: T16=1, NS=2, ER=1, RO=1, WT2=30, G=132; T16E/R/HI default 0.
Mask bits 0..7 and class bits 8..9 occupy Params tail padding (offset 124,
size 128); transported in pdl bits 8..37. Device-only runtime salt bits
10..29 are zero in every host arm. All controls use the SAME patched binary.

EST source/PTX accounting per unit, per compute thread, cold/hot:
HBM 36,864 -> 36,864 B/CTA; weight+scale LDS 9*16=144 ->144 B/thread;
hot x2s adds 16*16=256 B/thread, unchanged. Job-change x loads unchanged.
Base core: 9 LDS + 16*(PRMT+IMAD) +32*(8 unpack+4 HMUL2+1/2 MMA)
=457/489 instructions, plus 16 hot x2s LDS =505 hot. Address/move/control
instructions and epilogue excluded; SASS gate/inspection reports actual code.
DQ: unpack 8->3 INT/word (one AND + two runtime XORs), 256->96 INT/unit,
4 HMUL2 and 1/2 MMA retained; core 297/345 (minus160), plus register moves.
MM: 32/64 MMA->0, replaced by 96/128 XOR3 instructions/unit (all A/B
fragments feed the sink ONCE, avoiding hot A cancellation); core 521/569. A runtime-zero sink mask feeds the
existing zero-partial stores. Sink overhead makes this a conservative probe.
LD: 457/505 core unchanged; all nine volatile LDS.128 target slot+0 for
all lanes/warps, same 144 register bytes/thread, 16 unique bytes/unit.
DQ25 uses DQ in warps0,1 only; other six warps retain base arithmetic.

Protocol audit for every CTA c=0..G-1, unchanged in all arms:
* tid0 initializes full[NS] count1, pfull[NRB] count8, pempty[NRB] count1;
  tid0 issues UB-byte transactions; tid0..255 wait full and preload all wv.
* tid0..255 participate in bar.sync1 count256 and original ER refill;
  tid0 alone reissues q+NS. No consumer class changes this sequence.
* tid0..255 wait pempty and store red/optional red2; all32 lanes join;
  tid32*w (w=0..7) arrives pfull once, exactly8 arrivals.
* helper tid256..287 waits pfull, reduces all8 warps, joins; tid256 arrives
  pempty once. A stores/fence/join/doneA and B ybuf/ss loops are unchanged.
* All288 threads reach CTA joins; every CTA tid0 adds/waits the grid ctr.
  After grid wait, CTA0 tid0..63 resets ctr[8:72], tid0..127 clears next
  ss bank, tid0 increments generation. Combine CTAs c iterate it=c;m*4;+=G;
  tid0..255 write4 columns each, read all8 expert slots; helper alone breaks.
No added retirement/early exit or synchronization. Generator compares protocol
lines and the entire reduction/store/helper/grid-cleanup/combine suffix exactly.
"""
import ast
import ctypes
import pathlib
import re
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
OUT = ROOT / "kernels/lab/k3_vmsd_ab.py"
LIMIT = 1_000_000
GLOBALS = '''# VMSD LAB: class 0=ship, 1=DQ, 2=MM, 3=LD; mask warps 0..7 only.
VMSD_MASK = 0
VMSD_CLASS = 0
_VMSD = {"calls": {}}
'''
HOST = '''    mask, cls = (VMSD_MASK, VMSD_CLASS) if m == 16 else (0, 0)
    if type(mask) is not int or not 0 <= mask <= 255 or type(cls) is not int or not 0 <= cls <= 3:
        raise ValueError("VMSD_MASK/CLASS must be ints in 0..255 / 0..3")
'''
COUNT = '''    if m == 16:
        key = f"T{m}:{cls}:{mask}"
        _VMSD["calls"][key] = _VMSD["calls"].get(key, 0) + 1
'''


def sub(s, old, new, what):
    assert s.count(old) == 1, f"{what}: expected one anchor, got {s.count(old)}"
    return s.replace(old, new, 1)


def node(s, name):
    matches = [n for n in ast.parse(s).body if
               (isinstance(n, ast.FunctionDef) and n.name == name) or
               (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))]
    assert len(matches) == 1, f"unique definition: {name}"
    return matches[0]


def literal(s, name):
    return ast.literal_eval(node(s, name).value)


HELPERS = r'''
template<int C> struct VmsdClass { static constexpr int value = C; };
__device__ __forceinline__ uint4 vmsd_ld(const unsigned char* p) {
uint4 v;
asm volatile("ld.volatile.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(su32(p)) : "memory");
return v;
}
__device__ __forceinline__ void vmsd_sink(uint32_t& z, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
// Three XOR3s: retain ALL A/B inputs, including the hot x2s loads.
asm volatile("{ .reg .b32 t,u; lop3.b32 t,%1,%2,%3,0x96; lop3.b32 u,%4,%5,%6,0x96; lop3.b32 %0,%0,t,u,0x96; }"
: "+r"(z) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void vmsd_sink2(uint32_t& z, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3) {
// One A contribution per word: XORing it twice would cancel all hot dequant.
asm volatile("{ .reg .b32 t,u,v; lop3.b32 t,%1,%2,%3,0x96; lop3.b32 u,%4,%5,%6,0x96; lop3.b32 v,%7,%8,%0,0x96; lop3.b32 %0,t,u,v,0x96; }"
: "+r"(z) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3));
}
'''


def patch_cuda_host(base, cu):
    edits_cu = []

    def change(old, new, label):
        nonlocal cu
        cu = sub(cu, old, new, label)
        edits_cu.append((old, new, label))

    original = cu
    tail = "__nv_bfloat16* out; long long* marks; int m; float eps; int lo;\n};"
    change(tail, tail.replace("int lo;", "int lo; int flags; // VMSD: former tail padding") +
           '\nstatic_assert(sizeof(Params)==128 && __builtin_offsetof(Params,flags)==124, "VMSD ABI");', "Params")
    kernel = "__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {"
    change(kernel, HELPERS + kernel, "helpers")
    start = "const unsigned char* slot = ring + sl * UB;\n"
    end = "float4 o;\n"
    i, j = cu.index(start), cu.index(end, cu.index(start))
    body = cu[i:j]
    new = body
    # Decode once per unit, uniformly within a warp; helper never enters this block.
    decode = """#if VMS_T16
const int vmsd_cls = ((p.flags >> warp) & 1) ? ((p.flags >> 8) & 3) : 0;
#else
constexpr int vmsd_cls = 0;
#endif
"""
    new = sub(new, start, decode + start, "class decode")
    # Preload remains before the ORIGINAL early compute barrier/refill.
    scales = "*reinterpret_cast<const uint4*>(slot + UC + warp * 512 + lane * 16)"
    new = sub(new, scales, "vmsd_ld(slot + (vmsd_cls == 3 ? 0 : UC + warp * 512 + lane * 16))", "scale LDS")
    weights = "*reinterpret_cast<const uint4*>(slot + (warp * 8 + cc) * 512 + lane * 16)"
    assert new.count(weights) == 4
    new = new.replace(weights, "vmsd_ld(slot + (vmsd_cls == 3 ? 0 : (warp * 8 + cc) * 512 + lane * 16))")
    # ER's wv preloads are common; arithmetic variants specialize inside a generic
    # lambda. A single runtime branch surrounds the entire hot/cold unit, so NULL
    # does not merely predicate-off instructions while retaining their issue slots.
    loop_start = "#if VMS_T16\nfloat4 o2 = make_float4(0.f, 0.f, 0.f, 0.f);\n"
    new = sub(new, loop_start, loop_start + "#endif\nuint32_t vmsd_z = 0;\n"
              "auto vmsd_unit = [&](auto vmsd_tag) {\n"
              "constexpr int vmsd_mode = decltype(vmsd_tag)::value;\n#if VMS_T16\n", "unit dispatch start")
    unpack = """const uint32_t r0 = qw & M;
const uint32_t r1 = __funnelshift_l(qw, qw, 10) & M;
const uint32_t r2 = __funnelshift_l(qw, qw, 13) & M;
const uint32_t r3 = ((qw & 0x001B001Bu) * unp_ok32(0x1084u)) & M;"""
    dq = """const uint32_t r0 = qw & M;
uint32_t r1, r2, r3;
if constexpr (vmsd_mode == 1) {
// Upper launch bits are runtime unknown to the device compiler; host passes 0.
// r3=r0 has a DIFFERENT scale; r1/r2 cannot be equated with it or one another.
r1 = r0 ^ ((unsigned)p.flags >> 10);
r2 = r0 ^ ((unsigned)p.flags >> 20);
r3 = r0;
} else {
r1 = __funnelshift_l(qw, qw, 10) & M;
r2 = __funnelshift_l(qw, qw, 13) & M;
r3 = ((qw & 0x001B001Bu) * unp_ok32(0x1084u)) & M;
}"""
    assert new.count(unpack) == 3
    new = new.replace(unpack, dq)
    # Hot A fragments feed the sink once, not twice (which would cancel A).
    pair = ("mma(acc[s & 1], A0, A1, A2, A3, bb[2 * s], bb[2 * s + 1]);\n"
            "mma(acc[2 + (s & 1)], A0, A1, A2, A3, b2[2 * s], b2[2 * s + 1]);")
    assert new.count(pair) == 2
    # All three textual loops; the T16E loop is compile-disabled in the real hi.
    cold = "mma(acc[s], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);"
    new = sub(new, cold, "if constexpr (vmsd_mode == 2) vmsd_sink(vmsd_z, " + cold[len("mma(acc[s], "):] +
              "\nelse " + cold, "cold MMA sink")
    new = new.replace(pair, "if constexpr (vmsd_mode == 2) vmsd_sink2(vmsd_z, A0, A1, A2, A3, bb[2 * s], bb[2 * s + 1], b2[2 * s], b2[2 * s + 1]);\nelse {\n" + pair + "\n}")
    new += """};
if (vmsd_cls == 1) vmsd_unit(VmsdClass<1>{});
else if (vmsd_cls == 2) vmsd_unit(VmsdClass<2>{});
else vmsd_unit(VmsdClass<0>{});
if (vmsd_cls == 2) {
// Runtime-zero mask keeps the XOR chain live all the way to existing stores.
const float vmsd_zero = __uint_as_float(vmsd_z & ((unsigned)p.flags >> 10));
#pragma unroll
for (int s = 0; s < 4; ++s) {
#pragma unroll
for (int k = 0; k < 4; ++k) acc[s][k] = vmsd_zero;
}
#if VMS_T16
o2 = make_float4(vmsd_zero, vmsd_zero, vmsd_zero, vmsd_zero);
#endif
}
"""
    change(body, new, "consumer loops")
    transport = "p.flags = (int)((pdl >> 8) & 0x3fffffff); // VMSD mask/class/runtime salt\n"
    change("p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo;\n",
           "p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo;\n" + transport, "transport")
    change("programmaticStreamSerializationAllowed = pdl ? 1 : 0;",
           "programmaticStreamSerializationAllowed = (pdl & 1) ? 1 : 0;", "PDL bit")
    # Exact structural evidence: the protocol and cleanup suffix is unchanged.
    suffix = cu[cu.index(end):].replace(transport, "").replace(
        "programmaticStreamSerializationAllowed = (pdl & 1) ? 1 : 0;",
        "programmaticStreamSerializationAllowed = pdl ? 1 : 0;")
    assert suffix == original[original.index(end):]
    protocol = r"^.*(?:mbar_|cta_sync|__syncwarp|__syncthreads|issue\(|return|break|continue).*$"
    # New helpers contain a return, but kernel protocol itself must match exactly.
    def protocol_lines(text):
        text = text[text.index(kernel):]
        return re.findall(protocol, text, re.M)
    assert protocol_lines(cu) == protocol_lines(original)
    call = ast.get_source_segment(base, node(base, "_vms_call"))
    call2 = sub(call, '    md = vs["mod_hi"]', HOST + '    md = vs["mod_hi"]', "host flags")
    call2 = sub(call2, "K2A_POST_NORM_EPS, 1, 0)", "K2A_POST_NORM_EPS, 1 | ((mask | (cls << 8)) << 8), 0)", "host transport")
    call2 = sub(call2, "    return out", COUNT + "    return out", "counter")
    edits = [('VMS_CUDA_SRC = r"""' + original + '"""', 'VMS_CUDA_SRC = r"""' + cu + '"""', "CUDA"),
             ("VMS_ENABLE = True\n", GLOBALS + "VMS_ENABLE = True\n", "globals"), (call, call2, "call")]
    s = base
    for old, new, label in edits:
        s = sub(s, old, new, label)
    ast.parse(s)
    restored = s
    for old, new, label in reversed(edits):
        restored = sub(restored, new, old, "undo " + label)
    assert restored == base
    restored = cu
    for old, new, label in reversed(edits_cu):
        restored = sub(restored, new, old, "undo CUDA " + label)
    assert restored == original
    return s


def build(base):
    assert "VMSD" not in base
    cu = literal(base, "VMS_CUDA_SRC")
    assert "constexpr int NW = 8, NT = NW * 32, NTL = NT + 32" in cu
    params = re.search(r"struct Params \{(.*?)\n\};", cu, re.S).group(1)
    assert "flags" not in params
    # Derive and check the actual field layout, not a guessed spare word.
    fields = []
    for decl in params.split(";"):
        decl = decl.strip()
        if not decl:
            continue
        name = decl.split()[-1].lstrip("*")
        typ = ctypes.c_void_p if "*" in decl else ctypes.c_float if decl.startswith("float ") else ctypes.c_int
        fields.append((name, typ))
    before = type("BaseParams", (ctypes.Structure,), {"_fields_": fields})
    after = type("LabParams", (ctypes.Structure,), {"_fields_": fields + [("flags", ctypes.c_int)]})
    assert ctypes.sizeof(before) == ctypes.sizeof(after) == 128 and after.flags.offset == 124
    assert all(getattr(before, k).offset == getattr(after, k).offset for k, _ in fields)

    return patch_cuda_host(base, cu)


def dry_run(base, patched):
    """Execute real AST-selected host functions with CPU tensor/module sentinels.

    No invented state dictionary: execute the base's st['vms'] construction and
    its return {'st': st, 'ok': ok}; preserve the real layer outer key. All new
    host paths run, including prepare's zero-flag calls, hi fallback, bad flags,
    arm selection, liveness checks and report-only analysis boundaries.
    """
    class Tensor:
        def __init__(self, shape=(32,)):
            self.shape, self.device = shape, "cpu-dry"
        def to(self, *a): return self
        def contiguous(self): return self
        def float(self): return self
        def reshape(self, *a): return self
        def data_ptr(self): return 16
        def __getitem__(self, i): return 0
        def __mul__(self, x): return self
        def __radd__(self, x): return self
    class Module:
        def __init__(self): self.launches = []
        def run(self, *a): self.launches.append(a)
    torch = types.SimpleNamespace(int32=0, float32=1, bfloat16=2, int64=3,
                                  zeros=lambda shape, **kw: Tensor(shape), empty=lambda shape, **kw: Tensor(shape))
    mod, hi = Module(), Module()
    ns = dict(torch=torch, mod=mod, _VMS={"mod_hi": hi}, wA=Tensor(), wB=Tensor(), E=32, two62=float(2 ** 62),
              w={k: Tensor() for k in ("w13_gs", "w2_gs", "post_norm_w")}, dev="cpu-dry", st={},
              K2A_D_EXPERT=4096, K2A_D_HIDDEN=4096, K2A_POST_NORM_EPS=1e-6)
    prep = node(base, "_vms_prepare")
    st_vms = [n for n in ast.walk(prep) if isinstance(n, ast.Assign) and
              any(ast.unparse(t) == "st['vms']" for t in n.targets)]
    assert len(st_vms) == 1
    exec(compile(ast.Module(body=st_vms, type_ignores=[]), "<real base st[vms]>", "exec"), ns)
    ret = [n.value for n in ast.walk(node(base, "_v11_k2a33_prepare")) if isinstance(n, ast.Return) and
           isinstance(n.value, ast.Dict) and {ast.literal_eval(k) for k in n.value.keys} == {"st", "ok"}]
    assert len(ret) == 1 and 'state["k2a33"] = _v11_k2a33_prepare(weights, t0)' in base
    ns["ok"] = {8: True, 16: True}
    state = {"k2a33": eval(compile(ast.Expression(ret[0]), "<real base k33>", "eval"), ns)}
    state["k2a33"]["st"]["vms_ok"] = True
    assert 'k33["st"]' in base
    for name in ("VMS_HI_ON", "VMS_ENABLE", "VMS_MAXT", "HYBLO_T", "HYBLO_K", "V11_K2A33_ROUTE"):
        ns[name] = literal(base, name)
    exec(GLOBALS, ns)
    exec(compile(ast.Module(body=[node(patched, "_vms_call"), node(base, "_v11_k2a33_call")],
                            type_ignores=[]), "<real VMS call/route>", "exec"), ns)
    routed = []
    ns["_hyblo_route"] = lambda *a: routed.append(a[0].shape[0]) or Tensor()
    ns["_k3_disable"] = lambda *a: (_ for _ in ()).throw(AssertionError(f"unexpected fallback {a}"))
    runner = (ROOT / "tools/box/vmsd_run.py").read_text(encoding="utf-8")
    run_ns = dict(statistics=__import__("statistics"), math=__import__("math"),
                  copy=__import__("copy"), NAMES={2:"d16s",3:"d16f"})
    wanted = ("ProbeFailure", "require_vms", "calls", "require_live", "assess", "cpu_gate", "energy_schedule", "energy_report")
    rn = [n for n in ast.parse(runner).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    rn += [node(runner, n) for n in ("SHAPES", "ARMS", "NULLS", "POOL_US")]
    exec(compile(ast.Module(body=rn, type_ignores=[]), "<runner CPU paths>", "exec"), run_ns)
    proxy = types.SimpleNamespace(**ns)
    run_ns["require_vms"](proxy, state)
    run_ns["calls"](proxy)
    base_call = compile(ast.Module(body=[node(base, "_vms_call")], type_ignores=[]), "<base call>", "exec")
    # Execute every runtime class and mask, plus each unaffected prepare shape.
    for m in (1, 8, 9, 12, 16):
        for cls in range(4):
            for mask in range(256):
                ns["VMSD_MASK"], ns["VMSD_CLASS"] = mask, cls
                x, ids, tw = Tensor((m, 4096)), Tensor(), Tensor()
                ns["_vms_call"](state["k2a33"]["st"], x, ids, tw)
                args = (hi if m > 8 else mod).launches[-1]
                flags = mask | (cls << 8) if m == 16 else 0
                assert args[-2] == 1 | (flags << 8)
                if flags == 0:
                    original = dict(ns)
                    exec(base_call, original)
                    original["_vms_call"](state["k2a33"]["st"], x, ids, tw)
                    old = (hi if m > 8 else mod).launches[-1]
                    assert args[:12] == old[:12] and args[13:] == old[13:] and args[12].shape == old[12].shape
    for arm, settings in run_ns["ARMS"]:
        ns.update(settings)
        for shape in run_ns["SHAPES"]:
            n = len(hi.launches)
            ns["_v11_k2a33_call"](Tensor((16,4096)),Tensor(),Tensor(),state["k2a33"]["st"])
            assert len(hi.launches)==n+1 and hi.launches[-1][-2] == 1 | ((settings["VMSD_MASK"] | (settings["VMSD_CLASS"] << 8)) << 8)
    ns["_v11_k2a33_call"](Tensor((8,4096)),Tensor(),Tensor(),state["k2a33"]["st"])
    assert routed == [8]  # shipped T8 HYBLO interception remains unchanged
    run_ns["require_live"](ns["_VMSD"]["calls"], "CPU dry run")
    ns["VMSD_MASK"] = ns["VMSD_CLASS"] = 0
    state["k2a33"]["st"]["vms"]["mod_hi"] = None
    ns["_vms_call"](state["k2a33"]["st"], Tensor((16,4096)), Tensor(), Tensor())
    assert mod.launches[-1][-4] == 16
    try: run_ns["require_vms"](proxy, state)
    except run_ns["ProbeFailure"]: pass
    else: raise AssertionError("runner accepted missing hi build")
    for key, bads in (("VMSD_MASK",(-1,256,True,1.0)),("VMSD_CLASS",(-1,4,True,1.0))):
        for bad in bads:
            ns["VMSD_MASK"] = ns["VMSD_CLASS"] = 0
            ns[key]=bad
            try: ns["_vms_call"](state["k2a33"]["st"],Tensor((16,4096)),Tensor(),Tensor())
            except ValueError: pass
            else: raise AssertionError(f"bad {key} accepted: {bad}")
            try: run_ns["calls"](types.SimpleNamespace(**ns))
            except run_ns["ProbeFailure"]: pass
            else: raise AssertionError("runner accepted invalid flags")
    run_ns["cpu_gate"]()
    print("[vmsd] CPU dry run PASS: real state construction, 5120 class/mask/shape cases, both modules, every arm, report gates")


def main():
    import sys
    if len(sys.argv) != 1:
        raise SystemExit("usage: python -B -I tools/vmsd_patches.py (lab only)")
    raw = BASE.read_bytes()
    base = raw.decode("utf-8").replace("\r\n", "\n")
    source = build(base)
    dry_run(base, source)
    out = source.replace("\n", "\r\n").encode() if b"\r\n" in raw else source.encode()
    assert len(out) <= LIMIT, f"over cap: {len(out)}"
    OUT.write_bytes(out)
    print(f"[vmsd] {OUT.relative_to(ROOT)}: {len(out):,} B; over cap {max(0, len(out)-LIMIT):,} B")
    print("[vmsd] AST, ABI padding/offsets, protocol projection, exact reverse-edit gates PASS")


if __name__ == "__main__":
    main()

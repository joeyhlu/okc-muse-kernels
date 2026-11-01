"""R-vmsh-1: helper-only T16 probe; generator and CPU dry run, never a GPU run.

Kill risks first: helper work may wait/overlap instead of pacing SMSP0; extra
registers/code layout may perturb compute scheduling. This is a two-sided probe,
not an optimization. MEASURED R-vmsd-1/-2 (lab.md:1185/1187): DQ H=7.41 us,
SMSP0 H=3.87 us. EST conditional dose = 150*(7.41/320)=3.47 us slower,
75*(7.41/320)=1.74 us slower, IF helper issue is the exposed pacer. No
guaranteed saving: the report-only thresholds test that critical-path premise.

Real route: pz.py _v11_k2a33_call:7673/7684 -> _vms_call:4961 -> mod_hi:4967;
_vms_build:4919 flags T16=1, WT2=30, NS=2, ER=1, RO=1; G=132, NTL=288.
No prepare bank or module accessor is added. All controls share one binary.

EST source/PTX accounting per unit (SASS still requires compilation):
* Weight HBM 36,864 ->36,864 B/CTA; compute weight/scale LDS 144 ->144
  B/thread, hot x2s LDS 256 ->256 B/thread; compute instructions unchanged.
* HRED helper cold: 8 LDS.128+28 FADD ->1 LDS.128 (128 ->16 B/thread).
  Hot: 16 LDS.128+60 FADD ->2 LDS.128 (256 ->32 B/thread). The extra four
  FADD are the real base's red2 leading +0, retained exactly in mode zero.
* HSWI A: two/four outputs/thread keep 4/8 B stores and one BF16 conversion
  per output; 3 MUL+NEG+expf+ADD+fdividef per output ->0 (source operations;
  intrinsic expansion/SASS counts unverified). The data-dependent stand-in
  still needs conversion. B-phase math, stores, shuffles and atomics unchanged.
* HADD150/75: 0 ->100/50 independent ADD +50/25 balanced XOR3, respectively,
  per helper thread. No serial recurrence across units or pipeline input;
  only a short XOR tree feeds the sink. One lane0 STS of 4 B/CTA/unit
  (0 ->4 B), no LDS/global traffic, no shuffle/sync. Padding holds the sink.
  Guard/address overhead is additional; exact SASS counts are not asserted.

Protocol coverage for every CTA c=0..131, EVERY mode, with no new exits:
* tid0 initializes full/pfull/pempty and refills; compute tid0..255 keep
  full/pempty waits, bar.sync1 count256, red/red2 writes, and eight pfull
  arrivals from tid32*w, w=0..7. Whole compute branch is byte-identical.
* helper tid256..287 keeps pfull wait and __syncwarp; tid256 releases pempty.
  A stores (all helper lanes with tk<cnt), fence, join and tid256 doneA
  atomics stay; B ybuf stores and g==0 (tid256..259) ss atomics stay.
* All288 threads keep CTA joins. Every CTA tid0 adds/waits grid ctr; after
  the wait CTA0 tid0..63 resets ctr[8:72], tid0..127 clears next ss bank,
  tid0 increments generation. Combine c=0..63 at m16, tid0..255, reads
  all8 expert slots and writes4 columns/thread; tid256..287 alone break.
  Entire job-end/B/cleanup/combine suffix is byte-identical. Dummy word is
  write-only, has one writer (tid256), and needs no reset or publication.
"""
import ast
import ctypes
import pathlib
import re
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
OUT = ROOT / "kernels/lab/k3_vmsh_ab.py"
LIMIT = 1_000_000
GLOBALS = '''# VMSH LAB: 1=HRED, 2=HSWI, 4=HADD150, 8=HADD75; timing only.
VMSH_MODE = 0
_VMSH = {"calls": {}}
'''
HOST = '''    mode = VMSH_MODE if m == 16 else 0
    if type(mode) is not int or not 0 <= mode <= 15:
        raise ValueError("VMSH_MODE must be int in 0..15")
'''
COUNT = '''    if m == 16:
        key = f"T{m}:{mode}"
        _VMSH["calls"][key] = _VMSH["calls"].get(key, 0) + 1
'''


def sub(s, old, new, what):
    assert s.count(old) == 1, f"{what}: expected one anchor, got {s.count(old)}"
    return s.replace(old, new, 1)


def node(s, name):
    matches = [n for n in ast.parse(s).body if
               (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name) or
               (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))]
    assert len(matches) == 1, f"unique definition: {name}"
    return matches[0]


def literal(s, name):
    return ast.literal_eval(node(s, name).value)


def burst(n):
    """Exactly n live PTX ALUs: independent leaves, balanced XOR tree to salt.

    All leaves depend only on runtime salt, not each other or pipeline state.
    Nonlinear ADD carries prevent folding the tree to an XOR of constants.
    Every leaf contributes once. No accumulator-chain throughput confound.
    """
    assert n in (75, 150)
    leaves = n * 2 // 3
    ops = [f"add.u32 h{i},%1,0x{0x13579 + n * 1009 + i * 0x10203:x};" for i in range(leaves)]
    live, depths = list(range(leaves)), {i: 1 for i in range(leaves)}
    used = {i: {i} for i in live}
    while len(live) > 2:
        nxt = []
        for i in range(0, len(live), 3):
            group = live[i:i + 3]
            if len(group) == 3:
                a, b, c = group
                assert not (used[a] & used[b] or used[a] & used[c] or used[b] & used[c])
                ops.append(f"lop3.b32 h{a},h{a},h{b},h{c},0x96;")
                depths[a] = 1 + max(depths[x] for x in group)
                used[a] = used[a] | used[b] | used[c]
                nxt.append(a)
            else:
                nxt.extend(group)
        live = nxt
    a, b = live
    ops.append(f"lop3.b32 %0,h{a},h{b},%0,0x96;")
    assert len(ops) == n and used[a] | used[b] == set(range(leaves))
    assert max(depths[a], depths[b]) + 1 <= 6
    # Compact adjacent string literals keep the generated submission below 1 MB.
    strings = "\n".join('"' + "".join(ops[i:i + 5]) + '"' for i in range(0, len(ops), 5))
    return (f'asm volatile("{{ .reg .b32 h<{leaves}>;"\n' + strings +
            '\n"}" : "+r"(vmsh_z) : "r"(vmsh_seed) : "memory");\n')


def patch_cuda_host(base, cu):
    original, edits_cu = cu, []

    def change(old, new, label):
        nonlocal cu
        cu = sub(cu, old, new, label)
        edits_cu.append((old, new, label))

    tail = "__nv_bfloat16* out; long long* marks; int m; float eps; int lo;\n};"
    change(tail, tail.replace("int lo;", "int lo;\n#if VMS_T16\nint flags; // VMSH: former tail padding\n#endif") +
           '\n#if VMS_T16\nstatic_assert(sizeof(Params)==128 && __builtin_offsetof(Params,flags)==124, "VMSH ABI");\n#endif', "Params")
    wait = "      mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));\n"
    add = '''#if VMS_T16
// VMSH HADD: warp-uniform, no pipeline inputs; write-only padding sink.
static_assert(RING_OFF >= sizeof(Smem)+sizeof(uint32_t), "VMSH padding sink");
if (p.flags & 12) {
const uint32_t vmsh_seed = (uint32_t)p.flags >> 20;
uint32_t vmsh_z = vmsh_seed;
if (p.flags & 4) {
''' + burst(150) + '''}
if (p.flags & 8) {
''' + burst(75) + '''}
if (lane == 0) asm volatile("st.volatile.shared.b32 [%0],%1;" :: "r"(su32(smem+sizeof(Smem))), "r"(vmsh_z) : "memory");
}
#endif
'''
    change(wait, wait + add, "HADD150/HADD75")
    red = "#pragma unroll\nfor (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }"
    change(red, "#if VMS_T16\nif (!(p.flags & 1)) {\n#endif\n" + red + "\n#if VMS_T16\n}\n#endif", "HRED cold")
    red2 = "#pragma unroll\nfor (int w = 0; w < NW; ++w) { const float4 u = S.red2[bq][w][lane]; sm2.x += u.x; sm2.y += u.y; sm2.z += u.z; sm2.w += u.w; }"
    # Already enclosed in the base's #if VMS_T16, including the if(two) guard.
    change(red2, "if (p.flags & 1) sm2 = S.red2[bq][0][lane];\nelse {\n" + red2 + "\n}", "HRED hot")
    for val in ("sm", "sm2"):
        math = (f"const float ga = (h ? {val}.y : {val}.x) * gs, up = (h ? {val}.w : {val}.z) * gs;\n"
                "const float sv = __fdividef(ga, 1.f + __expf(-ga)) * up;\n"
                "p.inter[(size_t)(st + tk) * ID + rb * 8 + g] = __float2bfloat16(sv);")
        standin = f"p.inter[(size_t)(st + tk) * ID + rb * 8 + g] = __float2bfloat16(h ? {val}.y : {val}.x);"
        change(math, "#if VMS_T16\nif (p.flags & 2) {\n" + standin + "\n} else\n#endif\n{\n" + math + "\n}", "HSWI " + val)
    transport = "#if VMS_T16\np.flags = (int)((pdl >> 8) & 0x3fffffff); // VMSH mode/runtime salt\n#endif\n"
    change("p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo;\n",
           "p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo;\n" + transport, "transport")
    pdl_old = "programmaticStreamSerializationAllowed = pdl ? 1 : 0;"
    pdl_new = "programmaticStreamSerializationAllowed = (pdl & 1) ? 1 : 0;"
    change(pdl_old, pdl_new, "PDL bit")
    # Byte equality of every executed compute expression, barrier and refill.
    start, stop = "if (warp < NW) {", "} else {\nconst int g = lane >> 2, t = lane & 3;"
    assert cu[cu.index(start):cu.index(stop)] == original[original.index(start):original.index(stop)]
    suffix = "if (q == nA - 1 || ((v + 1) >> 9) != j) {"
    assert cu[cu.index(suffix):].replace(transport, "").replace(pdl_new, pdl_old) == original[original.index(suffix):]
    protocol = r"^.*(?:mbar_|cta_sync|__syncwarp|__syncthreads|__threadfence|fence\.|atomic|issue\(|return|break|continue).*$"
    assert re.findall(protocol, cu, re.M) == re.findall(protocol, original, re.M)
    call = ast.get_source_segment(base, node(base, "_vms_call"))
    call2 = sub(call, '    md = vs["mod_hi"]', HOST + '    md = vs["mod_hi"]', "host flags")
    call2 = sub(call2, "K2A_POST_NORM_EPS, 1, 0)", "K2A_POST_NORM_EPS, 1 | (mode << 8), 0)", "host transport")
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
    for old, new, label in edits_cu:
        pos = s.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""') + cu.index(new)
        lo = s[:pos].count("\n") + 1
        print(f"[vmsh] changed {label}: k3_vmsh_ab.py:{lo}-{lo + new.count(chr(10))}")
    return s


def build(base):
    assert "VMSH" not in base
    cu = literal(base, "VMS_CUDA_SRC")
    assert "constexpr int NW = 8, NT = NW * 32, NTL = NT + 32" in cu
    params = re.search(r"struct Params \{(.*?)\n\};", cu, re.S).group(1)
    fields = []
    for decl in params.split(";"):
        decl = decl.strip()
        if decl:
            typ = ctypes.c_void_p if "*" in decl else ctypes.c_float if decl.startswith("float ") else ctypes.c_int
            fields.append((decl.split()[-1].lstrip("*"), typ))
    before = type("BaseParams", (ctypes.Structure,), {"_fields_": fields})
    after = type("LabParams", (ctypes.Structure,), {"_fields_": fields + [("flags", ctypes.c_int)]})
    assert ctypes.sizeof(before) == ctypes.sizeof(after) == 128 and after.flags.offset == 124
    assert all(getattr(before, k).offset == getattr(after, k).offset for k, _ in fields)
    return patch_cuda_host(base, cu)


def dry_run(base, patched):
    """Execute the real state construction/route, with CPU sentinel tensors.

    Adapted from vmsd's proven padz-derived harness, not a fabricated state map.
    No new prepare-side path/bank exists; all original prepare shapes exercise
    the new host validation/dispatch and default-zero launch argument.
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
    runner = (ROOT / "tools/box/vmsh_run.py").read_text(encoding="utf-8")
    run_ns = dict(statistics=__import__("statistics"), math=__import__("math"), copy=__import__("copy"),
                  NAMES={2: "d16s", 3: "d16f"})
    wanted = ("ProbeFailure", "require_vms", "calls", "require_live", "assess", "cpu_gate", "energy_schedule", "energy_report")
    rn = [n for n in ast.parse(runner).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    rn += [node(runner, n) for n in ("SHAPES", "ARMS", "NULLS")]
    exec(compile(ast.Module(body=rn, type_ignores=[]), "<runner CPU paths>", "exec"), run_ns)
    proxy = types.SimpleNamespace(**ns)
    run_ns["require_vms"](proxy, state)
    run_ns["calls"](proxy)
    base_call = compile(ast.Module(body=[node(base, "_vms_call")], type_ignores=[]), "<base call>", "exec")
    for m in (1, 8, 9, 12, 16):
        for mode in range(16):
            ns["VMSH_MODE"] = mode
            x, ids, tw = Tensor((m, 4096)), Tensor(), Tensor()
            previous = dict(ns["_VMSH"]["calls"])
            ns["_vms_call"](state["k2a33"]["st"], x, ids, tw)
            args = (hi if m > 8 else mod).launches[-1]
            flags = mode if m == 16 else 0
            assert args[-2] == 1 | (flags << 8)
            assert (args[-2] >> 8) & 0x3fffffff == flags and flags >> 4 == 0
            if m != 16:
                assert ns["_VMSH"]["calls"] == previous
            if flags == 0:
                original = dict(ns)
                exec(base_call, original)
                original["_vms_call"](state["k2a33"]["st"], x, ids, tw)
                old = (hi if m > 8 else mod).launches[-1]
                assert args[:12] == old[:12] and args[13:] == old[13:] and args[12].shape == old[12].shape
    ns["_VMSH"]["calls"].clear()
    for arm, settings in run_ns["ARMS"]:
        ns.update(settings)
        for shape in run_ns["SHAPES"]:
            n = len(hi.launches)
            ns["_v11_k2a33_call"](Tensor((16,4096)),Tensor(),Tensor(),state["k2a33"]["st"])
            assert len(hi.launches) == n+1 and hi.launches[-1][-2] == 1 | (settings["VMSH_MODE"] << 8)
    run_ns["require_live"](ns["_VMSH"]["calls"], "CPU dry run")
    assert set(ns["_VMSH"]["calls"]) == {"T16:0", "T16:1", "T16:2", "T16:3", "T16:4", "T16:8"}
    ns["_v11_k2a33_call"](Tensor((8,4096)),Tensor(),Tensor(),state["k2a33"]["st"])
    assert routed == [8]
    ns["VMSH_MODE"] = 0
    state["k2a33"]["st"]["vms"]["mod_hi"] = None
    ns["_vms_call"](state["k2a33"]["st"], Tensor((16,4096)), Tensor(), Tensor())
    assert mod.launches[-1][-4] == 16
    try: run_ns["require_vms"](proxy, state)
    except run_ns["ProbeFailure"]: pass
    else: raise AssertionError("runner accepted missing hi build")
    for bad in (-1, 16, 256, True, 1.0, None):
        ns["VMSH_MODE"] = bad
        try: ns["_vms_call"](state["k2a33"]["st"],Tensor((16,4096)),Tensor(),Tensor())
        except ValueError: pass
        else: raise AssertionError(f"bad mode accepted: {bad}")
        try: run_ns["calls"](types.SimpleNamespace(**ns))
        except run_ns["ProbeFailure"]: pass
        else: raise AssertionError("runner accepted invalid flags")
    run_ns["cpu_gate"]()
    print("[vmsh] CPU dry run PASS: real state, 80 mode/shape cases, flag-off arguments, all arms, report thresholds")


def main():
    import sys
    if len(sys.argv) != 1:
        raise SystemExit("usage: python -B -I tools/vmsh_patches.py (lab only)")
    raw = BASE.read_bytes()
    base = raw.decode("utf-8").replace("\r\n", "\n")
    source = build(base)
    dry_run(base, source)
    out = source.replace("\n", "\r\n").encode() if b"\r\n" in raw else source.encode()
    assert len(out) <= LIMIT, f"over cap: {len(out)}"
    OUT.write_bytes(out)
    print(f"[vmsh] {OUT.relative_to(ROOT)}: {len(out):,} B; headroom {LIMIT-len(out):,} B")
    print("[vmsh] AST, ABI, exact compute/suffix/protocol, reverse-edit gates PASS")


if __name__ == "__main__":
    main()

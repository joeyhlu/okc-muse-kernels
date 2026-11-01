"""R-vmsn-1: VMS compute-warp null dose response, LAB ONLY.

Run: python -B -I tools/vmsn_patches.py (generator + CPU dry run, no GPU).
Bits 0..7 null compute warps 0..7; helper warp 8 is never nulled. The job-change
xr[16] / xr2 / x2s loads STAY: this probe excludes x-fragment loading, but skips
fragment USE (including x2s reads), weight/scale LDS, dequant and mma.sync.

VMS has no flags word. Params has 4 bytes of trailing padding: add flags there
(offset 124, sizeof stays 128). Transport via bits 8..15 of the existing int64
pdl argument; bit 0 retains PDL. No function signature or old field moves;
mask zero passes precisely the old Python launch arguments. Lab-only encoding.

Kill risks: consumer work may overlap the stream/helper; a changed register
allocation may dominate the probe. This is an intentionally incorrect-output
upper-bound experiment, not a speedup claim. The compile gate rejects resource
changes; paired controls and the registered VOID rule qualify the measurement.

Per unit, EST source accounting (not SASS): HBM 36,864 -> 36,864 B. Per compute
thread weight/scale LDS 9 x 16 = 144 -> 0 B when nulled; hot T16 adds 16 x 16
= 256 -> 0 B x2s USE. Kept job-change x loads are up to 256/512 B per thread.
Active path: 9 LDS.128, 16 scd calls (PRMT + IMAD), 32 words x [8 unpack ops
+ 4 bf16x2 MUL + 1/2 MMA] = 457/489 core instructions, plus hot T16's 16 LDS
= 505. These are source/PTX operation estimates, excluding address/move/reduce
instructions. Null path has 0 of those operations; one warp-uniform decision
per unit. Initial accumulators, reductions of those zeros and stores stay.
Empty volatile register uses keep the original x job loads on both arms;
they emit no PTX instructions, and the compile gate checks resource effects.

Protocol audit, for EVERY CTA c=0..G-1, independent of mask:
* tid 0 initializes full[NS] (count 1), pfull[NRB] (8), pempty[NRB] (1).
* tid 0 issues each UB-byte full transaction; tid 0..255 wait each full slot.
* bar.sync 1 has 256 arrivals/unit. ER: both branch arms execute the SAME
  named-barrier ID/count once; tid 0 reissues q+NS in whichever arm owns it.
  Non-ER: the original common post-compute barrier/refill is untouched.
* tid 0..255 wait pempty; all store red[slot][warp][lane] (and red2 if two).
  All 32 lanes join; tid 32*w (w=0..7) arrives pfull: exactly 8 arrivals.
* helper tid 256..287 waits pfull, reduces all 8 warps, joins; tid 256 arrives
  pempty once. All helper A stores/fence/join, tid 256 doneA atomic, B ybuf/ss
  loops and final helper fence remain byte-for-byte unchanged.
* All 288 threads reach all CTA joins. Every CTA tid 0 adds/waits grid ctr.
  After grid wait: CTA 0 tid 0..63 resets ctr[8:72], tid 0..127 clears next ss
  bank, tid 0 increments generation. Combine: CTAs c iterate it=c;it<m*4;it+=G,
  tid 0..255 each writes 4 columns, all 8 expert slots; helper alone breaks.
No new return/break/continue, no CTA or thread retires. N=0 still runs cleanup.
"""
import ast
import ctypes
import pathlib
import re
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
OUT = ROOT / "kernels/lab/k3_vmsn_ab.py"
LIMIT = 1_000_000
GLOBALS = '''# VMSN LAB: xr/xr2 job loads stay; bits 0..7 null compute warps, never helper 8.
VMSN_MASK = 0
_VMSN = {"calls": {}}
'''
HOST = '''    mask = VMSN_MASK if m in (8, 16) else 0
    if type(mask) is not int or not 0 <= mask <= 255:
        raise ValueError("VMSN_MASK must be an int in 0..255")
'''
COUNT = '''    if m in (8, 16):
        key = f"T{m}:{mask}"
        _VMSN["calls"][key] = _VMSN["calls"].get(key, 0) + 1
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


def build(base):
    assert "VMSN" not in base
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

    begin, end = "const unsigned char* slot = ring + sl * UB;\n", "float4 o;\n"
    i, j = cu.index(begin), cu.index(end, cu.index(begin))
    body = cu[i:j]
    init = '''float acc[4][4];
#pragma unroll
for (int s = 0; s < 4; ++s) { acc[s][0] = 0.f; acc[s][1] = 0.f; acc[s][2] = 0.f; acc[s][3] = 0.f; }
'''
    o2 = "float4 o2 = make_float4(0.f, 0.f, 0.f, 0.f);\n"
    refill_start = body.index("cta_sync_compute();\n")
    refill_end = body.index("#endif\nfloat acc", refill_start)
    refill = body[refill_start:refill_end]
    assert refill.count("cta_sync_compute();") == refill.count("issue(unit_src(") == 1
    normal = sub(sub(body, init, "", "move zero acc init"), o2, "", "move zero o2 init")
    null = "#if VMS_ER && VMS_RO && !VMS_HI\n" + refill + "#endif\n"
    replacement = (init + "#if VMS_T16\n" + o2 + "#endif\n"
                   "// VMSN: one warp-uniform decision/unit; x job loads above are retained.\n"
                   "if ((p.flags >> warp) & 1) {\n" + null + "} else {\n" + normal + "}\n")
    assert not any(x in null for x in ("return", "continue", "break", "xr[", "T16X2", "mma(", "slot +"))
    tail = "__nv_bfloat16* out; long long* marks; int m; float eps; int lo;\n};"
    new_tail = tail.replace("int lo;", "int lo; int flags;  // VMSN uses the former tail padding") + '''
static_assert(sizeof(Params) == 128 && __builtin_offsetof(Params, flags) == 124, "VMSN ABI");'''
    cu2 = sub(cu, tail, new_tail, "Params padding")
    cu2 = sub(cu2, body, replacement, "null consumer body")
    # A source-level load before the branch is not enough: an optimizer could
    # sink/delete xr loads for null warps. Empty volatile uses pin the original
    # job-change loads without adding PTX instructions or reading weight slots.
    x_end = """}
#endif
}
#if VMS_MK2
ck = clock64();
#endif
      mbar_wait(&S.full[sl], par);"""
    keep_x = """// VMSN: keep job-change x loads observable even for a null warp.
#pragma unroll
for (int cc = 0; cc < 16; ++cc) {
asm volatile("" :: "r"(xr[cc].x), "r"(xr[cc].y), "r"(xr[cc].z), "r"(xr[cc].w));
}
"""
    cu2 = sub(cu2, x_end, x_end.replace("}\n#if VMS_MK2", keep_x + "}\n#if VMS_MK2", 1), "retain x job loads")
    x2_loads = """T16X2(2 * cc) = PB ? __ldcg(src2 + cc * 8) : __ldg(src2 + cc * 8);
T16X2(2 * cc + 1) = PB ? __ldcg(src2 + cc * 8 + 1) : __ldg(src2 + cc * 8 + 1);"""
    x2_kept = """const uint4 vmsn_x0 = PB ? __ldcg(src2 + cc * 8) : __ldg(src2 + cc * 8);
const uint4 vmsn_x1 = PB ? __ldcg(src2 + cc * 8 + 1) : __ldg(src2 + cc * 8 + 1);
asm volatile("" :: "r"(vmsn_x0.x), "r"(vmsn_x0.y), "r"(vmsn_x0.z), "r"(vmsn_x0.w));
asm volatile("" :: "r"(vmsn_x1.x), "r"(vmsn_x1.y), "r"(vmsn_x1.z), "r"(vmsn_x1.w));
T16X2(2 * cc) = vmsn_x0;
T16X2(2 * cc + 1) = vmsn_x1;"""
    cu2 = sub(cu2, x2_loads, x2_kept, "retain xr2/x2s job loads without extra shared reads")
    cu2 = sub(cu2, "p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo;\n",
              "p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo;\np.flags = (int)((pdl >> 8) & 255);  // VMSN lab transport\n", "mask transport")
    cu2 = sub(cu2, "programmaticStreamSerializationAllowed = pdl ? 1 : 0;",
              "programmaticStreamSerializationAllowed = (pdl & 1) ? 1 : 0;", "keep PDL bit")
    # Only duplicate the early named barrier/refill. Everything after compute,
    # including ALL cleanup, is unedited; all x-loading remains before branch.
    assert cu2[cu2.index(end):].replace("p.flags = (int)((pdl >> 8) & 255);  // VMSN lab transport\n", "").replace(
        "programmaticStreamSerializationAllowed = (pdl & 1) ? 1 : 0;", "programmaticStreamSerializationAllowed = pdl ? 1 : 0;") == cu[j:]
    assert literal(base, "VMS_CPP_SRC").count("int64_t pdl, int64_t lo);") == 1
    cu_anchor = 'VMS_CUDA_SRC = r"""' + cu + '"""'
    call = ast.get_source_segment(base, node(base, "_vms_call"))
    call2 = sub(call, '    md = vs["mod_hi"]', HOST + '    md = vs["mod_hi"]', "host mask")
    call2 = sub(call2, "K2A_POST_NORM_EPS, 1, 0)", "K2A_POST_NORM_EPS, 1 | (mask << 8), 0)", "host pdl")
    call2 = sub(call2, "    return out", COUNT + "    return out", "successful launches counter")
    edits = [(cu_anchor, 'VMS_CUDA_SRC = r"""' + cu2 + '"""', "CUDA"),
             ("VMS_ENABLE = True\n", GLOBALS + "VMS_ENABLE = True\n", "globals"),
             (call, call2, "_vms_call")]
    s = base
    for old, new, label in edits:
        s = sub(s, old, new, label)
    ast.parse(s)
    restored = s
    for old, new, label in reversed(edits):
        restored = sub(restored, new, old, "undo " + label)
    assert restored == base
    return s


def dry_run(base, patched):
    """Execute real AST-selected host functions with CPU tensor/module sentinels.

    No invented state dictionary: execute the base's st['vms'] construction and
    its return {'st': st, 'ok': ok}; preserve the real layer outer key. All new
    host paths run, including prepare's zero-mask calls, hi fallback, bad masks,
    runner route pin, mask liveness checks and report-only analysis boundaries.
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
    # Prove the base's T8 interception, then apply the runner's exact route pin.
    ns["_v11_k2a33_call"](Tensor((8, 4096)), Tensor(), Tensor(), state["k2a33"]["st"])
    assert routed == [8] and not mod.launches
    runner = (ROOT / "tools/box/vmsn_run.py").read_text(encoding="utf-8")
    run_ns = dict(statistics=__import__("statistics"), math=__import__("math"), NAMES={2:"d16s",3:"d16f",6:"v1s",7:"v1f"})
    wanted = ("ProbeFailure", "configure_vms", "require_vms", "calls", "require_live", "assess", "cpu_gate")
    rn = [n for n in ast.parse(runner).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    rn += [node(runner, n) for n in ("SHAPES", "ARMS", "NULLS")]
    exec(compile(ast.Module(body=rn, type_ignores=[]), "<runner CPU paths>", "exec"), run_ns)
    proxy = types.SimpleNamespace(**ns)
    run_ns["configure_vms"](proxy)
    ns["HYBLO_T"] = proxy.HYBLO_T
    assert ns["HYBLO_T"] == (1,)
    run_ns["require_vms"](proxy, state)
    for m in (1, 8, 9, 16):
        for mask in range(256):
            ns["VMSN_MASK"] = mask
            x, ids, tw = Tensor((m, 4096)), Tensor(), Tensor()
            ns["_vms_call"](state["k2a33"]["st"], x, ids, tw)
            args = (hi if m > 8 else mod).launches[-1]
            assert args[-2] == 1 | ((mask if m in (8, 16) else 0) << 8)
            if mask == 0:
                original = dict(ns)
                exec(compile(ast.Module(body=[node(base, "_vms_call")], type_ignores=[]), "<base call>", "exec"), original)
                original["_vms_call"](state["k2a33"]["st"], x, ids, tw)
                old = (hi if m > 8 else mod).launches[-1]
                assert args[:12] == old[:12] and args[13:] == old[13:] and args[12].shape == old[12].shape
    for arm, settings in run_ns["ARMS"]:
        ns.update(settings)
        for m in (8, 16):
            md = hi if m == 16 else mod
            n = len(md.launches)
            ns["_v11_k2a33_call"](Tensor((m, 4096)), Tensor(), Tensor(), state["k2a33"]["st"])
            assert len(md.launches) == n + 1 and md.launches[-1][-2] == 1 | (settings["VMSN_MASK"] << 8)
    assert routed == [8]
    run_ns["require_live"](ns["_VMSN"]["calls"], "CPU dry run")
    ns["VMSN_MASK"] = 0
    state["k2a33"]["st"]["vms"]["mod_hi"] = None
    ns["_vms_call"](state["k2a33"]["st"], Tensor((16, 4096)), Tensor(), Tensor())
    assert mod.launches[-1][-4] == 16
    try:
        run_ns["require_vms"](proxy, state)
    except run_ns["ProbeFailure"]:
        pass
    else:
        raise AssertionError("runner accepted missing hi build")
    for bad in (-1, 256, True, 1.0):
        ns["VMSN_MASK"] = bad
        try:
            ns["_vms_call"](state["k2a33"]["st"], Tensor((8, 4096)), Tensor(), Tensor())
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad mask accepted: {bad}")
    run_ns["cpu_gate"]()
    print("[vmsn] CPU dry run PASS: real state construction, 1024 mask/shape cases, both modules, every arm, report gates")


def main():
    import sys
    if len(sys.argv) != 1:
        raise SystemExit("usage: python -B -I tools/vmsn_patches.py (lab only)")
    raw = BASE.read_bytes()
    base = raw.decode("utf-8").replace("\r\n", "\n")
    source = build(base)
    dry_run(base, source)
    out = source.replace("\n", "\r\n").encode() if b"\r\n" in raw else source.encode()
    OUT.write_bytes(out)
    print(f"[vmsn] {OUT.relative_to(ROOT)}: {len(out):,} B; over cap {max(0, len(out)-LIMIT):,} B")
    print("[vmsn] AST, ABI padding/offsets, protocol projection, exact reverse-edit gates PASS")


if __name__ == "__main__":
    main()

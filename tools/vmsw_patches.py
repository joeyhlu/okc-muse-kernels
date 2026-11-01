"""Task AA / R-vmsw-2: separate role code from dynamic-smem reservation.

Kill risks FIRST: rs0 may retain the compiler's role-scheduling penalty;
m4 may leave PDL admission unchanged (the +4 KB hypothesis then fails).
MEASURED lab R-vmsw-1: d1 H=+1.28, rs H=-47.5, e H=-50.9 us.
These are diagnostic controls, not saving forecasts: no bandwidth/edge
arithmetic explains a 47.5 us admission/code penalty without the paired run.
rs0=4 is rs=2 with no fragB allocation; m4=5 is d1=1 with a bare
+4096 host dynamic-smem request, never an accessed member or changed offset.
Both keep 36864 HBM B/unit and 36864/102400 shared B/unit (cold/hot);
432/464 arithmetic ops/lane/unit before=after; role dispatch is unchanged
from rs, and m4 must have byte-identical kernel SASS to d1 (compile gate).
No protocol change: all CTA/thread coverage below remains identical.

Original Task W accounting/proof, retained for buildable rs/e:

Kill risks FIRST: compiler merging/serializing roles or >168 regs/spills;
extra shared traffic / recipient issue / +4096 B reservation erase the gain.
MEASURED lab:1187 S0=3.87 us, W0=2.94, W4=3.34; :1193 D1 H=1.31/1.40;
:1194 E_rec H=0.17 (noise), E_don is invalid runtime-branch evidence.
EST opportunity only: 2*(50 DQ+1 weight LDS-4 fragment LDS)=94 slots;
94*0.042=3.948 us gross (small-cut D1 calibration), capped by S0's 3.87.
This is NOT a predicted net saving: wait/address/dispatch and saturation
require SASS and paired e-vs-rs timing. No GPU work occurs in this generator.

Ledger per lane/unit, cold/hot, excluding addresses/control (SASS pending):
D1 compute=8*(2 scale + 4*(8 INT+4 HMUL2+1/2 HMMA))=432/464 ops.
E donor=382/414 (-50 DQ), plus 4 LDS.128 and one bounded wait;
E recipient=482/514 (+50 DQ), plus 2 LDS, 4 STS.128, syncwarp, lane0 arrive.
HMMA stays 32/64 per compute warp. D1/rs own weight/scale LDS=144 B/lane;
E donor=128 B +64 B fragment LDS, recipient=164 B +64 B fragment STS.
CTA weight HBM=36864 B unchanged. Shared cold/hot D1/rs=36864/102400 B;
E=45312/110848 B (+256 duplicate scale +4096 STS +4096 LDS).
Donor sc.w remains necessary for cc6; cc7 scale DECODE and wv[7] are removed.

Protocol proof: q is CTA-local unit ordinal starting at zero, not sl or bq.
frdy is init(count=2,phase=0) before the existing init fence/CTA barrier.
REC raw LDS precede bar.sync 1,256, so ER refill cannot overwrite them.
REC DQ/STS follow that sync; syncwarp then lane0 release-arrive publishes.
Both donors acquire-wait(q&1) at cc7 then consume four uint4 fragments.
No next-unit production crosses its bar.sync until BOTH donors consumed q.
Two arrivals per q toggle phase once; q parity survives NS/NRB wrap and job
boundaries. N=0 skips ALL unit roles; N=1/short/final units need no drain.
No new early exit: timeout sets p.ctr[2]=3 (actual VMS error accessor) then
continues all existing publication/cleanup; it cannot be accepted as correct.
Each lane waits independently; the wait contains NO warp collective.
All 132 CTAs: compute tids 0..255 publish red/red2 after pempty, each warp's
lane0 arrives pfull; helper tids 256..287 reduce all 8 warps, lane0 returns
pempty and publishes doneA. All 288 threads reach both final CTA barriers;
tid0 in every CTA publishes/waits grid count. CTA0 tids0..63 reset ctr[8:72],
tids0..127 clear next ss bank, tid0 advances generation. CTA c, tids0..255,
owns it=c;it<m*4;it+=132 and combines all eight routes. Suffix is byte-identical.
Smem: no safe 4 KB slack (red/red2 live NRB ring; x2s live through job;
pos live until combine; jC live to helper; dead base only 512 B). Both rs/e
reserve 4096 B so e-vs-rs compares equal allocation; frdy uses ring-8 within
the unchanged 16 B alignment gap. Total 178560 B (+4096), co-residency risk.
"""
import ast
import contextlib
import hashlib
import io
import pathlib
import re
import sys
import time
from types import SimpleNamespace as NS

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
OUT = ROOT / "kernels/lab/k3_vmsw_ab.py"
VARIANTS = {"d1": 1, "rs": 2, "e": 3, "rs0": 4, "m4": 5}
SMEM = {0: 174464, 1: 174464, 2: 178560, 3: 178560, 4: 174464, 5: 178560}


def node(s, name):
    n, = [n for n in ast.parse(s).body if
          (isinstance(n, ast.FunctionDef) and n.name == name) or
          (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))]
    return n


def literal(s, name):
    return ast.literal_eval(node(s, name).value)


def sub(s, old, new, count=1):
    assert s.count(old) == count, (old[:100], s.count(old), count)
    return s.replace(old, new)


# Verbatim D1 replacement blocks from vmsx (no dependency on its evolving bank).
D1_HELPER = '''template <bool H>
__device__ __forceinline__ uint32_t bmulh(uint32_t a, uint32_t b) {
const __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&b);
const __nv_bfloat162 scale = H ? __high2bfloat162(v) : __low2bfloat162(v);
const __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), scale);
return *reinterpret_cast<const uint32_t*>(&r);
}
'''
SCALE = "const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);\n"
MERGED = "const uint32_t s2 = scd(sw, (cc & 1) ? 0x4342u : 0x4140u, k16);\n"
MUL = "const uint32_t A0 = bmul(r0, sg), A1 = bmul(r1, sg8), A2 = bmul(r2, sg), A3 = bmul(r3, sg8);\n"
MULH = "const uint32_t A0 = bmulh<false>(r0, s2), A1 = bmulh<true>(r1, s2), A2 = bmulh<false>(r2, s2), A3 = bmulh<true>(r3, s2);\n"
COLD = "mma(acc[s], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);\n"
COLD_MMA = "mma(acc[s], A0, A1, A2, A3, bb[2 * s], bb[2 * s + 1]);\n"
HOT_MMA = "mma(acc[s & 1], A0, A1, A2, A3, bb[2 * s], bb[2 * s + 1]);\nmma(acc[2 + (s & 1)], A0, A1, A2, A3, b2[2 * s], b2[2 * s + 1]);\n"
UNPACK = '''const uint32_t qw = ww[s];
const uint32_t M = 0x81C081C0u;
const uint32_t r0 = qw & M;
const uint32_t r1 = __funnelshift_l(qw, qw, 10) & M;
const uint32_t r2 = __funnelshift_l(qw, qw, 13) & M;
const uint32_t r3 = ((qw & 0x001B001Bu) * unp_ok32(0x1084u)) & M;
'''
ROLE_HELPER = r'''#if VMS_W >= 2 && VMS_W <= 4
template<int C> struct VmswRole { static constexpr int value = C; };
static_assert(RING_OFF-sizeof(Smem)>=8 && SMEM_BYTES==(VMS_W==4 ? 174464 : 178560), "VMSW layout");
__device__ __forceinline__ void vmsw_wait(uint64_t* b, uint32_t phase, int* err) {
int n=0;
while (true) {
unsigned ready;
asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared.b64 p,[%1],%2,64; selp.u32 %0,1,0,p; }"
 : "=r"(ready) : "r"(su32(b)),"r"(phase) : "memory");
if (ready) break;
if (++n > WAIT_MAX) { atomicExch(err,3); break; }
__nanosleep(64);
}
}
#endif
'''


def role_body(d1):
    """Copy actual hot/cold arithmetic; split LAST=7 from cc7, no inner guards."""
    start = d1.index("const unsigned char* slot = ring + sl * UB;\n")
    end = d1.index("float4 o;\n", start)
    unit = d1[start:end]
    hot_start = unit.index("#pragma unroll\nfor (int cc = 0; cc < 8; ++cc) {")
    hot_end = unit.index("o2.x =", hot_start)
    hot = unit[hot_start:hot_end]
    cold_start = unit.index("#pragma unroll\nfor (int cc = 0; cc < 8; ++cc) {", hot_end)
    cold_end = unit.index("#if VMS_T16 && VMS_T16E", cold_start)
    cold = unit[cold_start:cold_end]
    post_hot = unit[hot_end:unit.index("} else\n", hot_end)]
    preload = unit[:unit.index("float acc[4][4];")]
    preload = sub(preload, "uint4 wv[8];", "uint4 wv[LAST];")
    preload = sub(preload, "cc < 8;", "cc < LAST;")
    preload = sub(preload, "cta_sync_compute();", '''uint4 rw; uint32_t rsw;
if constexpr (VMS_W==3 && C==2) {
const int donor = (warp-1)*4;
rw = *reinterpret_cast<const uint4*>(slot + (donor*8+7)*512 + lane*16);
rsw = *reinterpret_cast<const uint32_t*>(slot + UC + donor*512 + lane*16 + 12);
}
cta_sync_compute();''')
    production = '''#if VMS_W==3
if constexpr (C==2) {
const uint32_t s2 = scd(rsw,0x4342u,k16);
const uint32_t ww[4] = {rw.x,rw.y,rw.z,rw.w};
#pragma unroll
for (int s=0; s<4; ++s) {
''' + UNPACK + MULH + '''S.fragB[warp-1][lane][s] = make_uint4(A0,A1,A2,A3);
}
__syncwarp();
if (lane==0) mbar_arrive(reinterpret_cast<uint64_t*>(ring-8));
}
#endif
'''
    # Pre-register x B operands identical to the actual cc7 block.
    bb = '''const uint32_t bb[8] = {xr[14].x,xr[14].y,xr[14].z,xr[14].w,
xr[15].x,xr[15].y,xr[15].z,xr[15].w};
'''
    b2 = '''const uint4 y0=T16X2(14), y1=T16X2(15);
const uint32_t b2[8] = {y0.x,y0.y,y0.z,y0.w,y1.x,y1.y,y1.z,y1.w};
'''
    def consume(hot):
        return '''#if VMS_W==3
if constexpr (C==1) {
vmsw_wait(reinterpret_cast<uint64_t*>(ring-8),uint32_t(q&1),p.ctr+2);
''' + bb + (b2 if hot else "") + '''#pragma unroll
for (int s=0; s<4; ++s) {
const uint4 f = S.fragB[warp/4][lane][s];
const uint32_t A0=f.x,A1=f.y,A2=f.z,A3=f.w;
''' + (HOT_MMA if hot else COLD_MMA) + "}\n}\n#endif\n"
    body = '''float acc[4][4];
float4 o2;
auto vmsw_unit = [&](auto tag) {
constexpr int C = decltype(tag)::value;
constexpr int LAST = VMS_W==3 && C==1 ? 7 : 8;
asm volatile("// VMSW_ROLE_%0" :: "n"(C));
''' + preload + production + '''#pragma unroll
for (int s=0; s<4; ++s) { acc[s][0]=0.f; acc[s][1]=0.f; acc[s][2]=0.f; acc[s][3]=0.f; }
o2=make_float4(0.f,0.f,0.f,0.f);
if (two) {
''' + hot.replace("cc < 8;", "cc < LAST;") + consume(True) + post_hot + "} else {\n" + cold.replace("cc < 8;", "cc < LAST;") + consume(False) + '''}
};
if (warp==0 || warp==4) vmsw_unit(VmswRole<1>{});
else if (warp==1 || warp==2) vmsw_unit(VmswRole<2>{});
else vmsw_unit(VmswRole<0>{});
'''
    return unit, body


def source_recipe(base):
    """Small runtime construction; original CUDA literal/stock builds stay exact."""
    cu = literal(base, "VMS_CUDA_SRC")
    kernel = "__global__ void __launch_bounds__(NTL, 1) vms_kernel"
    edits = [(kernel, D1_HELPER + kernel, 1), (SCALE, MERGED, 3),
             (MUL, MULH, 2), (COLD, MULH+COLD_MMA, 1)]
    d1 = cu
    for a, b, count in edits: d1 = sub(d1, a, b, count)
    old_unit, roles = role_body(d1)
    # The generated function's strings are source/data, never module factories.
    lines = ["def _vmsw_source():", "    s = VMS_CUDA_SRC", "    def replace(a,b,n=1):",
             "        nonlocal s", "        assert s.count(a)==n, a[:80]", "        s=s.replace(a,b)"]
    for a, b, count in edits: lines.append(f"    replace({a!r},{b!r},{count})")
    header = "#if !VMS_T16 || !VMS_ER || !VMS_RO || VMS_HI || VMS_T16E || VMS_T16R\n#error VMSW requires the shipped T16 hi route\n#endif\n"
    # Header must follow definitions of native flags; only VMS_W is supplied by bank.
    lines.append("    replace(%r,%r)" % ("namespace vms {\n", "namespace vms {\n"+header))
    smem = "#if VMS_LPAD > 0\nunsigned char lpad[VMS_LPAD];\n#endif\n"
    lines.append("    replace(%r,%r)" % (smem, smem+"#if VMS_W==2 || VMS_W==3\nuint4 fragB[2][32][4];\n#endif\n"))
    request = "constexpr int SMEM_BYTES = RING_OFF + NS * UB;\n"
    assert cu.count("SMEM_BYTES") == 4, "dynamic size must remain host-only"
    lines.append("    replace(%r,%r)" % (request, "#if VMS_W==5\n" +
                 request.replace(";", " + 4096;") + "#else\n" + request + "#endif\n"))
    lines.append(f"    replace({kernel!r},{(ROLE_HELPER+kernel)!r})")
    init = "if (mk) mrow[0] = gt();\n"
    lines.append("    replace(%r,%r)" % (init, init+"#if VMS_W==3\nmbar_init(reinterpret_cast<uint64_t*>(ring-8),2);\n#endif\n"))
    lines += ["    i=s.index('const unsigned char* slot = ring + sl * UB;\\n')",
              "    j=s.index('float4 o;\\n',i)",
              f"    s=s[:i]+'#if VMS_W >= 2 && VMS_W <= 4\\n'+{roles!r}+'#else\\n'+s[i:j]+'#endif\\n'+s[j:]",
              "    return s", "VMSW_CUDA_SRC = _vmsw_source()", ""]
    return "\n".join(lines), d1, roles


def build(base, report=False):
    recipe, d1, roles = source_recipe(base)
    s = sub(base, "VMS_ENABLE = True\n", recipe + PYTHON + "VMS_ENABLE = True\n")
    anchor = '    tag_hi = hashlib.sha1((VMS_CUDA_SRC + " ".join(fl_hi)).encode()).hexdigest()[:10]\n'
    s = sub(s, anchor, '    _VMSW["flags"] = list(fl_hi)\n' + anchor)
    anchor = '    md = vs["mod_hi"] if (VMS_HI_ON and m > 8 and vs.get("mod_hi") is not None) else vs["mod"]\n'
    s = sub(s, anchor, anchor + SELECT)
    s = sub(s, '        st["vms_ok"] = good\n', '        st["vms_ok"] = good\n        _vmsw_prepare(st)\n')
    ast.parse(s)
    assert len(s.encode()) <= 1_000_000, len(s.encode())
    # Every base module constant stays literally unchanged, including low VMS.
    patched_nodes = {t.id:n for n in ast.parse(s).body if isinstance(n, ast.Assign)
                     for t in n.targets if isinstance(t, ast.Name)}
    for n in ast.parse(base).body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant):
            for target in n.targets:
                if isinstance(target, ast.Name):
                    assert ast.dump(patched_nodes[target.id]) == ast.dump(n), target.id
    ns = {"VMS_CUDA_SRC": literal(base, "VMS_CUDA_SRC")}
    exec(recipe, ns)
    active = ns["VMSW_CUDA_SRC"]
    source_gate(active)
    suffix = "float4 o;\n"
    assert active[active.index(suffix):] == d1[d1.index(suffix):]
    assert "return" not in roles and "continue" not in roles and "if (cc" not in roles
    assert roles.count("cta_sync_compute();") == 1
    assert roles.index("rw = ") < roles.index("cta_sync_compute();") < roles.index("S.fragB[warp-1]")
    assert roles.index("rsw = ") < roles.index("cta_sync_compute();")
    assert roles.count("VmswRole<") == 3
    # Only the added helper contains a bounded loop; suffix includes every reset.
    assert "__syncwarp" not in ROLE_HELPER and "WAIT_MAX" in ROLE_HELPER
    if report:
        for name in ("_vmsw_source", "VMSW_CUDA_SRC", "_vmsw_build_one", "_vmsw_prepare", "_vms_call", "_vms_prepare"):
            n = node(s, name); print(f"[vmsw] {OUT.name}:{n.lineno}-{n.end_lineno} {name}")
        print(f"[vmsw] stock constants/helper+cleanup suffix exact; lab {len(s.encode()):,} B (+{len(s.encode())-len(base.encode()):,})")
    return s


def source_gate(source):
    """Resolve only our variant guards; leave every native preprocessor line.

    This is a source identity gate, not a substitute for the nvcc/SASS gate.
    """
    def selected(mask):
        result, stack, keep = [], [], True
        for line in source.splitlines(keepends=True):
            directive = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", line)
            if directive:
                op, expr = directive.groups()
                if op in ("if", "ifdef", "ifndef"):
                    own = op == "if" and re.search(r"\bVMS_W\b", expr)
                    cond = None
                    if own:
                        expr = re.sub(r"\bVMS_W\b", str(mask), expr).replace("&&", " and ").replace("||", " or ")
                        cond = bool(eval(expr, {"__builtins__": {}}))
                    stack.append((keep, cond))
                    if own:
                        keep = keep and cond
                        continue
                elif op == "else" and stack[-1][1] is not None:
                    keep = stack[-1][0] and not stack[-1][1]
                    continue
                elif op == "elif":
                    assert stack[-1][1] is None, "variant elif unsupported"
                elif op == "endif":
                    parent, cond = stack.pop()
                    keep = parent
                    if cond is not None:
                        continue
            if keep:
                result.append(line)
        assert not stack
        return "".join(result)
    d1, rs, rs0, m4 = (selected(mask) for mask in (1, 2, 4, 5))
    request = "constexpr int SMEM_BYTES = RING_OFF + NS * UB"
    assert m4.replace(request + " + 4096;", request + ";") == d1
    assert rs.replace("uint4 fragB[2][32][4];\n", "") == rs0
    assert "fragB" not in rs0 and "VMSW_ROLE" in rs0
    assert all(token not in d1 + m4 for token in ("fragB", "vmsw_unit", "vmsw_wait"))
    assert rs0.count("vmsw_unit(VmswRole<") == 3


PYTHON = 'VMSW_ON = 0\n_VMSW = {"mods": {}, "flags": None, "errors": {}, "secs": {}, "calls": {}}\nVMSW_VARIANTS = {"d1": 1, "rs": 2, "e": 3, "rs0": 4, "m4": 5}\ndef _vmsw_build_one(mask, flags):\n    from torch.utils.cpp_extension import load_inline\n    tag = hashlib.sha1((VMSW_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]\n    return load_inline(name=f"k3_vmshi_d{mask}_{tag}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMSW_CUDA_SRC,\n                       functions=["run", "attrs"], extra_cuda_cflags=flags)\ndef _vmsw_prepare(st):\n    _VMSW["errors"] = {}\n    _VMSW["calls"].clear()\n    vs = st["vms"]\n    vs["diet"] = {}\n    if not st["vms_ok"] or vs.get("mod_hi") is None:\n        return\n    base = list(vs["mod_hi"].attrs())\n    if base[0] > 168 or base[1] or base[2] != 174464:\n        raise RuntimeError(f"VMSW base resources {base}")\n    def one(name, mask):\n        t0 = time.time()\n        try:\n            if mask not in _VMSW["mods"]:\n                flags = list(_VMSW["flags"]) + [f"-DVMS_W={mask}", "-Xptxas=-v"]\n                _VMSW["mods"][mask] = _vmsw_build_one(mask, flags)\n            md = _VMSW["mods"][mask]\n            a = list(md.attrs())\n            if a[0] > 168 or a[1] or a[2:] != [178560 if mask in (2, 3, 5) else 174464] + base[3:]:\n                raise RuntimeError(f"VMSW {name} resources {a}, base {base}")\n            vs["diet"][mask] = md\n            print(f"[vmsw] {name} resources {a}", flush=True)\n        except Exception as ex:\n            _VMSW["errors"][name] = repr(ex)\n            print(f"[vmsw] {name} unavailable: {ex!r}", flush=True)\n        _VMSW["secs"][name] = time.time() - t0\n    import threading\n    ths = [threading.Thread(target=one, args=(name, mask)) for name, mask in VMSW_VARIANTS.items()]\n    for th in ths:\n        th.start()\n    for th in ths:\n        th.join()\n'
SELECT = '    if m == 16:\n        if type(VMSW_ON) is not int or VMSW_ON not in (0, *VMSW_VARIANTS.values()):\n            raise RuntimeError(f"invalid VMSW_ON {VMSW_ON!r}")\n        if VMSW_ON:\n            if md is not vs.get("mod_hi") or VMSW_ON not in vs["diet"]:\n                raise RuntimeError(f"unavailable VMSW arm {VMSW_ON}")\n            md = vs["diet"][VMSW_ON]\n        key = f"T16:{VMSW_ON}"\n        _VMSW["calls"][key] = _VMSW["calls"].get(key, 0) + 1\n'


def dry_run(base, patched):
    """Execute real _vms_prepare, all new build/bank paths and real _vms_call.

    Only GPU primitives/prepack/reference arithmetic are doubled. The real
    prepare builds its own strict dict; no hand-invented VMS state keys.
    """
    tree = ast.parse(patched)
    names = {"VMSW_ON", "_VMSW", "VMSW_VARIANTS", "_vmsw_build_one", "_vmsw_prepare", "_vms_prepare", "_vms_call", "VMS_SELF_TEST_CASES", "_vmsw_source"}
    def named(n):
        return n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None
    ns = {"time": time, "hashlib": hashlib}
    exec(compile(ast.Module(body=[n for n in tree.body if named(n) in names], type_ignores=[]), "vmsw-dry", "exec"), ns)
    class Tensor:
        device = "cpu"
        def __init__(self, shape=(16, 4096)): self.shape = shape
        def __getitem__(self, k): return 0 if isinstance(k, int) and self.shape == (72,) else Tensor(self.shape[1:])
        def __setitem__(self, k, v): pass
        def __ge__(self, v): return self
        __le__ = __lt__ = __ge__
        def __and__(self, v): return self
        def __mul__(self, v): return self
        def __radd__(self, v): return self
        def all(self): return True
        def reshape(self, *s): return Tensor(s)
        def view(self, *s): return self
        def to(self, *a): return self
        def contiguous(self): return self
        clone = zero_ = float = contiguous
    seen, builds, stock_builds, failed, badattrs = [], [], [], set(), {}
    class Module:
        def __init__(self, mask): self.mask = mask
        def attrs(self): return badattrs.get(self.mask, [156, 0, SMEM.get(self.mask, 174464), 132, 2])
        def run(self, *a): seen.append((self.mask, a[14]))
    def inputs(bs, *a): return {k: Tensor((bs, 4096)) for k in ("hidden_states", "topk_ids", "topk_weights")}
    # Execute base's actual flags expressions, not copied approximations.
    flags_ns = {"G": 132}
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_vms_build")
    for n in fn.body:
        if isinstance(n, ast.Assign) and named(n) in ("flags", "fl_hi"):
            exec(compile(ast.Module(body=[n], type_ignores=[]), "flags", "exec"), flags_ns)
    def load_inline(**kw):
        flags = kw["extra_cuda_cflags"]
        if not any(f.startswith("-DVMS_W=") for f in flags):
            assert kw["cuda_sources"] == literal(base, "VMS_CUDA_SRC")
            assert kw["cpp_sources"] == literal(base, "VMS_CPP_SRC")
            hi = "-DVMS_T16=1" in flags
            assert flags == flags_ns["fl_hi" if hi else "flags"]
            stock_builds.append(hi)
            return Module(0 if hi else -1)
        mask = int(next(f.split("=")[1] for f in flags if f.startswith("-DVMS_W=")))
        assert flags == flags_ns["fl_hi"] + [f"-DVMS_W={mask}", "-Xptxas=-v"]
        assert kw["name"].startswith(f"k3_vmshi_d{mask}_") and kw["functions"] == ["run", "attrs"]
        assert kw["cuda_sources"] == ns["VMSW_CUDA_SRC"]
        assert kw["cpp_sources"] == literal(base, "VMS_CPP_SRC")
        builds.append(mask)
        if mask in failed: raise RuntimeError("injected build failure")
        return Module(mask)
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **kw):
        return NS(load_inline=load_inline) if name == "torch.utils.cpp_extension" else real_import(name, *a, **kw)
    # Function globals use this per-exec builtins map; no sys.modules mutation.
    local_builtins = dict(vars(builtins), __import__=fake_import)
    ns["__builtins__"] = local_builtins
    build_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_vmsw_build_one", "_vms_build")]
    exec(compile(ast.Module(body=build_nodes, type_ignores=[]), "build-paths", "exec"), ns)
    ns.update(torch=NS(int32=1, float32=2, bfloat16=3, uint8=4, int64=5,
              empty=lambda s, **kw: Tensor(s), zeros=lambda s, **kw: Tensor(s),
              full_like=lambda t, v: t, where=lambda c, a, b: a, isfinite=lambda t: t,
              cuda=NS(synchronize=lambda: None, get_device_properties=lambda d: NS(multi_processor_count=132))),
              _VMS={"mod": None, "mod_hi": None, "G": 132}, _K3_BUILD_S={}, VMS_ENABLE=True, VMS_HI_ON=True,
              VMS_MAXT=16, VMS_UB=36864, K2A_E_LOC=80, K2A_D_EXPERT=4096, K2A_D_HIDDEN=4096,
              K2A_POST_NORM_EPS=1e-6, _pt1_wait=lambda n: None,
              VMS_CUDA_SRC=literal(base, "VMS_CUDA_SRC"), VMS_CPP_SRC=literal(base, "VMS_CPP_SRC"),
              _vms_prepack_expert=lambda *a: Tensor(), _k2a_self_test_inputs=inputs, _k2a36_synth_inputs=inputs,
              _k2a_ref_fp32=lambda *a: Tensor(), _k2a_rel_err=lambda *a: (0, 0),
              _v11_k2a33_call=lambda *a: Tensor(), _k2a33_reset=lambda s: None)
    ns["VMSW_CUDA_SRC"] = ns["_vmsw_source"]()
    w = {k: Tensor((80, 8192, 2048)) for k in ("w13", "w2", "w13_sf", "w2_sf", "w13_gs", "w2_gs", "post_norm_w")}
    state = {"err": Tensor()}
    with contextlib.redirect_stdout(io.StringIO()): ns["_vms_prepare"](state, w, "cpu")
    masks = set(VARIANTS.values())
    assert ns["VMSW_VARIANTS"] == VARIANTS
    assert sorted(stock_builds) == [False,True] and ns["_VMSW"]["flags"] == flags_ns["fl_hi"]
    assert state["vms_ok"] and set(state["vms"]["diet"]) == masks and sorted(builds) == sorted(masks)
    for mask in (0, *VARIANTS.values()):
        ns["VMSW_ON"] = mask
        for bs in (1, 8, 12, 16):
            ns["_vms_call"](state, Tensor((bs, 4096)), Tensor(), Tensor())
            assert seen[-1] == ((mask if bs == 16 else 0 if bs > 8 else -1), bs)
    for bad in (-1, 6, 7, 8, 16, True, 1.0, None):
        ns["VMSW_ON"] = bad
        try: ns["_vms_call"](state, Tensor(), Tensor(), Tensor())
        except RuntimeError: pass
        else: raise AssertionError("invalid arm accepted")
    ns["VMSW_ON"] = 0
    with contextlib.redirect_stdout(io.StringIO()): ns["_vmsw_prepare"](state)
    assert len(builds) == len(masks), "cache miss on repeated prepare"
    for name, mask in VARIANTS.items():
        failed.add(mask); ns["_VMSW"]["mods"].pop(mask)
        with contextlib.redirect_stdout(io.StringIO()): ns["_vmsw_prepare"](state)
        assert set(state["vms"]["diet"]) == masks - {mask} and name in ns["_VMSW"]["errors"]
        ns["VMSW_ON"] = mask
        try: ns["_vms_call"](state, Tensor(), Tensor(), Tensor())
        except RuntimeError: pass
        else: raise AssertionError("missing build silently fell back")
        failed.clear()
        for index, value in ((0, 169), (1, 4), (2, SMEM[mask] ^ 4096)):
            a = [156, 0, SMEM[mask], 132, 2]; a[index] = value; badattrs[mask] = a
            with contextlib.redirect_stdout(io.StringIO()): ns["_vmsw_prepare"](state)
            assert mask not in state["vms"]["diet"] and name in ns["_VMSW"]["errors"]
        badattrs.clear()
        with contextlib.redirect_stdout(io.StringIO()): ns["_vmsw_prepare"](state)
        assert set(state["vms"]["diet"]) == masks
    state["vms_ok"] = False; ns["_vmsw_prepare"](state)
    assert not state["vms"]["diet"]
    state["vms_ok"] = True; state["vms"]["mod_hi"] = None; ns["_vmsw_prepare"](state)
    assert not state["vms"]["diet"]
    ns["VMS_ENABLE"] = False; ns["_vms_prepare"](state, w, "cpu"); assert not state["vms_ok"]
    print("[vmsw] CPU dry run PASS: real prepare/state, flags, five builds, cache, all arms/resource failures, no silent fallback")

def exact_cpu_gate():
    """Address, bit-pattern, MMA-order and phase/overwrite checks without CUDA."""
    import random
    rng = random.Random(0x574D53)
    # Member accounting includes 16-byte alignment of each vector array.
    def layout(extra):
        offset = 0
        for count, width, align in [(2,8,8),(4,8,8),(4,8,8),(2,4,4),(4*32,4,4),
                (3*64,4,4),(2*128,4,4),(4*8*32,16,16),(4*8*32,16,16),
                (8*16*32,16,16),(4,4,4),(extra,16,16)]:
            offset = (offset+align-1)//align*align + count*width
        size = (offset+15)//16*16
        ring = (size+127)//128*128
        return size, ring, ring+2*36864
    assert layout(0) == (100720,100736,174464)
    assert layout(256) == (104816,104832,178560)
    assert SMEM[1] == SMEM[4] == layout(0)[2]
    assert SMEM[5] == layout(0)[2] + 4096 == SMEM[2] == SMEM[3]
    assert 104816 <= 104832-8 and (104832-8)%8 == 0
    def perm(w, sel):
        b = [(w>>(8*i))&255 for i in range(4)] + [0]*4
        return sum(b[(sel>>(4*i))&7] << (8*i) for i in range(4))
    for a in range(8,127):
        for b in range(8,127):
            w = a<<16 | b<<24
            both = perm(w,0x4342)*16+0x5C005C00
            assert (both&65535)*0x10001 == perm(w,0x4242)*16+0x5C005C00
            assert (both>>16)*0x10001 == perm(w,0x4343)*16+0x5C005C00
    for hot in (False, True):
        for warp in range(8):
            original, exchanged = [], []
            ring = [rng.getrandbits(32) for _ in range(36864//4)]
            scratch = {}
            for rec, donor in ((1,0),(2,4)):
                for lane in range(32):
                    base = (donor*8+7)*512+lane*16
                    scale = 32768+donor*512+lane*16+12
                    assert 0 <= base < 32768 and 32768 <= scale < 36864
                    for word in range(4):
                        # Unpack is identical to the VMSE recipient and base.
                        qw=ring[base//4+word]; mask=0x81C081C0
                        rot=lambda n: ((qw<<n)|(qw>>(32-n)))&0xffffffff
                        raw=(qw&mask,rot(10)&mask,rot(13)&mask,((qw&0x001B001B)*0x1084)&mask)
                        # Each multiplication receives identical bf16 operand bits;
                        # raw tuple+scale is sufficient to prove the bmul operands.
                        scratch[donor,lane,word]=(raw,ring[scale//4])
            for cc in range(8):
                for word in range(4):
                    for tile in range(2 if hot else 1):
                        chain = (word&1)+tile*2 if hot else word
                        event=(cc,word,tile,chain,2*cc,2*word)
                        original.append(event)
                        exchanged.append(event)
                    if warp in (0,4) and cc==7:
                        for lane in range(32):
                            qw=ring[((warp*8+cc)*512+lane*16)//4+word]
                            rot=lambda n: ((qw<<n)|(qw>>(32-n)))&0xffffffff
                            raw=(qw&mask,rot(10)&mask,rot(13)&mask,((qw&0x001B001B)*0x1084)&mask)
                            assert scratch[warp,lane,word] == (raw,ring[(32768+warp*512+lane*16+12)//4])
            assert original==exchanged
    # Random legal warp schedules: all 8 preloads -> existing named barrier;
    # recipient publication and donor consumption; next unit barrier gates reuse.
    for n in (0,1,2,3,4,5,7,8,9,65,513,769):
        for trial in range(5):
            pc=[0]*8; q=[0]*8; phase=0; arrived=set(); fragments={}; consumed=set()
            steps=0
            while any(v<n for v in q):
                choices=[]
                for w in range(8):
                    if q[w]>=n: continue
                    if pc[w]==1 and not all(q[k]==q[w] and pc[k]>=1 for k in range(8)):
                        continue
                    if pc[w]==3 and w in (0,4) and len(arrived)!=2: continue
                    choices.append(w)
                assert choices, (n,q,pc,"deadlock")
                w=rng.choice(choices); u=q[w]; steps+=1
                if pc[w]==0: pc[w]=1
                elif pc[w]==1:
                    # Release all eight participants together; same bar.sync ID.
                    assert u==0 or consumed=={0,4}
                    assert phase==(u&1)
                    arrived=set(); consumed=set()
                    pc=[2]*8
                elif pc[w]==2:
                    if w in (1,2):
                        donor=(w-1)*4; fragments[donor]=u; arrived.add(w)
                        if len(arrived)==2: phase^=1
                    pc[w]=3
                else:
                    if w in (0,4):
                        assert phase!=(u&1) and fragments[w]==u
                        consumed.add(w)
                    q[w]+=1; pc[w]=0
            assert steps==n*25 and (n==0 or consumed=={0,4})
    print("[vmsw] CPU exact/protocol PASS: 14,161 scale pairs, all fragment addresses/MMA order, N=0/short/job/NS/NRB wraps, unchanged cleanup")


def main():
    if len(sys.argv)!=1: raise SystemExit("usage: python -B -I tools/vmsw_patches.py")
    base = BASE.read_bytes().decode().replace("\r\n", "\n")
    s = build(base, report=True)
    exact_cpu_gate()
    dry_run(base, s)
    OUT.write_bytes(s.encode())
    print("[vmsw] generator/AST/source/CPU prepare gates PASS; CUDA/SASS/timing NOT RUN; HOLD_EXACT")


if __name__ == "__main__":
    main()

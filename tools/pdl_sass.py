"""pdl_sass.py: $0 SASS audit of griddepcontrol.wait (ACQBULK) placement in the MoE kernels of a layer file (R-pdl-0).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/pdl_sass.py" <layer.py> [OUTDIR] [--only=vms,vm_bf2,vm_w72] [--reuse] [-v] [--allow-latent]

Kernels (compile-time instances taken from the layer file itself): vms::vms_kernel (VMS_CUDA_SRC, the flag list of _vms_build
with G = 132), vm::vm_main + vm::vm_fin for every VML variant the route tables use (VML_VAR, VML_ABVARS, VML_BVAR values ->
_VML_DEFS[v]).  Each is compiled device-only for sm_90a (nvcc -cubin, local WSL toolchain; ptxas 13.4) and its SASS is analysed
on the control-flow graph (not in linear order, so warp-specialised branches are handled):

  * ACQBULK = griddepcontrol.wait.  An instruction is "pre-wait" when some path from the kernel entry reaches it without an
    ACQBULK.  Wait skipped by the runtime flag test (vm_main: A.flags & 256) is analysed ARMED: the conditional branch that
    jumps over an ACQBULK is pruned, i.e. the question is what runs when the wait is armed.
  * Every pre-wait global access (LDG, LD, LDGSTS, UBLKCP / TMA loads, STG, ATOMG, REDG; UBLKPF / UTMAPF = prefetch hints) has
    its address slice traced back (reaching definitions over the CFG, through ALU ops and through loaded values) to kernel
    parameter words c[0x0][0x210+off]; offsets map to struct members through an offsetof probe (template kernel names).
    Member classes: WEIGHT (prepacked weights / constant tables written in prepare()), UPSTREAM2 (wts: written before the
    predecessor started; safe while that predecessor triggers launch_dependents after its own wait), DIAG (marks: timing buffer,
    null in the ship), PRED (everything else = predecessor output, or scratch shared with it: RACE above the wait), UNKNOWN (no
    parameter provenance).
  * Verdict per instance: SAFE, HAZARD (a pre-wait access to PRED / UNKNOWN data, read or write), PREFETCH-DEAD (INTENT lists
    the pre-wait loads the source writes before the wait; fewer land above ACQBULK).  Exit 1 when an instance whose wait is
    armed in the ship is a HAZARD; a HAZARD in an instance disarmed in the ship (vm_main with VML_FLAGS bit 256 clear: no PDL
    attribute, no wait) is printed LATENT-HAZARD and also exits 1 (the SASS has the load above the wait, the first flag change
    arms it) unless --allow-latent.

  --nc-below (R-pdl-1): additionally list EVERY non-coherent load (LDG.*.CONSTANT = ld.global.nc / __ldg / const __restrict__)
    of predecessor data, ABOVE or BELOW the wait, with its source line (the cubins carry -lineinfo; nvdisasm -g maps SASS idx ->
    .cu line -> layer-file line), for vms_kernel, vm_main / vm_fin (bf2 / w72), vgf_kernel, the K2A33 kernels of the routed
    variants (vid 0 1 2 6) and tail_kernel / tail_kernel2 (the writer of kernel_fn's `experts`); also counts the stores whose
    address slice reaches ids / wts / tid / lid (who writes the routing tensors).  PRED = predecessor output, UP2 = wts.
"""
import ast
import collections
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
OBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
NVD = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/nvdisasm"
PARAM0 = 0x210                                       # c[0x0][0x210]: first kernel parameter byte (sm_90)
TORCH_NV = ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr", "-DTORCH_EXTENSION_NAME=pdl"]

# ---- member classes --------------------------------------------------------------------------------------------------
WEIGHT = {"ws13", "ws2", "gsc13", "gsc2", "nw", "wA", "wB", "gsA", "gsB", "bias", "lstart"}   # bias / lstart: router bias, data["local_start"]
UPSTREAM2 = {"wts"}
DIAG = {"marks"}
RANK = {"WEIGHT": 0, "DIAG": 0, "UPSTREAM2": 1, "UNKNOWN": 3, "PRED": 4}

# what the source intends to run before the wait: kernel -> [(member, kind, min accesses per thread-path)].  vm_fin: w4[0..3] =
# __ldg(nw ..) and wv = __ldg(wts ..) sit above griddepcontrol.wait; vm_main: the optional flags&2 L2 prefetch of x (disarmed
# in the ship: VML_FLAGS & 2 = 0); vms: VMS_FX & 4 (pre-wait first weight copies) is off in the shipped flags.
INTENT = {"vm_fin": [("nw", "R", 4), ("wts", "R", 1)], "vm_main": [("x", "PF", 1)], "vms_kernel": []}


def literal_nodes(path, names):
    """exec the named top-level assignments / defs of a layer file (in file order) and return the namespace."""
    ns = {"__name__": "pdl_extract"}
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    want = set(names)
    for n in tree.body:
        tg = []
        if isinstance(n, ast.Assign):
            tg = [t.id for t in n.targets if isinstance(t, ast.Name)]
        elif isinstance(n, ast.FunctionDef):
            tg = [n.name]
        if want & set(tg):
            exec(compile(ast.Module(body=[n], type_ignores=[]), str(path), "exec"), ns)
    missing = [k for k in names if k not in ns]
    if missing:
        raise SystemExit(f"[pdl_sass] {path}: cannot extract {missing}")
    return ns


def lineno_of(path, name):
    for n in ast.parse(pathlib.Path(path).read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return n.value.lineno
    raise SystemExit(f"[pdl_sass] {name} not found in {path}")


def vms_flags(path):
    """the flag list of _vms_build with G = 132 (the H200), evaluated from the file's own AST."""
    for n in ast.parse(pathlib.Path(path).read_text(encoding="utf-8")).body:
        if isinstance(n, ast.FunctionDef) and n.name == "_vms_build":
            for s in ast.walk(n):
                if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id == "flags":
                    return eval(compile(ast.Expression(s.value), str(path), "eval"), {"G": 132})
    raise SystemExit("[pdl_sass] _vms_build flags not found")


def units_of(path, only, nc=False):
    ns = literal_nodes(path, ["_vml_JMAX", "_vml_SLOTS", "_vml_FIXW", "_vml_wg_spec", "_vml_wg_spec64", "_vml_CUDA_T", "_vml_CUDA",
                              "_vml_FLAGS", "_VML_DEFS", "VML_VAR", "VML_ABVARS", "VML_BVAR", "VML_BUCKETS", "VML_FLAGS",
                              "VML_ENABLE", "VMS_CUDA_SRC", "VMS_ENABLE"])
    us = [dict(name="vms", src=ns["VMS_CUDA_SRC"], flags=vms_flags(path), kernels=["vms_kernel"], struct="Params", ns="vms",
               routes="T<=16 (VMS_MAXT)", armed={"vms_kernel": True}, origin="VMS_CUDA_SRC")]   # _vms_call passes pdl=1: PDL attribute set
    vars_ = list(dict.fromkeys([ns["VML_VAR"]] + list(ns["VML_ABVARS"]) + list(ns["VML_BVAR"].values())))
    for v in vars_:
        bk = [b for b, on in ns["VML_BUCKETS"].items() if on and ns["VML_BVAR"].get(b, ns["VML_VAR"]) == v]
        fl = sorted({ns["VML_FLAGS"].get(b, 1) for b in bk})
        us.append(dict(name=f"vm_{v}", src=ns["_vml_CUDA"], flags=list(ns["_vml_FLAGS"]) + list(ns["_VML_DEFS"][v]),
                       kernels=["vm_main", "vm_fin"], struct="Args", ns="vm",
                       routes=f"T-buckets {bk or 'none (self-test / default only)'} flags {fl}",
                       armed={"vm_main": any(f & 256 for f in fl), "vm_fin": True}, origin="_vml_CUDA_T"))   # vm_fin: PDL attribute always set
    if nc:
        t = literal_nodes(path, ["V8_TAIL_CUDA_SRC", "VGF_CU", "VGF_HOST", "K2A33_PLAN", "SK1_VID", "K2A33_VARIANTS", "_k2a33_wg_specs",
                                 "K2A33_KERNEL_SRC"])
        us.append(dict(name="tail", src=t["V8_TAIL_CUDA_SRC"], flags=["-O3", "-lineinfo"], kernels=["tail_kernel2", "tail_kernel"], struct=None,
                       ns=None, routes="V8_TAIL_ROUTE 'cat' T>=16 / 'sep' T<=8 (TL1_TAIL2 -> tail_kernel2)", armed={}, origin="V8_TAIL_CUDA_SRC"))
        us.append(dict(name="vgf", src=t["VGF_CU"] + t["VGF_HOST"], flags=["-O3"], kernels=["vgf_kernel"], struct=None, ns=None,
                       routes="not routed (_vgf_on never called; self-test only)", armed={}, origin="VGF_CU"))
        vids = sorted(set(t["K2A33_PLAN"].values()) | {t["SK1_VID"]})
        cs, at = [], []
        for v in vids:
            ex = tuple(t["K2A33_VARIANTS"][v])
            assert len(ex) == 4, ex     # the routed vids are the plain launch3 variants
            cs.append("    case %d: launch3<%d, %d, %d, %d>(p, (int)G, (int)G3, (int)mask); break;" % ((v,) + ex))
            at.append("    case %d: return attrs3<%d, %d, %d, %d>();" % ((v,) + ex))
        ksrc = t["K2A33_KERNEL_SRC"].replace("@CASES3@", "\n".join(cs)).replace("@ATTRS3@", "\n".join(at)).replace("@WGSPECS@", t["_k2a33_wg_specs"]())
        us.append(dict(name="k2a", src=ksrc, flags=["-O3", "--expt-relaxed-constexpr"], kernels=["k1_kernel", "k2_kernel", "p0_kernel", "k3_kernel"],
                       struct="Params", ns="k3", routes=f"K2A33_PLAN {t['K2A33_PLAN']} + SK1_VID {t['SK1_VID']}: vids {vids} (fallback when vms / vml are off)",
                       armed={}, origin="K2A33_KERNEL_SRC", info=True))   # lazy waits (pdl_wait() inside loops): path-insensitive ABOVE / BELOW
    for u in us:
        u["layer"] = str(path)
    return [u for u in us if not only or u["name"] in only]


# ---- struct member probe: offsetof baked into template kernel names ---------------------------------------------------
def struct_members(src, struct):
    m = re.search(r"struct\s+(?:__align__\(\d+\)\s+)?%s\s*\{" % struct, src)
    i, d = m.end(), 1
    while d:
        d += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", src[m.end():i - 1], flags=re.S)
    body = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    names = {}
    for decl in body.split(";"):
        decl = decl.strip()
        if not decl or "(" in decl or decl.startswith(("static", "constexpr", "struct")):
            continue
        for part in re.sub(r"\[[^\]]*\]", "", decl).split(","):
            ids = re.findall(r"[A-Za-z_]\w*", part.split("=")[0])
            if ids:
                names[ids[-1]] = "*" in part or "*" in decl.split(ids[-1])[0]
    return names


SCALAR8 = ("long", "size_t", "double", "int64_t", "uint64_t", "ptrdiff_t")
SCALAR4 = ("int", "float", "unsigned", "uint32_t", "int32_t")


def sig_params(src, kname):
    """[(name, offset, is_ptr)] of a plain-parameter __global__ kernel (natural alignment); None if a parameter is a struct."""
    m = re.search(r"__global__\s+void\s+(?:__launch_bounds__\((?:[^()]|\([^()]*\))*\)\s+)?%s\s*\(" % re.escape(kname), src)
    if not m:
        return None
    i, d = m.end(), 1
    while d:
        d += {"(": 1, ")": -1}.get(src[i], 0)
        i += 1
    out, off = [], 0
    for decl in split_ops(src[m.end():i - 1]):
        decl = re.sub(r"__restrict__|__grid_constant__|\bconst\b|__align__\(\d+\)", " ", decl)
        ids = re.findall(r"[A-Za-z_]\w*", decl)
        if not ids:
            continue
        if "*" in decl:
            sz, ptr = 8, True
        elif ids[0] in SCALAR8 or decl.strip().startswith("long long"):
            sz, ptr = 8, False
        elif ids[0] in SCALAR4:
            sz, ptr = 4, False
        else:
            return None
        off = (off + sz - 1) // sz * sz
        out.append((ids[-1], off, ptr))
        off += sz
    return out


def probe_src(src, ns, struct, names):
    out = ["\n// ---- pdl_sass probe: offsetof(member) baked into template kernel names (-1: member compiled out) ----",
           "#include <stddef.h>",
           "template <int I, int N> __global__ void pdl_off_probe() {}"]
    for i, n in enumerate(names):
        out.append("template <class A> constexpr int pdl_o_%s() { if constexpr (requires { &A::%s; }) return (int)offsetof(A, %s); else return -1; }"
                   % (n, n, n))
        out.append("template __global__ void pdl_off_probe<%d, pdl_o_%s<%s::%s>()>();" % (i, n, ns, struct))
    return src + "\n".join(out) + "\n"


def compile_units(units, out, reuse):
    import sysconfig
    import torch.utils.cpp_extension as ce
    inc = [f"-I{p}" for p in ce.include_paths()] + [f"-I{sysconfig.get_paths()['include']}"]
    sem, res = threading.Semaphore(4), {}

    def job(u):
        with sem:
            n = u["name"]
            mem = struct_members(u["src"], u["struct"]) if u["struct"] else {}
            names = list(mem)
            cu, cub, sas = out / f"{n}.cu", out / f"{n}.cubin", out / f"{n}.sass"
            cu.write_text(probe_src(u["src"], u["ns"], u["struct"], names) if u["struct"] else u["src"])
            (out / f"{n}.members.json").write_text(json.dumps(mem))
            if not (reuse and cub.exists() and sas.exists()):
                fl = [f for f in u["flags"] if not f.startswith("-gencode") and f not in ("-Xptxas=-warn-spills", "-lineinfo")]
                r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", "-std=c++20", *TORCH_NV, *fl, "-lineinfo", "-Xptxas=-v", *inc,
                                    "-o", str(cub), str(cu)], capture_output=True, text=True)
                (out / f"{n}.ptxas.log").write_text(r.stdout + r.stderr)
                if r.returncode:
                    res[n] = "FAIL " + " | ".join(ln[-200:] for ln in (r.stdout + r.stderr).splitlines() if "error" in ln)[:600]
                    return
                sas.write_text(subprocess.run([OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout)
                (out / f"{n}.lines.txt").write_text(subprocess.run([NVD, "-g", "-c", str(cub)], capture_output=True, text=True).stdout)
            res[n] = "ok"
    ths = [threading.Thread(target=job, args=(u,)) for u in units]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return res


# ---- SASS parsing + CFG -----------------------------------------------------------------------------------------------
LINE = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\*\s*0x[0-9a-f]{16}\s*\*/")
PRED = re.compile(r"^(@!?U?P[0-9T]\w*)\s+")
REG = re.compile(r"\b(UR|R)(\d+)(\.64)?\b")
CB = re.compile(r"c\[0x0\]\[(0x[0-9a-f]+)\]")
NODEST = re.compile(r"^(ST|STG|STS|STL|BRA|BRX|JMX|EXIT|RET|CALL|BAR|BSSY|BSYNC|MEMBAR|ERRBAR|FENCE|WARPSYNC|NOP|DEPBAR|PREEXIT|ACQBULK|"
                    r"CCTL|RED|REDG|UBLK|UTMA|LDGSTS|LDGDEPBAR|YIELD|NANOSLEEP|USETMAXREG|UCGABAR|SETCTAID|ISETP|FSETP|PSETP|HSETP|DSETP|"
                    r"UISETP|UPSETP|PLOP3|UPLOP3|R2P|UP2UR|SYNCS\.ARRIVE|SYNCS\.EXCH|SYNCS\.PHASECHK|TCGEN|WARPGROUP|HGMMA|UTMA)")
PREDFIRST_REG = re.compile(r"^(LOP3|ULOP3|SHFL|VOTE|VOTEU|MATCH|ELECT|IADD3|UIADD3|LEA|ULEA|IMAD|IABS|FADD|HFMA2)")


class Ins:
    __slots__ = ("idx", "pred", "op", "ops", "text", "succ", "guard")

    def __init__(self, idx, text):
        self.idx, self.text = idx, text
        m = PRED.match(text)
        self.pred = m.group(1) if m else ""
        body = text[m.end():] if m else text
        sp = body.split(None, 1)
        self.op = sp[0]
        self.ops = split_ops(sp[1]) if len(sp) > 1 else []
        self.succ = []


def split_ops(s):
    out, d, cur = [], 0, ""
    for ch in s:
        if ch in "[(":
            d += 1
        elif ch in "])":
            d -= 1
        if ch == "," and d == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_sass(sass):
    fns = {}
    for blk in sass.split("Function : ")[1:]:
        name = blk.split("\n", 1)[0].strip()
        ins = []
        for ln in blk.splitlines():
            m = LINE.match(ln)
            if m:
                ins.append(Ins(int(m.group(1), 16) // 16, m.group(2)))
        fns[name] = ins
    return fns


def target_of(i):
    for o in reversed(i.ops):
        m = re.fullmatch(r"(?:`\()?(0x[0-9a-f]+)\)?", o)
        if m:
            return int(m.group(1), 16) // 16
    return None


class Fn:
    def __init__(self, name, ins):
        self.name, self.ins = name, ins
        self.n = len(ins)
        self.byidx = {i.idx: k for k, i in enumerate(ins)}
        calls = [k for k, i in enumerate(ins) if i.op.startswith("CALL")]
        bssy = collections.defaultdict(list)
        for k, i in enumerate(ins):
            if i.op == "BSSY":
                bssy[i.ops[0]].append(self.byidx.get(target_of(i), -1))
        self.warn = []
        for k, i in enumerate(ins):
            nxt = k + 1 if k + 1 < self.n else None
            cond = bool(i.pred) and i.pred not in ("@PT",)
            t = self.byidx.get(target_of(i)) if target_of(i) is not None else None
            s = []
            if i.op.startswith("BRA"):
                if t is not None:
                    s.append(t)
                if (cond or i.op != "BRA") and nxt is not None:
                    s.append(nxt)
            elif i.op.startswith(("BRX", "JMX")):
                s += [nxt] if nxt is not None else []
                self.warn.append((k, f"{i.op} at {i.idx}: indirect branch targets unknown (only fall-through followed)"))
            elif i.op == "EXIT":
                if cond and nxt is not None:
                    s.append(nxt)
            elif i.op.startswith("RET"):
                s += [c + 1 for c in calls if c + 1 < self.n]
            elif i.op.startswith("CALL"):
                if t is not None:
                    s.append(t)
                if cond and nxt is not None:
                    s.append(nxt)
            elif i.op == "BSYNC":
                if nxt is not None:
                    s.append(nxt)
                tg = [x for x in bssy.get(i.ops[0], []) if x >= 0]
                if tg and nxt not in tg:
                    s += tg
            elif nxt is not None:
                s.append(nxt)
            i.succ = list(dict.fromkeys(s))
        self.preds = [[] for _ in ins]
        for k, i in enumerate(ins):
            for s in i.succ:
                self.preds[s].append(k)
        self.acq = [k for k, i in enumerate(ins) if i.op == "ACQBULK"]

    def prune_armed(self):
        """arm the wait: drop the taken edge of a conditional branch that jumps over an ACQBULK (to the instruction after it)."""
        cut = []
        for a in self.acq:
            for k, i in enumerate(self.ins):
                if i.op == "BRA" and i.pred and k != a and self.byidx.get(target_of(i)) == a + 1:
                    i.succ = [s for s in i.succ if s != a + 1] or [k + 1]
                    cut.append((k, a))
        self.preds = [[] for _ in self.ins]
        for k, i in enumerate(self.ins):
            for s in i.succ:
                self.preds[s].append(k)
        return cut

    def prewait(self):
        seen, st = {0}, [0]
        while st:
            k = st.pop()
            if self.ins[k].op == "ACQBULK":
                continue
            for s in self.ins[k].succ:
                if s not in seen:
                    seen.add(s)
                    st.append(s)
        return {k for k in seen if self.ins[k].op != "ACQBULK"}


# ---- register dataflow: where does a pointer come from ----------------------------------------------------------------
def regs_of(opnd, pair_default=False):
    out = []
    for m in REG.finditer(opnd):
        kind, num, dot64 = m.group(1), int(m.group(2)), m.group(3)
        w = 2 if (dot64 or pair_default) else 1
        out += [(kind, num + j) for j in range(w)]
    return out


def dest_regs(i):
    if not i.ops or NODEST.match(i.op) or i.op.startswith(("ST", "RED")):
        return []
    first = i.ops[0]
    if re.fullmatch(r"U?P[0-9T]\w*", first):
        if not PREDFIRST_REG.match(i.op) or len(i.ops) < 2:
            return []
        first = i.ops[1]
    m = REG.fullmatch(first.split(".")[0]) or REG.match(first)
    if not m:
        return []
    kind, num = m.group(1), int(m.group(2))
    w = 1
    if re.search(r"\.128\b|\.x4\b|\.M88\.4\b", i.op):
        w = 4
    elif re.search(r"\.64\b|WIDE|\.U64\b", i.op) and not i.op.startswith(("SHF", "USHF")):
        w = 2
    return [(kind, num + j) for j in range(w)]


def src_regs(i):
    dst = set(dest_regs(i))
    out = []
    skip = 0
    if dst:
        skip = 2 if (re.fullmatch(r"U?P[0-9T]\w*", i.ops[0])) else 1
    for k, o in enumerate(i.ops):
        if k < skip:
            continue
        pair = (("WIDE" in i.op and k == len(i.ops) - 1) or (i.op.startswith(("UBLK", "UTMA")) and "[" in o)
                or (i.op.endswith(".64") and i.op.startswith(("UIADD3", "IADD3", "UMOV", "MOV"))))
        out += regs_of(o, pair)
    return out


class Slicer:
    def __init__(self, fn, offs, through_loads=True):
        """through_loads: a loaded value carries the provenance of ITS address (data dependence); False = pointer provenance only."""
        self.fn, self.offs, self.memo, self.busy, self.tl = fn, offs, {}, set(), through_loads
        self.dests = [dest_regs(i) for i in fn.ins]

    def member(self, boff):
        best = None
        for off, nm in self.offs:
            if off <= boff:
                best = nm
        return best

    def defs(self, k, reg):
        """reaching definitions of `reg` at instruction k (backward search over CFG predecessors)."""
        out, seen, st = [], set(), list(self.fn.preds[k])
        while st:
            p = st.pop()
            if p in seen:
                continue
            seen.add(p)
            if reg in self.dests[p]:
                out.append(p)
                if not self.fn.ins[p].pred:
                    continue
            st += self.fn.preds[p]
        return out

    def tags_of_def(self, d):
        if d in self.memo:
            return self.memo[d]
        if d in self.busy:
            return set()
        self.busy.add(d)
        i = self.fn.ins[d]
        tg = set()
        for m in CB.finditer(i.text):
            x = int(m.group(1), 16)
            tg.add(("p", self.member(x - PARAM0)) if x >= PARAM0 else ("c", x))
        if i.op.startswith(("LDS", "LDSM")):
            tg.add(("smem", None))
        elif i.op.startswith(("S2R", "S2UR", "CS2R")):
            tg.add(("sreg", None))
        elif not self.tl and i.op.startswith(("LDG", "LD.", "LDGSTS")):
            tg.add(("ld", None))
        else:
            for r in src_regs(i):
                if r[1] < 255 or r[0] == "UR":
                    for dd in self.defs(d, r):
                        tg |= self.tags_of_def(dd)
        self.busy.discard(d)
        self.memo[d] = tg
        return tg

    def tags(self, k, regs):
        tg = set()
        for r in regs:
            for d in self.defs(k, r):
                tg |= self.tags_of_def(d)
        return tg


MEMOPS = re.compile(r"^(LDG|LD\.|LD$|LDGSTS|STG|ST\.|ST$|ATOMG|ATOM\.|REDG|RED\.|UBLKCP|UBLKPF|UBLKRED|UTMALDG|UTMAPF|UTMASTG|UTMAREDG)")


def mem_kind(op):
    if op.startswith(("UBLKPF", "UTMAPF")):
        return "PF"
    if op.startswith(("STG", "ST.", "ST", "REDG", "RED.", "UBLKRED", "UTMASTG", "UTMAREDG")) and not op.startswith(("STS", "STL")):
        return "W"
    if op.startswith(("ATOMG", "ATOM.")):
        return "W"
    return "R"


def classify(members):
    c = "UNKNOWN"
    cl = []
    for m in members:
        if m in WEIGHT:
            cl.append("WEIGHT")
        elif m in UPSTREAM2:
            cl.append("UPSTREAM2")
        elif m in DIAG:
            cl.append("DIAG")
        else:
            cl.append("PRED")
    if cl:
        c = max(cl, key=lambda x: RANK[x])
    return c


def addr_regs(i):
    regs = []
    for o in i.ops:
        if o.startswith("desc["):
            o = o[o.index("]") + 1:]
        if "[" in o and not o.startswith("c["):
            regs += regs_of(o, pair_default=i.op.startswith(("UBLK", "UTMA")))
    return regs


def analyse_fn(fn, offs, ptrs, armed_prune=True):
    """-> dict(acq, preexit, pre_n, accesses[(idx, op, kind, members, cls)], notes)"""
    cut = fn.prune_armed() if armed_prune else []
    pre = fn.prewait()
    sl = Slicer(fn, offs)
    acc = []
    for k in sorted(pre):
        i = fn.ins[k]
        if not MEMOPS.match(i.op) or i.op.startswith(("LDGDEPBAR", "LDGSTS.BYPASS.COMMIT")):
            continue
        tg = sl.tags(k, addr_regs(i))
        mem = sorted({t[1] for t in tg if t[0] == "p" and t[1] in ptrs})
        cls = classify(mem) if mem else "UNKNOWN"
        acc.append((i.idx, i.op, mem_kind(i.op), mem, cls, [t[0] for t in tg if t[0] != "p"]))
    pre_exit = [fn.ins[k].idx for k in sorted(pre) if fn.ins[k].op == "PREEXIT"]
    return dict(acq=[fn.ins[a].idx for a in fn.acq], preexit=pre_exit, pre_n=len(pre), acc=acc, cut=[(fn.ins[a].idx, fn.ins[b].idx) for a, b in cut],
                warn=[w for k, w in fn.warn if k in pre])


def verdict(kernel, res):
    hz = [a for a in res["acc"] if a[2] != "PF" and a[4] in ("PRED", "UNKNOWN") and not (a[2] == "W" and a[4] == "DIAG")]
    cnt = collections.Counter()
    for a in res["acc"]:
        for m in a[3]:
            cnt[(m, a[2])] += 1
    dead = []
    for mem, kind, need in INTENT.get(kernel, []):
        have = cnt[(mem, kind)]
        if have < need:
            dead.append(f"{mem}:{have}/{need}")
    return hz, dead


def parse_lines(txt):
    """nvdisasm -g -c: {function: {idx: (file, line)}} (the innermost frame of an inlined chain)."""
    out, cur, fn = {}, None, None
    for ln in txt.splitlines():
        m = re.match(r"\s*\.text\.(\S+):\s*$", ln)
        if m:
            fn, cur = m.group(1), None
            out[fn] = {}
            continue
        m = re.match(r'\s*//## File "([^"]*)", line (\d+)', ln)
        if m:
            cur = (m.group(1), int(m.group(2)))
            continue
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/", ln)
        if m and fn is not None and cur is not None:
            out[fn][int(m.group(1), 16) // 16] = cur
    return out


def layer_map(u):
    """cu line (1-based) -> (layer-file line, exact?) by aligning the unit's source with its literal in the layer file."""
    import difflib
    lines = pathlib.Path(u["layer"]).read_text(encoding="utf-8").split("\n")
    start = lineno_of(u["layer"], u["origin"])
    cu = u["src"].split("\n")
    a = [x.strip() for x in cu]
    b = [x.strip() for x in lines[start - 1:start - 1 + len(cu) + 800]]
    m, last = {}, (start, False)
    blocks = difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    for i, j, n in blocks:
        for k in range(n):
            m[i + k + 1] = (start + j + k, True)
    cur = (start, False)
    for k in range(1, len(cu) + 1):
        if k in m:
            cur = m[k]
        else:
            m[k] = (cur[0], False)
    return m, lines


def demangle(names):
    r = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True)
    return dict(zip(names, r.stdout.splitlines())) if r.returncode == 0 else {n: n for n in names}


def where(u, lmap, ln, idx, mem=()):
    """'k3_gc_20.py:8564 `source text`' for SASS idx of a function with line table ln.  The leaf frame of __ldg / atomics is a CUDA
    header, so for those the nearest .cu line in the SASS (+-10 instr) is taken and snapped to a `__ldg(` line within +-3 source
    lines (marked ~: approximate)."""
    if ln is None or idx not in ln:
        return "(no line info)"
    f, line = ln[idx]
    m, lines = lmap
    mine = f.endswith(f"{u['name']}.cu")
    if not mine:
        near = [(abs(d), d > 0, ln[idx + d]) for d in range(-10, 11) if idx + d in ln and ln[idx + d][0].endswith(f"{u['name']}.cu")]
        if not near:
            return f"{pathlib.Path(f).name}:{line}"
        line = min(near)[2][1]
    ll, exact = m.get(line, (0, False))
    if not mine:
        ldg = r"__ldg\(|ld\.global\.nc"
        named = [x for x in range(ll - 15, ll + 16) if 0 < x <= len(lines) and re.search(ldg, lines[x - 1])
                 and any(re.search(r"\b(?:\w+\.)?%s\b" % m, lines[x - 1]) for m in mem)]
        cand = [x for x in range(ll - 3, ll + 4) if 0 < x <= len(lines) and re.search(ldg + r"|__restrict__", lines[x - 1])]
        pick = named or cand
        if pick:
            ll = min(pick, key=lambda x: abs(x - ll))
        exact = False
    txt = lines[ll - 1].strip() if 0 < ll <= len(lines) else ""
    return f"{pathlib.Path(u['layer']).name}:{ll}{'' if exact else '~'} `{txt[:96]}`"


def nc_scan(fn, sl, pre, ptrs):
    """every LDG with a .CONSTANT qualifier (ld.global.nc): [(idx, op, members, class, ABOVE|BELOW|NOWAIT)]."""
    rows = []
    for k, i in enumerate(fn.ins):
        if not (i.op.startswith("LDG") and "CONSTANT" in i.op):
            continue
        tg = sl.tags(k, addr_regs(i))
        mem = sorted({t[1] for t in tg if t[0] == "p" and t[1] in ptrs})
        cls = classify(mem) if mem else "UNKNOWN"
        rows.append((i.idx, i.op, mem, cls, "NOWAIT" if not fn.acq else ("ABOVE" if k in pre else "BELOW")))
    return rows


def store_scan(fn, sl, ptrs, watch=("ids", "wts", "tw", "tid", "lid")):
    """members reached by the address slice of every global store / reduction / atomic: (Counter, [(idx, op, members)] touching `watch`)."""
    cnt, hits = collections.Counter(), []
    for k, i in enumerate(fn.ins):
        if mem_kind(i.op) != "W" or not MEMOPS.match(i.op) or i.op.startswith(("STS", "STL")):
            continue
        tg = sl.tags(k, addr_regs(i))
        mm = sorted({t[1] for t in tg if t[0] == "p" and t[1] in ptrs})
        for m in mm:
            cnt[m] += 1
        if set(mm) & set(watch):
            hits.append((i.idx, i.op, mm))
    return cnt, hits


def short(a):
    return f"{a[0]}:{a[1]} {','.join(a[3]) or '?'} [{a[4]}{'' if a[2] == 'R' else '/' + a[2]}]"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    path = pathlib.Path(args[0])
    out = pathlib.Path(args[1]) if len(args) > 1 else pathlib.Path(tempfile.mkdtemp(prefix="pdl_sass_"))
    out.mkdir(parents=True, exist_ok=True)
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    verbose, strict, nc = "-v" in sys.argv, "--allow-latent" not in sys.argv, "--nc-below" in sys.argv
    units = units_of(path, only, nc)
    res = compile_units(units, out, "--reuse" in sys.argv)
    bad = latent = ninst = ndead = ncpred = 0
    for u in units:
        n = u["name"]
        if res[n] != "ok":
            print(f"[pdl_sass] {n}: BUILD {res[n]}")
            bad += 1
            continue
        sass = (out / f"{n}.sass").read_text()
        mem = json.loads((out / f"{n}.members.json").read_text())
        names = list(mem)
        ptrs = {k for k, v in mem.items() if v}
        offp = {int(a): int(b.replace("n", "-")) for a, b in re.findall(r"pdl_off_probeILi(\d+)ELi(n?\d+)E", sass)}
        offs = sorted((offp[i], names[i]) for i in offp if offp[i] >= 0)
        fns = parse_sass(sass)
        lines = parse_lines((out / f"{n}.lines.txt").read_text()) if (out / f"{n}.lines.txt").exists() else {}
        lmap = layer_map(u) if nc else None
        dem = demangle([f for f in fns if "pdl_off_probe" not in f])
        print(f"[pdl_sass] {n}: {u['routes']}" + (f"; params {' '.join(f'{nm}@{o}' for o, nm in offs[:10])} ..." if offs else ""))
        for fname, ins in fns.items():
            kn = next((k for k in u["kernels"] if k in fname), None)
            if kn is None:
                continue
            ninst += 1
            fn = Fn(fname, ins)
            if not u["struct"]:       # plain-parameter kernel: offsets from the signature
                base = re.search(r"(\w+)(?:<|\()", dem[fname].split("::")[-1] if "::" in dem[fname] else dem[fname].split()[-1]).group(1) \
                    if False else kn
                sp = sig_params(u["src"], kn)
                if sp is None:
                    print(f"  {kn}: struct-by-value / unparsable signature, skipped")
                    continue
                offs = sorted((o, nm) for nm, o, _ in sp)
                ptrs = {nm for nm, _, pt in sp if pt}
            r = analyse_fn(fn, offs, ptrs)
            hz, dead = verdict(kn, r)
            armed = u["armed"].get(kn, True)
            agg = collections.Counter((a[1], ",".join(a[3]) or "?", a[4], a[2]) for a in r["acc"])
            v = "HAZARD" if hz else ("PREFETCH-DEAD" if dead else "SAFE")
            if u.get("info"):
                v += " (info only: lazy in-loop waits, K2 reads 2-hop-upstream ids by design)" if hz else " (info)"
                hz, dead = [], []
            if hz and not armed:
                v = "LATENT-HAZARD"
                latent += 1
            elif hz:
                bad += 1
            ndead += bool(dead and not hz)
            title = dem[fname].replace("void ", "")[:70] if (nc or len(u["kernels"]) > 2) else kn
            print(f"  {title}: {len(ins)} instr, ACQBULK@{r['acq'] or 'none'} (wait {'ARMED' if armed else 'DISARMED in ship'}), "
                  f"PREEXIT pre-wait@{r['preexit'] or '-'}, pre-wait region {r['pre_n']} instr, {len(r['acc'])} global accesses above the wait "
                  f"-> {v}{'  (intended pre-wait loads missing: ' + ' '.join(dead) + ')' if dead else ''}")
            for (op, mm, cls, kind), c in sorted(agg.items()):
                print(f"      {c:3d} x {op:<22s} {mm:<12s} {cls}{'' if kind == 'R' else '/' + kind}")
            for a in hz[:6]:
                print(f"      !! {short(a)} via {sorted(set(a[5]))}")
            if verbose:
                for a in r["acc"]:
                    print(f"      {short(a)}")
                if r["cut"]:
                    print(f"      armed: pruned skip-branch(es) (branch idx, ACQBULK idx) {r['cut']}")
            for w in r["warn"][:3]:
                print(f"      INCOMPLETE (BRX above the wait): {w}")
            if nc:
                sl = Slicer(fn, offs, through_loads=False)
                pre = fn.prewait()
                rows = nc_scan(fn, sl, pre, ptrs)
                data = [x for x in rows if x[3] in ("PRED", "UPSTREAM2", "UNKNOWN")]
                ncpred += len(data)
                print(f"    [nc] {len(rows)} LDG.*.CONSTANT loads, {len(data)} of predecessor data / unknown source "
                      f"({sum(1 for x in data if x[4] == 'ABOVE')} above the wait, {sum(1 for x in data if x[4] == 'BELOW')} below, "
                      f"{sum(1 for x in data if x[4] == 'NOWAIT')} no wait):")
                grp = collections.OrderedDict()
                for idx, op, mm, cls, pos in data:
                    wh = where(u, lmap, lines.get(fname), idx, mm)
                    key = (pos, op, ",".join(mm) or "?", "UP2" if cls == "UPSTREAM2" else cls, wh)
                    grp.setdefault(key, []).append(idx)
                for (pos, op, mm, cls, wh), idxs in grp.items():
                    print(f"      {pos:<6s} {len(idxs):2d} x {op:<20s} {mm:<8s} {cls:<7s} idx {idxs[0]}{'..' + str(idxs[-1]) if len(idxs) > 1 else ''}  {wh}")
                if "--no-stores" not in sys.argv:
                    st, hits = store_scan(fn, sl, ptrs)
                    print("    [stores] address slices reaching: " + (", ".join(f"{m}:{c}" for m, c in sorted(st.items())) or "none"))
                    if verbose:
                        for idx, op, mm in hits[:8]:
                            print(f"      store to a routing pointer? idx {idx} {op} {mm}  {where(u, lmap, lines.get(fname), idx)}")
    print(f"[pdl_sass] {path.name}: {ninst} instance(s), {bad} hazard(s) (wait armed in ship), {latent} latent hazard(s), {ndead} prefetch-dead"
          + (f", {ncpred} non-coherent loads of predecessor data" if nc else "") + f" -> "
          f"{'FAIL' if bad or (strict and latent) else 'OK'}")
    return 1 if bad or (strict and latent) else 0


if __name__ == "__main__":
    sys.exit(main())

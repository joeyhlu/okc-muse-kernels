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
PARAM0 = 0x210                                       # c[0x0][0x210]: first kernel parameter byte (sm_90)
TORCH_NV = ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr", "-DTORCH_EXTENSION_NAME=pdl"]

# ---- member classes --------------------------------------------------------------------------------------------------
WEIGHT = {"ws13", "ws2", "gsc13", "gsc2", "nw", "wA", "wB", "gsA", "gsB"}
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


def vms_flags(path):
    """the flag list of _vms_build with G = 132 (the H200), evaluated from the file's own AST."""
    for n in ast.parse(pathlib.Path(path).read_text(encoding="utf-8")).body:
        if isinstance(n, ast.FunctionDef) and n.name == "_vms_build":
            for s in ast.walk(n):
                if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id == "flags":
                    return eval(compile(ast.Expression(s.value), str(path), "eval"), {"G": 132})
    raise SystemExit("[pdl_sass] _vms_build flags not found")


def units_of(path, only):
    ns = literal_nodes(path, ["_vml_JMAX", "_vml_SLOTS", "_vml_FIXW", "_vml_wg_spec", "_vml_wg_spec64", "_vml_CUDA_T", "_vml_CUDA",
                              "_vml_FLAGS", "_VML_DEFS", "VML_VAR", "VML_ABVARS", "VML_BVAR", "VML_BUCKETS", "VML_FLAGS",
                              "VML_ENABLE", "VMS_CUDA_SRC", "VMS_ENABLE"])
    us = [dict(name="vms", src=ns["VMS_CUDA_SRC"], flags=vms_flags(path), kernels=["vms_kernel"], struct="Params", ns="vms",
               routes="T<=16 (VMS_MAXT)", armed={"vms_kernel": True})]       # _vms_call passes pdl=1: PDL attribute set
    vars_ = list(dict.fromkeys([ns["VML_VAR"]] + list(ns["VML_ABVARS"]) + list(ns["VML_BVAR"].values())))
    for v in vars_:
        bk = [b for b, on in ns["VML_BUCKETS"].items() if on and ns["VML_BVAR"].get(b, ns["VML_VAR"]) == v]
        fl = sorted({ns["VML_FLAGS"].get(b, 1) for b in bk})
        us.append(dict(name=f"vm_{v}", src=ns["_vml_CUDA"], flags=list(ns["_vml_FLAGS"]) + list(ns["_VML_DEFS"][v]),
                       kernels=["vm_main", "vm_fin"], struct="Args", ns="vm",
                       routes=f"T-buckets {bk or 'none (self-test / default only)'} flags {fl}",
                       armed={"vm_main": any(f & 256 for f in fl), "vm_fin": True}))   # vm_fin: PDL attribute always set
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
    sem, res = threading.Semaphore(3), {}

    def job(u):
        with sem:
            n = u["name"]
            mem = struct_members(u["src"], u["struct"])
            names = list(mem)
            cu, cub, sas = out / f"{n}.cu", out / f"{n}.cubin", out / f"{n}.sass"
            cu.write_text(probe_src(u["src"], u["ns"], u["struct"], names))
            (out / f"{n}.members.json").write_text(json.dumps(mem))
            if not (reuse and cub.exists() and sas.exists()):
                fl = [f for f in u["flags"] if not f.startswith("-gencode") and f != "-Xptxas=-warn-spills"]
                r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", "-std=c++20", *TORCH_NV, *fl, "-Xptxas=-v", *inc,
                                    "-o", str(cub), str(cu)], capture_output=True, text=True)
                (out / f"{n}.ptxas.log").write_text(r.stdout + r.stderr)
                if r.returncode:
                    res[n] = "FAIL " + " | ".join(ln[-200:] for ln in (r.stdout + r.stderr).splitlines() if "error" in ln)[:600]
                    return
                sas.write_text(subprocess.run([OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout)
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
    def __init__(self, fn, offs):
        self.fn, self.offs, self.memo, self.busy = fn, offs, {}, set()
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
        regs = []
        for o in i.ops:
            if o.startswith("desc["):
                o = o[o.index("]") + 1:]
            if "[" in o and not o.startswith("c["):
                regs += regs_of(o, pair_default=i.op.startswith(("UBLK", "UTMA")))
        tg = sl.tags(k, regs)
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


def short(a):
    return f"{a[0]}:{a[1]} {','.join(a[3]) or '?'} [{a[4]}{'' if a[2] == 'R' else '/' + a[2]}]"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    path = pathlib.Path(args[0])
    out = pathlib.Path(args[1]) if len(args) > 1 else pathlib.Path(tempfile.mkdtemp(prefix="pdl_sass_"))
    out.mkdir(parents=True, exist_ok=True)
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    verbose, strict = "-v" in sys.argv, "--allow-latent" not in sys.argv
    units = units_of(path, only)
    res = compile_units(units, out, "--reuse" in sys.argv)
    bad = latent = ninst = ndead = 0
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
        u["ptrs"] = ptrs
        fns = parse_sass(sass)
        print(f"[pdl_sass] {n}: {u['routes']}; params {' '.join(f'{nm}@{o}' for o, nm in offs[:10])} ...")
        for fname, ins in fns.items():
            kn = next((k for k in u["kernels"] if k in fname), None)
            if kn is None:
                continue
            ninst += 1
            fn = Fn(fname, ins)
            r = analyse_fn(fn, offs, ptrs)
            hz, dead = verdict(kn, r)
            armed = u["armed"].get(kn, True)
            agg = collections.Counter((a[1], ",".join(a[3]) or "?", a[4], a[2]) for a in r["acc"])
            v = "HAZARD" if hz else ("PREFETCH-DEAD" if dead else "SAFE")
            if hz and not armed:
                v = "LATENT-HAZARD"
                latent += 1
            elif hz:
                bad += 1
            ndead += bool(dead and not hz)
            print(f"  {kn}: {len(ins)} instr, ACQBULK@{r['acq'] or 'none'} (wait {'ARMED' if armed else 'DISARMED in ship'}), "
                  f"PREEXIT pre-wait@{r['preexit'] or '-'}, pre-wait region {r['pre_n']} instr, {len(r['acc'])} global accesses above the wait "
                  f"-> {v}{'  (intended pre-wait loads missing: ' + ' '.join(dead) + ')' if dead else ''}")
            for (op, mem, cls, kind), c in sorted(agg.items()):
                print(f"      {c:3d} x {op:<22s} {mem:<12s} {cls}{'' if kind == 'R' else '/' + kind}")
            for a in hz[:6]:
                print(f"      !! {short(a)} via {sorted(set(a[5]))}")
            if verbose:
                for a in r["acc"]:
                    print(f"      {short(a)}")
                if r["cut"]:
                    print(f"      armed: pruned skip-branch(es) (branch idx, ACQBULK idx) {r['cut']}")
            for w in r["warn"][:3]:
                print(f"      INCOMPLETE (BRX above the wait): {w}")
    print(f"[pdl_sass] {path.name}: {ninst} instance(s), {bad} hazard(s) (wait armed in ship), {latent} latent hazard(s), {ndead} prefetch-dead -> "
          f"{'FAIL' if bad or (strict and latent) else 'OK'}")
    return 1 if bad or (strict and latent) else 0


if __name__ == "__main__":
    sys.exit(main())

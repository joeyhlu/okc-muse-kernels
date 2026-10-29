"""rpo_sass.py: SASS gate for GD's griddepcontrol.wait placement (R-rpo-0/1). Run in the local WSL toolchain:

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/rpo_sass.py" <layer.py> [OUTDIR]

Builds the file's GD_CUDA_SRC (device-only sm_90a cubin) with every GD_ROUTE key (+ RPO_GD_ALT alternates if present) and
prints, per gd_kernel instance, the loads ptxas scheduled before ACQBULK (= griddepcontrol.wait): weight stages
(LDG.E.NA.128[.CONSTANT]: ldw / ldwp), escape words (32-bit LDG.CONSTANT), LDGSTS (VF 2 router weights), and ANY other
global load (x = the predecessor's output, LDG.E.128 without .NA). Exit 1 if any instance has such a load above the wait
(a race: x read before the predecessor finished). Weight-stage counts show whether the pre-wait prefetch is real.
"""
import ast
import pathlib
import re
import subprocess
import sys
import tempfile

OBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
DIMS = {"qkv": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}


def key(n, k, rt):
    mt, wr, wk, ku, st_, s, xp = rt
    nt = wr * wk * 32
    ctas = (n // 16 // wr) * s
    return (0, mt, wr, wk, ku, st_, s, max(1, min(2048 // nt, -(-ctas // 132))), xp)


def main():
    path = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(tempfile.mkdtemp())
    out.mkdir(parents=True, exist_ok=True)
    vals = {}
    for n in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("GD_CUDA_SRC", "GD_ROUTE", "RPO_GD_ALT"):
            vals[n.targets[0].id] = ast.literal_eval(n.value)
    keys, names = [], {}
    for (proj, b, *al), rt in sorted(vals["GD_ROUTE"].items(), key=str) + sorted(vals.get("RPO_GD_ALT", {}).items(), key=str):
        if not rt:
            continue
        n, k = DIMS[proj]
        if proj == "unpooler" and b <= 8:
            n = 1024
        kk = key(n, k, rt)
        if kk not in keys:
            keys.append(kk)
        names.setdefault(kk, []).append(f"{proj}/{b}" + (f"/{al[0]}" if al else ""))
    src = vals["GD_CUDA_SRC"]
    src = src.replace("@INST@", "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(map(str, kk))) for i, kk in enumerate(keys)))
    src = src.replace("@INFO@", "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(map(str, kk))) for i, kk in enumerate(keys)))
    cu, cub = out / f"{path.stem}_gd.cu", out / f"{path.stem}_gd.cubin"
    cu.write_text(src)
    p = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-O3", str(cu), "-o", str(cub)], capture_output=True, text=True)
    if p.returncode:
        print("COMPILE FAILED", p.stderr[-1500:])
        return 2
    sass = subprocess.run([OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
    (out / f"{path.stem}_gd.sass").write_text(sass)
    bad = 0
    for blk in sass.split("Function : ")[1:]:
        name = blk.split("\n", 1)[0]
        m = re.search(r"gd_kernelILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", name)
        if not m:
            continue
        t = tuple(int(x) for x in m.groups())
        ins = [mm.group(1) for mm in (re.match(r"\s*/\*[0-9a-f]+\*/\s+(.*?)\s*;", ln) for ln in blk.splitlines()) if mm]
        acq = next((i for i, s in enumerate(ins) if s.startswith("ACQBULK") or " ACQBULK" in s), -1)
        pre = [s for s in ins[:acq]] if acq >= 0 else []
        w = sum(1 for s in pre if "LDG.E.NA.128" in s)
        esc = sum(1 for s in pre if re.search(r"LDG\.E\.CONSTANT ", s))
        sts = sum(1 for s in pre if "LDGSTS" in s)
        other = [s for s in pre if "LDG" in s and "LDGSTS" not in s and "LDGDEPBAR" not in s and "LDG.E.NA.128" not in s
                 and not re.search(r"LDG\.E\.CONSTANT ", s)]
        post = ins[acq + 1:] if acq >= 0 else ins
        xpost = sum(1 for s in post if re.search(r"LDG\.E(\.\w+)*\.128", s) and ".NA." not in s and not s.endswith(".NA"))
        xkind = sorted({re.search(r"LDG\.E[\w.]*", s).group(0) for s in post if re.search(r"LDG\.E(\.\w+)*\.128", s) and ".NA" not in s})
        st, xp = t[5], t[8]
        flag = ""
        if other:
            flag = "  <-- RACE: non-weight load above the wait: " + "; ".join(o[:60] for o in other[:3])
            bad += 1
        print(f"{t[1:9]} VF{t[9]} CAT{t[10]} {names.get(t[:9], ['(VF/CAT)'])}: ACQBULK@{acq} pre-wait weight LDG {w} "
              f"(ST x 3 = {3 * st}{', PF' if xp & 2 else ''}{'+xg' if xp & 4 else ''}) esc {esc} LDGSTS {sts} | post-wait x LDG.128 {xpost} {xkind}{flag}")
    print(f"[rpo_sass] {path.name}: {'FAIL' if bad else 'OK'} ({bad} racy instance(s))")
    return 1 if bad else 0


# ---- --audit: the other non-MoE PDL kernels that read predecessor data (R-rpo-2 step 0) -----------------------------------
# rpo_sass.py --audit <layer.py> [OUTDIR] [--units=ks1,sbb1,...] [-v]
# Each unit's CUDA source is rebuilt exactly as the layer's _build function does (literal strings + the same .replace()s), compiled
# device-only for sm_90a with -lineinfo, and disassembled with nvdisasm -g -gi. Per kernel: the pre-wait set = instructions
# reachable from the entry without passing an ACQBULK (CFG over BRA targets; a predicated ACQBULK counts as executed). Every
# pre-wait global access (LDG / LD / LDGSTS / ATOMG / RED / STG / UTMALDG / UBLKCP) is listed with its outermost source line
# (the kernel-level call site): SRC-PRE = written before the wait in the source (intended; must read graph inputs / weights
# only), HOISTED = written after the wait but scheduled above it by ptxas (only .nc loads can move: a RACE if the address is
# predecessor output). Verdicts need the source text printed next to each access (no pointer provenance here).
NVDIS = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/nvdisasm"
TORCH_NV = ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr", "-DTORCH_EXTENSION_NAME=rpoaudit"]
MEMOP = re.compile(r"^(?:@!?U?P\w+\s+)?(LDG|LD|LDGSTS|ATOMG|ATOM|RED|STG|ST|UTMALDG|UBLKCP|UTMAPF|CCTL)(\.[\w.]+)?\s")


def _layer_ns(path, names):
    ns = {"__name__": "rpo_audit"}
    for n in ast.parse(pathlib.Path(path).read_text(encoding="utf-8")).body:
        tg = [t.id for t in n.targets if isinstance(t, ast.Name)] if isinstance(n, ast.Assign) else []
        if set(tg) & set(names):
            exec(compile(ast.Module(body=[n], type_ignores=[]), str(path), "exec"), ns)
    return ns


def audit_units(path):
    ns = _layer_ns(path, ["K1_CUDA_SRC", "V8_TAIL_CUDA_SRC", "KS1_CUDA_SRC", "KS1_INSTS", "VSL_KS1_INSTS", "VSL_KS3_INSTS",
                          "VSL_ENABLE", "SBB1_CU", "SBB1_HOST", "VG_CU", "VG_HOST", "VGF_CU", "VGF_HOST", "SU1_CUDA_SRC",
                          "VD_CUDA_SRC"])
    ve = ns.get("VSL_ENABLE", False)
    ks1 = (ns["KS1_CUDA_SRC"].replace("@INSTS@", " ".join("KS1_X(%d, %d, %d)" % t for t in ns["KS1_INSTS"]))
           .replace("@MINSTS@", " ".join("KS1_XM(%d, %d, %d, %d)" % t for t in ns["VSL_KS1_INSTS"]) if ve else "")
           .replace("@M3INSTS@", " ".join("KS3_XM(%d, %d)" % t for t in ns["VSL_KS3_INSTS"]) if ve else ""))
    return {"k1": ns["K1_CUDA_SRC"], "tail": ns["V8_TAIL_CUDA_SRC"], "ks1": ks1, "sbb1": ns["SBB1_CU"] + ns["SBB1_HOST"],
            "vgf": ns["VG_CU"] + ns["VG_HOST"] + ns["VGF_CU"] + ns["VGF_HOST"], "su1": ns["SU1_CUDA_SRC"], "vd": ns["VD_CUDA_SRC"]}


def _compile(tag, src, out):
    import sysconfig
    import torch.utils.cpp_extension as ce
    inc = [f"-I{p}" for p in ce.include_paths()] + [f"-I{sysconfig.get_paths()['include']}"]
    cu, cub = out / f"a_{tag}.cu", out / f"a_{tag}.cubin"
    cu.write_text(src)
    r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", "-O3", "-lineinfo", "-std=c++20", *TORCH_NV, *inc, "-o", str(cub), str(cu)],
                       capture_output=True, text=True)
    if r.returncode:
        return None, " | ".join(ln[-160:] for ln in r.stderr.splitlines() if "error" in ln)[:500]
    return subprocess.run([NVDIS, "-c", "-g", "-gi", str(cub)], capture_output=True, text=True).stdout, ""


def _functions(dis):
    """nvdisasm text -> {mangled: [(addr, ins, outer_line)]}; outer_line = last '//## File ..., line N' before the instruction."""
    fns, cur, line, lab, pend = {}, None, None, {}, []
    for ln in dis.splitlines():
        m = re.match(r"^//-+ \.text\.(\S+)", ln.strip())
        if m:
            cur = fns.setdefault(m.group(1), [])
            continue
        m = re.match(r'\s*//## File "([^"]+)", line (\d+)', ln)
        if m:
            line = (m.group(1), int(m.group(2)) if m.group(1).endswith(".cu") else -1)
            continue
        m = re.match(r"^\s*(\.L_x_\d+):", ln)
        if m:
            pend.append(m.group(1))
            continue
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", ln)
        if m and cur is not None:
            cur.append((int(m.group(1), 16), m.group(2), line))
            for lb in pend:
                lab[lb] = int(m.group(1), 16)
            pend = []
    # branch targets are printed as labels: `(.L_x_12) -> 0x<addr> so the CFG walk can follow them
    rl = lambda s: re.sub(r"`\((\.L_x_\d+)\)", lambda q: "0x%x" % lab[q.group(1)] if q.group(1) in lab else q.group(0), s)
    return {f: [(a_, rl(s_), l_) for a_, s_, l_ in v] for f, v in fns.items()}


def _prewait(ins):
    idx = {a: i for i, (a, _, _) in enumerate(ins)}
    seen, todo = set(), [0]
    while todo:
        i = todo.pop()
        if i in seen or i >= len(ins):
            continue
        seen.add(i)
        s = ins[i][1]
        op = re.sub(r"^@!?U?P\w+\s+", "", s)
        if op.startswith("ACQBULK"):
            continue
        pred = s.startswith("@")
        m = re.match(r"(?:BRA|BRA\.\w+|JMP)\s+(?:`\(\.L_x_\d+\)|0x([0-9a-f]+))", op)
        if op.startswith(("EXIT", "RET")) and not pred:
            continue
        if m and m.group(1):
            todo.append(idx.get(int(m.group(1), 16), len(ins)))
            if not pred and not op.startswith("BRA.DIV"):
                continue
        if op.startswith(("BRX", "JMX")):
            todo.extend(range(i + 1, len(ins)))
            continue
        todo.append(i + 1)
    return seen


def audit(path, out, units, verbose):
    import concurrent.futures as cf
    srcs = audit_units(path)
    if units:
        srcs = {k: v for k, v in srcs.items() if k in units}
    with cf.ThreadPoolExecutor(4) as ex:
        res = dict(zip(srcs, ex.map(lambda kv: _compile(kv[0], kv[1], out), srcs.items())))
    names = demangle_all([f for k in res if res[k][0] for f in _functions(res[k][0])])
    bad = 0
    for tag, (dis, err) in res.items():
        if dis is None:
            print(f"[audit] {tag}: COMPILE FAILED {err}")
            bad += 1
            continue
        lines = srcs[tag].split("\n")
        for fn, ins in _functions(dis).items():
            acq = [i for i, (_, s, _) in enumerate(ins) if "ACQBULK" in s]
            if not acq:
                continue
            pre = _prewait(ins)
            wl = max((ins[i][2][1] for i in acq if ins[i][2]), default=0)   # src-pre = written before the LAST wait line
            ops = [(ins[i][1], ins[i][2]) for i in sorted(pre) if MEMOP.match(ins[i][1])]
            hz = [(s, l) for s, l in ops if l and l[1] > wl and not s.split()[0].endswith(("STG", "ST"))]
            short = names.get(fn, fn)[:70]
            print(f"[audit] {tag} {short}: ACQBULK x{len(acq)} @{acq[0]} (src line {wl}) pre-wait mem ops {len(ops)} "
                  f"(src-pre {len(ops) - len(hz)}, HOISTED {len(hz)})")
            show = {}
            for s, l in ops:
                k = (l[1] if l else -1)
                show.setdefault(k, [s.split()[0] if not s.startswith("@") else " ".join(s.split()[:2]), 0])[1] += 1
            for k, (op, c) in sorted(show.items()):
                tag2 = "HOISTED" if k > wl else ("src-pre" if k > 0 else "hdr")
                if verbose or tag2 == "HOISTED":
                    txt = lines[k - 1].strip()[:110] if 0 < k <= len(lines) else "?"
                    print(f"    {tag2} x{c} {op:22s} L{k}: {txt}")
            # reverse direction: loads the source writes BEFORE the (first) wait that ptxas scheduled below it = lost prefetch
            w0 = min((ins[i][2][1] for i in acq if ins[i][2] and ins[i][2][1] > 0), default=0)
            sunk = {}
            for i, (_, s, l) in enumerate(ins):
                if i not in pre and MEMOP.match(s) and l and 0 < l[1] < w0 and not re.match(r"(?:@!?U?P\w+\s+)?(STG|ST)\b", s):
                    op = re.sub(r"^@!?U?P\w+\s+", "", s).split()[0]
                    sunk.setdefault(l[1], []).append(op)
            srcpre = len(ops) - len(hz)
            print(f"    intended pre-wait (src line < {w0}): ABOVE {srcpre} | BELOW (sunk) {sum(len(v) for v in sunk.values())}")
            for k, v in sorted(sunk.items())[:10]:
                print(f"    SUNK x{len(v)} {','.join(sorted(set(v)))} L{k}: {lines[k - 1].strip()[:110] if k <= len(lines) else '?'}")
            if "--nc" in sys.argv:   # post-wait .nc loads: legal for ptxas to hoist above ACQBULK -> latent race on predecessor data
                post = {}
                for i, (_, s, l) in enumerate(ins):
                    if i not in pre and re.search(r"\bLDG\.[\w.]*CONSTANT", s) and l and l[1] > 0:
                        post[l[1]] = post.get(l[1], 0) + 1
                for k, c in sorted(post.items())[:12]:
                    print(f"    post-wait nc x{c} L{k}: {lines[k - 1].strip()[:120] if k <= len(lines) else '?'}")
    return bad


def demangle_all(fns):
    if not fns:
        return {}
    r = subprocess.run(["c++filt"], input="\n".join(fns), capture_output=True, text=True).stdout.splitlines()
    return dict(zip(fns, (re.sub(r"\(.*", "", x) for x in r)))


if __name__ == "__main__":
    if "--audit" in sys.argv:
        a = [x for x in sys.argv[1:] if not x.startswith("-")]
        o = pathlib.Path(a[1]) if len(a) > 1 else pathlib.Path(tempfile.mkdtemp())
        o.mkdir(parents=True, exist_ok=True)
        u = next((x.split("=", 1)[1].split(",") for x in sys.argv if x.startswith("--units=")), None)
        sys.exit(1 if audit(pathlib.Path(a[0]), o, u, "-v" in sys.argv) else 0)
    sys.exit(main())

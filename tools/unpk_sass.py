"""unpk_sass.py: per-stage sm_90a SASS by pipe for the UNPK arms (tools/unpk_patches.py) of a layer file's MoE consumers:
VML bf2 / w72 (stage_m<N> loops at N <= 16, stage<N> loops at N > 16) and VMS (the unit loop). Device-only cubins in WSL.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/unpk_sass.py" OUTDIR [layer.py] [arms=ship,A,B,AB] [ns=8,16,32,64]
A stage loop = a backward-branch loop whose body holds exactly 64 `LOP3 ... 0x81c081c0` (16 code words = one stage per
lane); N from HGMMA.64xN / HMMA count. Pipes: ALU (LOP3 SHF PRMT IADD3 MOV SEL ISETP LEA ...), FMA (IMAD* FFMA FMUL FADD
HMUL2 HFMA2 ...; HMUL2/HFMA2 shown separately), VI (VIADD / VIMNMX), MUFU, LDS (LDS / LDSM), UNI (U* / R2UR), TC, other.
"""
import collections
import pathlib
import re
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/box")]
import m32e_sass_local as S  # noqa: E402
import unpk_patches as P  # noqa: E402
import xks_sass as XS  # noqa: E402

ALU = {"LOP3", "LOP", "SHF", "PRMT", "IADD3", "MOV", "SEL", "ISETP", "LEA", "FSEL", "FMNMX", "IMNMX", "IABS", "PLOP3", "P2R",
       "R2P", "FSETP", "FLO", "POPC", "BMSK", "SGXT", "CS2R", "S2R", "IADD", "ISCADD", "SHL", "SHR", "BFE", "BFI", "MOV32I"}
FMA = {"IMAD", "FFMA", "FMUL", "FADD", "HMUL2", "HFMA2", "HADD2", "IMUL", "FFMA32I", "FMUL32I", "IMAD32I", "HSETP2", "HMNMX2"}
TC = {"HGMMA", "HMMA", "WARPGROUP"}


def pipe(op):
    b = op.split(".")[0]
    if b in ("HMUL2", "HFMA2"):
        return "HM2"
    if b in FMA:
        return "FMA"
    if b in ALU:
        return "ALU"
    if b.startswith("VI"):
        return "VI"
    if b == "MUFU":
        return "MUFU"
    if b in ("LDS", "LDSM"):
        return "LDS"
    if b.startswith("U") or b == "R2UR":
        return "UNI"
    if b in TC:
        return "TC"
    return "oth"


COLS = ("ALU", "FMA", "HM2", "VI", "MUFU", "LDS", "UNI", "TC", "oth")


def op_of(t):
    t = re.sub(r"^@!?U?P\w+\s+", "", t)
    return t.split()[0] if t else "?"


def fn_ins(sass, pat):
    ins, on = [], False
    for ln in sass.splitlines():
        m = re.search(r"Function : (\S+)", ln)
        if m:
            on = re.search(pat, m.group(1)) is not None
            continue
        mm = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?);", ln)
        if on and mm:
            ins.append((int(mm.group(1), 16), mm.group(2).strip()))
    return ins


def loops(ins):
    addr = {a: i for i, (a, _) in enumerate(ins)}
    out = []
    for i, (a, t) in enumerate(ins):
        m = re.search(r"BRA\s+(?:`\(\.L_x_\d+\)\s*)?.*?0x([0-9a-f]+)", t)
        if op_of(t) == "BRA" and m:
            tgt = int(m.group(1), 16)
            if tgt < a and tgt in addr:
                out.append(ins[addr[tgt]:i + 1])
    return out


def nmask(body):
    return sum(1 for _, t in body if op_of(t).startswith("LOP3") and "0x81c081c0" in t)


def hist(body):
    c = collections.Counter(pipe(op_of(t)) for _, t in body)
    c["n"] = len(body)
    c["IMADm"] = sum(1 for _, t in body if op_of(t).startswith("IMAD") and "0x1084" in t)
    c["MOVk"] = sum(1 for _, t in body if "0x1084" in t and not op_of(t).startswith("IMAD"))
    c["PRMT"] = sum(1 for _, t in body if op_of(t).startswith("PRMT"))
    c["IMADall"] = sum(1 for _, t in body if op_of(t).startswith("IMAD"))
    return c


def stage_n(body):
    hg = [re.search(r"HGMMA\.64x(\d+)x16", t) for _, t in body if op_of(t).startswith("HGMMA")]
    hg = [int(m.group(1)) for m in hg if m]
    if hg:
        return ("wg", hg[0])
    nh = sum(1 for _, t in body if op_of(t).startswith("HMMA"))
    return ("m", nh // 2)


def build(nm, dev_path, fl, od):
    cub = od / f"{nm}.cubin"
    r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *fl, "-Xptxas=-v", "-o", str(cub), str(dev_path)], capture_output=True, text=True)
    if r.returncode:
        return None, [ln[-300:] for ln in (r.stdout + r.stderr).splitlines() if "error" in ln][:6]
    regs = re.findall(r"Compiling entry function '(\w+)'[\s\S]*?Used (\d+) registers", r.stdout + r.stderr)
    spills = re.findall(r"(\d+) bytes spill stores", r.stdout + r.stderr)
    s = subprocess.run([S.OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
    (od / f"{nm}.sass").write_text(s)
    return s, (regs, spills)


def vms_dev(src):
    src = src[:src.index("void run(torch::Tensor")]
    for inc in ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>"):
        src = src.replace(inc, "")
    return src


def main():
    od = pathlib.Path(sys.argv[1])
    od.mkdir(parents=True, exist_ok=True)
    rest = sys.argv[2:]
    lay = next((x for x in rest if x.endswith(".py")), "kernels/muse_decode_layer/k3_gc_21cg_s1.py")
    arms = next((x.split("=", 1)[1] for x in rest if x.startswith("arms=")), "ship,A,B,AB").split(",")
    ns_want = [int(v) for v in next((x.split("=", 1)[1] for x in rest if x.startswith("ns=")), "8,16,32,64").split(",")]
    text = (ROOT / lay).read_text(encoding="utf-8")
    flags = {"ship": (0, 0), "A": (1, 0), "B": (0, 1), "AB": (1, 1)}
    jobs = []
    import ast
    for arm in arms:
        a, b = flags[arm]
        t = P.patch(text, bool(a), bool(b)) if (a or b) else text
        tp = od / f"_{arm}_layer.py"
        tp.write_text(t, encoding="utf-8")
        XS.LAYER = tp
        ns = XS.extract_ns()
        vd = od / f"{arm}_vml.cu"
        vd.write_text(XS.dev(ns["_vml_CUDA"]))
        fl0 = [f for f in ns["_vml_FLAGS"] if not f.startswith("-gencode")]
        for var in ("bf2", "w72"):
            jobs.append((arm, var, vd, fl0 + list(ns["_VML_DEFS"][var])))
        tree = ast.parse(t)
        vsrc = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "VMS_CUDA_SRC")
        sd = od / f"{arm}_vms.cu"
        sd.write_text(vms_dev(vsrc))
        jobs.append((arm, "vms", sd, ["-O3", "--expt-relaxed-constexpr", "-DVMS_G=132", "-DVMS_NS=3", "-DVMS_RO=1"]))
    res, sem = {}, threading.Semaphore(4)

    def job(arm, var, dp, fl):
        with sem:
            res[(arm, var)] = build(f"{arm}_{var}", dp, fl, od)
    ths = [threading.Thread(target=job, args=j) for j in jobs]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    print("arm  var  stage     n  " + " ".join(f"{c:>4}" for c in COLS) + "  IMADk MOVk PRMT IMAD  regs")
    for arm, var, _, _ in jobs:
        s, info = res[(arm, var)]
        if s is None:
            print(f"{arm} {var} FAIL {info}")
            continue
        regs = ",".join(f"{r}" for fn, r in info[0] if ("vm_main" in fn or "vms_kernel" in fn)) or "?"
        sp = max((int(x) for x in info[1]), default=0)
        fn = "vms_kernel" if var == "vms" else "vm_main"
        ins = fn_ins(s, fn)
        rows = {}
        for body in loops(ins):
            k = nmask(body)
            if var == "vms":
                if k >= 64 and k % 16 == 0:
                    rows.setdefault(("unit", k // 4), body)
            elif k == 64:
                kind, n = stage_n(body)
                if n in ns_want:
                    key = (kind, n)
                    if key not in rows or len(body) < len(rows[key]):
                        rows[key] = body
        for key in sorted(rows):
            h = hist(rows[key])
            print(f"{arm:4} {var:4} {key[0]:>4}{key[1]:<4} {h['n']:4}  " + " ".join(f"{h[c]:4}" for c in COLS)
                  + f"  {h['IMADm']:5} {h['MOVk']:4} {h['PRMT']:4} {h['IMADall']:4}  {regs} sp{sp}")
        if not rows:
            print(f"{arm:4} {var:4} no stage loop found (kernel {len(ins)} instr)")


if __name__ == "__main__":
    main()

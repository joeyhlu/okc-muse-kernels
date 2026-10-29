"""dr_ptxas.py: compile-only check of the VML variants (and the dr_probe kernels) for sm_90a without a GPU or nvcc:
NVRTC (libnvrtc from `pip install nvidia-cuda-nvrtc-cu12`) -> PTX, then ptxas (`pip install nvidia-cuda-nvcc-cu12`)
-v -warn-spills -warn-lmem-usage. Device code only: the CUDA text is cut at its host entry point and the torch / ATen /
c10 / <vector> includes are dropped (host code is not checked here). Flags mirror torch's load_inline build
(the __CUDA_NO_HALF / BF16 conversion defines). Prints per kernel: registers, spill
stores / loads, stack, and every ptxas warning (wgmma serialization = C7510..C7520 "Potential Performance Loss").

    python -I tools/dr_ptxas.py [--vars=bf2,dr,drp] [--probe] [--out=box_out/dr/ptxas.txt] [--extra=-DVM_DRALL=16]
                                [--cuda=<CUDA 13 wheel tree, e.g. D/nvidia/cu13>] [--layer=<generated layer file>]
With nvdisasm present (CUDA 13 tree: nvidia-cuda-nvdisasm), the spill instructions (STL / LDL) of each kernel are listed
by the device function of their source line (-lineinfo): stage loops vs segment / epilogue code.
"""
from __future__ import annotations

import ctypes
import glob
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
import dr_patches as P  # noqa: E402

TORCH_DEFS = ("-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
              "-D__CUDA_NO_HALF2_OPERATORS__")
STDINT = ("typedef unsigned char uint8_t; typedef unsigned short uint16_t; typedef unsigned int uint32_t; "
          "typedef unsigned long long uint64_t; typedef signed char int8_t; typedef short int16_t; typedef int int32_t; "
          "typedef long long int64_t;\n")


def nvidia_dir():
    import nvidia  # noqa: PLC0415
    return pathlib.Path(list(nvidia.__path__)[0])


def find_tools(root=None):
    """(libnvrtc, ptxas, nvdisasm or None, include dirs). root: a CUDA 13 wheel tree (<root>/bin, lib, include, e.g.
    `pip install --target D nvidia-cuda-nvcc nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cuda-nvdisasm` -> D/nvidia/cu13);
    default: the CUDA 12 wheels of this interpreter (nvidia/cuda_nvrtc, cuda_nvcc, cuda_runtime)."""
    if root:
        r = pathlib.Path(root)
        lib = sorted(glob.glob(str(r / "lib/libnvrtc.so*")))
        ptxas_bin, dis, inc = r / "bin/ptxas", r / "bin/nvdisasm", [str(r / "include")]
    else:
        nv = nvidia_dir()
        lib = sorted(glob.glob(str(nv / "cuda_nvrtc/lib/libnvrtc.so*")))
        ptxas_bin, dis, inc = nv / "cuda_nvcc/bin/ptxas", nv / "cuda_nvcc/bin/nvdisasm", [str(nv / "cuda_runtime/include")]
        if (nv / "cuda_cccl/include").exists():
            inc.append(str(nv / "cuda_cccl/include"))
    lib = [x for x in lib if "alt" not in x and "builtins" not in x]
    if not lib or not ptxas_bin.exists():
        raise SystemExit("[dr_ptxas] need `pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12` "
                         "(or --cuda=<CUDA 13 wheel tree>)")
    return lib[0], str(ptxas_bin), (str(dis) if dis.exists() else None), inc


def device_source(cuda, cut="void vml_run("):
    """device part of a load_inline CUDA text: host entry points cut, torch / std headers dropped."""
    if cut in cuda:
        cuda = cuda[:cuda.index(cut)]
    keep = []
    for ln in cuda.split("\n"):
        s = ln.strip()
        if s.startswith("#include") and not any(h in s for h in ("cuda_bf16.h", "cuda_fp16.h")):
            continue
        keep.append(ln)
    return STDINT + "\n".join(keep)


def nvrtc_ptx(lib, src, opts, name="vml.cu"):
    nv = ctypes.CDLL(lib)
    prog = ctypes.c_void_p()
    r = nv.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), name.encode(), 0, None, None)
    if r:
        raise RuntimeError(f"nvrtcCreateProgram {r}")
    arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    rc = nv.nvrtcCompileProgram(prog, len(opts), arr)
    n = ctypes.c_size_t()
    nv.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    nv.nvrtcGetProgramLog(prog, log)
    if rc:
        raise RuntimeError(f"NVRTC failed ({rc}):\n{log.value.decode(errors='replace')[:6000]}")
    nv.nvrtcGetPTXSize(prog, ctypes.byref(n))
    ptx = ctypes.create_string_buffer(n.value)
    nv.nvrtcGetPTX(prog, ptx)
    nv.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx.value.decode(), log.value.decode(errors="replace").strip()


def ptxas(ptxas_bin, ptx, arch="sm_90a", dis=None):
    """(exit code, -v output, SASS text or None)."""
    with tempfile.TemporaryDirectory() as d:
        p, cb = os.path.join(d, "k.ptx"), os.path.join(d, "k.cubin")
        pathlib.Path(p).write_text(ptx)
        r = subprocess.run([ptxas_bin, f"-arch={arch}", "-v", "-warn-spills", "-warn-lmem-usage", "-O3", p, "-o", cb],
                           capture_output=True, text=True)
        sass = None
        if dis and r.returncode == 0:
            q = subprocess.run([dis, "-c", "-gi", cb], capture_output=True, text=True)
            sass = q.stdout if q.returncode == 0 else None
    return r.returncode, r.stdout + r.stderr, sass


def demangle(n):
    try:
        r = subprocess.run(["c++filt", n], capture_output=True, text=True)
        return r.stdout.strip() or n
    except OSError:
        return n


def parse(out):
    """ptxas -v output -> [(function, regs, spill_st, spill_ld, stack)], [warnings]."""
    rows, cur, warns = [], None, []
    for ln in out.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'|Function properties for (\S+)", ln)
        if m:
            cur = m.group(1) or m.group(2)
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", ln)
        if m and cur:
            rows.append([cur, None, int(m.group(2)), int(m.group(3)), int(m.group(1))])
        m = re.search(r"Used (\d+) registers", ln)
        if m and rows:
            rows[-1][1] = int(m.group(1))
        if "warning" in ln.lower() or "error" in ln.lower():
            warns.append(ln.strip())
    return rows, warns


def spill_sites(sass, src=None):
    """per kernel: spill instructions (STL / LDL) inside stage loops vs elsewhere. A stage loop = a SASS loop (backward
    branch target .. branch) whose body holds HGMMA / HMMA and no global store / reduction / atomic (so not a phase or
    segment loop, whose body has the epilogue); spin-wait loops inside it count as inside. Elsewhere = route / gather /
    producers / segment ends / fixups / epilogues."""
    out = {}
    for blk in re.split(r"\n\s*\.text\.", sass)[1:]:
        name = blk.split(":", 1)[0].strip()
        if not ("vm_main" in name or "drp_" in name):
            continue
        ins, lab, pend = [], {}, []
        for ln in blk.splitlines():
            m = re.match(r"(\.L_x_\d+):", ln.strip())
            if m:
                pend.append(m.group(1))
                continue
            m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*)", ln)
            if m:
                ad = int(m.group(1), 16)
                for lb in pend:
                    lab[lb] = ad
                pend = []
                ins.append((ad, m.group(2)))
        loops = []
        for ad, tx in ins:
            m = re.search(r"\bBRA\b.*?(\.L_x_\d+)", tx)
            if m and m.group(1) in lab and lab[m.group(1)] <= ad:
                loops.append((lab[m.group(1)], ad))
        mma = [ad for ad, tx in ins if re.search(r"\b(HGMMA|HMMA)\b", tx)]
        gst = [ad for ad, tx in ins if re.search(r"\b(STG|REDG|RED|ATOMG|ATOM)\b", tx) and ".EXCH" not in tx]   # EXCH: error flags
        stage = [(lo, hi) for lo, hi in loops
                 if any(lo <= x <= hi for x in mma) and not any(lo <= x <= hi for x in gst)]
        k = demangle(name).split("(")[0].split("::")[-1]
        rec = out.setdefault(k, {"stage_loop": 0, "elsewhere": 0, "stage_loops": len(stage), "mma": len(mma)})
        for ad, tx in ins:
            if re.search(r"\b(STL|LDL)\b", tx):
                rec["stage_loop" if any(lo <= ad <= hi for lo, hi in stage) else "elsewhere"] += 1
    return out


def check_source(tools, cuda, defines, label, cut="void vml_run("):
    lib, ptxas_bin, dis, inc = tools
    src = device_source(cuda, cut)
    opts = ["--gpu-architecture=compute_90a", "--std=c++17", "-default-device", "-lineinfo",
            *[f"-I{i}" for i in inc], *TORCH_DEFS, *defines]
    ptx, log = nvrtc_ptx(lib, src, opts, f"{label}.cu")
    rc, out, sass = ptxas(ptxas_bin, ptx, dis=dis)
    rows, warns = parse(out)
    if rc:
        warns.append(f"ptxas exit {rc}: {out[-2000:]}")
    return rows, warns, log, (spill_sites(sass) if sass else None)


def nvrtc_version(lib):
    nv = ctypes.CDLL(lib)
    a, b = ctypes.c_int(), ctypes.c_int()
    nv.nvrtcVersion(ctypes.byref(a), ctypes.byref(b))
    return f"{a.value}.{b.value}"


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    vars_ = [v for v in args.get("vars", "bf2,dr,drp").split(",") if v]
    extra = tuple(x for x in args.get("extra", "").split(",") if x)
    tools = find_tools(args.get("cuda"))
    lib, ptxas_bin = tools[0], tools[1]
    ver = subprocess.run([ptxas_bin, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    lines = [f"[dr_ptxas] NVRTC {nvrtc_version(lib)}, ptxas {ver}, arch sm_90a"]
    t = (pathlib.Path(args["layer"]).read_text(encoding="utf-8") if "layer" in args
         else P.patch_layer(P.BASE.read_text(encoding="utf-8")))
    lines.append(f"[dr_ptxas] VML source: {args.get('layer', 'k3_gc_6 + tools/dr_patches.py')}")
    ns = {}
    exec(compile("import hashlib, time\n" + P.vml_block(t), "vml_block", "exec"), ns)   # noqa: S102 (our generated text)
    cuda = ns["_vml_CUDA"]
    bad, verdict = False, {}
    jobs = [(v, cuda, P.DEFS[v] + extra, "void vml_run(") for v in vars_]
    if "--probe" in sys.argv:
        import dr_probe  # noqa: PLC0415  (torch-free at import)
        for arm in [a for a in dr_probe.ARMS if a != "ctl"]:   # ctl = inc's code
            jobs.append((f"probe:{arm}", dr_probe.probe_cuda(cuda), dr_probe.arm_defines(arm) + extra, dr_probe.HOST_MARK))
    for label, src, defs, cut in jobs:
        try:
            rows, warns, log, sites = check_source(tools, src, defs, label.replace(":", "_"), cut)
        except RuntimeError as ex:
            lines.append(f"[dr_ptxas] {label}: FAILED {ex}")
            bad = True
            verdict[label] = "COMPILE FAILED"
            continue
        lines.append(f"[dr_ptxas] {label} ({' '.join(defs)})")
        spill, inloop = 0, 0
        for fn, regs, sst, sld, stack in rows:
            if fn.startswith("_Z") and ("vm_main" in fn or "vm_fin" in fn or "drp_" in fn):
                lines.append(f"    {demangle(fn)[:90]:90s} regs {regs:4} spill st/ld {sst}/{sld} B, stack {stack} B")
                spill += sst + sld
        wg = [w for w in warns if "wgmma" in w.lower() or re.search(r"C75\d\d", w)]
        for w in warns:
            lines.append(f"    WARN {w[:300]}")
        lines.append(f"    wgmma serialization warnings: {len(wg)}; other ptxas warnings: {len(warns) - len(wg)}")
        if sites is not None:
            for k, r in sorted(sites.items()):
                lines.append(f"    SASS {k}: {r['stage_loops']} stage loops ({r['mma']} HGMMA/HMMA); spill instructions in stage "
                             f"loops {r['stage_loop']}, elsewhere {r['elsewhere']}")
                inloop += r["stage_loop"]
        bad |= bool(wg) or inloop > 0
        verdict[label] = ("wgmma SERIALIZED" if wg else "STAGE-LOOP SPILLS" if inloop else "spill-free" if spill == 0 else
                          f"spills outside the stage loops only ({spill} B st+ld)" if sites is not None else
                          f"spills {spill} B st+ld (no nvdisasm: stage-loop placement not checked)")
        if log:
            lines.append(f"    nvrtc log: {log[:400]}")
    for label, v in verdict.items():
        lines.append(f"[dr_ptxas] verdict {label:12s} {v}")
    lines.append("[dr_ptxas] " + ("FAIL: a compile failure, wgmma serialization or a spill inside a stage loop" if bad else
                                  "OK: every build compiles, no wgmma serialization, no spill inside a stage loop"))
    txt = "\n".join(lines)
    print(txt)
    if "out" in args:
        o = ROOT / args["out"]
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(txt + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

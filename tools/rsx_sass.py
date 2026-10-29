"""rsx_sass.py: R-sfx-3 ring audit ($0, no GPU). Captures every load_inline unit a layer file builds (load_inline is
stubbed; the layer is imported under local torch), compiles each device-only for sm_90a (nvcc -cubin, <= 2 at once) and
scans every release-type SASS instruction (SYNCS.ARRIVE*, ARRIVES.*, BAR.ARV / BAR.SYNC / BAR.RED, UCGABAR_ARV, RED / ATOM
after MEMBAR) for LDS / LDSM write scoreboards still open when it issues (control bits from the hi word: wbar 46-48,
wait mask 52-57; DEPBAR.LE SBk, n keeps the last n loads of SBk open: in-order return). Linear back-scan of 240 instr,
so a site whose loads sit across a back edge is re-checked by hand.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/rsx_sass.py" OUTDIR capture [--layer=kernels/...py]
  ... rsx_sass.py OUTDIR compile UNIT[,UNIT]          # cubin + sass per unit (2 nvcc at once)
  ... rsx_sass.py OUTDIR scan UNIT[,UNIT] [--v] [--fn=substr]
"""
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import threading
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
OBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
TORCH_NV = ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr", "-DTORCH_EXTENSION_NAME=rsx"]   # load_inline defaults
REL = re.compile(r"SYNCS\.ARRIVE|ARRIVES|BAR\.(ARV|SYNC|RED)|UCGABAR_ARV|^RED|^ATOM")


class _Cap(Exception):
    pass


def capture(out, layer):
    units = {}

    def fake(name, cpp_sources=None, cuda_sources=None, functions=None, extra_cuda_cflags=(), **kw):
        cu = cuda_sources if isinstance(cuda_sources, str) else "\n".join(cuda_sources)
        units[name] = (cu, list(extra_cuda_cflags or ()))
        raise _Cap(name)
    import torch.utils.cpp_extension as ce
    ce.load_inline = fake
    import importlib.abc
    import importlib.machinery

    class _Stub(types.ModuleType):
        def __getattr__(self, k):
            if k.startswith("__"):
                raise AttributeError(k)
            return _Stub(f"{self.__name__}.{k}")

        def __call__(self, *a, **k):
            return a[0] if a and callable(a[0]) else self

    class _F(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in ("sglang", "sgl_kernel", "cutlass", "flash_attn", "deep_gemm", "flashinfer", "quack", "cuda"):
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None

        def create_module(self, spec):
            m = _Stub(spec.name)
            m.__path__ = []
            return m

        def exec_module(self, m):
            pass
    sys.meta_path.insert(0, _F())
    import ast
    L = types.ModuleType("lay")
    L.__file__ = str(ROOT / layer)
    sys.modules["lay"] = L
    nbad = 0
    for node in ast.parse((ROOT / layer).read_text(encoding="utf-8")).body:   # node by node: CuTe-DSL parts may fail
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), L.__file__, "exec"), L.__dict__)
        except Exception:
            nbad += 1
    print(f"[rsx] layer exec: {nbad} top-level nodes failed (stubbed deps)")
    L.PT1_PREFETCH = False
    calls = [("v10", L._v10_build_all), ("vms", L._vms_build), ("ks1", L._ks1_build), ("gd", L._gd_build),
             ("vfa", L._vfa_build), ("vd", L._vd_build), ("su1", L._su1_build), ("sbb1", L._sbb1_build),
             ("vg", L._vg_build), ("g2", L._g2_build), ("p1", lambda: L.p1_build(layer_only=True)), ("lt", L._lt_build)]
    calls += [(f"vml_{v}", (lambda v=v: L._vml_build(defines=L._VML_DEFS[v]))) for v in ("bf2", "w72")]
    for tag, fn in calls:
        n0 = len(units)
        try:
            fn()
        except BaseException as ex:  # noqa: BLE001
            if not isinstance(ex, _Cap) and len(units) == n0:
                print(f"[rsx] {tag}: no capture ({type(ex).__name__}: {str(ex)[:120]})")
        for k in list(units)[n0:]:
            cu, fl = units[k]
            (out / f"{tag}.cu").write_text(cu)
            (out / f"{tag}.json").write_text(json.dumps({"name": k, "flags": fl}))
            print(f"[rsx] {tag}: {k} {len(cu)} B flags {[f for f in fl if f.startswith('-D')]}")
            break


def compile_units(out, names):
    import torch.utils.cpp_extension as ce
    inc = [f"-I{p}" for p in ce.include_paths()] + [f"-I{__import__('sysconfig').get_paths()['include']}"]
    sem, res = threading.Semaphore(2), {}

    def job(u):
        with sem:
            fl = [f for f in json.loads((out / f"{u}.json").read_text())["flags"] if not f.startswith("-gencode")]
            fl = [f for f in fl if f != "-Xptxas=-v"]
            r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", "-std=c++20", *TORCH_NV, *fl, "-Xptxas=-v", *inc, "-o", str(out / f"{u}.cubin"),
                                str(out / f"{u}.cu")], capture_output=True, text=True)
            log = r.stdout + r.stderr
            (out / f"{u}.ptxas.log").write_text(log)
            if r.returncode:
                res[u] = "FAIL " + " | ".join(ln[-160:] for ln in log.splitlines() if "error" in ln)[:400]
                return
            s = subprocess.run([OBJ, "-sass", str(out / f"{u}.cubin")], capture_output=True, text=True).stdout
            (out / f"{u}.sass").write_text(s)
            res[u] = f"ok {len(s)} B sass"
    ths = [threading.Thread(target=job, args=(u,)) for u in names]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for u in names:
        print(f"[rsx] {u}: {res[u]}")


def parse(sass):
    """{function: [(addr, text, wbar, waitmask)]}"""
    fns, cur, lines, i = {}, None, sass.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if "Function :" in ln:
            cur = ln.split("Function :")[1].strip()
            fns[cur] = []
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;?\s*/\* (0x[0-9a-f]{16}) \*/", ln)
        if cur and m:
            c = int(re.search(r"/\* (0x[0-9a-f]{16}) \*/", lines[i + 1]).group(1), 16) >> 41
            fns[cur].append((int(m.group(1), 16), m.group(2).strip().rstrip(";").strip(), (c >> 5) & 7, (c >> 11) & 63))
            i += 2
            continue
        i += 1
    return fns


def scan(ins, back=240):
    out = []
    for k, (a, t, _, _) in enumerate(ins):
        tt = re.sub(r"^@!?U?P\w+\s+", "", t)
        if not REL.search(tt):
            continue
        if tt.startswith(("RED", "ATOM")) and not any("MEMBAR" in ins[j][1] for j in range(max(0, k - 6), k)):
            continue
        pend, gdep = {}, None
        for j in range(max(0, k - back), k + 1):
            aj, y, wb, wt = ins[j]
            for b in range(6):
                if wt >> b & 1:
                    pend.pop(b, None)
            mm = re.search(r"DEPBAR\.LE SB(\d), (0x[0-9a-f]+|\d+)", y)
            if mm:
                b, n = int(mm.group(1)), int(mm.group(2), 0)
                if b in pend:
                    pend[b] = pend[b][len(pend[b]) - n:] if n else []
                    if not pend[b]:
                        pend.pop(b)
            mg = re.search(r"WARPGROUP\.DEPBAR\.LE gsb0, (0x[0-9a-f]+|\d+)", y)   # wgmma smem reads (no SB): pseudo-SB "G"
            if mg:
                gdep = (aj, int(mg.group(1), 0))
            if mg and "G" in pend:
                n = int(mg.group(1), 0)
                pend["G"] = pend["G"][len(pend["G"]) - n:] if n else []
                if not pend["G"]:
                    pend.pop("G")
            if j < k and re.search(r"^HGMMA", re.sub(r"^@!?U?P\w+\s+", "", y)) and "gsb0" in y:
                pend.setdefault("G", []).append((aj, "HGMMA"))
            if j < k and wb != 7 and re.search(r"\bLDS|\bLDSM", re.sub(r"^@!?U?P\w+\s+", "", y)):
                pend.setdefault(wb, []).append((aj, y.split()[0] if not y.startswith("@") else y.split()[1]))
        if "G" in pend and gdep:
            pend["dep"] = [gdep]
        out.append((a, tt.split(" ")[0], tt, pend))
    return out


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[2]
    lay = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--layer=")), "kernels/muse_decode_layer/k3_gc_19.py")
    if mode == "capture":
        capture(out, lay)
        return
    names = sys.argv[3].split(",")
    if mode == "compile":
        compile_units(out, names)
        return
    fsub = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--fn=")), "")
    for u in names:
        log = (out / f"{u}.ptxas.log").read_text()
        fns = parse((out / f"{u}.sass").read_text())
        for f, ins in fns.items():
            if fsub not in f:
                continue
            sc = scan(ins)
            if not sc:
                continue
            nop = sum(1 for *_, p in sc if p)
            kinds = {}
            for _, op, _, p in sc:
                kinds.setdefault(op, [0, 0])
                kinds[op][0] += 1
                kinds[op][1] += bool(p)
            ks = " ".join(f"{k}:{v[1]}/{v[0]}" for k, v in sorted(kinds.items()))
            print(f"[rsx] {u} {f[:70]} {len(ins)} instr | open/all {ks}")
            if "--v" in sys.argv:
                for a, op, t, p in sc:
                    if p:
                        print(f"     {a:#07x} {t[:60]:60s} open " + " ".join(f"SB{b}:" + ",".join(f"{x:#x}{o}" for x, o in v[-3:])
                                                                       + (f"(+{len(v) - 3})" if len(v) > 3 else "") for b, v in p.items()))
        regs = sorted({ln.split("Used ")[1][:24] for ln in log.splitlines() if "Used" in ln})
        sp = sorted({m for m in re.findall(r"(\d+) bytes spill stores", log)})
        if "--regs" in sys.argv:
            print(f"[rsx] {u} regs {regs[:8]} spill-store sizes {sp}")


if __name__ == "__main__":
    main()

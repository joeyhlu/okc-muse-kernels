"""lat32_sass.py: E0 SASS / register audit of the lat32 lab files' VML (bf2 = bucket 32, w72 = 128 / 256), device-only sm_90a.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/lat32_sass.py" OUTDIR arm=layer[:define,define] ...
  e.g. ship=kernels/lab/k3_lat32_0.py nwf=kernels/lab/k3_lat32_1.py cf=kernels/lab/k3_lat32_c.py:-DVM_CFLAGS=193
Per arm and variant: ptxas regs / spill bytes of vm_main + vm_fin, vm_main instruction count, opcode mix (KEYS), and the
per-stage loop estimate (instructions between consecutive HGMMA blocks are not separated: whole-kernel counts only).
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
import xks_sass as XS  # noqa: E402

KEYS = ["HGMMA", "LOP3", "HMUL2", "PRMT", "SHF", "IMAD", "R2UR", "BAR", "SYNCS", "FENCE", "MEMBAR", "RED", "ATOM", "STL", "LDL",
        "LDG", "STG", "BRA", "NANOSLEEP"]


def funcs(sass):
    out, cur = {}, None
    for ln in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", ln)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(.*?);", ln)
        if cur and m:
            out[cur].append(m.group(1))
    return out


def opmix(ins):
    c = collections.Counter()
    for t in ins:
        op = re.sub(r"^@!?U?P\w+\s+", "", t).split()[0]
        for k in KEYS:
            if op.startswith(k):
                c[k] += 1
    fences = [re.sub(r"^@!?U?P\w+\s+", "", t) for t in ins if "FENCE" in t or "MEMBAR" in t]
    return c, collections.Counter(f.split(";")[0] for f in fences)


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    arms = {}
    for a in sys.argv[2:]:
        nm, spec = a.split("=", 1)
        lay, _, defs = spec.partition(":")
        arms[nm] = (lay, [d for d in defs.split(",") if d])
    res, sem = {}, threading.Semaphore(3)
    srcs = {}
    for nm, (lay, _) in arms.items():
        XS.LAYER = ROOT / lay
        ns = XS.extract_ns()
        (out / f"{nm}_dev.cu").write_text(XS.dev(ns["_vml_CUDA"]))
        srcs[nm] = ns

    def job(nm, var):
        with sem:
            ns, defs = srcs[nm], arms[nm][1]
            cub = out / f"{nm}_{var}.cubin"
            fl = [f for f in ns["_vml_FLAGS"] if not f.startswith("-gencode")] + list(ns["_VML_DEFS"][var]) + defs
            r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *fl, "-Xptxas=-v", "-o", str(cub), str(out / f"{nm}_dev.cu")],
                               capture_output=True, text=True)
            log = r.stdout + r.stderr
            if r.returncode:
                res[(nm, var)] = ("FAIL", [ln[-300:] for ln in log.splitlines() if "error" in ln][:6])
                return
            s = subprocess.run([S.OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
            (out / f"{nm}_{var}.sass").write_text(s)
            info = {}
            cur = None
            for ln in log.splitlines():
                m = re.search(r"Compiling entry function '(\w+)'", ln)
                if m:
                    cur = m.group(1)
                m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ln)
                if m and cur:
                    info.setdefault(cur, {})["spill"] = (int(m.group(1)), int(m.group(2)))
                m = re.search(r"Used (\d+) registers", ln)
                if m and cur:
                    info.setdefault(cur, {})["regs"] = int(m.group(1))
            res[(nm, var)] = ("ok", (info, funcs(s)))
    items = [(nm, v) for nm in arms for v in ("bf2", "w72")]
    ths = [threading.Thread(target=job, args=it) for it in items]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for nm, var in items:
        st, info = res[(nm, var)]
        if st != "ok":
            print(f"[lat32-sass] {nm}:{var} FAIL {info}")
            continue
        pt, fs = info
        for fn, ins in fs.items():
            if "vm_main" not in fn and "vm_fin" not in fn:
                continue
            key = next((k for k in pt if k == fn), None)
            r = pt.get(key, {})
            c, fe = opmix(ins)
            print(f"[lat32-sass] {nm}:{var} {'main' if 'vm_main' in fn else 'fin '} regs {r.get('regs')} spill {r.get('spill')} "
                  f"instr {len(ins)} | " + " ".join(f"{k} {c[k]}" for k in KEYS if c[k]) + f" | fences {dict(fe)}")


if __name__ == "__main__":
    main()

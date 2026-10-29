"""icx_sass.py: device-only sm_90a cubins of the ICX arms (tools/gen_k3_icx.py, k3_icx_0 VML source) in WSL: .text size of
vm_main / vm_fin (ELF section headers), ptxas regs / spills, SASS hash (== k3_gc_14 for 'ship' and '0'). Two nvcc at a time.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/icx_sass.py" OUTDIR bf2:ship,bf2:0,bf2:a,... [--dump] [-DX=Y ...]
"""
import hashlib
import pathlib
import re
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools")]
import gen_k3_icx as G  # noqa: E402
import m32e_sass_local as S  # noqa: E402
import vrb_sass as V  # noqa: E402


def text_sizes(cub):
    r = subprocess.run(["readelf", "-S", "-W", str(cub)], capture_output=True, text=True).stdout
    out = {}
    for ln in r.splitlines():
        m = re.search(r"\]\s+\.text\.(\S+)\s+PROGBITS\s+\S+\s+\S+\s+([0-9a-f]+)", ln)
        if m:
            k = "main" if "vm_main" in m.group(1) else ("fin" if "vm_fin" in m.group(1) else m.group(1)[:20])
            out[k] = int(m.group(2), 16)
    return out


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    arms = sys.argv[2].split(",")
    extra = tuple(a for a in sys.argv[3:] if a.startswith("-D"))
    nss = {"ship": V.ns_of(G.BASE), "icx": V.ns_of(ROOT / "kernels/lab/k3_icx_0.py")}
    cus = {}
    for k, ns in nss.items():
        src, fl = V.dev(ns)
        (out / f"{k}.cu").write_text(src)
        cus[k] = (out / f"{k}.cu", fl, ns["_VML_DEFS"])
    res, sem = {}, threading.Semaphore(2)

    def job(arm):
        var, code = arm.split(":")
        cu, fl, D = cus["ship" if code == "ship" else "icx"]
        cub = out / f"{var}_{code}.cubin"
        with sem:
            r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *fl, *D[var], *G.defs(var, code), *extra, "-Xptxas=-v",
                                "-o", str(cub), str(cu)], capture_output=True, text=True)
        if r.returncode:
            res[arm] = "FAIL\n" + "\n".join((r.stdout + r.stderr).splitlines()[-12:])
            return
        regs, fn = {}, None
        for ln in (r.stdout + r.stderr).splitlines():
            m = re.search(r"Compiling entry function '(\S+)'", ln)
            if m:
                fn = "main" if "vm_main" in m.group(1) else ("fin" if "vm_fin" in m.group(1) else None)
            m = re.search(r"Used (\d+) registers", ln)
            if m and fn:
                regs[fn] = int(m.group(1))
            m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ln)
            if m and fn and (int(m.group(1)) or int(m.group(2))):
                regs[fn + "_spill"] = m.group(0)
        s = subprocess.run([S.OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
        if "--dump" in sys.argv:
            (out / f"{var}_{code}.sass").write_text(s)
        per = S.counts(s)
        tot = {("main" if "vm_main" in f else "fin"): c["total"] for f, c in per.items() if "vm_main" in f or "vm_fin" in f}
        body = re.sub(r"/\*[0-9a-f]{4}\*/", "", s)
        h = hashlib.sha1("\n".join(ln for ln in body.splitlines() if "Function" in ln or ln.strip().startswith("/*")
                                   or re.match(r"\s+[A-Z@]", ln)).encode()).hexdigest()[:10]
        ts = text_sizes(cub)
        res[arm] = (f"text main {ts.get('main', 0) / 1024:.1f} KB fin {ts.get('fin', 0) / 1024:.1f} KB | regs {regs} "
                    f"sass {tot} hash {h}")

    ths = [threading.Thread(target=job, args=(a,)) for a in arms]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for a in arms:
        print(f"[icx-sass] {a:9s} {res[a]}", flush=True)


if __name__ == "__main__":
    main()

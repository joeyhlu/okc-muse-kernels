"""sfx_sass.py: device-only sm_90a cubin of k3_gc_18's VML (bf2 / w72) with and without tools/sfx_patches.py SFX=1, then a
scoreboard scan of every SYNCS.ARRIVE in vm_main: which LDS / LDSM write scoreboards are still open (set, never waited by a
wait mask or DEPBAR.LE SBk, 0) when the arrive issues. Control bits from the hi word (wbar 46-48, rbar 49-51, wait 52-57).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/sfx_sass.py" OUTDIR [--layer=kernels/...py]
"""
import pathlib
import re
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/box")]
import m32e_sass_local as S  # noqa: E402
import sfx_patches as F  # noqa: E402
import xks_sass as XS  # noqa: E402


def parse(sass, fsub="vm7vm_main"):
    lines, ins, inf, i = sass.split("\n"), [], False, 0
    while i < len(lines):
        ln = lines[i]
        if "Function :" in ln:
            inf = fsub in ln
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;?\s*/\* (0x[0-9a-f]{16}) \*/", ln)
        if inf and m:
            c = int(re.search(r"/\* (0x[0-9a-f]{16}) \*/", lines[i + 1]).group(1), 16) >> 41
            ins.append((int(m.group(1), 16), m.group(2).strip().rstrip(";").strip(), (c >> 5) & 7, (c >> 11) & 63))
            i += 2
            continue
        i += 1
    return ins


def scan(ins):
    """[(addr, barrier offset, open {sb: n_loads})] for every SYNCS.ARRIVE (linear back-scan of 200 instr)."""
    out = []
    for k, (a, t, _, _) in enumerate(ins):
        if "SYNCS.ARRIVE" not in t:
            continue
        pend = {}
        for j in range(max(0, k - 200), k + 1):
            _, y, wb, wt = ins[j]
            for b in range(6):
                if wt >> b & 1:
                    pend.pop(b, None)
            mm = re.search(r"DEPBAR\.LE SB(\d), (0x[0-9a-f]+|\d+)", y)
            if mm and int(mm.group(2), 0) == 0:
                pend.pop(int(mm.group(1)), None)
            if j < k and wb != 7 and re.search(r"\bLDS", y):
                pend[wb] = pend.get(wb, 0) + 1
        off = re.search(r"\+(0x[0-9a-f]+)\]", t)
        out.append((a, off.group(1) if off else "?", pend))
    return out


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    lay = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--layer=")), None)
    if lay:
        XS.LAYER = ROOT / lay
    ns = XS.extract_ns()
    srcs = {"ship": XS.dev(ns["_vml_CUDA"]), "sfx": XS.dev(F.patch(ns["_vml_CUDA"]))}
    for k, v in srcs.items():
        (out / f"{k}_dev.cu").write_text(v)
    res, sem = {}, threading.Semaphore(2)

    def job(var, arm):
        with sem:
            cub = out / f"{var}_{arm}.cubin"
            fl = [f for f in ns["_vml_FLAGS"] if not f.startswith("-gencode")] + list(ns["_VML_DEFS"][var]) + (["-DSFX=1"] if arm == "sfx" else [])
            r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *fl, "-Xptxas=-v", "-o", str(cub), str(out / f"{arm}_dev.cu")],
                               capture_output=True, text=True)
            log = r.stdout + r.stderr
            if r.returncode:
                res[(var, arm)] = ("FAIL", [ln[-300:] for ln in log.splitlines() if "error" in ln][:6])
                return
            s = subprocess.run([S.OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
            (out / f"{var}_{arm}.sass").write_text(s)
            regs = [ln.split("Used ")[1][:14] for ln in log.splitlines() if "Used" in ln]
            spill = sorted({m for m in re.findall(r"(\d+) bytes spill stores", log)})
            res[(var, arm)] = ("ok", (regs, spill, parse(s)))
    items = [(v, a) for v in ("bf2", "w72") for a in ("ship", "sfx")]
    ths = [threading.Thread(target=job, args=it) for it in items]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for var, arm in items:
        st, info = res[(var, arm)]
        if st != "ok":
            print(f"[sfx-sass] {var}:{arm} FAIL {info}")
            continue
        regs, spill, ins = info
        sc = scan(ins)
        offs = sorted({o for _, o, _ in sc})
        summ = "; ".join(f"{o}: {sum(1 for _, oo, p in sc if oo == o and p)}/{sum(1 for _, oo, _ in sc if oo == o)} open" for o in offs)
        print(f"[sfx-sass] {var}:{arm:4s} vm_main {len(ins)} instr, regs {' / '.join(regs[-2:])}, spill {spill} | arrives {summ}")
        for a, o, p in sc:
            if p and "--v" in sys.argv:
                print(f"     {a:#07x} {o} open {p}")


if __name__ == "__main__":
    main()

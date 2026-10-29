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
        st, xp = t[5], t[8]
        flag = ""
        if other:
            flag = "  <-- RACE: non-weight load above the wait: " + "; ".join(o[:60] for o in other[:3])
            bad += 1
        print(f"{t[1:9]} VF{t[9]} CAT{t[10]} {names.get(t[:9], ['(VF/CAT)'])}: ACQBULK@{acq} pre-wait weight LDG {w} "
              f"(ST x 3 = {3 * st}{', PF' if xp & 2 else ''}) esc {esc} LDGSTS {sts}{flag}")
    print(f"[rpo_sass] {path.name}: {'FAIL' if bad else 'OK'} ({bad} racy instance(s))")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

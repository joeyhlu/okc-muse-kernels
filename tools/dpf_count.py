"""dpf_count.py <cubin>...: per PDL kernel, count 128-bit global loads above / below ACQBULK by kind (NA = weight ldw, other)."""
import re, subprocess, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import rpo_sass as R
for cub in sys.argv[1:]:
    dis = subprocess.run([R.NVDIS, "-c", "-g", "-gi", cub], capture_output=True, text=True).stdout
    fns = R._functions(dis)
    names = R.demangle_all(list(fns))
    for fn, ins in fns.items():
        if not any("ACQBULK" in s for _, s, _ in ins):
            continue
        pre = R._prewait(ins)
        c = {}
        for i, (_, s, _) in enumerate(ins):
            m = re.search(r"\b(LDG\.[\w.]+|LDGSTS[\w.]*|UBLKCP[\w.]*)", s)
            if not m:
                continue
            k = ("pre " if i in pre else "post ") + m.group(1)
            c[k] = c.get(k, 0) + 1
        print(names.get(fn, fn)[:60], "|", ", ".join(f"{k} {v}" for k, v in sorted(c.items())))

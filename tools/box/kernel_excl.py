"""Per-kernel exclusive / busy us per shape (ours side) from a layer_budget sqlite, in launch order."""
import json, re, statistics, sys
from collections import defaultdict
sys.path.insert(0, "tools/box")
import layer_budget as LB

def short(n):
    n = re.sub(r"\(.*", "", n)
    n = n.replace("kernel_cutlass_kernel_submissionk1fa4_", "FA4:")
    return n[:44]

db, meta_p = sys.argv[1], sys.argv[2]
shapes = [s for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
meta = json.loads(open(meta_p).read())
by = LB.load_sqlite(db)
for idx, info in sorted(meta["shapes"].items(), key=lambda kv: int(kv[0])):
    if shapes and idx not in shapes:
        continue
    ex, bu, spans, order = defaultdict(list), defaultdict(list), [], None
    r = 0
    while f"G|sub|{idx}|{r}" in by:
        gk = by[f"G|sub|{idx}|{r}"]
        cnt = defaultdict(int)
        rows = []
        for k in gk:
            nm = short(k["name"]); cnt[nm] += 1
            rows.append((k["start"], k["end"], f"{nm}#{cnt[nm]}", "main"))
        e, span, b = LB.exclusive(rows)
        for kk, v in e.items(): ex[kk].append(v)
        for kk, v in b.items(): bu[kk].append(v)
        spans.append(span)
        if order is None:
            t0 = rows[0][0]
            order = [(nm, (a - t0) / 1e3) for a, _, nm, _ in rows]
        r += 1
    if not spans:
        continue
    print(f"[{idx}] {info.get('label', '')} span {statistics.median(spans)/1e3 if statistics.median(spans) > 1e4 else statistics.median(spans):.1f}")
    for nm, st in order:
        e = statistics.median(ex[nm]); b = statistics.median(bu[nm])
        e = e / 1e3 if e > 1e4 else e; b = b / 1e3 if b > 1e4 else b
        print(f"   {st:7.1f}  excl {e:6.1f}  busy {b:6.1f}  {nm}")

"""KT1 report: judge per-shape us (PS routed via kernel_fn) + the lab's KT1 lines, against the pre-registered rule
(notes/study/dense_independent.md section 6).  python -I tools/dense/kt1_report.py kernels/results/<job>.json"""
import json
import re
import sys

GO_T1 = {"pooler": 13.9, "qkv_gate": 9.6, "o": 5.4}
GO_T32 = {"pooler": 14.4, "qkv_gate": 10.0, "o": 5.8}
GO_SLICE_T8 = 3.9


def main(path):
    d = json.load(open(path, encoding="utf-8"))
    r = d.get("result", d)
    print(f"job {d.get('id')} status {r.get('status')} passed {r.get('passed')} speedup {r.get('speedup')}")
    bench = [x for x in r.get("records", []) if x.get("event") == "benchmark"]
    judge = {}
    for b in bench:
        c = b.get("case", {})
        sp = b.get("speedup") or (b.get("baseline_ns", 0) / b["ns"] if b.get("ns") else None)
        ours = b.get("ns") or b.get("median_ns")
        base = b.get("baseline_ns")
        if ours is None and base and sp:
            ours = base / sp
        judge[(c.get("layer"), c.get("m"))] = (base / 1e3 if base else None, ours / 1e3 if ours else None)
        print(f"  {c.get('layer'):9s} m={c.get('m'):<4} sglang {base / 1e3 if base else float('nan'):6.2f} "
              f"ours {ours / 1e3 if ours else float('nan'):6.2f} x{(base / ours) if base and ours else float('nan'):.3f}")
    for t in [x for x in r.get("records", []) if x.get("event") == "test"]:
        if t.get("status") != "pass":
            print("  TEST FAIL", t.get("case"), t.get("message"))
    tail = r.get("log_tail", "")
    for line in tail.splitlines():
        if line.startswith(("KT1", "PS test", "PS info", "PS routes", "PS build", "PS prepare", "PS err", "PS pack")):
            print("  " + line[:220])
    # rule on the judge's own per-shape times (best PS arm routed by kernel_fn)
    def j(layer, m):
        v = judge.get((layer, m))
        return v[1] if v else None
    print("GO check (judge us):")
    ok = True
    for layer, thr in GO_T1.items():
        v = j(layer, 1)
        ok &= v is not None and v <= thr
        print(f"  T=1 {layer:9s} {v if v is None else round(v, 2)} <= {thr}")
    for layer, thr in GO_T32.items():
        v = j(layer, 32)
        ok &= v is not None and v <= thr
        print(f"  T=32 {layer:9s} {v if v is None else round(v, 2)} <= {thr}")
    v = j("unpooler", 8)
    ok &= v is not None and v <= GO_SLICE_T8
    print(f"  T=8 unpooler slice {v if v is None else round(v, 2)} <= {GO_SLICE_T8}")
    print("GO" if ok else "no GO on the judge numbers")
    for layer in ("qkv_gate", "pooler"):
        a, b8, b32 = j(layer, 1), j(layer, 8), j(layer, 32)
        if a and b8 and b32:
            print(f"  per-token {layer}: t8-t1 {b8 - a:+.2f} (<= 0.3)  t32-t1 {b32 - a:+.2f} (<= 1.0)")
    m = re.search(r"KT1 sweep", tail)
    if not m:
        print("  (no KT1 sweep lines in the log tail)")


if __name__ == "__main__":
    main(sys.argv[1])

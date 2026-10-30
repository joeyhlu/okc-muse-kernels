"""lead_gate.py: the lead's $0 gates for a bundle before / after its judge checks.

  python -I tools/lead_gate.py file <bundle.py> [--raw=<raw base>]
      ast parse, ascii, < 1,000,000 B, okc header line 1, every decorated def re-parses from its own source text
      (tools/pyslim.decorated_sources), third-party notices: BSD / Copyright / License / Apache / NOTICE line counts
      >= the raw base's (default kernels/lab/k3_b29_raw.py), THIRD_PARTY_NOTICES present if the file names it.
  python -I tools/lead_gate.py jobs <job> [<job> ...] [--vs=<job>,<job>,...]
      per-shape judge speedup of the jobs (and their mean ratio to the --vs runs), pass / test counts / flags, and the
      attention + MoE kernels each shape's profile ran (FA4 / KS / VA1 / AX1 / vjeux / VMS / VML / hybrid live where intended).
"""
import ast
import glob
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pyslim  # noqa: E402

LIMIT = 1_000_000
NOTICE_WORDS = ("BSD", "Copyright", "License", "Apache", "NOTICE")
ATTN = re.compile(r"fa4|ks1|ks3|va1|vfa|ax1|ax2|k1::|attn|attention|flash|decode_kernel|split", re.I)
MOE = re.compile(r"vms|vm_main|vm_fin|k2a|moe|expert|marlin|fused", re.I)


def gate_file(path, raw):
    t = pathlib.Path(path).read_text(encoding="utf-8")
    ok = True
    n = len(t.encode())
    print(f"size {n} B ({'OK' if n < LIMIT else 'OVER'} the cap by {abs(LIMIT - n)} B)")
    ok &= n < LIMIT
    print("line 1:", t.splitlines()[0])
    ok &= t.splitlines()[0] == "# okc: problem=muse_decode_layer"
    ast.parse(t)
    asc = t.isascii()
    print("ascii", asc)
    ok &= asc
    d = pyslim.decorated_sources(t)
    bad = [s for s in d if s.startswith("UNPARSABLE")]
    print(f"decorated defs {len(d)}, unparsable {len(bad)} {bad[:2]}")
    ok &= not bad
    r = pathlib.Path(raw).read_text(encoding="utf-8")
    for w in NOTICE_WORDS:
        print(f"notice word {w!r}: {len(re.findall(w, t, re.I))} (raw {len(re.findall(w, r, re.I))})")
    # notices follow the code that is present: vendored FA4 (CuTe DSL) needs the raw base's BSD-3 block; public
    # Apache code (vjeux / alygg77 / HYA / HYB) needs THIRD_PARTY_NOTICES with the Apache text
    fa4 = bool(re.search(r"import cutlass|@cute\.(kernel|jit)", t))
    bsd = len(re.findall("BSD", t)) >= len(re.findall("BSD", r))
    print(f"FA4 CuTe code present: {fa4}; BSD block kept: {bsd}")
    ok &= bsd or not fa4
    apache = bool(re.search(r"vjeux|alygg77|HYA_|HYB_", t))
    m = re.search(r"^THIRD_PARTY_NOTICES\s*=", t, re.M)
    has = bool(m) and "Apache License" in t
    print(f"public Apache code present: {apache}; THIRD_PARTY_NOTICES assigned with Apache text: {has}"
          f" (referenced {len(re.findall('THIRD_PARTY_NOTICES', t)) - 1}x)")
    ok &= has or not apache
    print("GATE", "PASS" if ok else "FAIL")
    return ok


def load(j):
    f = glob.glob(str(ROOT / f"kernels/results/{j}*.json"))
    if not f:
        sys.exit(f"no results for {j}")
    return json.load(open(f[0]))


def speedups(d):
    return {(x["case"]["load"], x["case"]["layer"]): x["speedup"] for x in d["result"]["records"]
            if x.get("event") == "benchmark"}


def kernels(d):
    out = {}
    for x in d["result"]["records"]:
        if x.get("event") != "profile":
            continue
        names = []
        for line in x["report"].splitlines()[1:]:
            m = re.match(r"(.+?)\s{2,}([\d.]+)\s", line)
            if m:
                names.append((m.group(1).strip(), float(m.group(2))))
        a = [f"{n[:34]} {u:.0f}" for n, u in names if ATTN.search(n)][:2]
        e = [f"{n[:22]} {u:.0f}" for n, u in names if MOE.search(n)][:2]
        out[(x["case"]["load"], x["case"]["layer"])] = (a, e)
    return out


def gate_jobs(jobs, vs):
    ds = {j: load(j) for j in jobs}
    for j, d in ds.items():
        r = d["result"]
        tests = [x for x in r["records"] if x.get("event") == "test"]
        print(f"{j} {d['cluster']} {r['speedup']:.4f} passed {r['passed']} tests {sum(x['status'] == 'pass' for x in tests)}"
              f"/{len(tests)} flagged {r['checks'].get('flagged')} rejected {r['checks'].get('rejected')}")
    sp = {j: speedups(d) for j, d in ds.items()}
    base = [speedups(load(j)) for j in vs]
    kn = kernels(next(iter(ds.values())))
    for k in sp[jobs[0]]:
        a = [sp[j][k] for j in jobs]
        line = f"{k[0]:>9s} {k[1]:7s} " + " ".join(f"{x:6.3f}" for x in a)
        if base:
            b = [s[k] for s in base]
            line += f" | vs {sum(b) / len(b):6.3f} {sum(a) / len(a) / (sum(b) / len(b)) * 100 - 100:+5.1f}%"
        att, moe = kn.get(k, ([], []))
        print(line, "|", "; ".join(att), "|", "; ".join(moe))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if args[0] == "file":
        sys.exit(0 if gate_file(args[1], opts.get("raw", ROOT / "kernels/lab/k3_b29_raw.py")) else 1)
    gate_jobs(args[1:], [v for v in opts.get("vs", "").split(",") if v])


if __name__ == "__main__":
    main()

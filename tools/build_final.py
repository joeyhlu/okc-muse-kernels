"""build_final.py: one reproducible build of the final decode_layer candidate kernels/lab/k3_hh_3.py (review 8.1).

  python tools/build_final.py [--work=build_final] [--strict] [--no-gate] [--from-gc21] [--stop-after=N] [--start-at=N] [-v]
                              [--scored-copy=<path> [--note=<text>]]

Regenerates k3_hh_3 from the scored ship kernels/muse_decode_layer/k3_gc_21cg_s1.py by running the 12 patch steps below in
order (each as a fresh `python -I -X utf8 tools/<patcher>` process from the repo root; outputs go to --work, never over a
committed file). For every step it prints the input / output sha1 and size, compares the output with the committed
intermediate (kernels/lab/*, when one exists) and with the sha1 recorded here, and warns if a patcher / helper file differs
from the blob this recipe was recorded with. Last, it asserts the rebuilt file is byte-identical to the committed
kernels/lab/k3_hh_3.py (--strict: lines 1-2 too; default: a difference in lines 1-2 only is reported and tolerated) and runs
tools/lead_gate.py on it.

--from-gc21 adds step 0: kernels/muse_decode_layer/k3_gc_21.py (scored 2.1887) -> tools/cg_patches.py --trig=1 --ph=0 -> the ship
k3_gc_21cg_s1 (lines 3.. byte-identical; the line-2 nonce was typed by hand and is set here). Step 0's output must equal the
committed ship, because robust_patches.py (step 1) can only read its base by name from kernels/muse_decode_layer/.

The two things that were done by hand when the chain was first built are automated here: the BND attn-256 route entry (step 6,
an anchored one-line replace; no committed patcher exists) and the line-2 nonce of k3_b29_raw (step 8 post-edit; hya's --nonce
and rob2's derived nonce overwrite it later, so it only matters for byte-identity of the committed k3_b29_raw).

--scored-copy=<path>: also write a scored/check copy of the rebuilt file there (line 2 replaced by `# nonce <stem> (= kernels/lab/
k3_hh_3.py: <note>)`, everything else byte-identical; refuses to overwrite). See kernels/lab/RECIPE.md.
Exit code: 0 = rebuilt == committed (and, unless --no-gate, the gate passed), 1 = mismatch / step failure / gate FAIL.
"""
import argparse
import difflib
import hashlib
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = "kernels/muse_decode_layer/k3_gc_21cg_s1.py"
BASE_SHA1 = "9752060f1c503aff1557ade0599b96d89f9280d5"
BASE_GC21 = "kernels/muse_decode_layer/k3_gc_21.py"
BASE_GC21_SHA1 = "310ad0acffb9b92a0ea0eeef5edf784190f495fa"
FINAL = "kernels/lab/k3_hh_3.py"
LIMIT = 1_000_000

NONCE_GC21CG = ("# nonce k3-gc-21cg-s1 (k3_gc_21 + CG trig: _k1fa4_combine_gate PREEXIT after its wait + gate load first; "
                "tools/cg_patches.py --ph=0; R-cg-1/2)")
NONCE_B29_RAW = ("# nonce k3-b29-raw (UNSLIMMED patch base: b24_v -> dpin q128,q256,o128,u256 -> BND256 -> ax2 -> ax1 --route=1 "
                 "--slim=0; slim_patches(this) == k3_b29_1 from line 3)")
NONCE_HYA_3 = ("k3-hya-3 (b29_raw + tools/hya_patches.py --mode=ship: k3_hya_1 + prepare-only HYA workspace; "
               "slim_patches last)")
BND_OLD = "('ffn', 16): (16, None), ('ffn', 32): (16, None)}\n"
BND_NEW = "('ffn', 16): (16, None), ('ffn', 32): (16, None), ('attn', 256): (1, (1024, True))}\n"

# git blob sha1 of every file the patchers read (patcher + helper modules + vendored / third-party inputs) when the recipe was
# recorded. A mismatch is a warning (the byte comparison of the outputs is the real test); it names what to look at.
DEP_BLOBS = {
    "tools/cg_patches.py": "078710b494ae431d93997562f6d26ff2986dea98",   # only used by --from-gc21
    "tools/robust_patches.py": "8032d99ad0950aff72943b560a6dd951cf036bd2",
    "tools/fnc_patches.py": "bad466e26ad8c61bd198cd7d6f79256ff94e4503",
    "tools/rider3_patches.py": "dd3b09bd3a57f09072ab385373586b228983006c",
    "tools/vse_patches.py": "5d051a09dd5181407a06576d3adbd29abec19e9d",
    "tools/dpin_patches.py": "a86fc0cbac81d61bebb2d3ea043e116ff1b3f716",
    "tools/ulp_patches.py": "7df7e6ecc4a8446cc494e359de8015a0439d32da",
    "tools/ax2_patches.py": "d6b1e3aff85178f5986d08224554b1e643b6b906",
    "tools/ax1_patches.py": "485631dee36f8d2378650fb60b17b5e3f5ddce48",
    "tools/ax1_src.py": "c3a2a55f10c45a1640ba93b631c70c79540652ac",
    "tools/hya_patches.py": "c9799b894d400591231d2c8afaa83cadac6eb084",
    "tools/hya_src.py": "d310fa5f348910b7c832604368639455e47a7e5e",
    "tools/vendor/vjeux_okc2026_f35a41ee/LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
    "tools/vendor/vjeux_okc2026_f35a41ee/muse_attention_decode.py": "3ae30d24058745a28f13c50021f7f387fe3da7bc",
    "tools/hyb_patches.py": "9ec0059085fa15ea85c2ccdc804c673a99cf9ef2",
    "tools/hyb_unifdef.py": "49161db7ec9918d57610b88deb80d9acbe9e6456",
    "third_party/alygg77-okc-muse-spark-kernels-6cae49a/LICENSE": "d645695673349e3947e8e5ae42332d0ac3164cd7",
    "third_party/alygg77-okc-muse-spark-kernels-6cae49a/NOTICE": "8d1069372b982b1f4fc7bcf263635ba56594a73c",
    "third_party/alygg77-okc-muse-spark-kernels-6cae49a/kernels/muse_decode_layer.py": "02389585ece4c36bf983e2cd616a50c0f3a9364f",
    "tools/rob2_patches.py": "a45c9afaf4e482cf7e03653e5d5724678a5ff662",
    "tools/pyslim.py": "3d521f58a5c49ee884a0befc551b05afd06fe8bf",
    "tools/slim_patches.py": "af514d9055754a6e3ec2017dd4777ae868bf9229",
    "tools/lead_gate.py": "44da1a756ce9e6ef4185efc8a2620beee93d4189",
}


def _set_line2(text, line2):
    lines = text.split("\n")
    if not lines[1].startswith("# nonce "):
        raise SystemExit("[build_final] line 2 is not a '# nonce' line")
    lines[1] = line2
    return "\n".join(lines)


def _bnd256(text):
    if text.count(BND_OLD) != 1:
        raise SystemExit(f"[build_final] BND256 anchor (V9_BND_ROUTE tail): found {text.count(BND_OLD)}, expected 1")
    return text.replace(BND_OLD, BND_NEW, 1)


# Each step: n, name, out (file name in --work; its stem feeds the derived nonce of robust / fnc / rider3 / rob2), tool, argv
# ({in} / {out} = repo-relative posix paths), deps (files read), ref (committed intermediate), sha1 / size (recorded output),
# optional fn (in-process text edit instead of a tool) and post (text edit after the tool).
STEP0 = dict(n=0, name="cg", out="k3_gc_21cg_s1.py", tool="tools/cg_patches.py",
             argv=["--base={in}", "--out={out}", "--trig=1", "--ph=0"], deps=["tools/cg_patches.py"],
             post=lambda t: _set_line2(t, NONCE_GC21CG),
             ref=BASE, sha1=BASE_SHA1, size=983524)
STEPS = [
    dict(n=1, name="robust", out="k3_rob_1.py", tool="tools/robust_patches.py",
         argv=["--r1", "--r2", "--base=k3_gc_21cg_s1", "--out={out}"], deps=["tools/robust_patches.py"],
         ref="kernels/lab/k3_rob_1.py", sha1="4f7b213c09dc9715a16177cd219131e919b367ec", size=987732),
    dict(n=2, name="fnc", out="k3_b23_1.py", tool="tools/fnc_patches.py",
         argv=["--base={in}", "--out={out}", "--clean=1"], deps=["tools/fnc_patches.py"],
         ref="kernels/lab/k3_b23_1.py", sha1="57a34c50e9d89e87c092005805d36ae379030803", size=989388),
    dict(n=3, name="rider3", out="k3_rider3_1.py", tool="tools/rider3_patches.py",
         argv=["--base={in}", "--out={out}", "--r1", "--r2", "--r3"], deps=["tools/rider3_patches.py"],
         ref="kernels/lab/k3_rider3_1.py", sha1="c93b69240cea26609d54187fa2f4a4c290d61ac3", size=991535),
    dict(n=4, name="vse", out="k3_b24_v.py", tool="tools/vse_patches.py",
         argv=["{in}", "--ps=1", "--out={out}"], deps=["tools/vse_patches.py"],
         ref="kernels/lab/k3_b24_v.py", sha1="c0e6bb41e0407641beaa2556fc4e56bb6e17eee6", size=997861),
    dict(n=5, name="dpin", out="k3_dpin_2.py", tool="tools/dpin_patches.py",
         argv=["--base={in}", "--out={out}", "--pins=q128,q256,o128,u256"], deps=["tools/dpin_patches.py", "tools/ulp_patches.py"],
         ref="kernels/lab/k3_dpin_2.py", sha1="2aa4a8fe0c425588897c3bdb4feb207902af22f7", size=999922),
    dict(n=6, name="bnd256", out="k3_bnd256.py", fn=_bnd256, deps=[],
         ref=None, sha1="5a10ad4bfadb8817333c02d0a1ad7fdb3a20ac88", size=999956),
    dict(n=7, name="ax2", out="k3_ax2_b29.py", tool="tools/ax2_patches.py",
         argv=["--base={in}", "--out={out}", "--vx=3", "--st=3"], deps=["tools/ax2_patches.py"],
         ref=None, sha1="4501ba472a275d38c6bdc3b9a821f65c6e9c9bc7", size=1006851),
    dict(n=8, name="ax1", out="k3_b29_raw.py", tool="tools/ax1_patches.py",
         argv=["--base={in}", "--out={out}", "--route=1", "--slim=0"], deps=["tools/ax1_patches.py", "tools/ax1_src.py"],
         post=lambda t: _set_line2(t, NONCE_B29_RAW),
         ref="kernels/lab/k3_b29_raw.py", sha1="6933008fa2ad9388265e82cdb7e69d448e431e59", size=1054699),
    dict(n=9, name="hya", out="k3_hya_3r.py", tool="tools/hya_patches.py",
         argv=["--base={in}", "--out={out}", "--mode=ship", "--nonce=" + NONCE_HYA_3],
         deps=["tools/hya_patches.py", "tools/hya_src.py", "tools/vendor/vjeux_okc2026_f35a41ee/LICENSE",
               "tools/vendor/vjeux_okc2026_f35a41ee/muse_attention_decode.py"],
         ref="kernels/lab/k3_hya_3r.py", sha1="21e81c4fe9611c8e885f686b11638bb7b4ab73a9", size=957234),
    dict(n=10, name="hyb", out="k3_hh_3r.py", tool="tools/hyb_patches.py",
         argv=["--base={in}", "--out={out}", "--pdl=1"],
         deps=["tools/hyb_patches.py", "tools/hyb_unifdef.py", "third_party/alygg77-okc-muse-spark-kernels-6cae49a/LICENSE",
               "third_party/alygg77-okc-muse-spark-kernels-6cae49a/NOTICE",
               "third_party/alygg77-okc-muse-spark-kernels-6cae49a/kernels/muse_decode_layer.py"],
         ref="kernels/lab/k3_hh_3r.py", sha1="f38403522e88afc7caaef7c1f7362ebd6e9d98ef", size=1018977),
    dict(n=11, name="rob2", out="k3_hh_3rr.py", tool="tools/rob2_patches.py",
         argv=["--base={in}", "--out={out}", "--no-p2"], deps=["tools/rob2_patches.py", "tools/pyslim.py", "tools/slim_patches.py"],
         ref="kernels/lab/k3_hh_3rr.py", sha1="f2167cbcb618a6ecd000940cfe114cf471288870", size=1018927),
    dict(n=12, name="slim", out="k3_hh_3.py", tool="tools/slim_patches.py",
         argv=["--base={in}", "--out={out}"], deps=["tools/slim_patches.py"],
         ref="kernels/lab/k3_hh_3.py", sha1="88e2c90f936012b115aa6d18f5c32d57f605c7ae", size=935816),
]


def sha1(b):
    return hashlib.sha1(b).hexdigest()


def blob_sha1(b):
    return sha1(b"blob %d\0" % len(b) + b)


def rel(p):
    return pathlib.Path(p).resolve().relative_to(ROOT).as_posix()


def ref_bytes(path):
    """Committed bytes of a repo file: the HEAD blob when git answers, else the working-tree file."""
    try:
        r = subprocess.run(["git", "cat-file", "blob", "HEAD:" + path], cwd=ROOT, capture_output=True, timeout=60)
        if r.returncode == 0:
            return r.stdout, "HEAD"
    except (OSError, subprocess.SubprocessError):
        pass
    f = ROOT / path
    return (f.read_bytes(), "worktree") if f.exists() else (None, "missing")


def fmt(b):
    return f"{sha1(b)} {len(b):>9,d} B"


def run_tool(tool, argv, verbose):
    cmd = [sys.executable, "-I", "-X", "utf8", tool] + argv
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    dt = time.time() - t0
    lines = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
    if r.returncode != 0:
        print("\n".join("        | " + ln[:300] for ln in lines[-25:]))
        raise SystemExit(f"[build_final] step failed: exit {r.returncode}: {' '.join(cmd[3:])}")
    for ln in (lines if verbose else lines[-1:]):
        print("        | " + ln[:220] + (" ..." if len(ln) > 220 else ""))
    return dt


def report_diff(got, want, name):
    """Exactly what differs between two byte strings (line based)."""
    a, b = got.split(b"\n"), want.split(b"\n")
    print(f"        {name}: rebuilt {len(got):,d} B / {len(a):,d} lines vs committed {len(want):,d} B / {len(b):,d} lines")
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    shown = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        shown += 1
        if shown > 12:
            print("        ... more differing blocks omitted")
            break
        print(f"        {tag}: rebuilt lines {i1 + 1}-{i2}, committed lines {j1 + 1}-{j2}")
        for ln in a[i1:min(i2, i1 + 3)]:
            print("          - " + ln.decode("utf-8", "replace")[:200])
        for ln in b[j1:min(j2, j1 + 3)]:
            print("          + " + ln.decode("utf-8", "replace")[:200])


def scored_copy(final, dest, note):
    dest = pathlib.Path(dest)
    dest = dest if dest.is_absolute() else ROOT / dest
    if dest.exists():
        raise SystemExit(f"[build_final] {dest} exists (scored copies are never overwritten)")
    lines = final.split(b"\n")
    assert lines[1].startswith(b"# nonce ")
    name = dest.stem.replace("_", "-")
    lines[1] = f"# nonce {name} (= {FINAL}: {note})".encode()
    out = b"\n".join(lines)
    assert out.split(b"\n", 2)[2] == final.split(b"\n", 2)[2] and out.isascii() and len(out) < LIMIT and b"\r" not in out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(out)
    print(f"[build_final] scored copy {rel(dest)}: {len(out):,d} B, sha1 {sha1(out)}; lines 3.. identical to the rebuilt file")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", default="build_final", help="build directory inside the repo (default build_final/)")
    ap.add_argument("--strict", action="store_true", help="also require lines 1-2 (okc header + nonce) to match")
    ap.add_argument("--no-gate", action="store_true", help="skip tools/lead_gate.py")
    ap.add_argument("--from-gc21", action="store_true", help="add step 0: k3_gc_21 -> cg_patches -> the ship k3_gc_21cg_s1")
    ap.add_argument("--start-at", type=int, default=None, help="resume at step N (needs the step N-1 output in --work)")
    ap.add_argument("--stop-after", type=int, default=STEPS[-1]["n"])
    ap.add_argument("--scored-copy", default="", help="write a scored copy of the rebuilt file to this path")
    ap.add_argument("--note", default="rebuilt by tools/build_final.py; scored", help="text after the colon in the copy's nonce line")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the whole tool output")
    a = ap.parse_args()

    work = (ROOT / a.work).resolve()
    if ROOT not in work.parents or not work.name.startswith("build"):
        raise SystemExit("[build_final] --work must be a directory named build* inside the repo (the patchers print repo-relative paths)")
    work.mkdir(exist_ok=True)

    print(f"[build_final] repo {ROOT}, python {sys.version.split()[0]}, work dir {rel(work)}/")
    steps = ([STEP0] if a.from_gc21 else []) + STEPS
    needed = sorted({d for s in steps for d in s["deps"]} | {"tools/lead_gate.py"})
    bad = 0
    for path in needed:
        want = DEP_BLOBS[path]
        f = ROOT / path
        got = blob_sha1(f.read_bytes()) if f.exists() else "MISSING"
        if got != want:
            bad += 1
            print(f"[build_final] WARN {path}: blob {got[:12]} != recorded {want[:12]} (patcher / input changed since the recipe was recorded)")
    print(f"[build_final] patcher + helper + vendored inputs: {len(needed) - bad}/{len(needed)} match the recorded blobs")

    first = steps[0]["n"]
    start = first if a.start_at is None else a.start_at
    base_path, base_sha = (BASE_GC21, BASE_GC21_SHA1) if a.from_gc21 else (BASE, BASE_SHA1)
    base = (ROOT / base_path).read_bytes()
    if sha1(base) != base_sha:
        raise SystemExit(f"[build_final] base {base_path} sha1 {sha1(base)} != recorded {base_sha}")
    cur_path, cur = base_path, base
    if start > first:
        prev = [s for s in steps if s["n"] == start - 1][0]
        cur_path = rel(work / prev["out"])
        cur = (ROOT / cur_path).read_bytes()
    print(f"[build_final] base  {base_path}\n              {fmt(base)}")

    first_bad = None
    for s in steps:
        if s["n"] < start or s["n"] > a.stop_after:
            continue
        out_path = work / s["out"]
        if out_path.exists():
            out_path.unlink()          # fnc / vse / rob patchers refuse to overwrite; the work dir is ours
        out_rel = rel(out_path)
        desc = (f"{s['tool']} {' '.join(x for x in s['argv'] if not x.startswith('--nonce'))}".replace("{in}", cur_path).replace("{out}", out_rel)
                if "tool" in s else "inline: V9_BND_ROUTE ('attn', 256): (1, (1024, True))")
        print(f"\n[{s['n']:>2}/{len(STEPS)}] {s['name']}: {desc}")
        print(f"        in : {pathlib.PurePosixPath(cur_path).name:<18s} {fmt(cur)}")
        t0 = time.time()
        if "tool" in s:
            argv = [x.replace("{in}", cur_path).replace("{out}", out_rel) for x in s["argv"]]
            run_tool(s["tool"], argv, a.verbose)
            if not out_path.exists():
                raise SystemExit(f"[build_final] {s['tool']} did not write {out_rel}")
        else:
            out_path.write_bytes(s["fn"](cur.decode("utf-8")).encode("utf-8"))
        if "post" in s:
            out_path.write_bytes(s["post"](out_path.read_bytes().decode("utf-8")).encode("utf-8"))
        new = out_path.read_bytes()
        if b"\r" in new:
            print("        WARN output contains CR bytes (a patcher wrote CRLF; the committed files are LF-only)")
        marks = []
        if s["ref"]:
            rb, src = ref_bytes(s["ref"])
            marks.append(f"== committed {s['ref']} ({src})" if rb == new else f"!= committed {s['ref']}")
            if rb != new and rb is not None:
                report_diff(new, rb, s["out"])
        else:
            marks.append("(no committed intermediate)")
        marks.append("== recorded sha1" if sha1(new) == s["sha1"] else f"!= RECORDED sha1 {s['sha1']}")
        if sha1(new) != s["sha1"] and first_bad is None:
            first_bad = s
        if s["n"] == 0 and new != ref_bytes(BASE)[0]:
            raise SystemExit("[build_final] step 0 did not reproduce the committed ship; step 1 (robust_patches --base=<name>) reads the committed file")
        print(f"        out: {s['out']:<18s} {fmt(new)}  [{time.time() - t0:.1f}s]\n             " + "; ".join(marks))
        cur_path, cur = out_rel, new

    if a.stop_after < STEPS[-1]["n"]:
        print(f"\n[build_final] stopped after step {a.stop_after}")
        return 0

    # ---- the assertion: rebuilt == committed k3_hh_3
    want, src = ref_bytes(FINAL)
    print(f"\n[build_final] FINAL  rebuilt {cur_path}\n              {fmt(cur)}\n              committed {FINAL} ({src})\n              {fmt(want)}")
    rc = 0
    if cur == want:
        print("[build_final] RESULT: BYTE-IDENTICAL to the committed file (all lines)")
    elif cur.split(b"\n", 2)[2:] == want.split(b"\n", 2)[2:]:
        print("[build_final] RESULT: lines 3.. byte-identical; lines 1-2 differ:")
        for i, (x, y) in enumerate(zip(cur.split(b"\n")[:2], want.split(b"\n")[:2]), 1):
            if x != y:
                print(f"        line {i} rebuilt  : {x.decode()[:200]}\n        line {i} committed: {y.decode()[:200]}")
        rc = 1 if a.strict else 0
    else:
        print("[build_final] RESULT: MISMATCH from line 3 on")
        report_diff(cur, want, "k3_hh_3.py")
        if first_bad:
            print(f"        first step whose output differs from the recorded sha1: {first_bad['n']} ({first_bad['name']}, {first_bad['tool'] if 'tool' in first_bad else 'inline'})")
        rc = 1

    if not a.no_gate:
        r = subprocess.run([sys.executable, "-I", "-X", "utf8", "tools/lead_gate.py", "file", cur_path], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=600)
        last = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()][-1:] or ["(no output)"]
        verdict = "PASS" if r.returncode == 0 and "GATE PASS" in last[0] else "FAIL"
        print(f"[build_final] lead_gate file {cur_path}: {verdict}  ({last[0].strip()})")
        if verdict != "PASS":
            rc = 1

    if a.scored_copy:
        scored_copy(cur, a.scored_copy, a.note)
    return rc


if __name__ == "__main__":
    sys.exit(main())

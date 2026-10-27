"""gen_k3_v35.py: kernels/muse_decode_layer/k3_v35.py = k3_v34 + the explorer's x_4 levers + KW1.

Composition (each step's patches assert they hit exactly once in their scope; the steps touch different code paths):
  1. tools/su1_patches.py, then tools/su1b_patches.py (explorer, copied verbatim from claude/explorer-baba5f @ 46b2402):
     SU1 staged GEMV + K3 early trigger (probe bit 25) + K3 grid 4T, with the SU1_ENABLE master switch.
  2. x_4's configuration (explorer kernels/muse_decode_layer/x_4.py, R-explorer-1/2/3):
     * SU1 at T=1 only (unpooler/8 off);
     * K3 trigger -> PDL P1 unpooler at T=8 (SU1_K3_TRIG_AT {1: su1, 8: p1});
     * SU1_TRIG 2, SU1_SMEM unpooler 120 KB, SU1 o self-test T (1, 8);
     * DG1 o-projection at T=32.
     Unlike x_4, the trigger keeps su1b's SU1_ENABLE master switch, so the A/B arm "SU1_ENABLE False + DG1 o off + KW1 0" is
     exactly k3_v34.
  3. tools/kw1_patches.py (R3, adopted): K2-only hot weight 5 for K2A v3.7 vid 5 (T=128), hot role >= G/2.
  4. Hook (off by default): --with-fa206 applies tools/fa206_patches.apply last (PR206 core on the full verify layers at batch
     >= 16; the parent's other agent is making it submittable). Its patches anchor on k3_v34 text that steps 1-3 do not touch;
     the wrapper asserts every patch still hits once.
Checks: base sha1 pinned; step 1+2 reproduce x_4 except the documented master-switch line and the nonce; py_compile; ASCII;
line 1; nonce; no duplicate top-level names; judge-checker scan (tools/gen_k3_fa.py SCAN) identical to the base; < 1 MB.

    python -I tools/gen_k3_v35.py [--with-fa206] [--strip] [--check]   # --with-fa206 implies --strip (k3_v35_fa.py)
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import kw1_patches  # noqa: E402
import su1_patches  # noqa: E402
import su1b_patches  # noqa: E402

BASE = ROOT / "kernels/muse_decode_layer/k3_v34.py"
BASE_SHA1 = "1d5e2a5ddee39134e4cb9960bf08884d61671d45"
X4 = ROOT / ".claude/worktrees/explorer-baba5f/kernels/muse_decode_layer/x_4.py"   # reference (optional, checked when present)
WITH_FA = "--with-fa206" in sys.argv
STRIP = "--strip" in sys.argv or WITH_FA   # PR206 needs it to fit < 1 MB (parent: keep the K2A fallbacks, strip comments instead)
PT1 = "--pt1" in sys.argv   # tools/pt1_patches.py: threaded prefetch of the independent CUDA builds (prepare-time cut)
_STEM = "k3_v35" + ("_fa" if WITH_FA else "_s" if STRIP else "") + ("_pt1" if PT1 else "")
OUT = ROOT / f"kernels/muse_decode_layer/{_STEM}.py"
NONCE_OLD = "# nonce k3-v34-1 (k3_v19 + the v20 levers listed in tools/gen_k3_v20.py)"
NONCE = ("# nonce k3-v35-1 (k3_v34 + explorer x_4 levers: SU1 at T=1, K3 trigger -> P1 unpooler at T=8, DG1 o at T=32; "
         "+ KW1 K2 hot weight 5 at T=128" + ("; + PR206 full verify B>=16" if WITH_FA else "")
         + ("; + PT1 threaded build prefetch" if PT1 else "") + ")")

# step 2: x_4's configuration (old, new); each `old` must occur exactly once in the whole text
X4_CONFIG = [
    ("DG1 o at T=32",
     "DG1_ROUTE = {('cat', 32): True, ('qkv', 32): True}   #",
     "DG1_ROUTE = {('cat', 32): True, ('qkv', 32): True, ('o', 32): True}   #"),
    ("DG1 o geometry",
     "DG1_GEOMETRY = {('cat', 32): (32, 2, 1, 3), ('qkv', 32): (32, 2, 1, 4)}   #",
     "DG1_GEOMETRY = {('cat', 32): (32, 2, 1, 3), ('qkv', 32): (32, 2, 1, 4), ('o', 32): (32, 1, 1, 1)}   #"),
    ("SU1 routes: unpooler at T=1 only",
     'SU1_ROUTE = {("unpooler", 1): True, ("unpooler", 8): True, ("o", 1): False,',
     'SU1_ROUTE = {("unpooler", 1): True, ("unpooler", 8): False, ("o", 1): False,'),
    ("SU1 trigger at entry",
     "SU1_TRIG = 1         #", "SU1_TRIG = 2         #"),
    ("SU1 smem cap",
     "SU1_SMEM = {}        #", 'SU1_SMEM = {"unpooler": 120 * 1024}   #'),
    ("SU1 o self-test T",
     'SU1_SELF_TEST_T = {"unpooler": (1, 2, 5, 8), "o": (1, 2, 5, 8, 13, 16, 29, 32)}',
     'SU1_SELF_TEST_T = {"unpooler": (1, 2, 5, 8), "o": (1, 8)}'),
    ("K3 trigger buckets: SU1 at T=1, PDL P1 unpooler at T=8",
     'SU1_K3_TRIG_AT = {1: "su1", 8: "su1", 16: "p1m", 32: "p1m"}   #',
     'SU1_K3_TRIG_AT = {1: "su1", 8: "p1"}   #'),
]
# the one intended difference from x_4 (su1b @ 46b2402 added the master switch after x_4 was generated)
X4_MASTER_SWITCH = ("    if not (SU1_ENABLE and SU1_K3_TRIG):   # SU1_ENABLE is the master switch of every explorer lever (False == k3_v34)",
                    "    if not SU1_K3_TRIG:")
SCAN = (("importlib", r"importlib"), ("sys.modules", r"sys\.modules"), ("exec(", r"(?<![\w.])exec\("),
        ("builtin compile(", r"(?<![\w.])compile\("), ("__import__", r"__import__"), ("setattr(", r"(?<![\w])setattr\("),
        ("globals()[", r"globals\(\)\s*\["), ("open(", r"(?<![\w.])open\("), ("write_text", r"write_text"),
        ("mkdtemp", r"mkdtemp"), ("tempfile", r"tempfile"), ("sys.path", r"sys\.path"), ("os.environ", r"os\.environ"),
        ("environ", r"\benviron\b"), ("exec_module", r"exec_module"), ("spec_from_file_location", r"spec_from_file_location"),
        ("module_from_spec", r"module_from_spec"), ("__name__", r"__name__"), ("__dict__", r"__dict__"),
        (".write(", r"\.write\("), ("delattr(", r"delattr\("), ("import_module", r"import_module"))


def check(cond, msg):
    if not cond:
        raise SystemExit(f"[gen_k3_v35] FAIL: {msg}")


def scan(text):
    return {label: len(re.findall(pat, text)) for label, pat in SCAN}


def x4_levers(base):
    text = su1b_patches.apply(su1_patches.apply(base))
    for what, old, new in X4_CONFIG:
        check(text.count(old) == 1, f"x_4 config {what!r}: anchor found {text.count(old)} times")
        text = text.replace(old, new)
    return text


def generate():
    base = BASE.read_text(encoding="utf-8")
    check(hashlib.sha1(base.encode()).hexdigest() == BASE_SHA1, "k3_v34 changed (base sha1)")
    rep = []
    text = x4_levers(base)
    if X4.exists():   # steps 1 + 2 == x_4 up to the master-switch line and the nonce
        x4 = X4.read_text(encoding="utf-8")
        a = text.replace(X4_MASTER_SWITCH[0], X4_MASTER_SWITCH[1])
        la, lb = a.split("\n"), x4.split("\n")
        diff = [i for i in range(max(len(la), len(lb))) if (la[i] if i < len(la) else None) != (lb[i] if i < len(lb) else None)]
        check(len(la) == len(lb) and diff == [1], f"steps 1+2 differ from x_4 at lines {diff[:8]}")
        rep.append("steps 1+2 reproduce explorer x_4 line for line (except the nonce and su1b's SU1_ENABLE master switch)")
    else:
        rep.append("x_4 reference not present: reproduction check skipped")
    text = kw1_patches.apply(text)
    check(text.count("KW1_K2_W4 = 5   #") == 1, "KW1 default 5")
    if WITH_FA:
        import fa206_patches   # the hook: everything but its own nonce patch (this wrapper writes the nonce)
        text = fa206_patches.apply(text, [p for p in fa206_patches.PATCHES if p[0] != "nonce"])
        rep.append("PR206 hook applied (fa206_patches.apply minus its nonce patch; every patch hit once)")
    if PT1:   # before the strip: its anchors carry comments
        import pt1_patches
        text = pt1_patches.apply(text)
        rep.append(f"PT1 applied ({len(pt1_patches.PATCHES)} patches hit once)")
    check(text.count(NONCE_OLD) == 1, "base nonce line")
    text = text.replace(NONCE_OLD, NONCE)
    if STRIP:   # tools/strip_layer.py: comments + docstrings only, license blocks and lines 1-2 kept, kernel strings untouched
        import strip_layer
        before = len(text.encode())
        stripped = strip_layer.strip(text)
        ok, msg = strip_layer.equivalent(text, stripped)
        check(ok, f"strip: {msg}")
        lic0 = [l for l in text.splitlines() if strip_layer.LICENSE.search(l)]
        lic1 = [l for l in stripped.splitlines() if strip_layer.LICENSE.search(l)]
        check(lic0 == lic1, "strip: license lines changed")
        text = stripped
        rep.append(f"stripped {before} -> {len(text.encode())} B; {msg}; {len(lic1)} license lines kept")
    return text, base, rep


def verify(text, base):
    rep = []
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(tmp), doraise=True)
    finally:
        tmp.unlink()
    check(text.isascii(), "non-ASCII character")
    check(text.split("\n", 1)[0] == "# okc: problem=muse_decode_layer", "line 1")
    check(text.count("k3-v35-1") == 1 and "k3-v34-1" not in text, "nonce")
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None)
             for n in ast.parse(text).body]
    dup = sorted({n for n in names if n and names.count(n) > 1})
    check(not dup, f"duplicate top-level names {dup}")
    s0, s1 = scan(base), scan(text)
    if STRIP:   # removing comments may only remove scan hits, never add one
        check(all(s1[k] <= s0[k] for k in s0), f"judge-checker scan grew: { {k: (s0[k], s1[k]) for k in s0 if s1[k] > s0[k]} }")
    else:
        check(s0 == s1, f"judge-checker scan differs: { {k: (s0[k], s1[k]) for k in s0 if s0[k] != s1[k]} }")
    check(len(text.encode()) < 1_000_000, "file >= 1 MB")
    rep.append(f"compiles, ASCII, line 1, nonce, no duplicate names, checker scan == base, {len(text.encode())} B")
    return rep


def main():
    text, base, rep = generate()
    rep += verify(text, base)
    for r in rep:
        print(f"[gen_k3_v35] OK  {r}")
    if "--check" in sys.argv:
        check(OUT.exists() and OUT.read_text(encoding="utf-8") == text, f"{OUT.name} on disk differs (regenerate)")
        print("[gen_k3_v35] --check: up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"[gen_k3_v35] wrote {OUT.relative_to(ROOT)} (sha1 {hashlib.sha1(text.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

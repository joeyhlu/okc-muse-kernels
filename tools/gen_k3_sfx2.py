"""gen_k3_sfx2.py: SFX2 patch (tools/sfx2_patches.py: VA + K2A empty-arrive data dependencies; SFX2 = 1 in the file) on a
layer (default base k3_gc_19 = k3_gc_18 + SFX) -> kernels/lab/<out>.py (default k3_sfx_2). Refuses to overwrite (--dry: checks).

  python -I tools/gen_k3_sfx2.py [--base=k3_gc_19] [--out=k3_sfx_2] [--dry]
"""
import hashlib
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools")]
import sfx2_patches as X  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_19')}.py"
OUTN = ARGS.get("out", "k3_sfx_2")
OUT = ROOT / f"kernels/lab/{OUTN}.py"
NL = chr(10)


def check(c, msg):
    if not c:
        raise SystemExit(f"[gen_k3_sfx2] FAIL {msg}")


def main():
    check("--dry" in sys.argv or not OUT.exists(), f"{OUT.name} exists")
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    check(bn.startswith("# nonce "), "line 2 is not a nonce line")
    t = X.patch(b)
    nonce = f"# nonce {OUTN.replace('_', '-')} ({BASE.stem} + SFX2 VA / K2A empty-arrive data dependencies; tools/gen_k3_sfx2.py)"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    check(max(len(x) for x in t.split(NL)) <= max(len(x) for x in b.split(NL)), "line longer than the base's longest")
    check(len(re.findall(r"importlib|sys\.modules", t)) == len(re.findall(r"importlib|sys\.modules", b)), "checker")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[gen_k3_sfx2] OK; {len(t.encode())} B (base {len(b.encode())} B)")
    if "--dry" in sys.argv:
        return 0
    OUT.write_text(t, encoding="utf-8", newline=NL)
    print(f"[gen_k3_sfx2] wrote {OUT.relative_to(ROOT)} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

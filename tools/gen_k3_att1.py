"""gen_k3_att1.py: kernels/lab/<out>.py = a layer file (default k3_gc_7) + ATT1 (tools/att1_patches.py: B = 1 full-attention
handoff knobs on the FA4 chain), optionally + B1A (tools/b1a_patches.py, routed off by default, with the lab B1A_DBG knob).

Same contract as gen_k3_b1a.py: the base must be a gen_k3_m32.finish() fixed point, every anchor is asserted unique, the text
goes through gen_k3_m32.finish (strip + AST checks, ASCII, nonce, judge-checker scan <= base, < 1 MB) and the output is a
finish() fixed point. Refuses to overwrite.

    python -I tools/gen_k3_att1.py --out=k3_att1_lab [--base=k3_gc_7] [--b1a] [--sets=sets.json] [--check]
    python -I tools/gen_k3_att1.py --defaults-only --in=<file> --out=<name> --sets=sets.json    # box: knob defaults only
sets.json: {"NAME": "python literal", ...} replaces the module line NAME = ... (one each); the nonce lists them.
"""
import hashlib
import json
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
import att1_patches as AP  # noqa: E402

OUT_NAME = ARGS.get("out", "k3_att1_lab")
OUT = ROOT / f"kernels/lab/{OUT_NAME}.py"
SETS = json.loads(pathlib.Path(ARGS["sets"]).read_text()) if "sets" in ARGS else {}
TAG = OUT_NAME.replace("_", "-")


def nonce(base_stem, b1a):
    return (f"# nonce {TAG} ({base_stem} + ATT1: B = 1 full attention handoff knobs (prologue trigger-first, per-shape FA4 "
            f"prewait / trigger, combine warps){' + B1A (routed off, lab B1A_DBG)' if b1a else ''}"
            + (f"; defaults {sorted(SETS.items())}" if SETS else "") + ")")


def defaults_only():
    src = ROOT / ARGS["in"]
    t = src.read_text(encoding="utf-8")
    old = t.split(NL)[1]
    assert old.startswith("# nonce ")
    new = old[:-1] + f"; defaults {sorted(SETS.items())})" if old.endswith(")") else old + f" defaults {sorted(SETS.items())}"
    new = new.replace(old.split()[2], TAG, 1)
    t = AP.set_defaults(t.replace(old, new, 1), SETS)
    assert not OUT.exists(), f"{OUT} exists"
    OUT.write_text(t, encoding="utf-8", newline=NL)
    print(f"[gen_k3_att1] defaults-only: wrote {OUT.relative_to(ROOT)} ({len(t.encode())} B, sha1 "
          f"{hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


def main():
    if "--defaults-only" in FLAGS:
        return defaults_only()
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        import gen_k3_m32 as GM  # noqa: E402
        stub = False
    except ModuleNotFoundError as ex:
        if ex.name != "sbb1_patches":
            raise
        sys.modules["sbb1_patches"] = types.ModuleType("sbb1_patches")
        import gen_k3_m32 as GM  # noqa: E402
        stub = True
    sys.argv = _argv
    base_name = ARGS.get("base", "k3_gc_7")
    base = ROOT / f"kernels/muse_decode_layer/{base_name}.py"
    check = "--check" in FLAGS
    b1a = "--b1a" in FLAGS
    GM.check(base.exists(), f"{base.name} not found")
    if not check:
        GM.check(not OUT.exists(), f"{OUT.relative_to(ROOT)} exists; pick another --out")
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    chk_base = b
    if not stub:
        try:
            chk_base = GM.G37.generate()[1]
        except (SystemExit, Exception) as ex:   # noqa: BLE001
            print(f"[gen_k3_att1] note: k3_v37 chain not regenerable ({ex!r:.100}); checker baseline = {base.name}")
    fixed = GM.finish(b, bn, bn.split()[2], chk_base, []) == b   # k3_gc_7 came from m32e_compose (own checks, not finish)
    rep = [f"{base.name} {'is' if fixed else 'is NOT'} a finish() fixed point ({len(b.encode())} B); the output goes through finish()"]
    nn = nonce(base.stem, b1a)
    t = AP.apply(b.replace(bn, nn))
    rep.append("att1_patches.apply: knobs, prologue TF, per-shape PW / TRIG, combine warps")
    if b1a:
        import b1a_patches as BP  # noqa: E402
        saved = BP.B1A_BLOCK
        BP.B1A_BLOCK = AP.b1a_block_edits(saved)
        try:
            t = BP.apply(t, ())
        finally:
            BP.B1A_BLOCK = saved
        rep.append("b1a_patches.apply (route off by default, B1A_DBG)")
    if SETS:
        t = AP.set_defaults(t, SETS)
        rep.append(f"defaults {SETS}")
    s = GM.finish(t, nn, TAG, chk_base, rep)
    GM.check(GM.finish(s, nn, TAG, chk_base, []) == s, "output is not a finish() fixed point")
    for r in rep:
        print(f"[gen_k3_att1] OK  {r}")
    sha = hashlib.sha1(s.encode()).hexdigest()[:12]
    if check:
        GM.check(OUT.exists() and OUT.read_text(encoding="utf-8") == s, f"{OUT.name} on disk differs")
        print(f"[gen_k3_att1] --check: {OUT.relative_to(ROOT)} up to date ({len(s.encode())} B, sha1 {sha})")
        return 0
    OUT.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_att1] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, sha1 {sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""tfpw_patches.py: TFPW-S (notes/study/fresh_eyes_nonmoe.md s3) text patch on a layer file (k3_sfx_2 or any later gc_* base):
prologue trigger-at-entry (TF) + FA4 first-tile K/V pre-wait (ATT1_PW) extended from (full, 1) to the sliding verify loads
v16s / v32s = (sliding, B 16 / 32, L 8) = shapes 8 / 10.

  ATT1_TF      new {(kind, B[, L]): True} table; replaces the hard-coded `batch == 1 and not window` in k1fa4_run
               (ship form = {('full', 1): True}, so the base behaviour is one table entry).
  ATT1_PW      lookup (kind, B, L) first, then (kind, B) (the existing (full, 1) key keeps working).
  K1FA4_SPLITS lookup (kind, B, L) first, then (kind, B), then the heuristic (split re-tune with PW on).

Safety (read, R-tfpw-0, notes/lab.md): the PV8 producer (the class at sliding L 8) takes griddepcontrol.wait before the first
K / V block iff `n_block_max * tile_n > seqlen_k - seqlen_q` (blocks are loaded newest first, so a first block without new keys
means no block has them), Q is loaded after the wait, empty splits wait before Q, and the consumer waits on Q before any
o_part / lse write. The judge's pages are a permutation (reference.py), so old-key rows are never rewritten by the prologue.

  python -I tools/tfpw_patches.py [--base=kernels/lab/k3_sfx_2.py] [--out=k3_tfpw_1] [--ns16=0] [--ns32=0] [--dry]
  (--ns16 / --ns32 > 0: pin the FA4 split count at v16s / v32s; 0 = the heuristic, NS 8 / 4)
"""
import hashlib
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
NL = chr(10)
KEYS = "('sliding', 16, 8): True, ('sliding', 32, 8): True"


def _sub(s, old, new, n=1):
    if s.count(old) != n:
        raise SystemExit(f"[tfpw_patches] anchor count {s.count(old)} != {n}: {old[:70]!r}")
    return s.replace(old, new)


def patch(t, ns16=0, ns32=0):
    t = _sub(t, "ATT1_PRO_TF = True\nATT1_PW = {('full', 1): True}\n",
             "ATT1_PRO_TF = True\nATT1_TF = {('full', 1): True, " + KEYS + "}   # TFPW-S: prologue trigger-at-entry per (kind, B[, L])\n"
             "ATT1_PW = {('full', 1): True, " + KEYS + "}\n")
    sp = re.search(r"^K1FA4_SPLITS = \{(.*)\}\n", t, re.M)
    if not sp:
        raise SystemExit("[tfpw_patches] K1FA4_SPLITS anchor")
    extra = "".join(f", ('sliding', {b}, 8): {n}" for b, n in ((16, ns16), (32, ns32)) if n)
    if extra:
        t = t.replace(sp.group(0), "K1FA4_SPLITS = {" + sp.group(1) + extra + "}\n", 1)
    t = _sub(t, '    k1fa4_K["pw"] = bool(ATT1_PW.get((kind, B), False))\n',
             '    k1fa4_K["pw"] = bool(ATT1_PW.get((kind, B, L), ATT1_PW.get((kind, B), False)))\n')
    t = _sub(t, "        TF=bool(pdl and ATT1_PRO_TF and batch == 1 and not window), **pdl_kw)\n",
             "        TF=bool(pdl and ATT1_PRO_TF and ATT1_TF.get((kind, batch, length), ATT1_TF.get((kind, batch), False))),\n"
             "        **pdl_kw)\n")
    t = _sub(t, "    window = cfg[\"window\"]\n    tokens = batch * length\n    qkv = data[\"qkv\"].view(tokens, -1)\n",
             "    window = cfg[\"window\"]\n    kind = \"sliding\" if window else \"full\"\n    tokens = batch * length\n"
             "    qkv = data[\"qkv\"].view(tokens, -1)\n")
    t = _sub(t, '    ns = splits or K1FA4_SPLITS.get(("sliding" if window else "full", batch)) or k1fa4_num_splits_for(batch, hkv, keys)\n',
             "    ns = (splits or K1FA4_SPLITS.get((kind, batch, length)) or K1FA4_SPLITS.get((kind, batch))\n"
             "          or k1fa4_num_splits_for(batch, hkv, keys))\n")
    return t


def main():
    a = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
    base = ROOT / a.get("base", "kernels/lab/k3_sfx_2.py")
    outn = a.get("out", "k3_tfpw_1")
    out = ROOT / f"kernels/lab/{outn}.py"
    dry = "--dry" in sys.argv
    if not dry and out.exists():
        raise SystemExit(f"[tfpw_patches] {out.name} exists")
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    if not bn.startswith("# nonce "):
        raise SystemExit("[tfpw_patches] line 2 is not a nonce line")
    t = patch(b, int(a.get("ns16", 0)), int(a.get("ns32", 0)))
    t = t.replace(bn, f"# nonce {outn.replace('_', '-')} ({base.stem} + TFPW-S sliding verify TF / PW; tools/tfpw_patches.py)", 1)
    assert t.isascii() and len(t.encode()) < 1_000_000, len(t.encode())
    assert max(len(x) for x in t.split(NL)) <= max(len(x) for x in b.split(NL))
    assert len(re.findall(r"importlib|sys\.modules", t)) == len(re.findall(r"importlib|sys\.modules", b))
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[tfpw_patches] OK {len(t.encode())} B (base {len(b.encode())} B)")
    if not dry:
        out.write_text(t, encoding="utf-8", newline=NL)
        print(f"[tfpw_patches] wrote {out.relative_to(ROOT)} sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()

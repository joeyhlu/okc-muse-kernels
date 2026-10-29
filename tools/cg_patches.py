"""cg_patches.py (R-cg-1, notes/study/serial_chains.md candidates 2 + 1): k3_b21_1 -> kernels/lab/k3_cg_1.py.

  python tools/cg_patches.py [--base=kernels/lab/k3_b21_1.py] [--out=kernels/lab/k3_cg_1.py] [--trig=1] [--ph=1]

Changes (Triton FA4 / VFA helper kernels only; ship kernels kept verbatim, new copies selected by module flags so rev_ab3
can A/B them inside one file):
  1. _k1fa4_combine_gate_t = _k1fa4_combine_gate (full B=1 path: d1f / v1f) + gdc_launch_dependents right AFTER its own
     gdc_wait (not at entry: an entry trigger would put o's weight prefetch on FA4's KV stream) + the QKV gate `g` load moved
     to the top (post-wait). _k1fa4_combine_gate_hb untouched (its pre-wait gate read is safe only while FA4 triggers at 'epi').
     Flag CG_COMB_TRIG.
  2. _k1fa4_prologue_ph = _k1fa4_prologue with positions, the cos/sin gather and cache_loc loaded BEFORE gdc_wait (graph
     inputs / the static rope table, never written in-layer); the qkv rows (predecessor output) stay below the wait. Same
     arithmetic -> bitwise. Used by k1fa4_run and _vfa_run. Flag CG_PRO_PH.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)


def _rep(src, old, new, what, n=1):
    c = src.count(old)
    if c != n:
        raise SystemExit(f"[cg_patches] {what}: expected {n} match(es), got {c}")
    return src.replace(old, new)


def _fn_src(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = src.split("\n")
            return "\n".join(lines[node.decorator_list[0].lineno - 1:node.end_lineno])
    raise SystemExit(f"[cg_patches] {name} not found")


PRO_PDL = """    if USE_PDL:
        if TF:
            tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
        if not TF:
            tl.extra.cuda.gdc_launch_dependents()
"""
PRO_SRC_LINE = "    src = token * stride_qkv + head * D\n"
PRO_ROPE_LOADS = """        pos = tl.load(POS + token, mask=row_mask, other=0).to(tl.int64)
        freq = cols // 2
        cs_mask = row_mask[:, None] & (cols < D)[None, :]
        cos = tl.load(COS_SIN + pos[:, None] * D + freq[None, :], mask=cs_mask, other=1.0)
        sin = tl.load(COS_SIN + pos[:, None] * D + D // 2 + freq[None, :], mask=cs_mask, other=0.0)
"""
PRO_LOC = "    loc = tl.load(CACHE_LOC + token, mask=row_mask, other=0).to(tl.int64)\n"


def prologue_ph(ship):
    s = _rep(ship, "def _k1fa4_prologue(", "def _k1fa4_prologue_ph(", "pro name")
    s = _rep(s, PRO_PDL, "", "pro pdl block")
    s = _rep(s, PRO_ROPE_LOADS, "", "pro rope loads")
    s = _rep(s, PRO_LOC, "", "pro loc")
    pre = ("    # R-cg-1: graph inputs + the static rope table (never written in-layer) are read BEFORE the wait; qkv after it\n"
           "    if HAS_ROPE:\n" + PRO_ROPE_LOADS + PRO_LOC + PRO_PDL)
    return _rep(s, PRO_SRC_LINE, PRO_SRC_LINE + pre, "pro src line")


COMB_G = "    g = tl.load(QKV + t * stride_qkv + gate_off + h * D + d).to(tl.float32)\n"


def combine_t(ship):
    s = _rep(ship, "def _k1fa4_combine_gate(", "def _k1fa4_combine_gate_t(", "comb name")
    s = _rep(s, "        tl.extra.cuda.gdc_wait()\n",
             "        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()   # R-cg-1: o's GD pre-wait window\n",
             "comb trigger")
    s = _rep(s, COMB_G, "", "comb g")
    s = _rep(s, "    d = tl.arange(0, D)\n", "", "comb d")
    return _rep(s, "    smask = s < NS\n", "    smask = s < NS\n    d = tl.arange(0, D)\n" + COMB_G, "comb g top")


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_b21_1.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_cg_1.py")
    src = base.read_text(encoding="utf-8")
    trig, ph = ARGS.get("trig", "1") == "1", ARGS.get("ph", "1") == "1"
    src = _rep(src, "ATT1_COMB_NW = 0\n",
               f"ATT1_COMB_NW = 0\nCG_COMB_TRIG = {trig}   # R-cg-1 (1): full-B1 combine triggers o after its wait, gate load first\n"
               f"CG_PRO_PH = {ph}   # R-cg-1 (2): FA4/VFA prologue reads positions / cos_sin / cache_loc before gdc_wait\n",
               "flags")
    pro, comb = _fn_src(src, "_k1fa4_prologue"), _fn_src(src, "_k1fa4_combine_gate")
    src = _rep(src, comb + "\n", comb + "\n\n\n" + prologue_ph(pro) + "\n\n\n" + combine_t(comb) + "\n", "insert copies")
    src = _rep(src, "    _k1fa4_prologue[(triton.cdiv(n_rows, block_m),)](",
               "    (_k1fa4_prologue_ph if CG_PRO_PH else _k1fa4_prologue)[(triton.cdiv(n_rows, block_m),)](", "pro calls", 2)
    src = _rep(src, "        _k1fa4_combine_gate[(rows,)](",
               "        (_k1fa4_combine_gate_t if CG_COMB_TRIG else _k1fa4_combine_gate)[(rows,)](", "comb call")
    ast.parse(src)
    out.write_text(src, encoding="utf-8", newline="\n")
    n = out.stat().st_size
    print(f"[cg_patches] {base.name} -> {out.relative_to(ROOT)} {n} bytes (trig={trig} ph={ph})")
    if n >= 1_000_000:
        raise SystemExit("[cg_patches] file >= 1 MB")


if __name__ == "__main__":
    main()

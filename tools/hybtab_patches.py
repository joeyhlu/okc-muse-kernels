"""hybtab_patches.py (R-hybtab-*): layer-fitted WGS cost tables for HYB (the public alygg77 WGS MoE engine, A32; T 49..320).

  python -I tools/hybtab_patches.py [--base=kernels/lab/k3_hh_3k.py] [--out=kernels/lab/k3_hybtab_1.py]            (lab, A/B)
  python -I tools/hybtab_patches.py --mode=ship --tab=a1,..,a8,b1,..,b8 [--base=..] [--out=kernels/lab/k3_hybtab_s.py]  (ship)
  python -I tools/lead_gate.py file <out>

HYB's plan is static stream-K: every CTA gets a contiguous range of phase-A / phase-B stages with equal summed cost, where a
stage's cost is WGS_CA / WGS_CB[n-tiles - 1] (OKC_WGSA / OKC_WGSB, 8 values each, 1..255). alygg77 fit them on per-CTA phase
times at board bs 96..256 (t_fit2.py) and tuned them on the board (combo_m798); they were never fit on layer routings.

--mode=lab (default): adds a SECOND HYB module "l" (tools/hybx_common.py helper, own scratch, own PT1 build job "hybl") built
from _HYB_MOE_CUDA with three text edits applied at import (_hybl_src): OKC_DEBUG 1 (alygg77's own per-CTA %globaltimer marks
DBG(0..7), written only when p.dbg = scratch[3] is passed), a `wtab[2]` field at the end of Smem, and wgs_cost reading the
packed tables from S.wtab, which thread 0 loads from p.wA (w[0]: a placeholder slot the WGS build never reads) before route().
Globals for A/B arms: HYB_LAB (route T 49..320 to "l"), HYB_TAB (16 ints, None = the shipped tables), HYB_MARKS (eager
diagnostics: pass the dbg buffer, keep the ids). Ship tables / ship module untouched; HYB_LAB False = k3_hh_3k behaviour.
--mode=ship: only the two #define lines OKC_WGSA / OKC_WGSB change (same SASS up to immediates).
--check: verify --out == apply(--base) byte for byte.
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hybx_common  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[hybtab_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


LAB = r'''HYB_LAB = False
HYB_TAB = None
HYB_MARKS = False
HYBL_TESTS = ((64, 62, "skewed"), (128, 64, "skewed"), (200, 66, "skewed"), (256, 67, "normal"), (320, 69, "normal"))
def _hybl_src(s):
    for a, b in (("#define OKC_DEBUG 0\n", "#define OKC_DEBUG 1\n"),
                 ("uint32_t wtotB;\n", "uint32_t wtotB;\nunsigned long long wtab[2];\n"),
                 ("return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);\n",
                  "return (int)((reinterpret_cast<const Smem*>(smem_raw)->wtab[pb] >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);\n"),
                 ("S.xbuf_ptr = xbuf;\n", "S.xbuf_ptr = xbuf;\n{ const unsigned long long* tb = reinterpret_cast<const unsigned long long*>(p.wA);"
                  " S.wtab[0] = tb[0] ? tb[0] : WGS_PA; S.wtab[1] = tb[1] ? tb[1] : WGS_PB; }\n")):
        if s.count(a) != 1:
            return None
        s = s.replace(a, b)
    return s
def _hybl_deftab():
    out = []
    for k in ("A", "B"):
        i = _HYB_MOE_CUDA.index(f"#define OKC_WGS{k} ") + len(f"#define OKC_WGS{k} ")
        out += [int(v) for v in _HYB_MOE_CUDA[i:_HYB_MOE_CUDA.index("\n", i)].split(",")]
    return tuple(out)
_HYBX["l"] = {"src": _hybl_src(_HYB_MOE_CUDA), "mod": None, "ok": False, "pdl": -1, "tabs": {}}
def _hybl_tab(tab, dev):
    X = _HYBX["l"]
    key = tuple(int(v) for v in (tab if tab is not None else _hybl_deftab()))
    t = X["tabs"].get(key)
    if t is None:
        if len(key) != 16 or not all(1 <= v <= 255 for v in key):
            raise ValueError(f"HYB_TAB needs 16 values in 1..255, got {key}")
        pk = [sum(key[8 * h + i] << (8 * i) for i in range(8)) for h in range(2)]
        t = torch.tensor([v - (1 << 64) if v >= 1 << 63 else v for v in pk], dtype=torch.int64, device=dev)
        X["tabs"][key] = t
    return t
def _hybl_call(x, ids, wts):
    X = _HYBX["l"]
    w = list(X["w"])
    w[0] = _hybl_tab(HYB_TAB, x.device)
    dbg = None
    if HYB_MARKS:
        if "dbg" not in X:
            X["dbg"] = torch.zeros(_HYB["nsm"] * 8, dtype=torch.int64, device=x.device)
        dbg = X["dbg"]
        dbg.zero_()
        X["last"] = ids.clone()
    return _hybx_call("l", x, ids, wts, w=w, dbg=dbg)
def _hybl_prepare(st, w, dev):
    if _HYBX["l"]["src"] is None:
        print("[hybl] OFF: a source anchor of the lab transform is missing", flush=True)
        return
    _hybx_prepare("l", w, dev, HYBL_TESTS, wmod=lambda wl: wl.__setitem__(0, _hybl_tab(None, dev)))
'''

A_DISPATCH = "    if _hyb_on(x.shape[0]):\n"
N_DISPATCH = ("    if HYB_LAB and _hyb_on(x.shape[0]) and _HYBX[\"l\"][\"ok\"]:\n"
              "        return _hybl_call(x, local_ids.to(torch.int32).contiguous(), topk_weights.to(torch.float32).contiguous())\n"
              + A_DISPATCH)
A_PT1 = '        jobs.append(("hyb", _hyb_build))\n'
N_PT1 = A_PT1 + '        if _HYBX["l"]["src"] is not None:\n            jobs.append(("hybl", lambda: _hybx_build("l")))\n'
A_PREP = "    _hyb_prepare(st, w, dev)\n"
N_PREP = A_PREP + "    _hybl_prepare(st, w, dev)\n"
A_TA = "#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207\n"
A_TB = "#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174\n"


def apply(s, mode, tab):
    lines = s.split(NL)
    if mode == "ship":
        lines[1] = (f"# nonce k3-hybtab-s (k3_hh_3k + R-hybtab: layer-fitted HYB WGS cost tables "
                    f"A {','.join(map(str, tab[:8]))} B {','.join(map(str, tab[8:]))}; tools/hybtab_patches.py)")
        s = NL.join(lines)
        s = sub(s, A_TA, f"#define OKC_WGSA {', '.join(map(str, tab[:8]))}\n", "OKC_WGSA")
        s = sub(s, A_TB, f"#define OKC_WGSB {', '.join(map(str, tab[8:]))}\n", "OKC_WGSB")
        return s
    lines[1] = ("# nonce k3-hybtab-1 (k3_hh_3k + R-hybtab lab: second HYB module with runtime WGS cost tables + DBG marks, "
                "HYB_LAB / HYB_TAB / HYB_MARKS arms; tools/hybtab_patches.py)")
    s = NL.join(lines)
    s = hybx_common.ensure(s, sub)
    s = sub(s, hybx_common.ANCHOR, LAB + hybx_common.ANCHOR, "lab block")
    s = sub(s, A_DISPATCH, N_DISPATCH, "dispatch")
    s = sub(s, A_PT1, N_PT1, "pt1 job")
    s = sub(s, A_PREP, N_PREP, "prepare hook")
    return s


def main():
    mode = ARGS.get("mode", "lab")
    check(mode in ("lab", "ship"), f"--mode={mode}")
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3k.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hybtab_1.py" if mode == "lab" else "kernels/lab/k3_hybtab_s.py")
    tab = []
    if mode == "ship":
        tab = [int(v) for v in ARGS.get("tab", "").split(",") if v.strip()]
        check(len(tab) == 16 and all(1 <= v <= 255 for v in tab), f"--tab needs 16 values in 1..255, got {tab}")
    s = apply(base.read_bytes().decode("ascii"), mode, tab)
    ast.parse(s)
    check(s.isascii(), "non-ASCII output")
    check(len(s.encode()) < LIMIT, f"size {len(s.encode()):,} >= {LIMIT:,}")
    if "check" in FLAGS:
        check(out.read_bytes().decode("ascii") == s, f"{out.name} != apply({base.name})")
        print(f"[hybtab_patches] --check PASS: {out.name} == apply({base.name}) ({len(s):,} B)")
        return
    out.write_bytes(s.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"[hybtab_patches] {mode}: {base.name} -> {out.name} ({len(s):,} B, +{len(s) - base.stat().st_size:,} B)")


if __name__ == "__main__":
    main()

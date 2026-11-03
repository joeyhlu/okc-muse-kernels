"""RPWT lab: per-T margin for the wide RPW tier (q_len 1 full layers d1f / d16f / d32f). R-rp-4 near-boundary router-score noise there is
0.0021 / 0.0066 / 0.0045 and R-rp-9 / R-rp-11 consumed <= 0.0004 / 0.0024 of the margin on the decisions taken, so 0.020 is far wider than
needed. RPW_MARGIN_T (default {} = RPW_MARGIN 0.020 everywhere, byte-identical behaviour); candidate {1: 0.008, 16: 0.012, 32: 0.010}.
Python only (the margin reaches the kernels through deps / the rp_final argument). Base kernels/lab/k3_rpf2_ab.py (fin2) ->
kernels/lab/k3_rpwt_ab.py; --base= / --out= for other bases with the same anchors."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_rpf2_ab.py"
OUT = ROOT / "kernels/lab/k3_rpwt_ab.py"


def main():
    global BASE, OUT
    for a in sys.argv[1:]:
        if a.startswith("--base="):
            BASE = ROOT / a.split("=", 1)[1]
        if a.startswith("--out="):
            OUT = ROOT / a.split("=", 1)[1]
    s = BASE.read_text(encoding="utf-8").replace("\r\n", "\n")
    a = "(RPW_MARGIN if rpw else RP_MARGIN_T.get(tokens,RP_MARGIN))"
    assert s.count(a) == 2, s.count(a)
    s = s.replace(a, "(RPW_MARGIN_T.get(tokens,RPW_MARGIN) if rpw else RP_MARGIN_T.get(tokens,RP_MARGIN))")
    b = "RPW_MARGIN=0.020\n"
    assert s.count(b) == 1
    s = s.replace(b, b + "RPW_MARGIN_T={}  # RPWT: per-T wide-tier margin (tools/rpwt_patches.py)\n")
    compile(s, str(OUT), "exec")
    OUT.write_bytes(s.encode("ascii"))
    print("RPWT lab", OUT, len(s.encode("ascii")), "B")


if __name__ == "__main__":
    main()

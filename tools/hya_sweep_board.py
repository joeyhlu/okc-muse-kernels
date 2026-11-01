"""hya_sweep_board.py (attention lane, team joeyhlu, 2026-11-01): board sweep of HYA's launch plan on the shapes t0..t3 did not touch.

hya_t0 (vjeux f35a41ee + HYA M1 + flat full split plan t0, scored 2.6055) keeps vjeux's `_plan()` defaults for the six sliding
shapes and the two B = 1 full shapes. Each file below changes only the `_TUNED` table (a plan override per shape), so one board
check tests one candidate per shape; 12 shapes are independent measurements, the four large full shapes stay at t0 and serve as
replicate controls. Candidates are listed per shape; arm k takes the k-th entry.

    python -I tools/hya_sweep_board.py    # writes kernels/muse_attention_decode/hya_s1.py .. hya_s5.py and tools/hya_sweep_wave.txt
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels" / "muse_attention_decode" / "hya_t0.py"

OLD = """_TUNED = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
}
"""

# (B, L, sliding, pairs) -> candidates (index 0 = arm s1). nc is always explicit (an n_splits override alone would set nc = min(n, 4)).
# coop only where n_splits * B * pairs <= 2 * SMs (264): the host rule for a resident grid.
CANDS = {
    (1, 1, True): [dict(n_splits=16, nc=1), dict(n_splits=24, nc=1), dict(n_splits=32, nc=1, coop=1),
                   dict(n_splits=16, nc=1, coop=1), dict(n_splits=32, nc=2)],
    (16, 1, True): [dict(n_splits=8, nc=1), dict(n_splits=12, nc=1), dict(n_splits=24, nc=1),
                    dict(n_splits=32, nc=1), dict(n_splits=16, nc=1, coop=1)],
    (32, 1, True): [dict(n_splits=4, nc=1), dict(n_splits=6, nc=1), dict(n_splits=12, nc=1),
                    dict(n_splits=16, nc=1), dict(n_splits=8, nc=1, coop=1)],
    (1, 8, True): [dict(n_splits=16, nc=1, coop=1), dict(n_splits=24, nc=1, coop=1), dict(n_splits=33, nc=1, coop=0),
                   dict(n_splits=17, nc=1, coop=1), dict(n_splits=33, nc=2, coop=1)],
    (16, 8, True): [dict(n_splits=4, nc=1, coop=1), dict(n_splits=6, nc=1, coop=1), dict(n_splits=12, nc=1, coop=1),
                    dict(n_splits=16, nc=1, coop=1), dict(n_splits=8, nc=1, coop=0)],
    (32, 8, True): [dict(n_splits=2, nc=1, coop=1), dict(n_splits=3, nc=1, coop=1), dict(n_splits=6, nc=1, coop=1),
                    dict(n_splits=8, nc=1, coop=1), dict(n_splits=4, nc=1, coop=0)],
    (1, 1, False): [dict(n_splits=128, nc=8, coop=1), dict(n_splits=192, nc=8, coop=1), dict(n_splits=264, nc=8, coop=1),
                    dict(n_splits=252, nc=4, coop=1), dict(n_splits=252, nc=8, coop=0)],
    (1, 8, False): [dict(n_splits=128, nc=8, coop=1), dict(n_splits=192, nc=8, coop=1), dict(n_splits=264, nc=8, coop=1),
                    dict(n_splits=252, nc=4, coop=1), dict(n_splits=252, nc=8, coop=0)],
}


def fmt(d):
    return "dict(" + ", ".join(f"{k}={v}" for k, v in d.items()) + ")"


def main():
    src = SRC.read_text(encoding="ascii")
    assert src.count(OLD) == 1, "hya_t0 _TUNED block changed"
    lines = src.split("\n")
    assert lines[1].startswith("# nonce hya-tune-t0")
    wave = []
    for k in range(5):
        entries = "".join(f"    ({b}, {l}, {s}, 1): {fmt(c[k])},\n" for (b, l, s), c in CANDS.items())
        text = src.replace(OLD, "_TUNED = {\n" + entries + "}\n")
        ls = text.split("\n")
        ls[1] = (f"# nonce hya-sweep-s{k + 1} (attention lane Nov 1: hya_t0 + sliding / B=1 full plan overrides, "
                 "tools/hya_sweep_board.py)")
        out = ROOT / "kernels" / "muse_attention_decode" / f"hya_s{k + 1}.py"
        out.write_text("\n".join(ls), encoding="ascii", newline="\n")
        wave.append(f"muse_attention_decode kernels/muse_attention_decode/hya_s{k + 1}.py")
        print(out.relative_to(ROOT), len(out.read_bytes()), "B")
    (ROOT / "tools" / "hya_sweep_wave.txt").write_text("\n".join(wave) + "\n", newline="\n")


if __name__ == "__main__":
    main()

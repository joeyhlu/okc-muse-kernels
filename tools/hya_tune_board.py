"""hya_tune_board.py (R-hya-t, team joeyhlu): attention-board copies of vjeux's muse_attention_decode.py (github.com/vjeux/okc-2026
f35a41ee, Apache-2.0; vendored copy tools/vendor/vjeux_okc2026_f35a41ee/) that differ only in the flat full-layer split plan
(_TUNED entries for B 16 / 32, q_len 1 / 8), plus our no-trap modification M1 (same as tools/hya_patches.py).

    python -I tools/hya_tune_board.py      # writes kernels/muse_attention_decode/hya_t0.py .. hya_t3.py and tools/hya_tune_wave.txt
t0 = control (vjeux's plan: flat 264 CTAs, nc 1, coop at q_len 8); t1 = 528 CTAs (2 waves); t2 = 792 (3 waves);
t3 = q_len 8 last-CTA merges (coop 0) and q_len 1 two workspace copies (nc 2), 264 CTAs.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import hya_patches as H  # noqa: E402

OLD = """_TUNED = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
}
"""
FULL = [(16, 1), (32, 1), (16, 8), (32, 8)]
ARMS = {
    "t0": {},
    "t1": {k: "dict(n_splits=528, nc=1)" for k in FULL},
    "t2": {k: "dict(n_splits=792, nc=1)" for k in FULL},
    "t3": {(16, 1): "dict(n_splits=264, nc=2)", (32, 1): "dict(n_splits=264, nc=2)",
           (16, 8): "dict(n_splits=264, nc=1, coop=0)", (32, 8): "dict(n_splits=264, nc=1, coop=0)"},
}


def main():
    raw = (H.VENDOR / "muse_attention_decode.py").read_bytes().replace(b"\r\n", b"\n")
    if H.blob_sha(raw) != H.BLOBS["muse_attention_decode.py"]:
        sys.exit("vendored file hash mismatch")
    src = raw.decode("ascii")
    for old, new in zip(H.M1_OLD, H.M1_NEW):
        src = H.sub1(src, old, new, "M1")
    wave = []
    for name, arm in ARMS.items():
        extra = "".join(f"    ({b}, {l}, False, 1): {v},\n" for (b, l), v in arm.items())
        t = H.sub1(src, OLD, OLD[:-2] + extra + "}\n", "_TUNED")
        l1, rest = t.split("\n", 1)
        head = (f"# nonce hya-tune-{name} (R-hya-t: vjeux f35a41ee + HYA M1 + flat full split plan {name}; tools/hya_tune_board.py)\n"
                "# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41ee1c), licensed under the Apache\n"
                "# License 2.0 (full text: tools/vendor/vjeux_okc2026_f35a41ee/LICENSE). Modified by team joeyhlu, 2026-10-31: HYA M1\n"
                "# (spin_fail sets a sticky flag and the poll loop breaks instead of __trap) and the _TUNED entries for full B 16 / 32.\n")
        out = ROOT / "kernels" / "muse_attention_decode" / f"hya_{name}.py"
        out.write_text(l1 + "\n" + head + rest, newline="\n", encoding="ascii")
        wave.append(f"muse_attention_decode kernels/muse_attention_decode/hya_{name}.py")
        print(out.relative_to(ROOT), len(out.read_bytes()), "B;", arm or "control")
    (ROOT / "tools" / "hya_tune_wave.txt").write_text("\n".join(wave) + "\n", newline="\n")


if __name__ == "__main__":
    main()

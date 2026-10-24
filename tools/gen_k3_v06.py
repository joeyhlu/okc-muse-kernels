"""gen_k3_v06.py: k3_v06.py = k3_v05.py (from tools/gen_k3.py) with the reviewed Marlin block-size range (rule R3).

    python3 -I tools/gen_k3_v06.py          # verifies k3_v05 is current, writes k3_v06.py, asserts the invariants

Only two text patches: the nonce line, and BLOCK_SIZE_M_RANGES -> ((49, 1 << 30, 16),) with its comment
(kernels/review_scored_delta_1.md: only the >=49 range gains on the judge; small-T ranges were noise or negative).
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels/muse_decode_layer/k3_v05.py"
OUT = ROOT / "kernels/muse_decode_layer/k3_v06.py"


def main():
    chk = subprocess.run([sys.executable, "-I", str(ROOT / "tools/gen_k3.py"), "--check"], capture_output=True, text=True)
    assert chk.returncode == 0 and "up to date" in chk.stdout, "k3_v05.py is not what gen_k3.py produces:\n" + chk.stdout + chk.stderr
    s = SRC.read_text()
    lines = s.split("\n")
    assert lines[1].startswith("# nonce k3-v05-1"), lines[1]
    lines[1] = "# nonce k3-v06-1 (k3_v05 + Marlin block_size_m 16 at >=49 tokens only; review kernels/review_scored_delta_1.md)"
    s = "\n".join(lines)
    old = "BLOCK_SIZE_M_RANGES = ((9, 24, 16), (25, 48, 8), (49, 1 << 30, 16))"
    assert s.count(old) == 1, "ranges line not found exactly once"
    s = s.replace(old, "BLOCK_SIZE_M_RANGES = ((49, 1 << 30, 16),)  # judge: only >=49 gains (bs128 1.37x, bs256 1.29x); "
                       "small-T ranges were noise or negative", 1)
    OUT.write_text(s)
    a, b = SRC.read_text().split("\n"), s.split("\n")
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) and len(diff) == 2, f"expected exactly 2 changed lines, got {diff}"
    print(f"[gen_k3_v06] wrote {OUT.name}: 2 lines changed vs k3_v05 (lines {[i + 1 for i in diff]})")


if __name__ == "__main__":
    main()

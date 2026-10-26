"""K1 v24 patches for the layer file kernels/muse_decode_layer/k3_v23.py (its K1 section = kernels/muse_attention_decode/
k1_v7.py's CUDA / host code). Measured in kernels/muse_attention_decode/k1_v24_lab.py (box_out/v24/run1-4.log).

Two device-only, bitwise-neutral changes to the L=1 path of k1_main (texts: kernels/muse_attention_decode/k1_v24_sk.py):
  * SECT: loader chunk map ch = part + 4u (was 2 part + u): every 16-B cp.async instruction of a warp covers whole 32-B
    sectors of the key rows instead of every other 16-B chunk of all four sectors; same bytes, same smem positions.
  * PAIR: the per-tile __syncthreads of the L=1 main loop -> bar.sync (1 + q4), 64: warps 2 q4 / 2 q4 + 1 load exactly
    the 16 keys (both heads) they consume, so the barrier only has to order that pair (named barriers 1..4; no other
    named barrier exists in the kernel).
Measured (same process, interleaved graph replays, cold L2, judge inputs; run4 / run3 / run1): decode full B32 196.9 ->
189.3 us (-7.6, 3.32 -> 3.46 TB/s; PAIR alone -6.5..-6.9 in 4 runs), B16 103.3 -> 102.1 (-1.2), B8 57.6 -> 56.8 (-0.75),
B1 -0.2; decode sliding B32 within +-0.7 (noise there); outputs and cache rows bitwise equal to the shipped build
(box_out/v24/bitwise.log: decode full B32 / B16 / B1, sliding B32).

  python tools/k1_v24_patches.py            # self-check: each `old` once in k3_v23.py and in k1_v7.py, patched files parse
  python tools/k1_v24_patches.py OUT.py     # also writes the patched k3_v23.py text to OUT.py
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAYER = ROOT / "kernels/muse_decode_layer/k3_v23.py"
CANON = ROOT / "kernels/muse_attention_decode/k1_v7.py"
sys.path.insert(0, str(ROOT / "kernels/muse_attention_decode"))
import k1_v24_sk as S  # noqa: E402

# (what, old, new)
PATCHES = [S.SECT_PATCH, S.PAIR_PATCH]


def apply(text, patches=PATCHES):
    for what, old, new in patches:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"k1_v24_patches: {what!r} anchor occurs {n} times (expected 1)")
        text = text.replace(old, new)
    return text


def main():
    layer = LAYER.read_text(encoding="utf-8")
    canon = CANON.read_text(encoding="utf-8")
    for what, old, new in PATCHES:
        assert layer.count(old) == 1 and canon.count(old) == 1, what
    out = apply(layer)
    ast.parse(out)
    ast.parse(apply(canon))
    print(f"[k1_v24] {len(PATCHES)} patches: anchors unique in {LAYER.name} and {CANON.name}; patched files parse")
    if len(sys.argv) > 1:
        pathlib.Path(sys.argv[1]).write_text(out, encoding="utf-8")
        print(f"[k1_v24] wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()

"""vct_patches.py (VCT, lead 2026-10-29): VML_COST rows for buckets 128 / 256 re-fit to the gated per-class stage paces.

Paces = tools/e4_model.py fit() on box_out/lat32/m_ship (gated full-clock marks, R-lat32-1); refit[k] = 128 x pace[k] / pace[0]
per phase; arm 'blend' = round((table + refit) / 2) (the $0 replay in the lead's scratch vct0.py: max vm_main end -7.3 us median
at T=256, -4.4 at T=128; refit -7.7 / -5.4). Only call-time arguments change (VML_COST is passed to _vml_call), no CUDA source edit.

  python -I tools/vct_patches.py --base=kernels/muse_decode_layer/k3_gc_21.py --out=kernels/lab/k3_vct_1.py [--arm=blend|refit]
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
TABLE = {256: [128, 153, 172, 186, 198, 216, 232, 248, 128, 149, 168, 180, 191, 208, 219, 236],
         128: [128, 157, 174, 194, 208, 230, 246, 262, 128, 154, 179, 189, 219, 229, 250, 267]}
REFIT = {256: [128, 153, 153, 164, 170, 179, 188, 201, 128, 145, 159, 168, 174, 184, 185, 200],
         128: [128, 151, 154, 166, 172, 184, 193, 200, 128, 144, 161, 168, 174, 185, 188, 201]}


def main():
    base = ROOT / ARGS.get("base", "kernels/muse_decode_layer/k3_gc_21.py")
    out = ROOT / ARGS["out"]
    arm = ARGS.get("arm", "blend")
    assert not out.exists(), f"{out} exists (never overwrite)"
    src = base.read_text(encoding="utf-8")
    new = {T: (REFIT[T] if arm == "refit" else [int(round((a + b) / 2)) for a, b in zip(TABLE[T], REFIT[T])]) for T in TABLE}
    t = src
    for T in (256, 128):
        old = f"{T}: {TABLE[T]}"
        assert t.count(old) == 1, f"anchor {T} not unique / missing"
        t = t.replace(old, f"{T}: {new[T]}", 1)
    lines = t.split("\n")
    lines[1] = f"# nonce {out.stem.replace('_', '-')} ({base.stem} + VCT {arm}: VML_COST 256 {new[256]} / 128 {new[128]}; tools/vct_patches.py)"
    t = "\n".join(lines)
    assert t.isascii() and len(t.encode()) < 1_000_000
    compile(t, str(out), "exec")
    out.write_text(t, encoding="utf-8", newline="\n")
    print(f"[vct_patches] {base.name} -> {out.relative_to(ROOT).as_posix()} arm {arm}: 256 {new[256]} | 128 {new[128]}")


if __name__ == "__main__":
    main()

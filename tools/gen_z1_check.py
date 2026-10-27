"""Generate the Z1 step-0 check file: the shipped MoE-board K2A submission (study-a sa_k35moe.py, verbatim,
kernel_fn unchanged) + the Z1 lab (kernels/lab/z1_moe_lab.py) run from prepare() after K2A's own prepare.

  python tools/gen_z1_check.py [out=kernels/muse_moe_w4a16/z1_check.py] [nonce]
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / ".claude/worktrees/study-a-b48069/kernels/muse_moe_w4a16/sa_k35moe.py"
LAB = ROOT / "kernels/lab/z1_moe_lab.py"

WRAP = '''

# ======== Z1 step 0 (notes/study/moe_structure_fresh.md s7, request R-moe1-2): lab in prepare(), kernel_fn = K2A ========
def prepare(weights):
    state = _sa_prepare(weights)
    if state.get("st") is not None:
        try:
            z1_lab(state["st"], state["w"], kernel_fn, state, _k2a_reencode_scales, _k2a31_pack_words, K2A31_BITPOS3, log=lambda s: print(s, flush=True), rounds=60, inputs=4,
                   sizes=(1, 4, 8))
        except Exception as ex:  # noqa: BLE001
            print("[z1] lab FAILED: " + repr(ex)[-3000:], flush=True)
        torch.cuda.synchronize()
    return state
'''


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kernels/muse_moe_w4a16/z1_check.py"
    nonce = sys.argv[2] if len(sys.argv) > 2 else "z1-step0-a"
    base = SRC.read_text(encoding="utf-8")
    assert base.startswith("# okc: problem=muse_moe_w4a16\n")
    assert base.count("\ndef prepare(weights):\n") == 1
    base = base.replace("\ndef prepare(weights):\n", "\ndef _sa_prepare(weights):\n")
    lab = LAB.read_text(encoding="utf-8")
    body = lab[lab.index("Z1_CUDA = r"):]          # drop the lab's docstring and imports
    text = (base + f"\n\n# nonce: {nonce}\nimport statistics  # Z1 lab\n\n" + body + WRAP)
    ast.parse(text)
    out.write_text(text, encoding="utf-8", newline="\n")  # LF (repo rule: submission files stay LF)
    print(f"wrote {out} ({len(text)} bytes)")


if __name__ == "__main__":
    main()

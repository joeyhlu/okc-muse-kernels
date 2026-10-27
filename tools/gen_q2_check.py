"""Generate the Q2 step-0 judge check file: the shipped MoE-board K2A submission (study-a sa_k35moe.py, verbatim,
kernel_fn unchanged) + the Q2 lab (kernels/lab/q2_moe_lab.py) run from prepare() after K2A's own prepare.

  python tools/gen_q2_check.py [out=kernels/muse_moe_w4a16/q2_check.py] [nonce]
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / ".claude/worktrees/study-a-b48069/kernels/muse_moe_w4a16/sa_k35moe.py"
LAB = ROOT / "kernels/lab/q2_moe_lab.py"

WRAP = '''

# ======== Q2 step 0 (notes/study/moe_structure_fresh.md, request R-moe1-1): lab in prepare(), kernel_fn = K2A ========
def prepare(weights):
    state = _sa_prepare(weights)
    if state.get("st") is not None:
        try:
            q2_lab(state["st"], kernel_fn, state, log=lambda s: print(s, flush=True), rounds=60, inputs=4,
                   sizes=(1, 4, 8))
        except Exception as ex:  # noqa: BLE001
            print("[q2] lab FAILED: " + repr(ex)[-3000:], flush=True)
        torch.cuda.synchronize()
    return state
'''


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kernels/muse_moe_w4a16/q2_check.py"
    nonce = sys.argv[2] if len(sys.argv) > 2 else "q2-step0-a"
    base = SRC.read_text(encoding="utf-8")
    assert base.startswith("# okc: problem=muse_moe_w4a16\n")
    assert base.count("\ndef prepare(weights):\n") == 1
    base = base.replace("\ndef prepare(weights):\n", "\ndef _sa_prepare(weights):\n")
    lab = LAB.read_text(encoding="utf-8")
    body = lab[lab.index("Q2_CUDA = r"):]          # drop the lab's docstring and imports
    text = (base + f"\n\n# nonce: {nonce}\nimport statistics  # Q2 lab\n\n" + body + WRAP)
    ast.parse(text)
    out.write_text(text, encoding="utf-8", newline="\n")  # LF (repo rule: submission files stay LF)
    print(f"wrote {out} ({len(text)} bytes)")


if __name__ == "__main__":
    main()

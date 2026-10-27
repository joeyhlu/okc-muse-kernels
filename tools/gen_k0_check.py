"""Generate the K0 judge check file: the shipped MoE-board K2A submission (study-a sa_k35moe.py, verbatim, kernel_fn
unchanged) + the K0 lab (kernels/lab/k0_sweep_lab.py) run from prepare() after K2A's own prepare.

  python tools/gen_k0_check.py [out=kernels/muse_moe_w4a16/k0_check.py]
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / ".claude/worktrees/study-a-b48069/kernels/muse_moe_w4a16/sa_k35moe.py"
LAB = ROOT / "kernels/lab/k0_sweep_lab.py"

WRAP = '''

# ======== K0 (SWEEP kill test, notes/study/moe_redesign_independent.md s5): lab in prepare(), kernel_fn = K2A ========
def prepare(weights):
    state = _sa_prepare(weights)
    if state.get("st") is not None:
        try:
            k0_lab(state["st"], kernel_fn, state, log=lambda s: print(s, flush=True))
        except Exception as ex:  # noqa: BLE001
            print("[k0] lab FAILED: " + repr(ex)[-3000:], flush=True)
        torch.cuda.synchronize()
    return state
'''


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kernels/muse_moe_w4a16/k0_check.py"
    base = SRC.read_text(encoding="utf-8")
    assert base.startswith("# okc: problem=muse_moe_w4a16\n")
    assert base.count("\ndef prepare(weights):\n") == 1
    base = base.replace("\ndef prepare(weights):\n", "\ndef _sa_prepare(weights):\n")
    lab = LAB.read_text(encoding="utf-8")
    body = lab[lab.index("K0_CUDA = r"):]          # drop the lab's docstring and imports
    text = (base + "\n\nimport statistics  # K0 lab\n\n" + body + WRAP)
    ast.parse(text)
    out.write_text(text, encoding="utf-8", newline="\n")  # LF (repo rule: submission files stay LF)
    print(f"wrote {out} ({len(text)} bytes)")


if __name__ == "__main__":
    main()

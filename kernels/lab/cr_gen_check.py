"""Generate the CR-K1 step-0 check file: the shipped MoE-board K2A submission (study-a sa_k35moe.py, verbatim, kernel_fn
unchanged) + the CR-K1 lab (kernels/lab/cr_k1_lab.py) run from prepare() after K2A's own prepare.

  python kernels/lab/cr_gen_check.py [out=kernels/lab/cr_check_box.py]
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / ".claude/worktrees/study-a-b48069/kernels/muse_moe_w4a16/sa_k35moe.py"
LAB = ROOT / "kernels/lab/cr_k1_lab.py"

WRAP = '''

# ======== CR-K1 (clean-room MoE step 0, notes/study/moe_cleanroom.md): lab in prepare(), kernel_fn = K2A ========
def _cr_d2(d):
    x = d["hidden_states"].contiguous()
    return {"hidden_states": x, "topk_ids": d["topk_ids"], "topk_weights": d["topk_weights"],
            "config": {"local_start": int(d["config"]["local_start"])}}


def _cr_k1_only(st, d):
    d2 = _cr_d2(d)
    rep = _k2a_bucket_of(d2["hidden_states"].shape[0])
    r37 = st["r37"].get(rep) if (K2A37_ENABLE and st.get("r37")) else None
    if r37 is not None:
        return _k2a37_call(st, d2, r37[0], r37[1], r37[2], mask=1)
    return _k2a33_call(st, d2, K2A33_PLAN[rep], mask=1, hot_t=K2A33_HOT_T_BUCKET.get(rep),
                       hot_w4=(76 if rep == 256 else None))


def prepare(weights):
    state = _sa_prepare(weights)
    st = state.get("st")
    if st is not None and state["ok"].get(128) and state["ok"].get(256):
        try:
            cr_lab(state["w"], _k2a_dequant, lambda d: kernel_fn(d, state), lambda d: _cr_k1_only(st, d),
                   lambda: _k2a33_reset(st), log=lambda s: print(s, flush=True))
        except Exception as ex:  # noqa: BLE001
            print("[cr] lab FAILED: " + repr(ex)[-3000:], flush=True)
        torch.cuda.synchronize()
        _k2a33_reset(st)
        st["err"].zero_()
    else:
        print("[cr] K2A state not ready: lab skipped", flush=True)
    return state
'''


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kernels/lab/cr_check_box.py"
    base = SRC.read_text(encoding="utf-8")
    assert base.startswith("# okc: problem=muse_moe_w4a16\n")
    assert base.count("\ndef prepare(weights):\n") == 1
    base = base.replace("\ndef prepare(weights):\n", "\ndef _sa_prepare(weights):\n")
    lab = LAB.read_text(encoding="utf-8")
    body = lab[lab.index("CR_CUDA = r"):]          # drop the lab's docstring and imports
    text = base + "\n\nimport math  # CR lab\nimport statistics  # CR lab\n\n" + body + WRAP
    ast.parse(text)
    for bad in ("importlib", "sys.modules"):
        assert bad not in body + WRAP, bad
    assert len(text.encode()) < 1_000_000
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(text.encode())} bytes)")


if __name__ == "__main__":
    main()

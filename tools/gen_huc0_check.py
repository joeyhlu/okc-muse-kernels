"""Generate the HUC0 judge check file: the shipped MoE-board K2A submission (study-a sa_k35moe.py, verbatim, kernel_fn
unchanged) + the HUC0 lab (kernels/lab/huc0_lab.py) run from prepare() after K2A's own prepare.

  python tools/gen_huc0_check.py [out=kernels/muse_moe_w4a16/huc0_check.py]
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / ".claude/worktrees/study-a-b48069/kernels/muse_moe_w4a16/sa_k35moe.py"
LAB = ROOT / "kernels/lab/huc0_lab.py"

WRAP = '''

# ======== HUC0 (hot-under-cold step 0, notes/study/moe_hot_fresh.md): lab in prepare(), kernel_fn = K2A ========
def _huc0_d2(d):
    x = d["hidden_states"].contiguous()
    return {"hidden_states": x, "topk_ids": d["topk_ids"], "topk_weights": d["topk_weights"],
            "config": {"local_start": int(d["config"]["local_start"])}}


def _huc0_k1_only(st, d):
    d2 = _huc0_d2(d)
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
            huc0_lab(st, lambda d: kernel_fn(d, state), lambda d: _huc0_k1_only(st, d), lambda: _k2a33_reset(st),
                     state["w"], _k2a_dequant, log=lambda s: print(s, flush=True))
        except Exception as ex:  # noqa: BLE001
            print("[huc0] lab FAILED: " + repr(ex)[-3000:], flush=True)
        torch.cuda.synchronize()
        _k2a33_reset(st)
        st["err"].zero_()
    return state
'''


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kernels/muse_moe_w4a16/huc0_check.py"
    base = SRC.read_text(encoding="utf-8")
    assert base.startswith("# okc: problem=muse_moe_w4a16\n")
    assert base.count("\ndef prepare(weights):\n") == 1
    base = base.replace("\ndef prepare(weights):\n", "\ndef _sa_prepare(weights):\n")
    lab = LAB.read_text(encoding="utf-8")
    body = lab[lab.index("HUC0_CUDA = r"):]          # drop the lab's docstring and imports
    text = base + "\n\nimport math  # HUC0 lab\nimport statistics  # HUC0 lab\n\n" + body + WRAP
    ast.parse(text)
    assert len(text.encode()) < 1_000_000
    out.write_text(text, encoding="utf-8", newline="\n")  # LF (repo rule: submission files stay LF)
    print(f"wrote {out} ({len(text)} bytes)")


if __name__ == "__main__":
    main()

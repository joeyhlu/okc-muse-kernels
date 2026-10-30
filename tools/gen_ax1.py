"""gen_ax1.py: attention-board file = kernels/muse_attention_decode/vfa_gc16as.py + the AX1 route (tools/ax1_src.py).

    python -I tools/gen_ax1.py --out=kernels/muse_attention_decode/ax1_1.py [--set NAME=VALUE ...]
Anchored, verbatim-checked insertions: the AX1 block before `_AD = {"state": None}`, its build thread in `_ad_prepare`
(PT1 pool, beside VFA's), `_ax1_prepare` after `_vfa_prepare`, and the route first in `_ad_attention` (B = 1 full only;
every other shape keeps its route, and AX1 falls back to the FA4 chain on any prepare / call failure).
--set overrides a module-level AX1_* assignment in the inserted block (e.g. --set "AX1_SPLITS={1: 264}").
"""
import argparse
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sub1(text, old, new, what):
    n = text.count(old)
    if n != 1:
        sys.exit(f"anchor {what!r}: found {n} times")
    return text.replace(old, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="kernels/muse_attention_decode/vfa_gc16as.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    base = (ROOT / a.base).read_text()
    src = (ROOT / "tools" / "ax1_src.py").read_text()
    block = src[src.index('AX1_CPP_SRC = r"""'):]
    for kv in a.set:
        name, val = kv.split("=", 1)
        lines = [ln for ln in block.split("\n") if ln.startswith(name + " = ")]
        if len(lines) != 1:
            sys.exit(f"--set {name}: {len(lines)} assignments")
        block = block.replace(lines[0], f"{name} = {val}")
    text = sub1(base, '_AD = {"state": None}\n', block.rstrip() + "\n\n" + '_AD = {"state": None}\n', "AD state")
    text = sub1(text, '    _PT1["threads"]["vfa"] = th\n    th.start()\n',
                '    _PT1["threads"]["vfa"] = th\n    th.start()\n'
                '    if AX1_ENABLE:\n'
                '        th_ax1 = threading.Thread(target=_pt1_target("ax1", _ax1_build), name="pt1-ax1", daemon=True)\n'
                '        _PT1["threads"]["ax1"] = th_ax1\n'
                '        th_ax1.start()\n', "PT1 start")
    text = sub1(text, '    _vfa_prepare(state, t0)\n    torch.cuda.synchronize()\n',
                '    _vfa_prepare(state, t0)\n    _ax1_prepare(state, t0)\n    torch.cuda.synchronize()\n', "prepare call")
    text = sub1(text, '    o = None\n    ks1_ran = False\n    if _ks1_wanted(qkv, data, cfg, state):\n',
                '    o = None\n    ks1_ran = False\n'
                '    if _ax1_wanted(qkv, data, cfg, state):\n'
                '        try:\n'
                '            o = _ax1_call(qkv, data, cfg).view(tokens, hq * d)\n'
                '        except Exception as ex:\n'
                '            _k3_disable(state, "ax1", False, "AX1", ex)\n'
                '            o = None\n'
                '    if o is None and _ks1_wanted(qkv, data, cfg, state):\n', "route")
    ast.parse(text)
    out = ROOT / a.out
    out.write_text(text, newline="\n")
    print(f"{out}: {len(text.encode())} B (base {len(base.encode())} B)")


if __name__ == "__main__":
    main()

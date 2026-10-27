"""gen_vms_board.py: kernels/muse_moe_w4a16/vms_<n>.py = the board copy of our layer MoE (kernels/muse_moe_w4a16/k0_check.py
without its K0 lab: K2A v3.3 / v3.6 / v3.7 routes) + the VMS one-launch route at bs <= VMS_MAXT (tools/vms_patches.py block,
global ids mapped in-kernel with local_start). Check only (free muse_moe_w4a16 checks), never scored.

    python -I tools/gen_vms_board.py [--n=1] [--maxt=16]
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vms_patches  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N, MAXT, FL = int(ARGS.get("n", 1)), int(ARGS.get("maxt", 16)), int(ARGS.get("fl", 0))
SRC = ROOT / "kernels/muse_moe_w4a16/k0_check.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/vms_{N}.py"

TAIL = r'''
_K3_BUILD_S = {}


def _pt1_wait(name):
    return None


def kernel_fn(data, state):
    st = state["st"]
    x = data["hidden_states"]
    m = x.shape[0]
    if st is not None and st.get("vms_ok") and m <= VMS_MAXT:
        x = x.contiguous()
        if x.data_ptr() % 16:
            x = x.clone()
        vs = st["vms"]
        out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
        vs["mod"].run(x, data["topk_ids"].to(torch.int32).contiguous(), data["topk_weights"].to(torch.float32).contiguous(),
                      vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"], out,
                      vs["nomarks"], m, K2A_POST_NORM_EPS, 0, int(data["config"]["local_start"]), VMS_FL)
        return out
    return _sa_kernel_fn(data, state)


def prepare(weights):
    state = _sa_prepare(weights)
    if state.get("st") is not None:
        _vms_prepare(state["st"], state["w"], weights["w13"].device)
        torch.cuda.synchronize()
    return state
'''


def main():
    t = SRC.read_text(encoding="utf-8")
    cut = t.index("import statistics  # K0 lab")
    t = t[:cut]
    assert t.count("def kernel_fn(data, state):") == 1
    t = t.replace("def kernel_fn(data, state):", "def _sa_kernel_fn(data, state):")
    lines = t.split("\n")
    assert lines[0] == "# okc: problem=muse_moe_w4a16"
    lines[1] = (f"# vms-board-{N}: our layer MoE (K2A, board copy of k3_v35_pt1's) + VMS one-launch MoE at bs <= {MAXT} "
                "(notes/study/vm_small.md); check only, never scored.")
    t = "\n".join(lines)
    cuda, cpp = vms_patches._lab_sources(vms_patches.LAB_V1 if N == 1 else None)
    block = ('VMS_CUDA_SRC = r"""' + cuda + '"""\nVMS_CPP_SRC = r"""' + cpp + '"""\n'
             + vms_patches.PY_BLOCK.replace("@MAXT@", str(MAXT)).replace("@FL@", str(FL)))
    tail = TAIL
    if N == 1:   # frozen v1 lab revision: run() has no fl argument
        block = block.replace(", K2A_POST_NORM_EPS, 1, 0, VMS_FL)", ", K2A_POST_NORM_EPS, 1, 0)").replace(f"VMS_FL = {FL}\n", "")
        tail = tail.replace(', int(data["config"]["local_start"]), VMS_FL)', ', int(data["config"]["local_start"]))')
    s = t.rstrip("\n") + "\n\n" + block + tail
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    dup = sorted({n for n in names if n and names.count(n) > 1})
    assert not dup, dup
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s
    OUT.write_text(s, encoding="utf-8", newline="\n")
    py_compile.compile(str(OUT), doraise=True)
    print(f"[gen_vms_board] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")


if __name__ == "__main__":
    main()

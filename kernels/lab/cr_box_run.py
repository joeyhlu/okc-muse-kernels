"""R-cr-0 (notes/memory/gpu_requests.md): the CR-K1 step-0 lab on the box, run the way the judge runs a check file:
weights from the problem's reference.generate_weights(WEIGHTS_SEED), prepare() (K2A + the CR lab, which prints the arms
and the pre-registered decision), then kernel_fn sanity checks at bs 128 / 256 against the reference.

  python3 -u kernels/lab/cr_box_run.py      (on the box, through tools/box/cr_batch.sh)
"""
import importlib.util
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROB = ROOT / "problems/muse_moe_w4a16"


def load(name, path):
    sys.path.insert(0, str(PROB))
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(PROB))


def main():
    ref = load("cr_reference", PROB / "reference.py")
    t0 = time.time()
    w = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda")
    torch.cuda.synchronize()
    print(f"[box] weights in {time.time() - t0:.1f}s; {torch.cuda.get_device_name()}", flush=True)
    sub = load("cr_sub", ROOT / "kernels/lab/cr_check_box.py")
    t0 = time.time()
    state = sub.prepare(w)
    torch.cuda.synchronize()
    print(f"[box] prepare {time.time() - t0:.1f}s", flush=True)
    for bs in (128, 256):
        d = ref.generate_input(bs, "ep8", 4242 + bs)
        out = sub.kernel_fn(d, state)
        torch.cuda.synchronize()
        ok, msg = ref.check_implementation(d, out, w)
        print(f"[box] kernel_fn bs={bs} after the lab: {ok} {msg}", flush=True)


if __name__ == "__main__":
    main()

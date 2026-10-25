"""k1_v8f pod driver (k1_pp_design.md section 7.4). Box only (needs the GPU); laptop checks live in k1_v8f_emulate.py.

  python3 kernels/muse_attention_decode/k1_v8f_lab.py --unit    # self-test of every L=8 kernel: codes 7, 17, 6, 16, 3
  compute-sanitizer --tool memcheck python3 kernels/muse_attention_decode/k1_v8f_lab.py --san   # codes 7 and 17

--unit runs the file's own _self_test with l8_stages, so both v8f variants (rolled 6/7, unrolled 16/17) and v7's
k1_main (3) are checked against the E4M3-query reference in both merge modes, with the fold counters checked for reset
and NaN output (= an R9 timeout) caught. Register/spill/serialization checks are tools/box/sass_gate.py's job.
--san runs two small judge-generator shapes (L=8 sliding page 1, L=8 full page 64) per code and merge mode and checks
each with the problem's reference.check_implementation.
"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "problems" / "muse_attention_decode"))
import reference  # noqa: E402

spec = importlib.util.spec_from_file_location("k1v8f", str(Path(__file__).with_name("k1_v8f.py")))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
dev = torch.device("cuda")
assert mod._MOD is not None, "k1_v8f did not build"

if "--unit" in sys.argv:
    ok = mod._self_test(dev, l8_stages=(7, 17, 6, 16, 3))
    print(f"UNIT k1_v8f self-test (L=8 codes 7, 17, 6, 16, 3): {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)

if "--san" in sys.argv:
    bad = 0
    for kw in (dict(batch=3, ctx=2047, heads="tp8", layer="sliding", seed=5, page=1, q_len=8),
               dict(batch=2, ctx=3000, heads="tp8", layer="full", seed=6, q_len=8, page=64, dist="soft")):
        for st in (7, 17):
            for fold in (True, False):
                data = reference.generate_input(**kw)
                out = mod._k1(data, fold=fold, stages=st)
                torch.cuda.synchronize()
                ok, msg = reference.check_implementation(data, reference.collect(data, out))
                bad += int(not ok)
                print(f"SAN {kw['layer']} L=8 B={kw['batch']} stages={st} {'fold' if fold else 'sep'}: "
                      f"{'ok' if ok else 'FAIL'} {msg[:90]}", flush=True)
    data = reference.generate_input(batch=2, ctx=3000, heads="tp8", layer="full", seed=7, q_len=8, page=1)
    out = mod.kernel_fn(data)
    torch.cuda.synchronize()
    ok = reference.check_implementation(data, reference.collect(data, out))[0]
    print(f"SAN kernel_fn (routes as shipped: L=8 -> rung-1 FA3): {'ok' if ok else 'FAIL'}", flush=True)
    bad += int(not ok)
    sys.exit(1 if bad else 0)

print(__doc__)

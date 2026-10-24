"""compute-sanitizer driver for k1_v3 (or --file k1_v4.py) (two small shapes, both K1 kernels called directly + the routed kernel_fn).
  compute-sanitizer --tool memcheck python3 kernels/muse_attention_decode/k1_v3_san.py"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "problems" / "muse_attention_decode"))
import reference  # noqa: E402

FILE = sys.argv[sys.argv.index("--file") + 1] if "--file" in sys.argv else "k1_v3.py"
spec = importlib.util.spec_from_file_location("k1san", str(Path(__file__).with_name(FILE)))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for kw in (dict(batch=3, ctx=2047, heads="tp8", layer="sliding", seed=5, page=1),
           dict(batch=2, ctx=3000, heads="tp8", layer="full", seed=6, q_len=8, page=1, dist="soft")):
    for name, fn in (("k1 fold", lambda d: mod._k1(d, fold=True)), ("k1 sep", lambda d: mod._k1(d, fold=False)),
                     ("kernel_fn", mod.kernel_fn)):
        data = reference.generate_input(**kw)
        out = fn(data)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        print(f"SAN {kw['layer']} L={kw.get('q_len', 1)} B={kw['batch']} {name}: {'ok' if ok else 'FAIL'} {msg[:90]}", flush=True)

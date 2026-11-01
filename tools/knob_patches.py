"""R-knob-1: TINY / TINY2 (HYBLO, T <= 8 MoE) knob variants for a paired in-layer A/B.

  python tools/knob_patches.py [--base=kernels/lab/k3_vmsx_ship_rt.py]

writes kernels/lab/k3_knob_ab.py: the ship file with ALL variants built in prepare and picked at call/capture time by the
module global KNOB (layer_ab3.py variants set it); KNOB = "ship" reproduces the ship (same flags, same source except a
default-0 define).
  trim    = TINY smem = base + xa only (T <= 8 with SK_T 8 cannot make fat jobs, so the fat x region is never touched;
            ~210 KB -> ~144 KB; mechanism: R-hybsm-2 / R-vt16-3 unused-smem cost in-layer)
  ns4     = TINY ring 3 -> 4 slots (fits inside the existing max(xa, xb) allocation)
  trimns4 = both
Every anchor must match exactly once (asserted). Arithmetic is unchanged: outputs must stay bitwise equal to ship.
"""
import sys
from pathlib import Path

base = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--base=")), "kernels/lab/k3_vmsx_ship_rt.py"))
src = base.read_text(encoding="utf-8")


def sub(s, old, new):
    n = s.count(old)
    assert n == 1, f"anchor matched {n} times: {old[:90]!r}"
    return s.replace(old, new)


# 1. CUDA: trim switch (default 0 = ship behaviour)
s = sub(src,
        "size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;\n"
        "return base + (xa > xb ? xa : xb);\n}\n\nstatic int g_okc_pdl = 0;",
        "size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;\n"
        "#ifndef OKC_TRIM\n#define OKC_TRIM 0\n#endif\n"
        "return base + ((OKC_TINY && OKC_SKT == 8 && OKC_TRIM) ? xa : (xa > xb ? xa : xb));\n}\n\nstatic int g_okc_pdl = 0;")
# 2. variant keys: "tiny" / "tiny2" (ship) plus "<k>@<knob>"
s = sub(s,
        "_HYBLO_FLAGS = ('-O3',",
        "KNOB = 'ship'\n"
        "KNOB_VARS = {'ship': (), 'trim': ('-DOKC_TRIM=1',), 'ns4': ('-DOKC_NS4=1',), 'trimns4': ('-DOKC_TRIM=1', '-DOKC_NS4=1')}\n"
        "_HYBLO_FLAGS = ('-O3',")
s = sub(s,
        "def _hyblo_key(m):\n    return next((k for k, mx, _ in _HYBLO_KEYS if m <= mx), None)\n",
        "def _hyblo_key(m, knob=None):\n    k = next((k for k, mx, _ in _HYBLO_KEYS if m <= mx), None)\n"
        "    knob = KNOB if knob is None else knob\n    return k if (k is None or knob == 'ship') else f'{k}@{knob}'\n"
        "def _knob_flags(knob):\n"
        "    fl = list(_HYBLO_FLAGS) + [f for f in KNOB_VARS[knob] if f != '-DOKC_NS4=1']\n"
        "    return [('-DOKC_NS=4' if (f == '-DOKC_NS=3' and '-DOKC_NS4=1' in KNOB_VARS[knob]) else f) for f in fl]\n")
s = sub(s,
        "    for k, _, var in _HYBLO_KEYS:   # one nvcc at a time (R-hovl-1: parallel extra builds froze a container)\n"
        "        fl = list(_HYBLO_FLAGS) + list(var)\n",
        "    for k0, _, var in _HYBLO_KEYS:   # one nvcc at a time (R-hovl-1: parallel extra builds froze a container)\n"
        "      for knob in KNOB_VARS:\n"
        "        k = k0 if knob == 'ship' else f'{k0}@{knob}'\n"
        "        fl = _knob_flags(knob) + list(var)\n")
s = sub(s,
        '            _HYBLO["mods"][k] = load_inline(name=f"hyblo_{k}_{tag}",',
        '            _HYBLO["mods"][k] = load_inline(name=f"hyblo_{k.replace(\'@\', \'_\')}_{tag}",')
# 3. self-test every variant of the bucket
s = sub(s,
        "            for k in (_hyblo_key(bs),):\n                if k != \"w\" and k not in _HYBLO[\"mods\"]:",
        "            for k in tuple(_hyblo_key(bs, kn) for kn in KNOB_VARS):\n                if k != \"w\" and k not in _HYBLO[\"mods\"]:")
first, rest = s.split("\n", 1)
assert first.startswith("# okc: problem=muse_decode_layer")
s = f"{first}\n# knob lab R-knob-1 (tools/knob_patches.py): in-file TINY variants, KNOB selects\n{rest}"
out = Path("kernels/lab/k3_knob_ab.py")
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write(s)
print(out, out.stat().st_size)

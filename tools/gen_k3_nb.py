"""gen_k3_nb.py: k3_nb_<n> = k3_gc_16 + nearby-shape route tables (dispatch only, no kernel-body change).

  python tools/gen_k3_nb.py <n> [--vml="((33, 48, 128),)"] [--ks3="((17, 31, 32), (33, 64, 32))"]

NB_VML_REP: VML config bucket (var / flags / cost) for T in [lo, hi] instead of the K2A bucket; the K2A fallbacks keep theirs.
NB_KS3_B:   decode (q_len 1) batches B in [lo, hi] without their own KS3_ROUTE entry use KS3 with key (k, 1)'s ns / trig.
Scored T (1, 8, 16, 32, 128, 256; decode B 1/16/32) are outside every default range -> byte-identical paths.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels/muse_decode_layer/k3_gc_16.py"

EDITS = [
    # VML gate in the expert dispatcher: own bucket for the VML config, K2A fallbacks keep rep
    ("    rep = _k2a_bucket_of(x.shape[0])\n    if VML_ENABLE and VML_BUCKETS.get(rep) and _VML[\"ok\"].get(rep) and x.shape[0] <= _vml_MMAX:\n",
     "    rep = _k2a_bucket_of(x.shape[0])\n    vrep = _nb_vml_rep(x.shape[0])\n"
     "    if VML_ENABLE and VML_BUCKETS.get(vrep) and _VML[\"ok\"].get(vrep) and x.shape[0] <= _vml_MMAX:\n"),
    ("    x = data[\"hidden_states\"].contiguous()\n    rep = _k2a_bucket_of(x.shape[0])\n    var = _VML[\"force\"]",
     "    x = data[\"hidden_states\"].contiguous()\n    rep = _nb_vml_rep(x.shape[0])\n    var = _VML[\"force\"]"),
    ("_vml_E_LOC, _vml_TOPK, _vml_KDIM = 32, 8, 4096\n",
     "_vml_E_LOC, _vml_TOPK, _vml_KDIM = 32, 8, 4096\nNB_VML_REP = {vml}\n\n\n"
     "def _nb_vml_rep(m):\n    for lo, hi, rep in NB_VML_REP:\n        if lo <= m <= hi:\n            return rep\n"
     "    return _k2a_bucket_of(m)\n\n\n"),
    # KS3 for nearby decode batches
    ("def _ks3_routed(key, state):\n    return bool(KS3_ENABLE and key[1] == 1 and KS3_ROUTE.get(key) and KS3_NS.get(key) in",
     "NB_KS3_B = {ks3}\n\n\ndef _ks3_key(key):\n    if key[1] == 1 and key not in KS3_ROUTE:\n"
     "        for lo, hi, k in NB_KS3_B:\n            if lo <= key[0] <= hi:\n                return (k, 1)\n    return key\n\n\n"
     "def _ks3_routed(key, state):\n    key = _ks3_key(key)\n"
     "    return bool(KS3_ENABLE and key[1] == 1 and KS3_ROUTE.get(key) and KS3_NS.get(key) in"),
    ("    ns = KS3_NS[(B, 1)] if ns is None else ns\n    trig = KS3_TRIG.get((B, 1), 0) if trig is None else trig\n",
     "    ns = KS3_NS[_ks3_key((B, 1))] if ns is None else ns\n    trig = KS3_TRIG.get(_ks3_key((B, 1)), 0) if trig is None else trig\n"),
]


def main():
    n = sys.argv[1]
    opt = {"vml": "()", "ks3": "()"}
    for a in sys.argv[2:]:
        k, v = a[2:].split("=", 1)
        opt[k] = repr(ast.literal_eval(v))
    s = SRC.read_text(encoding="utf-8")
    for old, new in EDITS:
        assert s.count(old) == 1, old[:80]
        s = s.replace(old, new.format(**opt) if "{" in new and ("{vml}" in new or "{ks3}" in new) else new)
    ast.parse(s)
    out = ROOT / f"kernels/lab/k3_nb_{n}.py"
    out.write_bytes(s.encode("utf-8"))
    print(f"{out.name}: {len(s.encode('utf-8'))} B (base {SRC.stat().st_size}), NB_VML_REP={opt['vml']} NB_KS3_B={opt['ks3']}")


if __name__ == "__main__":
    main()

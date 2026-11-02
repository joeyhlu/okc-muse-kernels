"""TNW lab: TINY / TINY2 (HYBLO, T <= 8) built with more warps (latency hiding: 2 -> 4 warps per SMSP; NCH = 64 / NW halves the
per-warp x registers). Flag-only variants of the shipped CUDA (no source change). Module global TNW selects at call / capture time:
0 ship (NW 8), 1 NW 16, 2 NW 16 + NACC 2, 3 NW 4. Base kernels/muse_decode_layer/k3_moexr_c1.py -> kernels/lab/k3_tnw_ab.py."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexr_c1.py"
OUT = ROOT / "kernels/lab/k3_tnw_ab.py"
ARMS = {1: ("-DOKC_NW=16",), 2: ("-DOKC_NW=16", "-DOKC_NACC=2"), 3: ("-DOKC_NW=4",)}

GLUE = r'''
TNW = 0
_TNW_ARMS = %r

def _tnw_flags(base, extra):
 out = [f for f in base if not any(f.startswith(e.split("=")[0] + "=") for e in extra)]
 return out + list(extra)

def _tnw_build_all():
 from torch.utils.cpp_extension import load_inline
 for a, extra in _TNW_ARMS.items():
  for k, _, var in _HYBLO_KEYS:
   fl = _tnw_flags(list(_HYBLO_FLAGS) + list(var), extra)
   kk = f"{k}n{a}"
   try:
    tag = hashlib.sha1((_HYB_MOE_CPP + _HYBLO_CUDA + " ".join(fl)).encode()).hexdigest()[:12]
    _HYBLO["mods"][kk] = load_inline(name=f"hyblo_{kk}_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=_HYBLO_CUDA,
     functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
   except Exception as ex:
    _HYBLO["err"][kk] = repr(ex)[:300]
    print(f"[tnw] build {kk} failed: {ex!r}"[:600], flush=True)

def _tnw_prepare(st, w, dev):
 _pt1_wait("tnw")
 for a in _TNW_ARMS:
  for k, _, _v in _HYBLO_KEYS:
   kk = f"{k}n{a}"
   _HYBLO["ok"][kk] = False
   if kk not in _HYBLO["mods"] or not _HYBLO["ok"].get(k):
    print(f"[tnw] {kk} OFF (module {kk in _HYBLO['mods']}, ship ok {_HYBLO['ok'].get(k)})", flush=True)
    continue
   good = True
   try:
    for bs, seed, dist in HYBLO_SELF_TEST:
     if _hyblo_key(bs) != k:
      continue
     d = _hyblo_inputs(bs, seed, dev, dist)
     x, ids, wts = d["hidden_states"].contiguous(), d["topk_ids"], d["topk_weights"]
     ref = _k2a_ref_fp32(d, w)
     o_s = _hyblo_run(k, x, ids, wts).clone()
     o_q = _hyblo_run(kk, x, ids, wts).clone()
     o_q2 = _hyblo_run(kk, x, ids, wts).clone()
     torch.cuda.synchronize()
     ev = _k2a_rel_err(o_q, o_s)[0] if dist != "none" else 0.0
     tot, row = _k2a_rel_err(o_q, ref) if dist != "none" else (0.0, 0.0)
     rep = _k2a_rel_err(o_q2, o_q)[0] if dist != "none" else 0.0
     ok = bool(torch.isfinite(o_q.float()).all()) and ev < 2e-3 and tot < 1e-2 and row < 5e-2 and rep < 1e-3
     print(f"[tnw] {kk} bs={bs} {dist} vs ship {ev:.2e} vs FP32 {tot:.2e} row {row:.2e} rep {rep:.1e} bitwise {bool(torch.equal(o_q, o_s))} -> {'ok' if ok else 'FAIL'}", flush=True)
     good &= ok
   except Exception as ex:
    good = False
    print(f"[tnw] {kk} self-test raised {ex!r}"[:400], flush=True)
   _HYBLO["ok"][kk] = bool(good and _hyb_guards_ok())
   print(f"[tnw] {kk} {'LIVE' if _HYBLO['ok'][kk] else 'OFF'}", flush=True)
''' % (ARMS,)


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:100])
    return s.replace(a, b, 1)


def main():
    src = BASE.read_text(encoding="utf-8").replace("\r\n", "\n")
    src = once(src, "def _hyblo_key(m):\n", GLUE + "\ndef _hyblo_key(m):\n")
    src = once(src, ' k="w" if HYBLO_K=="w" else _hyblo_key(m)\n',
               ' k="w" if HYBLO_K=="w" else _hyblo_key(m)\n if k!="w" and TNW and _HYBLO["ok"].get(f"{k}n{TNW}"):\n  k=f"{k}n{TNW}"\n')
    src = once(src, '  jobs.append(("hyb",_hyb_build))\n', '  jobs.append(("hyb",_hyb_build))\n  jobs.append(("tnw",_tnw_build_all))\n')
    a = '  _hyblo_prepare(state["k2a33"]["st"],'
    assert src.count(a) == 1
    i = src.index(a); j = src.index("\n", i) + 1
    line = src[i:j]
    src = src[:j] + line.replace("_hyblo_prepare(", "_tnw_prepare(") + src[j:]
    compile(src, str(OUT), "exec")
    OUT.write_bytes(src.encode("ascii"))
    print("TNW lab", OUT, len(src.encode("ascii")), "B")


if __name__ == "__main__":
    main()

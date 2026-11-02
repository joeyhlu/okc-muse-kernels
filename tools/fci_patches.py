"""FCI lab (public-code port, keell0renz E374 'st2l': OKC_WGSFCI 4, 'FCI 3..7 sweep best'): the WGS/HYB deferred split-unit flush
runs behind commit group OKC_WGSFCI of the first stage; ours ships 3. Timing-only knob (bitwise-identical output expected).
Base kernels/muse_decode_layer/k3_moexr_c2.py -> kernels/lab/k3_fci_ab.py. Module global FCI: 0 ship bank, 4 / 5 / 6 variant banks
(separate -DOKC_WGSFCI=N builds of the same HYB source; the ship bank keeps the default 3 = same SASS as the row file)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexr_c2.py"
OUT = ROOT / "kernels/lab/k3_fci_ab.py"

GLUE = r'''
FCI = 0
FCI_VALUES = (4, 5, 6)
_HYBF = {}

def _fci_build_all():
 from torch.utils.cpp_extension import load_inline
 def one(n, f7):
  H = _HYBF[n]
  try:
   fl = list(HYB_FLAGS) + (list(HYB_F7) if f7 else []) + ['-DOKC_SC6=1', '-DOKC_MFIN=1', '-DOKC_PLAN=1', f'-DOKC_WGSFCI={n}']
   tag = hashlib.sha1((_HYB_MOE_CPP + _HYB_MOE_CUDA + " ".join(fl)).encode()).hexdigest()[:12]
   H["mod7" if f7 else "mod"] = load_inline(name="hyb_fci_" + tag, cpp_sources=_HYB_MOE_CPP, cuda_sources=_HYB_MOE_CUDA,
    functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
  except Exception as ex:
   H["err"] = repr(ex)[:500]
   print(f"[fci] HYB build FCI={n} f7={f7} failed: {ex!r}", flush=True)
 for n in FCI_VALUES:
  _HYBF[n] = {"mod": None, "mod7": None, "ok": False, "pdl": -1, "pdl7": -1, "w": None, "sc": None, "full": None, "nsm": 0, "err": None}
 ths = [threading.Thread(target=one, args=(n, f7), daemon=True) for n in FCI_VALUES for f7 in (False, True)]
 for th in ths: th.start()
 for th in ths: th.join()

def _fci_prepare(dev):
 _pt1_wait("fci")
 if not _HYB["ok"]:
  print("[fci] ship HYB bank unavailable; FCI arms off", flush=True)
  return
 for n in FCI_VALUES:
  H = _HYBF.get(n)
  if H is None:
   continue
  H["ok"] = False
  try:
   if H["mod"] is None or H["mod7"] is None:
    raise RuntimeError("missing compiled module: " + str(H["err"]))
   H["w"], H["nsm"] = _HYB["w"], _HYB["nsm"]
   H["sc"], H["full"] = _moexr_scratch(dev, True)
   good, nbit = True, 0
   for T in (64, 128, 193, 256, 320):
    for dist in ("normal", "skewed"):
     d = _k2a_self_test_inputs(T, 13000 + T + (dist == "skewed"), dev, dist)
     x, ids, tw = d["hidden_states"].contiguous(), d["topk_ids"].to(torch.int32).contiguous(), d["topk_weights"].float().contiguous()
     ship = _moexr_call_bank(_HYB, x, ids, tw).clone()
     out = _moexr_call_bank(H, x, ids, tw).clone()
     torch.cuda.synchronize()
     rel = _k2a_rel_err(out, ship)[0]
     bit = bool(torch.equal(out, ship))
     nbit += bit
     good &= bool(torch.isfinite(out.float()).all() and rel < 1e-3)
   good &= _moexr_guards(H)
   H["ok"] = bool(good)
   print(f"[fci] FCI={n} {'LIVE' if H['ok'] else 'OFF'}: bitwise {nbit}/10 vs ship", flush=True)
  except Exception as ex:
   H["err"] = repr(ex)[:500]
   print(f"[fci] FCI={n} OFF: {H['err']}", flush=True)
'''


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:100])
    return s.replace(a, b, 1)


def main():
    src = BASE.read_text(encoding="utf-8").replace("\r\n", "\n")
    src = once(src, "#define OKC_WGSFCI 3\n", "#ifndef OKC_WGSFCI\n#define OKC_WGSFCI 3\n#endif\n")
    src = once(src, "def _moexr_bank():\n return _HYB if MOEXR else None\n",
               GLUE + "\ndef _moexr_bank():\n if not MOEXR: return None\n"
               " H = _HYBF.get(FCI) if FCI else None\n return H if (H is not None and H['ok']) else _HYB\n")
    src = once(src, '  jobs.append(("hyb",_hyb_build))\n', '  jobs.append(("hyb",_hyb_build))\n  jobs.append(("fci",_fci_build_all))\n')
    a = '  _hyblo_prepare(state["k2a33"]["st"],'
    i = src.index(a); j = src.index("\n", i) + 1
    assert src[i:j].rstrip().endswith('weights["w13"].device)')
    src = src[:j] + '  _fci_prepare(weights["w13"].device)\n' + src[j:]
    compile(src, str(OUT), "exec")
    OUT.write_bytes(src.encode("ascii"))
    print("FCI lab", OUT, len(src.encode("ascii")), "B")


if __name__ == "__main__":
    main()

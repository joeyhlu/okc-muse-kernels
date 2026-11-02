"""DQS lab (copy of tools/dqw_patches.py with the shift variant): NVFP4 unpack (dq4) with the two funnel rotates done as IMAD.WIDE.U32 (FMA pipe) and folded into the masking LOP3,
so the INT pipe carries 5 instead of 7 ops per code word (same 12 issue slots, bitwise-identical output).
Base kernels/muse_decode_layer/k3_moexr_c1.py -> kernels/lab/k3_dqw_ab.py. Module global DQW: 0 ship, 1 HYB / MOEXR bank (T >= 49),
2 HYBLO TINY / TINY2 builds (T <= 8), 3 both. Separate compiled modules (-DOKC_DQW=1); the ship modules keep OKC_DQW=0 (same SASS)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexr_c1.py"
OUT = ROOT / "kernels/lab/k3_dqs_ab.py"

DQ4 = """__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
constexpr uint32_t M = 0x81C081C0u;
r0 = q & M;
r1 = __funnelshift_r(q, q, 6) & M;
r2 = __funnelshift_r(q, q, 10) & M;
r1 = (q * 16u) & M;
r2 = __funnelshift_l(q, q, 10) & M;
r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
return;
r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
}
"""
DQ4W = """#ifndef OKC_DQW
#define OKC_DQW 0
#endif
#if OKC_DQW
// DQS: the ship's q*16 / q*64 (ptxas emits IMAD.SHL on the FMA pipe) as funnel shifts with a runtime-zero low word -> SHF on the
// INT pipe (combo22 showed the FMA pipe, not INT, binds the unpack). r2 / the rotate in r3 are unchanged.
__device__ __forceinline__ uint32_t dqs_shl(uint32_t z, uint32_t q, int k) {
uint32_t r;
asm("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(z), "r"(q), "r"(k));
return r;
}
__device__ __forceinline__ void dq4w(uint32_t q, uint32_t z, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
constexpr uint32_t M = 0x81C081C0u;
r0 = q & M;
r1 = dqs_shl(z, q, 4) & M;
r2 = __funnelshift_l(q, q, 10) & M;
r3 = (dqs_shl(z, q, 6) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
}
#endif
"""

GLUE = r'''
DQW = 0
_HYBDQ = {"mod": None, "mod7": None, "ok": False, "pdl": -1, "pdl7": -1, "w": None, "sc": None, "full": None, "nsm": 0, "err": None}

def _dqw_build_hyb():
 from torch.utils.cpp_extension import load_inline
 H = _HYBDQ
 def one(f7):
  try:
   fl = list(HYB_FLAGS) + (list(HYB_F7) if f7 else []) + ['-DOKC_SC6=1', '-DOKC_MFIN=1', '-DOKC_PLAN=1', '-DOKC_DQW=1']
   tag = hashlib.sha1((_HYB_MOE_CPP + _HYB_MOE_CUDA + " ".join(fl)).encode()).hexdigest()[:12]
   H["mod7" if f7 else "mod"] = load_inline(name="hyb_dqw_" + tag, cpp_sources=_HYB_MOE_CPP, cuda_sources=_HYB_MOE_CUDA,
    functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
  except Exception as ex:
   H["err"] = repr(ex)[:500]
   print(f"[dqw] HYB build f7={f7} failed: {ex!r}", flush=True)
 ths = [threading.Thread(target=one, args=(f7,), daemon=True) for f7 in (False, True)]
 for th in ths: th.start()
 for th in ths: th.join()

def _dqw_build_all():
 _dqw_build_hyb()
 from torch.utils.cpp_extension import load_inline
 for k, _, var in _HYBLO_KEYS:
  fl = list(_HYBLO_FLAGS) + list(var) + ['-DOKC_DQW=1']
  try:
   tag = hashlib.sha1((_HYB_MOE_CPP + _HYBLO_CUDA + " ".join(fl)).encode()).hexdigest()[:12]
   _HYBLO["mods"][k + "q"] = load_inline(name=f"hyblo_{k}q_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=_HYBLO_CUDA,
    functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
  except Exception as ex:
   _HYBLO["err"][k + "q"] = repr(ex)[:300]
   print(f"[dqw] HYBLO {k} build failed: {ex!r}", flush=True)

def _dqw_prepare(st, w, dev):
 _pt1_wait("dqw")
 H = _HYBDQ
 H["ok"] = False
 if not _HYB["ok"]:
  print("[dqw] ship HYB bank unavailable; DQW arms off", flush=True)
  return
 # HYB / MOEXR arm: bitwise vs the ship bank on the HYB self-test inputs
 try:
  if H["mod"] is None or H["mod7"] is None:
   raise RuntimeError("missing compiled module: " + str(H["err"]))
  H["w"], H["nsm"] = _HYB["w"], _HYB["nsm"]
  H["sc"], H["full"] = _moexr_scratch(dev, True)
  good = True
  for T in (64, 128, 193, 256, 320):
   for dist in ("normal", "skewed"):
    d = _k2a_self_test_inputs(T, 13000 + T + (dist == "skewed"), dev, dist)
    x, ids, tw = d["hidden_states"].contiguous(), d["topk_ids"].to(torch.int32).contiguous(), d["topk_weights"].float().contiguous()
    ship = _moexr_call_bank(_HYB, x, ids, tw).clone()
    out = _moexr_call_bank(H, x, ids, tw).clone()
    torch.cuda.synchronize()
    rel = _k2a_rel_err(out, ship)[0]
    fin = bool(torch.isfinite(out.float()).all())
    ok = bool(fin and rel < 1e-3)
    print(f"[dqw] HYB T={T} {dist} vs ship {rel:.3e} bitwise {bool(torch.equal(out, ship))} -> {'ok' if ok else 'FAIL'}", flush=True)
    good &= ok
  good &= _moexr_guards(H)
  H["ok"] = bool(good)
 except Exception as ex:
  H["err"] = repr(ex)[:500]
 print(f"[dqw] HYB arm {'LIVE' if H['ok'] else 'OFF'}: {H['err']}", flush=True)
 # HYBLO arms: bitwise vs the ship TINY / TINY2 modules on the HYBLO self-test inputs
 for k, _, _v in _HYBLO_KEYS:
  kq = k + "q"
  _HYBLO["ok"][kq] = False
  if kq not in _HYBLO["mods"] or not _HYBLO["ok"].get(k):
   print(f"[dqw] HYBLO {kq} OFF (module {kq in _HYBLO['mods']}, ship ok {_HYBLO['ok'].get(k)})", flush=True)
   continue
  good = True
  for bs, seed, dist in HYBLO_SELF_TEST:
   if _hyblo_key(bs) != k:
    continue
   d = _hyblo_inputs(bs, seed, dev, dist)
   x, ids, wts = d["hidden_states"].contiguous(), d["topk_ids"], d["topk_weights"]
   o_s = _hyblo_run(k, x, ids, wts).clone()
   o_q = _hyblo_run(kq, x, ids, wts).clone()
   torch.cuda.synchronize()
   rel = _k2a_rel_err(o_q, o_s)[0] if dist != "none" else 0.0
   ok = bool(torch.isfinite(o_q.float()).all()) and rel < 1e-3
   print(f"[dqw] HYBLO {kq} bs={bs} {dist} vs ship {rel:.3e} bitwise {bool(torch.equal(o_q, o_s))} -> {'ok' if ok else 'FAIL'}", flush=True)
   good &= ok
  _HYBLO["ok"][kq] = bool(good and _hyb_guards_ok())
  print(f"[dqw] HYBLO {kq} {'LIVE' if _HYBLO['ok'][kq] else 'OFF'}", flush=True)
'''


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:100])
    return s.replace(a, b, 1)


def main():
    src = BASE.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert src.count(DQ4) == 2, src.count(DQ4)
    src = src.replace(DQ4, DQ4 + DQ4W)
    ARGS = "p.zero"
    src = once(src, "dq4(q, r0, r1, r2, r3);\n", "#if OKC_DQW\ndq4w(q, " + ARGS + ", r0, r1, r2, r3);\n#else\ndq4(q, r0, r1, r2, r3);\n#endif\n")
    src = once(src, "dq4(ww[st], r0, r1, r2, r3);\n", "#if OKC_DQW\ndq4w(ww[st], " + ARGS + ", r0, r1, r2, r3);\n#else\ndq4(ww[st], r0, r1, r2, r3);\n#endif\n")
    # HYB bank selection
    src = once(src, "def _moexr_bank():\n return _HYB if MOEXR else None\n",
               GLUE + "\ndef _moexr_bank():\n if not MOEXR: return None\n return _HYBDQ if (DQW & 1) and _HYBDQ['ok'] else _HYB\n")
    # HYBLO route selection
    src = once(src, ' k="w" if HYBLO_K=="w" else _hyblo_key(m)\n',
               ' k="w" if HYBLO_K=="w" else _hyblo_key(m)\n if k!="w" and (DQW & 2) and _HYBLO["ok"].get(str(k)+"q"):\n  k=str(k)+"q"\n')
    # builds + prepare
    src = once(src, '  jobs.append(("hyb",_hyb_build))\n', '  jobs.append(("hyb",_hyb_build))\n  jobs.append(("dqw",_dqw_build_all))\n')
    a = '  _hyblo_prepare(state["k2a33"]["st"],'
    assert src.count(a) == 1
    i = src.index(a); j = src.index("\n", i) + 1
    line = src[i:j]
    assert line.rstrip().endswith('weights["w13"].device)'), line
    src = src[:j] + line.replace("_hyblo_prepare(", "_dqw_prepare(") + src[j:]
    compile(src, str(OUT), "exec")
    if "--check" in sys.argv:
        assert OUT.read_bytes() == src.encode("ascii")
    else:
        OUT.write_bytes(src.encode("ascii"))
    print("DQW lab", OUT, len(src.encode("ascii")), "B")


if __name__ == "__main__":
    main()

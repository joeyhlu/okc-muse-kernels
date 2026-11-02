"""HCT lab = HYB stream-K cost tables (OKC_WGSA / OKC_WGSB) re-fit under gated timing: OKC_WGSV 1 / 2 / 3 = excess over entry 0 scaled
by 0.6 / 1.4 / 0.3 (banks t06 / t14 / t03); built on the PORT lab pattern. PORT doc follows: public-code ports from keell0renz's decode layer (Apache-2.0, same WGS lineage as our HYB) into our HYB (T >= 49):
  ybf  = WGS ybuf in bf16 (keell 'ybfl': epilogue stores + final-pass reads; y is SGLang-precision bf16, the sum of squares and the
         weighted sum stay FP32), our extra PLAN epilogue (c.jst) converted the same way;
  f4   = OKC_WGSFCI 4 (keell st2l / E374) instead of our 3;
  f4y  = both.
Base kernels/muse_decode_layer/k3_moexr_c2.py -> kernels/lab/k3_port_ab.py. Module global PORT selects a variant HYB bank ('' = ship
bank, same SASS as the row file); variants are separate -D builds of the same source, self-tested vs ship in prepare."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexr_c2.py"
OUT = ROOT / "kernels/lab/k3_hct_ab.py"

ST_A = ("st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);\n"
        "st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);\n")
ST_C = ST_A.replace("(jst + tk)", "(c.jst + tk)")
RD = ("const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n"
      "const float4 a = __ldcg(src), b = __ldcg(src + 1);\n")
RD_BF = ("#if OKC_YBF\n"
         "const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);\n"
         "const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)), y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));\n"
         "const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)), y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));\n"
         "const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);\n"
         "#else\n" + RD + "#endif\n")


def st_bf(st):
    bf = st.replace("st_keep_f32(p.ybuf + ", "st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ")
    bf = bf.replace("d4[h] * w0);", "__float2bfloat16(d4[h] * w0));").replace("d4[2 + h] * w1);", "__float2bfloat16(d4[2 + h] * w1));")
    return "#if OKC_YBF\n" + bf + "#else\n" + st + "#endif\n"


GLUE = r'''
PORT = ""
PORT_FLAGS = {"t06": ("-DOKC_WGSV=1",), "t14": ("-DOKC_WGSV=2",), "t03": ("-DOKC_WGSV=3",)}
_HYBP = {}

def _port_build_all():
 from torch.utils.cpp_extension import load_inline
 def one(n, f7):
  H = _HYBP[n]
  try:
   fl = list(HYB_FLAGS) + (list(HYB_F7) if f7 else []) + ['-DOKC_SC6=1', '-DOKC_MFIN=1', '-DOKC_PLAN=1'] + list(PORT_FLAGS[n])
   tag = hashlib.sha1((_HYB_MOE_CPP + _HYB_MOE_CUDA + " ".join(fl)).encode()).hexdigest()[:12]
   H["mod7" if f7 else "mod"] = load_inline(name="hyb_port_" + tag, cpp_sources=_HYB_MOE_CPP, cuda_sources=_HYB_MOE_CUDA,
    functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
  except Exception as ex:
   H["err"] = repr(ex)[:500]
   print(f"[port] HYB build {n} f7={f7} failed: {ex!r}", flush=True)
 for n in PORT_FLAGS:
  _HYBP[n] = {"mod": None, "mod7": None, "ok": False, "pdl": -1, "pdl7": -1, "w": None, "sc": None, "full": None, "nsm": 0, "err": None}
 ths = [threading.Thread(target=one, args=(n, f7), daemon=True) for n in PORT_FLAGS for f7 in (False, True)]
 for th in ths: th.start()
 for th in ths: th.join()

def _port_prepare(dev):
 _pt1_wait("port")
 if not _HYB["ok"]:
  print("[port] ship HYB bank unavailable; PORT arms off", flush=True)
  return
 for n, H in _HYBP.items():
  H["ok"] = False
  try:
   if H["mod"] is None or H["mod7"] is None:
    raise RuntimeError("missing compiled module: " + str(H["err"]))
   H["w"], H["nsm"] = _HYB["w"], _HYB["nsm"]
   H["sc"], H["full"] = _moexr_scratch(dev, True)
   good, nbit, worst = True, 0, 0.0
   for T in (64, 128, 193, 256, 320):
    for dist in ("normal", "skewed"):
     d = _k2a_self_test_inputs(T, 13000 + T + (dist == "skewed"), dev, dist)
     x, ids, tw = d["hidden_states"].contiguous(), d["topk_ids"].to(torch.int32).contiguous(), d["topk_weights"].float().contiguous()
     ship = _moexr_call_bank(_HYB, x, ids, tw).clone()
     out = _moexr_call_bank(H, x, ids, tw).clone()
     torch.cuda.synchronize()
     rel = _k2a_rel_err(out, ship)[0]
     worst = max(worst, rel)
     nbit += bool(torch.equal(out, ship))
     good &= bool(torch.isfinite(out.float()).all() and rel < 1e-2)
   good &= _moexr_guards(H)
   H["ok"] = bool(good)
   print(f"[port] {n} {'LIVE' if H['ok'] else 'OFF'}: bitwise {nbit}/10 vs ship, worst rel {worst:.2e}", flush=True)
  except Exception as ex:
   H["err"] = repr(ex)[:500]
   print(f"[port] {n} OFF: {H['err']}", flush=True)
'''


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:100])
    return s.replace(a, b, 1)


def main():
    src = BASE.read_text(encoding="utf-8").replace("\r\n", "\n")
    a = src.index('_HYB_MOE_CUDA=r"""'); b = src.index('"""', a + 18)
    h = src[a:b]
    h = once(h, "#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207\n#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174\n", "#ifndef OKC_WGSV\n#define OKC_WGSV 0\n#endif\n#if OKC_WGSV == 1\n#define OKC_WGSA 112, 126, 129, 140, 149, 156, 165, 169\n#define OKC_WGSB 100, 102, 109, 114, 119, 126, 134, 144\n#elif OKC_WGSV == 2\n#define OKC_WGSA 112, 144, 151, 176, 197, 214, 235, 245\n#define OKC_WGSB 100, 106, 121, 132, 145, 162, 178, 204\n#elif OKC_WGSV == 3\n#define OKC_WGSA 112, 119, 120, 126, 130, 134, 138, 140\n#define OKC_WGSB 100, 101, 104, 107, 110, 113, 117, 122\n#else\n#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207\n#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174\n#endif\n")
    src = src[:a] + h + src[b:]
    src = once(src, "def _moexr_bank():\n return _HYB if MOEXR else None\n",
               GLUE + "\ndef _moexr_bank():\n if not MOEXR: return None\n"
               " H = _HYBP.get(PORT) if PORT else None\n return H if (H is not None and H['ok']) else _HYB\n")
    src = once(src, '  jobs.append(("hyb",_hyb_build))\n', '  jobs.append(("hyb",_hyb_build))\n  jobs.append(("port",_port_build_all))\n')
    a = '  _hyblo_prepare(state["k2a33"]["st"],'
    i = src.index(a); j = src.index("\n", i) + 1
    assert src[i:j].rstrip().endswith('weights["w13"].device)')
    src = src[:j] + '  _port_prepare(weights["w13"].device)\n' + src[j:]
    compile(src, str(OUT), "exec")
    OUT.write_bytes(src.encode("ascii"))
    print("HCT lab", OUT, len(src.encode("ascii")), "B")


if __name__ == "__main__":
    main()

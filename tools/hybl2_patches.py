from pathlib import Path
import ast
import hashlib
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/muse_decode_layer/k3_moexr_c1.py'
OUT = ROOT / 'kernels/lab/k3_hybl2_ab.py'

def once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:120])
    return s.replace(old, new, 1)

def cuda_source(text):
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_HYB_MOE_CUDA' for t in n.targets)]
    assert len(nodes) == 1
    return ast.literal_eval(nodes[0].value)

def patch_cuda(src):
    # Reuse the exact MOEX YBF arithmetic, with scope-specific asserted anchors.
    src = once(src, "namespace okc {\n", "#ifndef OKC_YBF\n#define OKC_YBF 0\n#endif\nnamespace okc {\n")
    for start, end, row in (("__device__ __forceinline__ void wgs_epi_tile(", "template <bool PB>\n__device__ __forceinline__ void wgs_flush(", "jst"),
                            ("__device__ __forceinline__ void wgs_epi_tile_c(", "template <bool PB>\n__device__ __forceinline__ void wgs_flush_end(", "c.jst")):
        assert src.count(start) == src.count(end) == 1
        a, b = src.index(start), src.index(end)
        old = f"st_keep_f32(p.ybuf + ((size_t)({row} + tk) * HD + n0), d4[h] * w0);\nst_keep_f32(p.ybuf + ((size_t)({row} + tk) * HD + n0 + 8), d4[2 + h] * w1);\n"
        new = old.replace("st_keep_f32(p.ybuf", "st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf)").replace("d4[h] * w0);", "__float2bfloat16(d4[h] * w0));").replace("d4[2 + h] * w1);", "__float2bfloat16(d4[2 + h] * w1));")
        scope = once(src[a:b], old, "#if OKC_YBF\n" + new + "#else\n" + old + "#endif\n")
        src = src[:a] + scope + src[b:]
    donor = (ROOT / 'vendor_ext/keell0renz/muse_decode_layer.py').read_text(encoding='utf-8').splitlines(keepends=True)
    yload = ''.join(donor[6601:6610])
    assert 'const uint4 yr' in yload and yload.rstrip().endswith('#endif')
    old = 'const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\nconst float4 a = __ldcg(src), b = __ldcg(src + 1);\n'
    for start, end in (("float ps[8];", "ps[sl] = a.x * a.x"), ("const float r = p.topk_w[tk * 8 + sl]", "acc[0] += r * a.x;")):
        assert src.count(start) == src.count(end) == 1
        a, b = src.index(start), src.index(end)
        src = src[:a] + once(src[a:b], old, yload) + src[b:]
    anchor = '__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {\n'
    helper = r'''#ifndef OKC_L2RA
#define OKC_L2RA 0
#endif
#ifndef OKC_L2RAEF
#define OKC_L2RAEF 0
#endif
#ifndef OKC_HYBL2_DIET
#define OKC_HYBL2_DIET 0
#endif
__device__ __forceinline__ void hybl2_pf(const void* a, uint32_t n) {
#if OKC_L2RAEF
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;\n" :: "l"(a), "r"(n), "l"(0x12F0000000000000ull) : "memory");
#else
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(a), "r"(n) : "memory");
#endif
}
'''
    src = once(src, anchor, helper + anchor)
    anchor = 'uint32_t wph = 0;\n#pragma unroll 1\nfor (int pb = 0; pb < 2; ++pb) {\n'
    insert = r'''uint32_t wph = 0;
#if OKC_L2RA
const int nA0 = S.wgr[1] - S.wgr[0];
const int nTot = nA0 + S.wgr[3] - S.wgr[2];
auto wsrc = [&](int m) -> const unsigned char* {
  const int pb = m >= nA0, i = pb ? m - nA0 : m;
  const int g0 = S.wgr[2*pb], g1 = S.wgr[2*pb+1], ubl = pb ? 4 : 5;
  int gs = wgs_split(pb, g0, g1) + i; if (gs >= g1) gs -= g1 - g0;
  const int u = gs >> 4, j = u >> ubl, ub = u & ((1 << ubl) - 1);
  return (pb ? p.wBs : p.wAs) + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + (gs & 15))) * (size_t)WGS_COPY_BYTES;
};
#endif
#pragma unroll 1
for (int pb = 0; pb < 2; ++pb) {
'''
    src = once(src, anchor, insert)
    anchor = 'if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);\n'
    src = once(src, anchor, r'''#if OKC_L2RA
for (int m = n ? n + WGS_NS + OKC_L2RA - 1 : WGS_NS; m < n + WGS_NS + OKC_L2RA && m < nTot; ++m)
  hybl2_pf(wsrc(m), WGS_COPY_BYTES);
#endif
''' + anchor)
    anchor = 'while (ld_acquire(cp) < target && n < 4000000) ++n;\n'
    src = once(src, anchor, '#if OKC_HYBL2_DIET\nwhile (ld_acquire(cp) < target && n < 4000000) { ++n; __nanosleep(32); }\n#else\n' + anchor + '#endif\n')
    anchor = '''// MOEX:MFIN
#if OKC_MFIN
const int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;
#else
const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);
#endif
S.gen = gen;
S.ssp = (gen & 1) ? p.ss1 : p.ss;
}
for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;
'''
    moved = anchor.replace('// MOEX:MFIN\n', '#if !OKC_HYBL2_DIET\n// MOEX:MFIN\n', 1).replace('S.ssp = (gen & 1) ? p.ss1 : p.ss;\n}', 'S.ssp = (gen & 1) ? p.ss1 : p.ss;\n#endif\n}\n#if OKC_HYBL2_DIET\nif (tid == 32) {\nconst int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;\nS.gen = gen;\nS.ssp = (gen & 1) ? p.ss1 : p.ss;\n}\n#endif', 1)
    src = once(src, anchor, moved)
    return src

GLUE = r'''
HYBL2 = 0
_HYBL2 = {i: {"mod": None, "mod7": None, "ok": False, "pdl": -1, "pdl7": -1, "w": None, "sc": None, "full": None, "nsm": 0, "err": None} for i in (1, 2)}

def _hybl2_build():
 from torch.utils.cpp_extension import load_inline
 def one(i, f7):
  H = _HYBL2[i]
  try:
   flags = list(HYB_FLAGS) + (list(HYB_F7) if f7 else (["-DOKC_HYBST=4"] if i == 2 else []))
   flags += ["-DOKC_SC6=1", "-DOKC_MFIN=1", "-DOKC_PLAN=1", "-DOKC_L2RA=4", "-DOKC_L2RAEF=1", "-DOKC_HYBL2_DIET=%d" % (i == 2), "-DOKC_YBF=%d" % (i == 2)]
   tag = hashlib.sha1((_HYB_MOE_CPP + _HYBL2_CUDA + " ".join(flags)).encode()).hexdigest()[:12]
   H["mod7" if f7 else "mod"] = load_inline(name="hyb_l2_" + tag, cpp_sources=_HYB_MOE_CPP, cuda_sources=_HYBL2_CUDA, functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=flags, verbose=False)
  except Exception as ex:
   H["err"] = repr(ex)[:500]
   print(f"[hybl2] build arm {i} f7={f7} failed: {ex!r}", flush=True)
 ths = [threading.Thread(target=one, args=(i, f7), daemon=True) for i in (1, 2) for f7 in (False, True)]
 for th in ths: th.start()
 for th in ths: th.join()

def _hybl2_prepare(st, w, dev):
 _pt1_wait("hybl2")
 if not _PT1["threads"].get("hybl2") and all(H["mod"] is None and H["mod7"] is None for H in _HYBL2.values()):
  _hybl2_build()
 for H in _HYBL2.values(): H["ok"] = False
 if not _HYB["ok"]:
  print("[hybl2] ship bank unavailable; lab arms disabled", flush=True)
  return
 for arm in (1, 2):
  H = _HYBL2[arm]
  H["ok"] = False
  try:
   if H["mod"] is None or H["mod7"] is None:
    raise RuntimeError("missing compiled module: " + str(H["err"]))
   H["w"], H["nsm"] = _HYB["w"], _HYB["nsm"]
   H["sc"], H["full"] = _moexr_scratch(dev, True)
   good = True
   for T in (64, 128, 193, 256, 320):
    for dist in ("normal", "skewed"):
     d = _k2a_self_test_inputs(T, 12000 + T + (dist == "skewed"), dev, dist)
     d = {"hidden_states": d["hidden_states"].contiguous(), "topk_ids": d["topk_ids"].to(torch.int32).contiguous(), "topk_weights": d["topk_weights"].float().contiguous(), "config": {"local_start": 0, "local_count": K2A_E_LOC}}
     x, ids, tw = d["hidden_states"], d["topk_ids"], d["topk_weights"]
     ref = _k2a_ref_fp32(d, w)
     ship = _moexr_call_bank(_HYB, x, ids, tw).clone()
     out = _moexr_call_bank(H, x, ids, tw).clone()
     out2 = _moexr_call_bank(H, x, ids, tw).clone()
     torch.cuda.synchronize()
     total, row = _k2a_rel_err(out, ref)
     rel, rowship = _k2a_rel_err(out, ship)
     repeat = _k2a_rel_err(out2, out)[0]
     finite = bool(torch.isfinite(out.float()).all() and torch.isfinite(out2.float()).all())
     ok = bool(finite and total < 1e-2 and row < 5e-2 and rel < 1e-2 and repeat < 1e-3)
     print(f"[hybl2] arm={arm} T={T} {dist} fp32={total:.3e} row={row:.3e} ship={rel:.3e} ship_row={rowship:.3e} repeat={repeat:.3e} -> {'ok' if ok else 'FAIL'}", flush=True)
     good &= ok
   good &= _moexr_guards(H) and int(H["sc"][0][8+1024+_HYB_EMAX+65]) == 0
   H["ok"] = bool(good)
   if not good: H["err"] = "numerical, finite, guard, or MF_ERR check failed"
  except Exception as ex:
   H["err"] = repr(ex)[:500]
  print(f"[hybl2] arm {arm} {'LIVE' if H['ok'] else 'OFF; fallback to ship'}: {H['err']}", flush=True)
'''

def main():
    source = BASE.read_text(encoding='utf-8')
    cu = patch_cuda(cuda_source(source))
    # Keep the base CUDA literal and therefore the original arm binary untouched.
    source = once(source, 'def _moexr_bank():\n', '_HYBL2_CUDA = ' + repr(cu) + '\n' + GLUE + '\ndef _moexr_bank():\n')
    source = once(source, ' return _HYB if MOEXR else None\n', ' if not MOEXR: return None\n H = _HYBL2.get(HYBL2)\n return H if H is not None and H["ok"] else _HYB\n')
    source = once(source, ' _moexr_prepare_bank(st,w,dev,_HYB)\n', ' _moexr_prepare_bank(st,w,dev,_HYB)\n _hybl2_prepare(st,w,dev)\n')
    source = once(source, '  jobs.append(("hyb",_hyb_build))\n', '  jobs.append(("hyb",_hyb_build))\n  jobs.append(("hybl2",_hybl2_build))\n')
    if "--check" in sys.argv:
        assert OUT.read_bytes() == source.encode("utf-8"), "generated lab is stale"
    else:
        OUT.write_bytes(source.encode("utf-8"))
    print('HYBL2 source patch generated:', hashlib.sha256(source.encode()).hexdigest())

if __name__ == '__main__':
    main()
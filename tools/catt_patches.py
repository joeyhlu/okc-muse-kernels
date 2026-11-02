"""CATT in-file HYA bank; only the named lab output is written.

python -B -I tools/catt_patches.py
python -B -I tools/catt_patches.py --check
python -B -I tools/catt_patches.py --variants [--correctness]
No GPU work, imports of the layer, bytecode writes, or external generator dependencies.
"""
import argparse
import ast
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexr_c1.py"
OUT = ROOT / "kernels/lab/k3_catt_ab.py"
KEYS = (1, 2, 3, 9, 10, 11)
COEFF = (0.9999280735393952, 0.6932609854606236,
         0.24261112219321848, 0.05517166905803779)


def sub(s, old, new):
    assert s.count(old) == 1, ("anchor must occur exactly once", old[:100], s.count(old))
    return s.replace(old, new, 1)


def literal(s, name):
    nodes = [n for n in ast.parse(s).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, name
    return ast.literal_eval(nodes[0].value)


HELPER = r'''// CATT C1: relative minimax cubic, five equioscillating extrema.
// Max real relative error 7.478144e-5 on [-.5,.5]; CPU gate includes FP32 rounding.
// Upper clipping is above E4M3 saturation and preserves pack2's overflow repair.
__device__ __forceinline__ float catt_ex2p(float x) {
const float z = fminf(fmaxf(x, -126.f), 126.f);
const float t = __fadd_rn(z, 12582912.f);
const float r = __fsub_rn(t, 12582912.f);
const float f = __fsub_rn(z, r);
const float p = fmaf(fmaf(fmaf(0.05517166905803779f, f, 0.24261112219321848f),
                        f, 0.6932609854606236f), f, 0.9999280735393952f);
const unsigned bits = 0x3f800000u + (__float_as_uint(t) << 23);
// Multiply rather than adding exponent bits to p: correct at the normal/subnormal boundary.
return x == -INFINITY ? 0.f : __fmul_rn(p, __uint_as_float(bits));
}
'''
PROBE = r'''// CATT C0: TIMING ONLY, INVALID OUTPUT, NEVER SHIP.
__device__ __forceinline__ float catt_ex2_null(float x) {
// Keep the requested FFMA even if nvcc can fold a finite x*0+1.
float y;
asm volatile("fma.rn.f32 %0, %1, 0f00000000, 0f3f800000;" : "=f"(y) : "f"(x));
return y;
}
'''
TILE_END = '''float* od = &o[0][0][0];
wgmma_wait<0>();
#pragma unroll
for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
PROF_TILE;'''
K_LOAD_END = '''kw[nt][0] = kb.x; kw[nt][1] = kb.y; kw[nt][2] = kb.z; kw[nt][3] = kb.w;
}
#pragma unroll
for (int nt = 0; nt < 4; ++nt) {
float se[4] = {0.f, 0.f, 0.f, 0.f}, so[4] = {0.f, 0.f, 0.f, 0.f};'''
V_LOOP = '''#pragma unroll
for (int ks = 0; ks < 2; ++ks) {
uint32_t b0[8], b1[8];
load_v_frags(V, 32 * wsub + 16 * ks, hk, g, tig, b0, b1);
const uint32_t a0 = pack_h2(s[2 * ks][0], s[2 * ks][1]);
const uint32_t a2 = pack_h2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
#pragma unroll
for (int t = 0; t < 8; ++t) mma_f16(o[0][t], a0, 0u, a2, 0u, b0[t], b1[t]);
}'''
L1_END = '''if constexpr (WS) {
__syncwarp();
mbar_arrive(&S.empty[(tb + i) % STAGES]);
}
PROF_TILE;'''


def cuda_source(ship, key):
    assert key in (0,) + KEYS
    s = ship
    if key in (1, 2, 9):
        a, b = s.index("auto pack2 = [&]() {"), s.index("\n};\npack2();")
        block = s[a:b]
        assert block.count("ex2(") == 16
        if key == 9:
            new = block.replace("ex2(", "catt_ex2_null(")
        else:
            # Every packed word: positions 0/2 use MUFU; positions 1/3 use the cubic.
            new, count = re.subn(r"ex2\(fmaf\(sv\[(1|3|5|7|9|11|13|15)\]",
                                lambda m: m[0].replace("ex2(", "catt_ex2p("), block)
            assert count == 8
        s = sub(s, block, new)
        anchor = "template <int L, bool F>\nstruct Cfg {"
        s = sub(s, anchor, (PROBE if key == 9 else HELPER) + anchor)
    if key == 2:
        s = sub(s, TILE_END, "// CATT C3: FLAT retires PV at next QK wait; final drain stays.\n"
                "if constexpr (!FLAT) {\n" + TILE_END.removesuffix("PROF_TILE;") + "}\nPROF_TILE;")
    if key == 3:
        s = sub(s, "static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;",
                "static constexpr int STAGES = F ? ((L == 1) ? 5 : 4) : 3; // CATT C2")
        early = '''// CATT C2: retain both V fragments before releasing the K/V slot.
uint32_t catt_b0[2][8], catt_b1[2][8];
if constexpr (WS) {
load_v_frags(V, 32 * wsub, hk, g, tig, catt_b0[0], catt_b1[0]);
load_v_frags(V, 32 * wsub + 16, hk, g, tig, catt_b0[1], catt_b1[1]);
__syncwarp();
mbar_arrive(&S.empty[(tb + i) % STAGES]);
}
'''
        s = sub(s, K_LOAD_END, K_LOAD_END.replace("}\n#pragma unroll", "}\n" + early + "#pragma unroll", 1))
        use = V_LOOP.replace("uint32_t b0[8], b1[8];\nload_v_frags(V, 32 * wsub + 16 * ks, hk, g, tig, b0, b1);\n", "")
        use = use.replace("b0[t], b1[t]", "catt_b0[ks][t], catt_b1[ks][t]")
        s = sub(s, V_LOOP, "if constexpr (WS) {\n" + use + "\n} else {\n" + V_LOOP + "\n}")
        s = sub(s, L1_END, "// CATT C2: the WS slot was released after its final LDS.\nPROF_TILE;")
    return s


# Padz pattern: source edit tuples, independent build threads, per-arm ok/error,
# prepare-time validation, then call/capture-time host selection of a compiled module.
LAB = '''
# CATT lab only: ATT_V 0 ship, 1 C1, 2 C1+C3, 3 C2, 9 INVALID C0.
ATT_V = 0
ATT_TWIN = 0
ATT_ALLOW_C0 = False
_CATT_KEYS = (1, 2, 3, 9, 10, 11)
_CATT = {k: {"mod": None, "ok": False, "err": None, "attempted": False, "build_s": 0.0,
             "test_s": 0.0, "metrics": {}} for k in _CATT_KEYS}
_CATT_TEST = 0
_CATT_METRICS = {}
_CATT_BUILD_S = 0.0
_CATT_READY = False
# The base graph suite has no flat L8 case. Exercise C1/C3 in capture too.
_CATT_GRAPH_CASES = ((16,1500,8,0,1141), (32,800,8,0,1142))

def _catt_src(k):
    s = HYA_CUDA_SRC
    for a, b in _CATT_EDITS[k]:
        assert s.count(a) == 1, a[:80]
        s = s.replace(a, b, 1)
    return s

def _catt_compile(k, src):
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((HYA_CPP_SRC + src).encode()).hexdigest()[:12]
    # Key in the name is essential: mid and ship2 really are separate compiled twins.
    return load_inline(name=f"catt_hya_{k}_{tag}", cpp_sources=HYA_CPP_SRC, cuda_sources=src,
                       functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a",
                       "--expt-relaxed-constexpr"], verbose=False)

def _catt_build():
    global _CATT_BUILD_S
    _pt1_wait("catt")
    begin = time.time()
    def one(k):
        H = _CATT[k]
        if H["attempted"]:
            return
        H["attempted"] = True
        t = time.time()
        try:
            H["mod"] = _catt_compile(k, _catt_src(k))
            H["err"] = None
        except Exception as ex:
            H["err"] = repr(ex)[:600]
        finally:
            H["build_s"] = time.time() - t
    ths = [threading.Thread(target=one, args=(k,)) for k in _CATT_KEYS]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    _CATT_BUILD_S += time.time() - begin

def _catt_module():
    # No compilation, allocation of CUDA state, or handle mutation during capture.
    if _CATT_TEST:
        return _CATT[_CATT_TEST]["mod"]
    if not _CATT_READY:
        return _HYA["mod"]
    if ATT_V not in (0, 1, 2, 3, 9) or ATT_TWIN not in (0, 1, 2):
        raise ValueError("CATT selector out of range")
    k = (9 + ATT_TWIN if ATT_TWIN else 0) if ATT_V == 0 else ATT_V
    if k == 9 and not ATT_ALLOW_C0:
        raise RuntimeError("C0 requires ATT_ALLOW_C0=True: INVALID OUTPUT, TIMING ONLY")
    H = _CATT.get(k)
    return H["mod"] if H is not None and H["ok"] else _HYA["mod"]

def _catt_record(kind, key, metrics, ok):
    tag = (kind, tuple(key))
    bank = _CATT[_CATT_TEST]["metrics"] if _CATT_TEST else _CATT_METRICS
    bank[tag] = dict(metrics)
    if _CATT_TEST:
        ship = _CATT_METRICS[tag]  # A missing baseline is a validation failure.
        print(f"[catt] {_CATT_TEST} {kind} {key}: ship={ship} variant={metrics} {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise RuntimeError(f"CATT base tolerance failed: {kind} {key} {metrics}")

def _catt_warm_probe(dev):
    # Eager initialization of launch<1/8>'s static CUDA attributes, never a correctness test.
    for L in (1, 8):
        B = 16
        dk = _k1fa4_case(B,1500,L,0,1143+L,False,dev)
        qkv = dk["qkv"].view(B*L,-1).contiguous()
        cfg = dict(dk["config"],tokens=B*L)
        _hya_launch(qkv,dk,cfg,False)
    torch.cuda.synchronize(dev)

def _catt_prepare(state):
    global _CATT_TEST, _CATT_READY
    _CATT_READY = False
    _pt1_wait("catt")
    _catt_build()  # Also supports PT1_PREFETCH=False, like the padz build fallback.
    dev = state["router"].device
    begin = time.time()
    if not state.get("hya"):
        for H in _CATT.values():
            H["ok"] = False
            H["err"] = "ship HYA unavailable"
        print("[catt] OFF: ship HYA unavailable", flush=True)
        return
    saved_kvs = _HYA["kvs"]
    for k, H in _CATT.items():
        H["ok"] = False
        H["metrics"].clear()
        if H["mod"] is None:
            print(f"[catt] {k} OFF, fallback ship: build {H['err']}", flush=True)
            continue
        if k == 9:
            try:
                _CATT_TEST = 9
                _catt_warm_probe(dev)
                H["ok"] = True
                H["err"] = None
            except Exception as ex:
                H["err"] = repr(ex)[:600]
            finally:
                _CATT_TEST = 0
                for x in _HYA["ws"].get(str(dev), ()):
                    x.zero_()
            print(f"[catt] 9 {'READY' if H['ok'] else 'OFF, fallback ship'} TIMING ONLY: "
                  f"excluded from correctness, NEVER SHIP, reason={H['err']}", flush=True)
            continue
        t = time.time()
        try:
            _CATT_TEST = k
            _hya_self_test(dev)  # Verbatim base self/graph/kv-scale tests, instrumented below.
            if HYA_KVS and not _HYA["kvs"]:
                raise RuntimeError("base HYA_KVS_CASES failed")
            if H["metrics"].keys() != _CATT_METRICS.keys():
                raise RuntimeError("incomplete base self-test coverage")
            H["ok"] = True
            H["err"] = None
        except Exception as ex:
            H["err"] = repr(ex)[:600]
            for x in _HYA["ws"].get(str(dev), ()):
                x.zero_()
        finally:
            _CATT_TEST = 0
            _HYA["kvs"] = saved_kvs
            H["test_s"] = time.time() - t
        print(f"[catt] {k} {'LIVE' if H['ok'] else 'OFF, fallback ship'} build={H['build_s']:.3f}s "
              f"test={H['test_s']:.3f}s reason={H['err']}", flush=True)
    print(f"[catt] extra_build_count={len(_CATT_KEYS)} parallel_build_wall_s={_CATT_BUILD_S:.3f} "
          f"validation_wall_s={time.time()-begin:.3f} build_s="
          f"{[(k,H['build_s']) for k,H in _CATT.items()]}", flush=True)
    _CATT_READY = True
'''


def edit_pairs(ship, changed):
    # Preserve line-local anchors, rather than embedding six copies of HYA.
    import difflib
    a, b = ship.splitlines(True), changed.splitlines(True)
    edits = []
    for op, i, j, u, v in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        # Surround with enough unchanged context for a unique anchor.
        lo, hi, left, right = i, j, u, v
        while not a[lo:hi] or ship.count("".join(a[lo:hi])) != 1:
            if lo > 0 and left > 0 and a[lo-1] == b[left-1]:
                lo -= 1; left -= 1
            elif hi < len(a) and right < len(b) and a[hi] == b[right]:
                hi += 1; right += 1
            else:
                raise AssertionError("cannot make unique diff anchor")
        edits.append(("".join(a[lo:hi]), "".join(b[left:right])))
    s = ship
    for old, new in edits:
        s = sub(s, old, new)
    assert s == changed
    return edits


def generate(base):
    ship = literal(base, "HYA_CUDA_SRC")
    edits = {k: edit_pairs(ship, cuda_source(ship, k)) for k in KEYS}
    s = sub(base, 'def _hya_build():\n', LAB + '\n_CATT_EDITS = ' + repr(edits) + '\ndef _hya_build():\n')
    s = sub(s, '  jobs.append(("hya",_hya_build))\n',
            '  jobs.append(("hya",_hya_build))\n  jobs.append(("catt",_catt_build))\n')
    s = sub(s, ' _rq_prepare(state)\n', ' _rq_prepare(state)\n _catt_prepare(state)\n')
    s = sub(s, 'def _hya_prepare(state,t0):\n',
            'def _hya_prepare(state,t0):\n global _CATT_READY\n _CATT_READY=False\n _CATT_METRICS.clear()\n')
    s = sub(s, ' _HYA["mod"].run(qkv,data["seq_lens"]', ' _catt_module().run(qkv,data["seq_lens"]')
    s = sub(s, '  if not ok and HYA_SELF_TEST_STRICT:\n',
            '  _catt_record("self",(B,ctx,L,win,seed,row0,P),dict(rel=ea,pdl=eab,kv=ekv,clean=clean),ok)\n'
            '  if not ok and HYA_SELF_TEST_STRICT:\n')
    s = sub(s, '  if not e<=2e-3 and HYA_SELF_TEST_STRICT:\n',
            '  _catt_record("graph",(B,ctx,L,win,seed),dict(rel=e),e<=2e-3)\n'
            '  if not e<=2e-3 and HYA_SELF_TEST_STRICT:\n')
    s = sub(s, ' for B,ctx,L,win,seed in HYA_GRAPH_CASES:\n',
            ' for B,ctx,L,win,seed in HYA_GRAPH_CASES+_CATT_GRAPH_CASES:\n')
    s = sub(s, '   good&=ok\n   del ref,dk,oa\n',
            '   _catt_record("kvs",(B,ctx,L,win,seed),dict(rel=ea,clean=clean),ok)\n'
            '   good&=ok\n   del ref,dk,oa\n')
    compile(s, str(OUT), "exec")
    assert literal(s, "HYA_CUDA_SRC") == ship
    return s


def variants(correctness=False):
    arms = [{"name": n, "set": {"ATT_V": v, "ATT_TWIN": t, "ATT_ALLOW_C0": v == 9}}
            for n, v, t in (("ship", 0, 0), ("c1", 1, 0), ("c13", 2, 0), ("c2", 3, 0),
                            ("mid", 0, 1), ("c0", 9, 0), ("ship2", 0, 2))]
    return [a for a in arms if not correctness or a["set"]["ATT_V"] != 9]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--correctness", action="store_true")
    args = ap.parse_args()
    if args.variants:
        print(json.dumps(variants(args.correctness)))
        return
    base = BASE.read_text(encoding="utf-8")
    s = generate(base)
    if args.check:
        assert OUT.read_bytes() == s.encode(), "generated lab is stale"
    else:
        OUT.write_bytes(s.encode())
    print(json.dumps({"generator": "PASS", "bytes": len(s.encode()),
                      "sha256": hashlib.sha256(s.encode()).hexdigest(), "extra_build_count": len(KEYS)}))


if __name__ == "__main__":
    main()

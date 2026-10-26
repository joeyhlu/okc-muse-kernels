"""dg1_patches.py: DG1 = swap-AB wgmma dense projections (qkv, o, cat [pooler; router hi; lo] FP32-out, unpooler) at T = 32 / 128 /
256, replacing today's cuBLAS calls where routed. Exact text patches against kernels/muse_decode_layer/k3_v33.py; the CUDA source,
its generator and the weight packer are copied verbatim from kernels/lab/dg1_lab.py (DG1_CPP_SRC .. dg1_cuda_src, dg1_pack).

Kernel (see the lab's docstring): y^T = W . x^T with wgmma m64nTk16 (A = prepacked weight rows, one contiguous cp.async.bulk per
BM x 64 tile, L2 evict_first; B = x via a per-call TMA tensor map, 128B swizzle), 1 producer warp + NWG consumer warpgroups, a
ring of ST stages, split-K across a (S, 1, 1) thread-block cluster reduced in-kernel through DSMEM in rank order (deterministic,
no workspace / atomics / 2nd kernel). PDL: weights of the first ST stages are requested before griddepcontrol.wait, x and every
global store after it; launch_dependents at entry on one-wave grids. Bounded mbarrier waits (err flag, no trap).

Layer glue: DG1_ENABLE / DG1_ROUTE {(projection, bucket): True} read at call time (A/B variants switch routes off without a new
prepare); DG1_GEOMETRY {(projection, bucket): (T, NWG, MT, S)}; every DG1_ROUTE key is built (DG1_FOLD_KS1: the routed
instantiations ride in KS1's load_inline, +~1 s; else DG1's own small build, +~38 s cold) and self-tested in prepare: random x at T = bucket and bucket - 3 vs FP64 (rel-L2 < 4e-3 bf16 /
1e-4 fp32 out) and vs today's torch.mm (< 1e-2), NaN-prefilled output fully written, repeat bit-identical, no timeout flag. A
failed self-test turns that route off; a host exception at call time disables DG1 for the run (_k3_disable).

    python -I tools/dg1_patches.py [k3_v33.py] [out.py]
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v33.py"
LAB = ROOT / "kernels/lab/dg1_lab.py"

# Routed (projection, bucket) -> geometry (T, NWG, MT, S). Standalone DG1 missed the x1.10 lab gate (it only matches cuBLAS), but
# in-layer at T = 32 the PDL weight prefetch wins (box_out/dg1/ab1.log, layer_ab2 shapes 4,5 x 3 seeds, checks pass): decode-32
# load x1.0077 (cat) / x1.0099 (cat + qkv); sliding x1.012 / x1.018, full x1.002 / x0.998. T = 128 / 256 lose -> not routed.
ROUTE = {("cat", 32): True, ("qkv", 32): True}
GEOMETRY = {("cat", 32): (32, 2, 1, 3), ("qkv", 32): (32, 2, 1, 4)}


def _lab_block():
    """DG1_CPP_SRC .. dg1_cuda_src and dg1_pack, verbatim from the lab (dg1_build and the lab harness stay there)."""
    lab = LAB.read_text(encoding="utf-8")
    a, b = lab.index('DG1_CPP_SRC = r"""'), lab.index("def dg1_build(")
    c, d = lab.index("def dg1_pack(w):"), lab.index("# ---------------------------------------------------------------- lab")
    insts_line = next(ln for ln in lab.splitlines() if ln.startswith("DG1_INSTS = "))
    return insts_line + "\n\n" + lab[a:b] + lab[c:d].rstrip() + "\n"


def layer_py():
    return (
        "# ---------------------------------------------------------------- DG1 (tools/dg1_patches.py): swap-AB wgmma dense projections, T = 32 / 128 / 256\n"
        "DG1_ENABLE = True   # master switch (read at call time; False also skips the DG1 build in prepare)\n"
        f"DG1_ROUTE = {ROUTE!r}   # (projection, bucket) -> True = routed (read at call time); every key is built + self-tested in prepare\n"
        f"DG1_GEOMETRY = {GEOMETRY!r}   # (projection, bucket) -> (T, NWG, MT, S): kernels/lab/dg1_lab.py sweep winners\n"
        "DG1_PDL = True    # programmatic stream serialization (weights before griddepcontrol.wait; read at call time)\n"
        "DG1_TRIG = True   # griddepcontrol.launch_dependents at entry on one-wave grids (read at call time)\n"
        "DG1_TOL = (4e-3, 1e-4, 1e-2)   # self-test rel-L2: vs FP64 (bf16 out, fp32 out), vs today's torch.mm\n"
        "\n"
        + _lab_block() +
        "\n\n"
        '_DG1 = {"mod": None, "insts": None, "build_s": 0.0}\n'
        "\n\n"
        "def _dg1_build(insts):\n"
        '    """DG1\'s own small load_inline holding only the routed (T, NWG, MT) instantiations (cached). Prepare-time only."""\n'
        '    if _DG1["mod"] is not None and _DG1["insts"] == insts:\n'
        '        return _DG1["mod"]\n'
        "    from torch.utils.cpp_extension import load_inline\n"
        "    cuda = dg1_cuda_src(list(insts))\n"
        "    tag = hashlib.sha1((DG1_CPP_SRC + cuda).encode()).hexdigest()[:10]\n"
        "    t0 = time.time()\n"
        '    _DG1["mod"] = load_inline(name=f"dg1_{tag}", cpp_sources=DG1_CPP_SRC, cuda_sources=cuda, functions=["dg1_run", "dg1_info"],\n'
        '                              extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)\n'
        '    _DG1["insts"], _DG1["build_s"] = insts, time.time() - t0\n'
        '    return _DG1["mod"]\n'
        "\n\n"
        "def _dg1_weight(state, proj):\n"
        '    if proj == "cat":\n'
        '        return state["v8"]["wcat"] if state.get("v8") is not None else None\n'
        '    return state["dense"]["qkv_gate" if proj == "qkv" else proj]\n'
        "\n\n"
        "def _dg1(state, proj, x):\n"
        '    """DG1 for (projection, T = x.shape[0]) when routed and self-tested, else None -> the caller keeps today\'s path. proj "cat" =\n'
        "    [pooler; router hi; router lo] with FP32 out. Shape-only branches (graph safe); a host exception disables DG1 for the rest of\n"
        '    the run (once, with a line), as P1M / K1 / K2A do."""\n'
        '    st = state.get("dg1")\n'
        "    if not DG1_ENABLE or st is None:\n"
        "        return None\n"
        "    T = x.shape[0]\n"
        "    if T <= 16:\n"
        "        return None\n"
        "    key = (proj, _k2a_bucket_of(T))\n"
        "    if not DG1_ROUTE.get(key):   # call-time filter: A/B variants switch routes off without a new prepare\n"
        "        return None\n"
        '    r = st["route"].get(key)\n'
        '    if (r is None or T > r["T"] or x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != r["K"] or not x.is_contiguous()\n'
        "            or x.data_ptr() % 16 != 0):\n"
        "        return None\n"
        "    try:\n"
        '        out = torch.empty((T, r["N"]), dtype=torch.float32 if proj == "cat" else torch.bfloat16, device=x.device)\n'
        '        st["mod"].dg1_run(r["kid"], x, st["w"][proj], out, st["err"], r["S"], 1 if DG1_PDL else 0,\n'
        '                          1 if (DG1_TRIG and r["wave1"]) else 0, r["N"], 0)\n'
        "        return out\n"
        "    except Exception as ex:  # noqa: BLE001\n"
        '        _k3_disable(state, "dg1", None, "DG1", ex)\n'
        "        return None\n"
        "\n\n"
        "def _dg1_selftest(st, r, proj, w, T, gen):\n"
        '    """(ok, message) for one route at T and T - 3 tokens: random x vs FP64 and vs today\'s torch.mm, NaN-prefilled output fully\n'
        '    written, repeat bit-identical, timeout flag clear. Prepare-time only."""\n'
        '    f32 = proj == "cat"\n'
        "    dev = w.device\n"
        "    ok, parts = True, []\n"
        "    for t in (T, T - 3):\n"
        '        x = torch.randn((t, r["K"]), generator=gen, device=dev).to(torch.bfloat16)\n'
        "        outs = []\n"
        "        for _ in range(2):\n"
        '            out = torch.full((t, r["N"]), float("nan"), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)\n'
        '            st["mod"].dg1_run(r["kid"], x, st["w"][proj], out, st["err"], r["S"], 1, 0, r["N"], 0)\n'
        "            outs.append(out)\n"
        "        ref64 = x.double() @ w.double().t()\n"
        "        today = torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())\n"
        "        torch.cuda.synchronize()\n"
        "        got = outs[0]\n"
        "        fin = bool(torch.isfinite(got).all())\n"
        "        e64 = float((got.double() - ref64).norm() / ref64.norm()) if fin else float('inf')\n"
        "        etoday = float((got.float() - today.float()).norm() / today.float().norm()) if fin else float('inf')\n"
        "        same = bool(torch.equal(outs[0], outs[1]))\n"
        '        errf = int(st["err"][0])\n'
        "        good = fin and same and errf == 0 and e64 < DG1_TOL[1 if f32 else 0] and etoday < DG1_TOL[2]\n"
        "        ok &= good\n"
        '        parts.append(f"T={t}: rel vs fp64 {e64:.2e} vs today {etoday:.2e} finite {fin} repeat {same} err {errf}")\n'
        '        st["err"].zero_()\n'
        '    return ok, "; ".join(parts)\n'
        "\n\n"
        "def _dg1_prepare(state, t0):\n"
        '    """Build (routed instantiations only), weight packs and per-route self-tests. state["dg1"] stays None (today\'s path) when off,\n'
        '    the build fails or nothing passes."""\n'
        '    state["dg1"] = None\n'
        "    keys = sorted(k for k in DG1_ROUTE if k in DG1_GEOMETRY)\n"
        "    if not DG1_ENABLE or not keys:\n"
        "        return None\n"
        "    t1 = time.time()\n"
        "    insts = tuple(sorted({DG1_GEOMETRY[k][:3] for k in keys}))\n"
        "    try:\n"
        '        ks = _KS1.get("mod")\n'
        '        folded = ks is not None and _KS1.get("dg1_insts") == insts and hasattr(ks, "dg1_run")\n'
        "        mod = ks if folded else _dg1_build(insts)   # folded: built inside KS1's extension (DG1_FOLD_KS1)\n"
        "    except Exception as ex:  # noqa: BLE001\n"
        '        print(f"[dg1] build FAILED -> today\'s path: {ex!r}", flush=True)\n'
        "        return None\n"
        '    dev = state["dense"]["qkv_gate"].device\n'
        '    st = {"mod": mod, "w": {}, "route": {}, "err": torch.zeros(4, dtype=torch.int32, device=dev)}\n'
        "    gen = torch.Generator(device=dev).manual_seed(4242)\n"
        "    for key in keys:\n"
        "        proj, rep = key\n"
        "        T, nwg, mt, s = DG1_GEOMETRY[key]\n"
        "        w = _dg1_weight(state, proj)\n"
        "        if w is None:\n"
        '            print(f"[dg1] {key}: no weight (v8 tail off) -> route off", flush=True)\n'
        "            continue\n"
        "        try:\n"
        '            if proj not in st["w"]:\n'
        '                st["w"][proj] = dg1_pack(w)\n'
        "            kid = insts.index((T, nwg, mt))\n"
        "            info = [int(v) for v in mod.dg1_info(kid, s)]\n"
        "            n, k = w.shape\n"
        "            if n % info[3] or rep > T or k % 64:\n"
        '                print(f"[dg1] {key}: geometry {DG1_GEOMETRY[key]} does not fit W {tuple(w.shape)} -> route off", flush=True)\n'
        "                continue\n"
        '            r = {"kid": kid, "S": s, "T": T, "N": n, "K": k, "wave1": info[0] >= n // info[3]}\n'
        "            ok, msg = _dg1_selftest(st, r, proj, w, rep, gen)\n"
        '            print(f"[dg1] self-test {proj} bucket {rep} geo {DG1_GEOMETRY[key]} (BM {info[3]} ST {info[4]} regs {info[5]} clusters {info[0]}"\n'
        "                  f\"{' one wave' if r['wave1'] else ''}): {msg} -> {'ok' if ok else 'FAIL: route off'}\", flush=True)\n"
        "            if ok:\n"
        '                st["route"][key] = r\n'
        "        except Exception as ex:  # noqa: BLE001\n"
        '            print(f"[dg1] {key}: self-test raised {ex!r} -> route off", flush=True)\n'
        '    for proj in list(st["w"]):\n'
        '        if not any(k[0] == proj for k in st["route"]):\n'
        '            del st["w"][proj]\n'
        '    state["dg1"] = st if st["route"] else None\n'
        '    print(f"[dg1] routes live {sorted(st[\'route\'])} (build {_DG1[\'build_s\']:.1f}s, total {time.time() - t1:.1f}s; prepare so far"\n'
        "          f\" {time.time() - t0:.1f}s)\", flush=True)\n"
        '    return state["dg1"]\n'
        "\n\n"
    )


FOLD_PY = (
    "DG1_FOLD_KS1 = True   # DG1's kernels ride in KS1's extension (no 2nd torch-header compile: -30 s cold prepare); else own build\n"
    "\n\n"
    "def _dg1_insts():\n"
    "    return tuple(sorted({DG1_GEOMETRY[k][:3] for k in DG1_ROUTE if k in DG1_GEOMETRY}))\n"
    "\n\n"
    "def _dg1_fold_on():\n"
    "    return bool(DG1_ENABLE and DG1_FOLD_KS1 and _dg1_insts())\n"
    "\n\n"
    "def _dg1_fold(part):\n"
    '    """What _ks1_build adds for DG1: "cuda" source, "cpp" declarations, "fns" exports (empty when the fold is off)."""\n'
    "    if not _dg1_fold_on():\n"
    '        return [] if part == "fns" else ""\n'
    '    return {"cuda": "\\n" + dg1_cuda_src(list(_dg1_insts())), "cpp": DG1_CPP_SRC, "fns": ["dg1_run", "dg1_info"]}[part]\n'
    "\n\n"
)
KS1B = ("def _ks1_build():", "\ndef ")
FN = ("def kernel_fn(data, state):", "\n    return next_attn_input")
PREP = ("def prepare(weights):", "\n    occ = ")
DISABLE = "def _k3_disable(state, key, value, piece, ex):\n"


def patches():
    return [
        ("PY: DG1 block", ("# ---------------------------------------------------------------- KS1 (tools/ks1_patches.py)", None),
         DISABLE, layer_py() + DISABLE),
        ("PY: DG1 fold helpers", ("# ---------------------------------------------------------------- KS1 (tools/ks1_patches.py)", None),
         '_DG1 = {"mod": None, "insts": None, "build_s": 0.0}\n', FOLD_PY + '_DG1 = {"mod": None, "insts": None, "build_s": 0.0}\n'),
        ("KS1 build: DG1 cuda", KS1B, '    cuda = KS1_CUDA_SRC.replace("@INSTS@", insts)\n',
         '    cuda = KS1_CUDA_SRC.replace("@INSTS@", insts) + _dg1_fold("cuda")   # DG1 (tools/dg1_patches.py) rides in this build\n'),
        ("KS1 build: DG1 cpp + exports", KS1B,
         'cpp_sources=KS1_CPP_SRC, cuda_sources=cuda,\n                              functions=["ks1_forward", "ks1_info", "ks3_forward", "ks3_info"],',
         'cpp_sources=KS1_CPP_SRC + _dg1_fold("cpp"), cuda_sources=cuda,\n'
         '                              functions=["ks1_forward", "ks1_info", "ks3_forward", "ks3_info"] + _dg1_fold("fns"),'),
        ("KS1 build: DG1 insts", KS1B, '    _KS1["build_s"] = time.time() - t0\n',
         '    _KS1["build_s"] = time.time() - t0\n    _KS1["dg1_insts"] = _dg1_insts() if _dg1_fold_on() else None\n'),
        ("PREP: DG1 prepare", PREP,
         "    _ks1_prepare(state, t0)   # KS1: one-launch B = 1 sliding attention (FA4 v8 stays on failure)\n",
         "    _ks1_prepare(state, t0)   # KS1: one-launch B = 1 sliding attention (FA4 v8 stays on failure)\n"
         "    _dg1_prepare(state, t0)   # DG1: swap-AB wgmma projections at T = 32 / 128 / 256 (today's cuBLAS stays on failure)\n"),
        ("FN: qkv", FN,
         "    if qkv is None:\n        qkv = (_v8_gemm(",
         "    if qkv is None:\n        qkv = _dg1(state, \"qkv\", data[\"attn_input\"])   # DG1 (T >= 17, routed), else today's path\n"
         "    if qkv is None:\n        qkv = (_v8_gemm("),
        ("FN: o", FN,
         "    if attn_output is None:\n        attn_output = _v12_gemm(o, dense[\"o\"], \"o\", tokens)",
         "    if attn_output is None:\n        attn_output = _dg1(state, \"o\", o)   # DG1 (T >= 17, routed), else today's path\n"
         "    if attn_output is None:\n        attn_output = _v12_gemm(o, dense[\"o\"], \"o\", tokens)"),
        ("FN: cat", FN,
         "        if y32 is None:\n            y32 = torch.mm(ffn_input, state[\"v8\"][\"wcat\"].t(), out_dtype=torch.float32)\n",
         "        if y32 is None:\n            y32 = _dg1(state, \"cat\", ffn_input)   # DG1 [pooler; hi; lo] FP32 out (T >= 17, routed)\n"
         "        if y32 is None:\n            y32 = torch.mm(ffn_input, state[\"v8\"][\"wcat\"].t(), out_dtype=torch.float32)\n"),
        ("FN: unpooler", FN,
         "        if ffn_output is None:\n            ffn_output = _v12_gemm(expert_output, dense[\"unpooler\"], \"unpool\", tokens)\n",
         "        if ffn_output is None:\n            ffn_output = _dg1(state, \"unpooler\", expert_output)   # DG1 (T >= 17, routed)\n"
         "        if ffn_output is None:\n            ffn_output = _v12_gemm(expert_output, dense[\"unpooler\"], \"unpool\", tokens)\n"),
    ]


def _scope(text, scope):
    a = text.index(scope[0])
    if text.count(scope[0]) != 1:
        raise RuntimeError(f"dg1: scope start {scope[0]!r} occurs {text.count(scope[0])} times")
    if scope[1] is None:
        return a, len(text)
    b = text.find(scope[1], a + len(scope[0]))
    return a, (len(text) if b < 0 else b)


def apply(text, patch_list=None):
    if "DG1_ENABLE" in text:
        raise RuntimeError("dg1: layer already carries DG1")
    for what, scope, old, new in (patch_list if patch_list is not None else patches()):
        a, b = _scope(text, scope)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"dg1 patch {what!r}: `old` occurs {n} times in its scope (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_v33_dg1_check.py")
    text = src.read_text(encoding="utf-8")
    pl = patches()
    for what, scope, old, _ in pl:
        a, b = _scope(text, scope)
        print(f"  {text[a:b].count(old)}x  {what}")
    patched = apply(text, pl)
    out.write_bytes(patched.encode("utf-8"))   # LF as read
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(pl)} patches hit once in scope; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

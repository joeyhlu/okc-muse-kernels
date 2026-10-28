"""dg1p12_patches.py: DGP = DG1's swap-AB SS-wgmma dense GEMM fed with lossless 12-bit P12 weight stage blocks (kernels/lab/
dgp_lab.py, notes/study/dgp_dense.md) as the T = 17..32 route for the cat ([pooler; router hi; lo], FP32 out) and o, tried
before DG1 (which stays live as the fallback and as the A/B "ship" arm). Exact text patches against the stripped
kernels/muse_decode_layer/k3_slim_4.py; the CUDA source, its generator, the packer and the torch unpacker are copied verbatim
from the lab (DGP_INSTS .. dgp_unpack).

Kernel (lab docstring): one cp.async.bulk per stage of a P12 block (per warpgroup 3 planes x 128 threads x 16 B + an escape
header + entries); DGP_RS = 1: each consumer thread decodes its wgmma A fragment (3 LDS.128, 36 ALU) straight into registers
(double-buffered across stages), XOR-patches its escapes, then wgmma m64n32k16 RS; DGP_RS = 0: the SS form (decode into a BF16
smem A tile + fence.proxy.async; lost R-dgp-1). x TMA ring, cluster DSMEM split-K, PDL and bounded waits are DG1's.

Layer glue: DGP_ENABLE / DGP_ROUTE {(projection, bucket): True} read at call time (A/B variants switch routes off without a
new prepare); DGP_GEOMETRY {(projection, bucket): (T, NWG, S)}; the build runs as PT1 job "dgp" in parallel with the other
nvcc builds; prepare packs every routed weight (per-row exponent windows, escapes listed per stage block; a pack above the
escape cap raises -> route off, loud), checks the pack with a torch unpack (bitwise == W), then self-tests the kernel: random x
at T = bucket and bucket - 3 vs FP64 (rel-L2 < 4e-3 bf16 / 1e-4 fp32 out) and vs today's torch.mm (< 1e-2), NaN-prefilled
output fully written, repeat bit-identical, no timeout flag, and a one-hot x sweep over EVERY k (y == W bitwise through the
real kernel: decode + escapes). A failed check turns that route off (loud); a host exception at call time disables DGP for
the run (_k3_disable). Prepare past DGP_PREPARE_BY_S skips DGP (loud).

    python -I tools/dg1p12_patches.py [k3_slim_4.py] [out.py]
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_slim_4.py"
LAB = ROOT / "kernels/lab/dgp_lab.py"

ROUTE = {("cat", 32): True, ("o", 32): True}
GEOMETRY = {("cat", 32): (32, 2, 3), ("o", 32): (32, 1, 1)}   # (T, NWG, S): DG1's shipped geometries on P12 blocks


def _lab_block():
    """DGP_INSTS .. dgp_unpack verbatim from the lab (dgp_build and the lab harness stay there)."""
    lab = LAB.read_text(encoding="utf-8")
    a, b = lab.index("DGP_INSTS = "), lab.index("def dgp_build(")
    return lab[a:b].rstrip() + "\n"


def layer_py(route=ROUTE, geometry=GEOMETRY):
    return (
        "# ---------------------------------------------------------------- DGP (tools/dg1p12_patches.py): DG1 fed with lossless P12 stage blocks, T = 17..32\n"
        "DGP_ENABLE = True   # master switch (read at call time; False also skips the DGP build + pack in prepare)\n"
        f"DGP_ROUTE = {route!r}   # (projection, bucket) -> True = routed (read at call time); every key is packed + self-tested in prepare\n"
        f"DGP_GEOMETRY = {geometry!r}   # (projection, bucket) -> (T, NWG, S)\n"
        "DGP_RS = 1   # 1 = RS-wgmma consumer (P12 decoded into the A registers), 0 = SS (smem A tile + proxy fence; lost R-dgp-1)\n"
        "DGP_PDL = True    # programmatic stream serialization (weights before griddepcontrol.wait; read at call time)\n"
        "DGP_TRIG = True   # griddepcontrol.launch_dependents at entry on one-wave grids (read at call time)\n"
        "DGP_TOL = (4e-3, 1e-4, 1e-2)   # self-test rel-L2: vs FP64 (bf16 out, fp32 out), vs today's torch.mm\n"
        "DGP_PREPARE_BY_S = 265.0   # no DGP pack / self-test starts after this many seconds of prepare (route off, loud)\n"
        "\n"
        + _lab_block() +
        "\n\n"
        '_DGP = {"mod": None, "insts": None, "build_s": 0.0}\n'
        "\n\n"
        "def _dgp_insts():\n"
        "    return tuple(sorted({DGP_GEOMETRY[k][:2] for k in DGP_ROUTE if k in DGP_GEOMETRY}))\n"
        "\n\n"
        "def _dgp_build(insts=None):\n"
        '    """DGP\'s own small load_inline holding only the routed (T, NWG) instantiations (cached; PT1 job "dgp" runs it early)."""\n'
        '    _pt1_wait("dgp")\n'
        "    if insts is None:\n"
        "        insts = _dgp_insts()\n"
        '    if _DGP["mod"] is not None and _DGP["insts"] == insts:\n'
        '        return _DGP["mod"]\n'
        "    from torch.utils.cpp_extension import load_inline\n"
        "    cuda = dgp_cuda_src(list(insts))\n"
        "    tag = hashlib.sha1((DGP_CPP_SRC + cuda).encode()).hexdigest()[:10]\n"
        "    t0 = time.time()\n"
        '    _DGP["mod"] = load_inline(name=f"dgp_{tag}", cpp_sources=DGP_CPP_SRC, cuda_sources=cuda, functions=["dgp_run", "dgp_info"],\n'
        '                              extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)\n'
        '    _DGP["insts"], _DGP["build_s"] = insts, time.time() - t0\n'
        '    return _DGP["mod"]\n'
        "\n\n"
        "def _dgp_weight(state, proj):\n"
        '    if proj == "cat":\n'
        '        return state["v8"]["wcat"] if state.get("v8") is not None else None\n'
        '    return state["dense"]["qkv_gate" if proj == "qkv" else proj]\n'
        "\n\n"
        "def _dgp(state, proj, x):\n"
        '    """DGP for (projection, T = x.shape[0]) when routed and self-tested, else None -> the caller keeps today\'s path (DG1 ...).\n'
        '    Shape-only branches (graph safe); a host exception disables DGP for the rest of the run (once, with a line)."""\n'
        '    st = state.get("dgp")\n'
        "    if not DGP_ENABLE or st is None:\n"
        "        return None\n"
        "    T = x.shape[0]\n"
        "    if T <= 16:\n"
        "        return None\n"
        "    key = (proj, _k2a_bucket_of(T))\n"
        "    if not DGP_ROUTE.get(key):   # call-time filter: A/B variants switch routes off without a new prepare\n"
        "        return None\n"
        '    r = st["route"].get(key)\n'
        '    if (r is None or T > r["T"] or x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != r["K"] or not x.is_contiguous()\n'
        "            or x.data_ptr() % 16 != 0):\n"
        "        return None\n"
        "    try:\n"
        '        out = torch.empty((T, r["N"]), dtype=torch.float32 if proj == "cat" else torch.bfloat16, device=x.device)\n'
        '        st["mod"].dgp_run(r["kid"], x, r["w"], r["base"], out, st["err"], r["S"], 1 if DGP_PDL else 0,\n'
        '                          1 if (DGP_TRIG and r["wave1"]) else 0, r["N"], 0, r["wb"], DGP_RS)\n'
        "        return out\n"
        "    except Exception as ex:  # noqa: BLE001\n"
        '        _k3_disable(state, "dgp", None, "DGP", ex)\n'
        "        return None\n"
        "\n\n"
        "def _dgp_selftest(st, r, proj, w, T, gen):\n"
        '    """(ok, message): random x at T and T - 3 vs FP64 and vs today\'s torch.mm (NaN-prefilled output fully written, repeat\n'
        '    bit-identical, timeout flag clear), then a one-hot x sweep over every k: y == W bitwise through the kernel. Prepare only."""\n'
        '    f32 = proj == "cat"\n'
        "    dev = w.device\n"
        "    ok, parts = True, []\n"
        "\n"
        "    def run(x, out):\n"
        '        st["mod"].dgp_run(r["kid"], x, r["w"], r["base"], out, st["err"], r["S"], 1, 0, r["N"], 0, r["wb"], DGP_RS)\n'
        "    for t in (T, T - 3):\n"
        '        x = torch.randn((t, r["K"]), generator=gen, device=dev).to(torch.bfloat16)\n'
        "        outs = []\n"
        "        for _ in range(2):\n"
        '            out = torch.full((t, r["N"]), float("nan"), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)\n'
        "            run(x, out)\n"
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
        "        good = fin and same and errf == 0 and e64 < DGP_TOL[1 if f32 else 0] and etoday < DGP_TOL[2]\n"
        "        ok &= good\n"
        '        parts.append(f"T={t}: rel vs fp64 {e64:.2e} vs today {etoday:.2e} finite {fin} repeat {same} err {errf}")\n'
        '        st["err"].zero_()\n'
        "    n, k = w.shape\n"
        '    y = torch.empty((T, n), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)\n'
        "    x = torch.zeros((T, k), dtype=torch.bfloat16, device=dev)\n"
        "    wt = w.t()\n"
        "    ar = torch.arange(T, device=dev)\n"
        "    bad = 0\n"
        "    for k0 in range(0, k, T):\n"
        "        x.zero_()\n"
        "        x[ar, k0 + ar] = 1.0\n"
        "        run(x, y)\n"
        "        bad += int((y.float() != wt[k0:k0 + T].float()).sum())\n"
        "    torch.cuda.synchronize()\n"
        '    errf = int(st["err"][0])\n'
        '    st["err"].zero_()\n'
        "    ok &= bad == 0 and errf == 0\n"
        '    parts.append(f"one-hot sweep bad {bad} err {errf}")\n'
        '    return ok, "; ".join(parts)\n'
        "\n\n"
        "def _dgp_prepare(state, t0):\n"
        '    """Build (PT1-prefetched), P12 packs (torch unpack == W bitwise) and per-route self-tests. state["dgp"] stays None (today\'s\n'
        '    path, i.e. DG1) when off, late, the build fails or nothing passes."""\n'
        '    state["dgp"] = None\n'
        "    keys = sorted(k for k in DGP_ROUTE if k in DGP_GEOMETRY)\n"
        "    if not DGP_ENABLE or not keys:\n"
        "        return None\n"
        "    t1 = time.time()\n"
        "    if t1 - t0 > DGP_PREPARE_BY_S:\n"
        '        print(f"[dgp] prepare already at {t1 - t0:.0f}s > {DGP_PREPARE_BY_S:.0f}s -> DGP off (DG1 / today\'s path)", flush=True)\n'
        "        return None\n"
        "    insts = _dgp_insts()\n"
        "    try:\n"
        "        mod = _dgp_build(insts)\n"
        "    except Exception as ex:  # noqa: BLE001\n"
        '        print(f"[dgp] build FAILED -> today\'s path: {ex!r}", flush=True)\n'
        "        return None\n"
        '    dev = state["dense"]["qkv_gate"].device\n'
        '    st = {"mod": mod, "route": {}, "err": torch.zeros(4, dtype=torch.int32, device=dev)}\n'
        "    gen = torch.Generator(device=dev).manual_seed(4243)\n"
        "    for key in keys:\n"
        "        proj, rep = key\n"
        "        T, nwg, s = DGP_GEOMETRY[key]\n"
        "        w = _dgp_weight(state, proj)\n"
        "        if w is None:\n"
        '            print(f"[dgp] {key}: no weight (v8 tail off) -> route off", flush=True)\n'
        "            continue\n"
        "        try:\n"
        "            kid = insts.index((T, nwg))\n"
        "            info = [int(v) for v in mod.dgp_info(kid, s, DGP_RS)]\n"
        "            n, k = w.shape\n"
        "            if n % info[3] or rep > T or k % 64:\n"
        '                print(f"[dgp] {key}: geometry {DGP_GEOMETRY[key]} does not fit W {tuple(w.shape)} -> route off", flush=True)\n'
        "                continue\n"
        "            tp = time.time()\n"
        "            blocks, base, wb, stats = dgp_pack(w, nwg, rs=bool(DGP_RS))\n"
        "            back = dgp_unpack(blocks, base, wb, n, k, nwg, rs=bool(DGP_RS))\n"
        "            exact = bool(torch.equal(back.view(torch.int16), w.view(torch.int16)))\n"
        "            del back\n"
        '            print(f"[dgp] pack {proj} {n}x{k} NWG {nwg}: {stats[\'mb\']:.1f} MB (x{stats[\'mb\'] / stats[\'bf16_mb\']:.3f} of bf16), wb {wb},"\n'
        '                  f" {stats[\'nesc\']} escapes (max / block {stats[\'nmax\']}), {time.time() - tp:.1f}s; torch unpack bit-exact {exact}", flush=True)\n'
        "            if not exact:\n"
        '                print(f"[dgp] {key}: pack is NOT bit-exact -> route off", flush=True)\n'
        "                continue\n"
        '            r = {"kid": kid, "S": s, "T": T, "N": n, "K": k, "w": blocks, "base": base, "wb": wb, "wave1": info[0] >= n // info[3]}\n'
        "            ok, msg = _dgp_selftest(st, r, proj, w, rep, gen)\n"
        '            print(f"[dgp] self-test {proj} bucket {rep} geo {DGP_GEOMETRY[key]} (BM {info[3]} ST {info[4]} regs {info[5]} clusters {info[0]}"\n'
        "                  f\"{' one wave' if r['wave1'] else ''}): {msg} -> {'ok' if ok else 'FAIL: route off'}\", flush=True)\n"
        "            if ok:\n"
        '                st["route"][key] = r\n'
        "        except Exception as ex:  # noqa: BLE001\n"
        '            print(f"[dgp] {key}: pack / self-test raised {ex!r} -> route off", flush=True)\n'
        '    state["dgp"] = st if st["route"] else None\n'
        '    print(f"[dgp] routes live {sorted(st[\'route\'])} (build {_DGP[\'build_s\']:.1f}s, total {time.time() - t1:.1f}s; prepare so far"\n'
        "          f\" {time.time() - t0:.1f}s)\", flush=True)\n"
        '    return state["dgp"]\n'
        "\n\n"
    )


FN = ("def kernel_fn(data, state):", "\n    return next_attn_input")
PREP = ("def prepare(weights):", "\n    occ = ")
PT1 = ("def _pt1_start():", "\ndef ")
SU1 = "SU1_ENABLE = True\n"


def patches(route=ROUTE, geometry=GEOMETRY):
    return [
        ("PY: DGP block", (SU1, None), SU1, layer_py(route, geometry) + SU1),
        ("PT1: dgp build job", PT1, '    if KS1_ENABLE:\n        jobs.append(("ks1", _ks1_build))\n',
         '    if KS1_ENABLE:\n        jobs.append(("ks1", _ks1_build))\n'
         '    if DGP_ENABLE and _dgp_insts():\n        jobs.append(("dgp", lambda: _dgp_build(_dgp_insts())))   # DGP (tools/dg1p12_patches.py)\n'),
        ("PREP: DGP prepare", PREP, "    _dg1_prepare(state, t0)\n",
         "    _dg1_prepare(state, t0)\n    _dgp_prepare(state, t0)   # DGP: DG1 on P12 stage blocks at T = 17..32 (DG1 / today's path stays on failure)\n"),
        ("FN: o", FN,
         '    if attn_output is None:\n        attn_output = _dg1(state, "o", o)\n',
         '    if attn_output is None:\n        attn_output = _dgp(state, "o", o)   # DGP (T >= 17, routed), else DG1 / today\'s path\n'
         '    if attn_output is None:\n        attn_output = _dg1(state, "o", o)\n'),
        ("FN: cat", FN,
         '        if y32 is None:\n            y32 = _dg1(state, "cat", ffn_input)\n',
         '        if y32 is None:\n            y32 = _dgp(state, "cat", ffn_input)   # DGP [pooler; hi; lo] FP32 out on P12 blocks (T >= 17, routed)\n'
         '        if y32 is None:\n            y32 = _dg1(state, "cat", ffn_input)\n'),
    ]


def _scope(text, scope):
    if text.count(scope[0]) != 1:
        raise RuntimeError(f"dgp: scope start {scope[0]!r} occurs {text.count(scope[0])} times")
    a = text.index(scope[0])
    if scope[1] is None:
        return a, len(text)
    b = text.find(scope[1], a + len(scope[0]))
    return a, (len(text) if b < 0 else b)


def apply(text, route=ROUTE, geometry=GEOMETRY):
    if "DGP_ENABLE" in text:
        raise RuntimeError("dgp: layer already carries DGP")
    if "DG1_ENABLE" not in text or "_pt1_start" not in text:
        raise RuntimeError("dgp: layer must carry DG1 and PT1")
    for what, scope, old, new in patches(route, geometry):
        a, b = _scope(text, scope)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"dgp patch {what!r}: `old` occurs {n} times in its scope (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_slim_4_dgp_check.py")
    text = src.read_text(encoding="utf-8")
    pl = patches()
    for what, scope, old, _ in pl:
        a, b = _scope(text, scope)
        print(f"  {text[a:b].count(old)}x  {what}")
    patched = apply(text)
    out.write_bytes(patched.encode("utf-8"))   # LF as read
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(pl)} patches hit once in scope; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

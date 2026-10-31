"""vel_patches.py (R-vel-1): VEL = VMS early landing at T = 1 (RSP's remaining part; notes/study/indep_review_oct31.md s6).

  python -I tools/vel_patches.py [--base=kernels/lab/k3_hh_3rr.py] [--out=kernels/lab/k3_vel_1r.py] [--on]
  python -I tools/slim_patches.py --base=kernels/lab/k3_vel_1r.py --out=kernels/lab/k3_vel_1.py      (LAST)
  python tools/lead_gate.py file kernels/lab/k3_vel_1.py

What changes at T = 1 when VEL_ON (sliding and full; place 'side', tail 'sep'):
  1. Routing on the side stream: right after vf_router32, the SAME v8 tail kernel binary (tail_kernel3, same template instance:
     dummy pooler input of the real input's dtype) runs on the side stream. Its routing warps read only the router partials,
     bias and local_start, so tid / tw / lid are bitwise the ship's; its norm warps write a dummy xn. No PDL attribute.
     The main stream's tail3 is unchanged (it still produces xn); kernel_fn returns the side tid / tw / lid.
  2. Ordering: main.wait_stream(side) already precedes the main tail3, so the side routing completes before tail3 launches,
     hence before VMS (tail3's PDL secondary) launches: VMS may read the routing before its griddepcontrol.wait.
  3. VMS: a second module (-DVMS_EARLY=1 -DVMS_FX=4) moves griddepcontrol.wait from the kernel entry to after route()
     (warp 0: ids via __ldcg, job scan, first NS weight copies issued by lane 0 = the existing FX&4 path) and the CTA barrier,
     and loads phase-A x with __ldcg instead of __ldg (R-rpo-0: no .nc load of predecessor data once the wait moves).
     The ship module is built from the same source with VMS_EARLY 0: preprocessed text identical, SASS identical
     (tools/local/vel_sass.py). VMS's ctr[0] generation is written only at the end of the previous launch: safe pre-wait.
  Prepare: the early module builds in parallel (PT1 job 'vms_e'); it is used only if its outputs equal the ship module's
  bitwise on the T = 1 self-test cases (else VEL stays off with a print). Defaults: VEL_BUILD True, VEL_ON False (arms set it);
  --on writes VEL_ON = True (a ship candidate).
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
WAIT = 'asm volatile("griddepcontrol.wait;' + chr(92) + 'n" ::: "memory");'


def sub(text, old, new, count=1, lo=0, hi=None):
    hi = len(text) if hi is None else hi
    n = text.count(old, lo, hi)
    if n != count:
        sys.exit(f"vel_patches: anchor found {n}x (want {count}): {old[:100]!r}")
    i = text.index(old, lo, hi)
    return text[:i] + new + text[i + len(old):]


CU_DEFS = """#ifndef VMS_EARLY
#define VMS_EARLY 0
#endif
#if VMS_EARLY
#define VMS_XLD __ldcg
#else
#define VMS_XLD __ldg
#endif
"""

PY = '''
VEL_BUILD = True       # R-vel-1: build the VMS early-landing module (-DVMS_EARLY=1 -DVMS_FX=4) in prepare (PT1 job 'vms_e')
VEL_ON = False         # R-vel-1: at T = 1, side-stream routing + VMS reads it before its griddepcontrol.wait
_VEL = {"on": False, "seen": set()}


def _vms_build_e():
    from torch.utils.cpp_extension import load_inline
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1", "-DVMS_EARLY=1", "-DVMS_FX=4", "-Xptxas=-warn-spills",
             "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((VMS_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t1 = time.time()
    mod = load_inline(name=f"k3_vmse_{tag}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC, functions=["run", "attrs"],
                      extra_cuda_cflags=flags)
    _VMS["mod_e"] = mod
    _VMS["G_e"] = G
    _K3_BUILD_S["vms_e"] = time.time() - t1
    return mod


def _vel_selftest(st, dev):
    """Early module vs ship module on the T = 1 self-test inputs (normal + skewed, 2 calls each). VMS reduces split partials
    with FP32 atomics, so ship vs ship is not bitwise: accept iff no error flag and rel L2(early, ship) <= max(4 x the
    ship-vs-ship rel L2, 1e-3) on every input (the A/B and judge checks then validate the layer outputs)."""
    vs = st["vms"]
    try:
        _pt1_wait("vms_e")
        mod_e = _VMS.get("mod_e")
        if mod_e is None:
            mod_e = _vms_build_e()
        if _VMS.get("G_e") != _VMS["G"]:
            print("[vel] early module built for another SM count: VEL OFF", flush=True)
            return
        good, rows = True, []
        for seed, dist in ((41, "normal"), (42, "skewed"), (51, "normal"), (52, "skewed")):
            d = _k2a_self_test_inputs(1, seed, dev, dist)
            x = d["hidden_states"].contiguous()
            outs, errs = [], []
            for use_e in (False, True, False, True):
                vs.pop("mod_e", None)
                if use_e:
                    vs["mod_e"] = mod_e
                _VEL["on"] = use_e
                o = _vms_call(st, x, d["topk_ids"], d["topk_weights"])
                torch.cuda.synchronize()
                outs.append(o.float().clone())
                errs.append(int(vs["ctr"][2]))
            _VEL["on"] = False
            rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-30))  # noqa: E731
            ss, es, ee = rel(outs[2], outs[0]), rel(outs[1], outs[0]), rel(outs[3], outs[1])
            ok = not any(errs) and bool(torch.isfinite(outs[1]).all()) and max(es, ee) <= max(4 * ss, 1e-3)
            good &= ok
            rows.append(f"{dist}/{seed}: ship-ship {ss:.1e} early-ship {es:.1e} early-early {ee:.1e} err {errs}")
        vs.pop("mod_e", None)
        if good:
            vs["mod_e"] = mod_e
        print(f"[vel] early VMS module {'matches ship on T = 1 -> available' if good else 'DIFFERS -> VEL OFF'} "
              f"(regs/local/smem/G/NS {list(mod_e.attrs())}, build {_K3_BUILD_S.get('vms_e', 0):.0f}s): " + "; ".join(rows),
              flush=True)
    except Exception as ex:
        _VEL["on"] = False
        vs.pop("mod_e", None)
        print(f"[vel] early module raised {ex!r}: VEL OFF", flush=True)


def _vel_route(state, logits, data, tokens, place, tail, y):
    """Side-stream routing at T = 1 (same tail_kernel3 instance as the main tail, dummy pooler input); None = ship path."""
    if not (VEL_ON and tokens == 1 and place == "side" and tail == "sep" and y is not None and logits is not None
            and logits.dim() == 3 and state.get("v8") is not None):
        return None
    k33 = state.get("k2a33")
    vs = k33["st"].get("vms") if k33 is not None else None
    zy = state.get("vel_zy", {}).get(y.dtype)
    if vs is None or vs.get("mod_e") is None or not k33["st"].get("vms_ok") or zy is None:
        return None
    v8, dev = state["v8"], logits.device
    tw = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
    tid = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    lid = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    v8["mod"].v8_tail(zy[:tokens], logits, 0, -1, logits.stride(1), logits.shape[0], logits.stride(0), v8["nw"], v8["bias"],
                      data["local_start"], state["vel_xn"][:tokens], tw, tid, lid, RMS_EPS, 0.1767766952966369, 1.0, 1e-15,
                      (2 if TL1_TAIL2 else 0) | _ft_flag(tokens))
    if (tokens, str(y.dtype)) not in _VEL["seen"]:
        _VEL["seen"].add((tokens, str(y.dtype)))
        print(f"[vel] first side-routing call T={tokens} y {y.dtype} ft {_ft_flag(tokens)}", flush=True)
    return tw, tid, lid


'''


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3rr.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_vel_1r.py")
    s = base.read_bytes().decode("ascii")
    lines = s.split(NL)
    lines[1] = ("# nonce k3-vel-1r (k3_hh_3rr + R-vel-1 VMS early landing at T = 1, VEL_ON "
                + ("on" if "on" in FLAGS else "off") + "; tools/vel_patches.py)")
    s = NL.join(lines)
    lo = s.index("VMS_CUDA_SRC = r")
    hi = s.index('"""', lo + 20)
    n0 = hi - lo
    # CUDA (VMS_CUDA_SRC only)
    s = sub(s, "#ifndef VMS_G\n#define VMS_G 132\n#endif\n", "#ifndef VMS_G\n#define VMS_G 132\n#endif\n" + CU_DEFS, lo=lo, hi=hi)
    hi += len(CU_DEFS)
    w1 = "#if __CUDA_ARCH__ >= 900\n  " + WAIT + "\n#endif\n  if (warp == 0) {\n    const int m8 = p.m * 8;"
    s = sub(s, w1, w1.replace("#if __CUDA_ARCH__ >= 900\n", "#if __CUDA_ARCH__ >= 900 && !VMS_EARLY\n", 1), lo=lo, hi=hi)
    hi += len(" && !VMS_EARLY")
    add = "#if VMS_EARLY && __CUDA_ARCH__ >= 900\n  " + WAIT + "\n#endif\n"
    s = sub(s, "  __syncthreads();\n  const int njobs = S.njobs;\n",
            "  __syncthreads();\n" + add + "  const int njobs = S.njobs;\n", lo=lo, hi=hi)
    hi += len(add)
    for a in ("xr[2 * cc] = PB ? __ldcg(src + cc * 8) : __ldg(src + cc * 8);",
              "xr[2 * cc + 1] = PB ? __ldcg(src + cc * 8 + 1) : __ldg(src + cc * 8 + 1);"):
        s = sub(s, a, a.replace(": __ldg(", ": VMS_XLD("), lo=lo, hi=hi)
        hi += len("VMS_XLD") - len("__ldg")
    assert s.index('"""', lo + 20) == hi, "VMS_CUDA_SRC end moved unexpectedly"
    print(f"vel_patches: VMS_CUDA_SRC {n0:,} -> {hi - lo:,} chars")
    # Python
    s = sub(s, '    if VMS_ENABLE:\n        jobs.append(("vms", _vms_build))\n',
            '    if VMS_ENABLE:\n        jobs.append(("vms", _vms_build))\n    if VMS_ENABLE and VEL_BUILD:\n'
            '        jobs.append(("vms_e", _vms_build_e))\n')
    s = sub(s, '        st["vms_ok"] = good\n',
            '        st["vms_ok"] = good\n        if good and VEL_BUILD:\n            _vel_selftest(st, dev)\n')
    s = sub(s, '    tw = topk_weights.to(torch.float32).contiguous()\n    m = x.shape[0]\n',
            '    tw = topk_weights.to(torch.float32).contiguous()\n    m = x.shape[0]\n'
            '    if _VEL["on"] and m == 1 and vs.get("mod_e") is not None:\n        vs = dict(vs, mod=vs["mod_e"])\n')
    s = sub(s, '        "side_stream": torch.cuda.Stream(device),\n',
            '        "side_stream": torch.cuda.Stream(device),\n'
            '        "vel_zy": {dt: torch.zeros((8, D_EXPERT), dtype=dt, device=device) for dt in (torch.bfloat16, torch.float32)},\n'
            '        "vel_xn": torch.zeros((8, D_EXPERT), dtype=torch.bfloat16, device=device),\n')
    s = sub(s, "    mode, place = _v8_router_mode(tokens, state)\n",
            "    vel = None\n    mode, place = _v8_router_mode(tokens, state)\n")
    s = sub(s, "            with torch.cuda.stream(side_stream):\n"
               "                logits = _v8_logits(mode, ffn_input, router_input, state)\n",
            "            with torch.cuda.stream(side_stream):\n"
            "                logits = _v8_logits(mode, ffn_input, router_input, state)\n"
            "                vel = _vel_route(state, logits, data, tokens, place, tail,\n"
            "                                 expert_input if place != \"side_first\" else None)\n")
    s = sub(s, "        if tail is not None:\n"
               "            expert_x, topk_weights, topk_ids, local_ids = _v8_tail(expert_input, logits, state, data)\n",
            "        if tail is not None:\n"
            "            expert_x, topk_weights, topk_ids, local_ids = _v8_tail(expert_input, logits, state, data)\n"
            "            if vel is not None:\n"
            "                topk_weights, topk_ids, local_ids = vel\n"
            "                if not torch.cuda.is_current_stream_capturing():\n"
            "                    for tensor in vel:\n"
            "                        tensor.record_stream(torch.cuda.current_stream())\n")
    s = sub(s, '    _SU1_FLAG["k3trig"] = _su1_k3trig(state, tokens)\n    try:\n'
               '        expert_output = _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens)\n'
               '    finally:\n        _SU1_FLAG["k3trig"] = False\n',
            '    _SU1_FLAG["k3trig"] = _su1_k3trig(state, tokens)\n    _VEL["on"] = vel is not None\n    try:\n'
            '        expert_output = _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens)\n'
            '    finally:\n        _SU1_FLAG["k3trig"] = False\n        _VEL["on"] = False\n')
    py = PY.replace("VEL_ON = False ", "VEL_ON = True  ") if "on" in FLAGS else PY
    s = sub(s, NL + "def kernel_fn(data, state):" + NL, NL + py.rstrip(NL) + NL + NL + NL + "def kernel_fn(data, state):" + NL)
    ast.parse(s)
    out.write_bytes(s.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"vel_patches: {base.name} -> {out.name} ({len(s):,} B)")


if __name__ == "__main__":
    main()

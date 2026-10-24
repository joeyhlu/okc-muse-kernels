"""k3_v05 CPU emulation: runs kernels/muse_decode_layer/k3_v05.py UNMODIFIED on CPU, reusing every stand-in,
the reference copy and the checks of kernels/muse_decode_layer/k3_emulate.py (imported, not edited), with a K1 v1
fake extension in place of k3_emulate's K1 v0 one.

Run:  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -I k3_v05_emulate.py [scenario ...]

K1 v1 fake (load_inline replacement, functions == [k1_forward, k1_occupancy]):
  * asserts it was handed k1_v1.py's CPP_SRC/CUDA_SRC byte for byte and a k1_v1_<tag> module name;
  * k1_occupancy: 2 (L=1) / 1 (L=8), the occupancies the judge printed for k1_v1 (job d857095aa064476f);
  * k1_forward: re-applies every TORCH_CHECK of k1_v1's C++ wrapper (incl. counters / nsg / min_chunk), asserts the
    fold-counter buffer is ONE persistent 64K int32 buffer for the module's lifetime and all-zero at kernel entry,
    runs k1_emulate.emulate_k1(fold=..., nsg=...) (CTA-level, v1: fold merge or k1_combine, L=8 E4M3 QK) with the
    per-row key-coverage check, and plays the fold protocol on the counters (every CTA of a multi-split (b, hp)
    increments its counter, the last one resets it to 0). Injections: build failure, output x1.5, a counter left
    at 1 (kernel that does not reset), TORCH_CHECK raise at call time under capture.
Routing is checked per call against the tables read from k1_v2.py (not from k3_v05.py): which (kind, L, B) take
K1, fold vs separate, Gx (SMs x min(cps, occupancy) / HP), min_chunk, nsg.
"""

import ast
import importlib.util
import io
import pathlib
import sys
import time

ROOT = pathlib.Path("<scratch>/"
                    "8561677d-1e4a-44b9-b007-f962c76eecc2/scratch-2026-10-23-2c70b6")
HERE = pathlib.Path(__file__).resolve().parent
LAYER = ROOT / "kernels/muse_decode_layer"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load("k3_emulate", LAYER / "k3_emulate.py")     # installs the SGLang/torch.cuda stand-ins and capture guards
import torch  # noqa: E402

K1E = E.K1E


def _consts(path, names):
    tree = ast.parse(path.read_text())
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names:
            out[n.targets[0].id] = eval(compile(ast.Expression(n.value), str(path), "eval"), {"__builtins__": {}})
        if isinstance(n, ast.FunctionDef) and n.name in names:
            ns = {}
            exec(compile(ast.Module([n], []), str(path), "exec"), ns)   # noqa: S102 (our own file, one pure function)
            out[n.name] = ns[n.name]
    return out


K1V1 = _consts(ROOT / "kernels/muse_attention_decode/k1_v1.py", ("CPP_SRC", "CUDA_SRC"))
K1V1 = (K1V1["CPP_SRC"], K1V1["CUDA_SRC"])
TAB = _consts(ROOT / "kernels/muse_attention_decode/k1_v2.py",
              ("K1_ROUTE_MIN_BATCH", "K1_FOLD_MIN_BATCH", "K1_MIN_CHUNK_T", "K1_CPS_T", "K1_COUNTERS", "_nsg"))
OCC = {1: 2, 8: 1}
N_SM = 132
E.FAIL["k1_counter_leak"] = False
FAIL = E.FAIL
LOG = []
CNT = {"ptr": None}


def _chk(cond, msg):
    if not cond:
        raise RuntimeError(f"TORCH_CHECK failed: {msg}")


class FakeK1V1:
    @E.standin
    def k1_occupancy(self, L):
        return OCC[int(L)]

    @E.standin
    def k1_forward(self, qkv, seq_lens, cache_loc, page_table, k_cache, v_cache, k_scale, v_scale, inv_freq,
                   part_o, part_lse, out, counters, B, L, Hq, Hkv, W, page_shift, Gx, min_chunk, use_rope, qk_sm,
                   fold, nsg):
        if FAIL["k1_raise_at_call"] and E.CAPTURE[0]:
            raise RuntimeError("TORCH_CHECK failed: injected call-time failure")
        # k1_v1.py k1_forward's TORCH_CHECKs, same conditions
        _chk(L in (1, 8), "k1: q_len must be 1 or 8")
        _chk(Hq == 8 * Hkv and Hkv % 2 == 0 and Hkv > 0, "k1: need Hq == 8*Hkv and even Hkv")
        _chk(Gx > B, "k1: Gx must exceed B")
        _chk(min_chunk >= 1, "k1: min_chunk")
        _chk(qkv.dtype == torch.bfloat16 and qkv.is_contiguous(), "k1: qkv")
        _chk(qkv.numel() == B * L * (2 * Hq + 2 * Hkv) * 64, "k1: qkv shape")
        _chk(seq_lens.dtype == torch.int32 and seq_lens.is_contiguous() and seq_lens.numel() == B, "k1: seq_lens")
        _chk(cache_loc.dtype == torch.int64 and cache_loc.is_contiguous() and cache_loc.numel() == B * L, "k1: cache_loc")
        _chk(page_table.dtype == torch.int32 and page_table.is_contiguous() and page_table.dim() == 2
             and page_table.size(0) == B, "k1: page_table")
        _chk(k_cache.dtype == torch.uint8 and k_cache.is_contiguous(), "k1: k_cache")
        _chk(v_cache.dtype == torch.uint8 and v_cache.is_contiguous(), "k1: v_cache")
        _chk(k_scale.dtype == torch.float32 and v_scale.dtype == torch.float32, "k1: scales")
        _chk(inv_freq.dtype == torch.float32 and inv_freq.numel() >= 32, "k1: inv_freq")
        _chk(part_o.dtype == torch.float32 and part_o.numel() >= (Hkv // 2) * Gx * 16 * L * 64, "k1: part_o")
        _chk(part_lse.dtype == torch.float32 and part_lse.numel() >= (Hkv // 2) * Gx * 16 * L, "k1: part_lse")
        _chk(out.dtype == torch.bfloat16 and out.is_contiguous() and out.numel() == B * L * Hq * 64, "k1: out")
        _chk(counters.dtype == torch.int32 and (fold == 0 or counters.numel() >= B * (Hkv // 2)), "k1: counters")
        _chk(fold != 0 or nsg in (2, 4, 8, 16, 32), "k1: nsg")
        # graph safety of the fold counters: one persistent zeroed buffer, never re-allocated, zero at entry
        assert counters.numel() == TAB["K1_COUNTERS"], "counters are not the persistent K1_COUNTERS buffer"
        if CNT["ptr"] is None:
            CNT["ptr"] = counters.data_ptr()
        assert counters.data_ptr() == CNT["ptr"], "fold counters re-allocated between calls (not persistent)"
        assert int(counters.abs().sum()) == 0, "fold counters not zero at kernel entry"
        if use_rope:
            assert torch.equal(inv_freq.cpu(), E.INV_FREQ_500K), "inv_freq is not theta=5e5's"
            theta = 500000.0
        else:
            assert float(inv_freq.abs().sum()) == 0.0
            theta = 0.0
        d = {"qkv": qkv, "seq_lens": seq_lens, "cache_loc": cache_loc, "page_table": page_table,
             "k_cache": k_cache, "v_cache": v_cache, "k_scale": k_scale, "v_scale": v_scale,
             "config": {"batch": B, "q_len": L, "num_q_heads": Hq, "num_kv_heads": Hkv, "head_dim": 64,
                        "page_size": 1 << page_shift, "window": W, "rope_theta": theta,
                        "qk_scale_factor": qk_sm, "sm_scale": 1.0}}
        visits, stats = {}, {}
        res = K1E.emulate_k1(d, n_sm=Gx * (Hkv // 2), cps=1, min_chunk=min_chunk, stats=stats, visits=visits,
                             fold=bool(fold), nsg=int(nsg))
        assert stats["Gx"] == Gx
        K1E.check_coverage(d, visits)
        if fold:   # the fold protocol: each CTA of a multi-split (b, hp) does one atomicAdd; the last resets it
            hp_n = Hkv // 2
            for b, S in enumerate(stats["S"]):
                if S > 1:
                    for hp in range(hp_n):
                        for _ in range(S):
                            counters[b * hp_n + hp] += 1
                        counters[b * hp_n + hp] = 0
            if FAIL["k1_counter_leak"]:
                counters[0] += 1
        if FAIL["k1_corrupt"]:
            res = (res.float() * 1.5).to(torch.bfloat16)
        out.copy_(res.view_as(out))
        LOG.append(dict(B=B, L=L, W=W, Gx=Gx, min_chunk=min_chunk, fold=int(fold), nsg=int(nsg), capture=E.CAPTURE[0],
                        S=list(stats["S"])))


_ORIG_LOAD_INLINE = E.fake_load_inline


def fake_load_inline(name, cpp_sources, cuda_sources, functions, extra_cuda_cflags=None, verbose=False, **kw):
    if functions == ["k1_forward", "k1_occupancy"]:
        assert (cpp_sources, cuda_sources) == K1V1, "K1 source differs from k1_v1.py"
        assert name.startswith("k1_v1_"), name
        assert "-gencode=arch=compute_90,code=sm_90" in (extra_cuda_cflags or [])
        if FAIL["k1_build"]:
            raise RuntimeError("Error building extension (injected)")
        return FakeK1V1()
    return _ORIG_LOAD_INLINE(name, cpp_sources, cuda_sources, functions, extra_cuda_cflags, verbose, **kw)


E._cpp.load_inline = fake_load_inline


def expected_k1(cfg):
    """None = r1/FA3; else the K1 call the k1_v2 tables imply for this shape."""
    kind = "sliding" if cfg["window"] else "full"
    B, L = cfg["batch"], cfg["q_len"]
    key = (kind, L)
    if B < TAB["K1_ROUTE_MIN_BATCH"].get(key, 1 << 30):
        return None
    cps = max(1, min(TAB["K1_CPS_T"][key], OCC[L]))
    return dict(fold=int(B >= TAB["K1_FOLD_MIN_BATCH"][key]), Gx=max(N_SM * cps // (cfg["num_kv_heads"] // 2), B + 1),
                min_chunk=TAB["K1_MIN_CHUNK_T"][key], nsg=TAB["_nsg"](B))


class Tee(io.TextIOBase):
    def __init__(self, inner):
        self.inner, self.buf = inner, []

    def write(self, s):
        self.buf.append(s)
        return self.inner.write(s)

    def flush(self):
        self.inner.flush()

    def lines(self):
        return "".join(self.buf).splitlines()


CASES = dict(E.CASES)
CASES.update({
    "T16 sliding L1 B16 page1": dict(batch=16, ctx=500, layer="sliding", seed=307, page=1),
    "T4 full L1 B4 page64 kv1/64": dict(batch=4, ctx=900, layer="full", seed=308, kv_scale=0.015625),
    "T8 full L1 B8 page1 (fold threshold)": dict(batch=8, ctx=400, layer="full", seed=309, page=1),
})
FAST_K1_CASES = (0, 3, 6)   # L=1: sliding page 1 B2, full page 64 kv 1/64, sliding page 64 B1 ctx 9

SCENARIOS = {
    # name: (file, cases, FAIL overrides, expected attention path per case ("auto" = from k1_v2 tables), fast,
    #        expected [k3] failure-line substring or None, patch USE_K2A=True)
    "main": ("k3_v05.py", list(CASES), {}, "auto", False, None, False),
    "k1_build_fails": ("k3_v05.py", ["T1 sliding ctx10 page64"], {"k1_build": True}, ["FA3"], True,
                       "[k3] K1 CUDA build FAILED", False),
    "k1_selftest_fails": ("k3_v05.py", ["T1 full ctx3000 page1 soft"], {"k1_corrupt": True}, ["FA3"], True,
                          "[k3] K1 self-test FAILED", False),
    "k1_counter_leak": ("k3_v05.py", ["T1 sliding ctx10 page64"], {"k1_counter_leak": True}, ["FA3"], True,
                        "[k3] K1 self-test FAILED", False),
    "k1_raises_at_call": ("k3_v05.py", ["T1 sliding ctx10 page64", "T1 full ctx3000 page1 soft", "T33 full L1 page1 mixed"],
                          {"k1_raise_at_call": True}, ["K1", "FA3", "FA3"], True, "[k3] K1 raised at call time", False),
    "k2a_switch": ("k3_v05.py", ["T1 sliding ctx10 page64"], {}, "auto", True, None, True),
}
_PREPACK = {}


def main(argv):
    t00 = time.time()
    names = argv or [n for n in SCENARIOS if n != "k2a_switch"]
    weights = E.generate_weights(E.WEIGHTS_SEED)
    print(f"[emu] layer weights generated ({time.time() - t00:.0f}s)", flush=True)
    inputs = {}
    all_ok = True
    summary = []
    for sname in names:
        fname, cases, fails, expect, fast, k3_line, k2a_on = SCENARIOS[sname]
        for k in FAIL:
            FAIL[k] = fails.get(k, False)
        E.CALLS.clear()
        LOG.clear()
        CNT["ptr"] = None
        t0 = time.time()
        print(f"\n===== scenario {sname}: {fname} FAIL={ {k: v for k, v in FAIL.items() if v} }"
              f"{' USE_K2A patched True' if k2a_on else ''} =====", flush=True)
        E.CAPTURE[0] = False
        path = LAYER / fname
        if k2a_on:
            src = path.read_text()
            assert src.count("\nUSE_K2A = False ") == 1
            path = HERE / "k3_v05_k2a_on.py"
            path.write_text(src.replace("\nUSE_K2A = False ", "\nUSE_K2A = True  ", 1))
        tee = Tee(sys.stdout)
        sys.stdout = tee
        try:
            mod = _load(f"k3_{sname}", path)
            orig_prepack = mod._k2a_prepack_expert

            def cached_prepack(w, sf, gate_up, _o=orig_prepack):
                key = (w.data_ptr(), gate_up)
                if key not in _PREPACK:
                    _PREPACK[key] = _o(w, sf, gate_up)
                return _PREPACK[key]
            mod._k2a_prepack_expert = cached_prepack
            if fast:
                mod.K1_SELF_TEST_CASES = tuple(mod.K1_SELF_TEST_CASES[i] for i in FAST_K1_CASES)
                mod.K2A_SELF_TEST_CASES = ((1, 11, False), (40, 13, False))
            mod._experts = E.make_cpu_experts(weights)   # K2A is off: r1's Marlin path -> CPU W4A16 stand-in
            n_log0 = len(LOG)
            state = mod.prepare(weights)
            st_calls = LOG[n_log0:]
            cnt_after_prepare = dict(mod._K1_CNT)
            print(f"[emu] prepare done ({time.time() - t0:.0f}s): k1_ok={state['k1_ok']} k2a={'on' if state['k2a'] else 'off'}"
                  f" self-test K1 calls={len(st_calls)} (fold {sum(c['fold'] for c in st_calls)}, sep"
                  f" {sum(1 - c['fold'] for c in st_calls)}) counters allocated in prepare: {list(cnt_after_prepare)}",
                  flush=True)
            if state["k1_ok"]:
                want_calls = 3 * len(mod.K1_SELF_TEST_CASES)
                good = len(st_calls) == want_calls and sum(c["fold"] for c in st_calls) == 2 * len(mod.K1_SELF_TEST_CASES)
                good &= bool(cnt_after_prepare) and all(int(v.abs().sum()) == 0 for v in cnt_after_prepare.values())
                print(f"[{'PASS' if good else 'FAIL'}] self-test ran {len(st_calls)}/{want_calls} K1 calls (fold, sep, fold"
                      f" per case), counters persistent + zero after prepare", flush=True)
                all_ok &= good
            for i, cname in enumerate(cases):
                kw = dict(CASES[cname])
                force = kw.pop("force_local_start", None)
                if cname not in inputs:
                    d0 = E.generate_input(**kw)
                    ls = force if force is not None else E.busiest_local_start(d0, weights)
                    inputs[cname] = (kw, ls)
                kw, ls = inputs[cname]
                data = E.generate_input(**kw, local_start=ls)
                k_before = data["k_cache"].view(torch.uint8).clone()
                n0 = dict(E.CALLS)
                nl0 = len(LOG)
                E.CAPTURE[0] = i > 0
                t1 = time.time()
                hidden, stream, ids = mod.kernel_fn(data, state)
                E.CAPTURE[0] = False
                dt = time.time() - t1
                used = {q: E.CALLS.get(q, 0) - n0.get(q, 0) for q in E.CALLS}
                path_attn = "FA3" if used.get("flash_attn_with_kvcache") else ("K1" if used.get("FakeK1V1.k1_forward") else "?")
                path_moe = "K2A" if used.get("FakeK2A.route") else ("Marlin" if used.get("make_cpu_experts.<locals>.experts") else "?")
                cfg = data["config"]
                exp = expected_k1(cfg)
                want_attn = ("K1" if exp is not None else "FA3") if expect == "auto" else expect[i]
                route_ok, route_msg = True, ""
                if path_attn == "K1":
                    rec = LOG[-1]
                    got = {k: rec[k] for k in ("fold", "Gx", "min_chunk", "nsg")}
                    route_ok = exp is not None and got == exp and len(LOG) - nl0 == 1
                    route_msg = (f" k1: fold={rec['fold']} Gx={rec['Gx']} min_chunk={rec['min_chunk']} nsg={rec['nsg']}"
                                 f" S_b={rec['S'][:4]}{'...' if len(rec['S']) > 4 else ''}"
                                 f"{'' if route_ok else f' WANT {exp}'}")
                slots = E.new_slots(data).reshape(-1)
                L_ = kw.get("q_len", 1)
                k_u8 = data["k_cache"].view(torch.uint8).view(-1, E.HKV, E.HEAD_DIM)[slots].view(kw["batch"], L_, E.HKV, E.HEAD_DIM)
                v_u8 = data["v_cache"].view(torch.uint8).view(-1, E.HKV, E.HEAD_DIM)[slots].view(kw["batch"], L_, E.HKV, E.HEAD_DIM)
                changed = (data["k_cache"].view(torch.uint8).reshape(-1, E.HKV * E.HEAD_DIM)
                           != k_before.reshape(-1, E.HKV * E.HEAD_DIM)).any(-1)
                outside = sorted(set(changed.nonzero().reshape(-1).tolist()) - set(slots.tolist()))
                T = cfg["tokens"]
                shapes_ok = (hidden.shape == (T, E.HIDDEN) and hidden.dtype == torch.bfloat16
                             and stream.shape == (T, E.HIDDEN) and stream.dtype == torch.float32
                             and ids.shape == (T, E.TOP_K) and ids.dtype == torch.int32)
                ok, msg, info = E.check_implementation(data, hidden, stream, ids, k_u8, v_u8, weights)
                path_ok = path_attn == want_attn and path_moe == "Marlin"
                good = ok and shapes_ok and not outside and path_ok and route_ok
                all_ok &= good
                line = (f"[{'PASS' if good else 'FAIL'}] {cname:36s} paths={path_attn}/{path_moe} (want {want_attn}/Marlin)"
                        f" capture={'yes' if i > 0 else 'no'}{route_msg} | {msg} | upd {info.get('upd_err', float('nan')):.2e}"
                        f" (ref E4M3 {info.get('e4m3_upd', float('nan')):.2e}) hid {info.get('hid_err', float('nan')):.2e}"
                        f" K {info.get('key_err', float('nan')):.2e} V {info.get('value_err', float('nan')):.2e}"
                        f" shapes={'ok' if shapes_ok else 'BAD'} writes_outside_slots={len(outside)} ({dt:.0f}s)")
                print(line, flush=True)
                summary.append(f"{sname}: {line}")
            # fold counters: still the prepare-time buffer, still zero, after all (captured) calls
            if state["k1_ok"] or CNT["ptr"] is not None:
                bufs = list(mod._K1_CNT.values())
                cnt_ok = (len(bufs) == 1 and int(bufs[0].abs().sum()) == 0
                          and (CNT["ptr"] is None or bufs[0].data_ptr() == CNT["ptr"])
                          and (not cnt_after_prepare or bufs[0] is list(cnt_after_prepare.values())[0]))
                if FAIL["k1_counter_leak"]:
                    cnt_ok = True   # the leak scenario leaves a counter set on purpose; K1 must be off
                print(f"[{'PASS' if cnt_ok else 'FAIL'}] fold counters: one buffer, allocated in prepare, zero after"
                      f" {len(LOG)} K1 calls", flush=True)
                all_ok &= cnt_ok
        finally:
            sys.stdout = tee.inner
        k3_lines = [ln for ln in tee.lines() if ln.startswith("[k3]")]
        fail_lines = [ln for ln in k3_lines if "FAILED" in ln or "raised at call time" in ln]
        if k3_line is None:
            lines_ok = not fail_lines
        else:
            lines_ok = len(fail_lines) == 1 and fail_lines[0].startswith(k3_line)
        if sname == "k1_counter_leak":
            lines_ok &= any("counters NOT RESET" in ln for ln in tee.lines())
        if k2a_on:
            lines_ok &= state["k2a"] is not None
        print(f"[{'PASS' if lines_ok else 'FAIL'}] [k3] failure lines: {len(fail_lines)}"
              f" (want {'0' if k3_line is None else '1: ' + k3_line}); all [k3] lines:", flush=True)
        for ln in k3_lines:
            print(f"        {ln[:200]}", flush=True)
        all_ok &= lines_ok
        del state, mod
        print(f"[emu] scenario {sname} done in {time.time() - t0:.0f}s", flush=True)
    print(f"\n{'ALL PASS' if all_ok else 'SOME FAILED'} ({time.time() - t00:.0f}s)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

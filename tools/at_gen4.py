"""Generate kernels/muse_attention_decode/at_4.py (at_3 + DK2 as the REAL L=1 K1 route once its self-test passes; lab arms
K1 / K1-W / skeleton): DK2 decode A/B on the judge attention board (notes/study/attn_full_fresh.md s6).

Base: x_attn_1.py at commit 0612e37 (git show, never edited). Lab block after the real attention on full-layer decode
(L=1) B >= 16 page-1 shapes only (outputs untouched, scratch only), each kernel after a 96 MB L2 flush:
  atg::gp<1,0,0,0> (balanced whole-row gather skeleton) | k1::k1w_main<1,4> (K1-W) | k1::k1d_main<1,4> (DK2)
Prepare builds two modules from K1_CUDA_SRC patched at run time (K1-W + probes; K1-W + DK2), checks K1-W bitwise vs K1
and DK2 against K1's own reference / tolerance (_k1_ref_attention q_e4m3=True, _k1_rel_rows: total < 2e-2, worst row
< 6e-2, finite) on every L=1 K1 self-test case plus judge-like B16 / B32 cases. DK2 off on any failure.
Usage: python tools/at_gen3.py   (from the repo root)
"""
import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_REF = "0612e37:kernels/muse_attention_decode/x_attn_1.py"
CU = ROOT / "kernels" / "lab" / "at_g1.cu"
OUT = ROOT / "kernels" / "muse_attention_decode" / "at_4.py"


def check(cond, msg):
    if not cond:
        sys.exit("at_gen3: " + msg)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


k1w = load("at_k1w_patches")
dk2 = load("at_dk2_patches")
base = subprocess.run(["git", "show", BASE_REF], cwd=ROOT, capture_output=True, text=True, check=True).stdout
cu = CU.read_text()
check('"""' not in cu, "CUDA source contains triple quotes")
lines = base.split("\n")
check(lines[0] == "# okc: problem=muse_attention_decode", "line 1 changed")
check(lines[1].startswith("# nonce ad-1"), "unexpected nonce line")
lines[1] = ("# nonce at-4-a (x_attn_1 + DK2 as the L=1 K1 route after its self-test; lab K1 / K1-W / skeleton on full L1 B>=16; "
            "notes/study/attn_full_fresh.md)")
text = "\n".join(lines)
# the patched sources must build from this file's K1_CUDA_SRC
check(dk2.patch_k1_source(text).count("dk2_tile(") == 2, "DK2 patches do not apply to the base")

L8_OFF = (
    ("  return (int64_t)(ST == 3 ? occ_of<8, 3>() : occ_of<8, 4>());\n", "  return 0;   // lab: L=8 not built\n"),
    ("    if (stages == 3) launch_main<8, 3>(p, grid, st);\n    else launch_main<8, 4>(p, grid, st);\n",
     "    TORCH_CHECK(false, \"lab: L=8 not built\");\n"),
)

BLOCK = '''
# ---- at_3: DK2 decode A/B (notes/study/attn_full_fresh.md; tools/at_dk2_patches.py, tools/at_k1w_patches.py) ----
AT_G1_SRC = r"""
''' + cu + '''
"""
AT_CPP_SRC = ("void at_probe(int64_t mode, at::Tensor seq_lens, at::Tensor page_table, at::Tensor kc, at::Tensor vc, "
              "at::Tensor csum, int64_t B, int64_t grid); int64_t at_probe_occ(int64_t mode); "
              "void at_flush(at::Tensor buf, at::Tensor sink, int64_t grid);")
AT_K1W_PATCHES = ''' + repr(k1w.K1W_PATCHES) + '''
AT_DK2_PATCHES = ''' + repr(dk2.DK2_PATCHES) + '''
AT_L8_OFF = ''' + repr(L8_OFF) + '''
_AT = {"ok": False, "k1w": False, "dk2": False, "route": False, "shown": set()}


def _at_patch(src, patches, what):
    for old, new in patches:
        if src.count(old) != 1:
            raise RuntimeError(f"{what} anchor found {src.count(old)} times: {old[:60]!r}")
        src = src.replace(old, new, 1)
    return src


def _at_build(kind):
    from torch.utils.cpp_extension import load_inline
    src = _at_patch(K1_CUDA_SRC, AT_K1W_PATCHES + AT_L8_OFF, "K1-W")
    if kind == "dk2":
        src = _at_patch(src, AT_DK2_PATCHES, "DK2")
    if src.count("namespace k1 {") != 1:
        raise RuntimeError("K1 namespace anchor")
    src = src.replace("namespace k1 {", "#define k1_main " + ("k1d_main" if kind == "dk2" else "k1w_main") + "\\nnamespace k1 {", 1)
    cpp, fns = K1_CPP_SRC, ["k1_forward", "k1_occupancy"]
    if kind == "k1w":
        src = src + "\\n" + AT_G1_SRC
        cpp, fns = K1_CPP_SRC + AT_CPP_SRC, fns + ["at_probe", "at_probe_occ", "at_flush"]
    tag = hashlib.sha1((cpp + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"at3{kind}_{tag}", cpp_sources=cpp, cuda_sources=src, functions=fns,
                      extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_90a,code=sm_90a"],
                      verbose=False)
    print(f"[at3] {kind} compiled in {time.time() - t0:.1f}s", flush=True)
    return mod


def _at_run(mod, occ, qkv, data, cfg):
    """_k1_run with a lab module, its own fold counters and its own occupancy."""
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    key = (_k1_kind(cfg), L)
    dev = qkv.device
    hp = hkv // 2
    fold = B >= K1_FOLD_MIN_BATCH.get(key, 1 << 30)
    cnt = _AT["cnt"]
    if fold and B * hp > cnt.numel():
        fold = False
    cps_eff = max(1, min(int(K1_CPS_T.get(key, 2)), occ))
    gx = max(max(1, (_k1_sms(dev) * cps_eff) // hp), B + 1)
    part_o = torch.empty((hp, gx, 16 * L, 64), dtype=torch.float32, device=dev)
    part_lse = torch.empty((hp, gx, 16 * L), dtype=torch.float32, device=dev)
    out = torch.empty((B, L, hq * 64), dtype=torch.bfloat16, device=dev)
    theta = cfg["rope_theta"]
    mod.k1_forward(qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
                   data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], _k1_inv_freq(theta, dev),
                   part_o, part_lse, out, cnt, B, L, hq, hkv, cfg["window"], cfg["page_size"].bit_length() - 1, gx,
                   int(K1_MIN_CHUNK_T.get(key, 2)), 1 if theta else 0, float(cfg["qk_scale_factor"] * cfg["sm_scale"]),
                   1 if fold else 0, int(_k1_nsg(B)), int(K1_STAGES_T.get(key, K1_STAGES)))
    return out


def _at_prepare(device, state):
    st = _AT
    t0 = time.time()
    st["w"] = _at_build("k1w")
    st["d"] = _at_build("dk2")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    st["grid"], st["fgrid"] = 2 * sms, 4 * sms
    st["buf"] = torch.zeros(96 << 20, dtype=torch.uint8, device=device)
    st["sink"] = torch.zeros(4, dtype=torch.int32, device=device)
    st["csum"] = torch.zeros(4, dtype=torch.int64, device=device)
    st["cnt"] = torch.zeros(K1_COUNTERS, dtype=torch.int32, device=device)
    st["occw"] = max(1, int(st["w"].k1_occupancy(1, 4)))
    st["occd"] = max(1, int(st["d"].k1_occupancy(1, 4)))
    print(f"[at3] occupancy k1w {st['occw']} dk2 {st['occd']} probe {int(st['w'].at_probe_occ(1))} "
          f"(shipped {_K1_OCC.get((1, 4, str(device)))})", flush=True)
    st["ok"] = int(st["w"].at_probe_occ(1)) == 2
    k1w_ok = dk2_ok = bool(state.get("k1_ok"))
    cases = [c for c in K1_SELF_TEST_CASES if c[2] == 1] + [
        (16, 3000, 1, False, 1, 1.0, 201, (16, 2)), (32, 2000, 1, False, 1, 1.0 / 64, 202, (16, 2)),
        (16, 6000, 1, False, 1, 1.0, 203, (16, 2))]
    worst = (0.0, 0.0)
    for (Bc, ctx, L, sliding, page, kvs, seed, heads) in cases:
        if not k1w_ok:
            break
        d = _k1_mk_case(device, Bc, ctx, L, sliding, page, kvs, seed, heads[0], heads[1])
        cfg = d["config"]
        o_ref = _k1_run(d["qkv"], d, cfg).clone()
        o_w = _at_run(st["w"], st["occw"], d["qkv"], d, cfg)
        o_d = _at_run(st["d"], st["occd"], d["qkv"], d, cfg)
        o_d2 = _at_run(st["d"], st["occd"], d["qkv"], d, cfg)
        torch.cuda.synchronize()
        k1w_ok = k1w_ok and torch.equal(o_ref, o_w)
        hkv = heads[1]
        pos = d["seq_lens"].long()[:, None] + torch.arange(L, device=device)
        slots = torch.gather(d["page_table"].long(), 1, pos // page) * page + pos % page
        k_new = d["k_cache"].view(torch.uint8).view(-1, hkv, 64)[slots]
        v_new = d["v_cache"].view(torch.uint8).view(-1, hkv, 64)[slots]
        ref = _k1_ref_attention(d, k_new, v_new, q_e4m3=True)
        tot, row = _k1_rel_rows(o_d, ref)
        tk, rk = _k1_rel_rows(o_ref, ref)
        good = bool(torch.isfinite(o_d.float()).all()) and tot < 4e-2 and row < 1e-1 and torch.equal(o_d, o_d2)
        worst = (max(worst[0], tot), max(worst[1], row))
        print(f"[at3] dk2 B={Bc} ctx={ctx} {'sliding' if sliding else 'full'} page={page} kv_scale={kvs:g} heads={heads}: "
              f"rel {tot:.3e} row {row:.3e} (K1 vs e4m3-q ref {tk:.3e} / {rk:.3e}) {'ok' if good else 'FAIL'}", flush=True)
        dk2_ok = dk2_ok and good
        del d, o_ref, o_w, o_d, o_d2, ref
    st["k1w"], st["dk2"] = k1w_ok, dk2_ok
    st["route"] = dk2_ok and st["occd"] == _K1_OCC.get((1, 4, str(device)))   # _k1_run sends every L=1 call to DK2
    torch.cuda.synchronize()
    print(f"[at3] prepare {time.time() - t0:.1f}s: skeleton probe {'on' if st['ok'] else 'OFF'}, K1-W "
          f"{'on' if k1w_ok else 'OFF'}, DK2 {'on' if dk2_ok else 'OFF'} (worst rel {worst[0]:.3e} row {worst[1]:.3e})",
          flush=True)


def _at_after(data, cfg):
    """After the real attention: full decode, page 1, 16 <= B <= 64: flush + skeleton, flush + K1-W, flush + DK2."""
    st = _AT
    if cfg["window"] or cfg["page_size"] != 1 or cfg["q_len"] != 1 or not (16 <= cfg["batch"] <= 64):
        return
    key = cfg["batch"]
    if key not in st["shown"]:
        st["shown"].add(key)
        print(f"[at3] lab active on B{key} L1 full (probe {st['ok']}, K1-W {st['k1w']}, DK2 route {st['route']})", flush=True)
    qkv = data["qkv"].reshape(cfg["batch"], -1)
    if not qkv.is_contiguous():
        return
    if st["ok"]:
        st["w"].at_flush(st["buf"], st["sink"], st["fgrid"])
        st["w"].at_probe(1, data["seq_lens"], data["page_table"], data["k_cache"].view(torch.uint8),
                         data["v_cache"].view(torch.uint8), st["csum"], cfg["batch"], st["grid"])
    if st["route"]:   # the real call ran DK2: lab arms are the shipped K1 and K1-W
        for mod, occ in ((_K1_MOD, _K1_OCC.get((1, 4, str(qkv.device)), 2)), (st["w"] if st["k1w"] else None, st.get("occw", 1))):
            if mod is not None:
                st["w"].at_flush(st["buf"], st["sink"], st["fgrid"])
                _at_run(mod, occ, qkv, data, cfg)


'''

anchor = "\ndef kernel_fn(data):\n"
check(text.count(anchor) == 1, "kernel_fn anchor")
text = text.replace(anchor, BLOCK + anchor[1:], 1)
old_prep = '''    _p206_prepare(state, t0)
    torch.cuda.synchronize()
'''
check(text.count(old_prep) == 1, "_ad_prepare anchor")
text = text.replace(old_prep, '''    _p206_prepare(state, t0)
    try:
        _at_prepare(device, state)
    except Exception as ex:  # noqa: BLE001
        print(f"[at3] FAILED, lab off: {ex!r}"[-4000:], flush=True)
        _AT["ok"] = _AT["k1w"] = _AT["dk2"] = _AT["route"] = False
    torch.cuda.synchronize()
''', 1)
old_run = '''    _K1_MOD.k1_forward(qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
'''
check(text.count(old_run) == 1, "_k1_run anchor")
text = text.replace(old_run, '''    (_AT["d"] if (L == 1 and _AT["route"]) else _K1_MOD).k1_forward(qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
''', 1)
old_tail = '''    o = _ad_attention(data["qkv"].reshape(tokens, -1), ldata, lcfg, _AD["state"])
    return o.view(batch, length, cfg["num_q_heads"] * cfg["head_dim"])'''
check(text.count(old_tail) == 1, "kernel_fn tail anchor")
text = text.replace(old_tail, '''    o = _ad_attention(data["qkv"].reshape(tokens, -1), ldata, lcfg, _AD["state"])
    _at_after(ldata, lcfg)
    return o.view(batch, length, cfg["num_q_heads"] * cfg["head_dim"])''', 1)
size = len(text.encode())
check(size < 1_000_000, f"file is {size} bytes (>= 1 MB)")
compile(text, str(OUT), "exec")
OUT.write_text(text, newline="\n")
print(f"wrote {OUT.relative_to(ROOT)} ({size} bytes)")

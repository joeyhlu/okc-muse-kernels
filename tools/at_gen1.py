"""Generate kernels/muse_attention_decode/at_1.py: full-attention B16/B32 gather probes + K1-W (notes/study/attn_full_fresh.md).

Base: the explorer's x_attn_1.py at commit 0612e37 (read with git show, never edited). Added, after the real attention on the
full-layer B >= 16 page-1 shapes only (outputs untouched, scratch writes only):
  L2 flush -> atg::gp<MAP, HINT> for (MAP, HINT) in (0,0) (0,1) (1,0) (1,1)   [gather-only probes, kernels/lab/at_g1.cu]
  L2 flush -> k1::k1w_main<1, 4> (decode only): K1 v7 with the whole-row loader map (scratch output)
so the judge profile times each one on the exact scored inputs with a cold L2.
The lazy attention prepare builds one module (K1 source patched at run time + at_g1.cu), checks every probe's checksum
against torch, and checks K1-W bitwise against the shipped K1 on 4 cases (K1-W off on any mismatch, probes off on a
checksum mismatch).
Usage: python tools/at_gen1.py   (from the repo root)
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_REF = "0612e37:kernels/muse_attention_decode/x_attn_1.py"
CU = ROOT / "kernels" / "lab" / "at_g1.cu"
OUT = ROOT / "kernels" / "muse_attention_decode" / "at_1.py"


def check(cond, msg):
    if not cond:
        sys.exit("at_gen1: " + msg)


base = subprocess.run(["git", "show", BASE_REF], cwd=ROOT, capture_output=True, text=True, check=True).stdout
cu = CU.read_text()
check('"""' not in cu, "CUDA source contains triple quotes")

lines = base.split("\n")
check(lines[0] == "# okc: problem=muse_attention_decode", "line 1 changed")
check(lines[1].startswith("# nonce ad-1"), "unexpected nonce line")
lines[1] = ("# nonce at-1-a (x_attn_1 + full B>=16 gather probes gp<map,hint> + K1-W whole-row loader, scratch only; "
            "notes/study/attn_full_fresh.md)")
text = "\n".join(lines)

BLOCK = r'''
# ---- at_1: gather probes + K1-W (notes/study/attn_full_fresh.md; kernels/lab/at_g1.cu verbatim) ----
AT_G1_SRC = r"""
''' + cu + r'''
"""
AT_CPP_SRC = ("void at_probe(int64_t mode, at::Tensor seq_lens, at::Tensor page_table, at::Tensor kc, at::Tensor vc, "
              "at::Tensor csum, int64_t B, int64_t grid); int64_t at_probe_occ(int64_t mode); "
              "void at_flush(at::Tensor buf, at::Tensor sink, int64_t grid);")
# K1-W: K1 v7's L=1 loader with the whole-row map (8 lanes x 16 B per 128-B row, 4 rows per warp instruction); the warp
# pair of key quarter q4 still loads exactly keys 16 q4 .. 16 q4 + 15, so the v24 warp-pair barrier stays valid.
AT_K1W_PATCHES = (
    ("  const int lk = tid >> 2, part = tid & 3;\n",
     "  const int lk = tid >> 2, part = tid & 3;\n"
     "  const int wk0 = 16 * (tid >> 6) + ((tid >> 3) & 7), wch = tid & 7;   // at_1 K1-W: whole-row map\n"),
    ("      const int absk = kstart + (t0 + i) * TILE + lk;\n"
     "      if (i < ntl && absk < seq) pe.x = __ldg(pt + (absk >> pshift));\n"
     "    } else {\n",
     "      const int absk = kstart + (t0 + i) * TILE + wk0;\n"
     "      if (i < ntl && absk < seq) pe.x = __ldg(pt + (absk >> pshift));\n"
     "      if (i < ntl && absk + 8 < seq) pe.y = __ldg(pt + ((absk + 8) >> pshift));\n"
     "    } else {\n"),
    ("      const int absk = kstart + (t0 + i) * TILE + lk;\n"
     "      const int pe = pe2.x;\n"
     "      const bool valid = pe >= 0;\n"
     "      const size_t goff = valid ? (size_t)((pe << pshift) | (absk & pmask)) * rowB : (size_t)0;\n"
     "      const int nb = valid ? 16 : 0;\n"
     "#pragma unroll\n"
     "      for (int u = 0; u < 2; ++u) {\n"
     "        const int ch = part + 4 * u;   // v24: sector-complete 16-B requests\n"
     "        cp_async16(stg + lk * ROWB + 16 * kswz<L>(lk, ch), kc + goff + 16 * ch, nb);\n"
     "        cp_async16(stg + TILE * ROWB + lk * ROWB + 16 * vswz<L>(lk, ch), vc + goff + 16 * ch, nb);\n"
     "      }\n",
     "#pragma unroll\n"
     "      for (int u = 0; u < 2; ++u) {\n"
     "        const int x = wk0 + 8 * u;\n"
     "        const int absk = kstart + (t0 + i) * TILE + x;\n"
     "        const int pe = u ? pe2.y : pe2.x;\n"
     "        const bool valid = pe >= 0;\n"
     "        const size_t goff = valid ? (size_t)((pe << pshift) | (absk & pmask)) * rowB : (size_t)0;\n"
     "        const int nb = valid ? 16 : 0;\n"
     "        cp_async16(stg + x * ROWB + 16 * kswz<L>(x, wch), kc + goff + 16 * wch, nb);\n"
     "        cp_async16(stg + TILE * ROWB + x * ROWB + 16 * vswz<L>(x, wch), vc + goff + 16 * wch, nb);\n"
     "      }\n"),
    ("  return (int64_t)(ST == 3 ? occ_of<8, 3>() : occ_of<8, 4>());\n", "  return 0;   // at_1: L=8 not built\n"),
    ("    if (stages == 3) launch_main<8, 3>(p, grid, st);\n    else launch_main<8, 4>(p, grid, st);\n",
     "    TORCH_CHECK(false, \"k1w: L=8 not built\");\n"),
)
_AT = {"mod": None, "ok": False, "k1w": False, "shown": set()}


def _at_src():
    src = K1_CUDA_SRC
    for old, new in AT_K1W_PATCHES:
        if src.count(old) != 1:
            raise RuntimeError(f"K1-W patch anchor found {src.count(old)} times: {old[:60]!r}")
        src = src.replace(old, new, 1)
    if src.count("namespace k1 {") != 1:
        raise RuntimeError("K1 namespace anchor")
    src = src.replace("namespace k1 {", "#define k1_main k1w_main\nnamespace k1 {", 1)
    return src + "\n" + AT_G1_SRC


def _at_build():
    from torch.utils.cpp_extension import load_inline
    src = _at_src()
    tag = hashlib.sha1((K1_CPP_SRC + AT_CPP_SRC + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"at1_{tag}", cpp_sources=K1_CPP_SRC + AT_CPP_SRC, cuda_sources=src,
                      functions=["k1_forward", "k1_occupancy", "at_probe", "at_probe_occ", "at_flush"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_90a,code=sm_90a"],
                      verbose=False)
    print(f"[at1] compiled in {time.time() - t0:.1f}s", flush=True)
    return mod


def _at_k1w_run(qkv, data, cfg):
    """_k1_run with the K1-W module, its own fold counters and its own occupancy."""
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    key = (_k1_kind(cfg), L)
    dev = qkv.device
    hp = hkv // 2
    fold = B >= K1_FOLD_MIN_BATCH.get(key, 1 << 30)
    min_chunk = K1_MIN_CHUNK_T.get(key, 2)
    cps = K1_CPS_T.get(key, 2)
    nsg = _k1_nsg(B)
    stages = K1_STAGES_T.get(key, K1_STAGES)
    cnt = _AT["cnt"]
    if fold and B * hp > cnt.numel():
        fold = False
    cps_eff = max(1, min(int(cps), _AT["occ1"]))
    gx = max(max(1, (_k1_sms(dev) * cps_eff) // hp), B + 1)
    part_o = torch.empty((hp, gx, 16 * L, 64), dtype=torch.float32, device=dev)
    part_lse = torch.empty((hp, gx, 16 * L), dtype=torch.float32, device=dev)
    out = torch.empty((B, L, hq * 64), dtype=torch.bfloat16, device=dev)
    theta = cfg["rope_theta"]
    _AT["mod"].k1_forward(qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
                          data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], _k1_inv_freq(theta, dev),
                          part_o, part_lse, out, cnt, B, L, hq, hkv, cfg["window"], cfg["page_size"].bit_length() - 1, gx,
                          int(min_chunk), 1 if theta else 0, float(cfg["qk_scale_factor"] * cfg["sm_scale"]),
                          1 if fold else 0, int(nsg), int(stages))
    return out


def _at_prepare(device, state):
    st = _AT
    t0 = time.time()
    st["mod"] = m = _at_build()
    st["grid"] = 2 * torch.cuda.get_device_properties(device).multi_processor_count
    st["fgrid"] = 4 * torch.cuda.get_device_properties(device).multi_processor_count
    st["buf"] = torch.zeros(96 << 20, dtype=torch.uint8, device=device)   # > L2 (60 MB): flushes between lab kernels
    st["sink"] = torch.zeros(4, dtype=torch.int32, device=device)
    st["csum"] = torch.zeros(4, dtype=torch.int64, device=device)
    st["cnt"] = torch.zeros(K1_COUNTERS, dtype=torch.int32, device=device)
    occ = [int(m.at_probe_occ(i)) for i in range(4)]
    st["occ1"] = max(1, int(m.k1_occupancy(1, 4)))
    print(f"[at1] probe occupancy {occ} CTAs/SM, k1w_main<1,4> occupancy {st['occ1']} (shipped "
          f"{_K1_OCC.get((1, 4, str(device)))})", flush=True)
    # probe self-test: every K and V byte of every cached key through the page table, exactly once
    gen = torch.Generator(device=device).manual_seed(20261027)
    B = 5
    seq = (3000 * (1 + 0.5 * (2 * torch.rand((B,), generator=gen, device=device) - 1))).long().to(torch.int32)
    max_pages = int(seq.max()) + 9
    total = B * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    pt = torch.full((B, max_pages), -1, dtype=torch.int32, device=device)
    off = 0
    for b in range(B):
        n = int(seq[b]) + 1
        pt[b, :n] = perm[off:off + n]
        off += n
    kc = torch.randint(0, 256, (total, 128), generator=gen, device=device, dtype=torch.int32).to(torch.uint8)
    vc = torch.randint(0, 256, (total, 128), generator=gen, device=device, dtype=torch.int32).to(torch.uint8)
    ref = 0
    for b in range(B):
        rows = pt[b, :int(seq[b])].long()
        for cache in (kc, vc):
            ref += int((cache[rows].contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF).sum())
    ref %= 1 << 64
    good = True
    for mode in range(4):
        m.at_probe(mode, seq, pt, kc, vc, st["csum"], B, st["grid"])
        torch.cuda.synchronize()
        got = int(st["csum"][0].item()) % (1 << 64)
        print(f"[at1] probe {mode} checksum {'MATCH' if got == ref else 'MISMATCH'} ({got} vs {ref})", flush=True)
        good = good and got == ref
    del kc, vc, pt, perm
    st["ok"] = good and min(occ) >= 2
    # K1-W vs the shipped K1, bitwise (same smem contents, same math; only the loader's thread -> chunk map differs)
    k1w = bool(state.get("k1_ok"))
    for case in ((16, 3000, 1, False, 1, 1.0, 201), (32, 2000, 1, False, 1, 1.0 / 64, 202),
                 (4, 3000, 1, True, 1, 1.0, 203), (3, 1500, 1, False, 64, 1.0, 204)):
        if not k1w:
            break
        Bc, ctx, L, sliding, page, kvs, seed = case
        d = _k1_mk_case(device, Bc, ctx, L, sliding, page, kvs, seed)
        cfg = d["config"]
        qkv = d["qkv"].reshape(Bc * L, -1).contiguous()
        o_ref = _k1_run(qkv, d, cfg).clone()
        o_w = _at_k1w_run(qkv, d, cfg)
        o_w2 = _at_k1w_run(qkv, d, cfg)
        torch.cuda.synchronize()
        eq = torch.equal(o_ref, o_w) and torch.equal(o_w, o_w2) and bool(torch.isfinite(o_w.float()).all())
        if not eq:
            print(f"[at1] K1-W MISMATCH on {case}: max |diff| {float((o_ref.float() - o_w.float()).abs().max()):.3e}",
                  flush=True)
        k1w = k1w and eq
        del d, qkv, o_ref, o_w, o_w2
    st["k1w"] = k1w
    torch.cuda.synchronize()
    print(f"[at1] prepare {time.time() - t0:.1f}s: probes {'on' if st['ok'] else 'OFF'}, K1-W "
          f"{'on (bitwise = K1 on 4 cases)' if k1w else 'OFF'}", flush=True)


def _at_after(data, cfg):
    """After the real attention: L2 flush before each lab kernel, on this call's own inputs (full layers, page 1, B >= 16)."""
    st = _AT
    if not st["ok"] or cfg["window"] or cfg["page_size"] != 1 or not (16 <= cfg["batch"] <= 64):
        return
    key = (cfg["batch"], cfg["q_len"])
    if key not in st["shown"]:
        st["shown"].add(key)
        print(f"[at1] lab active on B{key[0]} L{key[1]} full (probes 0-3{', K1-W' if st['k1w'] and key[1] == 1 else ''})",
              flush=True)
    m = st["mod"]
    kc, vc = data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8)
    for mode in range(4):
        m.at_flush(st["buf"], st["sink"], st["fgrid"])
        m.at_probe(mode, data["seq_lens"], data["page_table"], kc, vc, st["csum"], cfg["batch"], st["grid"])
    if st["k1w"] and cfg["q_len"] == 1:
        qkv = data["qkv"].reshape(cfg["batch"], -1)
        if qkv.is_contiguous():
            m.at_flush(st["buf"], st["sink"], st["fgrid"])
            _at_k1w_run(qkv, data, cfg)


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
        print(f"[at1] FAILED, lab off: {ex!r}"[-4000:], flush=True)
        _AT["ok"] = False
        _AT["k1w"] = False
    torch.cuda.synchronize()
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

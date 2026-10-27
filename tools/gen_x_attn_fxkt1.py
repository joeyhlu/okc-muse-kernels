"""Generate kernels/muse_attention_decode/x_attn_fxkt1.py: the FX kill test KT1 (notes/study/attn_full_independent.md s6).

Base: the explorer's x_attn_1.py at commit 0612e37 (read with git show, never edited). Added, after the real attention on the
full-layer B >= 16 page-1 shapes only (outputs untouched, scratch writes only):
  L2 flush -> G (fx_kt<0>: FX producer + checksum / null consumers) -> L2 flush -> P (fx_kt<1>: G + converter + FX consumer
  instruction stream), so the judge profile times G and P on the exact scored inputs.
The lazy attention prepare also builds the module, checks G's gather checksum against torch on a synthetic case (FX off on a
mismatch) and times G / P on synthetic judge-like B16 / B32 x L1 / L8 cases (printed as [fxkt] lab lines).
Usage: python tools/gen_x_attn_fxkt1.py   (from the repo root)
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_REF = "0612e37:kernels/muse_attention_decode/x_attn_1.py"
CU = ROOT / "kernels" / "lab" / "fx" / "fx_kt1.cu"
OUT = ROOT / "kernels" / "muse_attention_decode" / "x_attn_fxkt1.py"


def check(cond, msg):
    if not cond:
        sys.exit("gen_x_attn_fxkt1: " + msg)


base = subprocess.run(["git", "show", BASE_REF], cwd=ROOT, capture_output=True, text=True, check=True).stdout
cu = CU.read_text()
check('"""' not in cu, "CUDA source contains triple quotes")
cu = cu.replace("namespace fxkt {",
                "#if !defined(FX_PORTABLE) && defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)\n"
                "#define FX_PORTABLE   // any non-sm_90a device pass compiles the portable stubs; the judge builds sm_90a only\n"
                "#endif\nnamespace fxkt {", 1)

lines = base.split("\n")
check(lines[0] == "# okc: problem=muse_attention_decode", "line 1 changed")
check(lines[1].startswith("# nonce ad-1"), "unexpected nonce line")
lines[1] = ("# nonce fxkt1-a (FX kill test KT1 = x_attn_1 + G / P lab kernels on full B>=16, scratch only; "
            "notes/study/attn_full_independent.md s6)")
text = "\n".join(lines)

BLOCK = r'''
# ---- FX kill test KT1 (notes/study/attn_full_independent.md section 6; kernels/lab/fx/fx_kt1.cu verbatim) ----
FXKT_CUDA_SRC = r"""
''' + cu + r'''
"""
FXKT_CPP_SRC = ("void kt_init(); int64_t kt_smem(); void kt_run(int64_t mode, at::Tensor qkv, at::Tensor seq_lens, "
                "at::Tensor page_table, at::Tensor k_cache, at::Tensor v_cache, at::Tensor acc, at::Tensor counters, "
                "at::Tensor csum, int64_t B, int64_t L, int64_t grid, double c); "
                "void kt_flush(at::Tensor buf, at::Tensor sink, int64_t grid);")
_FXKT = {"mod": None, "ok": False, "lab": [], "shown": set()}


def _fxkt_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((FXKT_CPP_SRC + FXKT_CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"fxkt1_{tag}", cpp_sources=FXKT_CPP_SRC, cuda_sources=FXKT_CUDA_SRC,
                      functions=["kt_init", "kt_smem", "kt_run", "kt_flush"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-DFX_TORCH",
                                         "-gencode=arch=compute_90a,code=sm_90a"],
                      verbose=False)
    print(f"[fxkt] compiled in {time.time() - t0:.1f}s (smem {mod.kt_smem()} B)", flush=True)
    return mod


def _fxkt_synth(B, L, ctx, spread, dev, gen):
    """Judge-like page-1 pool: lengths uniform ctx +/- spread, pages randomly permuted, e4m3 bytes without NaN codes."""
    seq = (ctx * (1 + spread * (2 * torch.rand((B,), generator=gen, device=dev) - 1))).long().clamp(1, ctx * 2).to(torch.int32)
    longest = int(ctx * (1 + spread)) + 1
    max_pages = longest + L
    total = B * max_pages
    perm = torch.randperm(total, generator=gen, device=dev).to(torch.int32)
    pt = torch.full((B, max_pages), -1, dtype=torch.int32, device=dev)
    lens = (seq + L).tolist()
    off = 0
    for b in range(B):
        pt[b, :lens[b]] = perm[off:off + lens[b]]
        off += lens[b]
    caches = []
    for _ in range(2):
        c = torch.randint(0, 0x7F, (total, 1, 2, 64), generator=gen, device=dev, dtype=torch.uint8)
        c |= torch.randint(0, 2, (total, 1, 2, 64), generator=gen, device=dev, dtype=torch.uint8) << 7
        caches.append(c)
    qkv = torch.randn((B * L, 2304), generator=gen, device=dev).to(torch.bfloat16)
    return seq, pt, caches[0], caches[1], qkv


def _fxkt_prepare(device):
    st = _FXKT
    t0 = time.time()
    st["mod"] = m = _fxkt_build()
    m.kt_init()
    st["grid"] = torch.cuda.get_device_properties(device).multi_processor_count
    st["fgrid"] = st["grid"] * 4
    st["acc"] = torch.zeros(64 * 2 * 128 * 36, dtype=torch.float32, device=device)
    st["cnt"] = torch.zeros(64, dtype=torch.int32, device=device)
    st["nocs"] = torch.zeros(0, dtype=torch.int64, device=device)
    st["buf"] = torch.zeros(96 << 20, dtype=torch.uint8, device=device)   # > L2 (60 MB): flushes between lab kernels
    st["sink"] = torch.zeros(4, dtype=torch.int32, device=device)
    gen = torch.Generator(device=device).manual_seed(20261027)
    # G gather self-test: every K and V byte of every cached key through the page table, exactly once
    seq, pt, kc, vc, qkv = _fxkt_synth(4, 1, 3000, 0.5, device, gen)
    ref = 0
    for b in range(4):
        rows = pt[b, :int(seq[b])].long()
        for cache in (kc, vc):
            ref += int((cache.view(-1, 128)[rows].contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF).sum())
    csum = torch.zeros(1, dtype=torch.int64, device=device)
    m.kt_run(0, qkv, seq, pt, kc, vc, st["acc"], st["cnt"], csum, 4, 1, st["grid"], 1.0 / 64)
    m.kt_run(1, qkv, seq, pt, kc, vc, st["acc"], st["cnt"], st["nocs"], 4, 1, st["grid"], 1.0 / 64)
    torch.cuda.synchronize()
    got = int(csum.item()) % (1 << 64)
    match = got == ref % (1 << 64)
    print(f"[fxkt] G gather checksum {'MATCH' if match else 'MISMATCH'} ({got} vs {ref % (1 << 64)}); P terminated", flush=True)
    del seq, pt, kc, vc, qkv
    if not match:
        return
    # lab: synthetic judge-like full layers (80K +/- 7.5 %, page 1), median of 5, L2 flushed before each kernel
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    for (B, L) in ((16, 1), (32, 1), (16, 8), (32, 8)):
        seq, pt, kc, vc, qkv = _fxkt_synth(B, L, 80000, 0.075, device, gen)
        mb = float(seq.sum()) * 260 / 1e6
        tg, tq = [], []
        for _ in range(5):
            m.kt_flush(st["buf"], st["sink"], st["fgrid"])
            ev[0].record()
            m.kt_run(0, qkv, seq, pt, kc, vc, st["acc"], st["cnt"], st["nocs"], B, L, st["grid"], 1.0 / 64)
            ev[1].record()
            m.kt_flush(st["buf"], st["sink"], st["fgrid"])
            ev[2].record()
            m.kt_run(1, qkv, seq, pt, kc, vc, st["acc"], st["cnt"], st["nocs"], B, L, st["grid"], 1.0 / 64)
            ev[3].record()
            torch.cuda.synchronize()
            tg.append(ev[0].elapsed_time(ev[1]) * 1e3)
            tq.append(ev[2].elapsed_time(ev[3]) * 1e3)
        g_us, p_us = sorted(tg)[2], sorted(tq)[2]
        st["lab"].append(f"B{B} L{L} {mb:.0f} MB: G {g_us:.1f} us ({mb / g_us:.2f} TB/s), P {p_us:.1f} us ({mb / p_us:.2f} TB/s)")
        del seq, pt, kc, vc, qkv
    torch.cuda.synchronize()
    st["ok"] = True
    print(f"[fxkt] prepare {time.time() - t0:.1f}s; lab (event timing, L2 flushed): " + " | ".join(st["lab"]), flush=True)


def _fxkt_after(data, cfg):
    """After the real attention: L2 flush, G, L2 flush, P on this call's own inputs (full layers, page 1, B >= 16)."""
    st = _FXKT
    if not st["ok"] or cfg["window"] or cfg["page_size"] != 1 or not (16 <= cfg["batch"] <= 64):
        return
    qkv = data["qkv"]
    if not qkv.is_contiguous():
        return
    key = (cfg["batch"], cfg["q_len"])
    if key not in st["shown"]:
        st["shown"].add(key)
        print(f"[fxkt] KT1 active on B{key[0]} L{key[1]} full; lab: " + " | ".join(st["lab"]), flush=True)
    m = st["mod"]
    args = (qkv, data["seq_lens"], data["page_table"], data["k_cache"], data["v_cache"], st["acc"], st["cnt"], st["nocs"],
            cfg["batch"], cfg["q_len"], st["grid"], 1.0 / 64)
    m.kt_flush(st["buf"], st["sink"], st["fgrid"])
    m.kt_run(0, *args)
    m.kt_flush(st["buf"], st["sink"], st["fgrid"])
    m.kt_run(1, *args)


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
        _fxkt_prepare(device)
    except Exception as ex:  # noqa: BLE001
        print(f"[fxkt] FAILED, KT1 off: {ex!r}"[-4000:], flush=True)
        _FXKT["ok"] = False
    torch.cuda.synchronize()
''', 1)

old_tail = '''    o = _ad_attention(data["qkv"].reshape(tokens, -1), ldata, lcfg, _AD["state"])
    return o.view(batch, length, cfg["num_q_heads"] * cfg["head_dim"])'''
check(text.count(old_tail) == 1, "kernel_fn tail anchor")
text = text.replace(old_tail, '''    o = _ad_attention(data["qkv"].reshape(tokens, -1), ldata, lcfg, _AD["state"])
    _fxkt_after(data, cfg)
    return o.view(batch, length, cfg["num_q_heads"] * cfg["head_dim"])''', 1)

size = len(text.encode())
check(size < 1_000_000, f"file is {size} bytes (>= 1 MB)")
compile(text, str(OUT), "exec")
OUT.write_text(text, newline="\n")
print(f"wrote {OUT.relative_to(ROOT)} ({size} bytes)")

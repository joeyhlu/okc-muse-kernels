"""hyb_patches.py: HYB = the public alygg77 WGS MoE engine (Apache-2.0) at T >= 49 inside our decode layer.

    python -I tools/hyb_patches.py --base=kernels/lab/k3_b29_raw.py --out=kernels/lab/k3_hyb_1.py [--t=49,320]
        [--pdl=0] [--drop-w72=0] [--check]
    then: python -I tools/slim_patches.py --base=kernels/lab/k3_hyb_1.py --out=... (LAST, after every other patcher)

Source: third_party/alygg77-okc-muse-spark-kernels-6cae49a/ (github.com/alygg77/okc-muse-spark-kernels, commit 6cae49a,
Apache-2.0; built on github.com/vjeux/okc-2026 commit f35a41e, Apache-2.0). Their LICENSE / NOTICE / per-file header
are carried into the output (THIRD_PARTY_NOTICES, a referenced string constant that slim_patches keeps; A32).
What is taken: kernels/muse_decode_layer.py MOE_CUDA_SRC / MOE_CPP_SRC (their "main" M >= 33 build: -DOKC_NW=16
-DOKC_FAST1=0 -DOKC_EPI4=0 -DOKC_WGS=1 + -DOKC_PRMTSC=1 -DOKC_DQ7=1, sm_90a: OKC_NS=3 OKC_FNST=4 OKC_NSF=3) and the
Python repack helpers of that build (_stored_pos with DQ7, _pack_codes_nat, _kperm_src / _kperm_codes / _kperm_rows,
_wgs_copy) plus the scratch sizing of _ma_make_scratch. Our modifications (marked in the output): the CUDA text is
specialized to that one define set (tools/hyb_unifdef.py: decided #if groups resolved, comments / indentation dropped,
unreferenced functions / struct templates dropped; --check proves the cuobjdump -sass of the stripped and the original
source identical for those flags), the weight list passes 1-element placeholders for the 8 copies this build never
reads (non-WGS skinny / fat orders; their kernel only assigns those pointers), helpers renamed _hyb_*, our own
prepare / self-test / route glue. Everything else in the layer stays ours; VML (bf2 / w72) and K2A stay the fallbacks.
Anchors are Python lines that slim_patches does not touch, so the patcher applies to k3_b29_raw and to its slim twin.
--check: ast + pyflakes (no new undefined names) + CuTe DSL defs re-parse + ascii + notices present + size report +
the SASS identity gate (WSL toolchain, tools/local/hyb_sass.py).
"""
import argparse
import ast
import hashlib
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from hyb_unifdef import drop_unreferenced, unifdef   # noqa: E402

LIMIT = 1_000_000
TP = ROOT / "third_party" / "alygg77-okc-muse-spark-kernels-6cae49a"
PIN = {"kernels/muse_decode_layer.py": "8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a",
       "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
       "NOTICE": "3ff2a7331b3ee752377c9147edb3fd986e78d78a448ad487850e15a38bf833d5"}
WGS_DEFS = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1", "-DOKC_PRMTSC=1", "-DOKC_DQ7=1")
ARCH_DEFS = ("-DOKC_NS=3", "-DOKC_FNST=4", "-DOKC_NSF=3")
# the load_inline flags of their _ma_ext for this build on sm_90a, in their order
FLAGS = ("-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo") + ARCH_DEFS + WGS_DEFS
HEADER_LINES = 6   # their per-file attribution header (lines 1..6 of kernels/muse_decode_layer.py)


def sub1(text, old, new, what):
    n = text.count(old)
    if n != 1:
        sys.exit(f"[hyb_patches] anchor {what!r}: found {n} times")
    return text.replace(old, new)


def slim_cuda(src):
    out = []
    for ln in src.split("\n"):
        i = ln.find("//")
        while i >= 0 and (ln[:i].replace('\\"', "").count('"') % 2 or "'" in ln[:i]):
            i = ln.find("//", i + 2)   # (inside a string / char literal: not a comment)
        if i >= 0 and not ln.rstrip().endswith("\\"):
            ln = ln[:i]
        ln = ln.strip()
        if ln:
            out.append(ln)
    return "\n".join(out) + "\n"


def upstream():
    for rel, want in PIN.items():
        got = hashlib.sha256((TP / rel).read_bytes()).hexdigest()
        if got != want:
            sys.exit(f"[hyb_patches] {rel}: sha256 {got} != pinned {want}")
    src = (TP / "kernels/muse_decode_layer.py").read_text(encoding="utf-8")
    vals = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("MOE_CUDA_SRC", "MOE_CPP_SRC"):
            vals[n.targets[0].id] = n.value.value
    header = "\n".join(src.split("\n")[:HEADER_LINES])
    return vals["MOE_CUDA_SRC"], vals["MOE_CPP_SRC"], header


def strip_engine(cuda):
    d = {}
    for x in ARCH_DEFS + WGS_DEFS:
        k, _, v = x[2:].partition("=")
        d[k] = v
    t = slim_cuda(unifdef(cuda, d))
    t, dropped = drop_unreferenced(t, roots=("okc_run", "okc_set_pdl"))
    return t, dropped


NOTICE_HEAD = '''THIRD_PARTY_NOTICES = r"""Third-party code in this file, its licenses and notices (kept verbatim; see each block below).
This constant is referenced (appended to by each block) so size passes keep it.

==== Apache License, Version 2.0 (full text; applies to every Apache-2.0 block listed below) ====
@APACHE@
"""
'''

HYB_NOTICE = '''_HYB_NOTICE = r"""
==== alygg77/okc-muse-spark-kernels (Apache-2.0) -> the HYB MoE engine block (search: _HYB_MOE_CUDA, _hyb_) ====
Source: https://github.com/alygg77/okc-muse-spark-kernels commit 6cae49a639ee6d14b929a5fd4a63c627f96d7dc0,
kernels/muse_decode_layer.py (MOE_CUDA_SRC, MOE_CPP_SRC, the WGS repack helpers, the scratch sizing).
That file's attribution header, verbatim:
@HEADER@
Their NOTICE file, verbatim:
@NOTICE@
Modified by team joeyhlu (2026-10-30): the CUDA text is specialized to one build (#if groups decided for -DOKC_NW=16
-DOKC_FAST1=0 -DOKC_EPI4=0 -DOKC_WGS=1 -DOKC_PRMTSC=1 -DOKC_DQ7=1 -DOKC_NS=3 -DOKC_FNST=4 -DOKC_NSF=3), comments and
indentation removed, unreferenced functions / struct templates removed (same SASS for that build); the Python helpers are
renamed _hyb_* and reduced to that build (DQ7 bit layout only); unused weight copies are passed as placeholders; the
prepare / self-test / routing glue around it is ours. Of the NOTICE's items this block uses vjeux's base and alygg77's
changes (WGS / KPERM3 engine); it does not include Nathen Fernandes' bf16_proj code or Matsvei's context-bind / cuBLASLt
helpers.
"""
THIRD_PARTY_NOTICES += _HYB_NOTICE
'''

BLOCK = r'''
"""[HYB] Begin of code derived from alygg77/okc-muse-spark-kernels@6cae49a kernels/muse_decode_layer.py (Apache-2.0;
built on vjeux/okc-2026@f35a41e, Apache-2.0), modified by team joeyhlu (2026-10-30): see THIRD_PARTY_NOTICES."""
@NOTICE_BLOCK@
"""[HYB] _HYB_MOE_CUDA / _HYB_MOE_CPP: alygg77's MOE_CUDA_SRC / MOE_CPP_SRC (their WGS 'main' build), specialized to that
build (decided #if groups, comments, unreferenced functions removed; cuobjdump -sass identical, tools/hyb_patches.py)."""
_HYB_MOE_CUDA = r"""@CUDA@"""
_HYB_MOE_CPP = r"""@CPP@"""
HYB_ENABLE = True
HYB_T = (@TLO@, @THI@)
HYB_PDL = @PDL@
HYB_FLAGS = @FLAGS@
_HYB = {"mod": None, "ok": False, "w": None, "sc": None, "full": None, "nsm": 0, "err": None, "build_s": 0.0, "pdl": -1}
_HYB_MMAX = 384
_HYB_EMAX = _HYB_MMAX * 8
_HYB_FIXF = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16
_HYB_XROWS = _HYB_EMAX + 8 * (32 + _HYB_EMAX // 32 + 8)
_HYB_GUARD = 1 << 16
_HYB_SLOTS = 160
HYB_SELF_TEST = ((49, 61, "normal"), (64, 62, "skewed"), (100, 63, "normal"), (128, 64, "skewed"), (160, 65, "normal"),
                 (200, 66, "skewed"), (256, 67, "normal"), (257, 68, "skewed"), (320, 69, "normal"))


def _hyb_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((_HYB_MOE_CPP + _HYB_MOE_CUDA + " ".join(HYB_FLAGS)).encode()).hexdigest()[:12]
    t0 = time.time()
    _HYB["mod"] = load_inline(name=f"hyb_moe_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=_HYB_MOE_CUDA,
                              functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=list(HYB_FLAGS), verbose=False)
    _HYB["build_s"] = time.time() - t0
    return _HYB["mod"]


"""[HYB] _hyb_pos .. _hyb_wgs_copy: alygg77's _stored_pos (DQ7 branch), _pack_codes_nat, _kperm_src, _kperm_codes,
_kperm_rows, _wgs_copy (renamed; logic unchanged)."""


def _hyb_pos():
    pos = {}
    for R in range(4):
        for up in range(2):
            outb = {0: 6 + 16 * up, 1: 7 + 16 * up, 2: 8 + 16 * up, 3: 15 + 16 * up}
            for b, o in outb.items():
                d = (0, 28, 22, 18 if b == 2 else 26)[R]
                pos[(R, up, b)] = (o + d) % 32
    assert sorted(pos.values()) == list(range(32))
    return pos


_HYB_POS = _hyb_pos()


def _hyb_pack_codes_nat(codes, rows):
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)
    n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)
    word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
    for Rg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for up in range(2):
            code = n[..., h, q, up]
            for b in range(4):
                word |= ((code >> b) & 1) << _HYB_POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _hyb_kperm_src():
    p = torch.arange(256)
    s, c = p >> 4, p & 15
    lg, j, t, i = s >> 2, s & 3, (c >> 1) & 3, (c & 1) | ((c >> 3) << 1)
    k = 16 * (4 * t + j) + 4 * lg + i
    assert sorted(k.tolist()) == list(range(256))
    return k


def _hyb_kperm_codes(codes):
    R, K = codes.shape[0], codes.shape[1] * 2
    nib = torch.stack((codes & 0xF, codes >> 4), -1).reshape(R, K // 256, 256)
    nib = nib.index_select(2, _hyb_kperm_src().to(codes.device)).reshape(R, K // 2, 2)
    return (nib[..., 0] | (nib[..., 1] << 4)).contiguous()


def _hyb_kperm_rows(rowsA):
    r = rowsA.view(-1, 2, 8)
    gate, up = r[:, 0, :].reshape(-1), r[:, 1, :].reshape(-1)
    P = torch.arange(gate.numel(), device=rowsA.device)
    cm = (P >> 8) * 256 + _hyb_kperm_src().to(rowsA.device)[P & 255]
    return torch.stack((gate.index_select(0, cm).view(-1, 8), up.index_select(0, cm).view(-1, 8)), 1).reshape(-1)


def _hyb_wgs_copy(codes, sf, rows):
    wn = _hyb_pack_codes_nat(_hyb_kperm_codes(codes), rows)
    R = rows.numel()
    nb, nub = R // 16, R // 256
    w = wn.view(nb, 8, 8, 8, 4, 4)
    w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)
    w = w.view(nub, 4, 4, 16, 4, 4, 32)
    w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()
    cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
    s = sf.view(torch.uint8).index_select(0, rows)
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)
    s = s.permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 16, 4, 1024)
    return torch.cat([cb, s], dim=3).contiguous().view(-1)


"""[HYB] End of the code derived from alygg77; _hyb_prmt_ok (their _ma_scales_prmt_exact rule), _hyb_scratch (their
_ma_make_scratch sizes + our sentinel guards), _hyb_call, _hyb_prepare are our glue."""


def _hyb_prmt_ok(*sfs):
    return all(not bool(((b < 0x08) | (b > 0x7E)).any()) for b in (sf.contiguous().view(torch.uint8) for sf in sfs))


def _hyb_scratch(dev):
    sizes = ((8 + 1024 + _HYB_EMAX, torch.int32, torch.int32, 0x5EADBEEF),
             (_HYB_EMAX + _HYB_EMAX * 4096 + _HYB_FIXF, torch.float32, torch.int32, 0x5EADBEEF),
             ((_HYB_EMAX + 2 * _HYB_XROWS) * 4096, torch.bfloat16, torch.int16, 0x5EAD))
    full, views = [], []
    for n, dt, vt, sv in sizes:
        f = torch.zeros(n + _HYB_GUARD, dtype=dt, device=dev)
        f[n:].view(vt).fill_(sv)
        full.append((f, n, vt, sv))
        views.append(f[:n])
    return views, full


def _hyb_guards_ok():
    return all(bool((f[n:].view(vt) == sv).all()) for f, n, vt, sv in _HYB["full"])


def _hyb_call(x, ids, wts):
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYB_PDL else 0   # (launch attribute: fixed per CUDA graph at capture)
    if _HYB["pdl"] != pdl:
        _HYB["mod"].okc_set_pdl(pdl)
        _HYB["pdl"] = pdl
    _HYB["mod"].okc_run(_HYB["w"], _HYB["sc"], x, ids, wts, out, 0, 32, _HYB["nsm"])
    return out


def _hyb_on(m):
    return HYB_ENABLE and _HYB["ok"] and HYB_T[0] <= m <= HYB_T[1]


def _hyb_prepare(st, w, dev):
    _HYB["ok"] = False
    if not HYB_ENABLE:
        return
    t1 = time.time()
    try:
        _pt1_wait("hyb")
        if _HYB["mod"] is None:
            _hyb_build()
        nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        if nsm > _HYB_SLOTS or not _hyb_prmt_ok(w["w13_sf"], w["w2_sf"]):
            print(f"[hyb] OFF: {nsm} SMs / scale bytes outside the PRMT decode range -> VML / K2A", flush=True)
            return
        I = H = 4096
        rb, g = torch.arange(I // 8, device=dev), torch.arange(8, device=dev)
        rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
        rowsA, rowsB = _hyb_kperm_rows(rowsA), torch.arange(H, device=dev)
        wa = torch.cat([_hyb_wgs_copy(w["w13"][e].view(torch.uint8), w["w13_sf"][e], rowsA) for e in range(32)])
        wb = torch.cat([_hyb_wgs_copy(w["w2"][e].view(torch.uint8), w["w2_sf"][e], rowsB) for e in range(32)])
        lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float()
               * 2.0**64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        ph = torch.zeros(16, dtype=torch.int32, device=dev)   # copies the WGS build never reads (wA sA wB sB, fat, unit)
        _HYB["w"] = [ph, ph, w["w13_gs"].float().contiguous(), ph, ph, w["w2_gs"].float().contiguous(),
                     (lut | (lut << 16)).to(torch.int32).contiguous(), w["post_norm_w"].contiguous(), ph, ph, ph, ph, wa, wb]
        _HYB["sc"], _HYB["full"] = _hyb_scratch(dev)
        _HYB["nsm"] = nsm
        torch.cuda.synchronize()
        good = True
        for bs, seed, dist in HYB_SELF_TEST:
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            d = {"hidden_states": d["hidden_states"].contiguous(), "topk_ids": d["topk_ids"].to(torch.int32).contiguous(),
                 "topk_weights": d["topk_weights"].float().contiguous(), "config": {"local_start": 0, "local_count": K2A_E_LOC}}
            ref = _k2a_ref_fp32(d, w)
            vrep = _nb_vml_rep(bs)
            o_v = _vml_route(st, d).clone() if (VML_ENABLE and VML_BUCKETS.get(vrep) and _VML["ok"].get(vrep)) else ref
            prev = None
            for k in range(2):
                o = _hyb_call(d["hidden_states"], d["topk_ids"], d["topk_weights"])
                torch.cuda.synchronize()
                ev, _ = _k2a_rel_err(o, o_v)
                tot, row = _k2a_rel_err(o, ref)
                rep_d = 0.0 if prev is None else _k2a_rel_err(o, prev)[0]
                # the FP32 gate of our own engines (K2A / VML: tot < 1e-2, row < 5e-2); vs VML only a sanity bound (R-hyb-1:
                # the two engines round differently: hyb 2.55e-3 / VML 3.75e-3 vs FP32, 3.4e-3 between them)
                ok = tot < 1e-2 and row < 5e-2 and ev < 1e-2 and rep_d < 1e-3 and bool(torch.isfinite(o.float()).all())
                print(f"[hyb] self-test bs={bs} {dist} call {k}: rel vs VML {ev:.2e} vs FP32 {tot:.3e} worst row {row:.3e} "
                      f"repeat {rep_d:.1e} -> {'ok' if ok else 'FAIL'}", flush=True)
                good &= ok
                prev = o.clone()
        gok = _hyb_guards_ok()
        _HYB["ok"] = bool(good and gok)
        print(f"[hyb] alygg77 WGS engine {'LIVE' if _HYB['ok'] else 'OFF'} at T {HYB_T[0]}..{HYB_T[1]} (guards {gok}, "
              f"pdl {int(HYB_PDL)}, nvcc {_HYB['build_s']:.0f}s, prepare {time.time() - t1:.1f}s; "
              f"{len(THIRD_PARTY_NOTICES)} chars of third-party notices kept)", flush=True)
    except Exception as ex:
        _HYB["ok"] = False
        _HYB["err"] = repr(ex)[:500]
        print(f"[hyb] prepare raised {ex!r}: VML / K2A keep T {HYB_T[0]}..{HYB_T[1]}", flush=True)


'''

ROUTE_OLD = '''    rep = _k2a_bucket_of(x.shape[0])
    vrep = _nb_vml_rep(x.shape[0])
    if VML_ENABLE and VML_BUCKETS.get(vrep)'''
ROUTE_NEW = '''    if _hyb_on(x.shape[0]):
        try:
            return _hyb_call(x, local_ids.to(torch.int32).contiguous(), topk_weights.to(torch.float32).contiguous())
        except Exception as ex:
            print(f"[hyb] route raised ({ex!r}): VML / K2A from here on", flush=True)
            _HYB["ok"] = False
    rep = _k2a_bucket_of(x.shape[0])
    vrep = _nb_vml_rep(x.shape[0])
    if VML_ENABLE and VML_BUCKETS.get(vrep)'''
PT1_OLD = '''    if VML_ENABLE:
        jobs.append(("vml", _vml_build_job))
'''
PT1_NEW = PT1_OLD + '''    if HYB_ENABLE:
        jobs.append(("hyb", _hyb_build))
'''
PREP_OLD = '''    _vml_prepare(st, w, dev)
'''
PREP_NEW = PREP_OLD + '''    _hyb_prepare(st, w, dev)
'''


def apply(base, a):
    cuda, cpp, header = upstream()
    eng, dropped = strip_engine(cuda)
    for s in (eng, cpp):
        assert '"""' not in s and not s.endswith("\\")
    lic = (TP / "LICENSE").read_text(encoding="utf-8").rstrip("\n")
    notice = (TP / "NOTICE").read_text(encoding="utf-8").rstrip("\n")
    for s in (lic, notice, header):
        assert '"""' not in s and "\\" not in s
    t = base
    if "\nTHIRD_PARTY_NOTICES = " not in t:   # first third-party block in the file (else hya's): carry the license text once
        # (convention shared with tools/hya_patches.py: one r-string constant at line 3, each block appends its _X_NOTICE)
        lines = t.split("\n")
        t = "\n".join(lines[:2] + [NOTICE_HEAD.replace("@APACHE@", lic).rstrip("\n")] + lines[2:])
    lo, hi = (int(v) for v in a.t.split(","))
    blk = (BLOCK.replace("@NOTICE_BLOCK@", HYB_NOTICE.replace("@HEADER@", header).replace("@NOTICE@", notice).rstrip("\n"))
           .replace("@CUDA@", eng).replace("@CPP@", cpp).replace("@TLO@", str(lo)).replace("@THI@", str(hi))
           .replace("@PDL@", str(bool(a.pdl))).replace("@FLAGS@", repr(FLAGS)))
    t = sub1(t, "\ndef _v11_k2a33_call(", "\n" + blk.lstrip("\n") + "def _v11_k2a33_call(", "def _v11_k2a33_call")
    t = sub1(t, ROUTE_OLD, ROUTE_NEW, "route")
    t = sub1(t, PT1_OLD, PT1_NEW, "pt1")
    t = sub1(t, PREP_OLD, PREP_NEW, "prepare")
    if a.drop_w72:
        t = sub1(t, "VML_BVAR = {256: 'w72', 128: 'w72'}", "VML_BVAR = {}", "VML_BVAR")
    if a.nonce:
        ln = t.split("\n", 2)
        assert ln[1].startswith("# nonce ")
        t = ln[0] + "\n" + a.nonce + "\n" + ln[2]
    return t, eng, dropped


def cute_ok(t):
    sys.path.insert(0, str(ROOT / "tools"))
    from ax1_patches import cute_sources_ok
    return cute_sources_ok(t)


def pyflakes_undef(path):
    r = subprocess.run([sys.executable, "-m", "pyflakes", str(path)], capture_output=True, text=True)
    return sorted({ln.split("undefined name ")[1] for ln in r.stdout.splitlines() if "undefined name" in ln})


def gate(t, out, base_path, eng, sass):
    ok = True
    ast.parse(t)
    cok, cbad = cute_ok(t)
    ub, uo = pyflakes_undef(base_path), pyflakes_undef(out)
    new_undef = sorted(set(uo) - set(ub))
    notices = all(s in t for s in ("Apache License", "Version 2.0, January 2004", "END OF TERMS AND CONDITIONS",
                                   "alygg77", "vjeux/okc-2026", "THIRD_PARTY_NOTICES += _HYB_NOTICE"))
    n = len(t.encode())
    print(f"[hyb_patches] gates: ast ok, CuTe defs {cok} ok / {cbad} FAIL, pyflakes new undefined {new_undef or 'none'}, "
          f"ascii {t.isascii()}, notices {notices}, {n} B ({'under' if n < LIMIT else 'OVER'} the cap by {abs(LIMIT - n)} B "
          f"before slim_patches)")
    ok &= cbad == 0 and not new_undef and t.isascii() and notices
    if sass:
        wsl = lambda p: "/mnt/c/" + pathlib.Path(p).resolve().as_posix()[3:]   # noqa: E731
        tmp = ROOT / "box_out" / "hyb"
        tmp.mkdir(parents=True, exist_ok=True)
        cuda, _, _ = upstream()
        (tmp / "orig.cu").write_text(cuda, newline="\n", encoding="utf-8")
        (tmp / "strip.cu").write_text(eng, newline="\n", encoding="utf-8")
        cmd = ["wsl", "-d", "Ubuntu", "-u", "root", "--", "bash", wsl(ROOT / "tools/local/cudaenv.sh"), "python", "-I",
               wsl(ROOT / "tools/local/hyb_sass.py"), wsl(tmp / "orig.cu"), wsl(tmp / "strip.cu"), wsl(tmp / "sass"), "--",
               *FLAGS]
        r = subprocess.run(cmd, capture_output=True, text=True, env={**__import__("os").environ, "MSYS_NO_PATHCONV": "1"})
        print((r.stdout + r.stderr).strip()[-1500:])
        ok &= r.returncode == 0
    print(f"[hyb_patches] gate {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="kernels/lab/k3_b29_raw.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", default="49,320", help="token range routed to the engine (VML / K2A outside)")
    ap.add_argument("--pdl", type=int, default=0, help="their okc_set_pdl (their file: off at T > 32)")
    ap.add_argument("--drop-w72", dest="drop_w72", type=int, default=0, help="1: VML 128 / 256 buckets fall back to bf2")
    ap.add_argument("--nonce", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sass", type=int, default=1, help="with --check: run the WSL SASS identity gate")
    a = ap.parse_args()
    base_path = ROOT / a.base
    base = base_path.read_text(encoding="utf-8")
    t, eng, dropped = apply(base, a)
    out = ROOT / a.out
    out.write_text(t, newline="\n", encoding="utf-8")
    print(f"[hyb_patches] {a.base} -> {a.out}: {len(t.encode())} B (+{len(t.encode()) - len(base.encode())} B; engine CUDA "
          f"{len(eng.encode())} B, sha256 {hashlib.sha256(eng.encode()).hexdigest()[:16]}; dropped {len(dropped)} unreferenced "
          f"defs: {' '.join(dropped)})")
    if a.check:
        sys.exit(0 if gate(t, out, base_path, eng, a.sass) else 1)


if __name__ == "__main__":
    main()

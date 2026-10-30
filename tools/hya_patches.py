"""hya_patches.py (R-hya, team joeyhlu): route attention shapes of a decode-layer file to vjeux's public attention decode
kernel (github.com/vjeux/okc-2026 at f35a41ee, muse_attention_decode.py, Apache License 2.0; vendored with its LICENSE in
tools/vendor/vjeux_okc2026_f35a41ee/), keeping our KS1 / KS3 (and AX1) where they win.

    python -I tools/hya_patches.py --base=kernels/lab/k3_b29_raw.py --out=<out.py> [--mode=ship|lab] [--drop-ax1]
        [--route=d16f,d32f,v1f,v16f,v32f,v16s,v32s] [--check]
    then tools/slim_patches.py LAST (it keeps license / provenance comment blocks, R-slim-1).

Both modes insert (each anchor exactly once):
  * THIRD_PARTY_NOTICES, ONE shared string constant at line 3 with the full Apache-2.0 text (created unless the base has
    it, e.g. from R-hyb's patcher), and `THIRD_PARTY_NOTICES += _HYA_NOTICE`: NOTICE-style attribution, the modification
    list (M1-M4), FA4's BSD-3-Clause text while FA4 code remains (lab mode). Referenced, so slim keeps it;
  * HYA_CPP_SRC / HYA_CUDA_SRC = vjeux's CPP_SRC / CUDA_SRC, verbatim except M1 (spin_fail: sticky flag + break
    instead of __trap); the HYA_* route constants; the host block of tools/hya_src.py, before `def _r2_page1(`;
  * kernel_fn: the explicit HYA_ROUTE pass first (before AX1 / KS1); the nvcc build in the PT1 pool; _hya_prepare.
--mode=lab (LAB ONLY, over 1 MB): everything of the base stays; HYA_ROUTE = all 12 scored shapes, HYA_DEFAULT False,
  so rev_ab3 arms {"HYA_ENABLE": false} (= the base) vs {} (= HYA everywhere) pair every shape in one process.
--mode=ship: also deletes what HYA replaces: the K1 module (+ VA1 / AX2, out of the combined v10 build), the FA4 v6
  chain and VFA (their self-test helpers that KS1 / KS3 / AX1 use stay), --drop-ax1 also AX1; HYA becomes the
  fallback after AX1 / KS1 (HYA_DEFAULT True), kv_scale != 1 and unsupported layouts take the r1 path (SGLang FA3).
--check: $0 gates: AST parse, pyflakes (no undefined names), ASCII, CuTe DSL defs re-parse, vendored-file hashes, and
  HYA_CUDA_SRC compiled for sm_90a by the local WSL toolchain (ptxas -v: spills printed).
"""
import argparse
import ast
import hashlib
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "tools" / "vendor" / "vjeux_okc2026_f35a41ee"
BLOBS = {"muse_attention_decode.py": "3ae30d24058745a28f13c50021f7f387fe3da7bc", "LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"}
LIMIT = 1_000_000
SHAPES = {"d1s": ("sliding", 1, 1), "d1f": ("full", 1, 1), "d16s": ("sliding", 16, 1), "d16f": ("full", 16, 1),
          "d32s": ("sliding", 32, 1), "d32f": ("full", 32, 1), "v1s": ("sliding", 1, 8), "v1f": ("full", 1, 8),
          "v16s": ("sliding", 16, 8), "v16f": ("full", 16, 8), "v32s": ("sliding", 32, 8), "v32f": ("full", 32, 8)}
LIC_TAIL = "---- Apache License, Version 2.0: full text (applies to every section below marked Apache-2.0) ----"
M1_OLD = ("__device__ __noinline__ void spin_fail() { __trap(); }   // (out of line: keeps the trap off the polling loop's code)\n",
          "#define SPIN_CHECK do { if (++_spin_n > (1u << 24)) spin_fail(); } while (0)   /* (>= 2^24 global polls: seconds) */\n")
M1_NEW = ("// HYA M1 (team joeyhlu, 2026-10-30), modification of vjeux's code (Apache-2.0): no __trap() in a bounded wait; after\n"
          "// 2^24 polls spin_fail() sets the sticky flag hya_spin_err and the wait loop breaks (vjeux: spin_fail() { __trap(); }\n"
          "// and SPIN_CHECK = do { if (...) spin_fail(); } while (0)). Every SPIN_CHECK sits directly in a for (;;) poll loop.\n"
          "__device__ int hya_spin_err = 0;\n"
          "__device__ __noinline__ void spin_fail() { atomicExch(&hya_spin_err, 1); }   // (out of line: keeps the flag store off the polling loop's code)\n",
          "#define SPIN_CHECK if (++_spin_n > (1u << 24)) { spin_fail(); break; }   /* (>= 2^24 global polls: seconds; HYA M1) */\n")
TUNED = "{(32, 8, True, 1): dict(n_splits=4, nc=1, coop=1), (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252)}"
SELF_TEST = ("((1, 3000, 1, 2048, 1101, False, 1), (16, 2500, 1, 2048, 1102, False, 1), (32, 2500, 1, 2048, 1103, False, 1), "
             "(1, 9000, 1, 0, 1104, False, 1), (16, 1500, 1, 0, 1105, False, 1), (33, 1000, 1, 0, 1106, True, 1), "
             "(1, 3000, 8, 0, 1107, False, 1), (16, 1500, 8, 0, 1108, False, 1), (32, 800, 8, 0, 1109, False, 1), "
             "(1, 3000, 8, 2048, 1110, True, 1), (16, 2500, 8, 2048, 1111, False, 1), (32, 2500, 8, 2048, 1112, False, 1), "
             "(4, 3000, 1, 2048, 1113, False, 64), (4, 3000, 8, 0, 1114, False, 64), (1, 1, 1, 0, 1115, False, 1), "
             "(2, 127, 8, 2048, 1116, False, 16), (5, 9000, 1, 0, 1117, False, 1))")
GRAPH_CASES = "((4, 6000, 8, 2048, 1121), (1, 3000, 1, 0, 1122), (16, 1500, 1, 0, 1123))"


def die(msg):
    sys.exit(f"[hya_patches] {msg}")


def sub1(text, old, new, what):
    n = text.count(old)
    if n != 1:
        die(f"anchor {what!r}: found {n} times")
    return text.replace(old, new)


def blob_sha(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def vendor():
    src = {}
    for name, sha in BLOBS.items():
        raw = (VENDOR / name).read_bytes().replace(b"\r\n", b"\n")
        if blob_sha(raw) != sha:
            die(f"vendored {name}: git blob {blob_sha(raw)} != upstream {sha}")
        src[name] = raw.decode("ascii")
    tree = ast.parse(src["muse_attention_decode.py"])
    consts = {n.targets[0].id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}
    cuda, cpp = consts["CUDA_SRC"], consts["CPP_SRC"]
    for old, new in zip(M1_OLD, M1_NEW):
        cuda = sub1(cuda, old, new, "M1")
    if "__trap" in cuda.replace("no __trap() in", "").replace("spin_fail() { __trap(); }", "") or '"""' in cuda + cpp:
        die("CUDA source still traps or holds a triple quote")
    for s in re.finditer(r"SPIN_CHECK;", cuda):   # M1's break must land in the poll loop itself
        head = cuda[max(0, s.start() - 400):s.start()]
        if head.rfind("for (") < max(head.rfind("}\n"), head.rfind("do {")):
            die("a SPIN_CHECK is not directly inside a for loop")
    return cuda, cpp, src["LICENSE"]


def notice_hya(fa4_left):
    return f"""[HYA] vjeux/okc-2026, https://github.com/vjeux/okc-2026, commit f35a41ee1c (2026-10-28), file muse_attention_decode.py
    (git blob 3ae30d24). Copyright vjeux. Licensed under the Apache License, Version 2.0 (full text below).
    Reused: HYA_CUDA_SRC and HYA_CPP_SRC are vjeux's CUDA_SRC / CPP_SRC (the fused gated attention decode kernel: QK-norm,
    RoPE, FP8 cache write, split-KV FP8 attention, merge, RMSNorm * sigmoid(gate)); _hya_plan and _hya_launch are derived
    from that file's _plan / _launch. The same kernel family ships unchanged in alygg77/okc-muse-spark-kernels
    (https://github.com/alygg77/okc-muse-spark-kernels, Apache-2.0, "Portions Copyright vjeux"), whose decode layer showed
    how it is called in-layer (with PDL); no alygg77 code is copied here.
    Modifications by team joeyhlu (2026-10-30), marked "HYA M<n>" in the code:
      M1 HYA_CUDA_SRC: the bounded spin waits no longer trap after 2^24 polls; spin_fail() sets a sticky flag
         (hya_spin_err) and the poll loop breaks (SPIN_CHECK). No other change to the CUDA or C++ source.
      M2 _hya_plan: SM count passed in, vjeux's _splits / _copies / _headroom inlined; the plan logic is unchanged.
      M3 _hya_launch / _hya_build / _hya_ws: qkv is our qkv projection output, passed separately; PDL per HYA_PDL /
         HYA_PDL_T; per-shape overrides HYA_TUNED = vjeux's _TUNED; workspace and RoPE tables made in prepare, never
         inside a CUDA-graph capture; the extension is built as hya_attn_v298_<hash>.
      M4 the routing around it (_hya_wanted, _hya_try, kernel_fn) and the self-test (_hya_paged, _hya_self_test,
         _hya_prepare) are team joeyhlu's own code.
    {'FA4: FlashAttention-4 CuTe-DSL code remains in this file; its BSD-3-Clause notice follows.' if fa4_left else
     'No FlashAttention-4 (BSD-3-Clause) code remains in this file (the kept _k1fa4_* helpers are our own test code).'}
"""


FA4_BSD = """[FA4] FlashAttention-4 CuTe-DSL Hopper forward (k1fa4_* classes; from sglang/kernels/ops/attention/flash_attn/cute/
    flash_fwd_sm90.py in lmsysorg/sglang:dev-muse-spark-open). License (BSD 3-Clause):
    BSD 3-Clause License
    Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
    All rights reserved.
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
      disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def notices_head(text, apache_text):
    """The shared THIRD_PARTY_NOTICES constant (convention agreed with R-hyb, 2026-10-30): the first patcher to land puts it
    at line 3 with the full Apache-2.0 text; every reused block appends its own section (`THIRD_PARTY_NOTICES += _X_NOTICE`)."""
    if '\nTHIRD_PARTY_NOTICES = r"""' in text:
        return text
    l1, l2, rest = text.split("\n", 2)
    const = ('THIRD_PARTY_NOTICES = r"""Third-party code in this file (team joeyhlu, Muse Spark Open Kernel Challenge). Each '
             "reused block appends its NOTICE-style section below the license text (THIRD_PARTY_NOTICES += _<X>_NOTICE); "
             "code not attributed there is team joeyhlu's own.\n" + LIC_TAIL + "\n" + apache_text.rstrip("\n") + '\n"""\n')
    return "\n".join((l1, l2, const + rest))


def stmt_names(n):
    if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
        return {n.name}
    if isinstance(n, ast.Assign):
        return {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in n.names}
    return set()


def delete_regions(text, regions, singles, keep):
    """Delete top-level statements: every statement from region first .. last (by defined name, inclusive) except those
    defining a kept name, plus the single named statements; with each one, the comment / blank lines directly above it."""
    lines = text.split("\n")
    tree = ast.parse(text)
    body = tree.body
    idx = {}
    for i, n in enumerate(body):
        for nm in stmt_names(n):
            idx.setdefault(nm, i)
    drop = set()
    for first, last in regions:
        if first not in idx or last not in idx:
            die(f"region {first}..{last} not found")
        for i in range(idx[first], idx[last] + 1):
            if not (stmt_names(body[i]) & keep):
                drop.add(i)
    for nm in singles:
        if nm not in idx:
            die(f"statement {nm} not found")
        drop.add(idx[nm])
    kill = set()
    for i in sorted(drop):
        n = body[i]
        a = n.lineno - 1 - len(getattr(n, "decorator_list", []) or [])
        if getattr(n, "decorator_list", None):
            a = min(d.lineno for d in n.decorator_list) - 1
        b = n.end_lineno
        prev_end = body[i - 1].end_lineno if i > 0 else 0
        while a - 1 >= prev_end and (lines[a - 1].strip() == "" or lines[a - 1].lstrip().startswith("#")):
            a -= 1
        kill.update(range(a, b))
    out = "\n".join(ln for k, ln in enumerate(lines) if k not in kill)
    return out, len(drop)


def ship_edits(t, drop_ax1):
    # combined v10 build without the K1 module (K2A33 / R1B / tail use no K1 macro or function: checked, FNC_* blocks of
    # K1 and R1B are identical)
    t = sub1(t, 'cpp = "\\n".join((K1_CPP_SRC, K2A33_CPP_SRC, R1B_CPP_SRC, V8_TAIL_CPP_SRC))',
             'cpp = "\\n".join((K2A33_CPP_SRC, R1B_CPP_SRC, V8_TAIL_CPP_SRC))', "v10 cpp")
    t = sub1(t, 'cuda = "\\n".join((K1_CUDA_SRC, k33, R1B_CUDA_SRC, V8_TAIL_CUDA_SRC))',
             'cuda = "\\n".join((k33, R1B_CUDA_SRC, V8_TAIL_CUDA_SRC))', "v10 cuda")
    t = sub1(t, 'functions=["k1_forward", "k1_occupancy", "k1_prof_read", "run3",', 'functions=["run3",', "v10 fns")
    t = sub1(t, "combined build (K1 + K2A v3.7", "combined build (K2A v3.7", "v10 print")
    # prepare: no K1 build / self-test, no FA4 / VA1 / VFA prepare; HYA where FA4's prepare was
    m = re.search(r'\n    state\["k1_ok"\] = False\n    state\["k2a"\] = None\n    if V10_COMBINED_BUILD:\n.*?'
                  r'            state\["k1_ok"\] = _k1_prepare\(device\)\n', t, re.S)
    if not m:
        die("prepare K1 section not found")
    t = t[:m.start()] + '\n    state["k2a"] = None\n    if V10_COMBINED_BUILD:\n        _v10_build_all()\n' + t[m.end():]
    t = sub1(t, "\n    _v18_fa4_prepare(state, t0)\n", "\n    _hya_prepare(state, t0)\n", "prepare fa4")
    t = sub1(t, "\n    _va1_prepare(state)\n", "\n", "prepare va1")
    t = sub1(t, "\n    _vfa_prepare(state, t0)\n", "\n", "prepare vfa")
    if drop_ax1:
        t = sub1(t, "\n    _ax1_prepare(state, t0)\n", "\n", "prepare ax1")
    t = sub1(t, '    occ = "/".join(str(v) for _, v in sorted(_K1_OCC.items(), key=lambda kv: str(kv[0]))) or "-"\n', "",
             "prepare occ")
    t = sub1(t, "attention={'K1' if state['k1_ok'] else 'r1-FA3'}", "attention={'HYA' if state.get('hya') else 'r1-FA3'}",
             "prepare print 1")
    t = sub1(t, " (USE_K1={USE_K1} USE_K2A={USE_K2A}); nvcc {builds}; k1 v7 occupancy {occ} CTAs/SM,\"\n"
                "          f\" route {K1_ROUTE_MIN_BATCH}, cps {K1_CPS_T}, fold counters {'ok' if _K1_CNT else 'none'};\"\n",
             " (USE_K2A={USE_K2A}); nvcc {builds};\"\n", "prepare print 2")
    t = sub1(t, "switches: USE_K1={USE_K1} USE_K2A={USE_K2A} K1=v7 route={K1_ROUTE_MIN_BATCH} fold={K1_FOLD_MIN_BATCH} "
                "min_chunk={K1_MIN_CHUNK_T} cps={K1_CPS_T} K2A=v2.2", "switches: USE_K2A={USE_K2A} K2A=v2.2", "k3 switches print")
    t = sub1(t, "P1_PROJ={P1_PROJ} K1 route={K1_ROUTE_MIN_BATCH} K2A33_PLAN", "P1_PROJ={P1_PROJ} K2A33_PLAN", "v13 print")
    t = sub1(t, '    if VFA_ENABLE:\n        jobs.append(("vfa", _vfa_build))\n', "", "pt1 vfa")
    if drop_ax1:
        t = sub1(t, '    if AX1_ENABLE:\n        jobs.append(("ax1", _ax1_build))\n', "", "pt1 ax1")
    # kernel_fn: VFA / FA4 / K1 (+ VA1) -> the HYA fallback pass, then r1
    m = re.search(r"\n    if o is None and _vfa_wanted\(qkv, data, cfg, state\):\n.*?\n    if o is None:\n"
                  r"        o = _r1_attention\(qkv, data, state\)\n", t, re.S)
    if not m:
        die("kernel_fn chain not found")
    t = (t[:m.start()] + "\n    if o is None:\n        o = _hya_try(qkv, data, cfg, state, False)\n"
         "    if o is None:\n        o = _r1_attention(qkv, data, state)\n" + t[m.end():])
    if drop_ax1:
        m = re.search(r"\n    if o is None and _ax1_wanted\(qkv, adata, adata\[\"config\"\], state\):\n.*?"
                      r"            o = None\n", t, re.S)
        if not m:
            die("kernel_fn AX1 block not found")
        t = t[:m.start()] + "\n" + t[m.end():]
    # R2 page-1 view: only KS1 still wants it (HYA reads any power-of-2 page natively)
    m = re.search(r"\n    if kind == \"full\" and _vfa_wanted\(x, data, cfg, state\):.*?\n        return data\n"
                  r"    if not \(_ks1_wanted\(x, view, cfg1, state\) or \(not _va1_b1\(cfg1, state\) and "
                  r"_v18_fa4_wanted\(x, view, cfg1, state\)\)\):\n        return data\n", t, re.S)
    if not m:
        die("_r2_page1 checks not found")
    t = (t[:m.start()] + "\n    if not _ks1_wanted(x, view, cfg1, state):   # HYA takes the other classes on their own pages\n"
         "        return data\n" + t[m.end():])
    regions = [("K1_TILE", "_k1_prepare"), ("k1fa4_partial", "_v18_fa4_prepare"), ("VFA_ENABLE", "_vfa_prepare")]
    keep = {"_K3_BUILD_S", "_K3_WARNED", "_k1_rms_t", "K1FA4_RMS_EPS", "K1FA4_ROPE_POSITIONS", "_k1fa4_rope",
            "k1fa4_cos_sin", "_k1fa4_case", "_k1fa4_reference", "_k1fa4_r1_path", "_vfa_ref_attention"}
    if drop_ax1:
        regions.append(("AX1_CPP_SRC", "_ax1_prepare"))
        keep -= {"_vfa_ref_attention", "_k1_rms_t"}
    singles = ["USE_K1", "K1_ROUTE_MIN_BATCH", "K1_FOLD_MIN_BATCH", "K1_MIN_CHUNK_T", "K1_CPS_T", "K1_STAGES_T", "K1_SELF_TEST"]
    t, n = delete_regions(t, regions, singles, keep)
    print(f"[hya_patches] ship: deleted {n} top-level statements (K1 + VA1 / AX2, FA4 v6 chain, VFA"
          f"{', AX1' if drop_ax1 else ''})")
    return t


def apply(base, mode, route, drop_ax1):
    cuda, cpp, apache = vendor()
    t = base.replace("\r\n", "\n")
    lab = mode == "lab"
    keys = list(SHAPES) if lab else route
    rt = "{" + ", ".join(f"{SHAPES[k]!r}: True" for k in keys) + "}"
    fa4_left = lab   # ship mode deletes the FA4 v6 chain (checked again below)
    consts = ('"HYA BEGIN (tools/hya_patches.py): code from here to HYA END is derived from vjeux/okc-2026 f35a41ee where marked '
              '(Apache-2.0); attribution and modifications in _HYA_NOTICE / THIRD_PARTY_NOTICES"\n'
              "# ---- HYA constants (tools/hya_patches.py). HYA_CPP_SRC / HYA_CUDA_SRC: derived from vjeux/okc-2026 f35a41ee\n"
              "# muse_attention_decode.py CPP_SRC / CUDA_SRC (Apache-2.0, see THIRD_PARTY_NOTICES), verbatim except HYA M1.\n"
              f'HYA_CPP_SRC = r"""{cpp}"""\nHYA_CUDA_SRC = r"""{cuda}"""\n'
              f"HYA_ENABLE = True\nHYA_ROUTE = {rt}\nHYA_DEFAULT = {not lab}\nHYA_PDL = True\nHYA_PDL_T = {{}}\n"
              f"HYA_TUNED = {TUNED}\nHYA_PREPARE_BY_S = 250.0\nHYA_WS_ROWS = 1 << 15\nHYA_SELF_TEST_STRICT = True\n"
              f"HYA_SELF_TEST = {SELF_TEST}\n"
              f"HYA_GRAPH_CASES = {GRAPH_CASES}\n"
              f'_HYA_NOTICE = r"""\n{notice_hya(fa4_left) + (FA4_BSD if fa4_left else "")}"""\n'
              "THIRD_PARTY_NOTICES += _HYA_NOTICE\n")
    if '"""' in notice_hya(fa4_left) + FA4_BSD:
        die("notice text holds a triple quote")
    src = (ROOT / "tools" / "hya_src.py").read_text(encoding="ascii")
    blk = src.split("# ---- HYA BLOCK BEGIN ----\n", 1)[1].split("# ---- HYA BLOCK END ----\n", 1)[0]
    blk += '"HYA END (derived code above: HYA_CPP_SRC, HYA_CUDA_SRC, _hya_plan, _hya_launch; see _HYA_NOTICE)"\n'
    t = sub1(t, "\ndef _r2_page1(", "\n" + consts + "\n\n" + blk + "\n\ndef _r2_page1(", "r2_page1")
    t = sub1(t, '    o = None\n    ks1_ran = False\n    if _ax1_wanted(qkv, adata, adata["config"], state):\n',
             '    o = _hya_try(qkv, data, cfg, state, True)\n    ks1_ran = False\n'
             '    if o is None and _ax1_wanted(qkv, adata, adata["config"], state):\n', "kernel_fn first pass")
    t = sub1(t, '    if AX1_ENABLE:\n        jobs.append(("ax1", _ax1_build))\n',
             '    if AX1_ENABLE:\n        jobs.append(("ax1", _ax1_build))\n    if HYA_ENABLE:\n        jobs.append(("hya", _hya_build))\n',
             "pt1 hya")
    if lab:
        t = sub1(t, "\n    _ax1_prepare(state, t0)\n", "\n    _ax1_prepare(state, t0)\n    _hya_prepare(state, t0)\n", "prepare hya")
    else:
        t = ship_edits(t, drop_ax1)
    if ("class k1fa4_FlashAttentionForwardSm90" in t) != fa4_left:
        die("FA4 code presence does not match the notice")
    return notices_head(t, apache), cuda


def cute_ok(t):
    sys.path.insert(0, str(ROOT / "tools"))
    from ax1_patches import cute_sources_ok
    return cute_sources_ok(t)


def gate(t, cuda, out):
    ast.parse(t)
    r = subprocess.run([sys.executable, "-m", "pyflakes", str(out)], capture_output=True, text=True)
    undef = [ln for ln in r.stdout.splitlines() if "undefined name" in ln]
    print(f"[hya_patches] pyflakes: {len(undef)} undefined names" + ("".join("\n  " + u for u in undef[:20])))
    cok, cbad = cute_ok(t)
    n = len(t.encode())
    print(f"[hya_patches] CuTe DSL defs re-parse {cok} ok / {cbad} FAIL; ascii {t.isascii()}; {n} B (pre-slim)")
    tmp = ROOT / "box_out" / "hya_dev.cu"
    tmp.write_text(cuda, newline="\n")
    wsl = "<repo>/tools/local/hya_compile.sh"
    rr = subprocess.run(["wsl", "-d", "Ubuntu", "-u", "root", "--", "bash", wsl, "<repo>/box_out/hya_dev.cu"],
                        capture_output=True, text=True, env={"MSYS_NO_PATHCONV": "1", **__import__("os").environ})
    log = rr.stdout + rr.stderr
    (ROOT / "box_out" / "hya_ptxas.log").write_text(log)
    res = [ln for ln in log.splitlines() if "spill" in ln or "error" in ln.lower()]
    spills = [ln for ln in res if "spill" in ln and " 0 bytes spill stores" not in ln]
    print(f"[hya_patches] sm_90a compile rc {rr.returncode}; {sum('spill' in ln for ln in res)} kernels; spilling: {len(spills)}"
          + "".join("\n  " + s.strip() for s in spills[:8]) + "".join("\n  " + s for s in res if "error" in s.lower())[:3000])
    ok = not undef and not cbad and t.isascii() and rr.returncode == 0
    print(f"[hya_patches] gate {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="kernels/lab/k3_b29_raw.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("ship", "lab"), default="ship")
    ap.add_argument("--route", default="d16f,d32f,v1f,v16f,v32f,v16s,v32s")
    ap.add_argument("--drop-ax1", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--nonce", default="", help="replace line 2 (the '# nonce' line) with '# nonce <this>'")
    a = ap.parse_args()
    route = [r for r in a.route.split(",") if r]
    if a.drop_ax1 and "d1f" not in route:
        route.append("d1f")
    base = (ROOT / a.base).read_text(encoding="utf-8")
    t, cuda = apply(base, a.mode, route, a.drop_ax1)
    if a.nonce:
        l1, l2, rest = t.split("\n", 2)
        if not l2.startswith("# nonce"):
            die("line 2 is not a nonce line")
        t = "\n".join((l1, "# nonce " + a.nonce, rest))
    out = ROOT / a.out
    out.write_text(t, newline="\n", encoding="utf-8")
    print(f"[hya_patches] {a.base} -> {a.out} ({a.mode}{', drop AX1' if a.drop_ax1 else ''}, route {route}): "
          f"{len(t.encode())} B ({len(t.encode()) - len(base.encode()):+d} B); HYA CUDA {len(cuda.encode())} B")
    if a.check:
        sys.exit(0 if gate(t, cuda, out) else 1)


if __name__ == "__main__":
    main()

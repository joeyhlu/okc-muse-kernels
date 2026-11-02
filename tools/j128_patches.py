"""J128: default-off, separately compiled HYB paired-job bank. No GPU work here.

Generate: python -B tools/j128_patches.py
Check:    python -B tools/j128_patches.py --check
CPU gate: python -B tools/local/j128_compile.py --cpu-only
"""
import ast
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/muse_decode_layer/k3_moexr_c1.py'
OUT = ROOT / 'kernels/lab/k3_j128_ab.py'


def once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:160])
    return s.replace(old, new, 1)


def scope(s, start, end):
    assert s.count(start) == s.count(end) == 1, (start, end)
    a, b = s.index(start), s.index(end)
    assert a < b
    return a, b, s[a:b]


def literal(text, name):
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, name
    return ast.literal_eval(nodes[0].value)


def patch_cuda(src):
    # The real jobs remain in jE/jS/jC. forder maps logical groups to their
    # first real job; dync (unused by WGS) holds npad | paired-leader bit 128.
    src = once(src, 'JMAX = 32 + EMAX / FT_T + 8;', 'JMAX = 112; // J128: CPU capacity gate includes all EMAX routes')
    src = once(src, 'int njF, njS, unit;', 'int njF, njS, unit;\nint j128_groups;')
    src = once(src, '__device__ void route(const Params& p, Smem& S) {', r'''// J128: even, balanced real jobs; every token belongs to exactly one job.
__device__ __forceinline__ int j128_njobs(int tot, int M) {
  return M >= 128 && tot > 64 ? 2 * ((tot + 127) / 128) : (tot + FT_T - 1) / FT_T;
}
__device__ __forceinline__ int j128_pad(const Smem& S, int j) { return S.dync[j] & 127; }
__device__ __forceinline__ int j128_pair(const Smem& S, int j) { return (S.dync[j] >> 7) & 1; }
__device__ void route(const Params& p, Smem& S) {''')
    src = once(src, 'const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;',
               'const int nfj = tot > SK_T ? j128_njobs(tot, p.M) : 0;')
    src = once(src, '''S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);''', r'''const bool pair = p.M >= 128 && tot > 64;
const int start = pair ? q * tot / nfj : q * FT_T;
const int end = pair ? (q + 1) * tot / nfj : min(tot, start + FT_T);
S.jE[fj + q] = lane; S.jS[fj + q] = off + start; S.jC[fj + q] = end - start;
const int q0 = q & ~1;
const int c0 = pair ? (q0 + 1) * tot / nfj - q0 * tot / nfj : end - start;
const int c1 = pair ? (q0 + 2) * tot / nfj - (q0 + 1) * tot / nfj : end - start;
const int pad = (max(c0, c1) + 7) & ~7;
S.dync[fj + q] = pad | ((pair && (q & 1) == 0) ? 128 : 0);''')
    src = once(src, 'if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }',
               'if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; S.dync[sj] = (tot + 7) & ~7; }')
    a, b, r = scope(src, '__device__ void route(', '__device__ __forceinline__ float probe_q(')
    assert r.endswith('CTA_SYNC();\n}\n')
    r = r[:-len('CTA_SYNC();\n}\n')] + r'''// jE/jC/dync were published by the CTA_SYNC preceding the ent loop.
if (tid == 0) {
  int ng = 0;
  for (int j = 0; j < S.njF + S.njS; j += 1 + j128_pair(S, j)) S.forder[ng++] = j;
  S.j128_groups = ng;
}
CTA_SYNC();
}
'''
    src = src[:a] + r + src[b:]
    src = once(src, 'constexpr int WGS_SLOTS = 160;',
               'constexpr int J128_SLOT_STRIDE = 160;\nconstexpr int WGS_SLOTS = 2 * J128_SLOT_STRIDE;')
    src = once(src, 'MF_XRS = 96,', 'MF_XRS = 112,')
    src = once(src, '#define OKC_WGSGS 2', '#define OKC_WGSGS 2\n#ifndef J128_REGS\n#define J128_REGS 96\n#endif')
    src = once(src, '__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p)',
               '__global__ void __maxnreg__(J128_REGS) moe_fused(Params p)')
    # Every physical x layout uses the common padding of the two real jobs.
    for start, end in (
        ('__device__ void wgs_gather(', 'struct WgsPend'),
        ('__device__ __forceinline__ void wgs_epi_tile(', 'template <bool PB>\n__device__ __forceinline__ void wgs_flush('),
    ):
        a, b, r = scope(src, start, end)
        assert '(cnt + 7) >> 3' in r
        r = r.replace('(cnt + 7) >> 3', 'j128_pad(S, j) >> 3')
        src = src[:a] + r + src[b:]
    assert src.count('(S.jC[j] + 7) >> 3') == 4  # two fixup paths + jrow scan + producer
    src = src.replace('(S.jC[j] + 7) >> 3', 'j128_pad(S, j) >> 3')
    src = once(src, 'struct WgsEpi { float c0, c1; int cnt, jst, jrow; };',
               'struct WgsEpi { float c0, c1; int cnt, jst, jrow, ntt; };')
    src = once(src, 'c.cnt = S.jC[j]; c.jst = S.jS[j]; c.jrow = S.jrow[j];',
               'c.cnt = S.jC[j]; c.jst = S.jS[j]; c.jrow = S.jrow[j]; c.ntt = j128_pad(S, j) >> 3;')
    src = once(src, 'const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;',
               'const int col = r16 * 8 + g, ntt = c.ntt;')
    # Correct the final expert credit target and the last-job cache hint.
    src = once(src, 'const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);',
               'const int target = 64 * (tot > SK_T ? j128_njobs(tot, p.M) : 1);')
    src = once(src, 'const int q = n > 1 ? (S.jS[j] - S.off[e]) / FT_T : 0;\nreturn q < n - 1 ?',
               'return S.jS[j] + S.jC[j] < S.off[e] + tot ?')
    # Each logical stage emits one weight copy, for both jobs when paired.
    a, b, r = scope(src, '__device__ void wgs_wproducer(', 'constexpr int WGS_XROWS')
    r = once(r, 'const int j = u >> ubl, ub = u & ((1 << ubl) - 1);',
             'const int j = S.forder[u >> ubl], ub = u & ((1 << ubl) - 1);')
    src = src[:a] + r + src[b:]
    a, b, r = scope(src, '__device__ void wgs_xproducer(', '__device__ void wgs_gather(')
    r = once(r, '++i, ++gs, ++n)', '++i, ++gs)')
    r = once(r, 'const int j = u >> ubl;', r'''const int firstj = S.forder[u >> ubl];
const int members = 1 + j128_pair(S, firstj);
// Slots advance per real x tile; weights advance per logical pair stage.
for (int member = 0; member < members; ++member, ++n) {
const int j = firstj + member;''')
    r = once(r, 'npad = (j128_pad(S, j) >> 3) << 3;', 'npad = j128_pad(S, j);')
    old_ready = '''if (j != curj) {
curj = j;
npad = j128_pad(S, j);
jb = xsrc + (size_t)S.jrow[j] * HD;
if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); }
}
'''
    r = once(r, old_ready, '')
    r = once(r, '// Slots advance per real x tile; weights advance per logical pair stage.', r'''if (firstj != curj) {
curj = firstj;
npad = j128_pad(S, firstj);
jb = xsrc + (size_t)S.jrow[firstj] * HD;
if (!pb && !(OKC_WGSX & 8)) {
  for (int member = 0; member < members; ++member)
    while (ld_acquire(mf_xr(p, S) + firstj + member) < npad) __nanosleep(32);
}
}
// Slots advance per real x tile; weights advance per logical pair stage.''')
    r = once(r, '"l"(jb + (size_t)st * npad * 256)',
             '"l"(jb + (size_t)member * npad * HD + (size_t)st * npad * 256)')
    r = once(r, 'if (++xs == WGS_NX) { xs = 0; xph ^= 1; }',
             'if (++xs == WGS_NX) { xs = 0; xph ^= 1; }\n} // member')
    src = src[:a] + r + src[b:]
    # Keep the unpaired compute loop free of a runtime pairing branch. Its
    # arithmetic and pipeline are unchanged; only the independent x sequence is new.
    a, b, seg = scope(src, 'template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg(',
                      'template <bool PB>\n__device__ __forceinline__ void wgs_phase(')
    original_seg = seg
    seg = once(seg, 'int& seq, WgsPend& pd)', 'int& seq, int& xseq, WgsPend& pd)')
    seg = once(seg, 'xs = seq % WGS_NX;', 'xs = xseq % WGS_NX;')
    seg = once(seg, '(uint32_t)(seq / WGS_NX)', '(uint32_t)(xseq / WGS_NX)')
    seg = once(seg, '++seq;', '++seq;\n++xseq;')
    # Reuse the exact shipped epilogue (including red.v4, deferred counters,
    # HYBST and MFIN) as a helper, once for each accumulator bank.
    ep = original_seg[original_seg.index('if (slot >= 0) {'):]
    finish = r'''template <bool PB, int NTT>
__device__ __forceinline__ void j128_finish(const Params& p, Smem& S, int u, int s0, int s1,
int slot, float (&d)[WGS_NT][4], WgsPend& pd) {
const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
constexpr int UBL = PB ? 4 : 5;
''' + ep
    pair = original_seg[:original_seg.index('#if OKC_HYBST & 1\nif (slot < 0)')]
    pair = once(pair, 'void wgs_seg(', 'void wgs_seg_pair(')
    pair = once(pair, 'int& seq, WgsPend& pd)', 'int& seq, int& xseq)')
    pair = once(pair, 'float d[WGS_NT][4];', 'float d[2][WGS_NT][4];')
    pair = once(pair, 'constexpr int GS = OKC_WGSGS;', 'constexpr int GS = 1; // two HGMMA per commit, sharing the same af')
    pair = once(pair, 'int prev = -1;\n', '')
    pair = once(pair, 'const int ws = seq % WGS_NS, xs = seq % WGS_NX;',
                'const int ws = seq % WGS_NS, xs = xseq % WGS_NX, xs1 = (xseq + 1) % WGS_NX;')
    pair = once(pair, '(uint32_t)(seq / WGS_NX)', '(uint32_t)(xseq / WGS_NX)')
    pair = once(pair, 'const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));', r'''wgs_wait(S.bars + BX_F + xs1, (uint32_t)((xseq + 1) / WGS_NX) & 1u, 4100000 + seq);
const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
const uint64_t xd1 = xd0 + (uint64_t)(uint32_t)(xs1 * (WGS_XS >> 4));''')
    h0 = '''if (FIRST && ci == 0 && jj == 0)
WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
else
WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));'''
    h1 = '''if (FIRST && ci == 0 && jj == 0) {
WGSMMA<NTT>::mma0(d[0], af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
WGSMMA<NTT>::mma0(d[1], af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd1);
} else {
WGSMMA<NTT>::mma(d[0], af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
WGSMMA<NTT>::mma(d[1], af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
xd1 + (uint64_t)((lg * 4 + h * GS + jj) * 16));
}'''
    pair = once(pair, h0, h1)
    pair = once(pair, '''if (ci == 0) {
if (prev >= 0) {
if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
prev = -1;
}
}
if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);
''', '')
    pair = once(pair, 'prev = xs;\n++seq;', r'''// Two held x slots + two next slots cannot coexist in a three-slot ring.
// Drain before releasing BOTH slots: conservative, deadlock-free pair-only path.
wg_wait<0>();
if (lane == 0) { mbar_arrive1(S.bars + BX_E + xs); mbar_arrive1(S.bars + BX_E + xs1); }
++seq;
xseq += 2;''')
    pair += r'''stage(std::true_type{});
#pragma unroll 1
for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});
WgsPend pd0{0,0,0,0}, pd1{0,0,0,0};
j128_finish<PB, NTT>(p, S, u, s0, s1, slot, d[0], pd0);
j128_finish<PB, NTT>(p, S, u + (1 << UBL), s0, s1,
                    slot < 0 ? -1 : slot + J128_SLOT_STRIDE, d[1], pd1);
// Complete both members independently, including resetting every consumed partial.
wgs_flush<PB>(p, S, pd0);
wgs_flush<PB>(p, S, pd1);
}
'''
    src = src[:a] + seg + finish + pair + src[b:]
    a, b, r = scope(src, 'template <bool PB>\n__device__ __forceinline__ void wgs_phase(', '// MOEX:PLAN\n#if OKC_PLAN\n#define OKC_WGSU')
    r = once(r, 'int& seq)', 'int& seq, int& xseq)')
    r = once(r, 'const int ntt = (S.jC[u >> UBL] + 7) >> 3;', r'''const int j = S.forder[u >> UBL], realu = (j << UBL) + (u & ((1 << UBL) - 1));
const int ntt = j128_pad(S, j) >> 3;
if (j128_pair(S, j)) {
  wgs_flush<PB>(p, S, pd); // no pending unpaired epilogue crosses a paired segment
  switch (ntt) {
''' + '\n'.join(f'  case {nt}: wgs_seg_pair<PB, {nt}>(p, S, ring, realu, s0, s1, slot, seq, xseq); break;' for nt in range(1, 9)) + r'''
  }
} else {''')
    assert r.count('(p, S, ring, u, s0, s1, slot, seq, pd)') == 8
    r = r.replace('(p, S, ring, u, s0, s1, slot, seq, pd)', '(p, S, ring, realu, s0, s1, slot, seq, xseq, pd)')
    r = once(r, '\ngs += s1 - s0;', '\n}\ngs += s1 - s0;')
    src = src[:a] + r + src[b:]
    # Double-cost logical jobs. Both members necessarily get the same K splits.
    # Retain the warp-parallel PLAN prefix and the exact inverse range mapping.
    a, b, r = scope(src, '// MOEX:PLAN\n#if OKC_PLAN\n#define OKC_WGSU', 'template <bool BIG>\n__global__')
    assert r.count('const int nj = S.njF + S.njS;') == 2
    r = r.replace('const int nj = S.njF + S.njS;', 'const int nj = S.j128_groups;')
    for x in ('j', 'lo'):
        r = r.replace(f'wgs_cost(pb, S.jC[{x}])', f'j128_cost(S, pb, {x})')
    for pb in (0, 1):
        r = r.replace(f'wgs_cost({pb}, S.jC[j])', f'j128_cost(S, {pb}, j)')
    assert 'wgs_cost(' not in r
    r = r.replace('(uint32_t)wgs_unitc(pb)', '(uint32_t)(wgs_unitc(pb) * (1 + j128_pair(S, S.forder[' + 'J128_INDEX' + '])))')
    # Two sites differ only in their group index.
    r = once(r, 'S.forder[J128_INDEX]))), cu', 'S.forder[lo]))), cu')
    r = once(r, 'S.forder[J128_INDEX])));', 'S.forder[j])));')
    assert 'J128_INDEX' not in r
    # Both the serial (unused) and the parallel prefix charge two unit epilogues.
    for cost in ('OKC_WGSU', 'OKC_WGSUB'):
        r = r.replace(f'+ {cost}) <<', f'+ {cost} * (1 + j128_pair(S, S.forder[j]))) <<')
    helper = r'''// Logical group costs use the shared padded N and retain both jobs' cost.
__device__ __forceinline__ int j128_cost(const Smem& S, int pb, int group) {
  const int j = S.forder[group];
  return wgs_cost(pb, j128_pad(S, j)) * (1 + j128_pair(S, j));
}
'''
    src = src[:a] + helper + r + src[b:]
    src = once(src, 'int seq = 0;\nwgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);',
               'int seq = 0, xseq = 0;\nwgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq, xseq);')
    src = once(src, 'wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);',
               'wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq, xseq);')
    src = once(src, 'static int g_okc_pdl = 0;', '''static_assert(((sizeof(okc::Smem) + 127) / 128) * 128 +
  okc::WGS_NS * okc::UNIT_BYTES + okc::WGS_NX * okc::WGS_XS <= 232448, "J128 exceeds H200 dynamic shared memory");
static int g_okc_pdl = 0;''')
    assert '__trap' not in src
    return src


GLUE = r'''
# J128 bit 0: pair hot experts at T>=128. Everything is allocated/validated before capture.
J128 = 0
_J128 = {"mod": None, "mod7": None, "ok": False, "pdl": -1, "pdl7": -1,
         "w": None, "sc": None, "full": None, "nsm": 0, "err": None, "regs": {}}

def _j128_resources(md):
 import re, shutil, subprocess
 obj = shutil.which("cuobjdump")
 if obj is None:
  import triton
  from pathlib import Path
  obj = str(Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump")
 r = subprocess.run([obj, "-res-usage", md.__file__], capture_output=True, text=True, check=True)
 rows = []
 for chunk in re.split(r"Function\s*:?\s+", r.stdout)[1:]:
  if "moe_fused" not in chunk.splitlines()[0]: continue
  reg = re.search(r"REG:(\d+)", chunk)
  loc = re.search(r"LOCAL:(\d+)", chunk)
  if reg and loc: rows.append((int(reg[1]), int(loc[1])))
 if not rows: raise RuntimeError("J128 resource gate found no moe_fused kernels")
 return rows

def _j128_build():
 from torch.utils.cpp_extension import load_inline
 H = _J128
 if H.get("attempted"): return
 H["attempted"], H["ok"] = True, False
 try:
  for f7 in (False, True):
   key = "mod7" if f7 else "mod"
   for regs in (96, 112):
    fl = list(HYB_FLAGS) + (list(HYB_F7) if f7 else [])
    fl += ["-DOKC_SC6=1", "-DOKC_MFIN=1", "-DOKC_PLAN=1", "-DJ128_REGS=%d" % regs,
           "--maxrregcount=%d" % regs, "-Xptxas=-v,-warn-spills"]
    tag = hashlib.sha1((_HYB_MOE_CPP + _J128_CUDA + " ".join(fl)).encode()).hexdigest()[:12]
    md = load_inline(name="j128_" + tag, cpp_sources=_HYB_MOE_CPP, cuda_sources=_J128_CUDA,
                     functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
    rows = _j128_resources(md)
    if all(local == 0 and used <= regs for used, local in rows):
     H[key], H["regs"][key] = md, regs
     break
   else: raise RuntimeError("J128 spills at both 96 and 112 registers: " + key)
 except Exception as ex:
  H["err"] = repr(ex)[:500]
  H["mod"], H["mod7"] = None, None
  print(f"[j128] build OFF: {H['err']}", flush=True)

def _j128_scratch(dev):
 # Copy _hyb_scratch's allocator/guards; only the FP32 split area is larger.
 # CUDA JMAX=112 fits the base Python x allocation's conservative JMAX=136.
 fixf = 2 * 320 * 4 * 4096 + 2 * 320 * 4 + 136 * 16
 sizes = ((8 + 1024 + _HYB_EMAX + 4096, torch.int32, torch.int32, 0x5EADBEEF),
          (_HYB_EMAX + _HYB_EMAX * 4096 + fixf, torch.float32, torch.int32, 0x5EADBEEF),
          ((_HYB_EMAX + 2 * _HYB_XROWS) * 4096, torch.bfloat16, torch.int16, 0x5EAD))
 full, views = [], []
 for n, dt, vt, sv in sizes:
  f = torch.zeros(n + _HYB_GUARD, dtype=dt, device=dev)
  f[n:].view(vt).fill_(sv)
  full.append((f, n, vt, sv))
  views.append(f[:n])
 return views, full

def _j128_prepare(st, w, dev):
 _pt1_wait("j128")
 if not _J128.get("attempted"): _j128_build()
 H = _J128
 H["ok"] = False
 if not _HYB["ok"] or H["mod"] is None or H["mod7"] is None: return
 try:
  H["w"], H["nsm"] = _HYB["w"], _HYB["nsm"]
  H["sc"], H["full"] = _j128_scratch(dev)
  good = True
  # Include all-hot, zero-local, odd partitions, threshold and changing graph inputs.
  for T, hot in ((128, 0), (128, 65), (128, 76), (128, 128), (193, 0),
                 (256, 0), (256, 129), (256, 225), (256, 256), (320, 320), (128, -1)):
   d = _k2a_self_test_inputs(T, 12800 + T + hot, dev, "skewed")
   x = d["hidden_states"].contiguous()
   ids = d["topk_ids"].to(torch.int32).contiguous()
   tw = d["topk_weights"].float().contiguous()
   if hot:
    ids.fill_(-1)
    if hot > 0: ids[:hot, 0] = 0
   d = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw,
        "config": {"local_start": 0, "local_count": K2A_E_LOC}}
   ref = _k2a_ref_fp32(d, w)
   ship = _moexr_call_bank(_HYB, x, ids, tw).clone()
   ship2 = _moexr_call_bank(_HYB, x, ids, tw).clone()
   out = _moexr_call_bank(H, x, ids, tw).clone()
   again = _moexr_call_bank(H, x, ids, tw).clone()
   torch.cuda.synchronize()
   total, row = _k2a_rel_err(out, ref)
   rel, shiprow = _k2a_rel_err(out, ship)
   shiprepeat = _k2a_rel_err(ship2, ship)[0]
   shiptol = max(2.0 * shiprepeat, 1e-6)
   repeat = _k2a_rel_err(again, out)[0]
   finite = bool(torch.isfinite(out.float()).all() and torch.isfinite(again.float()).all())
   ok = bool(finite and total < 1e-2 and row < 5e-2 and rel < 1e-2 and rel <= shiptol and shiprow < 5e-2 and repeat < 1e-3)
   print(f"[j128] T={T} hot={hot} fp32={total:.3e} row={row:.3e} ship={rel:.3e} shiptol={shiptol:.3e} shiprow={shiprow:.3e} repeat={repeat:.3e} -> {ok}", flush=True)
   good &= ok
   # Same captured call, changing routes + x, three replays (both flag parities).
   graph = torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph): go = _moexr_call_bank(H, x, ids, tw)
   for turn in range(3):
    if turn == 1:
     x.mul_(-0.5)
     ids.fill_(-1)
     ids[:max(1, T - 7), 0] = 1
    graph.replay()
    shipg = _moexr_call_bank(_HYB, x, ids, tw).clone()
    shipg2 = _moexr_call_bank(_HYB, x, ids, tw)
    torch.cuda.synchronize()
    gr, grow = _k2a_rel_err(go, shipg)
    gtol = max(2.0 * _k2a_rel_err(shipg2, shipg)[0], 1e-6)
    good &= bool(torch.isfinite(go.float()).all() and gr < 1e-2 and gr <= gtol and grow < 5e-2)
  good &= _moexr_guards(H) and int(H["sc"][0][8+1024+_HYB_EMAX+65]) == 0
  H["ok"] = bool(good)
  if not good: H["err"] = "numerical / graph / guard / MF_ERR gate failed"
 except Exception as ex:
  H["err"] = repr(ex)[:500]
  H["ok"] = False
 print(f"[j128] {'LIVE' if H['ok'] else 'OFF; fallback to ship'} regs={H['regs']}: {H['err']}", flush=True)
'''


def generate(source):
    cu = patch_cuda(literal(source, '_HYB_MOE_CUDA'))
    assert '"""' not in cu
    source = once(source, 'def _moexr_bank():\n', '_J128_CUDA = r"""' + cu + '"""\n' + GLUE + '\ndef _moexr_bank():\n')
    source = once(source, ' return _HYB if MOEXR else None\n',
                  ' if not MOEXR: return None\n return _J128 if J128 == 1 and _J128["ok"] else _HYB\n')
    source = once(source, ' _moexr_prepare_bank(st,w,dev,_HYB)\n',
                  ' _moexr_prepare_bank(st,w,dev,_HYB)\n _j128_prepare(st,w,dev)\n')
    source = once(source, '  jobs.append(("hyb",_hyb_build))\n',
                  '  jobs.append(("hyb",_hyb_build))\n  jobs.append(("j128",_j128_build))\n')
    ast.parse(source)
    return source


def main():
    source = generate(BASE.read_text(encoding='utf-8'))
    if '--check' in sys.argv:
        assert OUT.read_bytes() == source.encode('utf-8'), 'generated lab is stale'
    else:
        OUT.write_bytes(source.encode('utf-8'))
    print('J128 generator PASS', hashlib.sha256(source.encode()).hexdigest(), len(source.encode()))


if __name__ == '__main__':
    main()

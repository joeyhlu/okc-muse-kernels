"""HYA dynamic-tail lab generator; no GPU, imports of torch, or writes by default.

Run: python -B -I tools/hya_dyn_patch.py --check
Emit (lead only): python -B -I tools/hya_dyn_patch.py --write
The implementer runs --check: all four generated modules are built and checked IN
MEMORY, so only the two requested deliverables are created. --write emits dy..dy4.

tma[38:28] = F (0 or 1024 disables, 1..1023 enables); tma[42:39] = C
(zero means four tiles). All lower bits are preserved. Arms (F,C): dy=(870,4),
dy2=(780,2), dy3=(920,2), dy4=(920,8). All arms have identical CUDA/extension names.
For a ONE-build control use dy._launch(data, _mod=mod, tma=0); xa supplies the
original control plan. Do not compare independent extension builds for timing.

CODE FACT: xa passes tb=0 and drains/reinitializes barriers EVERY segment
(xa:998-1008,1132-1136,2409,2778-2780), despite its older continuous-ring comment.
Keep that protocol. Reuse S.pchunk/pnseg/pseg (already allocated), publish with
the EXISTING segment-end CTA barrier plus one descriptor-publication CTA barrier
per chunk. The sole producer leader (tid NT) claims the next ticket before its
cp_wait<0>, while current tiles/consumer epilogue may still be in flight. No new
tile-loop queue, per-tile bytes, setmaxnreg budget, or shared-memory allocation.
The next ticket is prefetched only on the final segment of each chunk.

Static bounds use the existing skew/SEGK mapping on floor(T_virtual*F/1024).
Tail chunks use REAL keys from vmap(that cut), C*64 at a time; ignoring SEGK
there avoids empty charge-only tickets. A request boundary produces two ordinary
segments, not duplicated keys. Each nonempty segment keeps the original merge.
This preserves attention structure, NOT fp32 reduction order/bitwise output.

Counter words (int32 offsets): 30024=ticket, 30025=departed CTAs. _workspace
already zero-allocates >=32768 words. Each CTA's tid 0 arrives exactly once,
AFTER every producer/consumer observes the done sentinel at a CTA barrier.
The last acq_rel arrival resets both words with atomicExch. No reset kernel or
grid wait. Sequential launches/graph replays reuse zero words; concurrent calls
sharing the base's workspace remain unsupported, as in xa. No new early PDL
trigger. The ticket issuer executes pdl_wait before accessing the pool, so an
incoming PDL launch cannot observe the previous launch's not-yet-reset ticket.
F=0/1024 touches neither counter. L8 consumers ALWAYS reach finalize.

Existing counters, unchanged: bp (L1 / non-coop L8 processed keys, last consumer tid0 resets);
8192+bp/12288+bp and 24576+bp (non-flat coop alternating arrivals/epoch);
bp*CS+rank (cluster arrival, last cluster rank leader resets);
30020/30021 and 30022 (L8 flat coop alternating CTA arrivals/epoch, last tid0
resets opposite parity and advances epoch); 30000 is dead fallback code in xa.
L8 coop segments do NOT increment bp; flat_finalize sums/zeros every ws_acc/ws_l
row via all CTA ids, consumer tids 0..255, r=c*ceil(rows/G)+(tid>>3), stride NT/8.
L1 last-request finalizer uses tids 0..127 (fin_nc1), non-coop L8 uses 0..255,
and tid0 resets bp. Producers
128..255 (L1), 256..383 (L8) never finalize. No CTA exits before its shared-data
cleanup barrier; empty CTAs run sentinel/departure and the same L8 finalizer.

Premortem: repeated segment prologue/merge can erase the tail gain; ticket
round trips may not hide under the drain. No speedup claim is made. The requested
6/10/2.5/3 us are untested hypotheses. R-att-4/5/6 current kernel us are
85.6/165.5/97/181 (d16/d32/v16/v32, task's measured source box_out/hmk,skew).
Idealized two-CTA equalization budgets from measured last-loop gaps are
(79.9-78.9)/2=.5, (160.7-158.2)/2=1.25, (94-79.6)/2=7.2,
(177.7-154)/2=11.85 us EST, BEFORE added costs, not predicted net savings
or rigorous bounds on a whole grid (the measured times are medians).
Only the final CTA/grid barrier exposes these gaps; no in-layer carry is proven.

Hot K/V payload remains 2*64*128=16384 B/tile, 8 cp.async16 per producer thread
(128 B/thread), unchanged. Common page_size=1 page indices: L1 bulk 256..272 B
per tile, L8 4 scalar loads/producer thread (2048 requested B, 256 unique B).
Per extra segment, workspace red payload is 16*L*(64+1)*4 = 4160*L B, plus
query/new-row staging and instruction overhead; this is a serious bandwidth risk.
Scheduler overhead: one 32-bit atomic/ticket by tid NT, one 4-byte smem store,
12 B descriptor stores/segment and 3 scalar loads/thread/segment, one 4-byte
pnseg load/thread/chunk, one 4-byte pnseg store/chunk,
one full-CTA publish barrier/chunk. Chunk mapping
is once by tid NT (binary searches + <=B request intersections), not per tile.
Bounded mbar polling replaces the base's unbounded asm loop: old PTX poll is
try_wait + predicated branch; new is try_wait + selp + exit test, and on failure
increment/limit test (SPIN_CHECK). SASS counts/resources are NOT guessed: the
Linux gate records exact instructions, spill bytes and before/after sizeof.
"""

import argparse
import ast
import difflib
import hashlib
import json
import math
from pathlib import Path
import random
import re
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kernels/lab/hya_xa.py"
SHAPES = ((16, 1, False, 1), (32, 1, False, 1),
          (16, 8, False, 1), (32, 8, False, 1))
ARMS = {"dy": (870, 4), "dy2": (780, 2), "dy3": (920, 2), "dy4": (920, 8)}
F_SHIFT, C_SHIFT = 28, 39
TICKET, DONE = 30024, 30025


def bits(f, c):
    assert 0 <= f <= 1024 and 0 <= c <= 15
    return (f << F_SHIFT) | (c << C_SHIFT)


BOUNDED = r'''// Bounded phase poll; sticky error, never a trap (same SPIN_CHECK as global waits).
__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {
  SPIN_T0;
  for (;;) {
    int ready;
    asm volatile("{ .reg .pred q; mbarrier.try_wait.parity.shared::cta.b64 q, [%1], %2;"
                 " selp.b32 %0, 1, 0, q; }" : "=r"(ready)
                 : "r"(smem_u32(bar)), "r"(parity) : "memory");
    if (ready) break;
    SPIN_CHECK;
  }
}
'''

HELPERS = r'''// HYA_DYN: existing shared scratch, no Smem size or producer register-budget change.
constexpr int HYA_DYN_TICKET = 30024, HYA_DYN_DONE = 30025;
__device__ __forceinline__ int dyn_ticket(const Params& p) {
  pdl_wait();  // preceding launch, including its pool resets, must have completed
  return atomicAdd(p.ws_cnt + HYA_DYN_TICKET, 1);
}
__device__ __forceinline__ int64_t dyn_static(const Params& p, int64_t total) {
  return p.dyn_f ? (total * p.dyn_f) / 1024 : total;
}
template <int L, bool F>
__device__ __forceinline__ int dyn_vmap(const Params& p, const Smem<L, F>& S, int v) {
  const int sk = p.segk ? p.segk : (L == 8 ? 600 : 400);
  int a = 0, z = p.B - 1;
  while (a < z) {
    const int m = (a + z + 1) >> 1;
    if (S.cum[m] + m * sk <= v) a = m; else z = m - 1;
  }
  return min(v - a * sk, S.cum[a + 1]);
}
template <int L, bool F>
__device__ __forceinline__ void dyn_describe(const Params& p, Smem<L, F>& S) {
  const int sk = p.segk ? p.segk : (L == 8 ? 600 : 400);
  const int total = S.cum[p.B];
  const int cut = dyn_vmap(p, S, (int)dyn_static(p, (int64_t)total + (p.B - 1) * sk));
  const int64_t cs = (int64_t)cut + (int64_t)S.pchunk * (p.dyn_c * TILE);
  if (cs >= total) { S.pnseg = -1; return; }  // done sentinel, including an initially empty pool
  const int ce = (int)min((int64_t)total, cs + p.dyn_c * TILE);
  int a = 0, z = p.B - 1;
  while (a < z) {
    const int m = (a + z) >> 1;
    if (S.cum[m + 1] <= cs) a = m + 1; else z = m;
  }
  int n = 0;
  for (int b = a; b < p.B && S.cum[b] < ce; ++b) {
    const int s = max((int)cs, S.cum[b]), e = min(ce, S.cum[b + 1]);
    if (s < e) {
      S.pseg[n][0] = b;
      S.pseg[n][1] = s - S.cum[b];
      S.pseg[n++][2] = e - S.cum[b];
    }
  }
  S.pnseg = n;  // B<=32 is checked by run(), so the MAXSEG=64 array cannot overflow
}
__device__ __forceinline__ void dyn_depart(const Params& p) {
  // All threads have passed the sentinel publication barrier: no more claims in this CTA.
  if (threadIdx.x == 0) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"
                 : "=r"(old) : "l"(p.ws_cnt + HYA_DYN_DONE) : "memory");
    if (old + 1 == (int)(gridDim.x * gridDim.y * gridDim.z)) {
      atomicExch(p.ws_cnt + HYA_DYN_TICKET, 0);
      atomicExch(p.ws_cnt + HYA_DYN_DONE, 0);
    }
  }
}

'''

DYNAMIC_RUN = r'''    if (p.dyn_f) {
      // A zero-static-work CTA did not enter segment() and has no prefetched ticket.
      if (tb == 0 && threadIdx.x == NT)
        S.pchunk = dyn_ticket(p);
      // The preceding segment-end barrier already published a prefetched pchunk.
      // The initial empty-static path has only one writer, which also builds the list.
      for (;;) {
        if (threadIdx.x == NT) dyn_describe(p, S);
        __syncthreads();  // release/acquire pseg, pnseg; every producer + consumer participates
        const int dn = S.pnseg;
        if (dn < 0) break;  // uniform sentinel, including CTAs claiming no tail work
        // Real-key chunks are nonempty; empty requests are skipped by dyn_describe.
#pragma unroll 1
        for (int k = 0; k < dn; ++k) {
          const int b = S.pseg[k][0];
          const int seq = __ldg(p.seq_lens + b);
          const int s0 = S.pseg[k][1], s1 = S.pseg[k][2];
          segment<L, FLAT, R>(p, S, b, pair, s0 == 0, s0, s1, seq, seq + L,
                              vc - fdiv_i(vc, p.nc) * p.nc, 0, 0, 0, false,
                              k + 1 == dn);
          // segment's existing full-CTA end barrier protects pseg reads before overwrite.
        }
      }
      dyn_depart(p);  // no waits; all L8 consumer CTAs still reach flat_finalize below
    }
'''


def assignment(tree, name):
    found = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(found) == 1, name
    return found[0]


def literal(text, name):
    return ast.literal_eval(assignment(ast.parse(text), name).value)


def splice_assignment(text, name, value):
    node = assignment(ast.parse(text), name)
    lines = text.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + value + "\n" + "".join(lines[node.end_lineno:])


def build():
    base = SOURCE.read_text(encoding="ascii")
    cuda = literal(base, "CUDA_SRC")
    original = cuda
    anchors = []

    def edit(name, old, new, count=1):
        nonlocal cuda
        assert cuda.count(old) == count, (name, cuda.count(old), count, old[:100])
        anchors.append({"anchor": name, "old": old, "new": new, "count": count})
        cuda = cuda.replace(old, new)

    edit("params", "  int segk;             // flat kernels: virtual keys charged per request boundary (0: the compiled default 600 / 400)\n",
         "  int segk;             // flat kernels: virtual keys charged per request boundary (0: the compiled default 600 / 400)\n"
         "  int dyn_f, dyn_c;      // tma[38:28] F, [42:39] C; F=0/1024 off, C=0 means 4\n")
    wait_old = r'''// unbounded wait for a phase (producer / consumer hand-off)
__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {
  asm volatile("{\n.reg .pred q;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra WAIT_%=;\n}\n"
               ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
'''
    edit("bounded_mbar_poll", wait_old, BOUNDED)
    edit("helpers", "// Cluster mode, several clusters per request (Params::ncl > 1):", HELPERS + "// Cluster mode, several clusters per request (Params::ncl > 1):")
    edit("segment_claim_argument", "                                        const bool skip_init_ = false) {",
         "                                        const bool skip_init_ = false, const bool dyn_claim_ = false) {")
    edit("prefetch_next_ticket", "        cp_wait<0>();\n        __syncthreads();   // matches the consumers' end-of-segment barrier",
         "        // Claim before the drain: current tiles / merge may still be in flight.\n"
         "        if (dyn_claim_ && q == 0) S.pchunk = dyn_ticket(p);\n"
         "        cp_wait<0>();\n        __syncthreads();   // matches the consumers' end-of-segment barrier")
    for name, old, new in (
        ("generic_decode_static_cut", "const int64_t Tg = (int64_t)S.cum[p.B] + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);",
         "const int64_t Tg = dyn_static(p, (int64_t)S.cum[p.B] + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0));"),
        ("decode_static_cut", "const int64_t Tf = (int64_t)__shfl_sync(FULL, incl_l, 31) + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);",
         "const int64_t Tf = dyn_static(p, (int64_t)__shfl_sync(FULL, incl_l, 31) + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0));"),
        ("verify_static_cut", "const int64_t T = T0 + (int64_t)(p.B - 1) * SEGK;",
         "const int64_t T = dyn_static(p, T0 + (int64_t)(p.B - 1) * SEGK);"),
    ):
        edit(name, old, new)
    edit("empty_static_keeps_cta", "  if (cs >= ce) {\n", "  if (cs >= ce && !p.dyn_f) {\n")
    edit("bounded_segment_scan", "    while (S.cum[b + 1] <= cs) ++b;",
         "    while (b < p.B && S.cum[b + 1] <= cs) ++b;")
    edit("nonempty_static_descriptor", "    for (; b < p.B && S.cum[b] < ce && n < MAXSEG; ++b) {",
         "    for (; cs < ce && b < p.B && S.cum[b] < ce && n < MAXSEG; ++b) {")
    edit("prefetch_on_last_static", "                            L == 1 && Cfg<L, FLAT>::WS && k == 0);",
         "                            L == 1 && Cfg<L, FLAT>::WS && k == 0,\n"
         "                            p.dyn_f && k + 1 == nseg);")
    edit("dynamic_tail_loop", "       // the producer leaves with its copies landed\n", DYNAMIC_RUN + "       // the producer leaves with its copies landed\n")
    host = r'''  p.segk = (int)((tma >> 18) & 1023);
  const int dyn_raw_f = (int)((tma >> 28) & 2047);
  TORCH_CHECK(dyn_raw_f <= 1024, "HYA dyn F must be <=1024");
  p.dyn_f = dyn_raw_f == 1024 ? 0 : dyn_raw_f;
  p.dyn_c = (int)((tma >> 39) & 15);
  if (!p.dyn_c) p.dyn_c = 4;
  TORCH_CHECK(!p.dyn_f || (p.flat && !p.window && p.B > 1 && p.B <= 32 && p.Hkv == 2),
              "HYA dyn is restricted to full FLAT B<=32 with one KV pair");
  TORCH_CHECK(!p.dyn_f || ws_cnt.numel() > HYA_DYN_DONE, "HYA dyn workspace too small");
'''
    edit("host_plan_bits", "  p.segk = (int)((tma >> 18) & 1023);\n", host)
    text = splice_assignment(base, "CUDA_SRC", 'CUDA_SRC = r"""' + cuda + '"""')
    assert text.count('name=f"okc_attn_v298x{tag}_90a"') == 1
    text = text.replace('name=f"okc_attn_v298x{tag}_90a"', 'name=f"okc_attn_v298dy{tag}_90a"')
    base_tuned = ast.get_source_segment(base, assignment(ast.parse(base), "_TUNED"))
    tuned_body = base_tuned[:-1]  # keep both original entries exactly
    arms = {}
    for name, (f, c) in ARMS.items():
        tuned = tuned_body + "".join(f"    {key!r}: dict(tma={bits(f,c)}),\n" for key in SHAPES) + "}"
        arms[name] = splice_assignment(text, "_TUNED", tuned)
        compile(arms[name], f"hya_{name}.py", "exec")
    for anchor in anchors:
        for side, full in (("old", base), ("new", arms["dy"])):
            snippet = anchor[side]
            assert full.count(snippet) == anchor["count"], (anchor["anchor"], side)
            line = full.count("\n", 0, full.index(snippet)) + 1
            anchor[side + "_lines"] = [line, line + snippet.count("\n") - int(snippet.endswith("\n"))]
    # CUDA diff lines are full module coordinates, not ambiguous substring positions.
    hunks = []
    for group in difflib.SequenceMatcher(None, base.splitlines(), arms["dy"].splitlines(), autojunk=False).get_opcodes():
        tag, i, j, a, b = group
        if tag != "equal":
            hunks.append({"kind": tag, "old": [i + 1, j], "new": [a + 1, b]})
    return base, arms, original, cuda, anchors, hunks


class FakeTensor:
    def __init__(self, shape, device="cpu-dry"):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.device = device

    def numel(self):
        return math.prod(self.shape)

    def float(self):
        return self

    def contig(self):
        return self

    contiguous = contig


def dry_namespace(text):
    """Execute REAL _plan/_launch/_workspace/etc; fake only tensor/device/CUDA boundary.

    No guessed layer-state accessor, no local reimplementation of plan selection.
    HYA is the attention module, not a decode_layer state wrapper.
    """
    tree = ast.parse(text)
    keep = {"_cached", "_num_sms", "_splits", "_copies", "_headroom", "_plan", "_launch",
            "_workspace", "_inv_freq"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in keep:
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            # Top-level simple data only; never build/import a CUDA extension.
            try:
                eval(compile(ast.Expression(node.value), "<constant-check>", "eval"), {"dict": dict})
            except (NameError, TypeError):
                continue
            nodes.append(node)
    torch = types.SimpleNamespace(zeros=lambda shape, **kw: FakeTensor(shape, kw["device"]),
                                  empty=lambda shape, **kw: FakeTensor(shape, kw["device"]),
                                  float32="float32", int32="int32", bfloat16="bf16",
                                  cuda=types.SimpleNamespace(get_device_properties=lambda _: types.SimpleNamespace(multi_processor_count=132)))
    ns = {"torch": torch}
    exec(compile(ast.Module(nodes, type_ignores=[]), "<real-HYA-Python>", "exec"), ns)
    return ns


def dry_run(base, arms):
    cases = []
    for name, source in {"xa": base, **arms}.items():
        ns = dry_namespace(source)
        calls = []
        mod = types.SimpleNamespace(run=lambda *args: calls.append(args))
        for b, l, sliding, pairs in (*SHAPES, (1, 1, False, 1), (32, 8, True, 1)):
            cfg = dict(batch=b, q_len=l, num_q_heads=16, num_kv_heads=2, head_dim=64,
                       page_size=1, window=2048 if sliding else 0, sm_scale=.125,
                       qk_scale_factor=1., rope_theta=0)
            data = {"config": cfg, "qkv": FakeTensor((b, l, 4096)), "page_table": FakeTensor((b, 32768))}
            data.update({k: FakeTensor((b,)) for k in ("seq_lens", "cache_loc", "k_cache", "v_cache", "k_scale", "v_scale")})
            for override in ({}, {"tma": 0}, {"tma": bits(1024, 4)}, {"tma": bits(1, 0)}):
                # Only exercise nonzero dyn override on the supported flat shapes.
                if (b, l, sliding, pairs) not in SHAPES and override.get("tma", 0):
                    continue
                out = ns["_launch"](data, _mod=mod, **override)
                args = calls[-1]
                assert out is args[9] and out.shape == (b, l, 1024)
                assert args[12] is ns["_WS"]["cpu-dry"][2]
                assert args[12].numel() > DONE
                assert args[13:15] == (b, l)
                packed = args[-1]
                expected = override.get("tma", bits(*ARMS[name]) if name in ARMS and (b, l, sliding, pairs) in SHAPES else 0)
                assert (packed >> 28) == (expected >> 28)
                if (b, l, sliding, pairs) in SHAPES:
                    assert args[20:23] == (264, 1, 1), args[20:23]
                    assert bool(packed & 2) == (l == 8)
                cases.append((name, b, l, sliding, packed))
            # Identical workspace on repeated launch; actual cache/resize branch also checked below.
            assert calls[-1][12] is ns["_WS"]["cpu-dry"][2]
        prior = ns["_workspace"](1, "cpu-dry")
        assert ns["_workspace"](1, "cpu-dry") is prior
        grown = ns["_workspace"](40000, "cpu-dry")
        assert grown is not prior and grown[2].numel() == 40000
    return {"real_python_launch_paths": len(cases), "workspace_allocation_reuse_growth": True}


def vmap(cum, sk, v):
    a, z = 0, len(cum) - 2
    while a < z:
        m = (a + z + 1) >> 1
        if cum[m] + m * sk <= v:
            a = m
        else:
            z = m - 1
    return min(v - a * sk, cum[a + 1])


def partition_check():
    """Key coverage, segmentation, completion, and repeated-launch ticket lifecycle model.

    This is a CPU protocol model, not a claim of CUDA memory-order/numerical proof.
    Test actual-size and tiny/zero contexts, segment crossings, static-empty CTAs,
    F0/F1024, empty tail at first claim, and alternating off/on repeated launches.
    """
    rng = random.Random(0x485941)
    checked = 0
    max_segments = 0
    pool = [0, 0]  # reused across all simulated launches, including alternating off/on cases
    for b in (16, 32):
        for l in (1, 8):
            for lengths in ([l] * b, [l + 32768] * b, [l + i * 71 for i in range(b)],
                            [l + rng.randrange(50000) for _ in range(b)]):
                cum = [0]
                for n in lengths:
                    cum.append(cum[-1] + n)
                for sk in (0, 1, 400 if l == 1 else 600, 1023):
                    sk = sk or (400 if l == 1 else 600)
                    total = cum[-1] + (b - 1) * sk
                    for f, c in ((0, 4), (1024, 4), (1, 4), (1023, 4), *ARMS.values()):
                        for dq in (0, -127, 127):
                            cut_v = total * f // 1024 if 0 < f < 1024 else total
                            def boundary(i):
                                w = i * (1024 + dq) if i <= 132 else 132 * (1024 + dq) + (i - 132) * (1024 - dq)
                                return vmap(cum, sk, cut_v * w // (1024 * 264))
                            intervals = [(boundary(i), boundary(i + 1)) for i in range(264)]
                            cut = vmap(cum, sk, cut_v)
                            tail = [(s, min(s + c * 64, cum[-1])) for s in range(cut, cum[-1], c * 64)]
                            intervals += tail
                            position = 0
                            per_request = [0] * b
                            for start, end in intervals:
                                assert start == position and end >= start
                                position = end
                                segs = [(j, max(start, cum[j]) - cum[j], min(end, cum[j + 1]) - cum[j])
                                        for j in range(b) if max(start, cum[j]) < min(end, cum[j + 1])]
                                max_segments = max(max_segments, len(segs))
                                assert len(segs) <= 64
                                for j, s, e in segs:
                                    per_request[j] += e - s
                            assert position == cum[-1] and per_request == lengths
                            if f in (0, 1024):
                                assert not tail and pool == [0, 0]
                                checked += 1
                                continue  # disabled launches must never touch the pool
                            # Arbitrary CTA arrival order; one outstanding prefetched ticket per CTA.
                            ticket, arrived = pool
                            assert ticket == arrived == 0
                            seen = []
                            pending = []
                            for owner in range(264):
                                pending.append((owner, ticket)); ticket += 1
                            while pending:
                                idx = rng.randrange(len(pending))
                                owner, value = pending.pop(idx)
                                if value < len(tail):
                                    seen.append(value)
                                    pending.append((owner, ticket)); ticket += 1
                                else:
                                    arrived += 1
                                    if arrived == 264:
                                        assert not pending
                                        ticket = arrived = 0
                            assert sorted(seen) == list(range(len(tail)))
                            assert ticket == arrived == 0  # next graph replay starts identically
                            pool[:] = [ticket, arrived]
                            checked += 1
    # Explicit zero-total tail: every first claim is a sentinel; all finalize owners remain.
    assert vmap([0] * 17, 400, 0) == 0
    for b in (16, 32):
        rows, nt = b * 8 * 16, 256
        rspan = (rows + 263) // 264
        owners = [rbase + (tid >> 3)
                  for c in range(264)
                  for rbase in range(c * rspan, min(rows, (c + 1) * rspan), nt // 8)
                  for tid in range(0, nt, 8)
                  if rbase + (tid >> 3) < min(rows, (c + 1) * rspan)]
        assert sorted(owners) == list(range(rows))
    return {"coverage_and_lifecycle_cases": checked, "max_segments": max_segments,
            "empty_pool_and_all_finalize_rows": True}


def gates(base, arms, original, cuda):
    assert len({literal(src, "CUDA_SRC") for src in arms.values()}) == 1
    normalized = {splice_assignment(src, "_TUNED", "_TUNED = {}") for src in arms.values()}
    assert len(normalized) == 1, "arms must differ ONLY in _TUNED"
    # Every low plan bit and unmodified payload loop survives verbatim.
    for marker, end in (("      for (int x = 0; x < ntiles; ++x) {", "      PROF_DUMP;"),
                        ("  if (do_red) {", "  // (last-CTA finalizes)")):
        a = original[original.index(marker):original.index(end, original.index(marker))]
        z = cuda[cuda.index(marker):cuda.index(end, cuda.index(marker))]
        assert a == z, marker
    for marker, end in (("struct Cfg {", "constexpr int TILE"), ("struct Smem {", "\n};")):
        a = original[original.index(marker):original.index(end, original.index(marker))]
        z = cuda[cuda.index(marker):cuda.index(end, cuda.index(marker))]
        assert a == z, marker
    assert "WAIT_%=" not in cuda and not re.search(r"__trap\s*\(", re.sub(r"//[^\n]*", "", cuda))
    assert cuda.count("dyn_depart(p);") == 1
    assert cuda.count("p.ws_cnt + HYA_DYN_TICKET, 1") == 1
    assert cuda.count("S.pchunk = dyn_ticket(p);") == 2
    # No accidental doubled escapes in generated Python/CUDA.
    for src in arms.values():
        assert literal(src, "CUDA_SRC") == cuda
        assert "\\n    (16, 1" not in src
    report = dry_run(base, arms)
    report.update(partition_check())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="in-memory generator + CPU gates (default)")
    parser.add_argument("--write", action="store_true", help="lead: emit four lab arms after CPU gates pass")
    args = parser.parse_args()
    base, arms, original, cuda, anchors, hunks = build()
    report = gates(base, arms, original, cuda)
    report.update(status="CPU_PASS", cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
                  arms={n: {"F": f, "C": c, "tma": bits(f, c)} for n, (f, c) in ARMS.items()},
                  changed_lines=hunks, anchors=[{k: a[k] for k in ("anchor", "old_lines", "new_lines")} for a in anchors],
                  cuda_compile_sass_numerics_timing="NOT_RUN")
    if args.write:
        for name, src in arms.items():
            path = ROOT / f"kernels/lab/hya_{name}.py"
            path.write_text(src, encoding="ascii", newline="\n")
        report["written"] = [f"kernels/lab/hya_{n}.py" for n in arms]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

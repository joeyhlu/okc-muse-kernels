"""vrts_patches.py (VRT-S, notes/memory/gpu_requests.md): ship decode_layer -> kernels/lab/k3_vrts_1.py.

  python tools/vrts_patches.py [--base=kernels/muse_decode_layer/k3_gc_21cg_s1.py] [--out=kernels/lab/k3_vrts_1.py]

vms_kernel (the T <= 16 MoE) routing front, compile-time VRTS (default 1; 0 = ship code verbatim):
  VRTS 1: warps 0-3 each own one 32-entry round r = warp (load, match_any slice count -> S.base[r], the same per-round
          histogram the ship's warp 0 builds serially); named barrier 2 (128 threads); warp 0 sums the 4 rounds, runs the
          same two scans, job table and S.base prefixes verbatim; barrier 2; each of warps 0-3 scatters its own round's
          ent / pos. Rounds with r * 32 >= m8 skip match_any (they hold no entries). sm_90: first ring slots q < NS are
          issued by lane 0 of warp q instead of tid 0.
  VRTS 2: ship's single warp 0, but rounds with r * 32 >= m8 are skipped (warp-uniform), no new barriers.
ent / pos / jobs / S.base / njobs / gen are bitwise the ship's (same scan order, same S.base semantics):
tools/vrts_local_test.py.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)


def _rep(src, old, new, what, n=1):
    c = src.count(old)
    if c != n:
        raise SystemExit(f"[vrts_patches] {what}: expected {n} match(es), got {c}")
    return src.replace(old, new)


ROUTE_HEAD = "  if (warp == 0) {\n    const int m8 = p.m * 8;\n    int idv[4]; unsigned mm[4];\n"
ROUTE_TAIL = "        else S.pos[i] = -1;\n      }\n    }\n  }\n  __syncthreads();\n  const int njobs = S.njobs;\n"

SCAN = r"""    int inc = tot, nj = (tot + 7) >> 3, jinc = nj;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, inc, d), b = __shfl_up_sync(0xffffffffu, jinc, d);
      if (lane >= d) { inc += a; jinc += b; }
    }
    const int off = inc - tot, joff = jinc - nj;
    for (int k = 0; k < nj; ++k) { S.jE[joff + k] = lane; S.jS[joff + k] = off + 8 * k; S.jC[joff + k] = min(8, tot - 8 * k); }
    int run = off;
#pragma unroll
    for (int r = 0; r < 4; ++r) { S.base[r][lane] = run; run += cr[r]; }
    if (lane == 31) S.njobs = jinc;
"""

VRTS1 = r"""  if (warp < 4) {   // VRTS 1: round r = warp
    const int m8 = p.m * 8, rr = warp, i = rr * 32 + lane;
    const bool live = rr * 32 < m8;   // warp-uniform
    int idv = i < m8 ? ((__ldcg(p.ids + i) - p.lo) & 255) : 255;
    if (tid == 0) S.gen = ld_rlx(p.ctr);
    if (idv >= NE || idv < 0) idv = -1;
    S.base[rr][lane] = 0;
    __syncwarp();
    unsigned mm = 0u;
    if (live) {
      mm = __match_any_sync(0xffffffffu, idv);
      if (idv >= 0 && lane == __ffs(mm) - 1) S.base[rr][idv] = __popc(mm);
    }
    asm volatile("bar.sync 2, 128;\n" ::: "memory");
    if (warp == 0) {
    int cr[4], tot = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r) { cr[r] = S.base[r][lane]; tot += cr[r]; }
""" + SCAN + r"""    }
    asm volatile("bar.sync 2, 128;\n" ::: "memory");
    if (i < m8) {
      if (idv >= 0) { const int pr = S.base[rr][idv] + __popc(mm & ((1u << lane) - 1u)); S.ent[pr] = i; S.pos[i] = pr; }
      else S.pos[i] = -1;
    }
  }
  __syncthreads();
"""

VRTS2 = r"""  if (warp == 0) {   // VRTS 2: ship warp 0, dead rounds skipped
    const int m8 = p.m * 8;
    int idv[4]; unsigned mm[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) { const int i = r * 32 + lane; idv[r] = i < m8 ? ((__ldcg(p.ids + i) - p.lo) & 255) : 255; }
    if (tid == 0) S.gen = ld_rlx(p.ctr);
#pragma unroll
    for (int r = 0; r < 4; ++r) { if (idv[r] >= NE || idv[r] < 0) idv[r] = -1; S.base[r][lane] = 0; }
    __syncwarp();
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      mm[r] = 0u;
      if (r * 32 < m8) {
        mm[r] = __match_any_sync(0xffffffffu, idv[r]);
        if (idv[r] >= 0 && lane == __ffs(mm[r]) - 1) S.base[r][idv[r]] = __popc(mm[r]);
      }
    }
    __syncwarp();
    int cr[4], tot = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r) { cr[r] = S.base[r][lane]; tot += cr[r]; }
""" + SCAN + r"""    __syncwarp();
    const unsigned lt = (1u << lane) - 1u;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int i = r * 32 + lane;
      if (r * 32 < m8 && i < m8) {
        if (idv[r] >= 0) { const int pr = S.base[r][idv[r]] + __popc(mm[r] & lt); S.ent[pr] = i; S.pos[i] = pr; }
        else S.pos[i] = -1;
      }
    }
  }
  __syncthreads();
"""

ISSUE_OLD = """#if __CUDA_ARCH__ >= 900
      if (tid == 0)
#endif
      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);"""
ISSUE_NEW = """#if __CUDA_ARCH__ >= 900
#if VRTS == 1
      if (tid == 32 * q)
#else
      if (tid == 0)
#endif
#endif
      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);"""


def patch(src):
    k = src.index("vms_kernel(Params p) {")
    a = src.index(ROUTE_HEAD, k)
    b = src.index(ROUTE_TAIL, a) + len(ROUTE_TAIL)
    old = src[a:b]
    if old.count("__syncthreads();") != 1:
        raise SystemExit("[vrts_patches] route block anchor drift")
    keep = "  const int njobs = S.njobs;\n"
    ship_block = old[: -len(keep)]
    new = ("#if VRTS == 1\n" + VRTS1 + "#elif VRTS == 2\n" + VRTS2 + "#else\n" + ship_block + "#endif\n" + keep)
    src = src[:a] + new + src[b:]
    k2 = src.index("vms_kernel(Params p) {")
    seg_end = src.index("const uint32_t k16 = 16u + (uint32_t)(p.m >> 30);", k2)
    if src.count(ISSUE_OLD, k2, seg_end) != 1:
        raise SystemExit("[vrts_patches] issue anchor drift")
    i = src.index(ISSUE_OLD, k2)
    src = src[:i] + ISSUE_NEW + src[i + len(ISSUE_OLD):]
    src = _rep(src, "#ifndef VMS_FX\n#define VMS_FX 0\n#endif\n",
               "#ifndef VMS_FX\n#define VMS_FX 0\n#endif\n#ifndef VRTS\n#define VRTS 1\n#endif\n", "VRTS define")
    return src


def main():
    base = ROOT / ARGS.get("base", "kernels/muse_decode_layer/k3_gc_21cg_s1.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_vrts_1.py")
    src = patch(base.read_text(encoding="utf-8"))
    out.write_text(src, encoding="utf-8", newline="\n")
    print(f"[vrts_patches] {base.name} -> {out.relative_to(ROOT)} ({len(src.encode())} bytes)")


if __name__ == "__main__":
    main()

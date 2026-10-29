"""vrts_local_test.py: local RTX 3070 check of VRT-S (tools/vrts_patches.py) against the ship's vms_kernel routing front.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vrts_local_test.py" [ship.py] [vrts.py]

Takes the vms namespace prelude and the routing region (after griddepcontrol.wait up to `const int njobs`) verbatim from
both files, wraps each in a probe kernel that sentinel-fills the Smem int block (njobs, gen, base, jE/jS/jC, ent, pos),
runs the region and dumps the block. Bitwise: ship vs VRTS 1 / 2 over 320+ routings (T = 1..16, lo 0 / 32 / 96 / 224,
random / hot / all-local / none-local / same-for-all) + a negative control. Timing: clock64 at region entry / after the
__syncthreads (tid 0, not in any loop), median over CTAs x launches, ids L2-warm.
"""
import pathlib
import random
import sys

import torch
from torch.utils.cpp_extension import load_inline

ROOT = pathlib.Path(__file__).resolve().parents[1]
pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
SHIP = ROOT / (pos_args[0] if len(pos_args) > 0 else "kernels/muse_decode_layer/k3_gc_21cg_s1.py")
VRTS = ROOT / (pos_args[1] if len(pos_args) > 1 else "kernels/lab/k3_vrts_1.py")
GRID = 46


def pieces(path):
    s = path.read_text(encoding="utf-8")
    a = s.index("namespace vms {\nconstexpr int HD = 4096")
    b = s.index("\n", s.index("__device__ __forceinline__ void cta_sync_compute()", a)) + 1
    k = s.index("vms_kernel(Params p) {", b)
    g = s.index("#endif\n", s.index('griddepcontrol.wait;\\n', k)) + len("#endif\n")
    e = s.index("  const int njobs = S.njobs;\n", g)
    return s[a:b], s[g:e]


KER = r"""
__global__ void __launch_bounds__(NTL, 1) rt_kernel(Params p, int* out, long long* tm, int init) {
  extern __shared__ __align__(128) unsigned char smem[];
  Smem& S = *reinterpret_cast<Smem*>(smem);
  unsigned char* ring = smem + RING_OFF; (void)ring;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c = blockIdx.x;
  constexpr int nI = 2 + 4 * NE + 3 * MAXJ + 2 * MAXP;
  int* SI = &S.njobs;
  if (init) for (int k = tid; k < nI; k += NTL) SI[k] = -7777;
  __syncthreads();
  const long long t0 = clock64();
@REGION@
  const long long t1 = clock64();
  if (tid == 0) tm[c] = t1 - t0;
  if (init) for (int k = tid; k < nI; k += NTL) out[c * nI + k] = SI[k];
}
}  // namespace vms
"""


def build():
    pre_s, reg_s = pieces(SHIP)
    pre_v, reg_v = pieces(VRTS)
    head = ("#include <cuda_bf16.h>\n#include <stdint.h>\n#include <torch/extension.h>\n"
            "#define VMS_NS 3\n#define VMS_MK2 0\n#define VMS_RO 0\n#define VMS_NRB 4\n#define VMS_FX 0\n#define VMS_G 132\n")
    src = head
    for name, pre, reg, v in (("V0", pre_s, reg_s, 0), ("V1", pre_v, reg_v, 1), ("V2", pre_v, reg_v, 2)):
        src += f"#undef VRTS\n#define VRTS {v}\nnamespace {name} {{\n" + pre + KER.replace("@REGION@", reg) + "}\n"
    src += r"""
void run(torch::Tensor ids, torch::Tensor ctr, torch::Tensor out, torch::Tensor tm, int64_t m, int64_t lo, int64_t init,
         int64_t var, int64_t grid) {
  V0::vms::Params p{};
  p.ids = ids.data_ptr<int>(); p.ctr = ctr.data_ptr<int>(); p.m = (int)m; p.lo = (int)lo;
  const size_t sm = sizeof(V0::vms::Smem);
  if (var == 0) V0::vms::rt_kernel<<<grid, V0::vms::NTL, sm>>>(p, out.data_ptr<int>(), (long long*)tm.data_ptr<int64_t>(), (int)init);
  else if (var == 1) V1::vms::rt_kernel<<<grid, V1::vms::NTL, sm>>>(*reinterpret_cast<V1::vms::Params*>(&p), out.data_ptr<int>(), (long long*)tm.data_ptr<int64_t>(), (int)init);
  else V2::vms::rt_kernel<<<grid, V2::vms::NTL, sm>>>(*reinterpret_cast<V2::vms::Params*>(&p), out.data_ptr<int>(), (long long*)tm.data_ptr<int64_t>(), (int)init);
}
"""
    return load_inline(name="vrts_lt", cpp_sources="void run(torch::Tensor ids, torch::Tensor ctr, torch::Tensor out, "
                       "torch::Tensor tm, int64_t m, int64_t lo, int64_t init, int64_t var, int64_t grid);",
                       cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", "-arch=sm_86"])


def routing(rng, m, lo, mode):
    loc = list(range(lo, lo + 32))
    rows = []
    same = rng.sample(range(256), 8)
    for t in range(m):
        if mode == "random":
            r = rng.sample(range(256), 8)
        elif mode == "hot":
            r = loc[:8]
        elif mode == "local":
            r = rng.sample(loc, 8)
        elif mode == "none":
            r = rng.sample([e for e in range(256) if e not in loc], 8)
        else:
            r = list(same)
        rng.shuffle(r)
        rows += r
    return rows


def main():
    mod = build()
    dev = "cuda"
    nI = 2 + 4 * 32 + 3 * 64 + 2 * 128
    out = {v: torch.empty(GRID * nI, dtype=torch.int32, device=dev) for v in (0, 1, 2)}
    tm = torch.zeros(GRID, dtype=torch.int64, device=dev)
    ctr = torch.zeros(64, dtype=torch.int32, device=dev)
    rng = random.Random(4242)
    n = bad = 0
    for lo in (0, 32, 96, 224):
        for mode in ("random", "hot", "local", "none", "same"):
            for m in range(1, 17):
                rows = routing(rng, m, lo, mode)
                ids = torch.tensor(rows + [rng.randrange(256) for _ in range(128 - len(rows))], dtype=torch.int32, device=dev)
                ctr[0] = rng.randrange(1 << 20)
                for v in (0, 1, 2):
                    mod.run(ids, ctr, out[v], tm, m, lo, 1, v, GRID)
                torch.cuda.synchronize()
                ref = out[0].view(GRID, nI)
                assert bool((ref == ref[0]).all()), "ship not CTA-uniform"
                for v in (1, 2):
                    if not torch.equal(out[v], out[0]):
                        bad += 1
                        if bad <= 3:
                            print(f"MISMATCH v{v} lo={lo} mode={mode} m={m}")
                n += 1
    # negative control: one id changed for the VRTS run only
    rows = routing(rng, 8, 0, "local")
    ids = torch.tensor(rows + [0] * (128 - 64), dtype=torch.int32, device=dev)
    ids2 = ids.clone(); ids2[5] = [e for e in range(32) if e not in rows[0:8]][0]
    mod.run(ids, ctr, out[0], tm, 8, 0, 1, 0, GRID)
    mod.run(ids2, ctr, out[1], tm, 8, 0, 1, 1, GRID)
    torch.cuda.synchronize()
    neg = not torch.equal(out[0], out[1])
    print(f"bitwise: {n} routings x 2 variants, mismatches {bad}; negative control caught: {neg}")
    # timing
    reps = 400
    for m in (1, 8, 16):
        rows = routing(rng, m, 0, "random")
        ids = torch.tensor(rows + [0] * (128 - len(rows)), dtype=torch.int32, device=dev)
        res = {v: [] for v in (0, 1, 2)}
        for r in range(reps):
            for v in ((0, 1, 2) if r % 2 == 0 else (2, 1, 0)):
                mod.run(ids, ctr, out[v], tm, m, 0, 0, v, GRID)
                res[v].append(tm.clone())
        torch.cuda.synchronize()
        med = {v: float(torch.cat(res[v][20:]).double().median()) for v in res}
        print(f"T={m:2d} region cycles median: ship {med[0]:.0f}  vrts1 {med[1]:.0f} ({med[1]-med[0]:+.0f})  "
              f"vrts2 {med[2]:.0f} ({med[2]-med[0]:+.0f})  [us @1.98 GHz: {med[0]/1980:.3f} / {(med[1]-med[0])/1980:+.3f} / "
              f"{(med[2]-med[0])/1980:+.3f}]")


if __name__ == "__main__":
    main()

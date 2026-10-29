"""aw_local_test.py (moe-aw step 0/1, $0): run the REAL VM_RT route() of kernels/lab/k3_aw_1.py with VM_AW 0 (ship) and 1 on the
local RTX 3070 (sm_86) for saved layer routings (box_out/m32e14, box_out/vml14 run.json per-expert counts, also expert order
reversed) + random stress cases, and check the plans on the CPU.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python3 "<repo>/tools/aw_local_test.py" OUTDIR [stress=400]

Checks (every case, both variants bf2 = JN 64 / 576 thr at T 32, w72 = JN 72 / WG5 2 / 512 thr at T 128 / 256; shipped
VML_COST per T, stress: random tables): all CTAs' Tabs identical; AW and ship give identical job tables (nj, jC, cpA, cpB);
bA / bB non-decreasing, bA[0] = bB[0] = 0, bA[G] = nj << 9, bB[G] = nj << 8; AW: every bA[c] % 16 == 0.
Prints per T (saved routings, G 132): A split shares per CTA (ship vs AW), max |A+B - target| / target per CTA, B share min,
max per-CTA A+B cost over target (the plan's end), route cycles.
"""
import json
import pathlib
import random
import re
import struct
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import m32e_patches as P  # noqa: E402

LAB = ROOT / "kernels/lab/k3_aw_1.py"
GATHER_HEAD = "__device__ void gather(const Args& A, const Tab& T, const uint16_t* ent) {\n"
NOUT = 4 + 161 * 2 + 81 * 3   # nj, G, cyc lo, cyc hi, bA, bB, cpA, cpB, jC
MAIN = r"""
using namespace vm;
__global__ void k_route(Args A, uint8_t* oT) {
extern __shared__ __align__(128) uint8_t sm[];
Tab& T = *reinterpret_cast<Tab*>(sm);
uint8_t* scr = sm + (((int)sizeof(Tab) + 127) & ~127);
for (int i = threadIdx.x; i < (int)sizeof(Tab); i += blockDim.x) sm[i] = 0x5A;
for (int i = threadIdx.x; i < 16384; i += blockDim.x) scr[i] = 0xEE;
__syncthreads();
const long long c0 = clock64();
route(A, T, reinterpret_cast<int*>(scr), reinterpret_cast<uint16_t*>(scr + 4096), reinterpret_cast<uint32_t*>(scr + 12288));
__syncthreads();
if (threadIdx.x == 0 && blockIdx.x == 0) A.marks[0] = clock64() - c0;
for (int i = threadIdx.x; i < (int)sizeof(Tab); i += blockDim.x) oT[(size_t)blockIdx.x * sizeof(Tab) + i] = sm[i];
}
int main(int argc, char** argv) {
FILE* fi = fopen(argv[1], "rb"); FILE* fo = fopen(argv[2], "wb");
const int nth = VM_WG5 >= 2 ? 32 * NCW : NTH;
const int smem = (((int)sizeof(Tab) + 127) & ~127) + 16384;
cudaFuncSetAttribute(k_route, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
int *ids, *rmap; uint8_t* oT; long long* mk;
cudaMalloc(&mk, 16 * 8); cudaMalloc(&ids, 2560 * 4); cudaMalloc(&rmap, 2560 * 4); cudaMalloc(&oT, 160 * sizeof(Tab));
static int h_ids[2560]; static uint8_t h_T[160 * sizeof(Tab)];
int hdr[3];
while (fread(hdr, 4, 3, fi) == 3) {
const int M = hdr[0], G = hdr[1], ls = hdr[2];
int cost[2 * VM_NTM];
fread(cost, 4, 2 * VM_NTM, fi); fread(h_ids, 4, M * 8, fi);
cudaMemcpy(ids, h_ids, M * 32, cudaMemcpyHostToDevice);
Args A; memset(&A, 0, sizeof(A));
A.ids = ids; A.marks = mk; A.M = M; A.ls = ls; A.rmap = rmap; A.flags = 193;
A.magG = ((1ull << 40) + G - 1) / G;
for (int k = 0; k < 2 * VM_NTM; ++k) { A.cost[k] = cost[k]; A.mag[k] = ((1ull << 40) + cost[k] - 1) / cost[k]; }
k_route<<<G, nth, smem>>>(A, oT);
cudaError_t e = cudaDeviceSynchronize();
if (e != cudaSuccess) { printf("CUDA error %s\n", cudaGetErrorString(e)); return 1; }
cudaMemcpy(h_T, oT, G * sizeof(Tab), cudaMemcpyDeviceToHost);
int same = 1;
for (int c = 1; c < G; ++c) same &= memcmp(h_T, h_T + c * sizeof(Tab), sizeof(Tab)) == 0;
long long cy; cudaMemcpy(&cy, mk, 8, cudaMemcpyDeviceToHost);
const Tab* t = reinterpret_cast<const Tab*>(h_T);
int o[4 + 161 * 2 + 81 * 3]; memset(o, 0, sizeof(o));
o[0] = same ? t->nj : -1; o[1] = G; o[2] = (int)(cy & 0x7fffffff); o[3] = 0;
for (int c = 0; c <= G; ++c) { o[4 + c] = t->bA[c]; o[4 + 161 + c] = t->bB[c]; }
for (int j = 0; j <= t->nj && j <= 80; ++j) { o[4 + 322 + j] = (int)t->cpA[j]; o[4 + 322 + 81 + j] = (int)t->cpB[j]; }
for (int j = 0; j < t->nj && j < 80; ++j) o[4 + 322 + 162 + j] = t->jC[j];
fwrite(o, 4, 4 + 161 * 2 + 81 * 3, fo);
}
fclose(fo);
return 0;
}
"""


def _defs():
    import ast
    t = LAB.read_text(encoding="utf-8")
    a = t.index("\n_VML_DEFS = {") + len("\n_VML_DEFS = ")
    return ast.literal_eval(t[a:t.index("\n_VML = {", a)])


VARIANTS = {v: tuple(d for d in _defs()[v] if d.startswith("-D")) for v in ("bf2", "w72")}
NTM = {"bf2": 8, "w72": 9}


def test_src():
    t = LAB.read_text(encoding="utf-8")
    a, b = P.cuda_span(t, "_vml_CUDA_T")
    cuda = t[a:b].replace("@JMAX@", "80").replace("@SLOTS@", "136").replace("@FIXW@", "4096")
    cuda = "\n".join(ln for ln in cuda.split("\n") if not re.match(r"#include <(torch|ATen|c10)", ln))
    head = cuda[:cuda.index("__device__ __forceinline__ int split_point")]
    bar = "\n".join(ln for ln in cuda.split("\n") if ln.startswith(("__device__ __forceinline__ void bar_sync(",
                                                                    "__device__ __forceinline__ void bar_arrive(")))
    body = cuda[cuda.index("__device__ __forceinline__ uint32_t mdiv"):cuda.index(GATHER_HEAD)]
    return "#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n" + head + bar + "\n" + body + "}\n" + MAIN


def ext_cost(c, ntm):   # = _vml_cost_ext
    if len(c) == 2 * ntm:
        return list(c)
    out = []
    for pb in (0, 1):
        h = c[8 * pb:8 * pb + 8]
        step = max(1, h[7] - h[6])
        out += h + [h[7] + step * (k - 8) for k in range(9, ntm + 1)]
    return out


def ids_from_counts(counts, M):
    ids = [[] for _ in range(M)]
    i = 0
    for e, n in enumerate(counts):
        for _ in range(n):
            ids[i % M].append(e)
            i += 1
    assert all(len(r) <= 8 for r in ids)
    return [r + [32 + q for q in range(8 - len(r))] for r in ids]


def cases():
    vc = None
    for src in ("box_out/m32e14/run.json", "box_out/vml14/run.json"):
        d = json.loads((ROOT / src).read_text())
        for r in d["routings"]:
            for rev in (0, 1):
                cnt = list(r["counts"])[::-1] if rev else list(r["counts"])
                yield dict(tag=f"{src.split('/')[1]}:{r['rs'] % 1000}:{r['T']}:{r['seed']}:{rev}", T=r["T"], M=r["T"], G=132, ls=0,
                           counts=cnt, cost=None)
    rng = random.Random(7)
    for i in range(int(STRESS)):
        M = rng.choice([32, 33, 40, 48, 63, 64, 96, 128, 129, 160, 200, 256, 300, 320])
        G = rng.choice([132, 132, 136, 77, 1, 100])
        mode = rng.randrange(4)
        E = rng.choice([1, 2, 5, 12, 20, 32])
        tot = min(8 * M, sum(min(M, rng.randrange(1, M + 1)) for _ in range(E)))
        cnt = [0] * 32
        for _ in range(tot):
            e = rng.randrange(E) if mode else min(E - 1, int(rng.expovariate(0.5)))
            if cnt[e] < M:
                cnt[e] += 1
        perm = list(range(32))
        rng.shuffle(perm)
        cnt = [cnt[perm[k]] for k in range(32)]
        cost = None if i % 3 == 0 else [100 + rng.randrange(300) for _ in range(20)]
        yield dict(tag=f"stress{i}", T=256 if M > 160 else 128 if M > 48 else 32, M=M, G=G, ls=0, counts=cnt, cost=cost)


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    src = out / "aw_test.cu"
    src.write_text(test_src(), encoding="utf-8")
    VC = {256: [128, 153, 172, 186, 198, 216, 232, 248, 128, 149, 168, 180, 191, 208, 219, 236],
          128: [128, 157, 174, 194, 208, 230, 246, 262, 128, 154, 179, 189, 219, 229, 250, 267],
          32: [128, 167, 183, 229, 171, 182, 193, 204, 128, 162, 184, 215, 171, 182, 193, 204]}
    lab_vc = re.search(r"\nVML_COST = (\{.*\})\n", LAB.read_text(encoding="utf-8")).group(1)
    assert eval(lab_vc) == VC, "VML_COST drifted"  # noqa: S307
    cs = list(cases())
    res, exes = {}, {}

    def build(v, aw):
        exe = out / f"aw_{v}_{aw}"
        p = subprocess.run(["nvcc", "-arch=sm_86", "-O3", "-std=c++17", "-w", f"-DVM_AW={aw}", *VARIANTS[v], str(src), "-o", str(exe)],
                           capture_output=True, text=True)
        exes[(v, aw)] = (p.returncode, p.stderr[-2500:], exe)
    jobs = [(v, aw) for v in VARIANTS for aw in AWS]
    for i in range(0, len(jobs), 2):
        ths = [threading.Thread(target=build, args=j) for j in jobs[i:i + 2]]
        [t.start() for t in ths]
        [t.join() for t in ths]
    for k, (rc, err, _) in exes.items():
        if rc:
            print(f"[aw] BUILD FAIL {k}\n{err}")
            return 1
    for v in VARIANTS:
        sel = [c for c in cs if (c["T"] == 32) == (v == "bf2")]
        fin = out / f"in_{v}.bin"
        with open(fin, "wb") as f:
            for c in sel:
                cost = ext_cost(c["cost"][:16] if c["cost"] else VC[c["T"]], NTM[v])
                f.write(struct.pack("3i", c["M"], c["G"], c["ls"]) + struct.pack(f"{len(cost)}i", *cost))
                f.write(struct.pack(f"{c['M'] * 8}i", *[x for r in ids_from_counts(c["counts"], c["M"]) for x in r]))
                c["xc"] = cost
        for aw in AWS:
            fo = out / f"out_{v}_{aw}.bin"
            p = subprocess.run([str(exes[(v, aw)][2]), str(fin), str(fo)], capture_output=True, text=True, timeout=120)
            if p.returncode:
                print(f"[aw] RUN FAIL {v} {aw}: {p.stdout[-500:]}")
                return 1
            raw = fo.read_bytes()
            assert len(raw) == 4 * NOUT * len(sel), (len(raw), len(sel))
            for i, c in enumerate(sel):
                res[(c["tag"], aw)] = struct.unpack_from(f"{NOUT}i", raw, 4 * NOUT * i)
        for c in sel:
            c["v"] = v
    return report(cs, res)


def plan(o):
    nj, G, cyc = o[0], o[1], o[2]
    bA, bB = list(o[4:4 + G + 1]), list(o[165:165 + G + 1])
    cpA, cpB = [x & 0xffffffff for x in o[326:326 + nj + 1]], [x & 0xffffffff for x in o[407:407 + nj + 1]]
    jC = list(o[488:488 + nj])
    return nj, G, cyc, bA, bB, cpA, cpB, jC


def costf(g, sh, cp, nj):
    j = g >> sh
    if j >= nj:
        return cp[nj]
    return cp[j] + (g - (j << sh)) * (cp[j + 1] - cp[j]) // (1 << sh)


def report(cs, res):
    import statistics as S
    bad, rows = [], {}
    for c in cs:
        if 2 in AWS and (res[(c["tag"], 1)][4:] != res[(c["tag"], 2)][4:] or res[(c["tag"], 1)][0] != res[(c["tag"], 2)][0]):
            bad.append((c["tag"], "AW 2 != AW 1"))
        if 2 in AWS:
            rows.setdefault(("cyc2", c["T"]), []).append(res[(c["tag"], 2)][2] - res[(c["tag"], 1)][2])
        p0, p1 = plan(res[(c["tag"], 0)]), plan(res[(c["tag"], 1)])
        for aw, p in ((0, p0), (1, p1)):
            nj, G, cyc, bA, bB, cpA, cpB, jC = p
            e = []
            if nj < 0:
                e.append("CTAs differ")
            else:
                if any(bA[i + 1] < bA[i] for i in range(G)) or any(bB[i + 1] < bB[i] for i in range(G)):
                    e.append("non-monotone")
                if bA[0] or bB[0] or bA[G] != nj << 9 or bB[G] != nj << 8:
                    e.append(f"ends {bA[0]} {bB[0]} {bA[G]} {bB[G]} nj {nj}")
                if aw and any(x % 16 for x in bA):
                    e.append("A cut off unit edge")
            if e:
                bad.append((c["tag"], aw, e))
        if p0[0] >= 0 and p0[0] != p1[0] or p0[5:] != p1[5:]:
            bad.append((c["tag"], "jobs differ"))
        if c["tag"].startswith("stress") or p0[0] <= 0:
            continue
        for aw, p in ((0, p0), (1, p1)):
            nj, G, cyc, bA, bB, cpA, cpB, jC = p
            tot = cpA[nj] + cpB[nj]
            tgt = tot / G
            sh = [int(bA[i] % 16 != 0) + int(bA[i + 1] % 16 != 0) for i in range(G)]
            ca = [costf(bA[i + 1], 9, cpA, nj) - costf(bA[i], 9, cpA, nj) for i in range(G)]
            cb = [costf(bB[i + 1], 8, cpB, nj) - costf(bB[i], 8, cpB, nj) for i in range(G)]
            dev = [abs(a + b - tgt) / tgt for a, b in zip(ca, cb)]
            mx = max(a + b for a, b in zip(ca, cb)) / tgt - 1
            r = rows.setdefault((c["T"], aw), {k: [] for k in ("sh", "shmax", "dev", "mx", "bmin", "amin", "cyc", "unit")})
            r["sh"].append(sum(sh) / G); r["shmax"].append(max(sh)); r["dev"].append(max(dev)); r["mx"].append(mx)
            r["bmin"].append(min(bB[i + 1] - bB[i] for i in range(G))); r["amin"].append(min(bA[i + 1] - bA[i] for i in range(G)))
            r["cyc"].append(cyc); r["unit"].append(16 * max(cpA[j + 1] - cpA[j] for j in range(nj)) / 512 / tgt)
    for T in (32, 128, 256):
        if ("cyc2", T) in rows:
            print(f"[aw] T={T:3d} route cycles AW2 - AW1 median {S.median(rows.pop(('cyc2', T))):+.0f}")
    for (T, aw), r in sorted(rows.items()):
        print(f"[aw] T={T:3d} {'AW  ' if aw else 'ship'} n {len(r['sh'])}: A split shares/CTA mean {S.mean(r['sh']):.2f} (max {max(r['shmax'])}) | "
              f"max|A+B-tgt|/tgt med {S.median(r['dev']):.3f} max {max(r['dev']):.3f} | plan end over tgt med {S.median(r['mx']):+.3f} "
              f"max {max(r['mx']):+.3f} | min B stages/CTA {min(r['bmin'])} min A {min(r['amin'])} | A unit/tgt {S.median(r['unit']):.2f} | "
              f"route cyc med {S.median(r['cyc']):.0f}")
    print(f"[aw] cases {len(cs)} (saved {sum(1 for c in cs if not c['tag'].startswith('stress'))}), failures {len(bad)}: {bad[:6]}")
    print("[aw] PLAN PASS" if not bad else "[aw] PLAN FAIL")
    return 1 if bad else 0


AWS = (0, 1)   # AW 2 (speculative B search) dropped: identical plans, +466..+632 route cycles
STRESS = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("stress=")), 400)
if __name__ == "__main__":
    sys.exit(main())

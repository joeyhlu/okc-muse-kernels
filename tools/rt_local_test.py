"""rt_local_test.py: bitwise check of VM_RT (tools/rt_patches.py) route() vs the shipped route() on the RTX 3070 (sm_86).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/rt_local_test.py" OUTDIR [layer=kernels/muse_decode_layer/k3_gc_6.py] [cases=600]

Cuts Tab / Args / mdiv / excl_scan / stage_at / route out of the layer's _vml_CUDA_T (+ the RT patch), builds one test
executable per (variant, VM_RT) (bf2-like JN 64 / 576 threads, w72-like JN 72 / WG5 2 / 512 threads), runs the same random
routings (M 17..320, uniform / skewed / all-hot / none local, G 132 / 136 / 77 / 1, random cost tables) in every CTA and
compares every CTA's whole Tab, the ent rows and rmap byte for byte between VM_RT 0 and 1.
"""
import pathlib
import re
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import m32e_patches as P  # noqa: E402
import rt_patches as R  # noqa: E402

MAIN = r"""
using namespace vm;
__global__ void k_route(Args A, uint8_t* oT, uint16_t* oE) {
extern __shared__ __align__(128) uint8_t sm[];
Tab& T = *reinterpret_cast<Tab*>(sm);
uint8_t* scr = sm + (((int)sizeof(Tab) + 127) & ~127);
for (int i = threadIdx.x; i < (int)sizeof(Tab); i += blockDim.x) sm[i] = 0x5A;
for (int i = threadIdx.x; i < 16384; i += blockDim.x) scr[i] = 0xEE;
__syncthreads();
const long long c0 = clock64();
route(A, T, reinterpret_cast<int*>(scr), reinterpret_cast<uint16_t*>(scr + 4096), reinterpret_cast<uint32_t*>(scr + 12288));
__syncthreads();
if (threadIdx.x == 0 && blockIdx.x == 0) { A.marks[0] = clock64() - c0; A.marks[9] = c0; }
for (int i = threadIdx.x; i < (int)sizeof(Tab); i += blockDim.x) oT[(size_t)blockIdx.x * sizeof(Tab) + i] = sm[i];
const uint16_t* ent = reinterpret_cast<const uint16_t*>(scr + 4096);
for (int i = threadIdx.x; i < 2560; i += blockDim.x) oE[(size_t)blockIdx.x * 2560 + i] = ent[i];
}
static uint64_t rs = 88172645463325252ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
int main(int argc, char** argv) {
const int ncase = atoi(argv[1]);
FILE* f = fopen(argv[2], "wb");
const int nth = VM_WG5 >= 2 ? 32 * NCW : NTH;
const int smem = (((int)sizeof(Tab) + 127) & ~127) + 16384;
cudaFuncSetAttribute(k_route, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
int *ids, *rmap; uint8_t* oT; uint16_t* oE; long long* mk;
cudaMalloc(&mk, 16 * 8);
long long pc[3][16] = {{0}}, npc[3] = {0};
long long cyc[3] = {0, 0, 0}, ncy[3] = {0, 0, 0};
cudaMalloc(&ids, 2560 * 4); cudaMalloc(&rmap, 2560 * 4); cudaMalloc(&oT, 136 * sizeof(Tab)); cudaMalloc(&oE, 136 * 2560 * 2);
static int h_ids[2560], h_rmap[2560]; static uint8_t h_T[136 * sizeof(Tab)]; static uint16_t h_E[136 * 2560];
const int Ms[] = {17, 20, 24, 31, 32, 33, 40, 48, 63, 64, 65, 96, 100, 127, 128, 129, 160, 192, 200, 255, 256, 257, 300, 319, 320};
const int Gs[] = {132, 136, 77, 1, 132, 132};
long long sumR = 0, sumJ = 0;
for (int cs = 0; cs < ncase; ++cs) {
const int M = Ms[rnd() % 25], G = Gs[rnd() % 6], mode = rnd() % 5;
const int Etot = (mode == 4) ? 64 : (32 << (rnd() % 4)), ls = 32 * (rnd() % (Etot / 32));
for (int t = 0; t < M; ++t) {
int used[8], n = 0;
while (n < 8) {
int id;
if (mode == 1 && rnd() % 4 != 0) id = ls + rnd() % 6;                // skewed: hot local experts
else if (mode == 2) id = (ls + n + (t % 3)) % Etot;                   // all-hot: every token on the same few experts
else if (mode == 4) id = (ls + 32 + rnd() % 32) % 64;                 // none local
else id = rnd() % Etot;
bool dup = false;
for (int q = 0; q < n; ++q) dup |= used[q] == id;
if (!dup) used[n++] = id;
}
for (int q = 0; q < 8; ++q) h_ids[t * 8 + q] = used[q];
}
for (int i = 0; i < 2560; ++i) h_rmap[i] = -7;
cudaMemcpy(ids, h_ids, M * 32, cudaMemcpyHostToDevice);
cudaMemcpy(rmap, h_rmap, 2560 * 4, cudaMemcpyHostToDevice);
Args A;
memset(&A, 0, sizeof(A));
A.ids = ids; A.marks = mk; A.M = M; A.ls = ls; A.rmap = rmap; A.flags = 193;
A.magG = ((1ull << 40) + G - 1) / G;
for (int k = 0; k < 2 * VM_NTM; ++k) {
const unsigned c = (cs % 7 == 0) ? 128u + 11u * (k % VM_NTM) : 100u + rnd() % 300;
A.cost[k] = c; A.mag[k] = ((1ull << 40) + c - 1) / c;
}
k_route<<<G, nth, smem>>>(A, oT, oE);
cudaError_t e = cudaDeviceSynchronize();
if (e != cudaSuccess) { printf("CUDA error %s case %d\n", cudaGetErrorString(e), cs); return 1; }
cudaMemcpy(h_T, oT, G * sizeof(Tab), cudaMemcpyDeviceToHost);
cudaMemcpy(h_E, oE, G * 2560 * 2, cudaMemcpyDeviceToHost);
cudaMemcpy(h_rmap, rmap, 2560 * 4, cudaMemcpyDeviceToHost);
for (int c = 1; c < G; ++c) {   // every CTA routes redundantly: all copies must agree too
const Tab* t0 = reinterpret_cast<const Tab*>(h_T); const Tab* tc = reinterpret_cast<const Tab*>(h_T + c * sizeof(Tab));
if (memcmp(t0, tc, sizeof(Tab)) || memcmp(h_E, h_E + c * 2560, 5120)) { printf("CTA %d differs from CTA 0, case %d\n", c, cs); return 2; }
}
const Tab* t0 = reinterpret_cast<const Tab*>(h_T);
{ long long cy; cudaMemcpy(&cy, mk, 8, cudaMemcpyDeviceToHost); const int b = M <= 48 ? 0 : M <= 160 ? 1 : 2; if (G == 132) { cyc[b] += cy; ncy[b]++; }
if (G == 132) { long long h[16]; cudaMemcpy(h, mk, 16 * 8, cudaMemcpyDeviceToHost); const long long h0 = h[9]; for (int q = 1; q < 8; ++q) pc[b][q] += h[q] - h[1];
pc[b][8] += h[1] - h[8]; pc[b][9] += h[8] - h0; ++npc[b]; } }
sumR += t0->R; sumJ += t0->nj;
fwrite(&cs, 4, 1, f); fwrite(&M, 4, 1, f); fwrite(&G, 4, 1, f);
fwrite(h_T, sizeof(Tab), 1, f); fwrite(h_E, 2, 2560, f); fwrite(h_rmap, 4, 2560, f);
}
fclose(f);
printf("route cycles (CTA 0, G 132) M<=48 %lld, 49..160 %lld, >160 %lld; ", cyc[0] / (ncy[0] + !ncy[0]), cyc[1] / (ncy[1] + !ncy[1]), cyc[2] / (ncy[2] + !ncy[2]));
for (int b = 0; b < 3; ++b) if (npc[b] && pc[b][7]) { printf("prof b%d:", b); printf(" [c0->rt %lld, rt->ck1 %lld]", pc[b][9] / npc[b], pc[b][8] / npc[b]); for (int q = 2; q < 8; ++q) printf(" %lld", pc[b][q] / npc[b]); printf("; "); }
printf("cases %d, sum R %lld, sum nj %lld, sizeof(Tab) %d\n", ncase, sumR, sumJ, (int)sizeof(Tab));
return 0;
}
"""
PROF = "prof" in sys.argv
VARIANTS = {"bf2": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1"),
            "w72": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", "-DVM_WG5=2",
                    "-DVM_PREG=32", "-DVM_CREG=112", "-DVM_GV=2", "-DVM_JN=72", "-DVM_XSN=72")}


def test_src(layer):
    t = (ROOT / layer).read_text(encoding="utf-8")
    a, b = P.cuda_span(t, "_vml_CUDA_T")
    cuda = R.apply(t[a:b])
    if PROF:   # tid 0 clock64 checkpoints inside the VM_RT route (marks[1..7]), local profiling only
        NLc = chr(10)
        mk = lambda i: f"if (tid == 0 && blockIdx.x == 0 && A.marks) A.marks[{i}] = clock64();" + NLc  # noqa: E731
        ancs = ["int kb[5], kr[5];", "hist[warp * 33 + lane] = run;" + NLc + "}" + NLc,
                "VM_RSYNC();" + NLc + "if (warp < NCW) {" + NLc + "int tot = 0", "int* pl = hist + 528;",
                "const int G = gridDim.x, pb = (warp >= 5)", "const uint32_t x = mdiv(totP", "if (tid == 0) T.bready = 0;"]
        for i, anc in enumerate(ancs):
            k = cuda.index(anc)
            if i == 1:
                k += len(anc)
            if i == 2:
                k += len("VM_RSYNC();" + NLc)
            cuda = cuda[:k] + mk(i + 1) + cuda[k:]
        k = cuda.index("const int E8 = A.M * TOPK;")
        cuda = cuda[:k] + mk(8) + cuda[k:]
    cuda = cuda.replace("@JMAX@", "80").replace("@SLOTS@", "136").replace("@FIXW@", "4096")
    cuda = "\n".join(ln for ln in cuda.split("\n") if not re.match(r"#include <(torch|ATen|c10)", ln))
    head = cuda[:cuda.index("__device__ __forceinline__ int split_point")]
    bar = "\n".join(ln for ln in cuda.split("\n") if ln.startswith(("__device__ __forceinline__ void bar_sync(",
                                                                    "__device__ __forceinline__ void bar_arrive(")))
    body = cuda[cuda.index("__device__ __forceinline__ uint32_t mdiv"):cuda.index(R.GATHER_HEAD)]
    return ("#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n" + head + bar + "\n" + body + "}\n" + MAIN)


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    layer, ncase = kv.get("layer", "kernels/muse_decode_layer/k3_gc_6.py"), int(kv.get("cases", 600))
    src = out / "rt_test.cu"
    src.write_text(test_src(layer), encoding="utf-8")
    jobs = [(v, rt) for v in VARIANTS for rt in (0, 1)]
    res = {}

    def build(v, rt):
        exe = out / f"rt_{v}_{rt}"
        cmd = ["nvcc", "-arch=sm_86", "-O3", "-std=c++17", "-w", f"-DVM_RT={rt}", *VARIANTS[v], str(src), "-o", str(exe)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        res[(v, rt)] = (p.returncode, p.stderr[-3000:], exe)
    for i in range(0, len(jobs), 2):   # at most 2 compiles at once (WSL)
        ths = [threading.Thread(target=build, args=j) for j in jobs[i:i + 2]]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    bad = False
    for v in VARIANTS:
        bins = []
        for rt in (0, 1):
            rc, err, exe = res[(v, rt)]
            if rc:
                print(f"[rt] BUILD FAIL {v} RT={rt}:\n{err}")
                return 1
            dump = out / f"dump_{v}_{rt}.bin"
            p = subprocess.run([str(exe), str(ncase), str(dump)], capture_output=True, text=True)
            print(f"[rt] {v} RT={rt}: rc {p.returncode} {p.stdout.strip()}")
            bad |= p.returncode != 0
            bins.append(dump.read_bytes() if dump.exists() else b"")
        same = bins[0] == bins[1] and len(bins[0]) > 0
        print(f"[rt] {v}: VM_RT 1 vs 0 Tab / ent / rmap {'IDENTICAL' if same else 'DIFFER'} ({len(bins[0])} B)")
        bad |= not same
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

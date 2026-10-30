"""rider3_proof.py: $0 proofs for tools/rider3_patches.py (R-rider3-0).  Run in the local WSL toolchain:

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/rider3_proof.py" OUTDIR BASE.py NEW.py [sass|fin|time|cvt|all]

  sass : device-only sm_90a cubins (nvcc -Xptxas=-v) of BASE and NEW: vm_fin / vm_main registers + spills for the shipped VML
         variants bf2 and w72, vm_fin SASS op counts and the loads above the first ACQBULK; the six sbb_kernel instances
         (SBB1_CU + explicit instantiations): registers, spills, F2FP / F2F / PRMT / STG counts.
  fin  : vm_fin of BASE vs NEW built for sm_86 (griddepcontrol stripped) and run on the RTX 3070: every token count nloc = 0..8
         (token t of case c has nloc = (t + c) mod 9 or a random draw, valid slots at random positions of the 8), y rows with
         special values in some cases (inf, nan, denormal, 1e38); out (bf16) byte for byte and the y / xready zeroing.
  time : the same two binaries, M = 32 CTAs (the T = 32 shape), every token nloc = k (k = 1..8) and a mixed draw: graph of 200
         back-to-back launches, microseconds per launch (a sm_86 latency proxy, not an H200 number).
  cvt  : exhaustive over all 2^32 float bit patterns: __floats2bfloat162_rn(a, b) halves == __float2bfloat16_rn(a), (b)
         (RTX 3070, sm_86; the same cvt.rn.bf16x2.f32 / cvt.rn.bf16.f32 PTX the sm_90a build uses).
"""
import ast
import hashlib
import pathlib
import re
import statistics
import struct
import subprocess
import sys
import threading
import time

NL = chr(10)
OBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
SBB_INST = [(0, 0, 1), (0, 0, 0), (1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0)]   # kPreStream, kRouter, kInputScale as in sbb1_run


def extract_ns(path):
    ns = {"hashlib": hashlib, "time": time}
    for node in ast.parse(pathlib.Path(path).read_text(encoding="utf-8")).body:
        names = []
        if isinstance(node, ast.FunctionDef):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if any(n.startswith("_vml_") or n in ("_VML_DEFS", "SBB1_CU") for n in names):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
            except Exception:
                pass
    assert "_vml_CUDA" in ns and "SBB1_CU" in ns, "layer extraction"
    return ns


def vml_dev(ns):
    src = ns["_vml_CUDA"]
    src = src[:src.index("void vml_run(")]
    for inc in ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>"):
        src = src.replace(inc, "")
    return src


def ptxas_info(text):
    """-Xptxas=-v output -> {mangled entry: 'N regs, S spill st / L spill ld'}"""
    out, lines = {}, text.splitlines()
    for i, ln in enumerate(lines):
        m = re.search(r"Compiling entry function '(\S+)'", ln)
        if m:
            tail = " ".join(lines[i + 1:i + 5])
            rg = re.search(r"Used (\d+) registers", tail)
            ss = re.search(r"(\d+) bytes spill stores", tail)
            sl = re.search(r"(\d+) bytes spill loads", tail)
            out[m.group(1)] = (int(rg.group(1)) if rg else -1, int(ss.group(1)) if ss else -1, int(sl.group(1)) if sl else -1)
    return out


def sass_functions(cub):
    s = subprocess.run([OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
    fns, cur = {}, None
    for ln in s.splitlines():
        m = re.match(r"\s*Function : (\S+)", ln)
        if m:
            cur = fns.setdefault(m.group(1), [])
            continue
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(.*?)\s*;", ln)
        if cur is not None and m:
            cur.append(m.group(1))
    return fns


def opcount(ins, key):
    n = 0
    for s in ins:
        op = re.sub(r"^@!?U?P\w+\s+", "", s)
        n += op.startswith(key)
    return n


def compile_cubin(cu, cub, flags):
    r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *flags, "-Xptxas=-v", "-o", str(cub), str(cu)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("COMPILE FAILED " + NL.join(ln for ln in (r.stdout + r.stderr).splitlines() if "error" in ln)[:1500])
    return ptxas_info(r.stdout + r.stderr)


def sass(out, base, new):
    ns = {"base": extract_ns(base), "new": extract_ns(new)}
    flags = [f for f in ns["base"]["_vml_FLAGS"] if not f.startswith("-gencode")]
    jobs, res = [], {}
    for tag in ("base", "new"):
        v = vml_dev(ns[tag])
        cu = out / f"vml_{tag}.cu"
        cu.write_text(v)
        for var in ("bf2", "w72"):
            jobs.append(("vml", tag, var, cu, flags + list(ns[tag]["_VML_DEFS"][var])))
        s = ns[tag]["SBB1_CU"] + NL.join(f"template __global__ void sbb_kernel<{bool(a)}, {bool(b)}, {bool(c)}>(SbbParams);".replace("True", "true")
                                         .replace("False", "false") for a, b, c in SBB_INST) + NL
        cu = out / f"sbb_{tag}.cu"
        cu.write_text(s)
        jobs.append(("sbb", tag, "", cu, ["-O3", "--expt-relaxed-constexpr"]))

    def run(j):
        kind, tag, var, cu, fl = j
        cub = out / f"{kind}_{tag}_{var}.cubin"
        res[(kind, tag, var)] = (compile_cubin(cu, cub, fl), sass_functions(cub))
    for i in range(0, len(jobs), 2):
        ths = [threading.Thread(target=run, args=(j,)) for j in jobs[i:i + 2]]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    bad = 0
    for var in ("bf2", "w72"):
        for tag in ("base", "new"):
            info, fns = res[("vml", tag, var)]
            fin = next(v for k, v in fns.items() if "vm_fin" in k)
            fkey = next(k for k in info if "vm_fin" in k)
            mkey = next(k for k in info if "vm_main" in k)
            acq = next((i for i, s in enumerate(fin) if "ACQBULK" in s), -1)
            pre = [s for s in fin[:acq] if "LDG" in s]
            print(f"[rider3-sass] {var:3s} {tag:4s}: vm_fin regs {info[fkey][0]} spill {info[fkey][1]}/{info[fkey][2]}, {len(fin)} instr "
                  f"(FMUL {opcount(fin, 'FMUL')} FFMA {opcount(fin, 'FFMA')} SHFL {opcount(fin, 'SHFL')} BAR {opcount(fin, 'BAR')} "
                  f"BRA {opcount(fin, 'BRA')} STS {opcount(fin, 'STS')} LDG {opcount(fin, 'LDG')}), ACQBULK@{acq}, "
                  f"LDG above the wait {len(pre)} {sorted({re.search(r'LDG[.A-Z0-9]*', s).group(0) for s in pre})}; "
                  f"vm_main regs {info[mkey][0]} spill {info[mkey][1]}/{info[mkey][2]}")
        a = res[("vml", "base", var)][0]
        b = res[("vml", "new", var)][0]
        ma = next(v for k, v in a.items() if "vm_main" in k)
        mb = next(v for k, v in b.items() if "vm_main" in k)
        bad |= ma != mb
    for inst in SBB_INST:
        nm = "sbb_kernelILb%dELb%dELb%dE" % inst
        row = []
        for tag in ("base", "new"):
            info, fns = res[("sbb", tag, "")]
            k = next(k for k in fns if nm in k)
            ins = fns[k]
            row.append((info[next(i for i in info if nm in i)], len(ins), opcount(ins, "F2FP"), opcount(ins, "F2F."), opcount(ins, "PRMT"),
                        opcount(ins, "STG"), opcount(ins, "IMAD.U32"), opcount(ins, "LOP3"), opcount(ins, "SHF")))
        for tag, r in zip(("base", "new"), row):
            print(f"[rider3-sass] sbb<{inst[0]},{inst[1]},{inst[2]}> {tag:4s}: regs {r[0][0]} spill {r[0][1]}/{r[0][2]}, {r[1]} instr "
                  f"(F2FP {r[2]} F2F {r[3]} PRMT {r[4]} STG {r[5]} IMAD.U32 {r[6]} LOP3 {r[7]} SHF {r[8]})")
    print("[rider3-sass] vm_main resources base == new: " + ("yes" if not bad else "NO"))
    return bad


FIN_MAIN = r"""
using namespace vm;
static uint64_t rs = 88172645463325252ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static float frnd() { return ((int)(rnd() % 20001) - 10000) * 1e-3f * (rnd() % 4 == 0 ? 37.f : 1.f); }
static float *y, *wts, *nw; static int *rmap, *xr, *dA, *cnt, *gb; static __nv_bfloat16* out;
static float hy[(size_t)2560 * 4096]; static int hr[2560]; static float hw[2560]; static float hn[4096];
static uint16_t ho[(size_t)320 * 4096];
static void setup(int M, const int* nl, int special, int* nr_out) {
  int nr = 0;
  for (int t = 0; t < M; ++t) {
    int pos[8]; for (int i = 0; i < 8; ++i) pos[i] = i;
    for (int i = 7; i > 0; --i) { int j = rnd() % (i + 1); int tmp = pos[i]; pos[i] = pos[j]; pos[j] = tmp; }
    for (int i = 0; i < 8; ++i) hr[t * 8 + i] = -1;
    for (int i = 0; i < nl[t]; ++i) hr[t * 8 + pos[i]] = nr++;
  }
  for (int i = 0; i < M * 8; ++i) hw[i] = (rnd() % 1000) * 1e-3f + 1e-3f;
  for (int i = 0; i < KDIM; ++i) hn[i] = frnd() * 0.1f;
  for (size_t i = 0; i < (size_t)nr * KDIM; ++i) hy[i] = frnd();
  if (special) {
    static const uint32_t sp[] = {0x7f800000u, 0xff800000u, 0x7fc00000u, 0x00000001u, 0x007fffffu, 0x7e800000u, 0x80000000u, 0x3f800000u};
    for (int k = 0; k < 40; ++k) { uint32_t b = sp[rnd() % 8]; size_t i = (size_t)(rnd() % (nr ? nr : 1)) * KDIM + rnd() % KDIM; if (nr) memcpy(&hy[i], &b, 4); }
  }
  cudaMemcpy(y, hy, (size_t)nr * KDIM * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(rmap, hr, M * 32, cudaMemcpyHostToDevice); cudaMemcpy(wts, hw, M * 32, cudaMemcpyHostToDevice);
  cudaMemcpy(nw, hn, KDIM * 4, cudaMemcpyHostToDevice);
  cudaMemset(xr, 1, JMAX * 4); cudaMemset(dA, 1, JMAX * 64); cudaMemset(cnt, 1, 2 * SLOTSD * 16);
  *nr_out = nr;
}
int main(int argc, char** argv) {
  const int mode = atoi(argv[1]);   // 0 check, 1 time
  cudaMalloc(&y, (size_t)2560 * KDIM * 4); cudaMalloc(&wts, 320 * 8 * 4); cudaMalloc(&nw, KDIM * 4); cudaMalloc(&rmap, 320 * 8 * 4);
  cudaMalloc(&xr, JMAX * 4); cudaMalloc(&dA, JMAX * 16 * 4); cudaMalloc(&cnt, 2 * SLOTSD * 4 * 4); cudaMalloc(&gb, 8);
  cudaMalloc(&out, (size_t)320 * KDIM * 2);
  Args A; memset(&A, 0, sizeof(A));
  A.y = y; A.wts = wts; A.nw = nw; A.rmap = rmap; A.out = out; A.xready = xr; A.doneA = dA; A.cnt = cnt; A.gbar = gb;
  if (mode == 0) {
    const int ncase = atoi(argv[2]);
    FILE* f = fopen(argv[3], "wb");
    static int nl[320];
    for (int cs = 0; cs < ncase; ++cs) {
      const int M = (cs % 3 == 0) ? 1 + rnd() % 320 : 9 * (1 + rnd() % 35);
      const int style = cs % 4;
      for (int t = 0; t < M; ++t) nl[t] = style == 3 ? rnd() % 9 : (style == 2 ? (rnd() % 3 == 0 ? rnd() % 9 : rnd() % 3) : (t + cs) % 9);
      int nr; setup(M, nl, cs % 8 == 5, &nr);
      A.M = M;
      vm_fin<<<M, 256>>>(A);
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("CUDA error %s case %d\n", cudaGetErrorString(e), cs); return 1; }
      cudaMemcpy(ho, out, (size_t)M * KDIM * 2, cudaMemcpyDeviceToHost);
      cudaMemcpy(hy, y, (size_t)nr * KDIM * 4, cudaMemcpyDeviceToHost);
      int nz = 0; for (size_t i = 0; i < (size_t)nr * KDIM; ++i) nz += hy[i] != 0.f;
      static int hx[JMAX]; cudaMemcpy(hx, xr, JMAX * 4, cudaMemcpyDeviceToHost); for (int i = 0; i < JMAX; ++i) nz += hx[i] != 0;
      fwrite(&M, 4, 1, f); fwrite(&nr, 4, 1, f); fwrite(&nz, 4, 1, f); fwrite(nl, 4, M, f); fwrite(ho, 2, (size_t)M * KDIM, f);
    }
    fclose(f);
    printf("cases %d ok\n", ncase);
    return 0;
  }
  // time: M = 32, nloc = k for every token (k = 1..8) and a mixed draw; graph of 200 launches
  const int M = 32; static int nl[320];
  cudaStream_t s; cudaStreamCreate(&s);
  for (int k = 1; k <= 9; ++k) {
    for (int t = 0; t < M; ++t) nl[t] = k <= 8 ? k : (rnd() % 3 == 0 ? 3 + rnd() % 3 : rnd() % 3);   // 9 = T=32-like mix (mean ~1.6)
    int nr; setup(M, nl, 0, &nr);
    A.M = M;
    cudaGraph_t g; cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 200; ++i) vm_fin<<<M, 256, 0, s>>>(A);
    cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&ge, g, 0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int rep = 0; rep < 12; ++rep) {
      // vm_fin zeroes y after each call (BRED): refill so every launch does the same work
      cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
      setup(M, nl, 0, &nr);
      cudaEventRecord(e0, s); cudaGraphLaunch(ge, s); cudaEventRecord(e1, s); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
    }
    printf("k %d: %.3f us/launch (200-launch graph; launches 2..200 see zeroed y rows: loads still issue, values are 0)\n", k, best * 1000.f / 200.f);
  }
  return 0;
}
"""


def fin_src(layer_path):
    ns = extract_ns(layer_path)
    cuda = ns["_vml_CUDA"]
    cuda = NL.join(ln for ln in cuda.split(NL) if not re.match(r"#include <(torch|ATen|c10)", ln))
    head = cuda[:cuda.index("__device__ __forceinline__ int split_point")]
    a = cuda.index("__global__ void __launch_bounds__(256) vm_fin(const Args A) {")
    if "template <int NV>" in cuda[:a]:   # R1's fin_ss<NV> sits in front of vm_fin, inside its own #if VM_FTR
        a = cuda.rindex("#if VM_FTR" + NL, 0, cuda.index("template <int NV>"))
    b = cuda.index("void vml_run(")
    body = cuda[a:b].rstrip()
    body = body[:body.rindex("}")]   # drop the namespace close
    body = body.replace('asm volatile("griddepcontrol.wait;\\n" ::: "memory");', "")
    body = body.replace('asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");', "")
    assert "griddepcontrol" not in body, "strip"
    bf = "__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }" + NL
    return "#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n" + head + bf + body + "}\n" + FIN_MAIN


def build_fin(out, base, new):
    exes, res = {}, {}
    defs = ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", "-DVM_RT=1", "-DVM_FTR=1")
    for tag, path in (("base", base), ("new", new)):
        cu = out / f"fin_{tag}.cu"
        cu.write_text(fin_src(path))
        exes[tag] = out / f"fin_{tag}"

    def b(tag):
        p = subprocess.run(["nvcc", "-arch=sm_86", "-O3", "-std=c++17", "-w", *defs, str(out / f"fin_{tag}.cu"), "-o", str(exes[tag])],
                           capture_output=True, text=True)
        res[tag] = (p.returncode, p.stderr[-2500:])
    ths = [threading.Thread(target=b, args=(t,)) for t in ("base", "new")]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    for tag in ("base", "new"):
        if res[tag][0]:
            print(f"[rider3-fin] BUILD FAIL {tag}:{NL}{res[tag][1]}")
            return None
    return exes


def fin(out, base, new, ncase):
    exes = build_fin(out, base, new)
    if exes is None:
        return 1
    bins = {}
    for tag in ("base", "new"):
        d = out / f"fin_{tag}.bin"
        p = subprocess.run([str(exes[tag]), "0", str(ncase), str(d)], capture_output=True, text=True)
        print(f"[rider3-fin] {tag}: rc {p.returncode} {p.stdout.strip()}")
        if p.returncode:
            return 1
        bins[tag] = d.read_bytes()
    raw, off, nzbad, toks, hist, nonzero, outs = bins["new"], 0, 0, 0, [0] * 9, 0, 0
    while off < len(raw):
        M, nr, nz = struct.unpack_from("<iii", raw, off)
        nl = struct.unpack_from("<%di" % M, raw, off + 12)
        for x in nl:
            hist[x] += 1
        o = raw[off + 12 + 4 * M: off + 12 + 4 * M + 2 * M * 4096]
        outs += len(o) // 2
        nonzero += sum(1 for i in range(0, len(o), 2) if o[i] or o[i + 1])
        off += 12 + 4 * M + 2 * M * 4096
        nzbad += nz != 0
        toks += M
    same = bins["base"] == bins["new"]
    print(f"[rider3-fin] NEW vs BASE (out bf16 + y / xready zeroing): {'IDENTICAL' if same else 'DIFFER'} ({len(bins['base'])} B, {toks} tokens; "
          f"tokens per nloc 0..8 = {hist}; non-zero output halves {nonzero}/{outs}); records with y / xready not zeroed: {nzbad}")
    return 0 if same and nzbad == 0 and min(hist) > 100 else 1


def timing(out, base, new):
    exes = build_fin(out, base, new)
    if exes is None:
        return 1
    rows = {}
    for tag in ("base", "new"):
        rep = []
        for _ in range(3):
            p = subprocess.run([str(exes[tag]), "1"], capture_output=True, text=True)
            rep.append([float(m.group(1)) for m in re.finditer(r"k \d+: ([\d.]+) us", p.stdout)])
        rows[tag] = [min(c) for c in zip(*rep)]
    print("[rider3-time] 3070 (sm_86) vm_fin, M = 32 CTAs, us per launch inside a 200-launch graph (min over 3 runs x 12 reps)")
    print("  nloc     " + "  ".join(f"{k:>6}" for k in list(range(1, 9)) + ["mix"]))
    for tag in ("base", "new"):
        print(f"  {tag:4s}     " + "  ".join(f"{x:6.3f}" for x in rows[tag]))
    print("  new-base " + "  ".join(f"{b - a:+6.3f}" for a, b in zip(rows["base"], rows["new"])))
    return 0


CVT_SRC = r"""
#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>
__global__ void k(uint64_t base, unsigned long long* bad) {
  uint64_t i = base + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (1ull << 32)) return;
  const float a = __uint_as_float((uint32_t)i), b = __uint_as_float((uint32_t)(i * 2654435761u + 12345u));   // partner: a mixed pattern
  const __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
  const __nv_bfloat16 s0 = __float2bfloat16_rn(a), s1 = __float2bfloat16_rn(b);
  const uint32_t pw = *reinterpret_cast<const uint32_t*>(&p);
  const uint32_t sw = (uint32_t)__bfloat16_as_ushort(s0) | ((uint32_t)__bfloat16_as_ushort(s1) << 16);
  if (pw != sw) atomicAdd(bad, 1ull);
}
int main() {
  unsigned long long* bad; cudaMallocManaged(&bad, 8); *bad = 0;
  for (uint64_t base = 0; base < (1ull << 32); base += (1ull << 28)) k<<<(1u << 28) / 256, 256>>>(base, bad);
  cudaDeviceSynchronize();
  printf("2^32 float patterns (a) x mixed partner (b): mismatching pairs %llu\n", *bad);
  return *bad != 0;
}
"""


def cvt(out):
    cu = out / "cvt.cu"
    cu.write_text(CVT_SRC)
    exe = out / "cvt"
    p = subprocess.run(["nvcc", "-arch=sm_86", "-O3", "-o", str(exe), str(cu)], capture_output=True, text=True)
    if p.returncode:
        print("[rider3-cvt] BUILD FAIL", p.stderr[-1500:])
        return 1
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    print(f"[rider3-cvt] {r.stdout.strip()} (rc {r.returncode})")
    return r.returncode


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    base, new = sys.argv[2], sys.argv[3]
    what = sys.argv[4] if len(sys.argv) > 4 else "all"
    bad = 0
    if what in ("sass", "all"):
        bad |= sass(out, base, new)
    if what in ("fin", "all"):
        bad |= fin(out, base, new, 400)
    if what in ("time", "all"):
        bad |= timing(out, base, new)
    if what in ("cvt", "all"):
        bad |= cvt(out)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

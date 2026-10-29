"""vtr_local_test.py: local checks of the VTR riders (tools/vtr_patches.py) on the A/B copy (kernels/lab/k3_vtr_ab.py), in WSL:

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vtr_local_test.py" OUTDIR [sass|fin|route|all] [cases=600]

  sass : sm_90a cubins of bf2 / w72 and their rider variants (2 compiles at once): ptxas regs / spills of vm_main + vm_fin,
         vm_fin SASS op counts (total, FMUL, FFMA, BAR, SHFL, VOTE).
  fin  : vm_fin VM_FTR 0 vs 1 on the RTX 3070 (sm_86; griddepcontrol stripped, VM_BRED 1): random y / rmap (-1 holes) / wts,
         M 1..320; out (bf16) and y after the call (the BRED zeroing) compared byte for byte.
  route: route() VM_TA 0 vs 1 (VM_RT 1, bf2-like JN 64 and w72-like JN 72; tools/rt_local_test.py's harness): ent / rmap
         byte-identical, every CTA's Tab agrees, and the job table invariants (per expert: jobs contiguous from est, sizes
         in 1..JN summing to tot, nj unchanged, padded tiles TA <= base, max job tiles equal).
"""
import ast
import pathlib
import re
import struct
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import m32e_patches as P  # noqa: E402
import rt_local_test as RT  # noqa: E402

LAYER = ROOT / "kernels/lab/k3_vtr_ab.py"
OBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
NL = chr(10)


def layer_cuda():
    t = LAYER.read_text(encoding="utf-8")
    ns = {}
    keep = []
    for n in ast.parse(t).body:
        if isinstance(n, ast.FunctionDef) and n.name in ("_vml_wg_spec", "_vml_wg_spec64"):
            keep.append(n)
        elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in (
                "_vml_CUDA_T", "_vml_CUDA", "_vml_FLAGS", "_vml_JMAX", "_vml_SLOTS", "_vml_FIXW", "_VML_DEFS", "_vml_RMAX", "_vml_KDIM"):
            keep.append(n)
        elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith("_vml_"):
            keep.append(n)
    keep.sort(key=lambda n: n.lineno)
    for n in keep:   # one node at a time: helpers that need the torch / runtime side just fail and are skipped
        try:
            exec(compile(ast.Module(body=[n], type_ignores=[]), "layer", "exec"), ns)
        except Exception:
            pass
    assert "_vml_CUDA" in ns, "layer _vml_CUDA"
    return ns


def sass(out):
    ns = layer_cuda()
    src = ns["_vml_CUDA"]
    src = src[:src.index("void vml_run(")]
    for inc in ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>"):
        src = src.replace(inc, "")
    cu = out / "vml_dev.cu"
    cu.write_text(src)
    flags = [f for f in ns["_vml_FLAGS"] if not f.startswith("-gencode")]
    names = ["bf2", "bf2A", "w72", "w72A"] + [a for a in sys.argv[3:] if a in ns["_VML_DEFS"]]
    res = {}

    def job(nm):
        cub = out / f"{nm}.cubin"
        r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *flags, *ns["_VML_DEFS"][nm], "-Xptxas=-v", "-o", str(cub), str(cu)],
                           capture_output=True, text=True)
        if r.returncode:
            res[nm] = ("FAIL " + NL.join((r.stdout + r.stderr).splitlines()[-12:]), {})
            return
        info = (r.stdout + r.stderr).splitlines()
        regs = {}
        for i, ln in enumerate(info):
            m = re.search(r"Compiling entry function '(\S+)'", ln)
            if m and ("vm_main" in m.group(1) or "vm_fin" in m.group(1)):
                k = "main" if "vm_main" in m.group(1) else "fin"
                tail = " ".join(info[i + 1:i + 4])
                rg = re.search(r"Used (\d+) registers", tail)
                sp = re.search(r"(\d+) bytes spill stores", tail)
                regs[k] = f"{k} regs {rg.group(1) if rg else '?'} spill {sp.group(1) if sp else '?'}"
        s = subprocess.run([OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
        cnt, cur = {}, None
        for ln in s.splitlines():
            m = re.match(r"\s*Function : (\S+)", ln)
            if m:
                cur = m.group(1)
                continue
            m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)", ln)
            if cur and "vm_fin" in cur and m:
                op = m.group(1).split(".")[0]
                cnt[op] = cnt.get(op, 0) + 1
                cnt["total"] = cnt.get("total", 0) + 1
        res[nm] = ("ok " + "; ".join(regs.values()), cnt)

    for i in range(0, len(names), 2):   # at most 2 compiles at once (WSL)
        ths = [threading.Thread(target=job, args=(nm,)) for nm in names[i:i + 2]]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    bad = 0
    for nm in names:
        st, c = res[nm]
        bad |= not st.startswith("ok")
        print(f"[vtr-sass] {nm}: {st}; vm_fin " + " ".join(f"{k} {c.get(k, 0)}" for k in ("total", "FMUL", "FFMA", "BAR", "SHFL", "VOTE", "LDG", "STG")))
    return bad


FIN_MAIN = r"""
using namespace vm;
static uint64_t rs = 88172645463325252ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static float frnd() { return ((int)(rnd() % 20001) - 10000) * 1e-3f * (rnd() % 4 == 0 ? 37.f : 1.f); }
int main(int argc, char** argv) {
const int ncase = atoi(argv[1]);
FILE* f = fopen(argv[2], "wb");
const int RMAXT = 2560;
float *y, *wts, *nw; int *rmap, *xr, *dA, *cnt, *gb; __nv_bfloat16* out;
cudaMalloc(&y, (size_t)RMAXT * KDIM * 4); cudaMalloc(&wts, 320 * 8 * 4); cudaMalloc(&nw, KDIM * 4); cudaMalloc(&rmap, 320 * 8 * 4);
cudaMalloc(&xr, JMAX * 4); cudaMalloc(&dA, JMAX * 16 * 4); cudaMalloc(&cnt, 2 * SLOTSD * 4 * 4); cudaMalloc(&gb, 8);
cudaMalloc(&out, (size_t)320 * KDIM * 2);
static float hy[(size_t)2560 * 4096]; static int hr[2560]; static float hw[2560]; static float hn[4096];
static uint16_t ho[(size_t)320 * 4096];
for (int cs = 0; cs < ncase; ++cs) {
const int M = 1 + rnd() % 320, pl = rnd() % 4;
int nr = 0;
for (int i = 0; i < M * 8; ++i) hr[i] = (rnd() % 4 < pl + 1 && pl < 3) || pl == 3 ? nr++ : -1;
for (int i = 0; i < M * 8; ++i) hw[i] = (rnd() % 1000) * 1e-3f;
for (int i = 0; i < KDIM; ++i) hn[i] = frnd() * 0.1f;
for (size_t i = 0; i < (size_t)nr * KDIM; ++i) hy[i] = frnd();
cudaMemcpy(y, hy, (size_t)nr * KDIM * 4, cudaMemcpyHostToDevice);
cudaMemcpy(rmap, hr, M * 32, cudaMemcpyHostToDevice); cudaMemcpy(wts, hw, M * 32, cudaMemcpyHostToDevice);
cudaMemcpy(nw, hn, KDIM * 4, cudaMemcpyHostToDevice);
cudaMemset(xr, 1, JMAX * 4); cudaMemset(dA, 1, JMAX * 64); cudaMemset(cnt, 1, 2 * SLOTSD * 16);
Args A; memset(&A, 0, sizeof(A));
A.y = y; A.wts = wts; A.nw = nw; A.rmap = rmap; A.out = out; A.xready = xr; A.doneA = dA; A.cnt = cnt; A.gbar = gb; A.M = M;
vm_fin<<<M, 256>>>(A);
cudaError_t e = cudaDeviceSynchronize();
if (e != cudaSuccess) { printf("CUDA error %s case %d\n", cudaGetErrorString(e), cs); return 1; }
cudaMemcpy(ho, out, (size_t)M * KDIM * 2, cudaMemcpyDeviceToHost);
cudaMemcpy(hy, y, (size_t)nr * KDIM * 4, cudaMemcpyDeviceToHost);
int nz = 0; for (size_t i = 0; i < (size_t)nr * KDIM; ++i) nz += hy[i] != 0.f;
static int hx[JMAX]; cudaMemcpy(hx, xr, JMAX * 4, cudaMemcpyDeviceToHost); for (int i = 0; i < JMAX; ++i) nz += hx[i] != 0;
fwrite(&M, 4, 1, f); fwrite(&nr, 4, 1, f); fwrite(&nz, 4, 1, f); fwrite(ho, 2, (size_t)M * KDIM, f);
}
fclose(f);
printf("cases %d ok\n", ncase);
return 0;
}
"""


def fin(out, ncase):
    ns = layer_cuda()
    cuda = ns["_vml_CUDA"]
    cuda = NL.join(ln for ln in cuda.split(NL) if not re.match(r"#include <(torch|ATen|c10)", ln))
    head = cuda[:cuda.index("__device__ __forceinline__ int split_point")]
    a = cuda.index("__global__ void __launch_bounds__(256) vm_fin(const Args A) {")
    b = cuda.index("void vml_run(")
    body = cuda[a:b].rstrip()
    check = body.endswith("}\n}") or body.endswith("}" + NL + "}")
    body = body[:body.rindex("}")]   # drop the namespace close
    body = body.replace('asm volatile("griddepcontrol.wait;\\n" ::: "memory");', "")
    body = body.replace('asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");', "")
    assert "griddepcontrol" not in body and check, "strip"
    bf = "__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }" + NL
    src = "#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n" + head + bf + body + "}\n" + FIN_MAIN
    cu = out / "fin_test.cu"
    cu.write_text(src)
    base = ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", "-DVM_RT=1")
    res = {}

    def build(r):
        exe = out / f"fin_{r}"
        p = subprocess.run(["nvcc", "-arch=sm_86", "-O3", "-std=c++17", "-w", *base, f"-DVM_FTR={r}", str(cu), "-o", str(exe)],
                           capture_output=True, text=True)
        res[r] = (p.returncode, p.stderr[-2500:], exe)
    ths = [threading.Thread(target=build, args=(r,)) for r in (0, 1)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    bins = []
    for r in (0, 1):
        rc, err, exe = res[r]
        if rc:
            print(f"[vtr-fin] BUILD FAIL FTR={r}:{NL}{err}")
            return 1
        d = out / f"fin_{r}.bin"
        p = subprocess.run([str(exe), str(ncase), str(d)], capture_output=True, text=True)
        print(f"[vtr-fin] FTR={r}: rc {p.returncode} {p.stdout.strip()}")
        if p.returncode:
            return 1
        bins.append(d.read_bytes())
    # every record's nz (nonzero y after the call + xready) must be 0
    raw, off, nzbad, rows = bins[0], 0, 0, 0
    while off < len(raw):
        M, nr, nz = struct.unpack_from("<iii", raw, off)
        off += 12 + 2 * M * 4096
        nzbad += nz != 0
        rows += nr
    same = bins[0] == bins[1]
    print(f"[vtr-fin] FTR 1 vs 0 out + zeroing: {'IDENTICAL' if same else 'DIFFER'} ({len(bins[0])} B, {rows} y rows); "
          f"records with y / xready not zeroed: {nzbad}")
    return 0 if same and nzbad == 0 else 1


def route(out, ncase):
    t = LAYER.read_text(encoding="utf-8")
    a, b = P.cuda_span(t, "_vml_CUDA_T")
    cuda = t[a:b].replace("@JMAX@", "80").replace("@SLOTS@", "136").replace("@FIXW@", "4096")
    cuda = NL.join(ln for ln in cuda.split(NL) if not re.match(r"#include <(torch|ATen|c10)", ln))
    head = cuda[:cuda.index("__device__ __forceinline__ int split_point")]
    bar = NL.join(ln for ln in cuda.split(NL) if ln.startswith(("__device__ __forceinline__ void bar_sync(",
                                                                 "__device__ __forceinline__ void bar_arrive(")))
    body = cuda[cuda.index("__device__ __forceinline__ uint32_t mdiv"):cuda.index(RT.R.GATHER_HEAD)]
    src = "#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n" + head + bar + NL + body + "}\n" + RT.MAIN
    cu = out / "rt_test.cu"
    cu.write_text(src)
    jobs = [(v, ta) for v in RT.VARIANTS for ta in (0, 1)]
    res = {}

    def build(v, ta):
        exe = out / f"rt_{v}_{ta}"
        p = subprocess.run(["nvcc", "-arch=sm_86", "-O3", "-std=c++17", "-w", "-DVM_RT=1", f"-DVM_TA={ta}", *RT.VARIANTS[v],
                            str(cu), "-o", str(exe)], capture_output=True, text=True)
        res[(v, ta)] = (p.returncode, p.stderr[-2500:], exe)
    for i in range(0, len(jobs), 2):
        ths = [threading.Thread(target=build, args=j) for j in jobs[i:i + 2]]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    # Tab layout (JMAX 80): nj, R, Rpad, pad0, wlast[4], jE[80], jC[80], jS[81], jrow[81], ...; record: cs, M, G, Tab, ent[2560] u16, rmap[2560]
    sz_tab = 0
    bad = 0
    for v in RT.VARIANTS:
        JN = 72 if "-DVM_JN=72" in RT.VARIANTS[v] else 64
        dumps = []
        for ta in (0, 1):
            rc, err, exe = res[(v, ta)]
            if rc:
                print(f"[vtr-route] BUILD FAIL {v} TA={ta}:{NL}{err}")
                return 1
            d = out / f"rt_{v}_{ta}.bin"
            p = subprocess.run([str(exe), str(ncase), str(d)], capture_output=True, text=True)
            print(f"[vtr-route] {v} TA={ta}: rc {p.returncode} {p.stdout.strip()[-150:]}")
            if p.returncode:
                return 1
            sz_tab = int(re.search(r"sizeof\(Tab\) (\d+)", p.stdout).group(1))
            dumps.append(d.read_bytes())
        rec = 12 + sz_tab + 2 * 2560 + 4 * 2560
        n = len(dumps[0]) // rec
        stats = {"same_ent_rmap": 0, "inv_fail": 0, "tiles0": 0, "tiles1": 0, "changed": 0, "R0": 0, "R1": 0}
        for i in range(n):
            r0, r1 = (dd[i * rec:(i + 1) * rec] for dd in dumps)
            tail = 12 + sz_tab
            stats["same_ent_rmap"] += r0[tail:] == r1[tail:] and r0[:12] == r1[:12]
            tabs = []
            for r in (r0, r1):
                tb = r[12:12 + sz_tab]
                nj, R = struct.unpack_from("<ii", tb, 0)
                jE = struct.unpack_from("<80i", tb, 32)
                jC = struct.unpack_from("<80i", tb, 352)
                jS = struct.unpack_from("<81i", tb, 672)
                tabs.append((nj, R, jE[:nj], jC[:nj], jS[:nj]))
            (n0, R0, e0, c0, s0), (n1, R1, e1, c1, s1) = tabs
            ok = n0 == n1
            per = {}
            for (e, c, s) in zip(e1, c1, s1):
                per.setdefault(e, []).append((s, c))
            per0 = {}
            for (e, c, s) in zip(e0, c0, s0):
                per0.setdefault(e, []).append((s, c))
            ok &= set(per) == set(per0)
            for e, js in per.items():
                js0 = per0.get(e, [])
                st = js0[0][0] if js0 else -1
                tot = sum(c for _, c in js0)
                ok &= len(js) == len(js0) and js[0][0] == st and sum(c for _, c in js) == tot
                pos = st
                for s, c in js:
                    ok &= s == pos and 1 <= c <= JN
                    pos += c
                t0 = [(c + 7) >> 3 for _, c in js0]
                t1 = [(c + 7) >> 3 for _, c in js]
                ok &= sum(t1) <= sum(t0) and max(t1) == max(t0)
                stats["tiles0"] += sum(t0)
                stats["tiles1"] += sum(t1)
            stats["changed"] += c0 != c1
            stats["R0"] += R0
            stats["R1"] += R1
            stats["inv_fail"] += not ok
        print(f"[vtr-route] {v} (JN {JN}): {n} cases; ent / rmap identical {stats['same_ent_rmap']}/{n}; job invariants failed "
              f"{stats['inv_fail']}; cases with a changed split {stats['changed']}; padded tiles {stats['tiles0']} -> {stats['tiles1']}; "
              f"sum R {stats['R0']} -> {stats['R1']}")
        bad |= stats["same_ent_rmap"] != n or stats["inv_fail"] != 0
    return bad


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    what = sys.argv[2] if len(sys.argv) > 2 else "all"
    kv = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
    nc = int(kv.get("cases", 600))
    bad = 0
    if what in ("sass", "all"):
        bad |= sass(out)
    if what in ("fin", "all"):
        bad |= fin(out, max(1, nc // 10))
    if what in ("route", "all"):
        bad |= route(out, nc)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""vfaws_patches.py (R-vfaws-1): workspace-release fix for the VFA merge (vfa_main, verify q_len 8, full attention).

Ship merge per segment: all 256 consumer threads red.add their absolute-frame partials into ws_o / ws_l, bar.sync(3), then
ONLY tid 0 runs __threadfence + atomicAdd(ticket); the CTA whose ticket completes the request reads ws (ld.cg), writes the
output and zeroes ws. A MEMBAR orders only the issuing warp's own memory operations, and bar.sync does not wait for other
warps' reds to be performed at L2, so a red from warps 1..7 can land after the last CTA has read and zeroed ws: that call
loses the partial and the next call inherits it. Fix (VFAWS): every consumer thread fences its own reds (fence.acq_rel.gpu)
before bar.sync(3), the classic threadFenceReduction pattern.

Flags in the generated file:
  VFA_WSFIX (Python) 1 = DIAG 0 carries the fix, 0 = ship merge (bit-for-bit the base's VFA source otherwise).
  VFA_WSLAB (Python) 1 = also build lab DIAG 24 = ship + fix (ST 4, 32-bit offsets only; bench only).

  python tools/vfaws_patches.py --base=kernels/muse_decode_layer/k3_gc_21cg_s1.py --out=kernels/lab/k3_vfaws_1.py [--fix=1]
  WSL: tools/local/cudaenv.sh python tools/vfaws_patches.py --sass=<dir> <layer.py>   (ptxas -v + MEMBAR count, ship vs fix)
"""
import ast
import pathlib
import re
import subprocess
import sys

NL = chr(10)
CUOBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"


def one(t, old, new, what):
    n = t.count(old)
    if n != 1:
        raise SystemExit(f"vfaws_patches: anchor '{what}' found {n} times (need exactly 1)")
    return t.replace(old, new)


PVT_OLD = "template <int DIAG> constexpr bool pvt_on() { return (DIAG >= 18 && DIAG <= 20) || (DIAG == 0 && true); }   // VFA3-PVT"
PVT_NEW = "template <int DIAG> constexpr bool pvt_on() { return (DIAG >= 18 && DIAG <= 20) || DIAG == 24 || (DIAG == 0 && true); }   // VFA3-PVT (+ VFAWS 24)"

FLAG_OLD = "  constexpr bool PVS = (DIAG == 13 || DIAG == 14) || (DIAG == 0 && false);   // FA16-PVS" + NL
FLAG_NEW = FLAG_OLD + "  constexpr bool WSF = (DIAG == 24) || (DIAG == 0 && VFA_WSFIX_ON);   // VFAWS: every writer fences its reds before the ticket" + NL

MERGE_OLD = '''      red1(p.ws_l + R1, acc[34] * f1);
    }
    bar_named(3, NCONS);
    if (tid == 0) {'''   # ship: tid 0 then __threadfence(); b23: FNC_WF()
MERGE_NEW = '''      red1(p.ws_l + R1, acc[34] * f1);
    }
    if constexpr (WSF) asm volatile("fence.acq_rel.gpu;" ::: "memory");   // VFAWS: this thread's reds performed before the ticket
    bar_named(3, NCONS);
    if (tid == 0) {'''

DISP_OLD = '''static void vfa_diag(const Params& p, dim3 grid, cudaStream_t stream, bool pdl, int diag) {
#if VFA_LAB'''
DISP_NEW = '''static void vfa_diag(const Params& p, dim3 grid, cudaStream_t stream, bool pdl, int diag) {
#if VFA_WSLAB
  if constexpr (O32 && ST == 4) {   // VFAWS lab arm: 24 = ship + per-thread release fence before the ticket
    if (diag == 24) { vfa_go<ST, 24, true>(p, grid, stream, pdl); return; }
  }
  TORCH_CHECK(diag != 24, "vfa: VFAWS lab arm needs stages 4 and 32-bit cache offsets");
#endif
#if VFA_LAB'''

PY_OLD = "VFA_DIAG = 0" + NL
BUILD_OLD = '''    src = VFA_CUDA_SRC.replace("#if VFA_LAB", "#if 1" if lab else "#if 0")'''
BUILD_NEW = '''    src = VFA_CUDA_SRC.replace("#if VFA_LAB", "#if 1" if lab else "#if 0")
    src = src.replace("#if VFA_WSLAB", "#if 1" if VFA_WSLAB else "#if 0").replace("VFA_WSFIX_ON", "true" if VFA_WSFIX else "false")   # VFAWS'''


def apply(t, fix=1):
    """t: layer text carrying the ship VFA block (k3_gc_21cg_s1 / k3_b23_1 family)."""
    t = one(t, PVT_OLD, PVT_NEW, "pvt_on")
    t = one(t, FLAG_OLD, FLAG_NEW, "PVS flag")
    t = one(t, MERGE_OLD, MERGE_NEW, "merge ticket")
    t = one(t, DISP_OLD, DISP_NEW, "vfa_diag dispatch")
    t = one(t, PY_OLD, PY_OLD + f"VFA_WSFIX = {int(fix)}   # VFAWS: 1 = every merge writer fences its reds before the ticket (DIAG 0)" + NL
            + "VFA_WSLAB = 0   # VFAWS: 1 = also build lab DIAG 24 (bench only)" + NL, "VFA_DIAG constant")
    t = one(t, BUILD_OLD, BUILD_NEW, "_vfa_build source replace")
    return t


def sass(layer, od):
    od = pathlib.Path(od)
    od.mkdir(parents=True, exist_ok=True)
    tree = ast.parse(pathlib.Path(layer).read_text(encoding="utf-8"))
    s = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "VFA_CUDA_SRC")
    cut = s.index("template <int ST, int DIAG, bool O32>" + NL + "static void vfa_go(")
    dev = re.sub(r"#include <(torch|ATen|c10)[^>]*>" + NL, "", s[:cut]) + "}  // namespace vfa" + NL
    rc = 0
    for name, on, diags in (("ship", "false", (0,)), ("fix", "true", (0, 24))):
        src = dev.replace("VFA_WSFIX_ON", on) + "".join(
            f"template __global__ void vfa::vfa_main<4, {d}, true>(const __grid_constant__ vfa::Params);" + NL for d in diags)
        cu, cub = od / f"vfa_{name}.cu", od / f"vfa_{name}.cubin"
        cu.write_text(src)
        r = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-O3", "--expt-relaxed-constexpr", "-Xptxas=-v", "-o", str(cub), str(cu)],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"[{name}] COMPILE FAILED" + NL + r.stderr[-3000:])
            return 2
        for ln in r.stderr.splitlines():
            if "warning" in ln.lower() or "C75" in ln:
                print(f"[{name}] {ln.strip()[:220]}")
                rc = 1
            elif "Used" in ln or "spill" in ln:
                print(f"[{name}] {ln.strip()[:160]}")
        out = subprocess.run([CUOBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
        for blk in out.split("Function : ")[1:]:
            tag = re.sub(r".*vfa_mainILi4ELi(\d+)ELb1E.*", r"D\1", blk.split(NL, 1)[0])
            ins = re.findall(r"/\*[0-9a-f]{4,}\*/\s+([^;]*);", blk)
            print(f"[{name} {tag}] SASS {len(ins)} instr; MEMBAR {sum('MEMBAR' in x for x in ins)}: "
                  + ", ".join(sorted(set(x.strip() for x in ins if "MEMBAR" in x))))
    return rc


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if "sass" in args:
        pos = [a for a in sys.argv[1:] if not a.startswith("--")]
        sys.exit(sass(pos[0], args["sass"]))
    base = pathlib.Path(args.get("base", "kernels/muse_decode_layer/k3_gc_21cg_s1.py"))
    out = pathlib.Path(args.get("out", "kernels/lab/k3_vfaws_1.py"))
    t = base.read_text(encoding="utf-8")
    t2 = apply(t, int(args.get("fix", "1")))
    out.write_text(t2, encoding="utf-8", newline=NL)
    print(f"[vfaws] {base} -> {out}: {len(t2) - len(t)} bytes added, VFA_WSFIX = {args.get('fix', '1')}")


if __name__ == "__main__":
    main()

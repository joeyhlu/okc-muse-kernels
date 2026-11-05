"""S7F-XR: K-half x ring with lookahead for the XP-0 GD routes at T = 16 / 32 (no GPU calls).

  python -I tools/s7f_xr_patches.py            # write box_out/s7/k3_xr_v1.py + k3_xr_v2.py, run CPU checks
  python3 -I tools/s7f_xr_patches.py --gate    # (WSL, via tools/local/cudaenv.sh) nvcc sm_90a ptxas gate

Mechanism. The XP-0 GD routes (cat = pooler+router at T 16 / 32, qkv at T 16) load each unit's x (2 uint4 per
token group = the two K-halves of the 64-K unit) at the TOP of the unit, after the previous unit's HMMAs
(asm volatile keeps that order), and consume half 0 at q = 0 / 1 and half 1 at q = 2 / 3 of the same unit.
R-gdxh-0 measured that x latency (L2, ~2.1 us of the T16 -> T32 growth) is exposed every unit; XP bit 1 (a full
second X buffer) costs 8 MT registers. XR replaces X[MT][2] with a ring of XR half-slots XH[XR][MT]: the half
j = 2 s + h lives in slot j % XR and, the moment it has been consumed (after q = 2h + 1), slot j % XR is refilled
with half j + XR. XR = 2 (XP bit 32) is register-neutral and gives half 0 of every unit the extra cover of the
previous unit's q = 2 / 3 HMMAs; XR = 3 (XP bit 64, +4 MT registers) gives EVERY half ~one full unit of lead
(two half-consumptions + decode + weight issue + escape patch) at a third of XP 1's register cost.
Arithmetic, operand values, accumulation order, escapes, weights, epilogue: unchanged (bitwise-identical outputs).
v1 = XR 2 on cat16 / cat32 / qkv16; v2 = XR 3 on cat16 (126 regs) + XR 2 on cat32 / qkv16 (XR 3 spills: 128-reg /
96-reg caps at MB 2 / 5). The gate reports regs / spills per route and the LDG/HMMA issue pattern.
Not on the dead list: R-gdxh-0 split TOKENS (fewer X regs, more exposure); GD-A / ST rings moved WEIGHTS.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROW = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUT = ROOT / "box_out/s7"
LIMIT = 1_000_000


def need(c, msg):
    if not c:
        raise SystemExit(f"[s7f_xr] FAIL: {msg}")


def sub(t, old, new, what, n=1):
    k = t.count(old)
    need(k == n, f"anchor {what!r}: found {k}, expected {n}")
    return t.replace(old, new)


# ---------------------------------------------------------------- CUDA edits (GD_CUDA_SRC)
K_XR = ("constexpr int XB = (XP & 1) ? 2 : 1;\n",
        "constexpr int XB = (XP & 1) ? 2 : 1;\n"
        "constexpr int XR = (XP & 64) ? 3 : (XP & 32) ? 2 : 0;   // S7F-XR: K-half x ring slots (0 = off)\n"
        "static_assert(XR == 0 || (XP & 1) == 0, \"XR: the half ring replaces the XP-1 double buffer\");\n")
K_PRO = ("""uint4 X[XB][MT][2];
#pragma unroll
for (int mt = 0; mt < MT; ++mt) {
X[0][mt][0] = ldxk<XP>(xb + xo[mt]);
X[0][mt][1] = ldxk<XP>(xb + xo[mt] + 1);
}
""", """uint4 X[XB][MT][2];
#pragma unroll
for (int mt = 0; mt < MT; ++mt) {
X[0][mt][0] = ldxk<XP>(xb + xo[mt]);
X[0][mt][1] = ldxk<XP>(xb + xo[mt] + 1);
}
uint4 XH[XR > 0 ? XR : 1][MT];   // S7F-XR: half j = 2 s + h of unit s lives in slot j % XR
if constexpr (XR > 0) {
#pragma unroll
for (int mt = 0; mt < MT; ++mt) {
XH[0][mt] = X[0][mt][0];
XH[1][mt] = X[0][mt][1];
}
if constexpr (XR > 2 && KU > 1) {
#pragma unroll
for (int mt = 0; mt < MT; ++mt) XH[2][mt] = ldxk<XP>(GC_XP(1) + xo[mt]);
}
}
""")
K_TOP = ("""} else if (s > 0) {
#pragma unroll
for (int mt = 0; mt < MT; ++mt) {
X[0][mt][0] = ldxk<XP>(GC_XP(s) + xo[mt]);""", """} else if (XR == 0 && s > 0) {
#pragma unroll
for (int mt = 0; mt < MT; ++mt) {
X[0][mt][0] = ldxk<XP>(GC_XP(s) + xo[mt]);""")
K_MMA = ("""for (int mt = 0; mt < MT; ++mt) {
const uint4 xv = X[xs][mt][q >> 1];
const uint32_t b0 = (q & 1) ? xv.z : xv.x;
const uint32_t b1 = (q & 1) ? xv.w : xv.y;
mma16816(acc[q % NC][mt], a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], b0, b1);
}
}
""", """for (int mt = 0; mt < MT; ++mt) {
const uint4 xv = XR > 0 ? XH[XR > 0 ? (2 * s + (q >> 1)) % (XR > 0 ? XR : 1) : 0][mt] : X[xs][mt][q >> 1];
const uint32_t b0 = (q & 1) ? xv.z : xv.x;
const uint32_t b1 = (q & 1) ? xv.w : xv.y;
mma16816(acc[q % NC][mt], a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], b0, b1);
}
if constexpr (XR > 0) {   // S7F-XR: half (2 s + h) consumed -> refill its slot with half (2 s + h + XR)
if (q & 1) {
const int jc = 2 * s + (q >> 1), jn = jc + XR;
if ((jn >> 1) < KU) {
#pragma unroll
for (int mt = 0; mt < MT; ++mt) XH[jc % (XR > 0 ? XR : 1)][mt] = ldxk<XP>(GC_XP(jn >> 1) + xo[mt] + (jn & 1));
}
}
}
}
""")
# host: the cat key validator accepts the XR bits
H_CAT = (" if wr!=1 or s!=1 or mt not in(2,4)or wk*(ku-GD_CAT_UNITS)!=HIDDEN//64 or st_>ku or xp not in(0,1):\n",
         " if wr!=1 or s!=1 or mt not in(2,4)or wk*(ku-GD_CAT_UNITS)!=HIDDEN//64 or st_>ku or xp not in(0,1,32,64):\n")
R_CAT = ("GD_CAT_ROUTE={16:(2,1,8,18,2,1,0),32:(4,1,8,18,1,1,0)}\n", "GD_CAT_ROUTE={16:(2,1,8,18,2,1,%d),32:(4,1,8,18,1,1,%d)}\n")
R_QKV = ('("qkv",16):(2,1,4,8,1,4,0)', '("qkv",16):(2,1,4,8,1,4,%d)')

VARIANTS = {
    "v1": dict(bits=32, note="S7F-XR v1: GD K-half x ring XR 2 (XP bit 32, register-neutral) on cat16 / cat32 / qkv16"),
    "v2": dict(bits=64, note="S7F-XR v2: GD K-half x ring XR 3 (XP bit 64) on cat16, XR 2 on cat32 / qkv16 (XR 3 spills there)",
               routes={"cat16": 64, "cat32": 32, "qkv16": 32}),
}


def cuda_src(t):
    a = t.index("GD_CUDA_SRC=r'''") + len("GD_CUDA_SRC=r'''")
    return t[a:t.index("'''", a)]


def build(base, v, bits_by_route=None):
    t = base
    lines = t.split("\n")
    need(lines[1].startswith("# nonce "), "line 2 nonce")
    lines[1] = f"# nonce xr-{v} ({VARIANTS[v]['note']}; base k3_tinys7_c1)"
    t = "\n".join(lines)
    src0 = cuda_src(t)
    src = src0
    for old, new in (K_XR, K_PRO, K_TOP, K_MMA):
        src = sub(src, old, new, old[:40])
    t = t.replace(src0, src, 1)
    t = sub(t, *H_CAT, what="cat validator")
    b = bits_by_route or {"cat16": VARIANTS[v]["bits"], "cat32": VARIANTS[v]["bits"], "qkv16": VARIANTS[v]["bits"]}
    t = sub(t, R_CAT[0], R_CAT[1] % (b["cat16"], b["cat32"]), what="GD_CAT_ROUTE")
    t = sub(t, R_QKV[0], R_QKV[1] % b["qkv16"], what="GD_ROUTE qkv16")
    return t


def ring_proof(KU, XR):
    """Symbolic replay of the half ring: every half is loaded once, before its use, never overwritten while live."""
    slot = {}
    for j in range(min(XR, 2 * KU)):   # prologue: halves 0, 1 (and 2 for XR 3, KU > 1)
        slot[j % XR] = j
    loads = set(range(min(XR, 2 * KU)))
    for s in range(KU):
        for q in range(4):
            j = 2 * s + (q >> 1)
            need(slot.get(j % XR) == j, f"KU {KU} XR {XR}: half {j} not in slot at s {s} q {q}")
            if q & 1:
                jn = j + XR
                if (jn >> 1) < KU:
                    need(jn not in loads, f"half {jn} loaded twice")
                    loads.add(jn)
                    slot[j % XR] = jn
    need(loads == set(range(2 * KU)), f"KU {KU} XR {XR}: halves loaded {sorted(loads)}")


def cpu_checks(base, outs):
    for KU in (1, 2, 4, 8, 16, 18):
        for XR in (2, 3):
            ring_proof(KU, XR)
    for v, t in outs.items():
        need(len(t.encode()) < LIMIT, f"{v} size {len(t.encode())}")
        compile(t, f"k3_xr_{v}.py", "exec")
        need("\r" not in t, "CRLF")
        # only the nonce line, the GD CUDA source, the cat validator and two route literals differ
        bl, tl = base.split("\n"), t.split("\n")
        diff = [i + 1 for i, (x, y) in enumerate(zip(bl, tl)) if x != y]
        need(len(bl) <= len(tl), "lines")
        print(f"[s7f_xr] {v}: {len(t.encode())} B, first changed lines {diff[:8]} (+{len(tl) - len(bl)} lines)")


def gd_keys_for(t):
    """The GD instantiation keys of the T 16 / 32 routes this lane touches (+ the untouched base ones for reference)."""
    ns = {}
    exec(compile("GD_SMS=132\nHIDDEN=8192\nD_EXPERT=4096\nN_EXPERTS=256\nGD_CAT_UNITS=2\n", "c", "exec"), ns)
    m = re.search(r"^GD_CAT_ROUTE=(\{.*\})$", t, re.M)
    cat = eval(m[1])
    m = re.search(r"^GD_ROUTE=(\{.*\})$", t, re.M)
    gdr = eval(m[1])
    keys = []
    for b in (16, 32):   # _gd_cat_key: ctas = D_EXPERT // 16, nt = wk * 32
        mt, wr, wk, ku, st_, s, xp = cat[b]
        nt = wk * 32
        keys.append((f"cat{b}", (0, mt, wr, wk, ku, st_, s, max(1, min(2048 // nt, -(-(4096 // 16) // 132))), xp, 2)))
    for proj, n, k in (("qkv", 2304, 8192), ("o", 8192, 1024), ("unpooler", 8192, 4096)):
        mt, wr, wk, ku, st_, s, xp = gdr[(proj, 16)]
        nt = wr * wk * 32
        ctas = (n // 16 // wr) * s
        keys.append((f"{proj}16", (0, mt, wr, wk, ku, st_, s, max(1, min(2048 // nt, -(-ctas // 132))), xp)))
    return keys


def issue_pattern(sass, sym):
    """Run-length LDG (L) / HMMA (H) order of one kernel's SASS: shows where ptxas put each unit's x / weight loads."""
    a = sass.find("Function : " + sym)
    need(a >= 0, f"SASS for {sym} not found")
    b = sass.find("Function : ", a + 10)
    seq = []
    for line in sass[a:b if b > 0 else len(sass)].splitlines():
        m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", line)
        if m and (m[1].startswith("HMMA") or m[1].startswith("LDG")):
            seq.append(m[1][0])
    return " ".join(f"{m[0][0]}{len(m[0])}" for m in re.finditer(r"H+|L+", "".join(seq)))


def gate():
    import shutil
    need(shutil.which("nvcc"), "nvcc not on PATH (run through tools/local/cudaenv.sh)")
    work = Path("/tmp/s7f_xr")
    work.mkdir(parents=True, exist_ok=True)
    rows = {}
    files = {"base": ROW}
    files.update({v: OUT / f"k3_xr_{v}.py" for v in VARIANTS})
    for tag, f in files.items():
        t = f.read_bytes().decode()
        keys = gd_keys_for(t)
        inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(map(str, k))) for i, (_, k) in enumerate(keys))
        info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(map(str, k))) for i, (_, k) in enumerate(keys))
        cu = work / f"{tag}.cu"
        cu.write_bytes(cuda_src(t).replace("@INST@", inst).replace("@INFO@", info).encode())
        cmd = ["nvcc", "-cubin", "-std=c++17", "-arch=sm_90a", "-O3", "-Xptxas=-v", str(cu), "-o", str(work / f"{tag}.cubin")]
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=600)
        (work / f"{tag}.ptxas.log").write_text(p.stdout + p.stderr)
        need(p.returncode == 0, f"{tag}: nvcc failed\n{p.stderr[-3000:]}")
        cur = None
        res = {}
        for line in p.stderr.splitlines():
            m = re.search(r"Compiling entry function '([^']+)'", line)
            if m:
                cur = m[1]
                res[cur] = {}
            m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
            if m and cur:
                res[cur].update(stack=int(m[1]), ss=int(m[2]), sl=int(m[3]))
            m = re.search(r"Used (\d+) registers", line)
            if m and cur:
                res[cur]["regs"] = int(m[1])
        sass = subprocess.run(["cuobjdump", "-sass", str(work / f"{tag}.cubin")], text=True, capture_output=True).stdout
        (work / f"{tag}.sass").write_text(sass)
        for name, key in keys:
            sym = [s for s in res if "gd_kernel" in s and s.startswith("_ZN2gd9gd_kernelILi%dELi%dELi%dELi%dELi%dELi%dELi%dELi%dELi%d" % key[:9])]
            need(sym, f"{tag} {name}: symbol for {key} not found")
            r = dict(res[sym[0]])
            r["pattern"] = issue_pattern(sass, sym[0])
            rows.setdefault(name, {})[tag] = (key, r)
    ok = True
    for name, d in rows.items():
        for tag, (key, r) in d.items():
            spill = r.get("ss", 0) + r.get("sl", 0) + r.get("stack", 0)
            ok &= spill == 0
            print(f"[gate] {name:10s} {tag:5s} {key} regs {r.get('regs')} stack {r.get('stack')} spill {r.get('ss')}/{r.get('sl')}")
            print(f"         LDG/HMMA issue order (first units): {r['pattern'][:110]}")
    print("[gate] PASS 0 spills" if ok else "[gate] FAIL: spills / stack")
    return ok


def main():
    if "--gate" in sys.argv:
        sys.exit(0 if gate() else 1)
    base = ROW.read_bytes().decode()
    outs = {v: build(base, v, VARIANTS[v].get("routes")) for v in VARIANTS}
    cpu_checks(base, outs)
    OUT.mkdir(parents=True, exist_ok=True)
    for v, t in outs.items():
        (OUT / f"k3_xr_{v}.py").write_bytes(t.encode())
        print(f"[s7f_xr] wrote box_out/s7/k3_xr_{v}.py")


if __name__ == "__main__":
    main()

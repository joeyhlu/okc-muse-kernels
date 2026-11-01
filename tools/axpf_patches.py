"""AXPF: exact-anchor, single-module runtime KV prefetch experiment.

Only writes kernels/lab/k3_axpf_ab.py. No imports of the layer, CUDA, or runners.
CPU launch/prepare dry run: python -B tools/local/axpf_compile.py --cpu-only
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_vmsx_ship_rt.py"
OUT = ROOT / "kernels/lab/k3_axpf_ab.py"

# Like padz_patches.sub: exact anchors, fail closed on base drift.
def sub(text, old, new, label, count=1):
    assert text.count(old) == count, (label, text.count(old), count)
    return text.replace(old, new)


KNOB = '''# AXPF fractions, read at launch/capture time; one module per kernel family.
AXPF = 0.0
def _axpf_quarters():
    if AXPF not in (0.0, 0.25, 0.5, 1.0):
        raise ValueError("AXPF must be 0, 0.25, 0.5, or 1.0")
    return int(4 * AXPF)
'''

AX_LOOP = r'''// AXPF AX1 BEGIN: only immutable cached rows; no new cache rows or stores.
#ifndef AXPF_COMPILE_OFF
if (p.axpf && p.prew) {
const int axpf_rows = max(0, cend - start);
const int axpf_n = (axpf_rows * p.axpf + 3) >> 2;
const int axpf_lo = cend - axpf_n;
for (int pos = axpf_lo + tid; pos < cend; pos += NTHR) {
const int r = pos - start;
// cpw<1>() + __syncthreads() above completed the first index group.
const int pg = r < (ST - 1) * TILE ? idx[r] : __ldg(p.pt + pos);
const size_t row = (size_t)(uint32_t)pg * ROWB;
#pragma unroll
for (int sector = 0; sector < ROWB; sector += 32) {
asm volatile("prefetch.global.L2::evict_last [%0];\n"
             "prefetch.global.L2::evict_last [%1];\n"
             :: "l"(p.kc + row + sector), "l"(p.vc + row + sector) : "memory");
}
}
}
#endif
// AXPF AX1 END
'''

HYA_LOOP = r'''// AXPF HYA BEGIN: the scored B=1 q8 full specialization only.
#ifndef AXPF_COMPILE_OFF
if constexpr (L == 8 && !FLAT && MODE == 11) {
if (p.axpf && p.B == 1 && p.P == 1 && p.Hkv == 2 && !p.window && seq_real_ == seq) {
const int axpf_end = min(end, seq_real_);
const int axpf_rows = max(0, axpf_end - start);
const int axpf_n = (axpf_rows * p.axpf + 3) >> 2;
for (int pos = axpf_end - axpf_n + tid; pos < axpf_end; pos += NT) {
// S.idx's async group is not ready here: use a dependent, bounded scalar read.
const int pg = __ldg(p.page_table + (ix_t)b * p.max_pages + pos);
const int64_t row = (int64_t)pg * p.row_bytes + pair * 128;
#pragma unroll
for (int sector = 0; sector < ROWB; sector += 32) {
asm volatile("prefetch.global.L2::evict_last [%0];\n"
             "prefetch.global.L2::evict_last [%1];\n"
             :: "l"(p.k_cache + row + sector), "l"(p.v_cache + row + sector) : "memory");
}
}
}
}
#endif
// AXPF HYA END
'''

AX_ANCHOR = '''for (int x = 0; x < ST - 1; ++x) {
if (x < ntiles) issue_kv(x, false);
cpc();
}
pdl_wait();'''
HYA_ANCHOR = '''partL = tidL & 7;
}
pdl_wait();
tfor<L * 16 * 8, NT>(tidL, [&](int e) {'''


def edits():
    # Appending Params members preserves every existing member offset.
    return [
        ("AX1 declaration/definition", "int64_t trig, int64_t prew)",
         "int64_t trig, int64_t prew, int64_t axpf)", 2),
        ("AX1 Params tail", "int prew;\nlong long* prof;\n};",
         "int prew;\nlong long* prof;\nint axpf;\n};", 1),
        ("AX1 host parameter", "P.prew = (int)prew;",
         'P.prew = (int)prew;\nTORCH_CHECK(axpf == 0 || axpf == 1 || axpf == 2 || axpf == 4, "AXPF quarters");\nP.axpf = (L == 1) ? (int)axpf : 0;', 1),
        ("AX1 pre-wait", AX_ANCHOR, AX_ANCHOR.replace("pdl_wait();", AX_LOOP + "pdl_wait();"), 1),
        ("runtime knob", "AX1_ENABLE = True\n", KNOB + "AX1_ENABLE = True\n", 1),
        ("AX1 wrapper", "int(AX1_PREW[L] if prew is None else prew))",
         "int(AX1_PREW[L] if prew is None else prew), _axpf_quarters() if L == 1 else 0)", 1),
        ("HYA declaration/definition", "int64_t tma)", "int64_t tma, int64_t axpf)", 2),
        ("HYA Params tail", "unsigned ns_magic;\n};", "unsigned ns_magic;\nint axpf;\n};", 1),
        ("HYA host parameter", "p.coop = (int)((tma >> 1) & 1);",
         'p.coop = (int)((tma >> 1) & 1);\nTORCH_CHECK(axpf == 0 || axpf == 1 || axpf == 2 || axpf == 4, "AXPF quarters");\np.axpf = (B == 1 && L == 8 && P == 1 && Hkv == 2 && !window && !flat) ? (int)axpf : 0;', 1),
        ("HYA pre-wait", HYA_ANCHOR, HYA_ANCHOR.replace("pdl_wait();", HYA_LOOP + "pdl_wait();"), 1),
        ("HYA wrapper", "None, cs, ncl, tma)\n    return out",
         "None, cs, ncl, tma,\n                    _axpf_quarters() if B == 1 and L == 8 and P == 1 and hkv == 2 and not window and not flat else 0)\n    return out", 1),
    ]


def generate(src):
    assert src.startswith("# okc: problem=muse_decode_layer\n# nonce ")
    s = src
    for label, old, new, n in edits():
        s = sub(s, old, new, label, n)
    lines = s.splitlines(keepends=True)
    lines[1] = "# nonce k3-axpf-ab (k3_vmsx_ship_rt + pre-wait KV L2 prefetch; AXPF=0/.25/.5/1)\n"
    s = "".join(lines)
    compile(s, str(OUT), "exec")   # no pyc or unlisted files
    assert "\r" not in s and s.isascii()
    # Exact inverse proves all post-wait arithmetic/protocol and prepare are untouched.
    inv = s
    for label, old, new, n in reversed(edits()):
        inv = sub(inv, new, old, "inverse " + label, n)
    assert inv.split("\n", 2)[2] == src.split("\n", 2)[2]
    return s


def sources(text):
    names = ("AX1_CPP_SRC", "AX1_CUDA_SRC", "HYA_CPP_SRC", "HYA_CUDA_SRC")
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(text).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in names}


def main():
    src = BASE.read_text(encoding="ascii")
    out = generate(src)
    OUT.write_bytes(out.encode("ascii"))
    print("AXPF generator PASS", len(out), "bytes; base SHA256", hashlib.sha256(src.encode()).hexdigest())
    print("Run python -B tools/local/axpf_compile.py --cpu-only for the CPU gates and prepare/launch dry run.")


if __name__ == "__main__":
    main()

"""rs1_patches.py: RS1 = R-setup rider: the K2A K1 / K2 prologue before the first weight copy, made warp-parallel (R8:
setup 2.4-2.6 us above the one-launch stream on every small-T shape; entry -> gemm start 1.5-2.6 us, gemm start -> first copy
0.9-1.8 us). Scoped text patches inside K2A33_KERNEL_SRC plus the Python call sites; identical results by construction;
switched per call by probe bit 7 (RS1_ENABLE, read at call time; bit 5 = EPI1 lab, bit 6 = study-a FIN1):
  * tail_first (v3.3 gemm: vid 0 / 1 / 2; its three callers are full warps: producer, epilogue, consumers) and tail_first2
    (v3.6 segs36): the serial scan `while (sb[e + 1] <= end - 1) ++e;` (up to 32 dependent shared loads) -> one ballot:
    e = popc(ballot(sb[lane + 1] <= end - 1)). sb is non-decreasing and end <= sb[32], so the lanes that vote form the prefix
    {0 .. e-1} and the popc equals the serial e (v3.7's find_e, shipped).
  * ranges36 (v3.6 role-split cost model): the serial 32-expert loop -> lane = expert and a butterfly sum of the integer
    weights (order-independent, exact), as v3.7's segs37; segs36 then runs on warp 0 (was thread 0) and lane 0 writes.
vid 5 (v3.7) already uses the warp versions; vid 2's WSK range search is untouched.

    python -I tools/rs1_patches.py in.py out.py      (after tools/k3p_patches.py and, if present, study-a fin1_patches.py: both edit these call sites)
"""
import pathlib
import py_compile
import sys

REGION = ('K2A33_KERNEL_SRC = r"""', '"""\n')

SERIAL_TF = ("  int e = 0;\n"
             "  while (S.sb[e + 1] <= end - 1) ++e;\n")
SERIAL_TF2 = ("  int e = 0;\n"
              "  while (sb[e + 1] <= end - 1) ++e;\n")

CUDA = [
    ("tail_first: rs1 ballot",
     "__device__ __forceinline__ int tail_first(const Route& S, int beg, int end) {\n"
     "  if (end <= beg) return beg;\n" + SERIAL_TF,
     "__device__ __forceinline__ int tail_first(const Route& S, int beg, int end, int rs1 = 0) {   // rs1: full warp\n"
     "  if (end <= beg) return beg;\n"
     "  int e = 0;\n"
     "  if (rs1) e = __popc(__ballot_sync(0xffffffffu, S.sb[(threadIdx.x & 31) + 1] <= end - 1));   // RS1: = the scan\n"
     "  else while (S.sb[e + 1] <= end - 1) ++e;\n", 1),
    ("tail_first callers", "const int mid = tail_first(S, beg, end);",
     "const int mid = tail_first(S, beg, end, (P.probe >> 7) & 1);", 3),
    ("tail_first2: rs1 ballot",
     "__device__ __forceinline__ int tail_first2(const int* sb, int beg, int end) {\n"
     "  if (end <= beg) return beg;\n" + SERIAL_TF2,
     "__device__ __forceinline__ int tail_first2(const int* sb, int beg, int end, int rs1 = 0) {   // rs1: full warp\n"
     "  if (end <= beg) return beg;\n"
     "  int e = 0;\n"
     "  if (rs1) e = __popc(__ballot_sync(0xffffffffu, sb[(threadIdx.x & 31) + 1] <= end - 1));   // RS1: = the scan\n"
     "  else while (sb[e + 1] <= end - 1) ++e;\n", 1),
    ("ranges36: rs1 lane-parallel cost model",
     "    long long Wh = 0, Wc = 0;\n"
     "    for (int x = 0; x < 32; ++x) {\n"
     "      const int n = S.cnt[x];\n",
     "    long long Wh = 0, Wc = 0;\n"
     "    if ((P.probe >> 7) & 1) {   // RS1: warp 0, lane = expert, exact integer butterfly sum (v3.7 segs37's scheme)\n"
     "      const int n = S.cnt[threadIdx.x & 31];\n"
     "      if (n > 0) {\n"
     "        if (n > P.hot_t) {\n"
     "          const int tp = 8 * NCH, full = n / tp, rem = n - full * tp;\n"
     "          Wh = (long long)full * (16 + P.hot_w4 * (NCH - 1)) + (rem ? 16 + P.hot_w4 * (((rem + 7) >> 3) - 1) : 0);\n"
     "        } else {\n"
     "          Wc = 16ll * ((n + 7) >> 3);\n"
     "        }\n"
     "      }\n"
     "#pragma unroll\n"
     "      for (int o = 16; o > 0; o >>= 1) { Wh += __shfl_xor_sync(0xffffffffu, Wh, o); Wc += __shfl_xor_sync(0xffffffffu, Wc, o); }\n"
     "    } else\n"
     "    for (int x = 0; x < 32; ++x) {\n"
     "      const int n = S.cnt[x];\n", 1),
    ("segs36: rs1 tail_first2 + lane-0 writes",
     "    const int mid = tail_first2(k[part] ? S.sbh : S.sbc, b[part], e[part]);\n"
     "    sg[6 * part + 0] = mid;     sg[6 * part + 1] = e[part]; sg[6 * part + 2] = k[part];\n"
     "    sg[6 * part + 3] = b[part]; sg[6 * part + 4] = mid;     sg[6 * part + 5] = k[part];\n",
     "    const int mid = tail_first2(k[part] ? S.sbh : S.sbc, b[part], e[part], (P.probe >> 7) & 1);\n"
     "    if ((threadIdx.x & 31) == 0) {   // RS1: segs36 may run on warp 0\n"
     "      sg[6 * part + 0] = mid;     sg[6 * part + 1] = e[part]; sg[6 * part + 2] = k[part];\n"
     "      sg[6 * part + 3] = b[part]; sg[6 * part + 4] = mid;     sg[6 * part + 5] = k[part];\n"
     "    }\n", 1),
    ("segs36 on warp 0 when rs1 (K1 and K2)",
     "  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr);\n",
     "  if (threadIdx.x < (((P.probe >> 7) & 1) ? 32 : 1)) segs36<NCH>(P, S, s_pr);   // RS1: warp 0\n", 2),
]

CALL36 = '((1 << 25) if _SU1_FLAG["k3trig"] else 0) | _k3p_bits(m)'   # the _k2a33 / _k2a36 launches, with or without FIN1
PY_ANCHOR = "\ndef kernel_fn(data, state):\n"
PY = '''

# RS1 (tools/rs1_patches.py): warp-parallel K2A setup (tail_first / segs36), probe bit 7; read at call time
RS1_ENABLE = True

'''


def apply(text):
    if text.count(REGION[0]) != 1:
        raise RuntimeError("rs1 patch: K2A33_KERNEL_SRC not unique")
    for name, old, new, count in CUDA:
        a = text.index(REGION[0])
        b = text.index(REGION[1], a + len(REGION[0]))
        n = text[a:b].count(old)
        if n != count:
            raise RuntimeError(f"rs1 patch {name!r}: `old` occurs {n} times in K2A33_KERNEL_SRC (want {count})")
        text = text[:a] + text[a:b].replace(old, new) + text[b:]
    if text.count(CALL36) != 2:
        raise RuntimeError(f"rs1 patch: _k2a33 / _k2a36 call sites x{text.count(CALL36)} (want 2; apply after k3p / fin1)")
    text = text.replace(CALL36, CALL36 + " | ((1 << 7) if RS1_ENABLE else 0)")
    if text.count(PY_ANCHOR) != 1:
        raise RuntimeError("rs1 patch: kernel_fn anchor")
    return text.replace(PY_ANCHOR, PY + PY_ANCHOR)


if __name__ == "__main__":
    s, d = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    t = s.read_text(encoding="utf-8")
    out = apply(t)
    d.write_text(out, encoding="utf-8", newline="\n")
    py_compile.compile(str(d), doraise=True)
    print(f"rs1: applied {s.name} -> {d} ({len(out) - len(t):+d} chars, compiles)")

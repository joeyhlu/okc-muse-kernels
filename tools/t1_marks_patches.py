"""t1_marks_patches.py: LAB ONLY (never in a scored file). Per-CTA %globaltimer marks in the vid 0 / vid 7 K2A33 kernels
(NC 1, not HOTCAP: the T <= 8 route) and in k3_kernel_b at M <= 8, written into an always-unused tail of the xw2 buffer
(byte offset 64 MiB: hot W2 passes need < 72 x 512 KiB at M <= 320), so no Params / host change is needed.

Record (k 0 = K1, 1 = K2, 2 = K3; 16 int64 per CTA, CTA < 1024) at ((k * 1024 + cta) * 16 + f):
  0 smid, 1 entry, 2 after K1's top griddepcontrol.wait (= routing tail complete), 3 route done (gemm start), 4 first weight
  copy issued (producer lane 0), 5 producer back from griddepcontrol.wait (K2: K1 -> K2 hand-off), 6 consumer warp 0 gemm end,
  7 epilogue warp exit, 8 stages issued by the producer, 9 last weight copy issued, 10 producer exit.
  K3: 1 entry, 2 after griddepcontrol.wait, 6 end.
Host: st["xw2"][64 << 20 : (64 << 20) + 3 * 1024 * 16 * 8].view(torch.int64).view(3, 1024, 16).

    text = apply_marks(text)   (after apply_sk1 + apply_t1)
"""

K33_HEAD = 'K2A33_KERNEL_SRC = r"""'
ROUTE_HEAD = "template <int NC, int TPH = 64, bool HGE = false, bool LIST = true>\n__device__ void route("
HELPER = r"""// ---- T1 marks lab (tools/t1_marks_patches.py) ----
__device__ __forceinline__ long long t1m_gt() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void t1m(const Params& P, int k, int f, long long v) {
  if (blockIdx.x < 1024) reinterpret_cast<long long*>(P.xw2 + ((size_t)64 << 20))[((size_t)k * 1024 + blockIdx.x) * 16 + f] = v;
}
"""
K1_HEAD = "template <int NC, int NST, int MINB, bool HOTCAP, bool SK1 = false>\n__global__ void __launch_bounds__(320, MINB) k1_kernel(const Params P) {"
K2_HEAD = "template <int NC, int NST, int MINB, bool HOTCAP, bool SK1 = false>\n__global__ void __launch_bounds__(320, MINB) k2_kernel(const Params P) {"
GEMM_HEAD = "template <int NC, int NST, int MODE, int HM, int STG, bool SKF = false>\n__device__ void gemm("
K3B_HEAD = "__global__ void __launch_bounds__(256) k3_kernel_b(const Params P) {"
ENTRY = ("  if (T1M && threadIdx.x == 0) { unsigned sm; asm volatile(\"mov.u32 %%0, %%%%smid;\" : \"=r\"(sm)); "
         "t1m(P, %d, 0, sm); t1m(P, %d, 1, t1m_gt()); }\n")
# raw (R-t1-4): + field 12 = arrival slot on the SM (atomicAdd on a per-SM counter at row 512 + smid of the kernel's block)
ENTRY_RAW = ("  if (T1M && threadIdx.x == 0) { unsigned sm; asm volatile(\"mov.u32 %%0, %%%%smid;\" : \"=r\"(sm)); "
             "t1m(P, %d, 0, sm); t1m(P, %d, 1, t1m_gt()); unsigned long long* ctr = reinterpret_cast<unsigned long long*>("
             "P.xw2 + ((size_t)64 << 20)) + ((size_t)%d * 1024 + 512 + sm) * 16; t1m(P, %d, 12, (long long)atomicAdd(ctr, 1ull)); }\n")
FIRST_LAND = ("      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);\n",
              "      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);\n"
              "      if (T1M && tid == 0 && it == 0) t1m(P, MODE - 1, 11, t1m_gt());   // R-t1-4: first stage landed\n")


def _region(text, head, end, edits, what):
    if text.count(head) != 1:
        raise ValueError(f"marks region head {what!r} found {text.count(head)} times")
    a = text.index(head)
    b = text.index(end, a + len(head))
    seg = text[a:b]
    for old, new in edits:
        if seg.count(old) != 1:
            raise ValueError(f"marks {what}: {old[:50]!r} found {seg.count(old)} times")
        seg = seg.replace(old, new)
    return text[:a] + seg + text[b:]


def apply_marks(text, raw=False):
    """raw (R-t1-4): + field 11 (first stage landed: consumer warp 0 past its first full wait) and 12 (arrival slot on the SM)."""
    if text.count(K33_HEAD) != 1:
        raise ValueError("marks: K2A33 head")
    k0 = text.index(K33_HEAD)
    k1 = text.index('"""', k0 + len(K33_HEAD))
    src = text[k0:k1]
    if src.count(ROUTE_HEAD) != 1:
        raise ValueError("marks: route head")
    src = src.replace(ROUTE_HEAD, HELPER + ROUTE_HEAD)
    t1m = "  constexpr bool T1M = (NC == 1 && !HOTCAP);\n"
    src = _region(src, K1_HEAD, "\n}\n", (
        ("  pdl_launch();\n", "  pdl_launch();\n" + t1m + (ENTRY_RAW % (0, 0, 0, 0) if raw else ENTRY % (0, 0))),
        ("  if constexpr (!HOTCAP) pdl_wait();   // v4.7: K1 is a PDL secondary of the routing tail (vid 0 / 1)\n",
         "  if constexpr (!HOTCAP) pdl_wait();   // v4.7: K1 is a PDL secondary of the routing tail (vid 0 / 1)\n"
         "  if (T1M && threadIdx.x == 0) t1m(P, 0, 2, t1m_gt());\n"),
        ("    gemm<NC, NST, 1, 0, STG>(P, S, s_list, smem, bars, false, blockIdx.x, gridDim.x);\n",
         "    if (T1M && threadIdx.x == 0) t1m(P, 0, 3, t1m_gt());\n"
         "    gemm<NC, NST, 1, 0, STG>(P, S, s_list, smem, bars, false, blockIdx.x, gridDim.x);\n"
         "    if (T1M && threadIdx.x == 0) t1m(P, 0, 6, t1m_gt());\n"),
    ), "k1")
    k2a = src.index(K2_HEAD)
    k2seg = src[k2a:src.index("\n}\n", k2a)]
    K2G = next(g for g in ("    gemm<NC, NST, 2, 0, STG, SK1>(P, S, s_list, smem, bars, true, blockIdx.x, gridDim.x);\n",
                           "    gemm<NC, NST, 2, 0, STG>(P, S, s_list, smem, bars, true, blockIdx.x, gridDim.x);\n") if g in k2seg)
    src = _region(src, K2_HEAD, "\n}\n", (
        ("  pdl_launch();\n", "  pdl_launch();\n" + t1m + (ENTRY_RAW % (1, 1, 1, 1) if raw else ENTRY % (1, 1))),
        (K2G, "    if (T1M && threadIdx.x == 0) t1m(P, 1, 3, t1m_gt());\n" + K2G
         + "    if (T1M && threadIdx.x == 0) t1m(P, 1, 6, t1m_gt());\n"),
    ), "k2")
    if raw:
        src = _region(src, GEMM_HEAD, "\n}\n", (FIRST_LAND,), "gemm first stage landed")
    src = _region(src, GEMM_HEAD, "\n}\n", (
        ("  constexpr bool HOTCAP = (HM == 1);\n", "  constexpr bool HOTCAP = (HM == 1);\n  constexpr bool T1M = (NC == 1 && HM == 0);\n"),
        ("          if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);\n",
         "          if (T1M) { if (it == 0) t1m(P, MODE - 1, 4, t1m_gt()); t1m(P, MODE - 1, 9, t1m_gt()); t1m(P, MODE - 1, 8, it + 1); }\n"
         "          if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);\n"),
        ("          pdl_wait();\n          asm volatile(\"fence.proxy.async;\\n\" ::: \"memory\");\n          waited = true;\n",
         "          pdl_wait();\n          if (T1M && lane == 0) t1m(P, MODE - 1, 5, t1m_gt());\n"
         "          asm volatile(\"fence.proxy.async;\\n\" ::: \"memory\");\n          waited = true;\n"),
        ("    if (!waited) {\n      pdl_wait();\n",
         "    if (!waited) {\n      pdl_wait();\n      if (T1M && lane == 0) t1m(P, MODE - 1, 5, t1m_gt());\n"),
        ("    return;\n  }\n\n  if (warp == 9) {\n", "    if (T1M && lane == 0) t1m(P, MODE - 1, 10, t1m_gt());\n    return;\n  }\n\n  if (warp == 9) {\n"),
        ("    return;\n  }\n\n  // ---------------- consumers", "    if (T1M && lane == 0) t1m(P, MODE - 1, 7, t1m_gt());\n"
         "    return;\n  }\n\n  // ---------------- consumers"),
    ), "gemm")
    src = _region(src, K3B_HEAD, "\n}\n", (
        ("  pdl_wait();\n", "  if (P.M <= 8 && threadIdx.x == 0) t1m(P, 2, 1, t1m_gt());\n  pdl_wait();\n"
         "  if (P.M <= 8 && threadIdx.x == 0) t1m(P, 2, 2, t1m_gt());\n"),
        ("    __syncthreads();\n  }", "    __syncthreads();\n  }\n  if (P.M <= 8 && threadIdx.x == 0) t1m(P, 2, 6, t1m_gt());"),
    ), "k3b")
    return text[:k0] + src + text[k1:]

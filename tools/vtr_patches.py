"""vtr_patches.py: VTR = small VML (T >= 17 MoE) tail riders on k3_gc_9's VML block (vm_main trigger point, vm_fin, route()
job-split math). Every rider is a compile-time flag, default 0 = the base source's behaviour.

  VM_FT  1: vm_fin fires griddepcontrol.launch_dependents at entry (all threads), so the next kernel (the GD unpooler at
            T = 32) lands on the idle SMs and preloads its weights before its own griddepcontrol.wait. vm_fin's consumer
            still waits for vm_fin's completion (its wait), so the data dependency is unchanged.
  VM_LD  2: vm_main triggers right after its (optional) griddepcontrol.wait instead of at exit (1 = the existing tid-0
            trigger after phase A; 0 = implicit at exit). vm_fin CTAs then land on SMs freed by early-finishing vm_main CTAs.
  VM_FTR 1: vm_fin trims, bitwise-identical output: topk weights loaded before griddepcontrol.wait (they are vm_main's input,
            written before vm_main started; vm_main triggers no earlier than after its own wait), warp-0 ballot compaction
            of the 8 routes (one CTA barrier less, no serial tid-0 loop), bf16r(v) * w products computed once and reused.
  VM_TA  1: route() job split tile-aligned: base = ceil(ceil(tot / 8) / nj) * 8 instead of ceil(tot / nj) (nj unchanged):
            the max job keeps its tile count, total padded tiles never grow (one fewer whenever ceil(tot / 8) % nj leaves
            room); every job still <= VM_JN rows and non-empty (checked for tot 1..2048 at JN 64 / 72).
"""

_FT = 'asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");'

REPL = [
    # defaults
    ("#ifndef VM_LD\n#define VM_LD 0\n#endif\n",
     "#ifndef VM_LD\n#define VM_LD 0\n#endif\n#ifndef VM_FT\n#define VM_FT 0\n#endif\n#ifndef VM_FTR\n#define VM_FTR 0\n#endif\n"
     "#ifndef VM_TA\n#define VM_TA 0\n#endif\n", 1),
    # VM_TA: route() job split (JN 64 closed form, the VM_RT route and the stock route)
    ("if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));\n#else\n",
     "#if VM_TA\nif (nj) base = ((((tot + 7) >> 3) + nj - 1) / nj) << 3;   // VTR: tile-aligned split\n#else\n"
     "if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));\n#endif\n#else\n", 1),
    ("const int base = nj ? (tot + nj - 1) / nj : VM_JN;\n",
     "#if VM_TA\nconst int base = nj ? ((((tot + 7) >> 3) + nj - 1) / nj) << 3 : VM_JN;   // VTR: tile-aligned split\n#else\n"
     "const int base = nj ? (tot + nj - 1) / nj : VM_JN;\n#endif\n", 2),
    # VM_LD 2: vm_main trigger right after its wait; VM_LD 1 keeps the after-phase-A trigger
    ('if (A.flags & 256) asm volatile("griddepcontrol.wait;\\n" ::: "memory");   // PDL: ids / x come from the previous kernel\n',
     'if (A.flags & 256) asm volatile("griddepcontrol.wait;\\n" ::: "memory");   // PDL: ids / x come from the previous kernel\n'
     "#if VM_LD == 2\nif (tid == 0) " + _FT + "   // VTR: vm_fin may launch now (after the wait: its pre-wait wts load is safe)\n#endif\n", 1),
    ("#if VM_LD\nif (tid == 0) asm volatile", "#if VM_LD == 1\nif (tid == 0) asm volatile", 1),
    # vm_fin: entry trigger + pre-wait wts
    ("const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\nfloat4 w4[4];\n",
     "const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n"
     "#if VM_FT\n" + _FT + "   // VTR: the next kernel lands + preloads weights now (it still waits for our completion)\n#endif\n"
     "#if VM_FTR\nconst float wv = (tid < 8) ? __ldg(A.wts + t * 8 + tid) : 0.f;   // vm_main's input: ready before vm_main ran\n#endif\n"
     "float4 w4[4];\n", 1),
    # vm_fin: ballot compaction
    ("if (tid < 8) { s_r8[tid] = __ldcg(A.rmap + t * 8 + tid); s_w8[tid] = __ldg(A.wts + t * 8 + tid); }\n__syncthreads();\n"
     "if (tid == 0) {\nint n = 0;\n#pragma unroll\nfor (int k = 0; k < 8; ++k) {\nconst int r = s_r8[k];\n"
     "if (r >= 0) { s_r[n] = r; s_w[n] = s_w8[k]; ++n; }\n}\ns_n = n;\n}\n__syncthreads();\n",
     "#if VM_FTR\nif (warp == 0) {   // VTR: ballot compaction, order kept (== the serial loop)\n"
     "const int r = (lane < 8) ? __ldcg(A.rmap + t * 8 + lane) : -1;\nconst unsigned bm = __ballot_sync(0xffffffffu, r >= 0);\n"
     "if (r >= 0) { const int pos = __popc(bm & ((1u << lane) - 1u)); s_r[pos] = r; s_w[pos] = wv; }\n"
     "if (lane == 0) s_n = __popc(bm);\n}\n__syncthreads();\n#else\n"
     "if (tid < 8) { s_r8[tid] = __ldcg(A.rmap + t * 8 + tid); s_w8[tid] = __ldg(A.wts + t * 8 + tid); }\n__syncthreads();\n"
     "if (tid == 0) {\nint n = 0;\n#pragma unroll\nfor (int k = 0; k < 8; ++k) {\nconst int r = s_r8[k];\n"
     "if (r >= 0) { s_r[n] = r; s_w[n] = s_w8[k]; ++n; }\n}\ns_n = n;\n}\n__syncthreads();\n#endif\n", 1),
    # vm_fin: products once (v <- bf16r(v) * w4 in place)
    ("float ss[4];\n#pragma unroll\nfor (int jj = 0; jj < 4; ++jj) {\nss[jj] = 0.f;\n",
     "#if VM_FTR\n#pragma unroll\nfor (int jj = 0; jj < 4; ++jj)\n#pragma unroll\nfor (int m = 0; m < 4; ++m) {   // VTR: products once\n"
     "v[jj][m].x = bf16r(v[jj][m].x) * w4[m].x; v[jj][m].y = bf16r(v[jj][m].y) * w4[m].y;\n"
     "v[jj][m].z = bf16r(v[jj][m].z) * w4[m].z; v[jj][m].w = bf16r(v[jj][m].w) * w4[m].w;\n}\n#endif\n"
     "float ss[4];\n#pragma unroll\nfor (int jj = 0; jj < 4; ++jj) {\nss[jj] = 0.f;\n", 1),
    ("f = bf16r(v[jj][m].x) * w4[m].x; ss[jj] += f * f; f = bf16r(v[jj][m].y) * w4[m].y; ss[jj] += f * f;\n"
     "f = bf16r(v[jj][m].z) * w4[m].z; ss[jj] += f * f; f = bf16r(v[jj][m].w) * w4[m].w; ss[jj] += f * f;\n",
     "#if VM_FTR\nf = v[jj][m].x; ss[jj] += f * f; f = v[jj][m].y; ss[jj] += f * f;\nf = v[jj][m].z; ss[jj] += f * f; f = v[jj][m].w; ss[jj] += f * f;\n#else\n"
     "f = bf16r(v[jj][m].x) * w4[m].x; ss[jj] += f * f; f = bf16r(v[jj][m].y) * w4[m].y; ss[jj] += f * f;\n"
     "f = bf16r(v[jj][m].z) * w4[m].z; ss[jj] += f * f; f = bf16r(v[jj][m].w) * w4[m].w; ss[jj] += f * f;\n#endif\n", 1),
    ("a[m][0] += bf16r(v[jj][m].x) * w4[m].x * fr; a[m][1] += bf16r(v[jj][m].y) * w4[m].y * fr;\n"
     "a[m][2] += bf16r(v[jj][m].z) * w4[m].z * fr; a[m][3] += bf16r(v[jj][m].w) * w4[m].w * fr;\n",
     "#if VM_FTR\na[m][0] += v[jj][m].x * fr; a[m][1] += v[jj][m].y * fr;\na[m][2] += v[jj][m].z * fr; a[m][3] += v[jj][m].w * fr;\n#else\n"
     "a[m][0] += bf16r(v[jj][m].x) * w4[m].x * fr; a[m][1] += bf16r(v[jj][m].y) * w4[m].y * fr;\n"
     "a[m][2] += bf16r(v[jj][m].z) * w4[m].z * fr; a[m][3] += bf16r(v[jj][m].w) * w4[m].w * fr;\n#endif\n", 1),
]


def apply(cuda):
    for old, new, n in REPL:
        c = cuda.count(old)
        if c != n:
            raise SystemExit(f"[vtr_patches] anchor count {c} != {n}: {old[:90]!r}")
        cuda = cuda.replace(old, new)
    return cuda

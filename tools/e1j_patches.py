"""E1J lab (API timeline review tl_opus.md E1): at T = 1 the FP32 router runs on the side stream and the main stream joins it right before
tail_kernel3 (cudaStreamWaitEvent), which removes the tail's PDL edge from the pooler (nsys c26 d1s: pooler ends 31.6, router 32.4, tail
starts 33.7). E1J: vf_router32_kernel<1> releases a device counter per CTA (bar.sync + thread-0 fence + atomicAdd); tail_kernel3's routing
warps acquire-spin on it (>= 128 router CTAs) after griddepcontrol.wait and the last consumer resets it; the main -> side join moves to the
end of kernel_fn. Router partials are read with __ldcg (L2-coherent). Module global E1J_ON (0 = ship behaviour; the flag pointer is null).
Base kernels/muse_decode_layer/k3_moexr_c2.py -> kernels/lab/k3_e1j_ab.py."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexr_c2.py"
OUT = ROOT / "kernels/lab/k3_e1j_ab.py"
Q, BS = chr(34), chr(92)
NL = BS + "n"


def once(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:120])
    return s.replace(a, b)


def main():
    s = BASE.read_text(encoding="utf-8").replace("\r\n", "\n")
    # --- router kernel (SBB1 module): signature + per-CTA release
    rs = "vf_router32_kernel(const float* __restrict__ x, const float* __restrict__ w,\nfloat* __restrict__ part, int m) {"
    s = once(s, rs, rs.replace("int m) {", "int m, int* rflag = nullptr) {"))
    s = once(s, "}\n}\n}\n\n// x through L1, <= 80 regs",
             "}\n}\nif (rflag != nullptr) {  // E1J: per-CTA release of this CTA's partials\n__syncthreads();\n"
             "if (threadIdx.x == 0) { __threadfence(); atomicAdd(rflag, 1); }\n}\n}\n\n// x through L1, <= 80 regs")
    s = once(s, "void vf_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part, int64_t kind) {",
             "void vf_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part, int64_t kind, torch::Tensor rflag) {\n"
             "int* rf = rflag.numel() ? rflag.data_ptr<int>() : nullptr;")
    s = once(s, "void vf_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part, int64_t kind);",
             "void vf_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part, int64_t kind, torch::Tensor rflag);")
    s = once(s, "vf_router32_kernel<1><<<dim3(16, 8), 256, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);",
             "vf_router32_kernel<1><<<dim3(16, 8), 256, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m, rf);")
    # --- tail kernels (V8 module): signature (all three: FT_K needs one function type), launches, wrapper, spin in tail_kernel3
    s = once(s, "float eps, float mult, float rscale, float deps) {", "float eps, float mult, float rscale, float deps, int* rflag) {", 3)
    s = once(s, "(float)deps));", "(float)deps, rf));", 4)
    s = once(s, "double deps,\nint64_t pdl);", "double deps,\ntorch::Tensor rflag, int64_t pdl);")
    s = once(s, "double deps,\nint64_t pdl) {",
             "double deps,\ntorch::Tensor rflag, int64_t pdl) {\nint* rf = rflag.numel() ? rflag.data_ptr<int>() : nullptr;")
    W = ("asm volatile(" + Q + "griddepcontrol.launch_dependents;" + NL + Q + " ::: " + Q + "memory" + Q + ");\nasm volatile("
         + Q + "griddepcontrol.wait;" + NL + Q + " ::: " + Q + "memory" + Q + ");\n")
    spin = ("if (rflag != nullptr) {  // E1J: acquire the side-stream router's 128 CTA releases, last consumer resets\n"
            "if (lane == 0) {\nint v;\nwhile (true) {\nasm volatile(" + Q + "ld.acquire.gpu.global.s32 %0, [%1];" + Q
            + " : " + Q + "=r" + Q + "(v) : " + Q + "l" + Q + "(rflag) : " + Q + "memory" + Q + ");\n"
            "if (v >= 128) break;\n__nanosleep(32);\n}\n"
            "const int old = atomicAdd(rflag, 1);\nif (old == 128 + (int)gridDim.x * RW - 1) atomicExch(rflag, 0);\n}\n__syncwarp();\n}\n")
    s = once(s, W + "float qa[NSMAX][EPT], qb[EPT];", W + spin + "float qa[NSMAX][EPT], qb[EPT];")
    # --- python plumbing
    old_r = ("_SBB1[" + Q + "mod" + Q + "].vf_router32(router_input,state[" + Q + "router" + Q + "],part,{" + Q + "lean4" + Q + ":1,"
             + Q + "lean5" + Q + ":2}.get(kind,0))")
    s = once(s, old_r, old_r[:-1] + ",_e1j_flag(state,router_input.shape[0]==1 and kind not in(" + Q + "lean4" + Q + "," + Q + "lean5"
             + Q + ")))")
    s = once(s, "mod.vf_router32(z,state[" + Q + "router" + Q + "],got,kind)",
             "mod.vf_router32(z,state[" + Q + "router" + Q + "],got,kind,_E1J_EMPTY(z.device))")
    old_t = " RMS_EPS,0.1767766952966369,1.0,1e-15,(1 if V9_TAIL_PDL else 0)"
    s = once(s, old_t, " RMS_EPS,0.1767766952966369,1.0,1e-15,_e1j_flag(state,True),(1 if V9_TAIL_PDL else 0)")
    glue = ("E1J_ON = 0\n_E1J_E = {}\n"
            "def _E1J_EMPTY(dev):\n"
            " t = _E1J_E.get(str(dev))\n"
            " if t is None:\n"
            "  t = _E1J_E[str(dev)] = torch.empty(0, dtype=torch.int32, device=dev)\n"
            " return t\n"
            "def _e1j_flag(state, cond):\n"
            " f = state.get('e1j_flag')\n"
            " if state.get('_e1j') and cond and f is not None:\n"
            "  return f\n"
            " return _E1J_EMPTY(state['router'].device)\n")
    s = once(s, "def _v8_logits(mode,ffn_input,router_input,state):\n", glue + "def _v8_logits(mode,ffn_input,router_input,state):\n")
    side = "   main_stream,side_stream=torch.cuda.current_stream(),state[" + Q + "side_stream" + Q + "]\n"
    s = once(s, side, side
             + "   if E1J_ON and tokens==1 and tail is not None and mode=='f32' and VF_ROUTER.get(1,'r32')=='lean' and _VF['r8'] and _SBB1['mod'] is not None:\n"
             "    if state.get('e1j_flag') is None:\n"
             "     state['e1j_flag']=torch.zeros(4,dtype=torch.int32,device=ffn_input.device)\n"
             "    state['_e1j']=True\n")
    s = once(s, "   main_stream.wait_stream(side_stream)\n",
             "   if state.get('_e1j'):\n    state['_e1j_join']=side_stream\n   else:\n    main_stream.wait_stream(side_stream)\n")
    tl = "  if tail is not None:\n   expert_x,topk_weights,topk_ids,local_ids=_v8_tail(expert_input,logits,state,data)\n"
    s = once(s, tl, tl + "   state['_e1j']=False\n")
    ret = (" next_attn_input,stream,_=_v9_bnd(" + Q + "ffn" + Q + ",stream,ffn_output,None,state,tokens)\n"
           " return next_attn_input,stream,topk_ids\n")
    s = once(s, ret, ret.replace(" return next_attn_input",
                                 " if state.get('_e1j_join') is not None:\n"
                                 "  torch.cuda.current_stream().wait_stream(state['_e1j_join'])\n"
                                 "  state['_e1j_join']=None\n"
                                 " return next_attn_input"))
    compile(s, str(OUT), "exec")
    OUT.write_bytes(s.encode("ascii"))
    print("E1J lab", OUT, len(s.encode("ascii")), "B")


if __name__ == "__main__":
    main()

"""RQT lab: rq (T = 128 qkv / unpooler GEMV, chenwainuo lineage) with griddepcontrol.launch_dependents at kernel entry (RQ_TRIG build), so
the PDL-launched HYA attention can become resident during the qkv GEMV at v16s / v16f (nsys c26: HYA starts 0.8 us after rq ends; rq has
no PDL at all). Safe with rq's co-residency polls: the dependent grid launches only after every rq CTA is resident and has triggered.
Module global RQ_TRIG_ON: 0 ship module; 1 trigger module for qkv@128 only (second load_inline build of the same variants, -DRQ_TRIG=1).
Default base = kernels/lab/k3_sktr_ab.py (so one lab file carries SKTR + RQT); --base= to change."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_sktr_ab.py"
OUT = ROOT / "kernels/lab/k3_rqt_ab.py"
Q, BS = chr(34), chr(92)

GLUE = r'''
RQ_TRIG_ON = 0
_RQT = {"mod": None, "err": None}

def _rqt_build():
 try:
  import hashlib as _h
  from torch.utils.cpp_extension import load_inline as _li
  src = _rqsrc(_rq_variants)
  fl = list(_rq_flags) + ["-DRQ_TRIG=1"]
  tag = _h.sha256((src + repr(fl)).encode()).hexdigest()[:12]
  _RQT["mod"] = _li(name="rq_trig_" + tag, cpp_sources=_rq_cpp, cuda_sources=src, functions=["rq_run"], extra_cuda_cflags=fl, verbose=False)
 except Exception as ex:
  _RQT["err"] = repr(ex)[:300]
  print(f"[rqt] build failed: {ex!r}", flush=True)
'''


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:100])
    return s.replace(a, b, 1)


def main():
    base = BASE
    for a in sys.argv[1:]:
        if a.startswith("--base="):
            base = ROOT / a.split("=", 1)[1]
    src = base.read_text(encoding="utf-8").replace("\r\n", "\n")
    ent = "rq_kernel(const rq_P p, const __grid_constant__ rq_Tmap tx, const __grid_constant__ rq_Tmap to) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n"
    trig = ("#ifdef RQ_TRIG\n  asm volatile(" + Q + "griddepcontrol.launch_dependents;" + Q + " ::: " + Q + "memory" + Q
            + ");  // RQT: dependents may become resident now\n#endif\n")
    src = once(src, ent, ent + trig)
    src = once(src, "def _rq_gemm(state,proj,x):\n", GLUE + "\ndef _rq_gemm(state,proj,x):\n")
    src = once(src, "  return rq_out if _rq_call(rq_p,x,rq_out)==0 else None\n",
               "  if RQ_TRIG_ON and proj==" + Q + "qkv" + Q + " and _RQT[" + Q + "mod" + Q + "] is not None:\n"
               "   rq_p=dict(rq_p,rq_mod=_RQT[" + Q + "mod" + Q + "])\n"
               "  return rq_out if _rq_call(rq_p,x,rq_out)==0 else None\n")
    src = once(src, '  jobs.append(("hyb",_hyb_build))\n', '  jobs.append(("hyb",_hyb_build))\n  jobs.append(("rqt",_rqt_build))\n')
    compile(src, str(OUT), "exec")
    OUT.write_bytes(src.encode("ascii"))
    print("RQT lab", OUT, len(src.encode("ascii")), "B")


if __name__ == "__main__":
    main()

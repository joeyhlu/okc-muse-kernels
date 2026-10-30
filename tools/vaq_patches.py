"""vaq_patches.py: VMS_AQ = acquire-pattern B wait in VMS (moe redesign, R-vaq).
Shipped (FX bit 1 off): lane 0 polls the doneA counter with ld.relaxed.gpu, then every lane issues one more ld.acquire.gpu of
the same counter -> a full extra L2 round trip on the A->B hop (at T=1 the hop is on the critical path). FX bit 1 polls with
ld.acquire directly: -2.4 % at bs1 board but +0.4..+0.5 % at bs 8 / 16 (every poll iteration = LDG.STRONG + CCTL.IVALL, an
L1 invalidate under the other warps). VMS_AQ 1: keep the relaxed poll, then lane 0 issues fence.acquire.gpu (relaxed read +
fence.acquire = the PTX acquire pattern, A29; ~0 cost under the ring, R-fnc-1) and the __syncwarp orders the other lanes' __ldcg
h loads after it. No extra round trip, no per-iteration L1 invalidate. At VMS_AQ 0 the source preprocesses to the base's.
    python -I tools/vaq_patches.py <file.py>
"""
import pathlib
import sys

OLD = ("            while (ld_rlx(p.ctr + 8 + jj) < RBA) {\n"
       "              if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }\n"
       "              __nanosleep(64);\n"
       "            }\n"
       "          }\n"
       "          __syncwarp();\n"
       "          (void)ld_acq(p.ctr + 8 + jj);\n")
NEW = r"""            while (ld_rlx(p.ctr + 8 + jj) < RBA) {
              if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }
              __nanosleep(64);
            }
#if VMS_AQ
#if __CUDA_ARCH__ >= 900
            asm volatile("fence.acquire.gpu;" ::: "memory");   // R-vaq: relaxed read + fence.acquire = acquire pattern
#else
            asm volatile("fence.acq_rel.gpu;" ::: "memory");   // sm_86 test builds (no fence.acquire)
#endif
#endif
          }
          __syncwarp();
#if !VMS_AQ
          (void)ld_acq(p.ctr + 8 + jj);
#endif
"""
DEF = "#ifndef VMS_AQ\n#define VMS_AQ 0   // R-vaq: 1 = B wait ends in lane-0 fence.acquire instead of a second ld.acquire round trip\n#endif\n"


def apply(s):
    assert s.count(OLD) == 1, f"vaq anchor found {s.count(OLD)} times"
    s = s.replace(OLD, NEW)
    a = "namespace vms {\n"
    assert s.count(a) == 1
    return s.replace(a, DEF + a)


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    print(f"[vaq] {sys.argv[1]}: +{len(apply(t)) - len(t)} B")

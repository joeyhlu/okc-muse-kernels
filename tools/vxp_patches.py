"""vxp_patches.py: two compile-time bs1-class levers on the layer VMS source (moe redesign, R-vxp; ext_codex_bs1.md items 4 / 1').
  VMS_L2H 0: the weight unit copies drop the L2::evict_first cache hint (plain cp.async.bulk; the public TINY build runs its
             ring with the default policy). Default 1 = ship.
  VMS_XPF 1: when unit q + 1 starts a new A job, the compute warps issue prefetch.global.L1 for that job's x rows (each lane two
             of the 8 128-B lines its g row needs; lanes t = 0..3 cover them) before unit q's full-barrier wait, so the switch's
             dependent __ldg x reload hits L1 instead of L2 (~1 RTT under the stream). A-phase x only: phase-B x (inter) is written
             by other CTAs and read with __ldcg after the doneA acquire, so it is never prefetched. Default 0 = ship.
At VMS_L2H 1 / VMS_XPF 0 the source preprocesses to the base's.
    python -I tools/vxp_patches.py <file.py>      (prints the size delta; apply() is for generators)
"""
import pathlib
import sys

OLD_ISSUE = r"""  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(su32(bar)), "r"(UB) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(su32(dst)), "l"(src), "r"(UB), "r"(su32(bar)), "l"(pol) : "memory");
"""
NEW_ISSUE = r"""#if VMS_L2H
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(su32(bar)), "r"(UB) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(su32(dst)), "l"(src), "r"(UB), "r"(su32(bar)), "l"(pol) : "memory");
#else
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(su32(bar)), "r"(UB) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(su32(dst)), "l"(src), "r"(UB), "r"(su32(bar)) : "memory");   // R-vxp: default L2 policy
#endif
"""
OLD_WAIT = "      mbar_wait(&S.full[sl], par);\n"
NEW_WAIT = r"""#if VMS_XPF
      if (!PB && q + 1 < nA && ((a0 + q + 1) >> 9) != j) {   // R-vxp: next A job's x rows -> L1 one unit ahead of the switch
        const int jn = (a0 + q + 1) >> 9;
        if (g < S.jC[jn]) {
          const char* px = reinterpret_cast<const char*>(p.x + (size_t)(S.ent[S.jS[jn] + g] >> 3) * HD) + warp * 1024;
          asm volatile("prefetch.global.L1 [%0];" :: "l"(px + (2 * t) * 128));
          asm volatile("prefetch.global.L1 [%0];" :: "l"(px + (2 * t + 1) * 128));
        }
      }
#endif
      mbar_wait(&S.full[sl], par);
"""
DEF_ANCHOR = "#ifndef VMS_G\n"
DEFS = ("#ifndef VMS_L2H\n#define VMS_L2H 1   // R-vxp: 0 = weight copies without the evict_first hint\n#endif\n"
        "#ifndef VMS_XPF\n#define VMS_XPF 0   // R-vxp: 1 = next-A-job x rows prefetched to L1 one unit ahead\n#endif\n")


def apply(s):
    for old, tag in ((OLD_ISSUE, "issue"), (OLD_WAIT, "full wait"), (DEF_ANCHOR, "defines")):
        assert s.count(old) == 1, f"vxp anchor '{tag}' found {s.count(old)} times"
    s = s.replace(OLD_ISSUE, NEW_ISSUE).replace(OLD_WAIT, NEW_WAIT)
    return s.replace(DEF_ANCHOR, DEFS + DEF_ANCHOR)


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    print(f"[vxp] {sys.argv[1]}: +{len(apply(t)) - len(t)} B, all anchors applied")

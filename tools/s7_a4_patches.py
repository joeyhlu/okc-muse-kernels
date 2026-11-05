"""S7 A4 hardening: patches the output of tools/s6a4s_patches.py (a4s) -> c2.  CPU-only generator.

Why: the free judge check of k3_s6a4s_c1 (job ae3145ac29764ade) failed at test 12
(B=16, ctx 20000, full, page 1, q_len 1) with CUDA_ERROR_ILLEGAL_INSTRUCTION.  Of the 13 judge tests only test 12 is routed through
the a4s kernels (tools/local/s6a4s_check.py: routed == [False]*12 + [True]), and it is the DECODE instantiation (L=1).  The verify
(L=8) instantiation was never launched.  Every instruction form that has no precedent in a judge-proven file is removed:

 1. cp.async.cg with BOTH a src-size operand and an L2::cache_hint operand (the only such use in the repo; 483 other uses carry the
    hint without src-size, HYA's gather carries src-size without a hint; a competitor's cp.async + L2::cache_hint raised an illegal
    instruction on the judge, notes/study/competitor_scan_oct28.md:115).  -> HYA's form: src-size, no hint.
 2. st.global.L1::evict_last.L2::cache_hint.f32 (SASS STG.E.EL, no precedent).  -> K2A's judge-proven
    st.global.L2::cache_hint.b32 with the immediate evict_last-class policy 0x14F0000000000000.
 3. setmaxnreg in the DECODE kernel.  Decode needs no register redistribution: with the role split removed ptxas keeps decode at the
    80-register launch cap with 0 spills (consumer max GPR 78).  Verify keeps its 32/104 split (without it verify spills 32 B).
 4. createpolicy.* is no longer emitted anywhere.
 5. The producer commits its (never committed) ring cp.async before the final wait_group, so no thread exits with copies in flight.

Nothing else changes: same protocol, geometry, host code, selector-off exact-recovery proof, ship pruning.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s6a4s_patches as P  # noqa: E402

ROOT = P.ROOT
LAB = ROOT / "kernels/lab/k3_s7a4_ab.py"
SHIP = ROOT / "kernels/lab/k3_s7a4_ship.py"
CAND = ROOT / "kernels/muse_decode_layer/k3_s6a4s_c2.py"
NONCE_OLD = "# nonce tinys7-c1 ("
NONCE_NEW = ("# nonce s6a4s-c2 (a4s c1 hardened after the test-12 illegal instruction: no cp.async L2 hint, no L1::evict_last store, "
             "no setmaxnreg in decode; segment-per-CTA full attention, verify 2 CTAs/SM, decode 3 CTAs/SM, deterministic merge, "
             "on tinys7 base; exact top-8 of our own FP32 scores; no placement-dependent routing)\n")


def once(text, old, new):
    assert text.count(old) == 1, ("anchor", old[:90], text.count(old))
    return text.replace(old, new, 1)


def patch_cuda(cuda):
    # 1. cp.async: HYA's form (src-size, default cache policy).
    cuda = once(cuda, '''__device__ __forceinline__ uint64_t s6_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void s6_copy(uint32_t dst,const void* src,int valid,uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2,%3;\\n"
      ::"r"(dst),"l"(src),"r"(valid),"l"(pol):"memory");
}
__device__ __forceinline__ uint64_t s6_last() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void s6_keep(float* p,float x,uint64_t pol) {
  asm volatile("st.global.L1::evict_last.L2::cache_hint.f32 [%0],%1,%2;" ::"l"(p),"f"(x),"l"(pol):"memory");
}
''', '''// S7: no cache policy on the K/V gather: HYA's form (cp.async.cg, src-size zero-fill, default policy).
__device__ __forceinline__ uint64_t s6_first() { return 0; }
__device__ __forceinline__ void s6_copy(uint32_t dst,const void* src,int valid,uint64_t) {
  asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\\n"
      ::"r"(dst),"l"(src),"r"(valid):"memory");
}
__device__ __forceinline__ uint64_t s6_last() { return 0; }
// S7: partial / (m,l) stores use K2A's judge-proven form (immediate evict_last-class policy, .b32).
__device__ __forceinline__ void s6_keep(float* p,float x,uint64_t) {
  asm volatile("st.global.L2::cache_hint.b32 [%0],%1,%2;" ::"l"(p),"f"(x),"l"(0x14F0000000000000ull):"memory");
}
''')
    # 3. decode keeps the 80-register launch allocation for every role (it needs <= 78); only verify redistributes.
    cuda = once(cuda, '''    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(S6C<L>::PREG): "memory");
''', '''    if constexpr(L==8) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(S6C<L>::PREG): "memory");
''')
    cuda = once(cuda, '''    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(S6C<L>::CREG): "memory");
''', '''    if constexpr(L==8) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(S6C<L>::CREG): "memory");
''')
    # 5. A producer thread must not leave with un-committed cp.async still in flight (wait_group only covers committed groups, and the
    #    ring's gathers are never committed): commit, then wait for all of this thread's copies (and with them their mbarrier arrives).
    cuda = once(cuda, """  cp_wait<0>();
  // Terminal role; no consumer-side epilogue or full-CTA barrier after return.""",
                """  cp_commit();cp_wait<0>();
  // Terminal role; no consumer-side epilogue or full-CTA barrier after return.""")
    assert "createpolicy" not in cuda and "cp.async.cg.shared.global.L2" not in cuda
    assert "L1::evict_last" not in cuda
    return cuda


def patch_host(host):
    """Judge-toolchain diagnostics + strict verify pool: the judge's ptxas may pick another launch register count than WSL's 13.4,
    and verify's 32/104 role split tiles exactly the 80-register pool (a smaller pool would never satisfy setmaxnreg.inc)."""
    host = once(host, '''  TORCH_CHECK(attr.numRegs<=S6C<L>::RINIT && attr.localSizeBytes==0,"S6A4 requires the designed entry regs and no local memory");
  int occ=0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,k,S6C<L>::NT,sizeof(S6Shared<L>))==cudaSuccess && occ>=S6C<L>::MINB,"S6A4 residency (CTAs per SM) below design");
''', '''  int occ=0;
  const bool occ_ok=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,k,S6C<L>::NT,sizeof(S6Shared<L>))==cudaSuccess;
  fprintf(stderr,"[s6a4s] segment<%d> judge toolchain: regs %d local %d static smem %d dyn smem %d binary %d ptx %d occ %d (design regs %d, CTAs/SM %d)\\n",
    L,attr.numRegs,(int)attr.localSizeBytes,(int)attr.sharedSizeBytes,(int)sizeof(S6Shared<L>),attr.binaryVersion,attr.ptxVersion,occ,S6C<L>::RINIT,S6C<L>::MINB);
  TORCH_CHECK(attr.numRegs<=S6C<L>::RINIT && attr.localSizeBytes==0,"S6A4 requires the designed entry regs and no local memory");
  // Verify redistributes registers (producer 32 / consumer 104): its pool is exactly NT*RINIT, so the launch count must be RINIT.
  if constexpr (L==8) TORCH_CHECK(attr.numRegs==S6C<L>::RINIT,"S6A4 verify launch registers differ from the pool the role split tiles");
  TORCH_CHECK(occ_ok && occ>=S6C<L>::MINB,"S6A4 residency (CTAs per SM) below design");
''')
    assert "setmaxnreg" not in host
    return host


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify the committed outputs are current")
    args = ap.parse_args()
    raw = P.BASE.read_bytes()
    base = raw.decode()
    assert hashlib.sha256(raw).hexdigest() == P.BASE_SHA256, "unreviewed base revision"
    assert "\r" not in base
    P.CUDA = patch_cuda(P.CUDA)
    P.HOST = patch_host(P.HOST)
    lab = P.generate(base)
    assert P.strip_new(lab).encode() == raw, "selector-off exact recovery failed"
    ship = P.ship_form(lab)
    assert NONCE_OLD in ship.split("\n")[1] + "\n" and ship.split("\n")[1].startswith(NONCE_OLD)
    lines = ship.split("\n")
    lines[1] = NONCE_NEW.rstrip("\n")
    cand = "\n".join(lines)
    assert "\r" not in cand
    cb = cand.encode()
    assert len(cb) < 1_000_000, len(cb)
    outs = ((LAB, lab.encode()), (SHIP, ship.encode()), (CAND, cb))
    if args.check:
        for p, b in outs:
            assert p.read_bytes() == b, ("stale", p)
    else:
        for p, b in outs:
            p.write_bytes(b)  # bytes: LF endings on Windows
    print(json.dumps(dict(status="PASS", lab_bytes=len(outs[0][1]), ship_bytes=len(outs[1][1]), cand_bytes=len(cb),
                          selector_off_exact=True, base_sha256=P.BASE_SHA256)))


if __name__ == "__main__":
    main()

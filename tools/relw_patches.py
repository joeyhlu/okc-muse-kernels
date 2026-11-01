"""R-relw-1, lab only: N1 release tokens at the A epilogue, four waiters.

Team assumption A27 (notes/assumptions.md:160): xB stores -> per-thread
fence.proxy.async -> syncwarp -> release.cta arrive -> acquire.cta wait ->
red.release.gpu. No waiter fence. Runner stale-input stress is MANDATORY.
Kill risks remain ring slack and waiter serialization; no speed claim here.

Task D2: four slots, parity (sequence >> 2) & 1; Pend stays EXACTLY base.
128 B barriers + 64 B slots + 64 B seq + 3*16 B counters = 304 B.
MEASURED base smem 212,912 B (box_out/relw_compile.log) + EST 304 B =
213,216 B, below the required 227,328 B ceiling (14,112 B margin).
bf2r inlines only publication; bf2s calls the same body out of line.
Both keep the bounded 64-bit clock poll in a cold noinline helper.
Only lane 0 reads/writes its volatile shared sequence, so no counter is
carried across stages/segments; call ABI/caller-save spills remain a risk.
All four warp leaders gate reuse on ack, so no warp can arrive on an old phase.
Only the WG's warp-0 leader writes its slot. Ack follows the release and slot
read, using CTA release/acquire shared accesses, without a waiter fence.

No tile/stage loops change: 36,864 B weights + 512*npad B x per CTA stage,
and all LDS/dequant/MMA instructions remain identical. Per completed A/WG:
before, each thread 1 proxy fence + 1 WG bar, leader 1 global RED;
after, each thread 1 proxy fence + 1 warp sync, each warp leader 1 mbar arrive,
WG leader 1 shared slot store, each warp leader 1 shared acquire ack load,
1 volatile shared seq load/store, 1 subtract/compare, 1 mask, 1 increment
(plus predicates/address arithmetic; bf2s also call/return). No hot poll.
Publisher scalar smem traffic = 4*(4+4+4)+4 = 52 B/WG publication, plus
four opaque mbarrier arrives; unchanged weight/x/HBM bytes per stage.
Waiter per release: 1 successful try_wait (plus failed polls), 1 slot load,
1 unchanged global RED, 1 shared release ack; failed polls nanosleep(64).
These are source/PTX operations, not measured SASS issue counts.

Protocol/cleanup audit (every launched vm_main CTA, tid 0..575):
* tid 0 initializes all 16 count-4 barriers and 44 int state entries before
  the existing init fence/route synchronization. No barrier is reinitialized.
* WG q owns tid 128*q..128*q+127. Four warp leaders arrive; only tid 128*q
  writes slot/pub/done. Each warp's lane 0 increments its own shared seq.
  Split non-last-arrivers return from seg only (not phase); seq does not move.
  seg's split decision T.wlast[q] is uniform after its original 128-thread
  barrier: either every lane of all four warps calls epilogue or none do.
  Row/store predicates reconverge before publication. VM_AD=0 statically
  excludes m32e_finish/resolve; their guarded source stays unchanged.
  Thus all four warp seq values are equal at completion (not necessarily
  during execution). Finish publishes seq[q][0]; waiter additionally checks
  all four shared seq values equal released before its normal return.
* Warp 17 lanes 1..4 (tid 545..548) independently drain WGs 0..3. Normal exit
  requires acquired done and released==pub; zero A means pub=0/done=1.
  A zero-work CTA still initializes, routes, gathers and finishes all four
  empty A phases; producers have empty loops; waiters take that normal exit.
* Ack polling exits on ack, A.err, or WAIT_BUDGET (writes err=11). Error exits
  break only the poll: no consumer skips subsequent arrivals/phase cleanup.
  Waiter exits on drain, A.err, or timeout (err=12). On error the output is
  invalid and the prepare test rejects it; no successful-drain claim is made.
* Original split resolve remains on its last-arriving CTA/WG, tid 128*q..+127:
  vse_take<N,2/1> clears the middle buffer, <N,0> reads the private head/tail,
  leader resets cnt. Every original B phase/weight/x-ring arrive is retained.
* vm_fin CTA 0, tid 0..255, resets xready[0:80], doneA[0:1280], cnt[0:1088]
  with stride 256; gbar reset is compiled out (VM_DP=0). All token CTAs t<M,
  all 256 threads, still combine/RMS/clear BRED y and store 4096 output columns
  (four float4s/thread at 1024*m+4*tid). No finalizer early exit was added.
* Per warp, all 32 lanes fence then sync BEFORE the leader's release.cta
  arrive; count 4 joins every warp before the waiter's acquire.cta succeeds.
  Under A27 this transitively orders all xB writes before red.release.gpu,
  replacing only the old pend_flush WG barrier, not split-resolve barriers.
* Slot reuse waits on a shared CTA acquire; ack/done use shared CTA release
  stores, not explicit fence instructions. SASS must audit their lowering.
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
OUT = ROOT / "kernels/lab/k3_relw_ab.py"
LIMIT = 1_000_000


def sub(s, old, new, what):
    assert s.count(old) == 1, (what, s.count(old))
    return s.replace(old, new, 1)


def literal(s, name):
    nodes = [n for n in ast.parse(s).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, name
    return ast.literal_eval(nodes[0].value)


LAYOUT = '''#ifndef VM_RELW
#define VM_RELW 0
#endif
#if VM_RELW
struct Relw {
uint64_t bar[4][4];
int rel_slot[4][4], seq[4][4], rel_ack[4], rel_pub[4], rel_done[4];
};
static_assert(sizeof(Relw) == 304, "RELW layout");
constexpr int REL_OFF = BAR_OFF + 8 * 2 * (NS + NX);
constexpr int TAB_OFF = REL_OFF + sizeof(Relw);
#else
constexpr int TAB_OFF = BAR_OFF + 8 * 2 * (NS + NX);
#endif
'''

HELPERS = r'''#if VM_RELW
__device__ __forceinline__ int relw_ld(const int* p) {
int v;
asm volatile("ld.acquire.cta.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(sa(p)) : "memory");
return v;
}
__device__ __forceinline__ void relw_st(int* p, int v) {
asm volatile("st.release.cta.shared.b32 [%0], %1;\n" :: "r"(sa(p)), "r"(v) : "memory");
}
__device__ __noinline__ void relw_wait_ack(const Args& A, const int* ack, int s) {
const long long t0 = clock64();
while (s - relw_ld(ack) >= 4) {
if (*reinterpret_cast<volatile int*>(A.err)) break;
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 11); break; }
__nanosleep(64);
}
}
#if VM_RELW == 2
__device__ __noinline__ void relw_publish_call(const Args& A, int slot, int q, int wp, int lane) {
#else
__device__ __forceinline__ void relw_publish(const Args& A, int slot, int q, int wp, int lane) {
#endif
extern __shared__ __align__(1024) uint8_t smem[];
Relw& r = *reinterpret_cast<Relw*>(smem + REL_OFF);
fence_proxy_async();
__syncwarp();
if (lane == 0) {
const int s = *reinterpret_cast<volatile int*>(&r.seq[q][wp]);
if (s - relw_ld(r.rel_ack + q) >= 4) relw_wait_ack(A, r.rel_ack + q, s);
if (wp == 0) r.rel_slot[q][s & 3] = slot;
mbar_arrive(sa(&r.bar[q][s & 3]));
*reinterpret_cast<volatile int*>(&r.seq[q][wp]) = s + 1;
}
}
__device__ __forceinline__ void relw_finish(int q, int wp, int lane) {
extern __shared__ __align__(1024) uint8_t smem[];
if (wp == 0 && lane == 0) {
Relw& r = *reinterpret_cast<Relw*>(smem + REL_OFF);
r.rel_pub[q] = *reinterpret_cast<volatile int*>(&r.seq[q][0]);
relw_st(r.rel_done + q, 1);
}
}
__device__ __noinline__ void rel_waiter(const Args& A, uint8_t* smem, int q) {
Relw& r = *reinterpret_cast<Relw*>(smem + REL_OFF);
int released = 0;
long long t0 = clock64();
for (;;) {
if (relw_ld(r.rel_done + q) && released == r.rel_pub[q]) {
const volatile int* seq = r.seq[q];
if (seq[0] == released && seq[1] == released && seq[2] == released && seq[3] == released) return;
}
if (*reinterpret_cast<volatile int*>(A.err)) return;
if (mbar_try(sa(&r.bar[q][released & 3]), (released >> 2) & 1)) {
const int slot = r.rel_slot[q][released & 3];
red_release_add(A.doneA + slot, 1);
relw_st(r.rel_ack + q, ++released);
t0 = clock64();
} else {
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 12); return; }
__nanosleep(64);
}
}
}
#endif
'''

GLOBALS = '''RELW_ON = 0
RELW_VAR = "bf2r"
_RELW = {"calls": {"bf2": 0, "bf2r": 0, "bf2s": 0}}
def relw_live(var=None) -> bool:
    var = RELW_VAR if var is None else var
    return bool(var in ("bf2r", "bf2s") and _VML["mods"].get(var) is not None and _VML["ok"].get(32) and _VML["okv"].get((32, var)))
'''


def build(base, report=False):
    assert "VM_RELW" not in base and "RELW_ON" not in base
    defs = literal(base, "_VML_DEFS")
    assert all(f"-D{k}=1" in defs["bf2"] for k in ("VM_DEFER", "VM_SYNC7", "SFX", "VSE_PS", "VM_NOB9"))
    assert not any(x.startswith(("-DVM_WG5=", "-DVM_DP=", "-DVM_AD=", "-DVSE_HOLD=")) for x in defs["bf2"])
    edits = []

    def edit(old, new, what):
        edits.append((old, new, what))

    edit("constexpr int TAB_OFF = BAR_OFF + 8 * 2 * (NS + NX);\n", LAYOUT, "smem release bank")
    anchor = "#ifndef VSE_PS\n#define VSE_PS 0\n#endif\n"
    edit(anchor, anchor + '''#if VM_RELW
static_assert(VM_DEFER == 1 && VM_SYNC7 == 1 && SFX == 1 && VSE_PS == 1 && VM_NOB9 == 1
&& !VM_WG5 && !VM_DP && !VM_AD && !VSE_HOLD && NCW == 16, "RELW requires bf2");
static_assert(VM_RELW == 1 || VM_RELW == 2, "RELW variant");
#endif
''', "bf2 protocol guards")
    cu = literal(base, "_vml_CUDA_T")
    a = cu.index("__device__ __forceinline__ void pend_flush(")
    b = cu.index("#ifndef VCP_NB", a)
    pend = cu[a:b]
    guarded = pend.replace("if (pd.i >= 0) {", "#if VM_RELW\n(void)A; (void)pd; (void)q; (void)wp; (void)lane;\n#else\nif (pd.i >= 0) {", 1)
    assert guarded.endswith("}\n}\n")
    guarded = guarded[:-2] + "#endif\n}\n"
    edit(pend, guarded + HELPERS, "pend no-op and release helpers")
    edit("pd.i = j * 16 + (ub >> 1);", "#if VM_RELW\n#if VM_RELW == 2\nrelw_publish_call(A, j * 16 + (ub >> 1), q, wp, lane);\n#else\nrelw_publish(A, j * 16 + (ub >> 1), q, wp, lane);\n#endif\n#else\npd.i = j * 16 + (ub >> 1);\n#endif", "A epilogue publish")
    anchor = "if constexpr ((VM_DEFER || VM_DEFER2) && !PB) pend_flush(A, pd, q, wp, lane);"
    edit(anchor, "#if VM_RELW\nif constexpr (!PB) relw_finish(q, wp, lane);\n#else\n" + anchor + "\n#endif", "phase end including zero A")
    anchor = "for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }\n"
    edit(anchor, anchor + '''#if VM_RELW
Relw& r = *reinterpret_cast<Relw*>(smem + REL_OFF);
for (int q = 0; q < 4; ++q) {
for (int s = 0; s < 4; ++s) { mbar_init(sa(&r.bar[q][s]), 4); r.rel_slot[q][s] = 0; r.seq[q][s] = 0; }
r.rel_ack[q] = r.rel_pub[q] = r.rel_done[q] = 0;
}
#endif
''', "tid0 initialization")
    anchor = "bar_sync(6, 32 * NCW + 32);\nif (lane == 0) prod_x(A, T, sa(smem + NS * US), bars);\n"
    edit(anchor, anchor + "#if VM_RELW\nelse if (lane <= 4) rel_waiter(A, smem, lane - 1);\n#endif\n", "four producer waiter lanes")
    edit("VML_ENABLE = True\n", GLOBALS + "VML_ENABLE = True\n", "globals and liveness")
    edit("VML_ABVARS = ()\n", 'VML_ABVARS = ("bf2r", "bf2s")\n', "build variants")
    line = next(l for l in base.splitlines(True) if l.startswith('             "bf2":'))
    new_lines = ''.join(line.replace('"bf2":', f'"{v}":').replace('"-DVSE_PS=1"),', f'"-DVSE_PS=1", "-DVM_RELW={i}"),')
                        for i, v in enumerate(("bf2r", "bf2s"), 1))
    edit(line, line + new_lines, "bf2r/bf2s exact defines")
    anchor = '    dp, sc = var in ("d", "d2"), var == "c"\n'
    edit(anchor, '''    if rep == 32 and RELW_ON == 1 and _VML["force"] is None and relw_live():
        var = RELW_VAR
    if var in ("bf2r", "bf2s") and rep != 32:
        var = VML_VAR
    if rep == 32 and var in ("bf2", "bf2r", "bf2s"):
        _RELW["calls"][var] = _RELW["calls"].get(var, 0) + 1
''' + anchor, "capture-time route and calls")
    anchor = '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]\n'
    edit(anchor, anchor.replace('"bf2", "w72"', '"bf2", "bf2r", "bf2s", "w72"'), "identical bf2 scratch ABI")
    anchor = 'def _vml_prepare(st, w, dev):\n    _VML["ok"] = {}\n'
    edit(anchor, anchor + '    _VML["okv"] = {}\n', "invalidate stale variant validation")
    anchor = '    pairs = [(rep, v) for rep in VML_BVAR for v in dict.fromkeys((VML_BVAR[rep],) + tuple(VML_ABVARS))]\n'
    edit(anchor, anchor.rstrip() + '\n    pairs = [(r, v) for r, v in pairs if v not in ("bf2r", "bf2s")] + [(32, "bf2r"), (32, "bf2s")]\n', "self-test pair construction")
    anchor = '                    _VML["force"] = var\n                    o = _vml_route(st, d)\n'
    edit(anchor, '''                    if var in ("bf2r", "bf2s"):
                        _VML["scb"]["xB"].fill_(float("nan"))
''' + anchor, "poison before each variant self-test")
    s = base
    for old, new, what in edits:
        s = sub(s, old, new, what)
    ast.parse(s)
    for i, v in enumerate(("bf2r", "bf2s"), 1):
        assert literal(s, "_VML_DEFS")[v] == defs["bf2"] + (f"-DVM_RELW={i}",)
    patched_cu = literal(s, "_vml_CUDA_T")
    assert "__trap" not in HELPERS
    assert re.findall(r"struct Pend .*", cu) == re.findall(r"struct Pend .*", patched_cu)
    assert "pd.seq" not in patched_cu
    hot = HELPERS[HELPERS.index("#if VM_RELW == 2"):HELPERS.index("__device__ __forceinline__ void relw_finish")]
    assert not re.search(r"clock64|long long|uint64_t|while\s*\(|for\s*\(", hot)
    assert hot.count("relw_ld(") == 1 and hot.count("__syncwarp();") == 1
    assert hot.index("__syncwarp();") < hot.index("if (lane == 0)")
    assert "fence" not in HELPERS[HELPERS.index("__device__ __noinline__ void rel_waiter"):]
    # All four stage/segment sites plus the #else phase-end site survive verbatim.
    assert re.findall(r"^.*pend_flush\(A, pd, q, wp, lane\);.*$", cu, re.M) == re.findall(r"^.*pend_flush\(A, pd, q, wp, lane\);.*$", patched_cu, re.M)
    restored = s
    for old, new, what in reversed(edits):
        restored = sub(restored, new, old, "undo " + what)
    assert restored == base, "not exactly reversible"
    if report:
        for old, new, what in edits:
            assert s.count(new) == 1, what
            first = s[:s.index(new)].count("\n") + 1
            print(f"[relw] {what}: {OUT.name}:{first}-{first + new.count(chr(10))}")
    return s


def main():
    if len(sys.argv) != 1:
        raise SystemExit("usage: python -B -I tools/relw_patches.py (lab only)")
    raw = BASE.read_bytes()
    base = raw.decode("utf-8")
    crlf = b"\r\n" in raw
    assert not crlf or base.count("\n") == base.count("\r\n")
    s = build(base.replace("\r\n", "\n"), report=True)
    out = (s.replace("\n", "\r\n") if crlf else s).encode("utf-8")
    assert len(out) <= LIMIT, f"file exceeds cap by {len(out) - LIMIT} B"
    OUT.write_bytes(out)
    print(f"[relw] {len(out):,} B (+{len(out)-len(raw):,}); headroom {LIMIT-len(out):,} B")
    print("[relw] unique anchors, AST, exact reverse, define equality, retained calls PASS")


if __name__ == "__main__":
    main()

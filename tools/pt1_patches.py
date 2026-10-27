"""pt1_patches.py: PT1 = start the layer's independent load_inline builds in threads at prepare() entry.

Why (box cold prepare of k3_v35, box_out/v35fa/prep_k3_v35_ab.log, 257.8 s): the six CUDA builds run one after the other
inside prepare. They are combined K1 + K2A + R1B + tail 55.4 s, P1 + P1M 41.9 s, G2 42.6 s, KS1 + DG1 43.5 s and SU1 38.3 s:
222 s of serial nvcc. Plus the FA4 v6 CuTe JIT, 7-23 s. Self-tests take only ~25 s, K1 16.8 s of it. SU1 (250 s budget) and
PR206 (235 s budget) are what fall off the end.

What: `_pt1_start()` (first statement of prepare) starts one daemon thread per enabled build: _v10_build_all, p1_build(layer_only=
True), _g2_build, _ks1_build, _su1_build. Each build function first joins its own prefetch thread (`_pt1_wait`, a no-op from
inside that thread) and then hits its existing module cache, so prepare's sequence, self-tests, budgets and fallbacks are
unchanged; only the wall time moves. G2's build gets the same cache the others already have. nvcc runs in torch's own build
subprocesses exactly as before; our code starts no process. A prefetch exception is recorded and ignored: the sequential call
then builds again and keeps today's fallback. `_pt1_join_all()` joins every thread before prepare returns, so nothing runs
after prepare. The CuTe DSL JITs (FA4 v6, PR206) stay in the main thread. PT1_PREFETCH = False is exactly today's behaviour.

Apply BEFORE tools/strip_layer.py (the anchors carry comments): tools/gen_k3_v35.py --pt1.

    python -I tools/pt1_patches.py in.py out.py
"""
import pathlib
import py_compile
import sys

HELPERS = '''PT1_PREFETCH = True   # PT1 (tools/pt1_patches.py): start the independent load_inline builds in threads at prepare entry
_PT1 = {"threads": {}, "errors": {}, "secs": {}, "g2_mod": None}


def _pt1_target(name, fn):
    def run():
        t1 = time.time()
        try:
            fn()
        except Exception as ex:  # noqa: BLE001  (the sequential call builds again and keeps today's fallback)
            _PT1["errors"][name] = repr(ex)[:300]
        _PT1["secs"][name] = time.time() - t1
    return run


def _pt1_start():
    """One daemon thread per enabled CUDA build (nvcc runs in torch's own subprocesses; prepare-time only)."""
    if not PT1_PREFETCH:
        return
    jobs = [("v10", _v10_build_all)] if V10_COMBINED_BUILD else []
    if P1_ENABLE:
        jobs.append(("p1", lambda: p1_build(layer_only=True)))
    if G2_ENABLE:
        jobs.append(("g2", _g2_build))
    if KS1_ENABLE:
        jobs.append(("ks1", _ks1_build))
    if SU1_ENABLE:
        jobs.append(("su1", _su1_build))
    for name, fn in jobs:
        th = threading.Thread(target=_pt1_target(name, fn), name=f"pt1-{name}", daemon=True)
        _PT1["threads"][name] = th
        th.start()
    print(f"[pt1] prefetching builds {[n for n, _ in jobs]}", flush=True)


def _pt1_wait(name):
    """Join this build's prefetch thread (no-op inside that thread or without prefetch)."""
    th = _PT1["threads"].get(name)
    if th is not None and th is not threading.current_thread():
        th.join()


def _pt1_join_all():
    for th in list(_PT1["threads"].values()):
        th.join()
    if _PT1["threads"]:
        secs = " ".join(f"{k}={v:.0f}s" for k, v in _PT1["secs"].items())
        print(f"[pt1] prefetch builds done: {secs}; errors {_PT1['errors'] or 'none'}", flush=True)


'''

PATCHES = [
    ("import threading", "import hashlib\nimport time\nimport types\n",
     "import hashlib\nimport threading\nimport time\nimport types\n"),
    ("helpers + prepare start",
     'def prepare(weights):\n    """r1\'s prepare, then K1\'s self-test (K1 compiled at import), then K2a\'s repack, build and self-test."""\n'
     '    t0 = time.time()\n',
     HELPERS + 'def prepare(weights):\n    """r1\'s prepare, then K1\'s self-test (K1 compiled at import), then K2a\'s repack, build and self-test."""\n'
     '    t0 = time.time()\n    _pt1_start()   # PT1: the independent CUDA builds start now, in threads\n'),
    ("prepare end: join",
     '    occ = "/".join(str(v) for _, v in sorted(_K1_OCC.items(), key=lambda kv: str(kv[0]))) or "-"\n',
     '    _pt1_join_all()   # PT1: no build thread outlives prepare\n'
     '    occ = "/".join(str(v) for _, v in sorted(_K1_OCC.items(), key=lambda kv: str(kv[0]))) or "-"\n'),
    ("v10 wait", "    global _V10_MOD\n    if _V10_MOD is not None:\n        return _V10_MOD\n",
     '    global _V10_MOD\n    _pt1_wait("v10")\n    if _V10_MOD is not None:\n        return _V10_MOD\n'),
    ("p1 wait", "    global _P1_MOD\n    if _P1_MOD is not None:\n        return _P1_MOD\n",
     '    global _P1_MOD\n    _pt1_wait("p1")\n    if _P1_MOD is not None:\n        return _P1_MOD\n'),
    ("g2 wait + cache",
     'def _g2_build():\n    """P1_CUDA_SRC verbatim + the G2-A kernel (T = 1 instantiation only) in one load_inline (layer-only P1 instantiation). Prepare-time only."""\n'
     '    from torch.utils.cpp_extension import load_inline\n',
     'def _g2_build():\n    """P1_CUDA_SRC verbatim + the G2-A kernel (T = 1 instantiation only) in one load_inline (layer-only P1 instantiation). Prepare-time only."""\n'
     '    _pt1_wait("g2")\n    if _PT1["g2_mod"] is not None:   # PT1: cached like the other builds\n        return _PT1["g2_mod"]\n'
     '    from torch.utils.cpp_extension import load_inline\n'),
    ("g2 cache store",
     '    print(f"[g2] compiled in {time.time() - t0:.1f}s", flush=True)\n    return mod\n',
     '    print(f"[g2] compiled in {time.time() - t0:.1f}s", flush=True)\n    _PT1["g2_mod"] = mod\n    return mod\n'),
    ("ks1 wait", '    if _KS1["mod"] is not None:\n        return _KS1["mod"]\n',
     '    _pt1_wait("ks1")\n    if _KS1["mod"] is not None:\n        return _KS1["mod"]\n'),
    ("su1 wait", '    if _SU1["mod"] is not None:\n        return _SU1["mod"]\n',
     '    _pt1_wait("su1")\n    if _SU1["mod"] is not None:\n        return _SU1["mod"]\n'),
]


def apply(text):
    if "PT1_PREFETCH" in text:
        raise RuntimeError("pt1: already applied")
    for what, old, new in PATCHES:
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"pt1 {what}: anchor found {n} times")
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    t = apply(src.read_text(encoding="utf-8"))
    out.write_text(t, encoding="utf-8", newline="\n")
    py_compile.compile(str(out), doraise=True)
    print(f"{out}: {len(t.encode())} B, all {len(PATCHES)} patches hit once, compiles")

"""vms_wsbl_patches.py: swap the layer's VMS kernel source (frozen v1) for the live lab source built with VMS_FL = 512
(WSBL: the CTA's last phase-B unit is reduced and epilogued by compute warp 0, so the last epilogue does not wait behind
the helper's queue; R-vms-7 marks: helper tail 0.84 us). Applied to a stripped layer file on disk (a finish() fixed point,
default k3_vml_5); anchors are code lines of the stripped text.

  * VMS_CUDA_SRC / VMS_CPP_SRC := kernels/lab/vms_moe_lab.py's sources; run() gains the (unused) fl argument.
  * _vms_build builds VMS_FL (main) and, unless VMS_FL_ALT is None, VMS_FL_ALT (0 = the v1-equivalent build) in parallel;
    VMS_USE_ALT (read at call time) selects the alternate: the A/B arms of one lab file.
  * --ship: VMS_FL_ALT = None (one build).

    python -I tools/vms_wsbl_patches.py [--n=3] [--base=k3_vml_5] [--ship] [--check]
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vms_patches  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 3))
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_vml_5')}.py"
SHIP = "--ship" in sys.argv
OUT = ROOT / f"kernels/muse_decode_layer/k3_vms_{N}.py"
LAB = ROOT / f"kernels/lab/k3_vms_{N}_ab.py"
TAG = f"k3-vms-{N}"
NL = chr(10)
Q3 = '"' * 3

BUILD_OLD = '''def _vms_build():
    from torch.utils.cpp_extension import load_inline
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((VMS_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t1 = time.time()
    mod = load_inline(name=f"k3_vms_{tag}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC, functions=["run", "attrs"],
                      extra_cuda_cflags=flags)
    _VMS["G"] = G
    _VMS["mod"] = mod
    _K3_BUILD_S["vms"] = time.time() - t1
    return mod
'''
BUILD_NEW = '''VMS_FL = 512
VMS_FL_ALT = @ALT@
VMS_USE_ALT = False


def _vms_build_fl(G, fl):
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", f"-DVMS_FL={fl}", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((VMS_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    return load_inline(name=f"k3_vms_{tag}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC, functions=["run", "attrs"],
                       extra_cuda_cflags=flags)


def _vms_build():
    G = torch.cuda.get_device_properties(0).multi_processor_count
    t1 = time.time()
    alt = {}
    th = None
    if VMS_FL_ALT is not None:
        th = threading.Thread(target=lambda: alt.setdefault("mod", _vms_build_fl(G, VMS_FL_ALT)), daemon=True)
        th.start()
    mod = _vms_build_fl(G, VMS_FL)
    if th is not None:
        th.join()
    _VMS["G"] = G
    _VMS["mod"] = mod
    _VMS["mod_alt"] = alt.get("mod")
    _K3_BUILD_S["vms"] = time.time() - t1
    return mod
'''
CALL_OLD = '''    vs["mod"].run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],
                  out, vs["nomarks"], m, K2A_POST_NORM_EPS, 1, 0)'''
CALL_NEW = '''    md = vs["mod_alt"] if (VMS_USE_ALT and vs.get("mod_alt") is not None) else vs["mod"]
    md.run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],
           out, vs["nomarks"], m, K2A_POST_NORM_EPS, 1, 0, VMS_FL)'''
PREP_OLD = "        good = True\n        for bs, seed, dist in VMS_SELF_TEST_CASES:\n"
PREP_NEW = "        st[\"vms\"][\"mod_alt\"] = _VMS.get(\"mod_alt\")\n" + PREP_OLD


def _swap_string(text, name, new):
    head = f"{name} = r{Q3}"
    i = text.index(head)
    assert text.count(head) == 1, name
    j = text.index(Q3, i + len(head))
    return text[:i] + head + new + text[j:]


def apply(text, ship=False):
    cuda, cpp = vms_patches._lab_sources()
    assert "#define VMS_WSBL" in cuda, "live lab without WSBL"
    assert "VMS_USE_ALT" not in text and "_vms_build_fl" not in text
    text = _swap_string(text, "VMS_CUDA_SRC", cuda)
    text = _swap_string(text, "VMS_CPP_SRC", cpp)
    for what, old, new in (("build", BUILD_OLD, BUILD_NEW.replace("@ALT@", "None" if ship else "0")),
                           ("call", CALL_OLD, CALL_NEW), ("prepare", PREP_OLD, PREP_NEW)):
        n = text.count(old)
        assert n == 1, f"vms_wsbl anchor {what!r} found {n} times"
        text = text.replace(old, new, 1)
    assert "import threading" in text
    return text


def main():
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    GM.check(fb == b, f"{BASE.name} is not a fixed point of finish()")
    rep = [f"base {BASE.name} {len(b.encode())} B is a finish() fixed point"]
    nonce = (f"# nonce {TAG} ({BASE.stem} + VMS lab kernel with WSBL (last B unit's epilogue on compute warp 0)"
             f"{'' if SHIP else ' + v1-equivalent alternate build for A/B'}, self-tested vs K2A with fallback)")
    s = GM.finish(apply(b.replace(bn, nonce), ship=SHIP), nonce, TAG, chk_base, rep)
    rep.append(f"size: {len(b.encode())} -> {len(s.encode())} B ({len(s.encode()) - len(b.encode()):+d} B, headroom "
               f"{1_000_000 - len(s.encode())} B)")
    for r in rep:
        print(f"[vms_wsbl] OK  {r}")
    if "--check" in sys.argv:
        GM.check(OUT.exists() and OUT.read_text(encoding="utf-8") == s, f"{OUT.name} on disk differs")
        print(f"[vms_wsbl] --check: {OUT.name} up to date")
        return 0
    OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vms_wsbl] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""gen_hhx_board.py (R-dense-probe): kernels/muse_bf16_proj/hhx_<n>.py = vd_1 + the layer's dense engines vd_1 never had.

  python -I tools/gen_hhx_board.py [--n=1] [--base=kernels/muse_bf16_proj/vd_1.py] [--layer=kernels/lab/k3_hh_3tkv.py] [--check]

The board file vd_1 (Oct 28, check 4e47053a = 1.4275, never scored) already carries the board form of the layer's GD engine (its
autotune picks the same geometries as the layer's GD_ROUTE), the cuBLAS / K-major / VD routes at m >= 128 and the router. What
the layer has since and the board never saw:
  * DG1 (kernels/lab/k3_hh_3tkv.py DG1_*): the BF16 bulk-copy + wgmma skinny GEMM, used by the layer for qkv and o at T = 32;
  * the dpin cuBLASLt pins (LT_PIN / ULP_LAYOUT: algorithm 66, explicit tile / custom option / cluster words, NN or batched
    layouts through lt_plan_b / lt_make) for qkv 128 / 256, o 128, unpooler 256 (+ two candidates the layer parked: o 256,
    unpooler 128, from the public Apache-2.0 file of alygg77).
Both enter as EXTRA CANDIDATES of vd_1's own search (pins inside _lt_search next to the heuristic algorithms, torchT, addmm and
VD; DG1 in a phase after it), verified with the judge metric and timed cold-L2 in a CUDA graph; nothing is taken on trust.
Source text of the engines is copied verbatim from the layer file by AST segments; vd_1's own text is patched at anchors that
must match exactly once (the run fails loudly otherwise).
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000
# layer-file names that become the board's proj names
PROJ = {"qkv": "qkv_gate", "o": "o", "unpooler": "unpooler"}
# candidates the layer parked or never needed (alygg77's public pins, decoded in notes/study/unpooler_lt.md)
EXTRA_PINS = {("o", 256): [((66, 20, 35, 1, 0, 0, 6, 0, 5), 1, 2, "alyo256")],
              ("unpooler", 128): [((66, 18, 35, 1, 0, 0, 3, 0, 5), 1, 2, "alyun128")]}


def check(c, msg):
    if not c:
        raise SystemExit(f"[gen_hhx_board] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


def seg(src, node):
    return ast.get_source_segment(src, node)


def layer_pieces(src):
    tree = ast.parse(src)
    want_assign = {"DG1_CPP_SRC", "DG1_CUDA_SRC", "DG1_INSTS", "LT_CPP", "LT_PIN", "ULP_LAYOUT"}
    want_def = {"dg1_wg_spec", "dg1_cuda_src", "dg1_pack"}
    got = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in want_assign:
            check(node.targets[0].id not in got, f"{node.targets[0].id} assigned twice in the layer file")
            got[node.targets[0].id] = node
        elif isinstance(node, ast.FunctionDef) and node.name in want_def:
            got[node.name] = node
    for k in want_assign | want_def:
        check(k in got, f"layer file lacks {k}")
    return got


def build_block(layer_src):
    got = layer_pieces(layer_src)
    out = []
    for name in ("DG1_CPP_SRC", "DG1_CUDA_SRC", "DG1_INSTS"):
        out.append(seg(layer_src, got[name]))
    for name in ("dg1_wg_spec", "dg1_cuda_src", "dg1_pack"):
        out.append(seg(layer_src, got[name]))
    lt = seg(layer_src, got["LT_CPP"])
    check(lt.startswith("LT_CPP = "), "LT_CPP segment shape")
    out.append("HHX_" + lt)
    pins_l = ast.literal_eval(got["LT_PIN"].value)
    ulp = ast.literal_eval(got["ULP_LAYOUT"].value)
    pins = {}
    for (proj, t), cfg in sorted(pins_l.items()):
        if proj not in PROJ or t not in (128, 256):
            continue   # 'cat' (pooler + router, FP32 out) has no board shape
        nn, bt = ulp.get((proj, t), (0, 1))
        pins.setdefault((PROJ[proj], t), []).append((tuple(cfg), nn, bt, f"dpin{t}"))
    for (proj, t), lst in EXTRA_PINS.items():
        pins.setdefault((PROJ[proj], t), []).extend(lst)
    glue = (ROOT / "tools/hhx_glue.txt").read_bytes().decode("ascii")
    glue = sub(glue, "@HHX_PINS@", repr(dict(sorted(pins.items()))), "pins placeholder")
    return NL.join(out) + NL + glue, pins


ROUTE_DG1 = ('    if r["kind"] == "dg1":\n'
             '        out = torch.empty((x.shape[0], n_out), dtype=torch.bfloat16, device=x.device)\n'
             '        r["mod"].dg1_run(r["kid"], x, r["w"], out, r["err"], r["S"], 0, 0, n_out, 0)\n'
             '        return out\n')
PIN_LOOP = ('            for r in _hhx_pin_cands(state, layer, m, w, x, probes, ref, dev):\n'
            '                cands.append(r)\n')
DG1_CALL = ('    try:\n'
            '        summary += _hhx_dg1_phase(state, timer, floor, dev)\n'
            '    except Exception as ex:  # noqa: BLE001\n'
            '        print(f"[hhx] dg1 phase raised {ex!r}", flush=True)\n')


def apply(base_src, layer_src, base_name):
    block, pins = build_block(layer_src)
    s = base_src
    lines = s.split(NL)
    check(lines[0].startswith("# okc: problem=muse_bf16_proj"), "base first line")
    lines[1] = (f"# nonce hhx-1 ({base_name} + the layer's DG1 at m = 32 and the dpin cuBLASLt pins at m = 128 / 256 as extra search candidates; "
                f"tools/gen_hhx_board.py)")
    s = NL.join(lines)
    s = sub(s, "\n\ndef prepare(weights):\n", NL + block + NL + NL + "def prepare(weights):" + NL, "prepare anchor")
    s = sub(s, '    if r["kind"] == "mmT":\n        return torch.mm(x, r["wt"])\n', ROUTE_DG1 + '    if r["kind"] == "mmT":\n        return torch.mm(x, r["wt"])\n',
            "_route_call mmT")
    s = sub(s, "    if r is not None and x.is_contiguous() and x.data_ptr() % 16 == 0:\n        return _route_call(r, x, n, layer)\n",
            '    if r is not None and x.is_contiguous() and x.data_ptr() % r.get("align", 16) == 0:\n        return _route_call(r, x, n, layer)\n',
            "kernel_fn alignment")
    s = sub(s, '            gvr = state["routes"].get((layer, _bucket(layer, m)))\n', PIN_LOOP + '            gvr = state["routes"].get((layer, _bucket(layer, m)))\n',
            "_lt_search pins")
    s = sub(s, '    if VD_ENABLE and VD_BOARD and _VD["mod"] is None and torch.cuda.get_device_capability(dev)[0] >= 9:\n',
            '    _hhx_start(dev)\n    if VD_ENABLE and VD_BOARD and _VD["mod"] is None and torch.cuda.get_device_capability(dev)[0] >= 9:\n', "prepare hhx start")
    s = sub(s, "    # free packs that no route uses\n", DG1_CALL + "    # free packs that no route uses\n", "prepare dg1 phase")
    return s, pins


def main():
    n = ARGS.get("n", "1")
    base = ROOT / ARGS.get("base", "kernels/muse_bf16_proj/vd_1.py")
    layer = ROOT / ARGS.get("layer", "kernels/lab/k3_hh_3tkv.py")
    out = ROOT / f"kernels/muse_bf16_proj/hhx_{n}.py"
    s, pins = apply(base.read_bytes().decode("ascii"), layer.read_bytes().decode("ascii"), base.stem)
    ast.parse(s)
    check(s.isascii(), "non-ASCII output")
    check(len(s.encode()) < LIMIT, f"size {len(s.encode()):,} >= {LIMIT:,}")
    if "check" in FLAGS:
        check(out.read_bytes().decode("ascii") == s, f"{out.name} != generated")
        print(f"[gen_hhx_board] --check PASS: {out.name} ({len(s):,} B)")
        return
    out.write_bytes(s.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"[gen_hhx_board] {base.name} + {layer.name} -> {out.name} ({len(s):,} B, {len(s) - base.stat().st_size:+,} B)")
    for k, v in sorted(pins.items()):
        print(f"   pins {k}: {v}")


if __name__ == "__main__":
    main()

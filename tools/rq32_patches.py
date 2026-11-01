#!/usr/bin/env python3
"""RQ32 lab generator; exact anchors on k3_diet_rq128, CPU-only by default.

python -B tools/rq32_patches.py          # writes only the named lab file
python -B tools/rq32_patches.py --check  # read-only gates and structural dry run
python -B tools/rq32_patches.py --check --torch-cpu  # real CPU tensor gates
At capture, set RQ_SEL to the default union any subset of NEW_KEYS. Prepare once.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path
import runpy
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_diet_rq128.py"
OUT = ROOT / "kernels/lab/k3_rq32_ab.py"
VENDOR = ROOT / "vendor_ext/chenwainuo/submission.py"
DEFAULT = frozenset({("qkv", 128), ("unpooler", 128)})
NEW_KEYS = (("qkv", 32), ("o", 32), ("unpooler", 32))
VARIANTS = [("t32s6hzw", 32, 6, 1, 0, 0, 0, 9437187),
            ("h32s5hze2", 32, 5, 1, 0, 0, 0, 1075)]
EDITS = [
    ("notice", "Modifications: rq-prefixed names, two fixed instances, selector and guarded\n",
     "Modifications: rq-prefixed names, fixed T128/T256 and RQ32 instances, selector and guarded\n"),
    ("pack_all", 'RQ_PACK=(("qkv",128),("unpooler",128))\n',
     'RQ_PACK=(("qkv",128),("unpooler",128),("qkv",32),("o",32),("unpooler",32))\n'),
    ("per_projection_picks", '_rq_picks={128:(0,"d2s5hzwP"),256:(1,"n256s5m1hzmEH")}\n',
     '_rq_picks={128:(0,"d2s5hzwP"),256:(1,"n256s5m1hzmEH"),\n'
     ' ("qkv",32):(2,"t32s6hzw-k7"),("unpooler",32):(2,"t32s6hzw-k2"),("o",32):(3,"h32s5hze2-k2")}\n'),
    ("keys_existing_first", ' return tuple(sorted((rq_p,rq_t)for rq_p,rq_t in RQ_PACK\n'
     ' if rq_p in _rq_ks and rq_t in _rq_picks))\n',
     ' return tuple(sorted({(rq_p,rq_t)for rq_p,rq_t in RQ_PACK\n'
     ' if rq_p in _rq_ks and ((rq_p,rq_t)in _rq_picks or rq_t in _rq_picks)},\n'
     ' key=lambda k:(k not in (("qkv",128),("unpooler",128)),k)))\n'),
    ("pick_lookup", '   rq_i,rq_name=_rq_picks[rq_t]\n',
     '   rq_i,rq_name=_rq_picks.get(rq_key,_rq_picks.get(rq_t))\n'),
    ("half_residency", '   if rq_occ<1 or(rq_n//128)*rq_ks>rq_sms*rq_occ:\n',
     '   rq_grid=(rq_n//128)*rq_ks*(2 if _rq_variants[rq_i][7]&16 else 1)\n'
     '   if rq_occ<1 or rq_grid>rq_sms*rq_occ:\n'),
    ("instances", "_rq_variants=[('d2s5hzwP',128,5,1,0,0,0,42999939),('n256s5m1hzmEH',256,5,1,0,1,0,65603)]\n",
     "_rq_variants=[('d2s5hzwP',128,5,1,0,0,0,42999939),('n256s5m1hzmEH',256,5,1,0,1,0,65603),\n"
     " ('t32s6hzw',32,6,1,0,0,0,9437187),('h32s5hze2',32,5,1,0,0,0,1075)]\n"),
]


def namespace(text):
    """Only rq definitions; importing the submission would initialize CUDA stacks."""
    nodes = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_rq"):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and
                (t.id.startswith("_rq") or t.id in ("RQ_SEL", "RQ_PACK")) for t in node.targets):
            nodes.append(node)
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "rq32-extracted", "exec"), ns)
    return ns


def node_named(text, name):
    return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)


def generate():
    base = BASE.read_text(encoding="utf-8")
    text, changes = base, []
    for name, old, new in EDITS:
        # Same unique-anchor discipline as tools/padz_patches.py:48-51.
        assert text.count(old) == 1, (name, "anchor count", text.count(old))
        text = text.replace(old, new, 1)
    undo = text
    for name, old, new in reversed(EDITS):
        assert undo.count(new) == 1, (name, "reverse anchor")
        undo = undo.replace(new, old, 1)
    assert undo == base, "unscoped edit"
    compile(text, str(OUT), "exec")  # no pyc write
    a, b = namespace(base), namespace(text)
    assert b["RQ_SEL"] == a["RQ_SEL"] == DEFAULT
    assert set(b["RQ_PACK"]) == DEFAULT | set(NEW_KEYS)
    assert b["_rq_keys"]()[:2] == a["_rq_keys"]()
    assert b["_rq_variants"] == a["_rq_variants"] + VARIANTS
    # No CUDA/protocol/mainloop/validation/dispatch edits, including all fallbacks.
    assert b["_rq_cuda"] == a["_rq_cuda"] and b["_rq_flags"] == a["_rq_flags"]
    for name in ("kernel_fn", "_rq_weight", "_rq_call", "_rq_pack_weight", "_rq_validate", "_rq_gemm"):
        assert ast.dump(node_named(base, name)) == ast.dump(node_named(text, name)), name
    vendor = ast.parse(VENDOR.read_text(encoding="utf-8"))
    tables = {n.targets[0].id: ast.literal_eval(n.value) for n in vendor.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ("RQV", "RQPICK32", "RQPICK32B")}
    assert all(v in tables["RQV"] for v in VARIANTS)
    for proj, t in NEW_KEYS:
        key = ("qkv_gate" if proj == "qkv" else proj, t)
        expected = tables["RQPICK32B" if proj == "o" else "RQPICK32"][key]
        idx, pick = b["_rq_picks"][(proj, t)]
        assert pick == expected == b["_rq_variants"][idx][0] + "-k%d" % b["_rq_ks"][proj]
    src = b["_rqsrc"](b["_rq_variants"])
    assert "@@" not in src and "trap;" not in src
    assert "Apache" in b["_rq_notice"] and "chenwainuo" in b["_rq_notice"]
    for name, old, new in EDITS:
        changes.append({"bit": name, "base_line": base[:base.index(old)].count("\n") + 1,
                        "lab_line": text[:text.index(new)].count("\n") + 1,
                        "old_lines": old.count("\n"), "new_lines": new.count("\n")})
    return text, changes


def dry_run(text):
    """Execute actual prepare/key/bank/capture paths against the real dense layout.

    CUDA/packing/numerical leaves are explicit doubles; the validation leaf is
    exercised separately with the established rq NumPy/torch CPU harness.
    """
    support = runpy.run_path(str(ROOT / "tools/rq_patches.py"))
    Tensor = support["_rq_FakeTensor"]
    ns = namespace(text)
    base = BASE.read_text(encoding="utf-8")
    assert ast.dump(node_named(base, "_dg1_weight").body[-1]) == ast.dump(node_named(text, "_rq_weight").body[-1])
    prep = node_named(base, "_r1_prepare")
    state_expr = next(n.value for n in prep.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "state" for t in n.targets))
    dense_expr = next(v for k, v in zip(state_expr.keys, state_expr.values) if k.value == "dense")
    shapes = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
    weights = {k: Tensor(v) for k, v in shapes.items()}
    dense = eval(compile(ast.Expression(dense_expr), "real-base-dense", "eval"), {"weights": weights})
    events = []
    occupancy = [2]
    def launch(*args):
        if args[-1] == 4:
            return occupancy[0]
        events.append(("launch", args[0], args[2].shape[0], args[7]))
        return 0
    module = types.SimpleNamespace(rq_run=launch)
    ns["torch"] = types.SimpleNamespace(
        bfloat16="bf16", int32="i32", int64="i64",
        empty=lambda shape, dtype, device: Tensor(shape, dtype, device),
        zeros=lambda shape, dtype, device: Tensor(shape, dtype, device),
        Generator=lambda **kw: types.SimpleNamespace(manual_seed=lambda s: s),
        cuda=types.SimpleNamespace(get_device_properties=lambda d: types.SimpleNamespace(multi_processor_count=132)))
    ns["_rq_build"] = lambda: (events.append(("build",)) or module)
    ns["_rq_pack_weight"] = lambda w: (events.append(("pack", w.shape)) or
        (Tensor((w.shape[0] // 128, w.shape[1] // 64, 8, 1536), "u8"), 42, []))
    ns["_rqesc"] = lambda e, nb, kc, ks, dev: (Tensor((ks * nb * 8 * 32 + 32,), "i32"), None)
    ns["_rqesc2"] = lambda e, nb, kc, ks, dev: (Tensor((ks * nb * 8 * 128,), "i32"), 0)
    ns["_rq_validate"] = lambda p, w, t, g: (events.append(("validate", p["rq_i"], t)) or (True, "numeric leaf double"))
    cases, dispatch = 0, 0
    expected_keys = DEFAULT | set(NEW_KEYS)
    def prepare():
        state = {"dense": dense}
        ns["_rq_prepare"](state)
        return state
    with contextlib.redirect_stdout(io.StringIO()):
        # RQ_PACK, not RQ_SEL, determines the bank. Test default and empty selection.
        for selection in (DEFAULT, frozenset()):
            events.clear()
            ns["RQ_SEL"] = selection
            state = prepare()
            assert set(state["rq_bank"]) == expected_keys
            assert [e for e in events if e[0] == "pack"] == [
                ("pack", shapes["qkv_gate"]), ("pack", shapes["unpooler"]), ("pack", shapes["o"])]
            assert len([e for e in events if e[0] == "build"]) == 1
        # One bank, all eight combinations; changing arms allocates no new bank.
        state_id = id(state["rq_bank"])
        for bits in itertools.product((False, True), repeat=3):
            arm = DEFAULT | {k for k, on in zip(NEW_KEYS, bits) if on}
            ns["RQ_SEL"] = arm
            events.clear()
            for proj in ("qkv", "o", "unpooler", "pooler"):
                for t in (1, 8, 16, 32, 128, 256, 17, 33):
                    k = "qkv_gate" if proj == "qkv" else proj
                    out = ns["_rq_gemm"](state, proj, Tensor((t, shapes[k][1])))
                    assert (out is not None) == ((proj, t) in arm), (arm, proj, t)
                    dispatch += 1
            assert all(e[0] == "launch" for e in events)
            assert id(state["rq_bank"]) == state_id
            cases += 1
        # Separate counters per key, shared raw pack per projection.
        bank = state["rq_bank"]
        assert len({id(p["rq_cnt"]) for p in bank.values()}) == len(bank)
        for proj in ("qkv", "unpooler"):
            assert bank[(proj, 32)]["rq_lay"] is bank[(proj, 128)]["rq_lay"]
        for key, p in bank.items():
            idx, _ = ns["_rq_picks"].get(key, ns["_rq_picks"].get(key[1]))
            grid_x = p["rq_n"] // 128 * (2 if ns["_rq_variants"][idx][7] & 16 else 1)
            assert p["rq_cnt"].numel() >= 2 * grid_x
        # HALF grid is 256, not 128: one CTA/SM must reject o32 only.
        occupancy[0] = 1
        assert set(prepare()["rq_bank"]) == expected_keys - {("o", 32)}
        occupancy[0] = 2
        faults = [("_rq_build", lambda: (_ for _ in ()).throw(RuntimeError("build fail"))),
                  ("_rq_pack_weight", lambda w: (_ for _ in ()).throw(ValueError("pack fail"))),
                  ("_rq_validate", lambda *a: (False, "bad numeric output")),
                  ("_rq_budget_s", 0), ("_rq_call", lambda *a: -1)]
        for name, fault in faults:
            saved = ns[name]
            ns[name] = fault
            assert not prepare()["rq_bank"], name
            ns[name] = saved
        saved = ns["_rqesc2"]
        ns["_rqesc2"] = lambda *a: (None, 5)
        assert set(prepare()["rq_bank"]) == {("o", 32)}
        ns["_rqesc2"] = saved
        saved = ns["_rqesc"]
        ns["_rqesc"] = lambda *a: (Tensor((1 << 30,), "i32"), None)
        assert set(prepare()["rq_bank"]) == expected_keys - {("o", 32)}
        ns["_rqesc"] = saved
        saved = dense.pop("qkv_gate")
        assert set(prepare()["rq_bank"]) == expected_keys - {("qkv", 128), ("qkv", 32)}
        dense["qkv_gate"] = saved
        ns["RQ_PACK"] = (("qkv", 16), ("pooler", 32), ("router", 32), ("qkv", 33))
        assert ns["_rq_keys"]() == () and "rq_bank" not in prepare()
        ns["RQ_SEL"] = {("o", 32)}
        for x in (Tensor((32, 1024), "f32"), Tensor((32, 4096)), Tensor((32, 1024), device="bad")):
            assert ns["_rq_gemm"](state, "o", x) is None
        saved = ns["_rq_call"]
        ns["_rq_call"] = lambda *a: -1
        assert ns["_rq_gemm"](state, "o", Tensor((32, 1024))) is None
        ns["_rq_call"] = saved
        state["rq_bank"].pop(("o", 32))
        assert ns["_rq_gemm"](state, "o", Tensor((32, 1024))) is None
    return {"status": "PASS", "dense_state_line": dense_expr.lineno, "arms": cases,
            "dispatch_checks": dispatch, "half_grid_admission": "PASS", "fallbacks": "PASS",
            "protocol_coverage": protocol_coverage(namespace(text)),
            "leaves": "CUDA and pack mocked; NumPy validation gate runs separately"}


def protocol_coverage(ns):
    """CPU index proof for newly activated FULL/HALF zero and stmatrix loops."""
    for key in NEW_KEYS:
        idx = ns["_rq_picks"][key][0]
        fl = ns["_rq_variants"][idx][7]
        cw, nw, pw, threads = (64, 4, 4, 256) if fl & 16 else (128, 8, 8, 384)
        ks, t = ns["_rq_ks"][key[0]], key[1]
        assert (pw + 4) * 32 == threads and nw * 32 == pw * 32
        # Every split owns a disjoint token slice; all 64 zeroing lanes remain.
        seen = []
        for split in range(ks):
            r0, r1 = split * t // ks, (split + 1) * t // ks
            for lane in range(64):
                for e in range(lane, (r1 - r0) * (cw // 8), 64):
                    seen.append((r0 + e // (cw // 8), (e % (cw // 8)) * 8))
        assert len(seen) == len(set(seen)) == t * cw // 8
        assert set(seen) == {(r, c) for r in range(t) for c in range(0, cw, 8)}
        # 128 threads per live WG cover every 16-byte vector in its SW128 slab.
        addresses = []
        for w in range(4):
            for lane in range(32):
                j, rr = lane >> 3, lane & 7
                for pp in range(t // 16):
                    i, ch = 2 * pp + (j >> 1), 2 * w + (j & 1)
                    addresses.append((8 * i + rr) * 128 + ((ch ^ rr) << 4))
        assert len(addresses) == len(set(addresses)) == t * 128 // 16
        assert set(addresses) == set(range(0, t * 128, 16))
    return "PASS (zero-slice partition and SW128 slab indices; no CUDA executed)"


def leaf_gates(text, real_torch=False):
    """Reuse proven rq harness verbatim, changing only the tested T arms."""
    support_text = (ROOT / "tools/rq_patches.py").read_text(encoding="utf-8")
    support = runpy.run_path(str(ROOT / "tools/rq_patches.py"))
    support["_rq_namespace"] = namespace
    name = "_rq_torch_cpu" if real_torch else "_rq_leaf_dry_run"
    node = node_named(support_text, name)
    # Only (128,256) test-arm literals change. Production validation stays identical.
    class Arms(ast.NodeTransformer):
        def visit_Tuple(self, n):
            if all(isinstance(x, ast.Constant) for x in n.elts) and [x.value for x in n.elts] == [128, 256]:
                return ast.copy_location(ast.Tuple(elts=[ast.Constant(32)], ctx=ast.Load()), n)
            return self.generic_visit(n)
    node = ast.fix_missing_locations(Arms().visit(node))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "rq32-leaf-gate", "exec"), support)
    result = support[name](text)
    if real_torch:
        result["validation_T"] = [32]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--torch-cpu", action="store_true")
    args = parser.parse_args()
    text, changes = generate()
    result = {"status": "PASS", "reverse_gate": "PASS", "CUDA_unchanged": "PASS",
              "dry_run": dry_run(text), "numpy_leaves": leaf_gates(text), "changes": changes,
              "base_bytes": len(BASE.read_bytes()), "lab_bytes": len(text.encode()),
              "sha256": hashlib.sha256(text.encode()).hexdigest()}
    if args.torch_cpu:
        result["torch_cpu"] = leaf_gates(text, real_torch=True)
    if args.check:
        assert OUT.read_bytes() == text.encode(), "stale lab file"
    else:
        OUT.write_bytes(text.encode())  # always LF, never writes a pyc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

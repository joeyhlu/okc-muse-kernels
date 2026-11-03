"""Apply evo_search winners to a layer file as plain source literals (no runtime overrides).

usage: python tools/evo_apply.py <base.py> <out.py> <space.json> <T.json>[+<T.json>...] [--only=T1,T2] [--drop=GENE_ID,...]

For every T.json whose final arm is declared (or listed in --only), the winner's sparse set is rebuilt with evo_search's own
Gene / build_settings, starting from the base file's literal globals (ast.literal_eval of top-level assignments). Per-T dict
entries compose across T; a global-scope scalar wanted with different values by two T classes is a CONFLICT (kept at base,
reported). Each changed top-level assignment `NAME=...` is replaced by `NAME=<repr>`; nothing else in the file changes.
RP_* / RPW_* / HYA_Q2* are refused by evo_search's Gene class.
"""
import ast, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "box"))
import evo_search as E  # noqa: E402


DEL = object()


def arg(name, default=None):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def keyfix(v):   # same decoding as tools/box/layer_ab3.py keyfix (JSON arm -> the layer's Python globals)
    if isinstance(v, dict) and any(isinstance(k, str) and "," in k for k in v):
        return {tuple(int(p) if p.isdigit() else (p == "True") if p in ("True", "False") else p for p in k.split(",")): keyfix(x)
                for k, x in v.items()}
    if isinstance(v, dict):
        return {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): keyfix(x) for k, x in v.items()}
    if isinstance(v, list):
        return tuple(keyfix(x) for x in v)
    return v


def top_assigns(src):
    tree = ast.parse(src)
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out[n.targets[0].id] = n
    return out


def main():
    base, dst, space = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    tfiles = [Path(p) for p in sys.argv[4].split("+")]
    only = {int(t) for t in arg("only", "").split(",") if t}
    drop = {d for d in arg("drop", "").split(",") if d}
    src = base.read_text(encoding="utf-8")
    nodes = top_assigns(src)
    lit = {}

    def getg(name):
        if name not in lit:
            if name not in nodes:
                raise SystemExit(f"[apply] {name}: no top-level literal assignment in {base}")
            seg = ast.get_source_segment(src, nodes[name].value)
            try:
                lit[name] = ast.literal_eval(seg)
            except ValueError:   # e.g. HYA_TUNED uses dict(...) calls: evaluate with only `dict` available
                lit[name] = eval(compile(seg, name, "eval"), {"__builtins__": {}, "dict": dict})
        return lit[name]

    merged = {}
    fv = arg("from-variants")   # <variants.json>@<arm name>: write that layer_ab3 arm's globals instead of T.json winners
    if fv:
        vpath, vname = fv.rsplit("@", 1)
        arm = next(x["set"] for x in json.loads(Path(vpath).read_text(encoding="utf-8")) if x["name"] == vname)
        for name, val in arm.items():
            getg(name)
            merged[name] = (keyfix(val), f"variant {vname}")
        tfiles = []
    for tf in tfiles:
        d = json.loads(tf.read_text(encoding="utf-8"))
        T = d["T"]
        fin = d.get("final") or {}
        arms = fin.get("arms") or []
        if not arms:
            print(f"[apply] T={T}: no final arm, skipped")
            continue
        if only and T not in only:
            print(f"[apply] T={T}: not in --only, skipped")
            continue
        if not only and not fin.get("declared"):
            print(f"[apply] T={T}: final not declared ({fin.get('why', '')}), skipped")
            continue
        sp = {k: v for k, v in arms[0]["set"].items() if k not in drop}
        genes = E.load_space(space, T)
        idx = {g.id: g for g in genes}
        geno = []
        for g in genes:
            vi = 0
            if g.id in sp:
                vi = next((i for i, x in enumerate(g.values) if E.norm(x) == E.norm(sp[g.id])), None)
                if vi is None:
                    raise SystemExit(f"[apply] T={T}: value {sp[g.id]!r} not in gene {g.id}")
            geno.append(vi)
        missing = [k for k in sp if k not in idx]
        if missing:
            raise SystemExit(f"[apply] T={T}: genes not in space: {missing}")
        st = E.build_settings(genes, tuple(geno), getg)
        print(f"[apply] T={T}: {len(sp)} gene-values -> globals {sorted(st)}")
        for name, val in st.items():
            b = getg(name)
            if isinstance(val, dict) and isinstance(b, dict):
                # per-key deltas vs base compose across T; the same key changed differently by two T's -> base
                deltas = merged.setdefault(name, {})
                for k in set(val) | set(b):
                    nv = val.get(k, DEL)
                    if nv is not DEL and k in b and nv == b[k]:
                        continue
                    if nv is DEL and k not in b:
                        continue
                    if k in deltas and deltas[k][0] is not None and (deltas[k][0] is DEL) != (nv is DEL) or \
                            (k in deltas and deltas[k][0] is not None and nv is not DEL and deltas[k][0] is not DEL
                             and deltas[k][0] != nv):
                        print(f"[apply] CONFLICT {name}[{k!r}] wanted by {deltas[k][1]} and T={T}: kept base")
                        deltas[k] = (None, "conflict")
                    elif k not in deltas:
                        deltas[k] = (nv, f"T={T}")
            else:
                if name in merged and merged[name][0] != val:
                    print(f"[apply] CONFLICT {name} wanted as {merged[name][0]!r} by {merged[name][1]} and {val!r} by T={T}: kept base")
                    merged[name] = (b, "conflict")
                elif name not in merged:
                    merged[name] = (val, f"T={T}")
    final = {}
    for name, m in merged.items():
        b = getg(name)
        if isinstance(m, dict):
            v = dict(b)
            for k, (nv, who) in m.items():
                if nv is None:
                    continue
                if nv is DEL:
                    v.pop(k, None)
                else:
                    v[k] = nv
            final[name] = v
        else:
            final[name] = m[0]
    merged = final
    out = src
    for name, val in sorted(merged.items(), key=lambda kv: -nodes[kv[0]].lineno):
        if val == getg(name):
            continue
        n = nodes[name]
        old = ast.get_source_segment(src, n)
        new = f"{name}={val!r}"
        assert out.count(old) >= 1
        lines = out.split("\n")
        seg = "\n".join(lines[n.lineno - 1:n.end_lineno])
        assert seg.strip().startswith(old.strip()), (name, seg[:80], old[:80])
        lines[n.lineno - 1:n.end_lineno] = [new]
        out = "\n".join(lines)
        print(f"[apply] {name}: {old[:150]} -> {new[:150]}")
    dst.write_text(out, encoding="utf-8", newline="\n")
    print(f"[apply] wrote {dst} ({len(out.encode())} B)")
    vp = arg("variants")
    if vp:   # in-process A/B on the BASE file: layer_ab3 variants that set the changed globals (tuple keys -> "a,b,c")
        def enc(v):
            if isinstance(v, dict):
                return {(",".join(str(x) for x in k) if isinstance(k, tuple) else str(k)): enc(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [enc(x) for x in v]
            return v
        st = {k: enc(v) for k, v in merged.items() if v != getg(k)}
        var = [{"name": "base", "set": {}}, {"name": "evo", "set": st}, {"name": "twin", "set": {}},
               {"name": "evo2", "set": st}, {"name": "base2", "set": {}}]
        Path(vp).write_text(json.dumps(var) + "\n", encoding="utf-8")
        print(f"[apply] variants -> {vp}: {sorted(st)}")


if __name__ == "__main__":
    main()

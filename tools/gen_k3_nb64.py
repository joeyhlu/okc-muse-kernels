"""gen_k3_nb64.py: k3_nb64_<n> = k3_gc_16 + the k3_nb_1 KS3 table + T=64 compile-time instances for the nearby band T 33..64.

  python3 tools/gen_k3_nb64.py <n> [--gd64=JSON] [--dg64=JSON] [--use=JSON] [--x2=JSON]

  --gd64  {"qkv": [rt64, rt63], "unpooler": [...]}: GD route tuples (mt 8). The first is bucket 64 ("gd8"), the second (screen
          only) pseudo-bucket 63 ("gd8b"). The kernel template is unchanged; mt 8 is a new instantiation.
  --dg64  {"o": geo, "cat": geo, ...}: DG1 geometries (64, nwg, mt, s) at key (proj, 64) (new T=64 instantiations).
  --x2    ["gd", "gc"]: also make the T=32 GD qkv/unpooler (pseudo-bucket 66, "gdx2") / GC cat (bucket 64, "gcx2") routes
          run T 33..64 as 2 token tiles (grid z = 2: the existing mt 4 instances, no new code).
  --use   NB64_USE defaults, e.g. {"qkv": "gd8", "unpooler": "gd8", "cat": "dg1", "o": "dg1", "sbb1": true}.

Every T=64 path is gated by _nb64(T) (33 <= T <= 64); _gd_bucket / the DG1 key only return 64 inside the band, so the scored
T (1, 8, 16, 32, 128, 256) take byte-identical Python paths and the same kernel instantiations (module source grows only by
the new instances / the gdc nz switch, which is 1 unless pdl >> 8 > 1).
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gen_k3_nb as NB  # noqa: E402

KS3 = ((17, 31, 32), (33, 64, 32))

HDR = '''

NB64_LO, NB64_HI = 33, 64   # nearby band: T 33..64 -> the T=64 instances (route key 64; scored T never map here)
NB64_USE = {use}
NB64_BUILD = {build}
NB64_GD_B = {{"gd8": 64, "gd8b": 63, "gdx2": 66}}


def _nb64(T):
    return NB64_LO <= T <= NB64_HI


def _nb_sbb(T):
    return 32 if (NB64_USE.get("sbb1") and _nb64(T)) else _k2a_bucket_of(T)


GD_ENABLE = True
'''


def edits(gd64, dg64, use, x2):
    gd_extra = "".join(f', ("{p}", {b}): {tuple(rt)}' for p, rts in sorted(gd64.items()) for b, rt in zip((64, 63), rts))
    st_extra = ", 64: (64, 33)" + (", 63: (64, 33)" if any(len(v) > 1 for v in gd64.values()) else "")
    dg_route = "".join(f", ('{p}', 64): True" for p in sorted(dg64))
    dg_geo = "".join(f", ('{p}', 64): {tuple(g)}" for p, g in sorted(dg64.items()))
    build = {"gdx2": "gd" in x2, "gcx2": "gc" in x2}
    E = [
        ("\nGD_ENABLE = True\n", HDR.format(use=repr(use), build=repr(build))),
        ("    return 1 if T == 1 else 8 if T <= 8 else 16 if T <= 16 else 32 if T <= 32 else None\n",
         "    return 1 if T == 1 else 8 if T <= 8 else 16 if T <= 16 else 32 if T <= 32 else 64 if _nb64(T) else None\n"),
        ("or nt > 1024 or mt not in (1, 2, 4):", "or nt > 1024 or mt not in (1, 2, 4, 8):"),
        ('("unpooler", 32): (4, 2, 4, 16, 1, 1, 0)}', '("unpooler", 32): (4, 2, 4, 16, 1, 1, 0)' + gd_extra + "}"),
        ("GD_SELF_TEST_T = {1: (1,), 8: (8, 2, 5), 16: (16, 9), 32: (32, 17)}",
         "GD_SELF_TEST_T = {1: (1,), 8: (8, 2, 5), 16: (16, 9), 32: (32, 17)" + st_extra + "}"),
        ("    b = _gd_bucket(tokens)\n    if b is None or not GD_ROUTE.get((proj, b)):\n",
         "    b = _gd_bucket(tokens)\n    if b == 64:   # NB64: the band's own route (or none)\n"
         "        r = st[\"route\"].get((proj, NB64_GD_B.get(NB64_USE.get(proj), -1)))\n"
         "        return r if r is not None and tokens <= r[\"T\"] else None\n"
         "    if b is None or not GD_ROUTE.get((proj, b)):\n"),
        ('x, out, T, r["K"], ldo, r["tb"], r["nbx"], 1, pdl)', 'x, out, T, r["K"], ldo, r["tb"], r["nbx"], r.get("nz", 1), pdl)'),
        ('x, o, T, k, ldo, r["tb"], r["nbx"], 1, pdl)', 'x, o, T, k, ldo, r["tb"], r["nbx"], r.get("nz", 1), pdl)'),
        ('    torch.cuda.empty_cache()\n    state["gd"] = {"mod": mod, "route": routes} if routes else None\n',
         '    for proj in (("qkv", "unpooler") if NB64_BUILD.get("gdx2") else ()):   # NB64 gdx2: the T=32 route, grid z = 2\n'
         '        if (proj, 32) in routes:\n'
         '            r32 = routes[(proj, 32)]\n'
         '            rx = dict(r32, T=2 * r32["T"], nz=2, desc=r32["desc"] + "z2")\n'
         '            try:\n'
         '                ok, msg = _gd_selftest(rx, dense[GD_WEIGHT[proj]][:r32["N"]].contiguous(), 64)\n'
         '            except Exception as ex:\n'
         '                ok, msg = False, f"EXC {ex!r}"\n'
         '            if ok:\n'
         '                routes[(proj, 66)] = rx\n'
         '            print(f"[gd] self-test {proj}/x2 {rx[\'desc\']}: {\'ok\' if ok else \'FAIL\'} ({msg})", flush=True)\n'
         '    torch.cuda.empty_cache()\n    state["gd"] = {"mod": mod, "route": routes} if routes else None\n'),
        ("GD_CAT = {16: True, 32: True}", "GD_CAT = {16: True, 32: True, 64: True}"),
        ("GD_CAT_SELF_TEST_T = {16: (16, 9), 32: (32, 17)}", "GD_CAT_SELF_TEST_T = {16: (16, 9), 32: (32, 17), 64: (64, 33)}"),
        ("    b = _gd_bucket(tokens)\n    if b is None or not GD_CAT.get(b):\n",
         "    b = _gd_bucket(tokens)\n    if b == 64 and NB64_USE.get(\"cat\") != \"gcx2\":\n        return None\n"
         "    if b is None or not GD_CAT.get(b):\n"),
        ('r["tb"], r["tbr"], r["nbx"], pdl)', 'r["tb"], r["tbr"], r["nbx"], pdl | ((r["nz"] << 8) if r.get("nz", 1) > 1 else 0))'),
        ('    torch.cuda.empty_cache()\n    state["gd_cat"] = routes or None\n',
         '    if NB64_BUILD.get("gcx2") and 32 in routes:   # NB64 gcx2: the T=32 GC route, grid z = 2\n'
         '        rx = dict(routes[32], T=64, nz=2, desc=routes[32]["desc"] + "z2")\n'
         '        try:\n'
         '            ok, msg = _gd_cat_selftest(rx, wp, hi, lo, w32, 64)\n'
         '        except Exception as ex:\n'
         '            ok, msg = False, f"EXC {ex!r}"\n'
         '        if ok:\n'
         '            routes[64] = rx\n'
         '        print(f"[gc] self-test cat/x2 {rx[\'desc\']}: {\'ok\' if ok else \'FAIL\'} ({msg})", flush=True)\n'
         '    torch.cuda.empty_cache()\n    state["gd_cat"] = routes or None\n'),
        ("DG1_ROUTE = {('cat', 32): True, ('qkv', 32): True, ('o', 32): True}",
         "DG1_ROUTE = {('cat', 32): True, ('qkv', 32): True, ('o', 32): True" + dg_route + "}"),
        ("DG1_GEOMETRY = {('cat', 32): (32, 2, 1, 3), ('qkv', 32): (32, 2, 1, 4), ('o', 32): (32, 1, 1, 1)}",
         "DG1_GEOMETRY = {('cat', 32): (32, 2, 1, 3), ('qkv', 32): (32, 2, 1, 4), ('o', 32): (32, 1, 1, 1)" + dg_geo + "}"),
        ("    lt_out = _lt(state, proj, x)\n",
         "    nb_dg1 = _nb64(x.shape[0]) and NB64_USE.get(proj) == \"dg1\"\n"
         "    lt_out = None if nb_dg1 else _lt(state, proj, x)\n"),
        ("    vd_out = _vd(state, proj, x)\n", "    vd_out = None if nb_dg1 else _vd(state, proj, x)\n"),
        ("    key = (proj, _k2a_bucket_of(T))\n    if not DG1_ROUTE.get(key):\n",
         "    key = (proj, 64 if _nb64(T) else _k2a_bucket_of(T))\n    if key[1] == 64 and not nb_dg1:\n        return None\n"
         "    if not DG1_ROUTE.get(key):\n"),
        ('if which == "attn" and SBB1_ROUTE.get(_k2a_bucket_of(tokens)):', 'if which == "attn" and SBB1_ROUTE.get(_nb_sbb(tokens)):'),
        ('not st["ok"].get(_k2a_bucket_of(tokens)):', 'not st["ok"].get(_nb_sbb(tokens)):'),
    ]
    if "gc" in x2:   # GC grid z (host-side only); without gcx2 the GD module source stays byte-identical to gc_16's
        E.append(("  const int nz = 1;\n", "  const int nz = (pdl >> 8) > 1 ? (pdl >> 8) : 1;   // NB64: grid z token tiles\n"))
    return E


def main():
    n = sys.argv[1]
    opt = {"gd64": "{}", "dg64": "{}", "use": "{}", "x2": "[]"}
    for a in sys.argv[2:]:
        k, v = a[2:].split("=", 1)
        opt[k] = v
    gd64, dg64, use, x2 = (json.loads(opt[k]) for k in ("gd64", "dg64", "use", "x2"))
    s = NB.SRC.read_text(encoding="utf-8")
    for old, new in NB.EDITS:   # k3_nb_1: NB_VML_REP () + NB_KS3_B
        assert s.count(old) == 1, old[:80]
        s = s.replace(old, new.format(vml="()", ks3=repr(KS3)) if ("{vml}" in new or "{ks3}" in new) else new)
    for old, new in edits(gd64, dg64, use, x2):
        assert s.count(old) == 1, (s.count(old), old[:80])
        s = s.replace(old, new)
    ast.parse(s)
    out = ROOT / f"kernels/lab/k3_nb64_{n}.py"
    out.write_bytes(s.encode("utf-8"))
    print(f"{out.name}: {len(s.encode('utf-8'))} B; gd64 {gd64} dg64 {dg64} x2 {x2} use {use}")


if __name__ == "__main__":
    main()

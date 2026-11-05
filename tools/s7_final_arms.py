"""Build box_out/s7/arms_final.json from every box_out/s7/k3_<lane>_v*.py candidate that parses (plus k3_qr.py), with the row file
as arm 0 and two middle controls; prints the sync path list for s7x_run_d.sh."""
import ast, glob, json, os, pathlib
B = 'kernels/muse_decode_layer/k3_tinys7_c1.py'
cands = sorted(glob.glob('box_out/s7/k3_*_v[12].py')) + ['box_out/s7/k3_qr.py']
skip = ('k3_dn_v', 'k3_rt_v', 'k3_d32_v', 'k3_d128_v', 'k3_mk_v', 'k3_ks_v')   # older lanes already measured
ok = []
for c in cands:
    if any(os.path.basename(c).startswith(s) for s in skip) or not os.path.exists(c):
        continue
    try:
        ast.parse(pathlib.Path(c).read_text(encoding='utf-8'))
        ok.append(c.replace(chr(92),'/'))
    except Exception as ex:
        print('skip (parse)', c, ex)
arms = [{"name": "row", "file": B, "set": {}}]
half = (len(ok) + 1) // 2
for i, c in enumerate(ok):
    if i == half:
        arms.append({"name": "mid", "file": B, "set": {}})
    arms.append({"name": os.path.basename(c)[3:-3][:12], "file": c, "set": {}})
arms.append({"name": "mid2", "file": B, "set": {}})
pathlib.Path('box_out/s7/arms_final.json').write_bytes(json.dumps(arms).encode())
print(len(arms), 'arms'); print('SYNC', ' '.join([B, 'box_out/s7/arms_final.json'] + ok))

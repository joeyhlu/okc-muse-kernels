"""Compact view of judge profile files: per-shape layer time + speedup and the key kernels of the T<=8 shapes.
usage: python -I tools/s7_prof.py <profile.txt> [...]   (never print a whole profile: they are long)"""
import re, sys
LAB = ['d1s', 'd1f', 'd16s', 'd16f', 'd32s', 'd32f', 'v1s', 'v1f', 'v16s', 'v16f', 'v32s', 'v32f']
KEY = ('front_pipe', 'front_check', 'branch_b', 'extended_o', 'combine_e', 'input_u', 'moe_fused', 'ks1', 'gd_kernel', 'sbb', 'tail_kernel', 'vf_router', 'attn_kernel')
def parse(path):
    t = open(path, encoding='utf-8', errors='replace').read()
    shapes = []
    for line in t.splitlines():
        m = re.match(r'^(batch=.*?)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)x\s+(\d+\.\d+)', line)
        if m: shapes.append((float(m.group(2)), float(m.group(3)), float(m.group(4))))
    sp = re.search(r'^speedup\s+(\S+)', t, re.M)
    ps = {}
    for blk in re.split(r'\n(?=profile \d+:)', t):
        m = re.match(r'profile (\d+):', blk)
        if not m: continue
        ks = []
        for line in blk.splitlines()[2:]:
            mm = re.match(r'^(.*?)\s+(\d+\.\d+) µs\s+(\d+\.\d+)%\s+(\d+)', line)
            if mm and not line.startswith('total'): ks.append((mm.group(1).strip()[:46], float(mm.group(2))))
        ps[int(m.group(1))] = ks
    return (sp.group(1) if sp else '?'), shapes, ps
for p in sys.argv[1:]:
    sp, shapes, ps = parse(p)
    print(f'== {p.split("/")[-1]}  speedup {sp}')
    print('   ' + '  '.join(f'{l}:{s[1]:.1f}' for l, s in zip(LAB, shapes)))
    for idx, nm in ((0, 'd1s'), (1, 'd1f'), (6, 'v1s'), (7, 'v1f')):
        ks = [f'{k.split("(")[0][-22:]}={v:.1f}' for k, v in ps.get(idx, []) if any(x in k for x in KEY)]
        print(f'   {nm}: ' + ' | '.join(ks[:10]))

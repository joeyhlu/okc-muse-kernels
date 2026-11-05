"""Compare judge profile files against the 41-run row baseline (notes/study/evo/s7/kernel_baseline.md).
usage: python -I tools/s7_cmp.py <profile.txt> [...] [--k=ks3,vm_fin,...]"""
import os, re, sys
LAB = ['d1s', 'd1f', 'd16s', 'd16f', 'd32s', 'd32f', 'v1s', 'v1f', 'v16s', 'v16f', 'v32s', 'v32f']
keys = [a[4:].split(',') for a in sys.argv[1:] if a.startswith('--k=')]
keys = keys[0] if keys else []
base, cur = {}, None
for line in open('notes/study/evo/s7/kernel_baseline.md', encoding='utf-8'):
    m = re.match(r'## (\w+): layer median ([\d.]+)', line)
    if m:
        cur = m.group(1); base[cur] = {'layer': float(m.group(2))}
    m = re.match(r'- (.*?): ([\d.]+) \[', line)
    if m and cur:
        base[cur][m.group(1)[:40]] = float(m.group(2))
for f in [a for a in sys.argv[1:] if not a.startswith('--')]:
    t = open(f, encoding='utf-8', errors='replace').read()
    sh = [float(m.group(1)) for m in re.finditer(r'^batch=.*?\s+\d+\.\d+\s+(\d+\.\d+)\s+\d+\.\d+x', t, re.M)]
    sp = re.search(r'^speedup\s+(\S+)', t, re.M)
    print(os.path.basename(f)[:34], (sp.group(1)[:6] if sp else '?'), ' '.join(f'{l}{sh[i] - base[l]["layer"]:+.1f}' for i, l in enumerate(LAB) if i < len(sh)))
    for blk in re.split(r'\n(?=profile \d+:)', t):
        m = re.match(r'profile (\d+):', blk)
        if not m:
            continue
        l = LAB[int(m.group(1))]
        for line in blk.splitlines()[2:]:
            mm = re.match(r'^(.*?)\s+(\d+\.\d+) µs', line)
            if mm and any(k in mm.group(1) for k in keys):
                print(f'   {l} {mm.group(1).strip()[:34]} {base[l].get(mm.group(1).strip()[:40])} -> {mm.group(2)}')

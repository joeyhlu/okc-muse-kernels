# R-hot-0: CPU estimate of one-pass PP for hot experts (72 < N <= 128) at T=256 (probe fits from notes/study/xc_moe.md s5)
import json, statistics as st
d = json.load(open('box_out/m32e1/run.json'))
w72 = lambda n: 0.999 + 0.0168 * n          # probe cold fit (32,1.537)-(64,2.075)
pp = lambda n: 1.258 + 0.0174 * n           # probe cold fit (64,2.372)-(112,3.208)
p8 = lambda n: (n + 7) // 8 * 8
G, STG = 132, 768
rows = []
for x in d['routings']:
    if x.get('T') != 256: continue
    sv, nh, hs = 0.0, 0, 0
    for n in x['counts']:
        if 72 < n <= 128:
            b = (p8(n) // 8 + 1) // 2 * 8
            sv += STG * (w72(p8(b)) + w72(p8(n - b)) - pp(p8(n))); nh += 1; hs += STG
    gross = sv / G
    rows.append((x['rs'], x['seed'], nh, round(gross, 2), round(hs / G, 1), round(gross - 1.25 - 2.5, 2)))
for r in rows: print(r)
lay = [r for r in rows if r[0] in (8677534258059892491, 12345, 777)]
print('layer6 median gross', st.median(r[3] for r in lay), 'net', st.median(r[5] for r in lay))
print('all12 median gross', st.median(r[3] for r in rows), 'net', st.median(r[5] for r in rows))

"""Paper model behind notes/study/indep_mid.md s3.3 / s4.3 (VMS-KS at T = 16 / 32, MXS at T = 128 / 256) on the layer
routings in box_out/**/run*.json. Constants: VMS unit 1.17 us, VML stage table F / M (R-vml-1 / 6 / 8 / 14), VML fixed
costs per T (moe_t32plus s1). python3 -I tools/indep_mid_model.py (from the repo root)."""
import json, glob, math, statistics as st
seen=set(); R={}
for f in sorted(glob.glob('box_out/**/*.json',recursive=True)):
    try: d=json.load(open(f))
    except Exception: continue
    if not isinstance(d,dict) or 'routings' not in d: continue
    for r in d['routings']:
        if not isinstance(r,dict) or 'counts' not in r: continue
        T=r.get('T'); c=tuple(sorted(r['counts'],reverse=True))
        if T is None or (T,c) in seen: continue
        seen.add((T,c)); R.setdefault(T,[]).append(c)
G=132; TU=1.17
# measured VML stage us per 36 KB (R-vml-1/6/8): mma path for N<=16 (L1b), wgmma above
S={8:1.124,16:1.312,24:1.49,32:1.618,40:1.70,48:1.77,56:1.90,64:2.04}
pad=lambda n:max(8,-(-n//8)*8)
def vml_jobs(c):
    out=[]
    for n in c:
        nj=-(-n//64); base=-(-n//nj)
        for k in range(nj): out.append(min(base,n-k*base))
    return out
def vml_body(c):
    return sum(768*S[pad(n)] for n in vml_jobs(c))/G
def vms(c):
    jobs=sum(-(-n//8) for n in c); return math.ceil(768*jobs/G)*TU+5.3
def vmsks(c,r1,r2,maxj=32):
    W=0
    for n in c:
        if n<=8: W+=768
        elif n<=16: W+=768*r1
        elif n<=32 and maxj>=32: W+=768*r2
        else: W+=768*(-(-n//16))*r1   # fall back to 16-token K-split jobs
    nk=sum(1 for n in c if n>8)
    return (W/G+1)*TU+5.3+0.2+0.05*nk*768/G*0  # fixup cost inside r (probe red.add); weighting routing +0.2
def mxs_body(c,r):
    jobs=[pad(n) for n in vml_jobs(c)]
    hot=sorted([n for n in jobs if n>=24],reverse=True); cold=sorted([n for n in jobs if n<=16])
    H=[(n,1536) for n in hot]; C=[(n,1536) for n in cold]
    saved=0.0
    hi=0; ci=0; H=[list(x) for x in H]; C=[list(x) for x in C]
    while hi<len(H) and ci<len(C):
        k=min(H[hi][1],C[ci][1])
        saved+=k*(1-r)*(S[H[hi][0]]+S[C[ci][0]])/2   # k hot + k cold half-stages -> k mixed stages
        H[hi][1]-=k; C[ci][1]-=k
        if H[hi][1]==0: hi+=1
        if C[ci][1]==0: ci+=1
    return vml_body(c)-saved/G
for T,fix in ((16,None),(32,25.7),(128,30.2),(256,None)):
    cs=R[T]; print(f"T={T}: {len(cs)} routings, experts {st.median(len(c) for c in cs)}, pairs {st.median(sum(c) for c in cs)}, top {st.median(c[0] for c in cs)}")
    if T==16:
        a=[vms(c) for c in cs]; b=[vmsks(c,1.08,1.20) for c in cs]; g=[vmsks(c,1.20,1.40) for c in cs]
        print(f"  VMS model {st.median(a):.1f}; VMS-KS r1 1.08 {st.median(b):.1f} (d {st.median([y-x for x,y in zip(a,b)]):+.1f}, min {min(y-x for x,y in zip(a,b)):+.1f}, max {max(y-x for x,y in zip(a,b)):+.1f}); at gate r1 1.20 {st.median([y-x for x,y in zip(a,g)]):+.1f}")
    if T==32:
        a=[vml_body(c)+fix for c in cs]; b=[vmsks(c,1.08,1.20) for c in cs]; g=[vmsks(c,1.20,1.40) for c in cs]; v=[vms(c) for c in cs]
        print(f"  VML model {st.median(a):.1f} (measured 74.2 standalone); VMS(MAXT32) model {st.median(v):.1f}; VMS-KS model {st.median(b):.1f} (d vs VML {st.median([y-x for x,y in zip(a,b)]):+.1f}, range {min(y-x for x,y in zip(a,b)):+.1f}..{max(y-x for x,y in zip(a,b)):+.1f}); at gate (1.20/1.40) {st.median([y-x for x,y in zip(a,g)]):+.1f}")
    if T in (128,256):
        a=[vml_body(c) for c in cs]
        print(f"  VML body model {st.median(a):.1f}" + (f" + fixed {fix} = {st.median(a)+fix:.1f} (measured 156.1)" if fix else ""))
        for r in (0.76,0.80,0.87,0.95):
            b=[mxs_body(c,r) for c in cs]
            print(f"    MXS r={r}: d {st.median([y-x for x,y in zip(a,b)]):+.1f} us (range {min(y-x for x,y in zip(a,b)):+.1f}..{max(y-x for x,y in zip(a,b)):+.1f})")

print("\n--- VMS-KS for every expert at T >= 32 (experts > 32 tokens split into <= 32-token K-split jobs, re-streamed)")
def ks_all(c, r1, r2, fixed):
    W = 0.0
    for n in c:
        nj = -(-n // 32)
        for k in range(nj):
            m = -(-n // nj) if k < n % nj or n % nj == 0 else n // nj
            W += 768 * (1.0 if m <= 8 else r1 if m <= 16 else r2)
    return W / G * TU + fixed
for T, vml_meas, vml_fix in ((32, 74.2, 25.7), (128, 156.1, 30.2), (256, 204.8, 34.1)):
    cs = R[T]
    a = [vml_body(c) + vml_fix for c in cs]
    for r1, r2, fx in ((1.08, 1.20, 8.0), (1.20, 1.40, 10.0)):
        b = [ks_all(c, r1, r2, fx) for c in cs]
        scale = vml_meas / st.median(a)
        print(f"T={T}: VML model {st.median(a):.1f} (meas {vml_meas}); KS-all r1 {r1} r2 {r2} fixed {fx}: {st.median(b):.1f} "
              f"-> d vs VML model {st.median([y - x for x, y in zip(a, b)]):+.1f}, vs measured-scaled VML {st.median(b) - vml_meas:+.1f}")

print("\n--- VMS-KS per routing (T=16 vs VMS model; T=32 vs VML measured-scaled), r1 / r2 grid")
def vmsks2(c, r1, r2):
    W = sum(768 * (1.0 if n <= 8 else r1 if n <= 16 else r2) for n in c)
    hot = any(n > 8 for n in c)
    return (W / G + 0.5) * TU + 5.3 + (0.2 if hot else 0.0) if hot else vms(c)
for r1, r2 in ((1.08, 1.20), (1.10, 1.30), (1.20, 1.40)):
    d16 = [vmsks2(c, r1, r2) - vms(c) for c in R[16]]
    a32 = [vml_body(c) + 25.7 for c in R[32]]; sc = 74.2 / st.median(a32)
    d32 = [vmsks2(c, r1, r2) - x * sc for c, x in zip(R[32], a32)]
    print(f"r1 {r1} r2 {r2}: T=16 d median {st.median(d16):+.1f} (min {min(d16):+.1f} max {max(d16):+.1f}); "
          f"T=32 d vs scaled VML median {st.median(d32):+.1f} (min {min(d32):+.1f} max {max(d32):+.1f})")
print("T=16 routings hot experts (>8):", [sum(1 for n in c if n > 8) for c in R[16]])
print("T=32 routings experts >16:", [sum(1 for n in c if n > 16) for c in R[32]], " 9-16:", [sum(1 for n in c if 8 < n <= 16) for c in R[32]])

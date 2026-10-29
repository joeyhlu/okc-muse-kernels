"""dsm_split_frac.py (R-dsm-0 design note, $0): which of ship's phase-A split merges fall inside one thread-block cluster?

Runs the REAL ship route() through tools/aw_local_test.py (local RTX 3070, saved routings m32e14 / vml14 + stress) and
records every ship plan (VM_AW 0). A split share sits at CTA boundary i when bA[i] % 16 != 0 (an A unit cut between CTA
i-1 and i); its merge is intra-cluster for cluster size C iff i % C != 0 (the static plan only splits adjacent CTAs).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python3 "<repo>/tools/dsm_split_frac.py" OUTDIR stress=0
"""
import pathlib
import statistics as S
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aw_local_test as A  # noqa: E402

seen = []
_plan = A.plan


def plan(o):
    p = _plan(o)
    seen.append(p)
    return p


A.plan = plan
rc = A.main()
# main() calls plan(ship) then plan(AW) per case: even entries are ship plans
rows = {}
for p in seen[0::2]:
    nj, G, cyc, bA = p[:4]
    if nj <= 0:
        continue
    cuts = [i for i in range(1, G) if bA[i] % 16]
    for C in (2, 4, 8):
        rows.setdefault(C, []).append(sum(1 for i in cuts if i % C) / max(len(cuts), 1))
    rows.setdefault("cuts", []).append(len(cuts) / (G - 1))
print(f"[dsm] ship plans {len(rows.get('cuts', []))}: split boundaries / CTA boundary {S.mean(rows['cuts']):.3f}")
for C in (2, 4, 8):
    print(f"[dsm] C{C}: intra-cluster share of A split merges mean {S.mean(rows[C]):.3f} min {min(rows[C]):.3f}")
sys.exit(rc)

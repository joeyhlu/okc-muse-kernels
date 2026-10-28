"""Paper-only accounting from observed routings; no GPU, imports or source edits."""
from pathlib import Path
import json
import math
import statistics as S

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / 'run.json').read_text())['routings']

def jobs(n, cap):
    k = (n + cap - 1) // cap
    q, r = divmod(n, k)
    return [q + 1] * r + [q] * (k - r)

def account(counts, cap, tile_rows):
    sizes = [v for n in counts for v in jobs(n, cap)]
    units = math.ceil(8192 / tile_rows) + math.ceil(4096 / tile_rows)
    packed_bytes = units * 16 * tile_rows * (128 + 16) * len(sizes)
    rounded_tokens = sum((v + 7) // 8 * 8 for v in sizes)
    return dict(jobs=len(sizes), weight_bytes=packed_bytes,
                padded_tensor_work=units * tile_rows * rounded_tokens,
                activation_stage_bytes=units * 16 * 256 * 2 * rounded_tokens,
                segments=units * len(sizes))

out = {'warning': 'Accounting only, not predicted runtime. Includes 192-row end padding; excludes barriers, splits, bandwidth and clocks.', 'by_tokens': {}}
for t in (32, 128, 256):
    rr = [r for r in rows if r['T'] == t]
    entries = {}
    for cap, tile in ((96, 192), (112, 192), (128, 192)):
        ratios = {k: [] for k in ('weight_bytes', 'padded_tensor_work', 'activation_stage_bytes', 'segments')}
        removed = []
        for r in rr:
            base = account(r['counts'], 72, 256)
            alt = account(r['counts'], cap, tile)
            for k in ratios:
                ratios[k].append(alt[k] / base[k])
            removed.append(base['jobs'] - alt['jobs'])
        entries[str(cap)] = dict(median_jobs_removed=S.median(removed),
                                changes_pct={k: round(100 * (S.median(v) - 1), 2) for k, v in ratios.items()})
    out['by_tokens'][str(t)] = entries
out['registers'] = {'consumer_144_producer_64': 384 * 144 + 128 * 64,
                    'consumer_152_producer_56': 384 * 152 + 128 * 56,
                    'launch_512_threads_128_regs': 512 * 128}
out['ring_bytes_excluding_metadata'] = {'N96_W2_X3': 2 * 192 * 144 + 3 * 96 * 256 * 2,
                                       'N128_W3_X2': 3 * 192 * 144 + 2 * 128 * 256 * 2}
rr = [r for r in rows if r['T'] == 32]
out['decode32_double_rows'] = {
    k: round(100 * (S.median(account(r['counts'], 32, 512)[k] / account(r['counts'], 72, 256)[k] for r in rr) - 1), 2)
    for k in ('weight_bytes', 'padded_tensor_work', 'activation_stage_bytes', 'segments')}
out['decode32_double_rows']['ring_bytes_excluding_metadata'] = 2 * 512 * 144 + 3 * 32 * 256 * 2
(HERE / 'research_model.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

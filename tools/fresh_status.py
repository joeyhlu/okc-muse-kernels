"""Read-only official rank and quota snapshot. Use authenticated WSL okc helper."""
import json
from okc_wave import okc

for args in [('leaderboard', 'muse_decode_layer', '--json'), ('problems', '--json')]:
    rc, out, err = okc(*args)
    print('command',args,'rc',rc)
    if rc:
        print(err[-500:])
        continue
    data = json.loads(out)
    if args[0] == 'leaderboard':
        print('you',data.get('you'))
        rows = data.get('rows',data.get('leaderboard',[]))
        for row in rows:
            if row.get('rank') in (1,10,50,100) or 'joeyhlu' in str(row):
                print(json.dumps(row))
    else:
        print(json.dumps(data))

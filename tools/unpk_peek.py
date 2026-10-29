"""unpk_peek.py: print the instructions of one unpk_sass stage loop that match a regex (<= 40 rows).
  python unpk_peek.py SASS fn kind N regex"""
import re
import sys
import unpk_sass as U
s = open(sys.argv[1]).read()
ins = U.fn_ins(s, sys.argv[2])
kind, n, rx = sys.argv[3], int(sys.argv[4]), sys.argv[5].replace(",", "|")
for body in U.loops(ins):
    if U.nmask(body) == 64 and U.stage_n(body) == (kind, n):
        rows = [t for _, t in body if re.search(rx, t)]
        print(len(body), len(rows))
        for t in rows[:40]:
            print("  " + t[:110])
        break

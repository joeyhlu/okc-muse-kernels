"""lat32_loops.py: E0 (b) hot-loop audit: backward-branch loop bodies of vm_main that contain HGMMA; opcode mix per body.
  python lat32_loops.py file.sass [maxlen]"""
import collections
import re
import sys

UNP = ("LOP3", "HMUL2", "PRMT", "SHF", "HFMA2", "HADD2")


def main():
    txt = open(sys.argv[1]).read()
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    fn = None
    for blk in re.split(r"\n\s*Function : ", txt)[1:]:
        if blk.startswith("_Z") and "vm_main" in blk.split("\n", 1)[0]:
            fn = blk
    ins = []
    for ln in fn.splitlines():
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?);", ln)
        if m:
            ins.append((int(m.group(1), 16), re.sub(r"^@!?U?P\w+\s+", "", m.group(2))))
    addr = [a for a, _ in ins]
    rows = []
    for i, (a, t) in enumerate(ins):
        m = re.match(r"BRA\S*\s+(0x[0-9a-f]+)", t)
        if m and int(m.group(1), 16) < a and int(m.group(1), 16) in addr:
            j0 = addr.index(int(m.group(1), 16))
            body = [x for _, x in ins[j0:i + 1]]
            ops = collections.Counter(x.split()[0].split(".")[0] for x in body)
            if ops["HGMMA"] and len(body) <= mx:
                rows.append((len(body), ops))
    seen = set()
    for n, ops in sorted(rows, key=lambda r: -r[0]):
        key = (n, ops["HGMMA"])
        if key in seen:
            continue
        seen.add(key)
        unp = sum(ops[k] for k in UNP)
        top = " ".join(f"{k} {v}" for k, v in ops.most_common(9))
        print(f"body {n} HGMMA {ops['HGMMA']} unpack {unp} ({unp / n:.0%}) non-unpack-non-mma {(n - unp - ops['HGMMA']) / n:.0%} | {top}")


if __name__ == "__main__":
    main()

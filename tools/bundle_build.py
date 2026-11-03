"""bundle_build.py (2026-11-03, spare session): compose the final decode-layer bundle from a base file with the three independent patch generators, in the order
HYA Q2 (tools/hya_q2_patches.py) -> floor flags (tools/rpflags_patches.py) -> per-T wide-tier margin (tools/rpwt_patches.py, main session), and print the size and hash.
usage: python tools/bundle_build.py --base=kernels/muse_decode_layer/k3_fin2_c1.py --out=kernels/lab/k3_bundle_ab.py [--phase=2] [--flags=128:2] [--rpwt=1:0.008,16:0.012,32:0.010]
  --phase=0   skip HYA Q2 entirely (flags / RPWT on the base only); 1 = v1f only; 2 = v1f + v16s (default)
  --flags=    rpflags per-T flags (default none); --rpwt= RPW_MARGIN_T values to set in the file (default none = RPWT not applied)
The output must stay below 1,000,000 bytes (the submission limit); the script fails if it does not.
"""
import hashlib
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


base = arg("base", "kernels/muse_decode_layer/k3_fin2_c1.py")
out = arg("out", "kernels/lab/k3_bundle_ab.py")
phase = arg("phase", "2")
flags = arg("flags", "")
rpwt = arg("rpwt", "")
cur = base
tmp = pathlib.Path(tempfile.mkdtemp(prefix="bundle_", dir=str(ROOT / "kernels" / "lab")))
step = 0


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def nxt():
    global step
    step += 1
    return str((tmp / f"s{step}.py").relative_to(ROOT))


if phase != "0":
    o = nxt()
    print(run(["tools/hya_q2_patches.py", f"--base={cur}", f"--phase={phase}", f"--out={o}"]))
    cur = o
if flags:
    o = nxt()
    print(run(["tools/rpflags_patches.py", f"--base={cur}", f"--flags={flags}", f"--out={o}"]))
    cur = o
if rpwt:
    o = nxt()
    print(run(["tools/rpwt_patches.py", f"--base={cur}", f"--out={o}"]))
    s = (ROOT / o).read_bytes().decode("ascii")
    a = "RPW_MARGIN_T={}  # RPWT"
    assert s.count(a) == 1
    vals = ",".join(f"{int(kv.split(':')[0])}:{float(kv.split(':')[1])}" for kv in rpwt.split(","))
    s = s.replace(a, "RPW_MARGIN_T={%s}  # RPWT" % vals)
    (ROOT / o).write_bytes(s.encode("ascii"))
    cur = o
data = (ROOT / cur).read_bytes()
compile(data.decode("utf-8"), out, "exec")
assert len(data) < 1_000_000, f"bundle is {len(data)} bytes (limit 1,000,000)"
dst = ROOT / out
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_bytes(data)
for f in tmp.iterdir():
    f.unlink()
tmp.rmdir()
base_n = len((ROOT / base).read_bytes())
print(f"bundle {out}: {len(data)} bytes (+{len(data) - base_n} vs {base}), sha256 {hashlib.sha256(data).hexdigest()[:16]}, phase {phase}, flags '{flags}', rpwt '{rpwt}'")

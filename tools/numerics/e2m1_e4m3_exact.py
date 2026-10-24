"""Check that every E2M1 value times every finite E4M3 value is exactly representable in BF16,
using the challenge's own nvfp4.py decode (problems/muse_moe_w4a16/nvfp4.py) for E2M1.
Run: python3 -I tools/numerics/e2m1_e4m3_exact.py problems/muse_moe_w4a16"""
import sys, importlib.util, torch
spec = importlib.util.spec_from_file_location("nvfp4", sys.argv[1] + "/nvfp4.py")
nv = importlib.util.module_from_spec(spec); spec.loader.exec_module(nv)
codes = torch.arange(16, dtype=torch.uint8)
packed = (codes[0::2] | (codes[1::2] << 4))            # low nibble = even element
e2m1 = nv.unpack_e2m1(packed).double()                  # 16 values incl. -0
e4m3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
finite = torch.isfinite(e4m3); e4m3 = e4m3[finite]
prod = e2m1[:, None] * e4m3[None, :]                    # exact in float64 (<= 6 + 4 significant bits)
bf = prod.to(torch.bfloat16).double()
fp16 = prod.to(torch.float16).double()
print("E2M1 values:", sorted(set(e2m1.tolist())))
print("finite E4M3 values:", int(finite.sum()), "min>0", float(e4m3[e4m3 > 0].min()), "max", float(e4m3.max()))
print("products:", prod.numel(), "| exact in BF16:", bool((bf == prod).all()), "| exact in FP16:", bool((fp16 == prod).all()),
      "| FP16 inexact count:", int((fp16 != prod).sum()))
# significant bits actually used by the products
nz = prod[prod != 0].abs()
m, e = torch.frexp(nz)  # nz = m * 2^e, m in [0.5,1)
bits = max(int(k) for k in range(1, 12) if True and ((m * 2**k) != torch.round(m * 2**k)).any()) + 1 if False else None
need = 0
for k in range(1, 25):
    if ((m * 2**k) == torch.round(m * 2**k)).all():
        need = k; break
print("max significant bits in any product:", need, "(BF16 has 8)")
print("product range:", float(nz.min()), "to", float(nz.max()))
# with the FP32 global scale folded in, BF16 is no longer exact in general:
gs = torch.tensor(1.0 / (8.6 * 4.0**7 * 4096) ** 0.5, dtype=torch.float32).double()
print("with a typical global scale gs=%.6g, exact in BF16: %s" % (float(gs), bool(((prod * gs).to(torch.bfloat16).double() == prod * gs).all())))

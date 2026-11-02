#!/usr/bin/env python3
"""Generate the HYB-only runtime L2 redirect experiment; never modifies the base."""
from pathlib import Path
import difflib

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/muse_moe_w4a16/mb_layer.py'
OUT = ROOT / 'kernels/muse_moe_w4a16/mb_layer_l2r.py'

def generate():
    source = BASE.read_text(encoding='utf-8')
    start = source.index('_HYB_MOE_CUDA=r"""')
    end = source.index('\n_HYB_MOE_CPP=', start)
    head, cuda, tail = source[:start], source[start:end], source[end:]
    changes = [
        ('uint32_t zero;\nconst __nv_bfloat16* x;',
         'uint32_t zero; const int* l2r_mode;\nconst __nv_bfloat16* x;'),
        ('p.zero = 0;\np.x =',
         'p.zero = 0; p.l2r_mode = (const int*)w[8].data_ptr();\np.x ='),
        ('const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;',
         '''const size_t l2r_unit = (size_t)(((S.jE[j] << ubl) + ub) * 16 + st);
const int l2r_mode = *p.l2r_mode;
const bool l2r_redirect = l2r_mode == 2 || (l2r_mode == 1 && S.jS[j] > S.off[S.jE[j]]);
const unsigned char* src = base + (l2r_redirect ? l2r_unit % 112 : l2r_unit) * (size_t)UNIT_BYTES;'''),
    ]
    for old, new in changes:
        n = cuda.count(old)
        if n != 1:
            raise AssertionError('patch match count %d: %s' % (n, old[:100]))
        cuda = cuda.replace(old, new, 1)
    result = head + cuda + tail
    assert result[:start] == source[:start] and result[result.index('\n_HYB_MOE_CPP=', start):] == tail
    OUT.write_bytes(result.encode('utf-8'))
    print(''.join(difflib.unified_diff(source.splitlines(True), result.splitlines(True), fromfile=str(BASE), tofile=str(OUT))), end='')
    print('Generated', OUT, flush=True)

if __name__ == '__main__':
    generate()

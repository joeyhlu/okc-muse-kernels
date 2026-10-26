"""Generate a LAB-only full layer A/B wrapper. PR snapshot is not judge-ready."""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'kernels/muse_decode_layer/k3_v34.py'
OUT=ROOT/'kernels/lab/fresh_fa4_layer.py'
TAIL='''
# LAB ONLY: pinned PR206 snapshot plus the independently validated attention lab.
from fresh_fa4_lab import prepare as _fresh_attn_prepare, run as _fresh_attn_run
FRESH_ATTENTION = False
_fresh_attn_buffers = {}
_fresh_attn_old_core = _v18_fa4_core

def _v18_fa4_core(qkv, data, pdl=True):
    cfg = data['config']
    if FRESH_ATTENTION and cfg['window'] == 0 and cfg['q_len'] == 8 and cfg['batch'] >= 16:
        dd = dict(data, qkv=qkv)
        key = (cfg['batch'], cfg['q_len'], str(qkv.device))
        if key not in _fresh_attn_buffers:
            _fresh_attn_buffers[key] = _fresh_attn_prepare(dd)
        return _fresh_attn_run(dd, _fresh_attn_buffers[key], (64,128))
    return _fresh_attn_old_core(qkv, data, pdl=pdl)

def _fresh_validate_capture(data, state):
    cfg=data['config']
    if FRESH_ATTENTION and cfg['window']==0 and cfg['q_len']==8 and cfg['batch']>=16:
        key=(cfg['batch'],cfg['q_len'],str(data['attn_input'].device))
        assert state.get('fa4v6') and K1FA4_STATE['ready'] and key in _fresh_attn_buffers, 'PR206 route was not active'
'''

def main():
    source=BASE.read_text()
    old='''        except Exception as ex:  # noqa: BLE001
            _k3_disable(state, "fa4v6", False, "FA4 v6 route", ex)'''
    new='''        except Exception as ex:  # noqa: BLE001
            if FRESH_ATTENTION:
                raise  # A lab candidate must not silently fall back to the control.
            _k3_disable(state, "fa4v6", False, "FA4 v6 route", ex)'''
    assert source.count(old)==1
    source=source.replace(old,new)+TAIL
    ast.parse(source);OUT.write_text(source)
    print('LAB_ONLY',OUT.name,len(source.encode()),hashlib.sha256(source.encode()).hexdigest())

if __name__=='__main__':main()

"""Single-expert Quack 0.6.5 NVFP4/BF16 GEMM; native FP32 output only.

prepare(q, sf, gs) repacks weights only, without a dense copy or descriptors.
configs(tokens, state) returns at most two source-derived, split1 geometries.
run(x, state, out, config, metadata=None) returns the supplied FP32 out.
No imports of torch/Quack or GPU work until prepare/run/configs is called.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _library():
    from importlib.metadata import version
    from types import SimpleNamespace
    import torch
    from quack.gemm_w4 import prepare_w4_weight, _w4a16_alpha
    from quack.operand_transform.host import pick_w4_cfg
    if version('quack-kernels') != '0.6.5':
        raise RuntimeError('requires the audited Quack 0.6.5 API')
    return SimpleNamespace(torch=torch, pack=prepare_w4_weight,
                           epi=_w4a16_alpha, pick=pick_w4_cfg)


def geometry_configs(tokens, n, k, analytic):
    """Library geometry with split1, plus one policy for more independent tiles.

    The alternate M64, N16/32 uses the transform's small-N occupancy2 rule.
    It trades weight rereads for coverage; no timing-based choice or sweep.
    """
    if tokens <= 0 or tokens > 128 or n % 128 or k % 64:
        raise ValueError('requires 1<=tokens<=128, N multiple128, K multiple64')
    tm, tn, recommended_split = map(int, analytic)
    choices = [('quack_default_sk1', tm, tn),
               ('quack_coverage_sk1', 64, 16 if tokens <= 32 else 32)]
    result, seen = [], set()
    for name, tm, tn in choices:
        if (tm, tn) in seen:
            continue
        if tm not in (64, 128) or tn % 16 or tn > 128 or n % tm:
            raise ValueError(f'unaudited W4 geometry {(tm, tn)}')
        seen.add((tm, tn))
        result.append(dict(name=name, tile_m=tm, tile_n=tn, tile_k=64,
                           cluster_m=1, cluster_n=1, pingpong=False,
                           dynamic_scheduler=False, split_k=1, use_pdl=False,
                           analytic_recommended_split=recommended_split,
                           logical_output_tiles=(n // tm) * ((tokens + tn - 1) // tn)))
    return result


def prepare(q, sf, gs):
    lib = _library()
    torch = lib.torch
    if q.dtype != torch.uint8 or q.ndim != 2 or not q.is_contiguous():
        raise TypeError('q must be contiguous uint8[N,K/2] raw E2M1, low nibble first')
    n, kb = q.shape
    if n % 128 or kb % 32:
        raise ValueError('lab requires N multiple128 and K multiple64; no hidden padding')
    if sf.dtype != torch.float8_e4m3fn or tuple(sf.shape) != (n, kb // 8):
        raise TypeError('sf must be raw E4M3[N,K/16]')
    if gs.dtype != torch.float32 or gs.numel() != 1 or not gs.is_contiguous():
        raise TypeError('gs must be a contiguous one-element raw FP32 global scale')
    if not q.is_cuda or sf.device != q.device or gs.device != q.device:
        raise ValueError('q, sf, gs must share a CUDA device')
    blob, sf_blob = lib.pack(q, sf, wformat='nvfp4')
    if blob.dtype != torch.uint8 or sf_blob.dtype != torch.uint8:
        raise TypeError('Quack repack must preserve raw byte storage')
    if blob.numel() != q.numel() or sf_blob.numel() != sf.numel():
        raise RuntimeError('unexpected expansion of packed weight/SF storage')
    return dict(blob=blob, sf_blob=sf_blob, gs=gs, n=int(n), k=int(2 * kb),
                format='nvfp4', metadata=dict(weight_bytes=blob.numel(),
                sf_bytes=sf_blob.numel(), global_bytes=4, prepared_dense_weights=False,
                precision='raw NVFP4 + group16 E4M3 -> exact BF16 fragments; FP32 accum/alpha/output'))


def configs(tokens, state):
    analytic = _library().pick(tokens, state['n'], state['k'] // 64, sm120=False)
    return geometry_configs(tokens, state['n'], state['k'], analytic)


def _record(plan, config):
    expected = dict(tile_M=config['tile_m'], tile_N=config['tile_n'],
                    cluster_M=1, cluster_N=1, split_k=1,
                    scheduler_uses_semaphore=False, d_transposed=True)
    if any(getattr(plan, key, None) != value for key, value in expected.items()):
        raise RuntimeError('Quack resolved an unexpected W4 dispatch')
    keys = dict(plan.epi_arg_keys)
    if keys.get('alpha', (None,))[0] != 'pointer':
        raise RuntimeError('global alpha must stay a live FP32 device scalar')
    return dict(requested=config.copy(), dispatch={**expected,
                'max_active_clusters':getattr(plan, 'max_active_clusters', None),
                'max_swizzle_size':getattr(plan, 'max_swizzle_size', None)},
                epi_arg_keys=[(name, str(key)) for name, key in plan.epi_arg_keys],
                gemm_class=f'{plan.gemm_cls.__module__}.{plan.gemm_cls.__qualname__}')


def run(x, state, out, config, metadata=None):
    lib = _library()
    torch = lib.torch
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != state['k'] or not x.is_contiguous():
        raise TypeError('x must be live contiguous BF16[T,K]')
    if out.dtype != torch.float32 or tuple(out.shape) != (x.shape[0], state['n']) or not out.is_contiguous():
        raise TypeError('supplied out must be contiguous FP32[T,N]; no downcast fallback')
    if x.device != state['blob'].device or out.device != x.device:
        raise ValueError('live x/out and packed weights must share a device')
    if config['split_k'] != 1 or config['tile_k'] != 64:
        raise ValueError('only split1 and canonical K64 are supported')
    plan = lib.epi.gemm(x, state['blob'], out, epi_args={'alpha':state['gs']},
                        transform_a='nvfp4', transform_sf=state['sf_blob'],
                        tile_M=config['tile_m'], tile_N=config['tile_n'], tile_K=64,
                        cluster_M=1, cluster_N=1, pingpong=False,
                        persistent=True, is_dynamic_persistent=False, split_k=1,
                        post_init_attrs=(('use_pdl', False),))
    if metadata is not None and not metadata:
        metadata.update(_record(plan, config))
    return out

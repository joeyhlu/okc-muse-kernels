# b40edb1cfe1547ff muse_attention_decode run status=done/ok passed=True
speedup=1.3485417393276633 ci=[1.3283136132377484, 1.3687698654175782] score_us=2792.868689313056 baseline_us=3766.3 run_s=337 error=None
tests: 16/16 pass
| load | shape | w | SGLang µs | ours µs | speedup | flags |
|---|---|---|---|---|---|---|
| decode 1 | B1 L1 sliding | 46 | 20.7 | 13.1 | 1.575 |  |
| decode 1 | B1 L1 full | 16 | 31.8 | 23.2 | 1.370 |  |
| decode 16 | B16 L1 sliding | 46 | 26.7 | 15.7 | 1.697 |  |
| decode 16 | B16 L1 full | 16 | 199.9 | 114.0 | 1.754 |  |
| decode 32 | B32 L1 sliding | 46 | 30.7 | 18.8 | 1.633 |  |
| decode 32 | B32 L1 full | 16 | 383.3 | 235.8 | 1.626 |  |
| verify 1 | B1 L8 sliding | 46 | 22.9 | 17.7 | 1.296 |  |
| verify 1 | B1 L8 full | 16 | 36.1 | 30.5 | 1.182 |  |
| verify 16 | B16 L8 sliding | 46 | 29.1 | 23.6 | 1.232 |  |
| verify 16 | B16 L8 full | 16 | 205.2 | 198.7 | 1.033 |  |
| verify 32 | B32 L8 sliding | 46 | 34.4 | 29.1 | 1.183 |  |
| verify 32 | B32 L8 full | 16 | 389.0 | 382.7 | 1.017 |  |
profile decode 1 sliding: void k1::k1_main<1>(k1::Params)                                   12.5 µs 100.0%      1 || total                                                             12.5 µs
profile decode 1 full: void k1::k1_main<1>(k1::Params)                                   15.1 µs  72.0%      1 || k1::k1_combine(k1::Params, int)                                    5.9 µs  28.0%      1 || total                                                             21.0 µs
profile decode 16 sliding: void k1::k1_main<1>(k1::Params)                                   13.3 µs 100.0%      1 || total                                                             13.3 µs
profile decode 16 full: void k1::k1_main<1>(k1::Params)                                  114.9 µs 100.0%      1 || total                                                            114.9 µs
profile decode 32 sliding: void k1::k1_main<1>(k1::Params)                                   16.4 µs 100.0%      1 || total                                                             16.4 µs
profile decode 32 full: void k1::k1_main<1>(k1::Params)                                  229.7 µs 100.0%      1 || total                                                            229.7 µs
profile verify 1 sliding: void cutlass::device_kernel<flash::enable_sm90_or_later<f...       7.6 µs  45.9%      1 || void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.8 µs  23.0%      1 || void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs  10.8%      1 || _qk_rms_norm_fwd_kernel                                            1.2 µs   7.3%      1 || void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.2 µs   6.9%      1
profile verify 1 full: void cutlass::device_kernel<flash::enable_sm90_or_later<f...      15.1 µs  52.8%      1 || void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       7.3 µs  25.7%      1 || void flash::prepare_varlen_num_blocks_kernel<1, true>(int...       2.8 µs   9.8%      1 || _qk_rms_norm_fwd_kernel                                            1.2 µs   4.3%      1 || void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.2 µs   4.0%      1
profile verify 16 sliding: void cutlass::device_kernel<flash::enable_sm90_or_later<f...      10.7 µs  51.2%      1 || void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.0 µs  19.1%      1 || _qk_rms_norm_fwd_kernel                                            1.9 µs   9.0%      1 || void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs   8.7%      1 || void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.4 µs   6.7%      1
profile verify 16 full: void cutlass::device_kernel<flash::enable_sm90_or_later<f...     190.0 µs  94.0%      1 || void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.1 µs   2.0%      1 || void flash::prepare_varlen_num_blocks_kernel<1, true>(int...       3.1 µs   1.6%      1 || _qk_rms_norm_fwd_kernel                                            2.0 µs   1.0%      1 || void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.7 µs   0.8%      1
profile verify 32 sliding: void cutlass::device_kernel<flash::enable_sm90_or_later<f...      15.0 µs  57.5%      1 || void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.1 µs  15.6%      1 || _qk_rms_norm_fwd_kernel                                            2.5 µs   9.4%      1 || void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.8 µs   6.9%      1 || void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.6 µs   6.3%      1
profile verify 32 full: void cutlass::device_kernel<flash::enable_sm90_or_later<f...     376.3 µs  96.7%      1 || void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.9 µs   1.0%      1 || void flash::prepare_varlen_num_blocks_kernel<2, true>(int...       3.5 µs   0.9%      1 || _qk_rms_norm_fwd_kernel                                            2.4 µs   0.6%      1 || void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.8 µs   0.5%      1
log:
  [k1] compiled in 35.5s
  [k1] v1 FALLBACK=False SELF_TEST=True built=True route={('sliding', 1): 1, ('full', 1): 1} fold={('sliding', 1): 1, ('sliding', 8): 1, ('full', 1): 8, ('full', 8): 8} min_chunk={('sliding', 1): 2, ('sliding', 8): 4, ('full', 1): 2, ('full', 8): 2} cps={('sliding', 1): 1, ('sliding', 8): 2, ('full', 
  [k1] k1_main<1> occupancy 2 CTAs/SM
  [k1] self-test fold B=2 ctx=3000 L=1 sliding page=1 kv_scale=1 heads=(16, 2): out rel 1.776e-03 worst row 1.799e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test sep  B=2 ctx=3000 L=1 sliding page=1 kv_scale=1 heads=(16, 2): out rel 1.776e-03 worst row 1.799e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=2 ctx=3000 L=1 sliding page=1 kv_scale=1 heads=(16, 2): out rel 1.776e-03 worst row 1.799e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] k1_main<8> occupancy 1 CTAs/SM
  [k1] self-test fold B=3 ctx=2047 L=8 sliding page=1 kv_scale=1 heads=(16, 2): out rel 1.650e-03 worst row 1.736e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test sep  B=3 ctx=2047 L=8 sliding page=1 kv_scale=1 heads=(16, 2): out rel 1.650e-03 worst row 1.736e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=3 ctx=2047 L=8 sliding page=1 kv_scale=1 heads=(16, 2): out rel 1.650e-03 worst row 1.736e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=2 ctx=3000 L=8 full page=64 kv_scale=1 heads=(16, 2): out rel 1.669e-03 worst row 1.812e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test sep  B=2 ctx=3000 L=8 full page=64 kv_scale=1 heads=(16, 2): out rel 1.669e-03 worst row 1.812e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=2 ctx=3000 L=8 full page=64 kv_scale=1 heads=(16, 2): out rel 1.669e-03 worst row 1.812e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=2 ctx=2500 L=1 full page=64 kv_scale=0.015625 heads=(16, 2): out rel 1.699e-03 worst row 1.768e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test sep  B=2 ctx=2500 L=1 full page=64 kv_scale=0.015625 heads=(16, 2): out rel 1.699e-03 worst row 1.768e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=2 ctx=2500 L=1 full page=64 kv_scale=0.015625 heads=(16, 2): out rel 1.699e-03 worst row 1.768e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=2 ctx=1800 L=8 sliding page=1 kv_scale=0.015625 heads=(16, 2): out rel 1.658e-03 worst row 1.762e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test sep  B=2 ctx=1800 L=8 sliding page=1 kv_scale=0.015625 heads=(16, 2): out rel 1.658e-03 worst row 1.762e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=2 ctx=1800 L=8 sliding page=1 kv_scale=0.015625 heads=(16, 2): out rel 1.658e-03 worst row 1.762e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=2 ctx=20 L=8 full page=1 kv_scale=1 heads=(16, 2): out rel 1.638e-03 worst row 1.754e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test sep  B=2 ctx=20 L=8 full page=1 kv_scale=1 heads=(16, 2): out rel 1.638e-03 worst row 1.754e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=2 ctx=20 L=8 full page=1 kv_scale=1 heads=(16, 2): out rel 1.638e-03 worst row 1.754e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=1 ctx=9 L=1 sliding page=64 kv_scale=1 heads=(16, 2): out rel 1.691e-03 worst row 1.691e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test sep  B=1 ctx=9 L=1 sliding page=64 kv_scale=1 heads=(16, 2): out rel 1.691e-03 worst row 1.691e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=1 ctx=9 L=1 sliding page=64 kv_scale=1 heads=(16, 2): out rel 1.691e-03 worst row 1.691e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=2 ctx=600 L=8 full page=64 kv_scale=1 heads=(128, 16): out rel 1.656e-03 worst row 1.676e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test sep  B=2 ctx=600 L=8 full page=64 kv_scale=1 heads=(128, 16): out rel 1.656e-03 worst row 1.676e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=2 ctx=600 L=8 full page=64 kv_scale=1 heads=(128, 16): out rel 1.656e-03 worst row 1.676e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=3 ctx=1500 L=1 full page=64 kv_scale=1 heads=(128, 16): out rel 1.743e-03 worst row 1.755e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test sep  B=3 ctx=1500 L=1 full page=64 kv_scale=1 heads=(128, 16): out rel 1.743e-03 worst row 1.755e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=3 ctx=1500 L=1 full page=64 kv_scale=1 heads=(128, 16): out rel 1.743e-03 worst row 1.755e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=1 ctx=9000 L=1 full page=1 kv_scale=1 heads=(16, 2): out rel 1.672e-03 worst row 1.672e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test sep  B=1 ctx=9000 L=1 full page=1 kv_scale=1 heads=(16, 2): out rel 1.672e-03 worst row 1.672e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=1 ctx=9000 L=1 full page=1 kv_scale=1 heads=(16, 2): out rel 1.672e-03 worst row 1.672e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=1 ctx=5000 L=8 full page=1 kv_scale=1 heads=(16, 2): out rel 1.688e-03 worst row 1.760e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test sep  B=1 ctx=5000 L=8 full page=1 kv_scale=1 heads=(16, 2): out rel 1.688e-03 worst row 1.760e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test fold B=1 ctx=5000 L=8 full page=1 kv_scale=1 heads=(16, 2): out rel 1.688e-03 worst row 1.760e-03; new K 0.000e+00 V 0.000e+00; counters 0
  [k1] self-test passed in 0.7s

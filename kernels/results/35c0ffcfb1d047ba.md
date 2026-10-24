# 35c0ffcfb1d047ba muse_moe_w4a16 check status=done/ok passed=True
speedup=1.3564823385078981 ci=[1.3361351034302795, 1.3768295735855165] score_us=9914.246295895799 baseline_us=13448.5 run_s=96 error=None
tests: 5/5 pass
| load | shape | w | SGLang µs | ours µs | speedup | flags |
|---|---|---|---|---|---|---|
| decode 1 | B1 L1 | 60 | 76.6 | 50.6 | 1.513 |  |
| decode 16 | B16 L1 | 60 | 194.8 | 148.6 | 1.311 |  |
| decode 32 | B32 L1 | 60 | 205.1 | 187.8 | 1.092 |  |
| verify 1 | B8 L1 | 60 | 150.6 | 109.4 | 1.376 |  |
| verify 16 | B128 L1 | 60 | 458.7 | 310.5 | 1.477 |  |
| verify 32 | B256 L1 | 60 | 623.5 | 440.5 | 1.415 |  |
profile decode 1 : void k2a2::k2a2_kernel<k2a2::Cfg<1, 4, 0, 1, 8192, 4> >(k...      39.6 µs 100.0%      1 || total                                                             39.6 µs
profile decode 16 : void k2a2::k2a2_kernel<k2a2::Cfg<1, 4, 0, 1, 8192, 4> >(k...     159.3 µs 100.0%      2 || total                                                            159.3 µs
profile decode 32 : void k2a2::k2a2_kernel<k2a2::Cfg<1, 4, 0, 1, 8192, 4> >(k...     171.6 µs 100.0%      1 || total                                                            171.6 µs
profile verify 1 : void k2a2::k2a2_kernel<k2a2::Cfg<1, 4, 0, 1, 8192, 4> >(k...      97.9 µs 100.0%      2 || total                                                             97.9 µs
profile verify 16 : void k2a2::k2a2_kernel<k2a2::Cfg<2, 4, 0, 1, 16384, 4> >(...     281.7 µs 100.0%      1 || total                                                            281.7 µs
profile verify 32 : void k2a2::k2a2_kernel<k2a2::Cfg<2, 4, 0, 1, 16384, 4> >(...     406.3 µs 100.0%      1 || total                                                            406.3 µs

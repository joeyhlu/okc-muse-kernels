# ff4ae9cda6f34ffc muse_moe_w4a16 check status=done/ok passed=True
speedup=0.8766763921355625 ci=[0.863526246253529, 0.8898265380175958] score_us=15340.324115766114 baseline_us=13448.5 run_s=96 error=None
tests: 5/5 pass
| load | shape | w | SGLang µs | ours µs | speedup | flags |
|---|---|---|---|---|---|---|
| decode 1 | B1 L1 | 60 | 76.1 | 57.5 | 1.324 |  |
| decode 16 | B16 L1 | 60 | 186.9 | 222.2 | 0.841 |  |
| decode 32 | B32 L1 | 60 | 204.2 | 301.6 | 0.677 |  |
| verify 1 | B8 L1 | 60 | 150.1 | 163.1 | 0.920 |  |
| verify 16 | B128 L1 | 60 | 471.2 | 604.0 | 0.780 |  |
| verify 32 | B256 L1 | 60 | 614.2 | 732.5 | 0.839 |  |
profile decode 1 : void k2a::gemm_kernel<1, 1>(uint4 const*, unsigned int co...      24.5 µs  46.9%      1 || void k2a::gemm_kernel<2, 1>(uint4 const*, unsigned int co...      19.7 µs  37.8%      1 || k2a::final_kernel(float const*, int const*, int const*, f...       6.3 µs  12.0%      1 || k2a::route_kernel(int const*, int*, int, int, int)                 1.7 µs   3.3%      1 || total                                                             52.3 µs
profile decode 16 : void k2a::gemm_kernel<1, 1>(uint4 const*, unsigned int co...     114.0 µs  61.2%      1 || void k2a::gemm_kernel<2, 1>(uint4 const*, unsigned int co...      64.9 µs  34.9%      1 || k2a::final_kernel(float const*, int const*, int const*, f...       4.8 µs   2.6%      1 || k2a::route_kernel(int const*, int*, int, int, int)                 2.4 µs   1.3%      1 || total                                                            186.1 µs
profile decode 32 : void k2a::gemm_kernel<1, 2>(uint4 const*, unsigned int co...     156.8 µs  61.7%      1 || void k2a::gemm_kernel<2, 2>(uint4 const*, unsigned int co...      87.8 µs  34.6%      1 || k2a::final_kernel(float const*, int const*, int const*, f...       6.3 µs   2.5%      1 || k2a::route_kernel(int const*, int*, int, int, int)                 3.1 µs   1.2%      1 || total                                                            253.9 µs
profile verify 1 : void k2a::gemm_kernel<1, 1>(uint4 const*, unsigned int co...      90.6 µs  60.1%      1 || void k2a::gemm_kernel<2, 1>(uint4 const*, unsigned int co...      52.8 µs  35.0%      1 || k2a::final_kernel(float const*, int const*, int const*, f...       5.5 µs   3.6%      1 || k2a::route_kernel(int const*, int*, int, int, int)                 2.0 µs   1.3%      1 || total                                                            150.9 µs
profile verify 16 : void k2a::gemm_kernel<1, 4>(uint4 const*, unsigned int co...     375.6 µs  66.5%      1 || void k2a::gemm_kernel<2, 4>(uint4 const*, unsigned int co...     173.3 µs  30.7%      1 || k2a::route_kernel(int const*, int*, int, int, int)                 9.0 µs   1.6%      1 || k2a::final_kernel(float const*, int const*, int const*, f...       6.8 µs   1.2%      1 || total                                                            564.8 µs
profile verify 32 : void k2a::gemm_kernel<1, 4>(uint4 const*, unsigned int co...     501.9 µs  65.5%      1 || void k2a::gemm_kernel<2, 4>(uint4 const*, unsigned int co...     239.4 µs  31.3%      1 || k2a::route_kernel(int const*, int*, int, int, int)                16.4 µs   2.1%      1 || k2a::final_kernel(float const*, int const*, int const*, f...       8.2 µs   1.1%      1 || total                                                            765.9 µs
log:
  ptxas info    : Used 106 registers, used 1 barriers, 16776 bytes smem
  ptxas info    : Compile time = 113.207 ms
  ptxas info    : Compiling entry function '_ZN3k2a11gemm_kernelILi2ELi1EEEvPK5uint4PKjPKfPK13__nv_bfloat16PKiPS8_Pfi' for 'sm_90'
  ptxas info    : Function properties for _ZN3k2a11gemm_kernelILi2ELi1EEEvPK5uint4PKjPKfPK13__nv_bfloat16PKiPS8_Pfi
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 94 registers, used 1 barriers, 8584 bytes smem
  ptxas info    : Compile time = 99.064 ms
  ptxas info    : Compiling entry function '_ZN3k2a11gemm_kernelILi1ELi4EEEvPK5uint4PKjPKfPK13__nv_bfloat16PKiPS8_Pfi' for 'sm_90'
  ptxas info    : Function properties for _ZN3k2a11gemm_kernelILi1ELi4EEEvPK5uint4PKjPKfPK13__nv_bfloat16PKiPS8_Pfi
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 119 registers, used 1 barriers, 33156 bytes smem
  ptxas info    : Compile time = 114.537 ms
  ptxas info    : Compiling entry function '_ZN3k2a11gemm_kernelILi1ELi2EEEvPK5uint4PKjPKfPK13__nv_bfloat16PKiPS8_Pfi' for 'sm_90'
  ptxas info    : Function properties for _ZN3k2a11gemm_kernelILi1ELi2EEEvPK5uint4PKjPKfPK13__nv_bfloat16PKiPS8_Pfi
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 101 registers, used 1 barriers, 16772 bytes smem
  ptxas info    : Compile time = 83.298 ms
  ptxas info    : Compiling entry function '_ZN3k2a11gemm_kernelILi1ELi1EEEvPK5uint4PKjPKfPK13__nv_bfloat16PKiPS8_Pfi' for 'sm_90'
  ptxas info    : Function properties for _ZN3k2a11gemm_kernelILi1ELi1EEEvPK5uint4PKjPKfPK13__nv_bfloat16PKiPS8_Pfi
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 93 registers, used 1 barriers, 8580 bytes smem
  ptxas info    : Compile time = 70.316 ms
  ptxas info    : Compiling entry function '_ZN3k2a12final_kernelEPKfPKiS3_S1_S1_P13__nv_bfloat16ii' for 'sm_90'
  ptxas info    : Function properties for _ZN3k2a12final_kernelEPKfPKiS3_S1_S1_P13__nv_bfloat16ii
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 66 registers, used 1 barriers, 64 bytes smem
  ptxas info    : Compile time = 17.134 ms
  ptxas info    : Compiling entry function '_ZN3k2a12route_kernelEPKiPiiii' for 'sm_90'
  ptxas info    : Function properties for _ZN3k2a12route_kernelEPKiPiiii
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 31 registers, used 1 barriers, 256 bytes smem
  ptxas info    : Compile time = 16.199 ms
  [k2a] compiled in 38.5s
  [k2a] self-test bs=1 skew=False: rel 3.889e-03 worst row 3.889e-03
  [k2a] self-test bs=5 skew=False: rel 3.759e-03 worst row 3.814e-03
  [k2a] self-test bs=40 skew=False: rel 3.762e-03 worst row 3.929e-03
  [k2a] self-test bs=100 skew=False: rel 3.774e-03 worst row 3.911e-03
  [k2a] self-test bs=128 skew=True: rel 3.759e-03 worst row 3.953e-03
  [k2a] self-test bs=257 skew=False: rel 3.767e-03 worst row 3.967e-03
  [k2a] prepare done in 39.4s

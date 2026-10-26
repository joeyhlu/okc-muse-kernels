# SPDX-License-Identifier: Apache-2.0
# Humming contributors; modified source port of vllm-project/humming v0.1.12.
"""Indexed SM90 BF16/NVFP4 overlap lab: prepare(weights, tokens), run(data, state).

Exact pinned WGMMA and non-specialized kernel sources are embedded below.
Only local copies change: typed FP32 accumulator fences, split issue/wait, and
issue -> next-fragment prefetch -> wait/promote. Arithmetic/layout/epilogue and
all routing helpers remain the pinned adapter's wg16 control. No GPU use on
import. --source-audit requires only Python; a passing audit is NOT a valid
GPU prepare. Missing installed headers/compiler/imports fail explicitly.
"""

import dataclasses
import hashlib
import json
import re
from pathlib import Path

PINNED_VERSION = "0.1.12"
PINNED_COMMIT = "636ba85648c30ae2c2bfb335c9399593a67ecc1d"

_PINNED_WGMMA_SHA256 = 'e3b1e50fa31b96a2f86b4dbec7d83db7d940356d2be653876359754514c355be'
_PINNED_KERNEL_SHA256 = '9a021dffc15f2ba2f3c355b4e508aeb3722b351a8be8130ffd4530743df3edd3'
_HEADER_SHA256 = {'humming/arith/epilogue_arith.cuh': 'f9e7a9965feaf24e16b871556024ad0f5f4534a7e2fb2b30c79937aa1d0fbbd5',
 'humming/arith/exp_offset.cuh': 'd2c71a8538a6cb2a9364a05f8615b3001b749acea5756559d153d5efc1754729',
 'humming/arith/mainloop_arith.cuh': '315d2c6b0bc965dd77485a15b522c96585b11de15e39a69108cdb3b3de02f75b',
 'humming/datatype/base_conversion.cuh': '3014588bf09c475f6c5d8aa4a2d82a48d6a9e058891ec28b24b3b7b4abd0aabc',
 'humming/datatype/dequant.cuh': '0533cd1ce425a3c071babeed20bc69814057cc3a8bab7c6baf1f03f7436fe3f4',
 'humming/datatype/dequant_fused.cuh': 'e2b76158fbe2568e733334408110c2fc7dba8bfadc7bcc18917089e2a0f09d32',
 'humming/datatype/dequant_prepare.cuh': 'd1b17ca2bfef89a836f4afad5d18fad50f6c331aa8118221261b13f99bbc5f28',
 'humming/datatype/dequant_single.cuh': '30071115ede684bb33ece4250c50cf52255565a1060a7fef6f59cd00ef598687',
 'humming/datatype/dtypes.cuh': '2963f4a16a910739553727120404ee18e920f1e148663ab4c1fd5b79ef2d95dd',
 'humming/epilogue/gmem_writer.cuh': 'fb52647224da17c52a2bef0cbe6c5f48e52e4057c9a61269a985677b643bdca8',
 'humming/epilogue/pipeline.cuh': '730493c5a7a92176bf3a5d47d54596a663d734018ac02c2b26ebb2da3e292ed3',
 'humming/epilogue/smem_reducer.cuh': '4402c9a767082de66f2e3a4310a3bc2e8a8e29048b3cd00e12818cc7a4cc5048',
 'humming/epilogue/smem_writer.cuh': '898589a11b27b9e14c34a44dbd29e137a211854be02e9ca6769bce4f5822f002',
 'humming/memory/g2s_loader/loader_a.cuh': '6912afcc5c5fb38f5465bf11995931b0ec94cfd761ca4c39cbc961e609d54f97',
 'humming/memory/g2s_loader/loader_as.cuh': 'cd4c4698a31ae1e63563191dd9160fd972db7fe0e52f3bcc227d2ff07016530d',
 'humming/memory/g2s_loader/loader_b.cuh': 'f2dcc1925193244bf6ac9cffb56aa2881fb602d92ef9ba710460e9a7f09d550b',
 'humming/memory/g2s_loader/loader_bias.cuh': 'ad206e2dc2cd43269c8a235d6fc1f545e74d2891c0996a0dba0c6ce5191cc77b',
 'humming/memory/g2s_loader/loader_bs.cuh': 'a03b01ceb6a9e1b32711d8e05aa0998df430ecc640299024a9ab60e215e12e53',
 'humming/memory/g2s_loader/loader_bs2.cuh': '82119886299e445bc9998fdd3639712d63c82056317123b094ee30684901f542',
 'humming/memory/g2s_loader/loader_bzp.cuh': '50ada5ce67458633807a6d31c149839a248ecdfbe5413ea52abf0356bcda7781',
 'humming/memory/g2s_pipeline.cuh': '479ff827ecf40b82d4685b3a104d796ef41bb643a1d9cc398feabcc77ccbd43b',
 'humming/memory/s2r_loader/loader_a.cuh': '70a5a882b97d46a140c99389998c6748aa6c38cfbe9438bf614bbe9726082d70',
 'humming/memory/s2r_loader/loader_as.cuh': '00169872ddfd21427913dd9f419ac82684ff619718ff7e1046434b731328df46',
 'humming/memory/s2r_loader/loader_b.cuh': '7b7cd6560ebb3fa8a15565e39890a578e4ec4b12735b5e6977670976e62149ef',
 'humming/memory/s2r_loader/loader_bias.cuh': 'd3d16e41fdba0f997a2d390a8acbefe57cb48ebdb86c25de98eb28534f78e3f6',
 'humming/memory/s2r_loader/loader_bs.cuh': '21cc0742c22101d9b62dd5276d735e57147dfad3c1031bb4079eaf7cbf26f37f',
 'humming/memory/s2r_loader/loader_bzp.cuh': 'ff7e40b02a8575f182d374041693085f4065e92f8654091ab1d4aa8b83a91362',
 'humming/memory/s2r_pipeline.cuh': 'aecdc089bc0c7b5cd9a8d08b7c5252a379468ae4fdee02fc038fd9030e5ef675',
 'humming/mma/all.cuh': '849874698ff35ff6fa95f14576a45052b356716e17852f293471be2ebc56f77a',
 'humming/mma/mxmma.cuh': 'ce5f1fbadbff2de0ab2d13c7d45352d30f76543c2cbc3e90d3c81a82c07946ca',
 'humming/mma/wgmma.cuh': 'e3b1e50fa31b96a2f86b4dbec7d83db7d940356d2be653876359754514c355be',
 'humming/mma/wmma.cuh': '0bdac29794d4ac3190f5988a83def98564c49f2759858162a5568b38231a9395',
 'humming/scheduler.cuh': '24443b74d10ac799d805d2ffc840503142696f744a1b3882379c55a6efc1f2e2',
 'humming/utils/all.cuh': '7cfd604795f05da3598333a6d8e4638a3523134036966c12d07b85b190f79ccb',
 'humming/utils/base.cuh': '490064dfc6b867ea04b2b2af9dbd43b94c5f15fba0dc15598735366b7a893dce',
 'humming/utils/context.cuh': '3185879ca6330e8b83ba1df2d37507a2a8be8c1aef85bbec288a14b0959b957d',
 'humming/utils/enum.cuh': 'c4b4b70ccb66ef729d884fcda326720bb682ea1b4b65b22acd43e7924ebe63e6',
 'humming/utils/ptx/barrier.cuh': 'e1fbbbdd0a74230787e3fa504dcee6a53a384ecf121dc2b7186bdfcff36987bf',
 'humming/utils/ptx/legacy_load.cuh': '0645ed91075b0701a9b940d407024a8a17e28704bd637e4e01d9da7fa37a93db',
 'humming/utils/ptx/math.cuh': '54d2f41b8a17e4ef16711c02c47a738f82425c6c0c2463487853bc461dbd5228',
 'humming/utils/ptx/shared.cuh': '3c4a118670f5ca773bdf153be632b2f111dd8ace89130ca9f4ca3efed27892bd',
 'humming/utils/ptx/tma.cuh': 'a5fef42410f0c3f57b2054902fdfb3bf02e774165747861b7e117f1a1b47f29a',
 'humming/utils/ptx/warp.cuh': '3849fd913dddbe787aabae9e0e2644f48cceab76a3d8dbce64dd543a8068051c',
 'humming/utils/ptx/wgmma.cuh': '3df6e0cb3d09af60e283bc89ea3fd9e535fc5278f6c0ba5d05740bb9d907d9a3',
 'humming/utils/storage.cuh': '066b4c47b65b07c88ed975ccc5413fea67786e210f92c8032d1c7d8eb4ccd5f5'}

_PINNED_WGMMA = r"""#pragma once

#include <humming/utils/all.cuh>


template <uint32_t swizzle_bytes = 128>
CUDA_INLINE uint64_t make_wgmma_smem_desc(uint32_t addr) {
  static_assert(swizzle_bytes == 128 || swizzle_bytes == 64);

  constexpr uint64_t swizzle_type = swizzle_bytes == 128 ? 1 : 2;
  constexpr uint64_t stride = (swizzle_bytes * 8) >> 4;
  constexpr uint64_t desc_base = (swizzle_type << 62) | (stride << 32);

  uint64_t desc = desc_base;
  reinterpret_cast<uint32_t *>(&desc)[0] = (addr >> 4);

  return desc;
};


template <class Ctx, class ArithClass>
struct WGMMA {
public:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using MmaShape = typename Ctx::MmaShape;
  using SharedStorage = typename Ctx::SharedStorage;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;
  using CRegistersType = typename MmaOpClass::CRegisters;
  using CRegistersArrayType = CRegistersType[WarpShape::N * 4 / MmaShape::N][WarpShape::M / MmaShape::M];

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kSwizzleBytes = ElementA::kBits * BlockShape::K >= 1024 ? 128 : 64;
  static constexpr uint32_t kNumWarpShapeNSplits = WarpShape::N == ElementA::kBits * 2 ? 2 : 1;

  static constexpr bool kUsePackedKLayout = Ctx::kUsePackedKLayout;
  static constexpr uint32_t kPackedKFactor = Ctx::kPackedKFactor;
  static constexpr uint32_t kWarpIters = Ctx::kWarpIters;
  static constexpr uint32_t kNumKSlabs = WarpShape::K / kPartMmaShapeK;
  static constexpr uint32_t kRegsBKDim = kUsePackedKLayout ? kNumKSlabs : (kPartMmaShapeK / MmaShape::K);

  Ctx &ctx;
  ArithClass &arith;
  uint32_t regs_qb[2][ElementB::kBits * (16 / ElementA::kBits)];
  typename MmaOpClass::BRegisters regs_b[2][kUsePackedKLayout ? 1 : (WarpShape::N * 4 / MmaShape::N / kPackedKFactor)][kRegsBKDim];
  CRegistersArrayType regs_c[2];
  uint32_t smem_offset = 0;

  CUDA_INLINE
  WGMMA(Ctx &ctx, ArithClass &arith)
      : ctx(ctx), arith(arith) {
    constexpr uint32_t kSwizzleSizeK = kSwizzleBytes * 8 / ElementA::kBits;
    static_assert(kSwizzleSizeK >= WarpShape::K);

    const uint32_t row_offset = ctx.m_warp_offset();
    const uint32_t col_offset = ctx.k_warp_offset();

    smem_offset = row_offset * (kSwizzleBytes / 16);
    smem_offset += (col_offset % kSwizzleSizeK) * ElementA::kBits / 128;
    smem_offset += (col_offset / kSwizzleSizeK) * (BlockShape::M * kSwizzleBytes / 16);
    smem_offset = smem_offset * sizeof(int4);
  }

  CUDA_INLINE
  void zero_accum() {
    uint32_t *regs_c_ptr = regs_c_as_ptr();
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs_c) / 4; i++) {
      regs_c_ptr[i] = 0;
    };
  };

  CUDA_INLINE
  void transform_b(uint32_t buffer_id, uint32_t iter_id) {
    if constexpr (std::is_same<ElementA, ElementB>::value) return;

    if constexpr (kUseFusedE8m0Scale) {
      uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id]);
      fused_dequant_for_mxfp4<ElementA, WarpShape::N / 16, true>(regs_qb[buffer_id], regs_b_ptr, arith.bs[buffer_id]);
    } else {
      if constexpr (ElementB::kBits == 1 && kNumWarpShapeNSplits == 2) {
        regs_qb[buffer_id][0] = regs_qb[buffer_id][0] >> (ctx.warp_id() % 2 * (ElementA::kBits / 2));
      }

      constexpr uint32_t kTransformIters = kUsePackedKLayout ? kNumKSlabs : (WarpShape::N / (MmaShape::N / 4));
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kTransformIters; i++) {
        uint32_t *regs_b_ptr;
        if constexpr (kUsePackedKLayout) {
          regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][0][i]);
        } else {
          regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][i / kPackedKFactor][i % kPackedKFactor]);
        }
        uint32_t zp_index = kUsePackedKLayout ? iter_id : i;
        uint4 zp_vals = arith.prepare_zp_for_dequant(buffer_id, zp_index);
        uint32_t *zp_vals_ptr = reinterpret_cast<uint32_t *>(&zp_vals);
        dequant<ElementB, ElementA, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(regs_qb[buffer_id], regs_b_ptr, i, zp_vals_ptr);
        arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
      };
    }
  };

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    static_assert(WarpShape::M == MmaShape::M);
    static_assert(kPartMmaShapeK == MmaShape::K);
    uint32_t buffer_id = iter_id % 2;

    const uint32_t smem_base = cast_smem_ptr_to_uint(&ctx.smem);
    constexpr uint32_t kItersPerHalf = kWarpIters / kPackedKFactor;
    constexpr uint32_t kNumIters = kUsePackedKLayout ? 1 : (WarpShape::N / (MmaShape::N / 4) / kPackedKFactor);
    constexpr uint32_t kRunKLoop = kUsePackedKLayout ? kNumKSlabs : kPackedKFactor;

    uint32_t delta_m = kUsePackedKLayout ? iter_id : 0;
    uint32_t delta_j = final_regs_c_index() == 0 ? delta_m : 0;

    wgmma_fence();
    may_fence_regs(delta_j);

    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kRunKLoop; k++) {
      uint32_t k_slab = kUsePackedKLayout ? k : ((iter_id % kItersPerHalf) * kPackedKFactor + k);
      uint32_t smem_addr = smem_base + offsetof(SharedStorage, stages) + stage_id * sizeof(typename SharedStorage::StageStorage);
      smem_addr += k_slab * 2 * sizeof(int4) + smem_offset;
      uint64_t desc = make_wgmma_smem_desc<kSwizzleBytes>(smem_addr);

      bool scale_d = true;
      if constexpr (ElementA::kBits != 16 && Ctx::kInputScaleGroupSize > 0) {
        scale_d = (k_slab * kPartMmaShapeK) % Ctx::kInputScaleGroupSize > 0;
      }
      if constexpr (!kUseFusedE8m0Scale && ElementA::kBits != 16 && Ctx::kWeightScaleGroupSize > 0) {
        scale_d = scale_d && (k_slab * kPartMmaShapeK) % Ctx::kWeightScaleGroupSize > 0;
      }

      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++) {
        MmaOpClass::fma(desc, regs_b[buffer_id][j][k], regs_c[0][delta_j + j][0], scale_d);
      }
    }

    wgmma_commit();
    wgmma_wait<0>();
    may_fence_regs(delta_j);

    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kRunKLoop; k++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++) {
        arith.may_apply_as_and_bs_on_wgmma_c(regs_c_as_ptr(), j, k, iter_id, delta_m);
      }
    }
  };

  CUDA_INLINE void may_fence_regs(uint32_t delta_j) {
    if constexpr (final_regs_c_index() != 0) {
      constexpr uint32_t kNumIters = kUsePackedKLayout ? 1 : (WarpShape::N / (MmaShape::N / 4) / kPackedKFactor);
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++)
        fence_regs(regs_c[0][delta_j + j][0]);
    }
  }

  template <class T>
  CUDA_INLINE void fence_regs(T &regs) {
    PRAGMA_UNROLL
    for (uint32_t r = 0; r < sizeof(T) / 4; r++) {
      warpgroup_fence_operand(reinterpret_cast<uint32_t *>(regs)[r]);
    }
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_qb_as_ptr(uint32_t buffer_id) {
    if constexpr (std::is_same<ElementA, ElementB>::value) {
      return reinterpret_cast<T *>(regs_b[buffer_id]);
    } else {
      return reinterpret_cast<T *>(regs_qb[buffer_id]);
    };
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr(uint32_t buffer_id = 0) {
    return reinterpret_cast<T *>(regs_c[buffer_id]);
  };

  static constexpr uint32_t final_regs_c_index() {
    if constexpr (ElementA::kBits < 16 && Ctx::kInputScaleGroupSize > 0) return 1;
    if constexpr (ElementA::kBits < 16 && !kUseFusedE8m0Scale && (Ctx::kIsGroupWeightScale || Ctx::kIsBlockWeightScale)) return 1;
    return 0;
  };

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    return regs_c_as_ptr<T>(final_regs_c_index());
  };
};
"""

_PINNED_KERNEL = r"""#pragma once

#include <humming/scheduler.cuh>
#include <humming/utils/all.cuh>

#include <humming/arith/epilogue_arith.cuh>
#include <humming/arith/mainloop_arith.cuh>

#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#include <humming/mma/all.cuh>

#include <humming/datatype/dequant.cuh>


template <bool kUseTma>
class KernelTensorParamType {
public:
  using Type = std::conditional_t<kUseTma, CUtensorMap const, void *const>;
};

CUDA_INLINE const void *param_to_ptr(const CUtensorMap &x) { return &x; }
CUDA_INLINE const void *param_to_ptr(void *const &x) { return x; }


template <
    class MmaOpClass,
    class ProblemShape, class BlockShape, class WarpShape, class PadShape,
    class ElementA, class ElementB, class ElementC, class ElementBS,
    class LayerConfig, class ComputeConfig, class TuningConfig>
__global__ __launch_bounds__(TuningConfig::kNumThreads, TuningConfig::kNumCtasPerSm) void humming(
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaA>::Type A,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaB>::Type B,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaC>::Type C,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaAS>::Type AS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBS>::Type BS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBZP>::Type BZP,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBias>::Type Bias,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBS2>::Type BS2,
    const uint32_t *sorted_ids_ptr,
    const uint32_t *expert_ids_ptr,
    const uint32_t *num_tokens_padded_ptr,
    const uint32_t *expert_layout_ptr,
    CUtensorMap *tensor_map_buffer,
    int32_t *locks,
    uint32_t shape_m,
    uint32_t top_k,
    bool use_int64_expert_layout) {

  uint64_t debug_start_clock = debug_kernel_timer_start();
  constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  constexpr uint32_t kNumStages = TuningConfig::kNumStages;

  using SharedStorage = SharedStorage<
      MmaOpClass, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using Ctx = KernelContext<
      MmaOpClass, ProblemShape, BlockShape, WarpShape, PadShape,
      ElementA, ElementB, ElementC, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using Scheduler = Scheduler<Ctx>;
  using ProducerPipeline = ProducerPipeline<Ctx>;
  using ConsumerPipeline = ConsumerPipeline<Ctx>;
  using MainloopArithmetic = MainloopArithmetic<Ctx>;
  using EpilogueArithmetic = EpilogueArithmetic<Ctx>;
  using MMA = Mma<Ctx, MainloopArithmetic>;
  using Epilogue = EpiloguePipeline<Ctx, MMA, EpilogueArithmetic>;
  using S2RMemoryPipeline = S2RMemoryPipeline<Ctx, MMA, Epilogue>;

  extern __shared__ int4 shared_memory[];
  auto &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  const KernelParams params{
      shape_m, top_k, use_int64_expert_layout,
      param_to_ptr(A), param_to_ptr(B), param_to_ptr(AS), param_to_ptr(BS),
      param_to_ptr(BZP), param_to_ptr(Bias), param_to_ptr(C), param_to_ptr(BS2),
      sorted_ids_ptr, expert_ids_ptr, num_tokens_padded_ptr, expert_layout_ptr,
      tensor_map_buffer, locks};
  auto ctx = Ctx(smem, params);

  auto scheduler = Scheduler(ctx);
  auto mainloop_arith = MainloopArithmetic();
  auto epilogue_arith = EpilogueArithmetic();
  auto mma = MMA(ctx, mainloop_arith);
  auto epilogue = Epilogue(ctx, epilogue_arith);
  auto producer = ProducerPipeline(ctx);
  auto consumer = ConsumerPipeline(ctx);
  auto s2r_pipe = S2RMemoryPipeline(ctx, mma, epilogue);

  producer.init_mbarrier();
  mbarrier_init_sync<((TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB) > 1)>();

  while (scheduler.get_next_block()) {
    debug_kernel_timeout_check(debug_start_clock);
    mma.zero_accum();
    __syncthreads();

    uint32_t &slice_iters = scheduler.slice_iters;
    producer.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.k_block_id, scheduler.current_shape_m, scheduler.m_offset);
    epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.current_shape_m, scheduler.m_offset);
    epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);

    if constexpr (TuningConfig::kUseTmaC) {
      tma_wait_store_group<0, true>();
      __syncthreads();
    }
    producer.template load_stage<true, true>(0);
    PRAGMA_UNROLL
    for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
      producer.load_stage(stage_id, stage_id < slice_iters);
    };

    consumer.template wait_stage<true>(kNumStages);
    s2r_pipe.template load_stage_iter<true>(0, 0);
    mma.transform_b(0, 0);

    while (slice_iters) {
      debug_kernel_timeout_check(debug_start_clock);
      PRAGMA_UNROLL
      for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
        debug_kernel_timeout_check(debug_start_clock);
        if (slice_iters == 1) producer.load_channel();
        PRAGMA_UNROLL
        for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
          mma.run(stage_id, warp_iter_id);
          if (warp_iter_id == Ctx::kWarpIters - 2) {
            if constexpr (kNumStages == 2) {
              __syncthreads();
              if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            } else {
              __syncthreads();
              producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
              if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            }
          }

          mma.transform_b(
              (warp_iter_id + 1) % 2,
              (warp_iter_id + 1) % Ctx::kWarpIters);
        }

        if constexpr (kNumStages == 2) producer.load_stage(stage_id, slice_iters > kNumStages);
        slice_iters--;
        if (!slice_iters) break;
      };
    };

    consumer.wait_channel();
    s2r_pipe.load_channel(scheduler.slice_id);
    __syncthreads();
    epilogue.call(mma.final_regs_c_as_ptr());
  }

  __syncthreads();
  if constexpr (TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB > 1) {
    asm volatile("barrier.cluster.arrive;\n");
    asm volatile("barrier.cluster.wait;\n");
  }
};
"""


def _once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("pinned source edit anchor mismatch: " + old[:70])
    return source.replace(old, new, 1)


def _strip_includes(source):
    return re.sub(r"^#(?:pragma once|include [^\n]+)\n", "", source, flags=re.M)


def _patched_sources():
    w = re.sub(r"\bWGMMA\b", "OverlapWGMMA", _PINNED_WGMMA)
    w = w.replace("make_wgmma_smem_desc", "overlap_make_wgmma_smem_desc")
    w = _once(w, "void run(uint32_t stage_id, uint32_t iter_id)",
              "void issue(uint32_t stage_id, uint32_t iter_id)")
    w = _once(w, "    wgmma_fence();\n    may_fence_regs(delta_j);",
              "    fence_accumulators(delta_j);\n    wgmma_fence();")
    w = _once(w, "    wgmma_commit();\n    wgmma_wait<0>();\n    may_fence_regs(delta_j);", r"""    wgmma_commit();
  }

  CUDA_INLINE
  void wait_and_promote(uint32_t stage_id, uint32_t iter_id) {
    constexpr uint32_t kNumIters = kUsePackedKLayout ? 1 : (WarpShape::N / (MmaShape::N / 4) / kPackedKFactor);
    constexpr uint32_t kRunKLoop = kUsePackedKLayout ? kNumKSlabs : kPackedKFactor;
    uint32_t delta_m = kUsePackedKLayout ? iter_id : 0;
    uint32_t delta_j = final_regs_c_index() == 0 ? delta_m : 0;
    wgmma_wait<0>();
    fence_accumulators(delta_j);""")
    old_fence = r"""  CUDA_INLINE void may_fence_regs(uint32_t delta_j) {
    if constexpr (final_regs_c_index() != 0) {
      constexpr uint32_t kNumIters = kUsePackedKLayout ? 1 : (WarpShape::N / (MmaShape::N / 4) / kPackedKFactor);
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++)
        fence_regs(regs_c[0][delta_j + j][0]);
    }
  }"""
    new_fence = r"""  CUDA_INLINE void fence_accumulators(uint32_t delta_j) {
    constexpr uint32_t kNumIters = kUsePackedKLayout ? 1 : (WarpShape::N / (MmaShape::N / 4) / kPackedKFactor);
    PRAGMA_UNROLL
    for (uint32_t j = 0; j < kNumIters; j++)
      fence_regs(regs_c[0][delta_j + j][0]);
  }"""
    w = _once(w, old_fence, new_fence)
    w = _once(w, "warpgroup_fence_operand(reinterpret_cast<uint32_t *>(regs)[r]);",
              "overlap_fence_operand(regs[r]);")
    w = _once(w, "    static_assert(WarpShape::M == MmaShape::M);", r"""    static_assert(WarpShape::M == MmaShape::M);
    static_assert(ElementA::kBits == 16 && Ctx::kWeightScaleGroupSize == 16);
    static_assert(Ctx::kUseWgmma && Ctx::kIsIndexedGemm && !Ctx::kUseWarpSpec);
    static_assert(!kUsePackedKLayout && kWarpIters >= 2 && kWarpIters % 2 == 0);""")
    k = _once(_PINNED_KERNEL, " void humming(", " void overlap_kernel(")
    k = _once(k, "using MMA = Mma<Ctx, MainloopArithmetic>;",
              "using MMA = OverlapWGMMA<Ctx, MainloopArithmetic>;")
    k = _once(k, "          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);\n          mma.run(stage_id, warp_iter_id);",
              "          mma.issue(stage_id, warp_iter_id);\n          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);\n          mma.wait_and_promote(stage_id, warp_iter_id);")
    includes = list(dict.fromkeys(re.findall(r"^#include <(humming/[^>]+)>", _PINNED_WGMMA + _PINNED_KERNEL, re.M)))
    return includes, _strip_includes(w), _strip_includes(k)


def source_audit():
    for text, expected in ((_PINNED_WGMMA, _PINNED_WGMMA_SHA256), (_PINNED_KERNEL, _PINNED_KERNEL_SHA256)):
        assert hashlib.sha256(text.encode()).hexdigest() == expected
    includes, w, k = _patched_sources()
    assert w.count("wgmma_commit();") == w.count("wgmma_wait<0>();") == 1
    assert w.index("fence_accumulators(delta_j);") < w.index("wgmma_fence();")
    assert "overlap_fence_operand(regs[r]);" in w and "may_fence_regs" not in w
    assert k.index("mma.issue(") < k.index("s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1)") < k.index("mma.wait_and_promote(")
    assert "mma.run(" not in k and "using MMA = OverlapWGMMA" in k
    # Schedules, global scale epilogue, dequantization, producer refills and CTA barriers are unchanged.
    for token in ("producer.load_stage(", "consumer.wait_stage(", "__syncthreads();", "epilogue.call("):
        assert k.count(token) == _PINNED_KERNEL.count(token)
    assert "__trap" not in w + k
    # Regress the standalone class-wrapper bug without importing Humming/CUDA.
    class DeclarationProbe:
        def to_cpp_str(self, cls):
            return "  static constexpr unsigned kAccessProbe = 1u;"
    config_types = tuple(type(name, (), {}) for name in ("LayerConfig", "ComputeConfig", "TuningConfig"))
    declarations = _config_declarations(DeclarationProbe(), config_types)
    _assert_public_config_declarations(declarations)
    assert declarations.count("kAccessProbe") == 3
    assert "inline KernelFn kernel_ptr()" in _CUDA_WRAPPER
    assert "return &fresh_humming_overlap::overlap_kernel<" in _CUDA_WRAPPER
    assert "inline auto kernel_ptr()" not in _CUDA_WRAPPER
    return {"status": "source_audit_only", "gpu_prepare_validated": False,
            "pinned_commit": PINNED_COMMIT, "vendor_bytes": len(w.encode()) + len(k.encode()),
            "installed_header_count": len(_HEADER_SHA256), "include_roots": includes,
            "public_config_declarations": True, "explicit_kernel_pointer_abi": True,
            "configuration": "indexed wg16: BM16 BN128 BK64, warp16/32/64, stages4, CTAs/SM2, StreamK"}


def dependency_info():
    from importlib.util import find_spec
    from importlib.metadata import PackageNotFoundError, version
    result = {"status": "dependency_probe_only", "gpu_prepare_validated": False}
    for name in ("torch", "triton", "humming", "fresh_humming_adapter"):
        try:
            spec = find_spec(name)
            result[name] = spec is not None
        except (ImportError, ValueError):
            result[name] = False
    try:
        result["humming_version"] = version("humming-kernels")
    except PackageNotFoundError:
        result["humming_version"] = None
    return result


def _check_headers(humming):
    include = Path(humming.__file__).resolve().parent / "include"
    for name, expected in _HEADER_SHA256.items():
        path = include / name
        if not path.is_file():
            raise RuntimeError(f"overlap installed-header dependency unavailable: {path}")
        if hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest() != expected:
            raise RuntimeError(f"overlap installed-header version mismatch (requires exact v0.1.12): {path}")
    return include


def _config_declarations(spec, config_classes):
    # BaseHummingConfig's include_class_name=True wrapper has private members.
    # Humming's own CODE_TEMPLATE supplies public: separately; do the same here.
    return "\n".join(
        f"class {cls.__name__} {{\npublic:\n{spec.to_cpp_str(cls)}\n}};"
        for cls in config_classes
    )


def _assert_public_config_declarations(declarations):
    for name in ("LayerConfig", "ComputeConfig", "TuningConfig"):
        if not re.search(r"\bclass\s+" + name + r"\s*\{\s*public\s*:", declarations):
            raise RuntimeError(f"overlap codegen requires public {name} members")
    if re.search(r"\b(?:private|protected)\s*:", declarations):
        raise RuntimeError("overlap config declarations must stay public")


def _make_config(rows, k, experts, H):
    from humming import dtypes
    from humming.config import LayerConfig, ComputeConfig, TuningConfig, MmaType
    from humming.kernel.humming import HummingKernel
    from humming.schema import HummingInputSchema, HummingWeightSchema
    from humming.transform import prepare_layer_config
    from humming.utils.smem import estimate_smem_size_config
    import torch

    schema = HummingWeightSchema(b_dtype=dtypes.float4e2m1, bs_dtype=dtypes.float8e4m3,
                                weight_scale_group_size=16, weight_scale_type="group",
                                weight_scale_2_type="tensor")
    layer = prepare_layer_config(shape_n=rows, shape_k=k, weight_schema=schema,
                                 input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),
                                 num_experts=experts, torch_dtype=torch.bfloat16)
    layer = dataclasses.replace(layer, mma_type=MmaType.WGMMA)
    tuning = H._preset_tuning(H.PRESETS["wg16"])

    @dataclasses.dataclass(kw_only=True)
    class SourceConfig(LayerConfig, ComputeConfig, TuningConfig):
        def __post_init__(self):
            LayerConfig.__post_init__(self)
            ComputeConfig.__post_init__(self)
            TuningConfig.__post_init__(self)
            self.sm_version = 90

    kwargs = {f.name: getattr(layer, f.name) for f in dataclasses.fields(layer)}
    spec = SourceConfig(**kwargs, gemm_type="indexed", use_f16_accum=False, **tuning)
    # Use only generator/validator methods. Never construct or patch HummingKernel,
    # call its JIT/prepare_kernels, or register a library cubin.
    for name in ("check_shape", "check_dtype", "check_scale", "check_config"):
        getattr(HummingKernel, name)(spec)
    spec.mma_op_class = HummingKernel.select_mma_op_class(spec)
    spec.estimated_smem_size = estimate_smem_size_config(spec, spec, spec)
    metadata = H._kernel_metadata(spec, layer)
    assert metadata["ptx_instruction"] == "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16"
    assert spec.num_threads == 128 and not spec.use_tma and not spec.use_warp_spec
    assert not spec.use_int_weight_scale and not spec.use_fused_e8m0_scale
    macros = "\n".join(spec.to_macro_cpp_str(cls) for cls in (LayerConfig, ComputeConfig, TuningConfig))
    classes = _config_declarations(spec, (LayerConfig, ComputeConfig, TuningConfig))
    return layer, spec, tuning, metadata, macros, classes


_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>
void overlap_run(torch::Tensor a, torch::Tensor b, torch::Tensor bs, torch::Tensor gs,
                 torch::Tensor c, torch::Tensor sorted, torch::Tensor experts,
                 torch::Tensor padded, torch::Tensor locks, int64_t grid, int64_t top_k);
std::vector<int64_t> overlap_info();
"""

_CUDA_WRAPPER = r"""
namespace fresh_humming_overlap_config {
class OverlapMmaOpClass {
public:
@MMA@
};
@CLASSES@
using Storage = ::SharedStorage<OverlapMmaOpClass, Shape<16,128,64>, Shape<16,32,64>,
                              BFloat16, Float4E2M1, Float8E4M3,
                              LayerConfig, ComputeConfig, TuningConfig>;
static constexpr int kSmemBytes = sizeof(Storage);
// Exact pinned ABI for the all-pointer, no-TMA indexed specialization. An
// explicit target type avoids auto deduction from a CUDA function-template id.
using KernelFn = void (*)(void *, void *, void *, void *, void *, void *, void *, void *,
                         const uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *,
                         CUtensorMap *, int32_t *, uint32_t, uint32_t, bool);
inline KernelFn kernel_ptr() {
  return &fresh_humming_overlap::overlap_kernel<
      OverlapMmaOpClass, Shape<0,@N@,@K@>, Shape<16,128,64>, Shape<16,32,64>, Shape<0,0,0>,
      BFloat16, Float4E2M1, BFloat16, Float8E4M3, LayerConfig, ComputeConfig, TuningConfig>;
}
}

std::vector<int64_t> overlap_info() {
  using namespace fresh_humming_overlap_config;
  auto kernel = kernel_ptr();
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kernel));
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, TuningConfig::kNumThreads, kSmemBytes));
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)fa.sharedSizeBytes,
          fa.maxDynamicSharedSizeBytes, fa.maxThreadsPerBlock, active, kSmemBytes};
}

void overlap_run(torch::Tensor a, torch::Tensor b, torch::Tensor bs, torch::Tensor gs,
                 torch::Tensor c, torch::Tensor sorted, torch::Tensor experts,
                 torch::Tensor padded, torch::Tensor locks, int64_t grid, int64_t top_k) {
  using namespace fresh_humming_overlap_config;
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16 && a.is_contiguous(), "BF16 contiguous CUDA A required");
  TORCH_CHECK(a.dim() == 2 && a.size(1) == @K@ && c.size(1) == @N@, "overlap projection shape mismatch");
  TORCH_CHECK((top_k == 8 || top_k == 1) && c.size(0) == a.size(0) * top_k, "indexed output rows mismatch");
  TORCH_CHECK(a.size(0) <= UINT32_MAX && grid > 0 && grid <= UINT32_MAX, "launch dimension overflow");
  auto kernel = kernel_ptr();
  void *a_ptr = a.data_ptr(), *b_ptr = b.data_ptr(), *c_ptr = c.data_ptr();
  void *bs_ptr = bs.data_ptr(), *gs_ptr = gs.data_ptr();
  void *as_ptr = nullptr, *bzp_ptr = nullptr, *bias_ptr = nullptr;
  const uint32_t *sorted_ptr = static_cast<const uint32_t *>(sorted.data_ptr());
  const uint32_t *experts_ptr = static_cast<const uint32_t *>(experts.data_ptr());
  const uint32_t *padded_ptr = static_cast<const uint32_t *>(padded.data_ptr());
  const uint32_t *layout_ptr = nullptr;
  CUtensorMap *maps_ptr = nullptr;
  int32_t *locks_ptr = static_cast<int32_t *>(locks.data_ptr());
  uint32_t shape_m = static_cast<uint32_t>(a.size(0));
  uint32_t top_k_u32 = static_cast<uint32_t>(top_k);
  bool use_int64_layout = false;
  void *args[] = {&a_ptr, &b_ptr, &c_ptr, &as_ptr, &bs_ptr, &bzp_ptr, &bias_ptr, &gs_ptr,
                  &sorted_ptr, &experts_ptr, &padded_ptr, &layout_ptr, &maps_ptr, &locks_ptr,
                  &shape_m, &top_k_u32, &use_int64_layout};
  C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void *>(kernel), dim3((unsigned)grid),
                                  dim3(TuningConfig::kNumThreads), args, kSmemBytes,
                                  at::cuda::getCurrentCUDAStream(a.get_device())));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _render_cuda(spec, macros, classes):
    _assert_public_config_declarations(classes)
    includes, w, k = _patched_sources()
    source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
""" + macros + "\n" + "\n".join(f"#include <{name}>" for name in includes)
    source += r"""
namespace fresh_humming_overlap {
CUDA_INLINE void overlap_fence_operand(float &reg) {
  asm volatile("" : "+f"(reg) :: "memory");
}
CUDA_INLINE void overlap_fence_operand(uint32_t &reg) {
  asm volatile("" : "+r"(reg) :: "memory");
}
""" + w + "\n" + k + "\n}\n"
    wrapper = _CUDA_WRAPPER.replace("@MMA@", spec.mma_op_class.to_cpp_str()).replace("@CLASSES@", classes)
    wrapper = wrapper.replace("@N@", str(spec.shape_n)).replace("@K@", str(spec.shape_k))
    source += wrapper
    assert len(source.encode()) < 1_000_000 and "@MMA@" not in source
    return source


_CUDA_FLAGS = ["-O3", "-lineinfo", "--expt-relaxed-constexpr", "--use_fast_math",
               "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v", "-Xptxas=-warn-spills",
               "--ptxas-options=--register-usage-level=10"]
_RESOURCE_KEYS = ("registers_per_thread", "local_bytes_per_thread", "static_smem_bytes",
                  "max_dynamic_smem_bytes", "max_threads_per_block", "resident_ctas_per_sm", "dynamic_smem_bytes")


def _build(spec, macros, classes, include):
    from torch.utils.cpp_extension import CUDA_HOME, load_inline
    if CUDA_HOME is None:
        raise RuntimeError("overlap compiler dependency unavailable: PyTorch CUDA_HOME is unset")
    import os
    nvcc = Path(CUDA_HOME) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    if not nvcc.is_file():
        raise RuntimeError(f"overlap compiler dependency unavailable: {nvcc}")
    cuda = _render_cuda(spec, macros, classes)
    digest = hashlib.sha256((_CPP_SOURCE + cuda + str(include) + repr(_CUDA_FLAGS)).encode()).hexdigest()
    try:
        module = load_inline(name="fresh_humming_overlap_" + digest[:16], cpp_sources=_CPP_SOURCE,
                             cuda_sources=cuda, functions=["overlap_run", "overlap_info"],
                             extra_include_paths=[str(include)], extra_cflags=["-O3"],
                             extra_cuda_cflags=_CUDA_FLAGS, verbose=True)
    except Exception as exc:
        raise RuntimeError("overlap load_inline compile/load failed; source audit does not validate GPU prepare") from exc
    resources = dict(zip(_RESOURCE_KEYS, map(int, module.overlap_info())))
    assert resources["dynamic_smem_bytes"] == spec.estimated_smem_size
    return module, {"source_sha256": digest, "cuda_source_bytes": len(cuda.encode()),
                    "extension_path": module.__file__, "compiler": str(nvcc), "cuda_flags": list(_CUDA_FLAGS),
                    "compiler_backend": "torch load_inline / nvcc", "pipeline": "issue-prefetch-wait",
                    "compiled_resources": resources, "has_local_memory": resources["local_bytes_per_thread"] > 0,
                    "ptxas_warnings": "inspect verbose build stdout/stderr; cached builds may emit no diagnostics",
                    "automatic_serialization_verified": False}


class _Launch:
    def __init__(self, module, grid):
        self.module, self.grid = module, grid

    def __call__(self, *, configs, inputs, weight, weight_scale, weight_scale_2, outputs,
                 sorted_ids, expert_ids, num_tokens_padded, expert_layout, locks, top_k, valid_shape_m):
        if expert_layout is not None:
            raise ValueError("overlap supports indexed wg16 only")
        # All pointers are extracted by C++ from these live tensors on this call.
        self.module.overlap_run(inputs, weight, weight_scale, weight_scale_2, outputs,
                                sorted_ids, expert_ids, num_tokens_padded, locks, self.grid, top_k)


def _prepare_projection(packed, scales, globals_, tokens, H, include, torch, tr):
    from humming.transform import transform_humming_tensors
    experts, rows, k_bytes = packed.shape
    layer, spec, tuning, metadata, macros, classes = _make_config(rows, k_bytes * 2, experts, H)
    tensors = transform_humming_tensors(layer, {"weight": packed.contiguous().view(torch.int32),
                                               "weight_scale": scales.contiguous(),
                                               "weight_scale_2": globals_.contiguous()})
    gs = tensors["weight_scale_2"]
    assert gs.dtype == torch.float32 and tuple(gs.shape) == (experts,)
    assert tensors["weight_scale"].dtype == torch.float8_e4m3fn
    module, build_meta = _build(spec, macros, classes, include)
    metadata.update(build_meta)
    slots, bm = tokens * 8, 16
    capacity = tr.cdiv(slots + experts * (bm - 1), bm) * bm
    num_sms = torch.cuda.get_device_properties(packed.device).multi_processor_count
    grid = num_sms * 2
    metadata["grid_ctas"] = grid
    return {"weight": tensors["weight"], "sf": tensors["weight_scale"], "gs": gs,
            "launch": _Launch(module, grid), "configs": None, "block_m": bm, "tuning": tuning,
            "kernel_metadata": metadata, "layer_config": layer.to_str(),
            "compute_config": {"gemm_type": "indexed", "use_f16_accum": False},
            "mma_type": "wgmma", "gemm_type": "indexed", "use_stream_k": True,
            "sorted_ids": torch.empty(capacity, dtype=torch.int32, device=packed.device),
            "expert_ids": torch.empty(capacity // bm, dtype=torch.int32, device=packed.device),
            "padded": torch.empty((), dtype=torch.int32, device=packed.device), "expert_layout": None,
            "locks": torch.zeros(max(2048, grid * 16), dtype=torch.int32, device=packed.device)}


def prepare(weights, tokens):
    """Return compiled, loaded, preallocated indexed state; never an audit-only fallback."""
    try:
        import torch
        import triton as tr
        import humming
        import fresh_humming_adapter as H
        from importlib.metadata import version
    except ImportError as exc:
        raise RuntimeError("overlap Python dependency unavailable (requires torch, triton, humming 0.1.12 and sibling adapter)") from exc
    if version("humming-kernels").split("+", 1)[0] != PINNED_VERSION:
        raise RuntimeError("overlap requires installed humming-kernels==0.1.12")
    include = _check_headers(humming)
    source_audit()
    if not isinstance(tokens, int) or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    w13, w2 = weights["w13"], weights["w2"]
    if w13.ndim != 3 or w2.ndim != 3 or not w13.is_cuda:
        raise ValueError("expected CUDA packed weights [E,N,K/2]")
    experts, _, k_bytes = w13.shape
    d = k_bytes * 2
    if tuple(w13.shape) != (experts, 2 * d, d // 2):
        raise ValueError("W13 must have gate rows then up rows, each D x D")
    if tuple(w2.shape) != (experts, d, d // 2) or d % 128:
        raise ValueError("expected W2 [E,D,D/2] and D divisible by 128")
    for name, rows in (("w13", 2 * d), ("w2", d)):
        if weights[name].dtype != torch.uint8:
            raise TypeError(f"{name} must be raw packed uint8 E2M1")
        sf, gs = weights[name + "_sf"], weights[name + "_gs"]
        if sf.dtype != torch.float8_e4m3fn or tuple(sf.shape) != (experts, rows, d // 16):
            raise TypeError(f"{name}_sf must be E4M3 [E,N,D/16]")
        if gs.dtype != torch.float32 or tuple(gs.shape) != (experts,):
            raise TypeError(f"{name}_gs must be FP32 [E], in multiply convention")
    if tuple(weights["post_norm_w"].shape) != (d,):
        raise ValueError("post_norm_w must have shape [D]")
    if any(value.device != w13.device for value in weights.values() if torch.is_tensor(value)):
        raise ValueError("all weights must share a CUDA device")
    if torch.cuda.get_device_capability(w13.device)[0] != 9:
        raise ValueError("overlap targets SM90")
    with torch.no_grad(), torch.cuda.device(w13.device):
        p13 = _prepare_projection(w13, weights["w13_sf"], weights["w13_gs"], tokens, H, include, torch, tr)
        p2 = _prepare_projection(w2, weights["w2_sf"], weights["w2_gs"], tokens, H, include, torch, tr)
        for key in ("sorted_ids", "expert_ids", "padded"):
            p2[key] = p13[key]
        return {"tokens": tokens, "d": d, "experts": experts, "mode": "overlap_wg16", "grouped": False,
                "w13": p13, "w2": p2, "version": PINNED_VERSION, "helpers": H,
                "prepare_status": "compiled_and_loaded", "gpu_numerics_validated": False,
                "post_norm_w": weights["post_norm_w"].contiguous(),
                "counts": torch.empty(experts, dtype=torch.int32, device=w13.device),
                "h13": torch.empty((tokens * 8, 2 * d), dtype=torch.bfloat16, device=w13.device),
                "inter": torch.empty((tokens * 8, d), dtype=torch.bfloat16, device=w13.device),
                "y": torch.empty((tokens * 8, d), dtype=torch.bfloat16, device=w13.device),
                "z": torch.empty((tokens * 8, d), dtype=torch.float32, device=w13.device),
                "out": torch.empty((tokens, d), dtype=torch.bfloat16, device=w13.device)}


def run(data, state):
    return state["helpers"].run(data, state)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", action="store_true")
    parser.add_argument("--dependency-info", action="store_true")
    args = parser.parse_args()
    if not (args.source_audit or args.dependency_info):
        parser.error("choose --source-audit and/or --dependency-info; this file owns no GPU harness")
    if args.source_audit:
        print(json.dumps(source_audit(), indent=2))
    if args.dependency_info:
        print(json.dumps(dependency_info(), indent=2))

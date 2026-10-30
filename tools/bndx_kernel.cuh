// BNDX (notes/study/bndx.md): the T >= 128 layer boundaries (post-norm + gated residual + pre-norm) as our own PDL kernel with
// SGLang muse_spark_v12/norm_residual.cuh's single-CTA arithmetic vendored verbatim (same thread count per row, element map,
// fmaf order and cta::reduce_sum tree -> bitwise equal outputs). Changes vs the stock kernel, none of them arithmetic:
//   - persistent rows: G CTAs (one per SM) loop rows r = blockIdx.x + k * G, so the channel vectors (alpha, beta, post /
//     pre weights) are loaded once per CTA into registers instead of once per row after the first reduction;
//   - every load that does not depend on the predecessor (channel vectors, and the first row's stream when kStreamPre) is
//     issued before griddepcontrol.wait;
//   - kTrig 1: griddepcontrol.launch_dependents at entry (the dependent's wait still covers this whole grid); 2: at the end
//     (the stock placement);
//   - kNext: the next row's stream + branch are loaded before this row's reductions (one round trip per two rows).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>
#include <cstdint>

namespace sglang {
namespace okc_bndx {
using namespace device;
constexpr int kVec = 8;
constexpr int kFVec = kMaxVecBytes >= 32 ? kVec : kVec / 2;
constexpr int kFParts = kVec / kFVec;
constexpr int kPairs = kVec / 2;
constexpr int kFPairs = kFVec / 2;
SGL_DEVICE fp32x2_t round_bf16(fp32x2_t v) { return cast<fp32x2_t>(cast<bf16x2_t>(v)); }
using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

struct P {
  const bf16_t* branch; const fp32_t* stream; bf16_t* out; fp32_t* new_stream;
  const bf16_t* post_weight; const bf16_t* pre_weight; const fp32_t* alpha; const fp32_t* beta;
  float post_eps, pre_eps; int rows; bool gate_channel_wise, has_pre_weight;
};

template <int64_t kH, int kNT, bool kPDL, bool kIsc, bool kPw, bool kGate, int kTrig, bool kNext, bool kStreamPre>
__global__ void __launch_bounds__(kNT, 1) bndx_kernel(const __grid_constant__ P p) {
  constexpr int kPacks = kH / (kNT * kVec);
  constexpr int kNW = kNT / kWarpThreads;
  static_assert(kH % (kNT * kVec) == 0);
  __shared__ float smem[kNW == 1 ? 1 : 2 * kNW];   // the stock kernel's two scratch areas (safe across rows: see bndx.md)
  if constexpr (kTrig == 1) PDLTriggerSecondary<kPDL>();
  const bool channel_wise = kGate && p.gate_channel_wise;
  const bool has_pre_weight = p.has_pre_weight;
  const int G = static_cast<int>(gridDim.x);
  int row = static_cast<int>(blockIdx.x);

  // ---- channel vectors: never written by a predecessor -> before the wait, once per CTA
  fp32x2_t post_w[kPw ? kPacks : 1][kPairs];
  F32Vec alpha_r[kGate ? kPacks : 1][kFParts], beta_r[kGate ? kPacks : 1][kFParts];
  BF16Vec pre_w_r[kPacks];
#pragma unroll
  for (int pk = 0; pk < kPacks; ++pk) {
    const int idx = threadIdx.x + pk * kNT;
    if constexpr (kPw) {
      BF16Vec weight_v;
      weight_v.load(p.post_weight, idx);
#pragma unroll
      for (int q = 0; q < kPairs; ++q) post_w[pk][q] = cast<fp32x2_t>(weight_v[q]);
    }
    if constexpr (kGate) {
      if (channel_wise) {
#pragma unroll
        for (int j = 0; j < kFParts; ++j) {
          alpha_r[pk][j].load(p.alpha, idx * kFParts + j);
          beta_r[pk][j].load(p.beta, idx * kFParts + j);
        }
      }
    }
    if (has_pre_weight) pre_w_r[pk].load(p.pre_weight, idx);
  }
  fp32x2_t gate_a = {1.0f, 1.0f};
  fp32x2_t gate_b = {1.0f, 1.0f};
  if constexpr (kGate) {
    if (!channel_wise) {
      gate_a.x = gate_a.y = *p.alpha;
      gate_b.x = gate_b.y = *p.beta;
    }
  }

  F32Vec stream_v[kPacks][kFParts];
  BF16Vec branch_v[kPacks];
  if constexpr (kStreamPre) {
    if (row < p.rows) {
      const fp32_t* srow = p.stream + static_cast<int64_t>(row) * kH;
#pragma unroll
      for (int pk = 0; pk < kPacks; ++pk)
#pragma unroll
        for (int j = 0; j < kFParts; ++j) stream_v[pk][j].load(srow, (threadIdx.x + pk * kNT) * kFParts + j);
    }
  }
  PDLWaitPrimary<kPDL>();
  if (row < p.rows) {
    const int64_t base = static_cast<int64_t>(row) * kH;
#pragma unroll
    for (int pk = 0; pk < kPacks; ++pk) {
      const int idx = threadIdx.x + pk * kNT;
      branch_v[pk].load(p.branch + base, idx);
      if constexpr (!kStreamPre) {
#pragma unroll
        for (int j = 0; j < kFParts; ++j) stream_v[pk][j].load(p.stream + base, idx * kFParts + j);
      }
    }
  }

  for (; row < p.rows; row += G) {
    const int64_t base = static_cast<int64_t>(row) * kH;
    fp32x2_t vals[kPacks][kPairs];
    float sum_sq = 0.0f;
#pragma unroll
    for (int pk = 0; pk < kPacks; ++pk) {
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        auto v = cast<fp32x2_t>(branch_v[pk][q]);
        if constexpr (kIsc && kPw) {
          v.x *= post_w[pk][q].x;
          v.y *= post_w[pk][q].y;
          v = round_bf16(v);
        }
        vals[pk][q] = v;
        sum_sq = fmaf(v.x, v.x, sum_sq);
        sum_sq = fmaf(v.y, v.y, sum_sq);
      }
    }
    F32Vec stream_c[kPacks][kFParts];
#pragma unroll
    for (int pk = 0; pk < kPacks; ++pk)
#pragma unroll
      for (int j = 0; j < kFParts; ++j) stream_c[pk][j] = stream_v[pk][j];
    const int nrow = row + G;
    if constexpr (kNext) {   // next row's loads before this row's reductions
      if (nrow < p.rows) {
        const int64_t nb = static_cast<int64_t>(nrow) * kH;
#pragma unroll
        for (int pk = 0; pk < kPacks; ++pk) {
          const int idx = threadIdx.x + pk * kNT;
          branch_v[pk].load(p.branch + nb, idx);
#pragma unroll
          for (int j = 0; j < kFParts; ++j) stream_v[pk][j].load(p.stream + nb, idx * kFParts + j);
        }
      }
    }
    sum_sq = cta::reduce_sum<kNT>(sum_sq, smem);
    const float post_rstd = math::rsqrt(sum_sq / kH + p.post_eps);

    float pre_sum_sq = 0.0f;
#pragma unroll
    for (int pk = 0; pk < kPacks; ++pk) {
      const int idx = threadIdx.x + pk * kNT;
      F32Vec new_stream_v[kFParts];
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        const int h = q / kFPairs;
        const int l = q % kFPairs;
        fp32x2_t normed_b = vals[pk][q];
        normed_b.x *= post_rstd;
        normed_b.y *= post_rstd;
        if constexpr (!kIsc && kPw) {
          normed_b.x *= post_w[pk][q].x;
          normed_b.y *= post_w[pk][q].y;
        }
        normed_b = round_bf16(normed_b);
        const fp32x2_t s = stream_c[pk][h][l];
        fp32x2_t new_s;
        if constexpr (kGate) {
          const fp32x2_t ga = channel_wise ? alpha_r[pk][h][l] : gate_a;
          const fp32x2_t gb = channel_wise ? beta_r[pk][h][l] : gate_b;
          new_s.x = ga.x * s.x + gb.x * normed_b.x;
          new_s.y = ga.y * s.y + gb.y * normed_b.y;
        } else {
          new_s.x = s.x + normed_b.x;
          new_s.y = s.y + normed_b.y;
        }
        new_stream_v[h][l] = new_s;
        const fp32x2_t pre_v = new_s;   // pre_norm_bf16 is false at T > 8 (the module passes pre_norm_dtype None)
        vals[pk][q] = pre_v;
        pre_sum_sq = fmaf(pre_v.x, pre_v.x, pre_sum_sq);
        pre_sum_sq = fmaf(pre_v.y, pre_v.y, pre_sum_sq);
      }
#pragma unroll
      for (int j = 0; j < kFParts; ++j) new_stream_v[j].store(p.new_stream + base, idx * kFParts + j);
    }

    pre_sum_sq = cta::reduce_sum<kNT>(pre_sum_sq, kNW == 1 ? smem : smem + kNW);
    const float pre_rstd = math::rsqrt(pre_sum_sq / kH + p.pre_eps);
#pragma unroll
    for (int pk = 0; pk < kPacks; ++pk) {
      const int idx = threadIdx.x + pk * kNT;
      BF16Vec out_v;
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        fp32x2_t v = vals[pk][q];
        v.x *= pre_rstd;
        v.y *= pre_rstd;
        if (has_pre_weight) {
          const auto wf = cast<fp32x2_t>(pre_w_r[pk][q]);
          v.x *= wf.x;
          v.y *= wf.y;
        }
        out_v[q] = cast<bf16x2_t>(v);
      }
      out_v.store(p.out + base, idx);
    }

    if constexpr (!kNext) {   // next row's loads after this row (one round trip per row)
      if (nrow < p.rows) {
        const int64_t nb = static_cast<int64_t>(nrow) * kH;
#pragma unroll
        for (int pk = 0; pk < kPacks; ++pk) {
          const int idx = threadIdx.x + pk * kNT;
          branch_v[pk].load(p.branch + nb, idx);
#pragma unroll
          for (int j = 0; j < kFParts; ++j) stream_v[pk][j].load(p.stream + nb, idx * kFParts + j);
        }
      }
    }
  }
  if constexpr (kTrig == 2) PDLTriggerSecondary<kPDL>();
}
}  // namespace okc_bndx

template <int64_t kH, int kNT, bool kPDL, bool kIsc, bool kPw, bool kGate, int kTrig, bool kNext, bool kStreamPre>
struct OkcBndx {
  using TV = tvm::ffi::TensorView;
  using OT = tvm::ffi::Optional<tvm::ffi::TensorView>;
  static const bf16_t* bp(const OT& t) { return t.has_value() ? static_cast<const bf16_t*>(t.value().data_ptr()) : nullptr; }
  static const fp32_t* fp(const OT& t) { return t.has_value() ? static_cast<const fp32_t*>(t.value().data_ptr()) : nullptr; }
  static void run(const TV branch, const TV stream, const TV out, const TV new_stream, OT post_weight, OT pre_weight, OT alpha,
                  OT beta, double post_eps, double pre_eps, bool channel_wise, int64_t grid) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto device_ = SymbolicDevice{};
    H.set_value(kH);
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(branch);
    const auto rows = static_cast<int>(M.unwrap());
    if (rows == 0) return;
    const bool has_pre = pre_weight.has_value();   // the stock kernel's rule (the module passes None when absent)
    const okc_bndx::P p{static_cast<const bf16_t*>(branch.data_ptr()), static_cast<const fp32_t*>(stream.data_ptr()),
                        static_cast<bf16_t*>(out.data_ptr()), static_cast<fp32_t*>(new_stream.data_ptr()), bp(post_weight),
                        has_pre ? bp(pre_weight) : nullptr, fp(alpha), fp(beta), (float)post_eps, (float)pre_eps, rows,
                        channel_wise, has_pre};
    const int g = static_cast<int>(grid < rows ? grid : rows);
    LaunchKernel(g, kNT, device_.unwrap()).config({.use_pdl = kPDL})(
        okc_bndx::bndx_kernel<kH, kNT, kPDL, kIsc, kPw, kGate, kTrig, kNext, kStreamPre>, p);
  }
};

}  // namespace sglang

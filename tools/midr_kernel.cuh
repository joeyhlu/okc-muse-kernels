// MIDR (tools/midr_patches.py): SGLang muse_spark_v12/norm_residual.cuh single-CTA kernel arithmetic, vendored as plain source.
// A: attention boundary without the FP32 mid-stream store. F: ffn boundary that rebuilds mid = a1 * s + b1 * postnorm1(o) from
// the layer input and the o-proj output with A's arithmetic (registers only), then runs SGLang's ffn-boundary arithmetic on it.
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

namespace okc_midr {
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
  const bf16_t* b1; const fp32_t* s; const bf16_t* b2; bf16_t* out; fp32_t* ns;
  const bf16_t* pw1; const fp32_t* a1; const fp32_t* g1; const bf16_t* pw2; const bf16_t* qw; const fp32_t* a2; const fp32_t* g2;
  float e1, e2, qe; bool cw1, cw2, hasqw;
};

// SGLang's first half: branch post-norm + gated residual -> sv (the new stream) in registers. kLoad: sv = stream row from
// memory first (else sv already holds the stream, e.g. the rebuilt mid).
template <int64_t kH, int kNT, bool kIsc, bool kPw, bool kGate, bool kLoad>
SGL_DEVICE void mid_half(const bf16_t* branch, const fp32_t* stream, const bf16_t* post_weight, const fp32_t* alpha_ptr,
                         const fp32_t* beta_ptr, float eps, bool channel_wise_, F32Vec (&sv)[kH / (kNT * kVec)][kFParts], float* smem) {
  constexpr int kPacks = kH / (kNT * kVec);
  const bool channel_wise = kGate && channel_wise_;
  fp32x2_t vals[kPacks][kPairs];
  fp32x2_t post_w[kPw ? kPacks : 1][kPairs];
  float sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNT;
    BF16Vec branch_v;
    branch_v.load(branch, idx);
    if constexpr (kLoad) {
#pragma unroll
      for (int j = 0; j < kFParts; ++j) sv[p][j].load(stream, idx * kFParts + j);
    }
    if constexpr (kPw) {
      BF16Vec weight_v;
      weight_v.load(post_weight, idx);
#pragma unroll
      for (int q = 0; q < kPairs; ++q) post_w[p][q] = cast<fp32x2_t>(weight_v[q]);
    }
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      auto v = cast<fp32x2_t>(branch_v[q]);
      if constexpr (kIsc && kPw) {
        v.x *= post_w[p][q].x;
        v.y *= post_w[p][q].y;
        v = round_bf16(v);
      }
      vals[p][q] = v;
      sum_sq = fmaf(v.x, v.x, sum_sq);
      sum_sq = fmaf(v.y, v.y, sum_sq);
    }
  }
  sum_sq = cta::reduce_sum<kNT>(sum_sq, smem);
  const float post_rstd = math::rsqrt(sum_sq / kH + eps);
  fp32x2_t gate_a = {1.0f, 1.0f};
  fp32x2_t gate_b = {1.0f, 1.0f};
  if constexpr (kGate) {
    if (!channel_wise) {
      gate_a.x = gate_a.y = *alpha_ptr;
      gate_b.x = gate_b.y = *beta_ptr;
    }
  }
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNT;
    F32Vec alpha_v[kFParts], beta_v[kFParts];
    if constexpr (kGate) {
      if (channel_wise) {
#pragma unroll
        for (int j = 0; j < kFParts; ++j) {
          alpha_v[j].load(alpha_ptr, idx * kFParts + j);
          beta_v[j].load(beta_ptr, idx * kFParts + j);
        }
      }
    }
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      const int h = q / kFPairs;
      const int l = q % kFPairs;
      fp32x2_t normed_b = vals[p][q];
      normed_b.x *= post_rstd;
      normed_b.y *= post_rstd;
      if constexpr (!kIsc && kPw) {
        normed_b.x *= post_w[p][q].x;
        normed_b.y *= post_w[p][q].y;
      }
      normed_b = round_bf16(normed_b);
      const fp32x2_t s = sv[p][h][l];
      fp32x2_t new_s;
      if constexpr (kGate) {
        const fp32x2_t ga = channel_wise ? alpha_v[h][l] : gate_a;
        const fp32x2_t gb = channel_wise ? beta_v[h][l] : gate_b;
        new_s.x = ga.x * s.x + gb.x * normed_b.x;
        new_s.y = ga.y * s.y + gb.y * normed_b.y;
      } else {
        new_s.x = s.x + normed_b.x;
        new_s.y = s.y + normed_b.y;
      }
      sv[p][h][l] = new_s;
    }
  }
}

// SGLang's second half: pre-norm of the new stream (FP32 pre-norm input, no router) -> out; kStore: new_stream store too.
template <int64_t kH, int kNT, bool kStore>
SGL_DEVICE void pre_half(F32Vec (&sv)[kH / (kNT * kVec)][kFParts], const bf16_t* pre_weight, bool has_pre_weight, float pre_eps,
                         bf16_t* out, fp32_t* new_stream, float* smem) {
  constexpr int kPacks = kH / (kNT * kVec);
  float pre_sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNT;
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      const fp32x2_t pre_v = sv[p][q / kFPairs][q % kFPairs];
      pre_sum_sq = fmaf(pre_v.x, pre_v.x, pre_sum_sq);
      pre_sum_sq = fmaf(pre_v.y, pre_v.y, pre_sum_sq);
    }
    if constexpr (kStore) {
#pragma unroll
      for (int j = 0; j < kFParts; ++j) sv[p][j].store(new_stream, idx * kFParts + j);
    }
  }
  pre_sum_sq = cta::reduce_sum<kNT>(pre_sum_sq, smem);
  const float pre_rstd = math::rsqrt(pre_sum_sq / kH + pre_eps);
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNT;
    BF16Vec weight_v;
    if (has_pre_weight) weight_v.load(pre_weight, idx);
    BF16Vec out_v;
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      fp32x2_t v = sv[p][q / kFPairs][q % kFPairs];
      v.x *= pre_rstd;
      v.y *= pre_rstd;
      if (has_pre_weight) {
        const auto wf = cast<fp32x2_t>(weight_v[q]);
        v.x *= wf.x;
        v.y *= wf.y;
      }
      out_v[q] = cast<bf16x2_t>(v);
    }
    out_v.store(out, idx);
  }
}

template <int64_t kH, int kNT, bool kPDL, bool kIsc1, bool kPw1, bool kGate1, bool kIsc2, bool kPw2, bool kGate2, bool kF>
__global__ void __launch_bounds__(kNT) midr_kernel(const __grid_constant__ P p) {
  constexpr int kNW = kNT / kWarpThreads;
  __shared__ float smem[3 * kNW];   // one scratch area per reduction (reduce_sum has no trailing barrier)
  F32Vec sv[kH / (kNT * kVec)][kFParts];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * kH;
  PDLWaitPrimary<kPDL>();
  mid_half<kH, kNT, kIsc1, kPw1, kGate1, true>(p.b1 + base, p.s + base, p.pw1, p.a1, p.g1, p.e1, p.cw1, sv, smem);
  if constexpr (kF) mid_half<kH, kNT, kIsc2, kPw2, kGate2, false>(p.b2 + base, nullptr, p.pw2, p.a2, p.g2, p.e2, p.cw2, sv, smem + kNW);
  pre_half<kH, kNT, kF>(sv, p.qw, p.hasqw, p.qe, p.out + base, kF ? p.ns + base : nullptr, smem + 2 * kNW);
  PDLTriggerSecondary<kPDL>();
}
}  // namespace okc_midr

template <int64_t kH, int kNT, bool kPDL, bool kIsc1, bool kPw1, bool kGate1, bool kIsc2, bool kPw2, bool kGate2>
struct OkcMidr {
  using TV = tvm::ffi::TensorView;
  using OT = tvm::ffi::Optional<tvm::ffi::TensorView>;
  template <bool kF>
  static void go(const okc_midr::P& p, const TV any) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto device_ = SymbolicDevice{};
    H.set_value(kH);
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(any);
    const auto rows = static_cast<int>(M.unwrap());
    if (rows == 0) return;
    LaunchKernel(rows, kNT, device_.unwrap()).config({.use_pdl = kPDL})(
        okc_midr::midr_kernel<kH, kNT, kPDL, kIsc1, kPw1, kGate1, kIsc2, kPw2, kGate2, kF>, p);
  }
  static const bf16_t* bp(const OT& t) { return t.has_value() ? static_cast<const bf16_t*>(t.value().data_ptr()) : nullptr; }
  static const fp32_t* fp(const OT& t) { return t.has_value() ? static_cast<const fp32_t*>(t.value().data_ptr()) : nullptr; }
  static bool cw(const OT& t) { return t.has_value() && t.value().size(0) == kH; }
  static void runA(const TV b1, const TV s, const TV out, OT pw1, OT qw, OT a1, OT g1, double e1, double qe) {
    const okc_midr::P p{static_cast<const bf16_t*>(b1.data_ptr()), static_cast<const fp32_t*>(s.data_ptr()), nullptr,
                        static_cast<bf16_t*>(out.data_ptr()), nullptr, bp(pw1), fp(a1), fp(g1), nullptr, bp(qw), nullptr, nullptr,
                        (float)e1, 0.f, (float)qe, cw(a1), false, qw.has_value()};
    go<false>(p, b1);
  }
  static void runF(const TV b1, const TV s, const TV b2, const TV out, const TV ns, OT pw1, OT a1, OT g1, double e1, OT pw2, OT qw,
                   OT a2, OT g2, double e2, double qe) {
    const okc_midr::P p{static_cast<const bf16_t*>(b1.data_ptr()), static_cast<const fp32_t*>(s.data_ptr()),
                        static_cast<const bf16_t*>(b2.data_ptr()), static_cast<bf16_t*>(out.data_ptr()),
                        static_cast<fp32_t*>(ns.data_ptr()), bp(pw1), fp(a1), fp(g1), bp(pw2), bp(qw), fp(a2), fp(g2), (float)e1,
                        (float)e2, (float)qe, cw(a1), cw(a2), qw.has_value()};
    go<true>(p, b1);
  }
};

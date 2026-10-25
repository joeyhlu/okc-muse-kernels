// Compile-only stand-in for the torch headers our kernels' CUDA_SRC strings include (tools/cuda_ccheck/ccheck.py).
// Enough for host glue that checks tensors and launches kernels; never linked or run. Device code is untouched, so
// ptxas register / spill / SASS output is the real thing.
#pragma once
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <initializer_list>
#include <cuda_runtime.h>
namespace c10 {
enum class ScalarType { Byte, Char, Short, Int, Long, Half, Float, Double, BFloat16, Bool };
struct Device { int idx = 0; };
namespace cuda {
struct OptionalCUDAGuard { explicit OptionalCUDAGuard(Device) {} };
}  // namespace cuda
}  // namespace c10
namespace at {
using c10::ScalarType;
constexpr ScalarType kByte = ScalarType::Byte, kChar = ScalarType::Char, kShort = ScalarType::Short,
                     kInt = ScalarType::Int, kLong = ScalarType::Long, kHalf = ScalarType::Half,
                     kFloat = ScalarType::Float, kDouble = ScalarType::Double, kBFloat16 = ScalarType::BFloat16,
                     kBool = ScalarType::Bool;
struct Tensor {
  void* p = nullptr;
  bool is_cuda() const { return true; }
  ScalarType scalar_type() const { return ScalarType::Float; }
  bool is_contiguous() const { return true; }
  int64_t numel() const { return 0; }
  int64_t dim() const { return 2; }
  int64_t size(int) const { return 0; }
  int64_t stride(int) const { return 1; }
  c10::Device device() const { return {}; }
  void* data_ptr() const { return p; }
  template <typename T> T* data_ptr() const { return static_cast<T*>(p); }
};
namespace cuda {
struct Stream {
  operator cudaStream_t() const { return nullptr; }
  cudaStream_t stream() const { return nullptr; }
};
inline Stream getCurrentCUDAStream(int = -1) { return {}; }
}  // namespace cuda
}  // namespace at
namespace torch {
using at::Tensor;
constexpr c10::ScalarType kInt64 = c10::ScalarType::Long, kInt32 = c10::ScalarType::Int,
                          kFloat32 = c10::ScalarType::Float, kUInt8 = c10::ScalarType::Byte;
using at::kBFloat16;
using at::kFloat;
using at::kInt;
using at::kLong;
struct Opts {
  c10::ScalarType t;
  Opts device(c10::Device) const { return *this; }
};
inline Opts dtype(c10::ScalarType t) { return {t}; }
inline Tensor empty(std::initializer_list<int64_t>, Opts) { return {}; }
inline Tensor zeros(std::initializer_list<int64_t>, Opts) { return {}; }
}  // namespace torch
#define TORCH_CHECK(cond, ...) do { if (!(cond)) throw 1; } while (0)
#define C10_CUDA_CHECK(x) do { if ((x) != cudaSuccess) throw 2; } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

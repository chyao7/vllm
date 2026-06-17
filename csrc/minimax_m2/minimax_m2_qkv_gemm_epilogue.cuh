#pragma once

#include <cuda_runtime.h>

#include "cub_helpers.h"

namespace minimax_m2 {
namespace gemm_epilogue {

template <typename scalar_t>
struct ScalarTraits;

template <>
struct ScalarTraits<float> {
  static __device__ float to_float(float x) { return x; }
  static __device__ float from_float(float x) { return x; }
};

template <>
struct ScalarTraits<c10::Half> {
  static __device__ float to_float(c10::Half x) {
    return __half2float(reinterpret_cast<const __half&>(x));
  }
  static __device__ c10::Half from_float(float x) {
    c10::Half out;
    reinterpret_cast<__half&>(out) = __float2half(x);
    return out;
  }
};

template <>
struct ScalarTraits<c10::BFloat16> {
  static __device__ float to_float(c10::BFloat16 x) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(x));
  }
  static __device__ c10::BFloat16 from_float(float x) {
    c10::BFloat16 out;
    reinterpret_cast<__nv_bfloat16&>(out) = __float2bfloat16(x);
    return out;
  }
};

// GEMM epilogue: optional bias add on q/k/v columns + per-token local sum(q^2), sum(k^2).
// Launched on the same stream immediately after GEMM so values are still hot in cache.
template <typename scalar_t>
__global__ void qkv_gemm_epilogue_sum_sq_kernel(
    scalar_t* qkv, float* qk_sum_sq, scalar_t const* bias, int num_tokens,
    int qkv_dim, int q_size, int kv_size) {
  int const t = blockIdx.x;
  if (t >= num_tokens) {
    return;
  }

  scalar_t* row = qkv + static_cast<int64_t>(t) * qkv_dim;
  int const k_base = q_size;

  float sum_sq_q = 0.0f;
  float sum_sq_k = 0.0f;

  for (int i = threadIdx.x; i < q_size; i += blockDim.x) {
    float v = ScalarTraits<scalar_t>::to_float(row[i]);
    if (bias != nullptr) {
      v += ScalarTraits<scalar_t>::to_float(bias[i]);
      row[i] = ScalarTraits<scalar_t>::from_float(v);
    }
    sum_sq_q += v * v;
  }
  for (int i = threadIdx.x; i < kv_size; i += blockDim.x) {
    int const col = k_base + i;
    float v = ScalarTraits<scalar_t>::to_float(row[col]);
    if (bias != nullptr) {
      v += ScalarTraits<scalar_t>::to_float(bias[col]);
      row[col] = ScalarTraits<scalar_t>::from_float(v);
    }
    sum_sq_k += v * v;
  }
  if (bias != nullptr) {
    int const v_base = q_size + kv_size;
    for (int i = threadIdx.x; i < kv_size; i += blockDim.x) {
      int const col = v_base + i;
      float v = ScalarTraits<scalar_t>::to_float(row[col]);
      v += ScalarTraits<scalar_t>::to_float(bias[col]);
      row[col] = ScalarTraits<scalar_t>::from_float(v);
    }
  }

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  sum_sq_q = BlockReduce(tmp).Reduce(sum_sq_q, CubAddOp{}, blockDim.x);
  __syncthreads();
  sum_sq_k = BlockReduce(tmp).Reduce(sum_sq_k, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    qk_sum_sq[t * 2 + 0] = sum_sq_q;
    qk_sum_sq[t * 2 + 1] = sum_sq_k;
  }
}

template <typename scalar_t>
void launch_qkv_gemm_epilogue_sum_sq(
    scalar_t* qkv, float* qk_sum_sq, scalar_t const* bias, int num_tokens,
    int qkv_dim, int q_size, int kv_size, cudaStream_t stream) {
  constexpr int block = 256;
  qkv_gemm_epilogue_sum_sq_kernel<scalar_t>
      <<<num_tokens, block, 0, stream>>>(qkv, qk_sum_sq, bias, num_tokens,
                                         qkv_dim, q_size, kv_size);
}

}  // namespace gemm_epilogue
}  // namespace minimax_m2

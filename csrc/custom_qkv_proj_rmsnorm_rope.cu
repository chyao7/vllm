/*
 * QKV GEMM + Q/K RMSNorm + RoPE (no paged KV cache write).
 * Output stays in qkv; use reshape_and_cache_flash separately if needed.
 */

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <optional>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"

#define CHECK_TYPE(x, st)                                              \
  TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), \
              ", while ", st, " is expected")
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_TH_CUDA(x);    \
  CHECK_CONTIGUOUS(x)

namespace {

template <typename scalar_t>
void qkv_gemm_cuda(torch::Tensor& out, torch::Tensor const& input,
                   torch::Tensor const& weight,
                   std::optional<torch::Tensor> const& bias) {
  int64_t const M = input.size(0);
  int64_t const K = input.size(1);
  int64_t const N = weight.size(0);
  TORCH_CHECK(weight.size(1) == K);

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(
      cublasSetStream(handle, at::cuda::getCurrentCUDAStream()));

  cudaDataType_t cuda_dtype =
      std::is_same_v<scalar_t, at::BFloat16> ? CUDA_R_16BF : CUDA_R_16F;

  float alpha = 1.0f;
  float beta = 0.0f;
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N),
      static_cast<int>(M), static_cast<int>(K), &alpha, weight.data_ptr(),
      cuda_dtype, static_cast<int>(K), input.data_ptr(), cuda_dtype,
      static_cast<int>(K), &beta, out.data_ptr(), cuda_dtype,
      static_cast<int>(N), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

  if (bias.has_value()) {
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == N);
    out.add_(*bias);
  }
}

template <typename scalar_t>
struct ScalarTraits;

template <>
struct ScalarTraits<float> {
  using scalar_t = float;
  static __device__ float to_float(float x) { return x; }
  static __device__ float from_float(float x) { return x; }
};

template <>
struct ScalarTraits<c10::Half> {
  using scalar_t = c10::Half;
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
  using scalar_t = c10::BFloat16;
  static __device__ float to_float(c10::BFloat16 x) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(x));
  }
  static __device__ c10::BFloat16 from_float(float x) {
    c10::BFloat16 out;
    reinterpret_cast<__nv_bfloat16&>(out) = __float2bfloat16(x);
    return out;
  }
};

constexpr int QK_NORM_ROPE_BLOCK_THREADS = 32;

__device__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename cache_scalar_t>
struct CacheScalarTraits;

template <>
struct CacheScalarTraits<float> {
  static __device__ float to_float(float x) { return x; }
};

template <>
struct CacheScalarTraits<c10::Half> {
  static __device__ float to_float(c10::Half x) {
    return __half2float(reinterpret_cast<const __half&>(x));
  }
};

template <>
struct CacheScalarTraits<c10::BFloat16> {
  static __device__ float to_float(c10::BFloat16 x) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(x));
  }
};

// Grid: (num_tokens, num_qk_heads). One warp (32 threads) per (token, head).
template <typename scalar_t, typename cache_scalar_t>
__global__ void custom_qk_rmsnorm_rope_kernel(
    scalar_t* qkv, int num_tokens, int qkv_dim, int num_heads_q, int num_heads_k,
    int head_dim, int rotary_dim, const scalar_t* q_weight,
    const scalar_t* k_weight, const int64_t* positions,
    const cache_scalar_t* cos_sin_cache, float eps) {
  int const t = blockIdx.x;
  int const head_id = blockIdx.y;
  int const total_qk_heads = num_heads_q + num_heads_k;
  if (t >= num_tokens || head_id >= total_qk_heads) {
    return;
  }

  int const embed_dim = rotary_dim / 2;
  int const lane = threadIdx.x;
  int const q_offset = 0;
  int const k_offset = num_heads_q * head_dim;

  bool const is_q_head = head_id < num_heads_q;
  int col_offset;
  const scalar_t* weight;
  if (is_q_head) {
    col_offset = q_offset + head_id * head_dim;
    weight = q_weight;
  } else {
    int const kv_head = head_id - num_heads_q;
    col_offset = k_offset + kv_head * head_dim;
    weight = k_weight;
  }

  scalar_t* row = qkv + static_cast<int64_t>(t) * qkv_dim + col_offset;

  int64_t const pos = positions[t];
  cache_scalar_t const* cache_row = cos_sin_cache + pos * rotary_dim;
  cache_scalar_t const* cos_ptr = cache_row;
  cache_scalar_t const* sin_ptr = cache_row + embed_dim;

  using Traits = ScalarTraits<scalar_t>;
  using CacheTraits = CacheScalarTraits<cache_scalar_t>;

  float sum_sq = 0.0f;
  for (int pair_base = 0; pair_base < embed_dim; pair_base += 32) {
    int const p = pair_base + lane;
    if (p < embed_dim) {
      float const x = Traits::to_float(row[p]);
      float const y = Traits::to_float(row[p + embed_dim]);
      sum_sq += x * x + y * y;
    }
  }
  sum_sq = warp_reduce_sum(sum_sq);
  float const inv_rms = rsqrtf(sum_sq / static_cast<float>(head_dim) + eps);

  for (int pair_base = 0; pair_base < embed_dim; pair_base += 32) {
    int const p = pair_base + lane;
    if (p < embed_dim) {
      float nx = Traits::to_float(row[p]) * inv_rms * Traits::to_float(weight[p]);
      float ny = Traits::to_float(row[p + embed_dim]) * inv_rms *
                 Traits::to_float(weight[p + embed_dim]);
      float const cos = CacheTraits::to_float(cos_ptr[p]);
      float const sin = CacheTraits::to_float(sin_ptr[p]);
      row[p] = Traits::from_float(nx * cos - ny * sin);
      row[p + embed_dim] = Traits::from_float(ny * cos + nx * sin);
    }
  }
}

template <typename scalar_t, typename cache_scalar_t>
void launch_custom_qk_rmsnorm_rope(
    scalar_t* qkv, int num_tokens, int qkv_dim, int num_heads_q, int num_heads_k,
    int head_dim, int rotary_dim, const scalar_t* q_weight,
    const scalar_t* k_weight, const int64_t* positions,
    const cache_scalar_t* cos_sin_cache, float eps, cudaStream_t stream) {
  int const total_qk_heads = num_heads_q + num_heads_k;
  dim3 const grid(num_tokens, total_qk_heads);
  dim3 const block(QK_NORM_ROPE_BLOCK_THREADS);
  custom_qk_rmsnorm_rope_kernel<scalar_t, cache_scalar_t>
      <<<grid, block, 0, stream>>>(
          qkv, num_tokens, qkv_dim, num_heads_q, num_heads_k, head_dim,
          rotary_dim, q_weight, k_weight, positions, cos_sin_cache, eps);
}

}  // namespace

void custom_qk_rmsnorm_rope(
    torch::Tensor& qkv, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, int64_t rotary_dim, double eps,
    torch::Tensor& q_weight, torch::Tensor& k_weight,
    torch::Tensor& cos_sin_cache, bool is_neox, torch::Tensor& position_ids) {
  CHECK_INPUT(qkv);
  CHECK_INPUT(position_ids);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);
  CHECK_INPUT(cos_sin_cache);
  CHECK_TYPE(position_ids, torch::kInt64);
  TORCH_CHECK(is_neox, "custom_qk_rmsnorm_rope only supports NeoX-style RoPE");

  TORCH_CHECK(qkv.dim() == 2);
  TORCH_CHECK(position_ids.dim() == 1);
  TORCH_CHECK(q_weight.dim() == 1 && k_weight.dim() == 1);
  TORCH_CHECK(cos_sin_cache.dim() == 2);
  TORCH_CHECK(rotary_dim > 0 && rotary_dim % 2 == 0);
  TORCH_CHECK(rotary_dim <= head_dim);
  TORCH_CHECK(q_weight.size(0) == head_dim && k_weight.size(0) == head_dim);
  TORCH_CHECK(cos_sin_cache.size(1) == rotary_dim);

  int64_t const num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens);
  int64_t const total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(qkv.size(1) == total_heads * head_dim);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "custom_qk_rmsnorm_rope", [&] {
    using qkv_scalar_t = scalar_t;
    VLLM_DISPATCH_FLOATING_TYPES(
        cos_sin_cache.scalar_type(), "custom_qk_rmsnorm_rope_cache", [&] {
          using cache_scalar_t = scalar_t;
          launch_custom_qk_rmsnorm_rope<qkv_scalar_t, cache_scalar_t>(
              qkv.data_ptr<qkv_scalar_t>(), static_cast<int>(num_tokens),
              static_cast<int>(qkv.size(1)), static_cast<int>(num_heads_q),
              static_cast<int>(num_heads_k), static_cast<int>(head_dim),
              static_cast<int>(rotary_dim), q_weight.data_ptr<qkv_scalar_t>(),
              k_weight.data_ptr<qkv_scalar_t>(),
              position_ids.data_ptr<int64_t>(),
              cos_sin_cache.data_ptr<cache_scalar_t>(),
              static_cast<float>(eps), stream);
        });
  });
}

void custom_qkv_proj_rmsnorm_rope(
    torch::Tensor& qkv, torch::Tensor const& hidden_states,
    torch::Tensor const& qkv_weight, std::optional<torch::Tensor> qkv_bias,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t head_dim, int64_t rotary_dim, double eps, torch::Tensor& q_weight,
    torch::Tensor& k_weight, torch::Tensor& cos_sin_cache, bool is_neox,
    torch::Tensor& position_ids) {
  CHECK_INPUT(qkv);
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(qkv_weight);
  CHECK_INPUT(position_ids);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);
  CHECK_INPUT(cos_sin_cache);
  CHECK_TYPE(position_ids, torch::kInt64);
  TORCH_CHECK(is_neox, "custom_qkv_proj_rmsnorm_rope only supports NeoX-style RoPE");

  TORCH_CHECK(hidden_states.dim() == 2);
  TORCH_CHECK(qkv_weight.dim() == 2);
  TORCH_CHECK(hidden_states.size(0) == qkv.size(0));
  TORCH_CHECK(qkv_weight.size(1) == hidden_states.size(1));
  TORCH_CHECK(rotary_dim > 0 && rotary_dim % 2 == 0);
  TORCH_CHECK(rotary_dim <= head_dim);
  TORCH_CHECK(q_weight.size(0) == head_dim && k_weight.size(0) == head_dim);
  TORCH_CHECK(cos_sin_cache.dim() == 2);
  TORCH_CHECK(cos_sin_cache.size(1) == rotary_dim);
  TORCH_CHECK(qkv.scalar_type() == hidden_states.scalar_type());
  TORCH_CHECK(qkv.scalar_type() == qkv_weight.scalar_type());
  TORCH_CHECK(qkv.scalar_type() == q_weight.scalar_type());

  int64_t const num_tokens = qkv.size(0);
  int64_t const total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(qkv.size(1) == total_heads * head_dim);
  TORCH_CHECK(qkv_weight.size(0) == qkv.size(1));
  TORCH_CHECK(position_ids.size(0) == num_tokens);
  if (qkv_bias.has_value()) {
    const torch::Tensor& bias = *qkv_bias;
    CHECK_INPUT(bias);
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == qkv.size(1));
    TORCH_CHECK(bias.scalar_type() == qkv.scalar_type());
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "custom_qkv_proj_rmsnorm_rope", [&] {
    using qkv_scalar_t = scalar_t;
    qkv_gemm_cuda<qkv_scalar_t>(qkv, hidden_states, qkv_weight, qkv_bias);
    VLLM_DISPATCH_FLOATING_TYPES(
        cos_sin_cache.scalar_type(), "custom_qkv_proj_rmsnorm_rope_cache", [&] {
          using cache_scalar_t = scalar_t;
          launch_custom_qk_rmsnorm_rope<qkv_scalar_t, cache_scalar_t>(
              qkv.data_ptr<qkv_scalar_t>(), static_cast<int>(num_tokens),
              static_cast<int>(qkv.size(1)), static_cast<int>(num_heads_q),
              static_cast<int>(num_heads_k), static_cast<int>(head_dim),
              static_cast<int>(rotary_dim), q_weight.data_ptr<qkv_scalar_t>(),
              k_weight.data_ptr<qkv_scalar_t>(),
              position_ids.data_ptr<int64_t>(),
              cos_sin_cache.data_ptr<cache_scalar_t>(),
              static_cast<float>(eps), stream);
        });
  });
}

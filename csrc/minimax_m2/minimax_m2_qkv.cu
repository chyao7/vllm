/*
 * MiniMax-M2 fused QKV: qkv_proj + flat Q/K RMSNorm + RoPE.
 *
 * Two-stage pipeline (communication between stages is done in Python):
 *   1. minimax_m2_fused_qkv_compute  — GEMM + post (fused if tp<=1, else local sum_sq)
 *   2. [tensor_model_parallel_all_reduce on qk_sum_sq when tp>1]
 *   3. minimax_m2_fused_qkv_finalize — norm + RoPE from global sum_sq (tp>1 only)
 */

#include <cmath>
#include <optional>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

#include "cub_helpers.h"
#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);     \
  CHECK_CONTIG(x)

#ifdef USE_ROCM
  #define FINAL_MASK 0xffffffffffffffffULL
  #if defined(HIP_VERSION) && HIP_VERSION < 70000000
__device__ inline void __syncwarp() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
  #endif
#else
  #define FINAL_MASK 0xffffffff
#endif

namespace minimax_m2 {

enum class PostMode : int {
  kFusedAll = 0,   // tp==1: sum_sq + norm + rope in one launch
  kSumLocal = 1,   // write local sum(q^2), sum(k^2) to qk_sum_sq[t,0:1]
  kApplyGlobal = 2,  // use global sum_sq after TP all_reduce
};

// ---------------------------------------------------------------------------
// cuBLAS GEMM: qkv = hidden @ weight^T (+ bias)
// ---------------------------------------------------------------------------
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

namespace kernels {

template <typename T, int num>
struct packed_as;
template <>
struct packed_as<uint, 1> {
  using type = uint;
};
template <>
struct packed_as<uint, 2> {
  using type = uint2;
};
template <>
struct packed_as<uint, 4> {
  using type = uint4;
};

template <typename scalar_t_in, typename scalar_t_cache, int head_dim,
          bool interleave, PostMode mode>
__global__ void fused_post_qkv_kernel(
    void* qkv_void, float* qk_sum_sq_void, int num_heads_q, int num_heads_k,
    int num_heads_v, int q_size, int kv_size, int q_size_global,
    int kv_size_global, float eps, void const* q_weight_void,
    void const* k_weight_void, void const* cos_sin_cache_void,
    int64_t const* position_ids, int num_tokens, int rotary_dim) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr ((std::is_same_v<scalar_t_in, c10::BFloat16>) ||
                std::is_same_v<scalar_t_cache, c10::BFloat16>) {
    return;
  } else {
#endif
    using Converter = vllm::_typeConvert<scalar_t_in>;
    using T_in = typename Converter::hip_type;
    using T2_in = typename Converter::packed_hip_type;
    using CacheConverter = vllm::_typeConvert<scalar_t_cache>;
    using T_cache = typename CacheConverter::hip_type;

    T_in* qkv = reinterpret_cast<T_in*>(qkv_void);
    float* qk_sum_sq = qk_sum_sq_void;
    T_in const* q_weight = reinterpret_cast<T_in const*>(q_weight_void);
    T_in const* k_weight = reinterpret_cast<T_in const*>(k_weight_void);
    T_cache const* cos_sin_cache =
        reinterpret_cast<T_cache const*>(cos_sin_cache_void);

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int row_stride = q_size + 2 * kv_size;
    int q_base = token_idx * row_stride;
    int k_base = q_base + q_size;

    float sum_sq_q = 0.f;
    float sum_sq_k = 0.f;

    if (mode != PostMode::kApplyGlobal) {
      for (int i = threadIdx.x; i < q_size; i += blockDim.x) {
        float v = static_cast<float>(qkv[q_base + i]);
        sum_sq_q += v * v;
      }
      for (int i = threadIdx.x; i < kv_size; i += blockDim.x) {
        float v = static_cast<float>(qkv[k_base + i]);
        sum_sq_k += v * v;
      }
      using BlockReduce = cub::BlockReduce<float, 256>;
      __shared__ typename BlockReduce::TempStorage tmp;
      sum_sq_q = BlockReduce(tmp).Reduce(sum_sq_q, CubAddOp{}, blockDim.x);
      __syncthreads();
      sum_sq_k = BlockReduce(tmp).Reduce(sum_sq_k, CubAddOp{}, blockDim.x);

      if (threadIdx.x == 0) {
        if (mode == PostMode::kSumLocal) {
          qk_sum_sq[token_idx * 2 + 0] = sum_sq_q;
          qk_sum_sq[token_idx * 2 + 1] = sum_sq_k;
        }
      }
      if (mode == PostMode::kSumLocal) {
        return;
      }
    }

    __shared__ float s_rms_q;
    __shared__ float s_rms_k;
    if (threadIdx.x == 0) {
      float gsq_q, gsq_k;
      if (mode == PostMode::kApplyGlobal) {
        gsq_q = qk_sum_sq[token_idx * 2 + 0];
        gsq_k = qk_sum_sq[token_idx * 2 + 1];
      } else {
        gsq_q = sum_sq_q;
        gsq_k = sum_sq_k;
      }
      int denom_q = (mode == PostMode::kApplyGlobal) ? q_size_global : q_size;
      int denom_k = (mode == PostMode::kApplyGlobal) ? kv_size_global : kv_size;
      s_rms_q = rsqrtf(gsq_q / static_cast<float>(denom_q) + eps);
      s_rms_k = rsqrtf(gsq_k / static_cast<float>(denom_k) + eps);
    }
    __syncthreads();

    int warps_per_block = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int total_qk_heads = num_heads_q + num_heads_k;

    static_assert(head_dim % 64 == 0);
    constexpr int num_elems = head_dim / 32;
    float elements[num_elems];
    constexpr int elem_bytes = num_elems * sizeof(__nv_bfloat16);
    static_assert(elem_bytes % 4 == 0);
    constexpr int vec_size = elem_bytes / 4;
    using vec_T = typename packed_as<uint, vec_size>::type;

    for (int hb = 0; hb < total_qk_heads; hb += warps_per_block) {
      int local_head = hb + warp_id;
      if (local_head >= total_qk_heads) continue;
      bool is_q = local_head < num_heads_q;
      int head_idx = is_q ? local_head : local_head - num_heads_q;
      float rms = is_q ? s_rms_q : s_rms_k;
      T_in const* w = is_q ? q_weight : k_weight;
      int off_warp =
          is_q ? (q_base + head_idx * head_dim) : (k_base + head_idx * head_dim);
      int off_thread = off_warp + lane_id * num_elems;

      {
        vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[off_thread]);
        constexpr int packed = elem_bytes / sizeof(T2_in);
#pragma unroll
        for (int i = 0; i < packed; i++) {
          T2_in pv = *(reinterpret_cast<T2_in*>(&vec) + i);
          float2 vals = Converter::convert(pv);
          int d0 = lane_id * num_elems + 2 * i;
          int d1 = d0 + 1;
          elements[2 * i] =
              vals.x * rms * Converter::convert(w[head_idx * head_dim + d0]);
          elements[2 * i + 1] =
              vals.y * rms * Converter::convert(w[head_idx * head_dim + d1]);
        }
      }

      float elements2[num_elems];
      int64_t pos = position_ids[token_idx];
      T_cache const* cache = cos_sin_cache + pos * rotary_dim;
      int embed = rotary_dim / 2;
      T_cache const* cos_ptr = cache;
      T_cache const* sin_ptr = cache + embed;
      int rotary_lanes = rotary_dim / num_elems;

      if (lane_id < rotary_lanes) {
        if constexpr (interleave) {
#pragma unroll
          for (int i = 0; i < num_elems / 2; ++i) {
            int i0 = 2 * i, i1 = i0 + 1;
            int dim_idx = lane_id * num_elems + i0;
            float v0 = elements[i0], v1 = elements[i1];
            int hd = dim_idx / 2;
            float c = CacheConverter::convert(VLLM_LDG(cos_ptr + hd));
            float s = CacheConverter::convert(VLLM_LDG(sin_ptr + hd));
            elements[i0] = v0 * c - v1 * s;
            elements[i1] = v0 * s + v1 * c;
          }
        } else {
          __syncwarp();
          int pair = (rotary_dim / 2) / num_elems;
#pragma unroll
          for (int i = 0; i < num_elems; i++) {
            elements2[i] = __shfl_xor_sync(FINAL_MASK, elements[i], pair);
            if (lane_id < pair) elements2[i] = -elements2[i];
            int dim_idx = lane_id * num_elems + i;
            dim_idx = (dim_idx * 2) % rotary_dim;
            int hd = dim_idx / 2;
            float c = CacheConverter::convert(VLLM_LDG(cos_ptr + hd));
            float s = CacheConverter::convert(VLLM_LDG(sin_ptr + hd));
            elements[i] = elements[i] * c + elements2[i] * s;
          }
          __syncwarp();
        }
      }

      {
        vec_T vec;
        constexpr int packed = elem_bytes / sizeof(T2_in);
#pragma unroll
        for (int i = 0; i < packed; i++) {
          T2_in pv = Converter::convert(
              make_float2(elements[2 * i], elements[2 * i + 1]));
          *(reinterpret_cast<T2_in*>(&vec) + i) = pv;
        }
        *reinterpret_cast<vec_T*>(&qkv[off_thread]) = vec;
      }
    }
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

template <typename scalar_t_in, typename scalar_t_cache, PostMode mode>
void launch_fused_post_qkv(
    void* qkv, float* qk_sum_sq, int num_tokens, int num_heads_q,
    int num_heads_k, int num_heads_v, int head_dim, int q_size, int kv_size,
    int q_size_global, int kv_size_global, int rotary_dim, float eps,
    void const* q_w, void const* k_w, void const* cache, bool interleave,
    int64_t const* pos_ids, cudaStream_t stream) {
  constexpr int block = 256;
  dim3 grid(num_tokens);
  switch (head_dim) {
    case 64:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fused_post_qkv_kernel<scalar_t_in, scalar_t_cache, 64, INTERLEAVE, mode>
            <<<grid, block, 0, stream>>>(
                qkv, qk_sum_sq, num_heads_q, num_heads_k, num_heads_v, q_size,
                kv_size, q_size_global, kv_size_global, eps, q_w, k_w, cache,
                pos_ids, num_tokens, rotary_dim);
      });
      break;
    case 128:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fused_post_qkv_kernel<scalar_t_in, scalar_t_cache, 128, INTERLEAVE, mode>
            <<<grid, block, 0, stream>>>(
                qkv, qk_sum_sq, num_heads_q, num_heads_k, num_heads_v, q_size,
                kv_size, q_size_global, kv_size_global, eps, q_w, k_w, cache,
                pos_ids, num_tokens, rotary_dim);
      });
      break;
    case 256:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fused_post_qkv_kernel<scalar_t_in, scalar_t_cache, 256, INTERLEAVE, mode>
            <<<grid, block, 0, stream>>>(
                qkv, qk_sum_sq, num_heads_q, num_heads_k, num_heads_v, q_size,
                kv_size, q_size_global, kv_size_global, eps, q_w, k_w, cache,
                pos_ids, num_tokens, rotary_dim);
      });
      break;
    default:
      TORCH_CHECK(false, "Unsupported head_dim: ", head_dim);
  }
}

template <typename qkv_t, PostMode mode>
void dispatch_post_qkv(torch::Tensor& qkv, torch::Tensor& qk_sum_sq,
                     int num_heads_q, int num_heads_k, int num_heads_v,
                     int head_dim, int q_size, int kv_size, int q_size_global,
                     int kv_size_global, float eps, torch::Tensor& q_norm_weight,
                     torch::Tensor& k_norm_weight,
                     torch::Tensor& cos_sin_cache, bool is_neox,
                     torch::Tensor& position_ids) {
  int num_tokens = qkv.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
  VLLM_DISPATCH_FLOATING_TYPES(cos_sin_cache.scalar_type(), "rope_cache", [&] {
    using cache_scalar_t = scalar_t;
    kernels::launch_fused_post_qkv<qkv_t, cache_scalar_t, mode>(
        qkv.data_ptr(), qk_sum_sq.data_ptr<float>(), num_tokens, num_heads_q,
        num_heads_k, num_heads_v, head_dim, q_size, kv_size, q_size_global,
        kv_size_global, static_cast<int>(cos_sin_cache.size(1)), eps,
        q_norm_weight.data_ptr(), k_norm_weight.data_ptr(),
        cos_sin_cache.data_ptr(), !is_neox,
        reinterpret_cast<int64_t const*>(position_ids.data_ptr()), stream);
  });
}

}  // namespace kernels

template <typename scalar_t>
void run_post_qkv(torch::Tensor& qkv, torch::Tensor& qk_sum_sq, PostMode mode,
                  int num_heads_q, int num_heads_k, int num_heads_v,
                  int head_dim, int q_size, int kv_size, int tp_world,
                  float eps, torch::Tensor& q_norm_weight,
                  torch::Tensor& k_norm_weight, torch::Tensor& cos_sin_cache,
                  bool is_neox, torch::Tensor& position_ids) {
  int q_size_global = q_size * tp_world;
  int kv_size_global = kv_size * tp_world;
  if (mode == PostMode::kFusedAll) {
    q_size_global = q_size;
    kv_size_global = kv_size;
  }
  switch (mode) {
    case PostMode::kFusedAll:
      kernels::dispatch_post_qkv<scalar_t, PostMode::kFusedAll>(
          qkv, qk_sum_sq, num_heads_q, num_heads_k, num_heads_v, head_dim,
          q_size, kv_size, q_size_global, kv_size_global, eps, q_norm_weight,
          k_norm_weight, cos_sin_cache, is_neox, position_ids);
      break;
    case PostMode::kSumLocal:
      kernels::dispatch_post_qkv<scalar_t, PostMode::kSumLocal>(
          qkv, qk_sum_sq, num_heads_q, num_heads_k, num_heads_v, head_dim,
          q_size, kv_size, q_size_global, kv_size_global, eps, q_norm_weight,
          k_norm_weight, cos_sin_cache, is_neox, position_ids);
      break;
    case PostMode::kApplyGlobal:
      kernels::dispatch_post_qkv<scalar_t, PostMode::kApplyGlobal>(
          qkv, qk_sum_sq, num_heads_q, num_heads_k, num_heads_v, head_dim,
          q_size, kv_size, q_size_global, kv_size_global, eps, q_norm_weight,
          k_norm_weight, cos_sin_cache, is_neox, position_ids);
      break;
  }
}

}  // namespace minimax_m2

namespace {

void check_fused_qkv_common(torch::Tensor const& qkv,
                            torch::Tensor const& qk_sum_sq,
                            torch::Tensor const& position_ids) {
  CHECK_INPUT(qkv);
  CHECK_INPUT(qk_sum_sq);
  CHECK_INPUT(position_ids);
  TORCH_CHECK(position_ids.scalar_type() == torch::kInt64);
  TORCH_CHECK(qk_sum_sq.scalar_type() == torch::kFloat32);
  TORCH_CHECK(qk_sum_sq.dim() == 2 && qk_sum_sq.size(1) == 2);
  TORCH_CHECK(qk_sum_sq.size(0) == qkv.size(0));
  TORCH_CHECK(position_ids.size(0) == qkv.size(0));
}

}  // namespace

// Stage 1: qkv_proj + post-process (fully fused when tp_world <= 1).
void minimax_m2_fused_qkv_compute(
    torch::Tensor& qkv, torch::Tensor const& hidden_states,
    torch::Tensor const& qkv_weight, std::optional<torch::Tensor> qkv_bias,
    torch::Tensor& q_norm_weight, torch::Tensor& k_norm_weight,
    torch::Tensor& cos_sin_cache, torch::Tensor& position_ids,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t head_dim, double eps, bool is_neox, int64_t tp_world,
    torch::Tensor& qk_sum_sq) {
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(qkv_weight);
  CHECK_INPUT(q_norm_weight);
  CHECK_INPUT(k_norm_weight);
  CHECK_INPUT(cos_sin_cache);
  check_fused_qkv_common(qkv, qk_sum_sq, position_ids);
  TORCH_CHECK(hidden_states.size(0) == qkv.size(0));

  int q_size = static_cast<int>(num_heads_q * head_dim);
  int kv_size = static_cast<int>(num_heads_k * head_dim);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "minimax_m2_fused_qkv_compute",
                           [&] {
    minimax_m2::qkv_gemm_cuda<scalar_t>(qkv, hidden_states, qkv_weight,
                                        qkv_bias);
    minimax_m2::PostMode mode = (tp_world <= 1)
                                    ? minimax_m2::PostMode::kFusedAll
                                    : minimax_m2::PostMode::kSumLocal;
    minimax_m2::run_post_qkv<scalar_t>(
        qkv, qk_sum_sq, mode, num_heads_q, num_heads_k, num_heads_v, head_dim,
        q_size, kv_size, static_cast<int>(tp_world), static_cast<float>(eps),
        q_norm_weight, k_norm_weight, cos_sin_cache, is_neox, position_ids);
  });
}

// Stage 3: norm + RoPE using global sum_sq in qk_sum_sq (after TP all_reduce).
void minimax_m2_fused_qkv_finalize(
    torch::Tensor& qkv, torch::Tensor& q_norm_weight,
    torch::Tensor& k_norm_weight, torch::Tensor& cos_sin_cache,
    torch::Tensor& position_ids, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, double eps, bool is_neox,
    int64_t tp_world, torch::Tensor& qk_sum_sq) {
  CHECK_INPUT(q_norm_weight);
  CHECK_INPUT(k_norm_weight);
  CHECK_INPUT(cos_sin_cache);
  check_fused_qkv_common(qkv, qk_sum_sq, position_ids);

  int q_size = static_cast<int>(num_heads_q * head_dim);
  int kv_size = static_cast<int>(num_heads_k * head_dim);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "minimax_m2_fused_qkv_finalize",
                           [&] {
    minimax_m2::run_post_qkv<scalar_t>(
        qkv, qk_sum_sq, minimax_m2::PostMode::kApplyGlobal, num_heads_q,
        num_heads_k, num_heads_v, head_dim, q_size, kv_size,
        static_cast<int>(tp_world), static_cast<float>(eps), q_norm_weight,
        k_norm_weight, cos_sin_cache, is_neox, position_ids);
  });
}

/*
 * MiniMax-M2 fused QKV: qkv_proj + flat Q/K RMSNorm + RoPE.
 *
 * Pipeline:
 *   tp == 1:
 *     GEMM: BF16 CUTLASS/cuBLAS, or FP8 block W8A8 (Python W8A8BlockFp8LinearOp)
 *     -> fused token kernel: load q/k once (SMEM) -> sum -> norm + NeoX RoPE
 *
 *   tp > 1:
 *     GEMM (BF16 or FP8 block W8A8)
 *     -> CUTLASS GEMM epilogue (SM90+ BF16) or standalone epilogue: bias + sum(q^2/k^2)
 *     -> [Python all_reduce on qk_sum_sq]
 *     -> token kernel: read q/k once -> norm + NeoX RoPE
 *
 * FP8 weights (float8_e4m3fn + weight_scale_inv, block 128x128) use the same
 * W8A8BlockFp8LinearOp path as Fp8LinearMethod; post-GEMM kernels operate on BF16.
 */

#include <optional>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cub_helpers.h"
#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "minimax_m2/minimax_m2_qkv_cutlass_gemm.cuh"

#define CHECK_TYPE(x, st)                                              \
  TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), \
              ", while ", st, " is expected")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIG(x)

namespace minimax_m2 {

enum class PostMode : int {
  kFusedAll = 0,     // tp==1: sum + norm + rope in one token kernel
  kApplyGlobal = 1,  // tp>1: norm + rope using global sum_sq in qk_sum_sq
};

namespace kernels {

constexpr int TOKEN_BLOCK_THREADS = 256;
constexpr int kWarpSize = 32;
// Per-block dynamic SMEM budget for caching local q/k (bytes).
constexpr int MAX_QK_SMEM_BYTES = 48 * 1024;

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

template <typename scalar_t, typename cache_scalar_t, bool use_smem_cache>
__device__ void norm_rope_one_head(
    scalar_t* qkv_row, scalar_t const* smem_q, scalar_t const* smem_k,
    int head_id, int num_heads_q, int head_dim, int q_size, int kv_size,
    float inv_rms, scalar_t const* q_weight, scalar_t const* k_weight,
    cache_scalar_t const* cos_ptr, cache_scalar_t const* sin_ptr,
    int embed_dim, int lane) {
  using Traits = ScalarTraits<scalar_t>;
  using CacheTraits = CacheScalarTraits<cache_scalar_t>;

  bool const is_q_head = head_id < num_heads_q;
  int col_offset;
  scalar_t const* weight;
  if (is_q_head) {
    col_offset = head_id * head_dim;
    weight = q_weight + col_offset;
  } else {
    int const kv_head = head_id - num_heads_q;
    col_offset = q_size + kv_head * head_dim;
    weight = k_weight + kv_head * head_dim;
  }

  scalar_t* row = qkv_row + col_offset;
  scalar_t const* src_row = row;
  if constexpr (use_smem_cache) {
    src_row = is_q_head ? smem_q + col_offset : smem_k + col_offset - q_size;
  }

  for (int pair_base = 0; pair_base < embed_dim; pair_base += kWarpSize) {
    int const p = pair_base + lane;
    if (p < embed_dim) {
      float nx =
          Traits::to_float(src_row[p]) * inv_rms * Traits::to_float(weight[p]);
      float ny = Traits::to_float(src_row[p + embed_dim]) * inv_rms *
                 Traits::to_float(weight[p + embed_dim]);
      float const cos = CacheTraits::to_float(cos_ptr[p]);
      float const sin = CacheTraits::to_float(sin_ptr[p]);
      row[p] = Traits::from_float(nx * cos - ny * sin);
      row[p + embed_dim] = Traits::from_float(ny * cos + nx * sin);
    }
  }
}

// tp==1: one token block loads q/k once, computes sum, then norm+RoPE.
template <typename scalar_t, typename cache_scalar_t, bool use_smem_cache>
__global__ void fused_qk_sum_norm_rope_token_kernel(
    scalar_t* qkv, int num_tokens, int qkv_dim, int num_heads_q, int num_heads_k,
    int head_dim, int q_size, int kv_size, float eps, scalar_t const* q_weight,
    scalar_t const* k_weight, int64_t const* positions,
    cache_scalar_t const* cos_sin_cache, int rotary_dim) {
  int const t = blockIdx.x;
  if (t >= num_tokens) {
    return;
  }

  extern __shared__ char smem_raw[];
  scalar_t* smem_q = reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* smem_k = smem_q + q_size;

  scalar_t* row = qkv + static_cast<int64_t>(t) * qkv_dim;
  using Traits = ScalarTraits<scalar_t>;

  float sum_sq_q = 0.0f;
  float sum_sq_k = 0.0f;

  for (int i = threadIdx.x; i < q_size; i += blockDim.x) {
    scalar_t v = row[i];
    if constexpr (use_smem_cache) {
      smem_q[i] = v;
    }
    float fv = Traits::to_float(v);
    sum_sq_q += fv * fv;
  }
  for (int i = threadIdx.x; i < kv_size; i += blockDim.x) {
    scalar_t v = row[q_size + i];
    if constexpr (use_smem_cache) {
      smem_k[i] = v;
    }
    float fv = Traits::to_float(v);
    sum_sq_k += fv * fv;
  }

  using BlockReduce = cub::BlockReduce<float, TOKEN_BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage tmp;
  sum_sq_q = BlockReduce(tmp).Reduce(sum_sq_q, CubAddOp{}, blockDim.x);
  __syncthreads();
  sum_sq_k = BlockReduce(tmp).Reduce(sum_sq_k, CubAddOp{}, blockDim.x);

  __shared__ float s_inv_rms_q;
  __shared__ float s_inv_rms_k;
  if (threadIdx.x == 0) {
    s_inv_rms_q = rsqrtf(sum_sq_q / static_cast<float>(q_size) + eps);
    s_inv_rms_k = rsqrtf(sum_sq_k / static_cast<float>(kv_size) + eps);
  }
  __syncthreads();

  int64_t const pos = positions[t];
  int const embed_dim = rotary_dim / 2;
  cache_scalar_t const* cache_row = cos_sin_cache + pos * rotary_dim;
  cache_scalar_t const* cos_ptr = cache_row;
  cache_scalar_t const* sin_ptr = cache_row + embed_dim;

  int const warps_per_block = blockDim.x / kWarpSize;
  int const warp_id = threadIdx.x / kWarpSize;
  int const lane = threadIdx.x % kWarpSize;
  int const total_qk_heads = num_heads_q + num_heads_k;

  for (int hb = 0; hb < total_qk_heads; hb += warps_per_block) {
    int const head_id = hb + warp_id;
    if (head_id >= total_qk_heads) {
      continue;
    }
    float const inv_rms =
        head_id < num_heads_q ? s_inv_rms_q : s_inv_rms_k;
    norm_rope_one_head<scalar_t, cache_scalar_t, use_smem_cache>(
        row, smem_q, smem_k, head_id, num_heads_q, head_dim, q_size, kv_size,
        inv_rms, q_weight, k_weight, cos_ptr, sin_ptr, embed_dim, lane);
  }
}

// tp>1 finalize: global sum_sq already in qk_sum_sq; read q/k once for norm+RoPE.
template <typename scalar_t, typename cache_scalar_t>
__global__ void qk_norm_rope_token_kernel(
    scalar_t* qkv, float const* qk_sum_sq, int num_tokens, int qkv_dim,
    int num_heads_q, int num_heads_k, int head_dim, int q_size, int kv_size,
    int q_size_global, int kv_size_global, float eps, scalar_t const* q_weight,
    scalar_t const* k_weight, int64_t const* positions,
    cache_scalar_t const* cos_sin_cache, int rotary_dim) {
  int const t = blockIdx.x;
  if (t >= num_tokens) {
    return;
  }

  __shared__ float s_inv_rms_q;
  __shared__ float s_inv_rms_k;
  if (threadIdx.x == 0) {
    s_inv_rms_q =
        rsqrtf(qk_sum_sq[t * 2 + 0] / static_cast<float>(q_size_global) + eps);
    s_inv_rms_k =
        rsqrtf(qk_sum_sq[t * 2 + 1] / static_cast<float>(kv_size_global) + eps);
  }
  __syncthreads();

  scalar_t* row = qkv + static_cast<int64_t>(t) * qkv_dim;
  int64_t const pos = positions[t];
  int const embed_dim = rotary_dim / 2;
  cache_scalar_t const* cache_row = cos_sin_cache + pos * rotary_dim;
  cache_scalar_t const* cos_ptr = cache_row;
  cache_scalar_t const* sin_ptr = cache_row + embed_dim;

  int const warps_per_block = blockDim.x / kWarpSize;
  int const warp_id = threadIdx.x / kWarpSize;
  int const lane = threadIdx.x % kWarpSize;
  int const total_qk_heads = num_heads_q + num_heads_k;

  for (int hb = 0; hb < total_qk_heads; hb += warps_per_block) {
    int const head_id = hb + warp_id;
    if (head_id >= total_qk_heads) {
      continue;
    }
    float const inv_rms =
        head_id < num_heads_q ? s_inv_rms_q : s_inv_rms_k;
    norm_rope_one_head<scalar_t, cache_scalar_t, false>(
        row, nullptr, nullptr, head_id, num_heads_q, head_dim, q_size, kv_size,
        inv_rms, q_weight, k_weight, cos_ptr, sin_ptr, embed_dim, lane);
  }
}

template <typename scalar_t, typename cache_scalar_t>
void launch_fused_qk_sum_norm_rope(
    scalar_t* qkv, int num_tokens, int qkv_dim, int num_heads_q,
    int num_heads_k, int head_dim, int q_size, int kv_size, float eps,
    scalar_t const* q_weight, scalar_t const* k_weight,
    int64_t const* positions, cache_scalar_t const* cos_sin_cache,
    int rotary_dim, cudaStream_t stream) {
  dim3 const grid(num_tokens);
  dim3 const block(TOKEN_BLOCK_THREADS);
  int const smem_bytes = (q_size + kv_size) * static_cast<int>(sizeof(scalar_t));
  if (smem_bytes <= MAX_QK_SMEM_BYTES) {
    fused_qk_sum_norm_rope_token_kernel<scalar_t, cache_scalar_t, true>
        <<<grid, block, smem_bytes, stream>>>(
            qkv, num_tokens, qkv_dim, num_heads_q, num_heads_k, head_dim, q_size,
            kv_size, eps, q_weight, k_weight, positions, cos_sin_cache,
            rotary_dim);
  } else {
    fused_qk_sum_norm_rope_token_kernel<scalar_t, cache_scalar_t, false>
        <<<grid, block, 0, stream>>>(
            qkv, num_tokens, qkv_dim, num_heads_q, num_heads_k, head_dim,
            q_size, kv_size, eps, q_weight, k_weight, positions, cos_sin_cache,
            rotary_dim);
  }
}

template <typename scalar_t, typename cache_scalar_t>
void launch_qk_norm_rope(
    scalar_t* qkv, float const* qk_sum_sq, int num_tokens, int qkv_dim,
    int num_heads_q, int num_heads_k, int head_dim, int q_size, int kv_size,
    int q_size_global, int kv_size_global, float eps, scalar_t const* q_weight,
    scalar_t const* k_weight, int64_t const* positions,
    cache_scalar_t const* cos_sin_cache, int rotary_dim, cudaStream_t stream) {
  dim3 const grid(num_tokens);
  dim3 const block(TOKEN_BLOCK_THREADS);
  qk_norm_rope_token_kernel<scalar_t, cache_scalar_t>
      <<<grid, block, 0, stream>>>(
          qkv, qk_sum_sq, num_tokens, qkv_dim, num_heads_q, num_heads_k,
          head_dim, q_size, kv_size, q_size_global, kv_size_global, eps,
          q_weight, k_weight, positions, cos_sin_cache, rotary_dim);
}

template <typename input_t, PostMode mode>
void dispatch_post_qkv(torch::Tensor& qkv, torch::Tensor& qk_sum_sq,
                       int num_heads_q, int num_heads_k, int head_dim,
                       int q_size, int kv_size, int q_size_global,
                       int kv_size_global, float eps,
                       torch::Tensor& q_norm_weight,
                       torch::Tensor& k_norm_weight,
                       torch::Tensor& cos_sin_cache,
                       torch::Tensor& position_ids) {
  int const num_tokens = static_cast<int>(qkv.size(0));
  int const qkv_dim = static_cast<int>(qkv.size(1));
  int const rotary_dim = static_cast<int>(cos_sin_cache.size(1));
  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  VLLM_DISPATCH_FLOATING_TYPES(cos_sin_cache.scalar_type(), "rope_cache", [&] {
    using cache_scalar_t = scalar_t;
    if constexpr (mode == PostMode::kFusedAll) {
      launch_fused_qk_sum_norm_rope<input_t, cache_scalar_t>(
          qkv.data_ptr<input_t>(), num_tokens, qkv_dim, num_heads_q,
          num_heads_k, head_dim, q_size, kv_size, eps,
          q_norm_weight.data_ptr<input_t>(), k_norm_weight.data_ptr<input_t>(),
          reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
          cos_sin_cache.data_ptr<cache_scalar_t>(), rotary_dim, stream);
    } else {
      launch_qk_norm_rope<input_t, cache_scalar_t>(
          qkv.data_ptr<input_t>(), qk_sum_sq.data_ptr<float>(), num_tokens,
          qkv_dim, num_heads_q, num_heads_k, head_dim, q_size, kv_size,
          q_size_global, kv_size_global, eps, q_norm_weight.data_ptr<input_t>(),
          k_norm_weight.data_ptr<input_t>(),
          reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
          cos_sin_cache.data_ptr<cache_scalar_t>(), rotary_dim, stream);
    }
  });
}

}  // namespace kernels

template <typename scalar_t>
void run_post_qkv(torch::Tensor& qkv, torch::Tensor& qk_sum_sq, PostMode mode,
                  int num_heads_q, int num_heads_k, int head_dim, int q_size,
                  int kv_size, int tp_world, float eps,
                  torch::Tensor& q_norm_weight, torch::Tensor& k_norm_weight,
                  torch::Tensor& cos_sin_cache, torch::Tensor& position_ids) {
  int q_size_global = q_size * tp_world;
  int kv_size_global = kv_size * tp_world;
  switch (mode) {
    case PostMode::kFusedAll:
      kernels::dispatch_post_qkv<scalar_t, PostMode::kFusedAll>(
          qkv, qk_sum_sq, num_heads_q, num_heads_k, head_dim, q_size, kv_size,
          q_size, kv_size, eps, q_norm_weight, k_norm_weight, cos_sin_cache,
          position_ids);
      break;
    case PostMode::kApplyGlobal:
      kernels::dispatch_post_qkv<scalar_t, PostMode::kApplyGlobal>(
          qkv, qk_sum_sq, num_heads_q, num_heads_k, head_dim, q_size, kv_size,
          q_size_global, kv_size_global, eps, q_norm_weight, k_norm_weight,
          cos_sin_cache, position_ids);
      break;
  }
}

}  // namespace minimax_m2

namespace {

void check_fused_qkv_common(torch::Tensor const& qkv,
                            torch::Tensor const& qk_sum_sq,
                            torch::Tensor const& position_ids,
                            int64_t num_heads_q, int64_t num_heads_k,
                            int64_t num_heads_v, int64_t head_dim) {
  CHECK_INPUT(qkv);
  CHECK_INPUT(qk_sum_sq);
  CHECK_INPUT(position_ids);
  CHECK_TYPE(position_ids, torch::kInt64);
  TORCH_CHECK(qk_sum_sq.scalar_type() == torch::kFloat32);
  TORCH_CHECK(qk_sum_sq.dim() == 2 && qk_sum_sq.size(1) == 2);
  TORCH_CHECK(qk_sum_sq.size(0) == qkv.size(0));
  TORCH_CHECK(position_ids.size(0) == qkv.size(0));
  TORCH_CHECK(head_dim > 0);
  int64_t const total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(qkv.dim() == 2);
  TORCH_CHECK(qkv.size(1) == total_heads * head_dim);
}

void check_fused_qkv_weights(torch::Tensor const& qkv,
                             torch::Tensor const& q_norm_weight,
                             torch::Tensor const& k_norm_weight,
                             torch::Tensor const& cos_sin_cache,
                             int64_t num_heads_q, int64_t num_heads_k,
                             int64_t head_dim, bool is_neox) {
  CHECK_INPUT(q_norm_weight);
  CHECK_INPUT(k_norm_weight);
  CHECK_INPUT(cos_sin_cache);
  TORCH_CHECK(is_neox,
              "minimax_m2_fused_qkv only supports NeoX-style RoPE");
  TORCH_CHECK(q_norm_weight.dim() == 1 && k_norm_weight.dim() == 1);
  TORCH_CHECK(cos_sin_cache.dim() == 2);
  int64_t const q_size = num_heads_q * head_dim;
  int64_t const kv_size = num_heads_k * head_dim;
  int64_t const rotary_dim = cos_sin_cache.size(1);
  TORCH_CHECK(q_norm_weight.size(0) == q_size);
  TORCH_CHECK(k_norm_weight.size(0) == kv_size);
  TORCH_CHECK(rotary_dim > 0 && rotary_dim % 2 == 0);
  TORCH_CHECK(rotary_dim <= head_dim);
  TORCH_CHECK(qkv.scalar_type() == q_norm_weight.scalar_type());
  TORCH_CHECK(qkv.scalar_type() == k_norm_weight.scalar_type());
}

}  // namespace

void minimax_m2_fused_qkv_gemm_only(
    torch::Tensor& qkv, torch::Tensor const& hidden_states,
    torch::Tensor const& qkv_weight, std::optional<torch::Tensor> qkv_bias) {
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(qkv_weight);
  CHECK_INPUT(qkv);
  TORCH_CHECK(hidden_states.dim() == 2);
  TORCH_CHECK(qkv_weight.dim() == 2);
  TORCH_CHECK(hidden_states.size(0) == qkv.size(0));
  TORCH_CHECK(qkv_weight.size(1) == hidden_states.size(1));
  TORCH_CHECK(qkv_weight.size(0) == qkv.size(1));
  TORCH_CHECK(qkv.scalar_type() == hidden_states.scalar_type());
  TORCH_CHECK(qkv.scalar_type() == qkv_weight.scalar_type());
  if (qkv_bias.has_value()) {
    const torch::Tensor& bias = *qkv_bias;
    CHECK_INPUT(bias);
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == qkv.size(1));
    TORCH_CHECK(bias.scalar_type() == qkv.scalar_type());
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "minimax_m2_fused_qkv_gemm_only",
                           [&] {
    minimax_m2::qkv_gemm_cuda<scalar_t>(qkv, hidden_states, qkv_weight,
                                        qkv_bias, {});
  });
}

void minimax_m2_fused_qkv_epilogue_sum_sq(
    torch::Tensor& qkv, torch::Tensor& qk_sum_sq,
    std::optional<torch::Tensor> qkv_bias, int64_t q_size, int64_t kv_size) {
  CHECK_INPUT(qkv);
  CHECK_INPUT(qk_sum_sq);
  TORCH_CHECK(qk_sum_sq.scalar_type() == torch::kFloat32);
  TORCH_CHECK(qk_sum_sq.dim() == 2 && qk_sum_sq.size(1) == 2);
  TORCH_CHECK(qk_sum_sq.size(0) == qkv.size(0));
  if (qkv_bias.has_value()) {
    const torch::Tensor& bias = *qkv_bias;
    CHECK_INPUT(bias);
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == qkv.size(1));
    TORCH_CHECK(bias.scalar_type() == qkv.scalar_type());
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));
  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "minimax_m2_fused_qkv_epilogue_sum_sq",
                           [&] {
    minimax_m2::QkvGemmOptions opts;
    opts.accumulate_local_sum_sq = true;
    opts.q_size = static_cast<int>(q_size);
    opts.kv_size = static_cast<int>(kv_size);
    opts.qk_sum_sq = &qk_sum_sq;
    minimax_m2::launch_gemm_epilogue_sum_sq<scalar_t>(
        qkv, qkv_bias, opts, stream);
  });
}

void minimax_m2_fused_qkv_compute(
    torch::Tensor& qkv, torch::Tensor const& hidden_states,
    torch::Tensor const& qkv_weight, std::optional<torch::Tensor> qkv_bias,
    torch::Tensor& q_norm_weight, torch::Tensor& k_norm_weight,
    torch::Tensor& cos_sin_cache, torch::Tensor& position_ids,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t head_dim, double eps, bool is_neox, int64_t tp_world,
    torch::Tensor& qk_sum_sq, bool use_standalone_sum_epilogue) {
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(qkv_weight);
  CHECK_INPUT(qkv);
  check_fused_qkv_common(qkv, qk_sum_sq, position_ids, num_heads_q,
                         num_heads_k, num_heads_v, head_dim);
  check_fused_qkv_weights(qkv, q_norm_weight, k_norm_weight, cos_sin_cache,
                          num_heads_q, num_heads_k, head_dim, is_neox);
  TORCH_CHECK(hidden_states.dim() == 2);
  TORCH_CHECK(qkv_weight.dim() == 2);
  TORCH_CHECK(hidden_states.size(0) == qkv.size(0));
  TORCH_CHECK(qkv_weight.size(1) == hidden_states.size(1));
  TORCH_CHECK(qkv_weight.size(0) == qkv.size(1));
  TORCH_CHECK(qkv.scalar_type() == hidden_states.scalar_type());
  TORCH_CHECK(qkv.scalar_type() == qkv_weight.scalar_type());
  if (qkv_bias.has_value()) {
    const torch::Tensor& bias = *qkv_bias;
    CHECK_INPUT(bias);
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == qkv.size(1));
    TORCH_CHECK(bias.scalar_type() == qkv.scalar_type());
  }

  int q_size = static_cast<int>(num_heads_q * head_dim);
  int kv_size = static_cast<int>(num_heads_k * head_dim);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "minimax_m2_fused_qkv_compute",
                           [&] {
    minimax_m2::QkvGemmOptions gemm_opts;
    if (tp_world > 1) {
      gemm_opts.accumulate_local_sum_sq = true;
      gemm_opts.use_standalone_sum_epilogue = use_standalone_sum_epilogue;
      gemm_opts.q_size = q_size;
      gemm_opts.kv_size = kv_size;
      gemm_opts.qk_sum_sq = &qk_sum_sq;
    }
    minimax_m2::qkv_gemm_cuda<scalar_t>(qkv, hidden_states, qkv_weight,
                                        qkv_bias, gemm_opts);
    if (tp_world <= 1) {
      minimax_m2::run_post_qkv<scalar_t>(
          qkv, qk_sum_sq, minimax_m2::PostMode::kFusedAll, num_heads_q,
          num_heads_k, head_dim, q_size, kv_size, static_cast<int>(tp_world),
          static_cast<float>(eps), q_norm_weight, k_norm_weight, cos_sin_cache,
          position_ids);
    }
  });
}

void minimax_m2_fused_qkv_post_fused(
    torch::Tensor& qkv, torch::Tensor& q_norm_weight,
    torch::Tensor& k_norm_weight, torch::Tensor& cos_sin_cache,
    torch::Tensor& position_ids, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, double eps, bool is_neox,
    torch::Tensor& qk_sum_sq) {
  CHECK_INPUT(qkv);
  check_fused_qkv_common(qkv, qk_sum_sq, position_ids, num_heads_q,
                         num_heads_k, num_heads_v, head_dim);
  check_fused_qkv_weights(qkv, q_norm_weight, k_norm_weight, cos_sin_cache,
                          num_heads_q, num_heads_k, head_dim, is_neox);

  int q_size = static_cast<int>(num_heads_q * head_dim);
  int kv_size = static_cast<int>(num_heads_k * head_dim);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "minimax_m2_fused_qkv_post_fused",
                           [&] {
    minimax_m2::run_post_qkv<scalar_t>(
        qkv, qk_sum_sq, minimax_m2::PostMode::kFusedAll, num_heads_q,
        num_heads_k, head_dim, q_size, kv_size, 1, static_cast<float>(eps),
        q_norm_weight, k_norm_weight, cos_sin_cache, position_ids);
  });
}

void minimax_m2_fused_qkv_finalize(
    torch::Tensor& qkv, torch::Tensor& q_norm_weight,
    torch::Tensor& k_norm_weight, torch::Tensor& cos_sin_cache,
    torch::Tensor& position_ids, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, double eps, bool is_neox,
    int64_t tp_world, torch::Tensor& qk_sum_sq) {
  CHECK_INPUT(qkv);
  check_fused_qkv_common(qkv, qk_sum_sq, position_ids, num_heads_q,
                         num_heads_k, num_heads_v, head_dim);
  check_fused_qkv_weights(qkv, q_norm_weight, k_norm_weight, cos_sin_cache,
                          num_heads_q, num_heads_k, head_dim, is_neox);

  int q_size = static_cast<int>(num_heads_q * head_dim);
  int kv_size = static_cast<int>(num_heads_k * head_dim);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "minimax_m2_fused_qkv_finalize",
                           [&] {
    minimax_m2::run_post_qkv<scalar_t>(
        qkv, qk_sum_sq, minimax_m2::PostMode::kApplyGlobal, num_heads_q,
        num_heads_k, head_dim, q_size, kv_size, static_cast<int>(tp_world),
        static_cast<float>(eps), q_norm_weight, k_norm_weight, cos_sin_cache,
        position_ids);
  });
}

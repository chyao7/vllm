#pragma once

#include <optional>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

#include "cutlass_extensions/common.hpp"
#include "minimax_m2/minimax_m2_qkv_gemm_epilogue.cuh"
#include "minimax_m2/minimax_m2_qkv_cutlass_sum_epilogue.cuh"

#ifndef USE_ROCM
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh"
#include "quantization/w8a8/cutlass/c3x/scaled_mm.cuh"
#endif  // USE_ROCM

namespace minimax_m2 {

struct QkvGemmOptions {
  bool accumulate_local_sum_sq = false;
  // If true with accumulate_local_sum_sq: TrivialEpilogue GEMM + standalone
  // sum kernel (scheme B). If false: CUTLASS epilogue fused sum (scheme C).
  bool use_standalone_sum_epilogue = false;
  int q_size = 0;
  int kv_size = 0;
  torch::Tensor* qk_sum_sq = nullptr;
};

// hidden [M, K] @ weight^T [N, K] -> out [M, N]
template <typename scalar_t>
void qkv_gemm_cublas(torch::Tensor& out, torch::Tensor const& input,
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

#ifndef USE_ROCM

template <typename Gemm>
void qkv_gemm_cutlass_sm90_impl(
    torch::Tensor& out, torch::Tensor const& input,
    torch::Tensor const& weight,
    typename Gemm::Epilogue::ArgumentType epilogue_args) {
  using GemmKernel = typename Gemm::GemmKernel;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  int const M = static_cast<int>(input.size(0));
  int const K = static_cast<int>(input.size(1));
  int const N = static_cast<int>(weight.size(0));

  auto problem_shape = cute::make_shape(M, N, K, 1);

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;

  StrideA a_stride = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(M, K, 1));
  StrideB b_stride = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(N, K, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(M, N, 1));

  auto* a_ptr = reinterpret_cast<ElementAB*>(input.data_ptr());
  auto* b_ptr = reinterpret_cast<ElementAB*>(weight.data_ptr());
  auto* c_ptr = reinterpret_cast<ElementD*>(out.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr,
                                                       b_stride};
  typename GemmKernel::EpilogueArguments epilogue_args_full{
      epilogue_args, c_ptr, c_stride, c_ptr, c_stride};

  vllm::c3x::cutlass_gemm_caller<GemmKernel>(
      out.device(), problem_shape, mainloop_args, epilogue_args_full);
}

template <typename ElementT>
void qkv_gemm_cutlass_sm90(torch::Tensor& out, torch::Tensor const& input,
                           torch::Tensor const& weight) {
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using Gemm =
      vllm::cutlass_3x_gemm<ElementT, ElementT, vllm::c3x::TrivialEpilogue,
                            TileShape, ClusterShape, KernelSchedule,
                            EpilogueSchedule>;
  qkv_gemm_cutlass_sm90_impl<Gemm>(out, input, weight,
                                   Gemm::Epilogue::prepare_args());
}

template <typename ElementT>
void qkv_gemm_cutlass_sm90_sum_sq(
    torch::Tensor& out, torch::Tensor const& input,
    torch::Tensor const& weight,
    std::optional<torch::Tensor> const& bias, float* qk_sum_sq, int q_size,
    int kv_size) {
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using Gemm = vllm::cutlass_3x_gemm<
      ElementT, ElementT, vllm::minimax_m2::cutlass_epilogue::QkvSumSqEpilogue,
      TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>;
  qkv_gemm_cutlass_sm90_impl<Gemm>(
      out, input, weight,
      Gemm::Epilogue::prepare_args(bias, qk_sum_sq, q_size, kv_size));
}

#endif  // USE_ROCM

template <typename scalar_t>
void launch_gemm_epilogue_sum_sq(
    torch::Tensor& out, std::optional<torch::Tensor> const& bias,
    QkvGemmOptions const& opts, cudaStream_t stream) {
  TORCH_CHECK(opts.qk_sum_sq != nullptr);
  scalar_t const* bias_ptr = nullptr;
  if (bias.has_value()) {
    bias_ptr = bias->data_ptr<scalar_t>();
  }
  gemm_epilogue::launch_qkv_gemm_epilogue_sum_sq<scalar_t>(
      out.data_ptr<scalar_t>(), opts.qk_sum_sq->data_ptr<float>(), bias_ptr,
      static_cast<int>(out.size(0)), static_cast<int>(out.size(1)), opts.q_size,
      opts.kv_size, stream);
}

template <typename scalar_t>
void qkv_gemm_cuda(torch::Tensor& out, torch::Tensor const& input,
                   torch::Tensor const& weight,
                   std::optional<torch::Tensor> const& bias,
                   QkvGemmOptions const& opts = {}) {
  auto stream = at::cuda::getCurrentCUDAStream(out.get_device());

#ifndef USE_ROCM
#if defined ENABLE_MINIMAX_M2_CUTLASS_SM90 && ENABLE_MINIMAX_M2_CUTLASS_SM90
  if (get_sm_version_num() >= 90 && get_sm_version_num() < 100) {
    if (opts.accumulate_local_sum_sq) {
      TORCH_CHECK(opts.qk_sum_sq != nullptr);
      const bool defer_bias = bias.has_value();
      if (opts.use_standalone_sum_epilogue) {
        if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          qkv_gemm_cutlass_sm90<cutlass::bfloat16_t>(out, input, weight);
        } else {
          qkv_gemm_cutlass_sm90<cutlass::half_t>(out, input, weight);
        }
        launch_gemm_epilogue_sum_sq<scalar_t>(
            out, defer_bias ? bias : std::nullopt, opts, stream);
      } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        qkv_gemm_cutlass_sm90_sum_sq<cutlass::bfloat16_t>(
            out, input, weight, bias, opts.qk_sum_sq->data_ptr<float>(),
            opts.q_size, opts.kv_size);
      } else {
        qkv_gemm_cutlass_sm90_sum_sq<cutlass::half_t>(
            out, input, weight, bias, opts.qk_sum_sq->data_ptr<float>(),
            opts.q_size, opts.kv_size);
      }
    } else {
      if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        qkv_gemm_cutlass_sm90<cutlass::bfloat16_t>(out, input, weight);
      } else {
        qkv_gemm_cutlass_sm90<cutlass::half_t>(out, input, weight);
      }
      if (bias.has_value()) {
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == weight.size(0));
        out.add_(*bias);
      }
    }
    return;
  }
#endif  // ENABLE_MINIMAX_M2_CUTLASS_SM90
#endif  // USE_ROCM

  if (opts.accumulate_local_sum_sq) {
    qkv_gemm_cublas<scalar_t>(out, input, weight, std::nullopt);
    launch_gemm_epilogue_sum_sq<scalar_t>(
        out, bias, opts, stream);
  } else {
    qkv_gemm_cublas<scalar_t>(out, input, weight, bias);
  }
}

}  // namespace minimax_m2

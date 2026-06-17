#pragma once

#include <optional>

#include <torch/all.h>

#ifndef USE_ROCM

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cute/tensor.hpp"
#include "cutlass_extensions/torch_utils.hpp"

namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;

// Aux visitor for Sm90SplitTreeVisitor: accumulate local sum(q^2) / sum(k^2) per
// token row while the GEMM epilogue still has acc+bias values in registers.
//
// Scheme C (CTA-tile reduction): visit() atomically accumulates into per-CTA smem
// buckets (one per row in the M-tile); end() flushes each row's partial sum to
// global qk_sum_sq with a single atomicAdd per q/k bucket. This avoids the
// per-element global atomic storm of the naive implementation (~7168 atomics/row).
template <class CtaTileShapeMNK>
struct Sm90QkvSumSqAccumulate {
  static constexpr int CtaM = cute::size<0>(CtaTileShapeMNK{});

  struct SharedStorage {
    float smem_sum_q[CtaM];
    float smem_sum_k[CtaM];
  };

  struct Arguments {
    float* ptr_qk_sum_sq = nullptr;
    int q_size = 0;
    int kv_size = 0;
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&,
                                                    Arguments const& args,
                                                    void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&,
                                              Arguments const&, void*,
                                              cudaStream_t,
                                              CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  CUTLASS_HOST_DEVICE Sm90QkvSumSqAccumulate() {}
  CUTLASS_HOST_DEVICE Sm90QkvSumSqAccumulate(Params const& params,
                                             SharedStorage const& storage)
      : params(params),
        smem_sum_q_(const_cast<float*>(storage.smem_sum_q)),
        smem_sum_k_(const_cast<float*>(storage.smem_sum_k)) {}

  Params params;
  float* smem_sum_q_ = nullptr;
  float* smem_sum_k_ = nullptr;

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      ProducerLoadArgs<Args...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class CTensor, class ThrResidue>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE ConsumerStoreCallbacks(
        CTensor tCcD, ThrResidue residue_tCcD, Params const& params,
        float* smem_sum_q, float* smem_sum_k, int m_tile, int problem_m,
        int num_threads, int thread_idx)
        : tCcD(tCcD),
          residue_tCcD(residue_tCcD),
          params(params),
          smem_sum_q_(smem_sum_q),
          smem_sum_k_(smem_sum_k),
          m_tile_(m_tile),
          problem_m_(problem_m),
          num_threads_(num_threads),
          thread_idx_(thread_idx) {}

    CTensor tCcD;
    ThrResidue residue_tCcD;
    Params params;
    float* smem_sum_q_;
    float* smem_sum_k_;
    int m_tile_;
    int problem_m_;
    int num_threads_;
    int thread_idx_;

    CUTLASS_DEVICE bool begin_sync_needed() const { return true; }

    CUTLASS_DEVICE void begin() {
      if (smem_sum_q_ == nullptr || smem_sum_k_ == nullptr) {
        return;
      }
      for (int r = thread_idx_; r < CtaM; r += num_threads_) {
        smem_sum_q_[r] = 0.f;
        smem_sum_k_[r] = 0.f;
      }
    }

    template <typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE Array<ElementInput, FragmentSize> visit(
        Array<ElementInput, FragmentSize> const& frg_input, int epi_v,
        int epi_m, int epi_n) {
      if (params.ptr_qk_sum_sq != nullptr && smem_sum_q_ != nullptr &&
          smem_sum_k_ != nullptr) {
        using ConvertInput = NumericArrayConverter<
            float, ElementInput, FragmentSize,
            FloatRoundStyle::round_to_nearest>;
        ConvertInput convert{};
        Array<float, FragmentSize> frg_f = convert(frg_input);
        Tensor tCcD_mn = tCcD(_, _, _, epi_m, epi_n);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          if (!elem_less(tCcD_mn(epi_v * FragmentSize + i), residue_tCcD)) {
            continue;
          }
          auto coord = tCcD_mn(epi_v * FragmentSize + i);
          int const row_local = get<0>(coord);
          int const col = get<1>(coord);
          if (row_local < 0 || row_local >= CtaM) {
            continue;
          }
          float const sq = frg_f[i] * frg_f[i];
          if (col < params.q_size) {
            atomicAdd(&smem_sum_q_[row_local], sq);
          } else if (col < params.q_size + params.kv_size) {
            atomicAdd(&smem_sum_k_[row_local], sq);
          }
        }
      }
      return frg_input;
    }

    CUTLASS_DEVICE void end() {
      if (params.ptr_qk_sum_sq == nullptr || smem_sum_q_ == nullptr ||
          smem_sum_k_ == nullptr) {
        return;
      }

      cutlass::arch::NamedBarrier::sync(
          num_threads_, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

      int const global_row_base = m_tile_ * CtaM;
      for (int row_local = thread_idx_; row_local < CtaM;
           row_local += num_threads_) {
        int const global_row = global_row_base + row_local;
        if (global_row >= problem_m_) {
          continue;
        }
        float const q_partial = smem_sum_q_[row_local];
        float const k_partial = smem_sum_k_[row_local];
        if (q_partial != 0.f) {
          atomicAdd(params.ptr_qk_sum_sq + global_row * 2 + 0, q_partial);
        }
        if (k_partial != 0.f) {
          atomicAdd(params.ptr_qk_sum_sq + global_row * 2 + 1, k_partial);
        }
      }
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    int const num_threads = size(args.tiled_copy);
    return ConsumerStoreCallbacks<decltype(args.tCcD),
                                  decltype(args.residue_cD)>(
        args.tCcD, args.residue_cD, params, smem_sum_q_, smem_sum_k_,
        static_cast<int>(m), static_cast<int>(M), num_threads, args.thread_idx);
  }
};

}  // namespace cutlass::epilogue::fusion

namespace vllm::minimax_m2::cutlass_epilogue {

using namespace cute;
using namespace cutlass::epilogue::fusion;

// GEMM epilogue: D = acc (+ bias) with fused local sum(q^2)/sum(k^2) into qk_sum_sq.
template <typename ElementAcc, typename ElementD, typename TileShape>
struct QkvSumSqEpilogue {
 private:
  using ElementCompute = float;
  using ElementScalar = float;
  using CtaTileShapeMNK = TileShape;
  static constexpr int AlignmentBias =
      128 / cutlass::sizeof_bits<ElementD>::value;
  using StrideAlpha = Stride<_0, _0, int64_t>;
  using StrideBias = Stride<_0, _1, int64_t>;

  using BiasBroadcast = Sm90RowBroadcast<0, CtaTileShapeMNK, ElementD,
                                         ElementCompute, StrideBias,
                                         AlignmentBias, true>;

  using AlphaBroadcast =
      Sm90ScalarBroadcast<ElementScalar, StrideAlpha, 1, cutlass::multiplies>;

  using InputTree = Sm90EVT<
      Sm90Compute<cutlass::homogeneous_multiply_add, ElementCompute,
                  ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
      AlphaBroadcast, Sm90AccFetch, BiasBroadcast>;

  using SumSqAux = Sm90QkvSumSqAccumulate<CtaTileShapeMNK>;

  using OutputTree = Sm90EVT<
      Sm90Compute<cutlass::epilogue::thread::Identity, ElementD, ElementCompute,
                  cutlass::FloatRoundStyle::round_to_nearest>,
      Sm90SplitTreeFetch>;

 public:
  using EVTCompute =
      Sm90SplitTreeVisitor<InputTree, OutputTree, SumSqAux>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(
      std::optional<torch::Tensor> const& bias, float* qk_sum_sq, int q_size,
      int kv_size) {
    using TorchElementD = equivalent_scalar_type_t<ElementD>;
    ElementD const* bias_ptr = nullptr;
    if (bias.has_value()) {
      bias_ptr = reinterpret_cast<ElementD const*>(
          bias->data_ptr<TorchElementD>());
    }

    typename AlphaBroadcast::Arguments alpha_args{};
    alpha_args.scalars[0] = ElementScalar(1);
    alpha_args.scalar_ptrs[0] = nullptr;
    alpha_args.dScalar[0] = StrideAlpha{_0{}, _0{}, 0};

    typename BiasBroadcast::Arguments bias_args{};
    bias_args.ptr_row = bias_ptr;
    bias_args.null_default = ElementD(0);
    bias_args.dRow = StrideBias{_0{}, _1{}, 0};

    typename SumSqAux::Arguments sum_args{};
    sum_args.ptr_qk_sum_sq = qk_sum_sq;
    sum_args.q_size = q_size;
    sum_args.kv_size = kv_size;

    return ArgumentType{
        typename InputTree::Arguments{
            alpha_args,
            {},
            bias_args,
            {},
        },
        sum_args,
        typename OutputTree::Arguments{{}, {}},
    };
  }
};

}  // namespace vllm::minimax_m2::cutlass_epilogue

#endif  // USE_ROCM

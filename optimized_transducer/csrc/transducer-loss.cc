// optimized_transducer/csrc/transducer_loss.cc
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
//
#include "optimized_transducer/csrc/transducer-loss.h"

namespace ot {

std::pair<torch::Tensor, torch::optional<torch::Tensor>>
ComputeTransducerLossCpu(torch::Tensor &logits, const torch::Tensor &targets,
                         const torch::Tensor &logit_lengths,
                         const torch::Tensor &target_lengths, int32_t blank,
                         bool from_log_softmax, bool one_sym_per_frame);

std::pair<torch::Tensor, torch::optional<torch::Tensor>>
ComputeTransducerLossCuda(torch::Tensor &logits, const torch::Tensor &targets,
                          const torch::Tensor &logit_lengths,
                          const torch::Tensor &target_lengths, int32_t blank,
                          bool from_log_softmax, bool one_sym_per_frame);

std::pair<torch::Tensor, torch::optional<torch::Tensor>> ComputeTransducerLoss(
    torch::Tensor &logits, const torch::Tensor &targets,
    const torch::Tensor &logit_lengths, const torch::Tensor &target_lengths,
    int32_t blank, bool from_log_softmax, bool one_sym_per_frame) {
  auto logits_arg = torch::TensorArg(logits, "logits", 0);
  auto targets_arg = torch::TensorArg(targets, "targets", 1);
  auto logit_lengths_arg = torch::TensorArg(logit_lengths, "logit_lengths", 2);
  auto target_lengths_arg =
      torch::TensorArg(target_lengths, "target_lengths", 3);

  torch::CheckedFrom checked_from = "ComputeTransducerLoss";

  torch::checkScalarType(checked_from, logits_arg, torch::kFloat);
  torch::checkScalarType(checked_from, targets_arg, torch::kInt32);
  torch::checkScalarType(checked_from, logit_lengths_arg, torch::kInt32);
  torch::checkScalarType(checked_from, target_lengths_arg, torch::kInt32);

  torch::checkAllContiguous(
      checked_from,
      {logits_arg, targets_arg, logit_lengths_arg, target_lengths_arg});

  torch::checkDim(checked_from, logits_arg, 2);
  torch::checkDim(checked_from, targets_arg, 2);
  torch::checkDim(checked_from, logit_lengths_arg, 1);
  torch::checkDim(checked_from, target_lengths_arg, 1);
  torch::checkSameSize(checked_from, logit_lengths_arg, target_lengths_arg);

  // + 1 here since each sequence is prepended with a blank
  int32_t sum_all_TU = (logit_lengths * (target_lengths + 1)).sum().item<int>();

  torch::checkSize(checked_from, logits_arg, 0, sum_all_TU);

  int32_t targets_max = targets.max().item<int32_t>();
  TORCH_CHECK(targets_max < logits.size(1),
              "targets.max() vs logits.size(1) is ", targets_max, " vs ",
              logits.size(1), ". Expect targets.max() < logits.size(1)");

  TORCH_CHECK((0 <= blank && blank < logits.size(1)),
              "blank vs logits.size(1) is ", blank, " vs ", logits.size(1),
              ". Expect 0 <= blank < logits.size(1)");

  TORCH_CHECK((targets.size(1) == target_lengths.max().item<int32_t>()),
              "targets.size(1) should be equal to target_lengths.max().",
              "But targets.size(1) is ", targets.size(1),
              ", and target_lengths.max() is ",
              target_lengths.max().item<int32_t>());

  if (logits.device().type() == torch::kCPU) {
    torch::checkDeviceType(checked_from,
                           {logits, targets, logit_lengths, target_lengths},
                           torch::kCPU);
    return ComputeTransducerLossCpu(logits, targets, logit_lengths,
                                    target_lengths, blank, from_log_softmax,
                                    one_sym_per_frame);
  } else {
#ifdef OT_WITH_CUDA
    torch::checkAllSameGPU(
        checked_from,
        {logits_arg, targets_arg, logit_lengths_arg, target_lengths_arg});

    return ComputeTransducerLossCuda(logits, targets, logit_lengths,
                                     target_lengths, blank, from_log_softmax,
                                     one_sym_per_frame);

#else
    throw std::runtime_error(
        R"(optimized_transducer was not compiled with CUDA support!
    Please use
      cmake -DOT_WITH_CUDA=ON
    while compiling it."
    )");

    return {};
#endif
  }
}

}  // namespace ot

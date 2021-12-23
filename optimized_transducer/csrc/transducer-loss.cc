// optimized_transducer/csrc/transducer_loss.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
//
#include "optimized_transducer/csrc/transducer-loss.h"

namespace ot {

std::pair<torch::Tensor, torch::optional<torch::Tensor>> ComputeTransducerLoss(
    torch::Tensor &logits, const torch::Tensor &targets,
    const torch::Tensor &logit_lengths, const torch::Tensor &target_lengths,
    int32_t blank, double clamp) {
  torch::Device device = logits.device();
  torch::ScalarType scalar_type = logits.scalar_type();

  return {};
}

}  // namespace ot

// optimized_transducer/csrc/transducer_loss.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
//
#include "optimized_transducer/csrc/transducer-loss.h"

namespace ot {

/* Compute the emit and non-emit log probabilities for each (t, u) in the grid.

   Return a tensor of shape (logits.size(0), 2). Column 0 contains the log-prob
   for non-emit, i.e., horizontal transition from t to t+1. Column 1 contains
   the log-prob for emit, i.e., vertical transition from u to u+1.
 */
static torch::Tensor ComputeLogProbs(const torch::Tensor &logits,
                                     const torch::Tensor &denominator,
                                     const torch::Tensor &targets,
                                     const torch::Tensor &logit_lengths,
                                     const torch::Tensor &target_lengths,
                                     int32_t blank) {
  torch::ArrayRef<int32_t> logit_len_arr(logit_lengths.data_ptr<int32_t>(),
                                         logit_lengths.numel());

  torch::ArrayRef<int32_t> target_len_arr(target_lengths.data_ptr<int32_t>(),
                                          target_lengths.numel());

  torch::ArrayRef<float> den_arr(denominator.data_ptr<float>(),
                                 denominator.numel());

  int32_t batch_size = targets.size(0);

  // column 0 is the log-prob for blanks
  // column 0 is the log-prob for symbols
  torch::Tensor log_probs = torch::empty({logits.size(0), 2}, logits.options());
  constexpr int32_t kBlankCol = 0;
  constexpr int32_t kSymCol = 1;

  const float *p_logits = logits.data_ptr<float>();
  float *p_log_probs = log_probs.data_ptr<float>();
  const float *p_den = denominator.data_ptr<float>();

  const int32_t *p_targets = targets.data_ptr<int32_t>();

  int32_t V = logits.size(1);  // vocabulary size including blank

  for (int32_t b = 0; b != batch_size; ++b, p_targets += targets.size(1)) {
    int32_t T = logit_len_arr[b];

    // p1 means plus one
    // We need to plus one since it is prepended with a blank
    int32_t U_p1 = target_len_arr[b] + 1;
    for (int32_t t = 0; t != T; ++t) {
      for (int32_t u = 0; u != U_p1;
           ++u, ++p_den, p_log_probs += 2, p_logits += V) {
        // for blank
        p_log_probs[kBlankCol] = p_logits[blank] - *p_den;

        // for non-blank
        if (u < U_p1 - 1) {
          p_log_probs[kSymCol] = p_logits[p_targets[u]] - *p_den;
        }
      }
    }
  }

  return log_probs;
}

std::pair<torch::Tensor, torch::optional<torch::Tensor>> ComputeTransducerLoss(
    torch::Tensor &logits, const torch::Tensor &targets,
    const torch::Tensor &logit_lengths, const torch::Tensor &target_lengths,
    int32_t blank, double clamp /*=0*/) {
  auto logits_arg = torch::TensorArg(logits, "logits", 0);
  auto targets_arg = torch::TensorArg(targets, "targets", 1);
  auto logit_lengths_arg = torch::TensorArg(logit_lengths, "logit_lengths", 2);
  auto target_lengths_arg =
      torch::TensorArg(target_lengths, "target_lengths", 3);

  torch::CheckedFrom checked_from = "ComputeTransducerLoss";
  if (logits.device().type() == torch::kCPU) {
    torch::checkDeviceType(checked_from,
                           {logits, targets, logit_lengths, target_lengths},
                           torch::kCPU);
  } else {
    torch::checkAllSameGPU(
        checked_from,
        {logits_arg, targets_arg, logit_lengths_arg, target_lengths_arg});
  }

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

  int32_t sum_all_TU = (logit_lengths * (target_lengths + 1)).sum().item<int>();

  torch::checkSize(checked_from, logits_arg, 0, sum_all_TU);

  int32_t targets_max = targets.max().item<int32_t>();
  TORCH_CHECK(targets_max < logits.size(1),
              "targets.max() vs logits.size(1) is ", targets_max, " vs ",
              logits.size(1), ". Expect targets.max() < logits.size(1)");

  TORCH_CHECK((0 <= blank && blank < logits.size(1)),
              "blank vs logits.size(1) is ", blank, " vs ", logits.size(1),
              ". Expect 0 <= blank < logits.size(1)");

  // The denominator for the log-softmax.
  // Note that it is positive at present.
  torch::Tensor denominator = logits.logsumexp(/*dim*/ 1, /*keepdim*/ false);

  torch::Tensor log_probs = ComputeLogProbs(
      logits, denominator, targets, logit_lengths, target_lengths, blank);
  std::cout << log_probs << "\n";
  return {};
}

}  // namespace ot

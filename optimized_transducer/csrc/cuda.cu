// optimized_transducer/csrc/cuda.cu
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

#include "moderngpu/kernel_load_balance.hxx"
#include "optimized_transducer/csrc/kernels.h"
#include "optimized_transducer/csrc/moderngpu-allocator.h"
#include "torch/script.h"

static constexpr int32_t kMaxThreadsPerBlock = 1024;
static constexpr int32_t kWarpSize = 32;

namespace ot {

// See https://github.com/k2-fsa/k2/blob/master/k2/csrc/utils.cu#L75
// for the meaning of row splits and row IDs.
/**

  @param row_splits  A 1-D tensor of dtype torch.int32. Its first
                     element should be zero.
  @param num_elems   If -1, it is equal to row_splits[-1].
                     If not -1, it must be equal to row_splits[-1].

  @return Return a 1-D tensor of dtype torch.int32. Its lengths
          equals to num_elems.
 */
torch::Tensor RowSplitsToRowIds(const torch::Tensor &row_splits,
                                int32_t num_elems = -1) {
  torch::CheckedFrom c = "RowSplitsToRowIds";
  auto row_splits_arg = torch::TensorArg(row_splits, "row_splits", 0);
  torch::checkScalarType(c, row_splits_arg, torch::kInt32);
  torch::checkDim(c, row_splits_arg, 1);
  torch::checkContiguous(c, row_splits_arg);

  int32_t num_rows = row_splits.size(0) - 1;
  const int32_t *p_row_splits = row_splits.data_ptr<int32_t>();
  if (num_elems == -1) {
    num_elems = row_splits.cpu().data_ptr<int32_t>()[num_rows];
  }

  torch::Tensor row_ids = torch::empty({num_elems}, row_splits.options());
  ModernGpuAllocator allocator;
  mgpu::load_balance_search(num_elems, p_row_splits, num_rows,
                            row_ids.data_ptr<int32_t>(), allocator);
  return row_ids;
}

static std::pair<torch::Tensor, torch::Tensor> ComputeLogProbs(
    const torch::Tensor &logits, const torch::Tensor &denominator,
    const torch::Tensor &targets, const torch::Tensor &logit_lengths,
    const torch::Tensor &target_lengths, int32_t blank) {
  // + 1 here since each sequence is prepended with a blank
  torch::Tensor sizes = logit_lengths * (target_lengths + 1);
  torch::Tensor row_splits = torch::cumsum(sizes, -1, torch::kInt);
  torch::Tensor zero = torch::zeros({1}, row_splits.options());
  row_splits = torch::cat({zero, row_splits}, -1);
  torch::Tensor row_ids = RowSplitsToRowIds(row_splits, logits.size(0));

  const float *p_logits = logits.data_ptr<float>();
  const float *p_den = denominator.data_ptr<float>();
  const int32_t *p_targets = targets.data_ptr<int32_t>();
  const int32_t *p_target_lengths = target_lengths.data_ptr<int32_t>();
  const int32_t *p_row_splits = row_splits.data_ptr<int32_t>();
  const int32_t *p_row_ids = row_ids.data_ptr<int32_t>();

  torch::Tensor log_probs = torch::empty({logits.size(0), 2}, logits.options());
  float *p_log_probs = log_probs.data_ptr<float>();

  int32_t num_blocks =
      (logits.size(0) + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;

  ComputeLogProbs<<<num_blocks, kMaxThreadsPerBlock>>>(
      p_logits, p_den, p_targets, p_target_lengths, blank, p_row_splits,
      p_row_ids, logits.size(0), logits.size(1), targets.size(1), p_log_probs);

  return {log_probs, row_splits};
}

static torch::Tensor ComputeAlpha(const torch::Tensor &log_probs,
                                  const torch::Tensor &logit_lengths,
                                  const torch::Tensor &target_lengths,
                                  const torch::Tensor &row_splits) {
  // it is prepended with a blank so we need to use +1 here
  int32_t max_T = logit_lengths.max().item<int32_t>();
  int32_t max_U_p1 = target_lengths.max().item<int32_t>() + 1;
  int32_t batch_size = logit_lengths.size(0);

  int32_t num_warps = (max_T + kWarpSize - 1) / kWarpSize;
  dim3 block_dims(num_warps, max_U_p1, batch_size);
  dim3 thread_dims(kWarpSize);

  // torch::Tensor alpha = torch::empty({log_probs.size(0)},
  // log_probs.options());
  torch::Tensor alpha = torch::ones({log_probs.size(0)}, log_probs.options());
  torch::Tensor counter =
      torch::zeros({batch_size * max_U_p1}, logit_lengths.options());

  const float *p_log_probs = log_probs.data_ptr<float>();
  const int32_t *p_logit_lengths = logit_lengths.data_ptr<int32_t>();
  const int32_t *p_target_lengths = target_lengths.data_ptr<int32_t>();
  const int32_t *p_row_splits = row_splits.data_ptr<int32_t>();
  int32_t *p_counter = counter.data_ptr<int32_t>();
  float *p_alpha = alpha.data_ptr<float>();

  ComputeAlpha<<<block_dims, thread_dims>>>(
      p_log_probs, p_logit_lengths, p_target_lengths, p_row_splits, max_T,
      max_U_p1, p_counter, p_alpha);
  return alpha;
}

std::pair<torch::Tensor, torch::optional<torch::Tensor>>
ComputeTransducerLossCuda(torch::Tensor &logits, const torch::Tensor &targets,
                          const torch::Tensor &logit_lengths,
                          const torch::Tensor &target_lengths, int32_t blank) {
  // The denominator for the log-softmax.
  // Note that it is positive at present.
  torch::Tensor denominator = logits.logsumexp(/*dim*/ 1, /*keepdim*/ false);

  torch::Tensor log_probs;
  torch::Tensor row_splits;

  std::tie(log_probs, row_splits) = ComputeLogProbs(
      logits, denominator, targets, logit_lengths, target_lengths, blank);

  std::cout << "cuda log_probs: \n" << log_probs << "\n";

  torch::Tensor alpha =
      ComputeAlpha(log_probs, logit_lengths, target_lengths, row_splits);
  std::cout << "cuda alpha: \n" << alpha << "\n";

  torch::Tensor total_scores =
      torch::zeros({logit_lengths.size(0)}, logits.options());

  return {total_scores, torch::Tensor()};
}

}  // namespace ot

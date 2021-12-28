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

static void CheckCuda(cudaError_t result, const char *file, int32_t line) {
  if (result != cudaSuccess) {
    std::ostringstream os;
    os << file << ":" << line << ": " << cudaGetErrorString(result) << "\n";
    throw std::runtime_error(os.str());
  }
}
#define OT_CHECK_CUDA(ret) CheckCuda(ret, __FILE__, __LINE__)

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

static torch::Tensor ComputeLogProbs(
    const torch::Tensor &logits, const torch::Tensor &denominator,
    const torch::Tensor &targets, const torch::Tensor &logit_lengths,
    const torch::Tensor &target_lengths, const torch::Tensor &row_splits,
    const torch::Tensor &row_ids, int32_t blank) {
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

  auto ret = cudaGetLastError();
  OT_CHECK_CUDA(ret);

  return log_probs;
}

static std::pair<torch::Tensor, torch::Tensor> ComputeAlpha(
    const torch::Tensor &log_probs, const torch::Tensor &logit_lengths,
    const torch::Tensor &target_lengths, const torch::Tensor &row_splits) {
  int32_t max_T = logit_lengths.max().item<int32_t>();

  // it is prepended with a blank so we need to use +1 here
  int32_t max_U_p1 = target_lengths.max().item<int32_t>() + 1;

  int32_t batch_size = logit_lengths.size(0);

  int32_t num_warps = (max_T + kWarpSize - 1) / kWarpSize;
  dim3 block_dims(num_warps, max_U_p1, batch_size);
  dim3 thread_dims(kWarpSize);

  torch::Tensor alpha = torch::empty({log_probs.size(0)}, log_probs.options());
  torch::Tensor total_scores = torch::empty({batch_size}, log_probs.options());
  torch::Tensor counter =
      torch::zeros({batch_size * max_U_p1}, logit_lengths.options());

  const float *p_log_probs = log_probs.data_ptr<float>();
  const int32_t *p_logit_lengths = logit_lengths.data_ptr<int32_t>();
  const int32_t *p_target_lengths = target_lengths.data_ptr<int32_t>();
  const int32_t *p_row_splits = row_splits.data_ptr<int32_t>();
  int32_t *p_counter = counter.data_ptr<int32_t>();
  float *p_alpha = alpha.data_ptr<float>();
  float *p_total_socres = total_scores.data_ptr<float>();

  ComputeAlpha<<<block_dims, thread_dims>>>(
      p_log_probs, p_logit_lengths, p_target_lengths, p_row_splits, max_T,
      max_U_p1, p_counter, p_alpha, p_total_socres);

  auto ret = cudaGetLastError();
  OT_CHECK_CUDA(ret);

  return {alpha, total_scores};
}

static torch::Tensor ComputeBeta(const torch::Tensor &log_probs,
                                 const torch::Tensor &logit_lengths,
                                 const torch::Tensor &target_lengths,
                                 const torch::Tensor &row_splits) {
  int32_t max_T = logit_lengths.max().item<int32_t>();

  // it is prepended with a blank so we need to use +1 here
  int32_t max_U_p1 = target_lengths.max().item<int32_t>() + 1;

  int32_t batch_size = logit_lengths.size(0);

  int32_t num_warps = (max_T + kWarpSize - 1) / kWarpSize;
  dim3 block_dims(num_warps, max_U_p1, batch_size);
  dim3 thread_dims(kWarpSize);

  // torch::Tensor beta = torch::empty({log_probs.size(0)},
  // log_probs.options());
  torch::Tensor beta = torch::ones({log_probs.size(0)}, log_probs.options());
  torch::Tensor counter =
      torch::zeros({batch_size * max_U_p1}, logit_lengths.options());

  const float *p_log_probs = log_probs.data_ptr<float>();
  const int32_t *p_logit_lengths = logit_lengths.data_ptr<int32_t>();
  const int32_t *p_target_lengths = target_lengths.data_ptr<int32_t>();
  const int32_t *p_row_splits = row_splits.data_ptr<int32_t>();
  int32_t *p_counter = counter.data_ptr<int32_t>();
  float *p_beta = beta.data_ptr<float>();

  ComputeBeta<<<block_dims, thread_dims>>>(p_log_probs, p_logit_lengths,
                                           p_target_lengths, p_row_splits,
                                           max_T, max_U_p1, p_counter, p_beta);

  auto ret = cudaGetLastError();
  OT_CHECK_CUDA(ret);

  return beta;
}

static void ComputeGradient(
    const torch::Tensor &logits, const torch::Tensor &logit_lengths,
    const torch::Tensor &targets, const torch::Tensor &target_lengths,
    const torch::Tensor &denominator, const torch::Tensor &alpha,
    const torch::Tensor &beta, int32_t blank, const torch::Tensor &row_splits,
    const torch::Tensor &row_ids, torch::Tensor *gradient) {
  const float *p_logits = logits.data_ptr<float>();
  const int32_t *p_logit_lengths = logit_lengths.data_ptr<int32_t>();
  const int32_t *p_targets = targets.data_ptr<int32_t>();
  const int32_t *p_target_lengths = target_lengths.data_ptr<int32_t>();
  const float *p_den = denominator.data_ptr<float>();
  const float *p_alpha = alpha.data_ptr<float>();
  const float *p_beta = beta.data_ptr<float>();
  const int32_t *p_row_splits = row_splits.data_ptr<int32_t>();
  const int32_t *p_row_ids = row_ids.data_ptr<int32_t>();

  float *p_grad = gradient->data_ptr<float>();

  int32_t num_blocks =
      (logits.size(0) + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;

  ComputeGradient<<<num_blocks, kMaxThreadsPerBlock>>>(
      p_logits, p_den, p_targets, p_logit_lengths, p_target_lengths, blank,
      p_row_splits, p_row_ids, logits.size(0), logits.size(1), targets.size(1),
      p_alpha, p_beta, p_grad);

  auto ret = cudaGetLastError();
  OT_CHECK_CUDA(ret);
}

std::pair<torch::Tensor, torch::optional<torch::Tensor>>
ComputeTransducerLossCuda(torch::Tensor &logits, const torch::Tensor &targets,
                          const torch::Tensor &logit_lengths,
                          const torch::Tensor &target_lengths, int32_t blank) {
  // The denominator for the log-softmax.
  // Note that it is positive at present.
  torch::Tensor denominator = logits.logsumexp(/*dim*/ 1, /*keepdim*/ false);

  // + 1 here since each sequence is prepended with a blank
  torch::Tensor sizes = logit_lengths * (target_lengths + 1);
  torch::Tensor row_splits = torch::cumsum(sizes, -1, torch::kInt);
  torch::Tensor zero = torch::zeros({1}, row_splits.options());
  row_splits = torch::cat({zero, row_splits}, -1);
  torch::Tensor row_ids = RowSplitsToRowIds(row_splits, logits.size(0));

  torch::Tensor log_probs =
      ComputeLogProbs(logits, denominator, targets, logit_lengths,
                      target_lengths, row_splits, row_ids, blank);
  torch::Tensor alpha;
  torch::Tensor total_scores;
  std::tie(alpha, total_scores) =
      ComputeAlpha(log_probs, logit_lengths, target_lengths, row_splits);

  torch::Tensor beta =
      ComputeBeta(log_probs, logit_lengths, target_lengths, row_splits);

  bool requires_grad = logits.requires_grad();
  if (requires_grad) {
    torch::Tensor &gradient = logits;
    ComputeGradient(logits, logit_lengths, targets, target_lengths, denominator,
                    alpha, beta, blank, row_splits, row_ids, &gradient);
  }

  return {total_scores, requires_grad ? logits : torch::Tensor()};
}

}  // namespace ot

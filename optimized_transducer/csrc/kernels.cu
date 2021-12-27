// optimized_transducer/csrc/kernels.cu
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

#define kBlankCol 0
#define kSymCol 1

namespace ot {

__global__ void ComputeLogProbs(const float *logits, const float *denominator,
                                const int32_t *targets,
                                const int32_t *target_lengths, int32_t blank,
                                const int32_t *row_splits,
                                const int32_t *row_ids, int32_t sum_all_TU,
                                int32_t vocab_size, int32_t targets_col,
                                float *log_probs) {
  int32_t idx01 = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx01 >= sum_all_TU) return;  // out-of-boundary

  int32_t b = row_ids[idx01];  // batch size

  // +1 since it is prepended with a blank
  int32_t U_p1 = target_lengths[b] + 1;
  int32_t offset = row_splits[b];
  int32_t idx1 = idx01 - offset;

  int32_t u = idx1 % U_p1;

  const float *p_logits = logits + idx01 * vocab_size;
  const float *p_denominator = denominator + idx01;
  const int32_t *p_targets = targets + b * targets_col;

  float d = *p_denominator;

  float *p_log_probs = log_probs + idx01 * 2;
  p_log_probs[kBlankCol] = p_logits[blank] - d;
  if (u < U_p1 - 1) {
    p_log_probs[kSymCol] = p_logits[p_targets[u]] - d;
  }
}

}  // namespace ot

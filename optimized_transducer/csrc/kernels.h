// optimized_transducer/csrc/kernels.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
#ifndef OPTIMIZED_TRANSDUCER_CSRC_KERNELS_H_
#define OPTIMIZED_TRANSDUCER_CSRC_KERNELS_H_

namespace ot {

__global__ void ComputeLogProbs(const float *logits, const float *denominator,
                                const int32_t *targets,
                                const int32_t *target_lengths, int32_t blank,
                                const int32_t *row_splits,
                                const int32_t *row_ids, int32_t sum_all_TU,
                                int32_t vocab_size, int32_t targets_col,
                                float *log_probs);

__global__ void ComputeAlpha(const float *log_probs,
                             const int32_t *logit_lengths,
                             const int32_t *target_lengths,
                             const int32_t *row_splits, int32_t max_T,
                             int32_t max_U_p1, int32_t *counter, float *alpha,
                             float *total_scores);

__global__ void ComputeBeta(const float *log_probs,
                            const int32_t *logit_lengths,
                            const int32_t *target_lengths,
                            const int32_t *row_splits, int32_t max_T,
                            int32_t max_U_p1, int32_t *counter, float *alpha);

__global__ void ComputeGradient(
    const float *logits, const float *denominator, const int32_t *targets,
    const int32_t *logit_lengths, const int32_t *target_lengths, int32_t blank,
    const int32_t *row_splits, const int32_t *row_ids, int32_t sum_all_TU,
    int32_t vocab_size, int32_t targets_col, const float *alpha,
    const float *beta, float *gradient);

}  // namespace ot

#endif  // OPTIMIZED_TRANSDUCER_CSRC_KERNELS_H_

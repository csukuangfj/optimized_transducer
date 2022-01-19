// optimized_transducer/csrc/kernels.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
#ifndef OPTIMIZED_TRANSDUCER_CSRC_KERNELS_H_
#define OPTIMIZED_TRANSDUCER_CSRC_KERNELS_H_

namespace ot {

/**
  @param logits  Pointer to a 2-D tensor of shape (sum_all_TU, vocab_size)
                 from the output of some `nn.Linear` layer.
  @param denominator Pointer to a 1-D tensor of shape (sum_all_TU,)
  @param targets Pointer to a 2-D tensor of shape (batch_size, max_U), where
                 max_U is the maximum number of tokens in a utterance with
                 a batch, excluding blank.
  @param target_lengths  Pointer to a 1-D tensor of shape (batch_size,)
                         containing number of tokens in each utterance before
                         padding, excluding blank.
  @param blank  The ID of the blank symbol.
  @param row_splits  Pointer to a 1-D tensor of shape (batch_size+1,)
  @param row_ids Pointer to a 1-D tensor of shape (sum_all_TU,)
  @param sum_all_TU  It equals to \sum_i^{batch_size-1} T_i * U_i
  @param vocab_size  It is the number of symbols including the blank
  @param targets_col It is max_U.
  @param log_probs  Pointer to a 2-D tensor of shape (sum_all_TU, 2).
 */
__global__ void ComputeLogProbs(const float *logits, const float *denominator,
                                const int32_t *targets,
                                const int32_t *target_lengths, int32_t blank,
                                const int32_t *row_splits,
                                const int32_t *row_ids, int32_t sum_all_TU,
                                int32_t vocab_size, int32_t targets_col,
                                float *log_probs);

/**
  @param logits  Pointer to a 2-D tensor of shape (sum_all_TU, vocab_size)
                 from the output of log-softmax.
  @param targets Pointer to a 2-D tensor of shape (batch_size, max_U), where
                 max_U is the maximum number of tokens in a utterance with
                 a batch, excluding blank.
  @param target_lengths  Pointer to a 1-D tensor of shape (batch_size,)
                         containing number of tokens in each utterance before
                         padding, excluding blank.
  @param blank  The ID of the blank symbol.
  @param row_splits  Pointer to a 1-D tensor of shape (batch_size+1,)
  @param row_ids Pointer to a 1-D tensor of shape (sum_all_TU,)
  @param sum_all_TU  It equals to \sum_i^{batch_size-1} T_i * U_i
  @param vocab_size  It is the number of symbols including the blank
  @param targets_col It is max_U.
  @param log_probs  Pointer to a 2-D tensor of shape (sum_all_TU, 2).
 */
__global__ void ComputeLogProbsForLogSoftmax(
    const float *logits, const int32_t *targets, const int32_t *target_lengths,
    int32_t blank, const int32_t *row_splits, const int32_t *row_ids,
    int32_t sum_all_TU, int32_t vocab_size, int32_t targets_col,
    float *log_probs);

/**
  @param log_probs  Pointer to a 2-D tensor of shape (sum_all_TU, 2).
  @param logit_lengths Pointer to a 1-D tensor of shape (batch_size,) containing
                       the number of encoder output frames before padding.
  @param target_lengths  Pointer to a 1-D tensor of shape (batch_size,)
                         containing number of tokens in each utterance before
                         padding, excluding blank.
  @param row_splits  Pointer to a 1-D tensor of shape (batch_size+1,)
  @param max_T  The maximum number of encoder output frames of the utterance
                within the batch.
  @param max_U_p1  It is max_U + 1
  @param counter Pointer to a 2-D tensor of shape (batch_size, max_U_p1). The
                 tensor should be zero initialized.
  @param alpha  Pointer to a 1-D tensor of shape (sum_all_TU,). On return, it
                contains the computed alpha.
  @param total_scores Pointer to a 1-D tensor of shape (batch_size,). On
                      return, it contains the total scores for each utterance
                      in the batch.
 */
__global__ void ComputeAlpha(const float *log_probs,
                             const int32_t *logit_lengths,
                             const int32_t *target_lengths,
                             const int32_t *row_splits, int32_t max_T,
                             int32_t max_U_p1, int32_t *counter, float *alpha,
                             float *total_scores);
/**
  @param log_probs  Pointer to a 2-D tensor of shape (sum_all_TU, 2).
  @param logit_lengths Pointer to a 1-D tensor of shape (batch_size,) containing
                       the number of encoder output frames before padding.
  @param target_lengths  Pointer to a 1-D tensor of shape (batch_size,)
                         containing number of tokens in each utterance before
                         padding, excluding blank.
  @param row_splits  Pointer to a 1-D tensor of shape (batch_size+1,)
  @param max_T  The maximum number of encoder output frames of the utterance
                within the batch.
  @param max_U_p1  It is max_U + 1
  @param counter Pointer to a 2-D tensor of shape (batch_size, max_U_p1). The
                 tensor should be zero initialized.
  @param alpha  Pointer to a 1-D tensor of shape (sum_all_TU,). On return, it
                contains the computed alpha.
  @param total_scores Pointer to a 1-D tensor of shape (batch_size,). On
                      return, it contains the total scores for each utterance
                      in the batch.
 */
__global__ void ComputeAlphaOneSymPerFrame(
    const float *log_probs, const int32_t *logit_lengths,
    const int32_t *target_lengths, const int32_t *row_splits, int32_t max_T,
    int32_t max_U_p1, int32_t *counter, float *alpha, float *total_scores);

/**
  @param log_probs  Pointer to a 2-D tensor of shape (sum_all_TU, 2).
  @param logit_lengths Pointer to a 1-D tensor of shape (batch_size,) containing
                       the number of encoder output frames before padding.
  @param target_lengths  Pointer to a 1-D tensor of shape (batch_size,)
                         containing number of tokens in each utterance before
                         padding, excluding blank.
  @param row_splits  Pointer to a 1-D tensor of shape (batch_size+1,)
  @param max_T  The maximum number of encoder output frames of the utterance
                within the batch.
  @param max_U_p1  It is max_U + 1
  @param counter Pointer to a 2-D tensor of shape (batch_size, max_U_p1). The
                 tensor should be zero initialized.
  @param beta    Pointer to a 1-D tensor of shape (sum_all_TU,). On return, it
                 contains the computed beta.
 */
__global__ void ComputeBeta(const float *log_probs,
                            const int32_t *logit_lengths,
                            const int32_t *target_lengths,
                            const int32_t *row_splits, int32_t max_T,
                            int32_t max_U_p1, int32_t *counter, float *beta);
/**
  @param log_probs  Pointer to a 2-D tensor of shape (sum_all_TU, 2).
  @param logit_lengths Pointer to a 1-D tensor of shape (batch_size,) containing
                       the number of encoder output frames before padding.
  @param target_lengths  Pointer to a 1-D tensor of shape (batch_size,)
                         containing number of tokens in each utterance before
                         padding, excluding blank.
  @param row_splits  Pointer to a 1-D tensor of shape (batch_size+1,)
  @param max_T  The maximum number of encoder output frames of the utterance
                within the batch.
  @param max_U_p1  It is max_U + 1
  @param counter Pointer to a 2-D tensor of shape (batch_size, max_U_p1). The
                 tensor should be zero initialized.
  @param beta    Pointer to a 1-D tensor of shape (sum_all_TU,). On return, it
                 contains the computed beta.
 */
__global__ void ComputeBetaOneSymPerFrame(const float *log_probs,
                                          const int32_t *logit_lengths,
                                          const int32_t *target_lengths,
                                          const int32_t *row_splits,
                                          int32_t max_T, int32_t max_U_p1,
                                          int32_t *counter, float *beta);
/**
  @param logits  Pointer to a 2-D tensor of shape (sum_all_TU, vocab_size).
                 The tensor is the output of `nn.Linear`.
  @param denominator Pointer to a 1-D tensor of shape (sum_all_TU,)
  @param targets Pointer to a 2-D tensor of shape (batch_size, max_U), where
                 max_U is the maximum number of tokens in a utterance with
                 a batch, excluding blank.
  @param logit_lengths Pointer to a 1-D tensor of shape (batch_size,) containing
                       the number of encoder output frames before padding.
  @param target_lengths  Pointer to a 1-D tensor of shape (batch_size,)
                         containing number of tokens in each utterance before
                         padding, excluding blank.
  @param blank  The ID of the blank symbol.
  @param row_splits  Pointer to a 1-D tensor of shape (batch_size+1,)
  @param row_ids Pointer to a 1-D tensor of shape (sum_all_TU,)
  @param sum_all_TU  It equals to \sum_i^{batch_size-1} T_i * U_i
  @param vocab_size  It is the number of symbols including the blank
  @param targets_col It is max_U.
  @param alpha  Pointer to a 1-D tensor of shape (sum_all_TU,).
  @param beta  Pointer to a 1-D tensor of shape (sum_all_TU,).
  @param gradient  Pointer to a 2-D tensor of shape (sum_all_TU, vocab_size).
                   Note: It can be equal to `logits`.
 */
__global__ void ComputeGradient(
    const float *logits, const float *denominator, const int32_t *targets,
    const int32_t *logit_lengths, const int32_t *target_lengths, int32_t blank,
    const int32_t *row_splits, const int32_t *row_ids, int32_t sum_all_TU,
    int32_t vocab_size, int32_t targets_col, const float *alpha,
    const float *beta, float *gradient);

/**
  @param logits  Pointer to a 2-D tensor of shape (sum_all_TU, vocab_size).
                 The tensor is the output of log-softmax.
  @param targets Pointer to a 2-D tensor of shape (batch_size, max_U), where
                 max_U is the maximum number of tokens in a utterance with
                 a batch, excluding blank.
  @param logit_lengths Pointer to a 1-D tensor of shape (batch_size,) containing
                       the number of encoder output frames before padding.
  @param target_lengths  Pointer to a 1-D tensor of shape (batch_size,)
                         containing number of tokens in each utterance before
                         padding, excluding blank.
  @param blank  The ID of the blank symbol.
  @param row_splits  Pointer to a 1-D tensor of shape (batch_size+1,)
  @param row_ids Pointer to a 1-D tensor of shape (sum_all_TU,)
  @param sum_all_TU  It equals to \sum_i^{batch_size-1} T_i * U_i
  @param vocab_size  It is the number of symbols including the blank
  @param targets_col It is max_U.
  @param alpha  Pointer to a 1-D tensor of shape (sum_all_TU,).
  @param beta  Pointer to a 1-D tensor of shape (sum_all_TU,).
  @param gradient  Pointer to a 2-D tensor of shape (sum_all_TU, vocab_size).
                   Note: It can be equal to `logits`.
 */
__global__ void ComputeGradientForLogSoftmax(
    const float *logits, const int32_t *targets, const int32_t *logit_lengths,
    const int32_t *target_lengths, int32_t blank, const int32_t *row_splits,
    const int32_t *row_ids, int32_t sum_all_TU, int32_t vocab_size,
    int32_t targets_col, const float *alpha, const float *beta,
    float *gradient);

/**
  @param logits  Pointer to a 2-D tensor of shape (sum_all_TU, vocab_size).
                 The tensor is the output of log-softmax.
  @param targets Pointer to a 2-D tensor of shape (batch_size, max_U), where
                 max_U is the maximum number of tokens in a utterance with
                 a batch, excluding blank.
  @param logit_lengths Pointer to a 1-D tensor of shape (batch_size,) containing
                       the number of encoder output frames before padding.
  @param target_lengths  Pointer to a 1-D tensor of shape (batch_size,)
                         containing number of tokens in each utterance before
                         padding, excluding blank.
  @param blank  The ID of the blank symbol.
  @param row_splits  Pointer to a 1-D tensor of shape (batch_size+1,)
  @param row_ids Pointer to a 1-D tensor of shape (sum_all_TU,)
  @param sum_all_TU  It equals to \sum_i^{batch_size-1} T_i * U_i
  @param vocab_size  It is the number of symbols including the blank
  @param targets_col It is max_U.
  @param alpha  Pointer to a 1-D tensor of shape (sum_all_TU,).
  @param beta  Pointer to a 1-D tensor of shape (sum_all_TU,).
  @param gradient  Pointer to a 2-D tensor of shape (sum_all_TU, vocab_size).
                   Note: It can be equal to `logits`.
 */
__global__ void ComputeGradientForLogSoftmaxOneSymPerFrame(
    const float *logits, const int32_t *targets, const int32_t *logit_lengths,
    const int32_t *target_lengths, int32_t blank, const int32_t *row_splits,
    const int32_t *row_ids, int32_t sum_all_TU, int32_t vocab_size,
    int32_t targets_col, const float *alpha, const float *beta,
    float *gradient);

}  // namespace ot

#endif  // OPTIMIZED_TRANSDUCER_CSRC_KERNELS_H_

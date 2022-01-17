// optimized_transducer/csrc/cpu.cc
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

#include "torch/script.h"

namespace ot {

// Column indexes for the log_prob tensor.
// Column 0 is the log-prob for blanks
// Column 1 is the log-prob for symbols
static constexpr int32_t kBlankCol = 0;
static constexpr int32_t kSymCol = 1;

static constexpr float kMinLogDiff =
    -15.9423847198486328125f;  // logf(FLT_EPSILON)

static float LogSumExp(float a, float b) {
  // copied from k2/csrc/utils.h
  float diff;

  if (a < b) {
    diff = a - b;
    a = b;
  } else {
    diff = b - a;
  }
  // diff is negative.  a is now the larger one.

  if (diff >= kMinLogDiff) {
    return a + log1pf(expf(diff));
  }

  return a;  // return the larger one.
}

/* Compute the emit and non-emit log probabilities for each (t, u) in the grid.

   Return a tensor of shape (logits.size(0), 2). Column 0 contains the log-prob
   for non-emit, i.e., horizontal transition from t to t+1. Column 1 contains
   the log-prob for emit, i.e., vertical transition from u to u+1.

   @param logits  Output of some `nn.Linear` layer with shape
                  (sum_all_TU, vocab_size).
   @param denominator  A 1-D tensor of shape (sum_all_TU,)
   @param targets  A 2-D tensor of shape (batch_size, max_U). It
                   is NOT prepended with a blank.
   @param logit_lengths A 1-D tensor of shape (batch_size,)
   @param target_lengths A 1-D tensor of shape (batch_size,)
   @param blank The ID of the blank symbol.
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

  torch::Tensor log_probs = torch::empty({logits.size(0), 2}, logits.options());

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

/* Compute the emit and non-emit log probabilities for each (t, u) in the grid.

   Return a tensor of shape (logits.size(0), 2). Column 0 contains the log-prob
   for non-emit, i.e., horizontal transition from t to t+1. Column 1 contains
   the log-prob for emit, i.e., vertical transition from u to u+1.

   @param logits  Output of some `log-softmax` layer with shape
                      (sum_all_TU, vocab_size).
   @param targets  A 2-D tensor of shape (batch_size, max_U). It
                   is NOT prepended with a blank.
   @param logit_lengths A 1-D tensor of shape (batch_size,)
   @param target_lengths A 1-D tensor of shape (batch_size,)
   @param blank The ID of the blank symbol.
 */
static torch::Tensor ComputeLogProbsForLogSoftmax(
    const torch::Tensor &logits, const torch::Tensor &targets,
    const torch::Tensor &logit_lengths, const torch::Tensor &target_lengths,
    int32_t blank) {
  torch::ArrayRef<int32_t> logit_len_arr(logit_lengths.data_ptr<int32_t>(),
                                         logit_lengths.numel());

  torch::ArrayRef<int32_t> target_len_arr(target_lengths.data_ptr<int32_t>(),
                                          target_lengths.numel());
  int32_t batch_size = targets.size(0);

  torch::Tensor log_probs = torch::empty({logits.size(0), 2}, logits.options());

  const float *p_logits = logits.data_ptr<float>();
  float *p_log_probs = log_probs.data_ptr<float>();
  const int32_t *p_targets = targets.data_ptr<int32_t>();

  int32_t V = logits.size(1);  // vocabulary size including blank

  for (int32_t b = 0; b != batch_size; ++b, p_targets += targets.size(1)) {
    int32_t T = logit_len_arr[b];

    // p1 means plus one
    // We need to plus one since it is prepended with a blank
    int32_t U_p1 = target_len_arr[b] + 1;
    for (int32_t t = 0; t != T; ++t) {
      for (int32_t u = 0; u != U_p1; ++u, p_log_probs += 2, p_logits += V) {
        // for blank
        p_log_probs[kBlankCol] = p_logits[blank];

        // for non-blank
        if (u < U_p1 - 1) {
          p_log_probs[kSymCol] = p_logits[p_targets[u]];
        }
      }
    }
  }

  return log_probs;
}

/**
   Return two tensors:
    - alpha, a tensor of shape (log_probs.size(0),)
    - total_scores, a tensor of shape (logit_lengths.size(0),)
 */
static std::pair<torch::Tensor, torch::Tensor> ComputeAlpha(
    const torch::Tensor &log_probs, const torch::Tensor &logit_lengths,
    const torch::Tensor &target_lengths) {
  int32_t batch_size = logit_lengths.size(0);
  torch::Tensor alpha = torch::empty({log_probs.size(0)}, log_probs.options());
  torch::Tensor total_scores = torch::empty({batch_size}, log_probs.options());

  torch::ArrayRef<int32_t> logit_len_arr(logit_lengths.data_ptr<int32_t>(),
                                         logit_lengths.numel());

  torch::ArrayRef<int32_t> target_len_arr(target_lengths.data_ptr<int32_t>(),
                                          target_lengths.numel());

  const float *p_log_probs = log_probs.data_ptr<float>();
  float *p_alpha = alpha.data_ptr<float>();

  float *p_total_scores = total_scores.data_ptr<float>();

  for (int32_t b = 0; b != batch_size; ++b) {
    int32_t T = logit_len_arr[b];

    // p1 means plus one
    // We need to plus one since it is prepended with a blank
    int32_t U_p1 = target_len_arr[b] + 1;

    p_alpha[0] = 0;
    float *p_alpha_tm1 = p_alpha;  // tm1 means t minus 1
    float *p_alpha_t = p_alpha_tm1 + U_p1;

    const float *p_log_probs_tm1 = p_log_probs;

    // when u = 0, alpha(t, 0) = alpha(t-1, 0) + log_probs(t-1, 0).blank
    for (int32_t t = 1; t != T; ++t) {
      p_alpha_t[0] = p_alpha_tm1[0] + p_log_probs_tm1[kBlankCol];

      p_alpha_tm1 = p_alpha_t;
      p_alpha_t += U_p1;
      p_log_probs_tm1 += U_p1 * 2;
    }

    const float *p_log_probs_um1 = p_log_probs;
    // when t = 0, alpha(0, u) = alpha(0, u-1) + log_probs(0, u-1).symbol
    for (int32_t u = 1; u != U_p1; ++u, p_log_probs_um1 += 2) {
      p_alpha[u] = p_alpha[u - 1] + p_log_probs_um1[kSymCol];
    }

    p_alpha_tm1 = p_alpha;
    p_alpha_t = p_alpha + U_p1;

    p_log_probs_tm1 = p_log_probs;
    const float *p_log_probs_t = p_log_probs + 2 * U_p1;

    for (int32_t t = 1; t != T; ++t) {
      for (int32_t u = 1; u != U_p1; ++u) {
        // Note: p_log_probs_tm1 points to the start of the row on entry, but
        // we want to use p_log_probs_tm1[u] here, so +2 is used.
        //
        // alpha(t, u) = log_sum_exp(alpha(t-1,u) + log_probs(t-1, u).blank,
        //                           alpha(t, u-1) + log_probs(t, u-1).symbol)
        p_alpha_t[u] =
            LogSumExp(p_alpha_tm1[u] + (p_log_probs_tm1 + 2)[kBlankCol],
                      p_alpha_t[u - 1] + p_log_probs_t[kSymCol]);
        p_log_probs_tm1 += 2;
        p_log_probs_t += 2;
      }
      p_alpha_tm1 = p_alpha_t;
      p_alpha_t += U_p1;

      // Note: p_log_probs_tm1 and p_log_probs_t are incremented only for
      // U_p1 - 1 steps, so we need to move it one step forward here
      p_log_probs_tm1 += 2;
      p_log_probs_t += 2;
    }

    p_alpha += T * U_p1;
    p_log_probs += T * U_p1 * 2;

    // total_scores =  alpha(T-1, U-1) + log_probs(T-1, U-1).blank
    p_total_scores[b] = p_alpha[-1] + (p_log_probs - 2)[kBlankCol];
  }

  return {alpha, total_scores};
}

static std::pair<torch::Tensor, torch::Tensor> ComputeAlphaOneSymPerFrame(
    const torch::Tensor &log_probs, const torch::Tensor &logit_lengths,
    const torch::Tensor &target_lengths) {
  int32_t batch_size = logit_lengths.size(0);
  // torch::Tensor alpha = torch::empty({log_probs.size(0)},
  // log_probs.options());
  torch::Tensor alpha =
      torch::ones({log_probs.size(0)}, log_probs.options()) * 10;
  torch::Tensor total_scores = torch::empty({batch_size}, log_probs.options());

  torch::ArrayRef<int32_t> logit_len_arr(logit_lengths.data_ptr<int32_t>(),
                                         logit_lengths.numel());

  torch::ArrayRef<int32_t> target_len_arr(target_lengths.data_ptr<int32_t>(),
                                          target_lengths.numel());

  const float *p_log_probs = log_probs.data_ptr<float>();
  float *p_alpha = alpha.data_ptr<float>();

  float *p_total_scores = total_scores.data_ptr<float>();

  for (int32_t b = 0; b != batch_size; ++b) {
    int32_t T = logit_len_arr[b];

    // p1 means plus one
    // We need to plus one since it is prepended with a blank
    int32_t U_p1 = target_len_arr[b] + 1;

    // alpha(0, 0) = 0
    p_alpha[0] = 0;

    float *p_alpha_tm1 = p_alpha;  // tm1 means t minus 1
    float *p_alpha_t = p_alpha_tm1 + U_p1;

    const float *p_log_probs_tm1 = p_log_probs;

    int32_t diff = T - 1 - (U_p1 - 1);
    // when u = 0, alpha(t, 0) = alpha(t-1, 0) + log_probs(t-1, 0).blank
    for (int32_t t = 1; t <= diff; ++t) {
      p_alpha_t[0] = p_alpha_tm1[0] + p_log_probs_tm1[kBlankCol];

      p_alpha_tm1 = p_alpha_t;
      p_alpha_t += U_p1;
      p_log_probs_tm1 += U_p1 * 2;
    }

    for (int32_t t = 1; t != T; ++t) {
      p_alpha_tm1 = p_alpha + (t - 1) * U_p1;
      p_alpha_t = p_alpha + t * U_p1;
      p_log_probs_tm1 = p_log_probs + (t - 1) * U_p1 * 2;

      for (int32_t u = 1; u != U_p1; ++u) {
        if (u > t || t - u > diff) {
          continue;
        } else if (t == u) {
          // alpha(t, u) = alpha(t-1, u-1) + log_probs(t-1, u-1)
          p_alpha_t[u] =
              p_alpha_tm1[u - 1] + (p_log_probs_tm1 + (u - 1) * 2)[kBlankCol];
        } else {
          // alpha(t, u) = log_sum_exp(alpha(t-1, u) + log_probs(t-1, u).blank,
          //                      alpha(t-1, u-1) + log_probs(t-1, u-1).symbol)
          p_alpha_t[u] = LogSumExp(
              p_alpha_tm1[u] + (p_log_probs_tm1 + u * 2)[kBlankCol],
              p_alpha_tm1[u - 1] + (p_log_probs_tm1 + (u - 1) * 2)[kSymCol]);
        }
      }
    }

    p_alpha += T * U_p1;
    p_log_probs += T * U_p1 * 2;
    // total_scores =  alpha(T-1, U-1) + log_probs(T-1, U-1).blank
    p_total_scores[b] = p_alpha[-1] + (p_log_probs - 2)[kBlankCol];
  }
  return {alpha, total_scores};
}

static std::pair<torch::Tensor, torch::Tensor> ComputeBeta(
    const torch::Tensor &log_probs, const torch::Tensor &logit_lengths,
    const torch::Tensor &target_lengths) {
  int32_t batch_size = logit_lengths.size(0);
  torch::Tensor beta = torch::empty({log_probs.size(0)}, log_probs.options());
  torch::Tensor total_scores = torch::empty({batch_size}, log_probs.options());

  torch::ArrayRef<int32_t> logit_len_arr(logit_lengths.data_ptr<int32_t>(),
                                         logit_lengths.numel());

  torch::ArrayRef<int32_t> target_len_arr(target_lengths.data_ptr<int32_t>(),
                                          target_lengths.numel());

  const float *p_log_probs = log_probs.data_ptr<float>();
  float *p_beta = beta.data_ptr<float>();
  float *p_total_scores = total_scores.data_ptr<float>();
  for (int32_t b = 0; b != batch_size; ++b) {
    int32_t T = logit_len_arr[b];

    // p1 means plus one
    // We need to plus one since it is prepended with a blank
    int32_t U_p1 = target_len_arr[b] + 1;
    float *p_beta_t = p_beta + T * U_p1;
    const float *p_log_probs_t = p_log_probs + T * U_p1 * 2;
    p_beta_t[-1] = (p_log_probs_t - 2)[kBlankCol];

    // when u = U_p1 - 1,
    // beta(t, U_p1-1) = beta(t+1, U_p1-1) + lop_probs(t, U_p1-1).blank
    for (int32_t t = T - 2; t >= 0; --t) {
      p_beta_t = p_beta + (t + 1) * U_p1 - 1;
      float *p_beta_t_p1 = p_beta_t + U_p1;
      p_log_probs_t = p_log_probs + (t + 1) * U_p1 * 2 - 2;

      *p_beta_t = *p_beta_t_p1 + p_log_probs_t[kBlankCol];
    }
    // when t = T - 1,
    // beta(T-1 u) =  beta(T-1, u+1) + log_probs(T-1, u).symbol
    float *p_beta_u_p1 = p_beta + T * U_p1 - 1;
    float *p_beta_u = p_beta_u_p1 - 1;
    const float *p_log_probs_u = p_log_probs + T * U_p1 * 2 - 4;

    for (int32_t u = U_p1 - 2; u >= 0;
         --u, --p_beta_u, --p_beta_u_p1, p_log_probs_u -= 2) {
      *p_beta_u = *p_beta_u_p1 + p_log_probs_u[kSymCol];
    }

    for (int32_t t = T - 2; t >= 0; --t) {
      p_beta_t = p_beta + t * U_p1;
      float *p_beta_t_p1 = p_beta_t + U_p1;
      p_log_probs_t = p_log_probs + (t + 1) * U_p1 * 2 - 4;
      for (int32_t u = U_p1 - 2; u >= 0; --u, p_log_probs_t -= 2) {
        // beta(t, u) = log_sum_exp(beta(t+1,u) + log_probs(t, u).blank,
        //                           beta(t, u+1) + log_probs(t, u).symbol)
        p_beta_t[u] = LogSumExp(p_beta_t_p1[u] + p_log_probs_t[kBlankCol],
                                p_beta_t[u + 1] + p_log_probs_t[kSymCol]);
      }
    }
    // total_scores =  beta(0, 0)
    p_total_scores[b] = p_beta[0];
    p_beta += T * U_p1;
    p_log_probs += T * U_p1 * 2;
  }
  return {beta, total_scores};
}

/**
   @param logits The output of `nn.Linear` with shape (sum_all_TU, vocab_size)
   @param logit_lengths A 1-D tensor of shape (batch_size,)
   @param targets  A 2-D tensor of shape (batch_size, max_U). It
                   is NOT prepended with a blank.
   @param target_lengths A 1-D tensor of shape (batch_size,)
   @param denominator A 1-D tensor of shape (sum_all_TU,)
   @param alpha  A 1-D tensor of shape (sum_all_TU,)
   @param beta  A 1-D tensor of shape (sum_all_TU,)
   @param blank The ID of the blank symbol.
   @param gradient A 2-D tensor of shape (sum_all_TU, vocab_size).
                   Note: It may share the same memory with `logits`.

   Caution: This function assumes `logits` is the output of `nn.Linear`.
 */
static void ComputeGradient(
    const torch::Tensor &logits, const torch::Tensor &logit_lengths,
    const torch::Tensor &targets, const torch::Tensor &target_lengths,
    const torch::Tensor &denominator, const torch::Tensor &alpha,
    const torch::Tensor &beta, int32_t blank, torch::Tensor *gradient) {
  // see
  // https://github.com/pytorch/audio/blob/main/torchaudio/csrc/rnnt/cpu/cpu_kernels.h#L317
  // for the formula to compute the gradient.

  torch::ArrayRef<int32_t> logit_len_arr(logit_lengths.data_ptr<int32_t>(),
                                         logit_lengths.numel());

  torch::ArrayRef<int32_t> target_len_arr(target_lengths.data_ptr<int32_t>(),
                                          target_lengths.numel());

  const float *p_logits = logits.data_ptr<float>();
  const float *p_alpha = alpha.data_ptr<float>();
  const float *p_beta = beta.data_ptr<float>();
  const float *p_den = denominator.data_ptr<float>();
  float *p_grad = gradient->data_ptr<float>();

  const int32_t *p_targets = targets.data_ptr<int32_t>();

  int32_t V = logits.size(1);  // vocabulary size including blank

  int32_t batch_size = logit_lengths.size(0);
  for (int32_t b = 0; b != batch_size; ++b, p_targets += targets.size(1)) {
    int32_t T = logit_len_arr[b];

    // p1 means plus one
    // We need to plus one since it is prepended with a blank
    int32_t U_p1 = target_len_arr[b] + 1;

    float loss = -p_beta[0];

    for (int32_t t = 0; t != T;
         ++t, p_alpha += U_p1, p_beta += U_p1, p_den += U_p1) {
      const float *p_beta_t_p1 = p_beta + U_p1;

      for (int32_t u = 0; u != U_p1; ++u, p_logits += V, p_grad += V) {
        int32_t target_u =
            (u < U_p1 - 1) ? p_targets[u] : -1;  // -1 is not used
        float c = p_alpha[u] + loss - p_den[u];

        for (int32_t v = 0; v != V; ++v) {
          float g = p_logits[v] + c;

          if (v == blank && t == T - 1 && u == U_p1 - 1) {
            // the last blank transition
            p_grad[v] = std::exp(g + p_beta[u]) - std::exp(g);
          } else if (v == blank && t < T - 1) {
            p_grad[v] = std::exp(g + p_beta[u]) - std::exp(g + p_beta_t_p1[u]);
          } else if (u < U_p1 - 1 && v == target_u) {
            p_grad[v] = std::exp(g + p_beta[u]) - std::exp(g + p_beta[u + 1]);
          } else {
            p_grad[v] = std::exp(g + p_beta[u]);
          }
        }
      }
    }
  }
}

/**
   @param logits The output of log-softmax with shape (sum_all_TU, vocab_size)
   @param logit_lengths A 1-D tensor of shape (batch_size,)
   @param targets  A 2-D tensor of shape (batch_size, max_U). It
                   is NOT prepended with a blank.
   @param target_lengths A 1-D tensor of shape (batch_size,)
   @param alpha  A 1-D tensor of shape (sum_all_TU,)
   @param beta  A 1-D tensor of shape (sum_all_TU,)
   @param blank The ID of the blank symbol.
   @param gradient A 2-D tensor of shape (sum_all_TU, vocab_size)

  Caution: This function assumes `logits` is the output of log-softmax.
 */
static void ComputeGradientForLogSoftmax(
    const torch::Tensor &logits, const torch::Tensor &logit_lengths,
    const torch::Tensor &targets, const torch::Tensor &target_lengths,
    const torch::Tensor &alpha, const torch::Tensor &beta, int32_t blank,
    torch::Tensor *gradient) {
  torch::ArrayRef<int32_t> logit_len_arr(logit_lengths.data_ptr<int32_t>(),
                                         logit_lengths.numel());

  torch::ArrayRef<int32_t> target_len_arr(target_lengths.data_ptr<int32_t>(),
                                          target_lengths.numel());

  const float *p_logits = logits.data_ptr<float>();
  const float *p_alpha = alpha.data_ptr<float>();
  const float *p_beta = beta.data_ptr<float>();
  float *p_grad = gradient->data_ptr<float>();
  const int32_t *p_targets = targets.data_ptr<int32_t>();
  int32_t V = logits.size(1);  // vocabulary size including blank
  int32_t batch_size = logit_lengths.size(0);
  for (int32_t b = 0; b != batch_size; ++b, p_targets += targets.size(1)) {
    int32_t T = logit_len_arr[b];

    // p1 means plus one
    // We need to plus one since it is prepended with a blank
    int32_t U_p1 = target_len_arr[b] + 1;

    float loss = -p_beta[0];
    for (int32_t t = 0; t != T; ++t, p_alpha += U_p1, p_beta += U_p1) {
      const float *p_beta_t_p1 = p_beta + U_p1;

      for (int32_t u = 0; u != U_p1; ++u, p_logits += V, p_grad += V) {
        int32_t target_u =
            (u < U_p1 - 1) ? p_targets[u] : -1;  // -1 is not used
        float c = p_alpha[u] + loss;

        for (int32_t v = 0; v != V; ++v) {
          float g = p_logits[v] + c;
          if (v == blank && t == T - 1 && u == U_p1 - 1) {
            // the last blank transition
            p_grad[v] = -std::exp(g);
          } else if (v == blank && t < T - 1) {
            p_grad[v] = -std::exp(g + p_beta_t_p1[u]);
          } else if (u < U_p1 - 1 && v == target_u) {
            p_grad[v] = -std::exp(g + p_beta[u + 1]);
          } else {
            p_grad[v] = 0;
          }
        }
      }
    }
  }
}

// See the documentation in transducer-loss.h for the meaning of the arguments.
std::pair<torch::Tensor, torch::optional<torch::Tensor>>
ComputeTransducerLossCpu(torch::Tensor &logits,  // NOLINT
                         const torch::Tensor &targets,
                         const torch::Tensor &logit_lengths,
                         const torch::Tensor &target_lengths, int32_t blank,
                         bool from_log_softmax, bool one_sym_per_frame) {
  torch::Tensor denominator;  // The denominator for the log-softmax.
                              // Used only when from_log_softmax is False

  torch::Tensor log_probs;
  if (from_log_softmax) {
    // logits is the output of `log-softmax`.
    log_probs = ComputeLogProbsForLogSoftmax(logits, targets, logit_lengths,
                                             target_lengths, blank);
  } else {
    // logits is the output of `nn.Linear`.
    denominator = logits.logsumexp(/*dim*/ 1, /*keepdim*/ false);
    log_probs = ComputeLogProbs(logits, denominator, targets, logit_lengths,
                                target_lengths, blank);
  }

  torch::Tensor alpha;
  torch::Tensor total_scores;
  if (one_sym_per_frame) {
    std::tie(alpha, total_scores) =
        ComputeAlphaOneSymPerFrame(log_probs, logit_lengths, target_lengths);
  } else {
    std::tie(alpha, total_scores) =
        ComputeAlpha(log_probs, logit_lengths, target_lengths);
  }
  return {alpha, total_scores};

  torch::Tensor beta =
      ComputeBeta(log_probs, logit_lengths, target_lengths).first;

  bool requires_grad = logits.requires_grad();
  torch::Tensor gradient;
  if (requires_grad) {
    if (from_log_softmax) {
      gradient = torch::empty_like(logits);
      ComputeGradientForLogSoftmax(logits, logit_lengths, targets,
                                   target_lengths, alpha, beta, blank,
                                   &gradient);
    } else {
      gradient = logits;  // shallow copy
      ComputeGradient(logits, logit_lengths, targets, target_lengths,
                      denominator, alpha, beta, blank, &gradient);
    }
  }

  return {total_scores, gradient};
}

}  // namespace ot

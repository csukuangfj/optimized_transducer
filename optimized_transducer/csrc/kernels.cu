// optimized_transducer/csrc/kernels.cu
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

#define kBlankCol 0
#define kSymCol 1

namespace ot {

// copied from
// https://github.com/danpovey/fast_rnnt/blob/master/torch_mutual_information/mutual_information_cuda_kernel.cu
// returns log(exp(x) + exp(y)).
__forceinline__ __device__ float LogAdd(float x, float y) {
  float diff;
  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.
  if (diff - diff != 0)
    return x;  // x and y are probably -inf.  Return the larger one.
  else
    return x + log1p(exp(diff));
}

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

// This function uses
// https://github.com/pytorch/audio/blob/main/torchaudio/csrc/rnnt/gpu/gpu_kernels.cuh#L61
// as a reference
__global__ void ComputeAlpha(const float *log_probs,
                             const int32_t *logit_lengths,
                             const int32_t *target_lengths,
                             const int32_t *row_splits, int32_t max_T,
                             int32_t max_U_p1, int32_t *counter, float *alpha,
                             float *total_scores) {
  int32_t b = blockIdx.z;
  int32_t T = logit_lengths[b];
  int32_t U_p1 = target_lengths[b] + 1;

  int32_t t = blockDim.x * blockIdx.x + threadIdx.x + 1;
  int32_t u = blockIdx.y + 1;

  if (t >= T || u >= U_p1) return;  // out-of-boundary

  int32_t offset = row_splits[b];

  int32_t *p_counter = counter + b * max_U_p1 + blockIdx.y;
  float *p_alpha = alpha + offset;
  float *p_alpha_t = p_alpha + t * U_p1;

  const float *p_log_probs = log_probs + offset * 2;

  const float *p_log_probs_t = p_log_probs + t * U_p1 * 2;

  // m1 means minus 1
  const float *p_log_probs_t_m1 = p_log_probs + (t - 1) * U_p1 * 2;

  if (t == 1 && u == 1) {
    // alpha(0, 0) = 0
    p_alpha[0] = 0;
  }
  if (blockIdx.x > 0) {
    // wait for previous warp to finish (t-axis)
    while (atomicAdd(p_counter, 0) < blockIdx.x) {
      // busy waiting
    }
  }

  if (blockIdx.y > 0) {
    // wait for previous warp to finish (u-axis)
    while (atomicAdd(p_counter - 1, 0) <= blockIdx.x) {
      // busy waiting
    }
  }

  if (t == 1) {
    // alpha(0, u) = alpha(0, u-1) + log_probs(0, u-1).symbol
    p_alpha[u] = p_alpha[u - 1] + (p_log_probs + (u - 1) * 2)[kSymCol];
  }

  if (u == 1) {
    float skip_prob = p_log_probs_t_m1[kBlankCol];
    float val;

#pragma unroll
    for (int32_t i = 1; i < warpSize; i <<= 1) {
      val = __shfl_up_sync(0xffffffff, skip_prob, i);
      if (i <= threadIdx.x) {
        skip_prob = skip_prob + val;
      }
    }

    val = *(p_alpha + blockIdx.x * blockDim.x * U_p1);
    p_alpha_t[0] = skip_prob + val;
  }

  // log_probs(t-1, u).blank
  float skip_prob = (p_log_probs_t_m1 + 2 * u)[kBlankCol];

  // log_probs(t, u-1).symbol
  float emit_prob = (p_log_probs_t + 2 * (u - 1))[kSymCol];

  float skip = *(p_alpha + blockIdx.x * blockDim.x * U_p1 + u) + skip_prob;
  float emit = *(p_alpha_t + u - 1) + emit_prob;
  float val = LogAdd(skip, emit);
  float out = val;

#pragma unroll
  for (int32_t i = 1; i < warpSize; ++i) {
    val = __shfl_up_sync(0xffffffff, val, 1);
    if (i == threadIdx.x) {
      val = LogAdd(val + skip_prob, emit);
      out = val;
    }
  }
  *(p_alpha_t + u) = out;

  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(p_counter, 1);
  }

  if ((t == T - 1) && (u == U_p1 - 1)) {
    total_scores[b] =
        p_alpha_t[U_p1 - 1] + (p_log_probs_t + (U_p1 - 1) * 2)[kBlankCol];
  }
}

// This function uses
// https://github.com/pytorch/audio/blob/main/torchaudio/csrc/rnnt/gpu/gpu_kernels.cuh#L159
// as a reference
__global__ void ComputeBeta(const float *log_probs,
                            const int32_t *logit_lengths,
                            const int32_t *target_lengths,
                            const int32_t *row_splits, int32_t max_T,
                            int32_t max_U_p1, int32_t *counter, float *beta) {
  int32_t b = blockIdx.z;
  int32_t T = logit_lengths[b];
  int32_t U_p1 = target_lengths[b] + 1;

  const int t = T - 2 - blockDim.x * blockIdx.x - threadIdx.x;
  const int u = U_p1 - 2 - blockIdx.y;

  if (t < 0 || u < 0) return;  // out-of-boundary

  int32_t offset = row_splits[b];
  int32_t *p_counter = counter + b * max_U_p1 + blockIdx.y;

  float *p_beta = beta + offset;
  float *p_beta_t = p_beta + t * U_p1;
  float *p_beta_t_p1 = p_beta + (t + 1) * U_p1;

  const float *p_log_probs = log_probs + offset * 2;
  const float *p_log_probs_t = p_log_probs + t * U_p1 * 2;
  const float *p_log_probs_t_p1 = p_log_probs + (t + 1) * U_p1 * 2;

  if (t == T - 2 && u == U_p1 - 2) {
    // beta(T-1, U_p1-1) = log_probs(T-1, U_p1-1).blank
    p_beta_t_p1[U_p1 - 1] = (p_log_probs_t_p1 + (U_p1 - 1) * 2)[kBlankCol];
  }

  if (blockIdx.x > 0) {
    // wait for previous warp to finish (t-axis)
    while (atomicAdd(p_counter, 0) < blockIdx.x) {
      // busy waiting
    }
  }

  if (blockIdx.y > 0) {
    // wait for previous warp to finish (u-axis)
    while (atomicAdd(p_counter - 1, 0) <= blockIdx.x) {
      // busy waiting
    }
  }

  if (t == T - 2) {
    // beta(T-1, u) = beta(T-1, u+1) + log_probs(T-1, u).symbol
    p_beta_t_p1[u] = p_beta_t_p1[u + 1] + (p_log_probs_t_p1 + u * 2)[kSymCol];
  }

  if (u == U_p1 - 2) {
    float skip_prob = (p_log_probs_t + (U_p1 - 1) * 2)[kBlankCol];
    float val;

#pragma unroll
    for (int i = 1; i < warpSize; i <<= 1) {
      val = __shfl_up_sync(0xffffffff, skip_prob, i);
      if (i <= threadIdx.x) {
        skip_prob = skip_prob + val;
      }
    }

    p_beta_t[U_p1 - 1] =
        (p_beta + (T - 1 - blockDim.x * blockIdx.x) * U_p1)[U_p1 - 1] +
        skip_prob;
  }

  float skip_prob = (p_log_probs_t + u * 2)[kBlankCol];
  float emit_prob = (p_log_probs_t + u * 2)[kSymCol];

  float skip = (p_beta + (t + threadIdx.x + 1) * U_p1)[u] + skip_prob;
  float emit = p_beta_t[u + 1] + emit_prob;

  float val = LogAdd(skip, emit);
  float out = val;

#pragma unroll
  for (int i = 1; i < warpSize; ++i) {
    val = __shfl_up_sync(0xffffffff, val, 1);
    if (i == threadIdx.x) {
      val = LogAdd(val + skip_prob, emit);
      out = val;
    }
  }

  p_beta_t[u] = out;

  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(p_counter, 1);
  }
}

__global__ void ComputeGradient(
    const float *logits, const float *denominator, const int32_t *targets,
    const int32_t *logit_lengths, const int32_t *target_lengths, int32_t blank,
    const int32_t *row_splits, const int32_t *row_ids, int32_t sum_all_TU,
    int32_t vocab_size, int32_t targets_col, const float *alpha,
    const float *beta, float *gradient) {
  int32_t idx01 = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx01 >= sum_all_TU) return;  // out-of-boundary

  int32_t b = row_ids[idx01];  // batch size

  // +1 since it is prepended with a blank
  int32_t U_p1 = target_lengths[b] + 1;
  int32_t T = logit_lengths[b];
  int32_t offset = row_splits[b];

  int32_t idx1 = idx01 - offset;
  int32_t t = idx1 / U_p1;
  int32_t u = idx1 % U_p1;

  const float *p_logits_t_u = logits + idx01 * vocab_size;
  const float *p_denominator = denominator + offset;
  const float *p_denominator_t = p_denominator + t * U_p1;
  const int32_t *p_targets = targets + b * targets_col;

  const float *p_alpha = alpha + offset;
  const float *p_alpha_t = p_alpha + t * U_p1;

  const float *p_beta = beta + offset;
  const float *p_beta_t = p_beta + t * U_p1;
  const float *p_beta_t_p1 = p_beta + (t + 1) * U_p1;

  float *p_grad_t_u = gradient + idx01 * vocab_size;

  float loss = -1 * p_beta[0];

  if (isinf(loss) || isnan(loss)) {
    for (int32_t v = 0; v != vocab_size; ++v) {
      p_grad_t_u[v] = 0;
    }
    return;
  }

  float c = p_alpha_t[u] + loss - p_denominator_t[u];

  int32_t target_u = (u < U_p1 - 1) ? p_targets[u] : -1;  // -1 is not used

  // TODO(fangjun): Use separate threads to compute the gradient
  // so that we don't have a `for` loop here
  for (int32_t v = 0; v != vocab_size; ++v) {
    float g = p_logits_t_u[v] + c;
    if (v == blank && t == T - 1 && u == U_p1 - 1) {
      // last blank transition
      p_grad_t_u[v] = expf(g + p_beta_t[u]) - expf(g);
    } else if (v == blank && t < T - 1) {
      p_grad_t_u[v] = expf(g + p_beta_t[u]) - expf(g + p_beta_t_p1[u]);
    } else if (u < U_p1 - 1 && v == target_u) {
      p_grad_t_u[v] = expf(g + p_beta_t[u]) - expf(g + p_beta_t[u + 1]);
    } else {
      p_grad_t_u[v] = expf(g + p_beta_t[u]);
    }
  }
}

}  // namespace ot

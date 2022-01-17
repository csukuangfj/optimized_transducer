// optimized_transducer/csrc/transducer-loss.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
#ifndef OPTIMIZED_TRANSDUCER_CSRC_TRANSDUCER_LOSS_H_
#define OPTIMIZED_TRANSDUCER_CSRC_TRANSDUCER_LOSS_H_

#include "torch/script.h"

namespace ot {

/** Compute the transducer loss given the output from the joint network.

  @param logits  A 2-D tensor of shape (TU, V), where TU equals to
                 \sum_i (T_i * U_i), where `T_i` is the logit_lengths[i] and
                 `U_i` is `target_lengths[i]` + 1. V is the vocabulary size
                 including the blank.
  @param targets A 2-D tensor of shape (N, U), where N is the batch size.
  @param logit_lengths A 1-D tensor of shape (N, ) containing number of output
                       acoustic frames from the encoder.
  @param target_lengths A 1-D tensor of shape (N, ) containing the input
                        sequence length of each utterance for the decoder.
  @param from_log_softmax If true, `logits` is the output of `log-softmax`;
                          If false, `logits` is the output of `nn.Linear`.
  @param one_sym_per_frame If true, it limits the maximum number of symbols per
                           frame to 1. This means, it always advances the time
                           step whether it emits a symbol or a blank.
                           If false, it is the standard transducer, i.e., there
                           can be multiple symbols per frame.

  Note:
    If one_sym_per_frame is false, we have the following formula:

      alpha(t, u) = log_sum_exp(alpha(t-1, u) + log_prob(t-1, u).blank,
                                alpha(t-1, u-1) + log_prob(t-1, u-1).symbol);
             (t-1, u) ---> (t, u)
                             ^
                             |
                             |
                             |
                          (t, u-1)

    If one_sym_per_frame is true, we have the following formula:

      alpha(t, u) = log_sum_exp(alpha(t-1, u) + log_prob(t-1, u).blank,
                                alpha(t, u-1) + log_prob(t, u-1).symbol);

             (t-1, u) ---> (t, u)
                          _
                          /|
                        /
                      /
                    /
             (t-1, u-1)

  @return Return a pair containing
          - the loss, which is a 1-D tensor of shape (N,)
          - the gradient, it is empty if logits does not require grad.
            Otherwise, its shape is the same as `logits`.
 */
std::pair<torch::Tensor, torch::optional<torch::Tensor>> ComputeTransducerLoss(
    torch::Tensor &logits, const torch::Tensor &targets,
    const torch::Tensor &logit_lengths, const torch::Tensor &target_lengths,
    int32_t blank, bool from_log_softmax, bool one_sym_per_frame);

}  // namespace ot

#endif  // OPTIMIZED_TRANSDUCER_CSRC_TRANSDUCER_LOSS_H_

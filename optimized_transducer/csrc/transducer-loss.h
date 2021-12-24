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
  @param targets A 1-D tensor of shape (N, ), where N is the batch size.
  @param logit_lengths A 1-D tensor of shape (N, ) containing number of output
                       acoustic frames from the encoder.
  @param target_lengths A 1-D tensor of shape (N, ) containing the input
                        sequence length of each utterance for the decoder.
  @param clamp If positive, limit the gradient to the range [-clamp, clamp].
  @return Return a pair containing
          - the loss, which is a 1-D tensor of shape (N,)
          - the gradient, it is empty if logits does not require grad.
            Otherwise, its shape is the same as `logits`.
 */
std::pair<torch::Tensor, torch::optional<torch::Tensor>> ComputeTransducerLoss(
    torch::Tensor &logits, const torch::Tensor &targets,
    const torch::Tensor &logit_lengths, const torch::Tensor &target_lengths,
    int32_t blank, double clamp = 0);

}  // namespace ot

#endif  // OPTIMIZED_TRANSDUCER_CSRC_TRANSDUCER_LOSS_H_

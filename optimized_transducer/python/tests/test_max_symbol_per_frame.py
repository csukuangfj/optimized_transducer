#!/usr/bin/env python3

# Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)

import torch

import optimized_transducer


def test_alpha(from_log_softmax: bool = False):

    T1 = 6
    T2 = 4

    U1 = 3
    U2 = 3

    V = 3

    logits = torch.rand(2, max(T1, T2), max(U1, U2), V, dtype=torch.float32)
    targets = torch.randint(
        low=1, high=V - 1, size=(2, max(U1, U2) - 1), dtype=torch.int32
    )
    logit_lengths = torch.tensor([T1, T2], dtype=torch.int32)
    target_lengths = torch.tensor([U1, U2], dtype=torch.int32) - 1

    logits0 = logits[0, :T1, :U1, :].reshape(-1, V).requires_grad_(True)
    logits1 = logits[1, :T2, :U2, :].reshape(-1, V).requires_grad_(True)

    logits = torch.cat([logits0, logits1])

    if from_log_softmax:
        logits = logits.log_softmax(dim=-1)

    loss, total_scores = optimized_transducer.transducer_loss(
        logits=logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=0,
        from_log_softmax=from_log_softmax,
        one_sym_per_frame=True,
    )
    print(loss[: T1 * U1].reshape(T1, U1))
    print(loss[T1 * U1 :].reshape(T2, U2))
    print(loss)
    print(total_scores)


def main():
    #  for from_log_softmax in [True, False]:
    for from_log_softmax in [True]:
        test_alpha(from_log_softmax)


if __name__ == "__main__":
    torch.manual_seed(20220117)
    main()

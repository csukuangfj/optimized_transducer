#!/usr/bin/env python3

# Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)

import optimized_transducer
import torch


def test_compute_transducer_loss():
    T1 = 2
    T2 = 3

    U1 = 4
    U2 = 3

    V = 3

    sum_TU = T1 * U1 + T2 * U2

    device = torch.device("cpu")

    logits = torch.rand(sum_TU, V, device=device)

    targets = torch.randint(
        low=1, high=V, size=(2, max(T1, T2)), dtype=torch.int32, device=device
    )

    logit_lengths = torch.tensor([T1, T2], dtype=torch.int32, device=device)

    target_lengths = torch.tensor([U1, U2], dtype=torch.int32, device=device) - 1

    optimized_transducer.compute_transducer_loss(
        logits=logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=0,
    )
    print(targets)
    print(logits.log_softmax(-1))


def main():
    test_compute_transducer_loss()


if __name__ == "__main__":
    torch.manual_seed(20211224)
    main()

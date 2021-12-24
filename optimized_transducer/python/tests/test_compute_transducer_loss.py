#!/usr/bin/env python3

# Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)

import optimized_transducer
import torch
import torchaudio


def test_compute_transducer_loss():
    T1 = 2
    T2 = 3

    U1 = 3
    U2 = 5

    V = 3

    sum_TU = T1 * U1 + T2 * U2

    device = torch.device("cpu")

    torch_logits = torch.rand(2, max(T1, T2), max(U1, U2), V, device=device)

    logits0 = torch_logits[0, :T1, :U1, :].reshape(-1, V)

    logits1 = torch_logits[1, :T2, :U2, :].reshape(-1, V)
    logits = torch.cat((logits0, logits1))

    targets = torch.randint(
        low=1, high=V, size=(2, max(U1, U2) - 1), dtype=torch.int32, device=device
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
    print(
        torch.ops.torchaudio.rnnt_loss_alphas(
            logits=torch_logits,
            targets=targets,
            logit_lengths=logit_lengths,
            target_lengths=target_lengths,
            blank=0,
            clamp=0,
        )
    )

    print(
        torch.ops.torchaudio.rnnt_loss(
            logits=torch_logits,
            targets=targets,
            logit_lengths=logit_lengths,
            target_lengths=target_lengths,
            blank=0,
            clamp=0,
        )
    )


def main():
    test_compute_transducer_loss()


if __name__ == "__main__":
    torch.manual_seed(20211224)
    main()

#!/usr/bin/env python3

# Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)

import torch
import torchaudio

import optimized_transducer


def test_compute_transducer_loss(reduction: str):
    T1 = torch.randint(1, 100, (1,)).item()
    T2 = torch.randint(1, 100, (1,)).item()

    U1 = torch.randint(5, 100, (1,)).item()
    U2 = torch.randint(5, 100, (1,)).item()

    V = torch.randint(5, 100, (1,)).item()

    device = torch.device("cpu")

    torch_logits = torch.rand(
        2, max(T1, T2), max(U1, U2), V, device=device
    ).requires_grad_(True)

    logits0 = (
        torch_logits[0, :T1, :U1, :]
        .reshape(-1, V)
        .detach()
        .clone()
        .requires_grad_(True)
    )

    logits1 = (
        torch_logits[1, :T2, :U2, :]
        .reshape(-1, V)
        .detach()
        .clone()
        .requires_grad_(True)
    )

    logits = torch.cat((logits0, logits1))

    targets = torch.randint(
        low=1,
        high=V,
        size=(2, max(U1, U2) - 1),
        dtype=torch.int32,
        device=device,
    )

    logit_lengths = torch.tensor([T1, T2], dtype=torch.int32, device=device)

    target_lengths = (
        torch.tensor([U1, U2], dtype=torch.int32, device=device) - 1
    )

    loss = optimized_transducer.transducer_loss(
        logits=logits.requires_grad_(True),
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=0,
        reduction=reduction,
    )

    torch_loss = torchaudio.functional.rnnt_loss(
        logits=torch_logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=0,
        clamp=0,
        reduction=reduction,
    )

    assert torch.allclose(loss, torch_loss)
    (loss.sum() * 3).backward()
    (torch_loss.sum() * 3).backward()

    assert torch.allclose(
        logits0.grad.reshape(T1, U1, -1),
        torch_logits.grad[0, :T1, :U1, :],
        rtol=1e-4,
    )

    assert torch.allclose(
        logits1.grad.reshape(T2, U2, -1),
        torch_logits.grad[1, :T2, :U2, :],
        rtol=1e-4,
    )


def main():
    for reduction in ["mean", "sum"]:
        test_compute_transducer_loss(reduction)


if __name__ == "__main__":
    torch.manual_seed(20211224)
    main()

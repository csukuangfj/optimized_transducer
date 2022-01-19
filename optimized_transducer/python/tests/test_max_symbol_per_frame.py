#!/usr/bin/env python3

# Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)

import torch

import optimized_transducer


def assert_allclose(a: torch.Tensor, b: torch.Tensor, atol=1e-6, **kwargs):
    assert torch.allclose(a, b, atol=atol, **kwargs), f"{(a - b).abs().max()}, {a}, {b}"


def test_one_symbol_per_frame():

    T1 = 100
    T2 = 200

    U1 = 50
    U2 = 60

    V = 1000

    logits = torch.rand(2, max(T1, T2), max(U1, U2), V, dtype=torch.float32)
    targets = torch.randint(
        low=1, high=V - 1, size=(2, max(U1, U2) - 1), dtype=torch.int32
    )
    logit_lengths = torch.tensor([T1, T2], dtype=torch.int32)
    target_lengths = torch.tensor([U1, U2], dtype=torch.int32) - 1

    logits0 = logits[0, :T1, :U1, :].reshape(-1, V).requires_grad_(True)
    logits1 = logits[1, :T2, :U2, :].reshape(-1, V).requires_grad_(True)

    logits0_clone = logits0.detach().clone().requires_grad_(True)
    logits1_clone = logits1.detach().clone().requires_grad_(True)

    logits = torch.cat([logits0, logits1])
    logits_clone = torch.cat([logits0_clone, logits1_clone])

    loss = optimized_transducer.transducer_loss(
        logits=logits.log_softmax(dim=-1),
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=0,
        from_log_softmax=True,
        one_sym_per_frame=True,
    )

    loss_clone = optimized_transducer.transducer_loss(
        logits=logits_clone,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=0,
        from_log_softmax=False,
        one_sym_per_frame=True,
    )

    loss.backward()
    loss_clone.backward()

    assert_allclose(loss, loss_clone)
    assert_allclose(logits0.grad, logits0_clone.grad, atol=1e-4)
    assert_allclose(logits1.grad, logits1_clone.grad, atol=1e-4)


def main():
    test_one_symbol_per_frame()


if __name__ == "__main__":
    torch.manual_seed(20220117)
    main()

#!/usr/bin/env python3

# Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)

import torch

import optimized_transducer


def test_loss(from_log_softmax: bool = False):
    if not torch.cuda.is_available():
        print("cuda is not available - skipping")
        return

    T1 = 3
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

    loss = optimized_transducer.transducer_loss(
        logits=logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=0,
        from_log_softmax=from_log_softmax,
    )
    loss.backward()
    print(loss)
    # now for cuda
    device = torch.device("cuda", 0)
    ot_logits0 = logits0.detach().clone().to(device).requires_grad_(True)
    ot_logits1 = logits1.detach().clone().to(device).requires_grad_(True)
    ot_logits = torch.cat([ot_logits0, ot_logits1])

    targets = targets.to(device)
    logit_lengths = logit_lengths.to(device)
    target_lengths = target_lengths.to(device)

    loss_cuda = optimized_transducer.transducer_loss(
        logits=ot_logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=0,
        from_log_softmax=False,
    )
    loss_cuda.backward()

    assert torch.allclose(loss, loss_cuda.cpu())
    assert torch.allclose(logits0.grad, ot_logits0.grad.cpu())
    assert torch.allclose(logits1.grad, ot_logits1.grad.cpu())


def main():
    for from_log_softmax in [True, False]:
        test_loss(from_log_softmax)


if __name__ == "__main__":
    torch.manual_seed(20211227)
    main()

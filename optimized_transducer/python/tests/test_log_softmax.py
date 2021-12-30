#!/usr/bin/env python3

# Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)

import torch

import optimized_transducer


def assert_allclose(a: torch.Tensor, b: torch.Tensor, atol=1e-6, **kwargs):
    assert torch.allclose(a, b, atol=atol, **kwargs), f"{(a - b).abs().max()}, {a}, {b}"


def get_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda", 0))
        if torch.cuda.device_count() > 1:
            devices.append(torch.device("cuda", 1))
            torch.cuda.set_device(devices[-1])
    return devices


def test(reduction, device="cpu"):
    torch.manual_seed(20211230)
    T1 = 3
    T2 = 2

    U1 = 2
    U2 = 3

    V = 3

    logits = torch.rand(2, max(T1, T2), max(U1, U2), V, dtype=torch.float32).to(device)
    targets = torch.randint(
        low=1,
        high=V - 1,
        size=(2, max(U1, U2) - 1),
        dtype=torch.int32,
    ).to(device)
    logit_lengths = torch.tensor([T1, T2], dtype=torch.int32).to(device)
    target_lengths = torch.tensor([U1, U2], dtype=torch.int32).to(device) - 1

    logits0 = logits[0, :T1, :U1, :].reshape(-1, V).requires_grad_(True)
    logits1 = logits[1, :T2, :U2, :].reshape(-1, V).requires_grad_(True)

    logits0_clone = logits0.detach().clone().requires_grad_(True)
    logits1_clone = logits1.detach().clone().requires_grad_(True)

    logits = torch.cat([logits0, logits1])
    logits_clone = torch.cat(
        [logits0_clone.log_softmax(-1), logits1_clone.log_softmax(-1)]
    )

    loss = optimized_transducer.transducer_loss(
        logits=logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=0,
        from_log_softmax=False,
        reduction=reduction,
    )

    loss_clone = optimized_transducer.transducer_loss(
        logits=logits_clone,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=0,
        from_log_softmax=True,
        reduction=reduction,
    )
    assert_allclose(loss, loss_clone.cpu(), atol=1e-3)

    loss.backward()
    loss_clone.backward()

    assert_allclose(logits0.grad, logits0_clone.grad, atol=1e-3)
    assert_allclose(logits1.grad, logits1_clone.grad, atol=1e-3)

    return loss, logits0.grad, logits1.grad


def main():
    devices = get_devices()
    print("devices", devices)
    for reduction in ["mean", "sum"]:
        ans = []
        for device in devices:
            loss, logits0_grad, logits1_grad = test(reduction, device=device)
            ans.append((loss.cpu(), logits0_grad.cpu(), logits1_grad.cpu()))

        for loss, logits0_grad, logits1_grad in ans[1:]:
            assert_allclose(ans[0][0], loss)
            assert_allclose(ans[0][1], logits0_grad)
            assert_allclose(ans[0][2], logits1_grad)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)

import _optimized_transducer
import torch

import optimized_transducer


def test_row_splits():
    size = torch.tensor([1, 5, 0, 2])
    row_splits = torch.cumsum(size, dim=-1)
    row_splits = torch.cat([torch.tensor([0]), row_splits])
    device = torch.device("cuda", 0)
    row_splits = row_splits.to(torch.int32).to(device)
    row_ids = _optimized_transducer.row_splits_to_row_ids(row_splits)
    print(row_ids)
    row_ids2 = _optimized_transducer.row_splits_to_row_ids(
        row_splits,
        row_splits[-1],
    )
    print(row_ids2)


def test_loss():
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

    logits0 = logits[0, :T1, :U1, :].reshape(-1, V)
    logits1 = logits[1, :T2, :U2, :].reshape(-1, V)

    ot_logits = torch.cat([logits0, logits1])
    print(ot_logits.shape)
    loss = optimized_transducer.transducer_loss(
        logits=ot_logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=0,
    )
    print(loss)
    # now for cuda
    device = torch.device("cuda", 0)
    ot_logits = ot_logits.to(device)
    targets = targets.to(device)
    logit_lengths = logit_lengths.to(device)
    target_lengths = target_lengths.to(device)

    loss_cuda = optimized_transducer.transducer_loss(
        logits=ot_logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=0,
    )
    print(loss_cuda)


def main():
    #  test_row_splits()
    test_loss()


if __name__ == "__main__":
    torch.manual_seed(20211227)
    main()

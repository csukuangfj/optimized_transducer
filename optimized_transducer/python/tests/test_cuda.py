#!/usr/bin/env python3

# Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)

import _optimized_transducer
import torch


def main():
    size = torch.tensor([1, 5, 0, 2])
    row_splits = torch.cumsum(size, dim=-1)
    row_splits = torch.cat([torch.tensor([0]), row_splits])
    device = torch.device("cuda", 0)
    row_splits = row_splits.to(torch.int32).to(device)
    row_ids = _optimized_transducer.row_splits_to_row_ids(row_splits)
    print(row_ids)
    row_ids2 = _optimized_transducer.row_splits_to_row_ids(
        row_splits, row_splits[-1]
    )
    print(row_ids2)


if __name__ == "__main__":
    main()

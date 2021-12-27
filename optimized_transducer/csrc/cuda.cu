// optimized_transducer/csrc/cuda.cu
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

#include "moderngpu/kernel_load_balance.hxx"
#include "optimized_transducer/csrc/moderngpu-allocator.h"
#include "torch/script.h"

namespace ot {

// See https://github.com/k2-fsa/k2/blob/master/k2/csrc/utils.cu#L75
// for the meaning of row splits and row IDs.
/**

  @param row_splits  A 1-D tensor of dtype torch.int32. Its first
                     element should be zero.
  @param num_elems   If -1, it is equal to row_splits[-1].
                     If not -1, it must be equal to row_splits[-1].

  @return Return a 1-D tensor of dtype torch.int32. Its lengths
          equals to num_elems.
 */
torch::Tensor RowSplitsToRowIds(const torch::Tensor &row_splits,
                                int32_t num_elems = -1) {
  torch::CheckedFrom c = "RowSplitsToRowIds";
  auto row_splits_arg = torch::TensorArg(row_splits, "row_splits", 0);
  torch::checkScalarType(c, row_splits_arg, torch::kInt32);
  torch::checkDim(c, row_splits_arg, 1);
  torch::checkContiguous(c, row_splits_arg);

  int32_t num_rows = row_splits.size(0) - 1;
  const int32_t *p_row_splits = row_splits.data_ptr<int32_t>();
  if (num_elems == -1) {
    num_elems = row_splits.cpu().data_ptr<int32_t>()[num_rows];
  }

  torch::Tensor row_ids = torch::empty({num_elems}, row_splits.options());
  ModernGpuAllocator allocator;
  mgpu::load_balance_search(num_elems, p_row_splits, num_rows,
                            row_ids.data_ptr<int32_t>(), allocator);
  return row_ids;
}

}  // namespace ot

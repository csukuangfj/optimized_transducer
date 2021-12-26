// optimized_transducer/python/csrc/optimized_transducer.cc
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

#include "optimized_transducer/python/csrc/optimized_transducer.h"

#include "optimized_transducer/csrc/transducer-loss.h"
#include "torch/extension.h"

namespace ot {

PYBIND11_MODULE(_optimized_transducer, m) {
  m.doc() = "Python wrapper for Optimized Transducer";

  m.def("compute_transducer_loss", &ComputeTransducerLoss, py::arg("logits"),
        py::arg("targets"), py::arg("logit_lengths"), py::arg("target_lengths"),
        py::arg("blank"));

#if 1
  // for test only, will remove this block
  torch::Tensor RowSplitsToRowIds(const torch::Tensor &row_splits,
                                  int32_t num_elems /*= -1*/);
  m.def("row_splits_to_row_ids", &RowSplitsToRowIds, py::arg("row_splits"),
        py::arg("num_elems") = -1);
#endif
}

}  // namespace ot

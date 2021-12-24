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
        py::arg("blank"), py::arg("clamp") = 0);
}

}  // namespace ot

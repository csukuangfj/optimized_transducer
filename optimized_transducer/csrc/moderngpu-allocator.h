// optimized_transducer/csrc/moderngpu-allocator.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
#ifndef OPTIMIZED_TRANSDUCER_CSRC_MODERNGPU_ALLOCATOR_H_
#define OPTIMIZED_TRANSDUCER_CSRC_MODERNGPU_ALLOCATOR_H_

#ifndef OT_WITH_CUDA
#error "Please use cmake -DOT_WITH_CUDA=ON while building optimized_transducer"
#endif

#include "c10/cuda/CUDACachingAllocator.h"
#include "c10/cuda/CUDAFunctions.h"
#include "moderngpu/context.hxx"
#include "torch/script.h"

namespace ot {

class ModernGpuAllocator : public mgpu::standard_context_t {
 public:
  ModernGpuAllocator()
      : mgpu::standard_context_t(false, c10::cuda::getCurrentCUDAStream()) {
    allocator_ = c10::cuda::CUDACachingAllocator::get();
  }

  void *alloc(size_t size, mgpu::memory_space_t space) override {
    TORCH_CHECK(space == mgpu::memory_space_device, "Support GPU only");

    void *p = allocator_->raw_allocate(size);
    return p;
  }

  void free(void *p, mgpu::memory_space_t space) override {
    TORCH_CHECK(space == mgpu::memory_space_device, "Support GPU only");
    allocator_->raw_deallocate(p);
  }

 private:
  torch::Allocator *allocator_;  // NOT owned here
};

}  // namespace ot

#endif  // OPTIMIZED_TRANSDUCER_CSRC_MODERNGPU_ALLOCATOR_H_

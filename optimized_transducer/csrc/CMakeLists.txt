

include_directories(${CMAKE_SOURCE_DIR})

set(srcs
  transducer-loss.cc
  cpu.cc
)

add_library(optimized_transducer_core ${srcs})
target_link_libraries(optimized_transducer_core PUBLIC ${TORCH_LIBRARIES})

if(OT_WITH_CUDA)
  set(cuda_srcs cuda.cu)
  add_library(optimized_transducer_core_cuda ${cuda_srcs})
  target_compile_options(optimized_transducer_core_cuda PRIVATE -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w) # for moderngpu
  target_link_libraries(optimized_transducer_core_cuda PUBLIC moderngpu ${TORCH_LIBRARIES})
  target_link_libraries(optimized_transducer_core PUBLIC optimized_transducer_core_cuda)
endif()

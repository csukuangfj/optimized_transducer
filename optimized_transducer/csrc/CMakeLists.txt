

include_directories(${CMAKE_SOURCE_DIR})

set(srcs
  transducer-loss.cc
  cpu.cc
)
add_library(optimized_transducer_core ${srcs})
target_link_libraries(optimized_transducer_core PUBLIC ${TORCH_LIBRARIES})

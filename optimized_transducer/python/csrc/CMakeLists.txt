include_directories(${CMAKE_SOURCE_DIR})

pybind11_add_module(_optimized_transducer
  optimized_transducer.cc
)
target_link_libraries(_optimized_transducer PRIVATE optimized_transducer_core)
if(UNIX AND NOT APPLE)
  target_link_libraries(_optimized_transducer PUBLIC ${PYTHON_LIBRARY})
endif()


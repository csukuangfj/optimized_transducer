include_directories(${CMAKE_SOURCE_DIR})

pybind11_add_module(_optimized_transducer
  optimized_transducer.cc
)
target_link_libraries(_optimized_transducer PRIVATE optimized_transducer_core )

if(UNIX AND NOT APPLE)
  target_link_libraries(_optimized_transducer
    PRIVATE
      ${PYTHON_LIBRARY}
      ${TORCH_DIR}/lib/libtorch_python.so
  )
endif()

if(APPLE)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  target_link_libraries(_optimized_transducer PRIVATE "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}")
endif()

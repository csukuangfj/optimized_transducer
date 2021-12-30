function(optimized_transducer_add_py_test source)
  get_filename_component(name ${source} NAME_WE)
  set(name "${name}_py")

  add_test(NAME ${name}
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
  )

  get_filename_component(optimized_transducer_path ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

  set_property(TEST ${name}
    PROPERTY ENVIRONMENT "PYTHONPATH=${optimized_transducer_path}:$<TARGET_FILE_DIR:_optimized_transducer>:$ENV{PYTHONPATH}"
  )
endfunction()

# please sort the files in alphabetic order
set(py_test_files
  test_compute_transducer_loss.py
  test_cuda.py
  test_log_softmax.py
)

foreach(source IN LISTS py_test_files)
  optimized_transducer_add_py_test(${source})
endforeach()

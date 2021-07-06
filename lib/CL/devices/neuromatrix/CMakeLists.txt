set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}")
include_directories(${LLVM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../../llvmopencl)

add_pocl_device_library("pocl-devices-neuromatrix" neuromatrix.c neuromatrix.h)
target_link_libraries(pocl-devices-neuromatrix PUBLIC mc12101load)
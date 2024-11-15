cmake_minimum_required(VERSION 3.10)
project(hdf5_recoder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HDF5 REQUIRED COMPONENTS CXX HL)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

include_directories(
  ${HDF5_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  "/usr/include/eigen3"
  "include"
)

add_library(hdf5_recoder
  src/hdf5_reader.cc
  src/hdf5_recorder.cc
)
target_link_libraries(hdf5_recoder
  ${HDF5_CXX_LIBRARIES}
  ${HDF5_HL_LIBRARIES}
  ${OpenCV_LIBS}
  fmt::fmt
)

add_executable(01_read_dataset test/01_read_dataset.cc )
target_link_libraries(01_read_dataset
  hdf5_recoder
)

add_executable(02_create_hdf5 test/02_create_hdf5.cc )
target_link_libraries(02_create_hdf5
  hdf5_recoder
)

add_executable(03_image_debug test/03_image_debug.cc)
target_link_libraries(03_image_debug
  hdf5_recoder
)

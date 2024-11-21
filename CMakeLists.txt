cmake_minimum_required(VERSION 3.10)
project(manipulation_recorder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs std_msgs cv_bridge image_transport
)


find_package(HDF5 REQUIRED COMPONENTS CXX HL)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  "/usr/include/eigen3"
  "include"
)

add_library(hdf5_recorder
  src/hdf5_reader.cc
  src/hdf5_recorder.cc
)
target_link_libraries(hdf5_recorder
  ${HDF5_CXX_LIBRARIES}
  ${HDF5_HL_LIBRARIES}
  ${OpenCV_LIBS}
  fmt::fmt
)

add_executable(01_read_dataset test/01_read_dataset.cc )
target_link_libraries(01_read_dataset
  hdf5_recorder
)

add_executable(02_create_hdf5 test/02_create_hdf5.cc )
target_link_libraries(02_create_hdf5
  hdf5_recorder
)

add_executable(03_image_debug test/03_image_debug.cc)
target_link_libraries(03_image_debug
  hdf5_recorder
)

add_executable(ros_recorder src/ros_recorder.cc)
target_link_libraries(ros_recorder
  hdf5_recorder
  ${catkin_LIBRARIES}
)

# To publish joint states of ARIBOT dual-arm robot
add_executable(joint_state_publisher src/joint_state_publisher.cc)
target_link_libraries(joint_state_publisher
  hdf5_recorder
  ${catkin_LIBRARIES}
  # add here
)

add_executable(image_pub_test src/image_pub_test.cc)
target_link_libraries(image_pub_test
  hdf5_recorder
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(receive_test src/receive_test.cc)
target_link_libraries(receive_test
  ${catkin_LIBRARIES}
)

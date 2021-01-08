#
# Copyright (c) 2019, New York University and Max Planck Gesellschaft.
#
# License BSD-3 clause
#

#
# set up the project
#
cmake_minimum_required(VERSION 3.10.2)

project(mim_estimation)

# specify the C++ 17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#
# Dependencies
#
find_package(pinocchio 2.0.0 REQUIRED)
find_package(Eigen3 REQUIRED)

# local depends
find_package(yaml_utils REQUIRED)
find_package(mpi_cmake_modules REQUIRED)

#
# Build main library and binaries
#

# variable to store list of targets
set(all_targets)

#
# Main library
#

# define the source files
set(${PROJECT_NAME}_SRC_FILES
    src/ekf_vicon_imu.cpp #
    src/end_effector_force_estimator.cpp #
    src/io_tools/data_collector.cpp #
    src/io_tools/data_reader.cpp #
    src/io_tools/non_rt_data_collector.cpp)
# create the library
add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SRC_FILES})
# link the dependencies
target_link_libraries(${PROJECT_NAME} pinocchio::pinocchio)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} yaml_utils::yaml_utils)
# Includes. Add the include dependencies
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)
# export the library
list(APPEND all_targets ${PROJECT_NAME})

#
# Tests.
#
include(CTest)
if(BUILD_TESTING)
  # C++ unit-tests framework used.
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)
  # List of all C++ tests.
  set(unit_tests test_butterworth_filter test_ekf_vicon_imu
                 test_extended_kalman_filter test_end_effector_force_estimator)
  # Config folder
  get_filename_component(CONFIG_PATH config ABSOLUTE)
  # Create all C++ tests.
  foreach(unit_test ${unit_tests})
    # create the executable
    add_executable(${unit_test} tests/main.cpp tests/${unit_test}.cpp)
    # link to the created librairies and its dependencies
    target_link_libraries(${unit_test} ${PROJECT_NAME} GTest::gtest)
    # create 2 macro: the path to the config file and the path to the data file
    target_compile_definitions(
      ${unit_test}
      PUBLIC
        CONFIG_PATH="${CONFIG_PATH}"
        TEST_PARAMETER_YAML_FILE_PATH="${CONFIG_PATH}/test_vicon_base_state_estimator.yaml"
        TEST_DATA_FILE_PATH="${CONFIG_PATH}/test_vicon_base_state_estimator.dat"
    )
    # declare the test as gtest
    gtest_add_tests(TARGET ${unit_test})
  endforeach(unit_test ${unit_tests})
endif()

#
# Demo
#

# estimator based on IMU and Vicon
add_executable(demo_ekf_vicon_imu demos/demo_ekf_vicon_imu.cpp)
target_link_libraries(demo_ekf_vicon_imu ${PROJECT_NAME})
# export the executable
list(APPEND all_targets demo_ekf_vicon_imu)

#
# building documentation
#
add_documentation()

#
# Install and export.
#

# command to install the library and binaries
install(
  TARGETS ${all_targets}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

# install the python package too
get_python_install_dir(python_install_dir)
install(
  DIRECTORY python/${PROJECT_NAME}
  DESTINATION "${python_install_dir}"
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE)

# Export this package as a cmake package.
generate_cmake_package()
#
# Copyright (c) 2019, New York University and Max Planck Gesellschaft.
#
# License BSD-3 clause
#

#
# set up the project
#
cmake_minimum_required(VERSION 3.10.2)

set(PROJECT_VERSION 0.1.0)
project(mim_estimation)

# specify the C++ 17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#
# Dependencies
#

# Local depends always first to find the proper python version.
find_package(mpi_cmake_modules REQUIRED)
find_package(yaml_utils REQUIRED)

# External dependencies
find_package(eigenpy REQUIRED)
find_package(pinocchio 2.0.0 REQUIRED)
find_package(Eigen3 REQUIRED)

# Optional dependencies
find_package(dynamic-graph QUIET)
find_package(dynamic-graph-python QUIET)
set(BUILD_DG_ENTITIES OFF)
if(dynamic-graph_FOUND AND dynamic-graph-python_FOUND)
  set(BUILD_DG_ENTITIES ON)
endif()

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
    src/robot_state_estimator.cpp
    src/base_ekf_with_imu_kin.cpp #
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
# Dynamic Graph Bindings.
#
if(${BUILD_DG_ENTITIES})
  # Create the python module with the entities and the python bindings.
  add_library(
    entities MODULE
    srcdg/dg_python_module.cpp #
    srcdg/robot_state_estimator.cpp #
    srcdg/robot_state_estimator.hpp #
    srcdg/signal_utils.cpp #
    srcdg/signal_utils.hpp #
  )
  # Add the include dependencies.
  target_include_directories(
    entities
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}> ${Python_INCLUDE_DIR}
           ${PYTHON_INCLUDE_DIR}) # Link the dependencies.
  target_link_libraries(entities ${PROJECT_NAME})
  target_link_libraries(entities dynamic-graph::dynamic-graph)
  target_link_libraries(entities dynamic-graph-python::dynamic-graph-python)
  # Install the python module.
  get_python_install_dir(python_install_dir)
  set(python_module_install_dir
      ${python_install_dir}/${PROJECT_NAME}/dynamic_graph)
  set_target_properties(entities PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
  get_dynamic_graph_plugin_install_path(plugin_install_path)
  set_target_properties(
    entities
    PROPERTIES
      INSTALL_RPATH
      "${CMAKE_INSTALL_RPATH}:${CMAKE_INSTALL_PREFIX}/${plugin_install_path}")
  install(TARGETS entities DESTINATION ${python_module_install_dir})
endif()

#
# Python Bindings
#
add_library(
  ${PROJECT_NAME}_cpp MODULE #
  srcpy/${PROJECT_NAME}.cpp #
  srcpy/base_ekf_with_imu_kin.cpp #
  srcpy/end_effector_force_estimator.cpp #
  srcpy/robot_state_estimator.cpp #
)
# includes
target_include_directories(${PROJECT_NAME}_cpp PRIVATE ${Python_INCLUDE_DIR})
# dependencies
target_link_libraries(${PROJECT_NAME}_cpp ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_cpp eigenpy::eigenpy)
target_link_libraries(${PROJECT_NAME}_cpp ${Python_LIBRARY})
target_link_boost_python(${PROJECT_NAME}_cpp)
# install the bindings
set_target_properties(${PROJECT_NAME}_cpp PROPERTIES PREFIX "")
get_python_install_dir(python_install_dir)
install(
  TARGETS ${PROJECT_NAME}_cpp
  LIBRARY DESTINATION ${python_install_dir}
  ARCHIVE DESTINATION ${python_install_dir}
  RUNTIME DESTINATION ${python_install_dir}
  INCLUDES
  DESTINATION include)

#
# Tests.
#
if(BUILD_TESTING)
  # Get the CTest framework
  include(CTest)
  # C++ unit-tests framework used.
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)
  # List of all C++ tests.
  set(unit_tests #
      test_butterworth_filter #
      test_end_effector_force_estimator)
  # Config folder
  get_filename_component(CONFIG_PATH config ABSOLUTE)
  # Create all C++ tests.
  foreach(unit_test ${unit_tests})
    # create the executable
    add_executable(${unit_test} tests/main.cpp tests/${unit_test}.cpp)
    # link to the created librairies and its dependencies
    target_link_libraries(${unit_test} ${PROJECT_NAME} GTest::gtest)
    # create 2 macro: the path to the config file and the path to the data file
    target_compile_definitions(${unit_test} PUBLIC CONFIG_PATH="${CONFIG_PATH}")
    # declare the test as gtest
    gtest_add_tests(TARGET ${unit_test})
  endforeach(unit_test ${unit_tests})
endif()

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

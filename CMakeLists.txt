cmake_minimum_required(VERSION 3.5.1)

project(robot_estimation)

# required to use std::shared_ptr in code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

# ensuring path to libraries are set during install
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

################
# Dependencies #
################

# these catkin packages must be installed or in the catkin workspace.
set(CATKIN_PKGS 
    mpi_cmake_modules
    yaml_cpp_catkin
)
find_package(catkin REQUIRED COMPONENTS ${CATKIN_PKGS})

catkin_add_required_dependency("pinocchio >= 2.0.0")
search_for_eigen()

# define the include directory of all ${CATKIN_PKGS}
include_directories(
    ${catkin_INCLUDE_DIRS}
    include
)

# export the package
catkin_package(
 LIBRARIES ${PROJECT_NAME}
 INCLUDE_DIRS include
 CATKIN_DEPENDS ${CATKIN_PKGS}
)

################
# Main library #
################

# define the source files
set(${PROJECT_NAME}_SRC_FILES
    src/ekf_vicon_imu.cpp
    src/io_tools/data_collector.cpp
    src/io_tools/data_reader.cpp
    src/io_tools/non_rt_data_collector.cpp
)
# create the library
add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SRC_FILES})
# link the dependencies
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
# export the library so other catkin package can use it
set(EXPORTED_LIBRAIRIES ${EXPORTED_LIBRAIRIES} ${PROJECT_NAME})

#############
# Unittests #
#############

# add unit tests
set(CONFIG_PATH ${PROJECT_SOURCE_DIR}/config)
catkin_add_gtest(test_estimator_${ROBOT}
  tests/main.cpp
  tests/ekf_imu_vicon_estimator_ut.cpp
)
# link to the created librairies and its dependencies
target_link_libraries(test_estimator_${ROBOT}
    ${PROJECT_NAME} ${catkin_LIBRARIES})
# create 2 macro: the path to the config file and the path to the data file
target_compile_definitions(test_estimator_${ROBOT} PUBLIC
    TEST_PARAMETER_YAML_FILE_PATH="${CONFIG_PATH}/demos_and_tests/test_vicon_base_state_estimator.yaml"
    TEST_DATA_FILE_PATH="${CONFIG_PATH}/demos_and_tests/test_vicon_base_state_estimator.dat"
)

########
# Demo #
########

# estimator based on IMU and Vicon
add_executable(demo_${PROJECT_NAME}
    demos/demo_ekf_vicon_imu.cpp
)
target_link_libraries(demo_${PROJECT_NAME} ${PROJECT_NAME} ${catkin_LIBRARIES})

##########################
# building documentation #
##########################
build_documentation()
/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft
 * 
 * Unit tests for the estimator based on Nick Rotella's PhD.
 * @see https://git-amd.tuebingen.mpg.de/amd-clmc/ci_example/wikis/catkin:-how-to-implement-unit-tests
 */

#include <gtest/gtest.h>
#include "estimator/demos_and_tests/ekf_vicon_imu_simple.hh"

/* ************************ SETUP CLASS **************************** */

/**
 * @brief The EstimatorTests class: test suit template for seting up the unit
 * tests for the IMU Vicon base state estimator
 *
 * In SetUp, for test, we create all the configuration file paths
 * In TearDown, we do nothing
 */
class EstimatorTests : public ::testing::Test {
protected:
  void SetUp() {
    yaml_file_ = TEST_PARAMETER_YAML_FILE_PATH;
    d_file_ = TEST_DATA_FILE_PATH;
  }

  void TearDown() {
    // do nothing the smart pointor are suppose to be dealt with automatically
  }

  std::string yaml_file_ ;
  std::string d_file_ ;
};

/* ************************ testing of estimator **************************** */

// Can we just read a dfile and estimate from it?
TEST_F(EstimatorTests, test_vicon_base_state_estimator){
  estimator::test::EstimatorViconImuTest evi_test(yaml_file_, d_file_) ;
  evi_test.run();

  // public stat on the base pose tracking
  ASSERT_NEAR(evi_test.avg_err_base_pose  , 4.16688e-06, 1e-6);
  ASSERT_NEAR(evi_test.min_err_base_pose  , 2.50663e-10, 1e-6);
  ASSERT_NEAR(evi_test.max_err_base_pose  , 0.00539555 , 1e-6);
  ASSERT_NEAR(evi_test.avg_err_base_pose_x, 9.43891e-06, 1e-6);
  ASSERT_NEAR(evi_test.min_err_base_pose_x, 2.50663e-10, 1e-6);
  ASSERT_NEAR(evi_test.max_err_base_pose_x, 0.00539555 , 1e-6);
  ASSERT_NEAR(evi_test.avg_err_base_pose_y, 7.47841e-06, 1e-6);
  ASSERT_NEAR(evi_test.min_err_base_pose_y, 1.08386e-08, 1e-6);
  ASSERT_NEAR(evi_test.max_err_base_pose_y, 0.00437135 , 1e-6);
  ASSERT_NEAR(evi_test.avg_err_base_pose_z, 3.3535e-06 , 1e-6);
  ASSERT_NEAR(evi_test.min_err_base_pose_z, 1.75194e-09, 1e-6);
  ASSERT_NEAR(evi_test.max_err_base_pose_z, 0.00224563 , 1e-6);
  ASSERT_NEAR(evi_test.avg_err_base_quat  , 2.12996e-06, 1e-6);
  ASSERT_NEAR(evi_test.min_err_base_quat  , 0.0        , 1e-6);
  ASSERT_NEAR(evi_test.max_err_base_quat  , 0.0031145  , 1e-6);
  ASSERT_NEAR(evi_test.avg_err_base_quat_x, 4.78398e-06, 1e-6);
  ASSERT_NEAR(evi_test.min_err_base_quat_x, 0.0        , 1e-6);
  ASSERT_NEAR(evi_test.max_err_base_quat_x, 0.00817328 , 1e-6);
  ASSERT_NEAR(evi_test.avg_err_base_quat_y, 3.42904e-06, 1e-6);
  ASSERT_NEAR(evi_test.min_err_base_quat_y, 0.0        , 1e-6);
  ASSERT_NEAR(evi_test.max_err_base_quat_y, 0.00498303 , 1e-6);
  ASSERT_NEAR(evi_test.avg_err_base_quat_z, 6.23457e-06, 1e-6);
  ASSERT_NEAR(evi_test.min_err_base_quat_z, 0.0        , 1e-6);
  ASSERT_NEAR(evi_test.max_err_base_quat_z, 0.00623243 , 1e-6);

  std::cout << "All assert passed, the vicon IMU base state estimator is OK!"
            << std::endl;
}

/* **************** testing RosParameters_configuration ? ******************* */
// because it uses rostest, RosParameters_configuration is tested
// in a separated file

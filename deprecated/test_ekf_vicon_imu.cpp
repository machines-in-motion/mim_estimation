/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * Unit tests for the estimator based on Nick Rotella's PhD.
 * @see
 * https://git-amd.tuebingen.mpg.de/amd-clmc/ci_example/wikis/catkin:-how-to-implement-unit-tests
 */

#include <gtest/gtest.h>
#include "mim_estimation/demos_and_tests/ekf_vicon_imu_simple.hpp"

/* ************************ SETUP CLASS **************************** */

/**
 * @brief The EstimatorTests class: test suit template for seting up the unit
 * tests for the IMU Vicon base state estimator
 *
 * In SetUp, for test, we create all the configuration file paths
 * In TearDown, we do nothing
 */
class EstimatorTests : public ::testing::Test
{
protected:
    void SetUp()
    {
        yaml_file_ = TEST_PARAMETER_YAML_FILE_PATH;
        d_file_ = TEST_DATA_FILE_PATH;
    }

    void TearDown()
    {
        // do nothing the smart pointor are suppose to be dealt with
        // automatically
    }

    std::string yaml_file_;
    std::string d_file_;
};

/* ************************ testing of estimator **************************** */

// Can we just read a dfile and estimate from it?
TEST_F(EstimatorTests, test_estimator_vicon_imu_test)
{
    mim_estimation::test::EstimatorViconImuTest evi_test(yaml_file_, d_file_);
    evi_test.run();

    // public stat on the base pose tracking
    ASSERT_LE(evi_test.avg_err_base_pose, 1e-4);
    ASSERT_LE(evi_test.min_err_base_pose, 1e-6);
    ASSERT_LE(evi_test.max_err_base_pose, 1e-1);
    //
    ASSERT_LE(evi_test.avg_err_base_pose_x, 1e-4);
    ASSERT_LE(evi_test.min_err_base_pose_x, 1e-6);
    ASSERT_LE(evi_test.max_err_base_pose_x, 1e-1);
    //
    ASSERT_LE(evi_test.avg_err_base_pose_y, 1e-4);
    ASSERT_LE(evi_test.min_err_base_pose_y, 1e-6);
    ASSERT_LE(evi_test.max_err_base_pose_y, 1e-1);
    //
    ASSERT_LE(evi_test.avg_err_base_pose_z, 1e-4);
    ASSERT_LE(evi_test.min_err_base_pose_z, 1e-6);
    ASSERT_LE(evi_test.max_err_base_pose_z, 1e-1);

    ASSERT_LE(evi_test.avg_err_base_quat_x, 1e-4);
    ASSERT_LE(evi_test.min_err_base_quat_x, 1e-6);
    ASSERT_LE(evi_test.max_err_base_quat_x, 1e-1);
    //
    ASSERT_LE(evi_test.avg_err_base_quat_y, 1e-4);
    ASSERT_LE(evi_test.min_err_base_quat_y, 1e-6);
    ASSERT_LE(evi_test.max_err_base_quat_y, 1e-1);
    //
    ASSERT_LE(evi_test.avg_err_base_quat_z, 1e-3);
    ASSERT_LE(evi_test.min_err_base_quat_z, 1e-6);
    ASSERT_LE(evi_test.max_err_base_quat_z, 1e-0);

    std::cout << "All assert passed, the vicon IMU base state estimator is OK!"
              << std::endl;
}

/* **************** testing RosParameters_configuration ? ******************* */
// because it uses rostest, RosParameters_configuration is tested
// in a separated file

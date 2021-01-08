/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * Unit tests for the end-effector force estimator.
 */

#include <gtest/gtest.h>
#include "mim_estimation/end_effector_force_estimator.hpp"

/* ************************ SETUP CLASS **************************** */

/**
 * @brief The EndEffectorForceEstimatorTests class: test suit template for
 * setting up the unit tests for the end-effector force estimator
 */
class EndEffectorForceEstimatorTests : public ::testing::Test
{
protected:
    void SetUp()
    {
        config_path_ = CONFIG_PATH;
    }

    void TearDown()
    {
        // do nothing the smart pointer are suppose to be dealt with
        // automatically
    }

    std::string config_path_;
};

/* ************************ testing of estimator **************************** */

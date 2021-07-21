/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2021, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implementation of the RobotStateEstimator class.
 */

#include "mim_estimation/robot_state_estimator.hpp"

namespace mim_estimation
{
RobotStateEstimator::RobotStateEstimator()
{
}

RobotStateEstimator::~RobotStateEstimator()
{
}

void RobotStateEstimator::initialize(RobotStateEstimatorSettings settings)
{
    settings_ = settings;

    ee_force_estimator_.initialize(settings_.urdf_path,
                                   settings_.end_effector_frame_names);
    BaseEkfWithImuKinSettings& ekf_settings = settings_;
    base_ekf_with_imu_kin_.initialize(ekf_settings);
}

void RobotStateEstimator::run(
    Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
    Eigen::Ref<const Eigen::Vector3d> imu_gyroscope,
    Eigen::Ref<const Eigen::VectorXd> joint_position,
    Eigen::Ref<const Eigen::VectorXd> joint_velocity,
    Eigen::Ref<const Eigen::VectorXd> joint_torque)
{
}

void RobotStateEstimator::get_state(
    Eigen::Ref<Eigen::VectorXd> robot_configuration,
    Eigen::Ref<Eigen::VectorXd> robot_velocity)
{
}

}  // namespace mim_estimation

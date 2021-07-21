/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2021, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implement a base state estimator using different classes in this
 * package.
 */

#pragma once

#include "mim_estimation/base_ekf_with_imu_kin.hpp"
#include "mim_estimation/end_effector_force_estimator.hpp"

namespace mim_estimation
{
struct RobotStateEstimatorSettings: public BaseEkfWithImuKinSettings
{
    /* BaseEkfWithImuKin settings (inherited) */

    /* EndEffectorForceEstimator settings. */

    /** @brief Path to th robot URDF file. */
    std::string urdf_path;
};

/**
 * @brief This class has for purpose to estimate the state of the robot in
 * the current environment. It really on different classes implemented in this
 * package and provide a uniforme way to interact with them.
 *
 * For example there is an end-effector force estimation class that can be used
 * for contact detection needed by an EKF. This class creates this pipeline
 * in a transparent way for the user. The user simply need to pick which
 * scientific method are to be used among the available ones.
 *
 */
class RobotStateEstimator
{
public:
    /** @brief Construct a new Base State Estimator object. */
    RobotStateEstimator();

    /** @brief Destroy by default the Base State Estimator object. */
    ~RobotStateEstimator();

    void initialize(RobotStateEstimatorSettings settings);

    void run(Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
             Eigen::Ref<const Eigen::Vector3d> imu_gyroscope,
             Eigen::Ref<const Eigen::VectorXd> joint_position,
             Eigen::Ref<const Eigen::VectorXd> joint_velocity,
             Eigen::Ref<const Eigen::VectorXd> joint_torque);

    void get_state(Eigen::Ref<Eigen::VectorXd> robot_configuration,
                   Eigen::Ref<Eigen::VectorXd> robot_velocity);

private:
    /** @brief End-effector force estimator. Estimate the forces in the base
     * frame. */
    EndEffectorForceEstimator ee_force_estimator_;

    /** @brief EKF that estimating the localization of the base relative to it's
     * initial position. */
    BaseEkfWithImuKin base_ekf_with_imu_kin_;

    /** @brief Settings of this class. */
    RobotStateEstimatorSettings settings_;
};

}  // namespace mim_estimation

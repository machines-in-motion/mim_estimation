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
    force_in_world_map_.clear();
}

RobotStateEstimator::~RobotStateEstimator()
{
}

void RobotStateEstimator::initialize(
    const RobotStateEstimatorSettings& settings)
{
    settings_ = settings;

    ee_force_estimator_.initialize(settings_.urdf_path,
                                   settings_.end_effector_frame_names);
    BaseEkfWithImuKinSettings& ekf_settings = settings_;
    base_ekf_with_imu_kin_.initialize(ekf_settings);
    detected_contact_.resize(settings_.end_effector_frame_names.size(), false);

    for (std::size_t i = 0; i < settings_.end_effector_frame_names.size(); ++i)
    {
        force_in_world_map_[settings_.end_effector_frame_names[i]] =
            Eigen::Vector3d::Zero();
    }
}

void RobotStateEstimator::set_initial_state(
    Eigen::Ref<const Eigen::VectorXd> initial_robot_configuration,
    Eigen::Ref<const Eigen::VectorXd> initial_robot_velocity)
{
    base_ekf_with_imu_kin_.set_initial_state(
        initial_robot_configuration.head<7>(),
        initial_robot_velocity.head<6>());

    current_robot_configuration_ = initial_robot_configuration;
    current_robot_velocity_ = initial_robot_velocity;
}

void RobotStateEstimator::run(
    const std::vector<bool>& contact_schedule,
    Eigen::Ref<const Eigen::VectorXd> joint_position,
    Eigen::Ref<const Eigen::VectorXd> joint_velocity)
{    
    base_ekf_with_imu_kin_.update(contact_schedule,
                                  joint_position,
                                  joint_velocity);

    base_ekf_with_imu_kin_.get_filter_output(current_robot_configuration_,
                                             current_robot_velocity_);
}

void RobotStateEstimator::run(
    Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
    Eigen::Ref<const Eigen::Vector3d> imu_gyroscope)
{
    base_ekf_with_imu_kin_.prediction(imu_accelerometer,
                                      imu_gyroscope);
    
    // base_ekf_with_imu_kin_.update_filter(contact_schedule,
    //                                      imu_accelerometer,
    //                                      imu_gyroscope,
    //                                      joint_position,
    //                                      joint_velocity);

    base_ekf_with_imu_kin_.get_filter_output(current_robot_configuration_,
                                             current_robot_velocity_);
}

void RobotStateEstimator::compute_midline(
    const std::vector<bool>& contact_schedule,
    Eigen::Ref<const Eigen::VectorXd> joint_position,
    Eigen::Ref<const Eigen::VectorXd> joint_velocity)
{
    base_ekf_with_imu_kin_.compute_base_pose_to_midline(contact_schedule,
                                                        joint_position,
                                                        joint_velocity);
}

void RobotStateEstimator::run(
    Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
    Eigen::Ref<const Eigen::Vector3d> imu_gyroscope,
    Eigen::Ref<const Eigen::VectorXd> joint_position,
    Eigen::Ref<const Eigen::VectorXd> joint_velocity,
    Eigen::Ref<const Eigen::VectorXd> joint_torque)
{
    // Extract the current base orientation.
    rot_base_in_world_.x() = current_robot_configuration_(3);
    rot_base_in_world_.y() = current_robot_configuration_(4);
    rot_base_in_world_.z() = current_robot_configuration_(5);
    rot_base_in_world_.w() = current_robot_configuration_(6);
    rot_base_in_world_.normalize();

    // Get the estimated force.
    ee_force_estimator_.run(joint_position, joint_torque);

    // Estimate the contact state.
    std::size_t nb_ee = settings_.end_effector_frame_names.size();
    for (std::size_t i = 0; i < nb_ee; ++i)
    {
        const std::string& frame_name = settings_.end_effector_frame_names[i];

        const Eigen::Matrix<double, 6, 1>& force =
            ee_force_estimator_.get_force(frame_name);

        Eigen::Vector3d& force_in_world = force_in_world_map_[frame_name];

        force_in_world =
            rot_base_in_world_.toRotationMatrix() * force.head<3>();
        if (force_in_world.norm() > settings_.force_threshold_up)
        {
            detected_contact_[i] = true;
        }
        if (force_in_world.norm() < settings_.force_threshold_down)
        {
            detected_contact_[i] = false;
        }
    }

    run(imu_accelerometer,
        imu_gyroscope);
}

void RobotStateEstimator::set_settings(
    const RobotStateEstimatorSettings& settings)
{
    settings_ = settings;
    settings_.force_threshold_up = settings_.force_threshold_up;
    settings_.force_threshold_down = settings_.force_threshold_down;
}

const Eigen::Vector3d& RobotStateEstimator::get_force(
    const std::string& frame_name)
{
    return force_in_world_map_[frame_name];
}

void RobotStateEstimator::get_state(
    Eigen::Ref<Eigen::VectorXd> robot_configuration,
    Eigen::Ref<Eigen::VectorXd> robot_velocity)
{
    robot_configuration = current_robot_configuration_;
    robot_velocity = current_robot_velocity_;
}

const Eigen::VectorXd& RobotStateEstimator::get_robot_configuration() const
{
    return current_robot_configuration_;
}

const Eigen::VectorXd& RobotStateEstimator::get_robot_velocity() const
{
    return current_robot_velocity_;
}

const std::vector<bool>& RobotStateEstimator::get_detected_contact() const
{
    return detected_contact_;
}

const RobotStateEstimatorSettings& RobotStateEstimator::get_settings() const
{
    return settings_;
}

}  // namespace mim_estimation

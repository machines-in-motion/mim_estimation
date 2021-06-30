/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2021, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implements the classes and struct from base_ekf_with_imu_kin.hpp.
 */

#include "mim_estimation/base_ekf_with_imu_kin.hpp"

namespace mim_estimation
{
BaseEkfWithImuKin::BaseEkfWithImuKin()
{
    gravity_ << 0.0, 0.0, -9.81;
    root_velocities_per_contact_.clear();
}

BaseEkfWithImuKin::~BaseEkfWithImuKin()
{
}

void BaseEkfWithImuKin::initialize(const BaseEkfWithImuKinSettings& settings)
{
    settings_ = settings;

    // Extract the end-effector frame ids.
    int nb_ee = settings.end_effector_frame_names.size();
    contact_frame_id_.resize(nb_ee);
    for (unsigned int i = 0; nb_ee; ++i)
    {
        contact_frame_id_[i] = settings_.pinocchio_model.getFrameId(
            settings_.end_effector_frame_names[i]);
    }
}

void BaseEkfWithImuKin::set_initial_state(
    const Eigen::Vector3d& base_position,
    const Eigen::Quaterniond& base_attitude,
    const Eigen::Vector3d& base_linear_velocity,
    const Eigen::Vector3d& base_angular_velocity)
{
    if (settings_.is_imu_frame)
    {
        pinocchio::SE3 pos_base_in_world(base_attitude.toRotationMatrix(),
                                         base_position);
        pinocchio::Motion vel_base_in_world(base_linear_velocity,
                                            base_angular_velocity);

        pinocchio::SE3 pos_imu_in_world =
            pos_base_in_world.act(settings_.imu_in_base);
        pinocchio::Motion vel_imu_in_world =
            settings_.imu_in_base.act(vel_base_in_world);

        mu_post_.position = pos_imu_in_world.translation();
        mu_post_.attitude = pos_imu_in_world.rotation();
        mu_post_.linear_velocity = vel_base_in_world.linear();
    }
    else
    {
        mu_post_.position = base_position;
        mu_post_.attitude = base_attitude;
        mu_post_.linear_velocity = base_linear_velocity;
    }
}

void BaseEkfWithImuKin::update_filter(
    const std::vector<bool>& contact_schedule,
    Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
    Eigen::Ref<const Eigen::Vector3d> imu_gyroscope,
    Eigen::Ref<const Eigen::VectorXd> joint_position,
    Eigen::Ref<const Eigen::VectorXd> joint_velocity)
{
}

void BaseEkfWithImuKin::get_filter_output(
    Eigen::Ref<Eigen::VectorXd> robot_configuration,
    Eigen::Ref<Eigen::VectorXd> robot_velocity)
{
}

void BaseEkfWithImuKin::compute_end_effectors_forward_kinematics(
    Eigen::Ref<const Eigen::VectorXd> joint_position,
    Eigen::Ref<const Eigen::VectorXd> joint_velocity)
{
}

void BaseEkfWithImuKin::integrate_process_model(
    Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
    Eigen::Ref<const Eigen::Vector3d> imu_gyroscope)
{
}

void BaseEkfWithImuKin::compute_discrete_prediction_jacobian()
{
}

void BaseEkfWithImuKin::compute_noise_jacobian()
{
}

void BaseEkfWithImuKin::construct_continuous_noise_covariance()
{
}

void BaseEkfWithImuKin::construct_discrete_noise_covariance()
{
}

void BaseEkfWithImuKin::construct_discrete_measurement_noise_covariance()
{
}

void BaseEkfWithImuKin::prediction_step()
{
}

void BaseEkfWithImuKin::measurement_model(
    const std::vector<bool>& contact_schedule,
    Eigen::Ref<const Eigen::VectorXd> joint_position,
    Eigen::Ref<const Eigen::VectorXd> joint_velocity)
{
}

}  // namespace mim_estimation

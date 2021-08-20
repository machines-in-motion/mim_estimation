/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2021, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implements the classes and struct from base_ekf_with_imu_kin.hpp.
 */

#include "mim_estimation/base_ekf_with_imu_kin.hpp"

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/math/quaternion.hpp"

namespace mim_estimation
{
BaseEkfWithImuKin::BaseEkfWithImuKin()
{
    gravity_ << 0.0, 0.0, -9.81;
    kin_meas_root_velocity_.clear();
    kin_ee_position_.clear();
    kin_ee_velocity_.clear();
    kin_ee_fid_.clear();
    cont_proc_jac_ = Eigen::MatrixXd::Zero(posterior_state_.state_dim,
                                           posterior_state_.state_dim);
    disc_proc_jac_ = Eigen::MatrixXd::Zero(posterior_state_.state_dim,
                                           posterior_state_.state_dim);
    proc_noise_jac_ = Eigen::MatrixXd::Zero(posterior_state_.state_dim,
                                            posterior_state_.noise_dim);
    cont_proc_noise_cov_ = Eigen::MatrixXd::Zero(posterior_state_.noise_dim,
                                                 posterior_state_.noise_dim);
    disc_proc_noise_cov_ = Eigen::MatrixXd::Zero(posterior_state_.noise_dim,
                                                 posterior_state_.noise_dim);
}

BaseEkfWithImuKin::~BaseEkfWithImuKin()
{
}

void BaseEkfWithImuKin::initialize(const BaseEkfWithImuKinSettings& settings)
{
    settings_ = settings;

    if (settings_.pinocchio_model == pinocchio::Model())
    {
        throw std::runtime_error(
            "BaseEkfWithImuKin::initialize(settings): "
            "Please initialize the pinocchio model "
            "in the setting object.");
    }

    // Number of end-effector.
    int nb_ee = settings.end_effector_frame_names.size();

    // resize all end-effector vectors.
    kin_meas_root_velocity_.resize(nb_ee, Eigen::Vector3d::Zero());
    kin_ee_position_.resize(nb_ee, Eigen::Vector3d::Zero());
    kin_ee_velocity_.resize(nb_ee, Eigen::Vector3d::Zero());
    kin_ee_fid_.resize(nb_ee);

    // Extract the end-effector frame ids.
    for (int i = 0; i < nb_ee; ++i)
    {
        std::string& frame_name = settings_.end_effector_frame_names[i];
        if (settings_.pinocchio_model.existFrame(frame_name))
        {
            kin_ee_fid_[i] = settings_.pinocchio_model.getFrameId(frame_name);
        }
        else
        {
            throw std::runtime_error(
                "The end effector frame name (" + frame_name +
                ") does not exist in the pinocchio::Model.");
        }
    }

    // Resize the measurement matrices.
    int measurement_size = settings_.end_effector_frame_names.size() * 3;
    meas_jac_ =
        Eigen::MatrixXd::Zero(measurement_size, posterior_state_.state_dim);
    meas_error_ = Eigen::VectorXd::Zero(measurement_size);
    kalman_gain_ =
        Eigen::MatrixXd::Zero(posterior_state_.state_dim, measurement_size);
    disc_meas_noise_cov_ =
        Eigen::MatrixXd::Zero(measurement_size, measurement_size);

    // Kinematic vectors.
    q_kin_ = Eigen::VectorXd::Zero(settings_.pinocchio_model.nq);
    q_kin_(6) = 1.0;
    dq_kin_ = Eigen::VectorXd::Zero(settings_.pinocchio_model.nv);

    // kinematics data.
    pinocchio_data_ = pinocchio::Data(settings_.pinocchio_model);

    // Reset the integration model.
    root_angular_velocity_.setZero();
}

void BaseEkfWithImuKin::set_initial_state(
    Eigen::Ref<const Eigen::Vector3d> base_position,
    const Eigen::Quaterniond& base_attitude,
    Eigen::Ref<const Eigen::Vector3d> base_linear_velocity,
    Eigen::Ref<const Eigen::Vector3d> base_angular_velocity)
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

        posterior_state_.position = pos_imu_in_world.translation();
        posterior_state_.attitude = pos_imu_in_world.rotation();
        posterior_state_.linear_velocity = vel_imu_in_world.linear();
    }
    else
    {
        posterior_state_.position = base_position;
        posterior_state_.attitude = base_attitude;
        posterior_state_.linear_velocity = base_linear_velocity;
    }
}

void BaseEkfWithImuKin::set_initial_state(
    Eigen::Ref<const Eigen::Matrix<double, 7, 1> > base_se3_position,
    Eigen::Ref<const Eigen::Matrix<double, 6, 1> > base_se3_velocity)
{
    Eigen::Quaterniond base_attitude;
    base_attitude.x() = base_se3_position(3);
    base_attitude.y() = base_se3_position(4);
    base_attitude.z() = base_se3_position(5);
    base_attitude.w() = base_se3_position(6);
    set_initial_state(base_se3_position.block<3, 1>(0, 0),
                      base_attitude,
                      base_se3_velocity.block<3, 1>(0, 0),
                      base_se3_velocity.block<3, 1>(3, 0));
}

void BaseEkfWithImuKin::update_filter(
    const std::vector<bool>& contact_schedule,
    Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
    Eigen::Ref<const Eigen::Vector3d> imu_gyroscope,
    Eigen::Ref<const Eigen::VectorXd> joint_position,
    Eigen::Ref<const Eigen::VectorXd> joint_velocity)
{
    // internal copy of the joint readings.
    joint_position_ = joint_position;
    joint_velocity_ = joint_velocity;

    // Compute the EKF output.
    integrate_process_model(imu_accelerometer, imu_gyroscope);
    prediction_step();
    update_step(contact_schedule, joint_position, joint_velocity);
}

void BaseEkfWithImuKin::get_filter_output(
    Eigen::Ref<Eigen::VectorXd> robot_configuration,
    Eigen::Ref<Eigen::VectorXd> robot_velocity)
{
    assert(robot_configuration.size() == settings_.pinocchio_model.nq &&
           "robot_configuration of wrong size.");
    assert(robot_velocity.size() == settings_.pinocchio_model.nv &&
           "robot_velocity of wrong size.");

    // mu post is expressed in the imu frame.
    if (settings_.is_imu_frame)
    {
        posterior_state_.attitude.normalize();
        pinocchio::SE3 pos_imu_in_world(
            posterior_state_.attitude.toRotationMatrix(),
            posterior_state_.position);
        pinocchio::Motion vel_imu_in_world(
            posterior_state_.linear_velocity,
            imu_gyroscope_ - posterior_state_.bias_gyroscope);

        pinocchio::SE3 pos_base_in_world =
            pos_imu_in_world.act(settings_.imu_in_base.inverse());
        pinocchio::Motion vel_base_in_world =
            settings_.imu_in_base.actInv(vel_imu_in_world);

        robot_configuration.head<3>() = pos_base_in_world.translation();
        Eigen::Quaterniond q;
        pinocchio::quaternion::assignQuaternion(q,
                                                pos_base_in_world.rotation());
        q.normalize();
        robot_configuration(3) = q.x();
        robot_configuration(4) = q.y();
        robot_configuration(5) = q.z();
        robot_configuration(6) = q.w();
        //
        robot_velocity.head<6>() = vel_base_in_world.toVector();
    }
    // mu post is expressed in the base frame.
    else
    {
        robot_configuration.head<3>() = posterior_state_.position;
        robot_configuration(3) = posterior_state_.attitude.x();
        robot_configuration(4) = posterior_state_.attitude.y();
        robot_configuration(5) = posterior_state_.attitude.z();
        robot_configuration(6) = posterior_state_.attitude.w();
        //
        robot_velocity.head<3>() = posterior_state_.linear_velocity;
        robot_velocity.segment<3>(3) =
            imu_gyroscope_ - posterior_state_.bias_gyroscope;
    }
    robot_configuration.tail(joint_position_.size()) = joint_position_;
    robot_velocity.tail(joint_velocity_.size()) = joint_velocity_;
}

const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >&
BaseEkfWithImuKin::get_measurement()
{
    return kin_meas_root_velocity_;
}

void BaseEkfWithImuKin::compute_end_effectors_forward_kinematics(
    Eigen::Ref<const Eigen::VectorXd> joint_position,
    Eigen::Ref<const Eigen::VectorXd> joint_velocity)
{
    const pinocchio::Model& pinocchio_model = settings_.pinocchio_model;
    int nb_joint_dof = pinocchio_model.nq - 7;
    assert(joint_position.size() == nb_joint_dof &&
           "'joint_position' has wrong size.");
    assert(joint_velocity.size() == nb_joint_dof &&
           "'joint_velocity' has wrong size.");

    // Fill in the robot configuration and velocity.
    q_kin_.fill(0.0);
    q_kin_(6) = 1.0;
    q_kin_.segment(7, nb_joint_dof) = joint_position;
    dq_kin_.head<6>().fill(0.0);
    dq_kin_.segment(6, nb_joint_dof) = joint_velocity;

    // Perform the Forward kinematics.
    pinocchio::forwardKinematics(
        pinocchio_model, pinocchio_data_, q_kin_, dq_kin_);
    pinocchio::framesForwardKinematics(
        pinocchio_model, pinocchio_data_, q_kin_);

    std::size_t nb_ee = settings_.end_effector_frame_names.size();
    for (std::size_t i = 0; i < nb_ee; ++i)
    {
        pinocchio::updateFramePlacement(
            pinocchio_model, pinocchio_data_, kin_ee_fid_[i]);

        kin_ee_position_[i] = pinocchio_data_.oMf[kin_ee_fid_[i]].translation();
        kin_ee_velocity_[i] =
            pinocchio::getFrameVelocity(pinocchio_model,
                                        pinocchio_data_,
                                        kin_ee_fid_[i],
                                        pinocchio::LOCAL_WORLD_ALIGNED)
                .linear();
    }
}

void BaseEkfWithImuKin::integrate_process_model(
    Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
    Eigen::Ref<const Eigen::Vector3d> imu_gyroscope)
{
    // Some shortcut for easier code writing.
    const double& dt = settings_.dt;
    const pinocchio::SE3& imu_in_base = settings_.imu_in_base;
    // Get the base to world rotation matrix.
    posterior_state_.attitude.normalize();
    attitude_post_ = posterior_state_.attitude.toRotationMatrix();

    if (settings_.is_imu_frame)
    {
    }
    else
    {
        // Get the base angular velocity
        root_angular_velocity_ =
            imu_in_base.rotation() *
            (imu_gyroscope - posterior_state_.bias_gyroscope);

        // Compute the base angular acceleration numerically.
        root_angular_acceleration_ =
            (root_angular_velocity_ - root_angular_velocity_prev_) / dt;
        root_angular_velocity_prev_ = root_angular_velocity_;

        // Compute the base linear acceleration.
        root_linear_acceleration_ =
            imu_in_base.rotation() *
                (imu_accelerometer - posterior_state_.bias_accelerometer) +
            root_angular_acceleration_.cross(-imu_in_base.translation()) +
            root_angular_velocity_.cross(
                root_angular_velocity_.cross(-imu_in_base.translation()));
    }

    predicted_state_.position =
        posterior_state_.position + posterior_state_.linear_velocity * dt;
    predicted_state_.linear_velocity =
        posterior_state_.linear_velocity +
        (-root_angular_velocity_.cross(posterior_state_.linear_velocity) +
         attitude_post_.transpose() * gravity_ + root_linear_acceleration_) *
            dt;
    predicted_state_.attitude =
        attitude_post_ * pinocchio::exp3(root_angular_velocity_ * dt);
    predicted_state_.bias_accelerometer = posterior_state_.bias_accelerometer;
    predicted_state_.bias_gyroscope = posterior_state_.bias_gyroscope;
}

void BaseEkfWithImuKin::compute_discrete_prediction_jacobian()
{
    const double& dt = settings_.dt;
    Eigen::Vector3d& v_pre = predicted_state_.linear_velocity;
    Eigen::Vector3d& omega_hat = root_angular_velocity_;
    /// @todo remove this tmp variable creation here and create an attribute.
    predicted_state_.attitude.normalize();
    Eigen::Matrix3d R_pre = predicted_state_.attitude.toRotationMatrix();
    // dr/ddelta_x;
    cont_proc_jac_.block<3, 3>(0, 3) = R_pre;
    cont_proc_jac_.block<3, 3>(0, 6) = -R_pre * pinocchio::skew(v_pre);
    // dv/ddelta_x;
    cont_proc_jac_.block<3, 3>(3, 3) = -pinocchio::skew(omega_hat);
    cont_proc_jac_.block<3, 3>(3, 6) =
        pinocchio::skew(R_pre.transpose() * gravity_);
    cont_proc_jac_.block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity();
    cont_proc_jac_.block<3, 3>(3, 12) = -pinocchio::skew(v_pre);
    // dtheta/ddelta_x;
    cont_proc_jac_.block<3, 3>(6, 6) = -pinocchio::skew(omega_hat);
    cont_proc_jac_.block<3, 3>(6, 12) = -Eigen::Matrix3d::Identity();
    // Discretise the continuous process jacobian.
    disc_proc_jac_ = Eigen::MatrixXd::Identity(predicted_state_.state_dim,
                                               predicted_state_.state_dim) +
                     cont_proc_jac_ * dt;
}

void BaseEkfWithImuKin::compute_noise_jacobian()
{
    proc_noise_jac_.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
    proc_noise_jac_.block<3, 3>(3, 3) =
        -pinocchio::skew(predicted_state_.linear_velocity);
    proc_noise_jac_.block<3, 3>(6, 3) = -Eigen::Matrix3d::Identity();
    proc_noise_jac_.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
    proc_noise_jac_.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();
}

void BaseEkfWithImuKin::construct_continuous_noise_covariance()
{
    cont_proc_noise_cov_.block<3, 3>(0, 0) =
        settings_.noise_accelerometer.asDiagonal();
    cont_proc_noise_cov_.block<3, 3>(3, 3) =
        settings_.noise_gyroscope.asDiagonal();
    cont_proc_noise_cov_.block<3, 3>(6, 6) =
        settings_.noise_bias_accelerometer.asDiagonal();
    cont_proc_noise_cov_.block<3, 3>(9, 9) =
        settings_.noise_bias_gyroscope.asDiagonal();
}

void BaseEkfWithImuKin::construct_discrete_noise_covariance()
{
    disc_proc_noise_cov_.fill(0.0);
    Eigen::MatrixXd& Fk = disc_proc_jac_;
    Eigen::MatrixXd& Lc = proc_noise_jac_;
    Eigen::MatrixXd& Qc = cont_proc_noise_cov_;
    disc_proc_noise_cov_ =
        (Fk * Lc * Qc * Lc.transpose() * Fk.transpose()) * settings_.dt;
}

void BaseEkfWithImuKin::construct_discrete_measurement_noise_covariance()
{
    disc_meas_noise_cov_.fill(0.0);
    // Construct the continuous measurement covariance matrix first.
    for (unsigned int i = 0; i < settings_.end_effector_frame_names.size(); ++i)
    {
        disc_meas_noise_cov_.block<3, 3>(i * 3, i * 3) =
            settings_.meas_noise_cov.asDiagonal();
    }
    // Discretise it.
    disc_meas_noise_cov_ /= settings_.dt;
}

void BaseEkfWithImuKin::prediction_step()
{
    compute_discrete_prediction_jacobian();
    compute_noise_jacobian();
    construct_continuous_noise_covariance();
    construct_discrete_noise_covariance();
    // Shortcuts.
    Eigen::MatrixXd& Fk = disc_proc_jac_;
    Eigen::MatrixXd& Qk = disc_proc_noise_cov_;
    // Update the state covariance.
    predicted_state_.covariance =
        (Fk * posterior_state_.covariance * Fk.transpose()) + Qk;
}

void BaseEkfWithImuKin::measurement_model(
    const std::vector<bool>& contact_schedule,
    Eigen::Ref<const Eigen::VectorXd> joint_position,
    Eigen::Ref<const Eigen::VectorXd> joint_velocity)
{
    // end effectors frame positions and velocities expressed in the world frame
    compute_end_effectors_forward_kinematics(joint_position, joint_velocity);

    for (unsigned int i = 0; i < settings_.end_effector_frame_names.size(); ++i)
    {
        // If the end-effector is in contact we update the measurement.
        if (contact_schedule[i])
        {
            if (settings_.is_imu_frame)
            {
            }
            else
            {
                // compute measurement jacobian
                meas_jac_.block<3, 3>(3 * i, 3) = Eigen::Matrix3d::Identity();
                meas_jac_.block<3, 3>(3 * i, 12) =
                    pinocchio::skew(kin_ee_position_[i]);

                // compute measurement error
                kin_meas_root_velocity_[i] =
                    -kin_ee_velocity_[i] -
                    root_angular_velocity_.cross(kin_ee_position_[i]);
                meas_error_.segment<3>(i * 3) =
                    kin_meas_root_velocity_[i] -
                    predicted_state_.linear_velocity;
            }
        }
        // Otherwise we set the error to 0.
        else
        {
            // compute measurement jacobian
            meas_jac_.middleRows<3>(i * 3).setZero();

            // compute measurement error
            meas_error_.segment<3>(i + 3).setZero();
        }
    }
}

void BaseEkfWithImuKin::update_step(
    const std::vector<bool>& contact_schedule,
    Eigen::Ref<const Eigen::VectorXd> joint_position,
    Eigen::Ref<const Eigen::VectorXd> joint_velocity)
{
    /// @todo remove this tmp variable creation here and create an attribute.
    predicted_state_.attitude.normalize();
    Eigen::Matrix3d R_pre = predicted_state_.attitude.toRotationMatrix();

    // Compute the measurement related values
    measurement_model(contact_schedule, joint_position, joint_velocity);
    construct_discrete_measurement_noise_covariance();
    meas_covariance_ =
        (meas_jac_ * predicted_state_.covariance * meas_jac_.transpose()) +
        disc_meas_noise_cov_;

    // Compute the kalman gain.
    ldlt.compute(meas_covariance_);
    kalman_gain_ = (ldlt.solve(meas_jac_)).transpose();
    kalman_gain_ = predicted_state_.covariance * kalman_gain_;

    // Compute the delta in the state.
    delta_state_ = kalman_gain_ * meas_error_;

    /// @todo remove this line, cancel the update
    // delta_state_.setZero();

    // update the current state (posterior)
    posterior_state_.covariance =
        (Eigen::MatrixXd::Identity(posterior_state_.state_dim,
                                   posterior_state_.state_dim) -
         kalman_gain_ * meas_jac_) *
        predicted_state_.covariance;
    //
    posterior_state_.position =
        predicted_state_.position + delta_state_.segment<3>(0);
    //
    posterior_state_.linear_velocity =
        predicted_state_.linear_velocity + delta_state_.segment<3>(3);
    //
    posterior_state_.attitude =
        R_pre * pinocchio::exp3(delta_state_.segment<3>(6));
    posterior_state_.attitude.normalize();
    //
    posterior_state_.bias_accelerometer =
        predicted_state_.bias_accelerometer + delta_state_.segment<3>(9);
    //
    posterior_state_.bias_accelerometer =
        predicted_state_.bias_gyroscope + delta_state_.segment<3>(12);
}

}  // namespace mim_estimation

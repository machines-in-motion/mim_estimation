/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implement the EkfViconImu class.
 */

#include "robot_estimation/ekf_vicon_imu.hpp"
#include "pinocchio/math/rpy.hpp"
#include "pinocchio/spatial/skew.hpp"

namespace robot_estimation
{
EkfViconImu::EkfViconImu(double dt, const YAML::Node& config)
    : config_(config), EKF(false, false, dt, 1)
{
}

void EkfViconImu::initialize(
    const Eigen::Ref<const Eigen::Matrix4d>& base_pose_data)
{
    initialize(base_pose_data.topRightCorner<3, 1>(),
               base_pose_data.topLeftCorner<3, 3>());
}

void EkfViconImu::initialize(
    const Eigen::Ref<const Eigen::Vector3d>& base_pose,
    const Eigen::Ref<const Eigen::Matrix3d>& base_ori_mat)
{
    base_quat_vicon_ = base_ori_mat;
    initialize(base_pose, base_quat_vicon_);
}

void EkfViconImu::initialize(
    const Eigen::Ref<const Eigen::Vector3d>& base_pose,
    const Eigen::Quaterniond& base_quat)
{
    // Load noise and other config parameters from YAML file:
    init_var_ = config_["init_var"].as<double>();

    grav_vec_ << 0.0, 0.0, -9.81;

    // TODO load this from the robot model
    base_to_imu_translation_ << 0.0828, 0.2025, 0.0743;

    // Initialize covariances:
    state_pre_.state_cov =
        init_var_ * Eigen::Matrix<double,
                                  ViconIMUState::state_dim,
                                  ViconIMUState::state_dim>::Identity();
    state_post_.state_cov =
        init_var_ * Eigen::Matrix<double,
                                  ViconIMUState::state_dim,
                                  ViconIMUState::state_dim>::Identity();
    meas_pred_.meas_cov =
        init_var_ * Eigen::Matrix<double,
                                  ViconIMUMeasure::meas_dim,
                                  ViconIMUMeasure::meas_dim>::Identity();
    meas_actual_.meas_cov =
        init_var_ * Eigen::Matrix<double,
                                  ViconIMUMeasure::meas_dim,
                                  ViconIMUMeasure::meas_dim>::Identity();

    // Initialize noise paremeters:
    q_proc_accel_ = config_["proc_noise_accel"].as<double>();
    q_proc_gyro_ = config_["proc_noise_gyro"].as<double>();
    q_proc_accel_bias_ = config_["proc_noise_accel_bias"].as<double>();
    q_proc_gyro_bias_ = config_["proc_noise_gyro_bias"].as<double>();
    q_meas_base_pos_init_ = config_["meas_noise_base_pos"].as<double>();
    q_meas_base_quat_init_ = config_["meas_noise_base_quat"].as<double>();
    // q_meas_base_weight_ = config_["q_meas_base_weight"].as<double>();
    q_meas_base_pos_ = q_meas_base_pos_init_;
    q_meas_base_quat_ = q_meas_base_quat_init_;
    // get initial yaw
    Eigen::Quaterniond base_pose_data_yaw;
    base_quat_vicon_ = base_quat;
    base_quat_vicon_.normalize();
    Eigen::Vector3d init_base_abg =
        pinocchio::rpy::matrixToRpy(base_quat_vicon_.toRotationMatrix());
    init_base_abg(2) = 0.0;  // zero the yaw
    base_pose_data_yaw = pinocchio::rpy::rpyToMatrix(init_base_abg);

    // remove the yaw component
    // initial_base_quat_vicon_ = base_pose_data_yaw * base_quat_vicon_;
    initial_base_quat_vicon_ = base_quat_vicon_;

    // copy the base pose as initial state
    initial_base_pos_vicon_ = base_pose;

#ifndef ZERO_INITIAL_STATE
    // initial imu pos
    state_post_.imu_pos.setZero();
    // initial imu quat
    state_post_.imu_pos.set(1.0, 0.0, 0.0, 0.0);
#else
    state_post_.imu_pos =
        initial_base_pos_vicon_ +
        base_quat.toRotationMatrix() * base_to_imu_translation_;
    state_post_.imu_quat = initial_base_quat_vicon_;
    initial_base_pos_vicon_.setZero();
    initial_base_quat_vicon_ = Eigen::Quaterniond::Identity();
#endif

    // Initialize the rest of the filter state:
    state_post_.imu_vel.setZero();
    state_post_.accel_bias.setZero();
    state_post_.accel_bias(2) = -9.81; // the bias is the gravity field
    state_post_.gyro_bias.setZero();
    state_pre_ = state_post_;

    //  std::cout << "Vicon BSE Initial State:" << std::endl;
    //  std::cout << state_post_.getState().transpose() << std::endl
    //              << state_post_.imu_pos.transpose() << " ; "
    //            << state_post_.imu_vel.transpose() << " ; "
    //            << state_post_.imu_quat.get().transpose() << " ; "
    //            << state_post_.accel_bias.transpose() << " ; "
    //            << state_post_.gyro_bias.transpose() << std::endl
    //            << std::endl;

    proc_jac_.setZero();
    noise_jac_.setZero();
    meas_jac_.setZero();

    is_new_frame_ = 0;
    frame_quality_ = 0;
}

void EkfViconImu::update(const Eigen::Vector3d& accelerometer,
                         const Eigen::Vector3d& gyroscope,
                         const Eigen::Ref<const Eigen::Matrix4d>& base_pose_data,
                         const bool is_new_frame)
{
    base_quat_vicon_ = base_pose_data.topLeftCorner<3, 3>();
    update(accelerometer,
           gyroscope,
           base_pose_data.topRightCorner<3, 1>(),
           base_quat_vicon_,
           is_new_frame);
}

void EkfViconImu::update(const Eigen::Vector3d& accelerometer,
                         const Eigen::Vector3d& gyroscope,
                         const Eigen::Ref<const Eigen::Vector3d>& base_pose,
                         const Eigen::Quaterniond& base_quat,
                         const bool is_new_frame)
{
    // store IMU, and MoCap data:
    imu_accel_ = accelerometer;
    imu_angvel_ = gyroscope;
    base_pos_vicon_ = base_pose - initial_base_pos_vicon_;
    base_quat_vicon_ = base_quat;
    base_quat_vicon_.normalize();
    base_quat_vicon_ = base_quat_vicon_ * initial_base_quat_vicon_.inverse();
    // manage the covariance of the measured model in function of the quality of
    // the measure
    is_new_frame_ = is_new_frame;

    // call the EKF
    updateFilter(is_new_frame);
}

Eigen::Matrix<double, ViconIMUState::state_dim, 1> EkfViconImu::processModel(
    ViconIMUState& s)
{
    Eigen::Matrix<double, ViconIMUState::state_dim, 1> out;

    // Predict the filter state:
    Eigen::Matrix3d C = s.imu_quat.toRotationMatrix().transpose();
    out.segment(0, 3) = s.imu_pos + dt_ * s.imu_vel;
    out.segment(3, 3) =
        s.imu_vel +
        dt_ * (C.transpose() * (imu_accel_ - s.accel_bias) + grav_vec_);
    // imu_rotation_3d = log3(exp3(dt * ang_vel) * imu_quat)
    out.segment(6, 3) = pinocchio::quaternion::log3(
        s.imu_quat *
        pinocchio::quaternion::exp3(dt_ * (imu_angvel_ - s.gyro_bias)));
    out.segment(9, 3) = s.accel_bias;
    out.segment(12, 3) = s.gyro_bias;

    return out;
}

void EkfViconImu::formProcessJacobian(void)
{
    proc_jac_.setZero();
    Eigen::Matrix3d world_R_base = state_post_.imu_quat.toRotationMatrix();

    proc_jac_.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();

    proc_jac_.block(3, 6, 3, 3) =
        -world_R_base * pinocchio::skew(imu_accel_ - state_post_.accel_bias);
    proc_jac_.block(3, 9, 3, 3) = -world_R_base;

    proc_jac_.block(6, 6, 3, 3) =
        -pinocchio::skew(imu_angvel_ - state_post_.gyro_bias);
    proc_jac_.block(6, 12, 3, 3) = -Eigen::Matrix3d::Identity();
    return;
}

void EkfViconImu::formProcessNoise(void)
{
    proc_noise_.block(3, 3, 3, 3) =
        q_proc_accel_ * q_proc_accel_ * Eigen::MatrixXd::Identity(3, 3);
    proc_noise_.block(6, 6, 3, 3) =
        q_proc_gyro_ * q_proc_gyro_ * Eigen::MatrixXd::Identity(3, 3);
    proc_noise_.block(9, 9, 3, 3) = q_proc_accel_bias_ * q_proc_accel_bias_ *
                                    Eigen::MatrixXd::Identity(3, 3);
    proc_noise_.block(12, 12, 3, 3) =
        q_proc_gyro_bias_ * q_proc_gyro_bias_ * Eigen::MatrixXd::Identity(3, 3);
    return;
}

/**
 * @brief EkfViconImu::formNoiseJacobian
 *
 * The vector of the noise is : [accel gyro accel_bias gyro_bias]^T
 */
void EkfViconImu::formNoiseJacobian(void)
{
    if (is_discrete_)
    {
        noise_jac_.setZero();
    }
    else
    {
        Eigen::Matrix3d world_R_base = state_post_.imu_quat.toRotationMatrix();

        noise_jac_.block(0, 0, 3, 3) = -world_R_base;  // acc
        noise_jac_.block(0, 6, 3, 3) = -world_R_base;  // + acc_bias

        noise_jac_.block(3, 3, 3, 3) = -Eigen::Matrix3d::Identity();  // gyro
        noise_jac_.block(3, 9, 3, 3) =
            -Eigen::Matrix3d::Identity();  // gyro_bias

        noise_jac_.block(6, 6, 3, 3) = Eigen::Matrix3d::Identity();  // acc_bias

        noise_jac_.block(9, 9, 3, 3) =
            Eigen::Matrix3d::Identity();  // gyro_bias
    }
}

Eigen::Matrix<double, ViconIMUMeasure::meas_dim, 1> EkfViconImu::measModel(
    ViconIMUState& s)
{
    // Measurement is simply the imu position and orientation from Vicon:
    Eigen::Matrix<double, ViconIMUMeasure::meas_dim, 1> out;
    out.segment(0, 3) = s.imu_pos;
    out.segment(3, 3) = pinocchio::quaternion::log3(s.imu_quat);

    return out;
}

void EkfViconImu::formMeasJacobian(void)
{
    meas_jac_.block(0, 0, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
    meas_jac_.block(3, 6, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
    return;
}

void EkfViconImu::formMeasNoise(void)
{
    meas_noise_.block(0, 0, 3, 3) =
        q_meas_base_pos_ * q_meas_base_pos_ * Eigen::MatrixXd::Identity(3, 3);
    meas_noise_.block(3, 3, 3, 3) =
        q_meas_base_quat_ * q_meas_base_quat_ * Eigen::MatrixXd::Identity(3, 3);
    return;
}

void EkfViconImu::formActualMeas(void)
{
    /*tflayols: [done?] need to translate vicon measurement to express IMU
     * position*/
    /*assuming imu and base share the same orientation*/
    Eigen::Matrix3d world_R_base = base_quat_vicon_.toRotationMatrix();
    meas_actual_.meas_imu_pos =
        base_pos_vicon_ + (world_R_base * base_to_imu_translation_);
    meas_actual_.meas_imu_quat = base_quat_vicon_;
}

}  // namespace robot_estimation

/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implements an Extended Kalman Filter fusing IMU and Vicon data.
 */

#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include "pinocchio/spatial/explog-quaternion.hpp"
#include "yaml_cpp_catkin/yaml_cpp_fwd.hpp"

#include "robot_estimation/filtering_tools/ekf.hpp"

// Option to initialise to zero x,y,z and yaw from vicon at startup
#define ZERO_INITIAL_STATE

namespace robot_estimation
{
/**
 * @brief State of the EkfViconImu.
 *
 * Contains in the following order:
 * - the imu position (Vector3d [m])
 * - the imu velocity (Vector3d [m/s])
 * - the imu orientation (Vector3d 'log3(Quaternion)' [-]).
 *   We work on the tangent space of SO3.
 * - the accelerometer bias (Vector3d [m/s^2])
 * - the gyroscope bias (Vector3d [rad/s])
 */
class ViconIMUState
{
public:
    /** @brief Construct a new ViconIMUState object */
    ViconIMUState()
    {
        imu_pos.setZero();
        imu_vel.setZero();
        imu_quat.setIdentity();
        accel_bias.setZero();
        gyro_bias.setZero();
    }

    /** @brief Destroy the ViconIMUState object */
    ~ViconIMUState()
    {
    }

    /**
     * @brief State dimension:
     * - imu position, dim 3
     * - imu velocity, dim 3
     * - imu orientation log(Quaternion). dim 3
     * - imu acclerometer bias, dim 3
     * - imu gyroscope bias, dim 3
     */
    static const int state_dim = 15;

    /**
     * @brief Noise dimension, we assume that the position and orientation are
     * without noise:
     * - position, dim 0
     * - position, dim 0
     */
    static const int noise_dim =
        12; /*!< [ pos(0) acc ang_vel acc_bias gyro_bias ] */

    ViconIMUState operator+(
        const Eigen::Matrix<double, state_dim, 1>& rhs) const
    {
        ViconIMUState s;
        s.imu_pos = this->imu_pos + rhs.segment(0, 3);
        s.imu_vel = this->imu_vel + rhs.segment(3, 3);
        s.imu_quat =
            pinocchio::quaternion::exp3(rhs.segment(6, 3)) * this->imu_quat;
        s.imu_quat.normalize();
        s.accel_bias = this->accel_bias + rhs.segment(9, 3);
        s.gyro_bias = this->gyro_bias + rhs.segment(12, 3);
        return s;
    }

    Eigen::Matrix<double, state_dim, 1> operator-(
        const ViconIMUState& rhs) const
    {
        Eigen::Matrix<double, state_dim, 1> tmp;
        tmp.segment(0, 3) = this->imu_pos - rhs.imu_pos;
        tmp.segment(3, 3) = this->imu_vel - rhs.imu_vel;
        tmp.segment(6, 3) = pinocchio::quaternion::log3(this->imu_quat *
                                                        rhs.imu_quat.inverse());
        tmp.segment(9, 3) = this->accel_bias - rhs.accel_bias;
        tmp.segment(12, 3) = this->gyro_bias - rhs.gyro_bias;
        return tmp;
    }

    ViconIMUState& operator=(const ViconIMUState& rhs)
    {
        this->imu_pos = rhs.imu_pos;
        this->imu_vel = rhs.imu_vel;
        this->imu_quat = rhs.imu_quat;
        this->accel_bias = rhs.accel_bias;
        this->gyro_bias = rhs.gyro_bias;
        return *this;
    }

    ViconIMUState& operator=(const Eigen::Matrix<double, state_dim, 1> rhs)
    {
        this->imu_pos = rhs.segment(0, 3);
        this->imu_vel = rhs.segment(3, 3);
        this->imu_quat = pinocchio::quaternion::exp3(rhs.segment(6, 3));
        this->accel_bias = rhs.segment(9, 3);
        this->gyro_bias = rhs.segment(12, 3);
        return *this;
    }

    Eigen::Matrix<double, state_dim, 1> getState(void)
    {
        Eigen::Matrix<double, state_dim, 1> tmp;
        tmp.segment(0, 3) = imu_pos;
        tmp.segment(3, 3) = imu_vel;
        tmp.segment(6, 3) = pinocchio::quaternion::log3(imu_quat);
        tmp.segment(9, 3) = accel_bias;
        tmp.segment(12, 3) = gyro_bias;
        return tmp;
    }

    friend std::ostream& operator<<(std::ostream& out, const ViconIMUState& s)
    {
        out << "IMU Position:" << std::endl;
        out << s.imu_pos << std::endl << std::endl;
        out << "IMU Quaternion:" << std::endl;
        out << "  x: " << s.imu_quat.x() << ", y: " << s.imu_quat.y()
            << ", z: " << s.imu_quat.z() << ", w: " << s.imu_quat.w()
            << std::endl
            << std::endl;
        out << "IMU Linear Velocity:" << std::endl;
        out << s.imu_vel << std::endl << std::endl;
        out << "Accelerometer Bias:" << std::endl;
        out << s.accel_bias << std::endl << std::endl;
        out << "Gyroscope Bias" << std::endl;
        out << s.gyro_bias << std::endl << std::endl;
    }

    Eigen::Vector3d imu_pos;
    Eigen::Vector3d imu_vel;
    Eigen::Quaterniond imu_quat;
    Eigen::Vector3d accel_bias;
    Eigen::Vector3d gyro_bias;
    Eigen::MatrixXd state_cov;
};

class ViconIMUMeasure
{
public:
    ViconIMUMeasure()
    {
        meas_imu_pos.setZero();
        meas_imu_quat.setIdentity();
    }

    ~ViconIMUMeasure()
    {
    }

    static const int meas_dim = 6;

    ViconIMUMeasure operator+(
        const Eigen::Matrix<double, meas_dim, 1>& rhs) const
    {
        ViconIMUMeasure s;
        s.meas_imu_pos = this->meas_imu_pos + rhs.segment(0, 3);
        s.meas_imu_quat = pinocchio::quaternion::exp3(rhs.segment(3, 3)) *
                          this->meas_imu_quat;
        return s;
    }

    Eigen::Matrix<double, meas_dim, 1> operator+(
        const ViconIMUMeasure& rhs) const
    {
        Eigen::Matrix<double, meas_dim, 1> tmp;
        tmp.segment(0, 3) = this->meas_imu_pos + rhs.meas_imu_pos;
        tmp.segment(3, 3) = pinocchio::quaternion::log3(this->meas_imu_quat *
                                                        rhs.meas_imu_quat);
        return tmp;
    }

    Eigen::Matrix<double, meas_dim, 1> operator-(
        const ViconIMUMeasure& rhs) const
    {
        Eigen::Matrix<double, meas_dim, 1> tmp;
        tmp.segment(0, 3) = this->meas_imu_pos - rhs.meas_imu_pos;
        tmp.segment(3, 3) = pinocchio::quaternion::log3(
            this->meas_imu_quat * rhs.meas_imu_quat.inverse());
        return tmp;
    }

    ViconIMUMeasure& operator=(const ViconIMUMeasure& rhs)
    {
        this->meas_imu_pos = rhs.meas_imu_pos;
        this->meas_imu_quat = rhs.meas_imu_quat;
        return *this;
    }

    ViconIMUMeasure& operator=(const Eigen::Matrix<double, meas_dim, 1>& rhs)
    {
        this->meas_imu_pos = rhs.segment(0, 3);
        this->meas_imu_quat = pinocchio::quaternion::exp3(rhs.segment(3, 3));
        return *this;
    }

    Eigen::Matrix<double, meas_dim, 1> getMeas(void)
    {
        Eigen::Matrix<double, meas_dim, 1> tmp;
        tmp.segment(0, 3) = meas_imu_pos;
        tmp.segment(3, 3) = pinocchio::quaternion::log3(meas_imu_quat);
        return tmp;
    }

    friend std::ostream& operator<<(std::ostream& out, const ViconIMUMeasure m)
    {
        out << "IMU Position Meas:" << std::endl;
        out << m.meas_imu_pos.transpose() << std::endl << std::endl;
        out << "IMU Quaternion Meas:" << std::endl;
        out << "x:" << m.meas_imu_quat.x() << " ";
        out << "y:" << m.meas_imu_quat.y() << " ";
        out << "z:" << m.meas_imu_quat.z() << " ";
        out << "w:" << m.meas_imu_quat.w() << " ";
        out << std::endl << std::endl;
    }

    Eigen::Vector3d meas_imu_pos;
    Eigen::Quaterniond meas_imu_quat;

    Eigen::MatrixXd meas_cov;
};

class EkfViconImu : public EKF<ViconIMUState, ViconIMUMeasure>
{
public:
    EkfViconImu(double dt, const YAML::Node& config);

    ~EkfViconImu()
    {
    }

    void initialize(const Eigen::Ref<const Eigen::Matrix4d>& base_pose_data);
    void initialize(const Eigen::Ref<const Eigen::Vector3d>& base_pose,
                    const Eigen::Ref<const Eigen::Matrix3d>& base_ori_mat);
    void initialize(const Eigen::Ref<const Eigen::Vector3d>& base_pose,
                    const Eigen::Quaterniond& base_quat);

    void update(const Eigen::Vector3d& accelerometer,
                const Eigen::Vector3d& gyroscope,
                const Eigen::Ref<const Eigen::Matrix4d>& base_pose_data,
                const bool is_new_frame);

    void update(const Eigen::Vector3d& accelerometer,
                const Eigen::Vector3d& gyroscope,
                const Eigen::Ref<const Eigen::Vector3d>& base_pose,
                const Eigen::Quaterniond& base_quat,
                const bool is_new_frame);

    Eigen::Matrix<double, ViconIMUState::state_dim, 1> getFilterState(void)
    {
        return state_post_.getState();
    }

    void getFilterState(Eigen::Vector3d& imu_pos,
                        Eigen::Quaterniond& imu_quat,
                        Eigen::Vector3d& imu_vel,
                        Eigen::Vector3d& accel_bias,
                        Eigen::Vector3d& gyro_bias)
    {
        imu_pos = state_post_.imu_pos;
        imu_vel = state_post_.imu_vel;
        imu_quat = state_post_.imu_quat;
        accel_bias = state_post_.accel_bias;
        gyro_bias = state_post_.gyro_bias;
    }

    ///
    /// \brief getBaseFromFilterState extract the base information from the imu
    /// state
    /// \param base_position: the base pose (3D vector)
    /// \param base_orientation: the base orientation (Quaternion)
    /// \param base_linear_velocity: the base linear velocity (3D vector)
    /// \param base_angular_velocity: the base angular velocity (3D vector)
    ///
    void getBaseFromFilterState(
        Eigen::Ref<Eigen::Vector3d> output_base_position,
        Eigen::Quaterniond& output_base_orientation,
        Eigen::Ref<Eigen::Vector3d> output_base_linear_velocity,
        Eigen::Ref<Eigen::Vector3d> output_base_angular_velocity)
    {
        Eigen::Matrix3d world_R_base = state_post_.imu_quat.toRotationMatrix();
        output_base_position =
            state_post_.imu_pos - world_R_base * base_to_imu_translation_;
        output_base_orientation = state_post_.imu_quat;
        output_base_orientation.normalize();
        output_base_angular_velocity = imu_angvel_ - state_post_.gyro_bias;
        output_base_linear_velocity =
            state_post_.imu_vel +
            base_to_imu_translation_.cross(output_base_angular_velocity);
    }

    double getFloorHeight(void)
    {
        return -initial_base_pos_vicon_(2);
    }

private:
    Eigen::Matrix<double, ViconIMUState::state_dim, 1> processModel(
        ViconIMUState& s);
    Eigen::Matrix<double, ViconIMUMeasure::meas_dim, 1> measModel(
        ViconIMUState& s);

    void formProcessJacobian(void);
    void formProcessNoise(void);
    void formNoiseJacobian(void);
    void formMeasJacobian(void);
    void formMeasNoise(void);
    void formActualMeas(void);

    std::map<std::string, int> object_ids_,
        marker_ids_;  // markers unused for now

    bool update_on_;
    bool is_initialized_;
    bool use_biases_;

    int is_new_frame_;
    int frame_quality_;

    double init_var_;
    double q_proc_accel_, q_proc_gyro_, q_proc_accel_bias_, q_proc_gyro_bias_;
    double q_meas_base_pos_, q_meas_base_quat_;
    double q_meas_base_pos_init_, q_meas_base_quat_init_;
    double q_meas_base_weight_;

    Eigen::Vector3d imu_accel_, imu_angvel_;
    Eigen::Vector3d initial_base_pos_vicon_, base_pos_vicon_;
    Eigen::Quaterniond initial_base_quat_vicon_, base_quat_vicon_;

    Eigen::Vector3d grav_vec_;
    Eigen::Vector3d base_to_imu_translation_;
    YAML::Node config_;
};

}  // namespace robot_estimation

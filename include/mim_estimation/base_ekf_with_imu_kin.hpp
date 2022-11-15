/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2021, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Defines a class implementing a base state estimator using
 * the imu and the kinematics information.
 */

#include <Eigen/StdVector>

#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

namespace mim_estimation
{
/** @brief System state data. */
struct SystemState
{
    /** @brief State dimension. */
    static const int state_dim = 15;

    /** @brief Process noise dimension. */
    static const int noise_dim = 12;

    /** @brief Base/Imu position. */
    Eigen::Vector3d position = Eigen::Vector3d::Zero();

    /** @brief Base/Imu linear velocity. */
    Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero();

    /** @brief Base/Imu linear attitude. */
    Eigen::Quaterniond attitude = Eigen::Quaterniond::Identity();

    /** @brief Accelerometer bias. */
    Eigen::Vector3d bias_accelerometer = Eigen::Vector3d::Zero();

    /** @brief Gyroscope bias. */
    Eigen::Vector3d bias_gyroscope = Eigen::Vector3d::Zero();

    /** @brief Process covariance. */
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(state_dim, state_dim);
};

struct BaseEkfWithImuKinSettings
{
    /** @brief Is the EKF estimating the imu or the base frame. */
    bool is_imu_frame = true;

    /** @brief Potential contact end effector frame names. */
    std::vector<std::string> end_effector_frame_names;

    /** @brief Rigid body dynamic model. */
    pinocchio::Model pinocchio_model;

    /** @brief Imu position and orientation in the base frame. */
    pinocchio::SE3 imu_in_base = pinocchio::SE3::Identity();

    /** @brief discretization time. */
    double dt = 0.001;

    /** @brief Accelerometer noise covariance. */
    Eigen::Vector3d noise_accelerometer =
        0.0001962 * 0.0001962 * (Eigen::Vector3d() << dt, dt, dt).finished();

    /** @brief Gyroscope noise covariance. */
    Eigen::Vector3d noise_gyroscope =
        0.0000873 * 0.0000873 * (Eigen::Vector3d() << dt, dt, dt).finished();

    /** @brief Accelerometer bias noise covariance. */
    Eigen::Vector3d noise_bias_accelerometer =
        0.0001 * 0.0001 * (Eigen::Vector3d() << 1.0, 1.0, 1.0).finished();

    /** @brief Gyroscope bias noise covariance. */
    Eigen::Vector3d noise_bias_gyroscope =
        0.000309 * 0.000309 * (Eigen::Vector3d() << 1.0, 1.0, 1.0).finished();

    /** @brief Continuous measurement noise covariance. */
    Eigen::Vector3d meas_noise_cov =
        (Eigen::Vector3d() << 1e-5, 1e-5, 1e-5).finished();

    virtual std::string to_string()
    {
        std::ostringstream oss;
        oss << "The state is expressed in the imu frame ("
            << (is_imu_frame ? "true" : "false") << ") or in the base frame ("
            << (!is_imu_frame ? "true" : "false") << ")" << std::endl
            << "End-effector frame names are = [";
        for (std::size_t i = 0; i < end_effector_frame_names.size(); ++i)
        {
            oss << end_effector_frame_names[i] << " ";
        }
        oss << "]" << std::endl
            << "The pinocchio model:\n"
            << pinocchio_model << std::endl
            << "The SE3 position of the imu in the base frame:\n"
            << imu_in_base << std::endl
            << "The control period (dt) = " << dt << std::endl
            << "The noise of the accelerometer = "
            << noise_accelerometer.transpose() << std::endl
            << "The noise of the gyroscope = " << noise_gyroscope.transpose()
            << std::endl
            << "The noise of the accelerometer bias = "
            << noise_bias_accelerometer.transpose() << std::endl
            << "The noise of the gyroscope bias = "
            << noise_bias_gyroscope.transpose() << std::endl
            << "The measurement noise for all end-effector = "
            << meas_noise_cov.transpose() << std::endl;
        return oss.str();
    }
};

/**
 * @brief Extended Kalman Filter implementation for base state estimation using
 * the imu and the kinematics information.
 *
 * Details:
 *
 * @todo explain here the math around the EKF
 *
 */
class BaseEkfWithImuKin
{
public:
    /** @brief Construct a new Base Ekf With Imu and Kinematics object. */
    BaseEkfWithImuKin();

    /** @brief Destroy the Base Ekf With Imu Kin object. */
    ~BaseEkfWithImuKin();

    /**
     * @brief Get the EKF settings and initialize the filter from them.
     *
     * @param settings
     */
    void initialize(const BaseEkfWithImuKinSettings& settings);

    /**
     * @brief Set initial state.
     *
     * @param base_position Base position with respect to the world frame.
     * @param base_attitude Base orientation with respect to the world frame.
     * @param base_linear_velocity Base linear velocity with respect to the base
     * frame.
     * @param base_angular_velocity Base angular velocity with respect to the
     * base frame.
     */
    void set_initial_state(
        Eigen::Ref<const Eigen::Vector3d> base_position,
        const Eigen::Quaterniond& base_attitude,
        Eigen::Ref<const Eigen::Vector3d> base_linear_velocity,
        Eigen::Ref<const Eigen::Vector3d> base_angular_velocity);

    /**
     * @brief Set initial state.
     *
     * @param base_se3_position [XYZ Quaternion] SE3 representation of the base
     * position with respect to the world frame.
     * @param base_se3_velocity [Linear Angular] base velocity in the base frame.
     */
    void set_initial_state(
        Eigen::Ref<const Eigen::Matrix<double, 7, 1> > base_se3_position,
        Eigen::Ref<const Eigen::Matrix<double, 6, 1> > base_se3_velocity);

    /**
     * @brief Feed in the sensor raw data from the robot and update the filter
     * output.
     *
     * @param contact_schedule vector of boolean, if one boolean is true it
     *                         means that the corresponding end-effector is in
     *                         contact.
     * @param imu_accelerometer imu raw accelerometer data.
     * @param imu_gyroscope imu raw gyroscope data.
     * @param joint_position joint position data (in pinocchio ordering).
     * @param joint_velocity joint velocity data (in pinocchio ordering).
     */
    void update_filter(const std::vector<bool>& contact_schedule,
                       Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
                       Eigen::Ref<const Eigen::Vector3d> imu_gyroscope,
                       Eigen::Ref<const Eigen::VectorXd> joint_position,
                       Eigen::Ref<const Eigen::VectorXd> joint_velocity);

    /**
     * @brief Get the filter output which is the robot state.
     *
     * @param robot_configuration
     * @param robot_velocity
     */
    void get_filter_output(Eigen::Ref<Eigen::VectorXd> robot_configuration,
                           Eigen::Ref<Eigen::VectorXd> robot_velocity);

    /**
     * @brief Get measurement.
     *
     * @param root_velocities
     */
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d> >&
    get_measurement();

    /*
     * Private methods.
     */
private:
    /**
     * @brief Compute the base/imu velocities
     *
     */
    void compute_end_effectors_forward_kinematics(
        Eigen::Ref<const Eigen::VectorXd> joint_position,
        Eigen::Ref<const Eigen::VectorXd> joint_velocity);

    /**
     * @brief Integrate the process model using the imu data.
     *
     * compute mu_pre.
     */
    void integrate_process_model(
        Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
        Eigen::Ref<const Eigen::Vector3d> imu_gyroscope);

    /**
     * @brief Fill in the process jacobians (cont_proc_jac_, disc_proc_jac_).
     */
    void compute_discrete_prediction_jacobian();

    /**
     * @brief Fill in the noise jacobian.
     */
    void compute_noise_jacobian();

    /**
     * @brief Use and additive white noise as continuous noise.
     */
    void construct_continuous_noise_covariance();

    /**
     * @brief Discretise the noise covariance using zero-order hold and
     * truncating higher-order terms.
     */
    void construct_discrete_noise_covariance();

    /**
     * @brief Discretise the measurement noise covariance.
     */
    void construct_discrete_measurement_noise_covariance();

    /**
     * @brief Propagate the state covariance (predicted_state_.covariance).
     */
    void prediction_step();

    /**
     * @brief Use the kinematics to measure the linear velocity of the base.
     *
     * @param contact_schedule This indicates which end-effector is currently
     *                         in contact.
     * @param joint_position joint positions.
     * @param joint_velocity joint velocities.
     */
    void measurement_model(const std::vector<bool>& contact_schedule,
                           Eigen::Ref<const Eigen::VectorXd> joint_position,
                           Eigen::Ref<const Eigen::VectorXd> joint_velocity);

    /**
     * @brief Update the current state posterior_state_ in function of the
     * measurements.
     *
     * @param contact_schedule This indicates which end-effector is currently
     *                         in contact.
     * @param joint_position joint positions.
     * @param joint_velocity joint velocities.
     */
    void update_step(const std::vector<bool>& contact_schedule,
                     Eigen::Ref<const Eigen::VectorXd> joint_position,
                     Eigen::Ref<const Eigen::VectorXd> joint_velocity);

    /*
     * Settings.
     */
private:
    BaseEkfWithImuKinSettings settings_;

    /*
     * Internal Data.
     */
private:
    // Sensor data

    /** @brief Joint positions reading. */
    Eigen::VectorXd joint_position_;

    /** @brief Joint velocities reading. */
    Eigen::VectorXd joint_velocity_;

    /** @brief Accelerometer reading. */
    Eigen::Vector3d imu_accelerometer_;

    /** @brief Joint velocities. */
    Eigen::Vector3d imu_gyroscope_;

    // Extended Kalman filter states.

    /** @brief Predicted system state (next state). */
    SystemState predicted_state_;

    /** @brief Posterior system state (current state). */
    SystemState posterior_state_;

    // Kinematics data.

    /** @brief Rigid body dynamics data storage class. */
    pinocchio::Data pinocchio_data_;

    /** @brief Measured end-effectors positions expressed in the imu/base
     * frame.
     */
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >
        kin_ee_position_;

    /** @brief Measured end-effectors velocities expressed in the imu/base
     * frame.
     */
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >
        kin_ee_velocity_;

    /** @brief Pinocchio frame id corresponding to the end-effector potentially
     * in contact. */
    std::vector<pinocchio::FrameIndex> kin_ee_fid_;

    /** @brief Robot configuration with base at the center of the world.*/
    Eigen::VectorXd q_kin_;

    /** @brief Robot velocity with static base frame. */
    Eigen::VectorXd dq_kin_;

    // Integrate model

    /** @brief Rotation matrix from the posterior state. */
    Eigen::Matrix3d attitude_post_;

    /** @brief Imu/Base angular velocity. */
    Eigen::Vector3d root_angular_velocity_;

    /** @brief Imu/Base angular velocity previous value. */
    Eigen::Vector3d root_angular_velocity_prev_;

    /** @brief Imu/Base angular acceleration. */
    Eigen::Vector3d root_angular_acceleration_;

    /** @brief Imu/Base angular velocity. */
    Eigen::Vector3d root_linear_acceleration_;

    /** @brief Gravity vector (default is negative along the z-axis). */
    Eigen::Vector3d gravity_;

    // Jacobians and Covariances computation.

    /** @brief Continuous process jacobian. */
    Eigen::MatrixXd cont_proc_jac_;

    /** @brief Discrete process jacobian. */
    Eigen::MatrixXd disc_proc_jac_;

    /** @brief Process noise jacobian. */
    Eigen::MatrixXd proc_noise_jac_;

    /** @brief Continuous process noise covariance. */
    Eigen::MatrixXd cont_proc_noise_cov_;

    /** @brief Discrete process noise covariance. */
    Eigen::MatrixXd disc_proc_noise_cov_;

    /** @brief Discrete measurement noise covariance. */
    Eigen::MatrixXd disc_meas_noise_cov_;

    /** @brief Linear velocity of the root (base/imu) measured from the
     * kinematics. */
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >
        kin_meas_root_velocity_;

    /** @brief Measurement jacobian */
    Eigen::MatrixXd meas_jac_;

    /** @brief Predicted base velocity, this is the predicted measurement. */
    Eigen::VectorXd meas_error_;

    /** @brief Measurement covariance. */
    Eigen::MatrixXd meas_covariance_;

    /** @brief Kalman Gain computed during the update_step. */
    Eigen::MatrixXd kalman_gain_;

    /** @brief LDLT decomposition to invert some matrix */
    Eigen::LDLT<Eigen::MatrixXd> ldlt;

    /** @brief Update vector which is a delta state. */
    Eigen::VectorXd delta_state_;
};

}  // namespace mim_estimation
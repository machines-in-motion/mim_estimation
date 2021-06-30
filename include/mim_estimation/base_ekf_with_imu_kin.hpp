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

    /** @brief Base/Imu position. */
    Eigen::Vector3d position = Eigen::Vector3d::Zero();

    /** @brief Base/Imu linear velocity. */
    Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero();

    /** @brief Base/Imu linear attitude. */
    Eigen::Quaterniond attitude = Eigen::Quaterniond::Identity();

    /** @brief Base/Imu acceleration bias. */
    Eigen::Vector3d bias_acceleration = Eigen::Vector3d::Zero();

    /** @brief Base/Imu attitude bias. */
    Eigen::Vector3d bias_attitude = Eigen::Vector3d::Zero();

    /** @brief Process covariance. */
    Eigen::MatrixXd sigma = Eigen::MatrixXd::Zero(state_dim, state_dim);
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
     * @param robot_configuration is the robot position in generalized
     *                            coordinates.
     * @param robot_velocity is the robot velocity in generalized
     *                       coordinates.
     */
    void set_initial_state(const Eigen::Vector3d& base_position,
                           const Eigen::Quaterniond& base_attitude,
                           const Eigen::Vector3d& base_linear_velocity,
                           const Eigen::Vector3d& base_angular_velocity);

    /**
     * @brief Feed in the sensor raw data from the robot and update the filter
     * output.
     *
     * @param contact_schedule vector of boolean, if one boolean is true it
     *                         means that the corresding end-effector is in
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
     * @brief Propagate the state covariance (mu_pre_.covariance).
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

    /*
     * Settings.
     */
private:
    BaseEkfWithImuKinSettings settings_;

    /*
     * Internal Data.
     */
private:
    /** @brief Rigid body dynamics data storage class. */
    pinocchio::Data pinocchio_data_;

    /** @brief Predicted system state (next state). */
    SystemState mu_pre_;

    /** @brief Posterior system state (current state). */
    SystemState mu_post_;

    /** @brief Gravity vector (default is negative along the z-axis). */
    Eigen::Vector3d gravity_;

    /** @brief Linear velocity of the root (base/imu) computed from the
     * kinematics. */
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >
        root_velocities_per_end_effector_;

    /** @brief Pinocchio frame id corresponding to the end-effector potentially
     * in contact. */
    std::vector<pinocchio::FrameIndex> contact_frame_id_;

    // /** @brief Continuous process jacobian. */
    // Eigen::MatrixXd cont_proc_jac_;

    // /** @brief Discrete process jacobian. */
    // Eigen::MatrixXd disc_proc_jac_;

    // /** @brief Process noise jacobian. */
    // Eigen::MatrixXd proc_noise_jac_;

    // Eigen::MatrixXd proc_noise_jac_;
};

}  // namespace mim_estimation
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

#include <sstream>

#include "mim_estimation/base_ekf_with_imu_kin.hpp"
#include "mim_estimation/end_effector_force_estimator.hpp"

namespace mim_estimation
{
struct RobotStateEstimatorSettings : public BaseEkfWithImuKinSettings
{
    /* BaseEkfWithImuKin settings (inherited) */

    /* EndEffectorForceEstimator settings. */

    /** @brief Path to th robot URDF file. */
    std::string urdf_path;

    /* Contact detection. */

    /** @brief Threshold on the rising force norm. */
    double force_threshold_up = 10;

    /** @brief Threshold on the decreasing force norm. */
    double force_threshold_down = 5;

    /* Public methods. */

    /** @brief Convert the current object in human readable string. */
    virtual std::string to_string()
    {
        std::ostringstream oss;
        oss << "The urdf path is: " << urdf_path << std::endl;
        oss << BaseEkfWithImuKinSettings::to_string();
        return oss.str();
    }
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
    typedef std::map<
        std::string,
        Eigen::Vector3d,
        std::less<std::string>,
        Eigen::aligned_allocator<std::pair<const std::string, Eigen::Vector3d>>>
        ForceInWorldMap;

public:
    /** @brief Construct a new Base State Estimator object. */
    RobotStateEstimator();

    /** @brief Destroy by default the Base State Estimator object. */
    ~RobotStateEstimator();

    void initialize(const RobotStateEstimatorSettings& settings);

    void set_initial_state(
        Eigen::Ref<const Eigen::VectorXd> initial_robot_configuration,
        Eigen::Ref<const Eigen::VectorXd> initial_robot_velocity);

    /**
     * @brief Estimate the robot base velocity and postion assuming the contact
     * schedule is known.
     * 
     * @param contact_schedule 
     * @param imu_accelerometer 
     * @param imu_gyroscope 
     * @param joint_position 
     * @param joint_velocity 
     * @param joint_torque 
     */
     void run(Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
         Eigen::Ref<const Eigen::Vector3d> imu_gyroscope);

    void run(const std::vector<bool>& contact_schedule,
         Eigen::Ref<const Eigen::VectorXd> joint_position,
         Eigen::Ref<const Eigen::VectorXd> joint_velocity);

    void compute_midline(const std::vector<bool>& contact_schedule,
                     Eigen::Ref<const Eigen::VectorXd> joint_position,
                     Eigen::Ref<const Eigen::VectorXd> joint_velocity);
                     
    void run(const std::vector<bool>& contact_schedule,
             Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
             Eigen::Ref<const Eigen::Vector3d> imu_gyroscope,
             Eigen::Ref<const Eigen::VectorXd> joint_position,
             Eigen::Ref<const Eigen::VectorXd> joint_velocity);
    
    /**
     * @brief Estimate the robot base velocity and postion and the contact
     * states.
     * 
     * @param imu_accelerometer 
     * @param imu_gyroscope 
     * @param joint_position 
     * @param joint_velocity 
     * @param joint_torque 
     */
    void run(Eigen::Ref<const Eigen::Vector3d> imu_accelerometer,
             Eigen::Ref<const Eigen::Vector3d> imu_gyroscope,
             Eigen::Ref<const Eigen::VectorXd> joint_position,
             Eigen::Ref<const Eigen::VectorXd> joint_velocity,
             Eigen::Ref<const Eigen::VectorXd> joint_torque);

    void set_settings(const RobotStateEstimatorSettings& settings);

    void get_state(Eigen::Ref<Eigen::VectorXd> robot_configuration,
                   Eigen::Ref<Eigen::VectorXd> robot_velocity);

    const Eigen::VectorXd& get_robot_configuration() const;

    const Eigen::VectorXd& get_robot_velocity() const;

    const std::vector<bool>& get_detected_contact() const;

    const Eigen::Vector3d& get_force(const std::string& frame_name);

    const RobotStateEstimatorSettings& get_settings() const;

private:
    /** @brief End-effector force estimator. Estimate the forces in the base
     * frame. */
    EndEffectorForceEstimator ee_force_estimator_;

    /** @brief EKF that estimating the localization of the base relative to it's
     * initial position. */
    BaseEkfWithImuKin base_ekf_with_imu_kin_;

    /** @brief Settings of this class. */
    RobotStateEstimatorSettings settings_;

    /** @brief Contact detection from force. */
    std::vector<bool> detected_contact_;

    /** @brief Current robot configuration. */
    Eigen::VectorXd current_robot_configuration_;

    /** @brief Current robot velocity. */
    Eigen::VectorXd current_robot_velocity_;

    /** @brief Current base orientation in the estimated world frame. */
    Eigen::Quaterniond rot_base_in_world_;

    /** @brief Force in world frame. */
    ForceInWorldMap force_in_world_map_;
};

}  // namespace mim_estimation

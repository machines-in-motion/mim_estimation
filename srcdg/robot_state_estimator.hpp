/**
 * @file vicon_entity.hh
 * @author Maximilien Naveau (maximilien.naveau@gmail.com)
 * @brief
 * @version 0.1
 * @date 2018-12-18
 *
 * @copyright Copyright (c) 2018
 *
 */

#pragma once

#include <dynamic-graph/all-commands.h>
#include <dynamic-graph/all-signals.h>
#include <dynamic-graph/entity.h>
#include <dynamic-graph/factory.h>
#include <dynamic-graph/linear-algebra.h>

#include <deque>

#include "mim_estimation/robot_state_estimator.hpp"

namespace mim_estimation
{
namespace dynamic_graph
{
/** @brief Simple shortcut for code writing convenience. */
typedef dynamicgraph::SignalTimeDependent<int, int> SignalTrigger;

/** @brief Simple shortcut for code writing convenience. */
typedef dynamicgraph::SignalTimeDependent<dynamicgraph::Vector, int> SignalOut;

/** @brief Simple shortcut for code writing convenience. */
typedef dynamicgraph::SignalPtr<dynamicgraph::Vector, int> SignalIn;

/**
 * @brief This class define a dynamic graph wrapper around the vicon client
 */
class RobotStateEstimator : public dynamicgraph::Entity
{
public:
    DYNAMIC_GRAPH_ENTITY_DECL();

    /**
     * @brief Construct a new Vicon Client Entity object.
     *
     * @param name Entity name.
     */
    RobotStateEstimator(const std::string& name);

    /**
     * @brief Destroy the Vicon Client Entity object.
     */
    ~RobotStateEstimator();

    /**
     * @brief Connect to the vicon system and start a
     *
     * @param host_name
     */
    void initialize(const RobotStateEstimatorSettings& settings);

    /**
     * @brief Add a signal that contains the pose of the desired object.
     */
    void set_initial_state(
        Eigen::Ref<const Eigen::VectorXd> initial_robot_configuration,
        Eigen::Ref<const Eigen::VectorXd> initial_robot_velocity);

    void set_settings(const RobotStateEstimatorSettings& settings);

private:
    /**
     * @brief Signal callback for the one_iteration_sout_ signal.
     */
    int& signal_callback_one_iteration(int& /*not used*/, const int& time);

    /**
     * @brief Signal callback for the robot_configuration_sout_ signal.
     */
    dynamicgraph::Vector& signal_callback_robot_configuration(
        dynamicgraph::Vector& res, const int& time);

    /**
     * @brief Signal callback for the robot_velocity_sout_ signal.
     */
    dynamicgraph::Vector& signal_callback_robot_velocity(
        dynamicgraph::Vector& res, const int& time);

    /**
     * @brief Signal callback for the base_posture_sout_ signal.
     */
    dynamicgraph::Vector& signal_callback_base_posture(
        dynamicgraph::Vector& res, const int& time);

    /**
     * @brief Signal callback for the base_velocity_body_sout_ signal.
     */
    dynamicgraph::Vector& signal_callback_base_velocity_body(
        dynamicgraph::Vector& res, const int& time);

    /**
     * @brief Signal callback for the detected_contact_sout_ signal.
     */
    dynamicgraph::Vector& signal_callback_detected_contact(
        dynamicgraph::Vector& res, const int& time);

    /**
     * @brief Signal callback for the robot_velocity_sout_ signal.
     */
    dynamicgraph::Vector& signal_callback_force(const std::string& frame_name,
                                                dynamicgraph::Vector& res,
                                                const int& time);

private:
    /** @brief Sensor reading input signal: imu accelerometer. */
    SignalIn imu_accelerometer_sin_;

    /** @brief Sensor reading input signal:  imu gyroscope. */
    SignalIn imu_gyroscope_sin_;

    /** @brief Sensor reading input signal:  joint position. */
    SignalIn joint_position_sin_;

    /** @brief Sensor reading input signal:  joint velocity. */
    SignalIn joint_velocity_sin_;

    /** @brief Sensor reading input signal:  joint torque. */
    SignalIn joint_torque_sin_;

    /**
     * @brief Create an output signal which computes the estimation. All other
     * signal depend one this one. This allow the computation to be ran only
     * once per graph evaluation. And to compute the estimation early on to
     * allow convergence of the algorithm during initialization.
     */
    SignalTrigger one_iteration_sout_;

    /**
     * @brief Output signal carrying the robot configuration in generalized
     * coordinates.
     */
    SignalOut robot_configuration_sout_;

    /**
     * @brief Output signal carrying the robot velocity in generalized
     * coordinates.
     */
    SignalOut robot_velocity_sout_;

    /**
     * @brief Output signal carrying the robot base SE3 position.
     */
    SignalOut base_posture_sout_;

    /**
     * @brief Output signal carrying the robot base SE3 velocity.
     */
    SignalOut base_velocity_body_sout_;

    /**
     * @brief Output signal carrying the list of detected contact using doubles.
     */
    SignalOut detected_contact_sout_;

    /**
     * @brief Signals for the end-effector external force expressed in the world
     * frame.
     */
    std::deque<std::unique_ptr<SignalOut> > force_sout_;
    int flag;
    int sum_cnt_pre;
    int sum_cnt_post;
    /** @brief State estimator wrapped by this entity. */
    mim_estimation::RobotStateEstimator estimator_;
};

}  // namespace dynamic_graph
}  // namespace mim_estimation

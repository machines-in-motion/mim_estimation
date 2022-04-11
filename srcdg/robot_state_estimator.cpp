/**
 * @file vicon_client_entity.cpp
 * @author Maximilien Naveau (maximilien.naveau@gmail.com)
 * @brief
 * @version 0.1
 * @date 2018-12-18
 *
 * @copyright Copyright (c) 2018
 *
 */

#include "robot_state_estimator.hpp"

#include <iostream>
#include <numeric>

#include "signal_utils.hpp"

namespace mim_estimation
{
namespace dynamic_graph
{
DYNAMICGRAPH_FACTORY_ENTITY_PLUGIN(RobotStateEstimator, "RobotStateEstimator");

RobotStateEstimator::RobotStateEstimator(const std::string& name)
    : Entity(name),
      // Input signals
      define_input_signal(imu_accelerometer_sin_, "vector(3)"),
      define_input_signal(imu_gyroscope_sin_, "vector(3)"),
      define_input_signal(joint_position_sin_, "vector"),
      define_input_signal(joint_velocity_sin_, "vector"),
      define_input_signal(joint_torque_sin_, "vector"),
      // Output signals
      define_output_signal(one_iteration_sout_,
                           "int",
                           imu_accelerometer_sin_
                               << imu_gyroscope_sin_ << joint_position_sin_
                               << joint_velocity_sin_ << joint_torque_sin_,
                           &RobotStateEstimator::signal_callback_one_iteration),
      define_output_signal(
          robot_configuration_sout_,
          "vector",
          one_iteration_sout_,
          &RobotStateEstimator::signal_callback_robot_configuration),
      define_output_signal(
          robot_velocity_sout_,
          "vector",
          one_iteration_sout_,
          &RobotStateEstimator::signal_callback_robot_velocity),
      define_output_signal(base_posture_sout_,
                           "vector",
                           one_iteration_sout_,
                           &RobotStateEstimator::signal_callback_base_posture),
      define_output_signal(
          base_velocity_body_sout_,
          "vector",
          one_iteration_sout_,
          &RobotStateEstimator::signal_callback_base_velocity_body),
      define_output_signal(
          detected_contact_sout_,
          "vector(int)",
          one_iteration_sout_,
          &RobotStateEstimator::signal_callback_detected_contact)
{
    // Register input signals.
    signalRegistration(imu_accelerometer_sin_);
    signalRegistration(imu_gyroscope_sin_);
    signalRegistration(joint_position_sin_);
    signalRegistration(joint_velocity_sin_);
    signalRegistration(joint_torque_sin_);
    // Register output signals.
    signalRegistration(one_iteration_sout_);
    signalRegistration(robot_configuration_sout_);
    signalRegistration(robot_velocity_sout_);
    signalRegistration(base_posture_sout_);
    signalRegistration(base_velocity_body_sout_);
    signalRegistration(detected_contact_sout_);

    // Commands: the commands will be binded with boost::python directly.
}

RobotStateEstimator::~RobotStateEstimator()
{
}

void RobotStateEstimator::initialize(
    const RobotStateEstimatorSettings& settings)
{
    // initialize the internal estimator.
    estimator_.initialize(settings);

    // add and register the end-effector force signals.
    for (std::size_t i = 0; i < settings.end_effector_frame_names.size(); ++i)
    {
        // Compute the signal name.
        std::ostringstream oss;
        const std::string& frame_name = settings.end_effector_frame_names[i];
        oss << "RobotStateEstimator(" << name
            << ")::output(vector)::" << frame_name << "_force_sout";
        std::string signal_name = oss.str();

        // Get the callback function.
        auto callback_func =
            boost::bind(&RobotStateEstimator::signal_callback_force,
                        this,
                        frame_name,
                        _1,
                        _2);  // using auto here to get the type...

        // create the signal
        std::unique_ptr<SignalOut> signal = std::make_unique<SignalOut>(
            callback_func, one_iteration_sout_, signal_name);

        // Register signal and save the pointer.
        signalRegistration(*signal);
        force_sout_.emplace_back(std::move(signal));
    }
}

void RobotStateEstimator::set_initial_state(
    Eigen::Ref<const Eigen::VectorXd> initial_robot_configuration,
    Eigen::Ref<const Eigen::VectorXd> initial_robot_velocity)
{
    estimator_.set_initial_state(initial_robot_configuration,
                                 initial_robot_velocity);
    flag = -1;
}

void RobotStateEstimator::set_settings(
    const RobotStateEstimatorSettings& settings)
{
    estimator_.set_settings(settings);
}

int& RobotStateEstimator::signal_callback_one_iteration(int& res,
                                                        const int& time)
{
    const dynamicgraph::Vector& imu_accelerometer =
        imu_accelerometer_sin_.access(time);
    const dynamicgraph::Vector& imu_gyroscope = imu_gyroscope_sin_.access(time);
    const dynamicgraph::Vector& joint_position =
        joint_position_sin_.access(time);
    const dynamicgraph::Vector& joint_velocity =
        joint_velocity_sin_.access(time);
    const dynamicgraph::Vector& joint_torque = joint_torque_sin_.access(time);

    estimator_.run(imu_accelerometer,
                   imu_gyroscope,
                   joint_position,
                   joint_velocity,
                   joint_torque);

    // one_iteration_sout_.access(time);
    std::vector<bool> detected_contact = estimator_.get_detected_contact();

    if (flag == -1){
        flag = time-1;
        sum_cnt_pre = 0;
        std::cout << "#####" << flag << "#####" <<std::endl; 
    }
    
    // if ((time > flag+6500) & (time < flag+35000))
    // {
    //     int sum_cnt =  std::accumulate(detected_contact.begin(), detected_contact.end(), 0);
    //     if (sum_cnt == 2)
    //     {
    //         sum_cnt_post = 2;
    //         if (sum_cnt_post - sum_cnt_pre == 2)
    //         {
    //             estimator_.compute_midline(detected_contact,
    //                                        joint_position,
    //                                        joint_velocity);
    //             sum_cnt_pre = sum_cnt_post;
    //         }

    //     }
    //     if (sum_cnt == 0)
    //     {
    //         sum_cnt_post = 0;
    //         sum_cnt_pre = 0;
    //     }
    // }

    if (time > flag+35000)
    {
        std::vector<bool> all_contact = { 1, 1, 1, 1 };
        estimator_.run(all_contact,
                   joint_position,
                   joint_velocity);
    } 
    else 
    {
        estimator_.run(detected_contact,
                   joint_position,
                   joint_velocity);
    }
    res = 1.0;
    return res;
}

dynamicgraph::Vector& RobotStateEstimator::signal_callback_robot_configuration(
    dynamicgraph::Vector& res, const int& time)
{
    one_iteration_sout_.access(time);
    res = estimator_.get_robot_configuration();
    return res;
}

dynamicgraph::Vector& RobotStateEstimator::signal_callback_robot_velocity(
    dynamicgraph::Vector& res, const int& time)
{
    one_iteration_sout_.access(time);
    res = estimator_.get_robot_velocity();
    return res;
}

dynamicgraph::Vector& RobotStateEstimator::signal_callback_base_posture(
    dynamicgraph::Vector& res, const int& time)
{
    one_iteration_sout_.access(time);
    res = estimator_.get_robot_configuration().head<7>();
    return res;
}

dynamicgraph::Vector& RobotStateEstimator::signal_callback_base_velocity_body(
    dynamicgraph::Vector& res, const int& time)
{
    one_iteration_sout_.access(time);
    res = estimator_.get_robot_velocity().head<6>();
    return res;
}

dynamicgraph::Vector& RobotStateEstimator::signal_callback_detected_contact(
    dynamicgraph::Vector& res, const int& time)
{
    one_iteration_sout_.access(time);
    std::vector<bool> detected_contact = estimator_.get_detected_contact();
    if (static_cast<std::size_t>(res.size()) != detected_contact.size())
    {
        res.resize(detected_contact.size());
        res.fill(0.0);
    }
    for (std::size_t i = 0; i < detected_contact.size(); ++i)
    {
        res(i) = detected_contact[i] ? 1.0 : 0.0;
    }
    return res;
}

dynamicgraph::Vector& RobotStateEstimator::signal_callback_force(
    const std::string& frame_name, dynamicgraph::Vector& res, const int& time)
{
    one_iteration_sout_.access(time);
    res = estimator_.get_force(frame_name);
    return res;
}

}  // namespace dynamic_graph
}  // namespace mim_estimation

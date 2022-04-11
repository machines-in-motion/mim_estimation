/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Python bindings for the StepperHead class
 */
// clang-format off
#include "pinocchio/bindings/python/fwd.hpp"
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "mim_estimation/robot_state_estimator.hpp"
// clang-format on

using namespace boost::python;

namespace mim_estimation
{
void bind_robot_state_estimator_settings()
{
    class_<RobotStateEstimatorSettings, bases<BaseEkfWithImuKinSettings>>(
        "RobotStateEstimatorSettings")
        .def_readwrite("urdf_path", &RobotStateEstimatorSettings::urdf_path)
        .def_readwrite("force_threshold_up",
                       &RobotStateEstimatorSettings::force_threshold_up)
        .def_readwrite("force_threshold_down",
                       &RobotStateEstimatorSettings::force_threshold_down)
        .def("__repr__", &RobotStateEstimatorSettings::to_string);
}

void bind_robot_state_estimator()
{
    bind_robot_state_estimator_settings();

    class_<RobotStateEstimator>("RobotStateEstimator")
        // Public methods.
        .def("initialize",
             &RobotStateEstimator::initialize,
             "Set the estimator settings.")
        .def("set_settings",
             &RobotStateEstimator::set_settings,
             "Set the estimator settings.")
        .def("set_initial_state",
             &RobotStateEstimator::set_initial_state,
             "Set the initial state using generalized coordinates.")
        .def("compute_midline",
             &RobotStateEstimator::compute_midline,
         "compute base position to midline of feet in contact, using joint positions and velocities.")
        .def("run",
             static_cast<void(
                 (RobotStateEstimator::*)(const std::vector<bool>&,
                                          Eigen::Ref<const Eigen::VectorXd>,
                                          Eigen::Ref<const Eigen::VectorXd>))>(
                 &RobotStateEstimator::run),
             "Execute the estimation (including contact) from input data.")
        .def("run",
             static_cast<void(
                 (RobotStateEstimator::*)(Eigen::Ref<const Eigen::Vector3d>,
                                          Eigen::Ref<const Eigen::Vector3d>))>(
                 &RobotStateEstimator::run),
         "Execute the estimation (prediction_step) from input data.")
        .def("run",
             static_cast<void(
                 (RobotStateEstimator::*)(Eigen::Ref<const Eigen::Vector3d>,
                                          Eigen::Ref<const Eigen::Vector3d>,
                                          Eigen::Ref<const Eigen::VectorXd>,
                                          Eigen::Ref<const Eigen::VectorXd>,
                                          Eigen::Ref<const Eigen::VectorXd>))>(
                 &RobotStateEstimator::run),
             "Execute the estimation from input data.")
        .def("get_state",
             &RobotStateEstimator::get_state,
             "Get the robot state in the generalized coordinates.")
        .def("get_detected_contact",
             make_function(&RobotStateEstimator::get_detected_contact,
                           return_value_policy<copy_const_reference>()),
             "Get the contact detection per end-effector.")
        .def("get_force",
             make_function(&RobotStateEstimator::get_force,
                           return_value_policy<copy_const_reference>()),
             "Get the end-effector force from its name.");
}

}  // namespace mim_estimation
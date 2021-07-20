/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Python bindings for the StepperHead class
 */

#include "mim_estimation/end_effector_force_estimator.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace mim_estimation
{
void bind_end_effector_force_estimator(py::module& module)
{
    py::class_<EndEffectorForceEstimator>(module, "EndEffectorForceEstimator")
        .def(py::init<>())
        .def("initialize", &EndEffectorForceEstimator::initialize)
        .def("run", &EndEffectorForceEstimator::run)
        .def("has_free_flyer", &EndEffectorForceEstimator::has_free_flyer)
        .def("add_contact_frame",
             static_cast<void(
                 (EndEffectorForceEstimator::*)(const std::string& frame_name))>(
                 &EndEffectorForceEstimator::add_contact_frame))
        .def("add_contact_frame",
             static_cast<void(
                 (EndEffectorForceEstimator::*)(const std::vector<std::string>&
                                                    frame_names))>(
                 &EndEffectorForceEstimator::add_contact_frame))
        .def("get_force", &EndEffectorForceEstimator::get_force)
        .def("__repr__", &EndEffectorForceEstimator::to_string);
}

}  // namespace mim_estimation
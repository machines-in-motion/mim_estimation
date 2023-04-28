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
#include <eigenpy/eigenpy.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "mim_estimation/end_effector_force_estimator.hpp"
// clang-format on

using namespace boost::python;

namespace mim_estimation
{
void bind_end_effector_force_estimator()
{
    class_<EndEffectorForceEstimator>("EndEffectorForceEstimator")
        .def("initialize", &EndEffectorForceEstimator::initialize)
        .def("run", &EndEffectorForceEstimator::run)
        .def("add_contact_frame",
             static_cast<void((
                 EndEffectorForceEstimator::*)(const std::string& frame_name))>(
                 &EndEffectorForceEstimator::add_contact_frame))
        .def("add_contact_frame",
             static_cast<void(
                 (EndEffectorForceEstimator::*)(const std::vector<std::string>&
                                                    frame_names))>(
                 &EndEffectorForceEstimator::add_contact_frame))
        .def("get_force",
             make_function(&EndEffectorForceEstimator::get_force,
                           return_value_policy<copy_const_reference>()))
        .def("__repr__", &EndEffectorForceEstimator::to_string);
}

}  // namespace mim_estimation

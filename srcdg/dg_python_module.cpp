/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Expose the Device and the periodic call to python.
 */

#include "dynamic-graph/python/module.hh"
#include "dynamic-graph/python/signal.hh"
#include "srcdg/robot_state_estimator.hpp"

typedef bp::return_value_policy<bp::reference_existing_object>
    reference_existing_object;

BOOST_PYTHON_MODULE(entities)
{
    boost::python::import("dynamic_graph");
    boost::python::object mim_estimation_cpp = bp::import("mim_estimation_cpp");

    bp::scope().attr("RobotStateEstimatorSettings") =
        mim_estimation_cpp.attr("RobotStateEstimatorSettings");

    using mim_estimation::dynamic_graph::RobotStateEstimator;
    dynamicgraph::python::exposeEntity<RobotStateEstimator>()
        .def(
            "initialize",
            +[](boost::python::object py_obj, boost::python::object py_settings)
            {
                // get the C++ pointer.
                RobotStateEstimator *cpp_obj =
                    boost::python::extract<RobotStateEstimator *>(py_obj);
                // Get the C++ settings.
                const mim_estimation::RobotStateEstimatorSettings
                    &cpp_settings = boost::python::extract<
                        const mim_estimation::RobotStateEstimatorSettings &>(
                        py_settings);
                // Initialize the objet with the settings.
                cpp_obj->initialize(cpp_settings);
                // Initialize the python signals.
                return dynamicgraph::python::entity::addSignals(py_obj);
            },
            "Initialize the RobotStateEstimator.")
        .def("set_initial_state",
             &RobotStateEstimator::set_initial_state,
             "Set the initial state using the generalized coordinates.")
        .def("set_settings",
             &RobotStateEstimator::set_settings,
             "Set the estimator settings.");
}

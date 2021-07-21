/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Python bindings for the StepperHead class
 */

#include "boost_python_compatibility.hpp"
#include "mim_estimation/base_ekf_with_imu_kin.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace mim_estimation
{
void set_pinocchio_model(BaseEkfWithImuKinSettings& obj, py::object py_src)
{
    obj.pinocchio_model = BoostPython::pybind_to_cpp<pinocchio::Model>(py_src);
}
py::object get_pinocchio_model(BaseEkfWithImuKinSettings& obj)
{
    return BoostPython::cpp_to_pybind(obj.pinocchio_model);
}
void set_imu_in_base(BaseEkfWithImuKinSettings& obj, py::object py_src)
{
    obj.imu_in_base = BoostPython::pybind_to_cpp<pinocchio::SE3>(py_src);
}
py::object get_imu_in_base(BaseEkfWithImuKinSettings& obj)
{
    return BoostPython::cpp_to_pybind(obj.imu_in_base);
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >
get_measurement(BaseEkfWithImuKin& obj)
{
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >
        tmp;
    obj.get_measurement(tmp);
    return tmp;
}

void bind_base_ekf_with_imu_kin(py::module& module)
{
    py::class_<BaseEkfWithImuKinSettings>(module, "BaseEkfWithImuKinSettings")
        .def(py::init<>())
        .def_readwrite("is_imu_frame", &BaseEkfWithImuKinSettings::is_imu_frame)
        .def_readwrite("end_effector_frame_names",
                       &BaseEkfWithImuKinSettings::end_effector_frame_names)
        .def_property(
            "pinocchio_model", &get_pinocchio_model, &set_pinocchio_model)
        .def_property("imu_in_base", &get_imu_in_base, &set_imu_in_base)
        .def_readwrite("dt", &BaseEkfWithImuKinSettings::dt)
        .def_readwrite("noise_accelerometer",
                       &BaseEkfWithImuKinSettings::noise_accelerometer)
        .def_readwrite("noise_gyroscope",
                       &BaseEkfWithImuKinSettings::noise_gyroscope)
        .def_readwrite("noise_bias_accelerometer",
                       &BaseEkfWithImuKinSettings::noise_bias_accelerometer)
        .def_readwrite("noise_bias_gyroscope",
                       &BaseEkfWithImuKinSettings::noise_bias_gyroscope)
        .def_readwrite("meas_noise_cov",
                       &BaseEkfWithImuKinSettings::meas_noise_cov)
        .def("__repr__", &BaseEkfWithImuKinSettings::to_string);

    py::class_<BaseEkfWithImuKin>(module, "BaseEkfWithImuKin")
        .def(py::init<>())
        // Public methods.
        .def("initialize",
             &BaseEkfWithImuKin::initialize,
             "Get the EKF settings and initialize the filter from them.")
        .def("set_initial_state",
             static_cast<void(
                 (BaseEkfWithImuKin::*)(Eigen::Ref<const Eigen::Vector3d>,
                                        const Eigen::Quaterniond&,
                                        Eigen::Ref<const Eigen::Vector3d>,
                                        Eigen::Ref<const Eigen::Vector3d>))>(
                 &BaseEkfWithImuKin::set_initial_state),
             "Set the initial state from the base position and velocity.")
        .def("set_initial_state",
             static_cast<void((
                 BaseEkfWithImuKin::*)(Eigen::Ref<
                                           const Eigen::Matrix<double, 7, 1> >,
                                       Eigen::Ref<const Eigen::
                                                      Matrix<double, 6, 1> >))>(
                 &BaseEkfWithImuKin::set_initial_state),
             "Set the initial state from the base position and velocity.")
        .def("update_filter", &BaseEkfWithImuKin::update_filter, "")
        .def("get_filter_output", &BaseEkfWithImuKin::get_filter_output, "")
        .def("get_measurement", &get_measurement, "");
}

}  // namespace mim_estimation
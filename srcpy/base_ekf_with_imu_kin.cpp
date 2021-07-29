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
#include "mim_estimation/base_ekf_with_imu_kin.hpp"
// clang-format on

using namespace boost::python;

namespace mim_estimation
{
template <class Vector>
inline boost::python::list std_vector_to_py_list(const Vector& vector)
{
    typename Vector::const_iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter)
    {
        list.append(*iter);
    }
    return list;
}

boost::python::list get_measurement(BaseEkfWithImuKin* obj)
{
    return std_vector_to_py_list(obj->get_measurement());
}

void bind_base_ekf_with_imu_kin()
{
    class_<BaseEkfWithImuKinSettings>("BaseEkfWithImuKinSettings")
        .def_readwrite("is_imu_frame", &BaseEkfWithImuKinSettings::is_imu_frame)
        .def_readwrite("end_effector_frame_names",
                       &BaseEkfWithImuKinSettings::end_effector_frame_names)
        .def_readwrite("pinocchio_model",
                       &BaseEkfWithImuKinSettings::pinocchio_model)
        .def_readwrite("imu_in_base", &BaseEkfWithImuKinSettings::imu_in_base)
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

    // for get_measurement
    class_<std::vector<Eigen::Vector3d,
                       Eigen::aligned_allocator<Eigen::Vector3d>>>(
        "ListOfVector3d")
        .def(vector_indexing_suite<
             std::vector<Eigen::Vector3d,
                         Eigen::aligned_allocator<Eigen::Vector3d>>>());

    class_<BaseEkfWithImuKin>("BaseEkfWithImuKin")
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
        .def(
            "set_initial_state",
            static_cast<void((
                BaseEkfWithImuKin::*)(Eigen::Ref<
                                          const Eigen::Matrix<double, 7, 1>>,
                                      Eigen::Ref<
                                          const Eigen::Matrix<double, 6, 1>>))>(
                &BaseEkfWithImuKin::set_initial_state),
            "Set the initial state from the base position and velocity.")
        .def("update_filter", &BaseEkfWithImuKin::update_filter, "")
        .def("get_filter_output", &BaseEkfWithImuKin::get_filter_output, "")
        .def("get_measurement", &get_measurement, "");
}

}  // namespace mim_estimation

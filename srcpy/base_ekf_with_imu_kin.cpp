/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Python bindings for the StepperHead class
 */

#include "mim_estimation/base_ekf_with_imu_kin.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/python.hpp>

#include "pinocchio/bindings/python/fwd.hpp"

/*
 * Helper to convert back and forth between C++, Boost::python, and pybind11.
 */
struct BoostPython
{
    template <typename T>
    static pybind11::object cpp_to_pybind(T&& t)
    {
        return pybind11::reinterpret_borrow<pybind11::object>(
            boost::python::api::object(t).ptr());
    }
    template <typename T>
    static pybind11::object cpp_to_pybind(T& t)
    {
        return pybind11::reinterpret_borrow<pybind11::object>(
            boost::python::api::object(t).ptr());
    }
    template <typename T>
    static pybind11::object cpp_to_pybind(T* t)
    {
        return pybind11::reinterpret_borrow<pybind11::object>(
            boost::python::api::object(t).ptr());
    }
    template <typename ReturnType>
    static ReturnType& pybind_to_cpp(pybind11::object model)
    {
        return boost::python::extract<ReturnType&>(model.ptr());
    }
    template <typename ReturnType>
    static ReturnType& pybind_to_cpp(pybind11::handle model)
    {
        return boost::python::extract<ReturnType&>(model.ptr());
    }
};

/*
 * Custom caster of the BaseEkfWithImuKinSettings.
 */
namespace pybind11
{
namespace detail
{
template <>
struct type_caster<pinocchio::Model>
{
public:
    /**
     * This macro establishes the name 'pinocchio::Model' in function
     * signatures and declares a local variable 'value' of type
     * pinocchio::Model
     */
    PYBIND11_TYPE_CASTER(pinocchio::Model, _("pinocchio::Model"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a
     * pinocchio::Model instance or return false upon
     * failure. The second argument indicates whether implicit conversions
     * should be applied.
     */
    bool load(handle src, bool)
    {
        /* Extract PyObject from handle */
        value = BoostPython::pybind_to_cpp<pinocchio::Model>(src);
        /* Ensure return code was OK (to avoid out-of-range errors etc) */
        return !PyErr_Occurred();
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an
     * pinocchio::Model instance into a Python object.
     * The second and third arguments are used to indicate the return value
     * policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(pinocchio::Model src,
                       return_value_policy /* policy */,
                       handle /* parent */)
    {
        object dst = BoostPython::cpp_to_pybind(src);
        return dst;
    }
};

template <>
struct type_caster<pinocchio::SE3>
{
public:
    /**
     * This macro establishes the name 'pinocchio::SE3' in function
     * signatures and declares a local variable 'value' of type
     * pinocchio::SE3
     */
    PYBIND11_TYPE_CASTER(pinocchio::SE3, _("pinocchio::SE3"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a
     * pinocchio::SE3 instance or return false upon
     * failure. The second argument indicates whether implicit conversions
     * should be applied.
     */
    bool load(handle src, bool)
    {
        /* Extract PyObject from handle */
        value = BoostPython::pybind_to_cpp<pinocchio::SE3>(src);
        /* Ensure return code was OK (to avoid out-of-range errors etc) */
        return !PyErr_Occurred();
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an
     * pinocchio::SE3 instance into a Python object.
     * The second and third arguments are used to indicate the return value
     * policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(pinocchio::SE3 src,
                       return_value_policy /* policy */,
                       handle /* parent */)
    {
        object dst = BoostPython::cpp_to_pybind(src);
        return dst;
    }
};
}  // namespace detail
}  // namespace pybind11

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

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > get_measurement(BaseEkfWithImuKin& obj)
{
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > tmp;
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
        .def("initialize", &BaseEkfWithImuKin::initialize,
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
/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2021, New York University and Max Planck Gesellschaft
 * 
 * @brief Helper function for the python bindings.
 */

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/spatial/se3.hpp"
#include "pinocchio/bindings/python/fwd.hpp"

#include <boost/python.hpp>

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

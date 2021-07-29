/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Python bindings for the mim_estimation package.
 */

#include <boost/python.hpp>
#include <boost/python/class.hpp>
// #include <boost/python/overloads.hpp>
#include <eigenpy/eigenpy.hpp>

/* make boost::python understand std::shared_ptr */
namespace boost
{
template <typename T>
T *get_pointer(std::shared_ptr<T> p)
{
    return p.get();
}
}  // namespace boost

namespace mim_estimation
{
void bind_base_ekf_with_imu_kin();
void bind_end_effector_force_estimator();
void bind_robot_state_estimator();

BOOST_PYTHON_MODULE(mim_estimation_cpp)
{
    boost::python::import("pinocchio");

    // Enabling eigenpy support, i.e. numpy/eigen compatibility.
    eigenpy::enableEigenPy();
    eigenpy::enableEigenPySpecific<Eigen::VectorXd>();
    eigenpy::enableEigenPySpecific<Eigen::Vector3d>();
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, 7, 1>>();
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, 6, 1>>();

    bind_base_ekf_with_imu_kin();
    bind_end_effector_force_estimator();
    bind_robot_state_estimator();
}

}  // namespace mim_estimation

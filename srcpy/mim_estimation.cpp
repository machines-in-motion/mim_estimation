/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Python bindings for the mim_estimation package.
 */

#include <pybind11/pybind11.h>

namespace mim_estimation
{
void bind_base_ekf_with_imu_kin(pybind11::module& module);
void bind_end_effector_force_estimator(pybind11::module& module);
void bind_robot_state_estimator(pybind11::module& module);

PYBIND11_MODULE(mim_estimation_cpp, m)
{
    m.import("pinocchio");
    m.doc() = R"pbdoc(
        mim_estimation python bindings
        ------------------------------
        .. currentmodule:: mim_estimation
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    bind_base_ekf_with_imu_kin(m);
    bind_end_effector_force_estimator(m);
}

}  // namespace mim_estimation

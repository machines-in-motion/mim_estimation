/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Computes the forces at the end-effector.
 */

#include "mim_estimation/end_effector_force_estimator.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/parsers/urdf.hpp"

namespace mim_estimation
{
EndEffectorForceEstimator::EndEffectorForceEstimator()
{
    contact_jacobians_.clear();
    contact_jacobians_transposed_inversed_.clear();
    end_effector_forces_.clear();
}

EndEffectorForceEstimator::~EndEffectorForceEstimator()
{
}

void EndEffectorForceEstimator::initialize(
    std::string urdf_path,
    const std::vector<pinocchio::FrameIndex>& frame_indexes)
{
    pinocchio::urdf::buildModel(urdf_path, robot_model_);
    robot_data_ = pinocchio::Data(robot_model_);
    add_contact_frame(frame_indexes);
}

void EndEffectorForceEstimator::add_contact_frame(
    const pinocchio::FrameIndex& frame_index)
{
    contact_jacobians_[frame_index] =
        pinocchio::Data::Matrix6x::Zero(6, robot_model_.nv);
    contact_jacobians_transposed_inversed_[frame_index] =
        pinocchio::Data::Matrix6x::Zero(6, robot_model_.nv);
    end_effector_forces_[frame_index] = Eigen::Matrix<double, 6, 1>::Zero();
}

void EndEffectorForceEstimator::add_contact_frame(
    const std::vector<pinocchio::FrameIndex>& frame_indexes)
{
    for (unsigned int i = 0; i < frame_indexes.size(); ++i)
    {
        add_contact_frame(frame_indexes[i]);
    }
}

void EndEffectorForceEstimator::run(const Eigen::VectorXd& joint_positions,
                                    const Eigen::VectorXd& joint_torques)
{
    // Compute the current contact Jacobians.
    pinocchio::computeJointJacobians(
        robot_model_, robot_data_, joint_positions);
    for (ContactJacobianMap::iterator cnt_it = contact_jacobians_.begin();
         cnt_it != contact_jacobians_.end();
         ++cnt_it)
    {
        const pinocchio::FrameIndex& frame_index = cnt_it->first;
        pinocchio::updateFramePlacement(robot_model_, robot_data_, frame_index);
        pinocchio::getFrameJacobian(robot_model_,
                                    robot_data_,
                                    frame_index,
                                    pinocchio::LOCAL_WORLD_ALIGNED,
                                    contact_jacobians_[frame_index]);

        contact_jacobians_transposed_inversed_[frame_index] =
            contact_jacobians_[frame_index].transpose().inverse();

        end_effector_forces_[frame_index] =
            contact_jacobians_transposed_inversed_[frame_index] * joint_torques;
    }
}

}  // namespace mim_estimation

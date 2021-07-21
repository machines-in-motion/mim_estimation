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
    end_effector_forces_.clear();
}

EndEffectorForceEstimator::~EndEffectorForceEstimator()
{
}

void EndEffectorForceEstimator::initialize(
    const std::string& urdf_path, const std::vector<std::string>& frame_names)
{
    pinocchio::urdf::buildModel(urdf_path, robot_model_);
    robot_data_ = pinocchio::Data(robot_model_);
    nb_joint_ = robot_model_.nv;
    q_.resize(robot_model_.nq);
    q_.fill(0.0);
    add_contact_frame(frame_names);

    // run once with dummy data in order to update the different internal
    // matrix size, especially inside the numerical solver.
}

void EndEffectorForceEstimator::add_contact_frame(const std::string& frame_name)
{
    if (!robot_model_.existFrame(frame_name))
    {
        throw std::runtime_error(
            "EndEffectorForceEstimator::add_contact_frame() "
            "the given frame name (" +
            frame_name + ") does not exists in the given URDF.");
    }
    end_effector_id_map_[frame_name] = robot_model_.getFrameId(frame_name);
    const pinocchio::FrameIndex& frame_index = end_effector_id_map_[frame_name];
    contact_jacobians_[frame_index] = Eigen::MatrixXd::Zero(6, robot_model_.nv);
    contact_jacobians_transpose_[frame_index] =
        Eigen::MatrixXd::Zero(robot_model_.nv, 3);
    end_effector_forces_[frame_index] = Eigen::Matrix<double, 6, 1>::Zero();
}

void EndEffectorForceEstimator::add_contact_frame(
    const std::vector<std::string>& frame_names)
{
    for (unsigned int i = 0; i < frame_names.size(); ++i)
    {
        add_contact_frame(frame_names[i]);
    }
}

void EndEffectorForceEstimator::run(const Eigen::VectorXd& joint_positions,
                                    const Eigen::VectorXd& joint_torques)
{
    q_.tail(joint_positions.size()) = joint_positions;
    // Compute the current contact Jacobians.
    pinocchio::computeJointJacobians(robot_model_, robot_data_, q_);
    for (Matrix6xMap::iterator cnt_it = contact_jacobians_.begin();
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

        contact_jacobians_transpose_[frame_index] =
            contact_jacobians_[frame_index].topRows<3>().transpose();

        solver_.compute(contact_jacobians_transpose_[frame_index]);

        end_effector_forces_[frame_index].head<3>() =
            -solver_.solve(joint_torques);
    }
}

const Eigen::Matrix<double, 6, 1>& EndEffectorForceEstimator::get_force(
    const std::string& frame_name)
{
    return end_effector_forces_[end_effector_id_map_[frame_name]];
}

}  // namespace mim_estimation

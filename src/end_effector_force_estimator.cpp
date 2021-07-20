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
    // solver_ = FullPivLU;
    // solver_ = HouseholderQR;
    solver_ = ColPivHouseholderQR;
    // solver_ = FullPivHouseholderQR;
    // solver_ = CompleteOrthogonalDecomposition;
    // solver_ = LLT;
    // solver_ = LDLT;
    // solver_ = BDCSVD;
    // solver_ = JacobiSVD;
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
    if (has_free_flyer())
    {
        q_(6) = 1.0;
        nb_joint_ -= 6;
    }
    add_contact_frame(frame_names);
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
    contact_jacobians_[frame_index] =
        pinocchio::Data::Matrix6x::Zero(6, robot_model_.nv);
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

        end_effector_forces_[frame_index].head<3>() =
            solve(contact_jacobians_[frame_index]
                      .topRows<3>()
                      .rightCols(nb_joint_)
                      .transpose(),
                  joint_torques);
    }
}

Eigen::VectorXd EndEffectorForceEstimator::solve(
    Eigen::Ref<const Eigen::MatrixXd> a, Eigen::Ref<const Eigen::VectorXd> b)
{
    switch (solver_)
    {
        case Solver::FullPivLU:
            return a.fullPivLu().solve(b);
            break;
        case Solver::HouseholderQR:
            return a.householderQr().solve(b);
            break;
        case Solver::ColPivHouseholderQR:
            return a.colPivHouseholderQr().solve(b);
            break;
        case Solver::FullPivHouseholderQR:
            return a.fullPivHouseholderQr().solve(b);
            break;
        case Solver::CompleteOrthogonalDecomposition:
            return a.completeOrthogonalDecomposition().solve(b);
            break;
        case Solver::LLT:
            return a.llt().solve(b);
            break;
        case Solver::LDLT:
            return a.ldlt().solve(b);
            break;
        case Solver::BDCSVD:
            return a.bdcSvd().solve(b);
            break;
        case Solver::JacobiSVD:
            return a.jacobiSvd().solve(b);
            break;
        default:
            return a.colPivHouseholderQr().solve(b);
            break;
    }
}

const Eigen::Matrix<double, 6, 1>& EndEffectorForceEstimator::get_force(
    const std::string& frame_name)
{
    return end_effector_forces_[end_effector_id_map_[frame_name]];
}

}  // namespace mim_estimation

/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Computes the forces at the end-effector.
 */

#pragma once

#include <Eigen/Eigen>
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

namespace robot_estimation
{
class EndEffectorForceEstimator
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::map<
        pinocchio::FrameIndex,
        pinocchio::Data::Matrix6x,
        std::less<pinocchio::FrameIndex>,
        Eigen::aligned_allocator<std::pair<const pinocchio::FrameIndex,
                                           pinocchio::Data::Matrix6x> > >
        ContactJacobianMap;

    typedef std::map<
        pinocchio::FrameIndex,
        Eigen::Matrix<double, 6, 1>,
        std::less<pinocchio::FrameIndex>,
        Eigen::aligned_allocator<std::pair<const pinocchio::FrameIndex,
                                           Eigen::Matrix<double, 6, 1> > > >
        EndEffectorForceMap;

public:
    /** @brief Construct a new EndEffectorForceEstimator object.
     *
     * No memory allocation, @see initialize.
     * Use the default constructor for eigen and pinocchio objects.
     */
    EndEffectorForceEstimator();

    /** @brief Destroy the EndEffectorForceEsstimator object. */
    ~EndEffectorForceEstimator();

    void initialize(std::string urdf_path,
                    const std::vector<pinocchio::FrameIndex>& frame_indexes =
                        std::vector<pinocchio::FrameIndex>());

    void add_contact_frame(const pinocchio::FrameIndex& frame_index);

    void add_contact_frame(
        const std::vector<pinocchio::FrameIndex>& frame_indexes);

    void run(const Eigen::VectorXd& joint_positions,
             const Eigen::VectorXd& joint_torques);

private:
    /** @brief Contact Jacobians associated to a robot frame. */
    ContactJacobianMap contact_jacobians_;

    /** @brief Transposed inverted contact Jacobians associated to a robot
     * frame. */
    ContactJacobianMap contact_jacobians_transposed_inversed_;

    /** @brief Forces applied by the environmnent at the end-effector [N]. */
    EndEffectorForceMap end_effector_forces_;

    /** @brief Rigid body model of the robot, constructed from a urdf file. */
    pinocchio::Model robot_model_;

    /** @brief Cache of the algorithm performed on the robot model. */
    pinocchio::Data robot_data_;
};

}  // namespace robot_estimation

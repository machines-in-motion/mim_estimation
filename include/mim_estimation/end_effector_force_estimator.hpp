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

namespace mim_estimation
{
class EndEffectorForceEstimator
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef Eigen::Matrix<double, Eigen::Dynamic, 3> MatrixX3;
    typedef std::map<std::string, pinocchio::FrameIndex> EndEffectorIdMap;

    typedef std::map<pinocchio::FrameIndex,
                     MatrixX3,
                     std::less<pinocchio::FrameIndex>,
                     Eigen::aligned_allocator<
                         std::pair<const pinocchio::FrameIndex, MatrixX3>>>
        MatrixX3Map;

    typedef std::map<
        pinocchio::FrameIndex,
        pinocchio::Data::Matrix6x,
        std::less<pinocchio::FrameIndex>,
        Eigen::aligned_allocator<
            std::pair<const pinocchio::FrameIndex, pinocchio::Data::Matrix6x>>>
        Matrix6xMap;

    typedef std::map<
        pinocchio::FrameIndex,
        Eigen::Matrix<double, 6, 1>,
        std::less<pinocchio::FrameIndex>,
        Eigen::aligned_allocator<std::pair<const pinocchio::FrameIndex,
                                           Eigen::Matrix<double, 6, 1>>>>
        EndEffectorForceMap;

public:
    /** @brief Construct a new EndEffectorForceEstimator object.
     *
     * No memory allocation, @see initialize.
     * Use the default constructor for eigen and pinocchio objects.
     */
    EndEffectorForceEstimator();

    /** @brief Destroy the EndEffectorForceEstimator object. */
    ~EndEffectorForceEstimator();

    void initialize(const std::string& urdf_path,
                    const std::vector<std::string>& frame_name =
                        std::vector<std::string>());

    void add_contact_frame(const std::string& frame_name);

    void add_contact_frame(const std::vector<std::string>& frame_names);

    void run(const Eigen::VectorXd& joint_positions,
             const Eigen::VectorXd& joint_torques);

    const Eigen::Matrix<double, 6, 1>& get_force(const std::string& frame_name);

    std::string to_string()
    {
        return "EndEffectorForceEstimator::to_string(): to implement.";
    }

private:
    /** @brief Contact Jacobians associated to a robot frame. */
    Matrix6xMap contact_jacobians_;

    /** @brief Transpose of the contact Jacobians associated to a robot frame.
     */
    MatrixX3Map contact_jacobians_transpose_;

    /** @brief Forces applied by the environment at the end-effector [N]. */
    EndEffectorForceMap end_effector_forces_;

    /** @brief Map from model frame name to frame id. */
    EndEffectorIdMap end_effector_id_map_;

    /** @brief Rigid body model of the robot, constructed from a urdf file. */
    pinocchio::Model robot_model_;

    /** @brief Cache of the algorithm performed on the robot model. */
    pinocchio::Data robot_data_;

    /** @brief Internal robot configuration to perform the rigid body
     * algorithms. */
    Eigen::VectorXd q_;

    /** @brief Linear solver for the equation \$f A x = B \$f */
    Eigen::ColPivHouseholderQR<MatrixX3> solver_;

    /** @brief Number of joint Dof */
    int nb_joint_;
};

}  // namespace mim_estimation

/**
 * \remarks      ...
 * \file         robot_state_estimation.cpp
 * \author       Alexander Herzog
 * \date         Oct 26, 2015
 */

#include <estimator/robot_state_estimation.h>
#include <example_tasks/utils/robot_part_iterators.h>

#ifdef HAS_VICON
#include <estimator_cpp/vicon_base_state_estimator.hpp>
#endif

using namespace example_tasks;
using namespace mim_estimation;

namespace estimation
{
RobotStateEstimation::RobotStateEstimation(
    YAML::Node n,
    std::unique_ptr<mim_estimation::RobotProperties> robot_prop,
    JointPositionSensorPtr joint_sensors,
    BaseStateSensorPtr base_state_sensor,
    FTSensorPtrArray wrench_sensors,
    IMUPtr imu_sensors,
    KinematicsPtr filtered_kinematics,
    KinematicsPtr unfiltered_kinematics,
    VisToolsPtr vis_tools_interface)
    : base_state_estimator_(n["base_state_ekf"]),
      vis_tools_interface_(vis_tools_interface)
#ifdef HAS_VICON
      ,
      athena_vicon_collect_(robot_math::RtThreadInfo(
          "vicon_base_state_estimator_athena_vicon_collect", 0, 0)),
      vicon_bse_(filtered_kinematics->getUpdatePeriod(),
                 vis_tools_interface,
                 n["vicon_bse"])
#endif
{
    robot_prop_ = std::move(robot_prop);
    unfiltered_joint_position_sensor_ = std::move(joint_sensors);
    simulated_base_state_ = std::move(base_state_sensor);
    unfiltered_forward_kinematics_ = std::move(unfiltered_kinematics);
    filtered_kinematics_ = std::move(filtered_kinematics);
    unfiltered_imu_ = std::move(imu_sensors);

    unfiltered_wrench_sensors_ = std::move(wrench_sensors);

    joint_sensors_filter_cutoff_ =
        n["joint_sensors_filter_cutoff"].as<double>();
    wrench_filter_cutoff_ = n["wrench_filter_cutoff"].as<double>();

    force_threshold_for_contact_ =
        n["force_threshold_for_contact"].as<double>();
}

void RobotStateEstimation::initialize(
    const ContactDescritpion (&contact_description)[Robot::n_endeffs_])
{
#ifdef HAS_VICON
    use_vicon_base_ = true;
#endif

    for (int i = 0; i < Robot::n_endeffs_; ++i)
    {
        contacts_[i] = contact_description[i];
    }

    // acquire init data of all sensors
    unfiltered_joint_position_sensor_->acquire();
    unfiltered_joint_position_sensor_->data().positions().segment<3>(
        Robot::dof_l_ar_ - 1) = Eigen::Vector3d(0., 0.329, 0.);
    unfiltered_joint_position_sensor_->data().positions().segment<3>(
        Robot::dof_r_ar_ - 1) = Eigen::Vector3d(0., 0.329, 0.);
    unfiltered_joint_position_sensor_->data().velocities().segment<3>(
        Robot::dof_l_ar_ - 1) = Eigen::Vector3d(0., 0., 0.);
    unfiltered_joint_position_sensor_->data().velocities().segment<3>(
        Robot::dof_r_ar_ - 1) = Eigen::Vector3d(0., 0., 0.);
    unfiltered_imu_->acquire();
    //  const_cast<geometry_utils::Quaternion&>(simulated_base_state_->data().orient()).set(1.,0.,0.,0.);
    for (int i = 0; i < unfiltered_wrench_sensors_.size(); ++i)
    {
        assert(i < unfiltered_wrench_sensors_.size());
        unfiltered_wrench_sensors_[i]->acquire();
    }

    // create unfiltered posture and initialize kinematics + filters
    mim_estimation::RobotPosture init_posture(
        unfiltered_joint_position_sensor_->data().positions());
    if (simulated_base_state_ != nullptr)
    {
        simulated_base_state_->acquire();
        init_posture.base_position_ = simulated_base_state_->data().pos();
        init_posture.base_orientation_ = simulated_base_state_->data().orient();
    }
    else
    {
#ifdef HAS_VICON
        if (use_vicon_base_)
        {
            // Initialize the Vicon data collector:
            athena_vicon_collect_.initialize(true);

            // Acquire most recent Vicon data and initialize state:
            athena_vicon_collect_.update();
            vicon_kinematics::ViconFrame base_pose_vicon;
            vicon_kinematics::AthenaViconCollect::PostureVector posture_vec =
                joint_sensors_data.positions();
            bool new_frame = athena_vicon_collect_.get_base_measurement(
                posture_vec, base_pose_vicon);

            std::cout << "Using vicon..." << std::endl;
            vicon_bse_.initialize(unfiltered_joint_position_sensor_->data(),
                                  base_pose_vicon.f_,
                                  new_frame);
            Eigen::Vector3d vicon_bse_base_vel, vicon_bse_accel_bias,
                vicon_bse_gyro_bias;
            vicon_bse_.get_filter_state(init_posture.base_position_,
                                        vicon_bse_base_vel,
                                        init_posture.base_orientation_,
                                        vicon_bse_accel_bias,
                                        vicon_bse_gyro_bias);
            floor_tf_.setIdentity();
            floor_tf_(2, 3) = vicon_bse_.getFloorHeight();
        }
#endif
    }

    mim_estimation::RobotVelocity dummy_vel =
        mim_estimation::RobotVelocity::Zero();
    mim_estimation::RobotAcceleration dummy_acc =
        mim_estimation::RobotAcceleration::Zero();
    unfiltered_forward_kinematics_->initialize(
        init_posture, dummy_vel, dummy_acc, contacts_);
    filtered_kinematics_->initialize(
        init_posture, dummy_vel, dummy_acc, contacts_);

    position_filter_.initialize(
        unfiltered_joint_position_sensor_->data().positions(),
        joint_sensors_filter_cutoff_);
    velocity_filter_.initialize(
        unfiltered_joint_position_sensor_->data().velocities(),
        joint_sensors_filter_cutoff_);
    for (int i = 0; i < wrench_filters_.size(); ++i)
    {
        assert(i < unfiltered_wrench_sensors_.size());
        wrench_filters_[i].initialize(
            unfiltered_wrench_sensors_[i]->data().data(),
            wrench_filter_cutoff_);
    }

    filt_foot_cops_.resize(3, Eigen::Vector3d::Zero());
    filt_foot_cops_rgb_.resize(3, Eigen::Vector3d::Zero());
    filt_contact_points_.resize(2, Eigen::Vector3d::Zero());
}

void RobotStateEstimation::update()
{
    const mim_estimation::RobotVelocity dummy_vel =
        mim_estimation::RobotVelocity::Constant(
            std::numeric_limits<double>::infinity());
    const mim_estimation::RobotAcceleration dummy_acc =
        mim_estimation::RobotAcceleration::Constant(
            std::numeric_limits<double>::infinity());

    // // read position sensors
    unfiltered_joint_position_sensor_->acquire();

    // filter joint position and velocity
    position_filter_.update(
        unfiltered_joint_position_sensor_->data().positions());
    velocity_filter_.update(
        unfiltered_joint_position_sensor_->data().velocities());

    // construct estimated robot joint state
    position_filter_.getData(filtered_posture_.joint_positions_);
    velocity_filter_.getData(filtered_velocity_.joint_velocities());

    // do the wrench acquisition and filtering
    for (int i = 0; i < Robot::n_endeffs_; ++i)
    {
        unfiltered_wrench_sensors_[i]->acquire();
        wrench_filters_[i].update(unfiltered_wrench_sensors_[i]->data().data());
        wrench_filters_[i].getData(filtered_wrench_[i].data());
        filtered_wrench_[i].point_of_action() =
            unfiltered_wrench_sensors_[i]->data().point_of_action();
    }

    // compute and draw the foot and overall cops using knowledge of the floor
    // height:
    // computeFootCoPsOnFloor();

    // Draw the projected foot contact points on the floor:
    int i = 0;
    for (auto eff_id : {EndeffId::right(), EndeffId::left()})
    {
        filt_contact_points_.at(i) =
            getFootContactPointProjected(EndeffItr(eff_id) + 1);
        ++i;
    }
    vis_tools_interface_->drawBalls(
        filt_contact_points_, Eigen::Vector3d(1.0, 0.0, 0.0), 1.0, 0.02);

    // read IMU data
    unfiltered_imu_->acquire();

    // construct forward kinematics from unfiltered joint positions only
    mim_estimation::RobotPosture raw_posture_without_base(
        unfiltered_joint_position_sensor_->data().positions());
    unfiltered_forward_kinematics_->update(
        raw_posture_without_base, dummy_vel, dummy_acc, contacts_);

    // estimate base pose
    if (simulated_base_state_ != nullptr)
    {
        simulated_base_state_->acquire();
    }
    if (simulated_base_state_ != nullptr)
    {
        filtered_posture_.base_position_ = simulated_base_state_->data().pos();
        filtered_velocity_.base_linear_velocity() =
            simulated_base_state_->data().vel();
        filtered_posture_.base_orientation_ =
            simulated_base_state_->data().orient();
        filtered_velocity_.base_angular_velocity() =
            simulated_base_state_->data().ang_vel();
    }
    else
    {
#ifdef HAS_VICON
        if (use_vicon_base_)
        {
            update_vicon_base_state_ekf();
            update_opengl(filtered_posture_);
        }
        else
        {
#endif
            update_base_estimation_ekf();
            update_opengl(filtered_posture_);
#ifdef HAS_VICON

            // draw base state
        }
#endif
    }
    assert(filtered_posture_.allFinite() && filtered_velocity_.allFinite());
    filtered_kinematics_->update(
        filtered_posture_, filtered_velocity_, dummy_acc, contacts_);

    for (int i = 0; i < Robot::n_endeffs_; ++i)
    {
        endeff_wrench_[i] = filtered_wrench_[i].express_wrench_at(
            filtered_kinematics_->link_pose(
                unfiltered_wrench_sensors_[i]->link_id()),
            filtered_kinematics_->endeffector_position(i + 1));
    }
}

void RobotStateEstimation::computeFootCoPsOnFloor()
{
#ifdef HAS_VICON

    // Compute CoPs assuming flat feet contacts with the ground (specific to
    // Athena!):
    Eigen::Vector3d endeff_frc_world_frame, endeff_trq_world_frame;
    Eigen::Matrix4d ft_sensor_tf;
    double cop_torque;

    int i = 0;
    for (auto eff_id : {EndeffId::right(), EndeffId::left()})
    {
        assert(EndeffItr(eff_id) < filtered_wrench_.size());

        // Transform the filtered wrenches into world frame:
        ft_sensor_tf = filtered_kinematics_->link_pose(
            unfiltered_wrench_sensors_[EndeffItr(eff_id)]->link_id());
        endeff_frc_world_frame = ft_sensor_tf.topLeftCorner(3, 3) *
                                 filtered_wrench_[EndeffItr(eff_id)].force();
        endeff_trq_world_frame = ft_sensor_tf.topLeftCorner(3, 3) *
                                 filtered_wrench_[EndeffItr(eff_id)].torque();

        assert(EndeffItr(eff_id) < filt_foot_cops_.size());
        // Compute the CoPs from the world frame wrenches, assuming knowledge of
        // the ground plan:
        filt_foot_cops_.at(i).setZero();
        filt_foot_cops_.at(i).segment(0, 2) =
            mim_estimation::ContactHelper::center_of_pressure(
                endeff_frc_world_frame,
                endeff_trq_world_frame,
                geometry_utils::Transformations::transformVector(
                    ft_sensor_tf,
                    filtered_wrench_[EndeffItr(eff_id)].point_of_action()),
                floor_tf_,
                cop_torque);
        filt_foot_cops_.at(i) =
            geometry_utils::Transformations::transformVector(
                floor_tf_, filt_foot_cops_.at(i));
        filt_foot_cops_rgb_.at(i)(i) = 1.0;
        ++i;
    }

    // compute the overall CoP by transforming the right FT sensor force/torque
    // to the left FT sensor location:
    Eigen::Matrix<double, 6, 1> total_wrench_left_FT =
        filtered_wrench_[Robot::id_left_foot_ - 1].wrench();
    Eigen::Matrix4d right_FT_to_left_FT_tf =
        geometry_utils::Transformations::invertHomogeneousTransform(
            filtered_kinematics_->link_pose(
                unfiltered_wrench_sensors_[Robot::id_left_foot_ - 1]
                    ->link_id())) *
        filtered_kinematics_->link_pose(
            unfiltered_wrench_sensors_[Robot::id_right_foot_ - 1]->link_id());
    total_wrench_left_FT += geometry_utils::Transformations::transformWrench(
        right_FT_to_left_FT_tf,
        filtered_wrench_[Robot::id_right_foot_ - 1].wrench());
    Eigen::Matrix<double, 6, 1> total_wrench_world =
        geometry_utils::Transformations::rotateWrench(
            filtered_kinematics_->link_pose(
                unfiltered_wrench_sensors_[Robot::id_left_foot_ - 1]
                    ->link_id()),
            total_wrench_left_FT);

    // compute the overall cop:
    ft_sensor_tf = filtered_kinematics_->link_pose(
        unfiltered_wrench_sensors_[Robot::id_left_foot_ - 1]->link_id());
    filt_overall_cop_.setZero();
    filt_overall_cop_.segment(0, 2) =
        mim_estimation::ContactHelper::center_of_pressure(
            total_wrench_world.segment(0, 3),
            total_wrench_world.segment(3, 3),
            geometry_utils::Transformations::transformVector(
                ft_sensor_tf,
                filtered_wrench_[Robot::id_left_foot_ - 1].point_of_action()),
            floor_tf_,
            cop_torque);
    filt_foot_cops_.at(2) = geometry_utils::Transformations::transformVector(
        floor_tf_, filt_overall_cop_);
    filt_foot_cops_rgb_.at(2)(2) = 1.0;

    vis_tools_interface_->drawBalls(
        filt_foot_cops_, filt_foot_cops_rgb_, 1.0, 0.03);

#endif
}

void RobotStateEstimation::update_vicon_base_state_ekf()
{
#ifdef HAS_VICON
    athena_vicon_collect_.update() vicon_kinematics::ViconFrame base_pose_vicon;
    vicon_kinematics::AthenaViconCollect::PostureVector posture_vec =
        joint_sensors_data.positions();
    bool new_frame = athena_vicon_collect_.get_base_measurement(
        posture_vec, base_pose_vicon);
    vicon_bse_.update(unfiltered_imu_->data(),
                      unfiltered_joint_position_sensor_->data(),
                      base_pose_vicon.f_,
                      new_frame);
    Eigen::Vector3d vicon_bse_base_vel, vicon_bse_accel_bias,
        vicon_bse_gyro_bias;
    vicon_bse_.get_filter_state(filtered_posture_.base_position_,
                                vicon_bse_base_vel,
                                filtered_posture_.base_orientation_,
                                vicon_bse_accel_bias,
                                vicon_bse_gyro_bias);
    filtered_velocity_.base_linear_velocity() = vicon_bse_base_vel;
    filtered_velocity_.base_angular_velocity() =
        unfiltered_imu_->data().gyroscope() - vicon_bse_gyro_bias;
#endif
}

void RobotStateEstimation::update_base_estimation_ekf()
{
    // update base_estimator add X value of FT measurement
    assert(EndeffItr(EndeffId::left()) < unfiltered_wrench_sensors_.size());
    assert(EndeffItr(EndeffId::right()) < unfiltered_wrench_sensors_.size());

    geometry_utils::Quaternion left_orien, right_orien;
    left_orien.rotation_matrix_to_quaternion(
        unfiltered_forward_kinematics_->endeffector_pose(Robot::id_left_foot_)
            .topLeftCorner<3, 3>());
    right_orien.rotation_matrix_to_quaternion(
        unfiltered_forward_kinematics_->endeffector_pose(Robot::id_right_foot_)
            .topLeftCorner<3, 3>());
    base_state_estimator_.update(
        unfiltered_imu_->data().accelerometer(),
        unfiltered_imu_->data().gyroscope(),
        unfiltered_forward_kinematics_->endeffector_position(
            Robot::id_right_foot_),
        unfiltered_forward_kinematics_->endeffector_position(
            Robot::id_left_foot_),
        right_orien,
        left_orien,
        contacts_[Robot::id_left_foot_].activation_(0),
        contacts_[Robot::id_right_foot_].activation_(0));

    // update the robot base state
    Eigen::Vector3d dummy, base_linvel, base_angvel;
    base_state_estimator_.getState(filtered_posture_.base_position_,
                                   base_linvel,
                                   dummy,
                                   filtered_posture_.base_orientation_,
                                   base_angvel,
                                   dummy);
    filtered_velocity_.base_linear_velocity() = base_linvel;
    filtered_velocity_.base_angular_velocity() = base_angvel;
}

void RobotStateEstimation::subscribe_to_data_collector(
    data_collection::DataCollector& data_collector, std::string name)
{
#ifdef HAS_VICON
    if (use_vicon_base_)
    {
        athena_vicon_collect_.subscribe_to_data_collector(data_collector);
        vicon_bse_.subscribe_to_data_collector(data_collector);
    }
#endif
    std::vector<std::string> jnames(Robot::n_dofs_), units(Robot::n_dofs_);
    for (int i = 0; i < Robot::n_dofs_; ++i)
    {
        jnames[i] = std::string(robot_prop_->get_joint_name(i + 1));
        units[i] = std::string("-");
    }
    data_collector.addVector(filtered_posture_.joint_positions_,
                             name + std::string("_jfilt"),
                             jnames,
                             units);
    data_collector.addVector(filtered_velocity_.joint_velocities(),
                             name + std::string("_djfilt"),
                             jnames,
                             units);

    data_collector.addVector(
        unfiltered_joint_position_sensor_->data().positions(),
        name + std::string("_jraw"),
        jnames,
        units);
    data_collector.addVector(
        unfiltered_joint_position_sensor_->data().velocities(),
        name + std::string("_djraw"),
        jnames,
        units);
    data_collector.addVector3d(
        unfiltered_imu_->data().gyroscope(), name + "_imu_base_gyro", units[0]);
    data_collector.addVector3d(unfiltered_imu_->data().accelerometer(),
                               name + "_imu_base_acc",
                               units[0]);

    assert(EndeffItr(EndeffId::left()) < filtered_wrench_.size());
    assert(EndeffItr(EndeffId::right()) < filtered_wrench_.size());
    data_collector.addVector6d(
        endeff_wrench_[Robot::id_left_foot_ - 1].data(), name + "_LF_FT", "-");
    data_collector.addVector6d(
        endeff_wrench_[Robot::id_right_foot_ - 1].data(), name + "_RF_FT", "-");

    data_collector.addVector3d(
        filtered_posture_.base_position_, name + "_base_pos", "-");
    data_collector.addQuaternion(filtered_posture_.base_orientation_.get(),
                                 name + "_base_orient");
    data_collector.addVector3d(
        filtered_velocity_.base_linear_velocity(), name + "_base_lvel", "-");
    data_collector.addVector3d(
        filtered_velocity_.base_angular_velocity(), name + "_base_avel", "-");

    if (simulated_base_state_ != nullptr)
    {
        data_collector.addVector3d(
            simulated_base_state_->data().pos(), name + "_sim_base_pos", "m");
        data_collector.addQuaternion(
            simulated_base_state_->data().orient().get(),
            name + "_sim_base_orient");
    }
}

}  // namespace estimation

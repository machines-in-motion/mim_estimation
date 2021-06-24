/*********************************************************************
 Autonomous Motion Department
 Max-Planck Intelligent Systems
 Prof. Stefan Schaal
 *********************************************************************
 \remarks      ...

 \file         robot_state_estimation.h

 \author       Alexander Herzog
 \date         Oct 26, 2015

 *********************************************************************/

#pragma once

#include <array>
#include <memory>

#include <data_collection/data_collector.h>
#include <mim_estimation/contact_helper.h>
#include <mim_estimation/kinematics.h>
#include <mim_estimation/robot_properties.h>
#include <mim_estimation/robot_state_representation.h>
#include <mim_estimation/sensors.h>
#include <robot_properties/robot.h>
#include <standard_filters/butterworth_filter.h>
#include <yaml-cpp/yaml.h>
#include "geometry_utils/Transformations.h"
#include "hinvdyn_balance/ankle_ft_sensor.h"

#ifdef HAS_VICON
#include "base_state_estimators/ViconBaseStateEstimator.h"
#include "vicon_kinematics/athena_vicon_collect.h"
#endif

#include "base_state_estimators/BaseEstEKF.h"

#include "visualization_tools/VisualizationToolsInterface.h"

namespace estimation
{
struct RobotStateEstimation
{
    typedef Eigen::Matrix<double, 6, 1> EigenVector6d;
    typedef mim_estimation::RobotPosture::EigenVectorNDofsd EigenVectorNDofsd;

    typedef standard_filters::ButterworthFilter<EigenVectorNDofsd, 2>
        JointDOFFilter;
    typedef standard_filters::ButterworthFilter<EigenVector6d, 2> WrenchFilter;

    typedef std::array<mim_estimation::ContactDescritpion, Robot::n_endeffs_>
        ContactDescriptionArray;

    typedef std::unique_ptr<mim_estimation::JointPositionSensors>
        JointPositionSensorPtr;
    typedef std::unique_ptr<mim_estimation::BaseStateSensor> BaseStateSensorPtr;
    typedef std::unique_ptr<mim_estimation::IMU> IMUPtr;
    typedef std::unique_ptr<mim_estimation::FTSensor> FTSensorPtr;
    typedef std::array<FTSensorPtr, Robot::n_endeffs_> FTSensorPtrArray;

    typedef std::unique_ptr<mim_estimation::Kinematics> KinematicsPtr;

    typedef std::shared_ptr<VisualizationToolsInterface> VisToolsPtr;

    RobotStateEstimation(
        YAML::Node n,
        std::unique_ptr<mim_estimation::RobotProperties> robot_prop,
        JointPositionSensorPtr joint_sensors,
        BaseStateSensorPtr base_state_sensor,
        FTSensorPtrArray wrench_sensors,
        IMUPtr imu_sensors,
        KinematicsPtr filtered_kinematics,
        KinematicsPtr unfiltered_kinematics,
        VisToolsPtr vis_tools_interface);
    RobotStateEstimation(const RobotStateEstimation&) = delete;
    RobotStateEstimation(RobotStateEstimation&&) = delete;
    virtual ~RobotStateEstimation(){};
    void initialize(const mim_estimation::ContactDescritpion (
        &contact_description)[Robot::n_endeffs_]);

    void update();

    void computeFootCoPsOnFloor();

    Eigen::Vector3d getFootContactPointProjected(int endeff_id)
    {
        Eigen::Vector3d foot_pos_proj =
            filtered_kinematics_->endeffector_position(endeff_id);
        foot_pos_proj(2) = floor_tf_(2, 3);
        return foot_pos_proj;
    }

    bool isSimulated(void)
    {
        if (simulated_base_state_ != nullptr)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void subscribe_to_data_collector(
        data_collection::DataCollector& data_collector, std::string name);

    std::unique_ptr<mim_estimation::RobotProperties> robot_prop_;

    // unfiltered quantites
    JointPositionSensorPtr unfiltered_joint_position_sensor_;
    BaseStateSensorPtr simulated_base_state_;
    KinematicsPtr unfiltered_forward_kinematics_;
    IMUPtr unfiltered_imu_;
    FTSensorPtrArray unfiltered_wrench_sensors_;

    // filters
    int joint_sensors_filter_cutoff_, wrench_filter_cutoff_;
    JointDOFFilter position_filter_;
    JointDOFFilter velocity_filter_;
    std::array<WrenchFilter, Robot::n_endeffs_> wrench_filters_;
    BaseEstEKF base_state_estimator_;

    // visualization
    std::shared_ptr<VisualizationToolsInterface> vis_tools_interface_;

    double force_threshold_for_contact_;

    // estimated posture/velocity/wrench and cops
    mim_estimation::RobotPosture filtered_posture_;
    mim_estimation::RobotVelocity filtered_velocity_;
    std::array<mim_estimation::FTSensorData, Robot::n_endeffs_>
        filtered_wrench_;
    std::array<mim_estimation::FTSensorData, Robot::n_endeffs_> endeff_wrench_;
    std::vector<Eigen::Vector3d> filt_foot_cops_, filt_foot_cops_rgb_;
    std::vector<Eigen::Vector3d> filt_contact_points_;
    Eigen::Vector3d filt_overall_cop_;
    Eigen::Matrix4d floor_tf_;

    mim_estimation::ContactDescritpion contacts_[Robot::n_endeffs_];

    KinematicsPtr filtered_kinematics_;

#ifdef HAS_VICON
    vicon_kinematics::AthenaViconCollect athena_vicon_collect_;
    ViconBaseStateEstimator vicon_bse_;
#endif
    bool use_vicon_base_;

private:
    void update_base_estimation_ekf();
    void update_vicon_base_state_ekf();
    virtual bool update_opengl(const mim_estimation::RobotPosture& posture) = 0;
};

}  // namespace estimation

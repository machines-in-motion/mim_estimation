/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft
 * 
 * @brief Implements a testing class for the Vicon and IMU based estimator.
 * It loads a yaml configuration file and a SL data file, estimate the state
 * from these data and compare to the vicon trajectory.
 */

#include <fstream>
#include <string>

#include "robot_estimation/ekf_vicon_imu.hpp"

namespace robot_estimation {
namespace test {

class EstimatorViconImuTest{

  ///
  /// \brief This class tests the robot_estimation based on IMU and Vicon measurement
  ///

public:
  EstimatorViconImuTest(
      std::string yaml_file, std::string d_file)
  {
    /* get the parameters from the yaml file */
    std::cout << "Creating the robot_estimation dependence..." << std::endl;
    std::cout << "Get the robot_estimation parameters from " << yaml_file << std::endl;
    config_ = YAML::LoadFile(yaml_file);
    update_frequency_ = config_["task_servo_rate"].as<double>();
    config_est_ = config_["robot_state_est"];
    /* load the data set */
    std::cout << "Get the data from " << d_file << std::endl;
    data_reader_.read(d_file);
    vicon_bse_.reset(new robot_estimation::EkfViconImu(
                       update_frequency_, config_est_["vicon_bse"]));
    time_ = 0.0;
    get_data_indexes();
    subscribe_to_data_collector();

    error_base_pose_.resize(data_reader_.getNbRows(), 3);
    error_base_pose_.setZero();
    error_base_quat_.resize(data_reader_.getNbRows(), 4);
    error_base_quat_.setZero();

    avg_err_base_pose   = -1.0;
    avg_err_base_pose_x = -1.0;
    avg_err_base_pose_y = -1.0;
    avg_err_base_pose_z = -1.0;
    max_err_base_pose   = -1.0;
    max_err_base_pose_x = -1.0;
    max_err_base_pose_y = -1.0;
    max_err_base_pose_z = -1.0;
    min_err_base_pose   = -1.0;
    min_err_base_pose_x = -1.0;
    min_err_base_pose_y = -1.0;
    min_err_base_pose_z = -1.0;
    avg_err_base_quat   = -1.0;
    avg_err_base_quat_x = -1.0;
    avg_err_base_quat_y = -1.0;
    avg_err_base_quat_z = -1.0;
    max_err_base_quat   = -1.0;
    max_err_base_quat_x = -1.0;
    max_err_base_quat_y = -1.0;
    max_err_base_quat_z = -1.0;
    min_err_base_quat   = -1.0;
    min_err_base_quat_x = -1.0;
    min_err_base_quat_y = -1.0;
    min_err_base_quat_z = -1.0;
  }

  ///
  /// \brief run, simple method that execute the estimation of the base position
  /// from motion capture and IMU data saved from SL
  ///
  void run(){
    /* *** initialization *** */
    time_ = 0.0;
    /* *** initialize the EKF *** */
    // read the first data
    get_data_from_file(0,
                 input_imu_data_,
                 input_vicon_base_pose_,
                 input_vicon_base_quat_,
                 input_vicon_is_new_frame_);
    // initialize the EKF
    vicon_bse_->initialize(input_vicon_base_pose_);
    // fill in the joint pose from the sensors. We only fill the base here
    output_posture_.joint_positions_.setZero();
    // get the state after initialization
    vicon_bse_->getFilterState(output_imu_position_,
                               output_imu_orientation_,
                               output_imu_linear_velocity_,
                               output_accel_bias_,
                               output_gyro_bias_);
    vicon_bse_->getBaseFromFilterState(
          output_posture_.base_position_,
          output_posture_.base_orientation_,
          output_velocity_.base_linear_velocity(),
          output_velocity_.base_angular_velocity());
    error_base_pose_.row(0) = (output_posture_.base_position_ -
                              input_vicon_base_pose_.topRightCorner(3, 1))
                              .transpose();
    error_base_quat_.row(0) = (output_posture_.base_orientation_.get() -
                              input_vicon_base_quat_.get()).transpose();

    data_collector_.startDataCollection();
    data_collector_.updateDataCollection();

    // main loop
    for(int i=1; i<data_reader_.getNbRows() ; ++i){
      if (i%100 == 0)
      {
        //std::cout << "iteration number " << i << std::endl;
      }
      get_data_from_file(i,
                         input_imu_data_,
                         input_vicon_base_pose_,
                         input_vicon_base_quat_,
                         input_vicon_is_new_frame_);
      vicon_bse_->update(input_imu_data_,
                         input_vicon_base_pose_,
                         input_vicon_is_new_frame_);
      vicon_bse_->getFilterState(output_imu_position_,
                                 output_imu_orientation_,
                                 output_imu_linear_velocity_,
                                 output_accel_bias_,
                                 output_gyro_bias_);
      output_imu_angular_velocty_ = input_imu_data_.gyroscope() -
                                    output_gyro_bias_ ;
      vicon_bse_->getBaseFromFilterState(
            output_posture_.base_position_,
            output_posture_.base_orientation_,
            output_velocity_.base_linear_velocity(),
            output_velocity_.base_angular_velocity());

      time_ += 0.002;
      data_collector_.updateDataCollection();
      error_base_pose_.row(i) = (output_posture_.base_position_ -
                                input_vicon_base_pose_.topRightCorner(3, 1))
                                .transpose();
      error_base_quat_.row(i) = (output_posture_.base_orientation_.get() -
                                input_vicon_base_quat_.get()).transpose();
    }

    compute_all_statistics();
    data_collector_.stopDataCollection();
  }

  void dump(std::string file_name){
    data_collector_.dump(file_name);
  }

  ///
  /// \brief get_data_from_file
  /// \param row: row index homogenous to time/sampling_period.
  /// \param imu_data: output imu data copied from the data set
  /// \param unfiltered_joint_data: joint position copied from the data set
  /// \param filtered_joint_data: filtered joint position copied from the data set
  /// \param vicon_base_pose: position of the based measured form the Vicon system
  /// copied from the data set
  /// \param vicon_is_new_frame: True is the Vicon data if actually a new one.
  ///
  void get_data_from_file(
      const int row,
      robot_estimation::IMUData& imu_data,
      Eigen::Matrix4d& vicon_base_pose,
      geometry_utils::Quaternion &input_vicon_base_quat,
      bool& vicon_is_new_frame){

    // get the ids of the data.
    data_reader_.fillVector(row, gyroscope_ids_,
                            imu_data.gyroscope());
    data_reader_.fillVector(row, accelerometer_ids_,
                            imu_data.accelerometer());
    Eigen::VectorXd vicon_base_pose_vec ;
    vicon_base_pose_vec.resize(16);
    data_reader_.fillVector(row, vicon_base_pose_ids_,
                            vicon_base_pose_vec);
    for(unsigned i=0 ; i<4 ; ++i){
      for(unsigned j=0 ; j<4 ; ++j){
        vicon_base_pose(i, j) = vicon_base_pose_vec(4*i + j);
      }
    }
    assert(vicon_base_pose.bottomRows<1>() ==
           (Eigen::MatrixXd(1,4) << 0, 0, 0, 1).finished() &&
           "This matrix is not a homogeneous matrix");
    input_vicon_base_quat.rotation_matrix_to_quaternion(
          vicon_base_pose.topLeftCorner<3,3>().transpose());
    vicon_is_new_frame = data_reader_.getValue(row, vicon_is_new_frame_id_);
  }

  ///
  /// \brief subscribe_to_data_collector, colelct the address of all variables
  /// to save
  ///
  void subscribe_to_data_collector()
  {
    /* *** intialize *** */
    std::vector<std::string> units;
    std::vector<std::string> names;
    std::vector<std::string> jnames = JOINT_NAMES();
    /* *** time_ *** */
    data_collector_.addVariable(&time_,
                                "time", "-");
    // input
    /* *** input_imu_data_ *** */
    data_collector_.addVector3d(input_imu_data_.gyroscope(),
                               "input_imu_gyro", "rad/s");
    data_collector_.addVector3d(input_imu_data_.accelerometer(),
                                "input_imu_acc", "m/s^2");
    /* *** input_vicon_base_pose_ *** */
    data_collector_.addMatrix(input_vicon_base_pose_,
                              "input_vicon_base_pose", "-");
    /* *** input_vicon_base_pose_ *** */
    data_collector_.addQuaternion(input_vicon_base_quat_.get(),
                                  "input_vicon_base_quat");
    /* *** input_vicon_is_new_frame_ *** */
    data_collector_.addVariable(&input_vicon_is_new_frame_,
                                "input_vicon_is_new_frame", "-");
    // input with original names
    std::string task_name = "AthenaTestTask_";
    /* *** input_imu_data_ *** */
    data_collector_.addVector3d(input_imu_data_.gyroscope(),
                                task_name + "imu_base_gyro", "rad/s");
    data_collector_.addVector3d(input_imu_data_.accelerometer(),
                                task_name + "imu_base_acc", "m/s^2");
    /* *** input_vicon_base_pose_ *** */
    data_collector_.addMatrix(input_vicon_base_pose_,
                              "vicon_hip_frame", "-");
    /* *** input_vicon_is_new_frame_ *** */
    data_collector_.addVariable(&input_vicon_is_new_frame_,
                                "vicon_IMU_BaseEstimator_is_new_frame", "-");


    // output
    /* *** output_imu_position_ *** */
    data_collector_.addVector3d(output_imu_position_,
                               "output_imu_position", "m");
    /* *** output_imu_orientation_ *** */
    data_collector_.addQuaternion(output_imu_orientation_.get(),
                                  "output_imu_orientation");
    /* *** output_imu_linear_velocity_ *** */
    data_collector_.addVector3d(output_imu_linear_velocity_,
                                "output_imu_linear_velocity", "m/s");
    /* *** output_imu_angular_velocty_ *** */
    data_collector_.addVector3d(output_imu_angular_velocty_,
                                "output_imu_angular_velocity", "rad/s");
    /* *** output_base_position_ *** */
    data_collector_.addVector3d(output_posture_.base_position_,
                                "output_base_position", "m");
    /* *** output_base_orientation_ *** */
    data_collector_.addQuaternion(output_posture_.base_orientation_.get(),
                                  "output_base_quat");
    /* *** output_base_linear_velocity_ *** */
    data_collector_.addVector3d(output_velocity_.base_linear_velocity(),
                                "output_base_linear_velocity", "m/s");
    /* *** output_base_angular_velocty_ *** */
    data_collector_.addVector3d(output_velocity_.base_angular_velocity(),
                                "output_base_angular_velocity", "rad/s");
    /* *** output_accel_bias_ *** */
    data_collector_.addVector3d(output_accel_bias_,
                                "output_accel_bias", "m/s^2");
    /* *** input_vicon_is_new_frame_ *** */
    data_collector_.addVector3d(output_gyro_bias_,
                                "output_gyro_bias", "rad/s");
  }

  ///
  /// \brief get_data_indexes
  /// In order to parse the SL data file one need the indexes associated to the
  /// data. This method computes the different names and indexes.
  ///
  void get_data_indexes(){
    // create data stream names.
    std::vector<std::string>
        gyroscope_names (3),
        accelerometer_names (3),
        unfiltered_pos_names (Robot::n_dofs_),
        unfiltered_vel_names (Robot::n_dofs_),
        filtered_pos_names (Robot::n_dofs_),
        filtered_vel_names (Robot::n_dofs_),
        vicon_base_pose_names(16);
    // root name of all the data
    std::string task_name = "AthenaTestTask_";
    std::string estimator_name = "vicon_IMU_BaseEstimator_";
    // get the pos and vel names
    std::vector<std::string> joint_names = JOINT_NAMES();
    for(int i=0; i<Robot::n_dofs_; ++i){
      unfiltered_pos_names[i] = task_name + "jraw_" + joint_names[i+1];
      unfiltered_vel_names[i] = task_name + "djraw_" + joint_names[i+1];
      filtered_pos_names[i] = task_name + "jfilt_" + joint_names[i+1];
      filtered_vel_names[i] = task_name + "djfilt_" + joint_names[i+1];
    }
    // get the imu related names
    gyroscope_names[0] = task_name + "imu_base_gyro_x";
    gyroscope_names[1] = task_name + "imu_base_gyro_y";
    gyroscope_names[2] = task_name + "imu_base_gyro_z";
    accelerometer_names[0] = task_name + "imu_base_acc_x";
    accelerometer_names[1] = task_name + "imu_base_acc_y";
    accelerometer_names[2] = task_name + "imu_base_acc_z";
    // get the name of the vicon hip frame
    for(unsigned i=0 ; i<4 ; ++i){
      for(unsigned j=0 ; j<4 ; ++j){
        std::ostringstream tmp;
        tmp << "vicon_hip_frame_" << i+1 << "_" << j+1 ;
        vicon_base_pose_names[4*i + j] = tmp.str();
      }
    }
    // get the ids of the data.
    data_reader_.getIndexes(gyroscope_names, gyroscope_ids_);
    data_reader_.getIndexes(accelerometer_names, accelerometer_ids_);
    data_reader_.getIndexes(vicon_base_pose_names, vicon_base_pose_ids_);
    vicon_is_new_frame_id_ = data_reader_.getIndex(estimator_name +
                                                   "is_new_frame");
  }

  void display_one_statistics(double avg, double min,
                              double max, std::string name )
  {
    std::cout << "average error in " << name << ": " << avg << std::endl;
    std::cout << "min absolute error in " << name << ": " << min << std::endl;
    std::cout << "max absolute error in " << name << ": " << max << std::endl;
    std::cout << std::endl;
  }

  void display_all_statistics()
  {
    std::cout << "########### Statistics ##########" << std::endl << std::endl;
    display_one_statistics(avg_err_base_pose, min_err_base_pose, max_err_base_pose, "BASE_POSE");
    display_one_statistics(avg_err_base_pose_x, min_err_base_pose_x, max_err_base_pose_x, "BASE_POSE_X");
    display_one_statistics(avg_err_base_pose_y, min_err_base_pose_y, max_err_base_pose_y, "BASE_POSE_Y");
    display_one_statistics(avg_err_base_pose_z, min_err_base_pose_z, max_err_base_pose_z, "BASE_POSE_Z");
    display_one_statistics(avg_err_base_quat, min_err_base_quat, max_err_base_quat, "BASE_QUAT");
    display_one_statistics(avg_err_base_quat_x, min_err_base_quat_x, max_err_base_quat_x, "BASE_QUAT_X");
    display_one_statistics(avg_err_base_quat_y, min_err_base_quat_y, max_err_base_quat_y, "BASE_QUAT_Y");
    display_one_statistics(avg_err_base_quat_z, min_err_base_quat_z, max_err_base_quat_z, "BASE_QUAT_Z");
    std::cout << "#################################" << std::endl << std::endl;
  }

  void compute_all_statistics(){
    compute_one_statistics(error_base_pose_, avg_err_base_pose, min_err_base_pose, max_err_base_pose);
    compute_one_statistics(error_base_pose_.col(0), avg_err_base_pose_x, min_err_base_pose_x, max_err_base_pose_x);
    compute_one_statistics(error_base_pose_.col(1), avg_err_base_pose_y, min_err_base_pose_y, max_err_base_pose_y);
    compute_one_statistics(error_base_pose_.col(2), avg_err_base_pose_z, min_err_base_pose_z, max_err_base_pose_z);
    compute_one_statistics(error_base_quat_.col(0), avg_err_base_quat, min_err_base_quat, max_err_base_quat);
    compute_one_statistics(error_base_quat_.col(1), avg_err_base_quat_x, min_err_base_quat_x, max_err_base_quat_x);
    compute_one_statistics(error_base_quat_.col(2), avg_err_base_quat_y, min_err_base_quat_y, max_err_base_quat_y);
    compute_one_statistics(error_base_quat_.col(3), avg_err_base_quat_z, min_err_base_quat_z, max_err_base_quat_z);
  }

  void compute_one_statistics(Eigen::Ref<Eigen::MatrixXd> mat,
                              double & avg,
                              double & min,
                              double & max){
    Eigen::MatrixXd mat2 = mat;
    avg = mat2.norm() / mat2.size();
    min = mat2.cwiseAbs().minCoeff();
    max = mat2.cwiseAbs().maxCoeff();
  }

  // public stat on the base pose tracking
  double avg_err_base_pose ;
  double avg_err_base_pose_x ;
  double avg_err_base_pose_y ;
  double avg_err_base_pose_z ;
  double max_err_base_pose ;
  double max_err_base_pose_x ;
  double max_err_base_pose_y ;
  double max_err_base_pose_z ;
  double min_err_base_pose ;
  double min_err_base_pose_x ;
  double min_err_base_pose_y ;
  double min_err_base_pose_z ;
  // public stat on the base orientation tracking
  double avg_err_base_quat ;
  double avg_err_base_quat_x ;
  double avg_err_base_quat_y ;
  double avg_err_base_quat_z ;
  double max_err_base_quat ;
  double max_err_base_quat_x ;
  double max_err_base_quat_y ;
  double max_err_base_quat_z ;
  double min_err_base_quat ;
  double min_err_base_quat_x ;
  double min_err_base_quat_y ;
  double min_err_base_quat_z ;

private:
  // parameters
  YAML::Node config_, config_est_;
  double update_frequency_;
  data_collection::DataReader data_reader_;
  data_collection::NonRtDataCollector data_collector_;
  double time_;

  // create data stream ids.
  std::vector<int>
      gyroscope_ids_,
      accelerometer_ids_,
      vicon_base_pose_ids_;
  int vicon_is_new_frame_id_ ;

  // input
  robot_estimation::IMUData input_imu_data_;
  Eigen::Matrix4d input_vicon_base_pose_;
  geometry_utils::Quaternion input_vicon_base_quat_;
  bool input_vicon_is_new_frame_;

  // output
  Eigen::Vector3d output_imu_position_;
  geometry_utils::Quaternion output_imu_orientation_;
  Eigen::Vector3d output_imu_linear_velocity_;
  Eigen::Vector3d output_imu_angular_velocty_;
  Eigen::Vector3d output_accel_bias_, output_gyro_bias_;

  // reconstruct the generalized posture and velocity
  robot_estimation::RobotPosture output_posture_;
  robot_estimation::RobotVelocity output_velocity_;

  // used for statistics
  Eigen::MatrixXd error_base_pose_, error_base_quat_;

  // the class to test
  std::unique_ptr<robot_estimation::EkfViconImu> vicon_bse_ ;
};

} // namespace robot_estimation::test
} // namespace robot_estimation

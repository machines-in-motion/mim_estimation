"""Trotting motion for solo12
License BSD-3-Clause
Copyright (c) 2022, New York University and Max Planck Gesellschaft.
Author: Shahram Khorshidi
"""

from mim_estimation_cpp import RobotStateEstimator, RobotStateEstimatorSettings
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from mim_data_utils import DataLogger, DataReader
from bullet_utils.env import BulletEnvWithGround
import matplotlib.pyplot as plt
import pinocchio as pin
from pathlib import Path
from copy import deepcopy
from numpy import genfromtxt
import numpy as np
import time


def plot(
    data_list,
    legend_list,
    title,
    xlim=(None, None),
    ylim=(None, None),
):
    plt.figure(title)
    max_nb_it = data_list[0].shape[0]
    t = np.arange(max_nb_it, step=1) / 1000
    string = "XYZ"
    for i in range(3):
        plt.subplot(int("31" + str(i + 1)))
        for data, legend in zip(data_list, legend_list):
            plt.plot(t, data[:, i], label=legend, linewidth=0.75)
        plt.ylabel("_" + string[i] + "_")
        plt.xlim(xmin=xlim[0], xmax=xlim[1])
        plt.ylim(ymin=ylim[0], ymax=ylim[1])
        plt.grid()
    plt.legend(loc="upper right", shadow=True, fontsize="medium")
    plt.xlabel("time [sec]")
    plt.suptitle(title)


def demo(sim_time):
    # Create a Pybullet simulation environment.
    env = BulletEnvWithGround()

    # Create a robot instance in the simulator.
    robot = Solo12Robot()
    robot_config = Solo12Config()
    robot = env.add_robot(robot)

    # Reset the robot to some initial state.
    q0 = np.array(robot_config.initial_configuration)
    print(robot_config.initial_configuration)
    q0[0] = 0.0
    dq0 = np.array(robot_config.initial_velocity)
    robot.reset_state(q0, dq0)
    pin_robot = robot.pin_robot
    pin_robot.framesForwardKinematics(q0)
    Kp = 25*np.eye(12)
    Kd = 0.3*np.eye(12)

    # Read the desired trajectory.
    data_folder = Path(__file__).resolve().parent / "solo12_trot"
    tau_ff = genfromtxt(
        str(data_folder / "solo12_feedforward_torque.dat")
    )
    q_des = genfromtxt(
        str(data_folder / "solo12_robot_position.dat")
    )
    dq_des = genfromtxt(
        str(data_folder / "solo12_robot_velocity.dat")
    )

    # Create the Estimator setting instance
    estimator_settings = RobotStateEstimatorSettings()
    estimator_settings.is_imu_frame = False
    estimator_settings.pinocchio_model = robot_config.pin_robot.model
    estimator_settings.imu_in_base = pin.SE3(
        robot.rot_base_to_imu.T, robot.r_base_to_imu
    )
    estimator_settings.end_effector_frame_names = (
        robot_config.end_effector_names
    )
    estimator_settings.urdf_path = robot_config.urdf_path
    robot_weight_per_ee = robot_config.mass * 9.81 / 4
    estimator_settings.force_threshold_up = 0.2 * robot_weight_per_ee
    estimator_settings.force_threshold_down = 0.8 * robot_weight_per_ee
    estimator_settings.meas_noise_cov = np.array([1e-5, 1e-5, 1e-5])

    # Create the estimator and initialize it.
    estimator = RobotStateEstimator()
    estimator.initialize(estimator_settings)

    # Set the initial values of estimator.
    estimator.set_initial_state(
        q0,
        dq0,
    )

    # Initialise data collection
    path_dir = Path("/tmp") / "demo_est_trotting_motion_cpp"
    path_dir.mkdir(parents=True, exist_ok=True)
    out_logger_file_name = str(
        path_dir
        / ("out_" + deepcopy(time.strftime("%Y_%m_%d_%H_%M_%S")) + ".mds")
    )
    out_logger = DataLogger(out_logger_file_name)

    # Input the data fields.
    id_time = out_logger.add_field("sim_time", 1)
    id_sim_base_pos = out_logger.add_field("sim_base_pos", 3)
    id_sim_base_vel = out_logger.add_field("sim_base_vel", 3)
    id_sim_base_rpy = out_logger.add_field("sim_base_rpy", 3)
    id_est_base_pos = out_logger.add_field("est_base_pos", 3)
    id_est_base_vel = out_logger.add_field("est_base_vel", 3)
    id_est_base_rpy = out_logger.add_field("est_base_rpy", 3)
    id_est_ee_force = out_logger.add_field("est_ee_force", 4)
    est_contact = np.zeros(4)

    # Standing for 5 [sec], so the EKF values converge
    for i in range(5000):
        q, dq = robot.get_state()
        tau =  tau_ff[0, :] + Kp @ (q_des[0] - q[7:])+ Kd @ (dq_des[0]- dq[6:])
        robot.send_joint_command(tau)
        env.step(sleep=False)

        # Estimator computation:
        estimator.run(
            robot.get_base_imu_linacc(),
            robot.get_base_imu_angvel(),
            q[7:],
            dq[6:],
            tau,
        )

    # Use higher force thresholds for contact detection in trotting
    estimator_settings.force_threshold_up = 1.6 * robot_weight_per_ee
    estimator_settings.force_threshold_down = 1.45 * robot_weight_per_ee
    estimator.set_settings(estimator_settings)

    # Trotting motion from desired trajectory
    for time_idx in range(sim_time * 1000):
        time_stamp = time_idx * 0.001

        # ------------- Run the CTRL ------------- #
        q, dq = robot.get_state()
        tau =  tau_ff[time_idx, :] + Kp @ (q_des[time_idx] - q[7:])+ Kd @ (dq_des[time_idx]- dq[6:])
        robot.send_joint_command(tau)
        env.step(sleep=False)

        # ---------- Run the Estimator ---------- #
        # Contact detection and running base EKF.
        estimator.run(
            robot.get_base_imu_linacc(),
            robot.get_base_imu_angvel(),
            q[7:],
            dq[6:],
            tau,
        )

        # --------------- Log data --------------- #
        # Read the values of position, velocity and orientation from Robot.
        sim_base_pos = q[:3]
        sim_base_vel = dq[:3]
        q_base = pin.Quaternion(q[3:7])
        sim_base_rpy = pin.utils.matrixToRpy(q_base.matrix())

        # Read the values from Estimator
        q_est = np.zeros(robot_config.pin_robot.nq)
        dq_est = np.zeros(robot_config.pin_robot.nv)
        estimator.get_state(q_est, dq_est)
        est_base_pos = q_est[:3]
        est_base_vel = dq_est[:3]
        est_base_quat = pin.Quaternion(q_est[3:7])
        est_base_rpy = pin.utils.matrixToRpy(est_base_quat.matrix())

        detected_contact = estimator.get_detected_contact()
        for j, ee in enumerate(estimator_settings.end_effector_frame_names):
            est_contact[j] = int(detected_contact[j]) # Estimated contact array

        forces = [
            estimator.get_force(ee)
            for ee in estimator_settings.end_effector_frame_names
        ]
        forces_norm = [np.linalg.norm(f) for f in forces]
        est_ee_force = forces_norm # Estimated end-effectors force

        # Log the data
        out_logger.begin_timestep()
        out_logger.log(id_time, time_stamp)
        out_logger.log(id_sim_base_pos, sim_base_pos)
        out_logger.log(id_sim_base_vel, sim_base_vel)
        out_logger.log(id_sim_base_rpy, sim_base_rpy)
        out_logger.log(id_est_base_pos, est_base_pos)
        out_logger.log(id_est_base_vel, est_base_vel)
        out_logger.log(id_est_base_rpy, est_base_rpy)
        out_logger.end_timestep()
    
    out_logger.close_file()

    # -------------- Plot data -------------- #
    out_reader = DataReader(out_logger_file_name)
    plot(
        [out_reader.data["sim_base_pos"], out_reader.data["est_base_pos"]],
        ["Sim data", "EKF data"],
        "Base_Position",
    )
    plot(
        [out_reader.data["sim_base_vel"], out_reader.data["est_base_vel"]],
        ["Sim data", "EKF data"],
        "Base_Velocity",
    )
    plot(
        [out_reader.data["sim_base_rpy"], out_reader.data["est_base_rpy"]],
        ["Sim data", "EKF data"],
        "Base_Orientation(roll_pitch-yaw)",
    )
    plt.show()


if __name__ == "__main__":
    # Run the demo
    simulation_time = 5 # sec
    demo(simulation_time)
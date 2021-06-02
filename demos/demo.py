"""Squatting motion for solo12

License BSD-3-Clause
Copyright (c) 2020, New York University and Max Planck Gesellschaft.

Author: Majid Khadiv
"""

import argparse
import numpy as np
from mim_control.robot_impedance_controller import RobotImpedanceController
from mim_control.robot_centroidal_controller import RobotCentroidalController
from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
# from robot_properties_bolt.bolt_wrapper import BoltRobot, BoltConfig
# ---------------------------------#
from mim_estimation.ekf import EKF
import mim_estimation.conf as conf
# ---------------------------------$$


def demo(robot_name, sim_time):

    # Create a Pybullet simulation environment
    env = BulletEnvWithGround()

    # Create a robot instance in the simulator.
    if robot_name == "solo":
        robot = env.add_robot(Solo12Robot)
        robot_config = Solo12Config()
        mu = 0.2
        kc = [200, 200, 200]
        dc = [5, 5, 5]
        kb = [200, 200, 200]
        db = [1.0, 1.0, 1.0]
        qp_penalty_lin = 3 * [1e6]
        qp_penalty_ang = 3 * [1e6]
    elif robot_name == "bolt":
        # robot = env.add_robot(BoltRobot)
        # robot_config = BoltConfig()
        mu = 0.2
        kc = [0, 0, 100]
        dc = [0, 0, 10]
        kb = [100, 100, 100]
        db = [10.0, 10.0, 10.0]
        qp_penalty_lin = [1, 1, 1e6]
        qp_penalty_ang = [1e6, 1e6, 1]
    else:
        raise RuntimeError(
            "Robot name [" + str(robot_name) + "] unknown. "
            "Try 'solo' or 'bolt'"
        )

    # Initialize control
    tau = np.zeros(robot.nb_dof)

    # simulation time (ms)
    T = sim_time

    # Reset the robot to some initial state.
    q0 = np.matrix(robot_config.initial_configuration).T
    q0[0] = 0.0
    dq0 = np.matrix(robot_config.initial_velocity).T
    robot.reset_state(q0, dq0)

    # Desired initial center of mass position and velocity.
    x_com = np.zeros((T,3))
    xd_com = np.zeros((T,3))
    x_com[0,:] = [0.0, 0.0, 0.18]
    xd_com[0,:] = [0.0, 0.0, 0.0]

    # Generate squatting motion
    period = [1000, 1000, 1000] #ms
    amplitude = [0.0, 0.0, 0.05] #m
    for j in range(T):
        for i in range(3):
            x_com[j,i] = x_com[0,i] + amplitude[i] * np.sin(2 * np.pi * j / period[i])
            xd_com[j,i] = amplitude[i] * np.cos(2 * np.pi * j / period[i]) * 2 * np.pi

    # The base should be flat.
    x_ori = [0.0, 0.0, 0.0, 1.0]
    x_angvel = [0.0, 0.0, 0.0]
    # Alle end-effectors are in contact.
    cnt_array = robot.nb_ee * [1]

    # Impedance controller gains
    kp = robot.nb_ee * [0.0, 0.0, 0.0]  # Disable for now
    kd = robot.nb_ee * [0.0, 0.0, 0.0]
    x_des = robot.nb_ee * [0.0, 0.0, -q0[2].item()]  # Desired leg length
    xd_des = robot.nb_ee * [0.0, 0.0, 0.0]

    # config_file = "./solo_impedance.yaml"
    config_file = robot_config.paths["imp_ctrl_yaml"]
    robot_cent_ctrl = RobotCentroidalController(
        robot_config,
        mu=mu,
        kc=kc,
        dc=dc,
        kb=kb,
        db=db,
        qp_penalty_lin=qp_penalty_lin,
        qp_penalty_ang=qp_penalty_ang,
    )
    robot_leg_ctrl = RobotImpedanceController(robot, config_file)

    # --------------------------------------------------------##
    # Set all the noise terms to zero
    # robot.base_imu_accel_thermal_noise = 0  # m/s^2/sqrt(Hz)
    # robot.base_imu_gyro_thermal_noise = 0  # rad/s/sqrt(Hz)
    # robot.base_imu_accel_bias_noise = 0  # m/s^3/sqrt(Hz)
    # robot.base_imu_gyro_bias_noise = 0  # rad/s^2/sqrt(Hz)

    # Initialize the vectors for data collecting
    # imu_linacc = np.zeros((T, 3))
    # imu_angvel = np.zeros((T, 3))
    # base_acc = np.zeros((T, 3))
    # base_vel = np.zeros((T, 3))
    # base_angvel = np.zeros((T, 3))
    base_pos_post = np.zeros((T, 3))
    base_vel_post = np.zeros((T, 3))
    # Create the base EKF
    solo_ekf = EKF(conf)
    # ---------------------------------------------------------$$$$$$$$$$

    # Run the simulator for the trajectory
    for i in range(T):
        # Step the simulator.
        env.step(
            sleep=True
        )  # You can sleep here if you want to slow down the replay
        # Read the final state and forces after the stepping.
        q, dq = robot.get_state()
        # computing forces to be applied in the centroidal space
        w_com = robot_cent_ctrl.compute_com_wrench(
            q, dq, x_com[i,:], xd_com[i,:], x_ori, x_angvel
        )
        # distributing forces to the active end effectors
        F = robot_cent_ctrl.compute_force_qp(q, dq, cnt_array, w_com)
        # passing forces to the impedance controller
        tau = robot_leg_ctrl.return_joint_torques(
            q, dq, kp, kd, x_des, xd_des, F
        )
        # passing torques to the robot
        robot.send_joint_command(tau)

    # --------------------------------------------------------##
        # Read the values of IMU and Base
        # imu_linacc[i, :] = robot.get_base_imu_linacc()
        # imu_angvel[i, :] = robot.get_base_imu_angvel()
        # base_acc[i, :] = robot.get_base_acceleration_world()[:3]
        # base_vel[i, 0:3] = robot.get_base_velocity_world()[0:3, 0]
        # base_angvel[i, 0:3] = robot.get_base_velocity_world()[3:6, 0]
        # imu_accel_bias[i, :] = robot.base_imu_accel_bias

        # Run the EKF prediction step
        solo_ekf.integrate_model(robot.get_base_imu_linacc(), robot.get_base_imu_angvel())
        solo_ekf.prediction_step()
        base_state_pre = solo_ekf.get_mu_pre()

        # Run the EKF update step
        contacts_schedule = {'FL': False, 'FR': False, 'HL': False, 'HR': False}
        solo_ekf.update_step(contacts_schedule, q[7:], dq[6:])
        base_state_post = solo_ekf.get_mu_post()

        base_pos_post[i, :] = base_state_post.get("base_position")
        base_vel_post[i, :] = base_state_post.get("base_velocity")

    # return base_pos_post, base_vel_post, x_com, xd_com
    # ---------------------------------------------------------$$$$$$$$$$


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solo", help="Demonstrate Solo.", action="store_true"
    )
    parser.add_argument(
        "--bolt", help="Demonstrate Bolt.", action="store_true"
    )
    args = parser.parse_args()
    if args.solo:
        robot_name = "solo"
    elif args.bolt:
        robot_name = "bolt"
    else:
        robot_name = "solo"

    demo(robot_name, 1000)

"""Squatting motion for solo12
License BSD-3-Clause
Copyright (c) 2020, New York University and Max Planck Gesellschaft.
Author: Maximilien Naveau
"""

import argparse
import numpy as np
from mim_control.robot_impedance_controller import RobotImpedanceController
from mim_control.robot_centroidal_controller import RobotCentroidalController
from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from robot_properties_bolt.bolt_wrapper import BoltRobot, BoltConfig
from mim_estimation_cpp import BaseEkfWithImuKinSettings, BaseEkfWithImuKin
import mim_estimation.conf as conf
import matplotlib.pyplot as plt
import pinocchio


def demo(robot_name, sim_time):

    # Create a Pybullet simulation environment
    env = BulletEnvWithGround()

    # Create a robot instance in the simulator.
    if robot_name == "solo":
        robot = Solo12Robot()
        env.add_robot(robot)
        robot_config = Solo12Config()
        mu = 0.2
        kc = [200, 200, 200]
        dc = [5, 5, 5]
        kb = [200, 200, 200]
        db = [1.0, 1.0, 1.0]
        qp_penalty_lin = 3 * [1e6]
        qp_penalty_ang = 3 * [1e6]
    elif robot_name == "bolt":
        robot = env.add_robot(BoltRobot)
        robot_config = BoltConfig()
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
    x_com = np.zeros((T, 3))
    xd_com = np.zeros((T, 3))
    x_com[0, :] = [0.0, 0.0, 0.18]
    xd_com[0, :] = [0.0, 0.0, 0.0]

    # Generate squatting motion
    period = [1000, 1000, 1000]  # ms
    amplitude = [0.0, 0.0, 0.05]  # m
    for j in range(T):
        for i in range(3):
            x_com[j, i] = x_com[0, i] + amplitude[i] * np.sin(
                2 * np.pi * j / period[i]
            )
            xd_com[j, i] = (
                amplitude[i] * np.cos(2 * np.pi * j / period[i]) * 2 * np.pi
            )

    # The base should be flat.
    x_ori = [0.0, 0.0, 0.0, 1.0]
    x_angvel = [0.0, 0.0, 0.0]
    # All end-effectors are in contact.
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

    # initialize vectors for data collecting
    base_pos = np.zeros((T, 3), float)
    base_vel = np.zeros((T, 3), float)
    base_pos_ekf = np.zeros((T, 3), float)
    base_vel_ekf = np.zeros((T, 3), float)
    rpy_base = np.zeros((T, 3), float)
    rpy_base_ekf = np.zeros((T, 3), float)

    # Create EKF instance and set the SE3 from IMU to Base
    solo_ekf = EKF(conf)
    solo_ekf.set_SE3_imu_in_base(robot.rot_base_to_imu.T, robot.r_base_to_imu)

    # Run the simulator for the trajectory
    for i in range(T):
        # Step the simulator.
        env.step(
            sleep=False
        )  # You can sleep here if you want to slow down the replay
        # Read the final state and forces after the stepping.
        q, dq = robot.get_state()
        # computing forces to be applied in the centroidal space
        w_com = robot_cent_ctrl.compute_com_wrench(
            q, dq, x_com[i, :], xd_com[i, :], x_ori, x_angvel
        )
        # distributing forces to the active end effectors
        F = robot_cent_ctrl.compute_force_qp(q, dq, cnt_array, w_com)
        # passing forces to the impedance controller
        tau = robot_leg_ctrl.return_joint_torques(
            q, dq, kp, kd, x_des, xd_des, F
        )
        # passing torques to the robot
        robot.send_joint_command(tau)

        # -------------- Run the EKF -------------- #
        # Set the initial values of EKF
        if i == 0:
            solo_ekf.set_mu_post("base_position", q[:3])
            solo_ekf.set_mu_post("base_velocity", dq[:3])
            solo_ekf.set_mu_post("base_orientation", pin.Quaternion(q[3:7]))

        # EKF prediction step
        solo_ekf.integrate_model(
            robot.get_base_imu_linacc(), robot.get_base_imu_angvel()
        )
        solo_ekf.prediction_step()

        # EKF update step with all feet in contact
        contacts_schedule = {"FL": True, "FR": True, "HL": True, "HR": True}
        solo_ekf.update_step(contacts_schedule, q[7:], dq[6:])

        # Read the values of position, velocity and orientation of the robot
        base_pos[i, :] = q[:3]
        base_vel[i, :] = dq[:3]
        q_base = q[3:7]
        rpy_base[i, :] = pin.utils.matrixToRpy(pin.Quaternion(q_base).matrix())

        # Read the values of position, velocity and orientation from EKF
        base_state_post = solo_ekf.get_mu_post()
        base_pos_ekf[i, :] = base_state_post.get("base_position")
        base_vel_ekf[i, :] = base_state_post.get("base_velocity")
        q_ekf = base_state_post.get("base_orientation")
        rpy_base_ekf[i, :] = pin.utils.matrixToRpy(q_ekf.matrix())
    return (
        base_pos,
        base_vel,
        base_pos_ekf,
        base_vel_ekf,
        rpy_base,
        rpy_base_ekf,
    )


def plot(x, y, x_legend, y_legend, title):
    t = np.arange(simulation_time)
    string = "XYZ"
    for i in range(3):
        plt.subplot(int("31" + str(i + 1)))
        plt.plot(t, x[:, i], "b", label=x_legend, linewidth=0.75)
        plt.plot(t, y[:, i], "r--", label=y_legend, linewidth=0.75)
        plt.ylabel("_" + string[i] + "_")
        plt.grid()
    plt.legend(loc="upper right", shadow=True, fontsize="large")
    plt.xlabel("time(ms)")
    plt.suptitle(title)
    plt.show()


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

    # Run the demo
    simulation_time = 5000  # ms
    (
        base_pos,
        base_vel,
        base_pos_ekf,
        base_vel_ekf,
        rpy_base,
        rpy_base_ekf,
    ) = demo("solo", simulation_time)

    # Plot the results
    plot(base_pos, base_pos_ekf, "Squatting", "EKF", "Base_Position")
    plot(base_vel, base_vel_ekf, "Squatting", "EKF", "Base_Velocity")
    plot(
        rpy_base,
        rpy_base_ekf,
        "Squatting",
        "EKF",
        "Base_Orientation(roll_pitch-yaw)",
    )

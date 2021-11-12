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
from robot_properties_bolt.bolt_wrapper import BoltRobot, BoltConfig
from mim_estimation.ekf import EKF
import mim_estimation.conf as conf
import matplotlib.pyplot as plt
import pinocchio as pin


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
    try:
        config_file = robot_config.paths["imp_ctrl_yaml"]
    except:    
        config_file = robot_config.resources.imp_ctrl_yaml_path
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
    frame_pos = np.zeros((T, 3), float)
    frame_vel = np.zeros((T, 3), float)
    frame_pos_ekf = np.zeros((T, 3), float)
    frame_vel_ekf = np.zeros((T, 3), float)
    frame_rpy = np.zeros((T, 3), float)
    frame_rpy_ekf = np.zeros((T, 3), float)

    # Create EKF instance, and set the EKF_frame
    solo_ekf = EKF(conf)
    solo_ekf.set_ekf_in_imu_frame(False)

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
        imu_frame_pos, imu_frame_vel = robot.get_imu_frame_position_velocity()
        if i == 0:
            if solo_ekf.get_ekf_frame():
                solo_ekf.set_mu_post("ekf_frame_position", imu_frame_pos)
                solo_ekf.set_mu_post("ekf_frame_velocity", imu_frame_vel)
                ekf_orien = (
                    pin.Quaternion(q[3:7]).matrix() @ robot.rot_base_to_imu.T
                )
                solo_ekf.set_mu_post(
                    "ekf_frame_orientation", pin.Quaternion(ekf_orien)
                )
            else:
                solo_ekf.set_mu_post("ekf_frame_position", q[:3])
                solo_ekf.set_mu_post("ekf_frame_velocity", dq[:3])
                solo_ekf.set_mu_post(
                    "ekf_frame_orientation", pin.Quaternion(q[3:7])
                )

        # EKF prediction step
        solo_ekf.integrate_model(
            robot.get_base_imu_linacc(), robot.get_base_imu_angvel()
        )
        solo_ekf.prediction_step()

        # EKF update step with all feet in contact
        contacts_schedule = {"FL": True, "FR": True, "HL": True, "HR": True}
        solo_ekf.update_step(contacts_schedule, q[7:], dq[6:])

        # Read the values of position, velocity and orientation from the robot
        if solo_ekf.get_ekf_frame():
            frame_pos[i, :] = imu_frame_pos
            frame_vel[i, :] = imu_frame_vel
            frame_rot = (
                pin.Quaternion(q[3:7]).matrix() @ robot.rot_base_to_imu.T
            )
            frame_rpy[i, :] = pin.utils.matrixToRpy(frame_rot)
        else:
            frame_pos[i, :] = q[:3]
            frame_vel[i, :] = dq[:3]
            q_base = pin.Quaternion(q[3:7])
            frame_rpy[i, :] = pin.utils.matrixToRpy(q_base.matrix())

        # Read the values of position, velocity and orientation from EKF
        base_state_post = solo_ekf.get_mu_post()
        frame_pos_ekf[i, :] = base_state_post.get("ekf_frame_position")
        frame_vel_ekf[i, :] = base_state_post.get("ekf_frame_velocity")
        q_ekf = base_state_post.get("ekf_frame_orientation")
        frame_rpy_ekf[i, :] = pin.utils.matrixToRpy(q_ekf.matrix())
    return (
        frame_pos,
        frame_vel,
        frame_rpy,
        frame_pos_ekf,
        frame_vel_ekf,
        frame_rpy_ekf,
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
    simulation_time = 20000  # ms
    (
        frame_pos,
        frame_vel,
        frame_rpy,
        frame_pos_ekf,
        frame_vel_ekf,
        frame_rpy_ekf,
    ) = demo("solo", simulation_time)

    # Plot the results
    plt.figure("Position")
    plot(frame_pos, frame_pos_ekf, "Squatting", "EKF", "EKF_Frame_Position")

    plt.figure("Velocity")
    plot(frame_vel, frame_vel_ekf, "Squatting", "EKF", "EKF_Frame_Velocity")

    plt.figure("Orientation")
    plot(
        frame_rpy,
        frame_rpy_ekf,
        "Squatting",
        "EKF",
        "EKF_Frame_Orientation(roll-pitch-yaw)",
    )

    plt.show()

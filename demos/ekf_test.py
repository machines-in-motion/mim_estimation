"""This module runs the EKF on the data obtained by vicon from solo12 motion. 
Download the data from 'rsec-motion-group' page, in 'ReactiveStepping', Squatting and Wobbling motions for solo12.
Modify the 'path' variable to the path where the data is located.
"""
import numpy as np
import matplotlib.pyplot as plt
from mim_estimation.ekf import EKF, EKF_Vicon
import mim_estimation.conf as conf
import pinocchio as pin
from math import pi


def read_data(file_name):
    data = np.loadtxt(file_name)
    return data[:T, 1:]

def plot(x, y, x_label, y_label, title):
    t = np.arange(T)
    string = "XYZ"
    for i in range(3):
        plt.subplot(int("31" + str(i + 1)))
        plt.plot(t, x[:, i], "b", label=x_label, linewidth=0.75)
        plt.plot(t, y[:, i], "r--", label=y_label, linewidth=0.75)
        plt.ylabel("_" + string[i] + "_")
        plt.grid()
    plt.legend(loc="upper right", shadow=True, fontsize="large")
    plt.xlabel("time(ms)")
    plt.suptitle(title) 

def run_ekf(path):
    # Read the data from robot motion
    imu_lin_acc = read_data(path + "dg_solo12-imu_accelerometer.dat")
    imu_ang_vel = read_data(path + "dg_solo12-imu_gyroscope.dat")
    joint_positions = read_data(path + "dg_solo12-joint_positions.dat")
    joint_velocities = read_data(path + "dg_solo12-joint_velocities.dat")
    base_position = read_data(path + "dg_vicon_entity-solo12_position.dat")
    base_velocity_body = read_data(
        path + "dg_vicon_entity-solo12_velocity_body.dat"
    )

    # Initialize vectors for data collecting
    ekf_pos = np.zeros((T, 3), float)
    ekf_vel = np.zeros((T, 3), float)
    ekf_rpy = np.zeros((T, 3), float)
    ekf_vicon_pos = np.zeros((T, 3), float)
    ekf_vicon_vel = np.zeros((T, 3), float)
    ekf_vicon_rpy = np.zeros((T, 3), float)
    base_rpy = np.zeros((T, 3), float)

    # Create EKF instances and set the initial values
    solo_ekf = EKF(conf)
    solo_ekf.set_mu_post("ekf_frame_position", base_position[0, :3])
    solo_ekf.set_mu_post("ekf_frame_velocity", base_velocity_body[0, :3])
    solo_ekf.set_mu_post("ekf_frame_orientation", pin.Quaternion(base_position[0, 3:]))

    solo_ekf_vicon = EKF_Vicon(conf)
    solo_ekf_vicon.set_mu_post("ekf_frame_position", base_position[0, :3])
    solo_ekf_vicon.set_mu_post("ekf_frame_velocity", base_velocity_body[0, :3])
    solo_ekf_vicon.set_mu_post("ekf_frame_orientation", pin.Quaternion(base_position[0, 3:]))

    for i in range(T):
        # Run the EKF prediction step
        solo_ekf.integrate_model(imu_lin_acc[i, :], imu_ang_vel[i, :])
        solo_ekf.prediction_step()
        solo_ekf_vicon.integrate_model(imu_lin_acc[i, :], imu_ang_vel[i, :])
        solo_ekf_vicon.prediction_step()

        # Run the EKF update step with all feet in contact
        contacts_schedule = {"FL": True, "FR": True, "HL": True, "HR": True}
        solo_ekf.update_step(
            contacts_schedule, joint_positions[i, :], joint_velocities[i, :]
        )
        solo_ekf_vicon.update_step(base_position[i, :3])

        # Read the values of position, velocity and orientation from EKF
        ekf_state_post = solo_ekf.get_mu_post()
        ekf_pos[i, :] = ekf_state_post.get("ekf_frame_position")
        ekf_vel[i, :] = ekf_state_post.get("ekf_frame_velocity")
        q_ekf = ekf_state_post.get("ekf_frame_orientation")
        ekf_rpy[i, :] = pin.utils.matrixToRpy(q_ekf.matrix()) * (180/pi)

        ekf_vicon_state_post = solo_ekf_vicon.get_mu_post()
        ekf_vicon_pos[i, :] = ekf_vicon_state_post.get("ekf_frame_position")
        ekf_vicon_vel[i, :] = ekf_vicon_state_post.get("ekf_frame_velocity")
        q_ekf = ekf_vicon_state_post.get("ekf_frame_orientation")
        ekf_vicon_rpy[i, :] = pin.utils.matrixToRpy(q_ekf.matrix()) * (180/pi)
        
        # Read the value of orientation from the robot
        q_base = pin.Quaternion(base_position[i, 3:])
        base_rpy[i, :] = pin.utils.matrixToRpy(q_base.matrix()) * (180/pi)

    # Plot the results from feet_kinematics measurement
    plt.figure("Position(Feet_Kin)")
    plot(base_position, ekf_pos, "Vicon", "EKF", "Base_Position")

    plt.figure("Velocity(Feet_Kin)")
    plot(base_velocity_body, ekf_vel, "Vicon", "EKF", "Base_Velocity")

    plt.figure("Orientation(Feet_Kin)")
    plot(base_rpy, ekf_rpy, "Vicon", "EKF", "Base_Orientation(roll-pitch-yaw)_degree")

    # Plot the results from vicon measurement
    plt.figure("Position(Vicon)")
    plot(base_position, ekf_vicon_pos, "Vicon", "EKF", "Base_Position")

    plt.figure("Velocity(Vicon)")
    plot(base_velocity_body, ekf_vicon_vel, "Vicon", "EKF", "Base_Velocity")

    plt.figure("Orientation(Vicon)")
    plot(base_rpy, ekf_vicon_rpy, "Vicon", "EKF", "Base_Orientation(roll-pitch-yaw)_degree")

    plt.show()


if __name__ == "__main__":
    path = "/home/skhorshidi/Documents/files/files_first_wobbling/"
    T = 35000
    run_ekf(path)

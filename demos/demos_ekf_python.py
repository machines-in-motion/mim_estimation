"""This module runs the EKF on the data obtained by vicon from solo12 motion. 
Download the data from 'rsec-motion-group' page, in 'ReactiveStepping', Squatting and Wobbling motions for solo12.
Modify the 'path' variable to the path where the data is located.
"""
import numpy as np
import matplotlib.pyplot as plt
from mim_estimation.ekf import EKF
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
    base_pos_ekf = np.zeros((T, 3), float)
    base_vel_ekf = np.zeros((T, 3), float)
    base_rpy_ekf = np.zeros((T, 3), float)
    base_rpy = np.zeros((T, 3), float)

    # Create EKF instance and set the initial values
    solo_ekf = EKF(conf)
    solo_ekf.set_mu_post("ekf_frame_position", base_position[0, :3])
    solo_ekf.set_mu_post("ekf_frame_velocity", base_velocity_body[0, :3])
    solo_ekf.set_mu_post("ekf_frame_orientation", pin.Quaternion(base_position[0, 3:]))

    for i in range(T):
        # Run the EKF prediction step
        solo_ekf.integrate_model(imu_lin_acc[i, :], imu_ang_vel[i, :])
        solo_ekf.prediction_step()

        # Run the EKF update step with all feet in contact
        contacts_schedule = {"FL": True, "FR": True, "HL": True, "HR": True}
        solo_ekf.update_step(
            contacts_schedule, joint_positions[i, :], joint_velocities[i, :]
        )

        # Read the values of position, velocity and orientation from EKF
        base_state_post = solo_ekf.get_mu_post()
        base_pos_ekf[i, :] = base_state_post.get("ekf_frame_position")
        base_vel_ekf[i, :] = base_state_post.get("ekf_frame_velocity")
        q_ekf = base_state_post.get("ekf_frame_orientation")
        base_rpy_ekf[i, :] = pin.utils.matrixToRpy(q_ekf.matrix()) * (180/pi)
        
        # Read the value of orientation from the robot
        q_base = pin.Quaternion(base_position[i, 3:])
        base_rpy[i, :] = pin.utils.matrixToRpy(q_base.matrix()) * (180/pi)

    # Plot the results
    plt.figure("Position")
    plot(base_position, base_pos_ekf, "Vicon", "EKF", "Base_Position")

    plt.figure("Velocity")
    plot(base_velocity_body, base_vel_ekf, "Vicon", "EKF", "Base_Velocity")

    plt.figure("Orientation")
    plot(base_rpy, base_rpy_ekf, "Vicon", "EKF", "Base_Orientation(roll-pitch-yaw)_degree")

    plt.show()


if __name__ == "__main__":
    path = "/home/Documents/files/files_first_wobbling/"
    T = 35000
    run_ekf(path)

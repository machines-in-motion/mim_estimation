import numpy as np
import matplotlib.pyplot as plt
from mim_estimation.ekf import EKF
import mim_estimation.conf as conf
from scipy.spatial.transform import Rotation as Rot
from pinocchio import Quaternion


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
    plt.show()


def plot_results(lable):
    base_pos = np.loadtxt(path + "base_pos")
    base_vel = np.loadtxt(path + "base_vel")
    base_pos_ekf = np.loadtxt(path + "base_pos_ekf")
    base_vel_ekf = np.loadtxt(path + "base_vel_ekf")
    euler_angels_real = np.loadtxt(path + "euler_angels_real")
    euler_angels_ekf = np.loadtxt(path + "euler_angels_ekf")
    plot(base_pos, base_pos_ekf, lable, "ekf", "Position")
    plot(base_vel, base_vel_ekf, lable, "ekf", "Velocity")
    plot(euler_angels_real, euler_angels_ekf, "wobbling", "ekf", "Rotation")


def run_ekf():
    # Read the data from wobbling motion
    imu_lin_acc = read_data(path + "dg_solo12-imu_accelerometer.dat")
    imu_ang_vel = read_data(path + "dg_solo12-imu_gyroscope.dat")
    joint_positions = read_data(path + "dg_solo12-joint_positions.dat")
    joint_velocities = read_data(path + "dg_solo12-joint_velocities.dat")
    base_position = read_data(path + "dg_vicon_entity-solo12_position.dat")
    base_velocity_body = read_data(
        path + "dg_vicon_entity-solo12_velocity_body.dat"
    )

    # Create EKF instance and initialise the position
    solo_ekf = EKF(conf)
    solo_ekf.set_mu_post("base_position", base_position[0, 0:3])
    solo_ekf.set_mu_post("base_orientation", Quaternion(base_position[0, 3:]))
    base_pos_ekf = np.zeros((T, 3), float)
    base_vel_ekf = np.zeros((T, 3), float)
    euler_angels_ekf = np.zeros((T, 3), float)
    euler_angels_real = np.zeros((T, 3), float)
    for i in range(T):
        # Run the EKF prediction step
        solo_ekf.integrate_model(imu_lin_acc[i, :], imu_ang_vel[i, :])
        solo_ekf.prediction_step()

        # Run the EKF update step
        contacts_schedule = {"FL": True, "FR": True, "HL": True, "HR": True}
        # contacts_schedule = {'FL': False, 'FR': False, 'HL': False, 'HR': False}
        solo_ekf.update_step(
            contacts_schedule, joint_positions[i, :], joint_velocities[i, :]
        )
        # solo_ekf.update_step(contacts_schedule, base_position[i, :], joint_positions[i, :], joint_velocities[i, :])
        base_state_post = solo_ekf.get_mu_post()
        base_pos_ekf[i, :] = base_state_post.get("base_position")
        base_vel_ekf[i, :] = base_state_post.get("base_velocity")
        q = base_state_post.get("base_orientation")
        euler_angels_ekf[i, :] = Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler(
            "xyz", degrees=True
        )
        euler_angels_real[i, :] = Rot.from_quat(base_position[i, 3:]).as_euler(
            "xyz", degrees=True
        )
    np.savetxt(path + "base_pos", base_position)
    np.savetxt(path + "base_vel", base_velocity_body)
    np.savetxt(path + "base_pos_ekf", base_pos_ekf)
    np.savetxt(path + "base_vel_ekf", base_vel_ekf)
    np.savetxt(path + "euler_angels_real", euler_angels_real)
    np.savetxt(path + "euler_angels_ekf", euler_angels_ekf)


if __name__ == "__main__":
    # T = 24000
    # path = "files_squatting/"
    # run_ekf()
    # plot_results("squatting")

    # T = 35000
    # path = "files_first_wobbling/"
    # run_ekf()
    # plot_results("wobbling")

    T = 33000
    path = "files_second_wobbling/"
    # run_ekf()
    plot_results("wobbling")

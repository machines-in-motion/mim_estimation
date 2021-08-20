"""Squatting motion for solo12
License BSD-3-Clause
Copyright (c) 2020, New York University and Max Planck Gesellschaft.
Author: Maximilien Naveau
"""

from copy import deepcopy
import time
from pathlib import Path
import numpy as np
import pinocchio
from mim_data_utils import DataLogger, DataReader

from solo12_simulations import SimuController
from mim_estimation_cpp import BaseEkfWithImuKinSettings, BaseEkfWithImuKin


def demo(nb_iteration):
    # -------------- Init CTRL and Est ------------- #
    ctrl = SimuController()
    est_settings = BaseEkfWithImuKinSettings()
    est_settings.is_imu_frame = False
    est_settings.pinocchio_model = ctrl.robot_config.pin_robot.model
    est_settings.imu_in_base = pinocchio.SE3(
        ctrl.robot.rot_base_to_imu.T, ctrl.robot.r_base_to_imu
    )
    est_settings.end_effector_frame_names = [
        "FL_ANKLE",
        "FR_ANKLE",
        "HL_ANKLE",
        "HR_ANKLE",
    ]
    # Create the ekf and initialize it.
    ekf = BaseEkfWithImuKin()
    ekf.initialize(est_settings)

    # -------------- Init data collection ------------- #
    path_dir = Path("/tmp") / "demo_est_squatting_motion_cpp"
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
    id_base_vel = {}
    for ee in est_settings.end_effector_frame_names:
        id_base_vel[ee] = out_logger.add_field("est_base_vel_" + ee, 3)
    out_logger.init_file()

    # Set the initial values of est
    ekf.set_initial_state(
        ctrl.q0[:7],
        ctrl.dq0[:6],
    )

    for i in range(nb_iteration):

        time_stamp = i * 0.001

        # -------------- Run the CTRL ------------- #
        ctrl.run_squatting_motion(i)
        sim_imu_linacc = ctrl.out_imu_linacc.copy()
        sim_imu_angvel = ctrl.out_imu_angvel.copy()
        sim_joint_position = ctrl.out_joint_position.copy()
        sim_joint_velocity = ctrl.out_joint_velocity.copy()
        sim_base_pos = ctrl.out_base_pos.copy()
        sim_base_vel = ctrl.out_base_vel.copy()
        sim_base_rpy = ctrl.out_base_rpy.copy()

        # -------------- Run the EKF -------------- #
        # EKF computation:
        contacts_schedule = [True, True, True, True]
        ekf.update_filter(
            contacts_schedule,
            sim_imu_linacc,
            sim_imu_angvel,
            sim_joint_position,
            sim_joint_velocity,
        )

        # -------------- Log data -------------- #
        # extract some data
        q_ekf = np.zeros(ctrl.robot_config.pin_robot.nq)
        dq_ekf = np.zeros(ctrl.robot_config.pin_robot.nv)
        ekf.get_filter_output(q_ekf, dq_ekf)
        base_lin_vels = ekf.get_measurement()
        base_attitude_ekf = pinocchio.Quaternion(q_ekf[3:7])
        rpy_base_ekf = pinocchio.utils.matrixToRpy(base_attitude_ekf.matrix())
        # Log the ekf data
        out_logger.begin_timestep()
        out_logger.log(id_time, time_stamp)
        out_logger.log(id_sim_base_pos, sim_base_pos)
        out_logger.log(id_sim_base_vel, sim_base_vel)
        out_logger.log(id_sim_base_rpy, sim_base_rpy)
        out_logger.log(id_est_base_pos, q_ekf[:3])
        out_logger.log(id_est_base_vel, dq_ekf[:3])
        out_logger.log(id_est_base_rpy, rpy_base_ekf)
        for id_ee, ee in enumerate(est_settings.end_effector_frame_names):
            out_logger.log(id_base_vel[ee], base_lin_vels[id_ee])
        out_logger.end_timestep()
    out_logger.close_file()

    # -------------- Plot data -------------- #
    from matplotlib import pyplot as plt

    out_reader = DataReader(out_logger_file_name)

    def plot(
        data_list,
        legend_list,
        title,
        xlim=(None, None),
        ylim=(None, None),
    ):
        plt.figure(title)
        max_nb_it = data_list[0].shape[0]
        t = np.arange(max_nb_it, step=1)
        string = "XYZ"
        for i in range(3):
            plt.subplot(int("31" + str(i + 1)))
            for data, legend in zip(data_list, legend_list):
                plt.plot(t, data[:, i], label=legend, linewidth=0.75)
            plt.ylabel("_" + string[i] + "_")
            plt.xlim(xmin=xlim[0], xmax=xlim[1])
            plt.ylim(ymin=ylim[0], ymax=ylim[1])
            plt.grid()
        plt.legend(loc="upper right", shadow=True, fontsize="large")
        plt.xlabel("time(ms)")
        plt.suptitle(title)

    # Plot the results
    plot(
        [out_reader.data["sim_base_pos"], out_reader.data["est_base_pos"]],
        ["Sim data", "est data"],
        "Base_Position",
        # ylim=(-2, 2),
    )
    plot(
        [out_reader.data["sim_base_vel"], out_reader.data["est_base_vel"]],
        ["Sim data", "est data"],
        "Base_Velocity",
        # ylim=(-2, 2),
    )
    plot(
        [out_reader.data["sim_base_rpy"], out_reader.data["est_base_rpy"]],
        ["Sim data", "est data"],
        "Base_Orientation(roll_pitch-yaw)",
        # ylim=(-2*np.pi, 2*np.pi),
    )
    plot(
        [
            out_reader.data["est_base_vel_" + ee]
            for ee in est_settings.end_effector_frame_names
        ],
        [ee for ee in est_settings.end_effector_frame_names],
        "Measured Base Vel",
    )
    plt.show()


if __name__ == "__main__":
    # Run the demo
    demo(3000)  # ms

"""Squatting motion for solo12
License BSD-3-Clause
Copyright (c) 2020, New York University and Max Planck Gesellschaft.
Author: Maximilien Naveau
"""

import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import pinocchio
from mim_data_utils import DataLogger, DataReader

from solo12_simulations import SimuController
from mim_estimation_cpp import RobotStateEstimator, RobotStateEstimatorSettings


def demo(nb_iteration):
    # Create the controller.
    ctrl = SimuController()
    # Create the Estimator instance.
    estimator_settings = RobotStateEstimatorSettings()
    estimator_settings.is_imu_frame = False
    estimator_settings.pinocchio_model = ctrl.robot_config.pin_robot.model
    estimator_settings.imu_in_base = pinocchio.SE3(
        ctrl.robot.rot_base_to_imu.T, ctrl.robot.r_base_to_imu
    )
    estimator_settings.end_effector_frame_names = (
        ctrl.robot_config.end_effector_names
    )
    estimator_settings.urdf_path = ctrl.robot_config.urdf_path
    robot_weight_per_ee = ctrl.robot_config.mass * 9.81 / 4
    estimator_settings.force_threshold_up = 0.8 * robot_weight_per_ee
    estimator_settings.force_threshold_down = 0.2 * robot_weight_per_ee
    print("force_threshold_up = ", estimator_settings.force_threshold_up)
    print("force_threshold_down = ", estimator_settings.force_threshold_down)

    # Create the estimator and initialize it.
    estimator = RobotStateEstimator()
    estimator.initialize(estimator_settings)

    # -------------- Init data collection ------------- #
    path_dir = Path("/tmp") / "demo_est_squatting_motion_cpp"
    path_dir.mkdir(parents=True, exist_ok=True)
    logger_file_name = str(
        path_dir
        / ("out_" + deepcopy(time.strftime("%Y_%m_%d_%H_%M_%S")) + ".mds")
    )
    logger = DataLogger(logger_file_name)
    # Input the data fields.
    id_time = logger.add_field("sim_time", 1)
    id_sim_imu_linacc = logger.add_field("sim_imu_linacc", 3)
    id_sim_imu_angvel = logger.add_field("sim_imu_angvel", 3)
    id_sim_base_pos = logger.add_field("sim_base_pos", 3)
    id_sim_base_vel = logger.add_field("sim_base_vel", 3)
    id_sim_base_rpy = logger.add_field("sim_base_rpy", 3)
    id_est_base_pos = logger.add_field("est_base_pos", 3)
    id_est_base_vel = logger.add_field("est_base_vel", 3)
    id_est_base_rpy = logger.add_field("est_base_rpy", 3)
    id_est_force = {}
    id_est_contact = {}
    id_est_force_norm = {}
    for ee in estimator_settings.end_effector_frame_names:
        id_est_force[ee] = logger.add_field("est_" + ee + "_force", 3)
        id_est_contact[ee] = logger.add_field("est_" + ee + "_contact", 1)
        id_est_force_norm[ee] = logger.add_field(
            "est_" + ee + "_force_norm", 1
        )
    logger.init_file()

    # Set the initial values of est
    estimator.set_initial_state(
        ctrl.q0,
        ctrl.dq0,
    )

    for i in range(nb_iteration):
        time_sec = i * 0.001

        # -------------- Run controller ------------------#
        # ctrl.run_fake_walking_motion(i, gait="standing_gait")
        # ctrl.run_fake_walking_motion(i, gait="static_walking_gait")
        ctrl.run_fake_walking_motion(i, gait="static_walking_gait")

        # -------------- Run the estimator -------------- #
        # print(ctrl.out_contact_array)
        # est computation:
        estimator.run(
            ctrl.out_contact_array,
            ctrl.out_imu_linacc,
            ctrl.out_imu_angvel,
            ctrl.out_joint_position,
            ctrl.out_joint_velocity,
        )

        # Read the values of position, velocity and orientation from est
        q_est = np.zeros(ctrl.robot_config.pin_robot.nq)
        dq_est = np.zeros(ctrl.robot_config.pin_robot.nv)
        estimator.get_state(q_est, dq_est)
        detected_contact = estimator.get_detected_contact()
        forces = [
            estimator.get_force(ee)
            for ee in estimator_settings.end_effector_frame_names
        ]
        forces_norm = [np.linalg.norm(f) for f in forces]

        # Log the simu data
        logger.begin_timestep()
        logger.log(id_time, time_sec)
        logger.log(id_sim_imu_linacc, ctrl.out_imu_linacc.copy())
        logger.log(id_sim_imu_angvel, ctrl.out_imu_angvel.copy())
        logger.log(id_sim_base_pos, ctrl.out_base_pos.copy())
        logger.log(id_sim_base_vel, ctrl.out_base_vel.copy())
        logger.log(id_sim_base_rpy, ctrl.out_base_rpy.copy())
        # log the estimator data.
        base_attitude_est = pinocchio.Quaternion(q_est[3:7])
        rpy_base_est = pinocchio.utils.matrixToRpy(base_attitude_est.matrix())
        logger.log(id_est_base_pos, q_est[:3])
        logger.log(id_est_base_vel, dq_est[:3])
        logger.log(id_est_base_rpy, rpy_base_est)
        for j, ee in enumerate(estimator_settings.end_effector_frame_names):
            logger.log(id_est_force[ee], forces[j])
            logger.log(id_est_contact[ee], detected_contact[j])
            logger.log(id_est_force_norm[ee], forces_norm[j])
        logger.end_timestep()
    logger.close_file()

    # ------------ plot data ---------#
    import matplotlib.pyplot as plt

    reader = DataReader(logger_file_name)

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
        [reader.data["sim_base_pos"], reader.data["est_base_pos"]],
        ["Sim data", "est data"],
        "Base_Position",
        # ylim=(-2, 2),
    )
    plot(
        [reader.data["sim_base_vel"], reader.data["est_base_vel"]],
        ["Sim data", "est data"],
        "Base_Velocity",
        # ylim=(-2, 2),
    )
    plot(
        [reader.data["sim_base_rpy"], reader.data["est_base_rpy"]],
        ["Sim data", "est data"],
        "Base_Orientation(roll_pitch-yaw)",
        # ylim=(-2*np.pi, 2*np.pi),
    )
    # for ee, f in zip(estimator_settings.end_effector_frame_names, forces):
    #     title = "Force and contact (" + ee + ")"
    #     plt.figure(title)
    #     max_nb_it = reader.data["est_" + ee + "_force"].shape[0]
    #     t = np.arange(max_nb_it, step=1)
    #     plt.subplot(311)
    #     plt.plot(
    #         t,
    #         reader.data["est_" + ee + "_force"][:, 0],
    #         label="Fx",
    #         linewidth=0.75,
    #     )
    #     plt.plot(
    #         t,
    #         reader.data["est_" + ee + "_force"][:, 1],
    #         label="Fy",
    #         linewidth=0.75,
    #     )
    #     plt.plot(
    #         t,
    #         reader.data["est_" + ee + "_force"][:, 2],
    #         label="Fz",
    #         linewidth=0.75,
    #     )
    #     plt.legend(loc="upper right", shadow=True, fontsize="large")
    #     plt.grid()
    #     plt.subplot(312)
    #     plt.plot(
    #         t,
    #         reader.data["est_" + ee + "_force_norm"][:, 0],
    #         label="Force norm",
    #         linewidth=0.75,
    #     )
    #     plt.grid()
    #     plt.legend(loc="upper right", shadow=True, fontsize="large")
    #     plt.subplot(313)
    #     plt.plot(
    #         t,
    #         reader.data["est_" + ee + "_contact"][:, 0],
    #         label="Contact",
    #         linewidth=0.75,
    #     )
    #     plt.grid()
    #     plt.legend(loc="upper right", shadow=True, fontsize="large")

    #     plt.xlabel("time(ms)")
    #     plt.suptitle(title)

    plt.show()


if __name__ == "__main__":
    # Run the demo
    demo(3000)

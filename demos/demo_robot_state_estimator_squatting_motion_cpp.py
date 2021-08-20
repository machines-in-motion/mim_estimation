"""Squatting motion for solo12
License BSD-3-Clause
Copyright (c) 2020, New York University and Max Planck Gesellschaft.
Author: Maximilien Naveau
"""

import argparse
import numpy as np
from solo12_simulations import (
    SimuController,
    DataCollection,
)
from mim_estimation_cpp import RobotStateEstimator, RobotStateEstimatorSettings
import matplotlib.pyplot as plt
import pinocchio





class DataCollection(object):
    def __init__(self, max_nb_it):
        self.max_nb_it = max_nb_it
        self.data = {}

    def collect_data(self, data_name, data):
        if not (data_name) in self.data:
            self.data[data_name] = np.array(data)
        else:
            self.data[data_name] = np.vstack([self.data[data_name], data])


def demo(robot_name, nb_iteration):
    # Create the controller.
    ctrl = SimuController(robot_name)
    # Initialize the data collection.
    logger = DataCollection(nb_iteration)
    # Create the Estimator instance.
    estimator_settings = RobotStateEstimatorSettings()
    estimator_settings.is_imu_frame = False
    estimator_settings.pinocchio_model = ctrl.robot_config.pin_robot.model
    estimator_settings.imu_in_base = pinocchio.SE3(
        ctrl.robot.rot_base_to_imu.T, ctrl.robot.r_base_to_imu
    )
    estimator_settings.end_effector_frame_names = [
        "FL_ANKLE",
        "FR_ANKLE",
        "HL_ANKLE",
        "HR_ANKLE",
    ]
    estimator_settings.urdf_path = ctrl.robot_config.urdf_path
    robot_weight_per_ee = ctrl.robot_config.mass * 9.81 / 4
    estimator_settings.force_threshold_up = 0.8 * robot_weight_per_ee
    estimator_settings.force_threshold_down = 0.2 * robot_weight_per_ee

    print("force_threshold_up = ", estimator_settings.force_threshold_up)
    print("force_threshold_down = ", estimator_settings.force_threshold_down)

    # Create the estimator and initialize it.
    estimator = RobotStateEstimator()
    estimator.initialize(estimator_settings)

    for i in range(nb_iteration):
        # -------------- Run controller ------------------#
        ctrl.run_squatting_motion(i)

        # -------------- Run the estimator -------------- #
        # Set the initial values of est
        if i == 0:
            estimator.set_initial_state(
                ctrl.out_q,
                ctrl.out_dq,
            )

        # est computation:
        estimator.run(
            ctrl.out_imu_linacc,
            ctrl.out_imu_angvel,
            ctrl.out_joint_position,
            ctrl.out_joint_velocity,
            ctrl.out_joint_torque,
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
        logger.collect_data("sim_imu_linacc", ctrl.out_imu_linacc.copy())
        logger.collect_data("sim_imu_angvel", ctrl.out_imu_angvel.copy())
        logger.collect_data(
            "sim_joint_position", ctrl.out_joint_position.copy()
        )
        logger.collect_data(
            "sim_joint_velocity", ctrl.out_joint_velocity.copy()
        )
        logger.collect_data("sim_base_pos", ctrl.out_base_pos.copy())
        logger.collect_data("sim_base_vel", ctrl.out_base_vel.copy())
        logger.collect_data("sim_base_rpy", ctrl.out_base_rpy.copy())
        # log the estimator data.
        base_attitude_est = pinocchio.Quaternion(q_est[3:7])
        rpy_base_est = pinocchio.utils.matrixToRpy(base_attitude_est.matrix())
        logger.collect_data("est_base_pos", q_est[:3])
        logger.collect_data("est_base_vel", dq_est[:3])
        logger.collect_data("est_base_rpy", rpy_base_est)
        for i, (ee, f) in enumerate(
            zip(estimator_settings.end_effector_frame_names, forces)
        ):
            logger.collect_data("est_" + ee + "_force", f)
            logger.collect_data("est_" + ee + "_contact", detected_contact[i])
            logger.collect_data("est_" + ee + "_force_norm", forces_norm[i])

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
        [logger.data["sim_base_pos"], logger.data["est_base_pos"]],
        ["Sim data", "est data"],
        "Base_Position",
        # ylim=(-2, 2),
    )
    plot(
        [logger.data["sim_base_vel"], logger.data["est_base_vel"]],
        ["Sim data", "est data"],
        "Base_Velocity",
        # ylim=(-2, 2),
    )
    plot(
        [logger.data["sim_base_rpy"], logger.data["est_base_rpy"]],
        ["Sim data", "est data"],
        "Base_Orientation(roll_pitch-yaw)",
        # ylim=(-2*np.pi, 2*np.pi),
    )
    for ee, f in zip(estimator_settings.end_effector_frame_names, forces):
        title = "Force and contact (" + ee + ")"
        plt.figure(title)
        max_nb_it = logger.data["est_" + ee + "_force"].shape[0]
        t = np.arange(max_nb_it, step=1)
        plt.subplot(311)
        plt.plot(
            t,
            logger.data["est_" + ee + "_force"][:, 0],
            label="Fx",
            linewidth=0.75,
        )
        plt.plot(
            t,
            logger.data["est_" + ee + "_force"][:, 1],
            label="Fy",
            linewidth=0.75,
        )
        plt.plot(
            t,
            logger.data["est_" + ee + "_force"][:, 2],
            label="Fz",
            linewidth=0.75,
        )
        plt.legend(loc="upper right", shadow=True, fontsize="large")
        plt.grid()
        plt.subplot(312)
        plt.plot(
            t,
            logger.data["est_" + ee + "_force_norm"][:, 0],
            label="Force norm",
            linewidth=0.75,
        )
        plt.grid()
        plt.legend(loc="upper right", shadow=True, fontsize="large")
        plt.subplot(313)
        plt.plot(
            t,
            logger.data["est_" + ee + "_contact"][:, 0],
            label="Contact",
            linewidth=0.75,
        )
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
    args = parser.parse_args()
    if args.solo:
        robot_name = "solo"
    else:
        robot_name = "solo"

    # Run the demo
    simulation_time = 10000  # ms
    demo(robot_name, simulation_time)

"""Squatting motion for solo12
License BSD-3-Clause
Copyright (c) 2020, New York University and Max Planck Gesellschaft.
Author: Maximilien Naveau
"""

import copy
import pickle
from pathlib import Path
import argparse
import numpy as np
from mim_control.robot_impedance_controller import RobotImpedanceController
from mim_control.robot_centroidal_controller import RobotCentroidalController
from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from robot_properties_bolt.bolt_wrapper import BoltRobot, BoltConfig
from mim_estimation_cpp import BaseEkfWithImuKinSettings, BaseEkfWithImuKin
import matplotlib.pyplot as plt
import pinocchio


class SimuController(object):
    def __init__(self, robot_name):
        # copy args
        self.robot_name = robot_name

        # Create a Pybullet simulation environment
        self.env = BulletEnvWithGround()

        # Create a robot instance in the simulator.
        if self.robot_name == "solo":
            self.robot = Solo12Robot()
            self.env.add_robot(self.robot)
            self.robot_config = Solo12Config()
            self.mu = 0.2
            self.kc = [200, 200, 200]
            self.dc = [5, 5, 5]
            self.kb = [200, 200, 200]
            self.db = [1.0, 1.0, 1.0]
            self.qp_penalty_lin = 3 * [1e6]
            self.qp_penalty_ang = 3 * [1e6]
        elif robot_name == "bolt":
            self.robot = self.env.add_robot(BoltRobot)
            self.robot_config = BoltConfig()
            self.mu = 0.2
            self.kc = [0, 0, 100]
            self.dc = [0, 0, 10]
            self.kb = [100, 100, 100]
            self.db = [10.0, 10.0, 10.0]
            self.qp_penalty_lin = [1, 1, 1e6]
            self.qp_penalty_ang = [1e6, 1e6, 1]
        else:
            raise RuntimeError(
                "Robot name [" + str(self.robot_name) + "] unknown. "
                "Try 'solo' or 'bolt'"
            )

        # Initialize control
        self.tau = np.zeros(self.robot.nb_dof)

        # Reset the robot to some initial state.
        self.q0 = np.array(self.robot_config.initial_configuration)
        self.q0[0] = 0.0
        self.dq0 = np.array(self.robot_config.initial_velocity)
        self.robot.reset_state(self.q0, self.dq0)

        # Desired initial center of mass position and velocity.
        self.com_pos_init = np.array([0.0, 0.0, 0.18])
        self.com_vel_init = np.array([0.0, 0.0, 0.0])

        # Generate squatting motion
        self.period = [1000, 1000, 1000]  # ms
        self.amplitude = [0.0, 0.0, 0.05]  # m

        # The base should be flat.
        self.x_ori = [0.0, 0.0, 0.0, 1.0]
        self.x_angvel = [0.0, 0.0, 0.0]
        # All end-effectors are in contact.
        self.cnt_array = self.robot.nb_ee * [1]

        # Impedance controller gains
        self.kp = self.robot.nb_ee * [0.0, 0.0, 0.0]  # Disable for now
        self.kd = self.robot.nb_ee * [0.0, 0.0, 0.0]

        # Desired leg length
        self.x_des = self.robot.nb_ee * [0.0, 0.0, -self.q0[2].item()]
        self.xd_des = self.robot.nb_ee * [0.0, 0.0, 0.0]

        # config_file = "./solo_impedance.yaml"
        config_file = self.robot_config.paths["imp_ctrl_yaml"]
        self.robot_cent_ctrl = RobotCentroidalController(
            self.robot_config,
            mu=self.mu,
            kc=self.kc,
            dc=self.dc,
            kb=self.kb,
            db=self.db,
            qp_penalty_lin=self.qp_penalty_lin,
            qp_penalty_ang=self.qp_penalty_ang,
        )
        self.robot_leg_ctrl = RobotImpedanceController(self.robot, config_file)

    def _com_squatting_motion(self, t):
        com_pos = np.zeros(3, float)
        com_vel = np.zeros(3, float)
        for i in range(3):
            com_pos[i] = self.com_pos_init[i] + self.amplitude[i] * np.sin(
                2 * np.pi * t / self.period[i]
            )
            com_vel[i] = self.com_vel_init[i] + self.amplitude[
                i
            ] * 2 * np.pi * np.cos(2 * np.pi * t / self.period[i])
        return com_pos, com_vel

    def run_squatting_motion(self, t):
        # Step the simulator.
        self.env.step(
            sleep=False
        )  # You can sleep here if you want to slow down the replay
        # Read the final state and forces after the stepping.
        q, dq = self.robot.get_state()
        # Get the squatting motion of the base
        com_pos, com_vel = self._com_squatting_motion(t)
        # computing forces to be applied in the centroidal space
        w_com = self.robot_cent_ctrl.compute_com_wrench(
            q, dq, com_pos, com_vel, self.x_ori, self.x_angvel
        )
        # distributing forces to the active end effectors
        F = self.robot_cent_ctrl.compute_force_qp(q, dq, self.cnt_array, w_com)
        # passing forces to the impedance controller
        tau = self.robot_leg_ctrl.return_joint_torques(
            q, dq, self.kp, self.kd, self.x_des, self.xd_des, F
        )
        # passing torques to the robot
        self.robot.send_joint_command(tau)

        # collect sim data
        self.out_q = q.copy()
        self.out_dq = dq.copy()
        self.out_imu_linacc = self.robot.get_base_imu_linacc()
        self.out_imu_angvel = self.robot.get_base_imu_angvel()
        self.out_joint_position = q[7:]
        self.out_joint_velocity = dq[6:]
        self.out_base_pos = q[:3]
        self.out_base_vel = dq[:3]
        quat_base = q[3:7]
        self.out_base_rpy = pinocchio.utils.matrixToRpy(
            pinocchio.Quaternion(quat_base).matrix()
        )


class DataCollection(object):
    def __init__(self, max_nb_it):
        self.max_nb_it = max_nb_it
        self.data = {}
        # data folder for sim data caching
        self.path_dir = Path("/tmp") / "demo_ekf_squatting_motion_cpp"
        self.path_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file_name = "cached_data.pkl"
        self.cache_file_path = self._file_path(self.cache_file_name)
        # nb figures
        self.nb_figures = 0

    def collect_data(self, data_name, it, data):
        if not (data_name) in self.data:
            self.data[data_name] = np.array(data)
        else:
            self.data[data_name] = np.vstack([self.data[data_name], data])

    def clear_ekf_data(self):
        data = copy.deepcopy(self.data)
        for key in data:
            if key.startswith("ekf_"):
                self.clear_data(key)

    def clear_data(self, data_name):
        if not (data_name) in self.data:
            return
        else:
            self.data.pop(data_name)

    def _file_path(self, file_name):
        if not file_name.endswith(".pkl"):
            file_name += ".pkl"
        file_path = self.path_dir / file_name
        return str(file_path)

    def dump_data(self):
        with open(self.cache_file_path, "wb") as f:
            pickle.dump(self.data, f)

    def load_data(self):
        with open(self.cache_file_path, "rb") as f:
            self.data = pickle.load(f)

    def has_cache(self):
        found = Path(self.cache_file_path).exists()
        if found:
            print("Found data cache in: ", self.cache_file_path)
        else:
            print("Data cache not found.")
        return found

    def plot(
        self,
        data_list,
        legend_list,
        title,
        xlim=(None, None),
        ylim=(None, None),
    ):
        plt.figure(title)
        t = np.arange(self.max_nb_it)
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

    def plot_all(self):
        # Plot the results
        self.plot(
            [self.data["sim_base_pos"], self.data["ekf_base_pos"]],
            ["Sim data", "EKF data"],
            "Base_Position",
            # ylim=(-2, 2),
        )
        self.plot(
            [self.data["sim_base_vel"], self.data["ekf_base_vel"]],
            ["Sim data", "EKF data"],
            "Base_Velocity",
            # ylim=(-2, 2),
        )
        self.plot(
            [self.data["sim_base_rpy"], self.data["ekf_base_rpy"]],
            ["Sim data", "EKF data"],
            "Base_Orientation(roll_pitch-yaw)",
            # ylim=(-2*np.pi, 2*np.pi),
        )
        self.plot(
            [
                self.data["ekf_root_velocities[" + str(ee) + "]"]
                for ee in range(4)
            ],
            ["ekf_root_velocities[" + str(ee) + "]" for ee in range(4)],
            "Measured base velocities",
        )
        plt.show()


def demo(robot_name, nb_iteration):
    # Create the controller.
    ctrl = SimuController(robot_name)
    # Initialize the data collection.
    logger = DataCollection(nb_iteration)

    use_cache = True

    # Look for some cache files.
    if logger.has_cache() and use_cache:
        # If there is a cache we load it and use it.
        logger.load_data()
        logger.clear_ekf_data()

    else:
        # if not we run the simu
        for i in range(nb_iteration):
            ctrl.run_squatting_motion(i)
            logger.collect_data("sim_q", i, ctrl.out_q.copy())
            logger.collect_data("sim_dq", i, ctrl.out_dq.copy())
            logger.collect_data(
                "sim_imu_linacc", i, ctrl.out_imu_linacc.copy()
            )
            logger.collect_data(
                "sim_imu_angvel", i, ctrl.out_imu_angvel.copy()
            )
            logger.collect_data(
                "sim_joint_position", i, ctrl.out_joint_position.copy()
            )
            logger.collect_data(
                "sim_joint_velocity", i, ctrl.out_joint_velocity.copy()
            )
            logger.collect_data("sim_base_pos", i, ctrl.out_base_pos.copy())
            logger.collect_data("sim_base_vel", i, ctrl.out_base_vel.copy())
            logger.collect_data("sim_base_rpy", i, ctrl.out_base_rpy.copy())

    # Create EKF instance and set the SE3 from IMU to Base
    ekf_settings = BaseEkfWithImuKinSettings()
    ekf_settings.is_imu_frame = False
    ekf_settings.pinocchio_model = ctrl.robot_config.pin_robot.model
    ekf_settings.imu_in_base = pinocchio.SE3(
        ctrl.robot.rot_base_to_imu.T, ctrl.robot.r_base_to_imu
    )
    ekf_settings.end_effector_frame_names = [
        "FL_ANKLE",
        "FR_ANKLE",
        "HL_ANKLE",
        "HR_ANKLE",
    ]
    # Create the ekf and initialize it.
    ekf = BaseEkfWithImuKin()
    ekf.initialize(ekf_settings)

    # Run the Ekf on the simulation data.
    for i in range(nb_iteration):
        # -------------- Run the EKF -------------- #
        # Set the initial values of EKF
        if i == 0:
            ekf.set_initial_state(
                logger.data["sim_q"][i, :7],
                logger.data["sim_dq"][i, :6],
            )

        # EKF computation:
        contacts_schedule = [True, True, True, True]
        ekf.update_filter(
            contacts_schedule,
            logger.data["sim_imu_linacc"][i, :],
            logger.data["sim_imu_angvel"][i, :],
            logger.data["sim_joint_position"][i, :],
            logger.data["sim_joint_velocity"][i, :],
        )

        # Read the values of position, velocity and orientation from EKF
        q_ekf = np.zeros(ctrl.robot_config.pin_robot.nq)
        dq_ekf = np.zeros(ctrl.robot_config.pin_robot.nv)
        ekf.get_filter_output(q_ekf, dq_ekf)
        root_velocities = ekf.get_measurement()

        # Log the ekf data
        base_attitude_ekf = pinocchio.Quaternion(q_ekf[3:7])
        rpy_base_ekf = pinocchio.utils.matrixToRpy(base_attitude_ekf.matrix())
        logger.collect_data("ekf_base_pos", i, q_ekf[:3])
        logger.collect_data("ekf_base_vel", i, dq_ekf[:3])
        logger.collect_data("ekf_base_rpy", i, rpy_base_ekf)
        for ee, vel in enumerate(root_velocities):
            logger.collect_data(
                "ekf_root_velocities[" + str(ee) + "]", i, vel
            )

    logger.dump_data()
    logger.plot_all()
    return


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
    demo("solo", simulation_time)

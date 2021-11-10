"""Squatting motion for solo12
License BSD-3-Clause
Copyright (c) 2020, New York University and Max Planck Gesellschaft.
Author: Maximilien Naveau
"""

import numpy as np
import pinocchio
from mim_control.robot_impedance_controller import RobotImpedanceController
from mim_control.robot_centroidal_controller import RobotCentroidalController
from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config


class SimuController(object):
    def __init__(self):
        # Create a Pybullet simulation environment
        self.env = BulletEnvWithGround()

        # Create a robot instance in the simulator.
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
        try:
            config_file = self.robot_config.paths["imp_ctrl_yaml"]
        except:
            config_file = self.robot_config.resources.imp_ctrl_yaml_path

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
        self.env.step()
        # Read the final state and forces after the stepping.
        q, dq = self.robot.get_state()
        # Get the squatting motion of the base
        com_pos, com_vel = self._com_squatting_motion(t)
        # computing forces to be applied in the centroidal space
        w_com = self.robot_cent_ctrl.compute_com_wrench(
            q, dq, com_pos, com_vel, self.x_ori, self.x_angvel
        )
        # distributing forces to the active end effectors
        cnt_array = 4 * [True]
        F = self.robot_cent_ctrl.compute_force_qp(q, dq, cnt_array, w_com)
        # passing forces to the impedance controller
        tau = self.robot_leg_ctrl.return_joint_torques(
            q, dq, self.kp, self.kd, self.x_des, self.xd_des, F
        )
        # passing torques to the robot
        self.robot.send_joint_command(tau)

        # collect sim data
        self.out_joint_torque = tau.copy()
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

    def run_fake_walking_motion(self, t, gait="standing_gait"):
        # run squatting motion
        self.run_squatting_motion(t)
        # Get the gait
        if gait == "standing_gait":
            self.cnt_array = self._standing_gait()
        elif gait == "static_walking_gait":
            self.cnt_array = self._static_walking_gait(t)
        elif gait == "trotting_gait":
            self.cnt_array = self._trotting_gait(t)

        # collect sim data
        self.out_contact_array = self.cnt_array

    def _standing_gait(self):
        return [True, True, True, True]

    def _static_walking_gait(self, iteration):
        t = iteration / 0.001
        phase_period = 0.5
        phase = int(t / phase_period)
        state = phase % 4
        if state == 0:
            return [True, True, True, False]
        elif state == 1:
            return [True, True, False, True]
        elif state == 2:
            return [True, False, True, True]
        elif state == 3:
            return [False, True, True, True]
        else:
            print("Error in gait generation. Using all contacts.")
            return [True, True, True, True]

    def _trotting_gait(self, t):
        phase_period = 0.5
        phase = int(t / phase_period)
        state = phase % 2
        if state == 0:
            return [False, True, True, False]
        elif state == 1:
            return [True, False, False, True]

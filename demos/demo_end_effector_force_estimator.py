"""Demo of force estimation on the teststand (hopper) robot.
License BSD-3-Clause
Copyright (c) 2020, New York University and Max Planck Gesellschaft.
"""

import sys
from pathlib import Path
import numpy as np
from numpy import genfromtxt
from robot_properties_teststand.config import TeststandConfig
from mim_estimation_cpp import EndEffectorForceEstimator


def main(argv):

    # collect data
    data_folder = Path(__file__).resolve().parent / "2019-07-04_16-25-06"
    joint_positions = genfromtxt(
        str(data_folder / "dg_hopper_teststand-joint_positions.dat")
    )
    joint_torques = genfromtxt(
        str(data_folder / "dg_hopper_teststand-joint_torques.dat")
    )
    forces = genfromtxt(str(data_folder / "dg_hopper_teststand-ati_force.dat"))

    max_time = min(
        joint_positions.shape[0],
        joint_torques.shape[0],
        forces.shape[0],
    )
    joint_positions = joint_positions[:max_time, :]
    joint_torques = joint_torques[:max_time, :]
    forces = forces[:max_time, :]

    # reconstruct the vectors.
    joint_positions = joint_positions[:, 1:]
    joint_torques = joint_torques[:, 1:]
    forces = forces[:, 1:]

    # Robot model.
    robot_config = TeststandConfig()

    # Create the estimator.
    ee_force_estimator = EndEffectorForceEstimator()
    ee_force_estimator.initialize(
        robot_config.urdf_path_no_prismatic, ["contact"]
    )

    estimated_force = []

    # Estimate forces.
    for i in range(max_time):
        ee_force_estimator.run(joint_positions[i, :], joint_torques[i, :])

        estimated_force += [ee_force_estimator.get_force("contact").copy()]

    # Plot the data
    import matplotlib.pyplot as plt

    def compare_plot(t, x, y, x_legend, y_legend, title):
        plt.figure(title)
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

    compare_plot(
        range(max_time),
        forces,
        np.vstack(estimated_force),
        "forces",
        "estimated_force",
        "Force estimation.",
    )
    print(estimated_force)
    plt.show()


if __name__ == "__main__":
    main(sys.argv.copy())

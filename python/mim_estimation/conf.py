import numpy as np
import pinocchio as pin

robot_name = "solo12"
GUI = "Gepetto"

# Frame names
base_link_name = "base_link"
end_effectors_frame_names = {
    "FL": "FL_ANKLE",
    "FR": "FR_ANKLE",
    "HL": "HL_ANKLE",
    "HR": "HR_ANKLE",
}

# SE3 transformation from IMU to Base for solo12
SE3_imu_to_base = pin.SE3(np.identity(3), np.array([0.10407, -0.00635, 0.01540]))

# noise covariances (TO-BE-TUNED)
# prediction noise
eta_a = (0.00078 ** 2) * np.ones([3])
eta_omega = (0.000523 ** 2) * np.ones([3])
eta_b_a = (0.0001 ** 2) * np.ones([3])
eta_b_omega = (0.000618 ** 2) * np.ones([3])

Q_a = np.diag(eta_a)
Q_omega = np.diag(eta_omega)
Qb_a = np.diag(eta_b_a)
Qb_omega = np.diag(eta_b_omega)

# measurement noise
R = np.zeros((12, 12), dtype=float)
np.fill_diagonal(R, np.array([1e-5, 1e-5, 1e-8]))

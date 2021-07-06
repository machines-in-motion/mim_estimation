import numpy as np

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

# gravity vector
g_vector = np.zeros(3)
g_vector[2] = -9.81

# discretization time (s)
dt = 0.001

# noise covariances (TO-BE-TUNED)
# prediction noise
eta_a = 0.00078 ** 2 * np.array([dt, dt, dt])
eta_omega = 0.000523 ** 2 * np.array([dt, dt, dt])
eta_b_a = 0.0001 ** 2 * np.array([1, 1, 1])
eta_b_omega = 0.000618 ** 2 * np.array([1, 1, 1])

Q_a = np.diag(eta_a)
Q_omega = np.diag(eta_omega)
Qb_a = np.diag(eta_b_a)
Qb_omega = np.diag(eta_b_omega)

# measurement noise
R = np.zeros((12, 12), dtype=float)
np.fill_diagonal(R, np.array([1e-5, 1e-5, 1e-8]))

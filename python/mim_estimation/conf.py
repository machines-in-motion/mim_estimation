import numpy as np
robot_name = 'solo12'
GUI = 'Gepetto'

#Frame names
base_link_name = 'base_link'
end_effectors_frame_names = {'FL':'FL_ANKLE', 'FR':'FR_ANKLE', 'HL':'HL_ANKLE', 'HR':'HR_ANKLE'}

#gravity vector
g_vector = np.zeros(3)
g_vector[2] = -9.81
dt = 0.1

#noise covariances (TO-BE-TUNED)
#prediction noise
eta_a = np.array([0.01**2, 0.01**2, 0.01**2])
eta_omega = np.array([0.01**2, 0.01**2, 0.01**2])
eta_b_a = np.array([0.01**2, 0.01**2, 0.01**2])
eta_b_omega = np.array([0.01**2, 0.01**2, 0.01**2])
Q_a = np.dot(np.eye(3), eta_a)
Qb_a = np.dot(np.eye(3), eta_b_a)
Q_omega = np.dot(np.eye(3), eta_omega)
Qb_omega = np.dot(np.eye(3), eta_b_omega)

# measurement noise
R = 0.01*(np.eye(12))


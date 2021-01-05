import numpy as np
robot_name = 'solo12'
GUI = 'Gepetto'

#Frame names
base_link_name = 'base_link'
FL_end_effector_frame_name ='FL_ANKLE'
FR_end_effector_frame_name = 'FR_ANKLE'
HL_end_effector_frame_name = 'HL_ANKLE'
HR_end_effector_frame_name = 'HR_ANKLE'

#gravity vector
g_vector = np.zeros(3)
g_vector[2] = -9.81
dt = 0.1

#imu offset?

#noise covariances (TO-BE-TUNED)
#prediction noise
var_qf = np.array([0.01**2, 0.01**2, 0.01**2])
var_qw = np.array([0.01**2, 0.01**2, 0.01**2])
var_q_p1 = np.array([0.01**2, 0.01**2, 0.01**2])
var_q_p2 = np.array([0.01**2, 0.01**2, 0.01**2])
var_q_p3 = np.array([0.01**2, 0.01**2, 0.01**2])
var_q_p4 = np.array([0.01**2, 0.01**2, 0.01**2])
var_q_bf = np.array([0.01**2, 0.01**2, 0.01**2])
var_q_bw = np.array([0.01**2, 0.01**2, 0.01**2])

Qf = np.dot(np.eye(3), var_qf)
Qw = np.dot(np.eye(3), var_qw)
Q_p1 = np.dot(np.eye(3), var_q_p1)
Q_p2 = np.dot(np.eye(3), var_q_p2)
Q_p3 = np.dot(np.eye(3), var_q_p3)
Q_p4 = np.dot(np.eye(3), var_q_p4)
Q_bf = np.dot(np.eye(3), var_q_bf)
Q_bw = np.dot(np.eye(3), var_q_bw)

#feet position noise based on slippage and kinematics errors (TO-BE-TUNED)
R = 0.01*np.eye(12) 
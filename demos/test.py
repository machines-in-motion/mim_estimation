import numpy as np

T = 20000
path = "files_standing/"
imu_lin_acc = np.zeros((T, 3), float)
imu_ang_vel = np.zeros((T, 3), float)
joint_positions = np.zeros((T, 12), float)
joint_velocities = np.zeros((T, 12), float)
position = np.zeros((T, 7), float)

for i in range(20000):
    imu_lin_acc[:, 2] = 9.81
    joint_positions[:, :] = [-0.0494384, 0.728283, -1.51256, 0.0538153, 0.755718, -1.50994, -0.0318702, -0.737972,
                             1.49963, 0.0258502, -0.73721, 1.52228]
    position[:, :] = [-0.665763, -0.0062731, 0.250201, 0.0, 0.0, 0.0, 1]

np.savetxt(path + "dg_solo12-imu_accelerometer.dat", imu_lin_acc)
np.savetxt(path + "dg_solo12-imu_gyroscope.dat", imu_ang_vel)
np.savetxt(path + "dg_solo12-joint_positions.dat", joint_positions)
np.savetxt(path + "dg_solo12-joint_velocities.dat", joint_velocities)
np.savetxt(path + "dg_vicon_entity-solo12_position.dat", position)

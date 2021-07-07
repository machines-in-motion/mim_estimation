from mim_estimation_cpp import BaseEkfWithImuKinSettings, BaseEkfWithImuKin
from robot_properties_solo.config import Solo12Config

# get the pinocchio model
robot_config = Solo12Config()

# create the ekf settings.
s = BaseEkfWithImuKinSettings()
s.pinocchio_model = robot_config.pin_robot.model

# create the ekf and apply the settings.
ekf = BaseEkfWithImuKin()
ekf.initialize(s)

print(s)

mim_estimation
-----------

# Introduction

This package contains estimation algorithm C++ implementations with their Python
bindings and their Python prototypes.

# Getting started

### Installation

#### Dependencies

```
- Pinocchio
- Mim_Control (Optional, needed to run demos)
- Matplotlib (Optional, needed to run demos)
- BulletUtils (Optional, needed to run demos)
- Robot_Properties_Solo (Optional, needed to run demos)
- Robot_Properties_Bolt (Optional, needed to run demos)
```

#### Download the package

External dependencies:
See [this tutorial](https://github.com/machines-in-motion/machines-in-motion.github.io/wiki/laas_package_from_binaries)
in order to install `pinocchio` or you can follow [this tutorial](https://github.com/machines-in-motion/ubuntu_installation_scripts) 
to install related robotpkg software.
We use treep in order to manage the download of our code base. See [treep_installation](https://github.com/machines-in-motion/treep_machines_in_motion) for 
the installation, and [treep](https://gitlab.is.tue.mpg.de/amd-clmc/treep), 
and [colcon](https://github.com/machines-in-motion/machines-in-motion.github.io/wiki/use_colcon) 
for their usage.

Local and specific dependencies and the actual repo we need to compile:
```
mkdir devel
cd devel
pip install -U treep
git clone git@github.com:machines-in-motion/treep_machines_in_motion
treep --clone mim_estimation
```

For full installation of the package and related dependencies, use following:
```
treep --clone MIM_ESTIMATION
```

With this command all dependencies needed to run the demos are also downloaded, 
see [this tutorial](https://github.com/machines-in-motion/mim_control) to 
install related dependencies for Mim_Control package.

#### Build the package

We use [colcon](https://github.com/machines-in-motion/machines-in-motion.github.io/wiki/use_colcon)
to build this package:
```
cd devel/workspace
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select mim_estimation
```

Source the environment:
```
source ~/devel/workspace/install/setup.bash
```

### Usage

#### Running Demos

To run the base EKF estimator on Solo12 with pre-defined contact schedule, follow 
the below steps:
```
source /opt/openrobots/setup.bash (source open robots)
cd demos
python3 demo_ekf_squatting_motion_cpp.py
```

To run the robot state estimator (including contact-detection and base-EKF) on 
Solo12, follow the below steps:
```
source /opt/openrobots/setup.bash (source open robots)
cd demos
python3 demo_ekf_trotting_motion_cpp.py
```

# License

BSD 3-Clause

# Authors

- Maximilien Naveau (mnaveau@tue.mpg.de)
- Julian Viereck (jviereck@tue.mpg.de)
- Avadesh Meduri (ameduri@tue.mpg.de)
- Majid Kadhiv (mkadhiv@tue.mpg.de)
- Ludovic Righetti (lrighetti@tue.mpg.de)
- Nick Rotella
- Brahayam Ponton

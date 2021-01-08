from pyquaternion import Quaternion
import example_robot_data as robex
from numpy.linalg import inv
import pinocchio as pin 
import numpy as np
import conf

class EKF:
    # constructor
    def __init__(self, conf):
        # private members
        self.__robot = robex.load(conf.robot_name)
        self.__rmodel = self.__robot.model
        self.__rdata = self.__rmodel.createData() 
        self.__nx = 27 
        self.__init_robot_config = np.copy(self.__robot.q0)  
        self.__dt = conf.dt             # discretization time
        self.__g_vector = conf.g_vector # gravity acceleration vector                     
        self.__base_frame_name = conf.base_link_name
        self.__p1_frame_name = conf.FL_end_effector_frame_name
        self.__p2_frame_name = conf.FR_end_effector_frame_name
        self.__p3_frame_name = conf.HL_end_effector_frame_name
        self.__p4_frame_name = conf.HR_end_effector_frame_name
        self.__mu = dict.fromkeys(['base_position', 'base_velocity', 'base_orientation',
                                   'p1_position', 'p2_position','p3_position','p4_position',
                                    'bias_acceletation', 'bias_orientation'])
        self.__Sigma = np.zeros((self.__nx, self.__nx)) #27x27                            
        # call private methods
        self.__init_viewer(conf.GUI) 
        self.__init_filter()

    #private methods
    def __init_filter(self):
        C = self.__compute_base_pose_se3(self.__init_robot_config)
        end_effectors_positions = self.__compute_end_effectors_positions(self.__init_robot_config)
        self.__mu['base_position'] = C.translation  
        self.__mu['base_velocity'] = np.zeros(3)
        self.__mu['base_orientation'] = Quaternion(matrix = C.rotation)
        self.__mu['p1_position'] = end_effectors_positions['p1']
        self.__mu['p2_position'] = end_effectors_positions['p2']
        self.__mu['p3_position'] = end_effectors_positions['p3']
        self.__mu['p4_position'] = end_effectors_positions['p4']
        self.__mu['bias_acceletation'] = np.zeros(3)
        self.__mu['bias_orientation'] = np.zeros(3)
        
    def __init_viewer(self, gui):
        if gui == 'Gepetto':
            Viewer = pin.visualize.GepettoVisualizer
        elif gui == 'Meshcat':
            Viewer = pin.visualize.MeshcatVisualizer
        self.__viewer = Viewer(self.__rmodel, self.__robot.collision_model, 
                                                  self.__robot.visual_model)  

    def __compute_base_pose_se3(self, robot_configuration):
        pin.framesForwardKinematics(self.__rmodel, self.__rdata, robot_configuration)
        base_link_index = self.__rmodel.getFrameId(self.__base_frame_name)
        return self.__rdata.oMf[base_link_index]  

    def __compute_end_effectors_positions(self, robot_configuration):
        pin.framesForwardKinematics(self.__rmodel, self.__rdata, robot_configuration)
        p1_index = self.__rmodel.getFrameId(self.__p1_frame_name)
        p1_position = self.__rdata.oMf[p1_index].translation
        p2_index = self.__rmodel.getFrameId(self.__p2_frame_name)
        p2_position = self.__rdata.oMf[p2_index].translation
        p3_index = self.__rmodel.getFrameId(self.__p3_frame_name)
        p3_position = self.__rdata.oMf[p3_index].translation
        p4_index = self.__rmodel.getFrameId(self.__p4_frame_name)
        p4_position = self.__rdata.oMf[p4_index].translation  
        return {'p1':p1_position,'p2':p2_position,
                'p3':p3_position,'p4':p4_position}      

    # public methods
    # Accesors     
    def get_mu(self):return self.__mu
    def get_base_position(self):return self.__mu['base_position'] 
    def get_base_velocity(self):return self.__mu['base_velocity']
    def get_base_orientation(self):return self.__mu['base_orientation']
    def get_p1_position(self):return self.__mu['p1_position']
    def get_p2_position(self):return self.__mu['p2_position']
    def get_p3_position(self):return self.__mu['p3_position']
    def get_p4_position(self):return self.__mu['p4_position']
    def get_bias_acceleration(self):return self.__mu['bias_acceletation']
    def get_bias_orientation(self):return self.__mu['bias_orientation']
    
    # expected motion model (mean of the state vector)
    def transient_model(self, f_tilde, w_tilde):
        dt, g_vector = self.__dt, self.__g_vector
        C = self.get_base_orientation().rotation_matrix
        bf, bw = self.__mu['bias_acceletation'], self.__mu['bias_orientation'] 
        # expected IMU readings in the inertial frame 
        f = np.dot(C.T, (f_tilde - bf)) + g_vector # acceletation 
        w = w_tilde - bw                           # angular velocity 
        # discrete nonlinear motion model                      
        self.__mu['base_position'] = self.__mu['base_position'] + dt*self.__mu['base_velocity'] + (0.5*dt**2)*f
        self.__mu['base_velocity'] = self.__mu['base_velocity'] + dt*f 
        self.__mu['base_orientation'].integrate(w, dt)
        return f, w
    
    # jacobians 
    def compute_prediction_jacobian(self, f, w):
        dt = self.__dt
        Fk = np.eye(self.__nx, dtype=float) #27x27
        C = self.get_base_orientation().rotation_matrix
        Ct = C.T
        Fk[0:3,3:6] = dt*np.eye(3)
        Fk[3:6,6:9] = dt*np.dot(-Ct, pin.skew(f))
        Fk[3:6,21:24] = dt*-Ct
        Fk[6:9,6:9] = np.eye(3)-(dt*pin.skew(w))
        Fk[6:9,24:27] = -dt*np.eye(3)
        return Fk 

    def compute_measurement_jacobian(self):
        C = self.get_base_orientation().rotation_matrix
        p1 = self.get_p1_position()-self.get_base_position()
        p2 = self.get_p2_position()-self.get_base_position()
        p3 = self.get_p3_position()-self.get_base_position()
        p4 = self.get_p4_position()-self.get_base_position()
        Hk = np.zeros((12, self.__nx))  #12x27
        Hk[0:3,0:3] = Hk[3:6,0:3] = Hk[6:9,0:3] =  Hk[9:12,0:3] = -C
        Hk[0:3,9:12] = Hk[3:6,12:15] = Hk[6:9,15:18] = Hk[9:12,18:21] = C
        Hk[0:3,6:9] = pin.skew(np.dot(C, p1))
        Hk[3:6,6:9] = pin.skew(np.dot(C, p2))
        Hk[6:9,6:9] = pin.skew(np.dot(C, p3))
        Hk[9:12,6:9] = pin.skew(np.dot(C, p4))
        return Hk
    
    def compute_noise_jacobian(self):
        C = self.get_base_orientation().rotation_matrix
        Ct = C.T
        Lc = np.zeros((self.__nx,  self.__nx-3)) #27x24
        Lc[3:6, 0:3] = -Ct
        Lc[6:9, 3:6] = -np.eye(3)
        Lc[9:12,6:9] = Ct
        Lc[12:15, 9:12] = Ct
        Lc[15:18, 12:15] = Ct
        Lc[18:21, 15:18] = Ct
        Lc[21:24, 18:21] = np.eye(3)
        Lc[24:27, 21:24] = np.eye(3)
        return Lc     
    
    # additive white noise 
    def construct_continuous_noise_covariance(self, conf):
        Qc = np.eye(self.__nx-3, dtype=float) #24x24
        Qc[0:3, 0:3] = conf.Qf
        Qc[3:6, 3:6] = conf.Qw
        Qc[6:9, 6:9] = conf.Q_p1
        Qc[9:12,9:12] = conf.Q_p2
        Qc[12:15, 12:15] = conf.Q_p3
        Qc[15:18, 15:18] = conf.Q_p4
        Qc[18:21, 18:21] = conf.Q_bf
        Qc[21:24, 21:24] = conf.Q_bw
        return Qc
    
    # using zero-order hold and truncating higher-order terms
    def construct_discrete_noise_covariance(self, Fk, Lc, Qc):
        Qk_left = np.dot(np.dot(Fk, Lc), Qc)    
        Qk_right = np.dot(Lc.T, Fk.T)
        Qk = np.dot(Qk_left, Qk_right)*self.__dt
        return Qk
    
    def construct_discrete_measurement_noise_covariance(self, conf):
        R = (1/self.__dt)*conf.R
        return R
    
    # progagate covariance 
    def prediction_step(self, Fk, Qk):
        self.__Sigma = np.dot(np.dot(Fk, self.__Sigma), Fk.T) + Qk 
    
    def predict_measurements(self):
        z = np.zeros(12, dtype=float)
        C = self.get_base_orientation().rotation_matrix
        z[0:3] = np.dot(C, (self.get_p1_position()-self.get_base_position()))
        z[3:6] = np.dot(C, (self.get_p2_position()-self.get_base_position()))
        z[6:9] = np.dot(C, (self.get_p3_position()-self.get_base_position()))
        z[9:12] = np.dot(C, (self.get_p4_position()-self.get_base_position()))
        return z
    
    def get_new_measurments(self, robot_configuration):
        s = np.zeros(12, dtype=float)
        end_effectors_positions = self.__compute_end_effectors_positions(robot_configuration)
        C = self.__compute_base_pose_se3(robot_configuration).rotation_matrix
        r = C.translation 
        s[0:3] = np.dot(C, (end_effectors_positions['p1']-r))
        s[3:6] = np.dot(C, (end_effectors_positions['p2']-r))
        s[6:9] = np.dot(C, (end_effectors_positions['p3']-r))
        s[9:12] = np.dot(C, (end_effectors_positions['p4']-r))
        return s

    #update mean and covariance based on new foot measurements
    #TODO update mu and sigma
    #TODO update only the feet in contact to avoid seeing dragons
    def update_step(self, conf, robot_configuration, Hk, Rk):
        Sk = np.dot(np.dot(Hk, self.__Sigma), Hk.T) + Rk #innovation covariance 
        K = np.dot(np.dot(self.__Sigma, Hk.T), inv(Sk))  #kalman gain
        z = self.predict_measurements()
        s = self.get_new_measurments(robot_configuration)
        return 0              

if __name__=='__main__':
    import numpy.random as random
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from numpy import nditer
    print('visualize your jacobians like a neo ! '.center(60,'*'))
    solo_EKF = EKF(conf)
    f_tilde = random.rand(3)
    w_tilde = random.rand(3)
    
    f,w = solo_EKF.transient_model(f_tilde, w_tilde)
    Fk = solo_EKF.compute_prediction_jacobian(f, w)
    Lc = solo_EKF.compute_noise_jacobian()
    Qc = solo_EKF.construct_continuous_noise_covariance(conf)
    Qk = solo_EKF.construct_discrete_noise_covariance(Fk, Lc, Qc)
    solo_EKF.prediction_step(Fk, Qk)
    Hk = solo_EKF.compute_measurement_jacobian()

    with nditer(Fk, op_flags=['readwrite']) as it:
        for x in it:
            if x[...] != 0:
                x[...] = 1
    with nditer(Lc, op_flags=['readwrite']) as it:
        for y in it:
            if y[...] != 0:
                y[...] = 1
    with nditer(Hk, op_flags=['readwrite']) as it:
        for z in it:
            if z[...] != 0:
                z[...] = 1                             
    plt.figure(1)
    plt.suptitle('Structure of prediction Jacobian Fk')
    plt.imshow(Fk, cmap='Greys', extent=[0,Fk.shape[1],Fk.shape[0],0],
    interpolation = 'nearest')
    plt.figure(2)
    plt.suptitle('Structure of noise Jacobian Lc')
    plt.imshow(Lc, cmap='Greys', extent=[0,Lc.shape[1],Lc.shape[0],0],
    interpolation = 'nearest')
    plt.figure(3)
    plt.suptitle('Structure of measurement Jacobian Hk')
    plt.imshow(Hk, cmap='Greys', extent=[0,Hk.shape[1],Hk.shape[0],0],
    interpolation = 'nearest')
    plt.show()
    


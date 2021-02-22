import example_robot_data as robex
from pinocchio import Quaternion
from numpy.linalg import inv
from numpy import random
import pinocchio as pin 
import numpy as np
import conf

# plus and minus operators on SO3
def box_plus(R, theta):return R @ pin.exp(theta)
def box_minus(R_plus, R):return pin.log(R.T @ R_plus)

class EKF:
    # constructor
    def __init__(self, conf):
        # private members
        self.__robot = robex.load(conf.robot_name)
        self.__rmodel = self.__robot.model
        self.__rdata = self.__rmodel.createData() 
        self.__nx = 5*3 # delta_x = [delta_r, delta_v, delta_theta, delta_b_a, delta_b_omega] 
        self.__init_robot_config = np.copy(self.__robot.q0)  
        self.__dt = conf.dt             # discretization time
        self.__g_vector = conf.g_vector # gravity acceleration vector                     
        self.__base_frame_name = conf.base_link_name
        self.__end_effectors_frame_names = conf.end_effectors_frame_names
        self.__mu_pre = dict.fromkeys(['base_position', 'base_velocity', 'base_orientation',
                                        'bias_acceletation', 'bias_orientation'])
        self.__mu_post = dict.fromkeys(['base_position', 'base_velocity', 'base_orientation',
                                        'bias_acceletation', 'bias_orientation']) 
        self.__Sigma_pre = np.zeros((self.__nx, self.__nx))  #15x15
        self.__Sigma_post = np.zeros((self.__nx, self.__nx)) #15x15                            
        self.__Q_a = conf.Q_a              
        self.__Q_omega = conf.Q_omega   
        self.__Qb_a =  conf.Qb_a
        self.__Qb_omega =  conf.Qb_omega
        self.__R = conf.R     
        # call private methods
        self.__init_viewer(conf.GUI) 
        self.__init_filter()

    #private methods
    def __init_filter(self):
        joint_positions = self.__init_robot_config[7::]
        joint_velocities = np.zeros(12)
        M = self.__compute_base_pose_se3(self.__init_robot_config)
        R_worldTobase = M.rotation.T  
        q = Quaternion(R_worldTobase)
        q.normalize()
        end_effectors_positions, _ = self.compute_end_effectors_FK_quantities(joint_positions, joint_velocities)
        self.__end_effectors_positions_post = end_effectors_positions
        self.__mu_post['base_position'] = -M.translation  
        self.__mu_post['base_velocity'] = np.zeros(3, dtype=float)
        self.__mu_post['base_orientation'] = q
        self.__mu_post['bias_acceletation'] = np.zeros(3, dtype=float)
        self.__mu_post['bias_orientation'] = np.zeros(3, dtype=float)
        self.__mu_pre['bias_acceletation'] = np.zeros(3, dtype=float)
        self.__mu_pre['bias_orientation'] = np.zeros(3, dtype=float)

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

    # public methods
    # accesors
    def get_robot_model(self):return self.__rmodel  
    def get_robot_data(self):return self.__rdata
    def get_dt(self):return self.__dt
    def get_g_vector(self):return self.__g     
    def get_mu_pre(self):return self.__mu_pre
    def get_mu_post(self):return self.__mu_post
 
    # mutators (use only if you know what you are doing)
    def set_mu_post(self, mu_post):self.__mu_post = mu_post
    def set_mu_post(self, mu_pre):self.__mu_re = mu_pre

    # compute end effector placements and velocities w.r.t. the world frame 
    def compute_end_effectors_FK_quantities(self, joint_positions, joint_velocities):
        # locking the base frame to the world frame 
        q = np.zeros(7)
        q[6] = 1.0
        robot_configuration = np.concatenate([q, joint_positions])
        robot_velocity = np.concatenate([np.zeros(6), joint_velocities])
        end_effectors_positions = {}
        end_effectors_velocities = {}
        pin.forwardKinematics(self.__rmodel, self.__rdata, robot_configuration, robot_velocity)
        for key, frame_name in (self.__end_effectors_frame_names.items()):
            frame_index = self.__rmodel.getFrameId(frame_name)
            frame_position = -self.__rdata.oMf[frame_index].translation 
            frame_velocity = pin.getVelocity(self.__rmodel, self.__rdata, frame_index, pin.ReferenceFrame.WORLD)
            end_effectors_positions[key] = frame_position
            end_effectors_velocities[key] = frame_velocity.vector[0:3] #only linear velocity part
        return end_effectors_positions, end_effectors_velocities

    # discrete nonlinear motion model (mean of the state vector)
    def integrate_model(self, a_tilde, omega_tilde):
        dt, g = self.__dt, self.__g_vector
        mu_post = self.__mu_post
        r_post = mu_post['base_position']
        v_post = mu_post['base_velocity']
        q_post = mu_post['base_orientation']
        b_a_post, b_omega_post = self.__mu_post['bias_acceletation'], self.__mu_post['bias_orientation'] 
        R_post = pin.XYZQUATToSE3([r_post[0], r_post[1], r_post[2],
                 q_post.x, q_post.y, q_post.z, q_post.w]).rotation
        # IMU readings in the base frame 
        a_hat = a_tilde - b_a_post              # acceletation 
        omega_hat = omega_tilde - b_omega_post  # angular velocity 
        R_plus = box_plus(R_post, omega_hat*self.__dt) 
        q_pre = Quaternion(R_plus)
        q_pre.normalize()  
        self.__mu_pre['base_position'] = self.__mu_post['base_position'] + v_post*dt 
        self.__mu_pre['base_velocity'] = self.__mu_post['base_velocity'] + \
                       (-pin.skew(omega_hat)@v_post + R_post.T@g + a_hat)*dt
        self.__mu_pre['base_orientation'] = q_pre
        self.__a_hat, self.__omega_hat = a_hat, omega_hat 
        
    # jacobians 
    def compute_discrete_prediction_jacobian(self):
        dt, g = self.__dt, self.__g_vector
        Fc = np.zeros((self.__nx, self.__nx), dtype=float) #15x15
        mu_pre = self.get_mu_pre() 
        r_pre = mu_pre['base_position']
        q_pre = mu_pre['base_orientation']
        v_pre = mu_pre['base_velocity']
        omega_hat = self.__omega_hat
        R_pre = pin.XYZQUATToSE3([r_pre[0], r_pre[1], r_pre[2],
                q_pre.x, q_pre.y, q_pre.z, q_pre.w]).rotation
        Rt_pre = R_pre.T
        #dr/ddelta_x
        Fc[0:3,3:6] = R_pre
        Fc[0:3,6:9] = -R_pre @ pin.skew(v_pre)
        #dv/ddelta_x
        Fc[3:6,3:6] = -pin.skew(omega_hat)
        Fc[3:6,6:9] = pin.skew(Rt_pre @ g)
        Fc[3:6,9:12] = -np.eye(3)
        Fc[3:6,12:15] = -pin.skew(v_pre)
        #dtheta/ddelta_x
        Fc[6:9,6:9] = -pin.skew(omega_hat)
        Fc[6:9,12:15] = -np.eye(3)
        Fk = np.eye(self.__nx) + Fc*dt 
        return Fk 
    
    def compute_noise_jacobian(self):
        v_pre = self.__mu_pre['base_velocity']
        Lc = np.zeros((self.__nx,  self.__nx-3), dtype=float) #15x12
        Lc[3:6, 0:3] = -np.eye(3)
        Lc[3:6, 3:6] = -pin.skew(v_pre)
        Lc[6:9, 3:6] = -np.eye(3)
        Lc[9:12,6:9] =  np.eye(3)
        Lc[12:15, 9:12] = np.eye(3)
        return Lc     

    # additive white noise 
    def construct_continuous_noise_covariance(self):
        Qc = np.zeros((self.__nx-3, self.__nx-3), dtype=float) #12x12
        Qc[0:3, 0:3] = self.__Q_a
        Qc[3:6, 3:6] = self.__Q_omega
        Qc[6:9, 6:9] = self.__Qb_a
        Qc[9:12,9:12] = self.__Qb_omega
        return Qc
    
    # using zero-order hold and truncating higher-order terms
    def construct_discrete_noise_covariance(self, Fk, Lc, Qc):
        Qk_left = (Fk @ Lc)     
        Qk_right = Lc.T @ Fk.T
        return (Qk_left @ Qc @ Qk_right)*self.__dt
   
    def construct_discrete_measurement_noise_covariance(self):return (1/self.__dt)*self.__R

    # progagate covariance 
    def prediction_step(self):
        Fk = self.compute_discrete_prediction_jacobian()
        Lc = self.compute_noise_jacobian()
        Qc = self.construct_continuous_noise_covariance()
        Qk = self.construct_discrete_noise_covariance(Fk, Lc, Qc)
        self.__Sigma_pre = (Fk @ self.__Sigma_post @ Fk.T) + Qk 
    
    # assuming contact logic variables coming from the contact schedule for now
    # TODO contact logic prediction using a probabilistic model
    def measurement_model(self, contacts_schedule, joint_positions, joint_velocities):
        Hk = np.zeros((12, self.__nx))  #12x15
        predicted_base_velocity = np.zeros(12)
        measured_base_velocity = np.zeros(12)
        dt = self.__dt
        r_pre = self.__mu_pre['base_position']
        q_pre = self.__mu_pre['base_orientation']
        R_pre = pin.XYZQUATToSE3([r_pre[0], r_pre[1], r_pre[2],
                q_pre.x, q_pre.y, q_pre.z, q_pre.w]).rotation
        r_post = self.__mu_post['base_position']
        q_post = self.__mu_post['base_orientation']
        R_post = pin.XYZQUATToSE3([r_post[0], r_post[1], r_post[2],
                 q_post.x, q_post.y, q_post.z, q_post.w]).rotation
        # end effectors frame positions and velocities expressed in the world frame 
        ee_placements, ee_velocities = self.compute_end_effectors_FK_quantities(joint_positions, joint_velocities)
        # compute measurement jacobian
        Hk[0:3,3:6] = Hk[3:6,3:6] = Hk[6:9,3:6] = Hk[9:12,3:6] = -np.eye(3)
        Hk[0:3,9:12] = -R_pre @ pin.skew(ee_placements['FL'])
        Hk[3:6,9:12] = -R_pre @ pin.skew(ee_placements['FR'])
        Hk[6:9,9:12] = -R_pre @ pin.skew(ee_placements['HL'])
        Hk[9:12,9:12] = -R_pre @ pin.skew(ee_placements['HR'])
        i = 0
        for key, value in (contacts_schedule.items()):
            # check if foot is in contact based on contact schedule 
            if value:
                base_position_ee_post  = R_post @ self.__end_effectors_positions_post[key] 
                base_position_ee_pre = R_pre @ ee_placements[key]
                predicted_base_v_i = (base_position_ee_post - base_position_ee_pre)/dt
                measured_base_v_i = -R_pre @ ee_velocities[key] - pin.skew(self.__omega_hat) @ (base_position_ee_pre)
                predicted_base_velocity[i:i+3] = predicted_base_v_i
                measured_base_velocity[i:i+3] = measured_base_v_i 
            else:
                predicted_base_velocity[i:i+3] = np.zeros(3)
                measured_base_velocity[i:i+3] = np.zeros(3)
            i+=3     
        error = predicted_base_velocity - measured_base_velocity   
        self.__end_effectors_positions_post = ee_placements   
        return Hk, error

    def compute_innovation_covariance(self, Hk, Rk):return (Hk @ self.__Sigma_pre) @ Hk.T + Rk 
    
    # update mean and covariance based on new kinematic measurements
    def update_step(self, contacts_schedule, joint_positions, joint_velocities):
        r_pre = self.__mu_pre['base_position']
        q_pre = self.__mu_pre['base_orientation']
        R_pre = pin.XYZQUATToSE3([r_pre[0], r_pre[1], r_pre[2],
                q_pre.x, q_pre.y, q_pre.z, q_pre.w]).rotation
        Hk, measurement_error = self.measurement_model(contacts_schedule, joint_positions, joint_velocities)
        Rk = self.construct_discrete_measurement_noise_covariance()
        Sk = self.compute_innovation_covariance(Hk, Rk)
        K = (self.__Sigma_pre @ Hk.T) @ inv(Sk) # kalman gain
        delta_x = K @ measurement_error
        self.__Sigma_post = (np.eye(self.__nx) - (K @ Hk)) @ self.__Sigma_pre 
        self.__mu_post['base_position'] = self.__mu_pre['base_position'] + delta_x[0:3] 
        self.__mu_post['base_velocity'] = self.__mu_pre['base_velocity'] + delta_x[3:6]
        self.__mu_post['base_orientation'] = box_plus(R_pre, delta_x[6:9])
        self.__mu_post['bias_acceletation'] =  self.__mu_pre['bias_acceletation'] + delta_x[9:12]
        self.__mu_post['bias_orientation'] =  self.__mu_pre['bias_orientation'] + delta_x[12:15]

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from numpy import nditer
    solo_EKF = EKF(conf)
    f_tilde = random.rand(3)
    w_tilde = random.rand(3)
    solo_EKF.integrate_model(f_tilde, w_tilde)
    solo_EKF.prediction_step()
    contacts_schedule = {'FL':True, 'FR':True, 'HL':True,'HR':True}
    joint_positions = random.rand(12)
    joint_velocities = random.rand(12)
    solo_EKF.update_step(contacts_schedule, joint_positions, joint_velocities)

    


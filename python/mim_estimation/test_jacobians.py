import numpy as np
from ekf import EKF
import numpy.random as random
import conf
import pinocchio as pin
from pinocchio import Quaternion

def box_minus(R2, R1):return pin.log(R1.T@R2)
def box_plus(R, theta):return R@pin.exp(theta)

def continuous_transient_model(x, g, a_tilde, omega_tilde):
        f = dict.fromkeys(['base_position', 'base_velocity', 'base_orientation',
                                        'bias_acceleration', 'bias_orientation'])
        v, q = x['base_velocity'], x['base_orientation']
        b_a, b_omega = x['bias_acceleration'], x['bias_orientation']
        R = q.matrix()  
        # IMU readings in the base frame 
        a_hat = a_tilde - b_a              # acceleration
        omega_hat = omega_tilde - b_omega  # angular velocity
        R_plus = box_plus(R, omega_hat)
        q_plus = Quaternion(R_plus)
        q_plus.normalize()
        f['base_position'] = R @ v
        f['base_velocity'] = (-pin.skew(omega_hat))@v + R.T@g + a_hat
        f['base_orientation'] = q_plus
        f['bias_acceleration'] = np.zeros(3)
        f['bias_orientation'] = np.zeros(3)
        return f

def compute_prediction_jacobian(x, conf, a_hat, omega_hat):
        g = conf.g_vector
        Fc = np.zeros((15, 15), dtype=float) #15x15
        q_pre, v_pre = x['base_orientation'], x['base_velocity']
        R_pre = q_pre.matrix()
        Rt_pre = R_pre.T
        #ddeltap/ddelta_x
        Fc[0:3,3:6] = R_pre
        Fc[0:3,6:9] = -R_pre@pin.skew(v_pre)
        #ddelta_v/ddelta_x
        Fc[3:6,3:6] = -pin.skew(omega_hat)
        Fc[3:6,6:9] = pin.skew(Rt_pre@g)
        Fc[3:6,9:12] = -np.eye(3)
        Fc[3:6,12:15] = -pin.skew(v_pre)
        #ddelta_theta/ddelta_x
        Fc[6:9,6:9] = -pin.skew(omega_hat)
        Fc[6:9,12:15] = -np.eye(3)
        return Fc

def compute_measurement_jacobian(q_pre, p1, p2, p3, p4):
        R_pre = q_pre.matrix()
        Hk = np.zeros((12, 15))  #15x15
        Hk[0:3,3:6] = Hk[3:6,3:6] = Hk[6:9,3:6] =  Hk[9:12,3:6] = -np.eye(3)
        Hk[0:3,12:15] = -pin.skew(R_pre@p1)
        Hk[3:6,12:15] = -pin.skew(R_pre@p2)
        Hk[6:9,12:15] = -pin.skew(R_pre@p3)
        Hk[9:12,12:15] = -pin.skew(R_pre@p4)
        return Hk

if __name__=='__main__':
    # generates random pose    
    solo_EKF = EKF(conf)
    a_tilde = random.rand(3)
    omega_tilde = random.rand(3)
    p1 = random.rand(3)
    p2 = random.rand(3)
    p3 = random.rand(3)
    p4 = random.rand(3)
    v1 = random.rand(3)
    v2 = random.rand(3)
    v3 = random.rand(3)
    v4 = random.rand(3)
    g = conf.g_vector
    x = solo_EKF.get_mu_post()
    r, v, q = x['base_position'], x['base_velocity'], x['base_orientation']
    b_a, b_omega = x['bias_acceleration'], x['bias_orientation']
    a_hat = a_tilde - b_a              # acceleration
    omega_hat = omega_tilde - b_omega  # angular velocity
    R = q.matrix()
    f_x = continuous_transient_model(x, g, a_tilde, omega_tilde)
    q_pre = f_x['base_orientation']
    R_pre = q_pre.matrix()
    f1 = -v1 - (pin.skew(omega_hat)@R_pre@p1)
    f2 = -v2 - (pin.skew(omega_hat)@R_pre@p2)
    f3 = -v3 - (pin.skew(omega_hat)@R_pre@p3)
    f4 = -v4 - (pin.skew(omega_hat)@R_pre@p4)
    
    delta = 1e-9
    precision = 1e-0
    delta_vec = delta*np.ones(3)
    delta_exp = pin.exp(delta_vec)

    v_plus_dx = v + delta*np.array([1,0,0])
    v_plus_dy = v + delta*np.array([0,1,0])
    v_plus_dz = v + delta*np.array([0,0,1])
 
    a_hat_plus_dx = a_hat - delta*np.array([1,0,0])
    a_hat_plus_dy = a_hat - delta*np.array([0,1,0])
    a_hat_plus_dz = a_hat - delta*np.array([0,0,1])
    
    omega_hat_plus_dx = omega_hat - delta*np.array([1,0,0])
    omega_hat_plus_dy = omega_hat - delta*np.array([0,1,0])
    omega_hat_plus_dz = omega_hat - delta*np.array([0,0,1])
                                                                               
    theta_plus_dx = delta*np.array([1,0,0])
    theta_plus_dy = delta*np.array([0,1,0])
    theta_plus_dz = delta*np.array([0,0,1])
    theta_plus = delta*np.ones(3)

    # compute jacobian with numerical diff.
    J_num_prediction = np.zeros((15,15))
    J_num_meas = np.zeros((12,15))  
    
    # ------------------------  Analytical prediction jacobian-----------------------------------------------    
    # base position - base veclocity 
    J_num_prediction[0:3,3] = (R@v_plus_dx - f_x['base_position'])/delta
    J_num_prediction[0:3,4] = (R@v_plus_dy - f_x['base_position'])/delta
    J_num_prediction[0:3,5] = (R@v_plus_dz - f_x['base_position'])/delta
    # base position - base orientation
    J_num_prediction[0:3,6] = (box_plus(R, theta_plus_dx)@v - f_x['base_position'])/delta
    J_num_prediction[0:3,7] = (box_plus(R, theta_plus_dy)@v - f_x['base_position'])/delta
    J_num_prediction[0:3,8] = (box_plus(R, theta_plus_dz)@v - f_x['base_position'])/delta
    # base velocity-base velocity
    J_num_prediction[3:6,3] = (-pin.skew(omega_hat)@v_plus_dx + R.T@g + a_hat - f_x['base_velocity'])/delta
    J_num_prediction[3:6,4] = (-pin.skew(omega_hat)@v_plus_dy + R.T@g + a_hat - f_x['base_velocity'])/delta
    J_num_prediction[3:6,5] = (-pin.skew(omega_hat)@v_plus_dz + R.T@g + a_hat - f_x['base_velocity'])/delta
    # base velocity-base orientation
    J_num_prediction[3:6,6] = (-pin.skew(omega_hat)@v + box_plus(R, theta_plus_dx).T@g + a_hat - f_x['base_velocity'])/delta
    J_num_prediction[3:6,7] = (-pin.skew(omega_hat)@v + box_plus(R, theta_plus_dy).T@g + a_hat - f_x['base_velocity'])/delta 
    J_num_prediction[3:6,8] = (-pin.skew(omega_hat)@v + box_plus(R, theta_plus_dz).T@g + a_hat - f_x['base_velocity'])/delta 
    # base velocity-bias acceleration
    J_num_prediction[3:6,9] =  (-pin.skew(omega_hat)@v + R.T@g + a_hat_plus_dx - f_x['base_velocity'])/delta
    J_num_prediction[3:6,10] = (-pin.skew(omega_hat)@v + R.T@g + a_hat_plus_dy - f_x['base_velocity'])/delta
    J_num_prediction[3:6,11] = (-pin.skew(omega_hat)@v + R.T@g + a_hat_plus_dz - f_x['base_velocity'])/delta
    # base velocity - bias orientation
    J_num_prediction[3:6,12] = (-pin.skew(omega_hat_plus_dx)@v + R.T@g + a_hat - f_x['base_velocity'])/delta
    J_num_prediction[3:6,13] = (-pin.skew(omega_hat_plus_dy)@v + R.T@g + a_hat - f_x['base_velocity'])/delta
    J_num_prediction[3:6,14] = (-pin.skew(omega_hat_plus_dz)@v + R.T@g + a_hat - f_x['base_velocity'])/delta
    # base orientation - base orienation
    J_num_prediction[6:9,6] = -pin.exp(omega_hat)@(box_minus(box_plus(R, theta_plus_dx), R))/delta
    J_num_prediction[6:9,7] = -pin.exp(omega_hat)@(box_minus(box_plus(R, theta_plus_dy), R))/delta
    J_num_prediction[6:9,8] = -pin.exp(omega_hat)@(box_minus(box_plus(R, theta_plus_dz), R))/delta
    # base orientation - bias orientation
    J_num_prediction[6:9,12] = R@(omega_hat_plus_dx-omega_hat)/delta
    J_num_prediction[6:9,13] = R@(omega_hat_plus_dy-omega_hat)/delta
    J_num_prediction[6:9,14] = R@(omega_hat_plus_dz-omega_hat)/delta

    # ------------------------  Anayltical measurement jacobian-----------------------------------------------
    # base velocity from foot1-base velocity
    J_num_meas[0:3, 3] = (-(v1+delta*np.array([1,0,0])) - (pin.skew(omega_hat)@R_pre@p1) - f1)/delta
    J_num_meas[0:3, 4] = (-(v1+delta*np.array([0,1,0])) - (pin.skew(omega_hat)@R_pre@p1) - f1)/delta
    J_num_meas[0:3, 5] = (-(v1+delta*np.array([0,0,1])) - (pin.skew(omega_hat)@R_pre@p1) - f1)/delta
    # base velocity from foot1-bias orientation    
    J_num_meas[0:3, 12] =  (-v1 - (pin.skew(omega_hat_plus_dx)@R_pre@p1) - f1)/delta
    J_num_meas[0:3, 13] = (-v1 - (pin.skew(omega_hat_plus_dy)@R_pre@p1) - f1)/delta
    J_num_meas[0:3, 14] = (-v1 - (pin.skew(omega_hat_plus_dz)@R_pre@p1) - f1)/delta

    # base velocity from foot2-base velocity
    J_num_meas[3:6, 3] = (-(v2+delta*np.array([1,0,0])) - (pin.skew(omega_hat)@R_pre@p2) - f2)/delta
    J_num_meas[3:6, 4] = (-(v2+delta*np.array([0,1,0])) - (pin.skew(omega_hat)@R_pre@p2) - f2)/delta
    J_num_meas[3:6, 5] = (-(v2+delta*np.array([0,0,1])) - (pin.skew(omega_hat)@R_pre@p2) - f2)/delta
    # base velocity from foot2-bias orientation    
    J_num_meas[3:6, 12] = (-v2 - (pin.skew(omega_hat_plus_dx)@R_pre@p2) - f2)/delta
    J_num_meas[3:6, 13] = (-v2 - (pin.skew(omega_hat_plus_dy)@R_pre@p2) - f2)/delta
    J_num_meas[3:6, 14] = (-v2 - (pin.skew(omega_hat_plus_dz)@R_pre@p2) - f2)/delta

    # base velocity from foot3-base velocity
    J_num_meas[6:9, 3] = (-(v3+delta*np.array([1,0,0])) - (pin.skew(omega_hat)@R_pre@p3) - f3)/delta
    J_num_meas[6:9, 4] = (-(v3+delta*np.array([0,1,0])) - (pin.skew(omega_hat)@R_pre@p3) - f3)/delta
    J_num_meas[6:9, 5] = (-(v3+delta*np.array([0,0,1])) - (pin.skew(omega_hat)@R_pre@p3) - f3)/delta
    # base velocity from foot3-bias orientation    
    J_num_meas[6:9, 12] =  (-v3 - (pin.skew(omega_hat_plus_dx)@R_pre@p3) - f3)/delta
    J_num_meas[6:9, 13] = (-v3 - (pin.skew(omega_hat_plus_dy)@R_pre@p3) - f3)/delta
    J_num_meas[6:9, 14] = (-v3 - (pin.skew(omega_hat_plus_dz)@R_pre@p3) - f3)/delta

    # base velocity from foot4-base velocity
    J_num_meas[9:12, 3] = (-(v4+delta*np.array([1,0,0])) - (pin.skew(omega_hat)@R_pre@p4) - f4)/delta
    J_num_meas[9:12, 4] = (-(v4+delta*np.array([0,1,0])) - (pin.skew(omega_hat)@R_pre@p4) - f4)/delta
    J_num_meas[9:12, 5] = (-(v4+delta*np.array([0,0,1])) - (pin.skew(omega_hat)@R_pre@p4) - f4)/delta
    # base velocity from foot4-bias orientation    
    J_num_meas[9:12, 12] = (-v4 - (pin.skew(omega_hat_plus_dx)@R_pre@p4) - f4)/delta
    J_num_meas[9:12, 13] = (-v4 - (pin.skew(omega_hat_plus_dy)@R_pre@p4) - f4)/delta
    J_num_meas[9:12, 14] = (-v4 - (pin.skew(omega_hat_plus_dz)@R_pre@p4) - f4)/delta

    # compare numerical diff. against analytical diff.    
    J_anal = compute_prediction_jacobian(x, conf, a_hat, omega_hat)
    print(np.testing.assert_allclose(J_num_prediction, J_anal, atol=np.sqrt(precision)))

    # J_anal_meas = compute_measurement_jacobian(q_pre, p1, p2, p3, p4)
    # print(np.testing.assert_allclose(J_num_meas, J_anal_meas, atol=np.sqrt(precision)))

import numpy as np
import math

LENGTH = 5.5*10**(-2)   # [m]  # half width of the thymio
RADIUS = 2.2*10**(-2)   # [m]  # wheel radius of the thymio 
THYMIO_SPEED_TO_RADS = (0.43478260869565216 * 10**(-3))/RADIUS 

def convert_m2pix(x, m2pix_coeff):
    return x*m2pix_coeff

def motors(left, right):
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

def get_motor_speed(node):
    return np.array([node["motor.right.speed"], node["motor.left.speed"]])

def get_line_parameters(start, goal):
    length = np.linalg.norm(np.array(start)-np.array(goal))
    theta = math.atan2(goal[1]-start[1],goal[0]-start[0]) 
    return length,theta

def discretize_path(length, desired_speed, sampling_time):
    total_time = length/desired_speed
    nb_divisions = round(total_time/sampling_time)
    delta_L = length/nb_divisions
    return delta_L

def goal_reached(x_goal, tol, x):
    if (abs(x[0] - x_goal[0]) < tol ) and (abs(x[1] - x_goal[1]) < tol):
        return True   
    return False

def saturate_scalar(u:float, lim_down:float, lim_up:float):
    """
    simple saturation of a signal between 2 limits
    """
    if u > lim_up:
        return lim_up
    elif u < lim_down:
        return lim_down
    else:
        return u

def  Astolfi_controller(target, x, m2pix_coeff):
    x = np.squeeze(x)
    l = convert_m2pix(LENGTH, m2pix_coeff)
    r = convert_m2pix(RADIUS, m2pix_coeff)
    
    kp = 0.1 
    kalpha = 0.4
    
    delta_x = target[0] - x[0]
    delta_y = target[1] - x[1]
    
    rho = np.linalg.norm(target - x[0:2])
    
    tol = 0.2
    if abs(math.pi - abs(math.atan2(delta_y, delta_x))) < tol :
        alpha = -abs(x[2]) + abs(math.atan2(delta_y, delta_x))
    else :
        alpha = -x[2] + math.atan2(delta_y, delta_x)
    
    v = kp*rho
    w = kalpha*alpha
    
    phi1 = (v/r) - (l*w/r)
    phi2 = (v/r) + (l*w/r)
    
    phi1 = phi1/THYMIO_SPEED_TO_RADS
    phi2 = phi2/THYMIO_SPEED_TO_RADS
    
    phi1 = saturate_scalar(phi1, -500.0, 500.0)
    phi2 = saturate_scalar(phi2, -500.0, 500.0)
    
    return np.array([phi1, phi2])

def jacobianF(x, u, m2pix_coeff, ts):
    r = convert_m2pix(RADIUS, m2pix_coeff)
    F = np.array([[1.0, 0.0, -ts*np.sin(float(x[2]))*(u[0]*r + u[1]*r)/2.0],
                [0.0, 1.0, ts*np.cos(float(x[2]))*(u[0]*r + u[1]*r)/2.0],
                [0.0, 0.0, 1.0]]) 
    return F

def kalman_filter(speed, x_est_prev, P_est_prev, x_meas, m2pix_coeff, ts):
    """
    Estimates the current state using input sensor data and the previous state
    
    param speed: measured speed (Thymio units)
    param ground_prev: previous value of measured ground sensor
    param ground: measured ground sensor
    param pos_last_trans: position of the last transition detected by the ground sensor
    param x_est_prev: previous state a posteriori estimation
    param P_est_prev: previous state a posteriori covariance
    
    return pos_last_trans: updated if a transition has been detected
    return x_est: new a posteriori state estimation
    return P_est: new a posteriori state covariance
    """
    r1 = convert_m2pix(0.0017, m2pix_coeff)
    r2 = convert_m2pix(0.0017, m2pix_coeff)
    r3 = 0.1
    
    H = np.array([[1.0,0,0],[0,1.0,0],[0,0,1.0]])
    Q = np.array([[r1, 0, 0],[0, r2, 0], [0, 0, r3]])
    R = np.array([[1,0,0],[0,1,0],[0,0,0.04]])
    
    l = convert_m2pix(LENGTH, m2pix_coeff)
    r = convert_m2pix(RADIUS, m2pix_coeff)
    
    F = jacobianF(x_est_prev, speed, r, ts)
    
    ## Prediciton through the a priori estimate
    # estimated mean of the state
    x_est_a_priori = np.array([[0.0],[0.0],[0.0]])
    x_est_a_priori[0] = x_est_prev[0] + ts*np.cos(x_est_prev[2])*(speed[0]*r + speed[1]*r)/2.0
    x_est_a_priori[1] = x_est_prev[1] + ts*np.sin(x_est_prev[2])*(speed[0]*r + speed[1]*r)/2.0
    x_est_a_priori[2] = x_est_prev[2] + ts*(speed[1]*r - speed[0]*r)/(2*l)
    
    # Estimated covariance of the state
    P_est_a_priori = np.dot(F, np.dot(P_est_prev, F.T));
    P_est_a_priori = P_est_a_priori + Q if type(Q) != type(None) else P_est_a_priori
    
    y = np.array([x_meas]).T
    
    # innovation / measurement residual
    i = y - np.dot(H, x_est_a_priori)
    
    # measurement prediction covariance
    S = np.array(np.dot(H, np.dot(P_est_a_priori, H.T)) + R)
    
    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)))
    
    # a posteriori estimate
    x_est = x_est_a_priori + np.dot(K,i)
    P_est = P_est_a_priori - np.dot(K,np.dot(H, P_est_a_priori))
    
    return x_est, P_est

def update_instant_target(start, goal, speed_target, sampling_time):
    line_length, line_angle = get_line_parameters(start, goal)
    division_length = discretize_path(line_length, speed_target, sampling_time)
    instant_target = np.array([start[0] + division_length*np.cos(line_angle), start[1] + division_length*np.sin(line_angle)])
    delta_target = np.array([division_length*np.cos(line_angle), division_length*np.sin(line_angle)])
    return instant_target, delta_target

def update_motor(thymio_state, x_meas, x_est, P_est, instant_target, delta_target, goal_i, m2pix_coeff, node, ts):
    x_meas.append(thymio_state[-1])
    
    speed = get_motor_speed(node)*THYMIO_SPEED_TO_RADS
    
    x_new, P_new = kalman_filter(speed, x_est[-1], P_est[-1], x_meas[-1], m2pix_coeff, ts)
    x_est.append(x_new)
    P_est.append(P_new)
    
    vel_motors = Astolfi_controller(instant_target[-1], x_est[-1], m2pix_coeff)
    
    node.send_set_variables(motors(round(vel_motors[1]),round(vel_motors[0])))
    if not (all(abs(instant_target[-1] - (goal_i[-1] + 10*delta_target[-1])) < 10**(-3))):
        instant_target.append(instant_target[-1] + delta_target[-1])
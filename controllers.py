import numpy as np


def kmb_sample_controller_1(x, goal, dt):
    """Left Turn Controller

    Args:
        x (STATE): Current Agent Pose [px,py,theta,psi]
        goal (STATE): Goal State [px,py,theta,psi]

    Returns:
        control u = (v,psi_dot) 
    """
    return np.array([10, 0.2])


def kmb_sample_controller_2(x, goal, dt):
    """Random Controller

    Args:
        x (STATE): Current Agent Pose [px,py,theta,psi]
        goal (STATE): Goal State [px,py,theta,psi]

    Returns:
        control u = (v,psi_dot) 
    """

    sign_drive = 1
    if (np.random.random() > 0.5):
        sign_drive = 1

    sign_steer = 1
    if (np.random.random() > 0.5):
        sign_steer = -1

    return np.array([sign_drive*10, sign_steer*np.random.random()])


def stabilizing_control(x, goal, dt):
    """See homework 5 controller which is supposed to stablize
    This is an attempt to get a reference MVP

    Args:
        x (STATE): Current Agent Pose [px,py,theta,psi]
        goal (STATE): Goal State [px,py,theta,psi]

    Returns:
        control u = (v,psi_dot) 
    """

    # TODO make this work
    # TODO: Clean this function up  

    px, py, theta, psi = x
    px_des, py_des, theta_des, psi_des = goal

    theta = theta % (2*np.pi)

    theta_des = theta_des % (2*np.pi)
    
    diff = (theta_des - theta) % (2*np.pi)
    
    z_component = np.sin(theta_des) * np.cos(theta) - np.cos(theta_des) * np.sin(theta)

    if z_component>0:
        diff = -1*diff 

    kv = -1
    
    ktheta = 0.01 

    v = kv * ((px-px_des)*np.cos(theta) + (py-py_des)*np.sin(theta))

    # psi_dot = ktheta * ((theta-theta_des) - np.arctan2(py_des-py,px_des-px)) This is from the HW should work? 
    
    psi_dot = ktheta * ((diff))

    return np.array([v, psi_dot])


def smalled_angle_diff(a, b):
    return (a - b + np.pi) % (np.pi*2) - np.pi




def stabilizing_control_ignore_heading(x, goal, dt):
    """
    Regulate to the goal (px, py) while ignoring the goal theta.

    The idea here is that psi is sort of like theta_dot
    """

    px, py, theta, psi = x
    px_des, py_des, theta_des, psi_des = goal

    theta_from_goal = np.arctan2(py_des - py, px_des - px)
    theta_error = smalled_angle_diff(theta_from_goal, theta)

    # print(f"{theta=:0.3f}\t{theta_from_goal=:0.3f}")
    # print(f"theta error: {theta_error}")

    # tunable control gains
    k_v = 0.1
    k_psi = 0.2
    k_psi_dot = 0.2

    v = -k_v/dt * ((px - px_des) * np.cos(theta_from_goal) + (py - py_des) * np.sin(theta_from_goal))

    psi_desired = -k_psi/dt * -theta_error
    psi_dot = -k_psi_dot/dt * (psi - psi_desired)

    return np.array([v, psi_dot])



# TODO: Create MPC controller function which calls MPC.py utility 

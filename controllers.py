import numpy as np
from mpc import *


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

def distance_between_states(x0, x1):
    dist = np.linalg.norm(x0 - x1) # full vector
    # dist = np.linalg.norm(x0[:2] - x1[:2]) # ignore heading
    return dist

def find_closest_state_idx(x, x_ref):
    closest_state_idx = -1
    closest_state_distance = np.inf
    for i,xN in enumerate(x_ref):
        new_dist = distance_between_states(x, xN)
        if new_dist < closest_state_distance:
            closest_state_idx = i
            closest_state_distance = new_dist


    return closest_state_idx

def make_ref_horizon(T, i_ref, x_ref, u_ref):
    stop_idx = min(i_ref + T, len(u_ref))
    x_ref_T = x_ref[i_ref:stop_idx]
    u_ref_T = u_ref[i_ref:stop_idx]

    n_missing_entries = i_ref+T - len(u_ref)
    if n_missing_entries > 0:
        for _ in range(n_missing_entries):
            x_ref_T.append(x_ref[-1])
            u_ref_T.append(u_ref[-1])
        
    return x_ref_T, u_ref_T

# TODO: Create MPC controller function which calls MPC.py utility
def mpc_controller(x, goal, dt, x_ref, u_ref, k, rover):
    
    n = 4  # state dimension
    m = 2  # control dimension
    T = 20  # MPC horizon

    iterative_linearization = True

    # cost functions
    Q = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    R = np.array([[1, 0],
                  [0, 1]])
    Qf = 1e3 * np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    # u_ref = np.ones([m*T, 1])
    # x_ref = 2*np.ones([n*T, 1])
    i_ref = find_closest_state_idx(x, x_ref)
    x_ref_T, u_ref_T = make_ref_horizon(T, i_ref, x_ref, u_ref)

    u_0 = u_ref_T[0] # first timestep of reference control
    A, B, xeq, ueq = linearize_dynamics(x, u_0, dt, rover.wheel_base)

    x_ref_T = np.array(x_ref_T).reshape((n*T, 1))
    u_ref_T = np.array(u_ref_T).reshape((m*T, 1))

    # initial MPC
    u, res = mpc_control(x, x_ref_T, u_ref_T, Q, R, Qf, T, A, B, xeq, ueq, rover)

    if iterative_linearization:
    # iterative linearized MPC with state traj from the previous MPC
        u = mpc_control_iterative(x, x_ref_T, u_ref_T, Q, R, Qf, T, res, xeq, ueq, rover)

    v = u[0] + ueq[0]
    psi_dot = u[1] + ueq[1]
    return np.array([v, psi_dot])

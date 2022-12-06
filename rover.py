import numpy as np
from state import *

class Rover:
    def __init__(self, init_state):
        self.wheel_angle_limit = 0.5 # [radians]
        self.wheel_angle_velocity_limit = 1 # [radians / sec]
        self.velocity_limit = 1 # [m/s]
        self.wheel_base = 1 # [m]
        self.dt = 0.01 # [seconds]
        self.max_iters = 100000 # [iterations]
        self.goal_threshold = 0.05 # [meters]
        self.state = init_state.state

        self.dynamics = 'ackerman_kbm'

    def ackermann_kbm_dynamics(self, u):
        """
        Idealized Ackermann Steering Dynamics

        State: px, py, heading_angle, wheel angles
        Control: velocity in body frame, wheel angle velocity

        Slip (process noise) is modeled as a random normal added to the turning radius
        """
        # TODO encorporate slip

        # get state 
        px, py, theta, psi = self.state
        dt = self.dt

        # get control
        v, psi_dot = u

        # update state 
        px_new = px + (v*dt*np.cos(psi)*np.cos(theta))
        py_new = py + (v*dt*np.cos(psi)*np.sin(theta))
        theta_new = theta + (v*dt*np.sin(psi)*self.wheel_base/2)%(2*np.pi)

        # clamp turn radius
        psi_attempt = psi + (psi_dot*dt)
        psi_new = max(-self.wheel_angle_limit, min(psi_attempt, self.wheel_angle_limit))

        # update state stored inside class
        self.state = np.array([px_new, py_new, theta_new, psi_new])

        return self.state

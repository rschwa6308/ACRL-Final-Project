import numpy as np
from state import *

class Rover:
    def __init__(self, init_state, slip=False, dt=0.1):
        self.wheel_angle_limit = 0.5 # [radians]
        self.wheel_angle_velocity_limit = 1 # [radians / sec]
        self.velocity_limit = 1 # [m/s]
        self.wheel_base = 1 # [m]
        self.goal_threshold = 0.05 # [meters]
        self.state = init_state.state
        self.min_turning_radius = (self.wheel_base/(2 * np.sin(self.wheel_angle_limit))) * 4
        self.sKx = 0
        self.sKy = 0
        self.sKtheta = 0
        self.sKpsi = 0
        if slip:
            self.sKx = 0.08
            self.sKy = 0.08
            self.sKtheta = 0.01
            self.sKpsi = 0.001
        self.heartBeat = 0
        self.Sk = 0
        self.SkN = int(2/dt)
        self.dynamics = 'ackerman_kbm'

    def ackermann_kbm_dynamics(self, u, dt):
        """
        Idealized Ackermann Steering Dynamics

        State: px, py, heading_angle, wheel angles
        Control: velocity in body frame, wheel angle velocity

        Slip (process noise) is modeled as a random normal added to the turning radius
        """

        if ((self.heartBeat%self.SkN) == 0):
            self.Sk = np.random.uniform(-1,1,4)

        # get state 
        px, py, theta, psi = self.state

        # get control
        v, psi_dot = u

        # update state 
        px_new = px + (v*dt*(self.sKx*self.Sk[0] + np.cos(psi)*np.cos(theta))) 
        py_new = py + (v*dt*(self.sKy*self.Sk[1] + np.cos(psi)*np.sin(theta)))

        # This line is sus, need to determine how we want to represent angle (0:2pi) (-pi:pi) ?
        theta_new = (theta + (v*dt*(self.sKtheta*self.Sk[2] + np.sin(psi)*self.wheel_base/2))%(2*np.pi))%(2*np.pi)

        # clamp turn radius
        psi_attempt = psi + (psi_dot*dt) + self.sKpsi*self.Sk[3]
        psi_new = max(-self.wheel_angle_limit, min(psi_attempt, self.wheel_angle_limit))

        # update state stored inside class
        self.state = np.array([px_new, py_new, theta_new, psi_new])
        self.heartBeat += 1

        return self.state


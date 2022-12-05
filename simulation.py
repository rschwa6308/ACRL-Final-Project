import numpy as np
from matplotlib import pyplot as plt
from controllers import *
from observations import *
from rover import *
from state import *

def simulate(rover, goal, observation_func, control_law):
    """ Simulate discrete system

    Args:
        rover (ROVER): rover class to simulate
        goal (STATE): goal pose rover should drive to
        observation_func (function): points to function in observations.py
        control_law (function): points to function in controllers.py

    Returns:
        xs, ys, us: nparrays of all STATES, MEASUREMENTS, CONTROLS
    """
    N = rover.max_iters

    xs = [rover.state]
    ys = []
    us = []

    for k in range(N):

        # observe current state
        ys.append(observation_func(xs[k]))

        # desired control
        v, psi_dot = control_law(ys[k],goal)

        # clamped control
        v = max(-rover.velocity_limit, min(v, rover.velocity_limit))
        psi_dot = max(-rover.wheel_angle_velocity_limit, min(psi_dot, rover.wheel_angle_velocity_limit))            

        # appended control 
        us.append(np.array([v, psi_dot]))
        
        # forward dynamics with control
        xs.append(rover.ackermann_kbm_dynamics(us[k]))

        # TODO add break function for exiting once goal threshold is reached as defined in rover class
    
    plot(goal,xs,us)

    return xs, ys, us



def plot(goal,xs,us):

    # TODO: make plotting better 

    # plot goal arrow
    plt.arrow(goal[0], goal[1], np.cos(goal[2]), np.sin(goal[2]), color='red', head_width = 0.2, width = 0.05)

    path_length = len(us)

    pxs, pys, heading_angles, steering_angles = zip(*xs)
    speeds = [u[0] for u in us]

    # plot the trajectory
    plt.plot(pxs, pys)
    plt.axis("equal")

    # plot some nice pose arrows
    arrows = [(
        pxs[k], pys[k],
        np.cos(heading_angles[k]),
        np.sin(heading_angles[k])
    ) for k in range(path_length)]

    arrows_select = arrows[::path_length//10] + [arrows[-1]]   # plot sparse selection of states (including final)
    # print(len(arrows_select))
    plt.quiver(
        *zip(*arrows_select),
        color=["red"] + ["black"]*(len(arrows_select)-2) + ["green"]
    )
        

    plt.show()
    # TODO: make plotting control better 
    plt.plot(us)
    plt.show()
    


# Example Usage
if __name__ == "__main__":
    print("test")

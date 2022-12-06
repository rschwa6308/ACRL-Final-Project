import numpy as np
from matplotlib import pyplot as plt
from controllers import *
from observations import *
from rover import *
from state import *

def simulate(rover, goal, observation_func, control_law, stopping_condition=None, max_iters=1000, dt=0.1):
    """ Simulate discrete system

    Args:
        rover (ROVER): rover class to simulate
        goal (STATE): goal pose rover should drive to
        observation_func (function): points to function in observations.py
        control_law (function): points to function in controllers.py
        stopping_condition (function): points to function in stopping_conditions.py
        max_iters (int): terminate if stopping_condition has not been reached after a set time
        dt (float): timestep

    Returns:
        xs, ys, us: nparrays of all STATES, MEASUREMENTS, CONTROLS
    """

    xs = [rover.state]
    ys = []
    us = []

    for k in range(max_iters):

        # observe current state
        ys.append(observation_func(xs[k]))

        # desired control
        v, psi_dot = control_law(ys[k], goal, dt)

        # clamped control
        v = max(-rover.velocity_limit, min(v, rover.velocity_limit))
        psi_dot = max(-rover.wheel_angle_velocity_limit, min(psi_dot, rover.wheel_angle_velocity_limit))            

        # appended control 
        us.append(np.array([v, psi_dot]))
        
        # forward dynamics with control
        xs.append(rover.ackermann_kbm_dynamics(us[k], dt))

        # break early if stopping condition has been met
        if stopping_condition is not None:
            if stopping_condition(xs[-1], goal):
                break
    
    return xs, ys, us



def plot_traj(rover,goal,xs,us):

    # --------------------------------------------------------------------
    # plot goal arrow
    plt.arrow(goal[0], goal[1], np.cos(goal[2]), np.sin(goal[2]), color='red', head_width = 0.2, width = 0.05)

    path_length = len(us)
    vs, psi_dots = zip(*us)

    pxs, pys, heading_angles, steering_angles = zip(*xs)

    # plot the trajectory
    plt.plot(pxs, pys)
    plt.axis("equal")

    # plot some nice pose arrows
    arrows = [(
        pxs[k], pys[k],
        np.cos(heading_angles[k]) * np.sign(vs[k]),
        np.sin(heading_angles[k]) * np.sign(vs[k])
    ) for k in range(path_length)]

    arrows_select = arrows[::path_length//10] + [arrows[-1]]   # plot sparse selection of states (including final)
    # print(len(arrows_select))
    plt.quiver(
        *zip(*arrows_select),
        color=["red"] + ["black"]*(len(arrows_select)-2) + ["green"]
    )
    plt.show()

    
def plot_control(rover,goal,xs,us):

    speeds = [u[0] for u in us]
    delta_steer = [u[1] for u in us]

    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(2, 1, 1)
    plt.axhline(y = rover.velocity_limit, color = 'r', linestyle = 'dotted')
    plt.axhline(y = -rover.velocity_limit, color = 'r', linestyle = 'dotted')
    plt.plot(speeds)
    plt.title("Control Inputs - Velocity")
    plt.subplot(2, 1, 2)
    plt.axhline(y = rover.wheel_angle_velocity_limit, color = 'r', linestyle = 'dotted')
    plt.axhline(y = -rover.wheel_angle_velocity_limit, color = 'r', linestyle = 'dotted')
    plt.plot(delta_steer)
    plt.title("Control Inputs - Change in Steer")
    plt.show()
    
def plot_states(rover,goal,xs,us):

    pxs, pys, heading_angles, steering_angles = zip(*xs)

    pxs_g, pys_g, heading_angles_g, steering_angles_g = goal

    speeds = [u[0] for u in us]
    delta_steer = [u[1] for u in us]

    plt.figure(figsize=(8, 6), dpi=80)

    plt.subplot(2, 2, 1)
    plt.axhline(y = pxs_g, color = 'r', linestyle = 'dotted')
    plt.plot(pxs)
    plt.title("Px vs. Goal")

    plt.subplot(2, 2, 2)
    plt.axhline(y = pys_g, color = 'r', linestyle = 'dotted')
    plt.plot(pys)
    plt.title("Py vs. Goal")

    plt.subplot(2, 2, 3)
    plt.axhline(y = heading_angles_g, color = 'r', linestyle = 'dotted')
    plt.plot(heading_angles)
    plt.title("Heading vs. Goal")

    plt.subplot(2, 2, 4)
    plt.axhline(y = steering_angles_g, color = 'r', linestyle = 'dotted')
    plt.plot(steering_angles)
    plt.title("Steering vs. Goal")

    plt.show()

# Example Usage
if __name__ == "__main__":
    print("test")

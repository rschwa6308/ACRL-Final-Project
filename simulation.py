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



def simulate_with_MPC(rover, goal, observation_func, MPC_controller, reference_trajectory_xs, reference_trajectory_us, stopping_condition=None, max_iters=1000, dt=0.1):
    """ Simulate discrete system for use with the MPC controller

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
        v, psi_dot = MPC_controller(ys[k], goal, dt, reference_trajectory_xs, reference_trajectory_us, k, rover)

        """
        In theory, MPC_controller (above) will internally use reference_trajectory_xs[k:k+T]
        as the reference trajectory where T is the MPC horizon

        OR

        We try to find the point in the reference trajectory that is closest (L2-norm)
        to our current actual state. (WARNING: HACKY lol, but not a terrible idea)
        """

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

def plot_traj_dicts(goal, results):
    # Goal arrow
    plt.arrow(goal[0], goal[1], np.cos(goal[2]), np.sin(goal[2]), color='red', head_width = 0.2, width = 0.05)

    for result in results:
        us = result['us']
        xs = result['xs']
        label = result['label']
        color = result['color']

        path_length = len(us)
        vs, psi_dots = zip(*us)
        pxs, pys, heading_angles, steering_angles = zip(*xs)

        # plot the trajectory
        plt.plot(pxs, pys, label=label, color=color)
        plt.axis("equal")

        # plot some nice pose arrows, forwards only
        arrow_size = 1
        arrow_head_length = 0.2 * arrow_size
        arrow_head_width = 0.15 * arrow_size
        arrow_body_length = 0.8 * arrow_size
        arrow_body_width = 0.01 * arrow_size
        # arrows = [(
        #     pxs[k], pys[k],
        #     arrow_body_length * np.cos(heading_angles[k]),
        #     arrow_body_length * np.sin(heading_angles[k])
        # ) for k in range(path_length)]

        # arrows_select = arrows[::path_length//10] + [arrows[-1]]   # plot sparse selection of states (including final)
        
        # Start
        arrow_color='red'
        plt.arrow(pxs[0], pys[0], 
                    arrow_body_length * np.cos(heading_angles[0]), 
                    arrow_body_length * np.sin(heading_angles[0]), 
                    width=arrow_body_width, head_width=arrow_head_width, head_length=arrow_head_length, 
                    fc=arrow_color, ec=arrow_color, zorder=3)
        # End
        arrow_color='green'
        plt.arrow(pxs[-1], pys[-1], 
                    arrow_body_length * np.cos(heading_angles[-1]), 
                    arrow_body_length * np.sin(heading_angles[-1]), 
                    width=arrow_body_width, head_width=arrow_head_width, head_length=arrow_head_length, 
                    fc=arrow_color, ec=arrow_color, zorder=3)

        # Trajectory
        arrow_color='black'
        step_size = 10
        for k in range(0,path_length, step_size):
            plt.arrow(pxs[k], pys[k], 
                    arrow_body_length * np.cos(heading_angles[k]), 
                    arrow_body_length * np.sin(heading_angles[k]), 
                    width=arrow_body_width, head_width=arrow_head_width, head_length=arrow_head_length, 
                    fc=arrow_color, ec=arrow_color, zorder=2)
        # plt.quiver(
        #     *zip(*arrows_select),
        #     color=["red"] + ["black"]*(len(arrows_select)-2) + ["green"]
        # )
    plt.title("Trajectory Results")
    plt.legend()
    plt.show()

def plot_us_dicts(rover, results):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(2, 1, 1)
    plt.axhline(y = rover.velocity_limit, color = 'r', linestyle = 'dotted', label='Bound')
    plt.axhline(y = -rover.velocity_limit, color = 'r', linestyle = 'dotted')

    for result in results:
        us = result['us']
        label = result['label']
        color = result['color']
        zorder = result['zorder']

        speeds = [u[0] for u in us]
        delta_steer = [u[1] for u in us]
        plt.plot(speeds, label=label, color=color, zorder=zorder)
    plt.title("Control Inputs - Velocity")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.axhline(y = rover.wheel_angle_velocity_limit, color = 'r', linestyle = 'dotted', label='Bound')
    plt.axhline(y = -rover.wheel_angle_velocity_limit, color = 'r', linestyle = 'dotted')

    for result in results:
        us = result['us']
        label = result['label']
        color = result['color']
        zorder = result['zorder']
        
        delta_steer = [u[1] for u in us]
        plt.plot(delta_steer, label=label, color=color, zorder=zorder)
    plt.title("Control Inputs - Change in Steer")
    plt.legend()
    plt.show()

def plot_states_dicts(goal, results):
    pxs_g, pys_g, heading_angles_g, steering_angles_g = goal

    plt.figure(figsize=(8, 6), dpi=80)

    plt.subplot(2, 2, 1)
    plt.axhline(y = pxs_g, color = 'k', linestyle = 'dotted', label='Goal')
    for result in results:
        xs = result['xs']
        label = result['label']
        color = result['color']
        zorder = result['zorder']

        pxs, pys, heading_angles, steering_angles = zip(*xs)
        plt.plot(pxs, label=label, color=color, zorder=zorder)
    plt.title("Px vs. Goal")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.axhline(y = pys_g, color = 'k', linestyle = 'dotted', label='Goal')
    for result in results:
        xs = result['xs']
        label = result['label']
        color = result['color']
        zorder = result['zorder']

        pxs, pys, heading_angles, steering_angles = zip(*xs)
        plt.plot(pys, label=label, color=color, zorder=zorder)
    plt.title("Py vs. Goal")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.axhline(y = heading_angles_g, color = 'k', linestyle = 'dotted', label='Goal')
    for result in results:
        xs = result['xs']
        label = result['label']
        color = result['color']
        zorder = result['zorder']

        pxs, pys, heading_angles, steering_angles = zip(*xs)
        plt.plot(heading_angles, label=label, color=color, zorder=zorder)
    plt.title("Heading vs. Goal")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.axhline(y = steering_angles_g, color = 'k', linestyle = 'dotted', label='Goal')
    for result in results:
        xs = result['xs']
        label = result['label']
        color = result['color']
        zorder = result['zorder']

        pxs, pys, heading_angles, steering_angles = zip(*xs)
        plt.plot(steering_angles, label=label, color=color, zorder=zorder)
    plt.title("Steering vs. Goal")
    plt.legend()

    plt.show()

def plot_traj(rover,goal,xs,us,rover_name, rover2=None, xs2=None, us2=None, rover2_name=None):

    # --------------------------------------------------------------------
    # plot goal arrow
    plt.arrow(goal[0], goal[1], np.cos(goal[2]), np.sin(goal[2]), color='red', head_width = 0.2, width = 0.05)

    path_length = len(us)
    vs, psi_dots = zip(*us)
    pxs, pys, heading_angles, steering_angles = zip(*xs)

    # plot the trajectory
    plt.plot(pxs, pys, label=rover_name)
    plt.axis("equal")

    # plot some nice pose arrows
    # arrows = [(
    #     pxs[k], pys[k],
    #     np.cos(heading_angles[k]) * np.sign(vs[k]),
    #     np.sin(heading_angles[k]) * np.sign(vs[k])
    # ) for k in range(path_length)]
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
    plot_rover2 = (rover2 is not None) and (xs2 is not None) and (us2 is not None) and (rover2_name is not None)
    if plot_rover2:
        path_length = len(us2)
        vs, psi_dots = zip(*us2)
        pxs, pys, heading_angles, steering_angles = zip(*xs2)

        # plot the trajectory
        plt.plot(pxs, pys, label=rover2_name, zorder=0)
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
    plt.title("Trajectory Results")
    plt.legend()
    plt.show()

def plot_control(rover,goal,xs,us,rover_name, rover2=None, xs2=None, us2=None, rover2_name=None):

    speeds = [u[0] for u in us]
    delta_steer = [u[1] for u in us]

    plot_rover2 = (rover2 is not None) and (xs2 is not None) and (us2 is not None) and (rover2_name is not None)
    if plot_rover2:
        speeds2 = [u[0] for u in us2]
        delta_steer2 = [u[1] for u in us2]

    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(2, 1, 1)
    plt.axhline(y = rover.velocity_limit, color = 'r', linestyle = 'dotted', label='Bound')
    plt.axhline(y = -rover.velocity_limit, color = 'r', linestyle = 'dotted')
    plt.plot(speeds, label=rover_name)
    if plot_rover2: plt.plot(speeds2, label=rover2_name, zorder=0)
    plt.title("Control Inputs - Velocity")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.axhline(y = rover.wheel_angle_velocity_limit, color = 'r', linestyle = 'dotted', label='Bound')
    plt.axhline(y = -rover.wheel_angle_velocity_limit, color = 'r', linestyle = 'dotted')
    plt.plot(delta_steer, label=rover_name)
    if plot_rover2: plt.plot(delta_steer2, label=rover2_name, zorder=0)
    plt.title("Control Inputs - Change in Steer")
    plt.legend()
    plt.show()
    
def plot_states(rover,goal,xs,us,rover_name, rover2=None, xs2=None, us2=None, rover2_name=None):

    pxs, pys, heading_angles, steering_angles = zip(*xs)

    pxs_g, pys_g, heading_angles_g, steering_angles_g = goal

    plot_rover2 = (rover2 is not None) and (xs2 is not None) and (us2 is not None) and (rover2_name is not None)
    if plot_rover2:
        pxs2, pys2, heading_angles2, steering_angles2 = zip(*xs2)

    plt.figure(figsize=(8, 6), dpi=80)

    plt.subplot(2, 2, 1)
    plt.axhline(y = pxs_g, color = 'k', linestyle = 'dotted', label='Goal')
    plt.plot(pxs, label=rover_name)
    if plot_rover2: plt.plot(pxs2, label=rover2_name, zorder=0)
    plt.title("Px vs. Goal")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.axhline(y = pys_g, color = 'k', linestyle = 'dotted', label='Goal')
    plt.plot(pys, label=rover_name)
    if plot_rover2: plt.plot(pys2, label=rover2_name, zorder=0)
    plt.title("Py vs. Goal")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.axhline(y = heading_angles_g, color = 'k', linestyle = 'dotted', label='Goal')
    plt.plot(heading_angles, label=rover_name)
    if plot_rover2: plt.plot(heading_angles2, label=rover2_name, zorder=0)
    plt.title("Heading vs. Goal")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.axhline(y = steering_angles_g, color = 'k', linestyle = 'dotted', label='Goal')
    plt.plot(steering_angles, label=rover_name)
    if plot_rover2: plt.plot(steering_angles2, label=rover2_name, zorder=0)
    plt.title("Steering vs. Goal")
    plt.legend()

    plt.show()

# Example Usage
if __name__ == "__main__":
    print("test")

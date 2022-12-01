import numpy as np
from matplotlib import pyplot as plt



def ackermann_dynamics(x, u, dt=0.1, slip_magnitude=0.2):
    """
    Idealized Ackermann Steering Dynamics

    State: px, py, heading_angle
    Control: speed, curvature (i.e. 1/signed_turning_radius)

    Note: The mapping from the robot's "steering control"
    (which presumably goes from -1 to +1) to the
    signed_turning_radius (-inf, +inf) will depend on the details
    of the robot geometry

    Slip (process noise) is modeled as a random normal added to the turning radius
    """
    print(x, u)
    px, py, heading_angle = x
    speed, curvature = u

    # move in a straight line if curvature is sufficiently small
    if abs(curvature) < 1e-6:
        return np.array([
            px + np.cos(heading_angle) * speed * dt,
            py + np.sin(heading_angle) * speed * dt,
            heading_angle
        ])
        

    turning_radius = 1/curvature

    # model slip
    turning_radius += np.random.normal(0.0, slip_magnitude)

    # common center of turning circle
    turn_circle_center = np.array([
        px + np.cos(heading_angle + np.pi/2) * turning_radius,
        py + np.sin(heading_angle + np.pi/2) * turning_radius
    ])
    print(turn_circle_center)

    # turn amount measured in radians about the turn circle center (signed!)
    turn_amount = speed * dt / turning_radius

    # angle with respect to turning center
    current_center_angle = np.arctan2(py - turn_circle_center[1], px - turn_circle_center[0])
    new_center_angle = current_center_angle + turn_amount
    
    px_new = turn_circle_center[0] + np.cos(new_center_angle) * abs(turning_radius)
    py_new = turn_circle_center[1] + np.sin(new_center_angle) * abs(turning_radius)
    heading_angle_new = heading_angle + turn_amount

    return np.array([px_new, py_new, heading_angle_new])



def simulate(dynamics_func, observation_func, control_law, x0, N=100):
    """
    Simulate the discrete controlled system. Arugments:
     - `dynamics_func`: plant dynamics f(x, u) -> x'
     - `observation_func`: state observations h(x) -> y
     - `control_law`: controller g(x) -> u OR an array of predefined controls (of length N)
     - `x0`: starting state
     - `N`: number of time steps to simulate
    """

    xs = [x0]
    ys = []
    us = []

    for k in range(N):
        # observe current state
        ys.append(observation_func(xs[k]))

        # reference control law to get next control input
        if callable(control_law):
            us.append(control_law(ys[k]))
        else:
            us.append(control_law[k])
        
        # forward dynamics with control
        xs.append(dynamics_func(xs[k], us[k]))
    
    return xs, ys, us



# Example Usage
if __name__ == "__main__":
    N = 100
    x0 = np.array([0, 0, np.pi/2])


    def perfect_observations(x):
        "Lossless state measurement"
        return x


    def sample_controller_1(x):
        "Return a constant control (turn left)"
        return np.array([10, 0.2])


    def sample_controller_2(x):
        "Regulate heading_angle to 0 with basic control gain"
        px, py, heading_angle = x
        return np.array([10, -0.1 * heading_angle])


    "Turn back and forth a bit"
    sample_controller_3 = [np.array([10, np.sin(k/10) * 0.2]) for k in range(N)]


    for control_law in (sample_controller_1, sample_controller_2, sample_controller_3):
        xs, ys, us = simulate(
            ackermann_dynamics, perfect_observations,
            control_law, x0, N
        )

        pxs, pys, heading_angles = zip(*xs)
        speeds = [u[0] for u in us]

        # plot the trajectory
        plt.plot(pxs, pys)
        plt.axis("equal")

        # plot some nice pose arrows
        arrows = [(
            pxs[k], pys[k],
            speeds[k] * np.cos(heading_angles[k]),
            speeds[k] * np.sin(heading_angles[k])
        ) for k in range(N)]
        arrows_select = arrows[::N//10] + [arrows[-1]]   # plot sparse selection of states (including final)
        print(len(arrows_select))
        plt.quiver(
            *zip(*arrows_select),
            color=["red"] + ["black"]*(len(arrows_select)-2) + ["green"]
        )     

        plt.show()

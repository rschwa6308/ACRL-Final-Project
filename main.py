import numpy as np
from matplotlib import pyplot as plt
from controllers import *
from simulation import *
from observations import *
from goalFactory import *
from stopping_conditions import *
from rover import *
from state import *


# initialize robot 
x_0 = State(
    np.random.uniform(-3, 3),
    np.random.uniform(-3, 3),
    np.random.uniform(0, 2*np.pi),
    0
)    # starting pose, [px,py,theta,psi]
rover = Rover(x_0)


# get goal pose
# goal = generate_easy_goal_straight(x_0.state)
goal = generate_easy_goal_turn(x_0.state)

                         
# do reference trajectory "fake" control
reference_trajectory_xs, _, reference_trajectory_us = simulate(
    rover, goal, perfect_observations,
    stabilizing_control_ignore_heading,
    stopping_condition=goal_reached_ignore_heading,
    max_iters=1000,
    dt=0.1
)


# run MPC!  :D
xs, ys, us = simulate_with_MPC(
    rover, goal, perfect_observations,
    MPC_controller_wrapper_TODO,
    reference_trajectory_xs,
    reference_trajectory_us,
    stopping_condition=goal_reached_ignore_heading,
    max_iters=1000,
    dt=0.1
)

print(f"Simulation terminated after {len(us)} iterations")


plot_traj(rover,goal,xs,us)
plot_control(rover,goal,xs,us)
plot_states(rover,goal,xs,us)


# do MPC and "real" control 

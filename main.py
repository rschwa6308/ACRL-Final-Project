import numpy as np
from matplotlib import pyplot as plt
from controllers import *
from simulation import *
from observations import *
from goalFactory import *
from stopping_conditions import *
from rover import *
from state import *

np.random.seed(0)

# initialize robot 
x_0 = State(
    np.random.uniform(-3, 3),
    np.random.uniform(-3, 3),
    np.random.uniform(0, 2*np.pi),
    0
)    # starting pose, [px,py,theta,psi]
rover_mpc = Rover(x_0)
rover_ref = Rover(x_0)


# get goal pose
# goal = generate_easy_goal_straight(x_0.state)
goal = generate_easy_goal_turn(x_0.state)

                         
# do reference trajectory "fake" control
reference_trajectory_xs, _, reference_trajectory_us = simulate(
    rover_ref, goal, perfect_observations,
    stabilizing_control_ignore_heading,
    stopping_condition=goal_reached,
    max_iters=1000,
    dt=0.1
)

print(f"Reference simulation terminated after {len(reference_trajectory_us)} iterations")

# run MPC!  :D
xs, ys, us = simulate_with_MPC(
    rover_mpc, goal, perfect_observations,
    mpc_controller,
    reference_trajectory_xs,
    reference_trajectory_us,
    stopping_condition=goal_reached,
    max_iters=1000,
    dt=0.1
)

print(f"MPC simulation terminated after {len(us)} iterations")


plot_traj(rover_mpc,goal,xs,us,"MPC", rover_ref,reference_trajectory_xs,reference_trajectory_us,"Reference")
plot_control(rover_mpc,goal,xs,us,"MPC", rover_ref,reference_trajectory_xs,reference_trajectory_us,"Reference")
plot_states(rover_mpc,goal,xs,us,"MPC", rover_ref,reference_trajectory_xs,reference_trajectory_us,"Reference")

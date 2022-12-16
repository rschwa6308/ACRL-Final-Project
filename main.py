import numpy as np
from matplotlib import pyplot as plt
from controllers import *
from simulation import *
from observations import *
from goalFactory import *
from stopping_conditions import *
from rover import *
from state import *
from dubins import dubins_main

np.random.seed(0)

# initialize robot 
x_0 = State(
    np.random.uniform(-3, 3),
    np.random.uniform(-3, 3),
    np.random.uniform(0, 2*np.pi),
    0
)    # starting pose, [px,py,theta,psi]

# get goal pose
# goal = generate_easy_goal_straight(x_0.state)
goal = generate_easy_goal_turn(x_0.state)

# x_0 = State(0, 0, np.pi/2, 0)
# goal = np.array([10, 20, np.pi/4, 0])

# Create rovers
rover_mpc = Rover(x_0)
rover_ref = Rover(x_0)

use_dubins = True

if not use_dubins:
    # do reference trajectory "fake" control
    reference_trajectory_xs, _, reference_trajectory_us = simulate(
        rover_ref, goal, perfect_observations,
        stabilizing_control_ignore_heading,
        stopping_condition=goal_reached_ignore_heading,
        max_iters=1000,
        dt=0.05
    )

else:
    dubins_start_pose = np.array([x_0.state[0],x_0.state[1],x_0.state[2]])
    dubins_end_pose = np.array([goal[0],goal[1],goal[2]])
    print("min turning r:", rover_mpc.min_turning_radius)
    reference_trajectory_xs = dubins_main(dubins_start_pose, dubins_end_pose, rover_mpc.min_turning_radius*4, rover_mpc.wheel_angle_limit, rover_mpc.velocity_limit, dt=0.05)

for _ in range(10):
    reference_trajectory_xs.append(goal)

reference_trajectory_us = [np.array([1.0, 0.0]) for _ in range(len(reference_trajectory_xs))]

print(f"Reference simulation terminated after {len(reference_trajectory_us)} iterations")

# run MPC!  :D
xs, ys, us = simulate_with_MPC(
    rover_mpc, goal, perfect_observations,
    mpc_controller,
    reference_trajectory_xs,
    reference_trajectory_us,
    stopping_condition=goal_reached,
    max_iters=1000,
    dt=0.05
)

# print(f"MPC simulation terminated after {len(us)} iterations")

plot_traj(rover_mpc,goal,xs,us,"MPC", rover_mpc,reference_trajectory_xs,reference_trajectory_us,"Dubins")
plot_control(rover_mpc,goal,xs,us,"MPC", rover_ref,reference_trajectory_xs,reference_trajectory_us,"Reference")
plot_states(rover_mpc,goal,xs,us,"MPC", rover_ref,reference_trajectory_xs,reference_trajectory_us,"Reference")

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

def compute_control_accels(us, dt):
    us = np.array(us)
    dv_dt = np.diff(us[:,0]) / dt
    dpsidot_dt = np.diff(us[:,1]) / dt

    return np.sum(np.absolute(dv_dt)), np.sum(np.absolute(dpsidot_dt))

if __name__ == '__main__':
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

    dt = 0.1

    # Create rovers
    rover = Rover(x_0)
    rover_stabilizing = Rover(x_0)
    rover_mpc_stabilizing = Rover(x_0, True, dt)
    rover_mpc_dubins = Rover(x_0, True, dt)


    # Reference stabilizing controller
    ref_traj_xs_stablizing, _, ref_traj_us_stablizing = simulate(
        rover_stabilizing, goal, perfect_observations,
        stabilizing_control_ignore_heading,
        stopping_condition=goal_reached_ignore_heading,
        max_iters=1000,
        dt=dt
    )
    print(f"Stabilizing controller simulation terminated after {len(ref_traj_xs_stablizing)} iterations")

    # Reference Dubin's planner
    dubins_start_pose = np.array([x_0.state[0],x_0.state[1],x_0.state[2]])
    dubins_end_pose = np.array([goal[0],goal[1],goal[2]])

    ref_traj_xs_dubins = dubins_main(dubins_start_pose, dubins_end_pose, rover_mpc_dubins.min_turning_radius, rover_mpc_dubins.wheel_angle_limit, rover_mpc_dubins.velocity_limit, dt=dt)
    print(f"Dubin's planner terminated after {len(ref_traj_xs_dubins)} iterations")

    # Dubin's control inputs
    ref_traj_us_dubins = [np.array([1.0, 0.0]) for _ in range(len(ref_traj_xs_dubins))]

    # run MPC!  :D
    xs_dubins, ys_dubins, us_dubins = simulate_with_MPC(
        rover_mpc_dubins, goal, perfect_observations,
        mpc_controller,
        ref_traj_xs_dubins,
        ref_traj_us_dubins,
        stopping_condition=goal_reached_separate_pos_heading,
        max_iters=1000,
        dt=dt
    )
    print(f"MPC Dubin's terminated after {len(xs_dubins)} iterations")

    xs_stabilizing, ys_stabilizing, us_stabilizing = simulate_with_MPC(
        rover_mpc_stabilizing, goal, perfect_observations,
        mpc_controller,
        ref_traj_xs_stablizing,
        ref_traj_us_stablizing,
        stopping_condition=goal_reached_separate_pos_heading,
        max_iters=400,
        dt=dt
    )
    print(f"MPC Stabilizing terminated after {len(xs_stabilizing)} iterations")


    results = [
        {'label':"Ref Stabilizing", 'us': ref_traj_us_stablizing, 'xs': ref_traj_xs_stablizing, 'dt': dt, 'color': 'tab:blue', 'zorder': 0}, 
        {'label':"Ref Dubin's", 'us': ref_traj_us_dubins, 'xs': ref_traj_xs_dubins, 'dt': dt, 'color': 'tab:orange', 'zorder': 0},
        {'label':"MPC Stabilizing", 'us': us_stabilizing, 'xs': xs_stabilizing, 'dt': dt, 'color': 'tab:green', 'zorder': 1},
        {'label':"MPC Dubin's", 'us': us_dubins, 'xs': xs_dubins, 'dt': dt, 'color': 'tab:red', 'zorder': 1}
              ]

    mpc_stabilizing_term = xs_stabilizing[-1] - goal
    mpc_dubins_term = xs_dubins[-1] - goal
    energy_vel_stabilizing, energy_psi_stabilizing = compute_control_accels(us_stabilizing, dt)
    energy_vel_dubins, energy_psi_dubins = compute_control_accels(us_dubins, dt)
    print(f"MPC Stabilizing Termination: dx {mpc_stabilizing_term[0]:.3f}, dy {mpc_stabilizing_term[1]:.3f}, dhead {mpc_stabilizing_term[2]:.3f}, dsteer {mpc_stabilizing_term[3]:.3f} | energy vel / psi: {energy_vel_stabilizing:.3f} / {energy_psi_stabilizing:.3f}")
    print(f"MPC Dubins Termination     : dx {mpc_dubins_term[0]:.3f}, dy {mpc_dubins_term[1]:.3f}, dhead {mpc_dubins_term[2]:.3f}, dsteer {mpc_dubins_term[3]:.3f} | energy vel / psi: {energy_vel_dubins:.3f} / {energy_psi_dubins:.3f}")

    plot_traj_dicts(goal, results)
    plot_us_dicts(rover, results)
    plot_states_dicts(goal, results)
    # plot_traj(rover_mpc,goal,xs,us,"MPC", rover_mpc,reference_trajectory_xs,reference_trajectory_us,"Dubins")
    # plot_control(rover_mpc,goal,xs,us,"MPC", rover_ref,reference_trajectory_xs,reference_trajectory_us,"Reference")
    # plot_states(rover_mpc,goal,xs,us,"MPC", rover_ref,reference_trajectory_xs,reference_trajectory_us,"Reference")

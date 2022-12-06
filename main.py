import numpy as np
from matplotlib import pyplot as plt
from controllers import *
from simulation import *
from observations import *
from goalFactory import *
from rover import *
from state import *


# initialize robot 
x_0 = State(0,0,0.312,0) # starting pose, [px,py,theta,psi]
rover = Rover(x_0) # other parameters associated with rover are defined in the class initliaztion

# get goal pose
goal = generate_easy_goal_turn(x_0.state)
# goal = generate_easy_goal_straight(x_0.state)

# do reference trajectory "fake" control
# xs, ys, us = simulate(rover, goal, perfect_observations, kmb_sample_controller_1)
# xs, ys, us = simulate(rover, goal, perfect_observations, kmb_sample_controller_2)
xs, ys, us = simulate(rover, goal, perfect_observations, stabalizing_control)

# do MPC and "real" control 

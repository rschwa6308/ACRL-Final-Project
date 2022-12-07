import numpy as np
from state import *


def generate_easy_goal_straight(state):
    """Generate a goal directly in front of the agent at some distance

    Args:
        current_state (STATE): Current Agent Pose [px,py,theta,psi]

    Returns:
        goal_state (STATE): Goal State [px,py,theta,psi]
    """
    # TODO: Clean this function up
    
    x, y, theta, _ = state

    x_goal = x + 10 * np.cos(theta)
    y_goal = y + 10 * np.sin(theta)

    return np.array([x_goal,y_goal,theta,0])


def generate_easy_goal_turn(state):
    """Generate a goal with some randomness infront of the agent so it must turn slightly 

    Args:
        current_state (STATE): Current Agent Pose [px,py,theta,psi]

    Returns:
        goal_state (STATE): Goal State [px,py,theta,psi]
    """
    # TODO: Clean this function up

    x, y, theta, _ = state

    sign = 1
    if np.random.random() > 0.5:
        sign = -1

    # ang = sign* np.random.uniform(np.pi/3, np.pi/2)
    ang = sign* np.pi/2
    dist1 = 10
    dist2 = 8

    x_goal = x + dist1 * np.cos(theta) + dist2* np.cos(theta + ang)
    y_goal = y + dist1 * np.sin(theta) + dist2* np.sin(theta + ang)

    return np.array([x_goal,y_goal,theta+ang,0])

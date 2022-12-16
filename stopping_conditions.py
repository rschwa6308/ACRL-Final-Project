import numpy as np


def goal_reached(x, goal, epsilon=1e-3):
    return np.linalg.norm(x - goal) < epsilon

def goal_reached_separate_pos_heading(x, goal, pos_epsilon=0.2, head_epsilon=0.1):
    return np.linalg.norm(x[:2] - goal[:2]) < pos_epsilon and np.linalg.norm(x[2] - goal[2]) < head_epsilon


def goal_reached_ignore_heading(x, goal, epsilon=1e-2):
    return np.linalg.norm(x[:2] - goal[:2]) < epsilon

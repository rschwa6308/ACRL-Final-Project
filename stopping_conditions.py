import numpy as np


def goal_reached(x, goal, epsilon=1e-3):
    return np.linalg.norm(x - goal) < epsilon


def goal_reached_ignore_heading(x, goal, epsilon=1e-2):
    return np.linalg.norm(x[:2] - goal[:2]) < epsilon

import numpy as np
from matplotlib import pyplot as plt

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
    print("test")
    
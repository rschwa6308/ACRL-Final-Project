import numpy as np

class State:
    def __init__(self, px, py, theta, psi):
        self.state = np.array([px, py, theta, psi])

        

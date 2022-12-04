import numpy as np

def sample_controller_1(x):
    "Return a constant control (turn left)"
    return np.array([10, 0.2])

def sample_controller_2(x):
    "Regulate heading_angle to 0 with basic control gain"
    px, py, heading_angle = x
    return np.array([10, -0.1 * heading_angle])

def sample_controller_3(x):
    "random action"

    sign_drive = 1
    if (np.random.random() > 0.5):
        sign_drive = -1

    sign_steer = 1
    if (np.random.random() > 0.5):
        sign_steer = -1

    return np.array([sign_drive*10, sign_steer*np.random.random()])

# TODO: Create P control for velocity,

# TODO: Create STANLEY CONTROLLER

# TODO: Generate Refrence Trajcetory function 
    # take a goal and current pose, and output the full list of states X and controlls U 

# TODO: Create MPC controller function which calls MPC.py utility 


# Example Usage
if __name__ == "__main__":
    print("test")

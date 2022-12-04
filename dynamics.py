import numpy as np

def ackermann_dynamics(x, u, dt=0.1, slip_magnitude=0.2):
    """
    Idealized Ackermann Steering Dynamics

    State: px, py, heading_angle
    Control: speed, curvature (i.e. 1/signed_turning_radius)

    Note: The mapping from the robot's "steering control"
    (which presumably goes from -1 to +1) to the
    signed_turning_radius (-inf, +inf) will depend on the details
    of the robot geometry

    Slip (process noise) is modeled as a random normal added to the turning radius
    """
    print(x, u)
    px, py, heading_angle = x
    speed, curvature = u

    # move in a straight line if curvature is sufficiently small
    if abs(curvature) < 1e-6:
        return np.array([
            px + np.cos(heading_angle) * speed * dt,
            py + np.sin(heading_angle) * speed * dt,
            heading_angle
        ])
        

    turning_radius = 1/curvature

    # model slip
    turning_radius += np.random.normal(0.0, slip_magnitude)

    # common center of turning circle
    turn_circle_center = np.array([
        px + np.cos(heading_angle + np.pi/2) * turning_radius,
        py + np.sin(heading_angle + np.pi/2) * turning_radius
    ])
    print(turn_circle_center)

    # turn amount measured in radians about the turn circle center (signed!)
    turn_amount = speed * dt / turning_radius

    # angle with respect to turning center
    current_center_angle = np.arctan2(py - turn_circle_center[1], px - turn_circle_center[0])
    new_center_angle = current_center_angle + turn_amount
    
    px_new = turn_circle_center[0] + np.cos(new_center_angle) * abs(turning_radius)
    py_new = turn_circle_center[1] + np.sin(new_center_angle) * abs(turning_radius)
    heading_angle_new = heading_angle + turn_amount

    return np.array([px_new, py_new, heading_angle_new])



# Example Usage
if __name__ == "__main__":
    print("test")

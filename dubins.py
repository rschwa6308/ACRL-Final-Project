import numpy as np

# ---------------------------------------
# Utility functions
def pick_circle_center(pose, turn_radius, clockwise):
  """
  Pick circle center
  Args:
    clockwise (bool): driving direction
  Returns:
    [x,y] (np.ndarray): circle center
  """
  pose_x = pose[0]
  pose_y = pose[1]
  pose_theta = pose[2]

  if clockwise:
    # Right turn
    cx = pose_x + turn_radius * np.cos(pose_theta - np.pi/2)
    cy = pose_y + turn_radius * np.sin(pose_theta - np.pi/2)
  else:
    # Left turn
    cx = pose_x - turn_radius * np.cos(pose_theta - np.pi/2)
    cy = pose_y - turn_radius * np.sin(pose_theta - np.pi/2)
  
  return np.array([cx, cy])

def calc_inner_tangent():
  """
  Calculate inner tangent points
  """
  # TODO
  pass

def calc_outer_tangent(start_circle, end_circle, turn_radius, clockwise):
  """
  Calculate outer tangent points
  Args:
    clockwise (bool): outer tangent direction
  Returns:
    p_t1 [x,y,theta] (np.ndarray): tangent point on start circle
    p_t2 [x,y,theta] (np.ndarray): tangent point on end circle
  """
  c1_x = start_circle[0]
  c1_y = start_circle[1]
  c2_x = end_circle[0]
  c2_y = end_circle[1]

  if clockwise:
    # Right turn
    phi = np.arctan2(c2_y-c1_y, c2_x-c1_x) + np.pi/2
  else:
    # Left turn
    phi = np.arctan2(c2_y-c1_y, c2_x-c1_x) - np.pi/2

  p_t1x = c1_x + turn_radius * np.cos(phi)
  p_t1y = c1_y + turn_radius * np.sin(phi)
  p_t2x = c2_x + turn_radius * np.cos(phi)
  p_t2y = c2_y + turn_radius * np.sin(phi)

  theta = np.arctan2(p_t2y-p_t1y,p_t2x-p_t1x)

  return np.array([p_t1x, p_t1y, theta]), np.array([p_t2x, p_t2y, theta])

def calc_arc_length_to_tangent(pose1, pose2, circle_center, turn_radius, clockwise):
  """
  Calculate signed arc length to tangent point (+ for ccw; - for cw by RH rule)
  - Goes from pose1 to pose2
  """
  # Angle between start and end poses
  psi = np.arctan2(pose2[1]-circle_center[1],pose2[0]-circle_center[0]) - np.arctan2(pose1[1]-circle_center[1],pose1[0]-circle_center[0])
  
  # Check for direction
  if clockwise:
    if psi > 0:
      # Force right turn
      psi -= 2 * np.pi
  else:
    if psi < 0:
      # Force left turn
      psi += 2 * np.pi

  # Final arc length
  return turn_radius * psi

def calc_straight_length(p_t1, p_t2):
  """
  Calculate length of straight segment
  """
  return np.linalg.norm(p_t1 - p_t2)

def calc_traj_length(path):
  """
  Calculate word trajectory length
  """
  return sum([abs(l_seg) for l_seg in path])

def calc_control_traj(start_pose, turn_radius, max_steer_angle, path, velocity, dt):
  """
  Calculate control along trajectory
  """
  ds = velocity * dt # step size along trajectory
  traj = [[start_pose[0], start_pose[1], start_pose[2], 0]] # pack with [x,y,theta,psi]

  # First arc
  n_s = int(np.floor(abs(path[0]) / ds)) # number of segments
  for _ in range(n_s):
    last_x = traj[-1][0]
    last_y = traj[-1][1]
    last_theta = traj[-1][2]

    new_x = last_x + ds * np.cos(last_theta)
    new_y = last_y + ds * np.sin(last_theta)
    new_theta = (last_theta + ds / turn_radius * np.sign(path[0])) % (2*np.pi)
    psi = max_steer_angle * np.sign(path[0])

    traj.append([new_x, new_y, new_theta, psi])

  # Straight
  n_s = int(np.floor(abs(path[1]) / ds)) # number of segments
  for _ in range(n_s):
    last_x = traj[-1][0]
    last_y = traj[-1][1]
    last_theta = traj[-1][2]

    new_x = last_x + ds * np.cos(last_theta)
    new_y = last_y + ds * np.sin(last_theta)
    new_theta = (last_theta) % (2*np.pi)
    psi = 0.0

    traj.append([new_x, new_y, new_theta, psi])

  # Second arc
  n_s = int(np.floor(abs(path[2]) / ds)) # number of segments
  for _ in range(n_s):
    last_x = traj[-1][0]
    last_y = traj[-1][1]
    last_theta = traj[-1][2]

    new_x = last_x + ds * np.cos(last_theta)
    new_y = last_y + ds * np.sin(last_theta)
    new_theta = (last_theta + ds / turn_radius * np.sign(path[0])) % (2*np.pi)
    psi = max_steer_angle * np.sign(path[0])

    traj.append(np.array([new_x, new_y, new_theta, psi]))

  return traj
# ---------------------------------------

# ---------------------------------------
# Dubin's functions
def dubins_rsr(start_pose, end_pose, turn_radius):
  """
  RSR case
  - Outer tangent
  Args:
    start_pose [x,y,theta] (np.ndarray)
  Returns:
    [first arc length, straight length, second arc length] : signed radii with lengths
  """
  # TODO
  # Calculate right turn circle for start and end poses
  start_circle = pick_circle_center(start_pose, turn_radius, clockwise=True)
  end_circle = pick_circle_center(end_pose, turn_radius, clockwise=True)
  # print("start pose:", start_pose, "end_pose:", end_pose)
  # print("start_circle:", start_circle, "end_circle:", end_circle)

  # Calculate outer tangent points between circles
  p_t1, p_t2 = calc_outer_tangent(start_circle, end_circle, turn_radius, clockwise=True)
  # print("p_t1:", p_t1, "p_t2:", p_t2)

  # Select arc lengths for both circles
  l_arc1 = calc_arc_length_to_tangent(start_pose, p_t1, start_circle, turn_radius, clockwise=True)
  l_arc2 = calc_arc_length_to_tangent(p_t2, end_pose, end_circle, turn_radius, clockwise=True)
  # print("l_arc1:", l_arc1, "l_arc2:", l_arc2)

  # Calculate straight segment
  l_straight = calc_straight_length(p_t1, p_t2)
  # print("l_straight:", l_straight)
  
  return [l_arc1, l_straight, l_arc2]

def dubins_lsl():
  """
  LSL case
  - Outer tangent
  """
  # TODO
  pass

def dubins_rsl():
  """
  RSL case
  - Inner tangent
  """
  # TODO
  pass

def dubins_lsr():
  """
  LSR case
  - Inner tangent
  """
  # TODO
  pass

def dubins_main(start_pose, end_pose, turn_radius, max_steer_angle, velocity, dt):
  """
  Main algorithm
  Returns:
    [x,y,theta,psi] (np.ndarray, Nx4)
  """
  # For each word (4 cases): signed arc length --> negative := right turn, position := left turn
  path_rsr = dubins_rsr(start_pose, end_pose, turn_radius)

  # Select shortest length
  l_path = calc_traj_length(path_rsr)

  # Calculate desired reference control along trajectory (descretization based on ds)
  shortest_path = path_rsr # TODO: update based on actual lengths
  dubins_traj = calc_control_traj(start_pose, turn_radius, max_steer_angle, shortest_path, velocity, dt)


  return dubins_traj
# ---------------------------------------


if __name__ == '__main__':
  # Example usage
  # Start pose: <x,y,theta>
  start_pose = np.array([0,0,np.pi/2])

  # End pose: <x,y,theta>
  end_pose = np.array([10,0,3*np.pi/2])

  # Dubins parameters
  turn_radius = 1 
  dt = 0.1 # time step for trajectory
  velocity = 1
  max_steer_angle = 0.5 # get from rover

  traj = dubins_main(start_pose, end_pose, turn_radius, max_steer_angle, velocity, dt)

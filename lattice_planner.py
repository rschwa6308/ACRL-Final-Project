import numpy as np
from lattice import *
from agent import *
from queue import PriorityQueue


def calculate_traj_heuristic(goal_pose, trajectories, epsilon=1):
  traj_h = []

  # Cost from goal
  for traj in trajectories:
    dist_from_goal = epsilon * np.sqrt((traj[0,-1] - goal_pose.x)**2 + (traj[1,-1] - goal_pose.y)**2)
    traj_h.append(dist_from_goal)
    
  return traj_h

def calculate_traj_cost(trajectories, s):
  '''
  w_topo (float): Tune cost of topography variation relative to path length
  '''
  traj_costs = []

  for traj in trajectories:
    # Path length
    path_cost = s
    traj_costs.append(path_cost)

  return traj_costs

class AstarNode:
  def __init__(self, g, parent_idx, idx, pose, traj=[]):
    self.g = g
    self.parent_idx = parent_idx
    self.idx = idx
    self.pose = pose
    self.traj = traj

  def __lt__(self, obj):
      """self < obj."""
      return self.g < obj.g

  def __le__(self, obj):
      """self <= obj."""
      return self.g <= obj.g

  def __eq__(self, obj):
      """self == obj."""
      return self.g == obj.g

  def __ne__(self, obj):
      """self != obj."""
      return self.g != obj.g

  def __gt__(self, obj):
      """self > obj."""
      return self.g > obj.g

  def __ge__(self, obj):
      """self >= obj."""
      return self.g >= obj.g

def smallest_angle_difference(angle1, angle2):
  '''
  Returns smallest difference between two angles
  All units in radians
  '''
  diff = abs(angle1 % (2*np.pi) - angle2 % (2*np.pi)) % (2*np.pi)
  if diff > np.pi:
    diff = abs(diff - 2*np.pi)
  return diff

def same_pose_with_thresh(pose1, pose2, thresh_pos, thresh_yaw):
  if np.sqrt((pose1.x - pose2.x)**2 + (pose1.y - pose2.y)**2) <= thresh_pos and \
      smallest_angle_difference(pose1.yaw, pose2.yaw) <= thresh_yaw:
    return True
  return False

def get_traj_pos_closest_to_goal(traj, goal_pose):
  # Track the best pose
  best_idx = 0
  best_pose = Pose(traj[0,best_idx], traj[1,best_idx], traj[2,best_idx])
  best_pos = np.inf

  # Check all points in trajectory
  for col in range(traj.shape[1]):
    # Get new difference in position and yaw
    new_pos = np.sqrt((traj[0,col] - goal_pose.x)**2 + (traj[1,col] - goal_pose.y)**2)

    if new_pos < best_pos:
      # Update best
      best_pose.x = traj[0,col]
      best_pose.y = traj[1,col]
      best_pose.yaw = traj[2,col]
      best_idx = col
      
      # Update trackers
      best_pos = new_pos

  return best_pose, best_idx

def pose_within_thresh_to_visited(pose, visited_nodes, thresh_pos, thresh_yaw):
  for node in visited_nodes:
    if same_pose_with_thresh(pose, node.pose, thresh_pos, thresh_yaw):
      return True
  return False

def astar_search(start_pose, goal_pose, lattice, s, epsilon, thresh_pos=0.2, thresh_yaw=np.deg2rad(10), plot_search_end=False, plot_search_live=False, plot_time_step=0.01, arrow_size=1):
  '''
  State Lattice Algorithm:
  1. Initialize agent, map, goals (goal = pose = 2D x,y,heading)
  2. Initialize priority queue with current agent pose
  3. Until agent final pose within threshold of goal, or pqueue empty (thresh for position, thresh for heading)
  4.   Pop next agent successor pose from pqueue
  5.   Generate agent successor poses using parameterized lattice
  6.   Calculate node cost g(s') = g(s) + c(s,s') for each lattice trajectory by weighting cell crossings (prune collisions)
  7.   Calculate heuristic h(s') using L2 norm of state vector (i.e. x,y,heading)
  8.   Push each lattice pose into priority queue f(s') = g(s') + h(s')

  Pros: Ensures kinodynamic feasibility, relatively easy to apply costs and constraints
  Cons: May not find a solution if thresholds too small, search space might get large
  '''

  # assert(map.xy_on_map(start_pose.x, start_pose.y)) # Make sure start is on the map

  final_path = [] # list of final trajectories, composed from lattice primitives
  visited_traj = [] # trajectories visited 
  expanded_traj = []
  visited_nodes = [] 

  # Add first agent pose to pqueue
  pqueue = PriorityQueue() # Lower cost has higher priority i.e. earlier in queue
  start_traj = np.array([[start_pose.x], [start_pose.y], [start_pose.yaw]])
  start_node = AstarNode(g=0, parent_idx=0, idx=0, pose=start_pose, traj=start_traj)
  pqueue.put((0.0, start_node))

  print(f"Searching for A* path from start pose (x={start_pose.x:.2f}, y={start_pose.y:.2f}, yaw={np.rad2deg(start_pose.yaw):.2f}) to goal pose (x={goal_pose.x:.2f}, y={goal_pose.y:.2f}, yaw={np.rad2deg(goal_pose.yaw):.2f})...")
    
  num_iter = 0
  while not pqueue.empty():
    num_iter += 1
    if num_iter % 100 == 0:
      print("Live lattice planner iterations:", num_iter)
    # Pop new pose from pqueue
    curr_node = pqueue.get()[1]
    expanded_traj.append(curr_node.traj)

    # Transform lattice to new pose to get successors
    lattice_trajectories = transform_lattice_to_pose(lattice, curr_node.pose)

    # Check for goal along current best trajectory
    closest_traj_pose, closest_traj_idx = get_traj_pos_closest_to_goal(curr_node.traj, goal_pose)
    if same_pose_with_thresh(closest_traj_pose, goal_pose, thresh_pos, thresh_yaw):
      curr_idx = curr_node.idx

      # Add final trajectory to final path
      if len(visited_traj) == 0:
        # Edge case: already at goal, so trajectory is just the same point
        visited_traj.append(curr_node.traj)
        visited_nodes.append(curr_node)
      final_path.append(visited_traj[curr_idx][:, :closest_traj_idx+1])

      # Get final path
      curr_idx = visited_nodes[curr_idx].parent_idx
      while curr_idx != 0:
        final_path.append(visited_traj[curr_idx])
        curr_idx = visited_nodes[curr_idx].parent_idx
      break

    # Remove trajectories that have any points outside map bounds
    valid_trajectories = []
    for traj in lattice_trajectories:
      # if not traj_outside_bounds(map, traj):
      valid_trajectories.append(traj)

    # Make trajectory heuristics h(s')
    traj_h = calculate_traj_heuristic(goal_pose, valid_trajectories, epsilon)
    
    # Get trajectory costs c(s,s')
    traj_costs = calculate_traj_cost(valid_trajectories, s)
    
    for h,c,traj in zip(traj_h,traj_costs,valid_trajectories):

      # Skip if trajectory is oustide map bounds
      # if traj_outside_bounds(map, traj):
        # continue

      # Calculate next node
      succ_pose = Pose(x=traj[0,-1], y=traj[1,-1], yaw=traj[2,-1])
      succ_g = curr_node.g + c

      # Skip if node pose is close to another node that's already been visited
      skip_new_node = False
      for visited_node in visited_nodes:
        if same_pose_with_thresh(succ_pose, visited_node.pose, thresh_pos=0.05, thresh_yaw=np.deg2rad(25)) and succ_g >= visited_node.g:
            skip_new_node = True
            break
      if skip_new_node:
        continue

      # Update node costs g(s') = g(s) + c(s,s')
      visited_traj.append(traj)
      succ_node = AstarNode(g=succ_g, parent_idx=curr_node.idx, idx=len(visited_traj)-1, pose=succ_pose, traj=traj)
      visited_nodes.append(succ_node)
      
      # Add new poses to pqueue using f(s') = g(s') + h(s')
      f = succ_g + h
      pqueue.put((f, succ_node))
      
  if len(final_path) == 0:
    print("[WARN] No solution found, returning empty path...")
  print(f"Final lattice planner iterations: {num_iter}")
  return final_path

def plot_path(ax, path, path_color='lime'):
  for traj in path:
    ax.plot(traj[0,:], traj[1,:], linestyle='-', marker='.', markersize=10, linewidth=4, color='black', zorder=np.inf)
    # ax.plot(traj[0,:], traj[1,:], linestyle='-', marker='.', markersize=5, linewidth=2, color=path_color, zorder=np.inf)

if __name__ == '__main__':

  init_x = 3.91 # coordinates
  init_y = 2.99 # coordinates
  init_yaw = np.deg2rad(21.49) # radians (plotted with RH rule: x-forward, y-left, z-up)

  min_turn_radius = 0.8 # assume kinematic motion [m]
  agent = Agent(init_x, init_y, init_yaw, min_turn_radius)
  
  # goal_pose = Pose(x=2.0, y=2.2, yaw=np.deg2rad(270.0))
  goal_pose = Pose(x=2.0, y=7.0, yaw=np.deg2rad(90.00))
  
  # ---------------------------------------------------------------------------
  # Lattice
  # Generate lattice offline
  turn_radii = [min_turn_radius, min_turn_radius*2, 0, -min_turn_radius*2, -min_turn_radius] # Positive = left turn, negative = right turn (RH rule)
  s = 0.5 # trajectory length [m]
  ds = 0.05 # trajectory parameterization resolution [m]
  lattice = generate_agent_lattice(turn_radii, s, ds)
  
  # Convert agent space to map space
  agent_lattice = transform_lattice_to_pose(lattice, agent.pose)

  # Search nodes for best trajectory
  start_pose = Pose(agent.pose.x, agent.pose.y, agent.pose.yaw)
  thresh_pos = 0.5
  thresh_yaw = np.deg2rad(20)
  w_topo = 0.0
  epsilon = 2.0
  path = astar_search(start_pose, goal_pose, lattice, s, epsilon, thresh_pos, thresh_yaw, plot_search_live=False, plot_time_step=0.0000000001)

  # print(np.shape(path))
  # print(path)

  # ---------------------------------------------------------------------------
  # Plot
  fig,ax = plt.subplots()
  arrow_size = 1.0
  plot_pose(ax, agent.pose, pose_color='blueviolet', arrow_size=arrow_size)
  plot_lattice(ax, agent_lattice)
  plot_pose(ax, goal_pose, pose_color='gold', arrow_size=arrow_size)
  plot_path(ax, path)
  plt.show()

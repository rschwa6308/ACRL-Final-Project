from queue import PriorityQueue
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent, plot_pose
import copy

class Point:
  def __init__(self,x,y):
    self.x = x
    self.y = y

def generate_agent_lattice(turn_radii, s, ds):
  '''
  Generate successor agent poses from given agent, following kinematic constraints TODO: could extend to kinodynamic to include velocity/accel. For now assume a constant velocity
  agent_curr : the current agent to generate successors for
  s : total arc length for successor trajectory [m]
  ds : max arc distance between parameterized points in trajectory [m]
  '''
  # Initialize list of next agent poses [x,y,yaw]
  lattice = []
  
  # Set number of parameterization points in trajectories, by the maximum distance between points 
  # -> If not exact, last segment will be slightly smaller
  num_s = int(np.ceil(s / ds))

  # Forward cases
  for turn_radius in turn_radii:

    # Forward cases
    if turn_radius == 0:
      # Straight case
      t = np.linspace(0, s, num_s+1) # Parameterization steps
      traj_straight = np.vstack([t, 
                                 np.zeros(num_s+1),
                                 np.zeros(num_s+1)])
      lattice.append(traj_straight)
    else:
      if turn_radius > 0:
        # Left turn
        start_rad = -np.pi/2 # Start from bottom of unit circle
        stop_rad = s / abs(turn_radius) - (np.pi/2) # Subtract pi/2 from positive angle for offset to lower part of unit circle
        t = np.linspace(start_rad, stop_rad, num_s+1) # Parameterization steps
        traj_left = np.vstack([abs(turn_radius) * np.cos(t),
                               abs(turn_radius) * np.sin(t) + abs(turn_radius),
                               np.pi + np.arctan2(-np.cos(t), np.sin(t))]) # Add radius to offset from unit circle to origin
        lattice.append(traj_left)
      else: # turn radius < 0
        # Right turn
        start_rad = np.pi/2 # start from top of unit circle
        stop_rad = -s / abs(turn_radius) + (np.pi/2) # Add pi/2 to negative angle for offset to upper part of unit circle
        t = np.linspace(start_rad, stop_rad, num_s+1) # Parameterization steps
        traj_right = np.vstack([abs(turn_radius) * np.cos(t),
                                abs(turn_radius) * np.sin(t) - abs(turn_radius),
                                np.arctan2(-np.cos(t), np.sin(t))]) # Subtract radius to offset from unit circle to origin; subtract pi/2 from heading to re-center with map
        lattice.append(traj_right)

    # Reverse cases
    if turn_radius == 0:
      # Straight case
      t = np.linspace(0, s, num_s+1) # Parameterization steps
      traj_straight = np.vstack([-t, 
                                 np.zeros(num_s+1),
                                 np.zeros(num_s+1)])
      lattice.append(traj_straight)
    else:
      if turn_radius > 0:
        # Left turn
        start_rad = -np.pi/2 # Start from bottom of unit circle
        stop_rad = -s / abs(turn_radius) - (np.pi/2) # Subtract pi/2 from positive angle for offset to lower part of unit circle
        t = np.linspace(start_rad, stop_rad, num_s+1) # Parameterization steps
        traj_left = np.vstack([abs(turn_radius) * np.cos(t),
                               abs(turn_radius) * np.sin(t) + abs(turn_radius),
                               np.pi + np.arctan2(-np.cos(t), np.sin(t))]) # Add radius to offset from unit circle to origin; subtract pi/2 from heading to re-center with map
        lattice.append(traj_left)
      else: # turn radius < 0
        # Right turn
        start_rad = np.pi/2 # start from top of unit circle
        stop_rad = s / abs(turn_radius) + (np.pi/2) # Add pi/2 to negative angle for offset to upper part of unit circle
        t = np.linspace(start_rad, stop_rad, num_s+1) # Parameterization steps
        traj_right = np.vstack([abs(turn_radius) * np.cos(t),
                                abs(turn_radius) * np.sin(t) - abs(turn_radius),
                                np.arctan2(-np.cos(t), np.sin(t))]) # Subtract radius to offset from unit circle to origin; subtract pi/2 from heading to re-center with map
        lattice.append(traj_right)

  return lattice

def transform_2d_pose_pt_to_world(pt_agent, pose):
  # Assuming agent has x-forward, y-left, z-up 
  R = np.array([[np.cos(pose.yaw), -np.sin(pose.yaw), 0, pose.x], 
                [np.sin(pose.yaw),  np.cos(pose.yaw), 0, pose.y], 
                [                0,                  0, 1,       0],
                [                0,                  0, 0,       1]])

  pt_map_np = R @ np.array([pt_agent.x, pt_agent.y, 0, 1])
  
  return Point(pt_map_np[0], pt_map_np[1])

def transform_lattice_to_pose(lattice_template, pose):
  lattice = copy.deepcopy(lattice_template) # Don't modify by reference, so template can be reused
  for traj in lattice:
    for col in range(traj.shape[1]):
      transformed_pt = transform_2d_pose_pt_to_world(Point(traj[0,col], traj[1,col]), pose)
      traj[0,col] = transformed_pt.x
      traj[1,col] = transformed_pt.y
      traj[2,col] += pose.yaw
  return lattice 

def plot_lattice(ax, lattice, arrow_size=0.1):
  # Make heading marker
  arrow_head_length = 0.2 * arrow_size
  arrow_head_width = 0.15 * arrow_size
  arrow_body_length = 0.8 * arrow_size
  minor_axis_scale = 0.6
  arrow_color = 'black'
  for traj in lattice:
    ax.plot(traj[0,:], traj[1,:], marker='.')
    for pt_i in range(traj.shape[1]):
      ax.arrow(traj[0,pt_i], traj[1,pt_i], 
            arrow_body_length*minor_axis_scale * np.cos(traj[2,pt_i]), 
            arrow_body_length*minor_axis_scale * np.sin(traj[2,pt_i]), 
            head_width=arrow_head_width, head_length=arrow_head_length, 
            fc=arrow_color, ec=arrow_color,
            zorder=np.inf-1)

if __name__ == '__main__':
  # Agent
  init_x = 0.0 # coordinates
  init_y = 0.0 # coordinates
  init_yaw = np.deg2rad(0) # radians (plotted with RH rule)
  min_turn_radius = 1.0 # assume kinematic motion [m]
  agent = Agent(init_x, init_y, init_yaw, min_turn_radius)

  # Generate lattice offline
  turn_radii = [min_turn_radius, min_turn_radius*2, 0, -min_turn_radius*2, -min_turn_radius] # Positive = left turn, negative = right turn (RH rule)
  s = np.pi/2 # trajectory length [m]
  ds = 0.1 # trajectory parameterization resolution [m]
  lattice = generate_agent_lattice(turn_radii, s, ds)
  
  # Convert agent space to map space
  lattice = transform_lattice_to_pose(lattice, agent.pose)

  # Plot
  fig,ax = plt.subplots()
  plot_lattice(ax, lattice)
  plot_pose(ax, agent.pose, arrow_size=0.5)
  plt.axis('equal') # Makes circles looks like circles
  plt.show()

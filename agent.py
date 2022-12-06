from queue import PriorityQueue
import numpy as np
import matplotlib.pyplot as plt
from pose import *

class Agent:
  def __init__(self, init_x, init_y, init_yaw, min_turn_radius):
    self.pose = Pose(init_x, init_y, init_yaw)
    self.min_turn_radius = min_turn_radius

if __name__ == '__main__':
  # Agent
  init_x = 0.0 # coordinates
  init_y = 0.0 # coordinates
  init_yaw = np.deg2rad(45) # radians (plotted with RH rule; x-forward, y-left, z-up )
  min_turn_radius = 0.5 # assume kinematic motion [m]
  agent = Agent(init_x, init_y, init_yaw, min_turn_radius)

  fig,ax = plt.subplots()
  plot_pose(ax, agent.pose)
  plt.axis('equal') # Makes circles looks like circles
  plt.show()

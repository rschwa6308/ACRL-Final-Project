import numpy as np
class Pose:
  def __init__(self, x, y, yaw):
    self.x = x
    self.y = y
    self.yaw = yaw % (2*np.pi)

def plot_pose(ax, pose, pose_color='black', arrow_size=1, marker='*'):
  ax.plot(pose.x, pose.y, marker=marker, markersize=12, color=pose_color, zorder=np.inf)
  # Make heading marker
  arrow_head_length = 0.2 * arrow_size
  arrow_head_width = 0.15 * arrow_size
  arrow_body_length = 0.8 * arrow_size
  minor_axis_scale = 0.6
  # x-axis
  ax.arrow(pose.x, pose.y, 
           arrow_body_length * np.cos(pose.yaw), 
           arrow_body_length * np.sin(pose.yaw), 
           head_width=arrow_head_width, head_length=arrow_head_length, 
           fc=pose_color, ec=pose_color,
           zorder=np.inf)
  # y-axis
  ax.arrow(pose.x, pose.y, 
          arrow_body_length*minor_axis_scale * np.cos(pose.yaw + np.pi/2), 
          arrow_body_length*minor_axis_scale * np.sin(pose.yaw + np.pi/2), 
          head_width=arrow_head_width, head_length=arrow_head_length, 
          fc=pose_color, ec=pose_color,
          zorder=np.inf)
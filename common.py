import numpy as np

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
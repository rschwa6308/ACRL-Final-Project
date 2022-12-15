import numpy as np

# ---------------------------------------
# Utility functions
def pick_circle_center():
  """
  Pick circle center
  """
  # TODO
  pass

def calc_inner_tangent():
  """
  Calculate inner tangent points
  """
  # TODO
  pass

def calc_outer_tangent():
  """
  Calculate outer tangent points
  """
  # TODO
  pass

def calc_arc_length_to_tangent():
  """
  Calculate arc length to tangent point
  """
  # TODO
  pass

def calc_traj_length():
  """
  Calculate word trajectory length
  """
  # TODO
  pass

def calc_control_length():
  """
  Calculate control along trajectory
  """
  # TODO
  pass
# ---------------------------------------

# ---------------------------------------
# Dubin's functions
def dubins_rsr():
  """
  RSR case
  - Outer tangent
  """
  # TODO
  pass

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

def dubins_main(start_pose, end_pose, turn_radius, ds):
  # TODO
  # Main algorithm
    # For each word (4 cases)
      # Select circle centers
      # Calculate tangent points between circles (inner or outer)
      # Select arc lengths for both circles
      # Compute total word length
    # Select shortest length
    # Calculate desired reference control along trajectory (descretization based on ds)
  pass
# ---------------------------------------


if __name__ == '__main__':
  # Example usage
  # Start pose: <x,y,theta>
  start_pose = np.array([0,0,np.pi/2])

  # End pose: <x,y,theta>
  end_pose = np.array([10,0,3*np.pi/2])

  # Dubins parameters
  turn_radius = 1 
  ds = 0.1 # step size between control inputs for final trajecotry

  traj = dubins_main(start_pose, end_pose, turn_radius, ds)

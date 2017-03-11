import numpy as np
from DDP import *

import pdb


if __name__ == '__main__':
  desired_destination_x = 5.0
  desired_destination_y = 10.0
  lr = 0.00025

  # node: start
  node_start = DynamicsSystemNode('start', next_primitive_name='shoot')
  node_start.ssa.updateStateElement('shooter', np.array([0.0, 0.0]))
  node_start.ssa.updateActionElement('shoot_angle', np.array([0.0]))

  # node: destination
  node_destination = DynamicsSystemNode('destination', prev_primitive_name='shoot')

  # dynamics: shoot (start -> destination)
  dynamics_shoot = DynamicsSystemEdgeDynamicsModel()

  def dynamics_shoot_dynamics_func (ssa):
    next_ssa = ssa.copy()
    next_ssa.updateSelection(None)
    destination_x = desired_destination_x
    destination_y = (destination_x - ssa.retrive('shooter')[0]) * np.tan(ssa.retrive('shoot_angle')[0]) + ssa.retrive('shooter')[1]
    next_ssa.updateStateElement('destination', np.array([destination_x, destination_y]))
    return next_ssa
  dynamics_shoot.dynamics_func = dynamics_shoot_dynamics_func

  def dynamics_shoot_dynamics_dfunc (ssa):
    d = {}
    d_destx_theta = 0
    d_desty_theta = (desired_destination_x - ssa.retrive('shooter')[0]) * (1 + np.tan(ssa.retrive('shoot_angle')[0])**2)
    d['destination'] = {}
    d['destination']['shoot_angle'] = np.array([[d_destx_theta, d_desty_theta]])
    return d
  dynamics_shoot.dynamics_dfunc = dynamics_shoot_dynamics_dfunc

  def dynamics_shoot_reward_func (next_ssa):
    r = - (next_ssa.retrive('destination')[1] - desired_destination_y)**2
    return r
  dynamics_shoot.reward_func = dynamics_shoot_reward_func

  def dynamics_shoot_reward_dfunc (next_ssa):
    d = {}
    d_r_destx = 0
    d_r_desty = - 2 * (next_ssa.retrive('destination')[1] - desired_destination_y)
    d['destination'] = np.array([d_r_destx, d_r_desty])
    return d
  dynamics_shoot.reward_dfunc = dynamics_shoot_reward_dfunc

  # dynamics system
  dynamics_system = LinearDynamicsSystem()
  dynamics_system.updateNode(node_start.name, node_start)
  dynamics_system.updateNode(node_destination.name, node_destination)
  dynamics_system.updateRootNodeName(node_start.name)
  dynamics_system.updateDynamics('shoot', 'start', dynamics_shoot, 'destination', alpha=lr)

  # execute
  """
  last_J = None
  for i in range(1000):
    J, a_err = dynamics_system.optimizeActionsOnce()
    print('Round: {}, Value: {}, Error: {}'.format(i, J, a_err))

    if last_J is not None:
      delta_J = abs(J - last_J)
      if delta_J < 0.000001:
        break
    last_J = J
  """
  def optimizeActions_shouldStop(indicators):
    if indicators['rounds'] > 1000:
      return True
    if abs(indicators['value'] - indicators['last_value']) < 0.000001:
      return True
    return False
  dynamics_system.optimizeActions(optimizeActions_shouldStop, verbose=1)

  print('shooter: ({}, {}), shoot_angle: {}, destination: ({}, {})'.format(
    node_start.ssa.retrive('shooter')[0], node_start.ssa.retrive('shooter')[1],
    node_start.ssa.retrive('shoot_angle')[0],
    node_destination.ssa.retrive('destination')[0], node_destination.ssa.retrive('destination')[1]
  ))



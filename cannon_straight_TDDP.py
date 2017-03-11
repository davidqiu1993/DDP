import numpy as np
from DDP import *

import pdb


if __name__ == '__main__':
  desired_destination_x = 5.0
  desired_destination_y = 2.0

  # node: start
  node_start = DynamicsSystemNode('start', next_primitive_name='shoot')
  node_start.ssa.updateStateElement('shooter', np.array([0.0, 0.0]))
  node_start.ssa.updateActionElement('shoot_angle', np.array([0.0]))

  # node: destination
  node_destination = DynamicsSystemNode('destination', prev_primitive_name='shoot')

  # transition: shoot
  trans_shoot = DynamicsSystemTransitionModel()

  def trans_shoot_transition_func (ssa):
    prob = { 'destination': 1.0 }
    return prob
  trans_shoot.transition_func = trans_shoot_transition_func

  def trans_shoot_transition_dfunc (ssa):
    d = {}
    T = trans_shoot_transition_func(ssa)
    for b in T:
      d[b] = {}
      for k in ssa.keys():
        if k == 'selection':
          d[b][k] = np.zeros((1))
        else:
          d[b][k] = np.zeros((ssa.retrive(k).shape[0]))
    return d
  trans_shoot.transition_dfunc = trans_shoot_transition_dfunc

  # edge dynamics: shoot -> destination
  edge_shoot_destination = DynamicsSystemEdgeDynamicsModel()

  def edge_shoot_destination_dynamics_func (ssa):
    next_ssa = ssa.copy()
    next_ssa.updateSelection(None)
    destination_x = desired_destination_x
    destination_y = (destination_x - ssa.retrive('shooter')[0]) * np.sin(ssa.retrive('shoot_angle')[0]) + ssa.retrive('shooter')[1]
    next_ssa.updateStateElement('destination', np.array([destination_x, destination_y]))
    return next_ssa
  edge_shoot_destination.dynamics_func = edge_shoot_destination_dynamics_func

  def edge_shoot_destination_dynamics_dfunc (ssa):
    next_ssa = edge_shoot_destination_dynamics_func(ssa)
    d = {}
    for next_ssa_elem in next_ssa.keys():
      d[next_ssa_elem] = {}
      for ssa_elem in ssa.keys():
        if next_ssa_elem == 'selection' or ssa_elem == 'selection':
          d[next_ssa_elem][ssa_elem] = np.zeros((1, 1))
        elif next_ssa_elem == ssa_elem:
          assert(ssa.retrive(ssa_elem).shape[0] == next_ssa.retrive(next_ssa_elem).shape[0])
          d[next_ssa_elem][ssa_elem] = np.eye(ssa.retrive(ssa_elem).shape[0])
        else:
          d[next_ssa_elem][ssa_elem] = np.zeros((ssa.retrive(ssa_elem).shape[0], next_ssa.retrive(next_ssa_elem).shape[0]))
    d_destx_theta = 0
    d_desty_theta = (desired_destination_x - ssa.retrive('shooter')[0]) * np.cos(ssa.retrive('shoot_angle')[0])
    d['destination']['shoot_angle'] = np.array([[d_destx_theta, d_desty_theta]])
    return d
  edge_shoot_destination.dynamics_dfunc = edge_shoot_destination_dynamics_dfunc

  def edge_shoot_destination_reward_func (next_ssa):
    r = - (next_ssa.retrive('destination')[1] - desired_destination_y) ** 2
    return r
  edge_shoot_destination.reward_func = edge_shoot_destination_reward_func

  def edge_shoot_destination_reward_dfunc (next_ssa):
    d = {}
    for k in next_ssa.keys():
      if k == 'selection':
        d[k] = np.zeros((1))
      else:
        d[k] = np.zeros((next_ssa.retrive(k).shape[0]))
    d_destx = 0
    d_desty = - 2 * (next_ssa.retrive('destination')[1] - desired_destination_y)
    d['destination'] = np.array([d_destx, d_desty])
    return d
  edge_shoot_destination.reward_dfunc = edge_shoot_destination_reward_dfunc

  # primitive: shoot
  primitive_shoot = DynamicsSystemPrimitive('shoot', alpha=0.0025)
  primitive_shoot.prev_node_name = node_start.name
  primitive_shoot.transition = trans_shoot
  primitive_shoot.dynamics_dict = {}
  primitive_shoot.dynamics_dict[node_destination.name] = edge_shoot_destination

  # dynamics system
  dynamics_system = TreeDynamicsSystem()
  dynamics_system.updateNode(node_start.name, node_start)
  dynamics_system.updateNode(node_destination.name, node_destination)
  dynamics_system.updateRootNodeName(node_start.name)
  dynamics_system.updatePrimitive(primitive_shoot.name, primitive_shoot)

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



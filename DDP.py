#!/usr/bin/env python

"""
DDP.py
Differential dynamic programming toolkit.
"""

__version__     = "1.0.0"
__author__      = "David Qiu"
__email__       = "david@davidqiu.com"
__website__     = "www.davidqiu.com"
__copyright__   = "Copyright (C) 2017, David Qiu. All rights reserved."


import numpy as np
from collections import deque

import pdb



class SSA(object):
  """
  Super-state-action dictionary consists of state elements, action elements and 
  branch selection. All elements, except for the selection preference, should 
  be arrays, even if there is only a single value. Note that all names of the 
  elements should be unique. The name `selection` is reserved for the selection 
  element.

  @property state_dict The dictionary that maps state element names to 
                       corresponding state elements.
  @property action_dict The dictionary that maps action element names to 
                        corresponding action elements.
  @property selection The branch selection preference.
  @property info_dict The dictionary that maps name to corresponding additional 
                      information.
  """

  def __init__(self, state_dict=None, action_dict=None, selection=None):
    super(SSA, self).__init__()

    if state_dict is None:
      self.state_dict = {}
    else:
      assert(type(state_dict) is dict)
      self.state_dict = state_dict

    if action_dict is None:
      self.action_dict = {}
    else:
      assert(type(action_dict) is dict)
      self.action_dict = action_dict

    self.selection = selection

    self.info_dict = {}

    self._formatCheck()

  def _formatCheck(self):
    """
    Check if the values of the properties in this super-state-action dictionary 
    match the valid format.
    """
    assert(type(self.state_dict) is dict)

    assert(type(self.action_dict) is dict)

    assert(len(self.state_dict.keys() & self.action_dict.keys()) == 0)

    for k in self.state_dict:
      assert(k != 'selection')
      assert(type(self.state_dict[k]) is np.ndarray)

    for k in self.action_dict:
      assert(k != 'selection')
      assert(type(self.action_dict[k]) is np.ndarray)

  def copy(self):
    """
    Duplicate this super-state-action dictionary.

    @return A super-state-action dictionary duplication of this one.
    """
    self._formatCheck()

    ssa_copy = SSA()

    for k in self.state_dict:
      ssa_copy.state_dict[k] = self.state_dict[k].copy()
    
    for k in self.action_dict:
      ssa_copy.action_dict[k] = self.action_dict[k].copy()

    ssa_copy.selection = self.selection

    for k in self.info_dict:
      ssa_copy.info_dict[k] = self.info_dict[k]

    return ssa_copy

  def keys(self):
    """
    Obtain all the names of the elements from the super-state-action 
    dictionary. Note that this operation is effective across state dictionary, 
    action dictionary and selection.

    @return All the names of the elements in this super-state-action dictionary.
    """
    state_keys = self.state_dict.keys()
    action_keys = self.action_dict.keys()
    selection_keys = { 'selection' }

    joint_keys = state_keys | action_keys | selection_keys

    return joint_keys

  def retrive(self, name):
    """
    Retrive an element from the super-state-action dictionary. Note that this 
    operation is effective across state dictionary, action dictionary and 
    selection.

    @param name The name of the super-state-action dictionary element in state 
                dictionary, action dictionary or selection.
    @return The super-state-action dictionary element. An error will be thrown 
            if the corresponding element does not exist.
    """
    self._formatCheck()

    for k in self.state_dict:
      if k == name:
        return self.state_dict[k]

    for k in self.action_dict:
      if k == name:
        return self.action_dict[k]

    if 'selection' == name:
      return self.selection

    assert(False) # Element not found

  def updateStateElement(self, name, value):
    """
    Update a super-state-action dictionary element in state dictionary.

    @param name The name of the super-state-action dictionary element in the 
                state dictionary.
    @param value The target value of the element.
    """
    self.state_dict[name] = value

    self._formatCheck()

  def updateActionElement(self, name, value):
    """
    Update a super-state-action dictionary element in action dictionary.

    @param name The name of the super-state-action dictionary element in the 
                action dictionary.
    @param value The target value of the element.
    """
    self.action_dict[name] = value

    self._formatCheck()

  def updateSelection(self, selection):
    """
    Update the super-state-action dictionary selection element.

    @param selection The target selection value.
    """
    self.selection = selection

    self._formatCheck()

  def retriveInfo(self, name):
    """
    Retrive additional information of the super-state-action dictionary.

    @param name The name of the information.
    """
    assert(len({ name } & self.info_dict.keys()) == 1)

    return self.info_dict[name]

  def updateInfo(self, name, info):
    """
    Update or append an additional information to this super-state-action 
    dictionary.

    @param name The name of the information.
    @param info The content of the information.
    """
    self.info_dict[name] = info



class DynamicsSystemNode(object):
  """
  Node of a dynamics system, which consists of a name, the corresponding 
  super-state-action dictionary and the corresponding reward it gained.

  @property name The name of the node.
  @property prev_primitive_name The name of the previous bifurcation primitive 
                                connected to this node, where `None` indicates 
                                there is no previous bifurcation primitive 
                                connected to it.
  @property next_primitive_name The name of the following bifurcation primitive 
                                connected to this node, where `None` indicates 
                                there is no previous bifurcation primitive 
                                connected to it.
  @property ssa The super-state-action dictionary of the node.
  @property reward The reward gained at this node.
  @property value The state-action-selection value gained for this node. Note 
                  that the value for the terminal nodes should be `0` because 
                  there is no successive actions or selection.
  @property d_value The derivative of the state-action-selection value with 
                    respect to the super-state-action dictionary for this node. 
                    Note that the derivative of the value for the terminal 
                    nodes should be `0` because there is no successive actions 
                    or selection.
  """

  def __init__(self, name, prev_primitive_name=None, next_primitive_name=None, \
                     ssa=None, reward=0):
    """
    Initialize a dynamics system node. Note that the value and value derivative 
    for the node will be initialized to default value `0`.

    @param name The name of the node.
    @property prev_primitive_name The name of the previous bifurcation primitive 
                                  connected to this node.
    @property next_primitive_name The name of the following bifurcation primitive 
                                  connected to this node.
    @param ssa The initial super-state-action dictionary of the node.
    @param reward The reward gained at this node.
    """
    super(DynamicsSystemNode, self).__init__()

    self.name = name

    self.prev_primitive_name = prev_primitive_name

    self.next_primitive_name = next_primitive_name

    if ssa is None:
      self.ssa = SSA()
    else:
      assert(type(ssa) is SSA)
      self.ssa = ssa

    self.reward = reward

    self.value = 0

    self.d_value = {}
    for k in self.ssa.keys():
      if k == 'selection':
        self.d_value[k] = np.zeros(1)
      else:
        self.d_value[k] = np.zeros(self.ssa.retrive(k).shape[0])



class DynamicsSystemTransitionModel(object):
  """
  Transition model of a dynamics system.

  @property transition_func The transition probability distribution prediction 
                            function. It takes in a super-state-action 
                            dictionary, and returns a dictionary that maps the 
                            names of the next nodes to the corresponding 
                            probability of selecting this branch.
  @property transition_dfunc The derivative of the transition probability 
                             distribution function. It takes in a 
                             super-state-action dictionary, and returns a 
                             dictionary that maps a pair of node name and 
                             super-state-action dictionary element name to the 
                             corresponding derivatives, where the first name 
                             indicates the branch selection and the second name 
                             indicates the element with respect to which the 
                             derivative is. Derivatives here should be vectors 
                             (probability scalar to element value vector).
  """

  def __init__(self, transition_func=None, transition_dfunc=None):
    """
    Initialize a dynamics system transition model.

    @param transition_func The transition probability distribution prediction 
                           function. This parameter for initializing property 
                           `transition_func` is optional during initialization, 
                           but the corresponding property is required before 
                           casting predictions.
    @param transition_dfunc The derivative of the transition probability 
                            distribution prediction function. This parameter 
                            for initializing property `transition_dfunc` is 
                            optional during initialization, but the 
                            corresponding property is required before being 
                            assigned to a bifurcation primitive.
    """
    super(DynamicsSystemTransitionModel, self).__init__()
    
    if transition_func is not None:
      assert(callable(transition_func))
      self.transition_func = transition_func # function (ssa) return { name: prob, ... }

    if transition_dfunc is not None:
      assert(callable(transition_dfunc))
      self.transition_dfunc = transition_dfunc # function (ssa) return { node: { elem: d, ... }, ... }

  def predict(self, ssa):
    """
    Predict the transition probability distribution with respect to the given 
    super-state-action dictionary. Note that this function should NOT be 
    override in customized dynamics system transition model, which should 
    derive from this class. If customized prediction is required, one should 
    alter the property `transition_func` to provide customized prediction.

    @param ssa The input super-state-action dictionary.
    @return A dictionary that indicates the possible following branches, or 
            says the nodes, that the system may transit to with corresponding 
            node names and transition probability, which is in whole as a 
            transition probability distribution.
    """
    assert(callable(self.transition_func))

    prediction = self.transition_func(ssa)

    accumulated_prob = 0.0
    for node_name in prediction:
      assert(prediction[node_name] >= 0)
      accumulated_prob += prediction[node_name]
    assert(abs(accumulated_prob - 1.0) < 0.000001)

    return prediction

  def derivative(self, ssa):
    """
    Compute the transition probability function derivatives for each branch 
    with respect to each element in the super-state-action dictionary. Note 
    that this function should NOT be override in customized dynamics system 
    transition model, which should derive from this class. If customized 
    prediction is required, one should alter the property `transition_dfunc` to 
    provide customized derivatives computation.

    @param ssa The input super-state-action dictionary, with respect to which 
               the derivatives are computed.
    @return A dictionary that maps a pair of node name and super-state-action 
            dictionary element name to corresponding derivatives.
    """
    assert(callable(self.transition_dfunc))

    derivative = self.transition_dfunc(ssa)

    return derivative



class DynamicsSystemEdgeDynamicsModel(object):
  """
  Edge dynamics model including both dynamics and reward models of a dynamics 
  system.

  @property dynamics_func The dynamics function of the edge dynamics model. It 
                          takes in a super-state-action dictionary, and returns 
                          a following super-state-action dictionary.
  @property dynamics_dfunc The derivative of the dynamics function. It takes in 
                           a super-state-action dictionary, and returns a 
                           dictionary that maps pairs of super-state-action 
                           element names to corresponding derivatives, where 
                           the first name indicates an element of the following 
                           super-state-action dictionary and the second name 
                           indicates the super-state-action dictionary element 
                           to which the derivative is. Derivatives here should 
                           be matrices (element value vector to element value 
                           vector).
  @property reward_func The reward function of the edge dynamics model. It 
                        takes in a super-state-action dictionary, and returns a 
                        corresponding reward.
  @property reward_dfunc The derivative of the reward function. It takes in a 
                         super-state-action dictionary, and returns a 
                         dictionary that maps super-state-action element names 
                         to corresponding derivatives, where the name indicates 
                         the super-state-action dictionary element to which the 
                         derivative is. Derivatives here should be vectors 
                         (reward scalar to element value vector).
  """

  def __init__(self, dynamics_func=None, dynamics_dfunc=None, \
                     reward_func=None,   reward_dfunc=None):
    """
    Initialize a dynamics system edge dynamics model.

    @param dynamics_func The function that predicts the super-state-action 
                         dictionary dynamics from a previous node to the 
                         following node. Note that this parameter for 
                         initializing property `dynamics_func` is optional 
                         during initialization, but the corresponding property 
                         is required before the edge dynamics model is assigned 
                         to a bifurcation primitive.
    @param dynamics_dfunc The derivative of the dynamics prediction function. 
                          Note that this parameter for initializing property 
                          `dynamics_dfunc` is optional during initialization, 
                          but the corresponding property is required before the 
                          edge dynamics model is assigned to a bifurcation 
                          primitive.
    @param reward_func The function that predicts the reward gained with 
                       respect to a super-state-action dictionary. Note that 
                       this parameter for initializing property `reward_func` 
                       is optional during initialization, but the corresponding 
                       property is required before the edge dynamics model is 
                       assigned to a bifurcation primitive.
    @param reward_dfunc The derivative of the reward function. Note that this 
                        parameter for initializing property `reward_dfunc` is 
                        optional during initialization, but the corresponding 
                        property is required before the edge dynamics model is 
                        assigned to a bifurcation primitive.
    """
    super(DynamicsSystemEdgeDynamicsModel, self).__init__()

    if dynamics_func is not None:
      assert(callable(dynamics_func))
      self.dynamics_func = dynamics_func # function (ssa) return next_ssa

    if dynamics_dfunc is not None:
      assert(callable(dynamics_dfunc))
      self.dynamics_dfunc = dynamics_dfunc # function (ssa) return { next_elem: { elem: d, ... }, ... }

    if reward_func is not None:
      assert(callable(reward_func))
      self.reward_func = reward_func # function (next_ssa) return reward

    if reward_dfunc is not None:
      assert(callable(reward_dfunc))
      self.reward_dfunc = reward_dfunc # function (next_ssa) return { next_elem: d, ... }

  def predict(self, ssa):
    """
    Predict the dynamics from the previous super-state-action dictionary to the 
    following super-state-action dictionary and the corresponding reward of the 
    following super-state-action dictionary. Note that this function should NOT 
    be override in a customized edge dynamics model. Instead, one can alter the 
    properties `dynamics_func` and `reward_func` for customization purpose.

    @param ssa The input super-state-action dictionary.
    @return A tuple `(next_ssa, reward)` indicating the following 
            super-state-action dictionary yielded by the dynamics model and the 
            corresponding reward yielded by the reward model with respect to 
            the following super-state-action dictionary.
    """
    assert(callable(self.dynamics_func))
    assert(callable(self.reward_func))

    next_ssa = self.dynamics_func(ssa)
    for k in (ssa.state_dict.keys() - next_ssa.state_dict.keys()):
      next_ssa.state_dict[k] = ssa.state_dict[k].copy()
    for k in (ssa.action_dict.keys() - next_ssa.action_dict.keys()):
      next_ssa.action_dict[k] = ssa.action_dict[k].copy()

    reward = self.reward_func(next_ssa)

    return (next_ssa, reward)

  def derivative(self, ssa, next_ssa):
    """
    Compute the derivatives of the dynamics function and reward function with 
    respect to each element in the super-state-action dictionaries. Note that 
    this function should NOT be override in customized dynamics system edge 
    dynamics model, which should derive from this class. If customized 
    derivatives computation is required, one should alter the properties 
    `dynamics_dfunc` and `reward_dfunc` to provide customized derivatives 
    computation.

    @param ssa The input super-state-action dictionary, with respect to which 
               the dynamics function derivative is computed.
    @param next_ssa The input super-state-action dictionary, with respect to 
                    which the reward function derivative is computed.
    @return A tuple with the two dictionaries. The first dictionary maps pairs 
            of super-state-action dictionary element names to corresponding 
            dynamics function derivatives, where the first name indicates an 
            element of the following super-state-action dictionary and the 
            second name indicates the super-state-action dictionary element to 
            which the derivative is. The second dictionary maps a element names 
            of the following super-state-action dictionary to corresponding 
            reward function derivatives.
    """
    assert(callable(self.dynamics_dfunc))
    assert(callable(self.reward_dfunc))

    dynamics_func_derivative = {}
    for next_ssa_elem in next_ssa.keys():
      dynamics_func_derivative[next_ssa_elem] = {}
      for ssa_elem in ssa.keys():
        if next_ssa_elem == 'selection' and ssa_elem == 'selection':
          dynamics_func_derivative[next_ssa_elem][ssa_elem] = np.zeros((1, 1))
        elif next_ssa_elem == 'selection' and ssa_elem != 'selection':
          dynamics_func_derivative[next_ssa_elem][ssa_elem] = np.zeros((ssa.retrive(ssa_elem).shape[0], 1))
        elif next_ssa_elem != 'selection' and ssa_elem == 'selection':
          dynamics_func_derivative[next_ssa_elem][ssa_elem] = np.zeros((1, next_ssa.retrive(next_ssa_elem).shape[0]))
        elif next_ssa_elem == ssa_elem:
          assert(ssa.retrive(ssa_elem).shape[0] == next_ssa.retrive(next_ssa_elem).shape[0])
          dynamics_func_derivative[next_ssa_elem][ssa_elem] = np.eye((ssa.retrive(ssa_elem).shape[0]))
        else:
          dynamics_func_derivative[next_ssa_elem][ssa_elem] = np.zeros((ssa.retrive(ssa_elem).shape[0], next_ssa.retrive(next_ssa_elem).shape[0]))
    altered_dynamics_func_derivative = self.dynamics_dfunc(ssa)
    for next_ssa_elem in altered_dynamics_func_derivative:
      for ssa_elem in altered_dynamics_func_derivative[next_ssa_elem]:
        dynamics_func_derivative[next_ssa_elem][ssa_elem] = altered_dynamics_func_derivative[next_ssa_elem][ssa_elem]

    reward_func_derivative = {}
    for next_ssa_elem in next_ssa.keys():
      if next_ssa_elem == 'selection':
        reward_func_derivative[next_ssa_elem] = np.zeros((1))
      else:
        reward_func_derivative[next_ssa_elem] = np.zeros((next_ssa.retrive(next_ssa_elem).shape[0]))
    altered_reward_func_derivative = self.reward_dfunc(next_ssa)
    for next_ssa_elem in altered_reward_func_derivative:
      reward_func_derivative[next_ssa_elem] = altered_reward_func_derivative[next_ssa_elem]

    return (dynamics_func_derivative, reward_func_derivative)



class DynamicsSystemPrimitive(object):
  """
  Bifurcation primitive of a dynamics system, which consists of a transition 
  model a multiple edge dynamics models. It connects a previous node and a list 
  of following nodes.

  @property name The name of the bifurcation primitive.
  @property prev_node_name The name of the previous node connected to this 
                           primitive.
  @property transition The transition probability distribution model.
  @property dynamics_dict A dictionary that maps the names of the following 
                          nodes to corresponding edge dynamics models.
  @property gamma The discount factor.
  @property alpha The learning rate in gradient-based optimization.
  @property action_error The action error, which indicates how much the values 
                         of the action elements are changed during the last 
                         backward computation.
  """

  def __init__(self, name, prev_node_name=None, transition=None, dynamics_dict=None, \
                           gamma=1.0, alpha=0.000025):
    """
    Initialize a dynamics system bifurcation primitive.

    @param name The name of the bifurcation primitive.
    @param prev_node_name The name of the previous node connected to this 
                          primitive. Note that this parameter is optional 
                          during initialization, but the corresponding property 
                          is required before computation.
    @param transition The transition probability distribution model. Note that 
                      this parameter is optional during initialization, but the 
                      corresponding property is required before computation.
    @param dynamics_dict A dictionary that maps the names of the following 
                         nodes to corresponding edge dynamics models. Note that 
                         this parameter is optional during initialization, but 
                         the corresponding property is required before 
                         computation.
    @param gamma The discount factor. Note that this parameter is optional and 
                 is set to `1.0` in default. Its range is between `0.0` to 
                 `1.0`.
    @param alpha The learning rate in gradient-based optimization. Note that 
                 this parameter is optional and is set to `0.000025` in 
                 default. It should be non-negative.
    """
    super(DynamicsSystemPrimitive, self).__init__()

    self.name = name

    if prev_node_name is not None:
      self.prev_node_name = prev_node_name

    if transition is not None:
      assert(type(transition) is DynamicsSystemTransitionModel)
      self.transition = transition

    if dynamics_dict is None:
      self.dynamics_dict = {}
    else:
      assert(type(dynamics_dict) is dict)
      for next_node in dynamics_dict:
        assert(type(dynamics_dict[next_node]) is DynamicsSystemEdgeDynamicsModel)
      self.dynamics_dict = dynamics_dict

    assert(0.0 <= gamma and gamma <= 1.0)
    self.gamma = gamma

    assert(alpha >= 0.0)
    self.alpha = alpha

    self._initInternalData()

  def _initInternalData(self):
    """
    Initialize internal data.

    @property _T Transition probability distribution cache. 
                 `(scalar)prob = _T[next_node]`
    @property _dT Transition probability distribution derivative cache. 
                  `(vector)d = _dT[next_node][ssa_element]`
    @property _dF Edge dynamics dynamics model derivative cache.
                  `(matrix)d = _dF[next_node][next_ssa_element][ssa_element]`
    @property _dR_next Edge dynamics reward model derivative cache.
                       `(vector)d = _dR_next[next_node][next_ssa_element]`
    """
    self._T = None
    self._dT = None

    self._dF = {}
    self._dR_next = {}
    for b in self.dynamics_dict:
      self._dF[b] = None
      self._dR_next[b] = None

  def _checkInternalData(self):
    """
    Check if the internal data are valid and ready for forward and backward 
    computations.
    """
    assert(type(self._T) is dict)
    accumulated_prob = 0
    for k in self._T:
      accumulated_prob += self._T[k]
    assert(abs(accumulated_prob - 1.0) < 0.000001)

    assert(type(self._dT) is dict)
    for next_node in self._dT:
      assert(type(self._dT[next_node]) is dict)
      for ssa_element in self._dT[next_node]:
        assert(type(self._dT[next_node][ssa_element]) is np.ndarray)
        assert(self._dT[next_node][ssa_element].ndim == 1)

    assert(type(self._dF) is dict)
    for next_node in self._dF:
      assert(type(self._dF[next_node]) is dict)
      for next_ssa_element in self._dF[next_node]:
        assert(type(self._dF[next_node][next_ssa_element]) is dict)
        for ssa_element in self._dF[next_node][next_ssa_element]:
          assert(type(self._dF[next_node][next_ssa_element][ssa_element]) is np.ndarray)
          assert(self._dF[next_node][next_ssa_element][ssa_element].ndim == 2)

    assert(type(self._dR_next) is dict)
    for next_node in self._dR_next:
      assert(type(self._dR_next[next_node]) is dict)
      for ssa_element in self._dR_next[next_node]:
        assert(type(self._dR_next[next_node][ssa_element]) is np.ndarray)
        assert(self._dR_next[next_node][ssa_element].ndim == 1)

  def _checkInputNodes(self, prev_node, next_nodes):
    """
    Check if the input nodes, including the previous node and the following 
    nodes match the node names in this bifurcation primitive.

    @param prev_node The input previous node.
    @param next_nodes The input dictionary that maps the node names to 
                      corresponding following nodes.
    """
    assert(type(prev_node) is DynamicsSystemNode)
    assert(prev_node.name == self.prev_node_name)
    assert(prev_node.next_primitive_name == self.name)

    assert(len(self.dynamics_dict.keys() & next_nodes.keys()) == len(self.dynamics_dict))
    for k in self.dynamics_dict:
      assert(type(next_nodes[k]) is DynamicsSystemNode)
      assert(next_nodes[k].prev_primitive_name == self.name)

  def compute_forward(self, prev_node, next_nodes):
    """
    Execute forward computation for this bifurcation primitive and 
    corresponding nodes.

    @param prev_node The input previous node to this bifurcation.
    @param next_nodes The input dictionary that maps the node names to 
                      corresponding following nodes to this bifurcation 
                      primitive.
    @return The processed previous node and a dictionary that maps the node 
            names to the corresponding following nodes.
    """
    self._checkInputNodes(prev_node, next_nodes)

    T = self.transition.predict(prev_node.ssa)
    assert(len(self.dynamics_dict.keys() & T.keys()) == len(self.dynamics_dict))
    accumulated_prob = 0
    for b in T:
      accumulated_prob += T[b]
    assert(abs(accumulated_prob - 1.0) < 0.000001)
    self._T = T

    self._dT = self.transition.derivative(prev_node.ssa)

    for b in self.dynamics_dict:
      edge_dynamics = self.dynamics_dict[b]
      
      next_ssa, reward = edge_dynamics.predict(prev_node.ssa)
      for k in next_ssa.state_dict:
        next_nodes[b].ssa.state_dict[k] = next_ssa.state_dict[k]
      for k in next_ssa.action_dict:
        next_nodes[b].ssa.action_dict[k] = next_ssa.action_dict[k]
      for k in next_ssa.info_dict:
        next_nodes[b].ssa.info_dict[k] = next_ssa.info_dict[k]
      next_nodes[b].reward = reward

      dF_b, dR_next_b = edge_dynamics.derivative(prev_node.ssa, next_nodes[b].ssa)
      self._dF[b] = dF_b
      self._dR_next[b] = dR_next_b

      for k in prev_node.ssa.action_dict:
        assert((prev_node.ssa.action_dict[k] == next_nodes[b].ssa.action_dict[k]).all())

      for next_ssa_action_element in next_nodes[b].ssa.action_dict:
        for ssa_element in self._dF[b][next_ssa_action_element]:
          if next_ssa_action_element != ssa_element:
            assert(self._dF[b][next_ssa_action_element][ssa_element].min() == 0)
            assert(self._dF[b][next_ssa_action_element][ssa_element].max() == 0)

    self._checkInternalData()

    return (prev_node, next_nodes)

  def compute_backward(self, prev_node, next_nodes):
    """
    Execute backward computation for this bifurcation primitive and 
    corresponding nodes.

    @param prev_node The input previous node to this bifurcation.
    @param next_nodes The input dictionary that maps the node names to 
                      corresponding following nodes to this bifurcation 
                      primitive.
    @return The processed previous node and a dictionary that maps the node 
            names to the corresponding following nodes.
    """
    self._checkInputNodes(prev_node, next_nodes)
    self._checkInternalData()

    prev_node.value = 0
    for b in self._T:
      prev_node.value += self._T[b] * (next_nodes[b].reward + \
                                       self.gamma * next_nodes[b].value)

    prev_node.d_value = {}
    for k in prev_node.ssa.keys():
      if k == 'selection':
        prev_node.d_value[k] = None
      else:
        prev_node.d_value[k] = np.zeros(prev_node.ssa.retrive(k).shape[0])
    for k in prev_node.ssa.keys():
      if k != 'selection':
        for b in self._T:
          dR_next_delem = np.zeros(prev_node.ssa.retrive(k).shape[0])
          for next_ssa_element in self._dR_next[b]:
            dR_next_delem += self._dF[b][next_ssa_element][k].dot(self._dR_next[b][next_ssa_element])

          dJ_next_delem = np.zeros(prev_node.ssa.retrive(k).shape[0])
          for next_ssa_element in next_nodes[b].d_value:
            if next_ssa_element != 'selection':
              dJ_next_delem += self._dF[b][next_ssa_element][k].dot(next_nodes[b].d_value[next_ssa_element])

          prev_node.d_value[k] += self._dT[b][k] * (next_nodes[b].reward + self.gamma * next_nodes[b].value) + \
                                  self._T[b] * (dR_next_delem + self.gamma * dJ_next_delem)
    
    """
    accumulated_error = 0
    for k in prev_node.ssa.state_dict:
      for o in prev_node.d_value[k]:
        accumulated_error += abs(o)
    assert(accumulated_error < 0.000001)
    """

    self.action_error = 0
    for k in prev_node.ssa.action_dict:
      prev_node.ssa.action_dict[k] += self.alpha * prev_node.d_value[k]
      self.action_error += np.linalg.norm(prev_node.d_value[k])

    return (prev_node, next_nodes)



class TreeDynamicsSystem(object):
  """
  Tree-structured dynamics system.
  """

  def __init__(self):
    """
    Initialize a tree-structured dynamics system.
    """
    super(TreeDynamicsSystem, self).__init__()

    self._initInternalData()

  def _initInternalData(self):
    """
    Initialize internal data.

    @property _nodes A dictionary that maps node names to corresponding nodes, 
                     which includes all the nodes in this dynamics system.
    @property _root_node_name The name of the root node.
    @property _primitives A dictionary that maps bifurcation primitive names to 
                          corresponding bifurcation primitives, which includes 
                          all the bifurcation primitives in this dynamics 
                          system.
    """
    self._nodes = {}
    self._root_node_name = None
    self._primitives = {}

  def updateNode(self, name, node):
    """
    Append or update a dynamics system node.

    @param name The name of the node.
    @param node The node to append or update. Note that its name should match 
                the node name.
    """
    assert(type(node) is DynamicsSystemNode)
    assert(node.name == name)

    self._nodes[name] = node

  def updateRootNodeName(self, name):
    """
    Designate or update a dynamics system node as the root node, with which the 
    optimization process will begin.

    @param name The name of the node. Note that the node with the corresponding 
                name should have been appended to the dynamics system.
    """
    assert(len({ name } & self._nodes.keys()) == 1)

    self._root_node_name = name

  def updatePrimitive(self, name, primitive):
    """
    Append or update a dynamics system bifurcation primitive.

    @param name The name of the bifurcation primitive.
    @param primitive The primitive to append or update. Note that its name 
                     should match the bifurcation primitive name. All related 
                     nodes should have been appended to this dynamics system. 
                     And this bifurcation primitive should have been 
                     initialized.
    """
    assert(type(primitive) is DynamicsSystemPrimitive)
    assert(primitive.name == name)

    assert(len({ primitive.prev_node_name } & self._nodes.keys()) == 1)
    assert(len(primitive.dynamics_dict.keys() & self._nodes.keys()) == len(primitive.dynamics_dict))

    self._primitives[name] = primitive

  #TODO
  def _checkStructure(self):
    """
    Check the structure of the dynamics system.
    """
    #TODO

    assert(self._root_node_name is not None)

  def optimizeActionsOnce(self):
    """
    Optimize for once the actions for different nodes in the dynamics system.

    @return A tuple with the system value and the accumulated action error.
    """
    self._checkStructure()

    forwardWorkQueue = deque() # nodes
    backwardWorkStack = [] # primitives in stack
    accumulatedActionError = 0

    forwardWorkQueue.append(self._root_node_name)
    while len(forwardWorkQueue) > 0:
      currentNodeName = forwardWorkQueue.pop()
      currentNode = self._nodes[currentNodeName]
      currentPrimitiveName = currentNode.next_primitive_name

      if currentPrimitiveName is not None:
        currentPrimitive = self._primitives[currentPrimitiveName]
        backwardWorkStack.append(currentPrimitiveName)

        nextNodes = {}
        for k in currentPrimitive.dynamics_dict:
          forwardWorkQueue.append(k)
          nextNodes[k] = self._nodes[k]

        currentPrimitive.compute_forward(currentNode, nextNodes)

    while len(backwardWorkStack) > 0:
      currentPrimitiveName = backwardWorkStack.pop()
      currentPrimitive = self._primitives[currentPrimitiveName]

      previousNode = self._nodes[currentPrimitive.prev_node_name]
      nextNodes = {}
      for k in currentPrimitive.dynamics_dict:
        nextNodes[k] = self._nodes[k]

      currentPrimitive.compute_backward(previousNode, nextNodes)
      accumulatedActionError += currentPrimitive.action_error

    systemValue = self._nodes[self._root_node_name].value

    return (systemValue, accumulatedActionError)

  def optimizeActions(self, shouldStop, verbose=0):
    """
    Optimize the actions until stopping criteria fulfilled.

    @param shouldStop The stopping criteria function that determines if the 
                      optimization procedure should stop. It is in the form of 
                      `function (indicators)`, where `indicators` is a 
                      dictionary with different indicator to judge if this 
                      optimization procedure should stop, including 
                        - `rounds`: the number of optimization iteration rounds 
                          past,  
                        - `last_value`: the system value before the last 
                          optimization, 
                        - `value` indicates the system value after the last 
                          optimization, 
                        - `action_error` indicates the action error accumulated 
                          at the last optimization round.
    @param verbose A number of either `0` or `1`, indicating if the verbose 
                   procedural information should be printed.
    @return The final system value.
    """
    rounds = 0
    last_J = 0
    while True:
      J, a_err = self.optimizeActionsOnce()
      if verbose:
        print('Round: {}, Value: {}, Error: {}'.format(rounds, J, a_err))
      rounds += 1
      indicators = {
        'rounds': rounds,
        'last_value': last_J,
        'value': J,
        'action_error': a_err
      }
      if shouldStop(indicators):
        return J
      last_J = J



class LinearDynamicsSystem(object):
  """
  Linear dynamics system.
  """

  def __init__(self):
    """
    Initialize a linear dynamics system.
    """
    super(LinearDynamicsSystem, self).__init__()

    self._initInternalData()

  def _initInternalData(self):
    """
    Initialize internal data.

    @property _treeDynamicsSystem The internal tree-structured dynamics system.
    """
    self._treeDynamicsSystem = TreeDynamicsSystem()

  def updateNode(self, name, node):
    """
    Append or update a dynamics system node.

    @param name The name of the node.
    @param node The node to append or update. Note that its name should match 
                the node name.
    """
    self._treeDynamicsSystem.updateNode(name, node)

  def updateRootNodeName(self, name):
    """
    Designate or update a dynamics system node as the root node, with which the 
    optimization process will begin.

    @param name The name of the node. Note that the node with the corresponding 
                name should have been appended to the dynamics system.
    """
    self._treeDynamicsSystem.updateRootNodeName(name)

  def updateDynamics(self, name, prev_node_name, dynamics, next_node_name, 
                           gamma=1.0, alpha=0.000025):
    """
    Append or update a dynamics system dynamics model, which will be appended 
    to a bifurcation primitive generated automatically with transition model.

    @param name The name of the bifurcation primitive.
    @param prev_node_name The name of the previous node that connects to the 
                          automatically generated bifurcation primitive.
    @param dynamics The edge dynamics model to append or update.
    @param next_node_name The name of the following node that connects to the 
                          automatically generated bifurcation primitive.
    @param gamma The discount factor. Note that this parameter is optional and 
                 is set to `1.0` in default. Its range is between `0.0` to 
                 `1.0`.
    @param alpha The learning rate in gradient-based optimization. Note that 
                 this parameter is optional and is set to `0.000025` in 
                 default. It should be non-negative.
    """
    assert(type(dynamics) is DynamicsSystemEdgeDynamicsModel)

    # transition model
    transition = DynamicsSystemTransitionModel()

    def transition_func (ssa):
      prob = { next_node_name: 1.0 }
      return prob
    transition.transition_func = transition_func

    def transition_dfunc (ssa):
      d = {}
      d[next_node_name] = {}
      for k in ssa.keys():
        if k == 'selection':
          d[next_node_name][k] = np.zeros((1))
        else:
          d[next_node_name][k] = np.zeros((ssa.retrive(k).shape[0]))
      return d
    transition.transition_dfunc = transition_dfunc

    # bifurcation primitive
    primitive = DynamicsSystemPrimitive(name, gamma=gamma, alpha=alpha)
    primitive.prev_node_name = prev_node_name
    primitive.transition = transition
    primitive.dynamics_dict = {}
    primitive.dynamics_dict[next_node_name] = dynamics

    # append primitive
    self._treeDynamicsSystem.updatePrimitive(name, primitive)

  def optimizeActionsOnce(self):
    """
    Optimize for once the actions for different nodes in the dynamics system.

    @return A tuple with the system value and the accumulated action error.
    """
    systemValue, accumulatedActionError = self._treeDynamicsSystem.optimizeActionsOnce()

    return (systemValue, accumulatedActionError)

  def optimizeActions(self, shouldStop, verbose=0):
    """
    Optimize the actions until stopping criteria fulfilled.

    @param shouldStop The stopping criteria function that determines if the 
                      optimization procedure should stop. It is in the form of 
                      `function (rounds, improvement, action_error)`, where 
                      `rounds` indicates the number of optimization iteration 
                      rounds past, `improvement` indicates the difference 
                      between the system values after the last optimization 
                      round and before the last optimization round, and 
                      `action_error` indicates the action error accumulated at 
                      the last optimization round.
    @param verbose A number of either `0` or `1`, indicating if the verbose 
                   procedural information should be printed.
    @return The final system value.
    """
    return self._treeDynamicsSystem.optimizeActions(shouldStop, verbose=verbose)


#TODO
class GraphDynamicsSystem(object):
  """
  Graph-based dynamics system.
  """

  def __init__(self, arg):
    super(GraphDynamicsSystem, self).__init__()
    self.arg = arg
    


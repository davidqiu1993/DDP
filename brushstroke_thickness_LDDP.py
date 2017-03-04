import numpy as np
from DDP import *
import cv2

import pdb


if __name__ == '__main__':
  SAMPLE_IMAGE_PATH = 'brushstroke_sample_004.png'
  BRUSHSTROKES_NUM = 50
  BRUSHSTROKES_THICKNESS = 15
  LEARNING_RATE = 0.025

  image_desired = cv2.imread(SAMPLE_IMAGE_PATH)
  image_desired = cv2.cvtColor(image_desired, cv2.COLOR_BGR2GRAY)
  ret_info, image_desired = cv2.threshold(image_desired, 127, 255, cv2.THRESH_BINARY)
  #cv2.imshow('image_desired', image_desired)

  # nodes
  nodes = []
  for i in range(BRUSHSTROKES_NUM + 1):
    node = DynamicsSystemNode('picture_' + str(i))
    
    if (i < BRUSHSTROKES_NUM):
      node.next_primitive_name = 'draw_' + str(i)
      bound = image_desired.shape
      init_p1 = np.random.sample(2) * np.array([bound[0], bound[1]])
      init_p2 = init_p1 + np.array([5., 5.])
      init_brushstroke = np.array([init_p1[0], init_p1[1], init_p2[0], init_p2[1], BRUSHSTROKES_THICKNESS])
      node.ssa.updateActionElement('brushstroke_' + str(i), init_brushstroke)
      node.ssa.updateSelection(True)
    
    if (i > 0):
      node.prev_primitive_name = 'draw_' + str(i-1)

    nodes.append(node)

  nodes[0].ssa.updateInfo('image_draw', np.zeros(image_desired.shape, np.uint8))

  # dynamics models
  def simulate_brushstroke(image_draw, brushstroke):
    image_brushstroke = np.zeros(image_desired.shape, np.uint8)
    bounded_brushstroke = np.zeros(brushstroke.shape[0], np.int)
    bounded_brushstroke[0] = max(0, min(image_desired.shape[1], np.round(brushstroke[0])))
    bounded_brushstroke[1] = max(0, min(image_desired.shape[0], np.round(brushstroke[1])))
    bounded_brushstroke[2] = max(0, min(image_desired.shape[1], np.round(brushstroke[2])))
    bounded_brushstroke[3] = max(0, min(image_desired.shape[0], np.round(brushstroke[3])))
    bounded_brushstroke_thickness = max(0, np.round(brushstroke[4]))
    cv2.line(image_brushstroke, \
             (bounded_brushstroke[0], bounded_brushstroke[1]), \
             (bounded_brushstroke[2], bounded_brushstroke[3]), \
             255, \
             thickness=int(bounded_brushstroke_thickness))
    #cv2.imshow('image_brushstroke', image_brushstroke)
    image_draw_next = cv2.bitwise_or(image_draw, image_brushstroke)
    #cv2.imshow('image_draw_next', image_draw_next)
    image_increment = cv2.bitwise_xor(image_draw_next, image_draw)
    #cv2.imshow('image_increment', image_increment)
    image_matched = cv2.bitwise_and(image_desired, image_increment)
    image_mismatched = cv2.bitwise_and(cv2.bitwise_xor(image_desired, image_increment), image_increment)
    reward_positive = cv2.countNonZero(image_matched)
    reward_negative = cv2.countNonZero(image_mismatched)
    reward = reward_positive - reward_negative * 10
    #cv2.imshow('image_matched', image_matched)
    #cv2.imshow('image_mismatched', image_mismatched)
    return (image_draw_next, image_increment, reward)

  dynamics_models = []
  for i in range(BRUSHSTROKES_NUM):
    dynamics = DynamicsSystemEdgeDynamicsModel()

    def make_dynamics_func (i):
      def dynamics_func (ssa):
        next_ssa = ssa.copy()
        image_draw = ssa.retriveInfo('image_draw')
        brushstroke = ssa.retrive('brushstroke_' + str(i)) # p1_x, p1_y, p2_x, p2_y, thickness
        image_draw_next, image_increment, reward = simulate_brushstroke(image_draw, brushstroke)
        next_ssa.updateInfo('image_draw', image_draw_next)
        next_ssa.updateInfo('image_increment', image_increment)
        next_ssa.updateInfo('reward', reward)
        reward_d = [0, 0, 0, 0, 0]
        for brushstroke_i in range(brushstroke.shape[0]):
          di = np.zeros(brushstroke.shape[0])
          if brushstroke_i == 4:
            di[brushstroke_i] += 2 # pixel precision
          else:
            di[brushstroke_i] += 2 # pixel precision
          brushstroke_di = brushstroke + di
          image_draw_next, image_increment, reward_di = simulate_brushstroke(image_draw, brushstroke_di)
          reward_d[brushstroke_i] = (reward_di - reward) / di[brushstroke_i]
        next_ssa.updateInfo('reward_d', reward_d)
        return next_ssa
      return dynamics_func
    dynamics.dynamics_func = make_dynamics_func(i)

    def make_dynamics_dfunc (i):
      def dynamics_dfunc (ssa):
        next_ssa = make_dynamics_func(i)(ssa)
        d = {}
        return d
      return dynamics_dfunc
    dynamics.dynamics_dfunc = make_dynamics_dfunc(i)

    def make_reward_func (i):
      def reward_func (next_ssa):
        reward = next_ssa.retriveInfo('reward')
        return reward
      return reward_func
    dynamics.reward_func = make_reward_func(i)

    def make_reward_dfunc (i):
      def reward_dfunc (next_ssa):
        d = {}
        for k in next_ssa.keys():
          if k == 'selection':
            d[k] = np.zeros((1))
          else:
            d[k] = np.zeros((next_ssa.retrive(k).shape[0]))
        for brushstroke_i in range(next_ssa.retrive('brushstroke_' + str(i)).shape[0]):
          d['brushstroke_' + str(i)][brushstroke_i] = next_ssa.retriveInfo('reward_d')[brushstroke_i]
        return d
      return reward_dfunc
    dynamics.reward_dfunc = make_reward_dfunc(i)

    dynamics_models.append(dynamics)

  # dynamics system
  dynamics_system = LinearDynamicsSystem()
  for i in range(len(nodes)):
    dynamics_system.updateNode('picture_' + str(i), nodes[i])
  dynamics_system.updateRootNodeName('picture_0')
  for i in range(len(dynamics_models)):
    dynamics_system.updateDynamics('draw_' + str(i), \
                                   'picture_' + str(i), \
                                   dynamics_models[i], \
                                   'picture_' + str(i+1), \
                                   alpha=LEARNING_RATE)

  # execute
  for i in range(100):
    J, a_err = dynamics_system.optimizeActionsOnce()
    print('Round: {}, Value: {}, Error: {}'.format(i, J, a_err))
    #print(nodes[0].ssa.retrive('brushstroke_0'))
    last_node_name = 'picture_' + str(len(nodes) - 1)
    last_image_draw = nodes[len(nodes) - 1].ssa.retriveInfo('image_draw')
    cv2.imshow(last_node_name, cv2.bitwise_xor(last_image_draw, image_desired))
    cv2.waitKey(25)
    if a_err < 0.5:
      break

  image_draw_final = np.zeros(image_desired.shape, np.uint8)
  for i in range(len(nodes) - 1):
    brushstroke = nodes[i].ssa.retrive('brushstroke_' + str(i))
    print('brushstroke #{}: ({}, {}), ({}, {}), {}'.format(
      i,
      np.round(brushstroke[0], 2), np.round(brushstroke[1], 2),
      np.round(brushstroke[2], 2), np.round(brushstroke[3], 2),
      np.round(brushstroke[4], 2)
    ))
    image_draw_final, image_increment, reward = simulate_brushstroke(image_draw_final, brushstroke)
  cv2.imshow('image_draw_final_fit', cv2.bitwise_xor(image_draw_final, image_desired))
  cv2.imshow('image_draw_final', image_draw_final)

  cv2.waitKey()



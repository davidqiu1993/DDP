import numpy as np
from DDP import *
import cv2
import sys

import pdb


def test_reward_model():
  SAMPLE_IMAGE_PATH = 'brushstroke_sample_001.jpg'

  image_desired = cv2.imread(SAMPLE_IMAGE_PATH)
  image_desired = cv2.cvtColor(image_desired, cv2.COLOR_BGR2GRAY)
  ret_info, image_desired = cv2.threshold(image_desired, 127, 255, cv2.THRESH_BINARY)
  cv2.imshow('image_desired', image_desired)

  n_white = cv2.countNonZero(image_desired)
  n_black = image_desired.shape[0] * image_desired.shape[1] - n_white
  print('#white: {}, #black: {}'.format(n_white, n_black))

  cv2.waitKey()


def test_layers_extraction():
  SAMPLE_IMAGE_PATH = 'brushstroke_sample_006.jpg'
  DIVISION_LEVEL = 0

  image_desired = cv2.imread(SAMPLE_IMAGE_PATH)
  cv2.imshow('image_desired', image_desired)

  print('divides color space into segments...', end='\r', flush=True)
  portion = int(256 / (DIVISION_LEVEL + 1))
  assert(portion * (DIVISION_LEVEL + 1) == 256)
  segments = []
  for i in range(DIVISION_LEVEL + 1):
    segments.append(portion * i)
  segments.append(255)
  print('\033[Kcolor space segments ready (segments per channel: {})'.format(len(segments)))

  print('prepares descriptions of layers...', end='\r', flush=True)
  layers = []
  for r in segments:
    for g in segments:
      for b in segments:
        layer = {}
        layer['color'] = np.array([r, g, b])
        layer['luminance'] = np.array([0.2126, 0.7152, 0.0722]).dot(layer['color'])
        layer['image'] = np.zeros((image_desired.shape[0], image_desired.shape[1]), np.uint8)
        layers.append(layer)
  print('\033[Klayer descriptions ready (layers: {})'.format(len(layers)))

  print('generates layer images...', end='\r', flush=True)
  for row in range(len(image_desired)):
    for col in range(len(image_desired[row])):
      pixel = image_desired[row][col]
      minNorm = 256
      layerIndex = -1
      for l in range(len(layers)):
        norm = np.linalg.norm(pixel - layers[l]['color'])
        if norm < minNorm:
          layerIndex = l
          minNorm = norm
      layers[layerIndex]['image'][row][col] = 255
    print('\033[Kgenerates layer images... (completion: {}%)'.format(np.round(100. * float(row) / float(len(image_desired)), 1)), end='\r', flush=True)
  print('\033[Klayer images ready (images: {})'.format(len(layers)))

  print('recomposes image from layers...', end='\r', flush=True)
  image_recomposed = np.zeros(image_desired.shape)
  for l in range(len(layers)):
    layer = layers[l]
    for row in range(len(layer['image'])):
      for col in range(len(layer['image'][row])):
        if layer['image'][row][col] == 255:
          image_recomposed[row][col] = layer['color']
      print('\033[Krecomposes image from layers... (layer: {}/{}, completion: {}%)'.format(l, len(layers), np.round(100. * float(row) / float(len(layer['image']))), 1), end='\r', flush=True)
  print('\033[Kimage recomposed from layers (layers: {})'.format(len(layers)))
  cv2.imshow('image_recomposed', image_recomposed)


  cv2.waitKey()


if __name__ == '__main__':
  #test_reward_model()
  test_layers_extraction()


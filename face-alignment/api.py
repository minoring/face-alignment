import os
from enum import Enum
try:
  import urllib.request as request_file
except:
  import urllib as request_file

import torch
from torch.utils.model_zoo import load_url
from skimage import io
from skimage import color
import numpy as np
import cv2

# from .models import FAN, ResNetDepth
# from .utils import *


class LandmarksType(Enum):
  """Enum class defining the type of landmarks to detect.

  ``_2D`` - the detected points ``(x, y)`` are detected in a 2D space and follow the visible contour of the face
  ``_2halfD`` - this points represents the projection of the 3D points into 3D
  ``_3D`` - detect the points ``(x, y, z)``` in a 3D space
  """
  _2D = 1
  _2halfD = 2
  _3D = 3


class NetworkSize(Enum):
  # TINY = 1
  # SMALL = 2
  # MEDIUM = 3
  LARGE = 4

  def __new__(cls, value):
    member = object.__new__(cls)
    member._value_ = value
    return member

  def __int__(self):
    return self.value


models_urls = {
    '2DFAN-4':
        'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar',
    '3DFAN-4':
        'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar',
    'depth':
        'https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar',
}


class FaceAlignment:

  def __init__(self,
               landmarks_type,
               network_size=NetworkSize.LARGE,
               device='cuda',
               flip_input=False,
               face_detector='sfd',
               verbose=False):
    self.device = device
    self.flip_input = flip_input
    self.landmarks_type = landmarks_type
    self.verbose = verbose

    network_size = int(network_size)

    if 'cuda' in device:
      torch.backends.cudnn.benchmark = True
    
    # Get the face detector
    face_detector_module = __import__('face_alignment.detection.' + face_detector,
                                      globals(), locals(), [face_detector], 0)
    

import os
import sys
sys.path.append(os.path.abspath('..'))

import base64
from io import BytesIO

import bwm
import imutils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from h2o_wave import Q, app, handle_on, main, on, ui
from icecream import ic
from PIL import Image
from src.config import args
from src.datasetmgr import getDataLoader
from src.models import YedNet, ZhuNet
from web.utils import layout1, layout2, empty, img_preprocess
from web.index import *

trainloader, testloader = getDataLoader(args)

models = {'YedNet': YedNet, 'ZhuNet': ZhuNet}





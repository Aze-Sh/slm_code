#%% 
import WGS
import numpy as np
import scipy as sp
import torch
import torch.fft as fft
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import os
import time
import json
from Aberration import Zernike
from SLMGeneration import SLM_class
import json
import torch
from BlinkSLM import BlinkSLM
SLM = SLM_class()

SLM.image_init()

blinkslm=BlinkSLM()
# %%
slm_phase=SLM.zernike_generate()
# SLM_screen=SLM.phase_to_screen(slm_phase.cpu().clone().numpy())
plt.imshow(slm_phase)
# %%
blinkslm.write_image((slm_phase).astype('uint8'))
# %%
# blinkslm.write_image(np.zeros((1024,1024)).astype('uint8'))
blinkslm.slm_lib.Delete_SDK()
# %%

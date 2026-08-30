from BlinkSLM import BlinkSLM
import numpy as np
from ctypes import *
import torch

blinkslm=BlinkSLM()

# cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink Plus\\SDK\\Blink_C_wrapper.dll")
# slm_lib = CDLL("Blink_C_wrapper")
# blinkslm.slm_lib.Delete_SDK()

blinkslm.write_image(np.zeros((1024,1024)).astype('uint8'))
blinkslm.slm_lib.Delete_SDK()


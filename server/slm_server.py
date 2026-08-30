import numpy as np
import copy

import WGS
import torch
from SLMGeneration import SLM_class
from BlinkSLM import BlinkSLM

class SLM_server():
    def __init__(self,array_size=(15,27),distance=(20, -13*7.5)):
        self.array_size = array_size
        self.distance = distance
        self.SLM = SLM_class()
        self.blinkslm=BlinkSLM()

    def _init_SLM(self):
        self.SLM.image_init()
        self.incident_amp = self.SLM.initGaussianAmp
        self.target_amp=np.nonzero(self.SLM.target_generate("Rec",distance=self.dist,arraysize=self.array_size))

    def generate_tweezer():
        pass

    def optmize(self,intensity_array):
        self.fluore_amp = copy.deepcopy(intensity_array)
        self.fluore_amp = self.SLM.camera_Amp_generate(self.target_amp, self.fluore_amp) 
        self.target_amp = self.SLM.target_adapt(self.target_amp, self.fluore_amp)
        # SLM_Phase = WGS.WGS_phase_generate(torch.from_numpy(self.incident_amp), SLM_Phase, torch.from_numpy(targetAmp), Loop=20, threshold=0.01)
        # SLM_screen=SLM.phase_to_screen(SLM_Phase.cpu().clone().numpy())
        # slm_screen_f_corrected = SLM_screen 
        
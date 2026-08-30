#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.special import laguerre,genlaguerre
from BlinkSLM import *
from SLMGeneration import SLM_class



pixelpitch=17
SLMResX=1024
SLMResY=1024
beamwaist=1e3
w0=beamwaist/pixelpitch

X,Y=np.meshgrid(np.linspace(1,SLMResX,SLMResX),np.linspace(1,SLMResY,SLMResY))
X=X-SLMResX/2
Y=Y-SLMResY/2
r=np.sqrt(X**2+Y**2)
theta=np.mod(np.arctan2(Y,X),2*np.pi)

initGaussianAmp = np.sqrt(2/np.pi)/w0*np.exp(-(r**2)/w0**2)
plt.imshow(initGaussianAmp)
plt.colorbar()
plt.show()

def Phase_LG(p,l):
    return np.mod(-l*theta+np.pi*np.heaviside(-genlaguerre(p,np.abs(l))(2*r**2/w0**2),1),2*np.pi)

p=0
l=1
plt.imshow(Phase_LG(p,l))
plt.colorbar()
plt.show()


#%%
SLM_screen=np.round(Phase_LG(p,l)/(2*np.pi)*255).astype('uint8')
plt.imshow(SLM_screen)
plt.colorbar()
plt.show()

# blink_slm_correction  = np.load('Blink_SLM_correction_785_convert.npy')
# plt.imshow(blink_slm_correction)
# plt.colorbar()
# plt.show()

SLM = SLM_class()
fresnel_lens_screen, _ = SLM.fresnel_lens_phase_generate(125,1024/2,1024/2)
plt.imshow(fresnel_lens_screen)
plt.colorbar()
plt.show()


# %%
blink_slm = BlinkSLM()

#%%
blink_slm.slm_lib.Read_SLM_temperature.restype = c_double
blink_slm.slm_lib.Read_SLM_temperature(blink_slm.board_number)

#%%
blink_slm.write_image((SLM_screen))
#%%
blink_slm.write_image(blink_slm_correction)
#%%
blink_slm.write_image(np.zeros_like(blink_slm_correction).astype('uint8'))


 # %%
blink_slm.slm_lib.Delete_SDK()

# %%

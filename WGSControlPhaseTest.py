
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.optimize as opt
import random

from SLMGeneration import SLM_class
from WGS import *

#%%
SLM = SLM_class()

SLM.image_init()
targetAmp = torch.from_numpy(SLM.target_generate(Lattice_type='Rec',spacing=[5,5],arraysize=[45,45],translate=True,Plot=True))
# point_list = torch.argwhere(targetAmp)
# print(point_list)
# # 寻找非零元素
# indices = torch.where(targetAmp)
# print(indices)
# select_point = point_list[random.sample(range(45*45),1000)]
# select_indices = (select_point[:,0],select_point[:,1])
# plt.figure(figsize=(20,20))
# plt.scatter(select_indices[0],select_indices[1])
# plt.show()
# # 选择1000个随机点
# targetAmp = torch.zeros_like(targetAmp)
# targetPhase = torch.zeros_like(targetAmp)
# # 新设置目标
# targetAmp[select_indices] = torch.empty(1000,dtype=torch.double).uniform_(-torch.pi, torch.pi)
# targetPhase[select_indices] = torch.empty(1000,dtype=torch.double).uniform_(-torch.pi, torch.pi)
# targetField = torch.multiply(targetAmp, torch.exp(1j*targetPhase))
# %%
# RandomPhase = np.load('PhaseImageForSLM\\2023-10-26\\00-42-49_10x10_initGaussianPhase.npy')
input_shape = (4096, 4096)  # 输入振幅分布的尺寸
RandomPhase = torch.rand(input_shape, dtype=torch.float32) * 2 * np.pi - np.pi
# %%
slm_phase = WGS_phase_generate(torch.tensor(SLM.initGaussianAmp), torch.tensor(RandomPhase), torch.tensor(targetAmp), Loop=10, threshold=0.01)
# %%
fftAmp,fftPhase = SLM.phase_to_fftField(slm_phase.cpu())
plt.figure(figsize=(20,20))
plt.imshow(fftAmp[1000:3000,1000:3000])
plt.show()
# %%
# initPhase = (torch.fft.ifft2(torch.fft.ifftshift(targetField))).angle()
# fftAmp,fftPhase = SLM.phase_to_fftField(initPhase)
# plt.figure(figsize=(20,20))
# plt.imshow(fftAmp[1000:3000,1000:3000])
# plt.show()
# plt.plot(targetPhase[select_indices],fftPhase[select_indices],'.')
# # %%
# slm_phase = WGS_phase_generate(torch.tensor(SLM.initGaussianAmp), torch.tensor(RandomPhase), torch.tensor(targetAmp), Loop=20, threshold=0.01)

# fftAmp,fftPhase = SLM.phase_to_fftField(slm_phase.cpu())
# plt.figure(figsize=(20,20))
# plt.imshow(fftAmp[1000:3000,1000:3000])
# plt.show()
# plt.plot(targetPhase[select_indices],fftPhase[select_indices],'.')
# # # %%

# # # %%

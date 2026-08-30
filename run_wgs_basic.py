import numpy as np
import torch
import matplotlib.pyplot as plt

from SLMGeneration import SLM_class
from WGS import WGS_phase_generate

SLM = SLM_class()

SLM.image_init(Plot=True)

targetAmp = SLM.target_generate(
    Lattice_type="Rec",
    spacing=[150, 150],
    arraysize=[3, 3],
    translate=False,
    rotate=False,
    Plot=True,
)

slm_phase = WGS_phase_generate(
    torch.tensor(SLM.initGaussianAmp),
    torch.tensor(SLM.initGaussianPhase),
    torch.tensor(targetAmp),
    Loop=10,
    threshold=0.01,
    Plot=True,
)

fftAmp, fftPhase = SLM.phase_to_fftField(slm_phase.cpu().numpy())

plt.figure()
plt.imshow(fftAmp)
plt.title("Simulated focal plane intensity")
plt.colorbar()
plt.show()

screen = SLM.phase_to_screen(slm_phase.cpu().numpy())
np.save("wgs_screen.npy", screen)

plt.figure()
plt.imshow(screen, cmap="gray")
plt.title("SLM screen image")
plt.colorbar()
plt.show()
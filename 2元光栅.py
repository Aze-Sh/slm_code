import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift

# 参数设置
N = 512  # 空间分辨率
L = 10.0  # 物理尺寸 (单位可以自定)
wavelength = 0.6328  # 波长，例如He-Ne激光的波长为632.8 nm
focal_length = 50.0  # 镜头焦距
pixel_size = L / N  # 像素大小
x = y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# 高斯光束参数
waist = 2.0  # 高斯光束腰宽
k = 2 * np.pi / wavelength  # 波数
z = focal_length  # 光栅后传播距离
R = z + (1j * k * waist**2) / (2 * z)  # 复数半径
G = (1j * k) / (2 * R)  # 复数曲率
phase_factor = np.exp(1j * k * (X**2 + Y**2) / (2 * R))  # 相位因子
amplitude = (waist / np.sqrt(waist**2 + (1j * L**2) / (k * waist * z)))  # 振幅

# 高斯光束
gaussian_beam = amplitude * np.exp(-((X**2 + Y**2) / waist**2)) * phase_factor

# 二元相位光栅
grating_period = 0.5  # 光栅周期
phase_shift = 1*np.pi  # 相移
binary_phase_grating = np.where(np.sin(2 * np.pi * X / grating_period) > 0, 0, phase_shift)

# 通过光栅后的光场
field_after_grating = gaussian_beam * np.exp(1j * binary_phase_grating)

# 二维傅里叶变换
fft_result = fft2(field_after_grating)
fft_result_shifted = fftshift(fft_result)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.imshow(np.abs(field_after_grating), extent=[-L/2, L/2, -L/2, L/2], cmap='gray')
plt.title('Field after Grating')
plt.colorbar()


plt.subplot(1, 3, 2)
plt.imshow(binary_phase_grating, extent=[-L/2, L/2, -L/2, L/2], cmap='gray')
plt.title('phase Grating')
plt.colorbar() 

plt.subplot(1, 3, 3)
plt.imshow(np.abs(fft_result_shifted), extent=[-1/(2*pixel_size), 1/(2*pixel_size), -1/(2*pixel_size), 1/(2*pixel_size)], cmap='jet')
plt.title('2D Fourier Transform')
plt.colorbar()
plt.show()
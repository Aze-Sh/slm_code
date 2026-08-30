#%%
from SLMGeneration import SLM_class

from  IDS_Peak_Camera import IDS_Camera
from Gaussian_2D_Fit import *
from TweezerArrayRegionsExtraction import *
# from sipyco.pc_rpc import Client

import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.optimize as opt
from scipy.ndimage import binary_dilation, rotate
import os
import time
import h5py
import scipy as sp

from WGS import *
from BlinkSLM import *
# import Image3DServer

#%%
blink_slm = BlinkSLM()


#%%
SLM = SLM_class()

#%%
SLM.file_path_generate_according_to_date_and_time(Date_user_defined='2024-09-09',Time_user_defined='22-19-00')
#%%
SLM.file_path_generate_according_to_date_and_time()

#%%
initGaussianPhase = np.load('PhaseImageForSLM\\2023-10-26\\00-42-49_10x10_initGaussianPhase.npy')

#%%
SLM.image_init(initGaussianPhase_user_defined=initGaussianPhase, initGaussianPhase_save=True)

#%%
targetAmp = SLM.target_generate(Lattice_type='Rec',Plot=True)

#%%
# targetAmp = SLM.target_generate(Lattice_type='Tri',Plot=True)
#%%
kagomeAmp = np.copy(targetAmp)
spacingx = round(SLM.spacing[0]/SLM.Focalpitchx)
spacingy = round((SLM.spacing[1]/SLM.Focalpitchy)*np.sqrt(3)*0.5)
for i in range(kagomeAmp.shape[1]):
    for j in range(kagomeAmp.shape[0]):
        if kagomeAmp[j, i] > 0:
            # Remove points to form Kagome lattice
            if (i // spacingy) % 2 == 1 and (j // spacingx) % 2 == 1 and (i // spacingy) % 4 ==1:
                kagomeAmp[j, i] = 0
            elif (i // spacingy) % 2 == 1 and (j // spacingx) % 2 == 1 and (i // spacingy) % 4 ==3:
                kagomeAmp[j+spacingx, i] = 0
                kagomeAmp[j-spacingx, i] = 0
SLM.plot_target(kagomeAmp)
#%%
HoneycombAmp = np.copy(targetAmp)
spacingx = round(SLM.spacing[0]/SLM.Focalpitchx)
spacingy = round((SLM.spacing[1]/SLM.Focalpitchy)*np.sqrt(3)*0.5)
for i in range(HoneycombAmp.shape[1]):
    for j in range(HoneycombAmp.shape[0]):
        if HoneycombAmp[j, i] > 0:
            # Remove points to form Kagome lattice
            if  (j // spacingx) % 3 == 1 and (i // spacingy) % 2==1:
                HoneycombAmp[j, i] = 0
            elif  (j // spacingx) % 3 == 1 and (i // spacingy) % 2 ==0:
                HoneycombAmp[j+spacingx, i] = 0            
SLM.plot_target(HoneycombAmp)
#%%
def stagger_lattice(size):
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and (j - 1) % 2 == 0):
                matrix[i, j] = 1
    return matrix

stagger_matrix = stagger_lattice(45)
interested_indices = np.where(stagger_matrix.flatten())[0]

plt.imshow(stagger_matrix, cmap='Greys')
plt.title('Stagger Lattice')
plt.show()
#%%
# blink_slm_correction  = np.load('Datasets/Blink_SLM_correction.npy')
blink_slm_correction = np.load('Blink_SLM_correction_785_convert.npy')
fresnel_lens_screen, _ = SLM.fresnel_lens_phase_generate(125,1024/2,1024/2)


#%%
slm_phase = WGS_phase_generate(torch.from_numpy(SLM.initGaussianAmp), torch.from_numpy(SLM.initGaussianPhase), torch.from_numpy(targetAmp), Loop=20, threshold=0.01)
slm_screen = phase_to_screen_cuda(slm_phase).cpu().clone().numpy()
slm_screen_f_corrected = slm_screen + fresnel_lens_screen + blink_slm_correction
SLM.target_and_phase_save(targetAmp, slm_phase.cpu().clone().numpy(), slm_screen_f_corrected, info = '', adapt_times=0)


#%%
blink_slm.write_image(slm_screen_f_corrected)

slm_screen = phase_to_screen_cuda(slm_phase).cpu().clone().numpy()
slm_screen_f_corrected = slm_screen + fresnel_lens_screen + blink_slm_correction
#%%

#%%
#%%
blink_slm.write_image(fresnel_lens_screen + blink_slm_correction)




#%%
#%% adapt phase according to light shift data

lightshift_array = np.load('LightShiftData\\2024-09-10\\scattering_light_shift_adapt2@RID20561.npy')
# lightshift_array = np.load('LightShiftData\\2024-05-11\\manual_adjust5@RID13867.npy')
# lightshift_array[[1214,1484]]=0.01
#%%
lightshift_array = np.flip(lightshift_array)
lightshiftAmp = SLM.camera_Amp_generate(targetAmp, lightshift_array)
targetAmp = SLM.target_adapt(targetAmp, lightshiftAmp)

slm_phase = WGS_phase_generate(torch.from_numpy(SLM.initGaussianAmp), slm_phase, torch.from_numpy(targetAmp), Loop=20, threshold=0.01)
slm_screen = phase_to_screen_cuda(slm_phase).cpu().clone().numpy()

slm_screen_f_corrected = slm_screen + fresnel_lens_screen + blink_slm_correction

# SLM.target_and_phase_save(targetAmp, slm_phase.cpu().clone().numpy(), slm_screen_f_corrected, 
#                           info = '',adapt_times='_scattering_light_shift_adapt0')
SLM.target_and_phase_save(targetAmp, slm_phase.cpu().clone().numpy(), slm_screen_f_corrected, 
                          info = 'light_shift',adapt_times='4')

#%%
dataset_path = 'Datasets/60x60_to_45x45_60C/'
np.save(dataset_path + 'targetSLMScreen_tri.npy',slm_screen_f_corrected)
# np.save(dataset_path + 'targetSLMScreen_46x46.npy', slm_phase.cpu().numpy())
#%%
blink_slm.write_image(slm_screen_f_corrected)


#%%
blink_slm.slm_lib.Delete_SDK()


#%%
blink_slm.slm_lib.Read_SLM_temperature.restype = c_double
blink_slm.slm_lib.Read_SLM_temperature(blink_slm.board_number)
#%%




#%%
temp_path = 'PhaseImageForSLM\\2024-05-16\\20-22-48_60x60-4x4-dis0-center_adapt0'
# temp_path = 'PhaseImageForSLM\\2024-05-16\\20-22-48_60x60-4x4-dis0-_adapt-sum-18'
# temp_path = 'PhaseImageForSLM\\2024-05-16\\21-24-48_60x60-3.5x3.5-dis0-_adapt1_lightshift_following_adjust12_manual_adjust6'
# temp_path = 'PhaseImageForSLM\\2024-05-16\\21-24-48_60x60-3.5x3.5-dis0-_adapt1_lightshift_following_adjust12'
temp_path = 'PhaseImageForSLM\\2024-05-16\\21-24-48_60x60-4x4-dis20-_adapt5'
# temp_path = 'PhaseImageForSLM\\2024-05-23\\21-24-48_45x45-5.3x5.3-dis0-light_shift__adapt5'
temp_path = "PhaseImageForSLM\\2024-06-07\\16-56-25_46x46-5.25x5.25-dis0-light_shift__adapt5"
temp_path = "PhaseImageForSLM\\2024-07-03\\22-21-57_30x35-8.125x8-dis0-honey_adapt-sum-3"
temp_path = "PhaseImageForSLM\\2024-07-05\\13-29-39_30x35-8.125x8-dis0-ailab_adapt-sum-5"
temp_path = "PhaseImageForSLM\\2024-05-23\\21-24-48_45x45-5.3x5.3-dis0-light_shift__adapt5"
temp_path = "PhaseImageForSLM\\2024-07-17\\16-15-21_16x16x4_12.75um_10um[-15,-5,5,15]_adapt0"
temp_path ="C:\\Users\\ZhuangZhou\\Desktop\\SlmRearrangement\\PhaseImageForSLM\\2024-09-10\\16-12-23_32x32-7.5x7.5-dis0-light_shift_adapt2"
# temp_path ="PhaseImageForSLM\\2024-07-13\\15-39-07_25x25-8x8-dis0-25x25x4_8um_10um[-15,-5,5,15]_adapt_scattering_light_shift_adapt67"


targetAmp = np.load(temp_path+'/targetAmp.npy')
slm_phase = torch.from_numpy(np.load(temp_path+'/slm_phase.npy'))
slm_screen_f_corrected = np.load(temp_path+'/slm_screen_f_corrected.npy')
# targetLayer = np.load(temp_path+'/targetLayer.npy')

# slm_screen = phase_to_screen_cuda(slm_phase).cpu().clone().numpy()
# slm_screen_f_corrected = slm_screen + fresnel_lens_screen + blink_slm_correction


#%%
blink_slm.write_image(slm_screen_f_corrected)
#%%
def phase_to_fftField_3d(SLM_Phase,fresnel_lens):
        
    SLM_Field = np.multiply(SLM.initGaussianAmp, np.exp(1j*SLM_Phase))
    SLM_Field_shift = sp.fft.fftshift(SLM_Field*np.exp(1j*(fresnel_lens_phase_generate(-fresnel_lens)).cpu().numpy()))
    fftSLM = sp.fft.fft2(SLM_Field_shift)
    fftSLMShift = sp.fft.fftshift(fftSLM)
    fftSLM_norm = np.sqrt(np.sum(np.square(np.abs(fftSLMShift))))
    fftSLMShift_norm = fftSLMShift/fftSLM_norm

    fftAmp = np.abs(fftSLMShift_norm)
    fftPhase = np.angle(fftSLMShift_norm)
    return fftAmp, fftPhase

#%%
dataset_path = "Datasets/60x60_to_45x45_60C/"
layer=[0,10,20]
point_list_total=[[],[],[]]
phase_list_total=[[],[],[]]
for i in range(len(layer)):
    fftAmp, fftPhase = phase_to_fftField_3d(slm_phase.clone().numpy(),fresnel_lens=layer[i])
    point_list, phase_list = SLM.get_point_and_phase_list(targetAmp[i], fftPhase)
    point_list_total[i]=point_list
    phase_list_total[i]=phase_list
np.save(dataset_path + 'phase_list_middle_3x21x21.npy', phase_list_total)
np.save(dataset_path + 'point_list_middle_3x21x21.npy', point_list_total)
#%%
fftAmp, fftPhase = SLM.phase_to_fftField(slm_phase.clone().numpy())
point_list, phase_list = SLM.get_point_and_phase_list(targetAmp, fftPhase)
np.save(dataset_path + 'phase_list_target_32x32.npy', phase_list)
np.save(dataset_path + 'point_list_target_32x32.npy', point_list)

#%%
blink_slm.write_image(slm_screen_f_corrected)

#%%
blink_slm.write_image(fresnel_lens_screen+blink_slm_correction)










# 以下代码使用于相机反馈，但我们发现相机反馈与原子阱深差距太大，慎用

#%% camera connection, set and initialize

IDS_Camera = Client('192.168.31.15','11000','IDS_Camera')


#%%
IDS_Camera.SetROI()
IDS_Camera.SetBitDepth(12)

#%%
exposure_time = 60
IDS_Camera.SetExposureTime(exposure_time)

# %%
IDS_Camera.PrepareAcquisition()
IDS_Camera.AllocAndAnnounceBuffers()
IDS_Camera.StartAcquisition()

#%%
image = IDS_Camera.GetImage()

# %%
plt.figure(figsize=(20,20))
plt.imshow(image)
plt.colorbar()
print('max:',image.max())

#%%

# %%
IDS_Camera.StopAcquisition()
IDS_Camera.Close()

#%%
cutted_image = image
# rotated_image = rotate(cutted_image,angle=-4.8,reshape=False)
rotated_image = rotate(cutted_image,angle=-4,reshape=False)

plt.figure(figsize=(20,20))
plt.imshow(cutted_image)
plt.colorbar()
plt.show()

plt.figure(figsize=(20,20))
plt.imshow(rotated_image)
plt.colorbar()
plt.show()

#%%
plt.imshow(rotated_image[640:928,674:1534])
plt.colorbar()
print('exposure time:',exposure_time)
print('max:',rotated_image[640:928,674:1534].max())
print('sum:',rotated_image[640:928,674:1534].sum())

#%%
plt.imshow(rotated_image[1062-100:1120+70,1045-100:1115+50])
plt.colorbar()
print('exposure time:',exposure_time)
print('max:',rotated_image[1062-100:1120+70,1045-100:1115+50].max())
print('sum:',rotated_image[1062-100:1120+70,1045-100:1115+50].sum())


#%%
file_path = 'rotated_image.h5'
with h5py.File(file_path, "w") as h5_file:
    h5_file.create_dataset("rotated_image", data = rotated_image)



#%%
points = generate_points([246,409],[242,2045], [1873,409],[1873,2047],[45,45])
# points = generate_points_Tri([266,423],[259,2058], [1876,428],[1872,2063],[43,37])
# points = generate_points_kagome([261,400],[258,2011], [1879,400],[1879,2012],[40,34])
regions = generate_regions(points, [20,20])
aoi = generate_aoi(regions, rotated_image, 2000)

plt.figure(figsize=(20,20))
plt.imshow(rotated_image+aoi)
#%%
# triangle
regions_p1 = np.zeros((22,25,4),dtype=np.int32)
regions_p2 = np.zeros((21,25,4),dtype=np.int32)
for i in range(regions_p1.shape[0]):
    regions_p1[i] = regions.reshape(43,25,4)[i*2]
for i in range(regions_p2.shape[0]):
    regions_p2[i] = regions.reshape(43,25,4)[i*2+1]
    
regions_p1 = np.transpose(regions_p1,axes=(1,0,2))
regions_p2 = np.transpose(regions_p2,axes=(1,0,2))

regions_trans = np.zeros((43*25,4),dtype = np.int32)
for i in range(regions_p1.shape[0]):
    for j in range(regions_p1.shape[1]):
        if i%2==0:
            regions_trans[43*i+j] = regions_p1[24-i,j]
        if i%2==1:
            regions_trans[43*i+j+21] = regions_p1[24-i,j]
                    
for i in range(regions_p2.shape[0]):
    for j in range(regions_p2.shape[1]):
        if i%2==0:
            regions_trans[43*i+j+22] = regions_p2[24-i,j]
        if i%2==1:
            regions_trans[43*i+j] = regions_p2[24-i,j]        
aoi = generate_aoi(regions_trans[0:180], rotated_image, 2000)

plt.figure(figsize=(20,20))
plt.imshow(rotated_image+aoi)
#%%
# kagome
regions_p1 = np.zeros((20,34,4),dtype=np.int32)
regions_p2 = np.zeros((10,17,4),dtype=np.int32)
regions_p3 = np.zeros((10,17,4),dtype=np.int32)

for i in range(regions_p1.shape[0]):
    regions_p1[i] = regions[i*(17*3):i*(17*3)+34]
for i in range(regions_p2.shape[0]):
    regions_p2[i] = regions[i*(17*6)+17*2:i*(17*6)+17*3]
for i in range(regions_p3.shape[0]):
    regions_p3[i] = regions[i*(17*6)+17*5:i*(17*6)+17*6]
    
regions_p1 = np.transpose(regions_p1,axes=(1,0,2))
regions_p2 = np.transpose(regions_p2,axes=(1,0,2))
regions_p3 = np.transpose(regions_p3,axes=(1,0,2))

regions_trans = np.zeros((30*34,4),dtype = np.int32)
for i in range(regions_p1.shape[0]):
    for j in range(regions_p1.shape[1]):
        regions_trans[30*i+j] = regions_p1[33-i,j]
for i in range(regions_p2.shape[0]):
    for j in range(regions_p2.shape[1]):
        regions_trans[(30*2)*i+20+j] = regions_p2[16-i,j]
for i in range(regions_p3.shape[0]):
    for j in range(regions_p3.shape[1]):
        regions_trans[(30*2)*i+50+j] = regions_p3[16-i,j]

      
aoi = generate_aoi(regions_trans[0:180], rotated_image, 2000)

plt.figure(figsize=(20,20))
plt.imshow(rotated_image+aoi)
#%%
regions_p1 = np.zeros((20,51,4),dtype=np.int32)
for i in range(regions_p1.shape[0]):
    regions_p1[i] = regions.reshape(20,51,4)[i]   
regions_p1 = np.transpose(regions_p1,axes=(1,0,2))

regions_trans = np.zeros((20*51,4),dtype = np.int32)
for i in range(2):
    if i %2==0:
        for j in range(regions_p1.shape[1]):
                regions_trans[30*i+j] = regions_p1[33-i,j]
        for l in range(20):
            if l%2==0:
                k=int(l/2)
                regions_trans[30*i+k+20] = regions_p1[50-i,l]
            else:
                k=int((l-1)/2)
                regions_trans[30*i+k+50] = regions_p1[50-i,l]
    else:
        for j in range(regions_p1.shape[1]):
                regions_trans[30*i+j] = regions_p1[33-i,j]        
    # for l in range(20):
    #     if l%2==0:
    #         k=int(l/2)
    #         regions_trans[30*i+k+20] = regions_p1[50-i,l]
    #     else:
    #         k=int((l-1)/2)
    #         regions_trans[30*i+k+50] = regions_p1[50-i,l]
aoi = generate_aoi(regions_trans[0:180], rotated_image, 2000)

plt.figure(figsize=(20,20))
plt.imshow(rotated_image+aoi)
#%%
#ustc
regions_trans = np.flip(np.transpose(regions.reshape(45,45,4),axes=(1,0,2)),axis=0).reshape(-1,4)
regions_trans = regions_trans[c]#point select order
aoi = generate_aoi(regions_trans, rotated_image, 2000)

plt.figure(figsize=(20,20))
plt.imshow(rotated_image+aoi)
#%%
regions = regions_trans
#%%
x1 = regions[:,0]
x2 = regions[:,1]
y1 = regions[:,2]
y2 = regions[:,3]


#%%
def non_uniformity_show(non_uniformity):
    non_uniformity = np.array(non_uniformity)
    min_index = np.argmin(non_uniformity)
    min_value = np.min(non_uniformity)
    plt.plot(non_uniformity)
    plt.plot(min_index,min_value,'.')
    plt.annotate(f'Minimum: {min_value:.3f}', xy=(min_index, min_value),
                xytext=(min_index + 1, min_value),
                arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.title('Tweezer Uniformity Optimization')
    plt.xlabel('Number of iterations')
    plt.ylabel('Non uniformity')
    plt.show()

# 适用于4f系统优化
def camera_intensity_array_generate(intensity):
    intensity = np.array(intensity)
    intensity_2d_array = intensity.reshape(25,43)
    intensity_flip = np.transpose(intensity_2d_array)
    intensity_flip = np.flip(intensity_flip, axis=1)
    camera_intensity_array = intensity_flip.flatten()

    return camera_intensity_array
# 适用于物镜后优化
# def camera_intensity_array_generate(intensity):
#     intensity = np.array(intensity)
#     intensity_flip = intensity.reshape(64,64)
     

#     camera_intensity_array = intensity_flip.flatten()

#     return camera_intensity_array


array_size = [1013,1]
index = 0
non_uniformity = []
intensity_data = []
targetAmp_data = []
optimize = True
time_interval = 1
rep = 5

while True:

    if optimize:
        blink_slm.write_image(slm_screen_f_corrected)

    time.sleep(time_interval)

    image = IDS_Camera.GetImage()
    cutted_image = image
    rotated_image = rotate(cutted_image,angle=-0.0,reshape=False)

    image_ROI = []
    intensity = []

    for i  in range(array_size[0]*array_size[1]):
        image_ROI.append(rotated_image[x1[i]:x2[i],y1[i]:y2[i]])

    for i  in range(array_size[0]*array_size[1]):
        # popt, pcov, fitted_image = gaussian_2d_fit(image_ROI[i], show=False)
        # intensity.append(popt[0])
        intensity.append(image_ROI[i].sum())


    intensity = np.array(intensity)
    # non_uniformity.append(intensity.std()/intensity.mean())
    # camera_intensity = camera_intensity_array_generate(intensity)
    camera_intensity = np.flip(intensity)
    intensity_data.append(camera_intensity)
    non_uniformity.append(camera_intensity.std()/camera_intensity.mean())
    print(camera_intensity.std()/camera_intensity.mean())

    if camera_intensity.std()/camera_intensity.mean()< 0.001 or index == rep:
        break

    index += 1
    if index % 1 ==0:
        print('\n')
        print(index)


    if optimize:
        
        # camera_intensity_array = camera_intensity_array_generate(intensity)
        camera_intensity_array = np.flip(intensity)
        cameraAmp = SLM.camera_Amp_generate(targetAmp, camera_intensity_array)
        targetAmp = SLM.target_adapt(targetAmp, cameraAmp)
        targetAmp_data.append(targetAmp)

        print('Start the new SLM phase generation ...')

        slm_phase = WGS_phase_generate(torch.from_numpy(SLM.initGaussianAmp), slm_phase, torch.from_numpy(targetAmp), Loop=20, threshold=0.01)
        slm_screen = phase_to_screen_cuda(slm_phase).cpu().clone().numpy()
        slm_screen_f_corrected = slm_screen + fresnel_lens_screen + blink_slm_correction

        SLM.target_and_phase_save(targetAmp, slm_phase.cpu().clone().numpy(), slm_screen_f_corrected, info = "stagger",
                                adapt_times='-sum-'+str(index))


non_uniformity_show(non_uniformity)

plt.figure(figsize=(20,20))
plt.imshow(rotated_image)
plt.colorbar()
plt.show()






































#  3D
# %%
file_path = 'rotated_image_list.h5'
with h5py.File(file_path, "w") as h5_file:
    h5_file.create_dataset("rotated_image", data = rotated_image_list)
# %%
points1 = generate_points([423,598], [1727,606],[413,1909],[1719,1917],[16,16])
# points1 = generate_points([484,693],[480,1696], [1480,695],[1480,1697],[25,25])
# points1 = generate_points([830,1365],[1447,1361], [833,1985],[1450,1980],[25,25])
regions1 = generate_regions(points1, [16,16])
aoi1 = generate_aoi(regions1, rotated_image_list[0], 2000)

plt.figure(figsize=(20,20))
plt.imshow(rotated_image_list[2]+aoi1)


points2 = generate_points([380,642], [1684,650],[370,1953],[1675,1962],[16,16])
regions2 = generate_regions(points2, [16,16])
aoi2 = generate_aoi(regions2, rotated_image_list[0], 2000)

plt.figure(figsize=(20,20))
plt.imshow(rotated_image_list[1]+aoi2)

x1of1 = regions1[:,0]
x2of1 = regions1[:,1]
y1of1 = regions1[:,2]
y2of1 = regions1[:,3]

x1of2 = regions2[:,0]
x2of2 = regions2[:,1]
y1of2 = regions2[:,2]
y2of2 = regions2[:,3]

# %%
def non_uniformity_show(non_uniformity):
    non_uniformity = np.array(non_uniformity)
    min_index = np.argmin(non_uniformity)
    min_value = np.min(non_uniformity)
    plt.plot(non_uniformity)
    plt.plot(min_index,min_value,'.')
    plt.annotate(f'Minimum: {min_value:.3f}', xy=(min_index, min_value),
                xytext=(min_index + 1, min_value),
                arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.title('Tweezer Uniformity Optimization')
    plt.xlabel('Number of iterations')
    plt.ylabel('Non uniformity')
    plt.show()


def camera_intensity_array_generate(intensity):
    intensity = np.array(intensity)
    intensity_2d_array = intensity.reshape(16,16)
    intensity_flip = intensity_2d_array
    # intensity_flip = np.transpose(intensity_2d_array)
    intensity_flip = np.flip(intensity_flip, axis=1)
    # intensity_flip = np.transpose(intensity_flip)
    camera_intensity_array = intensity_flip.flatten()

    return camera_intensity_array

array_size = [16,16]
non_uniformity = []
index = 0

# %%
targetLayer = np.array([-15,-5,5,15])
# image_3d = Image3DServer.Image3D(slm_phase, [0,-10,-20,-30,-40], 40)
# image_3d.slm_init()
rotated_image_list = []
for i in range(len(targetLayer)):
    slm_phase_fresnel = slm_phase.cuda() + fresnel_lens_phase_generate(-targetLayer[i])
    slm_screen = phase_to_screen_cuda(slm_phase_fresnel).cpu().clone().numpy()
    slm_screen_f_corrected = slm_screen + fresnel_lens_screen + blink_slm_correction
    blink_slm.write_image(slm_screen_f_corrected)
    image = IDS_Camera.GetImage()
    rotated_image = rotate(image,angle=-4,reshape=False)
    rotated_image_list.append(rotated_image)
# image_3d.slm_close()

intensity = [[] for _ in range(len(rotated_image_list))]
camera_intensity = []
image_ROI = [[] for _ in range(len(rotated_image_list))]
cameraAmp = []

for j in range(len(rotated_image_list)):
    for i  in range(array_size[0]*array_size[1]):
        if j % 2 == 0:
            image_ROI[j].append(rotated_image_list[j][x1of1[i]:x2of1[i],y1of1[i]:y2of1[i]])
        else:
            image_ROI[j].append(rotated_image_list[j][x1of2[i]:x2of2[i],y1of2[i]:y2of2[i]])
        intensity[j].append(image_ROI[j][i].sum())
    intensity[j] = np.array(intensity[j])
    camera_intensity.append(camera_intensity_array_generate(intensity[j]))
    cameraAmp.append(SLM.camera_Amp_generate(targetAmp[j], camera_intensity[j]))

cameraAmp = np.array(cameraAmp)
targetAmp = SLM.target_adapt(targetAmp, cameraAmp)

camera_intensity = np.array(camera_intensity)
non_uniformity.append(camera_intensity.std()/camera_intensity.mean())
print(camera_intensity.std()/camera_intensity.mean())

print('Start the new SLM phase generation ...')

slm_phase = WGS3D_phase_generate(torch.from_numpy(SLM.initGaussianAmp), slm_phase, torch.from_numpy(targetAmp), targetLayer, Loop=20, threshold=0.01)
slm_screen = phase_to_screen_cuda(slm_phase).cpu().clone().numpy()
slm_screen_f_corrected = slm_screen + fresnel_lens_screen + blink_slm_correction

index += 1
SLM.target_and_phase_save(targetAmp, slm_phase.cpu().clone().numpy(), slm_screen_f_corrected, '16x16x4_12.75um_10um[-15,-5,5,15]', adapt_times='-sum-'+str(index))
blink_slm.write_image(slm_screen_f_corrected)
# %%
non_uniformity_show(non_uniformity)
#%%

temp_path ="PhaseImageForSLM\\2024-07-25\\23-19-16_28x28-7x7-dis0-28x28x3_7um_10um[0,10,20]_adapt_scattering_light_shift_adapt17"


targetAmp = np.load(temp_path+'/targetAmp.npy')
slm_phase = torch.from_numpy(np.load(temp_path+'/slm_phase.npy'))
slm_screen_f_corrected = np.load(temp_path+'/slm_screen_f_corrected.npy')
# targetLayer = np.load(temp_path+'/targetLayer.npy')
#%%
blink_slm.write_image(slm_screen_f_corrected)
# %%
slm_phase = torch.from_numpy(np.load('PhaseImageForSLM\\2024-07-25\\23-19-16_21x21-9.5x9.5-dis0-21x21x3_9.5um_10um[0,10,20]_adapt_scattering_light_shift_adapt3\\slm_phase.npy'))
targetAmp = np.load('PhaseImageForSLM\\2024-07-25\\23-19-16_21x21-9.5x9.5-dis0-21x21x3_9.5um_10um[0,10,20]_adapt_scattering_light_shift_adapt3\\targetAmp.npy')

# %%

# adapt phase according to light shift data for 3D

lightshift_array = np.load('LightShiftData\\2024-07-16\\scattering_light_shift_adapt4@RID18909.npy')

# %%
lightshift_array = np.load('LightShiftData\\2024-07-28\\scattering_light_shift_adapt3@RID19232.npy')
#%%
lightshiftAmp = []
for i in range(len(targetAmp)):
    lightshiftAmp.append(SLM.camera_Amp_generate(targetAmp[i],np.flip(lightshift_array[i])))
lightshiftAmp = np.array(lightshiftAmp)
targetAmp = SLM.target_adapt(targetAmp, lightshiftAmp)
# %%
slm_phase = WGS3D_phase_generate(torch.from_numpy(SLM.initGaussianAmp), slm_phase, torch.from_numpy(targetAmp), torch.tensor([0,10,20]), Loop=20, threshold=0.01)
slm_screen = phase_to_screen_cuda(slm_phase).cpu().clone().numpy()
slm_screen_f_corrected = slm_screen + fresnel_lens_screen + blink_slm_correction

SLM.arraysize = [21,21]
SLM.spacing = [9.5,9.5]
SLM.target_and_phase_save(targetAmp, slm_phase.cpu().clone().numpy(), slm_screen_f_corrected, '21x21x3_9.5um_10um[0,10,20]',
                          adapt_times='_scattering_light_shift_adapt7')
#%%
blink_slm.write_image(slm_screen_f_corrected)





# %%
targetLayer = np.array([0, 20, 40])
path = 'PhaseImageForSLM/2024-04-12/21-40-59_8x8-13x13-dis0-20um AB stacked square_adapt0'
np.save(path + '/targetLayer', targetLayer)

# %%
IDS_Camera.StopAcquisition()
IDS_Camera.Close()
#%%
def arb_pattern(center,v1,v2,bound):
    coor = []
    for i in range(np.round(bound[0]*2).astype(np.int32),np.round(bound[1]*2).astype(np.int32)):
        for j in range(np.round(bound[2]/np.sqrt(3)*4).astype(np.int32),np.round(bound[3]/np.sqrt(3)*4).astype(np.int32)):
            if abs(j-i)%3==0:
                continue
            co = v1*i+v2*j+center
            if bound[0]<co[0]<bound[1] and bound[2]<co[1]<bound[3]:
                coor.append(co)
    return np.array(coor)
# %%
def rotate_matrix(coor,rad):
    r_matrix = np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
    for i in range(coor.shape[0]):
        coor[i] = np.dot(r_matrix,coor[i])
        
    return coor
# %%
v = np.array([[1,0],[1/2,np.sqrt(3)/2]])
v1,v2 = rotate_matrix(v,np.pi*0)
center = np.array([0,1])
x = arb_pattern(center,v1,v2,[-9,9,-8,8])
x = rotate_matrix(x,np.pi/4)
test_size = 3000
test = np.zeros((test_size,test_size))
for i in np.round(60*x).astype(np.int32):
    print(i)
    test[i[0]+int(test_size/2)-10:i[0]+int(test_size/2)+10,i[1]+int(test_size/2)-10:i[1]+int(test_size/2)+10] = 1
plt.figure(figsize=(20,20))
plt.imshow(test)
plt.show()
# %%
plt.figure(figsize=(8, 8))
plt.scatter(np.round(60*x).astype(np.int32)[:,0], np.round(60*x).astype(np.int32)[:,1], s=1, color='blue')
plt.title("Hexagonal Lattice of Graphene in 4096x4096 Matrix (First and Last Rows Removed)")
# plt.xlim(0, 4096)
# plt.ylim(0, 4096)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
# %%
targetAmp = np.load("C:\\Users\\ZhuangZhou\\Desktop\\SlmRearrangement\\PhaseImageForSLM\\2024-07-25\\23-19-16_28x28-7x7-dis0-28x28x3_7um_10um[0,10,20]_adapt0\\targetAmp.npy")
y = np.argwhere(targetAmp[0]>0)

# %%
plt.figure(figsize=(8, 8))
plt.scatter(np.round(60*x).astype(np.int32)[:,0], np.round(60*x).astype(np.int32)[:,1], s=1, color='blue')
plt.scatter(np.round(y-y.mean()).astype(np.int32)[:,0], np.round(y-y.mean()).astype(np.int32)[:,1], s=1, color='orange')
# %%

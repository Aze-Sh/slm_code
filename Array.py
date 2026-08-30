import matplotlib.pyplot as plt
import copy
import os
import sys
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from scipy import special
import scipy
import math
import numpy as np
from scipy.optimize import curve_fit
import cv2

np.set_printoptions(threshold=np.inf)

# plt.rcParams['font.sans-serif'] = ['STSong']
# plt.rcParams['axes.unicode_minus'] = False

def get_max_gray_value(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换为灰度图像
    max_value = gray_img.max() # 获取灰度图像的最大值
    return max_value

def func_gaussian(x, A, shift, sigma, offset):
    """Fit function with four parameters"""
    return A*np.exp(-0.5*(x-shift)**2/sigma**2) + offset

def func_gaussian2(x, A, sigma, offset):
    """Fit function with four parameters"""
    return A*np.exp(-0.5*(x)**2/sigma**2) + offset

from concurrent.futures import ThreadPoolExecutor

def point_exist(image, threshold):
    """Check whether a light point exists in the image by comparing each point's value with the threshold."""
    return np.any(image > threshold)

def find_max(image):
    """Find the maximum value in the image and its coordinates."""
    max_index = np.argmax(image)
    max_value = image.flat[max_index]
    max_coords = np.unravel_index(max_index, image.shape)
    return max_value, max_coords[0], max_coords[1]

def roi_region(image, y_pos, x_pos, width_y, width_x):
    """Calculate the region of interest in the image."""
    y_start = max(0, y_pos - width_y)
    y_end = min(image.shape[0], y_pos)
    x_start = max(0, x_pos - width_x)
    x_end = min(image.shape[1], x_pos)
    return y_start, y_end, x_start, x_end

def process_subimage(image, threshold, y_start, y_end, x_start, x_end):
    """Process a subimage to find light points."""
    sub_image_area = image[y_start:y_end, x_start:x_end]
    if point_exist(sub_image_area, threshold):
        point_peak, site_y, site_x = find_max(sub_image_area)
        site_y += y_start
        site_x += x_start
        return [point_peak, site_y, site_x]
    return None

def search_light_point(image, threshold=100, width_y=30, width_x=30):
    """Find the light point in the input image.
       Search subimage in the input image one by one.
       Return the y and x indices of the light point."""
    num_y = int(np.ceil(image.shape[0] / width_y))
    num_x = int(np.ceil(image.shape[1] / width_x))

    point_site = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for iy in range(num_y):
            for jx in range(num_x):
                y_start = iy * width_y
                x_start = jx * width_x
                y_end = min((iy + 1) * width_y, image.shape[0])
                x_end = min((jx + 1) * width_x, image.shape[1])
                futures.append(executor.submit(process_subimage, image, threshold, y_start, y_end, x_start, x_end))
        
        for future in futures:
            result = future.result()
            if result:
                point_site.append(result)
    
    print(len(point_site))
    return point_site

def check_position_diff(point_site_list, threshold=100):
    """Calculate the distance of every two points in the point_site_list.
       If the distance is too close, the new point is considered not a corresponding hole in the target."""
    new_point_site_list = []
    new_point_site_list.append(point_site_list[0])
    for i in range(1, len(point_site_list)):
        new_point_site_list.append(point_site_list[i])
        for j in range(len(new_point_site_list)-1):
            distance = np.sqrt( (point_site_list[i][1]-new_point_site_list[j][1])**2 + (point_site_list[i][2]-new_point_site_list[j][2])**2 )
            if distance < threshold:
                new_point_site_list.pop()
    return new_point_site_list


def fit_single_point_obj(image, point_site, magnification, pixel_size=1.85, y_plot_range=50, x_plot_range=50):
    """Fit every point in the subimage at x_axis and y_axis in objective plane."""
    img = copy.deepcopy(image)
    y_up    = np.int32(point_site[1] - y_plot_range/2)
    y_down  = np.int32(point_site[1] + y_plot_range/2)
    x_left  = np.int32(point_site[2] - x_plot_range/2)
    x_right = np.int32(point_site[2] + x_plot_range/2)
    roi = img[y_up:y_down, x_left:x_right]
    # plt.imshow(roi)
    
    # x_fit
    # x_array = roi[point_site[2],:].flatten('F')
    x_array = roi[np.int32(y_plot_range/2),:].flatten('F')
    xx = np.arange(len(x_array))
    x_guess = [180,np.int32(x_plot_range/2),9,10]
    x_para, _ = curve_fit(func_gaussian, xx, x_array, p0=x_guess, maxfev = 1000000)
    x_fit_value = func_gaussian(xx, x_para[0], x_para[1], x_para[2], x_para[3])
    
    
    # y_fit
    # y_array = roi[point_site[1],:].flatten('F')
    y_array = roi[:,np.int32(x_plot_range/2)].flatten('F')
    yy = np.arange(len(y_array))
    y_guess = [180,np.int32(y_plot_range/2),9,10]
    y_para, _ = curve_fit(func_gaussian, yy, y_array, p0=y_guess, maxfev = 1000000)
    y_fit_value = func_gaussian(yy, y_para[0], y_para[1], y_para[2], y_para[3])
    
    print('light point:'+ str(point_site) + 'fit done!')
    print('The light point diameter in the x-axis is ' + str(2*2.928515*x_para[2]*pixel_size/magnification) + 'um.')
    print('The light point diameter in the y-axis is ' + str(2*2.928515*y_para[2]*pixel_size/magnification) + 'um.')
    
    fig, axes = plt.subplots(1, 3, figsize=(9,4), constrained_layout=True)
    
    axes[0].imshow(roi)
    axes[0].set_title('point position: '+str(point_site[1])+' '+str(point_site[2]))
    axes[1].plot(xx, x_fit_value)
    axes[1].scatter(xx, x_array)
    axes[1].set_title('fit_x_diameter: '+ str(2*2.928515*x_para[2]*pixel_size/magnification)[:4]+ 'um.')
    axes[1].set_xlabel('pixel number')
    axes[1].set_ylabel('Intensity')
#     plt.text(0.75, 0.9, s=r'x_diameter: '+ str(2*2.928515*x_para[2]*pixel_size)[:4] + 'um.', transform=ax.transAxes)
    axes[2].plot(yy, y_fit_value)
    axes[2].scatter(yy, y_array)
    axes[2].set_title('fit_y_diameter: '+ str(2*2.928515*y_para[2]*pixel_size/magnification)[:4]+ 'um.')
    axes[2].set_xlabel('pixel number')
    axes[2].set_ylabel('Intensity')
#     plt.text(0.75, 0.9, s=r'y_diameter: '+ str(2*2.928515*y_para[2]*pixel_size)[:4] + 'um.', transform=ax.transAxes)

    # print(2.928515*(x_para[2]+y_para[1])/2)
    marker = patches.Circle((point_site[2], point_site[1]), 2.928515*(x_para[2]+y_para[1])/2/magnification, color='r', fc='none', ec='w', lw=1, alpha=0.75)
    axes[0].add_patch(marker)
    plt.show()

    return y_para, x_para


# def sum_radial_intensity(image, point_site, width=50):
#     """Get the angular mean intensity in the image plane."""
#     img = copy.deepcopy(image)
#     distance  = []
#     point_intensity = []    

#     site_y = point_site[1]
#     site_x = point_site[2]
#     # 中心点的坐标
#     region = roi_region(img, site_y, site_x, width, width)    # roi_region = [roi_y_up, roi_y_down, roi_x_left, roi_x_right]
        
#     for iy in range(region[0], region[1]):
#         for jx in range(region[2], region[3]):
#             # dis_ = np.sqrt((iy-site_y)**2 + (jx-site_x)**2)*pixel_size
#             dis_ = np.sqrt((iy-site_y)**2 + (jx-site_x)**2)
#             if dis_ < width:
#                 distance.append(dis_)
#                 point_intensity.append(img[iy, jx])
#             # distance 标记每一个点到中心点的距离
#             # point_intensity标记每一个点对应的灰度强度
#             # 两个数组之间相互对应
    
#     step = 0.1
#     r_list = np.arange(1e-9, width+step*1.1, step)
#     i_list = []
#     for i in range(len(r_list)):
#         i_list.append([])
#     # print('distribution split number:'+str(len(i_list)))
#     # print("index",np.int32(max(distance)/step)) 
    
#     for radius_i, intensity_j in zip(distance, point_intensity):
#         index = np.int32(radius_i/step)
#         i_list[index].append(intensity_j)
        
#     radius = []
#     intensity = []
#     intensity_deviation = []
    
#     for radius_i, intensity_j in zip(r_list, i_list):
#         if len(intensity_j) > 0:
#             # radius.append(radius_i*pixel_size)
#             radius.append(radius_i)
#             intensity.append(np.mean(intensity_j))
#             intensity_deviation.append(np.std(intensity_j)/np.sqrt(len(intensity_j)))
            
#     radial_psf = [np.array(radius), np.array(intensity), np.array(intensity_deviation)]
    
#     return radial_psf


def sum_radial_intensity(image, point_site, width=50):
    """
    Get the angular mean intensity in the image plane.
    """
    site_y, site_x = point_site[1], point_site[2]
    region = roi_region(image, site_y, site_x, width, width)  # roi_region = [roi_y_up, roi_y_down, roi_x_left, roi_x_right]

    # Create a meshgrid for the region of interest
    y, x = np.ogrid[region[0]:region[1], region[2]:region[3]]
    distances = np.sqrt((y - site_y) ** 2 + (x - site_x) ** 2)
    mask = distances < width

    # Extract distances and intensities within the mask
    distances = distances[mask]
    intensities = image[region[0]:region[1], region[2]:region[3]][mask]

    step = 0.1
    r_list = np.arange(1e-9, width + step * 1.1, step)
    i_list = [[] for _ in range(len(r_list))]

    for radius_i, intensity_j in zip(distances, intensities):
        index = np.int32(radius_i / step)
        i_list[index].append(intensity_j)

    radius = []
    intensity = []
    intensity_deviation = []

    for radius_i, intensity_j in zip(r_list, i_list):
        if len(intensity_j) > 0:
            radius.append(radius_i)
            intensity.append(np.mean(intensity_j))
            intensity_deviation.append(np.std(intensity_j) / np.sqrt(len(intensity_j)))

    radial_psf = [np.array(radius), np.array(intensity), np.array(intensity_deviation)]

    return radial_psf


def psf_radius_fit(image, point_site, width=50):
    """Gaussian fit for the selected picture.""" 
    r_distance, r_intensity, intensity_std = sum_radial_intensity(image, point_site, width)
    guess = [image[point_site[1], point_site[2]], 6, 0]
    popt, conv = curve_fit(func_gaussian2, r_distance, r_intensity, p0=guess, maxfev = 1000000)
    #print('fit parameters:')
    #print(popt)
    
    return popt


def psf_radius_fit_obj(image, point_site, magnification, pixel_size, width=100):
    """Gaussian fit for every selected picture."""
    r_distance, r_intensity, intensity_std = sum_radial_intensity(image, point_site, width)
    r_distance = r_distance*pixel_size/magnification*1000

    # guess = [200, 0, 4, 2]
    # popt, conv = curve_fit(func_gaussian, r_distance, r_intensity, p0=guess, maxfev = 1000000)
    guess = [255, 0, 210, 0]
    popt, conv = curve_fit(func_gaussian, r_distance, r_intensity, p0=guess, maxfev = 1000000)
    print('fit parameters:')
    print(popt)
    fiterr = np.sqrt(np.diag(conv))


    fig0, ax0 = plt.subplots(1, figsize=(5,3.7), constrained_layout=True)
    radius = 2.928515*popt[2]/1000/pixel_size*magnification
    print(radius)
    roi = image[point_site[1]-20:point_site[1]+20, point_site[2]-20:point_site[2]+20]
    mappable=ax0.imshow(roi)
    ax0.set_title('point position: '+str(point_site[1])+' '+str(point_site[2]))
    marker = patches.Circle((point_site[2], point_site[1]), radius, color='r', fc='none', ec='w', lw=1, alpha=0.75)
    ax0.add_patch(marker)
    fig0.colorbar(mappable)
    
    fig, ax = plt.subplots(1, figsize=(5,3.7), constrained_layout=True)
    ax.set_title('Ponit: ' + ' y:'+str(point_site[1]) + ' x:'+ str(point_site[2]))
    ax.set_xlabel('radial distance (nm)')
    ax.set_ylabel('Intensity')
    ax.errorbar(r_distance, r_intensity, yerr=intensity_std, fmt='b.', mfc='none', lw=1)


    ax.plot(r_distance, func_gaussian(r_distance, popt[0], popt[1], popt[2], popt[3]), 'r-', lw=2)
    ax.text(0.6, 0.95, s=r'fit radius = %.1f(%.1f) nm'%(2.928515*popt[2], 2.928515*fiterr[2]), transform=ax.transAxes)
    y_major_locator=plt.MultipleLocator(10)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.grid()
    plt.show()

    return popt

def psf_radius_fit_obj2_plot(image, point_site, magnification, pixel_size, width=100):
    """Gaussian fit for every selected picture."""
    """Set gaussian function shift parameter is 0!"""
    r_distance, r_intensity, intensity_std = sum_radial_intensity(image, point_site, width)
    r_distance = r_distance*pixel_size/magnification*1000

    # guess = [200, 0, 4, 2]
    # popt, conv = curve_fit(func_gaussian, r_distance, r_intensity, p0=guess, maxfev = 1000000)
    guess = [255, 210, 0]
    popt, conv = curve_fit(func_gaussian2, r_distance, r_intensity, p0=guess, maxfev = 1000000)
    print('fit parameters:')
    print(popt)
    fiterr = np.sqrt(np.diag(conv))


    fig0, ax0 = plt.subplots(1, figsize=(5,3.7), constrained_layout=True)
    radius = 2.928515*popt[2]/1000/pixel_size*magnification
    print(radius)
    roi = image[point_site[1]-20:point_site[1]+20, point_site[2]-20:point_site[2]+20]
    mappable=ax0.imshow(roi)
    ax0.set_title('point position: '+str(point_site[1])+' '+str(point_site[2]))
    marker = patches.Circle((point_site[2], point_site[1]), radius, color='r', fc='none', ec='w', lw=1, alpha=0.75)
    ax0.add_patch(marker)
    fig0.colorbar(mappable)
    
    fig, ax = plt.subplots(1, figsize=(5,3.7), constrained_layout=True)
    ax.set_title('Ponit: ' + ' y:'+str(point_site[1]) + ' x:'+ str(point_site[2]))
    ax.set_xlabel('radial distance (nm)')
    ax.set_ylabel('Intensity')
    ax.errorbar(r_distance, r_intensity, yerr=intensity_std, fmt='b.', mfc='none', lw=1)


    ax.plot(r_distance, func_gaussian2(r_distance, popt[0], popt[1], popt[2]), 'r-', lw=2)
    ax.text(0.6, 0.95, s=r'fit radius = %.1f(%.1f) nm'%(2.928515*popt[1], 2.928515*fiterr[1]), transform=ax.transAxes)
    y_major_locator=plt.MultipleLocator(10)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.grid()
    plt.show()

    return popt,2.928515*popt[2] #艾里斑大小 nm

def psf_radius_fit_obj2(image, point_site, magnification, pixel_size, width=100):
    r_distance, r_intensity, intensity_std = sum_radial_intensity(image, point_site, width)
    r_distance = r_distance * pixel_size / magnification * 1000  # Convert to nm
    
    guess = [255, 210, 0]
    popt, conv = curve_fit(func_gaussian2, r_distance, r_intensity, p0=guess, maxfev=1000000)

    psf_radius_nm = 2.928515 * popt[2]
    print(psf_radius_nm, "nm")

    return popt, psf_radius_nm 
    
def main_peak_power_ratio(image, point_site, radial_psf, width=100):
    img = copy.deepcopy(image)
    distance = []
    intensity = []

    site_y = point_site[1]
    site_x = point_site[2]
    y_width = width
    x_width = width
    region = roi_region(img, site_y, site_x, y_width, x_width)

    for iy in range(region[0], region[1]):
        for jx in range(region[2], region[3]):
            dis_ = np.sqrt((iy-site_y)**2 + (jx-site_x)**2)
            distance.append(dis_)
            intensity.append(img[iy, jx])

    first_order_total_intensity = 0
    other_order_total_intensity = 0

    for distance_, intensity_ in zip(distance, intensity):
        # print(distance_)
        if distance_ <= radial_psf:
            first_order_total_intensity += intensity_

        if distance_ > radial_psf:
            other_order_total_intensity += intensity_

    main_peak_power_ratio = first_order_total_intensity/(first_order_total_intensity+other_order_total_intensity)
    
    return main_peak_power_ratio

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# 计算平均间距像素点的个数
def find_close_points(points, dis,dis1):
    num_points = len(points)
    max_dis = dis
    min_dis = dis1
    flag = True
    
    while flag:
        close_distances = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                distance = calculate_distance(points[i][1:], points[j][1:])
                if distance < max_dis*1.1 and distance >min_dis:
                    max_dis = distance
                    close_distances.append(distance)
        # print("points number:",len(close_distances))
        if all(value < min(close_distances)*1.2 for value in close_distances):
            flag = False
        else:
            max_dis = min(close_distances)*1.2
    return np.mean(close_distances)






    


    
        

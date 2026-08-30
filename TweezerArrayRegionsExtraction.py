import numpy as np
import matplotlib.pyplot as plt
from Gaussian_2D_Fit import *
from scipy import signal

def generate_points(left_top, left_bottom, right_top, right_bottom, array_size):
    """
    Generate a series of two-dimensional periodic point coordinates based on the four corner coordinates and grid size.

    Args:
    - left_top: Coordinates of the top-left corner point [x, y]
    - left_bottom: Coordinates of the bottom-left corner point [x, y]
    - right_top: Coordinates of the top-right corner point [x, y]
    - right_bottom: Coordinates of the bottom-right corner point [x, y]
    - array_size: Size of the point array [rows, columns]

    Returns:
    - points: Array of generated rounded integer point coordinates

    """

    # 将输入转换为 NumPy 数组
    left_top = np.array(left_top)
    left_bottom = np.array(left_bottom)
    right_top = np.array(right_top)
    right_bottom = np.array(right_bottom)

    # # 计算行向量和列向量
    # top_vector = (right_top - left_top)
    # left_vector = (left_bottom - left_top) 
    # bottom_vector = (right_bottom - left_bottom) 
    # right_vector = (right_bottom - right_top)


    # 生成点坐标
    points = []
    for i in range(array_size[0]):
        for j in range(array_size[1]):
            v = i / (array_size[0] - 1)
            u = j / (array_size[1] - 1)
 
            point = (1-v) * ((1-u) * left_top + u * left_bottom) + \
                    v    * ((1-u) * right_top + u * right_bottom)
            points.append(point)

    points = np.array(points)
    points_round = np.round(points).astype(int)

    return np.array(points_round)

# def generate_points_Tri(left_top, left_bottom, right_top, right_bottom, array_size):
#     # 将输入转换为 NumPy 数组
#     left_top = np.array(left_top)
#     left_bottom = np.array(left_bottom)
#     right_top = np.array(right_top)
#     right_bottom = np.array(right_bottom)

#     # # 计算行向量和列向量
#     # top_vector = (right_top - left_top)
#     # left_vector = (left_bottom - left_top) 
#     # bottom_vector = (right_bottom - left_bottom) 
#     # right_vector = (right_bottom - right_top)

#     k=int(array_size[0]/2)
#     # 生成点坐标
#     points = []
#     for j in range(array_size[1]):
#         for i in range(k):
#             u = i / (k - 1)
#             v = j / (array_size[1] - 1)
#             point = (1-v) * ((1-u) * left_top + u * left_bottom) + \
#                     v    * ((1-u) * right_top + u * right_bottom)+[-25,56]   

#             points.append(point)
#         for i in range(k):
#             u = i / (k - 1)
#             v = j / (array_size[1] - 1)        
#             point = (1-v) * ((1-u) * left_top + u * left_bottom) + \
#                     v    * ((1-u) * right_top + u * right_bottom)
#             points.append(point)
           
#     points = np.array(points)
#     points_round = np.round(points).astype(int)

#     return np.array(points_round)
def generate_points_Tri(left_top, left_bottom, right_top, right_bottom, array_size):
    # 将输入转换为 NumPy 数组
    left_top = np.array(left_top)
    left_bottom = np.array(left_bottom)
    right_top = np.array(right_top)
    right_bottom = np.array(right_bottom)

    # # 计算行向量和列向量
    # top_vector = (right_top - left_top)
    # left_vector = (left_bottom - left_top) 
    # bottom_vector = (right_bottom - left_bottom) 
    # right_vector = (right_bottom - right_top)
    if array_size[0]%2==0:
       k=int(array_size[0]/2)
       s=0 
    else:
        k=int((array_size[0]+1)/2)
        s=1
    # 生成点坐标
    points = []
    for i in range(k):
        for j in range(array_size[1]):
            u = i / (k - 1)
            v = j / (array_size[1] - 1)
            point = (1-v) * ((1-u) * left_top + u * left_bottom) + \
                    v    * ((1-u) * right_top + u * right_bottom)

            points.append(point)
        if s==0 or (s==1 and i!=(k-1)):
            for l in range(array_size[1]):
                u = i / (k - 1)
                v = l / (array_size[1] - 1)        
                point = (1-v) * ((1-u) * left_top + u * left_bottom) + \
                        v    * ((1-u) * right_top + u * right_bottom)+[-30,48]   
                points.append(point)


    points = np.array(points)
    points_round = np.round(points).astype(int)

    return np.array(points_round)

def generate_points_honey(left_top, left_bottom, right_top, right_bottom, array_size):
    # 将输入转换为 NumPy 数组
    left_top = np.array(left_top)
    left_bottom = np.array(left_bottom)
    right_top = np.array(right_top)
    right_bottom = np.array(right_bottom)

    # # 计算行向量和列向量
    # top_vector = (right_top - left_top)
    # left_vector = (left_bottom - left_top) 
    # bottom_vector = (right_bottom - left_bottom) 
    # right_vector = (right_bottom - right_top)
    if array_size[0]%2==0:
       k=int(array_size[0]/2)
       s=0 
    else:
        k=int((array_size[0]+1)/2)
        s=1
    # 生成点坐标
    points = []
    for i in range(k):
        for j in range(array_size[1]):
            if j %3!=2:
                u = i / (k - 1)
                v = j / (array_size[1] - 1)
                point = (1-v) * ((1-u) * left_top + u * left_bottom) + \
                        v    * ((1-u) * right_top + u * right_bottom)
                points.append(point)
        if s==0 or (s==1 and i!=(k-1)):
            for l in range(array_size[1]):
                if l %3!=1:
                    u = i / (k - 1)
                    v = l / (array_size[1] - 1)        
                    point = (1-v) * ((1-u) * left_top + u * left_bottom) + \
                            v    * ((1-u) * right_top + u * right_bottom)+[-22,39]   
                    points.append(point)
    # for j in range(array_size[1]-1,-1,-1):
    #     if j%3!=2:
    #         for i in range(k):
    #                 u = i / (k - 1)
    #                 v = j / (array_size[1] - 1)
    #                 point = (1-v) * ((1-u) * left_top + u * left_bottom) + \
    #                         v    * ((1-u) * right_top + u * right_bottom)
    #                 points.append(point)
    #     if j%3!=1:
    #         for i in range(k-1):
    #             u = i / (k - 1)
    #             v = j / (array_size[1] - 1)        
    #             point = (1-v) * ((1-u) * left_top + u * left_bottom) + \
    #                     v    * ((1-u) * right_top + u * right_bottom)+[-22,39]   
    #             points.append(point)        

    points = np.array(points)
    points_round = np.round(points).astype(int)

    return np.array(points_round)

def generate_points_kagome(left_top, left_bottom, right_top, right_bottom, array_size):
    # 将输入转换为 NumPy 数组
    left_top = np.array(left_top)
    left_bottom = np.array(left_bottom)
    right_top = np.array(right_top)
    right_bottom = np.array(right_bottom)


    if array_size[0]%2==0:
       k=int(array_size[0]/2)
       s=0 
    else:
        k=int((array_size[0]+1)/2)
        s=1
    # 生成点坐标
    points = []
    for i in range(k):
        for j in range(array_size[1]):
                u = i / (k - 1)
                v = j / (array_size[1] - 1)
                point = (1-v) * ((1-u) * left_top + u * left_bottom) + \
                        v    * ((1-u) * right_top + u * right_bottom)
                points.append(point)
        if s==0 or (s==1 and i!=(k-1)):
            for l in range(array_size[1]):
                if i%2==0 and l%2==1:
                    u = i / (k - 1)
                    v = l / (array_size[1] - 1)        
                    point = (1-v) * ((1-u) * left_top + u * left_bottom) + \
                            v    * ((1-u) * right_top + u * right_bottom)+[-24,42]   
                    points.append(point)    
                if i%2==1 and l%2==0:
                    u = i / (k - 1)
                    v = l / (array_size[1] - 1)        
                    point = (1-v) * ((1-u) * left_top + u * left_bottom) + \
                            v    * ((1-u) * right_top + u * right_bottom)+[-24,42]   
                    points.append(point)
                    
    points = np.array(points)
    points_round = np.round(points).astype(int)

    return np.array(points_round)

def generate_regions(points, output_size):
    """
    Generate regions of interest based on point coordinates and output size.

    Args:
    - points: Array of point coordinates
    - output_size: Output size [height, width]

    Returns:
    - regions: Array of generated regions of interest in the format [x1, x2, y1, y2]

    """

    regions = []

    for point in points:

        x, y = point
        h, w = output_size

        x1 = int(x - h // 2)
        x2 = x1 + h
        y1 = int(y - w // 2)
        y2 = y1 + w

        region = [x1, x2, y1, y2]
        regions.append(region)

    return np.array(regions)




def generate_fitted_regions(regions, image):

    fitted_regions = []
    intensity = []

    for region in regions:

        x1, x2, y1, y2 = region

        intensity.append(image[x1:x2,y1:y2].sum())
    
    intensity = np.array(intensity)

    mean = intensity.mean()
    std = intensity.std()

    warning_indices = intensity < (mean - 2.5*std)
    safe_indices = intensity >= (mean - 2.5*std)


    for k, region in enumerate(regions):
        # if warning_indices[k]:
        #     fitted_regions.append(region)
        #     continue
        
        x1, x2, y1, y2 = region

        image_of_interest = image[x1:x2, y1:y2]
        x0_max, y0_max = np.argwhere(image_of_interest == np.max(image_of_interest)) [0]

        h = x2 - x1
        w = y2 - y1

        x1_max = x1 + x0_max - h //2
        x2_max = x1_max + h
        y1_max = y1 + y0_max - w //2
        y2_max = y1_max + w


        # try:
        #     popt, pcov, fitted_image = gaussian_2d_fit(image[x1_max:x2_max, y1_max:y2_max])
        #     x0_fitted = popt[1]
        #     y0_fitted = popt[2]
        #     print('0')
        #     x0_round = np.round(x0_fitted).astype(int)
        #     y0_round = np.round(y0_fitted).astype(int)

        #     h = x2 - x1
        #     w = y2 - y1
            
        #     x1_fitted = x1_max + int(x0_round - h // 2)
        #     x2_fitted = x1_fitted + h
        #     y1_fitted = y1_max + int(y0_round - w // 2)
        #     y2_fitted = y1_fitted + w

        #     fitted_region = [x1_fitted, x2_fitted, y1_fitted, y2_fitted]

        # except:
        #     # print('1')

        fitted_region = [x1_max, x2_max, y1_max, y2_max]

        delta = np.abs(fitted_region - region)

        if np.any(delta > 2):
            fitted_regions.append(region)
            print(f'warning {k}')

        else:
            fitted_regions.append(fitted_region)

    return np.array(fitted_regions)


def generate_maxmium_subregions(regions, image, output_size:list):

    intensity = []

    for region in regions:

        x1, x2, y1, y2 = region

        intensity.append(image[x1:x2,y1:y2].sum())
    
    intensity = np.array(intensity)

    mean = intensity.mean()
    std = intensity.std()

    warning_indices = (np.where(intensity < (mean - 1*std)))[0]
    safe_indices = (np.where(intensity >= (mean - 1*std)))[0]




    maxmium_subregions = []
    
    input_h = regions[0,1] - regions[0,0]
    input_w = regions[0,3] - regions[0,2]

    output_h = output_size[0]
    output_w = output_size[1]

    convolution_h = input_h - output_h + 1
    convolution_w = input_w - output_w + 1

    for i, region in enumerate(regions):

        if i in warning_indices:
            print(f'wanring: {i}')
            maxmium_subregions.append(region)

        else:
            x1, x2, y1, y2 = region
            sub_image = image[x1:x2, y1:y2]
            
            kernel = np.ones((output_h, output_w))

            convolution = signal.convolve(sub_image, kernel, mode='valid')

            max_index = convolution.argmax()

            max_x = max_index // convolution_w
            max_y = max_index % convolution_w

            x1_new = x1 + max_x
            x2_new = x1_new + output_h
            y1_new = y1 + max_y
            y2_new = y1_new + output_w

            maxmium_subregion = [x1_new, x2_new, y1_new, y2_new]

            maxmium_subregions.append(maxmium_subregion)

    return np.array(maxmium_subregions)













def generate_aoi(regions, image, edge_value, show=False):
    """
    Generate an Area of Interest (AOI) image based on regions, image, and edge value.

    Args:
    - regions: Array of regions of interest
    - image: Input image array
    - edge_value: Value assigned to the edge regions
    - show: Whether to display the AOI image, default is False

    Returns:
    - aoi: Generated AOI array

    """

    aoi = np.zeros_like(image)

    for region in regions:
        
        x1, x2, y1, y2 = region

        # 设置上下左右四个边缘区域的取值
        aoi[x1, y1:y2] = edge_value  # 上边缘
        aoi[x2 - 1, y1:y2] = edge_value  # 下边缘
        aoi[x1:x2, y1] = edge_value  # 左边缘
        aoi[x1:x2, y2 - 1] = edge_value  # 右边缘

    if show:
        plt.figure(figsize=(15,15))
        plt.imshow(aoi)
        plt.colorbar()
        plt.show()

    return aoi
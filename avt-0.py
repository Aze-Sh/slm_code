# from pymba import Vimba
# import numpy as np
# def get_image(etime):
#     with Vimba() as vimba:
#         cams = vimba.cameras()
#         with cams [0] as cam:
#             # Aquire single frame synchronously
#             exposure_time = cam.ExposureTime
#             exposure_time.set(etime)
#             updated_exposure_time = cam.ExposureTime.get()
#             frame = cam.get_frame ()
#             image=frame.as_numpy_ndarray()
#     return image.squeeze(axis=2)

from pymba import Vimba
import numpy as np
import matplotlib.pyplot as plt

def get_image(etime):
    with Vimba() as vimba:
        camera_ids = vimba.camera_ids()
        print(camera_ids[0])
        cam =  vimba.camera(camera_ids[0])
        cam.open()
        cam.arm(mode='SingleFrame')

        # 设置曝光时间
        cam.ExposureTime = etime  # 新版属性直接赋值
        
        # 获取单帧
        frame = cam.acquire_frame()
        image = frame.buffer_data_numpy()

        cam.disarm()
        print(image.shape)
        plt.figure(figsize=(10,10))
        plt.imshow(image)   
        plt.colorbar()
        plt.show()
    return image
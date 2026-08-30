from threading import Event
from contextlib import ExitStack
import numpy as np
from vmbpy import VmbSystem, FrameStatus


class VimbaCamera:
    def __init__(self, cam_index=0):
        self._stack = ExitStack()
        self._vmb = self._stack.enter_context(VmbSystem.get_instance())
        cameras = self._vmb.get_all_cameras()
        if not cameras:
            self._stack.close()
            raise RuntimeError("没有发现 Allied Vision 相机")

        if cam_index >= len(cameras):
            self._stack.close()
            raise IndexError("cam_index 超出范围")

        self._cam = self._stack.enter_context(cameras[cam_index])

    def capture(self, exposure_time_us, timeout=2.0):
        """获取一帧图像并以 numpy 数组返回"""
        grab_event = Event()
        grabbed = {}

        def handler(cam, stream, frame):
            if frame.get_status() == FrameStatus.Complete:
                grabbed["frame"] = frame
                grab_event.set()
            stream.queue_frame(frame)

        self._cam.get_feature_by_name('ExposureTime').set(exposure_time_us)
        self._cam.start_streaming(handler=handler, buffer_count=1)

        if not grab_event.wait(timeout=timeout):
            self._cam.stop_streaming()
            raise TimeoutError("等待帧超时")

        self._cam.stop_streaming()
        return grabbed["frame"].as_numpy_ndarray()[:,:,0]

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

# 使用示例
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    with VimbaCamera() as camera:
        img1 = camera.capture(1000)
        plt.imshow(img1)
        plt.show()
        img2 = camera.capture(2000)
        plt.imshow(img2)
        plt.show()


        # 示例：打印图像大小
        print(img1.shape, img2.shape)

#
#
# from threading import Event
# import matplotlib.pyplot as plt
# from vmbpy import VmbSystem,FrameStatus
#
#
# def get_image(exposure_time_us: float):
#     grab_event = Event()
#     grabbed_frame = {}
#
#     def handler(cam, stream, frame):
#         if frame.get_status() == FrameStatus.Complete:
#             grabbed_frame['frame'] = frame
#             grab_event.set()
#         stream.queue_frame(frame)
#
#     with VmbSystem.get_instance() as vmb:
#         cams = vmb.get_all_cameras()
#         if not cams:
#             raise RuntimeError("没有发现 Allied Vision 相机")
#
#         cam = cams[0]
#
#         with cam:  # 默认以 Full Access 打开
#             cam.get_feature_by_name('ExposureTime').set(exposure_time_us)
#
#             cam.start_streaming(handler=handler, buffer_count=1)
#             if not grab_event.wait(timeout=2.0):
#                 cam.stop_streaming()
#                 raise TimeoutError("等待帧超时")
#             cam.stop_streaming()
#
#             image = grabbed_frame['frame'].as_numpy_ndarray()
#             plt.imshow(image, cmap='gray')
#             plt.colorbar()
#             plt.show()
#
#             return image

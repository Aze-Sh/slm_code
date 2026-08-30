from ids_peak import ids_peak as peak
from ids_peak_ipl import ids_peak_ipl
import matplotlib.pyplot as plt
import numpy as np
from camera import AbstractCamera, ShutterMode
import cv2
import ctypes
import threading
import time
from threading import Event

class IDS_Peak_Camera(AbstractCamera):


    def __init__(self, init_camera: bool = False, selected_device=0) -> None:
        # self.logger: logging.Logger = logging.getLogger(__name__)
        super().__init__()
        if init_camera:
            # selected_device = 0
            self.InitCamera(selected_device)

    def InitCamera(self, selected_device:int = 0):
        self.selected_device = selected_device
        self.m_device = None
        self.m_dataStream = None
        self.m_node_map_remote_device = None
        self.capture_images = {}
        self.cur_num = 0

        try:
            # initialize library
            peak.Library.Initialize()

            # Create instance of the device manager
            device_manager = peak.DeviceManager.Instance()  

            # Update the device manager
            device_manager.Update()

            if device_manager.Devices().empty():
                raise DeviceConnectionError('No device found!')
            
            if not device_manager.Devices()[self.selected_device].IsOpenable():
                raise DeviceConnectionError('The device you selected is not openable!')

        except DeviceConnectionError as e:
            print(e.args)

        else:
            # list all available devices
            for i, device in enumerate(device_manager.Devices()):
                print(str(i) + ": " + device.ModelName() + " ("
                        + device.ParentInterface().DisplayName() + "; "
                        + device.ParentInterface().ParentSystem().DisplayName() + "v."
                        + device.ParentInterface().ParentSystem().Version() + ")")
                
            # whether the device is openable
            device_count = device_manager.Devices().size()
            for i in range(device_count):
                if device_manager.Devices()[i].IsOpenable():
                    print('The device',i,'is openable.')
                else:
                    print('The device',i,'is not openable.')

            # open the user selected device
            self.m_device = device_manager.Devices()[self.selected_device].OpenDevice(peak.DeviceAccessType_Control)
            # Get NodeMap of the RemoteDevice for all accesses to the GenICam NodeMap tree
            self.m_node_map_remote_device = self.m_device.RemoteDevice().NodeMaps()[self.selected_device]
            
            print('You select device',i,', now it has been connected!')
    

    def SetROI(self, x=0, y=0, width=2448, height=2048):
        """
        Set ROI.
        """

        # Get the minimum ROI and set it. After that there are no size restrictions anymore
        x_min = self.m_node_map_remote_device.FindNode("OffsetX").Minimum()
        y_min = self.m_node_map_remote_device.FindNode("OffsetY").Minimum()
        w_min = self.m_node_map_remote_device.FindNode("Width").Minimum()
        h_min = self.m_node_map_remote_device.FindNode("Height").Minimum()

        self.m_node_map_remote_device.FindNode("OffsetX").SetValue(x_min)
        self.m_node_map_remote_device.FindNode("OffsetY").SetValue(y_min)
        self.m_node_map_remote_device.FindNode("Width").SetValue(w_min)
        self.m_node_map_remote_device.FindNode("Height").SetValue(h_min)

        # Get the maximum ROI values
        x_max = self.m_node_map_remote_device.FindNode("OffsetX").Maximum()
        y_max = self.m_node_map_remote_device.FindNode("OffsetY").Maximum()
        w_max = self.m_node_map_remote_device.FindNode("Width").Maximum()
        h_max = self.m_node_map_remote_device.FindNode("Height").Maximum()

        if (x < x_min) or (y < y_min) or (x > x_max) or (y > y_max):
            print('The start point beyond the boundary!')
        elif (width < w_min) or (height < h_min) or ((x + width) > w_max) or ((y + height) > h_max):
            print('The width and height beyond the boundary!')
        else:
            # Now, set final AOI
            self.m_node_map_remote_device.FindNode("OffsetX").SetValue(x)
            self.m_node_map_remote_device.FindNode("OffsetY").SetValue(y)
            self.m_node_map_remote_device.FindNode("Width").SetValue(width)
            self.m_node_map_remote_device.FindNode("Height").SetValue(height)

    def SetTrigger(self, source:str = 'line0'):
        '''
        Activate the ExposureStart trigger and configure its source.

        
        :param `src`,  available parameters as
            - 'Software'
            - 'Line0' to 'Line6'
            - 'PWM0'
            - 'SignalMultiplier0'
            - 'UserOutput0' to 'UserOutput3'
            - 'Counter0Active', 'Counter1Active', 'Counter0End', 'Counter1End', 'Counter0Start', 'Counter1Start'
            - 'Timer0Active', 'Timer1Active', 'Timer0End', 'Timer1End', 'Timer0Start', 'Timer1Start'
        '''
        try:
            self.m_node_map_remote_device.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
            self.m_node_map_remote_device.FindNode("TriggerMode").SetCurrentEntry("On")
            self.m_node_map_remote_device.FindNode("TriggerSource").SetCurrentEntry(source)
        except Exception as e:
            print(e)
        
    def SetTriggerSelector(self, slc:str = 'ExposeureStart'):
        '''
        Selects the type of trigger to configure.

        :param `slc` available parameters as
            - 'AcquisitionStart'
            - 'AcquisitionEnd'
            - 'ExposureStart'
            - 'ExposureEnd'
            - 'FrameStart'
            - 'LineStart'
            - 'ReadOutStart'
        '''
        try:
            self.m_node_map_remote_device.FindNode("TriggerSelector").SetCurrentEntry(slc)
        except Exception as e:
            print(e)
        
    def SetTriggerSource(self, src:str = 'ExposeureStart'):
        '''
        Specifies the internal signal or physical input line to use as the trigger source. The selected trigger must have its `TriggerMode` set to "On".

        :param `src`,  available parameters as
            - 'Software'
            - 'Line0' to 'Line6'
            - 'PWM0'
            - 'SignalMultiplier0'
            - 'UserOutput0' to 'UserOutput3'
            - 'Counter0Active', 'Counter1Active', 'Counter0End', 'Counter1End', 'Counter0Start', 'Counter1Start'
            - 'Timer0Active', 'Timer1Active', 'Timer0End', 'Timer1End', 'Timer0Start', 'Timer1Start'
        '''
        try:
            self.m_node_map_remote_device.FindNode("TriggerSource").SetCurrentEntry(src)
        except Exception as e:
            print(e)

    def SetBitDepth(self, BitDepth:int = 12):
        """
        Set bit depth: 8,10,12.
        """
        try:
            if BitDepth == 8:
                self.m_node_map_remote_device.FindNode("PixelFormat").SetCurrentEntry("Mono8")
            if BitDepth == 10:
                self.m_node_map_remote_device.FindNode("PixelFormat").SetCurrentEntry("Mono10")
            if BitDepth == 12:
                self.m_node_map_remote_device.FindNode("PixelFormat").SetCurrentEntry("Mono12")
        except Exception as e:
            print(e)
 
    def GetExposureTime(self, ):
        """
        Return exposure time, values in microseconds.
        """
        try:
            return self.m_node_map_remote_device.FindNode("ExposureTime").Value()
        except Exception as e:
            print(e)

    def SetExposureTime(self, ExposureTime):
        """
        Set exposure time, values in microseconds. The value must be greater than or equal to 20.216216.
        """
        try:
            self.m_node_map_remote_device.FindNode("ExposureTime").SetValue(ExposureTime)
        except Exception as e:
            print(e)

    def GetFrameRate(self):
        """
        Return the real-time FPS
        """
        try:
            return self.m_node_map_remote_device.FindNode("AcquisitionFrameRate").Value()
        except Exception as e:
            print(e)

    def SetFrameRate(self, FPS):
        """
        Set frame rate, value in frame per second (FPS).
        """
        try:
            self.m_node_map_remote_device.FindNode("AcquisitionFrameRate").SetValue(FPS)
        except Exception as e:
            print(e)


    def SetRollingShutter(self):

        try:
            self.m_node_map_remote_device.FindNode("SensorShutterMode").SetCurrentEntry("Rolling")
        except Exception as e:
            print(e)
        
    def SetGlobalShutter(self):

        try:
            self.m_node_map_remote_device.FindNode("SensorShutterMode").SetCurrentEntry("Global")
        except Exception as e:
            print(e)

    def SetGlobalStart(self):

        try:
            self.m_node_map_remote_device.FindNode("SensorShutterMode").SetCurrentEntry("GlobalReset")
        except Exception as e:
            print(e)

    def SetSingleFrameAcquisition(self):
        '''
        Set the acquisition mode to the single frame, i.e. One image is captured
        one time.
        '''
        try:
            self.m_node_map_remote_device.FindNode("AcquisitionMode").SetCurrentEntry("SingleFrame")
        except Exception as e:
            print(e)
    
    def SetContinuousAcquisition(self):
        '''
        Set the acquisition mode to the continuous, i.e. Images are captured until stopped with the `StopAcquisition` command.
        '''
        try:
            self.m_node_map_remote_device.FindNode("AcquisitionMode").SetCurrentEntry("Continuous")
        except Exception as e:
            print(e)

    def PrepareAcquisition(self):
        """
        Prepare acquisition, open data stream.
        """
        self.data_streams = self.m_device.DataStreams()
        if self.data_streams.empty():
            print('no data streams available')
        
        self.m_dataStream = self.m_device.DataStreams()[0].OpenDataStream()

    
    def AllocAndAnnounceBuffers(self):
        """
        Alloc and announce buffers. Typically alloc 3 buffers.
        """
        if self.m_dataStream:

            # Flush queue and prepare all buffers for revoking
            self.m_dataStream.Flush(peak.DataStreamFlushMode_DiscardAll)

            # Clear all old buffers
            for buffer in self.m_dataStream.AnnouncedBuffers():
                self.m_dataStream.RevokeBuffer(buffer)

            payload_size = self.m_node_map_remote_device.FindNode("PayloadSize").Value()

            # Get number of minimum required buffers
            self.num_buffers_min_required = self.m_dataStream.NumBuffersAnnouncedMinRequired()

            # Alloc buffers
            for count in range(self.num_buffers_min_required):
                buffer = self.m_dataStream.AllocAndAnnounceBuffer(payload_size)
                self.m_dataStream.QueueBuffer(buffer)
            
    
    def StartAcquisition(self, release_buffer:bool = False):
        """
        Start continuous acquisition, and release the last buffer for the following single image.
        """
        try:
            # start acquisition
            self.m_dataStream.StartAcquisition(peak.AcquisitionStartMode_Default, peak.DataStream.INFINITE_NUMBER)
            self.m_node_map_remote_device.FindNode("TLParamsLocked").SetValue(1)
            self.m_node_map_remote_device.FindNode("AcquisitionStart").Execute()

            # get all the buffers, release the last buffer for the later use.
            if release_buffer:
                for count in range(self.num_buffers_min_required):
                    # Get buffer from device's DataStream. Wait 5000 ms. The buffer is automatically locked until it is queued again.
                    self.buffer = self.m_dataStream.WaitForFinishedBuffer(5000)
        except Exception as e:
            print(f"Error Acquisition: {e}")


    def GetImage(self) -> np.ndarray:
        """
        Create IDS peak IPL image from the last buffer, output numpy array.

        """
        try:
            # queue the last buffer
            # self.m_dataStream.QueueBuffer(self.buffer)
            trigger_source = self.m_node_map_remote_device.FindNode("TriggerSource").CurrentEntry().SymbolicValue()
            # print(trigger_source)
            if trigger_source == 'Software':
                self.m_node_map_remote_device.FindNode('TriggerSoftware').Execute()
                self.m_node_map_remote_device.FindNode("TriggerSoftware").WaitUntilDone()
                # print('Trigger successfully!')

            # Get buffer from device's DataStream. Wait 5000 ms. The buffer is automatically locked until it is queued again.
            self.buffer = self.m_dataStream.WaitForFinishedBuffer(5000)
            # print('11')

            image = ids_peak_ipl.Image.CreateFromSizeAndBuffer(
                self.buffer.PixelFormat(),
                self.buffer.BasePtr(),
                self.buffer.Size(),
                self.buffer.Width(),
                self.buffer.Height()
            )
            image_np = image.get_numpy().copy()
            # print(type(image_np))
            self.m_dataStream.QueueBuffer(self.buffer)
            return True, image_np
        except Exception as e:
                print(e)
    

    def StopAcquisition(self):
        """
        Stop acquisition.
        """
        self.m_node_map_remote_device.FindNode("AcquisitionStop").Execute()

        self.m_dataStream.KillWait()
        self.m_dataStream.StopAcquisition(peak.AcquisitionStopMode_Default)
        self.m_dataStream.Flush(peak.DataStreamFlushMode_DiscardAll)


    def Close(self):
        if hasattr(self,'event'):
            self.event.set()
            print("Thread is close!")
        peak.Library.Close()


    
    # Followings are rewriting abstract method.
    

    def _prepare_for_imaging(self, **args):
        self.PrepareAcquisition()
        self.AllocAndAnnounceBuffers()
        self.StartAcquisition(release_buffer=args['release_buffer'])

    def _get_image(self):
        success, image = self.GetImage()
        return success, image

    def set_exposure_time(self, exposure_time: float):
        return self.SetExposureTime(exposure_time)

    def set_frame_rate(self, frame_rate: float):
        return self.SetFrameRate(frame_rate)

    def set_shutter_mode(self, mode: int):
        if mode == 0:
            self.SetRollingShutter()
        elif mode == 1:
            self.SetGlobalShutter()
        elif mode == 2:
            self.SetGlobalStart()
        else:
            raise ValueError
        
    def run_realtime(self,x=1400,y=930,w=2000,h=1400,threshold=16,exposureTime=2000,gain=25) -> None:
        self.SetROI()
        self.SetExposureTime(exposureTime)
        self.SetTrigger(source='Software')
        self.SetContinuousAcquisition()
        self._prepare_for_imaging(release_buffer=False)
        self.set_analog_gain(gain)

        cv2.namedWindow('IDS', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('IDS', 500, 500)
        
        try:
            while True:
                success, image = self._get_image()
                if success:
                    print('sum ',self.draw_rectangle_and_sum(image, x, y, w, h,threshold))
                    
                    if len(image.shape) == 2:  # 如果是灰度图像
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    cv2.imshow('IDS',image)
                    
                    # 按 'q' 键退出实时显示
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
        finally:
                self.StopAcquisition()
                self.Close()
                
    def run_trigger(self,x=1400,y=930,w=2000,h=1400,threshold=16,exposureTime=2000,gain=25,trigger_num=1) -> None:
        self.SetROI()
        self.SetExposureTime(exposureTime)
        self.SetTrigger(source='Software')
        self.SetContinuousAcquisition()
        self._prepare_for_imaging(release_buffer=False)
        self.set_analog_gain(gain)
        
        num = trigger_num
        value_list = []
        try:
            while trigger_num > 0:
                trigger_num = trigger_num-1
                success, image = self._get_image()
                if success:
                    if len(image.shape) == 2:  # 如果是灰度图像
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    value = self.draw_rectangle_and_sum(image, x, y, w, h,threshold) 
                    value_list.append(value)
                else:
                    break
        finally:
                self.StopAcquisition()
        if trigger_num == 0 :
            return np.mean(value_list)
        else:
            return 0
        
    def set_analog_gain(self, gain_value):
        gain_node = self.m_node_map_remote_device.FindNode("Gain")
        if gain_node is not None:
            gain_node.SetValue(gain_value)
            print(f"Analog gain set to {gain_value}")
        else:
            print("Gain node not found")
            
    def draw_rectangle_and_sum(self,image, x, y, w, h,threshold=5):
        img_height, img_width = image.shape[:2]
        top_left = (max(0, x - w // 2), max(0, y - h // 2))
        bottom_right = (min(img_width, x + w // 2), min(img_height, y + h // 2))
        cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 4)
        
        # 获取矩形内的灰度值并求和
        roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        mean_value = np.sum(roi[roi>threshold])
        return int(mean_value/3)

    # set external trigger
    def configure_for_external_trigger(self):
        # Configure the camera for external trigger
        nodemap = self.m_node_map_remote_device

        # Set trigger source to external trigger (e.g., Line0)
        trigger_source_node = nodemap.FindNode("TriggerSource")
        trigger_source_node.Value = "Line0"

        # Set trigger activation to rising edge
        trigger_activation_node = nodemap.FindNode("TriggerActivation")
        trigger_activation_node.Value = "RisingEdge"
        
        acquisition_mode_node = nodemap.FindNode("AcquisitionMode")
        acquisition_mode_node.Value = "Continuous"

        # Set trigger mode to 'On'
        trigger_mode_node = nodemap.FindNode("TriggerMode")
        trigger_mode_node.Value = "On"

        print("Camera configured for external trigger.")

    # get image by external trigger
    def capture_image(self):
        try:
            buffer = self.m_dataStream.WaitForFinishedBuffer(5) 
            # Get the image from the buffer
            image_ptr = buffer.BasePtr()
            image_size = buffer.Size()
            image_data = (ctypes.c_ubyte * image_size).from_address(int(image_ptr))

            width = buffer.Width()
            height = buffer.Height()
            # print(
            #     "width:", width,"\n",
            #     "height:", height, "\n",
            #     "size:", buffer.size()
            # )
            image_np = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width))
            
            # Queue the buffer for the next acquisition
            self.m_dataStream.QueueBuffer(buffer)
            print("Get 1 image!",np.sum(image_np), "cur_num:", self.cur_num)
            return True, image_np
        except Exception as e:
            print("Buffer is empty!")
            return False, None
    
    def absorbtion_imaging(self,event,exposureTime,num=1,analogGain=1):
        # set external trigger:Rising Edge
        self.configure_for_external_trigger()
        self.SetExposureTime(exposureTime)
        self.set_analog_gain(analogGain)
        print("Waiting for external trigger...")
        
        self._prepare_for_imaging(release_buffer=False)
        self.capture_images = {}
        self.cur_num = 0
        flag = 0
        while len(self.capture_images) < num:
            if hasattr(self,'event') and event.is_set():
                break
            success, image = self.capture_image()
            if success:
                print(
                    "if success", self.cur_num, np.sum(image), 
                    (np.sum(self.capture_images[0]) if self.cur_num>0 else None)
                )
                self.capture_images[self.cur_num]=image.copy()
                self.cur_num += 1
            else:
                time.sleep(1e-5)
                flag +=1
                if flag > 6e3: 
                    self.event.set()
                    print("No Image for long time, camera is closed!")
                    break
        if len(self.capture_images)==num:
            print(f"Get {num} images successfully!")
        return self.capture_images
    
    def get_list(self):
        res = []
        for i, j in zip([1, 2, 3], [3, 5, 7]):
            print(i,j)
            res.append(i+j)
        return res
    
    def start_external_trigger(self,num=1,analogGain = 1,exposureTime=2000):
        self.event = Event()
        imaging_thread = threading.Thread(target=self.absorbtion_imaging, args=(self.event,exposureTime,num,analogGain,))
        imaging_thread.start()
        # imaging_thread.join()

    def threading_num(self):
        return len(threading.enumerate())
    
    def get_result(self,num:int):
        images = self.capture_images
        if len(images)>0:
            if num == -1:
                self.capture_images = {}
                self.cur_num = 0
                return images
            else:
                assert False, "num!=-1 is not realized"
                if len(self.capture_images)>0:
                    del self.capture_images[num]
                    return [images[num]]
        else:
            return None
        
class DeviceConnectionError(RuntimeError):
    def __init__(self, arg):
        self.args = arg


if __name__ == '__main__':
    
    IDS_Camera = IDS_Peak_Camera(True)
    

    IDS_Camera.SetROI()
    IDS_Camera.SetBitDepth(12)
    IDS_Camera.SetExposureTime(200)
    IDS_Camera.SetTrigger(source='Software')
    IDS_Camera.SetContinuousAcquisition()
    
    IDS_Camera._prepare_for_imaging(release_buffer=False)



    
    success, image = IDS_Camera._get_image()
    if success:
        # plt.imshow(image)
        # plt.colorbar()
        print('Mean value of Image is',np.mean(image))

    
    success, image = IDS_Camera._get_image()
    if success:
        # plt.imshow(image)
        # plt.colorbar()
        print('Mean value of Image is',np.mean(image))


    
    IDS_Camera.StopAcquisition()
    IDS_Camera.Close()


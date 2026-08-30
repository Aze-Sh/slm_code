from ctypes import *
import numpy as np
from time import sleep
import time

class BlinkSLM:
    def __init__(self):
        self.slm_lib = CDLL("C:\\Program Files\\Meadowlark Optics\\Blink Plus\\SDK\\Blink_C_wrapper")
        self.image_lib = CDLL("C:\\Program Files\\Meadowlark Optics\\Blink Plus\\SDK\\ImageGen")
        
        
        self.bit_depth = c_uint(12)
        self.num_boards_found = c_uint(0)
        self.constructed_okay = c_uint(1)
        self.is_nematic_type = c_bool(1)
        self.RAM_write_enable = c_bool(1)
        self.use_GPU = c_bool(1)
        self.max_transients = c_uint(20)
        self.board_number = c_uint(1)
        self.wait_For_Trigger = c_uint(0)
        self.flip_immediate = c_uint(0)
        self.timeout_ms = c_uint(1000)
        self.center_x = c_float(512)
        self.center_y = c_float(512)
        
        self.OutputPulseImageFlip = c_uint(0)
        self.OutputPulseImageRefresh = c_uint(0)
        
        self.slm_lib.Create_SDK(byref(self.num_boards_found), byref(self.constructed_okay))
        # self.slm_lib.Create_SDK(self.bit_depth, byref(self.num_boards_found), byref(self.constructed_okay),
        #                         self.is_nematic_type, self.RAM_write_enable, self.use_GPU, self.max_transients, 0)

        if self.constructed_okay.value == 0:
            print("Blink SDK did not construct successfully")

        if self.num_boards_found.value == 1:
            print("Blink SDK was successfully constructed")
            print("Found %s SLM controller(s)" % self.num_boards_found.value)
            self.height = c_uint(self.slm_lib.Get_image_height(self.board_number))
            self.width = c_uint(self.slm_lib.Get_image_width(self.board_number))
            self.depth = c_uint(self.slm_lib.Get_image_depth(self.board_number))
            self.Bytes = c_uint(self.depth.value // 8)
            self.center_x = c_uint(self.width.value // 2)
            self.center_y = c_uint(self.height.value // 2)
            
            # self.slm_lib.Load_LUT_file(self.board_number, b"lut\\slm_60.lut")
            self.slm_lib.Load_LUT_file(self.board_number,  b"C:\\Users\\Demeter\\Desktop\\slm.lut")



    def write_image(self, image_data):
        retVal = self.slm_lib.Write_image(self.board_number, image_data.ctypes.data_as(POINTER(c_ubyte)),
                                          self.height.value * self.width.value * self.Bytes.value,
                                          self.wait_For_Trigger, self.flip_immediate, self.OutputPulseImageFlip,
                                          self.OutputPulseImageRefresh, self.timeout_ms)
        if retVal == -1:
            print("DMA Failed")
            self.slm_lib.Delete_SDK()
        self.slm_lib.ImageWriteComplete(self.board_number, self.timeout_ms)
        
        
    def get_slm_temperature(self):
        self.slm_lib.Read_SLM_temperature.restype = c_double
        slm_temperature = self.slm_lib.Read_SLM_temperature(
            self.board_number
        )

        return slm_temperature
    def monitor_temperature(self):
        try:
            while True:
                temperature = self.get_slm_temperature()
                print(f"Current SLM Temperature: {temperature:.2f} °C")
                sleep(0.1)  # 每秒检查一次温度
        except KeyboardInterrupt:
            print("Temperature monitoring stopped.")
            
def read_and_store_temperature(slm, duration=10, interval=0.01):
     temperatures = []
     start_time = time.time()
    
     while (time.time() - start_time) < duration:
        temperature = slm.get_slm_temperature()
        temperatures.append(temperature)
        time.sleep(interval)
    
     return temperatures
 




    



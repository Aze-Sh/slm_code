# Example usage of Blink_C_wrapper.dll
# Meadowlark Optics Spatial Light Modulators
# September 12 2019

import os
import numpy
from ctypes import *
from scipy import misc
from time import sleep
from TLPM import TLPM
import pandas as pd
from PIL import Image
tlPM = TLPM()
deviceCount = c_uint32()
tlPM.findRsrc(byref(deviceCount))
resourceName = create_string_buffer(1024)

for i in range(0, deviceCount.value):
    tlPM.getRsrcName(c_int(i), resourceName)
    print(c_char_p(resourceName.raw).value)
    break

tlPM.close()

tlPM = TLPM()
#resourceName = create_string_buffer(b"COM1::115200")
#print(c_char_p(resourceName.raw).value)
tlPM.open(resourceName, c_bool(True), c_bool(True))

message = create_string_buffer(1024)
tlPM.getCalibrationMsg(message)
print(c_char_p(message.raw).value)

# Load the DLL
# Blink_C_wrapper.dll, Blink_SDK.dll, ImageGen.dll, FreeImage.dll and wdapi1021.dll
# should all be located in the same directory as the program referencing the
# library
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink Plus\\SDK\\Blink_C_wrapper")
slm_lib = CDLL("Blink_C_wrapper")

# Open the image generation library
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink Plus\\SDK\\ImageGen")
image_lib = CDLL("ImageGen")

# Basic parameters for calling Create_SDK
bit_depth = c_uint(12)
num_boards_found = c_uint(0)
constructed_okay = c_uint(-1)
is_nematic_type = c_bool(1)
RAM_write_enable = c_bool(1)
use_GPU = c_bool(1)
max_transients = c_uint(20)
board_number = c_uint(1)
wait_For_Trigger = c_uint(0)
flip_immediate = c_uint(0) #only supported on the 1024
timeout_ms = c_uint(5000)
center_x = c_float(256)
center_y = c_float(256)
VortexCharge = c_uint(3)
fork = c_uint(0)
RGB = c_uint(0)

# Both pulse options can be false, but only one can be true. You either generate a pulse when the new image begins loading to the SLM
# or every 1.184 ms on SLM refresh boundaries, or if both are false no output pulse is generated.
OutputPulseImageFlip = c_uint(0)
OutputPulseImageRefresh = c_uint(0); #only supported on 1920x1152, FW rev 1.8. 


# Call the Create_SDK constructor
# Returns a handle that's passed to subsequent SDK calls
slm_lib.Create_SDK( byref(num_boards_found), byref(constructed_okay))

if constructed_okay.value == 0:
	print ("Blink SDK did not construct successfully")

if num_boards_found.value == 1:
	print ("Blink SDK was successfully constructed")
	print ("Found %s SLM controller(s)" % num_boards_found.value)
	NumDataPoints=256
	NumRegions = 1  # Global LUT = 1, regional LUT = 64
	height = c_uint(slm_lib.Get_image_height(board_number))
	width = c_uint(slm_lib.Get_image_width(board_number))
	depth = c_uint(slm_lib.Get_image_depth(board_number)) #Bits per pixel
	Bytes = c_uint(depth.value//8)
	ImgSize = c_uint(height.value*width.value*Bytes.value)
	center_x = c_uint(width.value//2)
	center_y = c_uint(height.value//2)

	slm_lib.Load_LUT_file(board_number,  b"C:\\Users\\Demeter\\Desktop\\slm.lut")
	#slm_lib.Load_LUT_file(board_number,  b"C:\\Users\\10094\\Desktop\\slm4937_at813_75.lut")   
	#slm_lib.Load_LUT_file(board_number,  b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6674_at785_75C.LUT")
	#slm_lib.Load_LUT_file(board_number,  b"C:\\Users\\10094\\Desktop\\slm.lut")

	# slm_lib.Load_LUT_file(board_number, b"C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\1024x1024_linearVoltage.lut")

	image = numpy.zeros([width.value*height.value*Bytes.value], numpy.uint8, 'C')
#	filename ="255.bmp"   
#    # 打开图像并转换为灰度模式
#	x = Image.open(filename).convert("L")     
#	image = numpy.array(x)    

    # 将图像转换为NumPy数组
    
	WFC = numpy.zeros([width.value*height.value*Bytes.value], numpy.uint8, 'C')
    # Write a blank pattern to the SLM to get going
	# retVal = slm_lib.Write_image(board_number, image.ctypes.data_as(POINTER(c_ubyte)), height.value*width.value*Bytes.value, wait_For_Trigger, flip_immediate, OutputPulseImageFlip, OutputPulseImageRefresh, timeout_ms)
    
	retVal = slm_lib.Write_image(board_number, image.ctypes.data_as(POINTER(c_ubyte)), wait_For_Trigger, flip_immediate, OutputPulseImageFlip, timeout_ms)
	# slm_lib.ImageWriteComplete(board_number, 5000)
      
    
	if(retVal != 1):
		print ("DMA Failed")
		slm_lib.Delete_SDK()
	else:
		Reference=255
		Variable=255
		StepBy=-1
		PixelsPerStripe = 4


image_lib.Generate_Stripe(image.ctypes.data_as(POINTER(c_ubyte)), WFC.ctypes.data_as(POINTER(c_ubyte)), width.value, height.value, depth.value, Reference, Variable, PixelsPerStripe, 1, RGB.value)
retVal = slm_lib.Write_image(board_number, image.ctypes.data_as(POINTER(c_ubyte)), wait_For_Trigger, flip_immediate, OutputPulseImageFlip, 5000)
sleep(1)
#power_m=numpy.zeros_like(256, dtype=float)
#gray=[]


#image_lib.Generate_Stripe(image.ctypes.data_as(POINTER(c_ubyte)), WFC.ctypes.data_as(POINTER(c_ubyte)), width, height, depth, Reference, 255, PixelsPerStripe, 1, RGB)
#data_ptr = 	Image.ctypes.data_as(POINTER(c_ubyte))
#unsigned_char_ptr = cast(data_ptr, POINTER(c_ubyte))
#slm_lib.Write_image(board_number, unsigned_char_ptr, ImgSize, wait_For_Trigger, flip_immediate, OutputPulseImageFlip, OutputPulseImageRefresh, 5000)
#slm_lib.ImageWriteComplete(board_number, 5000)
power_m=list(numpy.zeros(256))
for DataPoint in range(NumDataPoints):
    print(f"Gray: {Variable}")
    
    image_lib.Generate_Stripe(image.ctypes.data_as(POINTER(c_ubyte)), WFC.ctypes.data_as(POINTER(c_ubyte)), width.value, height.value, depth.value, Reference, Variable, PixelsPerStripe, 1, RGB.value)
    #filename = "{:03d}.bmp".format(Variable)





    #x = Image.open(filename).convert("L")

    # 将图像转换为NumPy数组
    #image = numpy.array(x)
	
	#if Variable==255:
 #         print(Image)
        #sleep(5)
 #       print(sleep)
    #Image = numpy.full((1024, 1024), Variable)
    

	
    # Load the image to the SLM
    
    data_ptr = 	image.ctypes.data_as(POINTER(c_ubyte))
    unsigned_char_ptr = cast(data_ptr, POINTER(c_ubyte))
    retVal = slm_lib.Write_image(board_number, image.ctypes.data_as(POINTER(c_ubyte)), wait_For_Trigger, flip_immediate, OutputPulseImageFlip, 5000)
    slm_lib.ImageWriteComplete(board_number, 5000)

                # Give the LC time to settle into the image
    sleep(0.3)


    power =  c_double()
    tlPM.measPower(byref(power)) 

	
    print(power.value)   
    power_m[Variable]=power.value
    Variable += StepBy
    
	
    #gray.append(Variable)
    
slm_lib.Delete_SDK()
file_name = "Raw0_Calibrated.csv"

#file_name = "Raw0.csv"
with open(file_name, "w") as file:
    for i in range(NumDataPoints):
        line = "{}, {}\n".format(i, power_m[255-i])
        file.write(line)
        

		

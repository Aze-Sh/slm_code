from ctypes import *
import numpy as np
from time import sleep
import BlinkSLM
import time 
import csv
import matplotlib.pyplot as plt
# 创建 BlinkSLM 实例
blink_slm = BlinkSLM.BlinkSLM()

# 用于存储时间和温度数据的列表
data = []

try:
    # 记录起始时间
    start_time = time.time()
    
    while True:
        temperature = blink_slm.get_slm_temperature()
        current_time = time.time()  # 获取当前时间（以秒为单位）
        relative_time = current_time - start_time  # 计算相对于起始时间的相对时间
        data.append((relative_time, temperature))
        print(f"Current SLM temperature: {temperature:.2f}°C, Time: {relative_time:.2f}s")
        sleep(0.1)  # 每0.1秒读取一次温度
except KeyboardInterrupt:
    print("Temperature reading stopped.")

# 将数据保存到 CSV 文件
with open('temperature_data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Time (s)', 'Temperature (°C)'])  # 写入表头
    for row in data:
        csv_writer.writerow(row)

print("Data saved to temperature_data.csv")

# 绘制温度随时间变化的图表
times, temperatures = zip(*data)

plt.figure(figsize=(10, 6))
plt.plot(times, temperatures, marker='o', linestyle='-', color='b')
plt.title('SLM Temperature Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()
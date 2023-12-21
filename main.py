'''
pipenv shell
pip install pyinstaller
pip install PyWavelets
pip install matplotlib
pip install mplcursors
'''

import pywt
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import mplcursors

file_names = os.listdir('.\\')
for file_name in file_names:
    if file_name.endswith('.csv'):
        print(file_name)
        time=[]
        data=[]

        # 打开csv文件获得数据
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            head = next(reader)
            head2 = next(reader)
            for row in reader:
                if row:
                    time.append(float(row[0]))
                    data.append(float(row[1]))
        t = np.array(time)
        x = np.array(data)

        # 进行小波分解，选取 "db4" 小波基和 4 层分解
        coeffs = pywt.wavedec(x, "db4", level=4)

        # 对每个细节系数进行阈值处理，去除小于一定阈值的系数
        threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(x)))
        new_coeffs = [coeffs[0]]
        for i in range(1, len(coeffs)):
            new_coeffs.append(pywt.threshold(coeffs[i], threshold))

        # 重构信号
        denoised = pywt.waverec(new_coeffs, "db4")

        # 移动平均法进一步平滑（可选）
        #smoothed = np.convolve(denoised, np.ones(10)/10, mode="same")
        

        # 绘制原始和降噪后图形
        fig1, ax1 = plt.subplots()
        ax1.plot(t, x, label="Original")
        ax1.plot(t, denoised[:-1], label="Denoised")
        ax1.set_title(file_name)
        ax1.set_xlabel("time")
        ax1.set_ylabel("A.U.")
        ax1.legend()
        
        # 绘制降噪后图形，并使用 mplcursors 添加交互式功能
        fig2, ax2 = plt.subplots()
        ax2.plot(t, denoised[:-1], label="Denoised")
        ax2.set_title(file_name)
        ax2.set_xlabel("time")
        ax2.set_ylabel("A.U.")
        ax2.legend()

        cursor = mplcursors.cursor(ax2)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"({sel.target[0]:.2f}, {sel.target[1]:.2f})"))
        plt.show()

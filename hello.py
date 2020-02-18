# coding:utf-8
import matplotlib
# 使用 matplotlib中的FigureCanvas (在使用 Qt5 Backends中 FigureCanvas继承自QtWidgets.QWidget)
from PyQt5.QtCore import Qt, QBasicTimer
from PyQt5.QtGui import QFont, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QFileDialog, QLineEdit, QHBoxLayout, QSlider, \
    QVBoxLayout, QLabel, \
    QRadioButton, QProgressBar
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import joblib
import datetime
import threading
import time


# 特征向量统一维度，注意需与训练时采用的相同，这里为尽量通用，定为4000
min_length = 4000
turelabel = 1  # 目标数据的标签
floatp = 0.2  # 4个GB1峰的浮动差值范围


class Main_window(QDialog):
    # ===================变量定义函数==================
    # 定义所选文件名变量

    @property
    def classFilePath(self):
        return self.__FilePathName

    @classFilePath.setter
    def classFilePath(self, value):
        self.__FilePathName = value

    # 定义所选文件夹变量
    @property
    def classFileDirPath(self):
        return self.__FilePathDirName

    @classFileDirPath.setter
    def classFileDirPath(self, value):
        self.__FilePathDirName = value

    # 定义判断模式变量
    @property
    def judgeMode(self):
        return self._judgeMode  # 文件为0，文件夹为1

    @judgeMode.setter
    def judgeMode(self, value):
        self._judgeMode = value

    def setMode(self, value):
        self.judgeMode = value

    # 定义暂停变量
    @property
    def mystop(self):
        return self._mystop

    @mystop.setter
    def mystop(self, value):
        self._mystop = value


    # =============================主函数===============================
    def __init__(self):
        super().__init__()

        # 几个QWidgets
        self.setWindowTitle('单分子力学谱智能检测系统')  # 窗体名
        self.setWindowIcon(QIcon('Logo.jpg'))
        self.figure = plt.figure(facecolor='#FFD7C4')  # 可选参数,facecolor为背景颜色
        self.resize(800, 800)
        self.info=QLineEdit()
        self.info.setReadOnly(True)
        self.info.setText("文件/文件夹名")

        self.l1 = QLabel("下限")
        self.l1.setAlignment(Qt.AlignCenter)
        self.s1 = QSlider(Qt.Horizontal)
        self.s1.setMinimum(0)
        self.s1.setMaximum(4000)
        self.s1.setSingleStep(100)
        self.s1.setTickPosition(QSlider.TicksBelow)
        self.s1.setTickInterval(100)
        self.s1.setValue(2100)
        self.s1.valueChanged.connect(self.valuechange)

        self.l2 = QLabel("上限")
        self.l2.setAlignment(Qt.AlignCenter)
        self.s2 = QSlider(Qt.Horizontal)
        self.s2.setMinimum(0)
        self.s2.setMaximum(4000)
        self.s2.setSingleStep(100)
        self.s2.setTickPosition(QSlider.TicksBelow)
        self.s2.setTickInterval(100)
        self.s2.setValue(2600)
        self.s2.valueChanged.connect(self.valuechange)

        self.canvas = FigureCanvas(self.figure)
        self.button_choosefile = QPushButton("选择文件")
        self.button_judge = QLabel("文件输出：")
        self.text_judge = QLabel()
        self.button_choosefile2 = QPushButton("选择文件夹")
        self.button_judge2 = QLabel("文件夹输出：")
        self.text_judge2 = QLabel()
        self.method2 = QPushButton("SVM")
        self.method3 = QPushButton("决策树")
        self.button_pause=QPushButton("暂停/继续")
        # 进度条
        self.progressBar = QProgressBar(self)
        self.progressBar.setValue(0)
        self.timer = QBasicTimer()
        self.step = 0
        # 连接事件
        self.button_choosefile.clicked.connect(self.fileOpen_button)
        self.button_choosefile2.clicked.connect(self.fileDirOpen_button)
#       self.method1.clicked.connect(self.method1_click)
        self.method2.clicked.connect(self.method2_click_on)
        self.method3.clicked.connect(self.method3_click)
        self.button_pause.clicked.connect(self.mystop_click)
        # 设置布局
        layout = QVBoxLayout()

        layout.addWidget(self.canvas)
        layout.addWidget(self.info)
        layout.addWidget(self.l1)
        layout.addWidget(self.s1)
        layout.addWidget(self.l2)
        layout.addWidget(self.s2)
        layout.addWidget(self.button_choosefile)
        layout.addWidget(self.button_choosefile2)

        # 创建一个水平布局
        hbox = QHBoxLayout()
        hbox.addWidget(self.method2)
        hbox.addWidget(self.method3)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.button_judge)
        hbox1.addWidget(self.text_judge)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.button_judge2)
        hbox2.addWidget(self.text_judge2)
        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.progressBar)
        hbox3.addWidget(self.button_pause)

        layout.addLayout(hbox)
        layout.addLayout(hbox1)
        layout.addLayout(hbox2)
        layout.addLayout(hbox3)

        self.setLayout(layout)

        # 赋初值
        self.classFilePath = None
        self.classFileDirPath = None
        self.mystop=False
        self.setMode(0)
    # ================================按键触发函数======================


    # 文件打开按键
    def fileOpen_button(self):
        if self.classFilePath:
            filename, _ = QFileDialog.getOpenFileName(self, '选取文件', self.classFilePath)
        else:
            filename, _ = QFileDialog.getOpenFileName(self, '选取文件', './')
        if filename == "":
            return
        self.classFilePath = filename
        self.read_txt_for_onefile(filename, self.s1.value(), self.s2.value())
        self.info.clear()
        self.info.setText(filename)
        self.setMode(0)

    # 改变范围
    def valuechange(self):
        if self.classFilePath and self.s1.value()+5<self.s2.value():
            self.read_txt_for_onefile(self.classFilePath, self.s1.value(), self.s2.value())

    # 文件夹打开按键
    def fileDirOpen_button(self):
        if self.classFileDirPath:
            directory1 = QFileDialog.getExistingDirectory(self, "选择文件夹", self.classFileDirPath)
        else:
            directory1 = QFileDialog.getExistingDirectory(self, "选择文件夹", "./")
        if directory1 == "":
            return
        self.classFileDirPath = directory1
        self.info.clear()
        self.info.setText(directory1)
        self.setMode(1)

    # SVM按键
    def method2_click_on(self):
        t = threading.Thread(target=self.method2_click, name='t')
        t.setDaemon(True)
        t.start()

    def method2_click(self):
        # 判断文件内容

        curr_time3 = datetime.datetime.now()
        print(curr_time3)
        if self.judgeMode == 0:
            if self.classFilePath == None:
                self.text_judge.setText("判断失败，还未选择文件")
                return
            filename = self.classFilePath
            retract = self.read_txt(filename)
            judge_restlt = self.svm_method(retract, filename, '.\目标数据_svm')
            # judge_result=方法返回值
            if judge_restlt == 1:
                self.text_judge.setText("是目标数据")
            elif judge_restlt == 0:
                self.text_judge.setText("不是目标数据")
            else:
                self.text_judge.setText("判断出错")
        # 判断文件夹内容
        else:
            if self.classFileDirPath == None:
                self.text_judge2.setText("判断失败，还未选择文件夹")
                return
            count = 0
            for filename in os.listdir(self.classFileDirPath):
                if os.path.splitext(filename)[1] == '.txt':
                        count += 1
            print(count)
            if not os.path.exists(self.classFileDirPath + '\目标数据_svm'):  # 创建文件夹
                os.makedirs(self.classFileDirPath + '\目标数据_svm')
            self.mystop=False
            # 判断每个文件
            all_read_file_num = 0  # 当前读的文件数
            right_read_file_num = 0  # 目标文件数
            for filename in os.listdir(self.classFileDirPath):
                if os.path.splitext(filename)[1] == '.txt':
                    while self.mystop:
                        time.sleep(1)
                    all_read_file_num += 1
                    self.progressBar.setValue(int(all_read_file_num*100/count))
                    retract = self.read_txt(self.classFileDirPath + '\\' + filename)
                    judge_result = self.svm_method(retract, self.classFileDirPath + '\\' + filename,
                                                   self.classFileDirPath + '\目标数据_svm')
                    # judge_result=方法返回值
                    if judge_result == 1:
                        right_read_file_num += 1
#                    self.text_judge2.setText(
#                      "当前已检测" + str(all_read_file_num) + "个文件，其中" + str(right_read_file_num) + "个是目标文件")
            self.text_judge2.setText(
                "文件夹共有" + str(all_read_file_num) + "个文件，其中" + str(right_read_file_num) + "个是目标文件")
            curr_time4 = datetime.datetime.now()
            print(curr_time4)

    # 决策树按键
    def method3_click(self):
        # 判断文件内容
        curr_time5 = datetime.datetime.now()
        print(curr_time5)
        if self.judgeMode == 0:
            if self.classFilePath == None:
                self.text_judge.setText("判断失败，还未选择文件")
                return

            filename = self.classFilePath
            retract = self.read_txt(filename)
            judge_restlt = self.dtree_method(retract, filename, '.\目标数据_决策树')
            # judge_result=方法返回值
            if judge_restlt == 1:
                self.text_judge.setText("是目标数据")
            elif judge_restlt == 0:
                self.text_judge.setText("不是目标数据")
            else:
                self.text_judge.setText("判断出错")
        # 判断文件夹内容
        else:
            if self.classFileDirPath == None:
                self.text_judge2.setText("判断失败，还未选择文件夹")
                return

            if not os.path.exists(self.classFileDirPath + '\目标数据_决策树'):  # 创建文件夹
                os.makedirs(self.classFileDirPath + '\目标数据_决策树')

            # 判断每个文件
            all_read_file_num = 0  # 当前读的文件数
            right_read_file_num = 0  # 目标文件数
            for filename in os.listdir(self.classFileDirPath):
                if os.path.splitext(filename)[1] == '.txt':
                    all_read_file_num += 1
                    retract = self.read_txt(self.classFileDirPath + '\\' + filename)
                    judge_result = self.dtree_method(retract, self.classFileDirPath + '\\' + filename,
                                                     self.classFileDirPath + '\目标数据_决策树')
                    # judge_result=方法返回值
                    if judge_result == 1:
                        right_read_file_num += 1
            self.text_judge2.setText(
                "文件夹共有" + str(all_read_file_num) + "个文件，其中" + str(right_read_file_num) + "个是目标文件")
            curr_time5 = datetime.datetime.now()
            print(curr_time5)

    # =========================工具函数======================
    # 输入文件名画图
    def read_txt_for_onefile(self, filename, low_range, high_range):
        extend_x = []
        extend_y = []
        retract_x = []
        retract_y = []
        temp_x = []
        temp_y = []
        spring_constant=1
        second_flag = False
        with open(filename, 'r') as file_to_read:
            lines = file_to_read.readlines()  # 整行全部读完
            for line in lines:
                if line in ['\n', '\r\n']:  # 跳过空行
                    continue
                if line.strip() == "":  # 跳过只有空格的行
                    continue
                if 'springConstant' in line:
                    spring_constant = float(line.split()[2]) * 1000
                if 'segment: retract' in line:
                    second_flag = True
                temp = line.split()  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                if line.startswith('#'):  # 注释行
                    if len(temp) == 3 and temp[1] == 'segment:' and temp[2] == 'retract':  # 进入第二段
                        extend_x = [num * 1e9 for num in temp_x]
                        extend_y = [num * -1e12 for num in temp_y]
                        temp_x.clear()
                        temp_y.clear()
                    continue
                # 读入处理数据
                x, y = [float(i) for i in temp]
                temp_x.append(x)  # 添加新读取的数据
                temp_y.append(y)
            retract_x = [num * 1e9 for num in temp_x]
            retract_y = [num * -1e12 for num in temp_y]

        plt.clf() # 清屏
        plt.xlim([low_range, high_range])
        plt.plot(extend_x, extend_y, color='gray', label='extend')
        plt.plot(retract_x, retract_y, color='red', label='retract')
        plt.legend()
        plt.xlabel('extension(nm)')
        plt.ylabel('force(pN)')
        self.canvas.draw()

    # 读取文件
    def read_txt(self, filename):  # 读txt文件，画图
        extend_x = []
        extend_y = []
        retract_x = []
        retract_y = []
        temp_x = []
        temp_y = []
        with open(filename, 'r') as file_to_read:
            lines = file_to_read.readlines()  # 整行全部读完
            for line in lines:
                if line in ['\n', '\r\n']:  # 跳过空行
                    continue
                if line.strip() == "":  # 跳过只有空格的行
                    continue
                temp = line.split()  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                if line.startswith('#'):  # 注释行
                    if len(temp) == 3 and temp[1] == 'segment:' and temp[2] == 'retract':  # 进入第二段
                        extend_x = [num * 1e9 for num in temp_x]
                        extend_y = [num * -1e12 for num in temp_y]
                        temp_x.clear()
                        temp_y.clear()
                    continue
                # 读入处理数据
                x, y = [float(i) for i in temp]
                temp_x.append(x)  # 添加新读取的数据
                temp_y.append(y)
            retract_x = [num * 1e9 for num in temp_x]
            retract_y = [num * -1e12 for num in temp_y]

        return list(zip(retract_x, retract_y))

    # =====================判断方法==========================
    # 方法1：SVM
    def svm_method(self, retract, sourcefile, target):
        retract.sort(key=lambda x: x[0], reverse=True)  # 二维点按x降序排序
        data = [x[1] for x in retract]  # 排好序后的y值
        vecdata = []
        length = len(data)
        if length >= min_length:  # 长的切分
            vecdata.append(data[len(data) - min_length:])
        else:  # 短的填充
            number = (200 if (200 < length / 2) else int(length / 2))
            fill = np.mean(data[:number])  # 填充值，拿最后200个的平均值填充，也就是基线部分
            while len(data) < min_length:
                data.insert(0, fill)
            vecdata.append(data)
        # load PCA模型，降维
        pca = joblib.load('pca.pkl')
        xdata = pca.transform(vecdata)
        # load SVM模型
        model = joblib.load('model_svm.pkl')
        ypred = model.predict(xdata)
        # 复制目标数据txt文件
        if ypred[0] == turelabel:
            shutil.copy(sourcefile, target)  # 复制txt文件
            print(sourcefile)
            return 1
        else:
            print(sourcefile)
            return 0

    # 方法2：决策树
    def dtree_method(self, retract, sourcefile, target):
        retract.sort(key=lambda x: x[0], reverse=True)  # 二维点按x降序排序
        data = [x[1] for x in retract]  # 排好序后的y值
        # 统一特征向量维度，且变为合适类型
        vecdata = []
        length = len(data)
        if length >= min_length:  # 长的切分
            vecdata.append(data[len(data) - min_length:])
        else:  # 短的填充
            number = (200 if (200 < length / 2) else int(length / 2))
            fill = np.mean(data[:number])  # 填充值，拿最后200个的平均值填充，也就是基线部分
            while len(data) < min_length:
                data.insert(0, fill)
            vecdata.append(data)
        # load PCA模型，降维
        pca = joblib.load('pca.pkl')
        xdata = pca.transform(vecdata)
        # load 决策树模型
        model = joblib.load('model_dtree.pkl')
        ypred = model.predict(xdata)
        # 复制目标数据txt文件
        if ypred[0] == turelabel:
            shutil.copy(sourcefile, target)  # 复制txt文件
            return 1
        else:
            return 0

   # 暂停按键

    def mystop_click(self):
        if self.mystop==False:
            self.mystop=True
            print("哈哈")
        else:
            self.mystop=False
            print("呵呵")

# 运行程序
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = Main_window()
    main_window.show()
    app.exec()

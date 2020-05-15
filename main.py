# coding:utf-8
import random
import numpy as np
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QPushButton, QFileDialog, QLineEdit, QHBoxLayout, QSlider, \
    QVBoxLayout, QLabel, \
    QRadioButton, QProgressBar, QComboBox
import matplotlib.pyplot as plt
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
    # 进度条信号
    download_proess_signal = pyqtSignal(int)
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

    @property
    def shapeMode(self):
        return self._shapeMode  # 四峰为0，八峰为1

    # 定义暂停变量
    @property
    def mystop(self):
        return self._mystop

    @property
    def on_thread(self):
        return self._on_thread  # 无进行为0，有进行为1

    @judgeMode.setter
    def judgeMode(self, value):
        self._judgeMode = value

    def setMode(self, value):
        self.judgeMode = value

    @shapeMode.setter
    def shapeMode(self, value):
        self._shapeMode = value

    def setMode_2(self, value):
        self.shapeMode = value

    @on_thread.setter
    def on_thread(self, value):
        self._on_thread = value

    def setMode_3(self, value):
        self.on_thread = value
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
        self.info = QLineEdit()
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
        self.cb = QComboBox()
        self.cb.addItem("SVM")  # 添加一个项目
        self.cb.addItem("决策树")
        self.cb.addItem("随机森林")
        self.cb.addItem("SVM+决策树+随机森林联合判断")
        self.cb.currentIndexChanged.connect(self.choose_clf)
        self.button_begin = QPushButton("开始分类")
        self.button_stop = QPushButton("停止")
        self.button_pause = QPushButton("暂停/继续")
        self.rbtn1 = QRadioButton('四峰', self)
        self.rbtn1.toggled.connect(lambda: self.setMode_2(0))
        self.rbtn2 = QRadioButton('八峰', self)
        self.rbtn2.toggled.connect(lambda: self.setMode_2(1))
        self.rbtn1.setChecked(1)  # 设定初值
        # 进度条
        self.progressBar = QProgressBar(self)
        self.progressBar.setValue(0)
        self.timer = QBasicTimer()


        # 连接事件
        self.button_choosefile.clicked.connect(self.fileOpen_button)
        self.button_choosefile2.clicked.connect(self.fileDirOpen_button)
        self.button_begin.clicked.connect(self.mystart)
        #       self.method1.clicked.connect(self.method1_click)
        #self.method2.clicked.connect(self.method2_click_on)
        #self.method3.clicked.connect(self.method3_click_on)
        self.button_stop.clicked.connect(self.mystop_click)
        self.button_pause.clicked.connect(self.mypause_click)
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
        #hbox.addWidget(self.method2)
        #hbox.addWidget(self.method3)
        hbox.addWidget(self.cb)
        hbox.addWidget(self.button_begin)
        hbox.addWidget(self.button_stop)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.rbtn1)
        hbox1.addWidget(self.rbtn2)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.button_judge)
        hbox2.addWidget(self.text_judge)
        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.button_judge2)
        hbox3.addWidget(self.text_judge2)
        hbox4 = QHBoxLayout()
        hbox4.addWidget(self.progressBar)
        hbox4.addWidget(self.button_pause)

        layout.addLayout(hbox)
        layout.addLayout(hbox1)
        layout.addLayout(hbox2)
        layout.addLayout(hbox3)
        layout.addLayout(hbox4)

        self.setLayout(layout)

        # 赋初值
        self.classFilePath = None
        self.classFileDirPath = None
        self.mystop = 0
        self.setMode(0)  # 判断文件/文件夹
        self.setMode_2(0)  # 判断4峰/8峰
        self.setMode_3(0)  # 判断当前是否有进程
        self.download_proess_signal.connect(self.set_progressbar_value)
        self.judgeMethod=1
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

    # 暂停/继续按键
    def mypause_click(self):
        if self.mystop == 0:
            self.mystop = 1
        elif self.mystop==1:
            self.mystop = 0

    # 停止按键
    def mystop_click(self):
        self.mystop=2

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

    # 开始分类
    def mystart(self):
        if self.on_thread:
            return
        if self.judgeMethod==1:
            print(1)
            t = threading.Thread(target=self.method1_click, name='t')
            t.setDaemon(True)
            t.start()
        if self.judgeMethod==2:
            print(2)
            t = threading.Thread(target=self.method2_click, name='t')
            t.setDaemon(True)
            t.start()
        if self.judgeMethod==3:
            print(3)
            t = threading.Thread(target=self.method3_click, name='t')
            t.setDaemon(True)
            t.start()
        if self.judgeMethod==4:
            print(3)
            t = threading.Thread(target=self.test_click, name='t')
            t.setDaemon(True)
            t.start()

    # SVM
    def method1_click(self):
        # 判断文件内容
        self.setMode_3(1)
        curr_time3 = datetime.datetime.now()
        print(curr_time3)
        if self.judgeMode == 0:
            if self.classFilePath == None:
                self.text_judge.setText("判断失败，还未选择文件")
                return
            filename = self.classFilePath
            retract = self.read_txt(filename)
            if retract !=1:
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
            self.judgeMode=0
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
            self.mystop = 0
            # 判断每个文件
            all_read_file_num = 0  # 当前读的文件数
            right_read_file_num = 0  # 目标文件数
            wrong_file_num = 0  # 非法数据文件数
            #self.timer.start(100,self)

            for filename in os.listdir(self.classFileDirPath):
                while self.mystop==1:
                    time.sleep(1)
                if self.mystop==2:
                    self.setMode_3(0)
                    self.mystop=0
                    self.judgeMode = 1
                    return
                if os.path.splitext(filename)[1] == '.txt':
                    all_read_file_num += 1
                    retract = self.read_txt(self.classFileDirPath + '\\' + filename)
                    self.download_proess_signal.emit(int(all_read_file_num * 100 / count))

                    if retract == 1:
                        if not os.path.exists(self.classFileDirPath + '\非法数据'):  # 创建文件夹
                            os.makedirs(self.classFileDirPath + '\非法数据')
                        shutil.copy(self.classFileDirPath + '\\' + filename, self.classFileDirPath + '\非法数据')
                        judge_result = 2
                    else:
                        judge_result = self.svm_method(retract, self.classFileDirPath + '\\' + filename,
                                                       self.classFileDirPath + '\目标数据_svm')
                    # judge_result=方法返回值
                    if judge_result == 1:
                        right_read_file_num += 1
                    if judge_result == 2:
                        wrong_file_num += 1
                    self.text_judge2.setText("当前已检测" + str(all_read_file_num) + "个文件，其中" + str(right_read_file_num) + "个是目标文件"+ str(
                    wrong_file_num) + "个非法文件")
            self.text_judge2.setText(
                "文件夹共有" + str(all_read_file_num) + "个文件，其中" + str(right_read_file_num) + "个是目标文件," + str(
                    wrong_file_num) + "个非法文件")
            self.judgeMode=1
        curr_time4 = datetime.datetime.now()
        print(curr_time4)
        self.setMode_3(0)

    # 决策树
    def method2_click(self):
        # 判断文件内容
        self.setMode_3(1)
        curr_time5 = datetime.datetime.now()
        print(curr_time5)
        if self.judgeMode == 0:
            if self.classFilePath == None:
                self.text_judge.setText("判断失败，还未选择文件")
                return
            filename = self.classFilePath
            retract = self.read_txt(filename)
            if retract !=1:
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
            self.judgeMode=0
            if self.classFileDirPath == None:
                self.text_judge2.setText("判断失败，还未选择文件夹")
                return
            count = 0
            for filename in os.listdir(self.classFileDirPath):
                if os.path.splitext(filename)[1] == '.txt':
                    count += 1
            print(count)
            if not os.path.exists(self.classFileDirPath + '\目标数据_决策树'):  # 创建文件夹
                os.makedirs(self.classFileDirPath + '\目标数据_决策树')
            self.mystop = 0
            # 判断每个文件
            all_read_file_num = 0  # 当前读的文件数
            right_read_file_num = 0  # 目标文件数
            wrong_file_num = 0  # 非法文件数
            for filename in os.listdir(self.classFileDirPath):
                while self.mystop==1:
                    time.sleep(1)
                if self.mystop == 2:
                    self.setMode_3(0)
                    self.mystop = 0
                    self.judgeMode = 1
                    return
                if os.path.splitext(filename)[1] == '.txt':
                    all_read_file_num += 1
                    self.download_proess_signal.emit(int(all_read_file_num * 100 / count))
                    retract = self.read_txt(self.classFileDirPath + '\\' + filename)
                    if retract == 1:
                        if not os.path.exists(self.classFileDirPath + '\非法数据'):  # 创建文件夹
                            os.makedirs(self.classFileDirPath + '\非法数据')
                        shutil.copy(self.classFileDirPath + '\\' + filename, self.classFileDirPath+'\非法数据')
                        judge_result = 2
                    else:
                        judge_result = self.dtree_method(retract, self.classFileDirPath + '\\' + filename,
                                                         self.classFileDirPath + '\目标数据_决策树')
                        # judge_result=方法返回值
                    if judge_result == 1:
                        right_read_file_num += 1
                    if judge_result == 2:
                        wrong_file_num += 1
                    self.text_judge2.setText(
                        "当前已检测" + str(all_read_file_num) + "个文件，其中" + str(right_read_file_num) + "个是目标文件" + str(
                            wrong_file_num) + "个非法文件")
            self.text_judge2.setText(
                "文件夹共有" + str(all_read_file_num) + "个文件，其中" + str(right_read_file_num) + "个是目标文件," + str(
                    wrong_file_num) + "个非法文件")
            self.judgeMode=1
        curr_time5 = datetime.datetime.now()
        print(curr_time5)
        self.setMode_3(0)

    #随机森林
    def method3_click(self):
        # 判断文件内容
        self.setMode_3(1)
        curr_time5 = datetime.datetime.now()
        print(curr_time5)
        if self.judgeMode == 0:
            if self.classFilePath == None:
                self.text_judge.setText("判断失败，还未选择文件")
                return
            filename = self.classFilePath
            retract = self.read_txt(filename)
            if retract !=1:
                judge_restlt = self.forest_method(retract, filename, '.\目标数据_决策树')
            # judge_result=方法返回值
                if judge_restlt == 1:
                    self.text_judge.setText("是目标数据")
                elif judge_restlt == 0:
                    self.text_judge.setText("不是目标数据")
                else:
                    self.text_judge.setText("判断出错")
        # 判断文件夹内容
        else:
            self.judgeMode=0
            if self.classFileDirPath == None:
                self.text_judge2.setText("判断失败，还未选择文件夹")
                return
            count = 0
            for filename in os.listdir(self.classFileDirPath):
                if os.path.splitext(filename)[1] == '.txt':
                    count += 1
            print(count)
            if not os.path.exists(self.classFileDirPath + '\目标数据_随机森林'):  # 创建文件夹
                os.makedirs(self.classFileDirPath + '\目标数据_随机森林')
            self.mystop = 0
            # 判断每个文件
            all_read_file_num = 0  # 当前读的文件数
            right_read_file_num = 0  # 目标文件数
            wrong_file_num = 0  # 非法文件数
            for filename in os.listdir(self.classFileDirPath):
                while self.mystop==1:
                    time.sleep(1)
                if self.mystop == 2:
                    self.setMode_3(0)
                    self.mystop = 0
                    self.judgeMode = 1
                    return
                if os.path.splitext(filename)[1] == '.txt':
                    all_read_file_num += 1
                    self.download_proess_signal.emit(int(all_read_file_num * 100 / count))
                    retract = self.read_txt(self.classFileDirPath + '\\' + filename)
                    if retract == 1:
                        if not os.path.exists(self.classFileDirPath + '\非法数据'):  # 创建文件夹
                            os.makedirs(self.classFileDirPath + '\非法数据')
                        shutil.copy(self.classFileDirPath + '\\' + filename, self.classFileDirPath+'\非法数据')
                        judge_result = 2
                    else:
                        judge_result = self.forest_method(retract, self.classFileDirPath + '\\' + filename,
                                                         self.classFileDirPath + '\目标数据_随机森林')
                        # judge_result=方法返回值
                    if judge_result == 1:
                        right_read_file_num += 1
                    if judge_result == 2:
                        wrong_file_num += 1
                    self.text_judge2.setText(
                        "当前已检测" + str(all_read_file_num) + "个文件，其中" + str(right_read_file_num) + "个是目标文件" + str(
                            wrong_file_num) + "个非法文件")
            self.text_judge2.setText(
                "文件夹共有" + str(all_read_file_num) + "个文件，其中" + str(right_read_file_num) + "个是目标文件," + str(
                    wrong_file_num) + "个非法文件")
            self.judgeMode=1
        curr_time5 = datetime.datetime.now()
        print(curr_time5)
        self.setMode_3(0)

    def test_click(self):
        # 判断文件内容
        self.setMode_3(1)
        curr_time5 = datetime.datetime.now()
        print(curr_time5)
        if self.judgeMode == 0:
            if self.classFilePath == None:
                self.text_judge.setText("判断失败，还未选择文件")
                return
            filename = self.classFilePath
            retract = self.read_txt(filename)
            if retract !=1:
                judge_restlt = self.co_method(retract, filename, '.\目标数据_决策树')
            # judge_result=方法返回值
                if judge_restlt == 1:
                    self.text_judge.setText("是目标数据")
                elif judge_restlt == 0:
                    self.text_judge.setText("不是目标数据")
                else:
                    self.text_judge.setText("判断出错")
        # 判断文件夹内容
        else:
            self.judgeMode=0
            if self.classFileDirPath == None:
                self.text_judge2.setText("判断失败，还未选择文件夹")
                return
            count = 0
            for filename in os.listdir(self.classFileDirPath):
                if os.path.splitext(filename)[1] == '.txt':
                    count += 1
            print(count)
            if not os.path.exists(self.classFileDirPath + '\目标数据_test'):  # 创建文件夹
                os.makedirs(self.classFileDirPath + '\目标数据_test')
            self.mystop = 0
            # 判断每个文件
            all_read_file_num = 0  # 当前读的文件数
            right_read_file_num = 0  # 目标文件数
            wrong_file_num = 0  # 非法文件数
            for filename in os.listdir(self.classFileDirPath):
                while self.mystop==1:
                    time.sleep(1)
                if self.mystop == 2:
                    self.setMode_3(0)
                    self.mystop = 0
                    self.judgeMode = 1
                    return
                if os.path.splitext(filename)[1] == '.txt':
                    all_read_file_num += 1
                    self.download_proess_signal.emit(int(all_read_file_num * 100 / count))
                    retract = self.read_txt(self.classFileDirPath + '\\' + filename)
                    if retract == 1:
                        if not os.path.exists(self.classFileDirPath + '\非法数据'):  # 创建文件夹
                            os.makedirs(self.classFileDirPath + '\非法数据')
                        shutil.copy(self.classFileDirPath + '\\' + filename, self.classFileDirPath+'\非法数据')
                        judge_result = 2
                    else:
                        judge_result = self.co_method(retract, self.classFileDirPath + '\\' + filename,
                                                         self.classFileDirPath + '\目标数据_test')
                        # judge_result=方法返回值
                    if judge_result == 1:
                        right_read_file_num += 1
                    if judge_result == 2:
                        wrong_file_num += 1
                    self.text_judge2.setText(
                        "当前已检测" + str(all_read_file_num) + "个文件，其中" + str(right_read_file_num) + "个是目标文件" + str(
                            wrong_file_num) + "个非法文件")
            self.text_judge2.setText(
                "文件夹共有" + str(all_read_file_num) + "个文件，其中" + str(right_read_file_num) + "个是目标文件," + str(
                    wrong_file_num) + "个非法文件")
            self.judgeMode=1
        curr_time5 = datetime.datetime.now()
        print(curr_time5)
        self.setMode_3(0)

    # =========================工具函数======================
    # 输入文件名画图
    def read_txt_for_onefile(self, filename, low_range, high_range):
        extend_x = []
        extend_y = []
        retract_x = []
        retract_y = []
        temp_x = []
        temp_y = []
        spring_constant = 1
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
                if len(temp) != 2:
                    self.text_judge.setText("非法数据！")
                    return
                # 读入处理数据
                x, y = [float(i) for i in temp]
                temp_x.append(x)  # 添加新读取的数据
                temp_y.append(y)
            retract_x = [num * 1e9 for num in temp_x]
            retract_y = [num * -1e12 for num in temp_y]
            print(len(retract_x))
        plt.clf()  # 清屏
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
                if len(temp) != 2:
                    return 1  # 返回1表示为非法数据
                x, y = [float(i) for i in temp]
                temp_x.append(x)  # 添加新读取的数据
                temp_y.append(y)
            min_x = min(temp_x)
            temp_x = [num - min_x for num in temp_x]
            retract_x = [num * 1e9 for num in temp_x]
            retract_y = [num * -1e12 for num in temp_y]

        return list(zip(retract_x, retract_y))

    # 选择分类器
    def choose_clf(self,tag):
        if tag==0:
            self.judgeMethod=1
        if tag==1:
            self.judgeMethod=2
        if tag==2:
            self.judgeMethod=3
        if tag==3:
            self.judgeMethod=4

    # 进度条调整
    def set_progressbar_value(self,value):
        self.progressBar.setValue(value)

    # 改变画面范围
    def valuechange(self):
        if self.classFilePath and self.s1.value() + 5 < self.s2.value():
            self.read_txt_for_onefile(self.classFilePath, self.s1.value(), self.s2.value())

    # =====================判断方法==========================
    # 数据处理
    def deal(self,retract):
        retract.sort(key=lambda x: x[0], reverse=True)  # 二维点按x降序排序
        data_length = len(retract)
        ordery = []
        data = [x[1] for x in retract]
        for i in range(min_length):
            ordery.append(data[int(i * data_length / min_length)])
        y_min = np.min(ordery)
        y_max = np.max(ordery)
        ordery = [(y - y_min) / (y_max - y_min) for y in ordery]
        vecdata = []
        vecdata.append(ordery)
        return vecdata

    # 方法1：SVM
    def svm_method(self, retract, sourcefile, target):
        vecdata = self.deal(retract)
        # load PCA模型，降维
        if self.shapeMode==0:
            pca = joblib.load('pca_4.pkl')
            xdata = pca.transform(vecdata)
        # load SVM模型
            model = joblib.load('model_svm_4.pkl')
            ypred = model.predict(xdata)
        else:
            pca = joblib.load('pca_8.pkl')
            xdata = pca.transform(vecdata)
            # load SVM模型
            model = joblib.load('model_svm_8.pkl')
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
        vecdata = self.deal(retract)
        if self.shapeMode==0:
            # load PCA模型，降维
            pca = joblib.load('pca_4.pkl')
            xdata = pca.transform(vecdata)
        # load 决策树模型
            model = joblib.load('model_dtree_4.pkl')
            ypred = model.predict(xdata)
        else:
            pca = joblib.load('pca_8.pkl')
            xdata = pca.transform(vecdata)
            # load 决策树模型
            model = joblib.load('model_dtree_8.pkl')
            ypred = model.predict(xdata)
        # 复制目标数据txt文件
        if ypred[0] == turelabel:
            shutil.copy(sourcefile, target)  # 复制txt文件
            print(sourcefile)
            return 1
        else:
            print(sourcefile)
            return 0

    # 方法3：随机森林
    def forest_method(self, retract, sourcefile, target):
        vecdata= self.deal(retract)
        #load PCA模型，降维
        if self.shapeMode==0:
            pca = joblib.load('pca_4.pkl')
            xdata = pca.transform(vecdata)
        # load 决策树模型
            model = joblib.load('model_forest_4.pkl')
            ypred = model.predict(xdata)
        else:
            pca = joblib.load('pca_8.pkl')
            xdata = pca.transform(vecdata)
            # load 决策树模型
            model = joblib.load('model_forest_8.pkl')
            ypred = model.predict(xdata)
        # 复制目标数据txt文件
        if ypred[0] == turelabel:
            shutil.copy(sourcefile, target)  # 复制txt文件
            print(sourcefile)
            return 1
        else:
            print(sourcefile)
            return 0

    # 联合方法
    def co_method(self, retract, sourcefile, target):
        vecdata = self.deal(retract)
        if self.shapeMode == 0:
            num_of_peak = str(4)
        else:
            num_of_peak = str(8)
        pca = joblib.load('pca_'+num_of_peak+'.pkl')
        xdata = pca.transform(vecdata)
        model1 = joblib.load('model_dtree_'+num_of_peak+'.pkl')
        ypred1 = model1.predict(xdata)
        model2 = joblib.load('model_svm_'+num_of_peak+'.pkl')
        ypred2 = model2.predict(xdata)
        model3 = joblib.load('model_forest_'+num_of_peak+'.pkl')
        ypred3 = model3.predict(xdata)
        if ypred1[0] + ypred2[0] + ypred3[0] == 3:
            shutil.copy(sourcefile, target)  # 复制txt文件
            print(sourcefile)
            return 1
        else:
            print(sourcefile)
            return 0
# 运行程序
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = Main_window()
    main_window.show()
    app.exec()

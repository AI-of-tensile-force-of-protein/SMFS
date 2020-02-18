import os
import time
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn
# from sklearn.externals import joblib
import joblib
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import tree

components=500      #特征的维数
#y数据即特征向量的最小维数，统一维数时用，由于txt点的数量不一致，取最小值切割维数，为了尽量通用，目前取4000
min_length = 4000

def read_train(filedir,label):     #读指定路径下正例或反例数据，返回数据和标签，指定路径下均为同一标签数据
    x_train = []
    y_train = []
    # global min_length
    for filename in os.listdir(filedir):
        if os.path.splitext(filename)[1] == '.txt':
            temp = read_txt(filedir+'\\'+filename)
            x_train.append(temp)
            y_train.append(label)
            # if(min_length > len(temp)):
            #     min_length=len(temp)
    return x_train,y_train

def read_txt(filename):         #读txt文件，传回拉伸段y数据，排好序的
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
                    temp_x.clear()
                    temp_y.clear()
                continue
            # 读入处理数据
            x, y = [float(i) for i in temp]
            temp_x.append(x)  # 添加新读取的数据
            temp_y.append(y)
        retract_x = [num * 1e9 for num in temp_x]
        retract_y = [num * -1e12 for num in temp_y]

    retract = list(zip(retract_x,retract_y))
    retract.sort(key=lambda x:x[0], reverse=True)       #二维点按x降序排序
    ordery = [x[1] for x in retract]        #排好序后的y值
    return ordery
#SVM训练
def svm_train(train_data, train_label, test_data, test_label):
    clf=SVC(kernel='linear', verbose=True, gamma='auto')    #建模
    start=time.time()
    clf.fit(train_data,train_label)         #训练
    end=time.time()
    print('svm训练',end-start)
    joblib.dump(clf, 'model_svm.pkl')       #保存模型
    test_scores=clf.score(test_data,test_label)     #测试集得分
    print("得分为",test_scores)
    start=time.time()
    y_pred=clf.predict(test_data)           #输出测试集信息，准确率、召回率等
    end=time.time()
    print('svm测试',end-start)
    print(classification_report(test_label, y_pred))
#决策树训练
def decision_tree(train_data, train_label, test_data, test_label):
    # clf=DecisionTreeClassifier(max_depth=10, min_samples_split=2,random_state=0)
    clf = DecisionTreeClassifier()      #建模
    start=time.time()
    clf.fit(train_data,train_label)     #训练
    end=time.time()
    print('决策树训练',end-start)
    joblib.dump(clf,'model_dtree.pkl')      #保存模型
    test_scores=clf.score(test_data,test_label)     #测试集得分
    print("得分为",test_scores)
    start=time.time()
    y_pred=clf.predict(test_data)       #输出测试集信息，准确率、召回率等
    end=time.time()
    print('决策树训练',end-start)
    print(classification_report(test_label, y_pred))

#训练获得模型
targetfiledir='190315'      #目标数据文件夹
nottargetfiledir='190315原始'     #非目标数据文件夹
x_ture,y_ture=read_train(targetfiledir,1)       #读入目标数据和标签
x_flase,y_flase=read_train(nottargetfiledir,0)  #读入非目标数据和标签
x_train=x_ture+x_flase  #整体训练数据和标签
y_label=y_ture+y_flase
#统一特征向量维度，切分每个文件数据点，留最后4000个
#这里就存在一个统一维度的问题，目前的解决方法比较粗暴，若有更好可替换
for i in range(len(x_train)):
    x_train[i]=x_train[i][len(x_train[i])-min_length:]
#有个向量标准化的问题，尝试了一下效果不好，也许是使用的不对
# nortrain=preprocessing.scale(x_train)
#随机划分训练集与测试集
train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x_train,y_label, train_size=0.7, test_size=0.3)
# PCA对特征降维
pca = PCA(n_components=components)  # 建立pca模型
ptrain_data = pca.fit_transform(train_data)  # 训练并降维
ptest_data=pca.transform(test_data)
joblib.dump(pca,'pca.pkl')      #保存模型

svm_train(ptrain_data,train_label,ptest_data,test_label)          #训练svm模型
decision_tree(ptrain_data,train_label,ptest_data,test_label)        #训练决策树模型

#决策树模型导出文件，使用graphviz工具可以可视化
dtree=joblib.load('model_dtree.pkl')
with open("tree.dot", 'w') as f:
    tree.export_graphviz(dtree,out_file=f)
# graphviz.source()     #需要graphviz库才可画图

# xdata,ylabel=read_train('190224',1)
# for i in range(len(xdata)):
#     xdata[i]=xdata[i][len(xdata[i])-4375:]
# pca=joblib.load('pca.pkl')
# xd=pca.transform(xdata)
# dtree=joblib.load('model_dtree.pkl')
# ypred=dtree.predict(xd)
# print(classification_report(ylabel, ypred))

# xdata = read_txt('force-save-2019.02.24-20.41.40.486.txt')
# xtestdata = []
# pca=joblib.load('pca.pkl')
# xdata=xdata[len(xdata)-4375:]
# xtestdata.append(xdata)
# xdata=read_txt('force-save-2019.02.24-21.22.29.426.txt')
# xdata=xdata[len(xdata)-4375:]
# xtestdata.append(xdata)
# xd=pca.transform(xtestdata)
# dtree=joblib.load('model_dtree.pkl')
# print(dtree.predict(xd))

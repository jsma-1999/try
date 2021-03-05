# -*- coding: utf-8 -*-
_author_ = '马进山'
# import sys
'''
本次作业的资料是从中央气象局网站下载的真实观测资料，大家必须利用 linear regression 或其他方法预测 PM2.5 的数值。
观测记录被分成 train set 跟 test set，前者是每个月的前 20 天所有资料；后者则是从剩下的资料中随机取样出来的。
train.csv: 每个月前 20 天的完整资料。
test.csv: 从剩下的 10 天资料中取出 240 笔资料。每一笔资料都有连续 9 小时的观测数据，儿童学必须以此预测第十小时的 PM2.5。
'''

import pandas as pd
import numpy as np
import math
import csv

data = pd.read_csv('./train.csv', encoding='big5')  # 从train.csv载入数据 20*18*12=4320
data = data.iloc[:, 3:]  # 截取正式数据部分    data.to_csv('data_df.csv')
data[data == 'NR'] = 0  # 取需要的数值部分，将 'RAINFALL' 栏位全部替换为0。
raw_data = data.to_numpy()  # raw_data.shape  → (4320, 24)   已转换为可操作的数组数据

# Extract Features (1) #12*20=240天
month_data = {}  # 字典，有12个月，所有键值有12个  month_data[0].shape → (18, 480)   20*24=480
for month in range(12):
    sample = np.empty([18, 480])  # 20*24=480， sample.shape=(18, 480)  raw_data.shape=(4320, 24)
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# month_data = { 0: array([...],1: array([...],2: array([...], ... ,12: array([...])
'''
原来：
一天的数据：18*24
一月的数据：(18*20)*24
提取后：
一月的数据：18*(20*24)  装入字典中去
month_data[0].shape  =  (18, 480)
'''

# Extract Features (2)
x = np.empty([12 * 471, 18 * 9], dtype=float)  # 把数据平铺了，一个特征数据，就一行，一个月的就有471行特征数据
y = np.empty([12 * 471, 1], dtype=float)  #
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:  # 23-9=14
                continue  # 后面不执行，继续下一轮循环
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1,
                                                                                                                     -1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value

# month_data[0].shape  =  (18, 480)    →   (471, 18 * 9) 这个数据有重叠部分

# x.shape = (12 * 471, 18 * 9)
# y.shape = (12 * 471, 1)


# Normalize(1) （标准化）即x = (x - mean(x)) / std(x)
mean_x = np.mean(x, axis=0)  # axis=0表示按列求平均值，shape=(1，18*9)
std_x = np.std(x, axis=0)  # 按列求方差，shape=(1，18*9)
# std(x,0)等价于std(x),计算标准差的时候除的是根号N-1，这个是无偏的。

# 方差：每个样本值 与 全体样本值的平均数 之差的平方值 的平均数;

# 标准差：总体各单位标准值与其平均数离差平方的算术平均数的平方根;

# 标准差 s =方差的算术平方根


for i in range(len(x)):  # 12 * 471 = 5652 每行走起
    for j in range(len(x[0])):  # 18 * 9  #遍历每个数据 每行中的每列
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]  # （标准化）即x = (x - mean(x)) / std(x)
# 最常见的标准化方法就是Z标准化   标准差标准化
# 标准化后的变量值围绕0上下波动，大于0说明高于平均水平，小于0说明低于平均水平。


# Split Training Data Into "train_set" and "validation_set"
# 将训练集再分出训练集和测试集，前80%的数据为训练集，后20%的数据为测试集，math.floor意为向下取整

# len(x)=5652  len(x) * 0.8=4521.6    math.floor(len(x) * 0.8)=4521,向下取整
x_train_set = x[: math.floor(len(x) * 0.8), :]  # x_train_set.shape=(4521, 18 * 9)
x_validation = x[math.floor(len(x) * 0.8):, :]  # x_validation.shape=(1131, 18 * 9)

y_train_set = y[: math.floor(len(y) * 0.8), :]  # y_train_set.shape = (4521, 1)
y_validation = y[math.floor(len(y) * 0.8):, :]  # y_validation.shape =(1131, 1)

dim = 18 * 9 + 1  # +1表示加入偏置（b）
w = np.zeros([dim, 1])  # 权重和偏置,w.shape=(18*9+1, 1)
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)  # concatenate：连在一起
# 在x中的第一列前插入一列 全为1，相当于偏置（b）的系数, x.shape=(471*12, 18*9+1)

learning_rate = 100  # learning rate
iter_time = 1000  # 训练次数
adagrad = np.zeros([dim, 1])  # ada梯度矩阵，shape和w的shape一样  adagrad.shape=(18*9+1, 1)
eps = 0.0000000001  # 10的-6次方，防止出现除零错误

for t in range(iter_time):  ##y.shape = (12 * 471, 1)   x.shape=(471*12, 18*9+1)
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)  # rmse
    if (t % 100 == 0):
        print(str(t) + ":" + str(loss))  # 打印出来了

    if (t > (iter_time - 10)):
        print(str(t) + ":" + str(loss))  # 打印出来了
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1  # loss对w的梯度 x.transpose() 转置
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)  # Adagrad  w.shape=(18*9+1, 1)

np.save('weight.npy', w)  # 保存权重  w.shape=(18*9+1, 1)

"""
#******************************************************************
#  x_train_set(4521, 18*9)  y_train_set(4521, 1) ; x_validation(1131,18*9)  y_validation(1131, 1)    
#  w.shape=(18*9+1, 1)
x_train_set = np.concatenate((np.ones([4521, 1]), x_train_set), axis = 1).astype(float)#concatenate：连在一起
predict_y = np.dot(x_train_set, w)
cnt=0
for i in range(len(predict_y)):
    if abs(math.floor(predict_y[i])-math.floor(y_train_set[i])) < 5:
        cnt+=1
        #print(y_train_set[i])
print("准确率=",str(cnt*100/len(predict_y)),"%")
#******************************************************************
"""
# Testing  Predict PM2.5
test_data = pd.read_csv('./test.csv', header=None, encoding='big5')
test_data = test_data.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()  # test_data.shape = (18*240, 9)  18*240组=4320 9个小时的数据
test_x = np.empty([240, 18 * 9], dtype=float)  # test_x.shape=(240, 18*9)

for i in range(240):  # 数据的压缩，一组id数据，放在一行处理；由原来的的18 x 9变成1 x (18*9)
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)  # test_x.shape=(240, 162)

for i in range(len(test_x)):  # len(test_x)=240
    for j in range(len(test_x[0])):  # len(test_x[0])=18*9=162
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]  # 数据标准化处理
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)  # concatenate：连在一起
# 在test_x中的第一列前插入一列 全为1，相当于偏置（b）的系数, test_x.shape=(240,18*9+1)


# Predict PM2.5   预测第十小时的 PM2.5
w = np.load('weight.npy')  # w.shape=(18*9+1, 1)
ans_y = np.dot(test_x, w)  # test_x.shape = (240, 163)

# Save Prediction to CSV File
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'PM2.5_value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)

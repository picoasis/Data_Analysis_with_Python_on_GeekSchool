#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:12:32 2021

@author: picoasis
"""

'''
用CART分类树创建分类树，
基于基尼系数，
给iris数据集构造一棵分类决策树。

'''

#import seaborn as sns
#iris = sns.load_dataset('iris', cache=True)
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

#准备数据集
iris = load_iris()
# 获取特征及和分类标识 ： -用离散值 0，1，2代表类别
features = iris.data
labels = iris.target 
#随机抽取33%的数据作为测试集，其余为训练集
train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size=0.33,random_state=0)
#创建CART分类树
clf = DecisionTreeClassifier(criterion='gini')
#拟合构造CART分类树
clf = clf.fit(train_features,train_labels)
#用CART分类树做预测 
test_predict = clf.predict(test_features)
#比对 预测结果和测试集结果
score = accuracy_score(test_labels,test_predict)
print("CART分类树准确率 %.4lf" % score)

'''
'''
使用CART回归树做预测
sklearn自带的波士顿房价数据集。
该数据集给出了影响房价的一些指标，比如犯罪率，房产税等，最后给出了房价。

根据这些指标，我们使用CART回归树对波士顿房价进行预测。
'''

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor

#数据集
boston = load_boston()
#探索数据
print(boston.feature_names)
#获取特征及和房价 ：相较于分类树中代表类别的0，1，2，在回归树中，target是连续值
features = boston.data
prices = boston.target
#随机抽取33%的数据集作为测试集，其余为训练集
train_features,test_features,train_price,test_price = train_test_split(features,prices,test_size=0.33)
#创建CART回归树
dtr = DecisionTreeRegressor()
#拟合构造CART回归树
dtr.fit(train_features,train_price)
#预测测试集中的房价
predict_price = dtr.predict(test_features)
#测试集的结果评价
print( '回归树二乘偏差均值', mean_squared_error(test_price,predict_price))
print( '回归树绝对值偏差均值', mean_absolute_error(test_price,predict_price))
                                                                         

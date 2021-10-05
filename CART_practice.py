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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

#准备数据集
iris = load_iris()
# 获取特征及和分类标识
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
                                                                         


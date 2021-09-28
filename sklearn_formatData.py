#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 17:48:51 2021

@author: picoasis
"""

'''数据规范化 使用sklearn'''

from sklearn import preprocessing
import numpy as np


'''Min-max 规范化'''
'''
新数值 =（原数值 - 极小值）/（极大值 - 极小值）
MinMaxScaler 是专门做这个的，
它允许我们给定一个最大值与最小值，然后将原数据投射到[min, max]中。
默认情况下[min,max]是[0,1]，也就是把原始数据投放到[0,1]范围内。
'''

#每一行表示一个样本，每一列表示1个特征
x = np.array([[0.,-3.,1.],
              [3.,1.,2.,],
              [0.,1.,-1.]
    ])
#将数据进行【0，1】规范化

min_max_scaler = preprocessing.MinMaxScaler()
minmax_x = min_max_scaler.fit_transform(x)
print('minmax_x\n',minmax_x)


'''Z-score规范化'''

'''
新数值 =（原数值 - 均值）/ 标准差
在 SciKit-Learn 库中使用 preprocessing.scale() 函数，
可以直接将给定数据进行 Z-Score 规范化。
'''

scaled_x = preprocessing.scale(x)
print('scaled_x\n',scaled_x)


'''小数定标规范化'''
moveS = np.ceil(np.log10(np.max(abs(x))))
pointde_x = x/(10**moveS)
print('pointed_x\n',pointde_x)


'''假设属性 income 的最小值和最大值分别是 5000 元和 58000 元。
利用 Min-Max 规范化的方法将属性的值映射到 0 至 1 的范围内，
那么属性 income 的 16000 元将被转化为多少？'''

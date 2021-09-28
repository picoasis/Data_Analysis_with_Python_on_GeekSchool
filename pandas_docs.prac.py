#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 17:23:47 2021

@author: picoasis
"""
'''
pandas官网练习
'''
import numpy as np
import pandas as pd

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)

#axis=1 columns  axis=0 index
dfsortindex = df.sort_index(axis=0,ascending=False, inplace = False)
print('dfsortindex\n',dfsortindex )
print('df\n',df)

dfsortvalue=df.sort_values(by='2013-01-01',axis=1)
print('dfsortvalue\n',dfsortvalue)

#按标签选择，loc
#提取一行
df.loc[dates[0]]
#多列
df.loc[:,['A','B']]
#标签切片，部分行 部分列
df.loc[dates[0]:dates[3],['A','B']]
df.loc['20130101':'20130104',['A','B']]
#返回对象降维
df.loc['20130102',['A','B']]

#提取标量值
df.loc[dates[0],'A']
#访问at 单个数值 get or set a single value 
df.at[dates[0],'B']


#按位置选择 iloc  iat
#默认行
#整段切片，按数字提取切片,按指定位置输出
df.iloc[[1,2,4],[3,0]]

#布尔索引
df[df.A>0] #只返回满足条件的行
df[df>0] #返回的DF ，shape不变，不满足的数值成为了NaN，方便用if进行下一步操作
#isin()筛选
df2 = df.copy()
df2['E'] = ['One','One','Two','two','four','Three']
print('df2\n',df2)
df2[df2['E'].isin(['two','four'])]


#赋值  
#用索引相同的series赋值
s1 = pd.Series([1,2,3,4,5,6],index=dates)
df['F']=s1
#按标签 
#按位置
#按数组
df.loc[:,'D'] = np.array([5]*len(df))

#用where条件赋值
df2 = df.copy()
df2[df2>0]=-df2


#缺失值  np.nan





















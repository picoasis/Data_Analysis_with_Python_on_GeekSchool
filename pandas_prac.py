#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:05:37 2021

@author: picoasis
"""
'''
对于下表的数据，请使用 Pandas 中的 DataFrame 
进行创建，
并对数据进行清洗。
同时新增一列“总和”计算每个人的三科成绩之和。
'''
import numpy as np
import pandas as pd
from pandas import DataFrame

# 缺失值如何表示： numpy用  np.nan
data = {"语文":[66,95,95,90,80,80],"英语":[65,85,92,88,90,90],
        "数学":[np.nan ,98,96,77,90,90]}
scores = DataFrame(data,index=['张飞',"关羽","赵云","黄忠","典韦","典韦"])

#重复行
scores = scores.drop_duplicates()
#缺失值如何处理 用0代替/用成绩均值代替
scores["数学"] = scores["数学"].fillna(scores["数学"].mean())
#print(scores)

totalS = []
for index,r in scores.iterrows():
    totalS.append( r["数学"]+r["英语"]+r["语文"] )
    
#用apply求和    
#def total_score(df):
#    df['总分'] = df['语文'] + df['英语'] + df['数学']
#   return df
#df = df.apply(total_score, axis=1)
    
scores['总分'] = totalS
print(scores)
                     
                                 
                                 


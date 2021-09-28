import numpy as np
import pandas as pd 
from pandas import Series,DataFrame

# series
##创建
x1 = Series([1,2,3,4])
x2 = Series(data=[1,2,3,4],index=['a','b','c','d'])
print(x1)
print(x2)

## 字典方式创建series
d = {'a':1,'b':2,'c':3,'d':4}
x3 = Series(d)
print(x3)


# DataFrame,
##创建
data = {'Chinese':[66,95,93,90,80],'English':[65,85,90,98,87],'Math':[76,90,93,87,85]}
df1 = DataFrame(data)
df2= DataFrame(data,index=['ZhangFei','GuanYu','ZhaoYun','HuangZhong','DianWei'],columns=['English','Chinese','Math'])
print(df1)
print(df2)


# 数据导入和输出
score = DataFrame(pd.read_excel('data.xlsx'))
score.to_excel('data1.xlsx')
print(score)


#数据清洗
# # 删除DataFrame中不必要的行或列
df2  = df2.drop(columns=['Chinese'])
df2  = df2.drop(index=['ZhangFei','DianWei'])
## 重命名 列名，使更容易识别
# rename(columns = new_names,inplace=true)
df2.rename(columns={'English':'YingYu','Math':'ShuXue'}, inplace = True)
# #重复行  drop_duplicates()
df = df.drop_duplicates()

# 格式问题  astype

df2['ShuXue'].astype('U')
df2['YingYu'].astype(np.int64)

type(df2['ShuXue']['GuanYu'])#??结果是numpy.int64，不是str




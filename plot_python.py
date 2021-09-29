#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:52:01 2021

@author: picoasis
"""

'''Python可视化'''
import numpy as np
import pandas as pd

import  matplotlib.pyplot as plt
import seaborn as sns

'''
散点图   

plt.scatter(x,y,marker = None)
x、y 是坐标，marker 代表了标记的符号。比如“x”、“>”或者“o”。

sns.jointplot(x,y,data=None,kind='scatter')
x、y 是 data 中的下标。
data 就是我们要传入的数据，一般是 DataFrame 类型。
kind 这类我们取 scatter，代表散点的意思。
当然 kind 还可以取其他值，不同的 kind 代表不同的视图绘制方式。
'''

'''
# 绘制随机的1000个点

N =1000 
x = np.random.randn(N)
y = np.random.randn(N) 

plt.scatter(x,y,marker='x')
plt.show()

df = pd.DataFrame({'x':x,'y':y})
sns.jointplot(x="x", y="y", data=df, kind="scatter")
plt.show()
'''

'''
折线图：数据随时间变化的趋势
Matplotlib 中， plt.plot（）
需要提前把数据按照 x 轴的大小进行排序，要不画出来的折线图就无法按照 x 轴递增的顺序展示。

SeaBorn 中 ，sns.lineplot(x,y,data=None)
其中 x、y 是 data 中的下标。
data 就是我们要传入的数据，一般是 DataFrame 类型。

'''

'''
# x、y 的数组。x 数组代表时间（年），y 数组我们随便设置几个取值。

np.random
x = np.arange(2010,2020)
y = np.random.randint(2,50,10)

plt.plot(x,y)
plt.show()

df = pd.DataFrame({"x":x,"y":y})
sns.lineplot(x="x",y="y",data = df)
plt.show()
'''

'''
直方图
直方图可以看到变量的数值分布，
把横坐标等分成了一定数量的小区间，这个小区间也叫作“箱子”，
然后在每个“箱子”内用矩形条（bars）展示该箱子的箱子数（也就是 y 值），
这样就完成了对数据集的直方图分布的可视化。

Matplotlib 中，我们使用 plt.hist(x, bins=10) 函数，
其中参数 x 是一维数组，bins 代表直方图中的箱子数量，默认是 10。

Seaborn 中，我们使用 sns.distplot(x, bins=10, kde=True) 函数。
其中参数 x 是一维数组，
bins 代表直方图中的箱子数量，
kde 代表显示核密度估计，默认是 True，我们也可以把 kde 设置为 False，不进行显示。
核密度估计是通过核函数帮我们来估计概率密度的方法。

'''

'''
a = np.random.randn(100)
s = pd.Series(a) 
plt.hist(s) #直接对a画图也成立
plt.show()

sns.distplot(s,kde = False)
plt.show()
sns.distplot(s, kde = True)
'''

'''
条形图
条形图可以帮我们查看类别的特征。
在条形图中，长条形的长度表示类别的频数，宽度表示类别。

Matplotlib 中，我们使用 plt.bar(x, height) 函数，
其中参数 x 代表 x 轴的位置序列，
height 是 y 轴的数值序列，也就是柱子的高度。

Seaborn 中，我们使用 sns.barplot(x=None, y=None, data=None) 函数。
其中参数 data 为 DataFrame 类型，
x、y 是 data 中的变量。
'''
'''

x =['cat1','cat2','cat3','cat4','cat5']
y =[1,4,7,3,9]

plt.bar(x,y)
plt.show()

sns.barplot(x,y)
plt.show()
'''
'''
箱线图，盒式图，由五个数值点组成：
最大值 (max)、最小值 (min)、中位数 (median) 和上下四分位数 (Q3, Q1)。
它可以帮我们分析出数据的差异性、离散程度和异常值等。

Matplotlib 中，我们使用 plt.boxplot(x, labels=None) 函数，
其中参数 x 代表要绘制箱线图的数据，
labels 是缺省值，可以为箱线图添加标签。

在 Seaborn 中，我们使用 sns.boxplot(x=None, y=None, data=None) 函数。
其中参数 data 为 DataFrame 类型，
x、y 是 data 中的变量。

'''
'''

data = np.random.normal(size=(10,4))#0-1 之间的 10*4 维度数据
labels=['A','B','C','D']

plt.boxplot(data,labels = labels)
plt.show()

df = pd.DataFrame(data,columns=labels)
sns.boxplot(data = df)
plt.show()
'''
'''
饼图
Matplotlib 中，我们使用 plt.pie(x, labels=None) 函数，
其中参数 x 代表要绘制饼图的数据，
labels 是缺省值，可以为饼图添加标签。

'''
#labels 数组，分别代表高中、本科、硕士、博士和其他几种学历的分类标签。
#nums 代表这些学历对应的人数。
'''
nums = [25,37,33,37,6]
labels=['High-School','Bachelor','Master','Ph.D','Others']
plt.pie(x=nums, labels=labels)
plt.show()
'''

'''
热力图。
heat map，是一种矩阵表示方法，
是一种非常直观的多元变量分析方法。

其中矩阵中的元素值用颜色来代表，不同的颜色代表不同大小的值。

一般使用 Seaborn 中的 sns.heatmap(data) 函数，
其中 data 代表需要绘制的热力图数据。

'''

#使用 Seaborn 中自带的数据集 flights，
#数据集地址 https://github.com/mwaskom/seaborn-data
#该数据集记录了 1949 年到 1960 年期间，每个月的航班乘客的数量。
# 如果seaborn导入数据集出现错误，参考下面的链接
#https://blog.csdn.net/yue81560/article/details/106725687?ops_request_misc=&request_id=&biz_id=102&utm_term=seaborn%E5%AF%BC%E5%85%A5%E6%95%B0%E6%8D%AE%E9%9B%86%20mac&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-106725687.first_rank_v2_pc_rank_v29&spm=1018.2226.3001.4187
#https://blog.csdn.net/fightingoyo/article/details/106920773?ops_request_misc=&request_id=&biz_id=102&utm_term=load_dataset&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-106920773.nonecase&spm=1018.2226.3001.4187
'''
# 数据准备
flights = sns.load_dataset("flights")
data=flights.pivot('year','month','passengers')
# 用Seaborn画热力图
sns.heatmap(data)
plt.show()
'''

'''
pandas.DataFrame.pivot(index,columns,values)
#第一个index是重塑的新表的索引名称是什么，
#第二个columns是重塑的新表的列名称是什么，一般来说就是被统计列的分组，
#第三个values就是生成新列的值应该是多少，如果没有，则会对data_df剩下未统计的列进行重新排列放到columns的上层。
'''

'''
蜘蛛图
显示一对多关系的方法。
'''
'''
假设我们想要给王者荣耀的玩家做一个战力图，
指标一共包括:推进、KDA、生存、团战、发育和输出。那该如何做呢？
用 Matplotlib 来进行画图，
首先设置两个数组：labels 和 stats。
他们分别保存了这些属性的 名称 和 属性值。
蜘蛛图是一个圆形，你需要:
    计算每个坐标的角度，然后对这些数值进行设置。
    当画完最后一个点后，需要与第一个点进行连线。
    
因为需要计算角度，所以我们要准备 angles 数组；
又因为需要设定统计结果的数值，所以我们要设定 stats 数组。
并且需要在原有 angles 和 stats 数组上增加一位，也就是添加数组的第一个元素。

'''
#字体
# 在 Mac 下设置中文字体，可以使用以下路径：
# 设置中文字体
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc", size=14)
'''

labels = np.array([u"推进","KDA",u"生存",u"团战",u"发育",u"输出"])
stats = [83, 61, 95, 67, 76, 88]

angels = np.linspace(0,2*np.pi,len(labels),endpoint=False)
stats = np.concatenate((stats,[stats[0]]))
angels = np.concatenate((angels,[angels[0]]))
labels = np.concatenate((labels,[labels[0]]))

fig = plt.figure()#空白figure对象
ax = fig.add_subplot(111,polar=True)
ax.plot(angels,stats,'o-',linewidth =2)
ax.fill(angels,stats,alpha=0.25)

ax.set_thetagrids(angels*180/np.pi,labels)
plt.show()
'''

'''
二元变量分布

在 Seaborn 里，使用二元变量分布是非常方便的，
直接使用 sns.jointplot(x, y, data=None, kind) 函数即可。
其中用 kind 表示不同的视图类型：“kind='scatter'”代表散点图，
“kind='kde'”代表核密度图，
“kind='hex' ”代表 Hexbin 图，它代表的是直方图的二维模拟。
'''

# 使用 Seaborn 中自带的数据集 tips，
#这个数据集记录了不同顾客在餐厅的消费账单及小费情况。
#代码中 total_bill 保存了客户的账单金额，
#tip 是该客户给出的小费金额。
#我们可以用 Seaborn 中的 jointplot 来探索这两个变量之间的关系。

#cache=True 使用本地存储的seaborn-data，
#如果下载后放在用户 username 文件夹下，可以不输入 data_home 的值，会自动找到
'''
tips = sns.load_dataset('tips',cache=True)
print(tips.head(10))
sns.jointplot(x="total_bill",y="tip",data=tips, kind = 'scatter')
sns.jointplot(x="total_bill",y="tip",data=tips, kind = 'kde')
sns.jointplot(x="total_bill",y="tip",data=tips, kind = 'hex')
plt.show()
'''

'''
成对关系
探索数据集中的多个成对双变量的分布，可以直接采用 sns.pairplot() 函数。
它会同时展示出 DataFrame 中每对变量的关系，
另外在对角线上，你能看到每个变量自身作为单变量的分布情况。
它可以说是探索性分析中的常用函数，可以很快帮我们理解变量对之间的关系。
pairplot 函数的使用，
就像在 DataFrame 中使用 describe() 函数一样方便，
是数据探索中的常用函数。
'''

'''
示例：
iris 数据集，这个数据集也叫鸢尾花数据集。
鸢尾花可以分成 Setosa、Versicolour 和 Virginica 三个品种，
在这个数据集中，针对每一个品种，都有 50 个数据，
每个数据中包括了 4 个属性，分别是花萼长度、花萼宽度、花瓣长度和花瓣宽度。
通过这些数据，需要你来预测鸢尾花卉属于三个品种中的哪一种。
'''
'''
iris = sns.load_dataset('iris', cache=True)
sns.pairplot(iris)
plt.show()
'''


'''
练习
1. Seaborn 数据集中自带了 car_crashes 数据集，这是一个国外车祸的数据集，
你要如何对这个数据集进行成对关系的探索呢？
2. 第二个问题就是，请你用 Seaborn 画二元变量分布图，
如果想要画散点图，核密度图，Hexbin 图，函数该怎样写？
'''
carCrashes =sns.load_dataset('car_crashes',cache = True)
#探索成对关系
sns.pairplot(carCrashes)
plt.show()
#二元变量 酒驾和总量的关系
sns.jointplot(x="alcohol",y="total",data=carCrashes, kind = 'scatter')
sns.jointplot(x="alcohol",y="total",data=carCrashes, kind = 'kde')
sns.jointplot(x="alcohol",y="total",data=carCrashes, kind = 'hex')


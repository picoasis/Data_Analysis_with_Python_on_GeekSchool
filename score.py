# -*- coding: utf-8 -*-
'''
假设一个团队里有5名学员，成绩如下表所示。
1.用NumPy统计下这些人在语文、英语、数学中的平均成绩、最小成绩、最大成绩、方差、标准差。
2.总成绩排序，得出名次进行成绩输出。
'''

import numpy as np

 #初始化
score_dtype = np.dtype({
 	'names':['name','chinese','english','math','total','rank'],
 	'formats':['U','i','i','i','i','i']})
scores = np.array([
	('ZhangFei',66,65,30,0,0),
	('GuanYu',95,85,98,0,0),
	('ZhaoYun',93,92,96,0,0),
	('HuangZhong',90,88,77,0,0),
	('DianWei',80,90,90,0,0)
	],dtype = score_dtype)

cn_sc = scores['chinese']
en_sc = scores['english']
mt_sc = scores['math']
scores['total']= cn_sc + en_sc + mt_sc  #??? 引用字段名称产生的数组，是原始数组的视图，会改变原始数组呀
# ranking = np.sort(scores, order='total' )
ranking= sorted(scores,reverse=True,key = lambda x:x['total'])

i = 1
N = len(scores)
ranks=np.zeros(N)
while i <= N:
	ranks[i-1]=i
	i+=1

scores['rank'] = ranks
print(scores)


def show(xkname,xkscore):
	print('{} | {} | {} | {} | {:.2f} | {:.2f}'
		.format(xkname,np.mean(xkscore),np.min(xkscore),np.max(xkscore),np.var(xkscore),np.std(xkscore)))

print("科目|平均成绩|最小成绩|最大成绩|方差|标准差")
show('语文',cn_sc )
show('数学',mt_sc)
show('英语',en_sc)

	
#-----导入库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn import metrics
from pandas import read_csv
from matplotlib import pyplot
from sqlalchemy import values

from numpy import asarray, quantile
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.ensemble import RandomForestRegressor


#-----导入数据
series = series = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv', header = 0, index_col = 0)
print(series)
values = series.values

#-----可视化数据
pyplot.plot(values)
pyplot.show()

#-----预测
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else  data.shape[1]
    df = DataFrame(data)
    cols = list()

    for i in range(n_in,0,-1):
        cols.append(df.shift(i))

    for i in range(0,n_out):
        cols.append(df.shift(-i))
    
    agg = concat(cols,axis=1)

    if dropnan:
        agg.dropna(inplace=True)

    return agg.values

#-----准备数据集
series = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-max-temperatures.csv', header=0, index_col=0)
data = series.values
n_test = 100 #模型测试的数量是100个

def train_test_split(data,n_test): #把所有数据分为训练集和测试集
    return data[:-n_test,:],data[-n_test:,:]

train,test = train_test_split(data,n_test) # train_test_split（）函数将数据划分为训练集train和测试集test，数据库为data，随机数量为100

#--训练集
Train = series_to_supervised(train,n_in = 6) #将时间数据集转换为用于学习训练的数据集,从上面👆划分出来的train中选60%
X_train,y_train = Train[:,:-1],Train[:,-1] # X:要划分的样本特征集（输入的信息） y:需要划分的样本结果（输出结果）
print("--Train-- :",Train)
#--测试集
Test = series_to_supervised(train,n_in = 6)
X_test,y_test = Train[:,:-1],Train[:,-1]
print("--Test--: ",Test)

#-----建立回归模型，使用训练集train
model = RandomForestRegressor(n_estimators = 1000) #n_estimators：决策树的个数，越多越好，但是性能就会越差
model.fit(X_train,y_train)

#-----预测
row = values[-6:].flatten() # 上面n_in是6，所以倒序排列，第一个元素为-6，j值缺省默认是0，
                            # 具体见https://blog.csdn.net/qiushangren/article/details/103550923?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-4-103550923-null-null.pc_agg_new_rank&utm_term=Python中values%E3%80%94%3A%2C0%E3%80%95什么意思&spm=1000.2123.3001.4430 ,
                            # 在末尾加一个flatten() 变成一行的方便统计分析,默认缺省参数为0，也就是说flatten()和flatte(0)效果一样。
yhat = model.predict(asarray([row]))
print('Input: %s, Predicted: %.3f' % (row, yhat[0]))

#-----梯度提升回归
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html

from sklearn.ensemble import GradientBoostingRegressor
quantile = [0.01,0.05,0.50,0.95,0.99] # 四分位数

#-----梯度提升回归函数
def GBM(q): # 关于GBR函数，见https://blog.csdn.net/anshiquanshu/article/details/78542852 
    modele = GradientBoostingRegressor(loss='quantile',# loss指的是每一次节点分裂所要最小化的损失函数
                                        alpha=q, # 梯度下降算法中合适的学习率，更合适的叫法“步长”，步长决定了每一次迭代过程中，会往梯度下降的方向移动的距离，
                                                    #如果步长很大，算法会在局部最优点来回跳跃，不会收敛，如果步长很小，算法收敛速度很慢 https://www.zhihu.com/question/54097634?sort=created
                                        n_estimators=500, # 定义了需要使用到的决定树的数量
                                        max_depth=8,# max_depth定义了树的最大深度
                                        learning_rate=.01, # 决定着每一个决定树对于最终结果的影响。GBM设定了初始的权重值之后，每一次树分类都会更新这个值，
                                                             #而learning_ rate控制着每次更新的幅度。一般来说这个值不应该设的比较大，因为较小的learning rate使得模型对不同的树更加稳健，就能更好地综合它们的结果。
                                        min_samples_leaf=20, # min_samples_leaf 定义了树中一个节点所需要用来分裂的最少样本数
                                        min_samples_split=20) # min_samples_split 定义了树中终点节点所需要的最少的样本数 

    modele.fit (X_train,y_train) # X_train,y_train将填充模型
    predict = pd.Series(modele.predict(X_test).round(2)) # round取小数点后2位

    return predict,modele           
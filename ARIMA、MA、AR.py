# https://blog.csdn.net/WHYbeHERE/article/details/109277597
#数据导入
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6

#数据导入和查看
data=pd.read_csv('Classeur1.csv', sep=";")
print(data.head())
print('\nData Types:')
print(data.dtypes)

#格式转换
dateparse=lambda dates: pd.datetime.strptime(dates,'%d/%m/%Y')
data=pd.read_csv('Classeur1.csv', sep=";",parse_dates=[' date'],index_col=' date',date_parser=dateparse)
data.head()
# parse_dates：指定包含日期时间信息的列。例子里的列名是'date‘
# index_col：在TS数据中使用pandas的关键是索引必须是日期等时间变量。所以这个参数告诉pandas使用'date'列作为索引
# date_parser：它指定了一个将输入字符串转换为datetime可变的函数。
# pandas 默认读取格式为'YYYY-MM-DD HH:MM:SS'的数据。如果这里的时间格式不一样，就要重新定义时间格式，dataparse函数可以用于此目的。


#查看数据的索引
print(data.index)

#平稳性检验
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics 
    rolmean=timeseries.rolling(12).mean()
    rolstd=timeseries.rolling(12).std()

    #Plot rolling statistics:
    orig=plt.plot(timeseries,color='blue',label='Original') 
    mean=plt.plot(rolmean,color='red',label='Rolling Mean') #均值
    std=plt.plot(rolstd,color='black',label='Rolling Std') #标准差
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller Test:
    print('Results of Dickey-Fuller Test:')
    dftest=adfuller(timeseries,autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key]=value 
    print(dfoutput)

test_stationarity()






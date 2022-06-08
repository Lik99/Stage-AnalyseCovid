#解释器用3.9.12（'base':conda）

#-------------------------------------------import les libraries
import os
from cProfile import label
from operator import le
from time import process_time, time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

plt.style.use("seaborn-whitegrid")

import logging

logging.basicConfig(level=logging.WARNING)
import sys
from pathlib import Path

import statsmodels.tsa.arima.model
from fbprophet import Prophet
from neuralprophet import NeuralProphet

sys.path.insert(0,str(Path(os.getcwd()).parent/"utilities"))
import data


#-------------------------------------------fonctionner data
df = pd.DataFrame(pd.read_csv("Classeur1.csv",header=0,index_col=0,sep= ';', decimal='.'))
df_train,df_test = train_test_split(df)

ax = df_train["y"].plot(figsize=(10,6),label="train")
df_test["y"].plot(ax=ax,label="test")
ax.legend()
ax.set_xlabel("date")
ax.set_ylabel("cas")

df_train = df_train.reset_index()
df_test = df_test.reset_index()

fit_time_ar = []
fit_time_np = []
mae_np = []
lag_range = range(1,25)
logging.getLogger("nprophet").setLevel(logging.WARNING)
for lag in lag_range:
    # fit statsmodels
    t1 = process_time()
    model_arima = statsmodels.tsa.arima.model.ARIMA(endog=df_train.set_index('ds'), order=(lag,0,0), freq='1D').fit()
    fit_time_ar.append(process_time() - t1)

    # fit neuralprophet
    t1 = process_time()
    model_nprophet_ar = NeuralProphet(
        growth="off",
        n_changepoints=0,
        n_forecasts=1,
        n_lags=lag,
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        loss_func='MSE',
        normalize='off'
    )
    mae_np.append(model_nprophet_ar.fit(df_train, freq="D"))
    fit_time_np.append(process_time() - t1)

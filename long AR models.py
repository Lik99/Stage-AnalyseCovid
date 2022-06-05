#-------------------------------------------import les libraries
from cProfile import label
from operator import le
from time import process_time,time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
import utilities

#-------------------------------------------fonctionner data
data = pd.read_csv("Classeur1.csv",header=0,index_col=0,sep= ';', decimal='.')
df = pd.DataFrame(data)
df_train,df_test = utilities.split_ts(df)

ax = df_train["y"].plot(figsize=(10,6),label="train")
df_test["y"].plot(ax=ax,label="test")
ax.legend()
ax.set_xlabel("date")
ax.set_ylabel("code")

df_train = df_train.reset_index()
df_test = df_test.reset_index()

fit_time_ar = []
fit_time_np = []
mae_np = []
lag_range = range(1,25)
logging.getLogger("nprophet").setLevel(logging.WARNING)
